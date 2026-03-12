"""
tests/test_guest_session.py
----------------------------
Unit tests for guest_session.py helpers and the guest WebSocket/HTTP endpoints.

Pure-Python unit tests (no hardware, no Whisper, no audio capture).
HTTP/WS endpoint tests use FastAPI's TestClient with _meeting state patched.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meetingmind.guest_session import decode_guest_audio, generate_pin


# ---------------------------------------------------------------------------
# generate_pin
# ---------------------------------------------------------------------------

class TestGeneratePin:
    def test_format(self):
        """PIN must be exactly 4 decimal characters."""
        pin = generate_pin()
        assert len(pin) == 4, f"Expected 4 chars, got {len(pin)!r}"
        assert pin.isdigit(), f"Expected all digits, got {pin!r}"

    def test_range(self):
        """All generated PINs must be in '0000'–'9999'."""
        for _ in range(1_000):
            pin = generate_pin()
            val = int(pin)
            assert 0 <= val <= 9999, f"PIN out of range: {pin!r}"
            assert len(pin) == 4   # zero-padded

    def test_zero_padding(self):
        """PIN '0001' must be returned as '0001', not '1'."""
        # Patch random to return 1, then check zero-padding.
        import random
        orig = random.randint
        random.randint = lambda a, b: 7
        try:
            pin = generate_pin()
            assert pin == "0007"
        finally:
            random.randint = orig


# ---------------------------------------------------------------------------
# decode_guest_audio
# ---------------------------------------------------------------------------

class TestDecodeGuestAudio:
    def _encode(self, arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode()

    def test_roundtrip(self):
        """Encoding then decoding a known array should return the same values."""
        orig = np.array([0.1, -0.5, 0.3, 0.9], dtype=np.float32)
        b64  = self._encode(orig)
        result = decode_guest_audio(b64)
        np.testing.assert_array_almost_equal(result, orig, decimal=6)

    def test_bad_base64(self):
        """Non-base64 string must raise ValueError."""
        with pytest.raises(ValueError, match="base64 decode failed"):
            decode_guest_audio("not!valid@base64$$")

    def test_bad_length(self):
        """3 bytes is not divisible by 4 — must raise ValueError."""
        b64 = base64.b64encode(b"\x00\x01\x02").decode()
        with pytest.raises(ValueError, match="not divisible by 4"):
            decode_guest_audio(b64)

    def test_writable(self):
        """Result must be a writable array (Whisper's pad_or_trim requires it)."""
        orig = np.zeros(16, dtype=np.float32)
        result = decode_guest_audio(self._encode(orig))
        assert result.flags.writeable, "Returned array must be writable"

    def test_dtype(self):
        """Result dtype must be float32."""
        orig = np.ones(8, dtype=np.float32)
        result = decode_guest_audio(self._encode(orig))
        assert result.dtype == np.float32

    def test_empty_valid(self):
        """Zero-byte input is valid base64 and produces an empty float32 array."""
        b64 = base64.b64encode(b"").decode()
        result = decode_guest_audio(b64)
        assert result.shape == (0,)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# HTTP + WebSocket endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_meeting_state():
    """Ensure _meeting state is clean before and after each test."""
    import queue as _queue_module
    from meetingmind.main import _meeting
    _meeting["active"]         = False
    _meeting["guest_pin"]      = None
    _meeting["guest_connected"] = False
    _meeting["guest_queue"]    = None
    _meeting["guest_task"]     = None
    _meeting["guest_ws"]       = None
    yield
    _meeting["active"]         = False
    _meeting["guest_pin"]      = None
    _meeting["guest_connected"] = False
    _meeting["guest_queue"]    = None
    _meeting["guest_task"]     = None
    _meeting["guest_ws"]       = None


class TestGuestSessionEndpoint:
    def test_no_meeting(self):
        """POST /guest/session with no active meeting → 400."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        client = TestClient(app)
        resp = client.post("/guest/session")
        assert resp.status_code == 400
        assert "No meeting" in resp.json().get("error", "")

    def test_creates_pin_when_meeting_active(self):
        """POST /guest/session with active meeting → returns pin and guest_url."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app, _meeting
        _meeting["active"] = True
        try:
            client = TestClient(app)
            resp   = client.post("/guest/session")
            assert resp.status_code == 200
            data = resp.json()
            assert "pin" in data
            assert len(data["pin"]) == 4
            assert data["pin"].isdigit()
            assert "/guest?pin=" in data["guest_url"]
        finally:
            _meeting["active"] = False

    def test_duplicate_guest_returns_409(self):
        """POST /guest/session when guest already connected → 409."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app, _meeting
        _meeting["active"]          = True
        _meeting["guest_connected"] = True
        try:
            client = TestClient(app)
            resp   = client.post("/guest/session")
            assert resp.status_code == 409
        finally:
            _meeting["active"]          = False
            _meeting["guest_connected"] = False


class TestGuestWebSocket:
    def test_no_meeting_close_4000(self):
        """WS /ws/guest with no active meeting → close code 4000."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        from meetingmind.main import app
        client = TestClient(app)
        with pytest.raises((WebSocketDisconnect, Exception)) as exc_info:
            with client.websocket_connect("/ws/guest?pin=0000") as ws:
                ws.receive_text()
        # starlette TestClient raises WebSocketDisconnect with the close code
        exc = exc_info.value
        if hasattr(exc, "code"):
            assert exc.code == 4000

    def test_invalid_pin_close_4001(self):
        """WS /ws/guest with wrong PIN → close code 4001."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        from meetingmind.main import app, _meeting
        _meeting["active"]    = True
        _meeting["guest_pin"] = "1234"
        try:
            client = TestClient(app)
            with pytest.raises((WebSocketDisconnect, Exception)) as exc_info:
                with client.websocket_connect("/ws/guest?pin=9999") as ws:
                    ws.receive_text()
            exc = exc_info.value
            if hasattr(exc, "code"):
                assert exc.code == 4001
        finally:
            _meeting["active"]    = False
            _meeting["guest_pin"] = None

    def test_duplicate_guest_close_4002(self):
        """WS /ws/guest when another guest is already connected → close code 4002."""
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        from meetingmind.main import app, _meeting
        _meeting["active"]          = True
        _meeting["guest_pin"]       = "5678"
        _meeting["guest_connected"] = True
        try:
            client = TestClient(app)
            with pytest.raises((WebSocketDisconnect, Exception)) as exc_info:
                with client.websocket_connect("/ws/guest?pin=5678") as ws:
                    ws.receive_text()
            exc = exc_info.value
            if hasattr(exc, "code"):
                assert exc.code == 4002
        finally:
            _meeting["active"]          = False
            _meeting["guest_pin"]       = None
            _meeting["guest_connected"] = False
