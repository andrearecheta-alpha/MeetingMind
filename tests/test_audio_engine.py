"""
tests/test_audio_engine.py
--------------------------
Unit and integration tests for the Phase 1 audio engine.

Run with:
    cd MeetingMind
    pytest tests/test_audio_engine.py -v

Notes
-----
- Tests that load Whisper use the 'tiny' model to keep run time short.
- Tests marked @pytest.mark.hardware require a microphone and are skipped
  in CI environments where no audio device is present.
- No audio is recorded during tests — mocks are used where hardware access
  would otherwise be required.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src/ is importable when running tests from the project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


# ===========================================================================
# AudioCapture tests
# ===========================================================================

class TestAudioCaptureHelpers:
    """Test static helper methods that require no hardware."""

    def test_to_float32_range(self):
        """int16 max value should map to exactly 1.0 after normalisation."""
        from meetingmind.audio_capture import AudioCapture
        raw = np.array([32767, -32768, 0], dtype=np.int16).tobytes()
        result = AudioCapture._to_float32(raw)
        assert result[0] == pytest.approx(1.0,    abs=1e-4)
        assert result[1] == pytest.approx(-1.0,   abs=1e-4)
        assert result[2] == pytest.approx(0.0,    abs=1e-6)

    def test_resample_reduces_length(self):
        """Resampling 44 100 Hz → 16 000 Hz should shorten the array."""
        from meetingmind.audio_capture import AudioCapture
        data = np.random.rand(44_100).astype(np.float32)
        out  = AudioCapture._resample(data, src_rate=44_100, dst_rate=16_000)
        expected = int(44_100 * 16_000 / 44_100)
        assert abs(len(out) - expected) <= 2  # allow ±2 samples rounding

    def test_resample_identity_when_same_rate(self):
        """_resample should return the input unchanged when rates are equal."""
        from meetingmind.audio_capture import AudioCapture
        data = np.ones(1_600, dtype=np.float32)
        out  = AudioCapture._resample(data, src_rate=16_000, dst_rate=16_000)
        np.testing.assert_array_equal(out, data)

    def test_rms_silence(self):
        """RMS of a silent (all-zero) array must be 0.0."""
        from meetingmind.audio_capture import AudioCapture
        assert AudioCapture._rms(np.zeros(16_000, dtype=np.float32)) == 0.0

    def test_rms_full_scale_sine(self):
        """RMS of a unit sine wave should be ~0.707 (1/sqrt(2))."""
        from meetingmind.audio_capture import AudioCapture
        t    = np.linspace(0, 2 * np.pi, 16_000, dtype=np.float32)
        sine = np.sin(t)
        assert AudioCapture._rms(sine) == pytest.approx(0.707, abs=0.01)


class TestAudioCaptureInit:
    """Test AudioCapture initialisation without opening any hardware stream."""

    def test_creates_without_error(self):
        """AudioCapture() should not raise even if no mic is plugged in."""
        from meetingmind.audio_capture import AudioCapture
        cap = AudioCapture(privacy_mode=False)
        cap._pa.terminate()  # release PyAudio immediately

    def test_list_devices_returns_list(self):
        """list_devices() must return a list (empty is acceptable in CI)."""
        from meetingmind.audio_capture import AudioCapture
        cap = AudioCapture(privacy_mode=False)
        try:
            devices = cap.list_devices()
            assert isinstance(devices, list)
            for d in devices:
                assert "index" in d
                assert "name"  in d
        finally:
            cap._pa.terminate()

    def test_audio_chunk_dataclass(self):
        """AudioChunk should store all fields correctly."""
        from meetingmind.audio_capture import AudioChunk, AudioSource
        data  = np.zeros(48_000, dtype=np.float32)
        chunk = AudioChunk(data=data, source=AudioSource.MICROPHONE, timestamp=1.0, rms=0.0)
        assert chunk.source    == AudioSource.MICROPHONE
        assert chunk.timestamp == 1.0
        assert chunk.rms       == 0.0
        assert len(chunk.data) == 48_000


# ===========================================================================
# StreamingTranscriber tests
# ===========================================================================

class TestStreamingTranscriber:
    """Tests for the chunk-based Whisper transcriber."""

    def test_whisper_tiny_loads(self):
        """Whisper 'tiny' model should load without errors."""
        from meetingmind.transcriber import _load_whisper_model
        model = _load_whisper_model("tiny")
        assert model is not None

    def test_silent_chunk_returns_none(self):
        """A chunk with RMS=0 (pure silence) must be skipped without calling Whisper."""
        from meetingmind.audio_capture import AudioChunk, AudioSource
        from meetingmind.transcriber   import StreamingTranscriber

        transcriber = StreamingTranscriber(model_size="tiny", privacy_mode=False)
        silent = AudioChunk(
            data=np.zeros(48_000, dtype=np.float32),
            source=AudioSource.MICROPHONE,
            timestamp=time.time(),
            rms=0.0,
        )
        result = transcriber.transcribe_chunk(silent)
        assert result is None

    def test_transcript_chunk_to_dict(self):
        """TranscriptChunk.to_dict() must include all required keys."""
        from meetingmind.transcriber import TranscriptChunk
        chunk = TranscriptChunk(
            text="Hello world",
            language="en",
            source="microphone",
            timestamp=1_700_000_000.0,
            confidence=0.85,
        )
        d = chunk.to_dict()
        assert d["text"]       == "Hello world"
        assert d["language"]   == "en"
        assert d["source"]     == "microphone"
        assert d["confidence"] == 0.85
        assert "timestamp"     in d


# ===========================================================================
# FastAPI endpoint tests
# ===========================================================================

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self):
        """Health endpoint should always return HTTP 200."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app

        client   = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_schema(self):
        """Health response must include status, meeting_active, and connected_clients."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app

        client = TestClient(app)
        data   = client.get("/health").json()
        assert data["status"]            == "ok"
        assert data["meeting_active"]    is False
        assert "connected_clients"       in data


class TestWebSocketEndpoint:
    """Tests for WS /ws/transcribe."""

    def test_connects_and_pong(self):
        """WebSocket should accept connections and respond 'pong' to 'ping'."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app

        client = TestClient(app)
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text("ping")
            reply = ws.receive_text()
            assert reply == "pong"

    def test_multiple_clients_connect(self):
        """Multiple WebSocket clients should be able to connect simultaneously."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app

        client = TestClient(app)
        with client.websocket_connect("/ws/transcribe") as ws1:
            with client.websocket_connect("/ws/transcribe") as ws2:
                ws1.send_text("ping")
                ws2.send_text("ping")
                assert ws1.receive_text() == "pong"
                assert ws2.receive_text() == "pong"


class TestMeetingEndpoints:
    """Tests for POST /meeting/start and /meeting/stop."""

    def test_stop_without_start_returns_400(self):
        """/meeting/stop with no active meeting should return 400."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app, _meeting

        # Ensure clean state
        _meeting["active"] = False

        client   = TestClient(app)
        response = client.post("/meeting/stop")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_double_start_returns_400(self):
        """A second /meeting/start while one is active should return 400."""
        from fastapi.testclient import TestClient
        from meetingmind.main import app, _meeting

        _meeting["active"] = True  # Simulate an active meeting
        try:
            client   = TestClient(app)
            response = client.post("/meeting/start")
            assert response.status_code == 400
        finally:
            _meeting["active"] = False  # Reset state
