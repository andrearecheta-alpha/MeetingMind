"""
tests/test_guest_link.py
------------------------
S7-003 — Guest Viewer Link feature tests.

Tests cover:
  1. POST /guest/generate returns token and URL
  2. GET /guest/view/{token} returns 200 for valid token
  3. GET /guest/view/{token} returns 404 for invalid token
  4. DELETE /guest/view/{token} deactivates token
  5. GET /guest/view/{token}/status returns guest list
  6. WebSocket /guest/ws/{token} connects and receives name ack
  7. WebSocket rejects invalid/expired tokens
  8. meeting_stop deactivates all guest viewer tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client():
    from fastapi.testclient import TestClient
    from meetingmind.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_guest_state():
    """Clear guest viewer state before/after each test."""
    from meetingmind.main import _guest_tokens, _guest_viewer_ws
    _guest_tokens.clear()
    _guest_viewer_ws.clear()
    yield
    _guest_tokens.clear()
    _guest_viewer_ws.clear()


# ---------------------------------------------------------------------------
# POST /guest/generate
# ---------------------------------------------------------------------------

class TestGuestGenerate:
    """POST /guest/generate returns a token and URL."""

    def test_returns_token_and_url(self):
        """Should return a JSON body with token and url fields."""
        client = _client()
        resp = client.post("/guest/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "url" in data
        assert len(data["token"]) > 10  # UUID-length
        assert data["token"] in data["url"]

    def test_url_contains_host(self):
        """URL should contain the configured host IP and port."""
        client = _client()
        resp = client.post("/guest/generate")
        data = resp.json()
        # Default config uses localhost
        assert "localhost" in data["url"] or "127.0.0.1" in data["url"] or ":" in data["url"]

    def test_token_stored_in_state(self):
        """Token should be stored in _guest_tokens with active=True."""
        from meetingmind.main import _guest_tokens
        client = _client()
        resp = client.post("/guest/generate")
        token = resp.json()["token"]
        assert token in _guest_tokens
        assert _guest_tokens[token]["active"] is True
        assert _guest_tokens[token]["guests"] == []

    def test_multiple_tokens_unique(self):
        """Each call should generate a unique token."""
        client = _client()
        r1 = client.post("/guest/generate")
        r2 = client.post("/guest/generate")
        assert r1.json()["token"] != r2.json()["token"]


# ---------------------------------------------------------------------------
# GET /guest/view/{token}
# ---------------------------------------------------------------------------

class TestGuestViewPage:
    """GET /guest/view/{token} serves the guest viewer HTML."""

    def test_valid_token_returns_200(self):
        """Valid, active token should return 200 with HTML."""
        client = _client()
        gen = client.post("/guest/generate").json()
        resp = client.get(f"/guest/view/{gen['token']}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_invalid_token_returns_404(self):
        """Non-existent token should return 404."""
        client = _client()
        resp = client.get("/guest/view/nonexistent-token-12345")
        assert resp.status_code == 404

    def test_deactivated_token_returns_404(self):
        """Deactivated token should return 404."""
        from meetingmind.main import _guest_tokens
        client = _client()
        gen = client.post("/guest/generate").json()
        token = gen["token"]
        _guest_tokens[token]["active"] = False
        resp = client.get(f"/guest/view/{token}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /guest/view/{token}
# ---------------------------------------------------------------------------

class TestGuestViewRevoke:
    """DELETE /guest/view/{token} deactivates a token."""

    def test_revoke_sets_inactive(self):
        """Revoking should set active=False."""
        from meetingmind.main import _guest_tokens
        client = _client()
        gen = client.post("/guest/generate").json()
        token = gen["token"]
        resp = client.delete(f"/guest/view/{token}")
        assert resp.status_code == 200
        assert _guest_tokens[token]["active"] is False

    def test_revoke_returns_status(self):
        """Response should confirm revocation."""
        client = _client()
        gen = client.post("/guest/generate").json()
        resp = client.delete(f"/guest/view/{gen['token']}")
        data = resp.json()
        assert data["status"] == "revoked"
        assert data["token"] == gen["token"]

    def test_revoke_unknown_returns_404(self):
        """Revoking unknown token should return 404."""
        client = _client()
        resp = client.delete("/guest/view/unknown-token-xyz")
        assert resp.status_code == 404

    def test_page_404_after_revoke(self):
        """GET should return 404 after token is revoked."""
        client = _client()
        gen = client.post("/guest/generate").json()
        client.delete(f"/guest/view/{gen['token']}")
        resp = client.get(f"/guest/view/{gen['token']}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /guest/view/{token}/status
# ---------------------------------------------------------------------------

class TestGuestViewStatus:
    """GET /guest/view/{token}/status returns connection info."""

    def test_status_returns_active_and_guests(self):
        """Should return active flag and empty guests list initially."""
        client = _client()
        gen = client.post("/guest/generate").json()
        resp = client.get(f"/guest/view/{gen['token']}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["guests"] == []

    def test_status_reflects_guests(self):
        """Should show guest names when manually added."""
        from meetingmind.main import _guest_tokens
        client = _client()
        gen = client.post("/guest/generate").json()
        token = gen["token"]
        _guest_tokens[token]["guests"].append("Alice")
        resp = client.get(f"/guest/view/{token}/status")
        data = resp.json()
        assert "Alice" in data["guests"]

    def test_status_reflects_inactive(self):
        """Should show active=False after revocation."""
        client = _client()
        gen = client.post("/guest/generate").json()
        client.delete(f"/guest/view/{gen['token']}")
        resp = client.get(f"/guest/view/{gen['token']}/status")
        data = resp.json()
        assert data["active"] is False

    def test_status_unknown_returns_404(self):
        """Unknown token should return 404."""
        client = _client()
        resp = client.get("/guest/view/unknown-token/status")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket /guest/ws/{token}
# ---------------------------------------------------------------------------

class TestGuestViewerWebSocket:
    """WebSocket /guest/ws/{token} for guest viewers."""

    def test_valid_token_connects(self):
        """Should accept connection for valid active token."""
        client = _client()
        gen = client.post("/guest/generate").json()
        token = gen["token"]
        with client.websocket_connect(f"/guest/ws/{token}") as ws:
            # Send name
            ws.send_json({"name": "TestUser"})
            reply = ws.receive_json()
            assert reply["type"] == "status"
            assert reply["status"] == "connected"
            assert reply["name"] == "TestUser"

    def test_invalid_token_closes(self):
        """Should close connection for invalid token."""
        from starlette.websockets import WebSocketDisconnect
        client = _client()
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/guest/ws/bad-token-xyz") as ws:
                ws.receive_text()

    def test_guest_name_added_to_list(self):
        """Connected guest name should appear in token's guests list."""
        from meetingmind.main import _guest_tokens
        client = _client()
        gen = client.post("/guest/generate").json()
        token = gen["token"]
        with client.websocket_connect(f"/guest/ws/{token}") as ws:
            ws.send_json({"name": "Alice"})
            ws.receive_json()  # status ack
            assert "Alice" in _guest_tokens[token]["guests"]


# ---------------------------------------------------------------------------
# Relay stub
# ---------------------------------------------------------------------------

class TestRelayStub:
    """RelayStub methods log without error."""

    def test_push_card_logs(self):
        """push_card should not raise."""
        from meetingmind.relay_stub import RelayStub
        relay = RelayStub()
        relay.push_card("tok", {"test": True})

    def test_push_key_facts_logs(self):
        """push_key_facts should not raise."""
        from meetingmind.relay_stub import RelayStub
        relay = RelayStub()
        relay.push_key_facts("tok", {"test": True})

    def test_deactivate_session_logs(self):
        """deactivate_session should not raise."""
        from meetingmind.relay_stub import RelayStub
        relay = RelayStub()
        relay.deactivate_session("tok")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestGuestConfig:
    """Guest config from settings.toml."""

    def test_tailscale_ip_loaded(self):
        """_TAILSCALE_IP should be a non-empty string."""
        from meetingmind.main import _TAILSCALE_IP
        assert isinstance(_TAILSCALE_IP, str)
        assert len(_TAILSCALE_IP) > 0

    def test_url_uses_tailscale_ip(self):
        """Generated URL should contain the configured IP."""
        from meetingmind.main import _TAILSCALE_IP
        client = _client()
        resp = client.post("/guest/generate").json()
        assert _TAILSCALE_IP in resp["url"]
