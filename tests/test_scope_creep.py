"""
tests/test_scope_creep.py
--------------------------
Tests for S8-003: Scope Creep Detection.

Uses FastAPI TestClient — no hardware, no real Whisper.
Mock Claude API for _detect_scope_creep() tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def _run(coro):
    """Run an async function synchronously (no pytest-asyncio needed)."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestScopeCreepDetection:
    """Tests for _detect_scope_creep() function."""

    @staticmethod
    def _mock_claude_response(response_json):
        """Create mock anthropic client returning the given JSON."""
        mock_content = MagicMock()
        mock_content.text = json.dumps(response_json)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)
        return mock_client_cls

    def test_detect_returns_alert_for_out_of_scope(self):
        """_detect_scope_creep returns out_of_scope alert."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "out_of_scope",
                    "severity": "critical",
                    "text": "Request for mobile app is outside project scope",
                    "source_quote": "Can we also build a mobile app?",
                    "matched_term": "mobile app",
                    "suggestion": "Defer to a separate project",
                    "confidence": 0.9,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("Can we also build a mobile app?"))

        assert len(result["scope_alerts"]) == 1
        assert result["scope_alerts"][0]["alert_type"] == "out_of_scope"
        assert result["scope_alerts"][0]["severity"] == "critical"

    def test_detect_returns_alert_for_creep_language(self):
        """_detect_scope_creep returns scope_creep_language alert."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "scope_creep_language",
                    "severity": "warning",
                    "text": "Informal scope expansion detected",
                    "source_quote": "While we're at it, let's add analytics",
                    "matched_term": "while we're at it",
                    "suggestion": "Log as a separate backlog item",
                    "confidence": 0.8,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("While we're at it, let's add analytics"))

        assert len(result["scope_alerts"]) == 1
        assert result["scope_alerts"][0]["alert_type"] == "scope_creep_language"

    def test_low_confidence_filtered(self):
        """Alerts below 0.65 confidence are filtered out."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "scope_creep_language",
                    "severity": "warning",
                    "text": "Maybe scope creep",
                    "source_quote": "perhaps we could",
                    "confidence": 0.4,
                },
                {
                    "alert_type": "out_of_scope",
                    "severity": "critical",
                    "text": "Definitely out of scope",
                    "source_quote": "build a new CRM",
                    "confidence": 0.9,
                },
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("test transcript"))

        assert len(result["scope_alerts"]) == 1  # only 0.9 item
        assert result["scope_alerts"][0]["text"] == "Definitely out of scope"

    def test_out_of_scope_alert_type(self):
        """Correct alert_type value for out_of_scope."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "out_of_scope",
                    "severity": "warning",
                    "text": "Feature request outside scope",
                    "source_quote": "let's add SSO support",
                    "confidence": 0.75,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("let's add SSO support"))

        assert result["scope_alerts"][0]["alert_type"] == "out_of_scope"

    def test_gold_plating_detected(self):
        """gold_plating alert type returned correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "gold_plating",
                    "severity": "warning",
                    "text": "Adding unrequested dark mode feature",
                    "source_quote": "I went ahead and added dark mode too",
                    "matched_term": "dark mode",
                    "suggestion": "Verify with stakeholders before including",
                    "confidence": 0.85,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("I went ahead and added dark mode too"))

        assert len(result["scope_alerts"]) == 1
        assert result["scope_alerts"][0]["alert_type"] == "gold_plating"

    def test_no_alerts_for_clean_transcript(self):
        """Empty list returned when no scope issues found."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {"scope_alerts": []}
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("The sprint is on track"))

        assert result["scope_alerts"] == []

    def test_critical_severity(self):
        """Critical severity assigned correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_scope_creep

        response = {
            "scope_alerts": [
                {
                    "alert_type": "out_of_scope",
                    "severity": "critical",
                    "text": "Major scope violation",
                    "source_quote": "Let's rebuild the whole backend",
                    "confidence": 0.95,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_scope_creep("Let's rebuild the whole backend"))

        assert result["scope_alerts"][0]["severity"] == "critical"

    def test_project_context_injected(self):
        """Verify scope items from active project appear in API call."""
        import anthropic as _anthropic
        import meetingmind.main as m
        from meetingmind.main import _detect_scope_creep

        m._active_project = {
            "project_name": "Phoenix",
            "cpmai_phase": "Deployment",
            "scope_items": ["API redesign", "Dashboard v2"],
        }

        response = {"scope_alerts": []}
        mock_content = MagicMock()
        mock_content.text = json.dumps(response)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            _run(_detect_scope_creep("some transcript"))

        # Check the user prompt sent to Claude includes scope items
        call_args = mock_client_instance.messages.create.call_args
        user_msg = call_args[1]["messages"][0]["content"] if call_args[1] else call_args[0][0]
        assert "Phoenix" in user_msg
        assert "API redesign" in user_msg
        assert "Dashboard v2" in user_msg

        m._active_project = None


class TestScopeAlertEndpoints:
    """Tests for scope alert REST endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_state(self):
        """Reset module-level state between tests."""
        import meetingmind.main as m
        m._scope_alerts.clear()

    def setup_method(self):
        self._reset_state()

    def teardown_method(self):
        self._reset_state()

    def test_get_scope_alerts_empty(self):
        """GET /meeting/scope-alerts returns empty list initially."""
        client = self._client()
        resp = client.get("/meeting/scope-alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope_alerts"] == []
        assert data["count"] == 0

    def test_delete_scope_alert(self):
        """DELETE /meeting/scope-alerts/{id} removes correct alert."""
        import meetingmind.main as m
        m._scope_alerts.extend([
            {"id": "scope-001", "text": "First alert"},
            {"id": "scope-002", "text": "Second alert"},
        ])
        client = self._client()
        resp = client.delete("/meeting/scope-alerts/scope-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert len(m._scope_alerts) == 1
        assert m._scope_alerts[0]["id"] == "scope-002"

    def test_delete_nonexistent_returns_404(self):
        """DELETE /meeting/scope-alerts/{id} returns 404 for unknown id."""
        client = self._client()
        resp = client.delete("/meeting/scope-alerts/nonexistent")
        assert resp.status_code == 404

    def test_meeting_start_resets_alerts(self):
        """_scope_alerts cleared on meeting start."""
        import meetingmind.main as m
        m._scope_alerts.append({"id": "old", "text": "stale alert"})
        client = self._client()
        resp = client.post("/meeting/reset")
        assert resp.status_code == 200
        assert len(m._scope_alerts) == 0

    def test_meeting_stop_includes_alerts(self):
        """Meeting summary contains scope_alerts."""
        import meetingmind.main as m
        m._last_summary = {
            "meeting_id": "test_scope",
            "date": "2026-03-09",
            "stopped_at": "2026-03-09",
            "duration_minutes": 5.0,
            "duration_seconds": 300,
            "transcript_chunks": 10,
            "decisions": [],
            "action_items": [],
            "coaching_events": 0,
            "coaching_details": [],
            "speaker_counts": {},
            "fact_checks": [],
            "fact_check_count": 0,
            "ai_action_items": [],
            "ai_decisions": [],
            "scope_alerts": [
                {"id": "sa-001", "text": "Scope issue", "severity": "warning"}
            ],
        }
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["scope_alerts"]) == 1
        assert data["scope_alerts"][0]["text"] == "Scope issue"
