"""
tests/test_timeline_alerts.py
------------------------------
Tests for S8-004: Timeline & Delay Flag Detection.

Uses FastAPI TestClient — no hardware, no real Whisper.
Mock Claude API for _detect_timeline_risk() tests.
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


class TestTimelineRiskDetection:
    """Tests for _detect_timeline_risk() function."""

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

    def test_detect_returns_alert_for_delay_language(self):
        """_detect_timeline_risk returns delay_language alert."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "delay_language",
                    "severity": "warning",
                    "text": "Team reports running behind schedule",
                    "source_quote": "we're running behind on the API integration",
                    "affected_milestone": None,
                    "days_at_risk": 5,
                    "suggestion": "Request updated estimate and identify blockers",
                    "confidence": 0.85,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("we're running behind on the API integration"))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["alert_type"] == "delay_language"
        assert result["timeline_alerts"][0]["severity"] == "warning"

    def test_detect_returns_alert_for_optimism_bias(self):
        """_detect_timeline_risk returns optimism_bias alert."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "optimism_bias",
                    "severity": "warning",
                    "text": "Unrealistic recovery plan proposed",
                    "source_quote": "we can make up the time by working weekends",
                    "affected_milestone": None,
                    "days_at_risk": None,
                    "suggestion": "Assess realistic capacity before committing to overtime",
                    "confidence": 0.80,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("we can make up the time by working weekends"))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["alert_type"] == "optimism_bias"

    def test_detect_returns_alert_for_dependency_blocker(self):
        """_detect_timeline_risk returns dependency_blocker alert."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "dependency_blocker",
                    "severity": "critical",
                    "text": "External team dependency blocking progress",
                    "source_quote": "we're still waiting for the design team to deliver the mockups",
                    "affected_milestone": "UI Complete",
                    "days_at_risk": 10,
                    "suggestion": "Escalate to design team lead and set a hard deadline",
                    "confidence": 0.90,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk(
                "we're still waiting for the design team to deliver the mockups"
            ))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["alert_type"] == "dependency_blocker"
        assert result["timeline_alerts"][0]["severity"] == "critical"

    def test_low_confidence_filtered(self):
        """Alerts below 0.65 confidence are filtered out."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "delay_language",
                    "severity": "warning",
                    "text": "Maybe a delay",
                    "source_quote": "it might take a bit longer",
                    "confidence": 0.4,
                },
                {
                    "alert_type": "dependency_blocker",
                    "severity": "critical",
                    "text": "Definitely blocked",
                    "source_quote": "blocked by infrastructure team",
                    "confidence": 0.9,
                },
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("test transcript"))

        assert len(result["timeline_alerts"]) == 1  # only 0.9 item
        assert result["timeline_alerts"][0]["text"] == "Definitely blocked"

    def test_milestone_risk_detected(self):
        """milestone_risk alert type returned correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "milestone_risk",
                    "severity": "critical",
                    "text": "Beta release milestone at risk",
                    "source_quote": "the beta release is not going to make it by March",
                    "affected_milestone": "Beta Release",
                    "days_at_risk": 14,
                    "suggestion": "Reduce beta scope or adjust timeline",
                    "confidence": 0.92,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk(
                "the beta release is not going to make it by March"
            ))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["alert_type"] == "milestone_risk"
        assert result["timeline_alerts"][0]["affected_milestone"] == "Beta Release"

    def test_schedule_compression_detected(self):
        """schedule_compression alert type returned correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "schedule_compression",
                    "severity": "critical",
                    "text": "Dangerous shortcut proposed — skipping UAT",
                    "source_quote": "let's just skip UAT and go straight to production",
                    "affected_milestone": None,
                    "days_at_risk": None,
                    "suggestion": "UAT is critical for quality — negotiate reduced scope instead",
                    "confidence": 0.95,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk(
                "let's just skip UAT and go straight to production"
            ))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["alert_type"] == "schedule_compression"

    def test_no_alerts_for_clean_transcript(self):
        """Empty list returned when no timeline issues found."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {"timeline_alerts": []}
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("The sprint is on track and all milestones are green"))

        assert result["timeline_alerts"] == []

    def test_empty_transcript_returns_empty(self):
        """Empty transcript returns empty alerts without API call."""
        from meetingmind.main import _detect_timeline_risk

        result = _run(_detect_timeline_risk(""))
        assert result["timeline_alerts"] == []

    def test_project_context_injected(self):
        """Verify milestones from active project appear in API call."""
        import anthropic as _anthropic
        import meetingmind.main as m
        from meetingmind.main import _detect_timeline_risk

        m._active_project = {
            "project_name": "Phoenix",
            "cpmai_phase": "Deployment",
            "start_date": "2026-01-15",
            "end_date": "2026-06-30",
            "milestones": [
                {"name": "Beta Release", "date": "2026-03-15"},
                {"name": "Go-Live", "date": "2026-06-30"},
            ],
            "success_criteria": "All features deployed to production",
            "scope_items": [],
        }

        response = {"timeline_alerts": []}
        mock_content = MagicMock()
        mock_content.text = json.dumps(response)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            _run(_detect_timeline_risk("some transcript"))

        # Check the user prompt sent to Claude includes project context
        call_args = mock_client_instance.messages.create.call_args
        user_msg = call_args[1]["messages"][0]["content"] if call_args[1] else call_args[0][0]
        assert "Phoenix" in user_msg
        assert "Beta Release" in user_msg
        assert "Go-Live" in user_msg
        assert "2026-06-30" in user_msg

        m._active_project = None

    def test_critical_severity(self):
        """Critical severity assigned correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        response = {
            "timeline_alerts": [
                {
                    "alert_type": "delay_language",
                    "severity": "critical",
                    "text": "Project deadline will be missed",
                    "source_quote": "we are not going to make the June deadline",
                    "confidence": 0.95,
                }
            ]
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("we are not going to make the June deadline"))

        assert result["timeline_alerts"][0]["severity"] == "critical"

    def test_markdown_fenced_json_handled(self):
        """Claude response wrapped in markdown fences is parsed correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_timeline_risk

        fenced_json = '```json\n{"timeline_alerts": [{"alert_type": "delay_language", "severity": "warning", "text": "Delay detected", "source_quote": "running behind", "confidence": 0.8}]}\n```'
        mock_content = MagicMock()
        mock_content.text = fenced_json
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_timeline_risk("running behind"))

        assert len(result["timeline_alerts"]) == 1
        assert result["timeline_alerts"][0]["text"] == "Delay detected"


class TestTimelineAlertEndpoints:
    """Tests for timeline alert REST endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_state(self):
        """Reset module-level state between tests."""
        import meetingmind.main as m
        m._timeline_alerts.clear()

    def setup_method(self):
        self._reset_state()

    def teardown_method(self):
        self._reset_state()

    def test_get_timeline_alerts_empty(self):
        """GET /meeting/timeline-alerts returns empty list initially."""
        client = self._client()
        resp = client.get("/meeting/timeline-alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timeline_alerts"] == []
        assert data["count"] == 0

    def test_get_timeline_alerts_with_data(self):
        """GET /meeting/timeline-alerts returns populated list."""
        import meetingmind.main as m
        m._timeline_alerts.append({
            "id": "tl-001",
            "alert_type": "delay_language",
            "severity": "warning",
            "text": "Running behind",
        })
        client = self._client()
        resp = client.get("/meeting/timeline-alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["timeline_alerts"][0]["id"] == "tl-001"

    def test_delete_timeline_alert(self):
        """DELETE /meeting/timeline-alerts/{id} removes correct alert."""
        import meetingmind.main as m
        m._timeline_alerts.extend([
            {"id": "tl-001", "text": "First alert"},
            {"id": "tl-002", "text": "Second alert"},
        ])
        client = self._client()
        resp = client.delete("/meeting/timeline-alerts/tl-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert len(m._timeline_alerts) == 1
        assert m._timeline_alerts[0]["id"] == "tl-002"

    def test_delete_nonexistent_returns_404(self):
        """DELETE /meeting/timeline-alerts/{id} returns 404 for unknown id."""
        client = self._client()
        resp = client.delete("/meeting/timeline-alerts/nonexistent")
        assert resp.status_code == 404

    def test_meeting_reset_clears_alerts(self):
        """_timeline_alerts cleared on meeting reset."""
        import meetingmind.main as m
        m._timeline_alerts.append({"id": "old", "text": "stale alert"})
        client = self._client()
        resp = client.post("/meeting/reset")
        assert resp.status_code == 200
        assert len(m._timeline_alerts) == 0

    def test_meeting_summary_includes_timeline_alerts(self):
        """Meeting summary contains timeline_alerts."""
        import meetingmind.main as m
        m._last_summary = {
            "meeting_id": "test_timeline",
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
            "scope_alerts": [],
            "timeline_alerts": [
                {"id": "tl-001", "text": "Delay risk", "severity": "warning",
                 "affected_milestone": "Beta Release"}
            ],
        }
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["timeline_alerts"]) == 1
        assert data["timeline_alerts"][0]["text"] == "Delay risk"
        assert data["timeline_alerts"][0]["affected_milestone"] == "Beta Release"
