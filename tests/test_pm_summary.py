"""
tests/test_pm_summary.py
--------------------------
Tests for S8-005: PM Meeting Summary (_generate_pm_summary).

Uses mocked Claude API — no hardware, no real Whisper.
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
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Helpers ─────────────────────────────────────────────────────────────────

_VALID_PM_RESPONSE = {
    "overview": "2026-03-12 | 30 min | Project Phoenix | Modeling phase",
    "executive_summary": "Team discussed model selection and data pipeline issues.",
    "decisions": [
        {"text": "Use XGBoost for initial model", "owner": "Alice"},
    ],
    "action_items": [
        {
            "text": "Prepare training dataset",
            "owner": "Bob",
            "deadline": "Friday",
            "priority": "High",
        },
    ],
    "risks": [
        {
            "type": "scope",
            "description": "Adding NLP features not in original scope",
            "severity": "warning",
            "recommendation": "Create change request before proceeding",
        },
    ],
    "next_steps": [
        "Review change request with stakeholders",
        "Schedule data pipeline review",
    ],
}


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


# ── Tests ───────────────────────────────────────────────────────────────────

class TestGeneratePmSummary:
    """Tests for _generate_pm_summary() function."""

    def test_returns_correct_structure(self):
        """_generate_pm_summary returns all required keys."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_cls = _mock_claude_response(_VALID_PM_RESPONSE)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="We decided to use XGBoost for the model.",
                action_items=[{"text": "Prepare training dataset", "owner": "Bob", "deadline": "Friday"}],
                decisions=[{"text": "Use XGBoost", "made_by": "Alice"}],
                scope_alerts=[],
                timeline_alerts=[],
                project_name="Phoenix",
                cpmai_phase="Modeling",
                duration_minutes=30.0,
                speaker_counts={"microphone": 20, "system": 15},
                meeting_date="2026-03-12",
            ))

        assert result is not None
        assert "overview" in result
        assert "executive_summary" in result
        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert "action_items" in result
        assert isinstance(result["action_items"], list)
        assert "risks" in result
        assert isinstance(result["risks"], list)
        assert "next_steps" in result
        assert isinstance(result["next_steps"], list)

    def test_decision_has_owner(self):
        """PM summary decisions include owner field."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_cls = _mock_claude_response(_VALID_PM_RESPONSE)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="We decided to use XGBoost.",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result["decisions"][0]["owner"] == "Alice"

    def test_action_item_has_priority(self):
        """PM summary action items include priority field."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_cls = _mock_claude_response(_VALID_PM_RESPONSE)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Bob will prepare training dataset by Friday.",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result["action_items"][0]["priority"] == "High"

    def test_empty_transcript_returns_none(self):
        """Empty transcript returns None without API call."""
        from meetingmind.main import _generate_pm_summary

        result = _run(_generate_pm_summary(
            transcript="",
            action_items=[], decisions=[], scope_alerts=[],
            timeline_alerts=[], project_name=None, cpmai_phase=None,
            duration_minutes=0, speaker_counts={}, meeting_date="",
        ))
        assert result is None

    def test_whitespace_transcript_returns_none(self):
        """Whitespace-only transcript returns None without API call."""
        from meetingmind.main import _generate_pm_summary

        result = _run(_generate_pm_summary(
            transcript="   \n  \t  ",
            action_items=[], decisions=[], scope_alerts=[],
            timeline_alerts=[], project_name=None, cpmai_phase=None,
            duration_minutes=0, speaker_counts={}, meeting_date="",
        ))
        assert result is None

    def test_api_failure_returns_none(self):
        """Claude API failure returns None gracefully."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = Exception("API down")
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Some meeting transcript",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is None

    def test_api_key_unavailable_returns_none(self):
        """Missing API key returns None gracefully."""
        from meetingmind.main import _generate_pm_summary

        with patch("meetingmind._api_key.load_api_key", side_effect=Exception("No key")):
            result = _run(_generate_pm_summary(
                transcript="Some meeting transcript",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is None

    def test_invalid_json_returns_none(self):
        """Invalid JSON from Claude returns None."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_content = MagicMock()
        mock_content.text = "This is not valid JSON at all."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Some transcript",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is None

    def test_fenced_json_parsed(self):
        """Claude response wrapped in ```json ... ``` is parsed correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        fenced = '```json\n' + json.dumps(_VALID_PM_RESPONSE) + '\n```'
        mock_content = MagicMock()
        mock_content.text = fenced
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Meeting transcript here",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is not None
        assert result["overview"] == _VALID_PM_RESPONSE["overview"]

    def test_missing_keys_returns_none(self):
        """Response missing required keys returns None."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        incomplete = {"overview": "test", "executive_summary": "test"}
        mock_cls = _mock_claude_response(incomplete)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Some transcript",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is None

    def test_risks_include_type_and_severity(self):
        """PM summary risks include type and severity fields."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        mock_cls = _mock_claude_response(_VALID_PM_RESPONSE)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_generate_pm_summary(
                transcript="Adding NLP features was discussed.",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result["risks"][0]["type"] == "scope"
        assert result["risks"][0]["severity"] == "warning"
        assert result["risks"][0]["recommendation"] != ""


    def test_timeout_returns_none(self):
        """Claude API call that exceeds 30s returns None."""
        import anthropic as _anthropic
        from meetingmind.main import _generate_pm_summary

        def _slow_create(**kwargs):
            import time
            time.sleep(60)  # would block forever without timeout

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = _slow_create
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"), \
             patch("meetingmind.main.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = _run(_generate_pm_summary(
                transcript="Some meeting transcript",
                action_items=[], decisions=[], scope_alerts=[],
                timeline_alerts=[], project_name=None, cpmai_phase=None,
                duration_minutes=10.0, speaker_counts={}, meeting_date="2026-03-12",
            ))

        assert result is None


class TestSummaryEndpointPmSummary:
    """Tests that GET /meeting/summary includes pm_summary field."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def test_summary_includes_pm_summary_null(self):
        """Summary includes pm_summary=None before generation completes."""
        import meetingmind.main as m
        m._last_summary = {
            "meeting_id": "test_pm_001",
            "date": "2026-03-12",
            "stopped_at": "2026-03-12",
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
            "timeline_alerts": [],
            "pm_summary": None,
            "_transcript_text": "some text",
        }
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "pm_summary" in data
        assert data["pm_summary"] is None
        # Internal key should NOT be in response
        assert "_transcript_text" not in data

        # Cleanup
        m._last_summary = None

    def test_summary_includes_pm_summary_populated(self):
        """Summary includes pm_summary when generation completes."""
        import meetingmind.main as m
        m._last_summary = {
            "meeting_id": "test_pm_002",
            "date": "2026-03-12",
            "stopped_at": "2026-03-12",
            "duration_minutes": 30.0,
            "duration_seconds": 1800,
            "transcript_chunks": 50,
            "decisions": [],
            "action_items": [],
            "coaching_events": 0,
            "coaching_details": [],
            "speaker_counts": {"microphone": 20, "system": 15},
            "fact_checks": [],
            "fact_check_count": 0,
            "ai_action_items": [],
            "ai_decisions": [],
            "scope_alerts": [],
            "timeline_alerts": [],
            "pm_summary": _VALID_PM_RESPONSE,
            "_transcript_text": "some text",
        }
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pm_summary"] is not None
        assert data["pm_summary"]["overview"] == _VALID_PM_RESPONSE["overview"]
        assert len(data["pm_summary"]["decisions"]) == 1
        assert len(data["pm_summary"]["action_items"]) == 1
        assert len(data["pm_summary"]["risks"]) == 1
        assert len(data["pm_summary"]["next_steps"]) == 2
        # Internal key filtered
        assert "_transcript_text" not in data

        # Cleanup
        m._last_summary = None

    def test_summary_no_meeting(self):
        """GET /meeting/summary returns 404 when no meeting completed."""
        import meetingmind.main as m
        m._last_summary = None
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 404
