"""
tests/test_action_items.py
---------------------------
Tests for S8-002: Action Item & Decision Capture.

Uses FastAPI TestClient — no hardware, no real Whisper.
Mock Claude API for _detect_action_items() tests.
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


class TestActionItemDetection:
    """Tests for _detect_action_items() function."""

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

    def test_detect_returns_action_items(self):
        """_detect_action_items returns action items from transcript."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        response = {
            "action_items": [
                {
                    "text": "Send the report to Sarah",
                    "owner": "John",
                    "deadline": "Friday",
                    "source_quote": "I'll send the report to Sarah by Friday",
                    "confidence": 0.9,
                }
            ],
            "decisions": [],
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("I'll send the report to Sarah by Friday"))

        assert len(result["action_items"]) == 1
        assert result["action_items"][0]["text"] == "Send the report to Sarah"
        assert result["action_items"][0]["owner"] == "John"
        assert result["action_items"][0]["deadline"] == "Friday"

    def test_detect_returns_decisions(self):
        """_detect_action_items returns decisions from transcript."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        response = {
            "action_items": [],
            "decisions": [
                {
                    "text": "Use PostgreSQL for the database",
                    "made_by": "Alice",
                    "source_quote": "We've decided to go with PostgreSQL",
                    "confidence": 0.85,
                }
            ],
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("We've decided to go with PostgreSQL"))

        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["text"] == "Use PostgreSQL for the database"
        assert result["decisions"][0]["made_by"] == "Alice"

    def test_low_confidence_filtered(self):
        """Items below 0.6 confidence are filtered out."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        response = {
            "action_items": [
                {
                    "text": "Maybe update the docs",
                    "owner": None,
                    "deadline": None,
                    "source_quote": "maybe update the docs",
                    "confidence": 0.3,
                },
                {
                    "text": "Deploy to staging",
                    "owner": "Bob",
                    "deadline": "Monday",
                    "source_quote": "Bob will deploy to staging by Monday",
                    "confidence": 0.9,
                },
            ],
            "decisions": [
                {
                    "text": "Possibly use React",
                    "made_by": None,
                    "source_quote": "might use React",
                    "confidence": 0.4,
                },
            ],
        }
        mock_cls = self._mock_claude_response(response)
        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("test transcript"))

        assert len(result["action_items"]) == 1  # only Bob's item (0.9)
        assert result["action_items"][0]["owner"] == "Bob"
        assert len(result["decisions"]) == 0  # React item filtered (0.4)

    def test_empty_transcript_returns_empty(self):
        """Empty transcript returns empty lists without API call."""
        from meetingmind.main import _detect_action_items

        result = _run(_detect_action_items(""))
        assert result == {"action_items": [], "decisions": []}

    def test_api_failure_returns_empty(self):
        """Claude API failure returns empty lists gracefully."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.side_effect = Exception("API down")
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("some transcript text"))

        assert result == {"action_items": [], "decisions": []}

    def test_fenced_json_response(self):
        """Claude response wrapped in ```json ... ``` is parsed correctly."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        fenced = '```json\n{"action_items": [{"text": "Fix bug", "owner": "Zara", "deadline": null, "source_quote": "fix the bug", "confidence": 0.7}], "decisions": []}\n```'
        mock_content = MagicMock()
        mock_content.text = fenced
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("fix the bug"))

        assert len(result["action_items"]) == 1
        assert result["action_items"][0]["owner"] == "Zara"

    def test_invalid_json_returns_empty(self):
        """Invalid JSON from Claude returns empty lists."""
        import anthropic as _anthropic
        from meetingmind.main import _detect_action_items

        mock_content = MagicMock()
        mock_content.text = "This is not valid JSON at all."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            result = _run(_detect_action_items("some transcript"))

        assert result == {"action_items": [], "decisions": []}


class TestActionItemEndpoints:
    """Tests for action item & decision REST endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_state(self):
        """Reset module-level state between tests."""
        import meetingmind.main as m
        m._action_items.clear()
        m._decisions_ai.clear()

    def setup_method(self):
        self._reset_state()

    def teardown_method(self):
        self._reset_state()

    # ── GET endpoints return empty initially ──────────────────────────────

    def test_get_action_items_empty(self):
        """GET /meeting/action-items returns empty list initially."""
        client = self._client()
        resp = client.get("/meeting/action-items")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_items"] == []
        assert data["count"] == 0

    def test_get_decisions_empty(self):
        """GET /meeting/ai-decisions returns empty list initially."""
        client = self._client()
        resp = client.get("/meeting/ai-decisions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions"] == []
        assert data["count"] == 0

    # ── GET endpoints return items when populated ─────────────────────────

    def test_get_action_items_populated(self):
        """GET /meeting/action-items returns items when populated."""
        import meetingmind.main as m
        m._action_items.append({
            "id": "test-ai-001",
            "text": "Write tests",
            "owner": "Alice",
            "deadline": "Friday",
            "source_quote": "Alice will write tests by Friday",
            "cpmai_phase": None,
            "project_name": None,
            "timestamp": "2026-03-09T00:00:00+00:00",
            "confidence": 0.9,
        })
        client = self._client()
        resp = client.get("/meeting/action-items")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["action_items"][0]["text"] == "Write tests"
        assert data["action_items"][0]["owner"] == "Alice"

    def test_get_decisions_populated(self):
        """GET /meeting/ai-decisions returns items when populated."""
        import meetingmind.main as m
        m._decisions_ai.append({
            "id": "test-dec-001",
            "text": "Use Python 3.12",
            "made_by": "Bob",
            "source_quote": "We agreed to use Python 3.12",
            "cpmai_phase": "Modeling",
            "project_name": "Test Project",
            "timestamp": "2026-03-09T00:00:00+00:00",
            "confidence": 0.85,
        })
        client = self._client()
        resp = client.get("/meeting/ai-decisions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["decisions"][0]["text"] == "Use Python 3.12"
        assert data["decisions"][0]["made_by"] == "Bob"
        assert data["decisions"][0]["cpmai_phase"] == "Modeling"

    # ── DELETE endpoints ──────────────────────────────────────────────────

    def test_delete_action_item(self):
        """DELETE /meeting/action-items/{id} removes correct item."""
        import meetingmind.main as m
        m._action_items.extend([
            {"id": "ai-001", "text": "First"},
            {"id": "ai-002", "text": "Second"},
        ])
        client = self._client()
        resp = client.delete("/meeting/action-items/ai-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert len(m._action_items) == 1
        assert m._action_items[0]["id"] == "ai-002"

    def test_delete_decision(self):
        """DELETE /meeting/ai-decisions/{id} removes correct item."""
        import meetingmind.main as m
        m._decisions_ai.extend([
            {"id": "dec-001", "text": "First decision"},
            {"id": "dec-002", "text": "Second decision"},
        ])
        client = self._client()
        resp = client.delete("/meeting/ai-decisions/dec-002")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert len(m._decisions_ai) == 1
        assert m._decisions_ai[0]["id"] == "dec-001"

    def test_delete_action_item_not_found(self):
        """DELETE /meeting/action-items/{id} returns 404 for unknown id."""
        client = self._client()
        resp = client.delete("/meeting/action-items/nonexistent")
        assert resp.status_code == 404

    def test_delete_decision_not_found(self):
        """DELETE /meeting/ai-decisions/{id} returns 404 for unknown id."""
        client = self._client()
        resp = client.delete("/meeting/ai-decisions/nonexistent")
        assert resp.status_code == 404

    # ── Summary includes AI items ─────────────────────────────────────────

    def test_summary_includes_ai_items(self):
        """Meeting summary includes ai_action_items and ai_decisions."""
        import meetingmind.main as m
        m._last_summary = {
            "meeting_id": "test_001",
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
            "ai_action_items": [
                {"id": "ai-sum-001", "text": "Review PR", "owner": "Carol"}
            ],
            "ai_decisions": [
                {"id": "dec-sum-001", "text": "Adopt TypeScript", "made_by": "Dan"}
            ],
        }
        client = self._client()
        resp = client.get("/meeting/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ai_action_items"]) == 1
        assert data["ai_action_items"][0]["text"] == "Review PR"
        assert len(data["ai_decisions"]) == 1
        assert data["ai_decisions"][0]["text"] == "Adopt TypeScript"

    # ── Project context enrichment ────────────────────────────────────────

    def test_enrichment_with_project(self):
        """Items are enriched with cpmai_phase and project_name from active project."""
        import anthropic as _anthropic
        import meetingmind.main as m

        m._active_project = {
            "project_name": "Phoenix",
            "cpmai_phase": "Modeling",
        }
        m._action_items.clear()
        m._decisions_ai.clear()

        response = {
            "action_items": [
                {
                    "text": "Train the model",
                    "owner": "Eve",
                    "deadline": None,
                    "source_quote": "Eve will train the model",
                    "confidence": 0.8,
                }
            ],
            "decisions": [
                {
                    "text": "Use XGBoost",
                    "made_by": "Eve",
                    "source_quote": "We decided on XGBoost",
                    "confidence": 0.9,
                }
            ],
        }
        mock_content = MagicMock()
        mock_content.text = json.dumps(response)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_cls = MagicMock(return_value=mock_client_instance)

        with patch.object(_anthropic, "Anthropic", mock_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"), \
             patch.object(m._manager, "broadcast"):
            _run(m._process_ai_detection("Eve will train the model", 1000.0))

        assert len(m._action_items) == 1
        assert m._action_items[0]["cpmai_phase"] == "Modeling"
        assert m._action_items[0]["project_name"] == "Phoenix"
        assert len(m._decisions_ai) == 1
        assert m._decisions_ai[0]["cpmai_phase"] == "Modeling"
        assert m._decisions_ai[0]["project_name"] == "Phoenix"

        # Cleanup
        m._active_project = None
        m._action_items.clear()
        m._decisions_ai.clear()

    # ── Reset on meeting lifecycle ────────────────────────────────────────

    def test_reset_clears_items(self):
        """POST /meeting/reset clears action items and decisions."""
        import meetingmind.main as m
        m._action_items.append({"id": "test", "text": "x"})
        m._decisions_ai.append({"id": "test", "text": "y"})
        client = self._client()
        resp = client.post("/meeting/reset")
        assert resp.status_code == 200
        assert len(m._action_items) == 0
        assert len(m._decisions_ai) == 0
