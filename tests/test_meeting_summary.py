"""
tests/test_meeting_summary.py
------------------------------
Tests for the meeting summary endpoint and action item extraction.

Run with:
    cd MeetingMind
    pytest tests/test_meeting_summary.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


# ===========================================================================
# Action item extraction tests
# ===========================================================================

class TestExtractActionItems:
    """Tests for extract_action_items() in suggestion_engine.py."""

    def test_finds_ill_pattern(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["I'll send the report by Friday"]
        result = extract_action_items(chunks)
        assert len(result) == 1
        assert result[0]["pattern"] == "i'll"
        assert result[0]["chunk_index"] == 0

    def test_finds_we_need_to(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["We need to update the backlog before sprint planning"]
        result = extract_action_items(chunks)
        assert len(result) == 1
        assert result[0]["pattern"] == "we need to"

    def test_finds_follow_up(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["Let's follow up on the vendor proposal next week"]
        result = extract_action_items(chunks)
        assert len(result) == 1
        assert result[0]["pattern"] == "follow up on"

    def test_finds_make_sure(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["Make sure to test the staging environment"]
        result = extract_action_items(chunks)
        assert len(result) == 1
        assert result[0]["pattern"] == "make sure to"

    def test_finds_will_send(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["John will send the budget spreadsheet tomorrow"]
        result = extract_action_items(chunks)
        assert len(result) == 1
        assert result[0]["pattern"] == "will send"

    def test_no_action_in_generic_text(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = [
            "The quarterly results look good",
            "Revenue increased by 15 percent",
            "Market conditions are stable",
        ]
        result = extract_action_items(chunks)
        assert result == []

    def test_empty_chunks(self):
        from meetingmind.suggestion_engine import extract_action_items
        result = extract_action_items([])
        assert result == []

    def test_deduplicates_identical_chunks(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = [
            "I'll call the client",
            "I'll call the client",
        ]
        result = extract_action_items(chunks)
        assert len(result) == 1

    def test_multiple_actions_across_chunks(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = [
            "I'll prepare the presentation",
            "The weather is nice today",
            "We need to review the contract",
            "Good meeting everyone",
        ]
        result = extract_action_items(chunks)
        assert len(result) == 2
        assert result[0]["chunk_index"] == 0
        assert result[1]["chunk_index"] == 2

    def test_preserves_original_case(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["I'll Send The Report"]
        result = extract_action_items(chunks)
        assert result[0]["text"] == "I'll Send The Report"

    def test_case_insensitive_matching(self):
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["WE NEED TO fix the production issue"]
        result = extract_action_items(chunks)
        assert len(result) == 1

    def test_one_match_per_chunk(self):
        """Even if a chunk has multiple patterns, only one match per chunk."""
        from meetingmind.suggestion_engine import extract_action_items
        chunks = ["I'll follow up on the issue and will send the notes"]
        result = extract_action_items(chunks)
        assert len(result) == 1


class TestActionPatterns:
    """Tests for _ACTION_PATTERNS list integrity."""

    def test_patterns_exist(self):
        from meetingmind.suggestion_engine import _ACTION_PATTERNS
        assert isinstance(_ACTION_PATTERNS, list)
        assert len(_ACTION_PATTERNS) >= 10

    def test_all_patterns_are_lowercase(self):
        from meetingmind.suggestion_engine import _ACTION_PATTERNS
        for p in _ACTION_PATTERNS:
            assert p == p.lower(), f"Pattern not lowercase: {p!r}"


# ===========================================================================
# Meeting summary endpoint tests
# ===========================================================================

class TestSummaryEndpoint:
    """Tests for GET /meeting/summary."""

    def test_no_summary_returns_404(self):
        """GET /meeting/summary with no completed meeting returns 404."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient

        main_mod._last_summary = None
        client = TestClient(main_mod.app)
        response = client.get("/meeting/summary")
        assert response.status_code == 404
        assert "error" in response.json()

    def test_returns_summary_when_available(self):
        """GET /meeting/summary returns stored summary data."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient

        main_mod._last_summary = {
            "meeting_id":        "20260228_120000",
            "date":              "2026-02-28T12:00:00+00:00",
            "stopped_at":        "2026-02-28T12:30:00+00:00",
            "duration_minutes":  30.0,
            "duration_seconds":  1800.0,
            "transcript_chunks": 25,
            "decisions":         [{"text": "We agreed on plan A", "phrase": "agreed"}],
            "action_items":      [{"text": "I'll send the notes", "pattern": "i'll", "chunk_index": 3}],
            "coaching_events":   2,
            "coaching_details":  [
                {"trigger_id": "no_decision_owner", "prompt": "Who owns this?", "timestamp": 1.0},
            ],
            "speaker_counts":    {"microphone": 15, "system": 10},
        }
        try:
            client = TestClient(main_mod.app)
            response = client.get("/meeting/summary")
            assert response.status_code == 200
            data = response.json()
            assert data["meeting_id"] == "20260228_120000"
            assert data["duration_minutes"] == 30.0
            assert len(data["decisions"]) == 1
            assert len(data["action_items"]) == 1
            assert data["coaching_events"] == 2
            assert data["speaker_counts"]["microphone"] == 15
        finally:
            main_mod._last_summary = None

    def test_summary_schema_keys(self):
        """Summary response must include all required keys."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient

        main_mod._last_summary = {
            "meeting_id":        "test",
            "date":              "2026-02-28T12:00:00+00:00",
            "stopped_at":        "2026-02-28T12:30:00+00:00",
            "duration_minutes":  5.0,
            "duration_seconds":  300.0,
            "transcript_chunks": 0,
            "decisions":         [],
            "action_items":      [],
            "coaching_events":   0,
            "coaching_details":  [],
            "speaker_counts":    {},
        }
        try:
            client = TestClient(main_mod.app)
            data = client.get("/meeting/summary").json()
            for key in (
                "meeting_id", "date", "stopped_at", "duration_minutes",
                "duration_seconds", "transcript_chunks", "decisions",
                "action_items", "coaching_events", "coaching_details",
                "speaker_counts",
            ):
                assert key in data, f"Missing key: {key}"
        finally:
            main_mod._last_summary = None


# ---------------------------------------------------------------------------
# Role endpoint tests
# ---------------------------------------------------------------------------

class TestRoleEndpoint:
    """Test POST/GET /settings/role endpoints."""

    def test_set_role_pm(self):
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        resp = client.post("/settings/role", json={"role": "PM"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "PM"

    def test_set_role_ea(self):
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        resp = client.post("/settings/role", json={"role": "EA"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "EA"

    def test_get_role(self):
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        client.post("/settings/role", json={"role": "Sales"})
        resp = client.get("/settings/role")
        assert resp.status_code == 200
        assert resp.json()["role"] == "Sales"

    def test_invalid_role(self):
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        resp = client.post("/settings/role", json={"role": "InvalidRole"})
        assert resp.status_code == 400

    def test_role_persists_after_reset(self):
        """Simulate localStorage clear: role survives meeting/reset."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        client.post("/settings/role", json={"role": "EA"})
        client.post("/meeting/reset")
        resp = client.get("/settings/role")
        assert resp.json()["role"] == "EA"

    def test_role_survives_multiple_gets(self):
        """After setting role, multiple GETs return the same value."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        client.post("/settings/role", json={"role": "Custom"})
        for _ in range(3):
            resp = client.get("/settings/role")
            assert resp.json()["role"] == "Custom"

    def test_role_overwrite(self):
        """Changing role replaces the previous value."""
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        client.post("/settings/role", json={"role": "PM"})
        client.post("/settings/role", json={"role": "Sales"})
        resp = client.get("/settings/role")
        assert resp.json()["role"] == "Sales"


# ---------------------------------------------------------------------------
# Recalibration & weak-signal config sanity checks
# ---------------------------------------------------------------------------

class TestRecalibrationConfig:
    """Verify the auto-recalibrate and weak-signal constants exist and are sane."""

    def test_weak_rms_threshold_exists(self):
        import meetingmind.main as main_mod
        assert hasattr(main_mod, "_WEAK_RMS_THRESHOLD")
        assert 0 < main_mod._WEAK_RMS_THRESHOLD < 0.01

    def test_weak_rms_consecutive_exists(self):
        import meetingmind.main as main_mod
        assert main_mod._WEAK_RMS_CONSECUTIVE >= 3

    def test_recalib_window_exists(self):
        import meetingmind.main as main_mod
        assert main_mod._RECALIB_WINDOW >= 10

    def test_recalib_silent_ratio(self):
        import meetingmind.main as main_mod
        assert 0.3 <= main_mod._RECALIB_SILENT_RATIO <= 0.8

    def test_recalib_factor(self):
        import meetingmind.main as main_mod
        assert 0 < main_mod._RECALIB_FACTOR < 1.0

    def test_debug_endpoint_has_gate_breakdown(self):
        import meetingmind.main as main_mod
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app)
        resp = client.get("/debug")
        assert resp.status_code == 200
        data = resp.json()
        assert "gate_breakdown" in data
        assert "silent" in data["gate_breakdown"]
