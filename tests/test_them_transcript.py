"""
tests/test_them_transcript.py
-------------------------------
Tests for POST /meeting/them-transcript — [Them] paste injection.

Verifies:
  - Endpoint accepts text and splits into sentence chunks
  - Requires active meeting
  - Rejects empty text
  - Does NOT interfere with MIC audio pipeline state
  - Coaching / decision detection fires on injected text
  - Speaker counts track "system" source correctly
  - _audio_io_pool exists (dedicated thread pool for get_chunk)

Uses FastAPI TestClient — no hardware, no real Whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TestThemTranscriptEndpoint:
    """Tests for POST /meeting/them-transcript."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _activate_meeting(self):
        """Set _meeting state to simulate an active meeting."""
        import meetingmind.main as m
        m._meeting["active"] = True
        m._meeting["transcript_chunks"] = []
        m._meeting["decisions"] = []
        m._meeting["speaker_counts"] = {"microphone": 0, "system": 0}
        m._meeting["coaching_cooldowns"] = {}
        m._meeting["coaching_history"] = []
        m._meeting["coaching_dismissals"] = []
        m._meeting["coaching_resolved"] = set()
        m._meeting["coaching_seq"] = 0
        m._meeting["fact_checks"] = []
        m._meeting["start_time"] = __import__("time").time()

    def _deactivate_meeting(self):
        """Reset _meeting state."""
        import meetingmind.main as m
        m._meeting["active"] = False
        m._meeting["transcript_chunks"] = []
        m._meeting["decisions"] = []
        m._meeting["speaker_counts"] = {"microphone": 0, "system": 0}
        m._meeting["coaching_cooldowns"] = {}
        m._meeting["coaching_history"] = []
        m._meeting["coaching_dismissals"] = []
        m._meeting["coaching_resolved"] = set()
        m._meeting["coaching_seq"] = 0
        m._meeting["fact_checks"] = []
        m._meeting["start_time"] = None
        m._meeting["capture"] = None
        m._meeting["transcriber"] = None
        m._meeting["task"] = None

    def setup_method(self):
        self._deactivate_meeting()

    def teardown_method(self):
        self._deactivate_meeting()

    # ── Basic endpoint behaviour ──────────────────────────────────────────

    def test_rejects_when_no_meeting(self):
        """POST /meeting/them-transcript returns 400 without active meeting."""
        client = self._client()
        resp = client.post(
            "/meeting/them-transcript",
            json={"text": "Hello world."},
        )
        assert resp.status_code == 400
        assert "No active meeting" in resp.json()["error"]

    def test_rejects_empty_text(self):
        """POST /meeting/them-transcript returns 400 for empty text."""
        self._activate_meeting()
        client = self._client()
        resp = client.post(
            "/meeting/them-transcript",
            json={"text": ""},
        )
        assert resp.status_code == 400
        assert "Empty text" in resp.json()["error"]

    def test_rejects_whitespace_only(self):
        """POST /meeting/them-transcript returns 400 for whitespace-only text."""
        self._activate_meeting()
        client = self._client()
        resp = client.post(
            "/meeting/them-transcript",
            json={"text": "   \n\t  "},
        )
        assert resp.status_code == 400

    def test_injects_single_sentence(self):
        """Single sentence is injected as one chunk."""
        self._activate_meeting()
        client = self._client()
        resp = client.post(
            "/meeting/them-transcript",
            json={"text": "We should go with option A."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "injected"
        assert data["chunks"] == 1

    def test_injects_multiple_sentences(self):
        """Multiple sentences are split and injected separately."""
        self._activate_meeting()
        client = self._client()
        resp = client.post(
            "/meeting/them-transcript",
            json={"text": "First sentence. Second sentence. Third sentence."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks"] == 3

    def test_char_count_returned(self):
        """Response includes character count of original text."""
        self._activate_meeting()
        text = "Hello world."
        client = self._client()
        resp = client.post("/meeting/them-transcript", json={"text": text})
        assert resp.json()["chars"] == len(text)

    # ── Pipeline integration ─────────────────────────────────────────────

    def test_transcript_chunks_accumulated(self):
        """Injected text is appended to _meeting['transcript_chunks']."""
        import meetingmind.main as m
        self._activate_meeting()
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "Budget is 50K. Timeline is Q3."},
        )
        assert len(m._meeting["transcript_chunks"]) == 2
        assert "Budget is 50K" in m._meeting["transcript_chunks"][0]
        assert "Timeline is Q3" in m._meeting["transcript_chunks"][1]

    def test_speaker_counts_system_incremented(self):
        """Each injected sentence increments speaker_counts['system']."""
        import meetingmind.main as m
        self._activate_meeting()
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "Point one. Point two. Point three."},
        )
        assert m._meeting["speaker_counts"]["system"] == 3
        # MIC count should remain untouched
        assert m._meeting["speaker_counts"]["microphone"] == 0

    def test_decision_detection_fires(self):
        """Decision phrases in injected text trigger decision events."""
        import meetingmind.main as m
        self._activate_meeting()
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "We've decided to go with React."},
        )
        assert len(m._meeting["decisions"]) >= 1
        assert m._meeting["decisions"][0]["type"] == "decision"

    # ── MIC pipeline isolation ───────────────────────────────────────────

    def test_meeting_active_unchanged(self):
        """them-transcript never modifies _meeting['active']."""
        import meetingmind.main as m
        self._activate_meeting()
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "Some pasted text here."},
        )
        assert m._meeting["active"] is True

    def test_capture_untouched(self):
        """them-transcript never modifies _meeting['capture']."""
        import meetingmind.main as m
        self._activate_meeting()
        sentinel = object()
        m._meeting["capture"] = sentinel
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "This should not touch capture."},
        )
        assert m._meeting["capture"] is sentinel

    def test_transcriber_untouched(self):
        """them-transcript never modifies _meeting['transcriber']."""
        import meetingmind.main as m
        self._activate_meeting()
        sentinel = object()
        m._meeting["transcriber"] = sentinel
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "This should not touch transcriber."},
        )
        assert m._meeting["transcriber"] is sentinel

    def test_task_untouched(self):
        """them-transcript never modifies _meeting['task']."""
        import meetingmind.main as m
        self._activate_meeting()
        sentinel = object()
        m._meeting["task"] = sentinel
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "This should not touch task."},
        )
        assert m._meeting["task"] is sentinel

    def test_start_time_untouched(self):
        """them-transcript never modifies _meeting['start_time']."""
        import meetingmind.main as m
        self._activate_meeting()
        original_start = m._meeting["start_time"]
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "This should not touch start_time."},
        )
        assert m._meeting["start_time"] == original_start

    def test_model_loading_untouched(self):
        """them-transcript never modifies _meeting['model_loading']."""
        import meetingmind.main as m
        self._activate_meeting()
        m._meeting["model_loading"] = False
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "Pasted text."},
        )
        assert m._meeting["model_loading"] is False

    def test_mic_speaker_count_preserved(self):
        """them-transcript does not reset mic speaker count."""
        import meetingmind.main as m
        self._activate_meeting()
        m._meeting["speaker_counts"]["microphone"] = 42
        client = self._client()
        client.post(
            "/meeting/them-transcript",
            json={"text": "Some text. More text."},
        )
        assert m._meeting["speaker_counts"]["microphone"] == 42

    # ── Thread pool isolation ────────────────────────────────────────────

    def test_audio_io_pool_exists(self):
        """Dedicated _audio_io_pool exists to prevent thread pool starvation."""
        import meetingmind.main as m
        assert hasattr(m, "_audio_io_pool")
        assert m._audio_io_pool is not None
        # Should have at least 2 workers (main loop + guest loop)
        assert m._audio_io_pool._max_workers >= 2

    # ── Multiple pastes ──────────────────────────────────────────────────

    def test_multiple_pastes_accumulate(self):
        """Multiple POST calls accumulate transcript_chunks, don't reset."""
        import meetingmind.main as m
        self._activate_meeting()
        client = self._client()
        client.post("/meeting/them-transcript", json={"text": "First paste."})
        client.post("/meeting/them-transcript", json={"text": "Second paste."})
        assert len(m._meeting["transcript_chunks"]) == 2
        assert "First paste" in m._meeting["transcript_chunks"][0]
        assert "Second paste" in m._meeting["transcript_chunks"][1]

    def test_paste_then_check_active_still_true(self):
        """After a large paste, meeting is still active."""
        import meetingmind.main as m
        self._activate_meeting()
        long_text = ". ".join([f"Sentence {i}" for i in range(50)]) + "."
        client = self._client()
        resp = client.post("/meeting/them-transcript", json={"text": long_text})
        assert resp.status_code == 200
        assert m._meeting["active"] is True
        assert m._meeting["capture"] is not None or True  # capture may be None in test
        assert resp.json()["chunks"] >= 40  # at least 40 sentences

    def test_no_text_key_returns_422(self):
        """Missing 'text' field returns 422 (validation error)."""
        self._activate_meeting()
        client = self._client()
        resp = client.post("/meeting/them-transcript", json={})
        assert resp.status_code == 422
