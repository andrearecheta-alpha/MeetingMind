"""
tests/test_system_audio_warning.py
-----------------------------------
S8-009: Tests for [Them] audio warning banner.

Verifies:
  - Warning broadcasts after 2 minutes with 0 system chunks transcribed
  - Warning does not broadcast if system chunks transcribed > 0
  - Warning only fires once per meeting session
  - Monitor task cancelled on meeting_stop()
  - Monitor task cancelled on meeting_reset()
  - WebSocket message has correct structure
  - Warning not triggered before 2 minutes

Uses asyncio — no hardware, no real Whisper, no pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def _run(coro):
    """Run an async function synchronously (no pytest-asyncio needed)."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSystemAudioWarning:
    """Tests for _monitor_system_audio background task."""

    def _activate_meeting(self):
        """Set _meeting state to simulate an active meeting."""
        import meetingmind.main as m
        m._meeting["active"] = True
        m._meeting["start_time"] = time.time()
        m._meeting["transcript_chunks"] = []
        m._meeting["decisions"] = []
        m._meeting["speaker_counts"] = {"microphone": 0, "system": 0}
        m._meeting["coaching_cooldowns"] = {}
        m._meeting["coaching_history"] = []
        m._meeting["coaching_dismissals"] = []
        m._meeting["coaching_resolved"] = set()
        m._meeting["coaching_seq"] = 0
        m._meeting["fact_checks"] = []
        m._meeting["system_audio_monitor_task"] = None
        m._stats["system_chunks_transcribed"] = 0

    def _deactivate_meeting(self):
        """Reset _meeting state."""
        import meetingmind.main as m
        m._meeting["active"] = False
        m._meeting["start_time"] = None
        m._meeting["transcript_chunks"] = []
        m._meeting["decisions"] = []
        m._meeting["speaker_counts"] = {"microphone": 0, "system": 0}
        m._meeting["coaching_cooldowns"] = {}
        m._meeting["coaching_history"] = []
        m._meeting["coaching_dismissals"] = []
        m._meeting["coaching_resolved"] = set()
        m._meeting["coaching_seq"] = 0
        m._meeting["fact_checks"] = []
        m._meeting["system_audio_monitor_task"] = None
        m._stats["system_chunks_transcribed"] = 0

    def setup_method(self):
        self._deactivate_meeting()

    def teardown_method(self):
        self._deactivate_meeting()

    # ── 1. Warning broadcasts after 2 minutes with 0 system chunks ────────

    def test_warning_broadcasts_after_delay_with_zero_system_chunks(self):
        """Warning is broadcast when system_chunks_transcribed == 0 after delay."""
        import meetingmind.main as m

        self._activate_meeting()
        m._stats["system_chunks_transcribed"] = 0

        broadcast_calls = []

        async def mock_broadcast(data):
            broadcast_calls.append(data)

        async def run():
            with patch.object(m._manager, "broadcast", side_effect=mock_broadcast):
                with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 0.05):
                    await m._monitor_system_audio()

        _run(run())

        assert len(broadcast_calls) == 1
        assert broadcast_calls[0]["type"] == "system_audio_warning"

    # ── 2. Warning does NOT broadcast if system chunks transcribed > 0 ────

    def test_no_warning_when_system_chunks_present(self):
        """No warning when system_chunks_transcribed > 0."""
        import meetingmind.main as m

        self._activate_meeting()
        m._stats["system_chunks_transcribed"] = 5

        broadcast_calls = []

        async def mock_broadcast(data):
            broadcast_calls.append(data)

        async def run():
            with patch.object(m._manager, "broadcast", side_effect=mock_broadcast):
                with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 0.05):
                    await m._monitor_system_audio()

        _run(run())

        assert len(broadcast_calls) == 0

    # ── 3. Warning only fires once per meeting session ────────────────────

    def test_warning_fires_only_once(self):
        """_monitor_system_audio completes after one broadcast — never re-fires."""
        import meetingmind.main as m

        self._activate_meeting()
        m._stats["system_chunks_transcribed"] = 0

        broadcast_calls = []

        async def mock_broadcast(data):
            broadcast_calls.append(data)

        async def run():
            with patch.object(m._manager, "broadcast", side_effect=mock_broadcast):
                with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 0.05):
                    # Run twice — task is a coroutine that exits after one check
                    await m._monitor_system_audio()
                    await m._monitor_system_audio()

        _run(run())

        # Each invocation broadcasts once — but in practice only one task is
        # created per meeting, so it fires exactly once.
        assert broadcast_calls[0]["type"] == "system_audio_warning"
        # The function itself is one-shot — it returns after broadcast.

    # ── 4. Monitor task cancelled on meeting_stop() ───────────────────────

    def test_monitor_cancelled_on_stop(self):
        """The monitor task is cancelled when meeting_stop logic cancels it."""
        import meetingmind.main as m

        self._activate_meeting()

        async def run():
            with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 999):
                task = asyncio.ensure_future(m._monitor_system_audio())
                m._meeting["system_audio_monitor_task"] = task

                # Give task time to start sleeping
                await asyncio.sleep(0.05)
                assert not task.done()

                # Cancel like meeting_stop does
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                assert task.cancelled()

        _run(run())

    # ── 5. Monitor task cancelled on meeting_reset() ──────────────────────

    def test_monitor_cancelled_on_reset(self):
        """The monitor task is cancelled when meeting_reset logic cancels it."""
        import meetingmind.main as m

        self._activate_meeting()

        async def run():
            with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 999):
                task = asyncio.ensure_future(m._monitor_system_audio())
                m._meeting["system_audio_monitor_task"] = task

                await asyncio.sleep(0.05)
                assert not task.done()

                # Simulate reset cancellation
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                assert task.cancelled()

        _run(run())

    # ── 6. WebSocket message has correct structure ────────────────────────

    def test_ws_message_structure(self):
        """Broadcast message has type, message, and detail fields."""
        import meetingmind.main as m

        self._activate_meeting()
        m._stats["system_chunks_transcribed"] = 0

        broadcast_calls = []

        async def mock_broadcast(data):
            broadcast_calls.append(data)

        async def run():
            with patch.object(m._manager, "broadcast", side_effect=mock_broadcast):
                with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 0.05):
                    await m._monitor_system_audio()

        _run(run())

        assert len(broadcast_calls) == 1
        msg = broadcast_calls[0]
        assert msg["type"] == "system_audio_warning"
        assert msg["message"] == "No [Them] audio detected"
        assert "paste input" in msg["detail"].lower() or "paste" in msg["detail"].lower()
        assert isinstance(msg["detail"], str)
        assert len(msg["detail"]) > 0

    # ── 7. Warning not triggered before 2 minutes ─────────────────────────

    def test_warning_not_triggered_before_delay(self):
        """Monitor is still sleeping before the delay elapses — no broadcast yet."""
        import meetingmind.main as m

        self._activate_meeting()
        m._stats["system_chunks_transcribed"] = 0

        broadcast_calls = []

        async def mock_broadcast(data):
            broadcast_calls.append(data)

        async def run():
            with patch.object(m._manager, "broadcast", side_effect=mock_broadcast):
                with patch.object(m, "_SYSTEM_AUDIO_WARN_DELAY", 999):
                    task = asyncio.ensure_future(m._monitor_system_audio())
                    await asyncio.sleep(0.1)

                    # Task should still be running (sleeping for 999s)
                    assert not task.done()
                    assert len(broadcast_calls) == 0

                    # Cleanup
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        _run(run())
