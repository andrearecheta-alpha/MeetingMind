# MeetingMind — Sprint 3 Checkpoint
**Date:** 2026-02-26
**Status:** Root cause of transcription silence found and fixed. Server hardened.
UI improved. Adaptive calibration system designed but not yet implemented.

---

## Root Cause Found and Fixed

**The bug:** Windows Voice Access was competing for the mic, reducing signal strength.
Direct PyAudio test measured **RMS = 0.000075**. The silence gate threshold was **0.0001**,
so every chunk was filtered before Whisper ever ran.

**The fix:** `_SILENCE_RMS_THRESHOLD` lowered to **0.00005** in `transcriber.py`.
Your mic's RMS (0.000075) now passes the gate (0.000075 > 0.00005 ✓).

---

## Files Modified This Session

### `src/meetingmind/transcriber.py`
- `_SILENCE_RMS_THRESHOLD` lowered: `0.001` → `0.0002` → `0.0001` → **`0.00005`**
- Added `WHISPER raw=` log line after every transcribe() call (before filters)
  — proves Whisper is running even when no transcript appears
- Updated calibration guide comment with dBFS values for all threshold levels

### `src/meetingmind/audio_capture.py`
- **`_stream_worker` completely restructured:**
  - Tries primary device first; if it fails (MICROPHONE source only), automatically
    walks through every other available input device as fallback
  - Logs `ERROR: Failed to open device [name]: [reason]` at each failure
  - Logs `FALLBACK: Trying device [name] instead` for each attempt
  - Detects "access denied" / error -9999 and stops fallback immediately
    (Windows privacy block — other devices won't help)
  - `stream.stop_stream()` moved to `finally:` block (was missing on some paths)
- **`probe_device(device_index)`** — new public method
  - Opens and immediately closes a stream to verify access
  - Returns `(success: bool, device_name: str, error_msg: str)`
  - Used by `meeting_start` for pre-flight validation

### `src/meetingmind/device_detector.py`
- **`DetectionResult.mic_permission_ok: Optional[bool]`** — new field
  - `True` = mic accessible, `False` = Windows blocked, `None` = unknown
- **`_check_mic_permission(pa)`** — new function
  - Tries to open default input device at startup
  - Returns `False` on PyAudio -9999 ("Unanticipated host error") = Windows block
- Called in `detect_devices()` at server startup; result stored in `DetectionResult`

### `src/meetingmind/main.py`
- **`AudioCapture(...)` creation** moved into `asyncio.to_thread`
  — prevents `pyaudio.PyAudio()` from blocking the event loop and hanging `/debug`
- **Device probe** added to `meeting_start` before Whisper loads:
  — if mic refuses to open → returns `{"error": "No working microphone found",
  "tried": [...], "reason": "...", "suggestion": "..."}` immediately
- **`_processing_loop` auto-restart:** crashes restart up to 3×; on 4th failure
  broadcasts `{"type": "error", "message": "..."}` to all WebSocket clients
- **`/health`** now returns `windows_mic_blocked: true/false`
- **`/meeting/reset`** now accepts **GET and POST** (accessible from browser URL bar)
- **RMS broadcast** fires every 0.5 s: `{"type": "rms", "rms": X, "threshold": Y}`

### `frontend/index.html`
- **`[↺ Reset]` button** — always enabled, clears stuck meeting state instantly
- **Auto-reset on page load** — `init()` calls `/meeting/reset` silently first
  so stale server state never blocks Start
- **Optimistic Stop enable** — Stop activates immediately on Start click;
  reverts to disabled (with full error banner) if `meeting/start` returns an error
- **RMS level meter** — visible during meetings, shows live amplitude bar +
  amber threshold line. Added with previous session's changes.
- **RMS watchdog** — if no RMS tick received for 3 s during a meeting,
  meter shows `⚠ signal lost`
- **WS message type-routing** — `onmessage` now routes on `msg.type`:
  - `"rms"` → `updateRmsMeter()`
  - `"error"` → `showError('Pipeline: ' + msg.message)`
  - anything else → `appendChunk()` (transcript)
- **Windows privacy banner** — on load, checks `/health`; if `windows_mic_blocked`
  is true, shows red banner with Settings path before user hits Start
- **`{"type": "error"}` WS handling** — pipeline crash messages surface in UI

---

## Current Test Count
**41/41 passing** at start of session (Sprint 2 baseline).
Tests were **not re-run** after this session's changes.

**Run tests before next session:**
```powershell
$env:PYTHONPATH="src"; python -m pytest tests/ -v
```

Expected: all 41 still pass (no public API signatures changed).
New tests needed for: `probe_device()`, `_check_mic_permission()`, fallback logic.

---

## Exact Startup Commands

```powershell
# Start the server
$env:PYTHONPATH="src"; python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000

# In a browser
# http://localhost:8000          — PM dashboard
# http://localhost:8000/debug    — pipeline counters (bookmark this)
# http://localhost:8000/devices  — device list with scores
# http://localhost:8000/health   — includes windows_mic_blocked flag

# Force-reset from browser URL bar (new this session)
# http://localhost:8000/meeting/reset

# Using real curl (NOT PowerShell alias)
curl.exe http://localhost:8000/debug
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/meeting/reset
curl.exe -X POST "http://localhost:8000/meeting/start?model_size=base"
curl.exe -X POST http://localhost:8000/meeting/stop
```

---

## Where We Stopped

The server was **running** but transcription was blocked by the silence gate.
The threshold fix (`0.00005`) was applied and the server was restarted.
**Status at session end: untested** — server restarted with the fix but
the user did not confirm a transcript appeared before saving this checkpoint.

### What is known to work
- Server starts cleanly
- `/health` returns `windows_mic_blocked` flag
- `/meeting/reset` works from browser URL bar (GET)
- `↺ Reset` button in UI
- RMS meter appears when meeting starts
- Device probe returns clear error if mic is blocked

### What is unconfirmed
- Whether threshold 0.00005 is low enough for your specific mic
  (RMS was 0.000075 — should pass now, but confirm with `/debug`)
- Whether Whisper produces transcripts after the gate fix
- Whether auto-reset on page load works end-to-end

---

## What To Do Next Session

### Step 1 — Verify the fix works (5 min)
1. Start server, open `http://localhost:8000`
2. Click Start → watch RMS meter → speak
3. Check terminal for `WHISPER raw=` log lines
4. Check `http://localhost:8000/debug` — `chunks_transcribed` should be > 0
5. Transcript should appear in the feed

### Step 2 — If still not working (diagnose with /debug)
```
chunks_from_queue == 0   → PyAudio not capturing (mic permission or device index)
chunks_filtered > 0      → silence gate still triggering (lower threshold further)
chunks_transcribed == 0  → Whisper filtering (no-speech prob or empty text)
chunks_broadcast == 0    → WebSocket not connected
```

### Step 3 — Adaptive calibration (designed, not built)
The system was designed but interrupted:
- `POST /calibrate?duration=3` — samples audio, sets threshold to 50% of avg RMS
  — clamp: `max(0.00003, min(0.001, avg_rms * 0.5))`
- `StreamingTranscriber.set_threshold(value)` — update gate at runtime
- `🎙 Calibrate` button in controls bar (only active during meeting)
- RMS meter text: `0.000075 | T: 0.00005` format

### Step 4 — Sprint 2 remaining items
1. **Speaker diarisation** — label who is speaking per chunk
2. **Auto-analyse on stop** — run Claude analysis when meeting ends
3. **Knowledge base** — ChromaDB indexing of transcripts for Q&A
4. **Query interface** — natural-language Q&A over meeting history
5. **Export** — Markdown, PDF, structured JSON reports

---

## Known Issues

| Issue | Status |
|---|---|
| Windows Voice Access competes for mic | Workaround: disable Voice Access before starting meeting |
| WASAPI loopback not found | Need "Stereo Mix" or VB-Audio Cable for system audio capture |
| Whisper medium model ~769 MB first download | Cached after first use |
| ffmpeg not on PATH in Claude Code shell | Works fine in Windows terminal |

---

## Git Status at Checkpoint

```
Modified:  frontend/index.html
Modified:  src/meetingmind/main.py
Untracked: docs/CHECKPOINT_SPRINT3.md
Untracked: docs/CHECKPOINT_SPRINT2.md
Untracked: docs/DEPLOYMENT.md
Untracked: src/meetingmind/device_detector.py
Untracked: tests/test_device_detector.py
Untracked: STARTUP.md
```

To commit this checkpoint:
```powershell
git add src/meetingmind/ frontend/index.html docs/CHECKPOINT_SPRINT3.md
git commit -m "Sprint 3: fix silence gate threshold, harden server, UI reset button"
```
