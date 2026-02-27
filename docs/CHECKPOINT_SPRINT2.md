# MeetingMind — Sprint 2 Checkpoint

**Date:** 2026-02-25
**Status:** Sprint 2 in progress — server verified, audio devices mapped

---

## Sprint 1 Handoff

All Sprint 1 deliverables complete and verified. See `docs/SPRINT1_COMPLETE.md` for full detail.

| Deliverable | Status |
|---|---|
| File-based transcription pipeline | Done |
| Real-time audio capture + Whisper streaming | Done |
| FastAPI server + WebSocket broadcast | Done |
| 17/17 unit + integration tests passing | Done |
| Git repo pushed to GitHub | Done |

---

## Sprint 2 Roadmap

| # | Feature | Status | Notes |
|---|---|---|---|
| 1 | Speaker diarisation | Not started | Label who is speaking per chunk |
| 2 | System audio (Stereo Mix) | Not started | Stereo Mix enabled — index 13 confirmed |
| 3 | Auto-analyse on stop | Not started | Run Claude analysis when meeting ends |
| 4 | Knowledge base | Not started | ChromaDB indexing of transcripts |
| 5 | Query interface | Not started | Natural-language Q&A over meeting history |
| 6 | Export | Not started | Markdown, PDF, structured JSON reports |

---

## Environment State (as of 2026-02-25)

### Server
- **URL:** `http://0.0.0.0:8000`
- **Framework:** FastAPI + uvicorn
- **Start command:** `PYTHONPATH=src python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000`

### Endpoint Health
| Endpoint | Expected Response |
|---|---|
| `GET /health` | `{"status":"ok","meeting_active":false,...}` |
| `POST /meeting/start` | `{"status":"started","model":"medium",...}` |
| `POST /meeting/stop` | `{"status":"stopped","elapsed_seconds":...}` |
| `WS /ws/transcribe` | Real-time JSON transcript stream |

### Audio Devices (PyAudio index map)
| Index | Device | Type | Notes |
|---|---|---|---|
| 0 | Microsoft Sound Mapper - Input | Input | Default input alias |
| 1 | Microphone (Camo) | Input | iPhone webcam mic — current default |
| 2 | Microphone Array (Intel® Smart Sound Technology) | Input | Built-in laptop array (short name) |
| 5 | Primary Sound Capture Driver | Input | Windows alias |
| 7 | Microphone Array (Intel® Smart Sound Technology for Digital Microphones) | Input | Built-in laptop array (full WASAPI name) |
| 11 | Microphone Array (Intel® Smart Sound Technology for Digital Microphones) | Input | Same device, third enumeration |
| 12 | Microphone (Camo) | Input | Camo duplicate |
| 13 | Stereo Mix (Realtek HD Audio Stereo input) | Input | **System audio loopback — ENABLED** |
| 20 | Microphone (Realtek HD Audio Mic input) | Input | Physical 3.5mm jack |

**Key finding:** `Stereo Mix` is enabled at index 13 — system audio (Zoom/Teams/Meet) capture is available without VB-Cable.

**Current default:** Server opens `Microphone (Camo)` by default. Pass device index explicitly to use Intel mic or Stereo Mix.

### Known Issues
| Issue | Status | Notes |
|---|---|---|
| Default mic is Camo, not Intel | Open | Need to pass device index explicitly |
| WASAPI loopback not auto-detected | Open | Stereo Mix exists but is not found by loopback scan — direct index selection needed |
| ffmpeg not on PATH in Claude Code shell | Open | Works fine in Windows terminal |
| Whisper medium model first-run download | Resolved | Cached at `~/.cache/whisper/` after first run |
| Silence threshold too high (0.001) | Resolved | Fixed to 0.0002 in `transcriber.py` |
| Debug reported "Pipeline OK" despite 99% filtration | Resolved | Fixed in `main.py` |

---

## Files Changed Since Sprint 1

| File | Change |
|---|---|
| `src/meetingmind/transcriber.py` | Silence threshold lowered to 0.0002 |
| `src/meetingmind/main.py` | Debug pipeline diagnosis fixed |
| `docs/CHECKPOINT_SPRINT2.md` | This file |
| `STARTUP.md` | New — session startup guide |

---

## Next Session Priorities

1. Wire Stereo Mix (index 13) into `audio_capture.py` as the system audio source
2. Allow device selection via `/meeting/start` request body or `.env` config
3. Begin speaker diarisation research (pyannote.audio vs simpler energy-based approach)
