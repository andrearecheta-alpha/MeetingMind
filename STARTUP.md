# MeetingMind — Session Startup Guide

Use this file at the start of every session to get the server running and verified in under 2 minutes.

---

## Step 1 — Read Context

At session start, ask Claude to read:
- `STARTUP.md` ← this file
- `docs/CHECKPOINT_SPRINT2.md` ← current sprint state

---

## Step 2 — Check Audio Devices

Verify the mic and Stereo Mix are still visible:

```bash
python -c "
import pyaudio
p = pyaudio.PyAudio()
print('=== AUDIO DEVICES ===')
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f'  [{i}] {d[\"name\"]}')
p.terminate()
"
```

**Expected:** Intel mic and `Stereo Mix (Realtek HD Audio Stereo input)` appear in the list.

---

## Step 3 — Run Tests

Verify nothing is broken before starting the server:

```powershell
cd C:\Users\andre\Projects\MeetingMind
python -m pytest tests/test_audio_engine.py -v
```

**Expected:** `17 passed`

---

## Step 4 — Start the Server

```powershell
cd C:\Users\andre\Projects\MeetingMind
$env:PYTHONPATH="src"; python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

Run in a dedicated terminal — leave it open.

---

## Step 5 — Verify Endpoints

Run in a second terminal:

```powershell
# Health check
curl http://localhost:8000/health

# Start a test meeting
curl -X POST http://localhost:8000/meeting/start

# Stop it
curl -X POST http://localhost:8000/meeting/stop
```

**Expected responses:**

| Call | Expected |
|---|---|
| `GET /health` | `{"status":"ok","meeting_active":false,...}` |
| `POST /meeting/start` | `{"status":"started","model":"medium",...}` |
| `POST /meeting/stop` | `{"status":"stopped","elapsed_seconds":...}` |

---

## Step 6 — Open the Dashboard

Open in browser:

```
C:\Users\andre\Projects\MeetingMind\frontend\index.html
```

Or if serving statically:

```
http://localhost:8000/
```

---

## Step 7 — Connect WebSocket (optional smoke test)

Paste in browser DevTools console:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/transcribe");
ws.onopen = () => console.log("Connected — speak into your mic!");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## Stopping the Server

```powershell
# Find PID on port 8000
netstat -ano | findstr :8000

# Kill it (replace 12345 with actual PID)
taskkill /PID 12345 /F /T
```

---

## Key Device Indices (as of 2026-02-25)

| Index | Device | Use |
|---|---|---|
| 1 | Microphone (Camo) | iPhone webcam mic |
| 2 | Microphone Array (Intel® Smart Sound Technology) | Built-in laptop mic |
| 13 | Stereo Mix (Realtek HD Audio Stereo input) | System audio — Zoom/Teams/Meet |
| 20 | Microphone (Realtek HD Audio Mic input) | 3.5mm jack |

---

## CLI Reference (no server needed)

```powershell
# Transcribe an audio file locally
python -m meetingmind transcribe audio/your_recording.mp3

# Analyse a transcript with Claude
python -m meetingmind analyse transcripts/your_recording_<timestamp>UTC.json
```

---

## Current Sprint

**Sprint 2** — See `docs/CHECKPOINT_SPRINT2.md` for full roadmap and status.
