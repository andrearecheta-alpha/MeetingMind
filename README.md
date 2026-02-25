# MeetingMind — PM Edition

A real-time meeting assistant built for **Project Managers**.
MeetingMind listens to your meetings, produces live transcripts on-device,
and surfaces instant AI-generated response suggestions tailored to PM topics:
sprint planning, stakeholder updates, risk management, resource allocation,
delivery timelines, scope changes, budget tracking, and team performance.

---

## Features

### Live in Sprint 1
- [x] **Local transcription** — Whisper runs on-device; audio never leaves your machine
- [x] **Real-time WebSocket stream** — transcript chunks broadcast to any browser tab
- [x] **Dual-source capture** — microphone + system audio (Zoom/Teams/Meet via WASAPI loopback)
- [x] **PM response suggestions** — Claude generates 3 ready-to-use responses per transcript moment
- [x] **PM Edition UI** — dark-theme web dashboard with live transcript feed and suggestion panel
- [x] **Auto tone detection** — suggestions adapt to formal / semi-formal / casual meeting tone
- [x] **AI meeting analysis** — post-meeting summaries and action-item extraction via Claude

### Roadmap (Sprint 2+)
- [ ] Speaker diarisation — label who is speaking per chunk
- [ ] System audio — WASAPI loopback / VB-Cable for richer Zoom/Teams capture
- [ ] Auto-analyse on stop — run Claude analysis automatically when meeting ends
- [ ] Knowledge base — ChromaDB indexing of all transcripts for Q&A
- [ ] Query interface — ask natural-language questions across your meeting history
- [ ] Export — Markdown, PDF, and structured JSON reports

---

## Quick Start (PM Edition)

### 1. Install dependencies
```bash
pip install -r requirements.txt
# Also install ffmpeg (used by Whisper for audio decoding):
# Windows:  winget install ffmpeg --accept-source-agreements --accept-package-agreements
# Mac:      brew install ffmpeg
# Linux:    sudo apt install ffmpeg
```

### 2. Configure your API key
```bash
cp .env.example .env
# Edit .env — add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start the server
```powershell
cd C:\Users\andre\Projects\MeetingMind
$env:PYTHONPATH="src"; python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

### 4. Open the PM dashboard
Open `frontend/index.html` in your browser, then click **Start Meeting**.

### 5. Get live suggestions
As your meeting progresses, click **Get PM Suggestions** in the right panel to generate
3 Claude-powered response options tailored to the last 60 seconds of transcript.

---

## Architecture Overview

```
Microphone ──┐
             ├──► AudioCapture ──► StreamingTranscriber (Whisper, local)
System audio ┘         │
                        ▼
              FastAPI + WebSocket  /ws/transcribe
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  frontend/index.html          POST /suggestions
  (live transcript feed)    ──► suggestion_engine.py
  (PM suggestion panel)         (Claude API, PM prompt)
```

Full pipeline diagram: [docs/architecture.md](docs/architecture.md)

---

## Project Structure

```
MeetingMind/
├── frontend/
│   └── index.html              # PM Edition web dashboard
├── src/meetingmind/
│   ├── audio_capture.py        # Mic + system audio capture
│   ├── transcriber.py          # Whisper file + streaming transcription
│   ├── suggestion_engine.py    # Claude PM response suggestions
│   ├── analyser.py             # Post-meeting Claude analysis
│   ├── main.py                 # FastAPI server + WebSocket engine
│   └── cli.py                  # CLI: transcribe / analyse commands
├── tests/
│   └── test_audio_engine.py    # 17 passing tests
├── audio/                      # Drop audio files here (gitignored)
├── transcripts/                # Whisper output JSON (gitignored)
├── outputs/
│   ├── summaries/              # Claude summary Markdown (gitignored)
│   └── action_items/           # Claude action items (gitignored)
├── data/knowledge_base/        # ChromaDB vector store — future phase
├── docs/                       # Architecture diagrams, sprint notes
└── config/
    └── settings.example.toml  # All configuration options documented
```

---

## CLI — File-Based Pipeline

No server needed for offline analysis:

```bash
# Transcribe an audio file locally with Whisper
python -m meetingmind transcribe audio/my_meeting.mp3

# Use a more accurate model (downloads once, then cached at ~/.cache/whisper/):
python -m meetingmind transcribe audio/my_meeting.mp3 --model small

# Specify language to skip auto-detection:
python -m meetingmind transcribe audio/my_meeting.mp3 --model base --language en

# Analyse a saved transcript with Claude (summaries + action items)
python -m meetingmind analyse transcripts/my_meeting_20260225_103045UTC.json
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + meeting state |
| `POST` | `/meeting/start` | Begin audio capture and transcription |
| `POST` | `/meeting/stop` | End capture, release resources |
| `POST` | `/suggestions` | Generate 3 PM response suggestions from transcript |
| `WS` | `/ws/transcribe` | Real-time transcript stream |

---

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- See `requirements.txt` for Python package versions

---

## Privacy

- **Audio processing** is entirely local — Whisper runs on your machine
- **Transcript text** is sent to the Anthropic API over HTTPS only when you request suggestions or analysis
- **No audio bytes** are ever transmitted over the network
- See [SECURITY.md](SECURITY.md) for the full threat model

---

## License

MIT — see [LICENSE](LICENSE) for details.
