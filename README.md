# MeetingMind — PM Edition

A real-time meeting assistant built for **Project Managers**.
MeetingMind listens to your meetings, produces live transcripts on-device,
and surfaces instant AI-generated response suggestions tailored to PM topics:
sprint planning, stakeholder updates, risk management, resource allocation,
delivery timelines, scope changes, budget tracking, and team performance.

**Sprint 8 complete** | **465/465 tests passing** | 100% privacy-first architecture

---

## Features

### Core (Sprint 1-4)
- [x] **Local transcription** — Whisper runs on-device; audio never leaves your machine
- [x] **Real-time WebSocket stream** — transcript chunks broadcast to any browser tab
- [x] **Dual-source capture** — microphone + system audio (Zoom/Teams/Meet via WASAPI loopback)
- [x] **PM response suggestions** — Claude generates 3 ready-to-use responses per transcript moment
- [x] **PM Edition UI** — dark-theme web dashboard with live transcript feed and suggestion panel
- [x] **Auto tone detection** — suggestions adapt to formal / semi-formal / casual meeting tone
- [x] **AI meeting analysis** — post-meeting summaries and action-item extraction via Claude
- [x] **Decision detection** — real-time decision capture with pattern matching
- [x] **Coaching prompts** — talk-dominance tracking, time-based triggers, role-aware coaching
- [x] **Guest phone mic** — PIN-secured guest audio contribution via mobile device

### Intelligence Layer (Sprint 5-7)
- [x] **Knowledge base** — ChromaDB + SQLite ingestion of past meetings for semantic retrieval
- [x] **Context engine** — grounded suggestions from historical meeting data
- [x] **Fact-checking** — 3-layer hallucination defense with source attribution
- [x] **spaCy NER** — named entity extraction for key facts enrichment
- [x] **Role selector** — EA, PM, Sales, Custom role profiles with tailored coaching
- [x] **Guest viewer** — read-only guest link with live fact-checks and key facts
- [x] **PWA support** — installable progressive web app with service worker

### PM Intelligence (Sprint 8) — NEW
- [x] **Scope creep detection** — CRITICAL alerts validated in live client meetings
- [x] **Timeline & delay flags** — real-time risk flagging when timelines slip
- [x] **PM meeting summary** — auto-generated structured summaries on meeting stop
- [x] **Action item & decision capture** — automatic extraction with assignee detection
- [x] **PM Project Brief** — document intelligence with upload and extraction
- [x] **Global PM Knowledge Base** — PMBOK, Scrum, and CPMAI methodology integration
- [x] **Audio preprocessing** — chunk overlap, pre-emphasis, normalisation, fade-in

### Roadmap
- [ ] Speaker diarisation — label who is speaking per chunk
- [ ] Export — Markdown, PDF, and structured JSON reports

---

## What's Next

**Sprint 9: Even Realities G1 Integration**
Smartglass-powered meeting assistance — live transcript and coaching prompts
displayed directly in your field of view during meetings.

---

## Quick Start (PM Edition)

### 1. Clone and install
```bash
git clone https://github.com/andrearecheta-alpha/MeetingMind.git
cd MeetingMind
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

### 3. Configure settings
```bash
cp config/settings.example.toml config/settings.toml
# Edit config/settings.toml — adjust token budgets, audio thresholds, etc.
```

### 4. Start the server
```bash
PYTHONPATH=src python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

PowerShell:
```powershell
$env:PYTHONPATH="src"; python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000
```

### 5. Open the PM dashboard
Open `http://localhost:8000` in your browser, then click **Start Meeting**.

### 6. Get live suggestions
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
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
  Live transcript   Suggestions    PM Intelligence
  (WebSocket)       (Claude API)   (Scope creep, timeline,
                                    decisions, action items)
```

Full pipeline diagram: [docs/architecture.md](docs/architecture.md)

---

## Project Structure

```
MeetingMind/
├── frontend/
│   ├── index.html              # PM Edition web dashboard
│   ├── guest.html              # Guest phone mic interface
│   ├── guest_viewer.html       # Read-only guest viewer
│   ├── manifest.json           # PWA manifest
│   └── sw.js                   # Service worker (network-first)
├── src/meetingmind/
│   ├── main.py                 # FastAPI server + WebSocket engine
│   ├── audio_capture.py        # Mic + WASAPI loopback capture
│   ├── transcriber.py          # Whisper file + streaming transcription
│   ├── suggestion_engine.py    # Claude PM response suggestions
│   ├── analyser.py             # Post-meeting Claude analysis
│   ├── knowledge_base.py       # SQLite + ChromaDB ingestion
│   ├── context_engine.py       # Semantic search for grounded suggestions
│   ├── fact_checker.py         # 3-layer hallucination defense
│   ├── spacy_extractor.py      # spaCy NER extraction
│   ├── device_detector.py      # Audio device detection
│   ├── guest_session.py        # Guest session management
│   ├── token_budget.py         # Token budget management
│   ├── relay_stub.py           # Cloud relay placeholder
│   ├── _api_key.py             # API key loader
│   └── cli.py                  # CLI: transcribe / analyse commands
├── tests/                      # 465 passing tests
├── docs/                       # Architecture, deployment, pipeline docs
├── audio/                      # Drop audio files here (gitignored)
├── transcripts/                # Whisper output JSON (gitignored)
├── outputs/                    # Summaries + action items (gitignored)
├── data/                       # Knowledge base + analytics (gitignored)
└── config/
    └── settings.example.toml   # All configuration options documented
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
| `POST` | `/meeting/reset` | Reset stuck meeting state |
| `GET` | `/meeting/decisions` | List detected decisions |
| `GET` | `/meeting/coaching` | Coaching prompt history |
| `POST` | `/suggestions` | Generate 3 PM response suggestions |
| `POST` | `/guest/session` | Create guest mic session |
| `GET` | `/kb/status` | Knowledge base status |
| `WS` | `/ws/transcribe` | Real-time transcript stream |
| `WS` | `/ws/guest` | Guest audio WebSocket |

---

## Whisper Model Guide

| Model  | Load time | Accuracy | Real-time? |
|--------|-----------|----------|------------|
| tiny   | ~2s       | Low      | Yes        |
| base   | ~5s       | Good     | Yes        |
| small  | ~12s      | Better   | Yes        |
| medium | ~60s      | High     | Marginal   |
| large  | ~100s     | Best     | No         |

The `base` model is the default and works well for clear speech in English.

---

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- A working microphone
- ~500 MB disk space for the `base` Whisper model (downloaded once, cached)
- See `requirements.txt` for Python package versions

---

## Troubleshooting

**"No working microphone found"**
- Windows: Settings > Privacy & Security > Microphone > Allow apps to access your microphone
- Try selecting a specific device from the Mic dropdown instead of OS Default

**Transcript not appearing after model loads**
- Check the RMS meter — if the bar is flat, the mic is silent or muted
- Open `http://localhost:8000/debug` for a full pipeline diagnostic

**Server stuck in "meeting already active"**
- Click **Reset** in the UI, or visit `http://localhost:8000/meeting/reset`

---

## Changelog

### Sprint 8 — PM Intelligence Layer (2026-03-12)
- Scope creep detection (CRITICAL alerts validated in live client calls)
- Timeline & delay flag with real-time risk detection
- PM meeting summary generation (auto on Stop)
- Action item & decision capture with assignee detection
- PM Project Brief + document intelligence extraction
- Global PM KB: PMBOK + Scrum + CPMAI methodology seeding
- Audio preprocessing: chunk overlap, pre-emphasis, normalisation, fade-in
- 465/465 tests passing

### Sprint 7 — Fact-Checking & Guest Link (2026-03-08)
- 3-layer hallucination defense with source attribution
- Guest viewer with read-only live fact-checks
- Knowledge base fact-checking integration
- 327/327 tests passing

### Sprint 6 — Role System & Hardening (2026-02-28)
- Role selector (EA, PM, Sales, Custom)
- ProcessPoolExecutor for parallel processing
- Crash hardening across all async tasks
- Auto-recalibrate audio thresholds
- 230/230 tests passing

### Sprint 5 — Transcription Accuracy (2026-02-28)
- Transcription accuracy improvement
- Meeting summary on Stop
- WebSocket ping/keepalive with exponential backoff
- Key facts persistence
- 197/197 tests passing

### Sprint 4 — Coaching & Guest Mic (2026-02-27)
- Decision detection with pattern matching
- Coaching prompts (talk-dominance, time triggers)
- 3-panel layout (transcript, coaching, suggestions)
- Guest phone mic with PIN security
- 163/163 tests passing

### Sprint 3 — Knowledge Base (2026-02-26)
- ChromaDB + SQLite knowledge base
- Context engine for grounded suggestions
- spaCy NER extraction
- 51/51 tests passing

### Sprint 2 — Real-Time Pipeline (2026-02-25)
- Silence gate, dual-stream audio
- Whisper speed optimisation (base model, 1.5s chunks)
- Semaphore-based concurrent transcription
- 41/41 tests passing

### Sprint 1 — Foundation (2026-02-25)
- Local Whisper transcription
- WebSocket streaming
- Claude PM suggestions
- Dark-theme PM dashboard
- 17/17 tests passing

---

## Privacy

- **Audio processing** is entirely local — Whisper runs on your machine
- **Transcript text** is sent to the Anthropic API over HTTPS only when you request suggestions or analysis
- **No audio bytes** are ever transmitted over the network
- **No telemetry** — zero data collection
- See [SECURITY.md](SECURITY.md) for the full threat model

---

## License

MIT — see [LICENSE](LICENSE) for details.
