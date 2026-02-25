# MeetingMind — Architecture Overview (PM Edition)

MeetingMind is a real-time meeting assistant for **Project Managers**.
It captures audio locally, transcribes it with Whisper, and feeds live transcript
text to both a browser dashboard and a Claude-powered suggestion engine that
returns 3 PM-tailored response options per meeting moment.

---

## Phase 1 — Real-Time Audio Engine + PM Suggestions

### Component diagram

```
  Microphone ──────────────────────────────────────────────────────────┐
                                                                        │
  System audio                                                          ▼
  (WASAPI loopback) ──────────────────────────────►  AudioCapture  (audio_capture.py)
  Zoom / Teams / Meet                                 │  Two PyAudio streams
                                                      │  3-second chunks
                                                      │  Resampled to 16 kHz
                                                      │  RMS silence gate
                                                      ▼
                                               StreamingTranscriber  (transcriber.py)
                                                      │  Whisper medium (local)
                                                      │  Auto language detection
                                                      │  Confidence scoring
                                                      │
                                               TranscriptChunk
                                               { text, language,
                                                 source, timestamp,
                                                 confidence }
                                                      │
                                                      ▼
                                            FastAPI + WebSocket  (main.py)
                                            /ws/transcribe
                                                      │
                              ┌───────────────────────┼────────────────────────┐
                              ▼                        ▼                        ▼
                    Browser / PM Dashboard      Any WS client           POST /suggestions
                    frontend/index.html                                suggestion_engine.py
                    (live transcript feed)                             (Claude API)
                    (PM suggestion panel)                              PM-tailored responses
```

### PM Suggestion flow

```
Browser (PM dashboard)
  │
  │  POST /suggestions
  │  { "transcript": "..last 60s of meeting text.." }
  │
  ▼
main.py  →  suggestion_engine.get_suggestions(transcript_snippet)
                │
                │  System prompt (Project Manager edition):
                │  • sprint planning • stakeholder updates
                │  • risk management • resource allocation
                │  • delivery timelines • scope changes
                │  • budget tracking • team performance
                │  • tone auto-detection (formal/semi-formal/casual)
                │
                ▼
         Claude API (claude-sonnet-4-6)
                │
                ▼
         ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
                │
                ▼
         Browser — click-to-copy suggestion cards
```

### Audio pipeline flow

```
PyAudio (2 threads)
  │
  │  raw int16 PCM @ device native rate (44100 / 48000 Hz typical)
  │
  ├─ _to_float32()       int16 bytes → float32 in [-1, 1]
  ├─ _resample()         → 16 000 Hz  (numpy linear interpolation)
  ├─ buffer              accumulate samples
  ├─ chunk boundary      emit every chunk_duration seconds
  └─ AudioChunk(data, source, timestamp, rms)
         │
         ▼
  StreamingTranscriber
  │
  ├─ silence gate        rms < 0.01 → skip
  ├─ whisper.transcribe  float32 array → {text, segments, language}
  ├─ no_speech filter    no_speech_prob > 0.80 → skip
  ├─ empty text filter   punctuation-only → skip
  └─ TranscriptChunk(text, language, source, timestamp, confidence)
         │
         ▼
  ConnectionManager
  └─ broadcast JSON to all /ws/transcribe clients
```

### Privacy notes (Phase 1)

- **Audio capture**: Both PyAudio streams read directly from the Windows audio
  subsystem. No audio bytes are written to disk or transmitted over the network.
- **Transcription**: Whisper runs as an in-process Python model. The float32
  audio array is passed in memory — it never touches the filesystem.
- **WebSocket stream**: Only transcript text is broadcast, never raw audio PCM.
- **PM Suggestions**: Transcript text snippets are sent to the Anthropic API
  over HTTPS only when the PM explicitly requests suggestions. The payload
  contains transcript text only — no audio, no PII beyond what was spoken.
- **PRIVACY_MODE flag**: When `privacy_mode=True` (default), a banner is printed
  to the console at capture start confirming local-only audio processing.

---

## Phase 0 — File-Based Pipeline (completed)

```
Audio file  (audio/)
    │
    ▼
transcriber.py          Whisper local  →  transcripts/<stem>_<timestamp>UTC.json
    │
    ▼
analyser.py             Claude API     →  outputs/summaries/…_summary.md
                                       →  outputs/action_items/…_action_items.md
```

---

## Full pipeline (all phases)

```
Audio file / live mic
    │
    ▼
┌─────────────────────┐
│   Transcription     │  Whisper local — audio stays on device
│   audio_capture.py  │
│   transcriber.py    │
└────────┬────────────┘
         │  TranscriptChunk / transcript JSON
         ├────────────────────────────────────► suggestion_engine.py
         │                                      Claude — PM suggestions (live)
         ▼
┌─────────────────────┐
│   LLM Analysis      │  Claude API — transcript text sent over HTTPS
│   analyser.py       │  Post-meeting: summaries + action items
└────────┬────────────┘
         │  summary.md / action_items.md
         ▼
┌─────────────────────┐
│   Knowledge Base    │  ChromaDB — local vector store
│   kb.py  (future)   │
└────────┬────────────┘
         │
         ▼
    Query Interface    Natural-language Q&A over meeting history
```

---

## Component responsibilities

| File | Phase | Responsibility |
|---|---|---|
| `src/meetingmind/audio_capture.py` | 1 | Mic + system audio → AudioChunk queue |
| `src/meetingmind/transcriber.py` | 0+1 | File transcription + StreamingTranscriber |
| `src/meetingmind/suggestion_engine.py` | 1 | Claude PM response suggestions (3 per call) |
| `src/meetingmind/main.py` | 1 | FastAPI server, WebSocket broadcast, /suggestions endpoint |
| `src/meetingmind/analyser.py` | 0 | Post-meeting LLM summarisation & action-item extraction |
| `src/meetingmind/cli.py` | 0 | CLI entry point (transcribe / analyse) |
| `frontend/index.html` | 1 | PM Edition web dashboard |
| `src/meetingmind/kb.py` | future | Vector store embed & query |
| `config/settings.example.toml` | all | All configuration options documented |

## API surface

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| `GET` | `/health` | `health()` | Server liveness + meeting state |
| `POST` | `/meeting/start` | `meeting_start()` | Begin audio capture and transcription |
| `POST` | `/meeting/stop` | `meeting_stop()` | End capture, release resources |
| `POST` | `/suggestions` | `suggestions()` | Generate 3 PM response suggestions |
| `WS` | `/ws/transcribe` | `websocket_endpoint()` | Real-time transcript stream |

## Data flow & storage

| Directory | Contents | Committed? |
|---|---|---|
| `audio/` | Raw audio input files | No — gitignored |
| `transcripts/` | JSON transcripts from Whisper | No — gitignored |
| `outputs/summaries/` | Markdown summaries from Claude | No — gitignored |
| `outputs/action_items/` | Markdown action items from Claude | No — gitignored |
| `data/knowledge_base/` | ChromaDB vector index | No — gitignored |
