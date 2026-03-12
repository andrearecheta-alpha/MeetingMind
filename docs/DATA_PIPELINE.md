# MeetingMind — Data Pipeline

End-to-end data flow from raw audio to stored knowledge.

---

## Pipeline Overview

```
Microphone / System Audio / Guest Phone
         │
         ▼
   [ PyAudio Streams ]
   _stream_worker threads (one per source)
   16 000 Hz, mono, float32
         │
         ▼
   [ Energy VAD ]
   10-chunk rolling history
   peak_rms gate: threshold = 0.001
   Silent chunks → dropped (not enqueued)
         │
         ▼
   [ AudioChunk Queue ]
   _chunk_queue (thread-safe, maxsize=64)
   Fields: data (np.ndarray), source, timestamp, rms, peak_rms
         │
         ▼
   [ Whisper Transcription ]
   asyncio.to_thread (thread pool, MAX_CONCURRENT=1)
   Model: base (default) — tiny/small/medium/large available
   Language: en (fixed, skips per-chunk detection)
   fp16: False (CPU inference)
   Timeout: 3 s hard limit per chunk
         │
         ▼
   [ TranscriptChunk ]
   Fields: text, language, source, timestamp, confidence
   Silence filter: confidence < 0.4 OR text == "" → dropped
         │
         ├──► [ WebSocket Broadcast ]
         │    _manager.broadcast() → all connected clients
         │    JSON: {text, language, source, timestamp, confidence}
         │
         ├──► [ Decision Detector ]
         │    DECISION_PATTERNS regex scan
         │    Match → broadcast {"type":"decision",...}
         │    → _meeting["decisions"] list
         │
         ├──► [ Coaching Engine ]
         │    detect_coaching(text, speaker_window)
         │    detect_coaching_resolution(text, last_coaching)
         │    Match + cooldown check → broadcast {"type":"coaching",...}
         │    → _meeting["coaching_history"] list
         │
         └──► [ Knowledge Base Ingestion ] (on meeting stop)
              ChromaDB + SQLite
              Chunks → embeddings → 6 collections
              MQS computed → meetings.db
```

---

## Stage 1 — Audio Capture (`audio_capture.py`)

### Inputs
- Microphone: PyAudio input stream, device index from `/devices`
- System audio: WASAPI loopback (Stereo Mix, VB-Audio Cable)
- Guest phone: binary PCM frames from `/ws/guest`

### Processing
```
Sample rate:    16 000 Hz (Whisper native)
Channels:       1 (mono)
Format:         paFloat32
Frames/buffer:  chunk_duration × sample_rate
                (default: 3.0 s × 16 000 = 48 000 samples)
```

### Energy VAD
```python
HISTORY_SIZE = 10       # chunks in rolling window
ENERGY_MULTIPLIER = 1.5 # threshold = multiplier × mean(history)
MIN_THRESHOLD = 0.001   # floor — never gate above this

peak_rms = max(abs(audio_frame))
if peak_rms < max(mean(history) * ENERGY_MULTIPLIER, MIN_THRESHOLD):
    drop chunk  # silence
else:
    enqueue chunk
```

### Output: AudioChunk
```python
@dataclass
class AudioChunk:
    data:      np.ndarray   # float32, shape (N,)
    source:    AudioSource  # MICROPHONE | SYSTEM | GUEST
    timestamp: float        # Unix timestamp
    rms:       float        # mean RMS of chunk
    peak_rms:  float        # max absolute amplitude
```

---

## Stage 2 — Transcription (`transcriber.py`)

### Whisper call
```python
result = whisper_model.transcribe(
    audio,
    language="en",
    fp16=False,
    condition_on_previous_text=False,
)
```

### Output filters
| Condition | Action |
|---|---|
| `result["text"].strip() == ""` | Drop (empty output) |
| `no_speech_prob > 0.6` | Drop (high silence probability) |
| `avg_logprob < -1.0` | Drop (very low confidence) |
| Passed all filters | Return `TranscriptChunk` |

### Output: TranscriptChunk
```python
@dataclass
class TranscriptChunk:
    text:       str
    language:   str
    source:     str    # "microphone" | "system" | "guest"
    timestamp:  float
    confidence: float  # derived from avg_logprob
```

---

## Stage 3 — Real-Time Intelligence (in `_transcription_loop`)

### Decision detection
```python
decision = detect_decision(result.text)
if decision:
    _meeting["decisions"].append({...})
    await _manager.broadcast({"type": "decision", ...})
```

### Coaching detection
```python
coaching = detect_coaching(result.text, _meeting["speaker_window"])
if coaching:
    await _fire_coaching(coaching, timestamp)
```

### Speaker window update
```python
_meeting["speaker_window"].append(result.source)
_meeting["speaker_window"] = _meeting["speaker_window"][-10:]  # rolling 10
```

---

## Stage 4 — Knowledge Base Ingestion (on meeting stop)

Triggered by `POST /meeting/stop`. Runs in `asyncio.to_thread`.

### ChromaDB collections (6)

| Collection | Content | Embedding |
|---|---|---|
| `transcripts` | Raw transcript chunks | text-embedding-3-small |
| `decisions` | Detected decisions | text-embedding-3-small |
| `action_items` | AI-extracted action items | text-embedding-3-small |
| `coaching_events` | Fired coaching prompts | text-embedding-3-small |
| `meeting_summaries` | Claude-generated summary | text-embedding-3-small |
| `speaker_stats` | Talk-time ratios per meeting | metadata only |

### SQLite analytics (`data/analytics/meetings.db`)

```sql
CREATE TABLE meetings (
    id              TEXT PRIMARY KEY,   -- YYYYMMDD_HHMMSS
    date            TEXT,
    duration_s      REAL,
    model           TEXT,
    chunk_count     INTEGER,
    decision_count  INTEGER,
    mqs             REAL,               -- Meeting Quality Score (0–100)
    speaker_you_pct REAL,
    speaker_them_pct REAL
);
```

### Claude extraction (via `analyser.py`)
On meeting stop, Claude is called with the full transcript to extract:
- Summary (3–5 bullet points)
- Action items (assignee + deadline)
- Key decisions
- Follow-up questions

Results stored in ChromaDB + returned to client.

---

## Stage 5 — Context Engine (`context_engine.py`)

Used by `/suggestions` endpoint. Queries ChromaDB with the current transcript
to retrieve grounded context from past meetings.

```python
grounded = context_engine.query(
    current_transcript=recent_chunks,
    top_k=5,
)
# Returns: GroundedContext(relevant_decisions, relevant_actions, summary_snippets)
```

The `GroundedContext` is passed as `historical_context` to `suggestion_engine.get_suggestions()`,
allowing Claude to reference past meeting data when generating PM suggestions.

---

## RMS Broadcast Sub-Pipeline

Runs in parallel with transcription (`_rms_broadcast_loop`). Independent of Whisper.

```
AudioCapture._rms_queue  (mic-only writes)
         │
         ▼  every 0.5 s
   get_latest_rms()  → drains queue, returns freshest value
         │
         ▼
   WebSocket broadcast:
   {"type": "rms", "rms": X, "threshold": Y, "captured": N, "passed": M}
```

RMS is available to the browser **immediately** after `meeting/start` returns,
before Whisper has finished loading. This is the core UX improvement from Sprint 3.

---

## Data Retention

| Data type | Retention | Location |
|---|---|---|
| Raw audio | Never stored | In-memory queue only |
| Transcript chunks | Session lifetime + KB | `_meeting["transcript_chunks"]` + ChromaDB |
| Decisions | Session + KB | `_meeting["decisions"]` + ChromaDB |
| Coaching events | Session + KB | `_meeting["coaching_history"]` + ChromaDB |
| Meeting analytics | Indefinite | `data/analytics/meetings.db` |
| Whisper model weights | Indefinite (cache) | `~/.cache/whisper/` |

Audio bytes are **never written to disk**. The pipeline is purely in-memory from
PyAudio callback through to Whisper. Only the text output is persisted.
