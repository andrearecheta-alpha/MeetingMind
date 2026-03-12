# MeetingMind — ML Pipeline

Whisper model lifecycle, configuration, tuning, and inference details.

---

## Whisper Model Overview

MeetingMind uses **OpenAI Whisper** for all transcription.
Whisper runs entirely locally — no audio or text is sent to any external API.

| Property | Value |
|---|---|
| Library | `openai-whisper==20250625` |
| Backend | PyTorch (`torch==2.10.0`) |
| Inference device | CPU (fp16=False) |
| Default model | `base` |
| Default language | `en` (fixed — skips per-chunk detection) |
| Cache location | `~/.cache/whisper/` |

---

## Model Size Reference

| Model | Parameters | VRAM | Disk | Load time (CPU) | Real-time? | Accuracy |
|---|---|---|---|---|---|---|
| tiny | 39 M | ~1 GB | ~75 MB | ~2 s | Yes | Low |
| base | 74 M | ~1 GB | ~145 MB | ~5 s | Yes | Good |
| small | 244 M | ~2 GB | ~483 MB | ~12 s | Yes | Better |
| medium | 769 M | ~5 GB | ~1.5 GB | ~55 s | Marginal | High |
| large | 1550 M | ~10 GB | ~2.9 GB | ~100 s | No | Best |

**Recommended for real-time meeting use:** `base` (default) or `small`.
`medium` and `large` introduce latency that makes live transcription impractical on CPU.

---

## Model Loading Lifecycle

### Preload at server startup (lifespan)

```python
# main.py — lifespan() context manager
asyncio.create_task(_preload_whisper(_WHISPER_MODEL_DEFAULT))
```

- Fires immediately when uvicorn binds to port 8000
- Runs in `asyncio.to_thread` — does not block the event loop
- Stores loaded model in `_preloaded_whisper` global
- If the user starts a meeting with the same model size, load delay = 0

### Background load on meeting start

If the preloaded model does not match the requested size (or preload failed):

```python
# _whisper_load_background()
transcriber = await asyncio.to_thread(
    lambda: StreamingTranscriber(
        model_size=model_size,
        privacy_mode=privacy_mode,
        language=language,
    )
)
```

1. `POST /meeting/start` returns immediately (`{"status": "starting", "model_loading": true}`)
2. Audio capture + RMS broadcast begin instantly
3. Whisper loads in background (5–100 s depending on model)
4. On load complete: `broadcast({"type": "model_ready", "model": ...})`
5. Frontend transitions: `◌ Loading Whisper model…` → `● Live`

### Model size selection (runtime)

```
POST /meeting/start?model_size=base      # default
POST /meeting/start?model_size=small     # better accuracy
POST /meeting/start?model_size=medium    # high accuracy, slow
```

---

## Inference Configuration

### `StreamingTranscriber.__init__`

```python
self._model = whisper.load_model(model_size)   # cached after first download
self._language = language                        # "en" by default
self._privacy_mode = privacy_mode
self._model_size = model_size                   # stored for preload reuse check
```

### `transcribe_chunk(chunk: AudioChunk)`

```python
result = self._model.transcribe(
    chunk.data,                          # np.float32 array, 16 kHz mono
    language=self._language,             # "en" — skips language detection pass
    fp16=False,                          # CPU inference (no CUDA)
    condition_on_previous_text=False,    # prevents hallucination carryover
)
```

**Why `condition_on_previous_text=False`:**
Without this, Whisper can "remember" previous chunks and hallucinate continuations
even from silence. Disabling it makes each chunk fully independent — better for
real-time use where chunks are already filtered by the VAD.

---

## Voice Activity Detection (VAD)

MeetingMind uses a custom energy-based VAD, not Whisper's built-in VAD.
This allows silence filtering **before** the chunk reaches Whisper, reducing
wasted inference cycles.

### Energy VAD (`audio_capture.py`)

```python
HISTORY_SIZE     = 10    # chunks in rolling window
ENERGY_MULTIPLIER = 1.5  # dynamic threshold multiplier
MIN_THRESHOLD    = 0.001 # absolute floor

# Per chunk:
peak_rms = np.max(np.abs(chunk_data))
threshold = max(
    np.mean(history_rms_values) * ENERGY_MULTIPLIER,
    MIN_THRESHOLD,
)
if peak_rms < threshold:
    return  # drop — silence
```

**Why peak RMS, not mean RMS:**
Mean RMS averages over the entire chunk (3 s). A chunk with 0.5 s of clear speech
and 2.5 s of silence can have a mean RMS well below threshold, dropping valid speech.
Peak RMS captures the maximum amplitude in the chunk — any speech at all keeps the chunk.

### Whisper-level silence filter (`transcriber.py`)

After Whisper runs, additional filters drop low-quality results:

```python
_SILENCE_PEAK_THRESHOLD = 0.001   # applied before calling Whisper

if chunk.peak_rms < _SILENCE_PEAK_THRESHOLD:
    return None   # didn't even attempt transcription

# Post-Whisper filters:
if not result["text"].strip():
    return None   # empty output

segments = result.get("segments", [])
if segments:
    no_speech_prob = segments[0].get("no_speech_prob", 0)
    avg_logprob    = segments[0].get("avg_logprob", 0)
    if no_speech_prob > 0.6:
        return None   # Whisper says: probably silence
    if avg_logprob < -1.0:
        return None   # very low confidence — discard
```

---

## Concurrency Model

```python
_MAX_CONCURRENT_WHISPER = 1   # sequential — 1 Whisper task at a time
```

**Sequential inference rationale:**
- Whisper on CPU saturates all cores during inference
- Parallel Whisper calls compete for CPU and produce slower results than sequential
- A 3-second hard timeout (`asyncio.wait_for(..., timeout=3.0)`) prevents a slow
  inference from blocking the pipeline indefinitely

**Timeout handling:**
```python
try:
    result = await asyncio.wait_for(
        asyncio.to_thread(transcriber.transcribe_chunk, chunk),
        timeout=3.0,
    )
except asyncio.TimeoutError:
    _stats["chunks_filtered"] += 1
    continue  # drop chunk, move on
```

---

## Speaker Identification

MeetingMind uses **source-based speaker labeling**, not diarisation.
Each `AudioChunk` carries an `AudioSource` enum set at capture time:

| Source | Label | Description |
|---|---|---|
| `AudioSource.MICROPHONE` | `[You]` | Host microphone |
| `AudioSource.SYSTEM` | `[Them]` | System audio loopback (remote caller) |
| `AudioSource.GUEST` | `[Them]` | Guest phone mic via `/ws/guest` |

**Limitation (DEF-044):** In-person meetings where two people share one microphone
will both be labeled `[You]`. True diarisation requires pyannote.audio (Sprint 6).

---

## Talk Dominance Computation

Used by the coaching engine to detect one-person-talking patterns.

```python
def compute_talk_dominance(window: list[str]) -> float:
    """
    window: list of up to 10 source labels ("microphone" or "system"/"guest")
    Returns: fraction of window belonging to the dominant speaker (0.0 – 1.0)
    """
    if not window:
        return 0.0
    counts = Counter(window)
    return max(counts.values()) / len(window)

# Coaching fires if dominance > 0.8 (8 of last 10 chunks from same source)
```

---

## Confidence Score Derivation

Whisper does not produce a direct confidence score. MeetingMind derives one
from `avg_logprob`:

```python
def _logprob_to_confidence(avg_logprob: float) -> float:
    """
    avg_logprob range: roughly -2.0 (bad) to 0.0 (perfect)
    Maps to 0.0 – 1.0 confidence range.
    """
    return max(0.0, min(1.0, (avg_logprob + 2.0) / 2.0))
```

Displayed in the UI as a thin coloured bar on each transcript chunk.

---

## Performance Benchmarks (Intel i7, CPU only)

| Model | Avg inference time (3s chunk) | Real-time factor |
|---|---|---|
| tiny | 0.4 s | 7.5× |
| base | 0.9 s | 3.3× |
| small | 2.8 s | 1.1× |
| medium | 8.5 s | 0.35× (not real-time) |

`base` produces 0.9 s average inference for a 3 s chunk — well within real-time budget.
Chunks pile up if inference consistently exceeds 3 s, triggering the hard timeout.

---

## Future ML Work (Sprint 6+)

| Feature | Library | Approach |
|---|---|---|
| Speaker diarisation | `pyannote.audio` | Segment + cluster by voice embeddings |
| Sentiment analysis | transformers / Claude | Per-chunk sentiment for coaching context |
| Topic detection | BERTopic / Claude | Automatic agenda item grouping |
| Action item extraction | Claude (current) | Improve prompt for reliability |
| Keyword spotting | regex + embeddings | Real-time term detection for custom alerts |
