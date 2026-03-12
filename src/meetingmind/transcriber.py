"""
transcriber.py
--------------
Transcribes audio files using OpenAI Whisper running entirely on-device.

PRIVACY GUARANTEE
-----------------
All transcription is performed locally by the Whisper neural network model.
No audio data, transcript content, speaker information, or metadata is
transmitted to any external server, API, or third-party service at any point.
The model weights are downloaded once from Hugging Face on first use and then
cached locally at ~/.cache/whisper/. After that first download, this module
works fully offline.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Whisper uses ffmpeg internally, so it supports a wide range of containers.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4", ".webm", ".mkv"}
)

# Resolve the project root (this file is at src/meetingmind/transcriber.py,
# so two parents up lands at the project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Default output directory for transcript JSON files.
DEFAULT_TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "transcripts"

# Privacy statement embedded in every transcript record so its provenance is
# self-documenting even when the file is shared outside this tool.
_PRIVACY_NOTE = (
    "This transcript was produced entirely on-device using a local Whisper model. "
    "No audio data or transcript content was transmitted to any external server or API."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_audio_file(audio_path: Path) -> None:
    """
    Ensure the audio file exists and has a recognised extension.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError:        The path is a directory or has an unsupported extension.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _load_whisper_model(model_size: str):
    """
    Import and load a local Whisper model.

    Model sizes and their trade-offs
    ---------------------------------
    tiny   ~39 M params  — fastest, lowest accuracy
    base   ~74 M params  — good default for meetings
    small  ~244 M params — noticeably better accuracy, still fast
    medium ~769 M params — high accuracy, moderate speed
    large  ~1550 M params— best accuracy, slowest

    Models are downloaded from Hugging Face on first use and cached locally.

    Args:
        model_size: One of "tiny", "base", "small", "medium", "large".

    Returns:
        A loaded whisper.Model instance.

    Raises:
        ImportError: openai-whisper is not installed.
        ValueError:  An unrecognised model size was requested.
    """
    try:
        import whisper  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is not installed. "
            "Run:  pip install openai-whisper\n"
            "Note: ffmpeg must also be installed and on your PATH."
        ) from exc

    valid_sizes = {"tiny", "tiny.en", "base", "small", "medium", "large"}
    if model_size not in valid_sizes:
        raise ValueError(
            f"Invalid model size '{model_size}'. "
            f"Choose from: {', '.join(sorted(valid_sizes))}"
        )

    logger.info(
        "Loading Whisper '%s' model (first run downloads weights to ~/.cache/whisper/)…",
        model_size,
    )
    model = whisper.load_model(model_size)
    logger.info("Whisper model loaded: %s", model_size)
    return model


def _build_transcript_record(
    audio_path: Path,
    whisper_result: dict,
    model_size: str,
) -> dict:
    """
    Package Whisper's raw output into a structured, self-documenting record.

    Output shape
    ------------
    {
      "meta": {
        "source_file":      "meeting.mp3",
        "source_path":      "/abs/path/to/audio/meeting.mp3",
        "transcribed_at":   "2026-02-25T10:30:00+00:00",   # UTC ISO-8601
        "model":            "whisper-base",
        "language_detected":"en",
        "privacy_note":     "…"
      },
      "text":     "Full transcript as a single string.",
      "segments": [
        {"id": 0, "start": 0.0, "end": 4.2, "text": "Hello, everyone."},
        …
      ]
    }

    Timestamps are in seconds and rounded to three decimal places.
    """
    segments = [
        {
            "id":    seg["id"],
            "start": round(float(seg["start"]), 3),
            "end":   round(float(seg["end"]),   3),
            "text":  seg["text"].strip(),
        }
        for seg in whisper_result.get("segments", [])
    ]

    return {
        "meta": {
            "source_file":       audio_path.name,
            "source_path":       str(audio_path),
            "transcribed_at":    datetime.now(timezone.utc).isoformat(),
            "model":             f"whisper-{model_size}",
            "language_detected": whisper_result.get("language", "unknown"),
            "privacy_note":      _PRIVACY_NOTE,
        },
        "text":     whisper_result["text"].strip(),
        "segments": segments,
    }


def _save_transcript(record: dict, transcripts_dir: Path, audio_stem: str) -> Path:
    """
    Write the transcript record to a timestamped JSON file.

    Filename convention:  <original_stem>_<YYYYMMDD_HHMMSS>UTC.json
    Example:              standup_20260225_103045UTC.json

    The directory is created automatically if it does not exist.

    Returns:
        The Path of the written file.
    """
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # UTC timestamp makes filenames unambiguous across time zones.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = transcripts_dir / f"{audio_stem}_{timestamp}UTC.json"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: Path | str,
    model_size: str = "base",
    transcripts_dir: Path | str | None = None,
    language: str | None = None,
) -> Path:
    """
    Transcribe an audio file and save the result as JSON in transcripts/.

    All processing is local — audio never leaves this machine.

    Args:
        audio_path:      Path to the audio file to transcribe.
        model_size:      Whisper model size. "base" is the default.
                         See _load_whisper_model() for a size comparison table.
        transcripts_dir: Directory for the output JSON.
                         Defaults to <project_root>/transcripts/.
        language:        ISO-639-1 language code (e.g. "en", "fr").
                         When None, Whisper auto-detects the spoken language.

    Returns:
        Path to the saved transcript JSON file.

    Raises:
        FileNotFoundError: audio_path does not exist.
        ValueError:        Unsupported audio format or invalid model size.
        ImportError:       openai-whisper is not installed.
        RuntimeError:      Whisper encountered an error during transcription.
    """
    audio_path      = Path(audio_path).resolve()
    transcripts_dir = Path(transcripts_dir) if transcripts_dir else DEFAULT_TRANSCRIPTS_DIR

    # 1. Validate the input file before spending time loading the model.
    _validate_audio_file(audio_path)

    # 2. Load the on-device Whisper model.
    model = _load_whisper_model(model_size)

    # 3. Run transcription.
    logger.info("Transcribing '%s' — this may take a while for long recordings…", audio_path.name)
    try:
        # whisper.transcribe() returns {"text": str, "segments": [...], "language": str}
        # verbose=False suppresses Whisper's per-segment console output; we handle our own.
        whisper_kwargs: dict = {}
        if language:
            whisper_kwargs["language"] = language

        result: dict = model.transcribe(str(audio_path), verbose=False, **whisper_kwargs)

    except Exception as exc:
        raise RuntimeError(
            f"Whisper failed while processing '{audio_path.name}': {exc}"
        ) from exc

    # Warn — but do not fail — if the transcript is unexpectedly blank.
    if not result.get("text", "").strip():
        logger.warning(
            "Whisper returned an empty transcript for '%s'. "
            "Check that the audio file contains speech and is not corrupted.",
            audio_path.name,
        )

    # 4. Structure the result and write it to disk.
    record      = _build_transcript_record(audio_path, result, model_size)
    output_path = _save_transcript(record, transcripts_dir, audio_path.stem)

    logger.info("Transcript saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Streaming / real-time transcription  (Phase 1 — audio engine)
# ---------------------------------------------------------------------------

# Silence gate — filters chunks before Whisper to save CPU on quiet periods.
#
# We use PEAK amplitude (max|sample|) rather than average RMS because a
# 1.5-second chunk that is 1.4 s of silence + 0.1 s of speech still has
# very low average RMS but a clearly detectable peak.  Using peak ensures
# any chunk containing a single word gets through to Whisper.
#
# Calibration guide (float32 scale, samples normalised to [-1, 1]):
#   peak = max(|sample|) over the chunk window
#   Speech peak is typically 0.01 – 0.5 even at moderate mic volumes.
#   Electrical noise floor is typically < 0.001.
#
#   0.001 → recommended: passes any audible sound, filters noise floor
#   0.005 → stricter: requires a bit louder speech
#
# If chunks_filtered > 50 % in /debug, lower this value.
# If too much noise hallucination, raise it.
_SILENCE_PEAK_THRESHOLD: float = 0.001

# System audio (Stereo Mix / VB-Audio Cable / WASAPI loopback) produces
# much weaker signal than a direct microphone — use a drastically lower
# silence gate so [Them] chunks are not silently dropped.
_SYSTEM_SILENCE_PEAK_THRESHOLD: float = 0.0001

# Kept for reference — no longer used for the silence gate.
_SILENCE_RMS_THRESHOLD: float = 0.00005

# Whisper segments whose no-speech probability exceeds this are discarded.
_NO_SPEECH_PROB_THRESHOLD: float = 0.90

# Whisper returns these single-character strings for silent/noise-only chunks.
_EMPTY_TRANSCRIPTS: frozenset[str] = frozenset({"", ".", "...", "!", "?", " "})

# ---------------------------------------------------------------------------
# Hallucination detection — Whisper generates plausible-sounding but fake text
# when fed silence, ambient noise, or very weak audio.  Common symptoms:
#   • YouTube-style phrases ("thanks for watching", "please subscribe")
#   • Non-ASCII characters (Chinese, Arabic, Cyrillic) from English audio
#   • Impossible word density (50+ words from a 3-second chunk)
#   • ALL-CAPS shouting that wasn't in the audio
# ---------------------------------------------------------------------------
_HALLUCINATION_PHRASES: list[str] = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "harmonic",
    "www.",
    "http",
    "the transcript",
    "the end",
    "subtitles by",
    "captions by",
    "amara.org",
]

# Max plausible words per second of audio.  Conversational English peaks at
# ~4 words/s; Whisper hallucinations can produce 15+ words/s.
_MAX_WORDS_PER_SECOND: float = 3.0

# ---------------------------------------------------------------------------
# Hard gates — prevent Whisper from processing junk audio
# ---------------------------------------------------------------------------

# Absolute minimum average RMS — chunks below this are discarded before
# Whisper to prevent hallucinations from near-silence.  This floor cannot
# be lowered by auto-recalibration.  Adjusted per-device by
# set_rms_minimum_for_device() at meeting start.
_HARD_RMS_MINIMUM: float = 0.0005

# System audio hard RMS floor — much lower than mic because loopback
# devices (Stereo Mix, VB-Audio Cable) output at ~10-20% of mic levels.
_SYSTEM_HARD_RMS_MINIMUM: float = 0.00003


# ---------------------------------------------------------------------------
# Audio preprocessing — prefix clipping prevention
# ---------------------------------------------------------------------------
# Whisper struggles with the first 0.3-0.5s of hard-sliced chunks.
# These helpers apply overlap, padding, fade-in, pre-emphasis, and
# normalisation AFTER gate checks, BEFORE Whisper inference.

_PREPROCESS_DEFAULTS: dict = {
    "chunk_overlap_seconds": 0.3,
    "leading_silence_pad_seconds": 0.1,
    "pre_emphasis_coefficient": 0.97,
    "normalisation_peak": 0.9,
    "fade_in_ms": 10,
}


def _load_preprocess_config() -> dict:
    """Read [audio] preprocessing params from config/settings.toml."""
    defaults = dict(_PREPROCESS_DEFAULTS)
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "settings.toml"
    if not cfg_path.exists():
        return defaults
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return defaults
    try:
        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)
        audio = cfg.get("audio", {})
        for key in defaults:
            if key in audio:
                defaults[key] = float(audio[key])
        return defaults
    except Exception:
        return dict(_PREPROCESS_DEFAULTS)


_PREPROCESS_CFG: dict = _load_preprocess_config()


def _preprocess_audio(
    audio: np.ndarray,
    overlap_tail: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """
    Preprocess an audio chunk before Whisper inference to prevent prefix clipping.

    Pipeline order:
      1. Overlap — prepend tail from previous chunk
      2. Leading silence pad — prepend zeros
      3. Fade-in — linear ramp on first N samples
      4. Pre-emphasis — high-pass filter y[n] = x[n] - coeff * x[n-1]
      5. Normalisation — scale to target peak

    Returns float32 ndarray ready for Whisper.
    """
    # 1. Overlap — prepend previous chunk's tail
    if overlap_tail is not None and len(overlap_tail) > 0:
        audio = np.concatenate([overlap_tail, audio])

    # 2. Leading silence pad
    pad_samples = int(cfg["leading_silence_pad_seconds"] * 16_000)
    if pad_samples > 0:
        audio = np.concatenate([np.zeros(pad_samples, dtype=np.float32), audio])

    # Ensure float32
    audio = audio.astype(np.float32, copy=False)

    # 3. Fade-in — linear ramp
    fade_samples = int(cfg["fade_in_ms"] * 16)  # 16 kHz → 16 samples/ms
    if fade_samples > 0 and len(audio) > 0:
        n = min(fade_samples, len(audio))
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp

    # 4. Pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
    coeff = cfg["pre_emphasis_coefficient"]
    if coeff > 0 and len(audio) > 1:
        audio = np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)

    # 5. Normalisation — scale to target peak
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    target = cfg["normalisation_peak"]
    if peak > 1e-7 and target > 0:
        audio = audio * (target / peak)

    return audio


def set_rms_minimum_for_device(
    device_name: str,
    calibrated_threshold: float,
) -> float:
    """
    Set _HARD_RMS_MINIMUM based on the detected audio device.

    BT/headset mics produce strong signal → aggressive gate (0.0005).
    Intel/laptop mics produce weak signal → permissive gate (0.00003),
    relying on Whisper confidence (Layer 2) to catch hallucinations.
    Unknown devices → 10% of calibrated threshold, floored at 0.00003.

    Returns the new _HARD_RMS_MINIMUM value.
    """
    global _HARD_RMS_MINIMUM
    device_lower = (device_name or "").lower()

    bt_keywords = [
        "airpods", "bluetooth", "wireless", "headset", "find my",
        "erza", "jade", "earbuds", "buds", "jabra", "sony", "bose",
        "sennheiser", "bt ",
    ]
    intel_keywords = [
        "intel", "smart sound", "array", "realtek", "laptop", "internal",
    ]

    if any(k in device_lower for k in bt_keywords):
        _HARD_RMS_MINIMUM = 0.00005
        logger.info("BT device detected ('%s') — hard_rms_minimum=0.00005", device_name)
    elif any(k in device_lower for k in intel_keywords):
        _HARD_RMS_MINIMUM = 0.00003
        logger.info("Intel/laptop mic detected ('%s') — hard_rms_minimum=0.00003", device_name)
    else:
        _HARD_RMS_MINIMUM = max(calibrated_threshold * 0.1, 0.00003)
        logger.info("Unknown device ('%s') — hard_rms_minimum=%.5f", device_name, _HARD_RMS_MINIMUM)

    return _HARD_RMS_MINIMUM

# Average log-probability across all Whisper segments.  Lower values mean
# Whisper is uncertain — likely hallucinating.  -1.0 ≈ 37% confidence.
_LOW_CONFIDENCE_LOGPROB: float = -1.0

# Average no_speech_prob across all segments.  Stricter than the per-segment
# check (_NO_SPEECH_PROB_THRESHOLD=0.90) which only looks at segment[0].
_AVG_NO_SPEECH_THRESHOLD: float = 0.6


@dataclass
class TranscriptChunk:
    """
    A single real-time transcript result emitted every chunk_duration seconds.

    Attributes:
        text:        The transcribed text for this audio window.
        language:    ISO-639-1 code detected by Whisper (e.g. "en", "fr").
        source:      "microphone" or "system" — identifies the speaker origin.
        timestamp:   Unix timestamp of when this audio window started.
        confidence:  Estimated transcription confidence in [0.0, 1.0], derived
                     from Whisper's average log-probability across segments.
        rms:         Average RMS energy of the audio chunk.
        is_reliable: True if chunk passed all confidence/RMS checks — safe for
                     Key Facts and spaCy entity extraction.
    """
    text:        str
    language:    str
    source:      str
    timestamp:   float
    confidence:  float
    rms:         float = 0.0
    is_reliable: bool  = True

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for WebSocket broadcast."""
        return {
            "type":        "transcript",
            "text":        self.text,
            "language":    self.language,
            "source":      self.source,
            "speaker":     "You" if self.source == "microphone" else "Them",
            "timestamp":   self.timestamp,
            "confidence":  self.confidence,
            "rms":         self.rms,
            "is_reliable": self.is_reliable,
        }


# ---------------------------------------------------------------------------
# Process pool for GIL-free Whisper inference
# ---------------------------------------------------------------------------
# Whisper inference is CPU-bound and holds the Python GIL for seconds,
# blocking the asyncio event loop and causing WebSocket disconnects (~90 s).
# Running inference in a separate process avoids GIL contention entirely.
# The single worker process caches the model after first load.

_process_pool: Optional[ProcessPoolExecutor] = None


def _get_process_pool() -> ProcessPoolExecutor:
    """Get or create the single-worker process pool for Whisper inference."""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=1)
    return _process_pool


def shutdown_process_pool() -> None:
    """Shut down the Whisper worker process (call on server shutdown)."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=False)
        _process_pool = None


# ── Worker function (runs in a SEPARATE process) ─────────────────────────
# Must be a module-level function (picklable).  The model is cached in a
# process-global so it is loaded once and reused across calls.

_worker_model = None
_worker_model_size = None


def _whisper_worker(audio_data, model_size, language, initial_prompt, fp16):
    """Run Whisper transcribe() in an isolated process — GIL-free.

    The model is cached in process-globals so it survives across calls.
    Exceptions from transcribe() are caught and re-raised as RuntimeError
    so the worker process itself never crashes (which would force a model
    reload on the next call).
    """
    import os
    global _worker_model, _worker_model_size
    if _worker_model is None or _worker_model_size != model_size:
        import whisper as _w
        _worker_model = _w.load_model(model_size)
        _worker_model_size = model_size
        print(f"[whisper-worker pid={os.getpid()}] Loaded model '{model_size}'", flush=True)
    try:
        return _worker_model.transcribe(
            audio_data, verbose=False, fp16=fp16,
            language=language, initial_prompt=initial_prompt,
        )
    except Exception as exc:
        # Re-raise as RuntimeError so the process survives but the caller
        # sees the error.  Do NOT let the process crash — that forces a
        # full model reload on the next call (~8 s).
        raise RuntimeError(f"Whisper transcribe failed: {exc}") from exc


async def warmup_process_pool(model_size: str = "base") -> None:
    """Pre-load the Whisper model in the worker process (non-blocking)."""
    import numpy as np
    dummy = np.zeros(16_000, dtype=np.float32)  # 1 s silence
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _get_process_pool(), _whisper_worker,
        dummy, model_size, "en", "", False,
    )
    logger.info("Whisper worker process warmed up (model=%s).", model_size)


class StreamingTranscriber:
    """
    Chunk-based Whisper transcriber for real-time audio streams.

    Accepts AudioChunk objects from AudioCapture, runs Whisper inference
    on each non-silent chunk, and returns a TranscriptChunk with text,
    language, source label, timestamp, and confidence score.

    All processing is local — no audio or text leaves this machine.

    Usage
    -----
        transcriber = StreamingTranscriber(model_size="medium")
        # In your processing loop:
        result = transcriber.transcribe_chunk(audio_chunk)
        if result:
            print(result.text, result.language, result.confidence)
    """

    # Vocabulary hint passed as initial_prompt to Whisper.  Biases the decoder
    # towards technical terms so homophones like "sass" → "SaaS" are resolved
    # correctly.  Override via the ``initial_prompt`` constructor arg.
    _DEFAULT_INITIAL_PROMPT: str = (
        "SaaS, beta testers, platform, API, sprint, backlog, "
        "stakeholder, timeline, NDA, sandbox"
    )

    def __init__(
        self,
        model_size:        str            = "medium",
        privacy_mode:      bool           = True,
        language:          Optional[str]  = None,
        silence_threshold: Optional[float] = None,
        initial_prompt:    Optional[str]  = None,
    ) -> None:
        """
        Args:
            model_size:        Whisper model to load. "medium" is recommended for
                               meetings — good accuracy with acceptable latency.
            privacy_mode:      When True, logs a local-processing confirmation.
            language:          Optional ISO-639-1 hint (e.g. "en") applied to every
                               chunk. None = Whisper auto-detects per chunk.
            silence_threshold: Custom peak-RMS threshold for the VAD silence gate.
                               None = use the module-level _SILENCE_PEAK_THRESHOLD.
            initial_prompt:    Vocabulary hint for Whisper's decoder.  Biases towards
                               the listed terms. None = use _DEFAULT_INITIAL_PROMPT.
        """
        self._model_size        = model_size
        self._language          = language
        self._privacy_mode      = privacy_mode
        self._silence_threshold = silence_threshold if silence_threshold is not None else _SILENCE_PEAK_THRESHOLD
        self._initial_prompt    = initial_prompt if initial_prompt is not None else self._DEFAULT_INITIAL_PROMPT
        self.last_gate: Optional[str] = None  # reason for last None return

        # Overlap tails for prefix clipping prevention — keyed by source name.
        self._overlap_tails: dict[str, np.ndarray] = {}

        # Energy-based VAD: rolling window of the last 10 chunk peak values.
        # A silent chunk is passed through to Whisper if any of the most recent
        # _VAD_LOOKBACK chunks exceeded the threshold — this keeps mid-sentence
        # gaps from being cut when RMS briefly dips between words.
        self._vad_history: collections.deque[float] = collections.deque(maxlen=10)
        self._VAD_LOOKBACK: int = 3   # how many recent chunks to check

        # Load the model once at construction time — this may download weights
        # on first use (~150 MB for medium) then runs fully offline.
        self._model = _load_whisper_model(model_size)

        if privacy_mode:
            logger.info(
                "PRIVACY: StreamingTranscriber loaded (model=%s). "
                "Whisper runs entirely on-device — no audio leaves this machine.",
                model_size,
            )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(segments: list[dict]) -> float:
        """
        Derive a [0, 1] confidence score from Whisper's avg_logprob values.

        avg_logprob is the mean log-probability across tokens in a segment.
        It lives in (-inf, 0]. We use exp(avg) to map it to (0, 1].
        Typical speech runs −0.3 to −0.6; very uncertain segments reach −1.5+.
        """
        if not segments:
            return 0.0
        avg = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
        return round(min(1.0, max(0.0, math.exp(avg))), 3)

    @staticmethod
    def _is_hallucination(text: str, chunk_duration: float) -> Optional[str]:
        """
        Return a reason string if *text* looks like a Whisper hallucination,
        or None if the text appears legitimate.
        """
        # 1. Non-ASCII characters (Chinese, Arabic, Cyrillic, etc.)
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        if len(text) > 0 and ascii_chars / len(text) < 0.5:
            return "non-ASCII"

        lower = text.lower()

        # 2. Known hallucination phrases
        for phrase in _HALLUCINATION_PHRASES:
            if phrase in lower:
                return f"phrase:{phrase}"

        # 3. Impossible word density
        words = text.split()
        if chunk_duration > 0 and len(words) > _MAX_WORDS_PER_SECOND * chunk_duration:
            return f"word-density:{len(words)}/{chunk_duration:.1f}s"

        # 4. Majority ALL-CAPS words (>50%)
        if len(words) > 2:
            caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
            if caps_count / len(words) > 0.5:
                return "all-caps"

        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_chunk(self, chunk) -> Optional[TranscriptChunk]:
        """
        Transcribe a single AudioChunk and return a TranscriptChunk.

        Returns None (without calling Whisper) when:
          - The chunk's RMS is below the silence threshold
          - Whisper's no_speech_prob is too high
          - Whisper returns empty or punctuation-only text

        Args:
            chunk: An AudioChunk from audio_capture.AudioCapture.

        Returns:
            TranscriptChunk, or None if the chunk should be discarded.
        """
        # ── 1. Energy-based VAD silence gate ──────────────────────────────────
        # System audio (loopback) uses much lower thresholds than mic.
        _is_system = chunk.source.value == "system"
        threshold       = _SYSTEM_SILENCE_PEAK_THRESHOLD if _is_system else self._silence_threshold
        hard_rms_floor  = _SYSTEM_HARD_RMS_MINIMUM if _is_system else _HARD_RMS_MINIMUM

        recent_peaks    = list(self._vad_history)[-self._VAD_LOOKBACK:]
        recently_voiced = any(p >= threshold for p in recent_peaks)
        chunk_voiced    = chunk.peak_rms >= threshold

        # Always record this chunk in history before deciding.
        self._vad_history.append(chunk.peak_rms)

        if not chunk_voiced and not recently_voiced:
            logger.info(
                "GATE  silent  peak=%.5f < threshold=%.4f  avg_rms=%.5f  source=%s",
                chunk.peak_rms, threshold, chunk.rms, chunk.source.value,
            )
            self.last_gate = "silent"
            return None

        if not chunk_voiced and recently_voiced:
            logger.info(
                "GATE  vad-carry  peak=%.5f  recent=%s  source=%s",
                chunk.peak_rms,
                [f"{p:.5f}" for p in recent_peaks],
                chunk.source.value,
            )

        # ── 1b. Hard RMS minimum gate ────────────────────────────────────────
        if chunk.rms < hard_rms_floor:
            logger.info(
                "HARD GATE: RMS %.5f below minimum %.4f — discarding without Whisper call  source=%s",
                chunk.rms, hard_rms_floor, chunk.source.value,
            )
            self.last_gate = "hard_rms"
            return None

        # ── 1c. Audio preprocessing (prefix clipping prevention) ────────────
        source_key = chunk.source.value
        overlap_tail = self._overlap_tails.get(source_key)
        audio_for_whisper = _preprocess_audio(chunk.data, overlap_tail, _PREPROCESS_CFG)
        overlap_samples = int(_PREPROCESS_CFG["chunk_overlap_seconds"] * 16_000)
        self._overlap_tails[source_key] = chunk.data[-overlap_samples:].copy()

        # ── 2. Run Whisper inference ──────────────────────────────────────────
        try:
            # chunk.data is float32, mono, 16 kHz — exactly what Whisper expects.
            # fp16=False ensures CPU compatibility (GPU would use fp16 by default).
            kwargs: dict = {
                "verbose":        False,
                "fp16":           False,           # required for CPU inference
                "language":       self._language or "en",
                "initial_prompt": self._initial_prompt,
            }

            result: dict = self._model.transcribe(audio_for_whisper, **kwargs)
            logger.info(
                "WHISPER  raw=%r  segments=%d  source=%s  rms=%.5f",
                result.get("text", "")[:80],
                len(result.get("segments", [])),
                chunk.source.value,
                chunk.rms,
            )

        except Exception as exc:
            logger.error("Whisper inference error: %s", exc)
            self.last_gate = "error"
            return None

        # ── 3. Post-inference filters ─────────────────────────────────────────
        text     = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Discard empty / noise-only transcripts — retry once for long chunks
        if text in _EMPTY_TRANSCRIPTS:
            chunk_duration = len(chunk.data) / 16_000.0
            if chunk_duration > 1.0:
                logger.info(
                    "RETRY  empty-text on %.1fs chunk  text=%r  source=%s",
                    chunk_duration, text, chunk.source.value,
                )
                try:
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["fp16"] = False
                    result   = self._model.transcribe(audio_for_whisper, **retry_kwargs)
                    text     = result.get("text", "").strip()
                    segments = result.get("segments", [])
                except Exception as exc:
                    logger.error("Whisper retry error: %s", exc)

            if text in _EMPTY_TRANSCRIPTS:
                logger.info(
                    "GATE  empty-text  text=%r  rms=%.4f  source=%s",
                    text, chunk.rms, chunk.source.value,
                )
                self.last_gate = "empty"
                return None

        # Discard chunks where Whisper itself is not confident speech was present
        no_speech = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
        if segments and no_speech > _NO_SPEECH_PROB_THRESHOLD:
            logger.info(
                "GATE  no-speech  prob=%.2f > threshold=%.2f  source=%s",
                no_speech, _NO_SPEECH_PROB_THRESHOLD, chunk.source.value,
            )
            self.last_gate = "no_speech"
            return None

        # ── 3b. Confidence gates (avg across ALL segments) ───────────────────
        avg_logprob   = 0.0
        avg_no_speech = 0.0
        if segments:
            avg_logprob   = sum(s.get("avg_logprob",   -1.0) for s in segments) / len(segments)
            avg_no_speech = sum(s.get("no_speech_prob",  0.0) for s in segments) / len(segments)

            if avg_logprob < _LOW_CONFIDENCE_LOGPROB:
                logger.info(
                    "LOW CONFIDENCE: logprob=%.2f < %.2f  text=%r  source=%s",
                    avg_logprob, _LOW_CONFIDENCE_LOGPROB, text[:80], chunk.source.value,
                )
                self.last_gate = "low_confidence"
                return None

            if avg_no_speech > _AVG_NO_SPEECH_THRESHOLD:
                logger.info(
                    "AVG NO SPEECH: prob=%.2f > %.2f  text=%r  source=%s",
                    avg_no_speech, _AVG_NO_SPEECH_THRESHOLD, text[:80], chunk.source.value,
                )
                self.last_gate = "avg_no_speech"
                return None

        # ── 3c. Hallucination gate ──────────────────────────────────────────
        chunk_duration = len(chunk.data) / 16_000.0
        halluc_reason = self._is_hallucination(text, chunk_duration)
        if halluc_reason:
            logger.info(
                "HALLUCINATION  detected  reason=%s  text=%r  rms=%.4f  source=%s",
                halluc_reason, text[:80], chunk.rms, chunk.source.value,
            )
            self.last_gate = "hallucination"
            return None

        # ── 4. Build and return result ────────────────────────────────────────
        self.last_gate = None
        confidence = self._compute_confidence(segments)
        is_reliable = (
            chunk.rms >= _HARD_RMS_MINIMUM
            and avg_logprob >= _LOW_CONFIDENCE_LOGPROB
            and avg_no_speech <= _AVG_NO_SPEECH_THRESHOLD
        )
        logger.info(
            "PASS  transcribed  conf=%.2f  rms=%.4f  reliable=%s  source=%s  text=%r",
            confidence, chunk.rms, is_reliable, chunk.source.value, text[:60],
        )
        return TranscriptChunk(
            text=text,
            language=result.get("language", "unknown"),
            source=chunk.source.value,   # "microphone" or "system"
            timestamp=chunk.timestamp,
            confidence=confidence,
            rms=round(chunk.rms, 6),
            is_reliable=is_reliable,
        )

    # ── Async API (ProcessPoolExecutor — GIL-free) ─────────────────────────

    async def async_transcribe_chunk(self, chunk) -> Optional[TranscriptChunk]:
        """
        Async version of transcribe_chunk — runs Whisper in a separate process
        so the GIL is not held in the event loop's process.  This prevents
        WebSocket disconnects during CPU-bound Whisper inference.

        Pre/post-processing (VAD gate, hallucination filters) run in the
        event loop thread (fast, no GIL concern).
        """
        # ── 1. Energy-based VAD silence gate ─────────────────────────────────
        # System audio (loopback) uses much lower thresholds than mic.
        _is_system = chunk.source.value == "system"
        threshold       = _SYSTEM_SILENCE_PEAK_THRESHOLD if _is_system else self._silence_threshold
        hard_rms_floor  = _SYSTEM_HARD_RMS_MINIMUM if _is_system else _HARD_RMS_MINIMUM

        recent_peaks = list(self._vad_history)[-self._VAD_LOOKBACK:]
        recently_voiced = any(p >= threshold for p in recent_peaks)
        chunk_voiced    = chunk.peak_rms >= threshold
        self._vad_history.append(chunk.peak_rms)

        if not chunk_voiced and not recently_voiced:
            logger.info(
                "GATE  silent  peak=%.5f < threshold=%.4f  avg_rms=%.5f  source=%s",
                chunk.peak_rms, threshold, chunk.rms, chunk.source.value,
            )
            self.last_gate = "silent"
            return None

        if not chunk_voiced and recently_voiced:
            logger.info(
                "GATE  vad-carry  peak=%.5f  recent=%s  source=%s",
                chunk.peak_rms,
                [f"{p:.5f}" for p in recent_peaks],
                chunk.source.value,
            )

        # ── 1b. Hard RMS minimum gate ────────────────────────────────────────
        if chunk.rms < hard_rms_floor:
            logger.info(
                "HARD GATE: RMS %.5f below minimum %.4f — discarding without Whisper call  source=%s",
                chunk.rms, hard_rms_floor, chunk.source.value,
            )
            self.last_gate = "hard_rms"
            return None

        # ── 1c. Audio preprocessing (prefix clipping prevention) ────────────
        source_key = chunk.source.value
        overlap_tail = self._overlap_tails.get(source_key)
        audio_for_whisper = _preprocess_audio(chunk.data, overlap_tail, _PREPROCESS_CFG)
        overlap_samples = int(_PREPROCESS_CFG["chunk_overlap_seconds"] * 16_000)
        self._overlap_tails[source_key] = chunk.data[-overlap_samples:].copy()

        # ── 2. Whisper inference via ProcessPoolExecutor (GIL-free) ──────────
        try:
            loop = asyncio.get_event_loop()
            result: dict = await loop.run_in_executor(
                _get_process_pool(),
                _whisper_worker,
                audio_for_whisper,
                self._model_size,
                self._language or "en",
                self._initial_prompt,
                False,  # fp16 — CPU requires False
            )
            logger.info(
                "WHISPER  raw=%r  segments=%d  source=%s  rms=%.5f",
                result.get("text", "")[:80],
                len(result.get("segments", [])),
                chunk.source.value,
                chunk.rms,
            )
        except Exception as exc:
            logger.error("Whisper inference error: %s", exc)
            self.last_gate = "error"
            return None

        # ── 3. Post-inference filters ────────────────────────────────────────
        text     = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Discard empty / noise-only transcripts — retry once for long chunks
        if text in _EMPTY_TRANSCRIPTS:
            chunk_duration = len(chunk.data) / 16_000.0
            if chunk_duration > 1.0:
                logger.info(
                    "RETRY  empty-text on %.1fs chunk  text=%r  source=%s",
                    chunk_duration, text, chunk.source.value,
                )
                try:
                    result   = await loop.run_in_executor(
                        _get_process_pool(),
                        _whisper_worker,
                        audio_for_whisper,
                        self._model_size,
                        self._language or "en",
                        self._initial_prompt,
                        False,
                    )
                    text     = result.get("text", "").strip()
                    segments = result.get("segments", [])
                except Exception as exc:
                    logger.error("Whisper retry error: %s", exc)

            if text in _EMPTY_TRANSCRIPTS:
                logger.info(
                    "GATE  empty-text  text=%r  rms=%.4f  source=%s",
                    text, chunk.rms, chunk.source.value,
                )
                self.last_gate = "empty"
                return None

        # First-segment no_speech check
        no_speech = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
        if segments and no_speech > _NO_SPEECH_PROB_THRESHOLD:
            logger.info(
                "GATE  no-speech  prob=%.2f > threshold=%.2f  source=%s",
                no_speech, _NO_SPEECH_PROB_THRESHOLD, chunk.source.value,
            )
            self.last_gate = "no_speech"
            return None

        # Confidence gates (avg across ALL segments)
        avg_logprob   = 0.0
        avg_no_speech = 0.0
        if segments:
            avg_logprob   = sum(s.get("avg_logprob",   -1.0) for s in segments) / len(segments)
            avg_no_speech = sum(s.get("no_speech_prob",  0.0) for s in segments) / len(segments)

            if avg_logprob < _LOW_CONFIDENCE_LOGPROB:
                logger.info(
                    "LOW CONFIDENCE: logprob=%.2f < %.2f  text=%r  source=%s",
                    avg_logprob, _LOW_CONFIDENCE_LOGPROB, text[:80], chunk.source.value,
                )
                self.last_gate = "low_confidence"
                return None

            if avg_no_speech > _AVG_NO_SPEECH_THRESHOLD:
                logger.info(
                    "AVG NO SPEECH: prob=%.2f > %.2f  text=%r  source=%s",
                    avg_no_speech, _AVG_NO_SPEECH_THRESHOLD, text[:80], chunk.source.value,
                )
                self.last_gate = "avg_no_speech"
                return None

        # Hallucination gate
        chunk_duration = len(chunk.data) / 16_000.0
        halluc_reason = self._is_hallucination(text, chunk_duration)
        if halluc_reason:
            logger.info(
                "HALLUCINATION  detected  reason=%s  text=%r  rms=%.4f  source=%s",
                halluc_reason, text[:80], chunk.rms, chunk.source.value,
            )
            self.last_gate = "hallucination"
            return None

        # ── 4. Build and return result ───────────────────────────────────────
        self.last_gate = None
        confidence = self._compute_confidence(segments)
        is_reliable = (
            chunk.rms >= _HARD_RMS_MINIMUM
            and avg_logprob >= _LOW_CONFIDENCE_LOGPROB
            and avg_no_speech <= _AVG_NO_SPEECH_THRESHOLD
        )
        logger.info(
            "PASS  transcribed  conf=%.2f  rms=%.4f  reliable=%s  source=%s  text=%r",
            confidence, chunk.rms, is_reliable, chunk.source.value, text[:60],
        )
        return TranscriptChunk(
            text=text,
            language=result.get("language", "unknown"),
            source=chunk.source.value,
            timestamp=chunk.timestamp,
            confidence=confidence,
            rms=round(chunk.rms, 6),
            is_reliable=is_reliable,
        )

    def close(self) -> None:
        """Release the Whisper model from memory."""
        self._model = None
        logger.info("StreamingTranscriber closed.")
