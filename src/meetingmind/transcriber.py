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

import collections
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

    valid_sizes = {"tiny", "base", "small", "medium", "large"}
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
    logger.info("Model ready.")
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

# Kept for reference — no longer used for the silence gate.
_SILENCE_RMS_THRESHOLD: float = 0.00005

# Whisper segments whose no-speech probability exceeds this are discarded.
_NO_SPEECH_PROB_THRESHOLD: float = 0.80

# Whisper returns these single-character strings for silent/noise-only chunks.
_EMPTY_TRANSCRIPTS: frozenset[str] = frozenset({"", ".", "...", "!", "?", " "})


@dataclass
class TranscriptChunk:
    """
    A single real-time transcript result emitted every chunk_duration seconds.

    Attributes:
        text:       The transcribed text for this audio window.
        language:   ISO-639-1 code detected by Whisper (e.g. "en", "fr").
        source:     "microphone" or "system" — identifies the speaker origin.
        timestamp:  Unix timestamp of when this audio window started.
        confidence: Estimated transcription confidence in [0.0, 1.0], derived
                    from Whisper's average log-probability across segments.
    """
    text:       str
    language:   str
    source:     str
    timestamp:  float
    confidence: float

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for WebSocket broadcast."""
        return {
            "type":       "transcript",
            "text":       self.text,
            "language":   self.language,
            "source":     self.source,
            "speaker":    "You" if self.source == "microphone" else "Them",
            "timestamp":  self.timestamp,
            "confidence": self.confidence,
        }


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

    def __init__(
        self,
        model_size:   str            = "medium",
        privacy_mode: bool           = True,
        language:     Optional[str]  = None,
    ) -> None:
        """
        Args:
            model_size:   Whisper model to load. "medium" is recommended for
                          meetings — good accuracy with acceptable latency.
            privacy_mode: When True, logs a local-processing confirmation.
            language:     Optional ISO-639-1 hint (e.g. "en") applied to every
                          chunk. None = Whisper auto-detects per chunk.
        """
        self._model_size   = model_size
        self._language     = language
        self._privacy_mode = privacy_mode

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
        # Pass the chunk through if its own peak exceeds the threshold OR if
        # any of the last _VAD_LOOKBACK chunks did — this bridges the brief
        # RMS dips that occur between words mid-sentence (inter-word silence).
        # The rolling history stores peak_rms for every chunk (voiced or not)
        # so the window accurately reflects recency.
        recent_peaks = list(self._vad_history)[-self._VAD_LOOKBACK:]
        recently_voiced = any(p >= _SILENCE_PEAK_THRESHOLD for p in recent_peaks)
        chunk_voiced    = chunk.peak_rms >= _SILENCE_PEAK_THRESHOLD

        # Always record this chunk in history before deciding.
        self._vad_history.append(chunk.peak_rms)

        if not chunk_voiced and not recently_voiced:
            logger.info(
                "GATE  silent  peak=%.5f < threshold=%.3f  avg_rms=%.5f  source=%s",
                chunk.peak_rms, _SILENCE_PEAK_THRESHOLD, chunk.rms, chunk.source.value,
            )
            return None

        if not chunk_voiced and recently_voiced:
            logger.info(
                "GATE  vad-carry  peak=%.5f  recent=%s  source=%s",
                chunk.peak_rms,
                [f"{p:.5f}" for p in recent_peaks],
                chunk.source.value,
            )

        # ── 2. Run Whisper inference ──────────────────────────────────────────
        try:
            # chunk.data is float32, mono, 16 kHz — exactly what Whisper expects.
            # fp16=False ensures CPU compatibility (GPU would use fp16 by default).
            kwargs: dict = {
                "verbose":  False,
                "fp16":     False,           # required for CPU inference
                "language": self._language or "en",   # default English; skips language-detection overhead
            }

            result: dict = self._model.transcribe(chunk.data, **kwargs)
            logger.info(
                "WHISPER  raw=%r  segments=%d  source=%s  rms=%.5f",
                result.get("text", "")[:80],
                len(result.get("segments", [])),
                chunk.source.value,
                chunk.rms,
            )

        except Exception as exc:
            logger.error("Whisper inference error: %s", exc)
            return None

        # ── 3. Post-inference filters ─────────────────────────────────────────
        text     = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Discard empty / noise-only transcripts
        if text in _EMPTY_TRANSCRIPTS:
            logger.info(
                "GATE  empty-text  text=%r  rms=%.4f  source=%s",
                text, chunk.rms, chunk.source.value,
            )
            return None

        # Discard chunks where Whisper itself is not confident speech was present
        no_speech = segments[0].get("no_speech_prob", 0.0) if segments else 0.0
        if segments and no_speech > _NO_SPEECH_PROB_THRESHOLD:
            logger.info(
                "GATE  no-speech  prob=%.2f > threshold=%.2f  source=%s",
                no_speech, _NO_SPEECH_PROB_THRESHOLD, chunk.source.value,
            )
            return None

        # ── 4. Build and return result ────────────────────────────────────────
        confidence = self._compute_confidence(segments)
        logger.info(
            "PASS  transcribed  conf=%.2f  rms=%.4f  source=%s  text=%r",
            confidence, chunk.rms, chunk.source.value, text[:60],
        )
        return TranscriptChunk(
            text=text,
            language=result.get("language", "unknown"),
            source=chunk.source.value,   # "microphone" or "system"
            timestamp=chunk.timestamp,
            confidence=confidence,
        )

    def close(self) -> None:
        """Release the Whisper model from memory."""
        self._model = None
        logger.info("StreamingTranscriber closed.")
