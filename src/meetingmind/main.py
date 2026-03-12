"""
main.py
-------
FastAPI application — the real-time audio engine HTTP/WebSocket layer.

Endpoints
---------
GET  /health              Server liveness + meeting state
POST /meeting/start       Begin audio capture and transcription
POST /meeting/stop        End audio capture and transcription
GET  /meeting/summary     Structured summary of the last completed meeting
WS   /ws/transcribe       Stream transcript chunks to connected clients

Running the server
------------------
    # From the project root (MeetingMind/):
    python -m uvicorn meetingmind.main:app --host 0.0.0.0 --port 8000

    # Or via the module directly:
    python -m meetingmind.main

WebSocket message format (JSON)
--------------------------------
    {
      "text":       "What was discussed in Q2...",
      "language":   "en",
      "source":     "microphone",
      "timestamp":  1740484800.123,
      "confidence": 0.91
    }

PRIVACY: Audio is processed entirely on this machine. The WebSocket stream
         carries transcript text only — no raw audio is transmitted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as _queue_module
import sys
import time
import uuid as _uuid_mod
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src/ is importable.
# parents[0] = src/meetingmind/   parents[1] = src/   parents[2] = project root
_SRC          = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND     = _PROJECT_ROOT / "frontend"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Logging — ensure meetingmind.* INFO messages are visible when running under
# uvicorn.  uvicorn only attaches handlers to its own uvicorn.* loggers; it
# does NOT add a handler to the root logger, so application-level logger.info()
# calls are silently swallowed without this.
# ---------------------------------------------------------------------------
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
)
logging.root.addHandler(_log_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# RMS threshold shown as the marker line in the UI meter.
# The actual silence gate now uses peak_rms in transcriber._SILENCE_PEAK_THRESHOLD (0.001).
_SILENCE_THRESHOLD: float = 0.00005

# ---------------------------------------------------------------------------
# Transcription config — single place to tune defaults
# ---------------------------------------------------------------------------
_WHISPER_MODEL_DEFAULT:  str   = "tiny.en"   # tiny.en|base|small|medium|large
_CHUNK_SECONDS_DEFAULT:  float = 3.0      # seconds of audio per Whisper call
_LANGUAGE_DEFAULT:       str   = "en"     # skips per-chunk language detection
_MAX_CONCURRENT_WHISPER: int   = 1        # sequential — 1 Whisper task at a time

_WHISPER_TIMEOUT:        float = 10.0     # seconds before dropping a Whisper call
_WHISPER_SLOW_THRESHOLD: float = 3.0     # avg Whisper time above this triggers slow warning
_MIN_CHUNK_SECONDS:      float = 1.0      # chunks shorter than this are accumulated
_COACHING_COOLDOWN_S:    float = 60.0     # seconds before the same trigger can re-fire
_SPEAKER_WINDOW_SIZE:    int   = 10       # chunks in rolling talk-ratio window
_GUEST_QUEUE_MAXSIZE:    int   = 32       # ~96 s of backpressure at 3 s/chunk
_WS_PING_INTERVAL:       float = 15.0    # server→client keepalive ping every 15 s
_WEAK_RMS_THRESHOLD:     float = 0.0001  # RMS below this = "weak signal"
_WEAK_RMS_CONSECUTIVE:   int   = 5       # consecutive weak readings before warning
_RECALIB_WINDOW:         int   = 20      # rolling window for auto-recalibrate check
_RECALIB_SILENT_RATIO:   float = 0.50    # gate_silent > 50% triggers recalibration
_RECALIB_FACTOR:         float = 0.50    # multiply threshold by this on recalibration
_WHISPER_MIN_INTERVAL:   float = 1.5     # min seconds between Whisper calls (CPU rest)
_SYSTEM_AUDIO_WARN_DELAY: float = 120.0  # seconds before [Them] audio warning fires

# Dedicated thread pool for audio queue reads.
# asyncio.to_thread uses the *default* executor, which is shared with every
# other asyncio.to_thread call — including Claude API calls inside
# _process_ai_detection / _process_scope_detection / _process_timeline_detection.
# If many API tasks are in-flight, the default pool is exhausted and
# get_chunk cannot obtain a thread, stalling the transcription loop.
# A tiny dedicated pool guarantees audio I/O is never starved.
from concurrent.futures import ThreadPoolExecutor as _TPE
_audio_io_pool = _TPE(max_workers=2, thread_name_prefix="audio-io")


def _task_done_callback(task: asyncio.Task) -> None:
    """Log unhandled exceptions from fire-and-forget asyncio tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Background task %r crashed: %s", task.get_name(), exc, exc_info=exc)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages all active WebSocket connections and broadcasts messages to them.

    Dead connections (clients that closed the browser / lost network) are
    detected on the next broadcast and silently removed.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(
            "WebSocket client connected. Active connections: %d",
            len(self._connections),
        )

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            len(self._connections),
        )

    async def broadcast(self, data) -> None:
        """Send a JSON payload to every connected client.

        Accepts a dict (auto-serialised) or a pre-serialised JSON string.
        Dead connections are removed automatically.
        """
        if not self._connections:
            return
        message = data if isinstance(data, str) else json.dumps(data)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)
            logger.warning("Removed dead WS connection. Active: %d", len(self._connections))

    @property
    def active_connections(self) -> list[WebSocket]:
        return self._connections

    @property
    def count(self) -> int:
        return len(self._connections)


async def _ws_ping_loop(manager: "ConnectionManager") -> None:
    """
    Server-initiated keepalive: sends {"type":"ping","timestamp":...} every
    _WS_PING_INTERVAL seconds.  NEVER dies — exceptions are caught and logged
    inside the loop so it runs for the entire server lifetime.
    """
    while True:
        try:
            await asyncio.sleep(_WS_PING_INTERVAL)
            if len(manager.active_connections) == 0:
                continue  # skip if no clients
            await manager.broadcast(json.dumps({
                "type":      "ping",
                "timestamp": time.time(),
            }))
            logger.info("Ping sent to %d client(s)", len(manager.active_connections))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Ping loop error: %s", exc)
            continue  # NEVER die


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

_manager = ConnectionManager()

# Device detection result — populated once at startup.
from meetingmind.device_detector import DetectionResult, detect_devices
_detected: Optional[DetectionResult] = None

from meetingmind.suggestion_engine import (
    check_time_triggers,
    detect_coaching,
    detect_coaching_resolution,
    detect_decision,
    extract_action_items,
)

# Knowledge base + context engine — initialised in lifespan, None if unavailable.
from meetingmind.knowledge_base import KnowledgeBase
from meetingmind.context_engine import ContextEngine
_kb:  Optional[KnowledgeBase]  = None
_ctx: Optional[ContextEngine]  = None

from meetingmind.guest_session import generate_pin, decode_guest_audio
from meetingmind.fact_checker import FactChecker
_fact_checker: Optional[FactChecker] = None

# Meeting lifecycle state — mutated by /meeting/start and /meeting/stop.
_meeting: dict = {
    "active":             False,
    "capture":            None,   # AudioCapture instance
    "transcriber":        None,   # StreamingTranscriber instance (None while loading)
    "task":               None,   # asyncio.Task for _transcription_loop (None while loading)
    "rms_task":           None,   # asyncio.Task for _rms_broadcast_loop
    "load_task":          None,   # asyncio.Task for _whisper_load_background
    "start_time":         None,   # Unix timestamp
    "model":              None,   # Whisper model size in use
    "model_loading":      False,  # True while Whisper model is being loaded
    "meeting_id":         None,   # str "YYYYMMDD_HHMMSS", set at meeting_start
    "transcript_chunks":  [],     # list[str] accumulated by _transcription_loop
    "decisions":          [],     # list[dict] decision events detected this meeting
    "speaker_counts":      {"microphone": 0, "system": 0},  # cumulative chunks per source
    "speaker_window":      [],    # source labels for last N chunks (talk-ratio window)
    "coaching_cooldowns":  {},    # {trigger_id: last_fired_timestamp}
    "coaching_history":    [],    # all fired coaching events this meeting
    "coaching_dismissals": [],    # [{prompt_id, dismissed_at}]
    "coaching_resolved":   set(), # coaching event IDs auto-resolved this meeting
    "coaching_seq":        0,     # monotonic counter for unique coaching event IDs
    "time_coach_task":     None,  # asyncio.Task for _coaching_time_loop
    # Guest phone mic
    "guest_pin":           None,  # str | None — valid until meeting stops
    "guest_connected":     False,
    "guest_queue":         None,  # _queue_module.Queue[AudioChunk]
    "guest_task":          None,  # asyncio.Task for _guest_transcription_loop
    "guest_ws":            None,  # WebSocket | None
    "fact_checks":         [],    # list[dict] fact-check alerts this meeting
    "system_audio_monitor_task": None,  # asyncio.Task for _monitor_system_audio
}

# Last meeting summary — populated by meeting_stop, served by GET /meeting/summary.
# Persists until the next meeting stop overwrites it.
_last_summary: Optional[dict] = None

# Active role — set by POST /settings/role, defaults to "PM".
_active_role: str = "PM"

# ---------------------------------------------------------------------------
# Project Brief — S8-001
# ---------------------------------------------------------------------------
_active_project: Optional[dict] = None
_BRIEFS_DIR = _PROJECT_ROOT / "data" / "project_briefs"

# ---------------------------------------------------------------------------
# Guest viewer link — S7-003
# ---------------------------------------------------------------------------
# { token_str: { "active": True, "guests": ["Alice", "Bob"] } }
_guest_tokens: dict[str, dict] = {}
# WebSocket connections for guest viewers, keyed by token
_guest_viewer_ws: dict[str, list[WebSocket]] = {}

from meetingmind.relay_stub import RelayStub
_relay = RelayStub()

# ---------------------------------------------------------------------------
# S8-002: Action Item & Decision Capture (AI-powered, in-memory per meeting)
# ---------------------------------------------------------------------------
_action_items: list[dict] = []   # ActionItem dicts, reset on meeting_start
_decisions_ai: list[dict] = []   # Decision dicts (AI-detected), reset on meeting_start

# S8-003: Scope Creep Detection (AI-powered, in-memory per meeting)
# ---------------------------------------------------------------------------
_scope_alerts: list[dict] = []   # ScopeAlert dicts, reset on meeting_start

# S8-004: Timeline & Delay Flag (AI-powered, in-memory per meeting)
# ---------------------------------------------------------------------------
_timeline_alerts: list[dict] = []  # TimelineAlert dicts, reset on meeting_start

def _load_tailscale_ip() -> str:
    """Read guest.tailscale_ip from config/settings.toml, default 'localhost'."""
    cfg_path = Path(__file__).parents[2] / "config" / "settings.toml"
    if cfg_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                return "localhost"
        try:
            with open(cfg_path, "rb") as f:
                cfg = tomllib.load(f)
            return cfg.get("guest", {}).get("tailscale_ip", "localhost")
        except Exception:
            return "localhost"
    return "localhost"

_TAILSCALE_IP: str = _load_tailscale_ip()


def _load_audio_thresholds() -> dict:
    """Read [audio] threshold caps from config/settings.toml."""
    defaults = {"bt_mic_max_threshold": 0.0005, "mic_max_threshold": 0.002}
    cfg_path = Path(__file__).parents[2] / "config" / "settings.toml"
    if cfg_path.exists():
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
            return {
                "bt_mic_max_threshold": float(audio.get("bt_mic_max_threshold", defaults["bt_mic_max_threshold"])),
                "mic_max_threshold": float(audio.get("mic_max_threshold", defaults["mic_max_threshold"])),
            }
        except Exception:
            return defaults
    return defaults


_AUDIO_THRESHOLDS: dict = _load_audio_thresholds()

# Onboarding profile — set by POST /onboarding/profile.
_user_name: Optional[str] = None
_user_vocab_hints: Optional[str] = None

# Dev profile — auto-loaded on startup if KB is empty.
_DEV_PROFILE = {
    "name": "Andrea",
    "role": "PM",
    "context": (
        "Andrea is a Project Manager implementing AI projects "
        "using PMBOK, Scrum, and CPMAI methodologies. She manages "
        "project scope, budget, timeline, risks, and team "
        "delivery. Key PM terms: scope creep, change request, "
        "risk register, milestone, sprint, backlog, stakeholder, "
        "deliverable, acceptance criteria, go-live."
    ),
}

_kb_lock = asyncio.Lock()  # prevents double-init if two requests arrive simultaneously


async def _ensure_kb() -> None:
    """Lazily initialise KnowledgeBase + ContextEngine on first use.

    Safe to call from any endpoint. No-ops if already initialised or if a
    previous attempt failed (history stays disabled rather than retrying).
    """
    global _kb, _ctx
    if _kb is not None:
        return
    async with _kb_lock:
        if _kb is not None:   # double-checked locking
            return
        try:
            _kb  = await asyncio.to_thread(KnowledgeBase)
            _ctx = await asyncio.to_thread(ContextEngine)
            logger.info("KnowledgeBase and ContextEngine initialised (lazy).")
            # Ingest PM global KB if available
            try:
                count = await asyncio.to_thread(_kb.ingest_pm_global_kb)
                if count > 0:
                    logger.info("PM global KB ingested: %d sections.", count)
            except Exception as exc:
                logger.warning("PM global KB ingestion failed: %s", exc)
        except Exception as exc:
            logger.error("KnowledgeBase init failed (history disabled): %s", exc)
            _kb = _ctx = None


_pm_global_last_seeded: Optional[str] = None  # ISO timestamp of last reseed


async def _verify_pm_global_kb() -> None:
    """Verify PM global KB on startup; re-ingest if empty."""
    global _pm_global_last_seeded
    try:
        await _ensure_kb()
        if _kb is None:
            logger.warning("PM Global KB: KB unavailable — skipping verification.")
            return
        count = await asyncio.to_thread(_kb.pm_global_doc_count)
        if count > 0:
            logger.info("PM Global KB: already seeded (%d documents), skipping.", count)
        else:
            logger.info("PM Global KB: 0 documents — re-ingesting from pm_global/.")
            n = await asyncio.to_thread(_kb.ingest_pm_global_kb)
            if n > 0:
                _pm_global_last_seeded = datetime.now(timezone.utc).isoformat()
            logger.info("PM Global KB: %d documents loaded.", n)
    except Exception as exc:
        logger.warning("PM Global KB verification failed (non-blocking): %s", exc)


async def _load_dev_profile() -> None:
    """Auto-load _DEV_PROFILE on startup if no onboarding profile is set.

    Sets _user_name, _active_role, ingests context into KB, and builds
    Whisper vocab hints from spaCy NER — same logic as POST /onboarding/profile.
    """
    global _user_name, _user_vocab_hints, _active_role

    # Skip if a profile was already loaded (e.g. from a previous startup cycle)
    if _user_name is not None:
        logger.info("Dev profile skipped — user profile already set (%s).", _user_name)
        return

    _user_name   = _DEV_PROFILE["name"]
    _active_role = _DEV_PROFILE["role"]
    context      = _DEV_PROFILE["context"]

    # Ingest into KB
    await _ensure_kb()
    if _kb is not None:
        seed_text = f"User: {_user_name}\nRole: {_active_role}\n\n{context}"
        try:
            await asyncio.to_thread(_kb.ingest_text, "onboarding_seed", seed_text, "summaries")
        except Exception as exc:
            logger.warning("Dev profile KB ingest failed: %s", exc)

    # spaCy NER extraction → vocab hints
    try:
        from meetingmind.spacy_extractor import extract_entities
        entities = await asyncio.to_thread(extract_entities, context)
    except Exception as exc:
        logger.warning("Dev profile spaCy extraction failed: %s", exc)
        entities = {"people": [], "orgs": [], "money": []}

    vocab_terms: list[str] = []
    for name in entities.get("people", []):
        if name not in vocab_terms:
            vocab_terms.append(name)
    for org in entities.get("orgs", []):
        if org not in vocab_terms:
            vocab_terms.append(org)
    for money in entities.get("money", []):
        if money not in vocab_terms:
            vocab_terms.append(money)

    if vocab_terms:
        from meetingmind.transcriber import StreamingTranscriber
        default_hints = StreamingTranscriber._DEFAULT_INITIAL_PROMPT
        _user_vocab_hints = default_hints + ", " + ", ".join(vocab_terms)
    else:
        _user_vocab_hints = None

    entities_count = len(entities.get("people", [])) + len(entities.get("orgs", [])) + len(entities.get("money", []))
    logger.info(
        "Dev profile loaded: name=%s, role=%s, entities=%d, vocab_hints_len=%d",
        _user_name, _active_role, entities_count,
        len(_user_vocab_hints) if _user_vocab_hints else 0,
    )


# Pipeline diagnostic counters — reset on each /meeting/start.
_stats: dict = {
    "chunks_from_queue":        0,    # AudioChunks pulled from PyAudio queue
    "chunks_filtered":          0,    # Returned None by transcribe_chunk (silence/noise)
    "chunks_transcribed":       0,    # TranscriptChunks produced by Whisper
    "chunks_broadcast":         0,    # Successfully broadcast over WebSocket
    "last_rms":                 0.0,  # RMS of the most recent chunk (before filtering)
    "recent_rms":               [],   # RMS of last 5 chunks — use to tune silence threshold
    "whisper_times":            [],   # rolling last-10 Whisper durations (seconds)
    "avg_whisper_time_s":       0.0,  # mean of whisper_times
    "first_rms_delay_s":        None, # seconds from meeting start → first RMS broadcast
    "first_transcript_delay_s": None, # seconds from meeting start → first transcript
    # Per-gate diagnostic counters — see which gate is filtering the most.
    "gate_silent":              0,    # VAD silence gate (peak_rms below threshold)
    "gate_timeout":             0,    # Whisper took too long (>5s)
    "gate_no_speech":           0,    # Whisper no_speech_prob too high
    "gate_empty":               0,    # Empty/punctuation-only text from Whisper
    "gate_error":               0,    # Whisper raised an exception
    "gate_hallucination":       0,    # Hallucinated text detected and discarded
    # Per-source (system audio) counters — isolate loopback diagnostics from mic.
    "system_chunks_from_queue":  0,
    "system_chunks_filtered":    0,
    "system_chunks_transcribed": 0,
    "system_gate_silent":        0,
    "system_gate_hard_rms":      0,
    "system_gate_no_speech":     0,
    "system_gate_empty":         0,
    "system_gate_error":         0,
    "system_gate_hallucination": 0,
    "system_gate_low_confidence":   0,
    "system_gate_avg_no_speech":    0,
    "system_last_rms":           0.0,
    "system_recent_rms":         [],
}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Coaching helpers shared by _transcription_loop and _coaching_time_loop
# ---------------------------------------------------------------------------

async def _fire_coaching(coaching: dict, timestamp: float) -> None:
    """
    Apply cooldown, assign a unique ID, store in history, and broadcast.
    Shared by both text-triggered and time-triggered coaching events.
    Never raises — broadcast errors are logged and swallowed so the
    transcription loop is never interrupted by a coaching failure.
    """
    try:
        tid = coaching["trigger_id"]
        now = time.time()
        last = _meeting["coaching_cooldowns"].get(tid, 0.0)
        if now - last < _COACHING_COOLDOWN_S:
            return   # still in cooldown — suppress

        _meeting["coaching_cooldowns"][tid] = now
        _meeting["coaching_seq"] += 1
        cid = f"{tid}_{_meeting['coaching_seq']}"

        event = {
            "id":         cid,
            "type":       "coaching",
            "trigger_id": tid,
            "prompt":     coaching["prompt"],
            "confidence": coaching["confidence"],
            "matched":    coaching.get("matched"),
            "timestamp":  timestamp,
        }
        _meeting["coaching_history"].append(event)
        await _manager.broadcast(event)
        logger.info("COACHING %r fired: id=%r  prompt=%r", tid, cid, coaching["prompt"])
    except Exception as exc:
        logger.warning("_fire_coaching error (non-fatal): %s", exc)


async def _monitor_system_audio() -> None:
    """
    S8-009: Wait ``_SYSTEM_AUDIO_WARN_DELAY`` seconds after meeting start,
    then check whether any [Them] audio chunks have been transcribed.
    If ``system_chunks_transcribed`` is still 0, broadcast a one-shot
    ``system_audio_warning`` WebSocket message so the frontend can show
    a banner prompting the user to use the paste input instead.
    Fires at most once per meeting session.
    """
    try:
        await asyncio.sleep(_SYSTEM_AUDIO_WARN_DELAY)
        if not _meeting.get("active"):
            return
        if _stats.get("system_chunks_transcribed", 0) > 0:
            logger.info("System audio monitor: [Them] chunks detected — no warning needed.")
            return
        logger.warning(
            "System audio monitor: 0 [Them] chunks transcribed after %.0f s — broadcasting warning.",
            _SYSTEM_AUDIO_WARN_DELAY,
        )
        await _manager.broadcast({
            "type": "system_audio_warning",
            "message": "No [Them] audio detected",
            "detail": "Check SYSTEM audio routing or use the paste input below the transcript",
        })
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("System audio monitor crashed: %s", exc)


async def _coaching_time_loop() -> None:
    """
    Polls every 60 s and fires time-based coaching triggers (e.g. 20-min
    no-decision alert).  Runs independently of the transcription loop.
    """
    logger.info("Coaching time loop started.")
    try:
        while _meeting.get("active"):
            await asyncio.sleep(_COACHING_COOLDOWN_S)
            if not _meeting.get("active"):
                break
            start = _meeting.get("start_time")
            if start is None:
                continue
            elapsed        = time.time() - start
            decisions_count = len(_meeting.get("decisions", []))
            for coaching in check_time_triggers(elapsed, decisions_count):
                await _fire_coaching(coaching, timestamp=time.time())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Coaching time loop crashed: %s", exc)
    logger.info("Coaching time loop stopped.")


# Background processing loop
# ---------------------------------------------------------------------------

async def _transcription_loop() -> None:
    """
    Task B: pulls AudioChunks from the capture queue, runs Whisper sequentially
    in a thread pool, and broadcasts results to WebSocket clients.

    Sequential (MAX_CONCURRENT=1): one chunk at a time, 5 s hard timeout.
    RMS broadcasting runs independently via _rms_broadcast_loop().
    """
    logger.info("Transcription loop started.")
    _max_restarts  = 3
    _restart_count = 0
    _last_whisper_call: float = 0.0            # monotonic time of last Whisper call
    _accum_data: Optional[np.ndarray] = None   # short-chunk accumulation buffer
    _accum_source = None                         # AudioSource of buffered data
    _gate_window: list[str] = []               # rolling window of gate outcomes for recalib
    _recalib_done = False                       # only auto-recalibrate once per meeting

    _min_samples = int(_MIN_CHUNK_SECONDS * 16_000)  # 0.5 s @ 16 kHz = 8 000

    while _meeting["active"]:
        try:
            while _meeting["active"]:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    _audio_io_pool,
                    _meeting["capture"].get_chunk, 1.0,
                )
                if chunk is None:
                    continue

                _stats["chunks_from_queue"] += 1
                _is_sys_chunk = chunk.source.value == "system"
                if _is_sys_chunk:
                    _stats["system_chunks_from_queue"] = _stats.get("system_chunks_from_queue", 0) + 1

                # ── Accumulate short chunks ───────────────────────────────
                # If the audio buffer has < 0.5 s of data, Whisper returns
                # empty text.  Prepend any accumulated leftovers, then check
                # if the combined buffer is long enough.
                # IMPORTANT: only merge if same source — never mix MIC + SYSTEM.
                if _accum_data is not None:
                    if _accum_source == chunk.source:
                        chunk.data = np.concatenate([_accum_data, chunk.data])
                        chunk.rms  = float(np.sqrt(np.mean(chunk.data.astype(np.float64) ** 2)))
                        chunk.peak_rms = float(np.max(np.abs(chunk.data)))
                    else:
                        logger.info(
                            "ACCUM  source mismatch (%s → %s) — discarding %d buffered samples",
                            _accum_source.value if _accum_source else "?",
                            chunk.source.value,
                            len(_accum_data),
                        )
                    _accum_data = None
                    _accum_source = None

                if len(chunk.data) < _min_samples:
                    _accum_data = chunk.data
                    _accum_source = chunk.source
                    logger.info(
                        "ACCUM  samples=%d < min=%d (%.2fs) — buffering for next chunk",
                        len(chunk.data), _min_samples, len(chunk.data) / 16_000,
                    )
                    continue

                logger.info(
                    "CHUNK  rms=%.5f  peak=%.5f  samples=%d  source=%s",
                    chunk.rms, chunk.peak_rms, len(chunk.data), chunk.source.value,
                )

                try:
                    # Enforce minimum gap between Whisper calls to rest CPU.
                    _now = time.monotonic()
                    _gap = _now - _last_whisper_call
                    if _gap < _WHISPER_MIN_INTERVAL:
                        await asyncio.sleep(_WHISPER_MIN_INTERVAL - _gap)

                    _t0     = time.time()
                    result  = await asyncio.wait_for(
                        _meeting["transcriber"].async_transcribe_chunk(chunk),
                        timeout=_WHISPER_TIMEOUT,
                    )
                    _last_whisper_call = time.monotonic()
                    elapsed = time.time() - _t0
                    times   = _stats.setdefault("whisper_times", [])
                    times.append(round(elapsed, 3))
                    _stats["whisper_times"]      = times[-10:]
                    avg_t = round(
                        sum(_stats["whisper_times"]) / len(_stats["whisper_times"]), 3
                    )
                    _stats["avg_whisper_time_s"] = avg_t

                    # Slow Whisper warning — fire once when avg exceeds threshold
                    # after at least 3 samples for a stable reading.
                    # Skip if tiny model is already active (nothing faster to suggest).
                    _current_model = _meeting.get("model", "")
                    if (
                        avg_t > _WHISPER_SLOW_THRESHOLD
                        and len(_stats["whisper_times"]) >= 3
                        and not _stats.get("_slow_whisper_warned")
                        and _current_model != "tiny.en"
                    ):
                        _stats["_slow_whisper_warned"] = True
                        logger.warning(
                            "Whisper avg %.1fs > %.1fs threshold — suggesting tiny.en model.",
                            avg_t, _WHISPER_SLOW_THRESHOLD,
                        )
                        try:
                            await _manager.broadcast({
                                "type":    "slow_whisper",
                                "message": (
                                    f"Transcription slow (avg {avg_t:.1f}s per chunk) "
                                    f"\u2014 switch to tiny.en model for better real-time performance"
                                ),
                                "avg_whisper_time": avg_t,
                            })
                        except Exception as exc:
                            logger.warning("Broadcast error (slow_whisper): %s", exc)
                except asyncio.TimeoutError:
                    logger.warning("SKIP: whisper timeout >%.0f s — chunk skipped (%s)", _WHISPER_TIMEOUT, chunk.source.value)
                    _stats["chunks_filtered"] += 1
                    _stats["gate_timeout"]    += 1
                    continue
                except Exception as exc:
                    logger.error("transcribe_chunk error: %s", exc)
                    _stats["chunks_filtered"] += 1
                    _stats["gate_error"]      += 1
                    continue

                if result is None:
                    _stats["chunks_filtered"] += 1
                    if _is_sys_chunk:
                        _stats["system_chunks_filtered"] = _stats.get("system_chunks_filtered", 0) + 1
                    # Read per-gate reason from the transcriber.
                    gate = getattr(_meeting.get("transcriber"), "last_gate", None)
                    if gate == "silent":
                        _stats["gate_silent"] += 1
                    elif gate == "no_speech":
                        _stats["gate_no_speech"] += 1
                    elif gate == "empty":
                        _stats["gate_empty"] += 1
                    elif gate == "error":
                        _stats["gate_error"] += 1
                    elif gate == "hallucination":
                        _stats["gate_hallucination"] += 1
                    elif gate == "hard_rms":
                        _stats["gate_hard_rms"] = _stats.get("gate_hard_rms", 0) + 1
                        # Persistent UI warning after 5 hard_rms gates — MIC ONLY
                        # (system hard_rms gates are expected at very low RMS).
                        if (
                            not _is_sys_chunk
                            and _stats["gate_hard_rms"] == 5
                            and not _stats.get("_mic_quality_warned")
                        ):
                            _stats["_mic_quality_warned"] = True
                            try:
                                await _manager.broadcast({
                                    "type":    "mic_quality_warning",
                                    "message": (
                                        "Weak mic signal detected \u2014 "
                                        "switch to Bluetooth headset "
                                        "for accurate fact-checking"
                                    ),
                                    "gate_hard_rms": _stats["gate_hard_rms"],
                                })
                            except Exception as exc:
                                logger.warning("Broadcast error (mic_quality_warning): %s", exc)
                    elif gate == "low_confidence":
                        _stats["gate_low_confidence"] = _stats.get("gate_low_confidence", 0) + 1
                    elif gate == "avg_no_speech":
                        _stats["gate_avg_no_speech"] = _stats.get("gate_avg_no_speech", 0) + 1

                    # ── Per-source gate tracking ──────────────────────────
                    if _is_sys_chunk and gate:
                        sys_key = f"system_gate_{gate}"
                        _stats[sys_key] = _stats.get(sys_key, 0) + 1

                    # ── Auto-recalibrate: track gate outcomes (MIC ONLY) ──
                    # System silence is expected — don't let it inflate
                    # the MIC recalibration window.
                    if not _is_sys_chunk:
                        _gate_window.append(gate or "filtered")
                    if len(_gate_window) > _RECALIB_WINDOW:
                        _gate_window.pop(0)
                    if (
                        not _recalib_done
                        and len(_gate_window) >= _RECALIB_WINDOW
                    ):
                        silent_count = sum(1 for g in _gate_window if g == "silent")
                        silent_ratio = silent_count / len(_gate_window)
                        if silent_ratio > _RECALIB_SILENT_RATIO:
                            transcriber = _meeting.get("transcriber")
                            if transcriber:
                                old_th = transcriber._silence_threshold
                                new_th = old_th * _RECALIB_FACTOR
                                transcriber._silence_threshold = new_th
                                _recalib_done = True
                                logger.warning(
                                    "AUTO-RECALIBRATE: gate_silent %d/%d (%.0f%%) "
                                    "— threshold %.5f → %.5f",
                                    silent_count, len(_gate_window),
                                    silent_ratio * 100, old_th, new_th,
                                )
                                try:
                                    await _manager.broadcast({
                                        "type":    "recalibrated",
                                        "message": (
                                            f"Auto-recalibrated: silence gate was blocking "
                                            f"{silent_ratio:.0%} of chunks "
                                            f"\u2014 threshold lowered {old_th:.5f} \u2192 {new_th:.5f}"
                                        ),
                                        "old_threshold": old_th,
                                        "new_threshold": new_th,
                                    })
                                except Exception as exc:
                                    logger.warning("Broadcast error (recalibrated): %s", exc)
                    continue

                # Successful transcription — record "passed" in gate window (MIC only).
                if not _is_sys_chunk:
                    _gate_window.append("passed")
                    if len(_gate_window) > _RECALIB_WINDOW:
                        _gate_window.pop(0)

                _stats["chunks_transcribed"] += 1
                if _is_sys_chunk:
                    _stats["system_chunks_transcribed"] = _stats.get("system_chunks_transcribed", 0) + 1
                if _stats.get("first_transcript_delay_s") is None:
                    _stats["first_transcript_delay_s"] = round(
                        time.time() - _meeting["start_time"], 2
                    )
                try:
                    await _manager.broadcast(result.to_dict())
                    _stats["chunks_broadcast"] += 1
                except Exception as exc:
                    logger.warning("Broadcast error (transcript): %s", exc)
                # Only accumulate reliable chunks for summary/key-facts.
                if result.is_reliable:
                    _meeting["transcript_chunks"].append(result.text)

                # Decision detection — runs on every chunk, no API call.
                decision = detect_decision(result.text)
                if decision["is_decision"]:
                    event = {
                        "type":      "decision",
                        "text":      result.text,
                        "phrase":    decision["phrase"],
                        "timestamp": result.timestamp,
                    }
                    _meeting["decisions"].append(event)
                    try:
                        await _manager.broadcast(event)
                    except Exception as exc:
                        logger.warning("Broadcast error (decision): %s", exc)
                    logger.info("DECISION detected: phrase=%r  text=%r", decision["phrase"], result.text[:80])

                # Speaker tracking — cumulative counts.
                src = result.source   # "microphone" or "system"
                _meeting["speaker_counts"][src] = _meeting["speaker_counts"].get(src, 0) + 1

                # Coaching detection — runs on every chunk, no API call.
                # Snapshot coaching_history to avoid interleaving with
                # inject_them_transcript which also appends via _fire_coaching.
                try:
                    for coaching in detect_coaching(result.text, role=_active_role):
                        await _fire_coaching(coaching, timestamp=result.timestamp)

                    # Auto-resolve: check if any active coaching prompt was addressed.
                    resolved_set: set = _meeting.setdefault("coaching_resolved", set())
                    for event in list(_meeting["coaching_history"]):
                        cid = event["id"]
                        tid = event["trigger_id"]
                        if cid in resolved_set:
                            continue
                        is_resolved = detect_coaching_resolution(tid, result.text)
                        if is_resolved:
                            resolved_set.add(cid)
                            try:
                                await _manager.broadcast({
                                    "type": "coaching_resolved",
                                    "id":   cid,
                                    "note": "✓ Auto-detected in conversation",
                                })
                            except Exception as exc:
                                logger.warning("Broadcast error (coaching_resolved): %s", exc)
                            logger.info(
                                "COACHING auto-resolved: id=%r  trigger=%r", cid, tid
                            )
                except Exception as exc:
                    logger.warning("Coaching detection error (non-fatal): %s", exc)

                # Fact-checking — compare spoken numbers against KB facts.
                if _fact_checker is not None:
                    try:
                        from meetingmind.spacy_extractor import extract_numeric_entities
                        num_ents = await asyncio.to_thread(extract_numeric_entities, result.text)
                        if num_ents:
                            fc_results = await asyncio.to_thread(
                                _fact_checker.check_chunk, result.text, num_ents
                            )
                            for fcr in fc_results:
                                alert = {
                                    "type":         "fact_check_alert",
                                    "entity":       fcr.entity_text,
                                    "entity_type":  fcr.entity_type,
                                    "spoken_value": fcr.spoken_value,
                                    "stored_text":  fcr.stored_text,
                                    "stored_value": fcr.stored_value,
                                    "variance_pct": fcr.variance_pct,
                                    "severity":     fcr.severity.value,
                                    "context":      fcr.context_chunk,
                                    "timestamp":    result.timestamp,
                                }
                                _meeting["fact_checks"].append(alert)
                                try:
                                    await _manager.broadcast(alert)
                                except Exception as exc:
                                    logger.warning("Broadcast error (fact_check): %s", exc)
                                # Push fact-check to guest viewers
                                # TODO: CLOUD RELAY — push card payload to relay here
                                try:
                                    await _broadcast_to_guest_viewers({"type": "fact_check", "data": alert})
                                except Exception:
                                    pass
                                logger.info(
                                    "FACT CHECK: %s=%s vs KB=%s  variance=%.1f%%  severity=%s",
                                    fcr.entity_text, fcr.spoken_value,
                                    fcr.stored_value, fcr.variance_pct, fcr.severity.value,
                                )
                    except Exception as exc:
                        logger.warning("Fact-check error (non-fatal): %s", exc)

                # S8-002: AI action item & decision detection (async, non-blocking)
                if result.is_reliable:
                    _ai_task = asyncio.create_task(
                        _process_ai_detection(result.text, result.timestamp)
                    )
                    _ai_task.add_done_callback(_task_done_callback)

                    # S8-003: Scope creep detection (async, non-blocking)
                    _scope_task = asyncio.create_task(
                        _process_scope_detection(result.text, result.timestamp)
                    )
                    _scope_task.add_done_callback(_task_done_callback)

                    # S8-004: Timeline & delay flag detection (async, non-blocking)
                    _timeline_task = asyncio.create_task(
                        _process_timeline_detection(result.text, result.timestamp)
                    )
                    _timeline_task.add_done_callback(_task_done_callback)

            break

        except asyncio.CancelledError:
            logger.info("Transcription loop cancelled.")
            raise

        except Exception as exc:
            _restart_count += 1
            logger.exception("Transcription loop error (restart %d/%d): %s", _restart_count, _max_restarts, exc)
            if _restart_count > _max_restarts:
                logger.error("Transcription loop: max restarts exceeded — stopping.")
                await _manager.broadcast({
                    "type":    "error",
                    "message": f"Audio pipeline crashed after {_restart_count} errors. Stop and restart.",
                })
                break
            await asyncio.sleep(1.0)

    logger.info("Transcription loop stopped.")


async def _rms_broadcast_loop() -> None:
    """
    Task A: broadcasts latest per-frame RMS to all WebSocket clients every 0.5 s.
    Runs independently of Whisper — never blocks on transcription.

    Also monitors for sustained weak signal: if RMS stays below
    _WEAK_RMS_THRESHOLD for _WEAK_RMS_CONSECUTIVE readings, broadcasts
    a ``weak_signal`` warning so the UI can alert the user.
    """
    logger.info("RMS broadcast loop started.")
    meeting_start = time.time()
    first_sent    = False
    _weak_count   = 0          # consecutive low-RMS readings
    _weak_warned  = False      # only warn once per weak-signal episode

    try:
        while _meeting.get("active"):
            await asyncio.sleep(0.5)
            if not _meeting.get("active"):
                break
            capture = _meeting.get("capture")
            if capture is None:
                continue

            rms = capture.get_latest_rms()
            _stats["last_rms"] = round(rms, 5)
            recent = list(_stats.get("recent_rms", []))
            recent.append(_stats["last_rms"])
            _stats["recent_rms"] = recent[-5:]

            # ── System audio RMS (separate from mic) ──────────────────
            sys_rms = capture.get_latest_system_rms()
            _stats["system_last_rms"] = round(sys_rms, 5)
            sys_recent = list(_stats.get("system_recent_rms", []))
            sys_recent.append(_stats["system_last_rms"])
            _stats["system_recent_rms"] = sys_recent[-5:]

            if not first_sent:
                _stats["first_rms_delay_s"] = round(time.time() - meeting_start, 2)
                first_sent = True

            # ── Weak-signal monitor ───────────────────────────────────────
            if rms < _WEAK_RMS_THRESHOLD:
                _weak_count += 1
                if _weak_count >= _WEAK_RMS_CONSECUTIVE and not _weak_warned:
                    _weak_warned = True
                    logger.warning(
                        "Weak mic signal: RMS < %.5f for %d consecutive readings",
                        _WEAK_RMS_THRESHOLD, _weak_count,
                    )
                    try:
                        await _manager.broadcast({
                            "type":    "weak_signal",
                            "message": "Mic signal weak \u2014 check microphone and close other apps",
                            "rms":     _stats["last_rms"],
                            "count":   _weak_count,
                        })
                    except Exception as exc:
                        logger.warning("Broadcast error (weak_signal): %s", exc)
            else:
                if _weak_warned:
                    logger.info("Mic signal recovered: RMS=%.5f", rms)
                _weak_count  = 0
                _weak_warned = False

            await _manager.broadcast({
                "type":      "rms",
                "rms":       _stats["last_rms"],
                "threshold": _SILENCE_THRESHOLD,
                "captured":  _stats["chunks_from_queue"],
                "passed":    _stats["chunks_transcribed"],
            })
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("RMS broadcast loop crashed: %s", exc)
        await _manager.broadcast({
            "type":    "error",
            "message": f"RMS meter stopped unexpectedly: {exc}",
        })

    logger.info("RMS broadcast loop stopped.")


# ---------------------------------------------------------------------------
# Guest phone mic transcription loop
# ---------------------------------------------------------------------------

async def _guest_transcription_loop() -> None:
    """
    Pulls AudioChunks from the guest queue, transcribes them with Whisper, and
    broadcasts the results to all connected dashboard clients.

    Runs while ``_meeting["active"]`` and ``_meeting["guest_connected"]`` are
    both True.  A 10-second timeout per chunk gives 3× headroom over the 3-second
    window used by the guest browser.  Timeout and decode errors are logged and
    skipped so the loop never crashes on a bad frame.
    """
    logger.info("Guest transcription loop started.")
    guest_q = _meeting.get("guest_queue")
    if guest_q is None:
        logger.warning("Guest transcription loop: no guest queue — exiting.")
        return

    try:
        while _meeting.get("active") and _meeting.get("guest_connected"):
            try:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    _audio_io_pool, guest_q.get, 1.0,
                )
            except Exception:
                # queue.Empty from the blocking get (timeout) — just poll again.
                continue

            transcriber = _meeting.get("transcriber")
            if transcriber is None:
                continue

            try:
                result = await asyncio.wait_for(
                    transcriber.async_transcribe_chunk(chunk),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Guest chunk: Whisper timeout >10 s — skipped.")
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Guest chunk transcription error: %s", exc)
                continue

            if result is None:
                continue

            if result.is_reliable:
                _meeting["transcript_chunks"].append(result.text)
            await _manager.broadcast(result.to_dict())
            logger.info("GUEST transcript: %r", result.text[:80])
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Guest transcription loop crashed: %s", exc)

    logger.info("Guest transcription loop stopped.")


# ---------------------------------------------------------------------------
# Background calibration
# ---------------------------------------------------------------------------

async def _calibrate_background(capture) -> None:
    """
    Measure ambient noise for ~2 seconds without blocking meeting_start.

    Reads peak_rms from the RMS broadcast queue (not the chunk queue, so
    chunks still flow to Whisper).  Updates the calibrated threshold and
    applies it to the active transcriber once done.
    """
    try:
        cal_peaks: list[float] = []
        cal_deadline = time.time() + 2.0
        while time.time() < cal_deadline:
            if not _meeting.get("active"):
                return
            await asyncio.sleep(0.3)
            recent = list(_stats.get("recent_rms", []))
            for v in recent:
                if v > 0:
                    cal_peaks.append(v)

        if cal_peaks:
            ambient_peak = max(cal_peaks)
            calibrated = max(0.0001, min(0.002, ambient_peak * 3.0))
        else:
            calibrated = 0.001  # fallback

        # Cap threshold based on device type (BT mics have weaker signal)
        is_bt = _meeting.get("is_bt_device", False)
        cap = _AUDIO_THRESHOLDS["bt_mic_max_threshold"] if is_bt else _AUDIO_THRESHOLDS["mic_max_threshold"]
        if calibrated > cap:
            device_name = _meeting.get("device_name", "unknown")
            logger.info(
                "Threshold capped at %.5f for %s device '%s' (was %.5f)",
                cap, "BT" if is_bt else "MIC", device_name, calibrated,
            )
            calibrated = cap
            _meeting["threshold_capped"] = True
            _meeting["threshold_cap_reason"] = (
                f"{'BT' if is_bt else 'MIC'} cap {cap:.5f} applied (raw was {ambient_peak * 3.0:.5f})"
            )

        _meeting["calibrated_threshold"] = calibrated

        # Apply to the transcriber if it's already loaded.
        transcriber = _meeting.get("transcriber")
        if transcriber is not None:
            transcriber._silence_threshold = calibrated

        logger.info(
            "Background calibration: %d samples, ambient_peak=%.5f, threshold=%.4f",
            len(cal_peaks), max(cal_peaks) if cal_peaks else 0.0, calibrated,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Background calibration failed: %s", exc)


# ---------------------------------------------------------------------------
# S8-002: AI-powered action item & decision detection
# ---------------------------------------------------------------------------

_AI_DETECT_SYSTEM = (
    "You are a PM assistant detecting action items and decisions "
    "from meeting transcripts. Return ONLY valid JSON. "
    "An action item is a commitment to do something, with or "
    "without an explicit owner or deadline. "
    "A decision is a conclusion, agreement, or resolution that "
    "was reached during the conversation. "
    "Be precise — only extract items explicitly stated, "
    "never infer or assume."
)

_SCOPE_DETECT_SYSTEM = (
    "You are a PM scope management assistant. Analyse "
    "meeting transcript chunks for scope creep risks. "
    "You have access to the active project scope context. "
    "Return ONLY valid JSON. Never invent scope violations "
    "— only flag what is explicitly stated or clearly "
    "implied in the transcript."
)

_TIMELINE_DETECT_SYSTEM = (
    "You are a PM schedule management assistant. Analyse "
    "meeting transcript chunks for timeline and schedule "
    "risks. Use the active project timeline as reference. "
    "Return ONLY valid JSON. Only flag what is explicitly "
    "stated — never infer delays without evidence."
)


async def _detect_action_items(transcript_chunk: str) -> dict:
    """
    Call Claude API to extract action items and decisions from a transcript chunk.

    Returns {"action_items": [...], "decisions": [...]} or empty lists on failure.
    Items with confidence < 0.6 are filtered out.
    """
    if not transcript_chunk.strip():
        return {"action_items": [], "decisions": []}

    user_prompt = (
        "Analyse this transcript chunk and extract any action items "
        "and decisions. Return JSON:\n"
        "{\n"
        "  \"action_items\": [\n"
        "    {\n"
        "      \"text\": \"clear description of what needs to be done\",\n"
        "      \"owner\": \"person name or null\",\n"
        "      \"deadline\": \"deadline text or null\",\n"
        "      \"source_quote\": \"exact phrase from transcript\",\n"
        "      \"confidence\": 0.0\n"
        "    }\n"
        "  ],\n"
        "  \"decisions\": [\n"
        "    {\n"
        "      \"text\": \"clear description of what was decided\",\n"
        "      \"made_by\": \"person name or null\",\n"
        "      \"source_quote\": \"exact phrase from transcript\",\n"
        "      \"confidence\": 0.0\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Only include items with confidence >= 0.6.\n"
        f"Transcript: {transcript_chunk}"
    )

    try:
        from meetingmind._api_key import load_api_key as _load_key
        api_key = _load_key()
    except Exception as exc:
        logger.warning("_detect_action_items: API key unavailable: %s", exc)
        return {"action_items": [], "decisions": []}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_AI_DETECT_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
        )
    except Exception as exc:
        logger.warning("_detect_action_items: Claude API call failed: %s", exc)
        return {"action_items": [], "decisions": []}

    raw = response.content[0].text.strip()
    # Strip markdown fences
    import re as _re
    fenced = _re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("_detect_action_items: invalid JSON from Claude: %s", raw[:200])
        return {"action_items": [], "decisions": []}

    # Filter by confidence >= 0.6
    items = [
        ai for ai in data.get("action_items", [])
        if isinstance(ai, dict) and ai.get("confidence", 0) >= 0.6
    ]
    decs = [
        d for d in data.get("decisions", [])
        if isinstance(d, dict) and d.get("confidence", 0) >= 0.6
    ]

    return {"action_items": items, "decisions": decs}


async def _process_ai_detection(transcript_chunk: str, timestamp: float) -> None:
    """
    Fire-and-forget task: detect action items and decisions, enrich with project
    context, append to global lists, and broadcast to WebSocket clients.
    """
    try:
        result = await _detect_action_items(transcript_chunk)

        # Enrich with project context
        cpmai_phase = None
        project_name = None
        if _active_project:
            cpmai_phase = _active_project.get("cpmai_phase")
            project_name = _active_project.get("project_name")

        now_iso = datetime.now(timezone.utc).isoformat()

        for ai in result.get("action_items", []):
            item = {
                "id": str(_uuid_mod.uuid4()),
                "text": ai.get("text", ""),
                "owner": ai.get("owner"),
                "deadline": ai.get("deadline"),
                "source_quote": ai.get("source_quote", ""),
                "cpmai_phase": cpmai_phase,
                "project_name": project_name,
                "timestamp": now_iso,
                "confidence": ai.get("confidence", 0.6),
            }
            _action_items.append(item)
            try:
                await _manager.broadcast({"type": "action_item", "data": item})
            except Exception as exc:
                logger.warning("Broadcast error (action_item): %s", exc)
            # Push to guest viewers
            try:
                await _broadcast_to_guest_viewers({"type": "action_item", "data": item})
            except Exception:
                pass
            logger.info("ACTION ITEM detected: %r  owner=%s", item["text"][:60], item["owner"])

        for d in result.get("decisions", []):
            dec = {
                "id": str(_uuid_mod.uuid4()),
                "text": d.get("text", ""),
                "made_by": d.get("made_by"),
                "source_quote": d.get("source_quote", ""),
                "cpmai_phase": cpmai_phase,
                "project_name": project_name,
                "timestamp": now_iso,
                "confidence": d.get("confidence", 0.6),
            }
            _decisions_ai.append(dec)
            try:
                await _manager.broadcast({"type": "ai_decision", "data": dec})
            except Exception as exc:
                logger.warning("Broadcast error (ai_decision): %s", exc)
            try:
                await _broadcast_to_guest_viewers({"type": "ai_decision", "data": dec})
            except Exception:
                pass
            logger.info("AI DECISION detected: %r  made_by=%s", dec["text"][:60], dec["made_by"])

    except Exception as exc:
        logger.warning("_process_ai_detection error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# S8-003: Scope Creep Detection (AI-powered)
# ---------------------------------------------------------------------------

async def _detect_scope_creep(transcript_chunk: str) -> dict:
    """
    Call Claude API to detect scope creep risks in a transcript chunk.

    Returns {"scope_alerts": [...]} or empty list on failure.
    Alerts with confidence < 0.65 are filtered out.
    """
    if not transcript_chunk.strip():
        return {"scope_alerts": []}

    # Build scope context from active project
    scope_context = ""
    if _active_project:
        scope_items = _active_project.get("scope_items", [])
        project_name = _active_project.get("project_name", "Unknown")
        cpmai_phase = _active_project.get("cpmai_phase", "")
        scope_context = (
            f"Project: {project_name}\n"
            f"CPMAI Phase: {cpmai_phase}\n"
            f"Approved scope items: {json.dumps(scope_items)}\n"
        )

    user_prompt = (
        "Analyse this transcript chunk for scope creep risks.\n"
        f"{scope_context}\n"
        "Look for:\n"
        "- Requests outside approved scope (out_of_scope)\n"
        "- Language signaling informal scope expansion like "
        "'while we're at it', 'can we also', 'it would be nice if' (scope_creep_language)\n"
        "- Gold plating — adding unrequested extras (gold_plating)\n"
        "- Unclear or shifting requirements (requirement_drift)\n\n"
        "Return JSON:\n"
        "{\n"
        "  \"scope_alerts\": [\n"
        "    {\n"
        "      \"alert_type\": \"out_of_scope|scope_creep_language|gold_plating|requirement_drift\",\n"
        "      \"severity\": \"critical|warning\",\n"
        "      \"text\": \"clear description of the scope risk\",\n"
        "      \"source_quote\": \"exact phrase from transcript\",\n"
        "      \"matched_term\": \"the scope item or keyword that triggered this\",\n"
        "      \"suggestion\": \"actionable PM recommendation\",\n"
        "      \"confidence\": 0.0\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Only include alerts with confidence >= 0.65.\n"
        f"Transcript: {transcript_chunk}"
    )

    try:
        from meetingmind._api_key import load_api_key as _load_key
        api_key = _load_key()
    except Exception as exc:
        logger.warning("_detect_scope_creep: API key unavailable: %s", exc)
        return {"scope_alerts": []}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SCOPE_DETECT_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
        )
    except Exception as exc:
        logger.warning("_detect_scope_creep: Claude API call failed: %s", exc)
        return {"scope_alerts": []}

    raw = response.content[0].text.strip()
    # Strip markdown fences
    import re as _re
    fenced = _re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("_detect_scope_creep: invalid JSON from Claude: %s", raw[:200])
        return {"scope_alerts": []}

    # Filter by confidence >= 0.65
    alerts = [
        a for a in data.get("scope_alerts", [])
        if isinstance(a, dict) and a.get("confidence", 0) >= 0.65
    ]

    return {"scope_alerts": alerts}


async def _process_scope_detection(transcript_chunk: str, timestamp: float) -> None:
    """
    Fire-and-forget task: detect scope creep risks, enrich with project
    context, append to global list, and broadcast to WebSocket clients.
    """
    try:
        result = await _detect_scope_creep(transcript_chunk)

        cpmai_phase = None
        project_name = None
        if _active_project:
            cpmai_phase = _active_project.get("cpmai_phase")
            project_name = _active_project.get("project_name")

        now_iso = datetime.now(timezone.utc).isoformat()

        for a in result.get("scope_alerts", []):
            alert = {
                "id": str(_uuid_mod.uuid4()),
                "alert_type": a.get("alert_type", "out_of_scope"),
                "severity": a.get("severity", "warning"),
                "text": a.get("text", ""),
                "source_quote": a.get("source_quote", ""),
                "matched_term": a.get("matched_term"),
                "suggestion": a.get("suggestion", ""),
                "cpmai_phase": cpmai_phase,
                "project_name": project_name,
                "timestamp": now_iso,
                "confidence": a.get("confidence", 0.65),
            }
            _scope_alerts.append(alert)
            try:
                await _manager.broadcast({"type": "scope_alert", "data": alert})
            except Exception as exc:
                logger.warning("Broadcast error (scope_alert): %s", exc)
            try:
                await _broadcast_to_guest_viewers({"type": "scope_alert", "data": alert})
            except Exception:
                pass
            logger.info("SCOPE ALERT detected: %r  type=%s", alert["text"][:60], alert["alert_type"])

    except Exception as exc:
        logger.warning("_process_scope_detection error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# S8-004: Timeline & Delay Flag detection
# ---------------------------------------------------------------------------

async def _detect_timeline_risk(transcript_chunk: str) -> dict:
    """
    Call Claude API to detect timeline/schedule risks in a transcript chunk.

    Returns {"timeline_alerts": [...]} or empty list on failure.
    Alerts with confidence < 0.65 are filtered out.
    """
    if not transcript_chunk.strip():
        return {"timeline_alerts": []}

    # Build timeline context from active project
    timeline_context = ""
    if _active_project:
        milestones = _active_project.get("milestones") or []
        milestones_text = "\n".join(
            f"- {m.get('name', m) if isinstance(m, dict) else m}: {m.get('date', '') if isinstance(m, dict) else ''}"
            for m in milestones
        ) if milestones else "None defined"
        timeline_context = (
            f"\nActive Project: {_active_project.get('project_name', 'Unknown')}\n"
            f"Start Date: {_active_project.get('start_date', 'N/A')}\n"
            f"End Date: {_active_project.get('end_date', 'N/A')}\n"
            f"Milestones:\n{milestones_text}\n"
            f"Success Criteria: {_active_project.get('success_criteria', 'N/A')}\n"
        )

    user_prompt = (
        "Analyse this transcript for timeline and schedule risks.\n"
        f"{timeline_context}\n"
        "Detect these risk patterns:\n\n"
        "DELAY LANGUAGE — direct statements of slippage:\n"
        "'we need more time', 'taking longer than expected', "
        "'running behind', 'haven't started yet', 'push the "
        "deadline', 'move the date', 'not going to make it', "
        "'delayed', 'behind schedule', 'overdue'\n\n"
        "MILESTONE RISK — threat to a specific milestone:\n"
        "Any reference to a named milestone being at risk, "
        "late, or incomplete\n\n"
        "OPTIMISM BIAS — unrealistic recovery language:\n"
        "'we can make up the time', 'work faster', 'skip "
        "testing', 'compress the timeline', 'work weekends', "
        "'it won't take long'\n\n"
        "DEPENDENCY BLOCKER — external dependency causing risk:\n"
        "'waiting on', 'blocked by', 'depends on', 'haven't "
        "heard back', 'still waiting for approval', "
        "'other team hasn't delivered'\n\n"
        "SCHEDULE COMPRESSION — dangerous shortcuts:\n"
        "'skip the testing phase', 'cut UAT', 'go straight "
        "to production', 'no time for review'\n\n"
        "Return JSON:\n"
        "{\n"
        "  \"timeline_alerts\": [\n"
        "    {\n"
        "      \"alert_type\": \"delay_language|milestone_risk"
        "|optimism_bias|dependency_blocker|schedule_compression\",\n"
        "      \"severity\": \"warning|critical\",\n"
        "      \"text\": \"clear description of the timeline risk\",\n"
        "      \"source_quote\": \"exact phrase from transcript\",\n"
        "      \"affected_milestone\": \"milestone name or null\",\n"
        "      \"days_at_risk\": \"estimated days as integer or null\",\n"
        "      \"suggestion\": \"PM coaching response\",\n"
        "      \"confidence\": 0.0\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Only include alerts with confidence >= 0.65.\n"
        f"Transcript: {transcript_chunk}"
    )

    try:
        from meetingmind._api_key import load_api_key as _load_key
        api_key = _load_key()
    except Exception as exc:
        logger.warning("_detect_timeline_risk: API key unavailable: %s", exc)
        return {"timeline_alerts": []}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_TIMELINE_DETECT_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
        )
    except Exception as exc:
        logger.warning("_detect_timeline_risk: Claude API call failed: %s", exc)
        return {"timeline_alerts": []}

    raw = response.content[0].text.strip()
    # Strip markdown fences
    import re as _re
    fenced = _re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("_detect_timeline_risk: invalid JSON from Claude: %s", raw[:200])
        return {"timeline_alerts": []}

    # Filter by confidence >= 0.65
    alerts = [
        a for a in data.get("timeline_alerts", [])
        if isinstance(a, dict) and a.get("confidence", 0) >= 0.65
    ]

    return {"timeline_alerts": alerts}


async def _process_timeline_detection(transcript_chunk: str, timestamp: float) -> None:
    """
    Fire-and-forget task: detect timeline risks, enrich with project
    context, append to global list, and broadcast to WebSocket clients.
    """
    try:
        result = await _detect_timeline_risk(transcript_chunk)

        cpmai_phase = None
        project_name = None
        if _active_project:
            cpmai_phase = _active_project.get("cpmai_phase")
            project_name = _active_project.get("project_name")

        now_iso = datetime.now(timezone.utc).isoformat()

        for a in result.get("timeline_alerts", []):
            days_at_risk = a.get("days_at_risk")
            if days_at_risk is not None:
                try:
                    days_at_risk = int(days_at_risk)
                except (TypeError, ValueError):
                    days_at_risk = None

            alert = {
                "id": str(_uuid_mod.uuid4()),
                "alert_type": a.get("alert_type", "delay_language"),
                "severity": a.get("severity", "warning"),
                "text": a.get("text", ""),
                "source_quote": a.get("source_quote", ""),
                "affected_milestone": a.get("affected_milestone"),
                "days_at_risk": days_at_risk,
                "suggestion": a.get("suggestion", ""),
                "cpmai_phase": cpmai_phase,
                "project_name": project_name,
                "timestamp": now_iso,
                "confidence": a.get("confidence", 0.65),
            }
            _timeline_alerts.append(alert)
            try:
                await _manager.broadcast({"type": "timeline_alert", "data": alert})
            except Exception as exc:
                logger.warning("Broadcast error (timeline_alert): %s", exc)
            try:
                await _broadcast_to_guest_viewers({"type": "timeline_alert", "data": alert})
            except Exception:
                pass
            logger.info("TIMELINE ALERT detected: %r  type=%s", alert["text"][:60], alert["alert_type"])

    except Exception as exc:
        logger.warning("_process_timeline_detection error (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# S8-005: PM Meeting Summary — Claude-powered structured summary
# ---------------------------------------------------------------------------

async def _generate_pm_summary(
    transcript: str,
    action_items: list[dict],
    decisions: list[dict],
    scope_alerts: list[dict],
    timeline_alerts: list[dict],
    project_name: str | None,
    cpmai_phase: str | None,
    duration_minutes: float,
    speaker_counts: dict,
    meeting_date: str,
) -> dict:
    """
    Call Claude API to produce a structured PM meeting summary.

    Returns a dict with keys: overview, executive_summary, decisions,
    action_items, risks, next_steps.  Returns None on any failure so
    the frontend can fall back to raw data display.
    """
    if not transcript.strip():
        return None

    # Build context block for Claude
    proj_line = f"Project: {project_name}" if project_name else "Project: (none)"
    phase_line = f"CPMAI Phase: {cpmai_phase}" if cpmai_phase else "CPMAI Phase: (none)"
    speakers = ", ".join(f"{k}: {v} chunks" for k, v in speaker_counts.items()) or "unknown"

    ai_text = "\n".join(
        f"- {a.get('text', '')} (owner: {a.get('owner', 'TBD')}, deadline: {a.get('deadline', 'TBD')})"
        for a in action_items
    ) or "(none)"
    dec_text = "\n".join(
        f"- {d.get('text', '')} (by: {d.get('made_by') or d.get('owner', 'unknown')})"
        for d in decisions
    ) or "(none)"
    scope_text = "\n".join(
        f"- [{s.get('severity', 'warning')}] {s.get('text', '')} — {s.get('suggestion', '')}"
        for s in scope_alerts
    ) or "(none)"
    timeline_text = "\n".join(
        f"- [{t.get('severity', 'warning')}] {t.get('text', '')} — {t.get('suggestion', '')}"
        for t in timeline_alerts
    ) or "(none)"

    user_prompt = (
        f"Meeting date: {meeting_date}\n"
        f"Duration: {duration_minutes} minutes\n"
        f"{proj_line}\n{phase_line}\n"
        f"Speakers: {speakers}\n\n"
        f"ACTION ITEMS DETECTED:\n{ai_text}\n\n"
        f"DECISIONS DETECTED:\n{dec_text}\n\n"
        f"SCOPE ALERTS:\n{scope_text}\n\n"
        f"TIMELINE ALERTS:\n{timeline_text}\n\n"
        f"TRANSCRIPT:\n{transcript[:8000]}\n"
    )

    system_prompt = (
        "You are a PM meeting analyst. Generate a structured "
        "meeting summary with these sections:\n\n"
        "MEETING OVERVIEW — Date, duration, project, CPMAI phase, "
        "1-2 sentence executive summary of what was discussed.\n\n"
        "KEY DECISIONS — Bulleted list of decisions made, with owner if known.\n\n"
        "ACTION ITEMS — Bulleted list with owner, deadline, priority (High/Medium/Low).\n\n"
        "RISKS IDENTIFIED — Scope risks and timeline risks combined into a single "
        "risk register with severity and recommendation.\n\n"
        "NEXT STEPS — Recommended PM actions before next meeting based on "
        "risks and action items detected.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "overview": "date, duration, project, phase in one line",\n'
        '  "executive_summary": "1-2 sentence summary",\n'
        '  "decisions": [{"text": "...", "owner": "..."}],\n'
        '  "action_items": [{"text": "...", "owner": "...", '
        '"deadline": "...", "priority": "High|Medium|Low"}],\n'
        '  "risks": [{"type": "scope|timeline", "description": "...", '
        '"severity": "critical|warning", "recommendation": "..."}],\n'
        '  "next_steps": ["step 1", "step 2"]\n'
        "}\n"
    )

    try:
        from meetingmind._api_key import load_api_key as _load_key
        api_key = _load_key()
    except Exception as exc:
        logger.warning("_generate_pm_summary: API key unavailable: %s", exc)
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning("PM Summary generation timed out after 30s")
        return None
    except Exception as exc:
        logger.warning("PM Summary failed: %s", exc)
        return None

    raw = response.content[0].text.strip()
    # Strip markdown fences
    import re as _re
    fenced = _re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("PM Summary: invalid JSON response: %s", raw[:200])
        return None

    # Validate expected keys — return None if structure is wrong
    required = {"overview", "executive_summary", "decisions", "action_items", "risks", "next_steps"}
    if not required.issubset(data.keys()):
        logger.warning("PM Summary: missing required keys: %s", required - data.keys())
        return None

    return data


async def _generate_pm_summary_background() -> None:
    """
    Fire-and-forget task: generate PM summary and store in _last_summary.
    Never blocks meeting_stop — failures are silently logged.
    """
    global _last_summary
    if _last_summary is None:
        return

    try:
        # Gather inputs from _last_summary (snapshot captured at stop time)
        transcript_chunks = _last_summary.get("_transcript_text", "")
        ai_actions = _last_summary.get("ai_action_items", [])
        ai_decs = _last_summary.get("ai_decisions", [])
        scope = _last_summary.get("scope_alerts", [])
        timeline = _last_summary.get("timeline_alerts", [])
        proj = _active_project.get("project_name") if _active_project else None
        phase = _active_project.get("cpmai_phase") if _active_project else None
        duration = _last_summary.get("duration_minutes", 0)
        speakers = _last_summary.get("speaker_counts", {})
        meeting_date = _last_summary.get("date", "")

        result = await _generate_pm_summary(
            transcript=transcript_chunks,
            action_items=ai_actions,
            decisions=ai_decs,
            scope_alerts=scope,
            timeline_alerts=timeline,
            project_name=proj,
            cpmai_phase=phase,
            duration_minutes=duration,
            speaker_counts=speakers,
            meeting_date=meeting_date,
        )

        if result and _last_summary is not None:
            _last_summary["pm_summary"] = result
            logger.info("PM summary generated successfully.")
    except Exception as exc:
        logger.warning("_generate_pm_summary_background failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Background Whisper loader
# ---------------------------------------------------------------------------

async def _whisper_load_background(
    model_size:   str,
    privacy_mode: bool,
    language:     Optional[str],
) -> None:
    """
    Load the Whisper model in the thread pool, then start the transcription loop.

    Broadcasts {"type": "model_ready"} on success or {"type": "error"} on failure.
    Safe to cancel — cleans up gracefully if the meeting is stopped during loading.
    """
    # Read the calibrated threshold computed during meeting_start.
    cal_threshold = _meeting.get("calibrated_threshold")

    logger.info("Loading Whisper '%s' model in background…", model_size)
    try:
        from meetingmind.transcriber import StreamingTranscriber
        transcriber = await asyncio.to_thread(
            lambda: StreamingTranscriber(
                model_size=model_size,
                privacy_mode=privacy_mode,
                language=language,
                silence_threshold=cal_threshold,
                initial_prompt=_user_vocab_hints,
            )
        )
    except asyncio.CancelledError:
        logger.info("Whisper background load cancelled.")
        raise
    except Exception as exc:
        logger.exception("Whisper model load failed: %s", exc)
        _meeting["model_loading"] = False
        await _manager.broadcast({
            "type":    "error",
            "message": f"Whisper '{model_size}' failed to load: {exc}",
        })
        return

    # Meeting may have been stopped while we were loading — abort silently.
    if not _meeting["active"]:
        logger.info("Whisper loaded but meeting already stopped — discarding.")
        return

    _meeting["transcriber"]   = transcriber
    _meeting["model_loading"] = False
    _t = asyncio.create_task(_transcription_loop())
    _t.add_done_callback(_task_done_callback)
    _meeting["task"]          = _t

    # Edge case: guest connected while Whisper was still loading.
    # The guest_queue has been buffering chunks; start the loop now.
    if _meeting.get("guest_connected") and _meeting.get("guest_queue") is not None:
        _gt = asyncio.create_task(_guest_transcription_loop())
        _gt.add_done_callback(_task_done_callback)
        _meeting["guest_task"] = _gt
        logger.info("Guest transcription loop started (Whisper now ready).")

    logger.info("Whisper '%s' ready — transcription loop started.", model_size)
    await _manager.broadcast({
        "type":  "model_ready",
        "model": model_size,
    })


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------

async def _ingest_meeting_background(
    meeting_id:       str,
    started_at:       datetime,
    stopped_at:       datetime,
    duration_seconds: float,
    model_size:       str,
    transcript_text:  str,
) -> None:
    """Lazy-init KB if needed, then run ingestion in thread pool."""
    await _ensure_kb()
    if _kb is None:
        logger.warning("KB unavailable — skipping ingestion for %s", meeting_id)
        return
    try:
        result = await asyncio.to_thread(
            _kb.ingest_meeting,
            meeting_id,
            started_at,
            stopped_at,
            duration_seconds,
            model_size,
            transcript_text,
        )
        logger.info(
            "Ingestion complete: %s MQS=%.1f", meeting_id, result["mqs_score"]
        )
        await _manager.broadcast({
            "type":       "ingestion_complete",
            "meeting_id": meeting_id,
            "mqs_score":  result["mqs_score"],
            "decisions":  result["decision_count"],
            "actions":    result["action_count"],
            "project":    result.get("project_name"),
        })
    except Exception as exc:
        logger.exception(
            "Background ingestion failed for %s: %s", meeting_id, exc
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run device detection and KB init once at startup, then yield."""
    global _detected, _kb, _ctx, _active_project
    try:
        _detected = await asyncio.to_thread(detect_devices)
        logger.info(
            "Device detection complete — mic: %s | system: %s | warnings: %d",
            _detected.recommended_mic["name"] if _detected.recommended_mic else "None",
            _detected.recommended_system["name"] if _detected.recommended_system else "None",
            len(_detected.warnings),
        )
    except Exception as exc:
        logger.error("Device detection failed at startup: %s", exc)
        _detected = DetectionResult(
            recommended_mic=None,
            recommended_system=None,
            all_devices=[],
            warnings=[f"Device detection error: {exc}"],
        )

    # KB/ContextEngine are NOT initialised here.
    # ChromaDB's PersistentClient binds port 8000 by default, which conflicts
    # with uvicorn's bind (lifespan runs BEFORE uvicorn binds its socket).
    # Instead, _ensure_kb() initialises them lazily on first use.

    # Whisper is NOT preloaded at startup — saves ~400 MB between meetings.
    # It loads on-demand when the user clicks Start (via _whisper_load_background).

    # Lower process priority so Windows gives CPU to Chrome/CC when idle.
    try:
        import psutil, os
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL (pid=%d).", os.getpid())
    except Exception as exc:
        logger.warning("Could not set process priority: %s", exc)

    # Auto-load dev profile (sets name, role, vocab hints, ingests KB seed).
    _dp = asyncio.create_task(_load_dev_profile())
    _dp.add_done_callback(_task_done_callback)

    # S8-006: Verify PM global KB on startup (non-blocking).
    _pm_kb_task = asyncio.create_task(_verify_pm_global_kb())
    _pm_kb_task.add_done_callback(_task_done_callback)

    # Auto-load most recent project brief (non-blocking).
    try:
        import os as _os
        if _BRIEFS_DIR.exists():
            brief_files = list(_BRIEFS_DIR.glob("*.json"))
            if brief_files:
                newest = max(brief_files, key=lambda p: _os.path.getmtime(p))
                data = json.loads(newest.read_text(encoding="utf-8"))
                _active_project = data
                logger.info(
                    "Auto-loaded active project: %s (CPMAI phase: %s)",
                    data.get("project_name"),
                    data.get("cpmai_phase"),
                )
            else:
                logger.info("No project briefs found — user will be prompted on first load")
        else:
            logger.info("No project briefs found — user will be prompted on first load")
    except Exception as exc:
        logger.warning("Auto-load project brief failed (continuing startup): %s", exc)

    # Server-initiated WebSocket keepalive ping every 15 s.
    ping_task = asyncio.create_task(_ws_ping_loop(_manager))
    ping_task.add_done_callback(_task_done_callback)

    yield   # server runs here (uvicorn has bound port 8000 by this point)

    ping_task.cancel()
    if _kb:
        await asyncio.to_thread(_kb.close)
    # Shut down the Whisper worker process.
    from meetingmind.transcriber import shutdown_process_pool
    shutdown_process_pool()


app = FastAPI(
    title="MeetingMind",
    description=(
        "Real-time AI meeting assistant. "
        "Audio processing is on-device; only transcript text is sent to Claude "
        "when AI suggestions or analysis are requested."
    ),
    version="0.2.0",
    lifespan=_lifespan,
)

# Allow the frontend/index.html (file://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.endswith(".html") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

# Serve everything inside frontend/ under /static (JS, CSS, images added later)
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Root — serve the dashboard ────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the MeetingMind web dashboard."""
    return FileResponse(str(_FRONTEND / "index.html"), headers=_NO_CACHE_HEADERS)


@app.get("/static/icon-192.png", include_in_schema=False)
async def serve_icon() -> Response:
    """Serve a placeholder 192x192 PWA icon (purple square with 'MM')."""
    import io
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (192, 192), color="#6c63ff")
        draw = ImageDraw.Draw(img)
        draw.text((60, 80), "MM", fill="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    except ImportError:
        # Fallback: 1x1 transparent PNG so the 404 stops
        import base64
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
            "DUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        return Response(content=png_1x1, media_type="image/png")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve favicon.ico if present; return 204 so browsers stop 404-logging."""
    ico = _FRONTEND / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> Response:
    """Serve PWA manifest."""
    mf = _FRONTEND / "manifest.json"
    if mf.exists():
        return FileResponse(str(mf), media_type="application/manifest+json")
    return Response(status_code=204)


@app.get("/sw.js", include_in_schema=False)
async def service_worker() -> Response:
    """Serve PWA service worker from root scope."""
    sw = _FRONTEND / "sw.js"
    if sw.exists():
        return FileResponse(str(sw), media_type="application/javascript")
    return Response(status_code=204)


@app.get("/guest", include_in_schema=False)
async def guest_page() -> FileResponse:
    """Serve the guest phone mic page."""
    return FileResponse(str(_FRONTEND / "guest.html"), headers=_NO_CACHE_HEADERS)


# ── Settings ──────────────────────────────────────────────────────────────────

_VALID_ROLES = {"EA", "PM", "Sales", "Custom"}


class _RoleRequest(BaseModel):
    role: str


@app.post("/settings/role", summary="Set the active user role")
async def set_role(req: _RoleRequest) -> dict:
    """Set the active role (EA, PM, Sales, Custom). Affects coaching triggers."""
    global _active_role
    if req.role not in _VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {req.role}")
    _active_role = req.role
    logger.info("Active role set to: %s", _active_role)
    return {"status": "ok", "role": _active_role}


@app.get("/settings/role", summary="Get the active user role")
async def get_role() -> dict:
    """Return the currently active role."""
    return {"role": _active_role}


# ── Onboarding profile ───────────────────────────────────────────────────────

class _OnboardingProfileRequest(BaseModel):
    name: str = ""
    role: str = "PM"
    context: str = ""


@app.post("/onboarding/profile", summary="Save onboarding profile, extract vocab hints")
async def save_onboarding_profile(req: _OnboardingProfileRequest) -> dict:
    """
    Save the user's onboarding profile.

    1. Stores name and syncs active role.
    2. Ingests context text into ChromaDB as 'onboarding_seed'.
    3. Runs spaCy NER on context to extract people, orgs, money.
    4. Builds Whisper vocabulary hints from extracted entity names.
    """
    global _user_name, _user_vocab_hints, _active_role

    # Store name
    _user_name = req.name.strip() or None

    # Sync role
    if req.role in _VALID_ROLES:
        _active_role = req.role

    context = req.context.strip()
    entities_extracted = 0
    vocab_hints_count = 0

    if context:
        # Ingest into KB
        if _kb is not None:
            seed_text = f"User: {_user_name or 'unknown'}\nRole: {_active_role}\n\n{context}"
            try:
                await asyncio.to_thread(_kb.ingest_text, "onboarding_seed", seed_text, "summaries")
            except Exception as exc:
                logger.warning("Onboarding KB ingest failed: %s", exc)

        # spaCy NER extraction
        try:
            from meetingmind.spacy_extractor import extract_entities
            entities = await asyncio.to_thread(extract_entities, context)
        except Exception as exc:
            logger.warning("Onboarding spaCy extraction failed: %s", exc)
            entities = {"people": [], "orgs": [], "money": []}

        entities_extracted = len(entities.get("people", [])) + len(entities.get("orgs", [])) + len(entities.get("money", []))

        # Build vocab hints from entities
        vocab_terms = []
        for name in entities.get("people", []):
            if name not in vocab_terms:
                vocab_terms.append(name)
        for org in entities.get("orgs", []):
            if org not in vocab_terms:
                vocab_terms.append(org)
        for money in entities.get("money", []):
            if money not in vocab_terms:
                vocab_terms.append(money)

        if vocab_terms:
            from meetingmind.transcriber import StreamingTranscriber
            default_hints = StreamingTranscriber._DEFAULT_INITIAL_PROMPT
            _user_vocab_hints = default_hints + ", " + ", ".join(vocab_terms)
        else:
            _user_vocab_hints = None

        vocab_hints_count = len(_user_vocab_hints) if _user_vocab_hints else 0
    else:
        _user_vocab_hints = None

    logger.info(
        "Onboarding profile saved: name=%s, role=%s, entities=%d, vocab_hints_len=%d",
        _user_name, _active_role, entities_extracted, vocab_hints_count,
    )

    return {
        "status": "ok",
        "name": _user_name,
        "role": _active_role,
        "entities_extracted": entities_extracted,
        "vocab_hints_length": vocab_hints_count,
    }


@app.get("/onboarding/profile", summary="Get the current onboarding profile")
async def get_onboarding_profile() -> dict:
    """Return the stored onboarding name and role."""
    return {"name": _user_name, "role": _active_role}


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Server liveness check")
async def health() -> dict:
    """
    Returns server status and current meeting state.

    Safe to poll frequently — no side effects.
    """
    elapsed: Optional[float] = None
    if _meeting["active"] and _meeting["start_time"]:
        elapsed = round(time.time() - _meeting["start_time"], 1)

    mic_ok = _detected.mic_permission_ok if _detected else None

    return {
        "status":               "ok",
        "meeting_active":       _meeting["active"],
        "model":                _meeting["model"],
        "model_loading":        _meeting.get("model_loading", False),
        "connected_clients":    _manager.count,
        "elapsed_seconds":      elapsed,
        "windows_mic_blocked":  mic_ok is False,
    }


# ── Meeting lifecycle ─────────────────────────────────────────────────────────

@app.get("/devices", summary="List available audio input devices with recommendations")
async def list_devices() -> dict:
    """
    Returns categorised audio input devices with auto-detected recommendations.

    Device detection runs at startup and is cached for the lifetime of the server.
    Categories: REAL_MIC, VIRTUAL_MIC, SYSTEM_AUDIO.
    Scores range 0–10 (higher = preferred).

    Use recommended_mic['index'] / recommended_system['index'] as override values
    for POST /meeting/start if the auto-selected devices are wrong.
    """
    if _detected is None:
        return JSONResponse({"error": "Device detection not yet complete. Retry in a moment."}, status_code=503)

    return {
        "recommended_mic":    _detected.recommended_mic,
        "recommended_system": _detected.recommended_system,
        "all_devices":        [d.to_dict() for d in _detected.all_devices],
        "warnings":           _detected.warnings,
    }


# ── Guest phone mic endpoints ─────────────────────────────────────────────────

@app.post("/guest/session", summary="Create a guest phone mic session")
async def create_guest_session() -> dict:
    """
    Generate a one-time PIN for the guest phone mic feature.

    Requires an active meeting. Returns the 4-digit PIN and the URL the guest
    should open on their phone. Only one guest at a time is supported.
    """
    if not _meeting["active"]:
        return JSONResponse(
            {"error": "No meeting is active. Start a meeting first."},
            status_code=400,
        )
    if _meeting.get("guest_connected"):
        return JSONResponse(
            {"error": "A guest is already connected."},
            status_code=409,
        )
    pin = generate_pin()
    _meeting["guest_pin"] = pin
    logger.info("Guest session created: PIN=%s", pin)
    return {"pin": pin, "guest_url": f"/guest?pin={pin}"}


@app.websocket("/ws/guest")
async def guest_ws_endpoint(websocket: WebSocket, pin: str = "") -> None:
    """
    WebSocket endpoint for the guest phone mic.

    The guest browser connects here, sends raw Float32 PCM chunks as
    base64-encoded JSON messages, and receives status messages.

    Close codes:
        4000 — no meeting active
        4001 — invalid or missing PIN
        4002 — another guest is already connected
    """
    from meetingmind.audio_capture import AudioChunk, AudioSource

    # Validate before accepting — WebSocket close before accept() is a pre-close.
    if not _meeting.get("active"):
        await websocket.close(code=4000, reason="No meeting active")
        return
    if not pin or pin != _meeting.get("guest_pin"):
        await websocket.close(code=4001, reason="Invalid PIN")
        return
    if _meeting.get("guest_connected"):
        await websocket.close(code=4002, reason="Another guest is already connected")
        return

    await websocket.accept()
    _meeting["guest_connected"] = True
    _meeting["guest_ws"]        = websocket

    # Broadcast connection event to dashboard clients.
    await _manager.broadcast({"type": "guest_connected"})
    await websocket.send_text(json.dumps({"type": "status", "status": "connected"}))
    logger.info("Guest connected (PIN=%s).", pin)

    # Start transcription loop if Whisper is already ready.
    if _meeting.get("transcriber") is not None and _meeting.get("guest_queue") is not None:
        _gt = asyncio.create_task(_guest_transcription_loop())
        _gt.add_done_callback(_task_done_callback)
        _meeting["guest_task"] = _gt

    guest_q = _meeting.get("guest_queue")

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
                continue

            if msg.get("type") != "guest_audio":
                continue

            b64 = msg.get("data", "")
            try:
                audio = decode_guest_audio(b64)
            except ValueError as exc:
                logger.warning("Guest audio decode error: %s", exc)
                await websocket.send_text(
                    json.dumps({"type": "error", "message": str(exc)})
                )
                continue

            rms      = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            peak_rms = float(np.max(np.abs(audio))) if len(audio) else 0.0

            chunk = AudioChunk(
                data=audio,
                source=AudioSource.GUEST,
                timestamp=time.time(),
                rms=rms,
                peak_rms=peak_rms,
            )

            if guest_q is not None:
                try:
                    guest_q.put_nowait(chunk)
                except _queue_module.Full:
                    # Drop oldest, insert newest — same backpressure as AudioCapture.
                    try:
                        guest_q.get_nowait()
                    except _queue_module.Empty:
                        pass
                    guest_q.put_nowait(chunk)

    except WebSocketDisconnect:
        logger.info("Guest disconnected.")
    except Exception as exc:
        logger.error("Guest WS error: %s", exc)
    finally:
        # Cancel guest transcription task if it's ours.
        gt = _meeting.get("guest_task")
        if gt and not gt.done():
            gt.cancel()
            try:
                await gt
            except (asyncio.CancelledError, Exception):
                pass

        _meeting["guest_connected"] = False
        _meeting["guest_ws"]        = None
        _meeting["guest_task"]      = None

        await _manager.broadcast({"type": "guest_disconnected"})
        logger.info("Guest session ended.")


# ---------------------------------------------------------------------------
# Guest viewer link — S7-003  (read-only fact-check / key-facts viewer)
# ---------------------------------------------------------------------------

async def _broadcast_to_guest_viewers(payload: dict) -> None:
    """Push a JSON payload to all active guest viewer WebSockets."""
    for token, ws_list in list(_guest_viewer_ws.items()):
        info = _guest_tokens.get(token)
        if not info or not info["active"]:
            continue
        dead: list[WebSocket] = []
        for ws in ws_list:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_list.remove(ws)
        # TODO: CLOUD RELAY — push card payload to relay here
        _relay.push_card(token, payload)


@app.post("/guest/generate", summary="Generate a guest viewer link")
async def guest_generate() -> dict:
    """Create a shareable guest viewer link for the current meeting."""
    token = str(_uuid_mod.uuid4())
    _guest_tokens[token] = {"active": True, "guests": []}
    _guest_viewer_ws.setdefault(token, [])
    host_ip = _TAILSCALE_IP
    url = f"http://{host_ip}:8000/guest/view/{token}"
    return {"token": token, "url": url}


@app.get("/guest/view/{token}", summary="Serve guest viewer page")
async def guest_view_page(token: str):
    """Serve the read-only guest viewer HTML page."""
    info = _guest_tokens.get(token)
    if not info or not info["active"]:
        raise HTTPException(status_code=404, detail="Invalid or expired guest link")
    viewer_path = Path(__file__).parents[2] / "frontend" / "guest_viewer.html"
    if not viewer_path.exists():
        raise HTTPException(status_code=500, detail="Guest viewer page not found")
    return FileResponse(viewer_path, media_type="text/html", headers=_NO_CACHE_HEADERS)


@app.get("/guest/view/{token}/status", summary="Guest viewer connection status")
async def guest_view_status(token: str) -> dict:
    """Return active status and connected guest names for the host UI."""
    info = _guest_tokens.get(token)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown token")
    return {"active": info["active"], "guests": list(info["guests"])}


@app.delete("/guest/view/{token}", summary="Revoke a guest viewer link")
async def guest_view_revoke(token: str) -> dict:
    """Deactivate a guest viewer token and close all connected WebSockets."""
    info = _guest_tokens.get(token)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown token")
    info["active"] = False
    # Close all connected guest viewer WebSockets
    for ws in list(_guest_viewer_ws.get(token, [])):
        try:
            await ws.close(code=4010, reason="Link revoked by host")
        except Exception:
            pass
    _guest_viewer_ws.pop(token, None)
    _relay.deactivate_session(token)
    return {"status": "revoked", "token": token}


@app.websocket("/guest/ws/{token}")
async def guest_viewer_websocket(ws: WebSocket, token: str):
    """Read-only WebSocket for guest viewers receiving fact-checks and key facts."""
    info = _guest_tokens.get(token)
    if not info or not info["active"]:
        await ws.close(code=4010, reason="Invalid or expired guest link")
        return

    await ws.accept()
    _guest_viewer_ws.setdefault(token, []).append(ws)

    try:
        # First message must be the guest's name
        raw = await ws.receive_text()
        msg = json.loads(raw)
        guest_name = msg.get("name", "Anonymous")
        info["guests"].append(guest_name)
        logger.info("Guest viewer '%s' connected (token=%s…)", guest_name, token[:8])
        await ws.send_json({"type": "status", "status": "connected", "name": guest_name})

        # Keep connection alive — guest viewer is receive-only (server pushes)
        while True:
            # Wait for client messages (ping/pong keep-alive or disconnect)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                # Client can send pings
                try:
                    m = json.loads(data)
                    if m.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except (json.JSONDecodeError, Exception):
                    pass
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
            # Check if token was deactivated
            if not info["active"]:
                await ws.send_json({"type": "meeting_ended"})
                break

    except WebSocketDisconnect:
        logger.info("Guest viewer '%s' disconnected.", guest_name)
    except Exception as exc:
        logger.warning("Guest viewer WS error: %s", exc)
    finally:
        ws_list = _guest_viewer_ws.get(token, [])
        if ws in ws_list:
            ws_list.remove(ws)
        if guest_name in info.get("guests", []):
            info["guests"].remove(guest_name)


@app.post("/meeting/start", summary="Start audio capture and transcription")
async def meeting_start(
    model_size:          str            = _WHISPER_MODEL_DEFAULT,
    chunk_duration:      float          = _CHUNK_SECONDS_DEFAULT,
    privacy_mode:        bool           = True,
    language:            Optional[str]  = None,
    mic_device_index:    Optional[int]  = None,
    system_device_index: Optional[int]  = None,
) -> dict:
    """
    Initialises AudioCapture and StreamingTranscriber, then starts the
    background processing loop.

    Parameters
    ----------
    model_size:           Whisper model — tiny | base | small | medium | large.
                          'base' is the default — loads in ~5 s, real-time capable (~1 s/chunk).
                          Use 'small' for better accuracy (loads in ~12 s, ~4 s/chunk).
    chunk_duration:       Seconds of audio per transcript chunk (default 3).
    privacy_mode:         Print a local-processing confirmation banner (default True).
    language:             ISO-639-1 hint (e.g. 'en'). None = auto-detect per chunk.
    mic_device_index:     PyAudio device index for the microphone (from GET /devices).
                          None = auto-detected recommended mic.
    system_device_index:  PyAudio device index for system audio loopback.
                          None = auto-detected recommended system device (if any).
    """
    if _meeting["active"]:
        return JSONResponse(
            {"error": "A meeting is already active. POST /meeting/stop first."},
            status_code=400,
        )

    # Resolve device indices — use explicit override or fall back to auto-detected.
    resolved_mic = mic_device_index
    mic_source   = "user-specified"
    if resolved_mic is None and _detected and _detected.recommended_mic:
        resolved_mic = _detected.recommended_mic["index"]
        mic_source   = f"auto-detected ({_detected.recommended_mic['name']})"

    resolved_sys = system_device_index
    sys_source   = "user-specified"
    if resolved_sys is None and _detected and _detected.recommended_system:
        resolved_sys = _detected.recommended_system["index"]
        sys_source   = f"auto-detected ({_detected.recommended_system['name']})"
    elif resolved_sys is None:
        sys_source = "none available"

    logger.info(
        "Meeting start — mic index=%s (%s) | system index=%s (%s)",
        resolved_mic, mic_source, resolved_sys, sys_source,
    )

    # Import here to keep startup fast.
    from meetingmind.audio_capture import AudioCapture

    # AudioCapture.__init__ calls pyaudio.PyAudio() which can block on some
    # Windows audio drivers.  Run it in a thread to keep the event loop alive.
    try:
        capture = await asyncio.to_thread(
            lambda: AudioCapture(
                mic_device_index=resolved_mic,
                system_device_index=resolved_sys,
                chunk_duration=chunk_duration,
                privacy_mode=privacy_mode,
            )
        )
    except Exception as exc:
        logger.exception("AudioCapture init failed: %s", exc)
        return JSONResponse(
            {"error": f"PyAudio initialisation failed: {exc}"},
            status_code=500,
        )

    # Probe the mic device — fail fast with a clear error rather than silently
    # capturing nothing for the entire model-loading window.
    logger.info("Probing mic device (index=%s)…", resolved_mic)
    probe_ok, probe_name, probe_err = await asyncio.to_thread(
        capture.probe_device, resolved_mic
    )
    if not probe_ok:
        await asyncio.to_thread(capture._pa.terminate)
        err_lower = probe_err.lower()
        is_privacy = any(
            x in err_lower for x in ["access denied", "-9999", "-9997"]
        )
        suggestion = (
            "Check Windows Settings → Privacy & Security → Microphone → "
            "Allow apps to access your microphone"
            if is_privacy
            else "Try a different device index from GET /devices"
        )
        try:
            alt_names = [
                d["name"] for d in capture.list_devices()
                if d["index"] != resolved_mic
            ][:3]
        except Exception:
            alt_names = []
        logger.error("Device probe failed for '%s': %s", probe_name, probe_err)
        return JSONResponse(
            {
                "error":               "No working microphone found",
                "tried":               [probe_name],
                "reason":              probe_err,
                "suggestion":          suggestion,
                "alternative_devices": alt_names,
            },
            status_code=500,
        )

    # ── Phase 1: Start audio + RMS immediately ──────────────────────────────
    # Audio is live as soon as capture.start() returns (< 200 ms).
    # The RMS loop begins broadcasting at once — no Whisper wait.
    capture.start()

    # Use default threshold immediately — calibration runs in background.
    # Bluetooth/AirPods devices produce weaker RMS — use a lower threshold.
    _bt_keywords = (
        "airpods", "bluetooth", "wireless", "bt ", "buds", "earbuds",
        "headset", "find my", "jabra", "sony", "bose", "sennheiser",
        "erza", "jade",
    )
    _is_bt = any(kw in (probe_name or "").lower() for kw in _bt_keywords)
    _meeting["is_bt_device"] = _is_bt
    _meeting["device_name"] = probe_name or "unknown"
    _meeting["threshold_capped"] = False
    _meeting["threshold_cap_reason"] = ""
    _meeting["calibrated_threshold"] = 0.0005 if _is_bt else 0.001
    if _is_bt:
        logger.info("Bluetooth device detected ('%s') — using lower threshold 0.0005", probe_name)

    # Set device-aware hard RMS minimum — Intel mics get a permissive gate
    # (rely on Whisper confidence instead); BT mics get an aggressive gate.
    from meetingmind.transcriber import set_rms_minimum_for_device
    set_rms_minimum_for_device(probe_name or "", _meeting["calibrated_threshold"])

    # Reset diagnostic counters.
    _stats["chunks_from_queue"]        = 0
    _stats["chunks_filtered"]          = 0
    _stats["chunks_transcribed"]       = 0
    _stats["chunks_broadcast"]         = 0
    _stats["last_rms"]                 = 0.0
    _stats["recent_rms"]               = []
    _stats["whisper_times"]            = []
    _stats["avg_whisper_time_s"]       = 0.0
    _stats["_slow_whisper_warned"]     = False
    _stats["first_rms_delay_s"]        = None
    _stats["first_transcript_delay_s"] = None
    _stats["gate_silent"]              = 0
    _stats["gate_timeout"]             = 0
    _stats["gate_no_speech"]           = 0
    _stats["gate_empty"]               = 0
    _stats["gate_error"]               = 0
    _stats["gate_hallucination"]       = 0
    # System-audio per-source counters.
    _stats["system_chunks_from_queue"]  = 0
    _stats["system_chunks_filtered"]    = 0
    _stats["system_chunks_transcribed"] = 0
    _stats["system_gate_silent"]        = 0
    _stats["system_gate_hard_rms"]      = 0
    _stats["system_gate_no_speech"]     = 0
    _stats["system_gate_empty"]         = 0
    _stats["system_gate_error"]         = 0
    _stats["system_gate_hallucination"] = 0
    _stats["system_gate_low_confidence"]   = 0
    _stats["system_gate_avg_no_speech"]    = 0
    _stats["system_last_rms"]           = 0.0
    _stats["system_recent_rms"]         = []

    _meeting["active"]            = True
    _meeting["capture"]           = capture
    _meeting["transcriber"]       = None        # not ready yet
    _meeting["task"]              = None        # started after model loads
    _meeting["load_task"]         = None
    _meeting["start_time"]        = time.time()
    _meeting["model"]             = model_size
    _meeting["model_loading"]     = True
    _meeting["meeting_id"]        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _meeting["transcript_chunks"]  = []
    _meeting["decisions"]          = []
    _meeting["speaker_counts"]     = {"microphone": 0, "system": 0}
    _meeting["speaker_window"]     = []
    _meeting["coaching_cooldowns"] = {}
    _meeting["coaching_history"]   = []
    _meeting["coaching_dismissals"] = []
    _meeting["coaching_resolved"]  = set()
    _meeting["coaching_seq"]       = 0
    _rms = asyncio.create_task(_rms_broadcast_loop())
    _rms.add_done_callback(_task_done_callback)
    _meeting["rms_task"]           = _rms
    _tc = asyncio.create_task(_coaching_time_loop())
    _tc.add_done_callback(_task_done_callback)
    _meeting["time_coach_task"]    = _tc
    _meeting["guest_pin"]          = None
    _meeting["guest_connected"]    = False
    _meeting["guest_queue"]        = _queue_module.Queue(maxsize=_GUEST_QUEUE_MAXSIZE)
    _meeting["guest_task"]         = None
    _meeting["guest_ws"]           = None
    _meeting["fact_checks"]        = []

    # S8-009: Start [Them] audio monitor — warns after 2 min if no system transcription
    _sam = asyncio.create_task(_monitor_system_audio())
    _sam.add_done_callback(_task_done_callback)
    _meeting["system_audio_monitor_task"] = _sam

    # S8-002: Reset AI-detected action items & decisions
    _action_items.clear()
    _decisions_ai.clear()

    # S8-003: Reset scope alerts
    _scope_alerts.clear()

    # S8-004: Reset timeline alerts
    _timeline_alerts.clear()

    # Init fact checker from KB facts collection
    global _fact_checker
    _fact_checker = None
    if _kb is not None and _kb._collections.get("facts"):
        try:
            fc = FactChecker(_kb._collections["facts"])
            if fc.is_available():
                _fact_checker = fc
                logger.info("FactChecker initialised — facts collection available.")
        except Exception as exc:
            logger.warning("FactChecker init failed: %s", exc)

    # ── Phase 2: Load Whisper + calibrate in background ─────────────────────
    # Returns to the browser immediately; transcription starts once model is ready.
    # Default language to the config constant so the preloaded model language matches.
    _load = asyncio.create_task(
        _whisper_load_background(model_size, privacy_mode, language or _LANGUAGE_DEFAULT)
    )
    _load.add_done_callback(_task_done_callback)
    _meeting["load_task"] = _load
    _cal = asyncio.create_task(_calibrate_background(capture))
    _cal.add_done_callback(_task_done_callback)

    logger.info(
        "Meeting started — audio live, loading Whisper '%s' in background…",
        model_size,
    )
    return {
        "status":               "starting",
        "model":                model_size,
        "model_loading":        True,
        "chunk_duration":       chunk_duration,
        "language":             language or "auto-detect",
        "mic_device_index":     resolved_mic,
        "mic_source":           mic_source,
        "system_device_index":  resolved_sys,
        "system_source":        sys_source,
    }


@app.post("/meeting/stop", summary="Stop audio capture and transcription")
async def meeting_stop() -> dict:
    """
    Stops audio capture, cancels the processing loop, and releases resources.
    Connected WebSocket clients remain open but will receive no further chunks.
    """
    if not _meeting["active"]:
        return JSONResponse(
            {"error": "No meeting is currently active."},
            status_code=400,
        )

    global _last_summary

    # Capture ingestion data BEFORE clearing _meeting
    ingestion_meeting_id  = _meeting.get("meeting_id")
    ingestion_transcript_chunks = list(_meeting.get("transcript_chunks", []))
    ingestion_transcript  = " ".join(ingestion_transcript_chunks)
    ingestion_model_size  = _meeting.get("model") or "base"
    ingestion_started_at  = datetime.fromtimestamp(
        _meeting["start_time"] or time.time(), tz=timezone.utc
    )
    ingestion_stopped_at  = datetime.now(timezone.utc)
    ingestion_duration    = round(
        (ingestion_stopped_at - ingestion_started_at).total_seconds(), 1
    )

    # Build meeting summary snapshot BEFORE clearing _meeting state
    summary_decisions   = list(_meeting.get("decisions", []))
    summary_coaching    = list(_meeting.get("coaching_history", []))
    summary_speakers    = dict(_meeting.get("speaker_counts", {}))
    summary_actions     = extract_action_items(ingestion_transcript_chunks)
    summary_fact_checks = list(_meeting.get("fact_checks", []))
    # S8-002: Capture AI-detected action items & decisions before reset
    summary_ai_action_items = list(_action_items)
    summary_ai_decisions    = list(_decisions_ai)
    # S8-003: Capture scope alerts before reset
    summary_scope_alerts    = list(_scope_alerts)
    # S8-004: Capture timeline alerts before reset
    summary_timeline_alerts = list(_timeline_alerts)
    _last_summary = {
        "meeting_id":       ingestion_meeting_id,
        "date":             ingestion_started_at.isoformat(),
        "stopped_at":       ingestion_stopped_at.isoformat(),
        "duration_minutes": round(ingestion_duration / 60.0, 1),
        "duration_seconds": ingestion_duration,
        "transcript_chunks": len(ingestion_transcript_chunks),
        "decisions":        [
            {"text": d.get("text", ""), "phrase": d.get("phrase", "")}
            for d in summary_decisions
        ],
        "action_items":     summary_actions,
        "coaching_events":  len(summary_coaching),
        "coaching_details": [
            {
                "trigger_id": c.get("trigger_id", ""),
                "prompt":     c.get("prompt", ""),
                "timestamp":  c.get("timestamp"),
            }
            for c in summary_coaching
        ],
        "speaker_counts":   summary_speakers,
        "fact_checks":      summary_fact_checks,
        "fact_check_count": len(summary_fact_checks),
        # S8-002: AI-detected structured items
        "ai_action_items":  summary_ai_action_items,
        "ai_decisions":     summary_ai_decisions,
        # S8-003: Scope creep alerts
        "scope_alerts":     summary_scope_alerts,
        # S8-004: Timeline alerts
        "timeline_alerts":  summary_timeline_alerts,
        # S8-005: PM summary — populated asynchronously, None until ready
        "pm_summary":       None,
        # Internal: full transcript text for PM summary generation (not served in API)
        "_transcript_text": ingestion_transcript,
    }

    _meeting["active"] = False

    # Cancel all background tasks (load_task first — it may spawn the transcription task)
    for key in ("load_task", "task", "rms_task", "time_coach_task", "guest_task", "system_audio_monitor_task"):
        t = _meeting.get(key)
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    # Close guest WebSocket if still open
    guest_ws = _meeting.get("guest_ws")
    if guest_ws:
        try:
            await guest_ws.close()
        except Exception:
            pass

    # Stop audio capture and release resources
    if _meeting["capture"]:
        await asyncio.to_thread(_meeting["capture"].close)

    elapsed = round(time.time() - (_meeting["start_time"] or time.time()), 1)

    _meeting["capture"]           = None
    _meeting["transcriber"]       = None
    _meeting["task"]              = None
    _meeting["rms_task"]            = None
    _meeting["time_coach_task"]     = None
    _meeting["load_task"]           = None
    _meeting["system_audio_monitor_task"] = None
    _meeting["model_loading"]       = False
    _meeting["start_time"]          = None
    _meeting["model"]               = None
    _meeting["meeting_id"]          = None
    _meeting["transcript_chunks"]   = []
    _meeting["decisions"]           = []
    _meeting["speaker_counts"]      = {"microphone": 0, "system": 0}
    _meeting["speaker_window"]      = []
    _meeting["coaching_cooldowns"]  = {}
    _meeting["coaching_history"]    = []
    _meeting["coaching_dismissals"] = []
    _meeting["coaching_resolved"]   = set()
    _meeting["coaching_seq"]        = 0
    _meeting["guest_pin"]           = None
    _meeting["guest_ws"]            = None
    _meeting["guest_connected"]     = False
    _meeting["guest_queue"]         = None
    _meeting["guest_task"]          = None
    _meeting["fact_checks"]         = []

    # S8-002: Reset AI-detected action items & decisions
    _action_items.clear()
    _decisions_ai.clear()

    # S8-003: Reset scope alerts
    _scope_alerts.clear()

    # S8-004: Reset timeline alerts
    _timeline_alerts.clear()

    # Deactivate all guest viewer tokens for this meeting
    for tok, info in _guest_tokens.items():
        if info["active"]:
            info["active"] = False
            for gws in list(_guest_viewer_ws.get(tok, [])):
                try:
                    await gws.send_json({"type": "meeting_ended"})
                    await gws.close(code=4010, reason="Meeting ended")
                except Exception:
                    pass
            _guest_viewer_ws.pop(tok, None)
            # TODO: CLOUD RELAY — push deactivation signal to relay here
            _relay.deactivate_session(tok)

    # Unload Whisper process pool to free ~400 MB between meetings.
    try:
        from meetingmind.transcriber import shutdown_process_pool
        shutdown_process_pool()
        logger.info("Whisper process pool shut down — memory released.")
    except Exception as exc:
        logger.warning("Whisper process pool shutdown failed: %s", exc)

    # Launch background ingestion (non-blocking — returns in <1 s).
    # _ensure_kb() lazily initialises KB on first stop.
    if ingestion_meeting_id and ingestion_transcript.strip():
        _ig = asyncio.create_task(_ingest_meeting_background(
            ingestion_meeting_id,
            ingestion_started_at,
            ingestion_stopped_at,
            ingestion_duration,
            ingestion_model_size,
            ingestion_transcript,
        ))
        _ig.add_done_callback(_task_done_callback)

    # S8-005: Launch PM summary generation (non-blocking — never delays stop).
    if ingestion_transcript.strip():
        _pm = asyncio.create_task(_generate_pm_summary_background())
        _pm.add_done_callback(_task_done_callback)

    logger.info("Meeting stopped after %.1f seconds.", elapsed)
    return {
        "status":          "stopped",
        "elapsed_seconds": elapsed,
        "meeting_id":      ingestion_meeting_id,
    }


@app.get("/meeting/reset",  include_in_schema=False)
@app.post("/meeting/reset", summary="Force-reset any stuck meeting state")
async def meeting_reset() -> dict:
    """
    Force-clears all meeting state regardless of the current active flag.

    Use this to recover from a 'meeting already active' error caused by
    a previous session that didn't call /meeting/stop (e.g. a browser
    crash, server restart with stale in-memory state, or a lost connection).
    Safe to call even when no meeting is active.
    """
    was_active = _meeting["active"]
    _meeting["active"] = False

    for key in ("load_task", "task", "rms_task", "time_coach_task", "guest_task", "system_audio_monitor_task"):
        t = _meeting.get(key)
        if t:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # Close guest WebSocket if still open
    guest_ws = _meeting.get("guest_ws")
    if guest_ws:
        try:
            await guest_ws.close()
        except Exception:
            pass

    if _meeting["capture"]:
        try:
            await asyncio.to_thread(_meeting["capture"].close)
        except Exception:
            pass

    _meeting["capture"]           = None
    _meeting["transcriber"]         = None
    _meeting["task"]                = None
    _meeting["rms_task"]            = None
    _meeting["time_coach_task"]     = None
    _meeting["load_task"]           = None
    _meeting["system_audio_monitor_task"] = None
    _meeting["model_loading"]       = False
    _meeting["start_time"]          = None
    _meeting["model"]               = None
    _meeting["meeting_id"]          = None
    _meeting["transcript_chunks"]   = []
    _meeting["decisions"]           = []
    _meeting["speaker_counts"]      = {"microphone": 0, "system": 0}
    _meeting["speaker_window"]      = []
    _meeting["coaching_cooldowns"]  = {}
    _meeting["coaching_history"]    = []
    _meeting["coaching_dismissals"] = []
    _meeting["coaching_resolved"]   = set()
    _meeting["coaching_seq"]        = 0
    _meeting["guest_pin"]           = None
    _meeting["guest_ws"]            = None
    _meeting["guest_connected"]     = False
    _meeting["guest_queue"]         = None
    _meeting["guest_task"]          = None
    _meeting["fact_checks"]         = []

    # S8-002: Reset AI-detected action items & decisions
    _action_items.clear()
    _decisions_ai.clear()

    # S8-003: Reset scope alerts
    _scope_alerts.clear()

    # S8-004: Reset timeline alerts
    _timeline_alerts.clear()

    logger.info("Meeting state force-reset (was_active=%s).", was_active)
    return {"status": "reset", "was_active": was_active}


# ── Knowledge ingestion ───────────────────────────────────────────────────────


class _IngestTextRequest(BaseModel):
    text: str
    doc_id: str = "onboarding_seed"
    collection: str = "summaries"


@app.post("/knowledge/ingest/text", summary="Ingest plain text into the knowledge base")
async def ingest_text(req: _IngestTextRequest) -> dict:
    """Ingest onboarding or seed text directly into ChromaDB."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    if _kb is None:
        return {"status": "skipped", "reason": "knowledge base not available"}
    ok = await asyncio.to_thread(_kb.ingest_text, req.doc_id, req.text, req.collection)
    return {"status": "ok" if ok else "failed", "doc_id": req.doc_id}


# ── KB Seed endpoints ─────────────────────────────────────────────────────────

_seed_status: dict = {"facts_count": 0, "last_seeded": None, "entities": {}}


class _KBSeedRequest(BaseModel):
    content: str
    label: str = "pre_meeting_brief"


@app.post("/knowledge/seed", summary="Seed the knowledge base with pre-meeting context")
async def seed_knowledge_base(req: _KBSeedRequest) -> dict:
    """Seed the KB with context (people, budgets, dates) for fact-checking."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content must not be empty")

    await _ensure_kb()
    if _kb is None:
        raise HTTPException(status_code=500, detail="Knowledge base not available")

    # Split content into paragraphs for granular retrieval
    paragraphs = [p.strip() for p in req.content.split("\n\n") if p.strip()]
    if not paragraphs:
        raise HTTPException(status_code=400, detail="Content must not be empty")

    ts = datetime.now(timezone.utc).isoformat()
    stored = 0
    for i, para in enumerate(paragraphs):
        doc_id = f"seed_{ts}_{i}"
        ok = await asyncio.to_thread(
            _kb.ingest_text, doc_id, para, "facts",
            {"source": "kb_seed", "label": req.label, "seeded_at": ts},
        )
        if ok:
            stored += 1

    # Extract entities from the full content
    entities: dict = {}
    try:
        from meetingmind.spacy_extractor import extract_entities
        entities = await asyncio.to_thread(extract_entities, req.content)
    except Exception as exc:
        logger.warning("Entity extraction failed during KB seed: %s", exc)

    # Extract percentages (spaCy doesn't catch PERCENT well)
    import re as _re
    pct_matches = _re.findall(r"\d+(?:\.\d+)?%", req.content)
    if pct_matches:
        entities.setdefault("percentages", [])
        entities["percentages"].extend(pct_matches)

    # Update seed status
    _seed_status["facts_count"] = _seed_status.get("facts_count", 0) + stored
    _seed_status["last_seeded"] = ts
    _seed_status["entities"] = entities

    logger.info("KB seed complete: %d facts stored, entities=%s", stored, list(entities.keys()))
    return {"facts_stored": stored, "entities_extracted": entities}


@app.get("/knowledge/seed/status", summary="Get KB seed status")
async def get_seed_status() -> dict:
    """Return live facts_count from ChromaDB + last_seeded + entities."""
    # Try to get live count from ChromaDB
    live_count = _seed_status.get("facts_count", 0)
    if _kb is not None and _kb._collections.get("facts"):
        try:
            live_count = await asyncio.to_thread(_kb._collections["facts"].count)
        except Exception:
            pass

    return {
        "facts_count": live_count,
        "last_seeded": _seed_status.get("last_seeded"),
        "entities": _seed_status.get("entities", {}),
    }


# ── S8-006: KB status & reseed ────────────────────────────────────────────────


@app.get("/kb/status", summary="Knowledge base status overview")
async def kb_status() -> dict:
    """Return document counts for PM global KB and project KB."""
    await _ensure_kb()

    pm_count = 0
    pm_files: list[str] = []
    project_count = 0
    project_name = None
    facts_count = 0

    if _kb is not None:
        pm_count = await asyncio.to_thread(_kb.pm_global_doc_count)
        pm_files = await asyncio.to_thread(_kb.pm_global_files)
        project_count = await asyncio.to_thread(_kb.project_doc_count)
        if _kb._collections.get("facts"):
            try:
                facts_count = await asyncio.to_thread(_kb._collections["facts"].count)
            except Exception:
                pass

    if _active_project:
        project_name = _active_project.get("project_name")

    total = pm_count + project_count + facts_count

    return {
        "pm_global": {
            "document_count": pm_count,
            "files": pm_files,
            "last_seeded": _pm_global_last_seeded,
        },
        "project_kb": {
            "document_count": project_count + facts_count,
            "project_name": project_name,
        },
        "total_documents": total,
    }


@app.post("/kb/reseed", summary="Force reseed PM global knowledge base")
async def kb_reseed() -> dict:
    """Clear and re-ingest all PM global KB markdown files."""
    global _pm_global_last_seeded
    await _ensure_kb()
    if _kb is None:
        raise HTTPException(status_code=500, detail="Knowledge base not available")

    n = await asyncio.to_thread(_kb.force_ingest_pm_global_kb)
    _pm_global_last_seeded = datetime.now(timezone.utc).isoformat()
    logger.info("PM Global KB reseeded: %d documents.", n)
    return {"status": "reseeded", "documents": n}


# ── AI Suggestions endpoint ───────────────────────────────────────────────────

@app.post("/coaching/dismiss/{prompt_id}", summary="Log a coaching prompt dismissal")
async def dismiss_coaching(prompt_id: str) -> dict:
    """
    Record that the user dismissed a coaching prompt.

    Called by the frontend dismiss (✕) button and by the 8-second auto-dismiss
    timer.  The dismissal is appended to the current meeting's
    ``coaching_dismissals`` list for later analytics.

    Args:
        prompt_id: The ``id`` field from the coaching WS broadcast
                   (e.g. "no_decision_owner_3").
    """
    record = {"prompt_id": prompt_id, "dismissed_at": time.time()}
    _meeting["coaching_dismissals"].append(record)
    logger.info("COACHING dismissed: %r", prompt_id)
    return {"status": "ok", "prompt_id": prompt_id}


class _MilestoneItem(BaseModel):
    name: str
    date: str

class _TeamMember(BaseModel):
    name: str
    role: str

class _ProjectBriefRequest(BaseModel):
    project_id: Optional[str] = None
    project_name: str
    project_type: str = "Software"
    methodology: list[str] = []
    cpmai_phase: Optional[str] = None
    objectives: str = ""
    scope_items: list[str] = []
    out_of_scope: list[str] = []
    budget_total: Optional[float] = None
    budget_remaining: Optional[float] = None
    budget_currency: str = "USD"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    milestones: list[_MilestoneItem] = []
    team: list[_TeamMember] = []
    risks: list[str] = []
    success_criteria: str = ""


class _SuggestionsRequest(BaseModel):
    transcript: str
    context: Optional[str] = None
    model: str = "claude-sonnet-4-6"


@app.get("/meeting/decisions", summary="List decisions detected in the current (or last) meeting")
async def get_decisions() -> dict:
    """
    Returns all decision events detected during the current meeting.

    Each event contains:
        text      – the transcript chunk that triggered the detection
        phrase    – the matched decision phrase
        timestamp – Unix timestamp of the chunk

    If no meeting is active, returns the decisions from the most recent
    meeting (until a new meeting starts and resets the list).
    """
    return {
        "meeting_id": _meeting.get("meeting_id"),
        "active":     _meeting["active"],
        "decisions":  list(_meeting.get("decisions", [])),
        "count":      len(_meeting.get("decisions", [])),
    }


@app.get("/meeting/coaching", summary="All coaching prompts fired this meeting")
async def get_coaching() -> dict:
    """
    Returns the full coaching history for the current (or most recent) meeting.

    Each item in ``prompts_fired`` contains:
        id          – unique prompt ID (e.g. "no_decision_owner_3")
        trigger_id  – which COACHING_TRIGGERS entry fired
        prompt      – the text shown to the user
        confidence  – trigger confidence score
        matched     – matched phrase
        timestamp   – Unix time the prompt fired

    ``prompts_dismissed`` lists IDs the user explicitly dismissed and when.
    """
    return {
        "meeting_id":        _meeting.get("meeting_id"),
        "active":            _meeting["active"],
        "speaker_counts":    dict(_meeting.get("speaker_counts", {})),
        "prompts_fired":     list(_meeting.get("coaching_history", [])),
        "prompts_dismissed": list(_meeting.get("coaching_dismissals", [])),
    }


@app.get("/meeting/fact_checks", summary="Fact-check alerts from the current meeting")
async def get_fact_checks() -> dict:
    """
    Returns all fact-check alerts detected during the current meeting.

    Each alert compares a spoken numeric entity against a KB fact, showing
    variance percentage and severity (matches / notable / verify).
    """
    return {
        "meeting_id":  _meeting.get("meeting_id"),
        "active":      _meeting["active"],
        "fact_checks": list(_meeting.get("fact_checks", [])),
        "count":       len(_meeting.get("fact_checks", [])),
    }


# ---------------------------------------------------------------------------
# S8-002: Action Item & Decision Capture — REST endpoints
# ---------------------------------------------------------------------------

@app.get("/meeting/action-items", summary="List AI-detected action items from the current meeting")
async def get_action_items() -> dict:
    """Returns all AI-detected action items captured during the current meeting."""
    return {
        "meeting_id":   _meeting.get("meeting_id"),
        "active":       _meeting["active"],
        "action_items": list(_action_items),
        "count":        len(_action_items),
    }


@app.get("/meeting/ai-decisions", summary="List AI-detected decisions from the current meeting")
async def get_ai_decisions() -> dict:
    """Returns all AI-detected decisions captured during the current meeting."""
    return {
        "meeting_id": _meeting.get("meeting_id"),
        "active":     _meeting["active"],
        "decisions":  list(_decisions_ai),
        "count":      len(_decisions_ai),
    }


@app.delete("/meeting/action-items/{item_id}", summary="Dismiss an action item")
async def delete_action_item(item_id: str) -> dict:
    """Remove an action item by ID (PM dismissed it)."""
    for i, item in enumerate(_action_items):
        if item.get("id") == item_id:
            _action_items.pop(i)
            logger.info("Action item dismissed: %s", item_id)
            return {"status": "deleted", "id": item_id}
    raise HTTPException(status_code=404, detail="Action item not found")


@app.delete("/meeting/ai-decisions/{decision_id}", summary="Dismiss an AI decision")
async def delete_ai_decision(decision_id: str) -> dict:
    """Remove a decision by ID (PM dismissed it)."""
    for i, dec in enumerate(_decisions_ai):
        if dec.get("id") == decision_id:
            _decisions_ai.pop(i)
            logger.info("AI decision dismissed: %s", decision_id)
            return {"status": "deleted", "id": decision_id}
    raise HTTPException(status_code=404, detail="Decision not found")


# ---------------------------------------------------------------------------
# S8-003: Scope Creep Alert endpoints
# ---------------------------------------------------------------------------

@app.get("/meeting/scope-alerts", summary="List scope creep alerts from the current meeting")
async def get_scope_alerts() -> dict:
    """Returns all scope creep alerts detected during the current meeting."""
    return {
        "meeting_id":   _meeting.get("meeting_id"),
        "active":       _meeting["active"],
        "scope_alerts": list(_scope_alerts),
        "count":        len(_scope_alerts),
    }


@app.delete("/meeting/scope-alerts/{alert_id}", summary="Dismiss a scope alert")
async def delete_scope_alert(alert_id: str) -> dict:
    """Remove a scope alert by ID (PM dismissed it)."""
    for i, alert in enumerate(_scope_alerts):
        if alert.get("id") == alert_id:
            _scope_alerts.pop(i)
            logger.info("Scope alert dismissed: %s", alert_id)
            return {"status": "deleted", "id": alert_id}
    raise HTTPException(status_code=404, detail="Scope alert not found")


# ---------------------------------------------------------------------------
# S8-004: Timeline Alert endpoints
# ---------------------------------------------------------------------------

@app.get("/meeting/timeline-alerts", summary="List timeline alerts from the current meeting")
async def get_timeline_alerts() -> dict:
    """Returns all timeline/delay alerts detected during the current meeting."""
    return {
        "meeting_id":       _meeting.get("meeting_id"),
        "active":           _meeting["active"],
        "timeline_alerts":  list(_timeline_alerts),
        "count":            len(_timeline_alerts),
    }


@app.delete("/meeting/timeline-alerts/{alert_id}", summary="Dismiss a timeline alert")
async def delete_timeline_alert(alert_id: str) -> dict:
    """Remove a timeline alert by ID (PM dismissed it)."""
    for i, alert in enumerate(_timeline_alerts):
        if alert.get("id") == alert_id:
            _timeline_alerts.pop(i)
            logger.info("Timeline alert dismissed: %s", alert_id)
            return {"status": "deleted", "id": alert_id}
    raise HTTPException(status_code=404, detail="Timeline alert not found")


@app.get("/meeting/summary", summary="Structured summary of the last completed meeting")
async def get_meeting_summary() -> dict:
    """
    Returns a structured JSON summary of the most recently stopped meeting.

    Includes: date, duration, decisions detected, action items extracted
    (pattern-based, no API call), coaching events fired, and speaker counts.

    The summary is captured at meeting stop time and persists until the next
    meeting is stopped.  Returns 404 if no meeting has been completed yet.
    """
    if _last_summary is None:
        return JSONResponse(
            {"error": "No meeting summary available. Complete a meeting first."},
            status_code=404,
        )
    # Strip internal keys (leading underscore) before serving
    return {k: v for k, v in _last_summary.items() if not k.startswith("_")}


@app.post("/suggestions", summary="Extract key facts from transcript")
async def suggestions(body: _SuggestionsRequest) -> dict:
    """
    Sends the transcript snippet to Claude and extracts key facts:
    decision, owner, deadline, risk, and next action.

    Returns {"facts": {decision, owner, deadline, risk, action}} where each
    value is a short phrase extracted from the transcript, or null.

    The transcript text is the only data sent externally (Anthropic API over HTTPS).
    """
    if not body.transcript.strip():
        return JSONResponse({"error": "transcript must not be empty."}, status_code=422)

    try:
        from meetingmind.suggestion_engine import get_key_facts  # lazy import
    except ImportError as exc:
        return JSONResponse(
            {"error": f"suggestion_engine not available: {exc}"},
            status_code=500,
        )

    # Query ChromaDB for relevant KB context — lazy-init on first use.
    await _ensure_kb()
    historical_context: Optional[str] = None
    if _ctx is not None:
        try:
            grounded = await asyncio.to_thread(_ctx.get_context, body.transcript)
            historical_context = grounded.historical_context
            if historical_context:
                logger.info(
                    "KB context injected into key facts (%d items from %d meetings).",
                    grounded.item_count,
                    len(grounded.source_meeting_ids),
                )
        except Exception as exc:
            logger.warning("ContextEngine query failed (continuing without KB): %s", exc)

    project_ctx = _build_project_context(_active_project) if _active_project else None
    try:
        facts = await asyncio.to_thread(
            get_key_facts, body.transcript, body.model,
            historical_context=historical_context,
            project_context=project_ctx,
        )
    except (EnvironmentError, RuntimeError) as exc:
        logger.error("Key facts error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Push key facts to guest viewers
    # TODO: CLOUD RELAY — push key facts payload to relay here
    try:
        await _broadcast_to_guest_viewers({"type": "key_facts", "data": facts})
    except Exception:
        pass

    return {"facts": facts, "kb_grounded": historical_context is not None}


# ── Debug / diagnostics (not shown in API docs) ───────────────────────────────

@app.get("/debug", include_in_schema=False)
async def debug_pipeline() -> dict:
    """
    Real-time snapshot of the full pipeline.

    Use this to pinpoint where chunks stop flowing:
      chunks_from_queue  > 0  → PyAudio is capturing
      chunks_filtered    > 0  → Whisper is running but some chunks are silence/noise
      chunks_transcribed > 0  → Whisper is producing text
      chunks_broadcast   > 0  → WebSocket clients are receiving text

    If chunks_from_queue == 0 after 5+ seconds, PyAudio is not capturing.
    If chunks_transcribed == 0 but chunks_from_queue > 0, all chunks are filtered.
    If chunks_broadcast == 0 but chunks_transcribed > 0, no WS clients connected.
    """
    queue_size    = 0
    threads_alive: list[bool] = []
    thread_names: list[str]   = []

    if _meeting["capture"]:
        queue_size    = _meeting["capture"]._chunk_queue.qsize()
        threads_alive = [t.is_alive() for t in _meeting["capture"]._threads]
        thread_names  = [t.name      for t in _meeting["capture"]._threads]

    elapsed: Optional[float] = None
    if _meeting["active"] and _meeting["start_time"]:
        elapsed = round(time.time() - _meeting["start_time"], 1)

    return {
        "meeting_active":      _meeting["active"],
        "elapsed_seconds":     elapsed,
        "model":               _meeting["model"],
        "connected_ws_clients": _manager.count,
        "audio_queue_size":    queue_size,
        "audio_threads":       dict(zip(thread_names, threads_alive)),
        "pipeline_stats":      dict(_stats),
        "gate_breakdown": {
            "silent":         _stats["gate_silent"],
            "timeout":        _stats["gate_timeout"],
            "no_speech":      _stats["gate_no_speech"],
            "empty":          _stats["gate_empty"],
            "error":          _stats["gate_error"],
            "hallucination":  _stats["gate_hallucination"],
            "hard_rms":       _stats.get("gate_hard_rms", 0),
            "low_confidence": _stats.get("gate_low_confidence", 0),
            "avg_no_speech":  _stats.get("gate_avg_no_speech", 0),
        },
        "calibrated_threshold": _meeting.get("calibrated_threshold"),
        "threshold_capped":     _meeting.get("threshold_capped", False),
        "threshold_cap_reason": _meeting.get("threshold_cap_reason", ""),
        "system_rms": {
            "last":   _stats.get("system_last_rms", 0.0),
            "recent": _stats.get("system_recent_rms", []),
        },
        "system_pipeline": {
            "chunks_from_queue":  _stats.get("system_chunks_from_queue", 0),
            "chunks_filtered":    _stats.get("system_chunks_filtered", 0),
            "chunks_transcribed": _stats.get("system_chunks_transcribed", 0),
        },
        "system_gate_breakdown": {
            "silent":         _stats.get("system_gate_silent", 0),
            "hard_rms":       _stats.get("system_gate_hard_rms", 0),
            "no_speech":      _stats.get("system_gate_no_speech", 0),
            "empty":          _stats.get("system_gate_empty", 0),
            "error":          _stats.get("system_gate_error", 0),
            "hallucination":  _stats.get("system_gate_hallucination", 0),
            "low_confidence": _stats.get("system_gate_low_confidence", 0),
            "avg_no_speech":  _stats.get("system_gate_avg_no_speech", 0),
        },
        "system_thresholds": {
            "silence_peak":   0.0001,   # _SYSTEM_SILENCE_PEAK_THRESHOLD
            "hard_rms_floor": 0.00003,  # _SYSTEM_HARD_RMS_MINIMUM
            "note": "SYSTEM chunks bypass calibrated_threshold entirely",
        },
        "timing": {
            "first_rms_delay_s":        _stats.get("first_rms_delay_s"),
            "first_transcript_delay_s": _stats.get("first_transcript_delay_s"),
            "avg_whisper_time_s":       _stats.get("avg_whisper_time_s"),
            "queue_depth": (
                _meeting["capture"]._chunk_queue.qsize()
                if _meeting.get("capture") else 0
            ),
        },
        "diagnosis": (
            "PyAudio not capturing — check mic permissions / device"
            if _meeting["active"] and _stats["chunks_from_queue"] == 0 and (elapsed or 0) > 5
            else "All chunks silenced — mic muted or RMS threshold too high"
            if _stats["chunks_from_queue"] > 0 and _stats["chunks_transcribed"] == 0
            else (
                f"High filtration rate "
                f"({_stats['chunks_filtered']}/{_stats['chunks_from_queue']} chunks silenced) — "
                "mic too quiet; try a lower mic_device_index (GET /devices) or raise mic volume"
            )
            if (
                _stats["chunks_from_queue"] >= 10
                and _stats["chunks_filtered"] / _stats["chunks_from_queue"] > 0.90
            )
            else "No WS clients connected — check browser WebSocket URL"
            if _stats["chunks_transcribed"] > 0 and _stats["chunks_broadcast"] == 0
            else "Pipeline OK" if _stats["chunks_broadcast"] > 0
            else "Waiting for audio..."
        ),
    }


@app.post("/calibrate", summary="Re-calibrate silence threshold mid-meeting")
async def calibrate() -> dict:
    """
    Re-measure ambient noise and update the silence threshold.

    Reads ~2 seconds of audio from the capture queue, computes the ambient
    peak RMS, and sets the threshold to 3x ambient (clamped to [0.0001, 0.005]).
    The updated threshold is applied to the active StreamingTranscriber immediately.
    """
    if not _meeting.get("active") or not _meeting.get("capture"):
        return JSONResponse(
            {"error": "No active meeting. POST /meeting/start first."},
            status_code=400,
        )

    # Read RMS values from _stats (populated by _rms_broadcast_loop) over ~2 s.
    # This avoids stealing chunks from the transcription pipeline.
    cal_peaks: list[float] = []
    for _ in range(4):
        await asyncio.sleep(0.5)
        for v in list(_stats.get("recent_rms", [])):
            if v > 0:
                cal_peaks.append(v)

    if cal_peaks:
        ambient_peak = max(cal_peaks)
        new_threshold = max(0.0001, min(0.002, ambient_peak * 3.0))
    else:
        return JSONResponse(
            {"error": "No RMS data captured during calibration window."},
            status_code=500,
        )

    old_threshold = _meeting.get("calibrated_threshold", 0.001)
    _meeting["calibrated_threshold"] = new_threshold

    transcriber = _meeting.get("transcriber")
    if transcriber is not None:
        transcriber._silence_threshold = new_threshold

    logger.info(
        "Re-calibration: %d samples, ambient_peak=%.5f, old=%.4f → new=%.4f",
        len(cal_peaks), ambient_peak, old_threshold, new_threshold,
    )

    return {
        "status":            "calibrated",
        "samples":           len(cal_peaks),
        "ambient_peak":      round(ambient_peak, 5),
        "old_threshold":     round(old_threshold, 5),
        "new_threshold":     round(new_threshold, 5),
    }


# ---------------------------------------------------------------------------
# Manual [Them] transcript injection (Option B)
# ---------------------------------------------------------------------------

class _ThemTranscriptRequest(BaseModel):
    text: str


@app.post(
    "/meeting/them-transcript",
    summary="Inject manual [Them] transcript text into the pipeline",
)
async def inject_them_transcript(req: _ThemTranscriptRequest) -> dict:
    """
    Accept copy-pasted text from Zoom/Meet captions or manual [Them] input.

    The text is split into sentence-sized chunks, each injected as a
    ``source="system"`` TranscriptChunk.  Every chunk flows through the
    full pipeline: broadcast to WS clients, coaching detection, decision
    detection, speaker counts, and transcript accumulation.

    Requires an active meeting.
    """
    if not _meeting.get("active"):
        return JSONResponse(
            {"error": "No active meeting. POST /meeting/start first."},
            status_code=400,
        )

    raw = (req.text or "").strip()
    if not raw:
        return JSONResponse(
            {"error": "Empty text — nothing to inject."},
            status_code=400,
        )

    # Split into sentence-ish chunks so coaching/decision detection
    # sees individual statements, not one giant blob.
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
    if not sentences:
        sentences = [raw]

    from meetingmind.transcriber import TranscriptChunk

    injected = 0
    now = time.time()

    for i, sentence in enumerate(sentences):
        chunk = TranscriptChunk(
            text=sentence,
            language="en",
            source="system",
            timestamp=now + i * 0.1,
            confidence=1.0,
            rms=0.0,
            is_reliable=True,
        )

        # Broadcast to all WS clients.
        try:
            await _manager.broadcast(chunk.to_dict())
            _stats["chunks_broadcast"] += 1
        except Exception as exc:
            logger.warning("Broadcast error (them-transcript): %s", exc)

        # Accumulate for summary / key-facts.
        _meeting["transcript_chunks"].append(chunk.text)

        # Decision detection.
        decision = detect_decision(chunk.text)
        if decision["is_decision"]:
            event = {
                "type":      "decision",
                "text":      chunk.text,
                "phrase":    decision["phrase"],
                "timestamp": chunk.timestamp,
            }
            _meeting["decisions"].append(event)
            try:
                await _manager.broadcast(event)
            except Exception as exc:
                logger.warning("Broadcast error (them-decision): %s", exc)

        # Speaker tracking.
        _meeting["speaker_counts"]["system"] = (
            _meeting["speaker_counts"].get("system", 0) + 1
        )

        # Coaching detection.
        for coaching in detect_coaching(chunk.text, role=_active_role):
            await _fire_coaching(coaching, timestamp=chunk.timestamp)

        # Auto-resolve coaching (snapshot to avoid interleaving with
        # _transcription_loop which also iterates coaching_history).
        resolved_set: set = _meeting.setdefault("coaching_resolved", set())
        for event in list(_meeting["coaching_history"]):
            cid = event["id"]
            tid = event["trigger_id"]
            if cid in resolved_set:
                continue
            if detect_coaching_resolution(tid, chunk.text):
                resolved_set.add(cid)
                try:
                    await _manager.broadcast({
                        "type": "coaching_resolved",
                        "id":   cid,
                        "note": "✓ Auto-detected in conversation",
                    })
                except Exception as exc:
                    logger.warning("Broadcast error (them-coaching-resolved): %s", exc)

        injected += 1

    logger.info(
        "Injected %d [Them] transcript chunk(s) from manual input (%d chars)",
        injected, len(raw),
    )
    return {"status": "injected", "chunks": injected, "chars": len(raw)}


@app.post("/debug/inject", include_in_schema=False)
async def debug_inject() -> dict:
    """
    Inject a fake transcript chunk into the WebSocket broadcast.

    Proves end-to-end that the WebSocket → frontend pipeline works
    independently of PyAudio and Whisper. If this appears in the UI,
    the JS / WebSocket side is fine and the bug is in audio capture or
    transcription. If it does NOT appear, the WebSocket connection is broken.
    """
    if _manager.count == 0:
        return JSONResponse(
            {"error": "No WebSocket clients connected. Open the dashboard first."},
            status_code=400,
        )
    test_chunk = {
        "text":       "[DEBUG] WebSocket pipeline test — if you see this, WS is working.",
        "language":   "en",
        "source":     "microphone",
        "timestamp":  time.time(),
        "confidence": 1.0,
    }
    await _manager.broadcast(test_chunk)
    logger.info("Debug chunk injected to %d client(s).", _manager.count)
    return {"status": "injected", "clients": _manager.count}


@app.post("/debug/test-whisper", include_in_schema=False)
async def debug_test_whisper() -> dict:
    """
    Run a synthetic 3-second audio chunk through the loaded Whisper model.

    Requires a meeting to be active (so the model is already loaded).
    Reports exactly which stage of the pipeline filtered the chunk:
      passed_silence_gate=False  → chunk.rms < threshold (silence gate blocked it)
      whisper_result=null        → Whisper returned no_speech or empty text
      whisper_result=<text>      → Pipeline is fully working end-to-end

    Use case: start a meeting, wait for model_ready, then call this endpoint
    from curl to verify Whisper is responding without needing to speak.

    curl.exe -X POST http://127.0.0.1:8000/debug/test-whisper
    """
    if _meeting["transcriber"] is None:
        status = "loading" if _meeting.get("model_loading") else "not_started"
        return JSONResponse(
            {"error": f"No model loaded yet (status: {status}). "
                      "Start a meeting and wait for model_ready."},
            status_code=400,
        )

    import numpy as np
    from meetingmind.audio_capture import AudioChunk, AudioSource

    # Low-amplitude white noise: RMS ~0.001, well above silence threshold (0.00005).
    # Whisper will likely filter it as no_speech — that's expected and still confirms
    # the model is running correctly.  Real speech will always produce output.
    rng    = np.random.default_rng(42)
    audio  = rng.standard_normal(16_000 * 3).astype(np.float32) * 0.001
    chunk  = AudioChunk(
        data=audio,
        source=AudioSource.MICROPHONE,
        timestamp=time.time(),
        rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
    )

    passed_gate = chunk.rms >= _SILENCE_THRESHOLD
    whisper_out = None
    error_msg   = None

    if passed_gate:
        try:
            result = await _meeting["transcriber"].async_transcribe_chunk(chunk)
            whisper_out = result.to_dict() if result else None
        except Exception as exc:
            error_msg = str(exc)
            logger.error("debug/test-whisper transcription error: %s", exc)

    return {
        "rms":               round(chunk.rms, 6),
        "threshold":         _SILENCE_THRESHOLD,
        "passed_silence_gate": passed_gate,
        "whisper_result":    whisper_out,
        "filtered_by_whisper": passed_gate and whisper_out is None and error_msg is None,
        "error":             error_msg,
        "diagnosis": (
            f"Silence gate blocked chunk (rms {chunk.rms:.6f} < threshold {_SILENCE_THRESHOLD})"
            if not passed_gate
            else f"Whisper error: {error_msg}"
            if error_msg
            else "Whisper filtered chunk as no-speech or empty (expected for noise — real speech will pass)"
            if whisper_out is None
            else f"Pipeline OK — Whisper produced: {whisper_out['text'][:80]!r}"
        ),
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Real-time transcript stream.

    Connect to receive transcript chunks as JSON objects broadcast by the
    processing loop.  The server sends {"type":"ping"} every 15 s; the
    client replies {"type":"pong"}.

    The connection stays open until the client disconnects or the server shuts
    down. No raw audio passes through this endpoint.
    """
    await _manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            text = message.strip()
            # Text ping (legacy)
            if text.lower() == "ping":
                await websocket.send_text("pong")
                continue
            # JSON messages: ping (legacy client-initiated) and pong (keepalive reply)
            try:
                msg = json.loads(text)
                if msg.get("type") == "pong":
                    continue  # client acknowledging server ping — no action needed
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except (json.JSONDecodeError, AttributeError):
                pass
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
        await _manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket error (will disconnect): %s", exc)
        await _manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Project Brief — S8-001  CRUD endpoints
# ---------------------------------------------------------------------------

def _save_brief(brief: dict) -> Path:
    """Write a project brief to data/project_briefs/{id}.json."""
    _BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    path = _BRIEFS_DIR / f"{brief['project_id']}.json"
    path.write_text(json.dumps(brief, indent=2, default=str), encoding="utf-8")
    return path


def _load_brief(project_id: str) -> Optional[dict]:
    """Read a project brief from disk, or None if not found."""
    path = _BRIEFS_DIR / f"{project_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_briefs() -> list[dict]:
    """List all saved briefs (id, name, created_at, project_type), newest first."""
    if not _BRIEFS_DIR.exists():
        return []
    briefs = []
    for p in _BRIEFS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            briefs.append({
                "project_id":   data.get("project_id"),
                "project_name": data.get("project_name"),
                "created_at":   data.get("created_at"),
                "project_type": data.get("project_type"),
            })
        except Exception:
            continue
    briefs.sort(key=lambda b: b.get("created_at") or "", reverse=True)
    return briefs


def _build_project_context(brief: dict) -> str:
    """Build the system prompt block from an active project brief."""
    methodology = ", ".join(brief.get("methodology") or [])
    scope = "\n".join(f"  - {s}" for s in (brief.get("scope_items") or []))
    out_scope = "\n".join(f"  - {s}" for s in (brief.get("out_of_scope") or []))
    risks = "\n".join(f"  - {r}" for r in (brief.get("risks") or []))
    milestones = "\n".join(
        f"  - {m.get('name', m) if isinstance(m, dict) else m}"
        + (f" ({m['date']})" if isinstance(m, dict) and m.get('date') else "")
        for m in (brief.get("milestones") or [])
    )
    team = "\n".join(
        f"  - {t.get('name', t) if isinstance(t, dict) else t}"
        + (f" ({t['role']})" if isinstance(t, dict) and t.get('role') else "")
        for t in (brief.get("team") or [])
    )
    budget_parts = []
    if brief.get("budget_total") is not None:
        budget_parts.append(f"Total: {brief['budget_currency']} {brief['budget_total']:,.0f}")
    if brief.get("budget_remaining") is not None:
        budget_parts.append(f"Remaining: {brief['budget_currency']} {brief['budget_remaining']:,.0f}")
    budget = " | ".join(budget_parts) if budget_parts else "Not specified"

    lines = [
        "You are a PM intelligence assistant coaching a project manager in real time.",
        f"The active project context is:",
        f"Project: {brief.get('project_name', 'Unknown')}",
        f"Type: {brief.get('project_type', 'Unknown')}",
        f"Methodologies: {methodology or 'Not specified'}",
    ]
    if brief.get("cpmai_phase"):
        lines.append(f"CPMAI Phase: {brief['cpmai_phase']}")
    if brief.get("objectives"):
        lines.append(f"Objectives: {brief['objectives']}")
    if scope:
        lines.append(f"Approved Scope:\n{scope}")
    if out_scope:
        lines.append(f"Out of Scope:\n{out_scope}")
    lines.append(f"Budget: {budget}")
    if brief.get("start_date"):
        lines.append(f"Start Date: {brief['start_date']}")
    if brief.get("end_date"):
        lines.append(f"End Date: {brief['end_date']}")
    if milestones:
        lines.append(f"Milestones:\n{milestones}")
    if team:
        lines.append(f"Team:\n{team}")
    if risks:
        lines.append(f"Known Risks:\n{risks}")
    if brief.get("success_criteria"):
        lines.append(f"Success Criteria: {brief['success_criteria']}")
    lines.append("")
    lines.append("Flag immediately if anything discussed deviates from approved scope or exceeds budget.")
    return "\n".join(lines)


# CRITICAL: /project/brief/active MUST be declared BEFORE /project/brief/{project_id}

@app.post("/project/brief", summary="Create or update a project brief")
async def create_project_brief(body: _ProjectBriefRequest) -> dict:
    """Validate, assign uuid4 if missing, save to disk, set as active."""
    global _active_project
    data = body.model_dump()
    if not data.get("project_id"):
        data["project_id"] = str(_uuid_mod.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    data["created_at"] = data.get("created_at") or now
    data["updated_at"] = now
    # Serialize nested models to plain dicts
    data["milestones"] = [m if isinstance(m, dict) else m for m in data.get("milestones", [])]
    data["team"] = [t if isinstance(t, dict) else t for t in data.get("team", [])]
    _save_brief(data)
    _active_project = data
    logger.info("Project brief saved: %s (%s)", data["project_name"], data["project_id"])
    return {"project_id": data["project_id"], "status": "saved"}


_EXTRACT_SYSTEM = (
    "You are a project management assistant. Extract structured "
    "project information from the provided documents. Return ONLY "
    "valid JSON matching the schema provided. If a field cannot "
    "be found in the documents, set it to null. Do not invent "
    "or assume values — only extract what is explicitly stated."
)

_EXTRACT_SCHEMA = """\
Extract project brief information from these documents \
and return as JSON matching this exact schema:
{
  "project_name": "string or null",
  "project_type": "one of ['AI Project','Software','Infrastructure','Business Change','Other'] or null",
  "methodology": ["zero or more from: 'PMBOK','Scrum','CPMAI'"],
  "cpmai_phase": "one of ['Business Understanding','Data Understanding','Data Preparation','Modeling','Evaluation','Deployment'] or null",
  "objectives": "string or null",
  "scope_items": ["list of strings or empty"],
  "out_of_scope": ["list of strings or empty"],
  "budget_total": "number or null",
  "budget_remaining": "number or null",
  "budget_currency": "string or null",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "milestones": [{"name": "string", "date": "YYYY-MM-DD"}],
  "team": [{"name": "string", "role": "string"}],
  "risks": ["list of strings or empty"],
  "success_criteria": "string or null"
}

Documents:
"""

_ALLOWED_EXTENSIONS = {".md", ".txt", ".docx", ".pdf", ".csv"}

# All fields in the extraction schema for missing-field detection
_BRIEF_FIELDS = [
    "project_name", "project_type", "methodology", "cpmai_phase",
    "objectives", "scope_items", "out_of_scope", "budget_total",
    "budget_remaining", "budget_currency", "start_date", "end_date",
    "milestones", "team", "risks", "success_criteria",
]


def _read_docx(content: bytes) -> str:
    """Extract plain text from a .docx file."""
    import io
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_pdf(content: bytes) -> str:
    """Extract plain text from a .pdf file."""
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@app.post("/project/extract", summary="Extract project brief from uploaded documents")
async def extract_project_brief(
    files: list[UploadFile] = File(...),
) -> dict:
    """Upload .md/.txt/.docx/.pdf/.csv files → Claude extracts a structured brief."""
    if not files:
        return JSONResponse({"error": "No files provided."}, status_code=400)

    # Read and validate files
    parts: list[str] = []
    filenames: list[str] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            return JSONResponse(
                {"error": f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"},
                status_code=400,
            )
        raw = await f.read()
        if ext in (".md", ".txt"):
            text = raw.decode("utf-8", errors="replace")
        elif ext == ".docx":
            try:
                text = await asyncio.to_thread(_read_docx, raw)
            except Exception as exc:
                return JSONResponse({"error": f"Failed to parse {f.filename}: {exc}"}, status_code=400)
        elif ext == ".pdf":
            try:
                text = await asyncio.to_thread(_read_pdf, raw)
            except Exception as exc:
                return JSONResponse({"error": f"Failed to parse {f.filename}: {exc}"}, status_code=400)
        elif ext == ".csv":
            try:
                import csv as _csv_mod
                import io as _io_mod
                decoded = raw.decode("utf-8", errors="replace")
                reader = _csv_mod.reader(_io_mod.StringIO(decoded))
                rows = list(reader)
                if rows:
                    headers = rows[0]
                    lines = []
                    for row in rows[1:]:
                        line = ", ".join(
                            f"{headers[i]}: {row[i]}" if i < len(headers) else row[i]
                            for i in range(len(row))
                        )
                        lines.append(line)
                    text = "\n".join(lines) if lines else ", ".join(headers)
                else:
                    text = ""
            except Exception as exc:
                return JSONResponse({"error": f"Failed to parse {f.filename}: {exc}"}, status_code=400)
        else:
            text = ""

        if text.strip():
            parts.append(f"=== {f.filename} ===\n{text}")
            filenames.append(f.filename or "unknown")

    if not parts:
        return JSONResponse({"error": "No readable content found in uploaded files."}, status_code=400)

    combined_text = "\n\n".join(parts)

    # Call Claude API for extraction
    from meetingmind._api_key import load_api_key as _load_api_key
    try:
        api_key = _load_api_key()
    except EnvironmentError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    try:
        import anthropic
    except ImportError as exc:
        return JSONResponse({"error": f"anthropic package not available: {exc}"}, status_code=500)

    user_message = _EXTRACT_SCHEMA + combined_text

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_EXTRACT_SYSTEM,
                messages=[{"role": "user", "content": user_message}],
            )
        )
    except Exception as exc:
        logger.error("Claude extraction failed: %s", exc)
        return JSONResponse({"error": f"Claude API call failed: {exc}"}, status_code=500)

    raw_text = response.content[0].text.strip()

    # Strip fenced code block if present
    import re
    fenced = re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw_text)
    if fenced:
        raw_text = fenced.group(1).strip()

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("Claude extraction returned invalid JSON: %s", raw_text[:500])
        return JSONResponse(
            {"error": "Claude returned invalid JSON", "raw_response": raw_text[:2000]},
            status_code=500,
        )

    # Compute missing fields
    missing: list[str] = []
    for field in _BRIEF_FIELDS:
        val = extracted.get(field)
        if val is None or val == "" or val == []:
            missing.append(field)

    found_count = len(_BRIEF_FIELDS) - len(missing)

    logger.info(
        "Project brief extracted from %d file(s): %d/%d fields found.",
        len(filenames), found_count, len(_BRIEF_FIELDS),
    )

    return {
        "extracted": extracted,
        "missing_fields": missing,
        "found_count": found_count,
        "missing_count": len(missing),
        "files_processed": filenames,
    }


@app.get("/project/brief/active", summary="Get the currently active project brief")
async def get_active_project_brief() -> dict:
    """Return the active brief, or {active: false}."""
    if _active_project is None:
        return {"active": False, "brief": None}
    return {"active": True, "brief": _active_project}


@app.get("/project/brief/{project_id}", summary="Get a project brief by ID")
async def get_project_brief(project_id: str) -> dict:
    """Load a specific brief from disk."""
    brief = _load_brief(project_id)
    if brief is None:
        raise HTTPException(status_code=404, detail="Project brief not found")
    return brief


@app.get("/project/briefs", summary="List all saved project briefs")
async def list_project_briefs() -> dict:
    """Return summary list of all saved briefs."""
    return {"briefs": _list_briefs()}


@app.delete("/project/brief/{project_id}", summary="Delete a project brief")
async def delete_project_brief(project_id: str) -> dict:
    """Delete a brief file and clear active if it was the active one."""
    global _active_project
    path = _BRIEFS_DIR / f"{project_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project brief not found")
    path.unlink()
    if _active_project and _active_project.get("project_id") == project_id:
        _active_project = None
    logger.info("Project brief deleted: %s", project_id)
    return {"status": "deleted", "project_id": project_id}


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    uvicorn.run(
        "meetingmind.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        # Disable uvicorn's protocol-level WS pong timeout.
        # Whisper's CPU-bound work holds the GIL, which can delay pong
        # processing past uvicorn's default 20 s timeout → disconnects.
        # Our own _ws_ping_loop handles application-level keepalive.
        ws_ping_interval=20.0,
        ws_ping_timeout=None,
    )
