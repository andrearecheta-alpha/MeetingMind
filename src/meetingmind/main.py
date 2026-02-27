"""
main.py
-------
FastAPI application — the real-time audio engine HTTP/WebSocket layer.

Endpoints
---------
GET  /health              Server liveness + meeting state
POST /meeting/start       Begin audio capture and transcription
POST /meeting/stop        End audio capture and transcription
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
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
_WHISPER_MODEL_DEFAULT:  str   = "base"   # tiny|base|small|medium|large
_CHUNK_SECONDS_DEFAULT:  float = 2.0      # seconds of audio per Whisper call
_LANGUAGE_DEFAULT:       str   = "en"     # skips per-chunk language detection
_MAX_CONCURRENT_WHISPER: int   = 1        # sequential — 1 Whisper task at a time


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

    async def broadcast(self, data: dict) -> None:
        """Send a JSON payload to every connected client."""
        if not self._connections:
            return
        message  = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# Global application state
# ---------------------------------------------------------------------------

_manager = ConnectionManager()

# Device detection result — populated once at startup.
from meetingmind.device_detector import DetectionResult, detect_devices
_detected: Optional[DetectionResult] = None

# Knowledge base + context engine — initialised in lifespan, None if unavailable.
from meetingmind.knowledge_base import KnowledgeBase
from meetingmind.context_engine import ContextEngine
_kb:  Optional[KnowledgeBase]  = None
_ctx: Optional[ContextEngine]  = None

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
}

# Whisper model preloaded in the background at server startup.
# If the user starts a meeting with the same model size, no load delay.
_preloaded_whisper: Optional[object] = None

_kb_lock = asyncio.Lock()  # prevents double-init if two requests arrive simultaneously


async def _preload_whisper(model_size: str = _WHISPER_MODEL_DEFAULT) -> None:
    """Load Whisper in the background at server start so the first meeting has no delay."""
    global _preloaded_whisper
    try:
        from meetingmind.transcriber import StreamingTranscriber
        _preloaded_whisper = await asyncio.to_thread(
            lambda: StreamingTranscriber(
                model_size=model_size,
                privacy_mode=False,
                language=_LANGUAGE_DEFAULT,
            )
        )
        logger.info("Whisper '%s' preloaded at startup — first meeting starts instantly.", model_size)
    except Exception as exc:
        logger.warning("Whisper preload failed (will load on first meeting start): %s", exc)
        _preloaded_whisper = None


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
        except Exception as exc:
            logger.error("KnowledgeBase init failed (history disabled): %s", exc)
            _kb = _ctx = None


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
}


# ---------------------------------------------------------------------------
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

    while _meeting["active"]:
        try:
            while _meeting["active"]:
                chunk = await asyncio.to_thread(
                    _meeting["capture"].get_chunk, 1.0
                )
                if chunk is None:
                    continue

                _stats["chunks_from_queue"] += 1
                logger.info(
                    "CHUNK  rms=%.5f  peak=%.5f  source=%s",
                    chunk.rms, chunk.peak_rms, chunk.source.value,
                )

                try:
                    _t0     = time.time()
                    result  = await asyncio.wait_for(
                        asyncio.to_thread(
                            _meeting["transcriber"].transcribe_chunk, chunk
                        ),
                        timeout=5.0,
                    )
                    elapsed = time.time() - _t0
                    times   = _stats.setdefault("whisper_times", [])
                    times.append(round(elapsed, 3))
                    _stats["whisper_times"]      = times[-10:]
                    _stats["avg_whisper_time_s"] = round(
                        sum(_stats["whisper_times"]) / len(_stats["whisper_times"]), 3
                    )
                except asyncio.TimeoutError:
                    logger.warning("WHISPER TIMEOUT >5 s — chunk skipped (%s)", chunk.source.value)
                    _stats["chunks_filtered"] += 1
                    continue
                except Exception as exc:
                    logger.error("transcribe_chunk error: %s", exc)
                    _stats["chunks_filtered"] += 1
                    continue

                if result is None:
                    _stats["chunks_filtered"] += 1
                    continue

                _stats["chunks_transcribed"] += 1
                if _stats.get("first_transcript_delay_s") is None:
                    _stats["first_transcript_delay_s"] = round(
                        time.time() - _meeting["start_time"], 2
                    )
                await _manager.broadcast(result.to_dict())
                _stats["chunks_broadcast"] += 1
                _meeting["transcript_chunks"].append(result.text)

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
    """
    logger.info("RMS broadcast loop started.")
    meeting_start = time.time()
    first_sent    = False

    while _meeting.get("active"):
        await asyncio.sleep(0.5)
        if not _meeting.get("active"):
            break
        capture = _meeting.get("capture")
        if capture is None:
            continue

        rms = capture.get_latest_rms()  # drains mic-only RMS queue, returns freshest value
        _stats["last_rms"] = round(rms, 5)
        recent = list(_stats.get("recent_rms", []))
        recent.append(_stats["last_rms"])
        _stats["recent_rms"] = recent[-5:]

        if not first_sent:
            _stats["first_rms_delay_s"] = round(time.time() - meeting_start, 2)
            first_sent = True

        await _manager.broadcast({
            "type":      "rms",
            "rms":       _stats["last_rms"],
            "threshold": _SILENCE_THRESHOLD,
            "captured":  _stats["chunks_from_queue"],
            "passed":    _stats["chunks_transcribed"],
        })

    logger.info("RMS broadcast loop stopped.")


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
    global _preloaded_whisper

    # Fast path: reuse the model preloaded at startup if it matches.
    if (
        _preloaded_whisper is not None
        and getattr(_preloaded_whisper, "_model_size", None) == model_size
    ):
        transcriber        = _preloaded_whisper
        _preloaded_whisper = None   # hand ownership to this meeting
        logger.info("Using preloaded Whisper '%s' — no load delay.", model_size)
    else:
        logger.info("Loading Whisper '%s' model in background…", model_size)
        try:
            from meetingmind.transcriber import StreamingTranscriber
            transcriber = await asyncio.to_thread(
                lambda: StreamingTranscriber(
                    model_size=model_size,
                    privacy_mode=privacy_mode,
                    language=language,
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
    _meeting["task"]          = asyncio.create_task(_transcription_loop())

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
    global _detected, _kb, _ctx
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

    # Preload the default Whisper model in the background so the first meeting
    # start has no loading delay.  Fire-and-forget — failure is non-fatal.
    asyncio.create_task(_preload_whisper(_WHISPER_MODEL_DEFAULT))

    yield   # server runs here (uvicorn has bound port 8000 by this point)

    if _kb:
        await asyncio.to_thread(_kb.close)


app = FastAPI(
    title="MeetingMind — PM Edition",
    description=(
        "Real-time meeting assistant for Project Managers. "
        "Audio processing is on-device; only transcript text is sent to Claude "
        "when PM suggestions or analysis are requested."
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

# Serve everything inside frontend/ under /static (JS, CSS, images added later)
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ── Root — serve the PM dashboard ────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the PM Edition web dashboard."""
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve favicon.ico if present; return 204 so browsers stop 404-logging."""
    ico = _FRONTEND / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(status_code=204)


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

    # Reset diagnostic counters.
    _stats["chunks_from_queue"]        = 0
    _stats["chunks_filtered"]          = 0
    _stats["chunks_transcribed"]       = 0
    _stats["chunks_broadcast"]         = 0
    _stats["last_rms"]                 = 0.0
    _stats["recent_rms"]               = []
    _stats["whisper_times"]            = []
    _stats["avg_whisper_time_s"]       = 0.0
    _stats["first_rms_delay_s"]        = None
    _stats["first_transcript_delay_s"] = None

    _meeting["active"]            = True
    _meeting["capture"]           = capture
    _meeting["transcriber"]       = None        # not ready yet
    _meeting["task"]              = None        # started after model loads
    _meeting["load_task"]         = None
    _meeting["start_time"]        = time.time()
    _meeting["model"]             = model_size
    _meeting["model_loading"]     = True
    _meeting["meeting_id"]        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _meeting["transcript_chunks"] = []
    _meeting["rms_task"]          = asyncio.create_task(_rms_broadcast_loop())

    # ── Phase 2: Load Whisper in background ─────────────────────────────────
    # Returns to the browser immediately; transcription starts once model is ready.
    # Default language to the config constant so the preloaded model language matches.
    _meeting["load_task"] = asyncio.create_task(
        _whisper_load_background(model_size, privacy_mode, language or _LANGUAGE_DEFAULT)
    )

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

    # Capture ingestion data BEFORE clearing _meeting
    ingestion_meeting_id  = _meeting.get("meeting_id")
    ingestion_transcript  = " ".join(_meeting.get("transcript_chunks", []))
    ingestion_model_size  = _meeting.get("model") or "base"
    ingestion_started_at  = datetime.fromtimestamp(
        _meeting["start_time"] or time.time(), tz=timezone.utc
    )
    ingestion_stopped_at  = datetime.now(timezone.utc)
    ingestion_duration    = round(
        (ingestion_stopped_at - ingestion_started_at).total_seconds(), 1
    )

    _meeting["active"] = False

    # Cancel all background tasks (load_task first — it may spawn the transcription task)
    for key in ("load_task", "task", "rms_task"):
        t = _meeting.get(key)
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    # Stop audio capture and release resources
    if _meeting["capture"]:
        await asyncio.to_thread(_meeting["capture"].close)

    elapsed = round(time.time() - (_meeting["start_time"] or time.time()), 1)

    _meeting["capture"]           = None
    _meeting["transcriber"]       = None
    _meeting["task"]              = None
    _meeting["rms_task"]          = None
    _meeting["load_task"]         = None
    _meeting["model_loading"]     = False
    _meeting["start_time"]        = None
    _meeting["model"]             = None
    _meeting["meeting_id"]        = None
    _meeting["transcript_chunks"] = []

    # Launch background ingestion (non-blocking — returns in <1 s).
    # _ensure_kb() lazily initialises KB on first stop.
    if ingestion_meeting_id and ingestion_transcript.strip():
        asyncio.create_task(_ingest_meeting_background(
            ingestion_meeting_id,
            ingestion_started_at,
            ingestion_stopped_at,
            ingestion_duration,
            ingestion_model_size,
            ingestion_transcript,
        ))

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

    for key in ("load_task", "task", "rms_task"):
        t = _meeting.get(key)
        if t:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    if _meeting["capture"]:
        try:
            await asyncio.to_thread(_meeting["capture"].close)
        except Exception:
            pass

    _meeting["capture"]           = None
    _meeting["transcriber"]       = None
    _meeting["task"]              = None
    _meeting["rms_task"]          = None
    _meeting["load_task"]         = None
    _meeting["model_loading"]     = False
    _meeting["start_time"]        = None
    _meeting["model"]             = None
    _meeting["meeting_id"]        = None
    _meeting["transcript_chunks"] = []

    logger.info("Meeting state force-reset (was_active=%s).", was_active)
    return {"status": "reset", "was_active": was_active}


# ── PM Suggestions endpoint ───────────────────────────────────────────────────

class _SuggestionsRequest(BaseModel):
    transcript: str
    context: Optional[str] = None
    model: str = "claude-sonnet-4-6"


@app.post("/suggestions", summary="Generate 3 PM response suggestions")
async def suggestions(body: _SuggestionsRequest) -> dict:
    """
    Sends the transcript snippet to Claude with a Project Manager system prompt
    and returns 3 ready-to-use response suggestions.

    The transcript text is the only data sent externally (to the Anthropic API
    over HTTPS). No audio is involved.
    """
    if not body.transcript.strip():
        return JSONResponse({"error": "transcript must not be empty."}, status_code=422)

    try:
        from meetingmind.suggestion_engine import get_suggestions  # lazy import
    except ImportError as exc:
        return JSONResponse(
            {"error": f"suggestion_engine not available: {exc}"},
            status_code=500,
        )

    try:
        items = await asyncio.to_thread(
            get_suggestions, body.transcript, body.model, body.context
        )
    except (EnvironmentError, RuntimeError) as exc:
        logger.error("Suggestions error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return {"suggestions": items}


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
            result = await asyncio.to_thread(
                _meeting["transcriber"].transcribe_chunk, chunk
            )
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
    processing loop. Send "ping" to receive "pong" for keep-alive checks.

    The connection stays open until the client disconnects or the server shuts
    down. No raw audio passes through this endpoint.
    """
    await _manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; client may send "ping" at any time.
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await _manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)
        await _manager.disconnect(websocket)


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
    )
