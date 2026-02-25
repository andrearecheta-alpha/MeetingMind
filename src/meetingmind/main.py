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
from pathlib import Path
from typing import Optional

import uvicorn
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

# Meeting lifecycle state — mutated by /meeting/start and /meeting/stop.
_meeting: dict = {
    "active":      False,
    "capture":     None,   # AudioCapture instance
    "transcriber": None,   # StreamingTranscriber instance
    "task":        None,   # asyncio.Task for _processing_loop
    "start_time":  None,   # Unix timestamp
    "model":       None,   # Whisper model size in use
}

# Pipeline diagnostic counters — reset on each /meeting/start.
_stats: dict = {
    "chunks_from_queue":  0,   # AudioChunks pulled from PyAudio queue
    "chunks_filtered":    0,   # Returned None by transcribe_chunk (silence/noise)
    "chunks_transcribed": 0,   # TranscriptChunks produced by Whisper
    "chunks_broadcast":   0,   # Successfully broadcast over WebSocket
    "last_rms":           0.0, # RMS of the most recent chunk (before filtering)
    "recent_rms":         [],  # RMS of last 5 chunks — use to tune silence threshold
}


# ---------------------------------------------------------------------------
# Background processing loop
# ---------------------------------------------------------------------------

async def _processing_loop() -> None:
    """
    Background asyncio task: pulls AudioChunks from the capture queue,
    runs Whisper transcription in a thread pool (CPU-bound, non-blocking),
    and broadcasts results to all connected WebSocket clients.

    Runs until _meeting["active"] is set to False or the task is cancelled.
    """
    logger.info("Processing loop started.")
    try:
        while _meeting["active"]:
            # ── Pull one AudioChunk from the capture queue (blocks ≤ 1 s) ────
            chunk = await asyncio.to_thread(
                _meeting["capture"].get_chunk, 1.0
            )
            if chunk is None:
                continue  # Timeout — loop again to check _meeting["active"]

            # ── Per-chunk stats & logging ─────────────────────────────────────
            _stats["chunks_from_queue"] += 1
            _stats["last_rms"] = round(chunk.rms, 5)
            _stats["recent_rms"] = (_stats["recent_rms"] + [_stats["last_rms"]])[-5:]
            logger.info(
                "CHUNK  rms=%.5f  source=%s",
                chunk.rms, chunk.source.value,
            )

            # ── Whisper inference (CPU-bound — run in thread pool) ────────────
            try:
                result = await asyncio.to_thread(
                    _meeting["transcriber"].transcribe_chunk, chunk
                )
            except Exception as exc:
                logger.error("transcribe_chunk error (chunk skipped): %s", exc)
                _stats["chunks_filtered"] += 1
                continue

            if result is None:
                _stats["chunks_filtered"] += 1
                continue  # Silence or empty chunk — skip broadcast

            # ── Broadcast TranscriptChunk to all connected WS clients ─────────
            _stats["chunks_transcribed"] += 1
            await _manager.broadcast(result.to_dict())
            _stats["chunks_broadcast"] += 1

    except asyncio.CancelledError:
        logger.info("Processing loop cancelled.")
    except Exception as exc:
        logger.exception("Processing loop fatal error: %s", exc)
    finally:
        logger.info("Processing loop stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MeetingMind — PM Edition",
    description=(
        "Real-time meeting assistant for Project Managers. "
        "Audio processing is on-device; only transcript text is sent to Claude "
        "when PM suggestions or analysis are requested."
    ),
    version="0.2.0",
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

    return {
        "status":            "ok",
        "meeting_active":    _meeting["active"],
        "model":             _meeting["model"],
        "connected_clients": _manager.count,
        "elapsed_seconds":   elapsed,
    }


# ── Meeting lifecycle ─────────────────────────────────────────────────────────

@app.get("/devices", summary="List available audio input devices")
async def list_devices() -> dict:
    """
    Returns all audio input devices visible to PyAudio.

    Use the 'index' value from this response as the mic_device_index parameter
    when calling POST /meeting/start to select a specific microphone.

    Tip: if you see a virtual device (e.g. 'Camo', 'VB-Cable') capturing silence,
    use this endpoint to find your physical microphone's index.
    """
    from meetingmind.audio_capture import AudioCapture
    try:
        cap = AudioCapture.__new__(AudioCapture)
        import pyaudio
        cap._pa = pyaudio.PyAudio()
        devices = cap.list_devices()
        cap._pa.terminate()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    return {
        "devices": devices,
        "hint": "Pass the 'index' value as mic_device_index to POST /meeting/start",
    }


@app.post("/meeting/start", summary="Start audio capture and transcription")
async def meeting_start(
    model_size:      str            = "medium",
    chunk_duration:  float          = 3.0,
    privacy_mode:    bool           = True,
    language:        Optional[str]  = None,
    mic_device_index: Optional[int] = None,
) -> dict:
    """
    Initialises AudioCapture and StreamingTranscriber, then starts the
    background processing loop.

    Parameters
    ----------
    model_size:       Whisper model — tiny | base | small | medium | large.
                      'medium' balances accuracy and speed for meetings.
    chunk_duration:   Seconds of audio per transcript chunk (default 3).
    privacy_mode:     Print a local-processing confirmation banner (default True).
    language:         ISO-639-1 hint (e.g. 'en'). None = auto-detect per chunk.
    mic_device_index: PyAudio device index for the microphone (from GET /devices).
                      None = OS default. Use this to pick a specific mic if the
                      default is a virtual device (Camo, VB-Cable, etc.) that
                      outputs silence.
    """
    if _meeting["active"]:
        return JSONResponse(
            {"error": "A meeting is already active. POST /meeting/stop first."},
            status_code=400,
        )

    # Import here to keep startup fast — model load happens below.
    from meetingmind.audio_capture import AudioCapture
    from meetingmind.transcriber import StreamingTranscriber

    try:
        capture = AudioCapture(
            mic_device_index=mic_device_index,
            chunk_duration=chunk_duration,
            privacy_mode=privacy_mode,
        )
        # StreamingTranscriber loads the Whisper model (CPU-bound, up to 30 s
        # on first run). Run it in a thread so the asyncio event loop stays
        # responsive — WebSocket connections and health checks still work
        # while the model is loading.
        logger.info("Loading Whisper '%s' model in thread pool…", model_size)
        transcriber = await asyncio.to_thread(
            lambda: StreamingTranscriber(
                model_size=model_size,
                privacy_mode=privacy_mode,
                language=language,
            )
        )
    except Exception as exc:
        logger.exception("Failed to initialise audio engine: %s", exc)
        return JSONResponse(
            {"error": f"Initialisation failed: {exc}"},
            status_code=500,
        )

    capture.start()

    # Reset diagnostic counters for the new session (explicit reset preserves types).
    _stats["chunks_from_queue"]  = 0
    _stats["chunks_filtered"]    = 0
    _stats["chunks_transcribed"] = 0
    _stats["chunks_broadcast"]   = 0
    _stats["last_rms"]           = 0.0
    _stats["recent_rms"]         = []

    _meeting["active"]     = True
    _meeting["capture"]    = capture
    _meeting["transcriber"]= transcriber
    _meeting["start_time"] = time.time()
    _meeting["model"]      = model_size
    _meeting["task"]       = asyncio.create_task(_processing_loop())

    logger.info("Meeting started — model: %s, chunk: %.1fs", model_size, chunk_duration)
    return {
        "status":         "started",
        "model":          model_size,
        "chunk_duration": chunk_duration,
        "language":       language or "auto-detect",
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

    _meeting["active"] = False

    # Cancel background processing task
    if _meeting["task"]:
        _meeting["task"].cancel()
        try:
            await _meeting["task"]
        except asyncio.CancelledError:
            pass

    # Stop audio capture and release resources
    if _meeting["capture"]:
        await asyncio.to_thread(_meeting["capture"].close)

    elapsed = round(time.time() - (_meeting["start_time"] or time.time()), 1)

    _meeting["capture"]     = None
    _meeting["transcriber"] = None
    _meeting["task"]        = None
    _meeting["start_time"]  = None
    _meeting["model"]       = None

    logger.info("Meeting stopped after %.1f seconds.", elapsed)
    return {"status": "stopped", "elapsed_seconds": elapsed}


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

    if _meeting["task"]:
        _meeting["task"].cancel()
        try:
            await _meeting["task"]
        except asyncio.CancelledError:
            pass

    if _meeting["capture"]:
        try:
            await asyncio.to_thread(_meeting["capture"].close)
        except Exception:
            pass

    _meeting["capture"]     = None
    _meeting["transcriber"] = None
    _meeting["task"]        = None
    _meeting["start_time"]  = None
    _meeting["model"]       = None

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
