"""
main.py
-------
FastAPI application — the real-time audio engine HTTP/WebSocket layer.

Endpoints
---------
GET  /health              Server liveness + meeting state
POST /meeting/start       Begin audio capture and transcription
POST /meeting/stop        End audio capture and transcription
GET  /meeting/reset       Force-reset stuck meeting state
GET  /devices             List available audio input devices
GET  /debug               Pipeline diagnostic counters
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure src/ is importable.
_SRC          = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND     = _PROJECT_ROOT / "frontend"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Logging
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
_SILENCE_THRESHOLD: float = 0.00005

# ---------------------------------------------------------------------------
# Transcription config
# ---------------------------------------------------------------------------
_WHISPER_MODEL_DEFAULT:  str   = "base"   # tiny|base|small|medium|large
_CHUNK_SECONDS_DEFAULT:  float = 3.0      # seconds of audio per Whisper call
_LANGUAGE_DEFAULT:       str   = "en"     # skips per-chunk language detection
_MAX_CONCURRENT_WHISPER: int   = 1        # sequential — 1 Whisper task at a time


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("WebSocket client connected. Active: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("WebSocket client disconnected. Active: %d", len(self._connections))

    async def broadcast(self, data: dict) -> None:
        if not self._connections:
            return
        message = json.dumps(data)
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

_meeting: dict = {
    "active":             False,
    "capture":            None,
    "transcriber":        None,
    "task":               None,
    "rms_task":           None,
    "load_task":          None,
    "start_time":         None,
    "model":              None,
    "model_loading":      False,
    "meeting_id":         None,
    "transcript_chunks":  [],
    "speaker_counts":     {"microphone": 0, "system": 0},
}

_preloaded_whisper: Optional[object] = None


async def _preload_whisper(model_size: str = _WHISPER_MODEL_DEFAULT) -> None:
    """Load Whisper in the background at server start."""
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
        logger.info("Whisper '%s' preloaded at startup.", model_size)
    except Exception as exc:
        logger.warning("Whisper preload failed (will load on first meeting start): %s", exc)
        _preloaded_whisper = None


# Pipeline diagnostic counters — reset on each /meeting/start.
_stats: dict = {
    "chunks_from_queue":        0,
    "chunks_filtered":          0,
    "chunks_transcribed":       0,
    "chunks_broadcast":         0,
    "last_rms":                 0.0,
    "recent_rms":               [],
    "whisper_times":            [],
    "avg_whisper_time_s":       0.0,
    "first_rms_delay_s":        None,
    "first_transcript_delay_s": None,
}


# ---------------------------------------------------------------------------
# Background processing loops
# ---------------------------------------------------------------------------

async def _transcription_loop() -> None:
    """Pulls AudioChunks from the capture queue, runs Whisper, broadcasts results."""
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
                    _t0    = time.time()
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            _meeting["transcriber"].transcribe_chunk, chunk
                        ),
                        timeout=3.0,
                    )
                    elapsed = time.time() - _t0
                    times   = _stats.setdefault("whisper_times", [])
                    times.append(round(elapsed, 3))
                    _stats["whisper_times"]      = times[-10:]
                    _stats["avg_whisper_time_s"] = round(
                        sum(_stats["whisper_times"]) / len(_stats["whisper_times"]), 3
                    )
                except asyncio.TimeoutError:
                    logger.warning("SKIP: whisper timeout >3 s — chunk skipped")
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

                src = result.source
                _meeting["speaker_counts"][src] = _meeting["speaker_counts"].get(src, 0) + 1

            break

        except asyncio.CancelledError:
            logger.info("Transcription loop cancelled.")
            raise

        except Exception as exc:
            _restart_count += 1
            logger.exception(
                "Transcription loop error (restart %d/%d): %s",
                _restart_count, _max_restarts, exc,
            )
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
    """Broadcasts latest per-frame RMS to all WebSocket clients every 0.5 s."""
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

        rms = capture.get_latest_rms()
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


async def _whisper_load_background(
    model_size:   str,
    privacy_mode: bool,
    language:     Optional[str],
) -> None:
    """Load the Whisper model in a thread pool, then start the transcription loop."""
    global _preloaded_whisper

    if (
        _preloaded_whisper is not None
        and getattr(_preloaded_whisper, "_model_size", None) == model_size
    ):
        transcriber        = _preloaded_whisper
        _preloaded_whisper = None
        logger.info("Using preloaded Whisper '%s'.", model_size)
    else:
        logger.info("Loading Whisper '%s' in background…", model_size)
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

    if not _meeting["active"]:
        logger.info("Whisper loaded but meeting already stopped — discarding.")
        return

    _meeting["transcriber"]   = transcriber
    _meeting["model_loading"] = False
    _meeting["task"]          = asyncio.create_task(_transcription_loop())

    logger.info("Whisper '%s' ready — transcription loop started.", model_size)
    await _manager.broadcast({"type": "model_ready", "model": model_size})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    asyncio.create_task(_preload_whisper(_WHISPER_MODEL_DEFAULT))
    yield


app = FastAPI(
    title="MeetingMind",
    description="Real-time meeting transcription. Audio processing is on-device.",
    version="0.3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    ico = _FRONTEND / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(status_code=204)


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health", summary="Server liveness check")
async def health() -> dict:
    elapsed: Optional[float] = None
    if _meeting["active"] and _meeting["start_time"]:
        elapsed = round(time.time() - _meeting["start_time"], 1)

    return {
        "status":            "ok",
        "meeting_active":    _meeting["active"],
        "model":             _meeting["model"],
        "model_loading":     _meeting.get("model_loading", False),
        "connected_clients": _manager.count,
        "elapsed_seconds":   elapsed,
    }


# ── Devices ───────────────────────────────────────────────────────────────

@app.get("/devices", summary="List available audio input devices")
async def list_devices() -> dict:
    """Returns all PyAudio input devices available on this machine."""
    try:
        import pyaudio
        pa      = pyaudio.PyAudio()
        devices = []
        default_index: Optional[int] = None

        try:
            default_index = pa.get_default_input_device_info()["index"]
        except Exception:
            pass

        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({
                        "index":       i,
                        "name":        info["name"],
                        "channels":    info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                        "is_default":  i == default_index,
                    })
            except Exception:
                continue

        pa.terminate()
        return {"devices": devices, "default_index": default_index}

    except Exception as exc:
        logger.warning("Device enumeration failed: %s", exc)
        return {"devices": [], "default_index": None, "error": str(exc)}


# ── Meeting lifecycle ─────────────────────────────────────────────────────

@app.post("/meeting/start", summary="Start audio capture and transcription")
async def meeting_start(
    model_size:          str           = _WHISPER_MODEL_DEFAULT,
    chunk_duration:      float         = _CHUNK_SECONDS_DEFAULT,
    privacy_mode:        bool          = True,
    language:            Optional[str] = None,
    mic_device_index:    Optional[int] = None,
    system_device_index: Optional[int] = None,
) -> dict:
    """
    Initialises AudioCapture and StreamingTranscriber, then starts the
    background processing loop.

    model_size: tiny|base|small|medium|large (default: base)
    """
    if _meeting["active"]:
        return JSONResponse(
            {"error": "A meeting is already active. POST /meeting/stop first."},
            status_code=400,
        )

    from meetingmind.audio_capture import AudioCapture

    try:
        capture = await asyncio.to_thread(
            lambda: AudioCapture(
                mic_device_index=mic_device_index,
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

    probe_ok, probe_name, probe_err = await asyncio.to_thread(
        capture.probe_device, mic_device_index
    )
    if not probe_ok:
        await asyncio.to_thread(capture._pa.terminate)
        err_lower  = probe_err.lower()
        is_privacy = any(x in err_lower for x in ["access denied", "-9999", "-9997"])
        suggestion = (
            "Check Windows Settings → Privacy & Security → Microphone → "
            "Allow apps to access your microphone"
            if is_privacy
            else "Try a different device index or check device connection"
        )
        return JSONResponse(
            {
                "error":      "No working microphone found",
                "tried":      [probe_name],
                "reason":     probe_err,
                "suggestion": suggestion,
            },
            status_code=500,
        )

    capture.start()

    _stats.update({
        "chunks_from_queue":        0,
        "chunks_filtered":          0,
        "chunks_transcribed":       0,
        "chunks_broadcast":         0,
        "last_rms":                 0.0,
        "recent_rms":               [],
        "whisper_times":            [],
        "avg_whisper_time_s":       0.0,
        "first_rms_delay_s":        None,
        "first_transcript_delay_s": None,
    })

    _meeting.update({
        "active":            True,
        "capture":           capture,
        "transcriber":       None,
        "task":              None,
        "load_task":         None,
        "start_time":        time.time(),
        "model":             model_size,
        "model_loading":     True,
        "meeting_id":        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "transcript_chunks": [],
        "speaker_counts":    {"microphone": 0, "system": 0},
        "rms_task":          asyncio.create_task(_rms_broadcast_loop()),
    })

    _meeting["load_task"] = asyncio.create_task(
        _whisper_load_background(model_size, privacy_mode, language or _LANGUAGE_DEFAULT)
    )

    logger.info("Meeting started — audio live, loading Whisper '%s' in background…", model_size)
    return {
        "status":              "starting",
        "model":               model_size,
        "model_loading":       True,
        "chunk_duration":      chunk_duration,
        "language":            language or "auto-detect",
        "mic_device_index":    mic_device_index,
        "system_device_index": system_device_index,
    }


@app.post("/meeting/stop", summary="Stop audio capture and transcription")
async def meeting_stop() -> dict:
    if not _meeting["active"]:
        return JSONResponse({"error": "No meeting is currently active."}, status_code=400)

    _meeting["active"] = False

    for key in ("load_task", "task", "rms_task"):
        t = _meeting.get(key)
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    if _meeting["capture"]:
        await asyncio.to_thread(_meeting["capture"].close)

    elapsed = round(time.time() - (_meeting["start_time"] or time.time()), 1)

    _meeting.update({
        "capture":           None,
        "transcriber":       None,
        "task":              None,
        "rms_task":          None,
        "load_task":         None,
        "model_loading":     False,
        "start_time":        None,
        "model":             None,
        "meeting_id":        None,
        "transcript_chunks": [],
        "speaker_counts":    {"microphone": 0, "system": 0},
    })

    logger.info("Meeting stopped after %.1f seconds.", elapsed)
    return {"status": "stopped", "elapsed_seconds": elapsed}


@app.get("/meeting/reset",  include_in_schema=False)
@app.post("/meeting/reset", summary="Force-reset stuck meeting state")
async def meeting_reset() -> dict:
    """Force-clears all meeting state. Use to recover from a stuck 'meeting already active' error."""
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

    _meeting.update({
        "capture":           None,
        "transcriber":       None,
        "task":              None,
        "rms_task":          None,
        "load_task":         None,
        "model_loading":     False,
        "start_time":        None,
        "model":             None,
        "meeting_id":        None,
        "transcript_chunks": [],
        "speaker_counts":    {"microphone": 0, "system": 0},
    })

    logger.info("Meeting state force-reset (was_active=%s).", was_active)
    return {"status": "reset", "was_active": was_active}


# ── Debug / diagnostics ───────────────────────────────────────────────────

@app.get("/debug", include_in_schema=False)
async def debug_pipeline() -> dict:
    """Real-time pipeline snapshot. Use to pinpoint where chunks stop flowing."""
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
        "meeting_active":       _meeting["active"],
        "elapsed_seconds":      elapsed,
        "model":                _meeting["model"],
        "connected_ws_clients": _manager.count,
        "audio_queue_size":     queue_size,
        "audio_threads":        dict(zip(thread_names, threads_alive)),
        "pipeline_stats":       dict(_stats),
        "diagnosis": (
            "PyAudio not capturing — check mic permissions"
            if _meeting["active"] and _stats["chunks_from_queue"] == 0 and (elapsed or 0) > 5
            else "All chunks silenced — mic muted or RMS threshold too high"
            if _stats["chunks_from_queue"] > 0 and _stats["chunks_transcribed"] == 0
            else "No WS clients connected"
            if _stats["chunks_transcribed"] > 0 and _stats["chunks_broadcast"] == 0
            else "Pipeline OK" if _stats["chunks_broadcast"] > 0
            else "Waiting for audio…"
        ),
    }


@app.post("/debug/inject", include_in_schema=False)
async def debug_inject() -> dict:
    """Inject a fake transcript chunk to test the WebSocket pipeline end-to-end."""
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
    return {"status": "injected", "clients": _manager.count}


# ── WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time transcript stream. Connect to receive transcript chunks as JSON."""
    await _manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await _manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)
        await _manager.disconnect(websocket)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    uvicorn.run("meetingmind.main:app", host="0.0.0.0", port=8000, reload=False)
