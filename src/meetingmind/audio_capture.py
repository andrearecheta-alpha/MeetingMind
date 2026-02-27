"""
audio_capture.py
----------------
Dual-source real-time audio capture: microphone AND system audio simultaneously.

PRIVACY GUARANTEE
-----------------
All audio is captured and processed entirely on this device. No raw audio,
PCM data, or processed samples are transmitted to any external server, API,
or third-party service. Recording stops the moment stop() is called.

How system audio capture works (Windows WASAPI Loopback)
---------------------------------------------------------
Windows WASAPI exposes "loopback" input devices that record whatever is
currently playing through the speakers — this is how Zoom, Teams, and Meet
audio can be captured without a second microphone or extra hardware.

If no loopback device is found, MeetingMind logs a help message and falls
back to microphone-only mode. See find_wasapi_loopback() for setup guidance.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pyaudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Whisper requires 16 kHz mono float32 input.
TARGET_SAMPLE_RATE: int = 16_000

# Default length of each emitted audio chunk in seconds.
DEFAULT_CHUNK_DURATION: float = 1.5

# PyAudio reads this many raw frames per call — small enough for low latency.
_PA_FRAMES_PER_BUFFER: int = 1_024

# int16 PCM — widest compatibility across device drivers.
_PA_FORMAT: int = pyaudio.paInt16
_PA_CHANNELS: int = 1

# Maximum number of chunks held in the queue before oldest are dropped.
# At 3-second chunks this is 64 × 3 = ~3 minutes of backpressure.
_QUEUE_MAXSIZE: int = 64


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AudioSource(str, Enum):
    """Identifies which physical source an audio chunk came from."""
    MICROPHONE = "microphone"   # Local microphone — captures the user's voice
    SYSTEM     = "system"       # WASAPI loopback — captures Zoom/Teams/Meet etc.


@dataclass
class AudioChunk:
    """
    A fixed-duration window of audio ready for transcription.

    Attributes:
        data:      float32 numpy array, mono, resampled to TARGET_SAMPLE_RATE Hz.
        source:    MICROPHONE or SYSTEM — used for speaker labelling downstream.
        timestamp: Unix timestamp of when this chunk's recording window began.
        rms:       Root-mean-square amplitude in [0, 1]. Used for silence detection;
                   chunks with very low RMS are skipped before calling Whisper.
    """
    data:      np.ndarray
    source:    AudioSource
    timestamp: float
    rms:       float = field(default=0.0)   # average RMS over the chunk (for display)
    peak_rms:  float = field(default=0.0)   # max(|sample|) — used for silence gate


# ---------------------------------------------------------------------------
# AudioCapture
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Captures microphone and system audio in parallel using PyAudio.

    Each source runs in its own daemon thread. Audio is normalised, resampled
    to 16 kHz, buffered into fixed-duration chunks, and placed onto a shared
    queue for downstream processing by StreamingTranscriber.

    PRIVACY: All audio is processed locally on this device. Nothing is sent
             to any external server or API at any point.

    Basic usage
    -----------
        cap = AudioCapture(privacy_mode=True)
        cap.start()
        try:
            while True:
                chunk = cap.get_chunk(timeout=1.0)
                if chunk:
                    process(chunk)
        finally:
            cap.close()

    Context manager usage
    ---------------------
        with AudioCapture() as cap:
            chunk = cap.get_chunk()
    """

    def __init__(
        self,
        mic_device_index:    Optional[int]   = None,
        system_device_index: Optional[int]   = None,
        chunk_duration:      float           = DEFAULT_CHUNK_DURATION,
        sample_rate:         int             = TARGET_SAMPLE_RATE,
        privacy_mode:        bool            = True,
    ) -> None:
        """
        Args:
            mic_device_index:    PyAudio device index for the microphone.
                                 None = OS default input device.
            system_device_index: PyAudio device index for WASAPI loopback.
                                 None = auto-detect via find_wasapi_loopback().
            chunk_duration:      Length of each audio chunk in seconds (default 3).
            sample_rate:         Target Hz — must be 16 000 for Whisper.
            privacy_mode:        When True, prints a local-processing confirmation
                                 banner when capture starts.
        """
        self.mic_device_index    = mic_device_index
        self.system_device_index = system_device_index
        self.chunk_duration      = chunk_duration
        self.sample_rate         = sample_rate
        self.privacy_mode        = privacy_mode

        self._pa:           pyaudio.PyAudio            = pyaudio.PyAudio()
        self._chunk_queue:  queue.Queue[AudioChunk]    = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        # Separate queue for per-frame RMS (mic-only).  Only the mic stream
        # writes here; the system audio stream writes only to _chunk_queue.
        # This guarantees that _rms_broadcast_loop and _transcription_loop
        # both see the SAME mic signal — no competing-stream signal split.
        self._rms_queue:    queue.Queue[float]         = queue.Queue(maxsize=50)
        self._running:      bool                       = False
        self._threads:      list[threading.Thread]     = []

        if privacy_mode:
            logger.info(
                "PRIVACY: AudioCapture created — all audio stays on this device."
            )

    # ── Device discovery ──────────────────────────────────────────────────────

    def list_devices(self) -> list[dict]:
        """
        Return metadata for every available audio input device.

        Useful for identifying specific microphone or loopback device indices
        to pass to mic_device_index / system_device_index.
        """
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if int(info["maxInputChannels"]) > 0:
                host_api_name = self._pa.get_host_api_info_by_index(
                    int(info["hostApi"])
                )["name"]
                devices.append({
                    "index":       i,
                    "name":        info["name"],
                    "channels":    int(info["maxInputChannels"]),
                    "sample_rate": int(info["defaultSampleRate"]),
                    "host_api":    host_api_name,
                })
        return devices

    def find_wasapi_loopback(self) -> Optional[int]:
        """
        Locate a WASAPI loopback device for capturing system audio on Windows.

        WASAPI loopback lets us record audio playing through the speakers
        (Zoom, Teams, Meet, YouTube, etc.) without extra hardware.

        Search order
        ------------
        1. Any WASAPI device whose name contains "loopback" (case-insensitive).
        2. Any input device named "Stereo Mix" (an older Windows loopback option).

        If nothing is found, a help message is logged explaining how to enable
        system audio capture.

        Returns:
            PyAudio device index, or None if no loopback device is found.
        """
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except Exception:
            logger.warning(
                "WASAPI host API not available. "
                "System audio capture requires Windows with WASAPI support."
            )
            return None

        wasapi_api_index = int(wasapi_info["index"])

        # Pass 1: true WASAPI loopback device
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if (
                int(info.get("hostApi", -1)) == wasapi_api_index
                and int(info.get("maxInputChannels", 0)) > 0
                and "loopback" in info["name"].lower()
            ):
                logger.info(
                    "WASAPI loopback device found: '%s' (index %d)",
                    info["name"], i,
                )
                return i

        # Pass 2: Stereo Mix on WASAPI only (older Windows fallback).
        # Must be restricted to the WASAPI host API — WDM-KS Stereo Mix
        # devices (e.g. Realtek index 13) reject mono/int16 streams with
        # [Errno -9999] and crash the capture thread.
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if (
                int(info.get("hostApi", -1)) == wasapi_api_index
                and int(info.get("maxInputChannels", 0)) > 0
                and "stereo mix" in info["name"].lower()
            ):
                logger.info(
                    "Stereo Mix device found: '%s' (index %d)", info["name"], i
                )
                return i

        logger.warning(
            "No system audio loopback device found. Continuing mic-only.\n"
            "  To also capture Zoom / Teams / Meet audio, enable one of:\n"
            "  1. 'Stereo Mix' in Windows Sound Settings > Recording tab\n"
            "  2. Install VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/\n"
            "  3. Install PyAudioWPatch for native WASAPI loopback: "
            "pip install PyAudioWPatch"
        )
        return None

    # ── Audio helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_float32(raw_bytes: bytes) -> np.ndarray:
        """Convert raw int16 PCM bytes to float32 in the range [-1.0, 1.0]."""
        return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32_768.0

    @staticmethod
    def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        Resample mono float32 audio from src_rate to dst_rate Hz.

        Uses numpy linear interpolation. Adequate for speech (< 8 kHz content).
        For music or wideband audio, prefer resampy or scipy.signal.resample_poly.
        """
        if src_rate == dst_rate:
            return data
        new_length = int(len(data) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0.0, len(data) - 1, new_length),
            np.arange(len(data)),
            data,
        ).astype(np.float32)

    @staticmethod
    def _rms(data: np.ndarray) -> float:
        """Compute root-mean-square amplitude of an audio array (result in [0, 1])."""
        return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))

    # ── Stream worker ─────────────────────────────────────────────────────────

    def _stream_worker(
        self,
        device_index: Optional[int],
        source:       AudioSource,
    ) -> None:
        """
        Thread target: opens one PyAudio input stream and pushes AudioChunks
        onto the shared queue at the configured chunk_duration cadence.

        Device-open strategy
        --------------------
        Tries *device_index* first.  If that fails and this is the MICROPHONE
        source, automatically walks through every other available input device
        as a fallback.  Each attempt is logged clearly so failures are easy to
        diagnose.  An "access denied" error aborts the fallback immediately —
        no point trying other devices when Windows has blocked mic access.

        Handles:
          - Device's native sample rate → 16 kHz resampling
          - int16 PCM bytes → float32 normalisation
          - Chunk boundary alignment
          - RMS calculation for downstream silence detection
          - Graceful recovery from transient OSError read failures
          - Backpressure: drops the oldest buffered chunk if queue is full
        """
        chunk_samples = int(self.sample_rate * self.chunk_duration)

        # ── 1. Open a stream (primary device, then fallbacks for mic) ─────────
        stream:      Optional[object] = None
        device_name: str              = str(device_index)
        device_rate: int              = TARGET_SAMPLE_RATE
        tried_names: list[str]        = []

        candidates: list[Optional[int]] = [device_index]
        if source == AudioSource.MICROPHONE:
            for dev in self.list_devices():
                idx = dev["index"]
                if idx != device_index:
                    candidates.append(idx)

        for i, candidate in enumerate(candidates):
            cand_name = str(candidate)
            try:
                if candidate is not None:
                    info = self._pa.get_device_info_by_index(candidate)
                else:
                    info = self._pa.get_default_input_device_info()
                cand_name = str(info.get("name", cand_name))
                cand_rate = int(info.get("defaultSampleRate", TARGET_SAMPLE_RATE))
            except Exception as exc:
                tried_names.append(cand_name)
                logger.debug("Cannot query device %s: %s", candidate, exc)
                continue

            if i > 0:
                logger.warning(
                    "FALLBACK: Trying %s device '%s' (index %s) instead",
                    source.value, cand_name, candidate,
                )

            try:
                _stream = self._pa.open(
                    format=_PA_FORMAT,
                    channels=_PA_CHANNELS,
                    rate=cand_rate,
                    input=True,
                    input_device_index=candidate,
                    frames_per_buffer=_PA_FRAMES_PER_BUFFER,
                )
                stream      = _stream
                device_name = cand_name
                device_rate = cand_rate
                logger.info(
                    "Audio stream open — source: %s | device: '%s' (index %s) | rate: %d Hz",
                    source.value, device_name, candidate, device_rate,
                )
                break  # success

            except OSError as exc:
                tried_names.append(cand_name)
                logger.error(
                    "ERROR: Failed to open %s device '%s' (index %s): %s",
                    source.value, cand_name, candidate, exc,
                )
                if any(x in str(exc).lower() for x in ["access denied", "-9999", "-9997"]):
                    logger.error(
                        "  → Windows mic access may be blocked. "
                        "Check Settings → Privacy & Security → Microphone → "
                        "Allow apps to access your microphone"
                    )
                    break  # Permission error — fallbacks won't help

        if stream is None:
            logger.error(
                "No working %s device found. Tried: %s\n"
                "  → Check mic permissions and that the device is not in use by another app.",
                source.value, tried_names or [str(device_index)],
            )
            return

        # ── 2. Run the capture loop ───────────────────────────────────────────
        try:
            buffer:           np.ndarray = np.zeros(0, dtype=np.float32)
            chunk_start_time: float      = time.time()

            while self._running:
                # Read raw PCM from the device
                try:
                    raw = stream.read(_PA_FRAMES_PER_BUFFER, exception_on_overflow=False)
                except OSError as exc:
                    # Transient underflow/overflow — log at debug level and continue
                    logger.debug("Stream read error (%s): %s", source.value, exc)
                    continue

                # Convert: int16 bytes → float32 → resample to 16 kHz
                samples = self._to_float32(raw)
                samples = self._resample(samples, device_rate, self.sample_rate)

                # Fan out per-frame RMS to the RMS queue — MIC ONLY.
                # System audio must NOT write here; if it did, it would
                # overwrite the mic signal and cause the UI meter and the
                # silence gate to see different audio (the original bug).
                if source == AudioSource.MICROPHONE:
                    frame_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    try:
                        self._rms_queue.put_nowait(frame_rms)
                    except queue.Full:
                        try:
                            self._rms_queue.get_nowait()   # drop oldest
                        except queue.Empty:
                            pass
                        self._rms_queue.put_nowait(frame_rms)

                buffer  = np.concatenate([buffer, samples])

                # Emit as many complete chunks as the buffer holds
                while len(buffer) >= chunk_samples:
                    chunk_data = buffer[:chunk_samples].copy()
                    buffer     = buffer[chunk_samples:]

                    audio_chunk = AudioChunk(
                        data=chunk_data,
                        source=source,
                        timestamp=chunk_start_time,
                        rms=self._rms(chunk_data),
                        peak_rms=float(np.max(np.abs(chunk_data))),
                    )

                    # Non-blocking put; drop oldest if queue is full
                    try:
                        self._chunk_queue.put_nowait(audio_chunk)
                    except queue.Full:
                        try:
                            self._chunk_queue.get_nowait()   # discard oldest
                        except queue.Empty:
                            pass
                        self._chunk_queue.put_nowait(audio_chunk)

                    chunk_start_time = time.time()

        except Exception as exc:
            logger.exception(
                "Unexpected error in %s stream worker: %s", source.value, exc
            )
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            logger.info("Audio stream closed — source: %s", source.value)

    # ── Public interface ──────────────────────────────────────────────────────

    def probe_device(self, device_index: Optional[int]) -> tuple[bool, str, str]:
        """
        Briefly open and immediately close a device stream to verify it works.

        Call this (via asyncio.to_thread) before starting the full capture loop
        to give the API layer a chance to return a clear error instead of
        silently capturing nothing.

        Returns:
            (success, device_name, error_message).
            On success, error_message is "".
            On failure, error_message contains the OSError description.
        """
        name = f"device:{device_index}"
        try:
            if device_index is not None:
                info = self._pa.get_device_info_by_index(device_index)
            else:
                info = self._pa.get_default_input_device_info()
            name = str(info.get("name", name))
            rate = int(info.get("defaultSampleRate", TARGET_SAMPLE_RATE))
            stream = self._pa.open(
                format=_PA_FORMAT,
                channels=_PA_CHANNELS,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=_PA_FRAMES_PER_BUFFER,
            )
            stream.stop_stream()
            stream.close()
            logger.info("Device probe OK: '%s' (index %s)", name, device_index)
            return True, name, ""
        except Exception as exc:
            logger.warning(
                "Device probe FAILED: '%s' (index %s): %s", name, device_index, exc
            )
            return False, name, str(exc)

    def start(self) -> None:
        """
        Start audio capture on all available sources.

        Always starts the microphone. Attempts WASAPI loopback for system audio;
        logs a warning and continues in mic-only mode if no loopback is found.
        """
        if self._running:
            logger.warning("AudioCapture.start() called while already running — ignored.")
            return

        self._running = True

        if self.privacy_mode:
            print(
                "\n"
                "  +--------------------------------------------------+\n"
                "  |  PRIVACY  Audio capture is LIVE                  |\n"
                "  |  All recording and processing is 100% local.     |\n"
                "  |  No audio data is sent to any server or API.     |\n"
                "  +--------------------------------------------------+\n"
            )

        # ── Microphone (always started) ──────────────────────────────────────
        mic_thread = threading.Thread(
            target=self._stream_worker,
            args=(self.mic_device_index, AudioSource.MICROPHONE),
            daemon=True,
            name="meetingmind-mic",
        )
        self._threads.append(mic_thread)
        mic_thread.start()

        # ── System audio via WASAPI loopback ─────────────────────────────────
        # System audio is OPT-IN only — never auto-detected here.
        # Auto-detection (find_wasapi_loopback) was the original bug: it
        # opened a second stream that competed with the mic, splitting the
        # signal so both streams got ~half the audio.  The server layer
        # (meeting_start) is responsible for resolving the device index and
        # passing it explicitly; if it passes None, we run mic-only mode.

        if self.system_device_index is not None:
            sys_thread = threading.Thread(
                target=self._stream_worker,
                args=(self.system_device_index, AudioSource.SYSTEM),
                daemon=True,
                name="meetingmind-system",
            )
            self._threads.append(sys_thread)
            sys_thread.start()
            logger.info("Dual-source capture: microphone + system audio active.")
        else:
            logger.info("Single-source capture: microphone only.")

    def stop(self) -> None:
        """Signal all stream threads to exit and wait for them to finish."""
        self._running = False
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()
        logger.info("AudioCapture stopped.")

    def get_chunk(self, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Retrieve the next AudioChunk from the queue.

        Blocks for up to `timeout` seconds. Returns None on timeout so callers
        can check self._running and exit gracefully.
        """
        try:
            return self._chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop capture and release all PyAudio resources."""
        self.stop()
        self._pa.terminate()
        logger.info("PyAudio terminated.")

    def get_latest_rms(self) -> float:
        """
        Return the most recent mic per-frame RMS. Non-blocking.

        Drains the entire _rms_queue and returns the last value so the
        caller always sees the freshest reading.  Returns 0.0 if no mic
        data has arrived yet.

        Safe to call from any thread (queue.Queue is thread-safe).
        """
        rms = 0.0
        try:
            while True:
                rms = self._rms_queue.get_nowait()
        except queue.Empty:
            pass
        return rms
