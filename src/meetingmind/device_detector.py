"""
device_detector.py
------------------
Intelligent audio device categorisation and auto-selection engine.

On startup MeetingMind calls detect_devices() once. The result is cached and
served from GET /devices. POST /meeting/start uses the recommended indices
unless the user explicitly overrides them via query parameters.

PRIVACY: Device names are scanned locally only. No device information is
         transmitted externally.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------


class DeviceCategory(str, Enum):
    REAL_MIC     = "REAL_MIC"
    VIRTUAL_MIC  = "VIRTUAL_MIC"
    SYSTEM_AUDIO = "SYSTEM_AUDIO"
    UNKNOWN      = "UNKNOWN"


# Host API preference for deduplication — higher wins within a device group.
# Windows PyAudio exposes each physical device once per host API layer.
# WASAPI is the lowest-latency modern API; WDM-KS is kernel-level; DS/MME are legacy.
HOST_API_PRIORITY: dict[str, int] = {
    "windows wasapi":        4,
    "windows wdm-ks":        3,
    "windows directsound":   2,
    "mme":                   1,
}


@dataclass
class DetectedDevice:
    index:           int
    name:            str
    category:        DeviceCategory
    score:           int            # 0–10; higher = preferred
    channels:        int
    sample_rate:     float
    host_api:        str
    warning:         Optional[str] = None   # shown as tooltip in the UI
    duplicate_count: int = 1               # how many host-API instances were merged

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class DetectionResult:
    recommended_mic:    Optional[dict]  # best REAL_MIC serialised to dict
    recommended_system: Optional[dict]  # best SYSTEM_AUDIO serialised to dict
    all_devices:        list            # list of DetectedDevice (not yet serialised)
    warnings:           list[str] = field(default_factory=list)
    mic_permission_ok:  Optional[bool] = None  # None=unknown, True=ok, False=blocked


# ---------------------------------------------------------------------------
# Keyword sets (all comparisons are case-insensitive)
# ---------------------------------------------------------------------------

VIRTUAL_KEYWORDS = [
    "camo", "obs", "virtual", "vb-cable", "soundflower",
    "voicemeeter", "screencapture", "discord", "zoom mic",
]

SYSTEM_KEYWORDS = [
    "stereo mix", "loopback", "what u hear", "wave out",
    "speakers (loopback)", "blackhole",
]

HIGH_PRIORITY_KEYWORDS = [
    "array", "intel", "realtek", "built-in", "internal", "laptop", "integrated",
]

USB_KEYWORDS = ["usb", "external"]

HEADSET_KEYWORDS = ["headset", "headphone", "earphone"]

# Aliases and non-mic devices to skip entirely (substring match, case-insensitive).
# Includes Windows aliases that duplicate real devices, and "PC Speaker" which
# appears as an input device in WDM-KS on some Realtek drivers but is not a mic.
SKIP_PREFIXES = [
    "microsoft sound mapper",
    "primary sound capture",   # matches "Primary Sound Capture Driver"
    "primary sound driver",
    "pc speaker",              # Realtek WDM-KS loopback, not a real mic input
]


# ---------------------------------------------------------------------------
# Categorisation helpers
# ---------------------------------------------------------------------------

def _contains(name_lower: str, keywords: list[str]) -> bool:
    return any(kw in name_lower for kw in keywords)


def _categorise(name: str, in_channels: int) -> tuple[DeviceCategory, int, Optional[str]]:
    """
    Return (category, score, warning) for a device.

    Rules are evaluated top-to-bottom; first match wins.
    """
    if in_channels == 0:
        return DeviceCategory.UNKNOWN, 0, None   # output-only — caller should skip

    n = name.lower()

    # Substring check so "Microsoft Sound Mapper - Input" and "Primary Sound Capture Driver"
    # are caught even when Windows appends a suffix to the base alias name.
    if any(n.startswith(skip) for skip in SKIP_PREFIXES):
        return DeviceCategory.UNKNOWN, 0, None   # alias or non-mic — skip

    if _contains(n, SYSTEM_KEYWORDS):
        return DeviceCategory.SYSTEM_AUDIO, 10, None

    if _contains(n, VIRTUAL_KEYWORDS):
        return DeviceCategory.VIRTUAL_MIC, 2, f"Virtual mic detected ({name!r}) — may capture silence or low-quality audio"

    if _contains(n, HIGH_PRIORITY_KEYWORDS):
        return DeviceCategory.REAL_MIC, 10, None

    if _contains(n, USB_KEYWORDS):
        return DeviceCategory.REAL_MIC, 8, None

    if _contains(n, HEADSET_KEYWORDS):
        return DeviceCategory.REAL_MIC, 6, None

    # Anything left with input channels — treat as real mic, moderate priority
    return DeviceCategory.REAL_MIC, 5, None


def _dedup_by_host_api(devices: list[DetectedDevice]) -> list[DetectedDevice]:
    """
    Collapse duplicate entries for the same physical device across host APIs.

    Windows PyAudio reports each physical device once per host API
    (MME, DirectSound, WASAPI, WDM-KS).  We keep the highest-priority entry
    and record how many duplicates were merged in `duplicate_count`.

    Grouping key: first 20 chars of the lowercased, stripped name.  This is
    long enough to distinguish different mics, but short enough to match an
    MME-truncated name with its full-length WASAPI equivalent (MME caps device
    names at ~32 chars, so "Microphone Array (Intel® Smart" in MME and the
    full 70-char WASAPI name share identical first-20 chars).

    Priority: WASAPI(4) > WDM-KS(3) > DirectSound(2) > MME(1) > other(0).
    Tiebreak: lowest PyAudio device index (first device PyAudio enumerates).
    """
    groups: dict[str, list[DetectedDevice]] = {}
    for dev in devices:
        key = dev.name.strip().lower()[:20]
        groups.setdefault(key, []).append(dev)

    result: list[DetectedDevice] = []
    for group in groups.values():
        # max() with (priority DESC, index ASC via negation)
        best = max(
            group,
            key=lambda d: (HOST_API_PRIORITY.get(d.host_api.lower(), 0), -d.index),
        )
        best.duplicate_count = len(group)
        result.append(best)

    return result


# ---------------------------------------------------------------------------
# Windows microphone permission check
# ---------------------------------------------------------------------------

def _check_mic_permission(pa) -> Optional[bool]:
    """
    Attempt to open the default input device briefly to verify mic access.

    On Windows, if the user has not granted microphone permission in
    Settings → Privacy & Security → Microphone, PyAudio raises OSError -9999
    ("Unanticipated host error") instead of opening the stream.

    Returns:
        True   — device opened and closed successfully (permission granted)
        False  — access denied (Windows privacy block most likely)
        None   — result ambiguous (no default device, device busy, etc.)
    """
    try:
        info = pa.get_default_input_device_info()
        rate = int(info.get("defaultSampleRate", 16_000))
        stream = pa.open(
            format=4,        # pyaudio.paInt16 = 4 (avoid module-level import)
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=1_024,
        )
        stream.stop_stream()
        stream.close()
        logger.info("Mic permission check: OK (default input accessible)")
        return True
    except OSError as exc:
        err = str(exc).lower()
        # -9999 = Unanticipated host error (Windows privacy block)
        # -9997 = Invalid sample rate (can also indicate device locked)
        if "access denied" in err or "-9999" in str(exc) or "-9997" in str(exc):
            logger.warning(
                "Mic permission check: access may be blocked — %s\n"
                "  → Windows: Settings → Privacy & Security → Microphone → "
                "Allow apps to access your microphone",
                exc,
            )
            return False
        logger.debug("Mic permission check inconclusive (non-permission error): %s", exc)
        return None
    except Exception as exc:
        logger.debug("Mic permission check error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_devices() -> DetectionResult:
    """
    Enumerate all PyAudio devices, categorise them, and return a DetectionResult
    with recommended mic/system selections and any warnings.

    PRIVACY: Device names are scanned locally only. No device information is
             transmitted externally.
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("pyaudio is not installed — device detection unavailable.")
        return DetectionResult(
            recommended_mic=None,
            recommended_system=None,
            all_devices=[],
            warnings=["pyaudio not installed — device detection unavailable"],
        )

    pa = pyaudio.PyAudio()
    raw: list[DetectedDevice] = []
    mic_permission = _check_mic_permission(pa)

    try:
        count = pa.get_device_count()
        for i in range(count):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception as exc:
                logger.debug("Skipping device %d: %s", i, exc)
                continue

            in_ch    = int(info.get("maxInputChannels", 0))
            name     = str(info.get("name", f"Device {i}")).strip()
            rate     = float(info.get("defaultSampleRate", 44100))
            host_idx = int(info.get("hostApi", 0))

            try:
                host_info = pa.get_host_api_info_by_index(host_idx)
                host_api  = str(host_info.get("name", "Unknown"))
            except Exception:
                host_api = "Unknown"

            category, score, _ = _categorise(name, in_ch)

            if category == DeviceCategory.UNKNOWN:
                continue   # output-only or skipped alias

            raw.append(DetectedDevice(
                index=i,
                name=name,
                category=category,
                score=score,
                channels=in_ch,
                sample_rate=rate,
                host_api=host_api,
            ))

    finally:
        pa.terminate()

    # ── Deduplicate within each category, then sort by score ─────────────────
    # Dedup is done per-category so a real mic and a virtual mic with a
    # similar name are never merged into one entry.
    real_mics     = sorted(
        _dedup_by_host_api([d for d in raw if d.category == DeviceCategory.REAL_MIC]),
        key=lambda d: d.score, reverse=True,
    )
    virtual_mics  = sorted(
        _dedup_by_host_api([d for d in raw if d.category == DeviceCategory.VIRTUAL_MIC]),
        key=lambda d: d.score, reverse=True,
    )
    system_audios = sorted(
        _dedup_by_host_api([d for d in raw if d.category == DeviceCategory.SYSTEM_AUDIO]),
        key=lambda d: d.score, reverse=True,
    )

    # ── Update per-device warning field to reflect merge count ────────────────
    for dev in virtual_mics:
        if dev.duplicate_count > 1:
            dev.warning = (
                f"{dev.duplicate_count} virtual instances merged — "
                "may capture silence or low-quality audio"
            )
        else:
            dev.warning = "Virtual mic — may capture silence or low-quality audio"

    # ── Top-level warnings: one entry per unique virtual device ───────────────
    warnings: list[str] = []
    for dev in virtual_mics:
        if dev.duplicate_count > 1:
            warnings.append(
                f"Virtual mic '{dev.name}' detected "
                f"({dev.duplicate_count} instances across host APIs) — "
                "may capture silence or low-quality audio"
            )
        else:
            warnings.append(
                f"Virtual mic '{dev.name}' detected — "
                "may capture silence or low-quality audio"
            )

    # ── Pick recommendations ──────────────────────────────────────────────────
    recommended_mic: Optional[DetectedDevice] = None
    if real_mics:
        recommended_mic = real_mics[0]
    elif virtual_mics:
        recommended_mic = virtual_mics[0]
        logger.warning("No real mic found — falling back to virtual mic: %s", recommended_mic.name)

    recommended_system: Optional[DetectedDevice] = system_audios[0] if system_audios else None

    logger.info(
        "Device detection: %d real mics, %d virtual, %d system audio | "
        "recommended mic=%s score=%s host_api=%s | recommended system=%s",
        len(real_mics),
        len(virtual_mics),
        len(system_audios),
        recommended_mic.name if recommended_mic else "None",
        recommended_mic.score if recommended_mic else "—",
        recommended_mic.host_api if recommended_mic else "—",
        recommended_system.name if recommended_system else "None",
    )

    all_ordered = real_mics + virtual_mics + system_audios

    return DetectionResult(
        recommended_mic=recommended_mic.to_dict() if recommended_mic else None,
        recommended_system=recommended_system.to_dict() if recommended_system else None,
        all_devices=all_ordered,
        warnings=warnings,
        mic_permission_ok=mic_permission,
    )
