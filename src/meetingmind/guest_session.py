"""
guest_session.py
----------------
Helpers for the Guest Phone Mic feature.

A guest connects their phone browser to /guest?pin=XXXX, sends raw Float32 PCM
over WebSocket (/ws/guest?pin=XXXX), and their speech is labelled [Them] in the
host transcript.

WebSocket connection logic lives in main.py; only pure data-manipulation helpers
belong here to keep this module easily unit-testable without importing FastAPI.
"""

import base64
import random

import numpy as np


def generate_pin() -> str:
    """Return a random 4-digit zero-padded PIN string, e.g. '0472'."""
    return f"{random.randint(0, 9999):04d}"


def decode_guest_audio(b64: str) -> np.ndarray:
    """
    Decode a base64-encoded block of raw Float32 PCM bytes sent by the guest
    browser (from a JavaScript ``Float32Array.buffer``) into a writable float32
    numpy array suitable for Whisper.

    Args:
        b64: Base64 string produced by ``btoa()`` on the browser side.

    Returns:
        1-D float32 numpy array, values in [-1.0, 1.0].

    Raises:
        ValueError: If ``b64`` is not valid base64 or if the decoded byte length
                    is not divisible by 4 (i.e. not aligned to float32 words).

    Notes:
        ``np.frombuffer`` returns a read-only view; ``.copy()`` is mandatory
        because Whisper's internal ``pad_or_trim`` calls ``np.pad`` which
        requires a writable array.
    """
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"base64 decode failed: {exc}") from exc

    if len(raw) % 4 != 0:
        raise ValueError(
            f"Decoded byte length ({len(raw)}) is not divisible by 4 — "
            "data is not valid Float32 PCM"
        )

    return np.frombuffer(raw, dtype=np.float32).copy()
