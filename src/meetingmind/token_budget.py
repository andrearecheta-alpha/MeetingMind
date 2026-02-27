"""
token_budget.py
---------------
Lightweight token estimation and budget enforcement for Claude API calls.

Uses word-count * 1.3 as a fast approximation (no tokeniser required).
Typical error vs. actual cl100k tokens: ±10%.

Limits are loaded from config/settings.toml [token_budget] at import time.
Falls back to hardcoded defaults if the file is missing or the section absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults (used when settings.toml is absent or incomplete) ───────────────
_DEFAULTS = {
    "max_input_tokens":  800,
    "max_output_tokens": 120,
    "context_tokens":    200,
    "warn_at_percent":   80,
}

_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"


def _load_settings() -> dict:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning("tomllib unavailable — using default token budget.")
            return _DEFAULTS.copy()

    if not _SETTINGS_PATH.exists():
        logger.warning("config/settings.toml not found — using default token budget.")
        return _DEFAULTS.copy()

    try:
        with open(_SETTINGS_PATH, "rb") as f:
            data = tomllib.load(f)
        section = data.get("token_budget", {})
        cfg = {k: section.get(k, v) for k, v in _DEFAULTS.items()}
        return cfg
    except Exception as exc:
        logger.warning("Failed to parse settings.toml: %s — using defaults.", exc)
        return _DEFAULTS.copy()


_cfg = _load_settings()

MAX_INPUT_TOKENS:  int = _cfg["max_input_tokens"]
MAX_OUTPUT_TOKENS: int = _cfg["max_output_tokens"]
CONTEXT_TOKENS:    int = _cfg["context_tokens"]
WARN_AT_PERCENT:   int = _cfg["warn_at_percent"]

logger.info(
    "Token budget: input=%d output=%d context=%d warn_at=%d%%",
    MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, CONTEXT_TOKENS, WARN_AT_PERCENT,
)


# ── Public API ────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def enforce_budget(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    words = text.split()
    while len(words) > 0 and estimate_tokens(" ".join(words)) > max_tokens:
        words = words[10:]
    return " ".join(words)


def log_usage(stage: str, text: str) -> None:
    tokens  = estimate_tokens(text)
    warn_at = int(MAX_INPUT_TOKENS * WARN_AT_PERCENT / 100)
    if tokens > MAX_INPUT_TOKENS:
        status = "OVER_BUDGET"
    elif tokens >= warn_at:
        status = "WARN"
    else:
        status = "OK"
    logger.info("TOKENS %s %s: ~%d", status, stage, tokens)
