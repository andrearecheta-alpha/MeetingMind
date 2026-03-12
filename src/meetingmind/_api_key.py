"""Shared Anthropic API key resolver."""
from __future__ import annotations
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_api_key() -> str:
    """Resolve ANTHROPIC_API_KEY from env var or <project_root>/.env.

    Raises:
        EnvironmentError: if key not found in either location.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )
