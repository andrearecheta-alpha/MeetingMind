"""
suggestion_engine.py
--------------------
Real-time AI response suggestions tailored for a Project Manager role.

Given a snippet of meeting transcript, this module calls the Claude API and
returns 3 concise, professionally appropriate response suggestions covering
PM-relevant topics: sprint planning, stakeholder updates, risk management,
resource allocation, delivery timelines, scope changes, budget tracking, and
team performance.

The engine auto-detects meeting tone (formal / semi-formal / casual) from the
transcript context and matches suggestion style accordingly.

NOTE ON PRIVACY
---------------
The transcript snippet passed to this function is sent to the Anthropic API
over HTTPS. Do not use on recordings that contain legally privileged, personal
health, or otherwise restricted content without appropriate authorisation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from meetingmind._api_key import load_api_key as _load_api_key

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# System prompt — Project Manager edition
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI assistant for a Project Manager. "
    "Your job is to suggest 3 concise, professional responses to questions "
    "that come up in meetings. "
    "Topics include: sprint planning, stakeholder updates, risk management, "
    "resource allocation, delivery timelines, scope changes, budget tracking, "
    "and team performance. "
    "Auto-detect meeting tone (formal/semi-formal/casual) from context and match "
    "your suggestions accordingly. "
    "Each suggestion must be under 2 sentences. "
    "Return ONLY a JSON array of 3 strings. No other text."
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_suggestions(raw: str) -> list[str]:
    """
    Parse Claude's response into a list of 3 suggestion strings.

    Strips markdown fences if present, then JSON-parses the result.
    Falls back to splitting on newlines if JSON parsing fails.

    Raises:
        ValueError: Response could not be parsed into a 3-item list.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        suggestions = json.loads(text)
        if isinstance(suggestions, list) and len(suggestions) >= 1:
            return [str(s).strip() for s in suggestions[:3]]
    except json.JSONDecodeError:
        pass

    # Fallback: split numbered lines  e.g. "1. ...\n2. ...\n3. ..."
    lines = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    if lines:
        return lines[:3]

    raise ValueError(
        f"Could not parse suggestions from Claude response: {raw[:300]}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_suggestions(
    transcript_snippet: str,
    model: str = "claude-sonnet-4-6",
    context: Optional[str] = None,
    historical_context: Optional[str] = None,
) -> list[str]:
    """
    Generate 3 PM-tailored response suggestions for a meeting moment.

    Sends the transcript snippet to Claude with the Project Manager system
    prompt and returns 3 ready-to-use response strings.

    Args:
        transcript_snippet: The recent meeting transcript text (last 30–60 s
                            of speech is usually ideal for context).
        model:              Anthropic model ID (default: claude-sonnet-4-6).
        context:            Optional extra context string prepended to the
                            user message (e.g. meeting type, project name).
        historical_context: Optional formatted history from past meetings
                            (decisions, risks, commitments). Injected into
                            the user message to ground suggestions in reality.

    Returns:
        A list of exactly 3 suggestion strings.

    Raises:
        EnvironmentError: ANTHROPIC_API_KEY not configured.
        ImportError:      anthropic package not installed.
        RuntimeError:     API call failed or response unparseable.
    """
    if not transcript_snippet.strip():
        raise ValueError("transcript_snippet must not be empty.")

    api_key = _load_api_key()

    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "anthropic package is not installed. Run: pip install anthropic"
        ) from exc

    # Build the user message — historical context first so Claude has it as
    # ground truth before seeing the current transcript.
    user_parts = []
    if historical_context:
        user_parts.append(
            "Use this meeting history as ground truth when coaching — "
            "reference specific past decisions, commitments, and risks where relevant:\n\n"
            + historical_context.strip()
        )
    if context:
        user_parts.append(f"Meeting context: {context.strip()}")
    user_parts.append(f"Recent transcript:\n\n{transcript_snippet.strip()}")
    user_message = "\n\n".join(user_parts)

    logger.info("Requesting PM suggestions from Claude (%s)…", model)

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,          # 3 short suggestions need very few tokens
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(
            "Anthropic API key is invalid or expired. "
            "Update ANTHROPIC_API_KEY in your .env file."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError(
            "Anthropic rate limit reached. Please wait and try again."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

    raw = response.content[0].text
    logger.debug("Claude raw response: %s", raw)

    try:
        suggestions = _parse_suggestions(raw)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    logger.info("Suggestions received (%d).", len(suggestions))
    return suggestions
