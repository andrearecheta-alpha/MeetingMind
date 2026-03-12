"""
suggestion_engine.py
--------------------
Real-time AI response suggestions tailored for meeting professionals.

Given a snippet of meeting transcript, this module calls the Claude API and
returns 3 concise, professionally appropriate response suggestions covering
meeting-relevant topics: sprint planning, stakeholder updates, risk management,
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
from meetingmind.token_budget import enforce_budget, log_usage, CONTEXT_TOKENS

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Decision detection — pure local, no API call
# ---------------------------------------------------------------------------

DECISION_PATTERNS: list[str] = [
    "let's go with",
    "we've decided",
    "we'll proceed",
    "going forward",
    "decision is",
    "approved",
    "confirmed",
    "decided",
    "agreed",
    "approval process",
    "requesting access",
    "first wave",
    "background verification",
    "eligibility criteria",
    "granting access",
]


def detect_decision(text: str) -> dict:
    """
    Scan a transcript chunk for decision-signalling phrases.

    Returns a dict with:
        is_decision (bool)   – True when at least one pattern matches.
        confidence  (str)    – "high" | "none"
        phrase      (str|None) – first matched phrase, or None.

    Pure string matching — no API call, safe to call on every chunk.
    """
    tl = text.lower()
    matched = [p for p in DECISION_PATTERNS if p in tl]
    return {
        "is_decision": bool(matched),
        "confidence":  "high" if matched else "none",
        "phrase":      matched[0] if matched else None,
    }


# ---------------------------------------------------------------------------
# Coaching triggers — pure local, no API call
# ---------------------------------------------------------------------------

PM_COACHING_TRIGGERS: list[dict] = [
    {
        "id":         "no_decision_owner",
        "pattern":    ["someone should", "we should"],
        "prompt":     "No decision owner named — confirm accountability",
        "confidence": 0.85,
    },
    {
        "id":         "resistance_detected",
        "pattern":    ["not sure", "concerned", "worried", "but"],
        "prompt":     "Resistance detected — clarify ROI",
        "confidence": 0.80,
    },
    {
        "id":         "scope_creep",
        "pattern":    ["also add", "while we're at it", "one more thing"],
        "prompt":     "Scope change detected — confirm change control",
        "confidence": 0.90,
    },
    {
        "id":         "no_timeline",
        "pattern":    ["let's decide later", "tbd", "to be determined"],
        "prompt":     "Timeline missing — set deadline now",
        "confidence": 0.85,
    },
]

# Backwards-compatible alias
COACHING_TRIGGERS = PM_COACHING_TRIGGERS

EA_COACHING_TRIGGERS: list[dict] = [
    {
        "id":         "no_followup_owner",
        "pattern":    ["someone will", "we should followup", "someone should send",
                       "someone should", "we should", "we need to"],
        "prompt":     "No follow-up owner — confirm who sends the summary",
        "confidence": 0.85,
    },
    {
        "id":         "exec_preference",
        "pattern":    ["he prefers", "she wants", "the executive likes", "the CEO wants",
                       "he wants", "she prefers", "they want", "they prefer"],
        "prompt":     "Executive preference noted — log this for future reference",
        "confidence": 0.80,
    },
    {
        "id":         "protocol_deviation",
        "pattern":    ["skip approval", "bypass", "without authorization", "off the record"],
        "prompt":     "Protocol deviation detected — confirm authorization level",
        "confidence": 0.95,
    },
    {
        "id":         "commitment_without_auth",
        "pattern":    ["i'll make sure", "i'll arrange", "i'll confirm", "i'll send",
                       "i will", "i'll do", "i'll get", "i'll take care",
                       "i'll handle", "i'll follow up", "i'll check"],
        "prompt":     "Commitment made — log as action item",
        "confidence": 0.85,
    },
    {
        "id":         "sensitive_topic",
        "pattern":    ["confidential", "between us", "don't share", "off the record", "just between"],
        "prompt":     "Sensitive topic — discretion advised",
        "confidence": 0.95,
    },
    {
        "id":         "action_needed",
        "pattern":    ["next step", "action item", "follow up", "to do",
                       "need to", "have to", "must", "deadline", "by friday",
                       "by monday", "by end of", "asap", "urgent"],
        "prompt":     "Action item detected — capture owner and deadline",
        "confidence": 0.80,
    },
    {
        "id":         "scheduling",
        "pattern":    ["schedule", "calendar", "book a", "set up a meeting",
                       "send an invite", "block time", "reschedule"],
        "prompt":     "Scheduling request — confirm attendees and time",
        "confidence": 0.80,
    },
]

SALES_COACHING_TRIGGERS: list[dict] = PM_COACHING_TRIGGERS  # placeholder — uses PM triggers for now


def get_coaching_triggers(role: str = "PM") -> list[dict]:
    """Return the coaching trigger list for the given role."""
    _TRIGGERS_BY_ROLE = {
        "PM":     PM_COACHING_TRIGGERS,
        "EA":     EA_COACHING_TRIGGERS,
        "Sales":  SALES_COACHING_TRIGGERS,
        "Custom": PM_COACHING_TRIGGERS,
    }
    return _TRIGGERS_BY_ROLE.get(role, PM_COACHING_TRIGGERS)


# ---------------------------------------------------------------------------
# Time-based coaching triggers — evaluated on elapsed time, not text
# ---------------------------------------------------------------------------

_TIME_TRIGGERS: list[dict] = [
    {
        "id":                  "no_decision_20min",
        "prompt":              "20 min passed — no decisions yet. Redirect meeting.",
        "confidence":          1.0,
        "elapsed_threshold_s": 1200,   # 20 minutes
        "condition":           "no_decisions",
    },
]


def check_time_triggers(elapsed_seconds: float, decisions_count: int) -> list[dict]:
    """
    Evaluate time-based coaching triggers that are independent of transcript text.

    Args:
        elapsed_seconds:  Seconds since the meeting started.
        decisions_count:  Number of decisions detected so far this meeting.

    Returns:
        A (possibly empty) list of triggered coaching events.  Each item:
            trigger_id  (str)        – matches a _TIME_TRIGGERS entry id
            prompt      (str)        – coaching message to surface
            confidence  (float)      – always 1.0 (rule-based)
            matched     (None)       – always None for time triggers
    """
    triggered: list[dict] = []
    for trigger in _TIME_TRIGGERS:
        if elapsed_seconds < trigger["elapsed_threshold_s"]:
            continue
        if trigger["condition"] == "no_decisions" and decisions_count > 0:
            continue
        triggered.append({
            "trigger_id": trigger["id"],
            "prompt":     trigger["prompt"],
            "confidence": trigger["confidence"],
            "matched":    None,
        })
    return triggered


# ---------------------------------------------------------------------------
# Coaching resolution patterns — check if a prompt was addressed in transcript
# ---------------------------------------------------------------------------

COACHING_RESOLUTION_PATTERNS: dict[str, list[str]] = {
    "no_decision_owner": ["owner is", "responsible", "will handle", "assigned to"],
    "resistance_detected": ["roi is", "value is", "benefit", "saves us"],
}


def detect_coaching_resolution(trigger_id: str, text: str) -> bool:
    """
    Check if transcript text contains resolution signals for a coaching trigger.

    Returns True if the coaching prompt appears to have been addressed via
    keyword match.
    """
    patterns = COACHING_RESOLUTION_PATTERNS.get(trigger_id)
    if not patterns:
        return False
    tl = text.lower()
    return any(p in tl for p in patterns)


def detect_coaching(text: str, speaker_dominance: float = 0.0, role: str = "PM") -> list[dict]:
    """
    Scan a transcript chunk for coaching trigger conditions.

    Args:
        text:               Transcript text for this audio chunk.
        speaker_dominance:  Kept for API compatibility; currently unused.
        role:               Active user role (EA, PM, Sales, Custom).
                            Determines which trigger set is used.

    Returns:
        A (possibly empty) list of triggered coaching events.  Each item:
            trigger_id  (str)        – matches a trigger entry id
            prompt      (str)        – coaching message to surface to the user
            confidence  (float)      – confidence score for this trigger
            matched     (str|None)   – matched phrase

    Pattern matching is case-insensitive. Each trigger fires at most once per
    chunk even if multiple phrases from the same trigger match.
    """
    tl = text.lower()
    triggered: list[dict] = []
    triggers = get_coaching_triggers(role)

    logger.info(
        "COACHING INPUT: text=%r  role=%s  triggers=%d  ids=%s",
        text[:80], role, len(triggers), [t["id"] for t in triggers],
    )

    for trigger in triggers:
        for phrase in trigger["pattern"]:
            if phrase in tl:
                triggered.append({
                    "trigger_id": trigger["id"],
                    "prompt":     trigger["prompt"],
                    "confidence": trigger["confidence"],
                    "matched":    phrase,
                })
                logger.info(
                    "COACHING MATCH: trigger=%s  pattern=%r  text=%r",
                    trigger["id"], phrase, text[:60],
                )
                break   # fire each trigger at most once per chunk

    if not triggered:
        logger.info("COACHING: no matches in chunk")

    return triggered


# ---------------------------------------------------------------------------
# System prompt — meeting intelligence
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI meeting assistant. "
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
# Key facts extraction — structured facts from transcript
# ---------------------------------------------------------------------------

_KEY_FACTS_PROMPT = (
    "You are a meeting fact extractor. "
    "Read the transcript and extract exactly these 5 key facts as a JSON object:\n"
    '{"decision": "...", "owner": "...", "deadline": "...", "risk": "...", "action": "..."}\n'
    "Rules:\n"
    "- decision: the key decision made or being discussed (one short phrase)\n"
    "- owner: the name of the person assigned responsibility\n"
    "- deadline: any date or timeline mentioned (e.g. 'Friday', 'end of Q2', '2 weeks')\n"
    "- risk: any risk or concern flagged (one short phrase)\n"
    "- action: the next concrete action step mentioned (one short phrase)\n"
    "If a fact is NOT present in the transcript, return null for that field. "
    "Extract words directly from the transcript — do not invent or elaborate. "
    "Return ONLY the JSON object. No markdown fences, no explanation."
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
# Project context injection helper
# ---------------------------------------------------------------------------

def _with_project_context(base_prompt: str, project_context: Optional[str]) -> str:
    """Prepend project context to a system prompt if provided."""
    if not project_context:
        return base_prompt
    return project_context + "\n\n" + base_prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_suggestions(
    transcript_snippet: str,
    model: str = "claude-sonnet-4-6",
    context: Optional[str] = None,
    historical_context: Optional[str] = None,
    project_context: Optional[str] = None,
) -> list[str]:
    """
    Generate 3 AI-tailored response suggestions for a meeting moment.

    Sends the transcript snippet to Claude with the meeting assistant system
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

    transcript_snippet = enforce_budget(transcript_snippet)
    log_usage("transcript", transcript_snippet)
    if context:
        context = enforce_budget(context, max_tokens=CONTEXT_TOKENS)
        log_usage("context", context)

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
            "Relevant context from past meetings — reference this specifically in your "
            "suggestions (be specific, not generic; name actual decisions, owners, risks "
            "from the context below):\n\n"
            + historical_context.strip()
        )
    if context:
        user_parts.append(f"Meeting context: {context.strip()}")
    user_parts.append(f"Recent transcript:\n\n{transcript_snippet.strip()}")
    user_message = "\n\n".join(user_parts)

    logger.info("Requesting AI suggestions from Claude (%s)…", model)

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,          # 3 short suggestions need very few tokens
            system=_with_project_context(_SYSTEM_PROMPT, project_context),
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


def get_key_facts(
    transcript_snippet: str,
    model: str = "claude-sonnet-4-6",
    historical_context: Optional[str] = None,
    project_context: Optional[str] = None,
) -> dict:
    """
    Extract key facts from a meeting transcript snippet.

    Returns a dict with keys: decision, owner, deadline, risk, action.
    Each value is a short phrase extracted directly from the transcript, or None.

    Args:
        transcript_snippet: Recent meeting transcript text.
        model:              Anthropic model ID.
        historical_context: Optional past-meeting context for grounding.

    Raises:
        EnvironmentError: ANTHROPIC_API_KEY not set.
        RuntimeError:     API call failed.
    """
    if not transcript_snippet.strip():
        raise ValueError("transcript_snippet must not be empty.")

    transcript_snippet = enforce_budget(transcript_snippet)
    api_key = _load_api_key()

    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "anthropic package is not installed. Run: pip install anthropic"
        ) from exc

    user_parts = []
    if historical_context:
        user_parts.append("Context from past meetings:\n\n" + historical_context.strip())
    user_parts.append(f"Meeting transcript:\n\n{transcript_snippet.strip()}")
    user_message = "\n\n".join(user_parts)

    logger.info("Requesting AI key facts from Claude (%s)…", model)

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=_with_project_context(_KEY_FACTS_PROMPT, project_context),
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(
            "Anthropic API key is invalid or expired. "
            "Update ANTHROPIC_API_KEY in your .env file."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError("Anthropic rate limit reached. Please wait and try again.") from exc
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

    raw = response.content[0].text.strip()
    fenced = re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
    if fenced:
        raw = fenced.group(1).strip()

    _empty_fact = {"decision": None, "owner": None, "deadline": None, "risk": None, "action": None}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "decision": data.get("decision") or None,
                "owner":    data.get("owner")    or None,
                "deadline": data.get("deadline") or None,
                "risk":     data.get("risk")     or None,
                "action":   data.get("action")   or None,
            }
    except json.JSONDecodeError:
        pass

    logger.error("Could not parse key facts JSON from Claude: %s", raw[:200])
    return _empty_fact


# ---------------------------------------------------------------------------
# Action item extraction — pure local, no API call
# ---------------------------------------------------------------------------

_ACTION_PATTERNS: list[str] = [
    "i'll ",
    "i will ",
    "we need to ",
    "we should ",
    "make sure to ",
    "make sure we ",
    "follow up on ",
    "follow up with ",
    "let's schedule ",
    "action item",
    "todo",
    "to-do",
    "need to ",
    "will send ",
    "will update ",
    "will review ",
    "will prepare ",
    "will create ",
    "will set up ",
    "will share ",
    "can you ",
    "please ",
    "assigned to ",
    "take care of ",
    "responsible for ",
    "before granting ",
    "before the end ",
    "may impact ",
    "needs to complete ",
]


def extract_action_items(transcript_chunks: list[str]) -> list[dict]:
    """
    Extract action items from transcript text using pattern matching.

    Scans each transcript chunk for action-signalling phrases and returns
    a deduplicated list of action items.  Pure string matching — no API call.

    Args:
        transcript_chunks: List of transcript text chunks from the meeting.

    Returns:
        List of dicts with keys: text, pattern, chunk_index.
    """
    items: list[dict] = []
    seen: set[str] = set()

    for i, chunk in enumerate(transcript_chunks):
        cl = chunk.lower().strip()
        if not cl:
            continue
        for pattern in _ACTION_PATTERNS:
            if pattern in cl:
                # Use the original (un-lowered) chunk text
                normalised = chunk.strip()
                if normalised not in seen:
                    seen.add(normalised)
                    items.append({
                        "text":        normalised,
                        "pattern":     pattern.strip(),
                        "chunk_index": i,
                    })
                break  # one match per chunk is enough

    return items
