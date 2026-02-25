"""
analyser.py
-----------
LLM-powered analysis of meeting transcripts.

Reads a transcript JSON produced by transcriber.py, sends the text to the
Anthropic Claude API, and writes two human-readable Markdown files:

    outputs/summaries/<stem>_<timestamp>UTC.md
    outputs/action_items/<stem>_<timestamp>UTC.md

NOTE ON PRIVACY
---------------
Unlike transcription (which is fully local), this step sends the transcript
text to the Anthropic API over HTTPS. Do not use this on recordings that
contain legally privileged content, personal health information, or material
that your organisation's data-handling policy prohibits sending to third parties.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_SUMMARIES_DIR: Path    = PROJECT_ROOT / "outputs" / "summaries"
DEFAULT_ACTION_ITEMS_DIR: Path = PROJECT_ROOT / "outputs" / "action_items"

# Default model — Claude Sonnet 4.6 balances quality and speed well for
# meeting analysis. Switch to claude-opus-4-6 for higher quality on complex
# or technical meetings.
DEFAULT_MODEL = "claude-sonnet-4-6"

# The prompt instructs Claude to return a strict JSON object so we can parse
# the result reliably without brittle text scraping.
_SYSTEM_PROMPT = """\
You are an expert meeting analyst. Your job is to read meeting transcripts and
extract structured information. Always respond with a single valid JSON object
— no markdown fences, no commentary outside the JSON.

Return exactly this structure:
{
  "summary": "<3-5 sentence paragraph summarising the meeting>",
  "key_decisions": ["<decision 1>", "<decision 2>"],
  "action_items": [
    {
      "task":  "<clear description of what needs to be done>",
      "owner": "<name or role, or 'Unassigned' if not mentioned>",
      "due":   "<deadline if mentioned, otherwise 'Not specified'>"
    }
  ],
  "topics_discussed": ["<topic 1>", "<topic 2>"],
  "participants_mentioned": ["<name 1>", "<name 2>"],
  "sentiment": "<one word: productive | tense | neutral | inconclusive>"
}

If a field has no data, use an empty list [] or empty string "" as appropriate.
"""

_USER_PROMPT_TEMPLATE = """\
Please analyse the following meeting transcript:

---
{transcript_text}
---
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """
    Resolve the Anthropic API key.

    Search order:
    1. ANTHROPIC_API_KEY environment variable (already set in the process)
    2. .env file in the project root

    Raises:
        EnvironmentError: No API key could be found.
    """
    # Check the live environment first (covers CI, Docker, etc.)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    # Fall back to parsing the .env file directly.
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    return key

    raise EnvironmentError(
        "ANTHROPIC_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )


def _load_transcript(transcript_path: Path) -> dict:
    """
    Read and parse a transcript JSON file produced by transcriber.py.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError:        The file is not valid JSON or is missing 'text'.
    """
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    try:
        with transcript_path.open(encoding="utf-8") as fh:
            record = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse transcript JSON: {exc}") from exc

    if "text" not in record or not record["text"].strip():
        raise ValueError(
            f"Transcript file '{transcript_path.name}' has no 'text' field or it is empty."
        )

    return record


def _call_claude(transcript_text: str, api_key: str, model: str) -> dict:
    """
    Send the transcript to Claude and return the parsed analysis dict.

    The system prompt instructs Claude to respond with a JSON object only,
    but we defensively strip any markdown fences in case they appear.

    Raises:
        ImportError:  anthropic package not installed.
        RuntimeError: API call failed or response could not be parsed as JSON.
    """
    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "anthropic package is not installed. Run:  pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Sending transcript to Claude (%s) for analysis…", model)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        transcript_text=transcript_text
                    ),
                }
            ],
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

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped the JSON anyway.
    # Pattern matches ```json ... ``` or ``` ... ```
    fenced = re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw_text)
    if fenced:
        raw_text = fenced.group(1).strip()

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Claude returned a response that could not be parsed as JSON: {exc}\n"
            f"Raw response (first 500 chars): {raw_text[:500]}"
        ) from exc

    logger.info("Analysis received from Claude.")
    return analysis


def _render_summary_md(
    analysis: dict,
    source_filename: str,
    analysed_at: str,
    model: str,
) -> str:
    """Render the summary section of the analysis as a Markdown document."""
    topics = "\n".join(f"- {t}" for t in analysis.get("topics_discussed", []))
    decisions = "\n".join(f"- {d}" for d in analysis.get("key_decisions", []))
    participants = ", ".join(analysis.get("participants_mentioned", [])) or "Not identified"
    sentiment = analysis.get("sentiment", "—").capitalize()
    summary = analysis.get("summary", "No summary produced.")

    return f"""\
# Meeting Summary

**Source file:** {source_filename}
**Analysed:** {analysed_at}
**Model:** {model}

---

## Overview

{summary}

---

## Topics Discussed

{topics or "- None identified"}

---

## Key Decisions

{decisions or "- None recorded"}

---

## Participants Mentioned

{participants}

---

## Meeting Sentiment

{sentiment}
"""


def _render_action_items_md(
    analysis: dict,
    source_filename: str,
    analysed_at: str,
) -> str:
    """Render the action items as a Markdown document with a table."""
    items = analysis.get("action_items", [])

    if not items:
        table = "_No action items were identified in this meeting._"
    else:
        rows = "\n".join(
            f"| {i + 1} | {item.get('task', '—')} "
            f"| {item.get('owner', 'Unassigned')} "
            f"| {item.get('due', 'Not specified')} |"
            for i, item in enumerate(items)
        )
        table = (
            "| # | Task | Owner | Due |\n"
            "|---|------|-------|-----|\n"
            f"{rows}"
        )

    return f"""\
# Action Items

**Source file:** {source_filename}
**Extracted:** {analysed_at}

---

{table}
"""


def _save_markdown(content: str, directory: Path, stem: str, suffix: str) -> Path:
    """
    Write a Markdown string to a timestamped file.

    Filename:  <stem>_<YYYYMMDD_HHMMSS>UTC_<suffix>.md
    Example:   standup_20260225_103045UTC_summary.md
    """
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = directory / f"{stem}_{timestamp}UTC_{suffix}.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(
    transcript_path: Path | str,
    model: str = DEFAULT_MODEL,
    summaries_dir: Path | str | None = None,
    action_items_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Analyse a transcript JSON file with Claude and save Markdown outputs.

    WARNING: This function sends transcript text to the Anthropic API.
             See the module-level privacy note before use.

    Args:
        transcript_path:  Path to a transcript JSON file from transcriber.py.
        model:            Anthropic model ID. Defaults to claude-sonnet-4-6.
        summaries_dir:    Where to write the summary Markdown.
                          Defaults to <project_root>/outputs/summaries/.
        action_items_dir: Where to write the action items Markdown.
                          Defaults to <project_root>/outputs/action_items/.

    Returns:
        A dict with keys "summary" and "action_items", each mapping to the
        Path of the written Markdown file.

    Raises:
        FileNotFoundError: transcript_path does not exist.
        ValueError:        Transcript JSON is malformed or empty.
        EnvironmentError:  ANTHROPIC_API_KEY not configured.
        RuntimeError:      API call failed or response could not be parsed.
    """
    transcript_path  = Path(transcript_path).resolve()
    summaries_dir    = Path(summaries_dir)    if summaries_dir    else DEFAULT_SUMMARIES_DIR
    action_items_dir = Path(action_items_dir) if action_items_dir else DEFAULT_ACTION_ITEMS_DIR

    # 1. Load the transcript.
    logger.info("Loading transcript: %s", transcript_path.name)
    record = _load_transcript(transcript_path)
    transcript_text  = record["text"]
    source_filename  = record.get("meta", {}).get("source_file", transcript_path.name)

    # 2. Resolve the API key from .env or environment.
    api_key = _load_api_key()

    # 3. Send to Claude and get structured analysis.
    analysis = _call_claude(transcript_text, api_key, model)

    # 4. Render and save outputs.
    analysed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stem        = transcript_path.stem  # e.g. "standup_20260225_103045UTC"

    summary_md = _render_summary_md(analysis, source_filename, analysed_at, model)
    actions_md = _render_action_items_md(analysis, source_filename, analysed_at)

    summary_path = _save_markdown(summary_md, summaries_dir, stem, "summary")
    actions_path = _save_markdown(actions_md, action_items_dir, stem, "action_items")

    logger.info("Summary saved      → %s", summary_path)
    logger.info("Action items saved → %s", actions_path)

    return {
        "summary":      summary_path,
        "action_items": actions_path,
    }
