"""
cli.py
------
Command-line interface for MeetingMind.

Entry point:
    python -m meetingmind <command> [options]

Available commands:
    transcribe   Convert an audio file to a timestamped JSON transcript.
                 Processing is 100% local — no network required after the
                 one-time model download.
    analyse      Send a transcript JSON to Claude and produce a summary
                 and action-items Markdown file in outputs/.

Run `python -m meetingmind --help` or `python -m meetingmind <command> --help`
for full option details.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meetingmind import __version__
from meetingmind.analyser import DEFAULT_ACTION_ITEMS_DIR, DEFAULT_MODEL, DEFAULT_SUMMARIES_DIR, analyse
from meetingmind.transcriber import DEFAULT_TRANSCRIPTS_DIR, transcribe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Set up a simple console log handler."""
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meetingmind",
        description=(
            "MeetingMind — local meeting transcription and AI knowledge base.\n"
            "All Whisper processing happens on this machine; nothing is sent to the cloud."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"MeetingMind {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging."
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ── transcribe ────────────────────────────────────────────────────────────
    t = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file using local Whisper (no internet required).",
        description=(
            "Transcribe an audio file and save a timestamped JSON transcript.\n\n"
            "PRIVACY: audio is processed entirely on this device by a locally\n"
            "         stored Whisper model. No data leaves this machine.\n\n"
            "Supported formats: mp3, wav, m4a, ogg, flac, aac, mp4, webm, mkv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    t.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file (e.g. audio/standup.mp3).",
    )
    t.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        metavar="SIZE",
        help=(
            "Whisper model size — larger models are more accurate but slower.\n"
            "  tiny   ~39 M params  fastest\n"
            "  base   ~74 M params  recommended default\n"
            "  small  ~244 M params better accuracy\n"
            "  medium ~769 M params high accuracy\n"
            "  large  ~1.5 B params best accuracy, slowest\n"
            "(default: base)"
        ),
    )
    t.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help=(
            "ISO-639-1 language code of the spoken language, e.g. 'en', 'fr', 'de'.\n"
            "Omit to let Whisper auto-detect the language (adds a few seconds)."
        ),
    )
    t.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        metavar="DIR",
        help=f"Directory where transcript JSON files are saved. (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )

    # ── analyse ───────────────────────────────────────────────────────────────
    a = subparsers.add_parser(
        "analyse",
        help="Analyse a transcript with Claude — summary + action items.",
        description=(
            "Send a transcript JSON to the Anthropic Claude API and produce:\n"
            "  • outputs/summaries/<name>_<timestamp>UTC_summary.md\n"
            "  • outputs/action_items/<name>_<timestamp>UTC_action_items.md\n\n"
            "PRIVACY: transcript text is sent to the Anthropic API over HTTPS.\n"
            "         Do not use on legally privileged or sensitive recordings\n"
            "         without appropriate authorisation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    a.add_argument(
        "transcript_file",
        type=Path,
        help="Path to a transcript JSON file (e.g. transcripts/meeting_20260225_103045UTC.json).",
    )
    a.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Anthropic model ID to use for analysis. (default: {DEFAULT_MODEL})",
    )
    a.add_argument(
        "--summaries-dir",
        type=Path,
        default=DEFAULT_SUMMARIES_DIR,
        metavar="DIR",
        help=f"Directory for summary Markdown files. (default: {DEFAULT_SUMMARIES_DIR})",
    )
    a.add_argument(
        "--action-items-dir",
        type=Path,
        default=DEFAULT_ACTION_ITEMS_DIR,
        metavar="DIR",
        help=f"Directory for action-items Markdown files. (default: {DEFAULT_ACTION_ITEMS_DIR})",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_transcribe(args: argparse.Namespace) -> int:
    """
    Handle the 'transcribe' subcommand.

    Prints a privacy notice, runs the transcriber, and reports the result.
    Returns a Unix exit code (0 = success, non-zero = failure).
    """
    _print_privacy_banner()

    try:
        output_path = transcribe(
            audio_path=args.audio_file,
            model_size=args.model,
            transcripts_dir=args.output_dir,
            language=args.language,
        )
    except FileNotFoundError as exc:
        _error(str(exc), hint="Check that the audio file path is correct.")
        return 1
    except ValueError as exc:
        _error(str(exc))
        return 1
    except ImportError as exc:
        _error(
            str(exc),
            hint=(
                "Install dependencies with:\n"
                "    pip install openai-whisper\n"
                "and ensure ffmpeg is installed and on your PATH."
            ),
        )
        return 1
    except RuntimeError as exc:
        _error(str(exc), hint="Is the audio file corrupted or in an unexpected format?")
        return 1

    print(f"\n  Transcript saved to:\n    {output_path}\n")
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    """
    Handle the 'analyse' subcommand.

    Sends a transcript to Claude and writes summary + action-item Markdown files.
    Returns a Unix exit code (0 = success, non-zero = failure).
    """
    print(
        "\n  NOTE: Transcript text will be sent to the Anthropic API over HTTPS.\n"
        "  Ensure you have the right to process this recording externally.\n"
    )

    try:
        output_paths = analyse(
            transcript_path=args.transcript_file,
            model=args.model,
            summaries_dir=args.summaries_dir,
            action_items_dir=args.action_items_dir,
        )
    except FileNotFoundError as exc:
        _error(str(exc), hint="Check that the transcript file path is correct.")
        return 1
    except ValueError as exc:
        _error(str(exc))
        return 1
    except EnvironmentError as exc:
        _error(str(exc), hint="Add ANTHROPIC_API_KEY=<your-key> to your .env file.")
        return 1
    except ImportError as exc:
        _error(str(exc), hint="Run:  pip install anthropic")
        return 1
    except RuntimeError as exc:
        _error(str(exc))
        return 1

    print(  "  Analysis complete. Files written:\n")
    print(f"    Summary      →  {output_paths['summary']}")
    print(f"    Action items →  {output_paths['action_items']}\n")
    return 0


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def _print_privacy_banner() -> None:
    """Print a brief privacy reminder before any processing begins."""
    border = "─" * 62
    print(f"\n  ┌{border}┐")
    print(  "  │  PRIVACY                                               │")
    print(  "  │  Audio is processed entirely on this device by a      │")
    print(  "  │  local Whisper model. No audio, transcript, or         │")
    print(  "  │  metadata is sent to any external server or API.       │")
    print(f"  └{border}┘\n")


def _error(message: str, hint: str | None = None) -> None:
    """Print a formatted error message (and optional hint) to stderr."""
    print(f"\n  ERROR: {message}", file=sys.stderr)
    if hint:
        print(f"  HINT:  {hint}", file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_logging(getattr(args, "verbose", False))

    if args.command == "transcribe":
        sys.exit(_cmd_transcribe(args))
    elif args.command == "analyse":
        sys.exit(_cmd_analyse(args))
    else:
        # No subcommand given — print help and exit cleanly.
        parser.print_help()
        sys.exit(0)
