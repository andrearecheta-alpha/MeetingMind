"""
tests/test_decision_detection.py
---------------------------------
Unit tests for detect_decision() in suggestion_engine.

These tests are pure-Python — no API calls, no hardware, no mocks needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meetingmind.suggestion_engine import DECISION_PATTERNS, detect_decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(text: str) -> dict:
    return detect_decision(text)


# ---------------------------------------------------------------------------
# Positive cases — each DECISION_PATTERN phrase
# ---------------------------------------------------------------------------

class TestDecisionDetected:
    """Every phrase in DECISION_PATTERNS should trigger is_decision=True."""

    @pytest.mark.parametrize("phrase", DECISION_PATTERNS)
    def test_exact_phrase_match(self, phrase: str):
        result = _result(f"So {phrase} the new timeline.")
        assert result["is_decision"] is True
        assert result["confidence"] == "high"
        assert result["phrase"] == phrase

    def test_case_insensitive(self):
        result = _result("We ALL AGREED on the sprint scope.")
        assert result["is_decision"] is True
        assert result["phrase"] == "agreed"

    def test_phrase_in_longer_sentence(self):
        result = _result(
            "After reviewing the options the decision is to use vendor B."
        )
        assert result["is_decision"] is True
        assert result["phrase"] == "decision is"

    def test_returns_first_matched_phrase(self):
        # Both "confirmed" and "agreed" appear; first in DECISION_PATTERNS wins.
        text = "We agreed and confirmed the budget."
        result = _result(text)
        assert result["is_decision"] is True
        # "confirmed" comes before "agreed" in DECISION_PATTERNS
        assert result["phrase"] == "confirmed"

    def test_mixed_case_phrase(self):
        result = _result("Going Forward we will deploy weekly.")
        assert result["is_decision"] is True

    def test_approved(self):
        result = _result("The budget was approved by the board.")
        assert result["is_decision"] is True
        assert result["phrase"] == "approved"

    def test_lets_go_with(self):
        result = _result("Let's go with the React approach.")
        assert result["is_decision"] is True
        assert result["phrase"] == "let's go with"

    def test_well_proceed(self):
        result = _result("OK so we'll proceed with option two.")
        assert result["is_decision"] is True
        assert result["phrase"] == "we'll proceed"

    def test_weve_decided(self):
        result = _result("We've decided to freeze scope for this sprint.")
        assert result["is_decision"] is True
        assert result["phrase"] == "we've decided"


# ---------------------------------------------------------------------------
# Negative cases — ordinary text that must NOT trigger
# ---------------------------------------------------------------------------

class TestNoDecisionDetected:
    """Normal transcript text should return is_decision=False."""

    def test_empty_string(self):
        result = _result("")
        assert result["is_decision"] is False
        assert result["confidence"] == "none"
        assert result["phrase"] is None

    def test_question_only(self):
        result = _result("What do you think about the timeline?")
        assert result["is_decision"] is False

    def test_generic_update(self):
        result = _result("The team is on track and progress looks good.")
        assert result["is_decision"] is False

    def test_partial_word_no_match(self):
        # "going" alone (without "going forward") must not match
        result = _result("The project is going well.")
        assert result["is_decision"] is False

    def test_whitespace_only(self):
        result = _result("   \t\n  ")
        assert result["is_decision"] is False

    def test_numbers_only(self):
        result = _result("123 456 789")
        assert result["is_decision"] is False


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

class TestReturnShape:
    """detect_decision must always return a dict with the three required keys."""

    @pytest.mark.parametrize("text", [
        "",
        "Hello world",
        "We agreed on this.",
    ])
    def test_keys_present(self, text: str):
        result = _result(text)
        assert set(result.keys()) == {"is_decision", "confidence", "phrase"}

    def test_is_decision_is_bool(self):
        assert isinstance(_result("agreed")["is_decision"], bool)
        assert isinstance(_result("hello")["is_decision"], bool)

    def test_confidence_values(self):
        assert _result("agreed")["confidence"] == "high"
        assert _result("hello")["confidence"] == "none"

    def test_phrase_is_none_when_no_match(self):
        assert _result("nothing here")["phrase"] is None

    def test_phrase_is_string_when_match(self):
        assert isinstance(_result("agreed")["phrase"], str)


# ---------------------------------------------------------------------------
# DECISION_PATTERNS contract
# ---------------------------------------------------------------------------

class TestDecisionPatterns:
    """Sanity checks on the pattern list itself."""

    def test_patterns_is_list(self):
        assert isinstance(DECISION_PATTERNS, list)

    def test_patterns_not_empty(self):
        assert len(DECISION_PATTERNS) >= 1

    def test_all_patterns_are_lowercase_strings(self):
        for p in DECISION_PATTERNS:
            assert isinstance(p, str)
            assert p == p.lower(), f"Pattern not lowercase: {p!r}"

    def test_no_duplicate_patterns(self):
        assert len(DECISION_PATTERNS) == len(set(DECISION_PATTERNS))
