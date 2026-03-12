"""
tests/test_fact_checker.py
--------------------------
Unit tests for the KB Fact-Check Engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meetingmind.fact_checker import (
    FactCheckResult,
    FactCheckSeverity,
    FactChecker,
    classify_severity,
    compute_variance,
    parse_number,
)


# ---------------------------------------------------------------------------
# TestParseNumber
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_dollar_with_commas(self):
        assert parse_number("$18,500") == 18500.0

    def test_dollar_plain(self):
        assert parse_number("$500") == 500.0

    def test_percent(self):
        assert parse_number("6%") == 6.0

    def test_percent_decimal(self):
        assert parse_number("5.5%") == 5.5

    def test_plain_integer(self):
        assert parse_number("15") == 15.0

    def test_with_commas(self):
        assert parse_number("1,200") == 1200.0

    def test_empty_string(self):
        assert parse_number("") is None

    def test_no_number(self):
        assert parse_number("hello") is None

    def test_euro_with_k(self):
        assert parse_number("€10k") == 10000.0

    def test_dollar_with_k(self):
        assert parse_number("$10k") == 10000.0

    def test_decimal(self):
        assert parse_number("3.14") == 3.14

    def test_million_multiplier(self):
        assert parse_number("$5M") == 5_000_000.0


# ---------------------------------------------------------------------------
# TestComputeVariance
# ---------------------------------------------------------------------------

class TestComputeVariance:
    def test_exact_match(self):
        assert compute_variance(20000.0, 20000.0) == 0.0

    def test_ten_percent_over(self):
        result = compute_variance(22000.0, 20000.0)
        assert abs(result - 10.0) < 0.01

    def test_ten_percent_under(self):
        result = compute_variance(18000.0, 20000.0)
        assert abs(result - 10.0) < 0.01

    def test_stored_zero(self):
        assert compute_variance(100.0, 0.0) == 100.0

    def test_both_zero(self):
        assert compute_variance(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# TestClassifySeverity
# ---------------------------------------------------------------------------

class TestClassifySeverity:
    def test_ok(self):
        assert classify_severity(3.0) == FactCheckSeverity.OK

    def test_notable(self):
        assert classify_severity(10.0) == FactCheckSeverity.NOTABLE

    def test_flag(self):
        assert classify_severity(20.0) == FactCheckSeverity.FLAG

    def test_boundary_5(self):
        assert classify_severity(5.0) == FactCheckSeverity.NOTABLE

    def test_boundary_15(self):
        assert classify_severity(15.0) == FactCheckSeverity.NOTABLE

    def test_just_over_15(self):
        assert classify_severity(15.1) == FactCheckSeverity.FLAG


# ---------------------------------------------------------------------------
# TestFactChecker
# ---------------------------------------------------------------------------

def _mock_collection(docs=None, distances=None, count=None):
    """Create a mock ChromaDB collection."""
    col = MagicMock()
    col.count.return_value = count if count is not None else (len(docs) if docs else 0)

    if docs:
        col.query.return_value = {
            "documents": [docs],
            "distances": [distances] if distances else [[0.5] * len(docs)],
            "ids": [[f"id_{i}" for i in range(len(docs))]],
            "metadatas": [[{}] * len(docs)],
        }
    else:
        col.query.return_value = {"documents": [[]], "distances": [[]], "ids": [[]], "metadatas": [[]]}

    return col


class TestFactChecker:
    def test_no_entities_returns_empty(self):
        col = _mock_collection(["Revenue is $20,000"], [0.3])
        fc = FactChecker(col)
        assert fc.check_chunk("Revenue is $18,500", []) == []

    def test_empty_collection_returns_empty(self):
        col = _mock_collection(count=0)
        fc = FactChecker(col)
        result = fc.check_chunk(
            "Revenue is $18,500",
            [{"text": "$18,500", "label": "MONEY"}],
        )
        assert result == []

    def test_matching_fact(self):
        col = _mock_collection(["Weekly revenue target is $20,000"], [0.3])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Revenue was $18,500 this week",
            [{"text": "$18,500", "label": "MONEY"}],
        )
        assert len(results) == 1
        r = results[0]
        assert r.spoken_value == 18500.0
        assert r.stored_value == 20000.0
        assert abs(r.variance_pct - 7.5) < 0.1
        assert r.severity == FactCheckSeverity.NOTABLE

    def test_exact_match_severity(self):
        col = _mock_collection(["Budget is $20,000"], [0.2])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Budget is $20,000",
            [{"text": "$20,000", "label": "MONEY"}],
        )
        assert len(results) == 1
        assert results[0].severity == FactCheckSeverity.OK
        assert results[0].variance_pct == 0.0

    def test_high_variance_flag(self):
        col = _mock_collection(["Revenue target is $20,000"], [0.4])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Revenue was $10,000",
            [{"text": "$10,000", "label": "MONEY"}],
        )
        assert len(results) == 1
        assert results[0].severity == FactCheckSeverity.FLAG
        assert results[0].variance_pct == 50.0

    def test_distance_above_threshold_filtered(self):
        col = _mock_collection(["Revenue target is $20,000"], [2.0])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Revenue was $18,500",
            [{"text": "$18,500", "label": "MONEY"}],
        )
        assert results == []

    def test_is_available_true(self):
        col = _mock_collection(count=5)
        fc = FactChecker(col)
        assert fc.is_available() is True

    def test_is_available_false(self):
        col = _mock_collection(count=0)
        fc = FactChecker(col)
        assert fc.is_available() is False

    def test_is_available_none(self):
        fc = FactChecker(None)
        assert fc.is_available() is False

    def test_percentage_fact_check(self):
        col = _mock_collection(["ACOS target is 6%"], [0.4])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "ACOS is running at 8%",
            [{"text": "8%", "label": "PERCENT"}],
        )
        assert len(results) == 1
        r = results[0]
        assert r.spoken_value == 8.0
        assert r.stored_value == 6.0
        assert r.severity == FactCheckSeverity.FLAG

    def test_multi_number_doc_picks_relevant_number(self):
        """When KB doc has multiple numbers, extract the one matching context."""
        multi_doc = (
            "Last week revenue was $18,200. "
            "Current ACOS is 4.1%. "
            "Gross margin is 31%. "
            "Budget is $50,000."
        )
        col = _mock_collection([multi_doc], [0.3])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "ACOS is running at 50%",
            [{"text": "50%", "label": "PERCENT"}],
        )
        assert len(results) == 1
        r = results[0]
        assert r.stored_value == 4.1, f"Expected 4.1 (ACOS), got {r.stored_value}"
        assert r.spoken_value == 50.0

    def test_multi_number_doc_revenue_context(self):
        """Revenue context should pick the revenue number, not ACOS."""
        multi_doc = (
            "Current ACOS is 4.1%. "
            "Last week revenue was $18,200. "
            "Budget is $50,000."
        )
        col = _mock_collection([multi_doc], [0.3])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Revenue this week is $21,000",
            [{"text": "$21,000", "label": "MONEY"}],
        )
        assert len(results) == 1
        r = results[0]
        assert r.stored_value == 18200.0, f"Expected 18200 (revenue), got {r.stored_value}"

    def test_multi_number_doc_margin_context(self):
        """Margin context should pick the margin number."""
        multi_doc = (
            "Revenue was $18,200. "
            "ACOS is 4.1%. "
            "Gross margin is 31%."
        )
        col = _mock_collection([multi_doc], [0.3])
        fc = FactChecker(col)
        results = fc.check_chunk(
            "Margin dropped to 28%",
            [{"text": "28%", "label": "PERCENT"}],
        )
        assert len(results) == 1
        r = results[0]
        assert r.stored_value == 31.0, f"Expected 31 (margin), got {r.stored_value}"
