"""
tests/test_spacy_extractor.py
------------------------------
Unit tests for spacy_extractor.py — NER extraction and Key Facts enrichment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Skip entire module if spaCy or model not available
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from meetingmind.spacy_extractor import (
    extract_actions,
    extract_entities,
    extract_numeric_entities,
    enrich_key_facts,
    detect_obligations,
    detect_risks,
    detect_decisions,
    nlp,
)


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_returns_dict_keys(self):
        result = extract_entities("Hello world")
        assert set(result.keys()) == {"people", "dates", "orgs", "money", "actions"}

    def test_finds_person(self):
        result = extract_entities("John Smith will handle the report.")
        assert any("John" in p for p in result["people"])

    def test_finds_date(self):
        result = extract_entities("The deadline is March 15th.")
        assert len(result["dates"]) >= 1

    def test_finds_org(self):
        result = extract_entities("We signed a deal with Microsoft yesterday.")
        assert any("Microsoft" in o for o in result["orgs"])

    def test_finds_money(self):
        result = extract_entities("The budget is $50,000 for this quarter.")
        assert len(result["money"]) >= 1

    def test_empty_text(self):
        result = extract_entities("")
        assert result["people"] == []
        assert result["dates"] == []
        assert result["actions"] == []


# ---------------------------------------------------------------------------
# extract_actions
# ---------------------------------------------------------------------------

class TestExtractActions:
    def test_returns_list(self):
        doc = nlp("Sarah reviews the contract.")
        actions = extract_actions(doc)
        assert isinstance(actions, list)

    def test_action_has_keys(self):
        doc = nlp("Sarah reviews the contract.")
        actions = extract_actions(doc)
        if actions:
            assert "verb" in actions[0]
            assert "subject" in actions[0]
            assert "object" in actions[0]


# ---------------------------------------------------------------------------
# enrich_key_facts
# ---------------------------------------------------------------------------

class TestEnrichKeyFacts:
    def test_fills_owner_from_person(self):
        facts = enrich_key_facts("John will complete the review by Friday.")
        assert facts["owner"] is not None

    def test_fills_deadline_from_date(self):
        facts = enrich_key_facts("We need this done by March 15th.")
        assert facts["deadline"] is not None

    def test_fills_risk_from_money(self):
        facts = enrich_key_facts("The cost is $100,000 which exceeds our budget.")
        assert facts["risk"] is not None
        assert "Budget:" in facts["risk"]

    def test_does_not_overwrite_existing_owner(self):
        existing = {"decision": None, "owner": "Alice", "deadline": None, "risk": None, "action": None}
        facts = enrich_key_facts("John will handle it.", existing)
        assert facts["owner"] == "Alice"

    def test_does_not_overwrite_existing_deadline(self):
        existing = {"decision": None, "owner": None, "deadline": "Friday", "risk": None, "action": None}
        facts = enrich_key_facts("We need this by March 15th.", existing)
        assert facts["deadline"] == "Friday"

    def test_empty_text_returns_defaults(self):
        facts = enrich_key_facts("")
        assert facts["owner"] is None
        assert facts["deadline"] is None

    def test_none_current_facts(self):
        facts = enrich_key_facts("Test text", None)
        assert isinstance(facts, dict)
        assert "owner" in facts


# ---------------------------------------------------------------------------
# Error boundary — spaCy should never crash the caller
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# detect_obligations
# ---------------------------------------------------------------------------

class TestDetectObligations:
    def test_returns_list(self):
        assert isinstance(detect_obligations("Hello world"), list)

    def test_ill_send(self):
        result = detect_obligations("I'll send the report by Friday.")
        assert len(result) >= 1
        assert result[0]["subject"] == "I"

    def test_will_prepare(self):
        result = detect_obligations("John will prepare the presentation.")
        assert len(result) >= 1

    def test_we_need_to(self):
        result = detect_obligations("We need to review the contract.")
        assert len(result) >= 1

    def test_no_obligation_in_generic(self):
        result = detect_obligations("The weather is nice today.")
        assert result == []

    def test_empty_text(self):
        assert detect_obligations("") == []

    def test_returns_required_keys(self):
        result = detect_obligations("I'll handle the client call.")
        if result:
            for item in result:
                assert "text" in item
                assert "subject" in item
                assert "verb" in item
                assert "object" in item


# ---------------------------------------------------------------------------
# detect_risks
# ---------------------------------------------------------------------------

class TestDetectRisks:
    def test_returns_list(self):
        assert isinstance(detect_risks("Hello world"), list)

    def test_concerned(self):
        result = detect_risks("I'm concerned about the budget.")
        assert len(result) >= 1

    def test_worried(self):
        result = detect_risks("The team is worried about the deadline.")
        assert len(result) >= 1

    def test_risk_noun(self):
        result = detect_risks("There is a risk of delay.")
        assert len(result) >= 1

    def test_blocker(self):
        result = detect_risks("We have a blocker on the deployment.")
        assert len(result) >= 1

    def test_no_risk_in_generic(self):
        result = detect_risks("The meeting went well.")
        assert result == []

    def test_empty_text(self):
        assert detect_risks("") == []

    def test_returns_required_keys(self):
        result = detect_risks("I'm concerned about timelines.")
        if result:
            for item in result:
                assert "text" in item
                assert "signal" in item
                assert "detail" in item


# ---------------------------------------------------------------------------
# detect_decisions
# ---------------------------------------------------------------------------

class TestDetectDecisions:
    def test_returns_list(self):
        assert isinstance(detect_decisions("Hello world"), list)

    def test_decided(self):
        result = detect_decisions("We decided to go with plan A.")
        assert len(result) >= 1

    def test_agreed(self):
        result = detect_decisions("The team agreed on the timeline.")
        assert len(result) >= 1

    def test_approved(self):
        result = detect_decisions("The budget was approved by management.")
        assert len(result) >= 1

    def test_decision_noun(self):
        result = detect_decisions("The decision was to proceed.")
        assert len(result) >= 1

    def test_no_decision_in_generic(self):
        result = detect_decisions("We had lunch at noon.")
        assert result == []

    def test_empty_text(self):
        assert detect_decisions("") == []

    def test_returns_required_keys(self):
        result = detect_decisions("We agreed on the approach.")
        if result:
            for item in result:
                assert "text" in item
                assert "signal" in item
                assert "detail" in item


class TestErrorBoundary:
    def test_extract_entities_survives_nlp_crash(self):
        """If spaCy's nlp() throws, extract_entities returns empty result."""
        import meetingmind.spacy_extractor as mod
        original = mod.nlp
        try:
            mod.nlp = lambda text: (_ for _ in ()).throw(RuntimeError("boom"))
            result = extract_entities("test")
            assert result == {"people": [], "dates": [], "orgs": [], "money": [], "actions": []}
        finally:
            mod.nlp = original

    def test_enrich_key_facts_survives_extract_crash(self):
        """If extract_entities throws, enrich_key_facts returns current facts."""
        import meetingmind.spacy_extractor as mod
        original = mod.nlp
        try:
            mod.nlp = lambda text: (_ for _ in ()).throw(RuntimeError("boom"))
            facts = enrich_key_facts("test", {"decision": "X", "owner": None, "deadline": None, "risk": None, "action": None})
            assert facts["decision"] == "X"
        finally:
            mod.nlp = original


# ---------------------------------------------------------------------------
# extract_numeric_entities
# ---------------------------------------------------------------------------

class TestExtractNumericEntities:
    def test_returns_list(self):
        result = extract_numeric_entities("Hello world")
        assert isinstance(result, list)

    def test_finds_money(self):
        result = extract_numeric_entities("The budget is $50,000 for this quarter.")
        labels = [e["label"] for e in result]
        assert "MONEY" in labels or "CARDINAL" in labels
        assert any("50" in e["text"] for e in result)

    def test_finds_percentage_via_regex(self):
        result = extract_numeric_entities("ACOS is running at 6.5% this week.")
        texts = [e["text"] for e in result]
        assert any("%" in t for t in texts)

    def test_empty_text_returns_empty(self):
        assert extract_numeric_entities("") == []

    def test_finds_cardinal(self):
        result = extract_numeric_entities("We sold 1,200 units last week.")
        assert len(result) >= 1
