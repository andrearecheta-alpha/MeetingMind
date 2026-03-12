"""
tests/test_coaching.py
-----------------------
Unit tests for COACHING_TRIGGERS and detect_coaching() in suggestion_engine.

Pure-Python — no API calls, no hardware, no mocks required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meetingmind.suggestion_engine import (
    COACHING_TRIGGERS,
    EA_COACHING_TRIGGERS,
    PM_COACHING_TRIGGERS,
    check_time_triggers,
    detect_coaching,
    get_coaching_triggers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(results: list[dict]) -> list[str]:
    return [r["trigger_id"] for r in results]


def _one(text: str) -> list[dict]:
    return detect_coaching(text)


# ---------------------------------------------------------------------------
# Pattern-based triggers — positive cases
# ---------------------------------------------------------------------------

class TestNoDecisionOwner:
    def test_someone_should(self):
        r = _one("Someone should handle the client comms.")
        assert "no_decision_owner" in _ids(r)

    def test_we_should(self):
        r = _one("We should revisit the sprint goals.")
        assert "no_decision_owner" in _ids(r)

    def test_case_insensitive(self):
        r = _one("WE SHOULD consider hiring more staff.")
        assert "no_decision_owner" in _ids(r)

    def test_matched_phrase_returned(self):
        r = _one("Someone should own this deliverable.")
        hit = next(x for x in r if x["trigger_id"] == "no_decision_owner")
        assert hit["matched"] in ("someone should", "we should")


class TestResistanceDetected:
    def test_not_sure(self):
        r = _one("I'm not sure this will work in time.")
        assert "resistance_detected" in _ids(r)

    def test_concerned(self):
        r = _one("I'm concerned about the budget overrun.")
        assert "resistance_detected" in _ids(r)

    def test_worried(self):
        r = _one("The team is worried about the deadline.")
        assert "resistance_detected" in _ids(r)

    def test_but(self):
        r = _one("Great idea, but we might not have capacity.")
        assert "resistance_detected" in _ids(r)

    def test_confidence_value(self):
        r = _one("I'm concerned.")
        hit = next(x for x in r if x["trigger_id"] == "resistance_detected")
        assert hit["confidence"] == pytest.approx(0.80)


class TestScopeCreep:
    def test_also_add(self):
        r = _one("Also add the reporting dashboard to this sprint.")
        assert "scope_creep" in _ids(r)

    def test_while_were_at_it(self):
        r = _one("While we're at it, let's redo the login screen.")
        assert "scope_creep" in _ids(r)

    def test_one_more_thing(self):
        r = _one("One more thing — can we include the mobile app?")
        assert "scope_creep" in _ids(r)

    def test_confidence_value(self):
        r = _one("Also add the export feature.")
        hit = next(x for x in r if x["trigger_id"] == "scope_creep")
        assert hit["confidence"] == pytest.approx(0.90)


class TestNoTimeline:
    def test_lets_decide_later(self):
        r = _one("Let's decide later on the release date.")
        assert "no_timeline" in _ids(r)

    def test_tbd_uppercase_in_speech(self):
        # patterns are stored lowercase; matching is done on lowercased text
        r = _one("The delivery date is TBD for now.")
        assert "no_timeline" in _ids(r)

    def test_tbd_lowercase(self):
        r = _one("launch date: tbd")
        assert "no_timeline" in _ids(r)

    def test_to_be_determined(self):
        r = _one("That milestone is to be determined.")
        assert "no_timeline" in _ids(r)

    def test_confidence_value(self):
        r = _one("Launch date is TBD.")
        hit = next(x for x in r if x["trigger_id"] == "no_timeline")
        assert hit["confidence"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# No triggers fired — ordinary text
# ---------------------------------------------------------------------------

class TestNoTriggers:
    def test_empty_string(self):
        assert _one("") == []

    def test_generic_update(self):
        assert _one("The team completed the sprint retrospective.") == []

    def test_whitespace_only(self):
        assert _one("   ") == []

    def test_question_without_trigger(self):
        assert _one("What are the next steps?") == []

    def test_greeting_no_trigger(self):
        assert _one("Good morning everyone.") == []


# ---------------------------------------------------------------------------
# Multiple triggers in one chunk
# ---------------------------------------------------------------------------

class TestMultipleTriggers:
    def test_two_triggers_at_once(self):
        # "someone should" (no_decision_owner) + "concerned" (resistance_detected)
        r = _one("Someone should own this, I'm concerned about timelines.")
        ids = _ids(r)
        assert "no_decision_owner"   in ids
        assert "resistance_detected" in ids

    def test_each_trigger_fires_at_most_once(self):
        # "not sure" + "worried" — both match resistance_detected but it fires once
        r = _one("I'm not sure and also worried about this.")
        ids = _ids(r)
        assert ids.count("resistance_detected") == 1

    def test_two_text_triggers_at_once_v2(self):
        r = _one("Someone should handle this, it's TBD.")
        ids = _ids(r)
        assert "no_decision_owner" in ids


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_list(self):
        assert isinstance(_one("anything"), list)

    def test_each_item_has_required_keys(self):
        r = _one("I'm concerned about this.")
        for item in r:
            assert "trigger_id"  in item
            assert "prompt"      in item
            assert "confidence"  in item
            assert "matched"     in item

    def test_confidence_is_float(self):
        r = _one("Someone should decide.")
        for item in r:
            assert isinstance(item["confidence"], float)

    def test_trigger_id_is_known(self):
        known_ids = {t["id"] for t in COACHING_TRIGGERS}
        r = _one("Someone should decide.")
        for item in r:
            assert item["trigger_id"] in known_ids


# ---------------------------------------------------------------------------
# COACHING_TRIGGERS contract
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# check_time_triggers
# ---------------------------------------------------------------------------

class TestCheckTimeTriggers:
    def test_no_trigger_before_20_min(self):
        # 19 min 59 s — below threshold
        result = check_time_triggers(elapsed_seconds=1199, decisions_count=0)
        assert result == []

    def test_trigger_fires_at_exactly_20_min(self):
        result = check_time_triggers(elapsed_seconds=1200, decisions_count=0)
        assert len(result) == 1
        assert result[0]["trigger_id"] == "no_decision_20min"

    def test_trigger_fires_after_20_min(self):
        result = check_time_triggers(elapsed_seconds=2400, decisions_count=0)
        ids = [r["trigger_id"] for r in result]
        assert "no_decision_20min" in ids

    def test_no_trigger_when_decisions_exist(self):
        # 30 min but decisions present — should not trigger
        result = check_time_triggers(elapsed_seconds=1800, decisions_count=1)
        assert result == []

    def test_no_trigger_when_many_decisions(self):
        result = check_time_triggers(elapsed_seconds=3600, decisions_count=5)
        assert result == []

    def test_trigger_prompt_text(self):
        result = check_time_triggers(elapsed_seconds=1200, decisions_count=0)
        assert "20 min" in result[0]["prompt"]
        assert "no decisions" in result[0]["prompt"].lower()

    def test_trigger_confidence_is_one(self):
        result = check_time_triggers(elapsed_seconds=1200, decisions_count=0)
        assert result[0]["confidence"] == pytest.approx(1.0)

    def test_trigger_matched_is_none(self):
        result = check_time_triggers(elapsed_seconds=1200, decisions_count=0)
        assert result[0]["matched"] is None

    def test_return_shape(self):
        result = check_time_triggers(elapsed_seconds=1200, decisions_count=0)
        for item in result:
            assert "trigger_id"  in item
            assert "prompt"      in item
            assert "confidence"  in item
            assert "matched"     in item

    def test_zero_elapsed_no_trigger(self):
        assert check_time_triggers(0, 0) == []


class TestCoachingTriggersContract:
    def test_is_list(self):
        assert isinstance(COACHING_TRIGGERS, list)

    def test_not_empty(self):
        assert len(COACHING_TRIGGERS) >= 1

    def test_all_have_required_keys(self):
        for t in COACHING_TRIGGERS:
            assert "id"         in t
            assert "pattern"    in t
            assert "prompt"     in t
            assert "confidence" in t

    def test_ids_are_unique(self):
        ids = [t["id"] for t in COACHING_TRIGGERS]
        assert len(ids) == len(set(ids))

    def test_patterns_are_lowercase(self):
        for t in COACHING_TRIGGERS:
            for p in t["pattern"]:
                assert p == p.lower(), f"Pattern not lowercase: {p!r} in trigger {t['id']!r}"

    def test_confidence_in_valid_range(self):
        for t in COACHING_TRIGGERS:
            assert 0.0 <= t["confidence"] <= 1.0, (
                f"confidence={t['confidence']} out of range for {t['id']!r}"
            )

    def test_no_timeline_exists(self):
        tr = next(t for t in COACHING_TRIGGERS if t["id"] == "no_timeline")
        assert len(tr["pattern"]) >= 1


# ---------------------------------------------------------------------------
# EA coaching triggers
# ---------------------------------------------------------------------------

class TestEACoachingTriggers:
    """Verify EA-specific coaching triggers fire correctly."""

    def test_ea_triggers_is_list(self):
        assert isinstance(EA_COACHING_TRIGGERS, list)
        assert len(EA_COACHING_TRIGGERS) >= 5

    def test_no_followup_owner(self):
        results = detect_coaching("someone will handle that", role="EA")
        ids = _ids(results)
        assert "no_followup_owner" in ids

    def test_exec_preference(self):
        results = detect_coaching("she wants the report by Friday", role="EA")
        ids = _ids(results)
        assert "exec_preference" in ids

    def test_protocol_deviation(self):
        results = detect_coaching("let's skip approval this time", role="EA")
        ids = _ids(results)
        assert "protocol_deviation" in ids

    def test_commitment_without_auth(self):
        results = detect_coaching("I'll make sure the package arrives", role="EA")
        ids = _ids(results)
        assert "commitment_without_auth" in ids

    def test_sensitive_topic(self):
        results = detect_coaching("this is confidential information", role="EA")
        ids = _ids(results)
        assert "sensitive_topic" in ids

    def test_ea_no_fire_on_clean_text(self):
        results = detect_coaching("The weather is nice today", role="EA")
        assert len(results) == 0

    def test_pm_triggers_not_in_ea(self):
        """EA role should use EA triggers, not PM triggers."""
        results = detect_coaching("also add a new feature while we're at it", role="EA")
        ids = _ids(results)
        assert "scope_creep" not in ids


# ---------------------------------------------------------------------------
# Role-based trigger selection
# ---------------------------------------------------------------------------

class TestGetCoachingTriggers:
    """Verify get_coaching_triggers returns the correct list per role."""

    def test_pm_returns_pm_triggers(self):
        assert get_coaching_triggers("PM") is PM_COACHING_TRIGGERS

    def test_ea_returns_ea_triggers(self):
        assert get_coaching_triggers("EA") is EA_COACHING_TRIGGERS

    def test_sales_returns_triggers(self):
        assert get_coaching_triggers("Sales") is not None

    def test_custom_returns_pm_triggers(self):
        assert get_coaching_triggers("Custom") is PM_COACHING_TRIGGERS

    def test_unknown_role_returns_pm(self):
        assert get_coaching_triggers("Unknown") is PM_COACHING_TRIGGERS

    def test_coaching_triggers_alias(self):
        """COACHING_TRIGGERS should be aliased to PM_COACHING_TRIGGERS."""
        assert COACHING_TRIGGERS is PM_COACHING_TRIGGERS
