"""
spacy_extractor.py
------------------
Named Entity Recognition and action extraction using spaCy.

Extracts people, dates, organisations, money amounts, and verb-based actions
from meeting transcript text.  Used to auto-populate Key Facts fields.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load spaCy model ONCE at module level — never reload per chunk.
# ---------------------------------------------------------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy en_core_web_sm loaded at module level.")
except (ImportError, OSError) as exc:
    logger.warning("spaCy not available: %s", exc)
    nlp = None


_EMPTY_ENTITIES: dict = {"people": [], "dates": [], "orgs": [], "money": [], "actions": []}


def extract_entities(text: str) -> dict:
    """
    Extract named entities and actions from text.

    Returns:
        dict with keys: people, dates, orgs, money, actions
    """
    if nlp is None or not text.strip():
        return dict(_EMPTY_ENTITIES)

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error("spaCy NLP error: %s", exc)
        return dict(_EMPTY_ENTITIES)

    # Debug logging — shows exactly what spaCy detects
    if doc.ents:
        logger.info("spaCy found: %s", [(e.text, e.label_) for e in doc.ents])

    return {
        "people": [e.text for e in doc.ents if e.label_ == "PERSON"],
        "dates":  [e.text for e in doc.ents if e.label_ == "DATE"],
        "orgs":   [e.text for e in doc.ents if e.label_ == "ORG"],
        "money":  [e.text for e in doc.ents if e.label_ == "MONEY"],
        "actions": extract_actions(doc),
    }


_NUMERIC_LABELS = {"MONEY", "PERCENT", "CARDINAL", "QUANTITY"}
_PCT_RE = re.compile(r"\d+(?:\.\d+)?%")


def extract_numeric_entities(text: str) -> list[dict]:
    """
    Extract numeric entities (MONEY, PERCENT, CARDINAL, QUANTITY) from text.

    Uses spaCy NER with a regex fallback for percentage patterns that
    ``en_core_web_sm`` often labels as CARDINAL instead of PERCENT.

    Returns:
        List of dicts with keys: text, label.
        E.g. [{"text": "$18,500", "label": "MONEY"}, {"text": "6%", "label": "PERCENT"}]
    """
    if nlp is None or not text.strip():
        return []

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error("spaCy extract_numeric_entities error: %s", exc)
        return []

    results: list[dict] = []
    seen_spans: set[str] = set()

    # spaCy entities
    for ent in doc.ents:
        if ent.label_ in _NUMERIC_LABELS:
            # Skip bare small cardinals with no currency/percent context
            if ent.label_ == "CARDINAL":
                try:
                    val = float(re.sub(r'[,$%\s]', '', ent.text))
                    start = max(0, ent.start - 3)
                    end = min(len(doc), ent.end + 3)
                    context = doc[start:end].text.lower()
                    has_marker = any(
                        m in context
                        for m in ['$', '%', 'k', 'K', 'M',
                                  'million', 'billion',
                                  'percent', 'budget',
                                  'revenue', 'cost', 'acos']
                    )
                    if val < 100 and not has_marker:
                        continue
                except (ValueError, AttributeError):
                    pass
            results.append({"text": ent.text, "label": ent.label_})
            seen_spans.add(ent.text)

    # Regex fallback for percentages missed by spaCy
    for m in _PCT_RE.finditer(text):
        if m.group() not in seen_spans:
            results.append({"text": m.group(), "label": "PERCENT"})
            seen_spans.add(m.group())

    return results


def extract_actions(doc) -> list[dict]:
    """
    Extract subject-verb-object triples from the parsed document.

    Returns a list of dicts with keys: verb, subject, object.
    """
    actions = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subjects = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects  = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]
            if subjects:
                actions.append({
                    "verb":    token.text,
                    "subject": subjects[0],
                    "object":  objects[0] if objects else "",
                })
    return actions


# ---------------------------------------------------------------------------
# Obligation detection — dependency parsing for "I'll", "will", "need to"
# ---------------------------------------------------------------------------

_OBLIGATION_LEMMAS = {"will", "shall", "must", "need", "go"}
_OBLIGATION_AUX = {"'ll", "ll"}
_OBLIGATION_ROOT_LEMMAS = {"need", "must"}


def detect_obligations(text: str) -> list[dict]:
    """
    Detect obligation/commitment phrases using spaCy dependency parsing.

    Looks for subject + modal-verb patterns like "I'll send", "we need to review",
    "John will prepare". Returns a list of obligation dicts.

    Returns:
        List of dicts with keys: text, subject, verb, object.
        Empty list if spaCy unavailable or no obligations found.
    """
    if nlp is None or not text.strip():
        return []

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error("spaCy detect_obligations error: %s", exc)
        return []

    obligations: list[dict] = []
    seen_verbs: set[int] = set()

    for token in doc:
        # Pattern 1: Modal auxiliaries (will, 'll, shall, must)
        is_modal = (
            token.dep_ == "aux" and token.lemma_.lower() in _OBLIGATION_LEMMAS
        ) or (
            token.text.lower() in _OBLIGATION_AUX
        )

        if is_modal:
            main_verb = token.head
            if main_verb.i in seen_verbs or main_verb.pos_ not in ("VERB", "AUX"):
                continue
            seen_verbs.add(main_verb.i)

            subjects = [
                w.text for w in main_verb.subtree
                if w.dep_ in ("nsubj", "nsubjpass") and w.i < main_verb.i
            ]
            objects = [
                w.text for w in main_verb.children
                if w.dep_ in ("dobj", "pobj", "attr", "xcomp", "ccomp")
            ]
            if subjects:
                obligations.append({
                    "text":    text.strip(),
                    "subject": subjects[0],
                    "verb":    main_verb.text,
                    "object":  objects[0] if objects else "",
                })
            continue

        # Pattern 2: "need to X", "must X" as ROOT verb with xcomp child
        if (
            token.lemma_.lower() in _OBLIGATION_ROOT_LEMMAS
            and token.pos_ == "VERB"
            and token.i not in seen_verbs
        ):
            seen_verbs.add(token.i)
            subjects = [
                w.text for w in token.children
                if w.dep_ in ("nsubj", "nsubjpass")
            ]
            xcomps = [
                w.text for w in token.children
                if w.dep_ == "xcomp"
            ]
            if subjects:
                obligations.append({
                    "text":    text.strip(),
                    "subject": subjects[0],
                    "verb":    token.text,
                    "object":  xcomps[0] if xcomps else "",
                })

    return obligations


# ---------------------------------------------------------------------------
# Risk detection — dependency parsing for concern/worry phrases
# ---------------------------------------------------------------------------

_RISK_LEMMAS = {"concern", "worry", "risk", "fear", "doubt", "threaten", "delay", "block", "fail",
                "concerned", "worried"}  # spaCy lemmatises adj forms as themselves
_RISK_NOUNS = {"risk", "concern", "issue", "problem", "blocker", "threat", "delay", "bottleneck"}


def detect_risks(text: str) -> list[dict]:
    """
    Detect risk/concern phrases using spaCy dependency parsing and NER.

    Looks for risk-signalling verbs (concerned, worried, risk) and
    risk-related noun phrases. Returns a list of risk dicts.

    Returns:
        List of dicts with keys: text, signal, detail.
        Empty list if spaCy unavailable or no risks found.
    """
    if nlp is None or not text.strip():
        return []

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error("spaCy detect_risks error: %s", exc)
        return []

    risks: list[dict] = []
    seen: set[str] = set()

    # Check for risk-signalling verbs
    for token in doc:
        if token.lemma_.lower() in _RISK_LEMMAS and token.pos_ in ("VERB", "ADJ"):
            detail_tokens = [w.text for w in token.subtree]
            detail = " ".join(detail_tokens)
            key = f"verb:{token.lemma_.lower()}"
            if key not in seen:
                seen.add(key)
                risks.append({
                    "text":   text.strip(),
                    "signal": token.text,
                    "detail": detail,
                })

    # Check for risk-related noun phrases
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _RISK_NOUNS:
            key = f"noun:{chunk.root.lemma_.lower()}"
            if key not in seen:
                seen.add(key)
                risks.append({
                    "text":   text.strip(),
                    "signal": chunk.root.text,
                    "detail": chunk.text,
                })

    return risks


# ---------------------------------------------------------------------------
# Decision detection — dependency parsing for "agreed", "decided", "approved"
# ---------------------------------------------------------------------------

_DECISION_LEMMAS = {"decide", "agree", "approve", "confirm", "conclude", "resolve", "proceed"}
_DECISION_NOUNS = {"decision", "agreement", "approval", "consensus"}


def detect_decisions(text: str) -> list[dict]:
    """
    Detect decision-signalling language using spaCy dependency parsing.

    Looks for decision verbs (decided, agreed, approved) and decision-related
    noun phrases. Returns a list of decision dicts.

    Returns:
        List of dicts with keys: text, signal, detail.
        Empty list if spaCy unavailable or no decisions found.
    """
    if nlp is None or not text.strip():
        return []

    try:
        doc = nlp(text)
    except Exception as exc:
        logger.error("spaCy detect_decisions error: %s", exc)
        return []

    decisions: list[dict] = []
    seen: set[str] = set()

    # Check for decision-signalling verbs
    for token in doc:
        if token.lemma_.lower() in _DECISION_LEMMAS and token.pos_ in ("VERB", "ADJ"):
            detail_tokens = [w.text for w in token.subtree]
            detail = " ".join(detail_tokens)
            key = f"verb:{token.lemma_.lower()}"
            if key not in seen:
                seen.add(key)
                decisions.append({
                    "text":   text.strip(),
                    "signal": token.text,
                    "detail": detail,
                })

    # Check for decision-related noun phrases
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _DECISION_NOUNS:
            key = f"noun:{chunk.root.lemma_.lower()}"
            if key not in seen:
                seen.add(key)
                decisions.append({
                    "text":   text.strip(),
                    "signal": chunk.root.text,
                    "detail": chunk.text,
                })

    return decisions


def enrich_key_facts(text: str, current_facts: Optional[dict] = None) -> dict:
    """
    Use NER to fill in missing Key Facts fields.

    Args:
        text:          Transcript chunk to analyse.
        current_facts: Existing facts dict (decision, owner, deadline, risk, action).
                       Missing/null fields will be filled from NER if possible.

    Returns:
        Updated facts dict with NER-enriched values.
    """
    _default = {"decision": None, "owner": None, "deadline": None, "risk": None, "action": None}
    facts = dict(current_facts) if current_facts else dict(_default)

    try:
        entities = extract_entities(text)
    except Exception as exc:
        logger.error("spaCy enrich_key_facts error: %s", exc)
        return facts

    # First PERSON → Owner (if not already set)
    if not facts.get("owner") and entities["people"]:
        facts["owner"] = entities["people"][0]

    # First DATE → Deadline (if not already set)
    if not facts.get("deadline") and entities["dates"]:
        facts["deadline"] = entities["dates"][0]

    # First MONEY → Risk (if not already set — budget mentions are risk signals)
    if not facts.get("risk") and entities["money"]:
        facts["risk"] = f"Budget: {entities['money'][0]}"

    # First action → Action (if not already set)
    if not facts.get("action") and entities["actions"]:
        a = entities["actions"][0]
        facts["action"] = f"{a['subject']} {a['verb']} {a['object']}".strip()

    return facts
