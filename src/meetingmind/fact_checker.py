"""
fact_checker.py
---------------
KB Fact-Check Engine — compares spoken numbers against seeded KB facts.

When a speaker mentions a number during a meeting, this module queries the
ChromaDB "facts" collection, parses numeric values from both the spoken
entity and the stored document, computes the percentage variance, and
classifies severity (OK / NOTABLE / FLAG).

All public functions are synchronous — call via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class FactCheckSeverity(Enum):
    OK      = "OK"         # <5% variance
    NOTABLE = "NOTABLE"    # 5–15% variance
    FLAG    = "FLAG"       # >15% variance


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FactCheckResult:
    entity_text:   str               # e.g. "$18,500"
    entity_type:   str               # e.g. "MONEY"
    spoken_value:  float             # parsed number from spoken entity
    stored_text:   str               # full stored KB document text
    stored_value:  float             # parsed number from stored doc
    variance_pct:  float             # absolute % difference
    severity:      FactCheckSeverity
    context_chunk: str               # the transcript chunk that triggered
    distance:      float             # ChromaDB embedding distance


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}

def parse_number(text: str) -> Optional[float]:
    """
    Parse a numeric value from text, stripping currency symbols, commas, and
    handling k/M/B multipliers.

    Returns None if no number can be extracted.

    Examples:
        "$18,500"  → 18500.0
        "6%"       → 6.0
        "$10k"     → 10000.0
        "5.5%"     → 5.5
        "1,200"    → 1200.0
        "€10k"     → 10000.0
    """
    if not text or not text.strip():
        return None

    s = text.strip()

    # Strip currency symbols and whitespace
    s = re.sub(r"[£€$¥₹\s]", "", s)

    # Strip trailing percent sign
    s = s.rstrip("%")

    # Strip commas
    s = s.replace(",", "")

    if not s:
        return None

    # Check for multiplier suffix (k, M, B — case-insensitive)
    multiplier = 1.0
    if s and s[-1].lower() in _MULTIPLIERS:
        multiplier = _MULTIPLIERS[s[-1].lower()]
        s = s[:-1]

    if not s:
        return None

    # Try to parse the remaining string as a float
    try:
        return float(s) * multiplier
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Variance computation
# ---------------------------------------------------------------------------

def compute_variance(spoken: float, stored: float) -> float:
    """
    Compute the absolute percentage difference between spoken and stored values.

    Returns 0.0 if both values are zero.  If stored is zero but spoken is not,
    returns 100.0 (complete divergence).
    """
    if stored == 0.0:
        return 0.0 if spoken == 0.0 else 100.0
    return abs(spoken - stored) / abs(stored) * 100.0


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_severity(variance_pct: float) -> FactCheckSeverity:
    """
    Classify a variance percentage into a severity level.

    <5%   → OK      (green)
    5–15% → NOTABLE (yellow)
    >15%  → FLAG    (red)
    """
    if variance_pct < 5.0:
        return FactCheckSeverity.OK
    elif variance_pct <= 15.0:
        return FactCheckSeverity.NOTABLE
    else:
        return FactCheckSeverity.FLAG


# ---------------------------------------------------------------------------
# FactChecker — main engine
# ---------------------------------------------------------------------------

_DISTANCE_THRESHOLD = 1.5  # filter out semantically irrelevant KB matches

# Metric keyword groups — when context matches a group, only sentences
# containing those keywords are considered for number extraction.
_METRIC_KEYWORDS: dict[str, list[str]] = {
    "acos":      ["acos", "advertising cost"],
    "revenue":   ["revenue", "sales", "income"],
    "margin":    ["margin", "gross"],
    "budget":    ["budget", "spend", "allocated"],
    "inventory": ["inventory", "runway", "days"],
    "cpc":       ["cpc", "cost per click"],
    "cvr":       ["cvr", "conversion rate"],
    "roas":      ["roas", "return on ad"],
}


class FactChecker:
    """
    Compares spoken numeric entities against seeded KB facts via ChromaDB.

    Accepts a ChromaDB Collection object (reuses ``_kb._collections["facts"]``
    — no duplicate client).
    """

    def __init__(self, facts_collection) -> None:
        self._collection = facts_collection

    def is_available(self) -> bool:
        """True if the facts collection exists and has at least one item."""
        try:
            return self._collection is not None and self._collection.count() > 0
        except Exception:
            return False

    def check_chunk(
        self,
        text: str,
        numeric_entities: list[dict],
    ) -> list[FactCheckResult]:
        """
        Check spoken numeric entities against KB facts.

        Args:
            text:              The full transcript chunk text.
            numeric_entities:  List of dicts with "text" and "label" keys
                               (from extract_numeric_entities).

        Returns:
            List of FactCheckResult, deduplicated by entity (keeps lowest
            distance match per entity).
        """
        if not numeric_entities or not self.is_available():
            return []

        results: list[FactCheckResult] = []
        seen_entities: dict[str, FactCheckResult] = {}  # entity_text → best result

        for ent in numeric_entities:
            entity_text = ent.get("text", "")
            entity_type = ent.get("label", "")

            spoken_value = parse_number(entity_text)
            if spoken_value is None:
                continue

            # Query ChromaDB with the full chunk for semantic relevance
            try:
                query_result = self._collection.query(
                    query_texts=[text],
                    n_results=3,
                )
            except Exception as exc:
                logger.warning("ChromaDB query failed for entity %r: %s", entity_text, exc)
                continue

            if not query_result or not query_result.get("documents"):
                continue

            docs = query_result["documents"][0]
            distances = query_result["distances"][0] if query_result.get("distances") else []

            for i, doc in enumerate(docs):
                dist = distances[i] if i < len(distances) else 999.0

                # Filter out semantically irrelevant matches
                if dist > _DISTANCE_THRESHOLD:
                    continue

                # Try to extract a context-relevant number from the stored document
                stored_value = self._extract_number_from_doc(doc, entity_text, text)
                if stored_value is None:
                    continue

                variance = compute_variance(spoken_value, stored_value)
                severity = classify_severity(variance)

                result = FactCheckResult(
                    entity_text=entity_text,
                    entity_type=entity_type,
                    spoken_value=spoken_value,
                    stored_text=self._get_kb_context(doc, entity_text),
                    stored_value=stored_value,
                    variance_pct=round(variance, 1),
                    severity=severity,
                    context_chunk=text,
                    distance=round(dist, 4),
                )

                # Dedup: keep lowest distance per entity
                existing = seen_entities.get(entity_text)
                if existing is None or dist < existing.distance:
                    seen_entities[entity_text] = result

        return list(seen_entities.values())

    @staticmethod
    def _get_kb_context(doc_text: str, spoken_text: str) -> str:
        """Return the most relevant sentence from doc_text, capped at 100 chars."""
        # Split on period+space or newlines/semicolons (safe for decimals like 4.1%)
        sentences = re.split(r'(?<=\.\s)|(?<=\.\Z)|[\n;]', doc_text)
        spoken_words = set(spoken_text.lower().split())

        best = ""
        best_score = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            score = sum(1 for w in spoken_words if w in s.lower())
            if score > best_score:
                best_score = score
                best = s

        result = best or doc_text
        if len(result) > 100:
            result = result[:97] + "..."
        return result

    @staticmethod
    def _extract_number_from_doc(
        doc: str,
        spoken_entity: str = "",
        context: str = "",
    ) -> Optional[float]:
        """
        Extract the most contextually relevant number from a KB document.

        Strategy:
        1. Check metric keywords — if context mentions a known metric (e.g.
           "ACOS"), only consider sentences containing that metric's keywords.
        2. Score each sentence by overlap with context words and pick the best.
        3. Extract the first number from the winning sentence.
        4. Fallback: first number in the entire document.
        """
        # Split into sentences — avoid splitting on decimal points (e.g. "4.1%")
        # by only splitting on periods followed by a space or end-of-string.
        sentences = [s.strip() for s in re.split(r"(?:\.\s|\.\Z|[\n;])", doc) if s.strip()]

        if not sentences:
            return _extract_first_number(doc)

        # If only one sentence, just extract from it directly
        if len(sentences) == 1:
            return _extract_first_number(sentences[0])

        # ── Step 1: metric keyword filter ────────────────────────────────
        context_lower = (context + " " + spoken_entity).lower()
        metric_sentences: list[str] = []
        for _metric_name, keywords in _METRIC_KEYWORDS.items():
            if any(kw in context_lower for kw in keywords):
                for sent in sentences:
                    sent_lower = sent.lower()
                    if any(kw in sent_lower for kw in keywords):
                        metric_sentences.append(sent)

        if metric_sentences:
            for sent in metric_sentences:
                val = _extract_first_number(sent)
                if val is not None:
                    return val

        # ── Step 2: context-word overlap scoring ─────────────────────────
        # Build context words from both the spoken entity and the chunk
        context_words = set(
            w for w in context_lower.split()
            if len(w) > 2  # skip tiny words like "is", "at", "a"
        )

        best_sentence: Optional[str] = None
        best_score = 0
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for w in context_words if w in sent_lower)
            if score > best_score:
                best_score = score
                best_sentence = sent

        if best_sentence and best_score > 0:
            val = _extract_first_number(best_sentence)
            if val is not None:
                return val

        # ── Step 3: fallback — first number in entire doc ────────────────
        return _extract_first_number(doc)


def _extract_first_number(text: str) -> Optional[float]:
    """Extract the first parseable number from text.

    Tries currency patterns, then percentages, then plain numbers.
    """
    # Currency: $20,000  €10k  etc.
    for m in re.finditer(r"[$€£¥₹]\s*[\d,]+(?:\.\d+)?[kKmMbB]?", text):
        val = parse_number(m.group())
        if val is not None:
            return val

    # Percentages: 6%, 5.5%
    for m in re.finditer(r"\d+(?:\.\d+)?%", text):
        val = parse_number(m.group())
        if val is not None:
            return val

    # Plain numbers with optional multipliers: 1,200  10k
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?[kKmMbB]?", text):
        val = parse_number(m.group())
        if val is not None:
            return val

    return None
