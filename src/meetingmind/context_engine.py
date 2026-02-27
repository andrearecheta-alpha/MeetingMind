"""
context_engine.py
-----------------
Retrieves semantically relevant historical context from ChromaDB and formats
it for injection into Claude suggestion prompts.

Usage (always call get_context via asyncio.to_thread):

    engine  = ContextEngine()
    grounded = await asyncio.to_thread(engine.get_context, transcript_snippet)
    historical_context = grounded.historical_context  # None if KB empty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_PATH  = _PROJECT_ROOT / "data" / "knowledge_base"

# Collections to query — open_loops and summaries are too broad for grounding
_QUERY_COLLECTIONS = ("decisions", "commitments", "risks", "action_items")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroundedContext:
    current_transcript:  str
    historical_context:  Optional[str]       # None if KB empty or no relevant history
    source_meeting_ids:  list[str] = field(default_factory=list)
    item_count:          int = 0


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------

class ContextEngine:
    """
    Queries ChromaDB for past meeting items relevant to the current transcript
    and formats them as readable historical context for Claude.

    Gracefully returns GroundedContext(historical_context=None) if ChromaDB
    is unavailable or the knowledge base is empty.
    """

    def __init__(
        self,
        chroma_path: Path = _CHROMA_PATH,
        top_n:       int  = 5,
    ) -> None:
        self._chroma_path = Path(chroma_path)
        self._top_n       = top_n
        self._chroma      = None
        self._collections: dict = {}
        self._init_chromadb()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_chromadb(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._chroma = chromadb.PersistentClient(path=str(self._chroma_path))
            ef = DefaultEmbeddingFunction()
            for name in _QUERY_COLLECTIONS:
                try:
                    col = self._chroma.get_collection(name=name, embedding_function=ef)
                    self._collections[name] = col
                except Exception:
                    pass  # Collection doesn't exist yet — that's fine
            logger.info(
                "ContextEngine initialised — %d/%d collections available.",
                len(self._collections), len(_QUERY_COLLECTIONS),
            )
        except Exception as exc:
            logger.warning("ContextEngine ChromaDB init failed (history disabled): %s", exc)
            self._chroma = None
            self._collections = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self, transcript_snippet: str) -> GroundedContext:
        """
        Query ChromaDB for items semantically similar to the transcript snippet.

        Returns a GroundedContext with formatted historical_context ready for
        injection into a Claude prompt. Returns historical_context=None if the
        KB is empty or ChromaDB is unavailable.
        """
        if not self._collections:
            return GroundedContext(
                current_transcript=transcript_snippet,
                historical_context=None,
            )

        # Query each collection — 2 results per collection
        candidates: list[dict] = []
        for name, col in self._collections.items():
            candidates.extend(self._query_collection(name, col, transcript_snippet, n=2))

        if not candidates:
            return GroundedContext(
                current_transcript=transcript_snippet,
                historical_context=None,
            )

        # Dedup by document text, sort by distance (lower = more similar)
        seen:   set[str] = set()
        unique: list[dict] = []
        for c in sorted(candidates, key=lambda x: x["distance"]):
            if c["document"] not in seen:
                seen.add(c["document"])
                unique.append(c)

        top = unique[: self._top_n]

        historical_context = self._format_historical_context(top)
        meeting_ids = list({c["metadata"].get("meeting_id", "") for c in top})

        return GroundedContext(
            current_transcript=transcript_snippet,
            historical_context=historical_context,
            source_meeting_ids=meeting_ids,
            item_count=len(top),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _query_collection(
        self,
        name:       str,
        collection, # chromadb.Collection
        query_text: str,
        n:          int,
    ) -> list[dict]:
        """Query a single collection, returning a list of result dicts."""
        try:
            count = collection.count()
            if count == 0:
                return []
            n_actual = min(n, count)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_actual,
                include=["documents", "metadatas", "distances"],
            )
            items = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "collection": name,
                    "document":   doc,
                    "metadata":   meta,
                    "distance":   dist,
                })
            return items
        except Exception as exc:
            logger.debug("ContextEngine query to '%s' failed: %s", name, exc)
            return []

    def _format_historical_context(self, items: list[dict]) -> str:
        """Format retrieved items as a readable block for Claude."""
        lines = ["RELEVANT HISTORY FROM PAST MEETINGS:", ""]
        for item in items:
            col  = item["collection"]
            meta = item["metadata"]
            doc  = item["document"]

            ts      = meta.get("timestamp", "")[:10]  # date only
            project = meta.get("project",   "")

            if col == "decisions":
                header = f"[Decision — {ts}"
                if project:
                    header += f", Project: {project}"
                header += "]"
            elif col == "commitments":
                owner  = meta.get("owner", "Unknown")
                header = f"[Commitment — {ts}, Owner: {owner}"
                if project:
                    header += f", Project: {project}"
                header += "]"
            elif col == "risks":
                severity = meta.get("severity", "medium").capitalize()
                header   = f"[Risk — {ts}, Severity: {severity}"
                if project:
                    header += f", Project: {project}"
                header += "]"
            elif col == "action_items":
                owner  = meta.get("owner", "Unassigned")
                header = f"[Action — {ts}, Owner: {owner}"
                if project:
                    header += f", Project: {project}"
                header += "]"
            else:
                header = f"[{col.capitalize()} — {ts}]"

            lines.append(header)
            lines.append(doc)
            lines.append("")

        return "\n".join(lines).rstrip()
