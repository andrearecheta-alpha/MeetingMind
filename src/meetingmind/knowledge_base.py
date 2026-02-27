"""
knowledge_base.py
-----------------
Meeting intelligence ingestion pipeline.

Extracts structured data from meeting transcripts via Claude, persists it to
SQLite (analytics) and ChromaDB (semantic search), and computes a Meeting
Quality Score (MQS).

The public entry point is KnowledgeBase.ingest_meeting() which is synchronous
and CPU/IO-bound — always call it via asyncio.to_thread() from async code.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from meetingmind._api_key import load_api_key as _load_api_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH      = _PROJECT_ROOT / "data" / "analytics" / "meetings.db"
_CHROMA_PATH  = _PROJECT_ROOT / "data" / "knowledge_base"

# ---------------------------------------------------------------------------
# Claude extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an expert meeting intelligence extractor for a Project Manager.
Read the meeting transcript and extract structured data.
Respond with a single valid JSON object — no markdown fences, no extra commentary.

Return EXACTLY:
{
  "project_name": "<auto-detect or null>",
  "summary":      "<3-5 sentence paragraph>",
  "sentiment":    "<productive | tense | neutral | inconclusive>",
  "decisions":    [{"text": "...", "owner": "...", "deadline": "...", "confidence": 0.0}],
  "action_items": [{"task": "...", "owner": "...", "due": "..."}],
  "risks":        [{"text": "...", "severity": "high|medium|low", "mitigation": "..."}],
  "commitments":  [{"owner": "...", "text": "...", "deadline": "...", "is_firm": true}],
  "open_loops":   ["<unresolved question or issue>"],
  "stakeholders": [{"name": "...", "sentiment": "positive|neutral|negative|frustrated|disengaged"}]
}
Empty lists [] where no items found.\
"""

_EXTRACTION_USER = "Meeting transcript:\n\n{transcript}"


# ---------------------------------------------------------------------------
# MQS calculation
# ---------------------------------------------------------------------------

def _compute_mqs(analysis: dict, duration_seconds: float) -> float:
    """Compute Meeting Quality Score (0–100) from extracted analysis."""
    decisions    = analysis.get("decisions", [])
    action_items = analysis.get("action_items", [])

    # Decision score
    total_dec = len(decisions)
    if total_dec == 0:
        decision_score = 0.50
    else:
        with_owner    = sum(1 for d in decisions if d.get("owner", "").strip())
        with_deadline = sum(1 for d in decisions if d.get("deadline", "").strip())
        decision_score = 0.5 * (with_owner / total_dec) + 0.5 * (with_deadline / total_dec)

    # Action score
    total_act = len(action_items)
    if total_act == 0 and total_dec == 0:
        action_score = 0.40   # penalty: no outputs
    elif total_act == 0:
        action_score = 0.50   # neutral
    else:
        assigned = sum(1 for a in action_items if a.get("owner", "Unassigned") != "Unassigned")
        dated    = sum(1 for a in action_items if a.get("due", "Not specified") != "Not specified")
        action_score = 0.5 * (assigned / total_act) + 0.5 * (dated / total_act)

    # Inclusion score (placeholder — no diarisation yet)
    inclusion_score = 0.50

    # Time score
    time_score = 0.75 if duration_seconds <= 3600 else 0.50

    mqs = (
        0.35 * decision_score
        + 0.25 * action_score
        + 0.20 * inclusion_score
        + 0.20 * time_score
    ) * 100
    return round(mqs, 1)


# ---------------------------------------------------------------------------
# KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Stores and retrieves meeting intelligence from SQLite and ChromaDB.

    Usage (always call ingest_meeting via asyncio.to_thread):

        kb = KnowledgeBase()
        result = await asyncio.to_thread(
            kb.ingest_meeting, meeting_id, started_at, stopped_at,
            duration_seconds, model_size, transcript_text
        )
        kb.close()
    """

    def __init__(
        self,
        db_path:    Path = _DB_PATH,
        chroma_path: Path = _CHROMA_PATH,
    ) -> None:
        self._db_path    = Path(db_path)
        self._chroma_path = Path(chroma_path)
        self._lock       = threading.Lock()  # SQLite write mutex
        self._conn: Optional[sqlite3.Connection] = None
        self._chroma     = None
        self._collections: dict = {}

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._chroma_path.mkdir(parents=True, exist_ok=True)

        self._init_sqlite()
        self._init_chromadb()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS meetings (
                    meeting_id        TEXT PRIMARY KEY,
                    started_at        TEXT NOT NULL,
                    stopped_at        TEXT NOT NULL,
                    duration_seconds  REAL NOT NULL,
                    model_size        TEXT NOT NULL,
                    project_name      TEXT,
                    mqs_score         REAL,
                    decision_count    INTEGER NOT NULL DEFAULT 0,
                    action_count      INTEGER NOT NULL DEFAULT 0,
                    risk_count        INTEGER NOT NULL DEFAULT 0,
                    open_loop_count   INTEGER NOT NULL DEFAULT 0,
                    sentiment         TEXT,
                    transcript_text   TEXT NOT NULL,
                    raw_analysis_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS action_items (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT    NOT NULL REFERENCES meetings(meeting_id),
                    task       TEXT    NOT NULL,
                    owner      TEXT    NOT NULL DEFAULT 'Unassigned',
                    due        TEXT    NOT NULL DEFAULT 'Not specified',
                    status     TEXT    NOT NULL DEFAULT 'open',
                    created_at TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_action_items_meeting
                    ON action_items(meeting_id);
                CREATE INDEX IF NOT EXISTS idx_meetings_started_at
                    ON meetings(started_at);
            """)
            self._conn.commit()
        logger.info("SQLite initialised at %s", self._db_path)

    def _init_chromadb(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._chroma = chromadb.PersistentClient(path=str(self._chroma_path))
            ef = DefaultEmbeddingFunction()
            for name in ("decisions", "action_items", "risks",
                         "commitments", "open_loops", "summaries"):
                self._collections[name] = self._chroma.get_or_create_collection(
                    name=name,
                    embedding_function=ef,
                )
            logger.info(
                "ChromaDB initialised at %s — %d collections",
                self._chroma_path, len(self._collections),
            )
        except Exception as exc:
            logger.warning("ChromaDB init failed (vector search disabled): %s", exc)
            self._chroma = None
            self._collections = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_meeting(
        self,
        meeting_id:       str,
        started_at:       datetime,
        stopped_at:       datetime,
        duration_seconds: float,
        model_size:       str,
        transcript_text:  str,
        model:            str = "claude-sonnet-4-6",
    ) -> dict:
        """
        Extract intelligence from a transcript and persist to SQLite + ChromaDB.

        Returns a summary dict with mqs_score, decision_count, action_count,
        risk_count, open_loop_count, project_name, and sentiment.
        """
        logger.info("Ingesting meeting %s (%d chars)…", meeting_id, len(transcript_text))

        analysis  = self._extract_intelligence(transcript_text, model)
        mqs_score = _compute_mqs(analysis, duration_seconds)

        self._write_sqlite(
            meeting_id, started_at, stopped_at,
            duration_seconds, model_size, transcript_text,
            analysis, mqs_score,
        )
        self._write_chromadb(meeting_id, started_at, analysis)

        result = {
            "meeting_id":     meeting_id,
            "mqs_score":      mqs_score,
            "decision_count": len(analysis.get("decisions",   [])),
            "action_count":   len(analysis.get("action_items",[])),
            "risk_count":     len(analysis.get("risks",       [])),
            "open_loop_count":len(analysis.get("open_loops",  [])),
            "project_name":   analysis.get("project_name"),
            "sentiment":      analysis.get("sentiment"),
        }
        logger.info(
            "Ingestion complete for %s: MQS=%.1f decisions=%d actions=%d",
            meeting_id, mqs_score, result["decision_count"], result["action_count"],
        )
        return result

    def list_meetings(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Return recent meetings ordered newest-first."""
        if self._conn is None:
            return []
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT meeting_id, started_at, stopped_at, duration_seconds,
                   model_size, project_name, mqs_score, decision_count,
                   action_count, risk_count, open_loop_count, sentiment
            FROM meetings
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_meeting(self, meeting_id: str) -> Optional[dict]:
        """Return full meeting record including transcript and raw analysis."""
        if self._conn is None:
            return None
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        # Deserialise stored JSON
        try:
            record["raw_analysis_json"] = json.loads(record["raw_analysis_json"])
        except Exception:
            pass
        return record

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_intelligence(self, transcript_text: str, model: str) -> dict:
        """Call Claude to extract structured intelligence from transcript."""
        api_key = _load_api_key()
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is not installed.") from exc

        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_EXTRACTION_SYSTEM,
                messages=[{
                    "role":    "user",
                    "content": _EXTRACTION_USER.format(transcript=transcript_text),
                }],
            )
        except Exception as exc:
            raise RuntimeError(f"Claude extraction API call failed: {exc}") from exc

        raw = response.content[0].text.strip()
        # Strip markdown fences if Claude wrapped the JSON
        fenced = re.match(r"^```(?:json)?\s*([\s\S]+?)\s*```$", raw)
        if fenced:
            raw = fenced.group(1).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude returned unparseable JSON: {exc}\nRaw (first 500): {raw[:500]}"
            ) from exc

    def _write_sqlite(
        self,
        meeting_id:       str,
        started_at:       datetime,
        stopped_at:       datetime,
        duration_seconds: float,
        model_size:       str,
        transcript_text:  str,
        analysis:         dict,
        mqs_score:        float,
    ) -> None:
        started_iso  = started_at.astimezone(timezone.utc).isoformat()
        stopped_iso  = stopped_at.astimezone(timezone.utc).isoformat()
        created_iso  = datetime.now(timezone.utc).isoformat()

        decisions    = analysis.get("decisions",    [])
        action_items = analysis.get("action_items", [])
        risks        = analysis.get("risks",        [])
        open_loops   = analysis.get("open_loops",   [])

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO meetings (
                    meeting_id, started_at, stopped_at, duration_seconds,
                    model_size, project_name, mqs_score,
                    decision_count, action_count, risk_count, open_loop_count,
                    sentiment, transcript_text, raw_analysis_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    meeting_id, started_iso, stopped_iso, duration_seconds,
                    model_size,
                    analysis.get("project_name"),
                    mqs_score,
                    len(decisions), len(action_items),
                    len(risks), len(open_loops),
                    analysis.get("sentiment"),
                    transcript_text,
                    json.dumps(analysis),
                ),
            )
            # Delete existing action items for this meeting (upsert pattern)
            cur.execute(
                "DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,)
            )
            for item in action_items:
                cur.execute(
                    """
                    INSERT INTO action_items
                        (meeting_id, task, owner, due, status, created_at)
                    VALUES (?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        meeting_id,
                        item.get("task",  ""),
                        item.get("owner", "Unassigned"),
                        item.get("due",   "Not specified"),
                        created_iso,
                    ),
                )
            self._conn.commit()
        logger.debug("SQLite write complete for %s.", meeting_id)

    def _write_chromadb(
        self,
        meeting_id: str,
        started_at: datetime,
        analysis:   dict,
    ) -> None:
        if not self._collections:
            return

        ts        = started_at.astimezone(timezone.utc).isoformat()
        project   = analysis.get("project_name") or ""

        def _meta(extra: dict) -> dict:
            return {"meeting_id": meeting_id, "timestamp": ts,
                    "project": project, **extra}

        # decisions
        decisions = analysis.get("decisions", [])
        if decisions:
            self._upsert_collection(
                "decisions",
                [f"{meeting_id}_dec_{i}" for i in range(len(decisions))],
                [d.get("text", "") for d in decisions],
                [_meta({"confidence": str(d.get("confidence", 0.0))}) for d in decisions],
            )

        # action_items
        actions = analysis.get("action_items", [])
        if actions:
            self._upsert_collection(
                "action_items",
                [f"{meeting_id}_act_{i}" for i in range(len(actions))],
                [
                    f"{a.get('task','')} — owner: {a.get('owner','Unassigned')}, "
                    f"due: {a.get('due','Not specified')}"
                    for a in actions
                ],
                [_meta({"owner": a.get("owner",""), "due": a.get("due","")}) for a in actions],
            )

        # risks
        risks = analysis.get("risks", [])
        if risks:
            self._upsert_collection(
                "risks",
                [f"{meeting_id}_risk_{i}" for i in range(len(risks))],
                [r.get("text", "") for r in risks],
                [_meta({"severity": r.get("severity", "medium")}) for r in risks],
            )

        # commitments
        commitments = analysis.get("commitments", [])
        if commitments:
            self._upsert_collection(
                "commitments",
                [f"{meeting_id}_com_{i}" for i in range(len(commitments))],
                [
                    f"{c.get('owner','')} committed to: {c.get('text','')}"
                    for c in commitments
                ],
                [_meta({"owner": c.get("owner","")}) for c in commitments],
            )

        # open_loops
        open_loops = analysis.get("open_loops", [])
        if open_loops:
            self._upsert_collection(
                "open_loops",
                [f"{meeting_id}_loop_{i}" for i in range(len(open_loops))],
                open_loops,
                [_meta({}) for _ in open_loops],
            )

        # summary
        summary = analysis.get("summary", "").strip()
        if summary:
            self._upsert_collection(
                "summaries",
                [f"{meeting_id}_summary"],
                [summary],
                [_meta({
                    "sentiment":        analysis.get("sentiment", ""),
                    "duration_seconds": "",  # not available here; stored in SQLite
                })],
            )

        logger.debug("ChromaDB write complete for %s.", meeting_id)

    def _upsert_collection(
        self,
        name:      str,
        ids:       list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        try:
            col = self._collections[name]
            # Filter out empty documents — ChromaDB rejects them
            valid = [
                (id_, doc, meta)
                for id_, doc, meta in zip(ids, documents, metadatas)
                if doc.strip()
            ]
            if not valid:
                return
            v_ids, v_docs, v_metas = zip(*valid)
            col.upsert(
                ids=list(v_ids),
                documents=list(v_docs),
                metadatas=list(v_metas),
            )
        except Exception as exc:
            logger.warning("ChromaDB upsert to '%s' failed: %s", name, exc)
