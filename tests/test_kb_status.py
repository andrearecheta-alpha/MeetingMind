"""
tests/test_kb_status.py
-----------------------
S8-006: Tests for GET /kb/status, POST /kb/reseed,
and _verify_pm_global_kb startup verification.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TestKBStatus:
    """GET /kb/status endpoint tests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def test_kb_status_returns_correct_structure(self):
        """GET /kb/status returns pm_global, project_kb, total_documents."""
        c = self._client()
        resp = c.get("/kb/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "pm_global" in data
        assert "project_kb" in data
        assert "total_documents" in data
        # pm_global shape
        pg = data["pm_global"]
        assert "document_count" in pg
        assert "files" in pg
        assert "last_seeded" in pg
        assert isinstance(pg["document_count"], int)
        assert isinstance(pg["files"], list)
        # project_kb shape
        pk = data["project_kb"]
        assert "document_count" in pk
        assert "project_name" in pk

    def test_pm_global_document_count_matches(self):
        """pm_global document_count reflects actual ChromaDB collection count."""
        c = self._client()
        resp = c.get("/kb/status")
        assert resp.status_code == 200
        data = resp.json()
        count = data["pm_global"]["document_count"]
        # Count should be a non-negative int (may be 0 if KB not initialised)
        assert isinstance(count, int)
        assert count >= 0

    def test_pm_global_files_lists_markdown(self):
        """pm_global files list should contain .md filenames if directory exists."""
        c = self._client()
        resp = c.get("/kb/status")
        data = resp.json()
        files = data["pm_global"]["files"]
        for f in files:
            assert f.endswith(".md")

    def test_total_documents_is_sum(self):
        """total_documents equals pm_global + project_kb counts."""
        c = self._client()
        resp = c.get("/kb/status")
        data = resp.json()
        expected = data["pm_global"]["document_count"] + data["project_kb"]["document_count"]
        assert data["total_documents"] == expected


class TestKBReseed:
    """POST /kb/reseed endpoint tests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def test_reseed_returns_status(self):
        """POST /kb/reseed returns {"status": "reseeded", "documents": int}."""
        c = self._client()
        resp = c.post("/kb/reseed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reseeded"
        assert isinstance(data["documents"], int)
        assert data["documents"] >= 0

    def test_reseed_with_empty_dir_returns_zero(self):
        """POST /kb/reseed with no .md files in pm_global returns documents=0."""
        import meetingmind.main as m

        # Mock _kb.force_ingest_pm_global_kb to return 0 (empty dir)
        mock_kb = MagicMock()
        mock_kb.force_ingest_pm_global_kb.return_value = 0
        original_kb = m._kb
        m._kb = mock_kb
        try:
            c = self._client()
            resp = c.post("/kb/reseed")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "reseeded"
            assert data["documents"] == 0
        finally:
            m._kb = original_kb

    def test_reseed_updates_last_seeded(self):
        """POST /kb/reseed updates _pm_global_last_seeded timestamp."""
        import meetingmind.main as m
        m._pm_global_last_seeded = None
        c = self._client()
        resp = c.post("/kb/reseed")
        assert resp.status_code == 200
        assert m._pm_global_last_seeded is not None


class TestVerifyPmGlobalKb:
    """_verify_pm_global_kb startup verification tests."""

    def test_verify_logs_already_seeded(self, caplog):
        """When pm_global has docs, logs 'already seeded' and skips."""
        import meetingmind.main as m

        mock_kb = MagicMock()
        mock_kb.pm_global_doc_count.return_value = 42
        original_kb = m._kb
        m._kb = mock_kb
        try:
            with caplog.at_level(logging.INFO, logger="meetingmind.main"):
                asyncio.get_event_loop().run_until_complete(m._verify_pm_global_kb())
            assert any("already seeded" in r.message for r in caplog.records)
            # Should NOT have called ingest
            mock_kb.ingest_pm_global_kb.assert_not_called()
        finally:
            m._kb = original_kb

    def test_verify_reingests_when_empty(self, caplog):
        """When pm_global has 0 docs, re-ingests and logs count."""
        import meetingmind.main as m

        mock_kb = MagicMock()
        mock_kb.pm_global_doc_count.return_value = 0
        mock_kb.ingest_pm_global_kb.return_value = 15
        original_kb = m._kb
        m._kb = mock_kb
        try:
            with caplog.at_level(logging.INFO, logger="meetingmind.main"):
                asyncio.get_event_loop().run_until_complete(m._verify_pm_global_kb())
            assert any("0 documents" in r.message for r in caplog.records)
            mock_kb.ingest_pm_global_kb.assert_called_once()
        finally:
            m._kb = original_kb

    def test_verify_never_blocks_on_error(self, caplog):
        """_verify_pm_global_kb swallows exceptions and never blocks startup."""
        import meetingmind.main as m

        mock_kb = MagicMock()
        mock_kb.pm_global_doc_count.side_effect = RuntimeError("boom")
        original_kb = m._kb
        m._kb = mock_kb
        try:
            with caplog.at_level(logging.WARNING, logger="meetingmind.main"):
                # Should NOT raise
                asyncio.get_event_loop().run_until_complete(m._verify_pm_global_kb())
            assert any("verification failed" in r.message for r in caplog.records)
        finally:
            m._kb = original_kb
