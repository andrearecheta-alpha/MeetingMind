"""
tests/test_project_brief.py
----------------------------
Tests for the S8-001 Project Brief CRUD + document extraction endpoints.

Uses FastAPI TestClient — no hardware, no real Whisper.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TestProjectBrief:
    """POST/GET/DELETE /project/brief* endpoint tests."""

    _BRIEFS_DIR = Path(__file__).parents[1] / "data" / "project_briefs"

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_state(self):
        """Reset module-level project state between tests."""
        import meetingmind.main as m
        m._active_project = None

    def _cleanup_briefs(self):
        """Remove any test-created brief files."""
        if self._BRIEFS_DIR.exists():
            for f in self._BRIEFS_DIR.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("project_name", "").startswith("Test"):
                        f.unlink()
                except Exception:
                    pass

    def _minimal_payload(self, name="Test Project", **overrides):
        payload = {"project_name": name, "objectives": "Ship it"}
        payload.update(overrides)
        return payload

    def setup_method(self):
        self._reset_state()

    def teardown_method(self):
        self._reset_state()
        self._cleanup_briefs()

    # ── 1. Create brief returns id ────────────────────────────────────────
    def test_create_brief_returns_id(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert "project_id" in data
        assert len(data["project_id"]) == 36  # uuid4

    # ── 2. Create brief sets active ───────────────────────────────────────
    def test_create_brief_sets_active(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload())
        pid = resp.json()["project_id"]
        resp2 = client.get("/project/brief/active")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["active"] is True
        assert data["brief"]["project_id"] == pid

    # ── 3. Create brief persists to disk ──────────────────────────────────
    def test_create_brief_persists_to_disk(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload())
        pid = resp.json()["project_id"]
        path = self._BRIEFS_DIR / f"{pid}.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["project_name"] == "Test Project"

    # ── 4. Get brief by id ────────────────────────────────────────────────
    def test_get_brief_by_id(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload("Test Lookup"))
        pid = resp.json()["project_id"]
        resp2 = client.get(f"/project/brief/{pid}")
        assert resp2.status_code == 200
        assert resp2.json()["project_name"] == "Test Lookup"

    # ── 5. Get brief not found ────────────────────────────────────────────
    def test_get_brief_not_found(self):
        client = self._client()
        resp = client.get("/project/brief/nonexistent-id")
        assert resp.status_code == 404

    # ── 6. List briefs ────────────────────────────────────────────────────
    def test_list_briefs(self):
        client = self._client()
        client.post("/project/brief", json=self._minimal_payload("Test A"))
        self._reset_state()
        client.post("/project/brief", json=self._minimal_payload("Test B"))
        resp = client.get("/project/briefs")
        assert resp.status_code == 200
        briefs = resp.json()["briefs"]
        names = [b["project_name"] for b in briefs]
        assert "Test A" in names
        assert "Test B" in names

    # ── 7. Active when none ───────────────────────────────────────────────
    def test_active_when_none(self):
        client = self._client()
        resp = client.get("/project/brief/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["brief"] is None

    # ── 8. Delete brief ───────────────────────────────────────────────────
    def test_delete_brief(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload("Test Delete"))
        pid = resp.json()["project_id"]
        resp2 = client.delete(f"/project/brief/{pid}")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "deleted"
        # Verify file is gone
        resp3 = client.get(f"/project/brief/{pid}")
        assert resp3.status_code == 404

    # ── 9. Delete clears active ───────────────────────────────────────────
    def test_delete_clears_active(self):
        client = self._client()
        resp = client.post("/project/brief", json=self._minimal_payload("Test Active Del"))
        pid = resp.json()["project_id"]
        client.delete(f"/project/brief/{pid}")
        resp2 = client.get("/project/brief/active")
        assert resp2.json()["active"] is False

    # ── 10. Delete not found ──────────────────────────────────────────────
    def test_delete_not_found(self):
        client = self._client()
        resp = client.delete("/project/brief/nonexistent-id")
        assert resp.status_code == 404

    # ── 11. Create requires name ──────────────────────────────────────────
    def test_create_requires_name(self):
        client = self._client()
        resp = client.post("/project/brief", json={"objectives": "something"})
        assert resp.status_code == 422

    # ── 12. Create all fields round-trips ─────────────────────────────────
    def test_create_all_fields(self):
        client = self._client()
        payload = {
            "project_name": "Test Full",
            "project_type": "AI Project",
            "methodology": ["PMBOK", "CPMAI"],
            "cpmai_phase": "Modeling",
            "objectives": "Build the best AI",
            "scope_items": ["ML pipeline", "API integration"],
            "out_of_scope": ["Mobile app"],
            "budget_total": 500000,
            "budget_remaining": 350000,
            "budget_currency": "EUR",
            "start_date": "2026-03-01",
            "end_date": "2026-09-30",
            "milestones": [{"name": "MVP", "date": "2026-05-01"}],
            "team": [{"name": "Alice", "role": "ML Engineer"}],
            "risks": ["Data quality", "Vendor lock-in"],
            "success_criteria": "95% accuracy on test set",
        }
        resp = client.post("/project/brief", json=payload)
        pid = resp.json()["project_id"]
        resp2 = client.get(f"/project/brief/{pid}")
        data = resp2.json()
        assert data["project_name"] == "Test Full"
        assert data["project_type"] == "AI Project"
        assert data["methodology"] == ["PMBOK", "CPMAI"]
        assert data["cpmai_phase"] == "Modeling"
        assert data["objectives"] == "Build the best AI"
        assert data["scope_items"] == ["ML pipeline", "API integration"]
        assert data["out_of_scope"] == ["Mobile app"]
        assert data["budget_total"] == 500000
        assert data["budget_remaining"] == 350000
        assert data["budget_currency"] == "EUR"
        assert data["start_date"] == "2026-03-01"
        assert data["end_date"] == "2026-09-30"
        assert data["milestones"][0]["name"] == "MVP"
        assert data["team"][0]["name"] == "Alice"
        assert data["risks"] == ["Data quality", "Vendor lock-in"]
        assert data["success_criteria"] == "95% accuracy on test set"


class TestProjectExtract:
    """POST /project/extract document extraction tests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    @staticmethod
    def _mock_extraction(extracted_json, test_func):
        """Wrap test_func with mocks for Claude API + api key."""
        mock_content = MagicMock()
        mock_content.text = json.dumps(extracted_json)
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_client_cls = MagicMock(return_value=mock_client_instance)
        import anthropic as _anthropic
        with patch.object(_anthropic, "Anthropic", mock_client_cls), \
             patch("meetingmind._api_key.load_api_key", return_value="test-key"):
            return test_func()

    _SAMPLE_EXTRACTED = {
        "project_name": "Test Extraction",
        "project_type": "Software",
        "methodology": ["Scrum"],
        "cpmai_phase": None,
        "objectives": "Build a great product",
        "scope_items": ["Backend API", "Frontend UI"],
        "out_of_scope": [],
        "budget_total": 100000,
        "budget_remaining": None,
        "budget_currency": "USD",
        "start_date": "2026-04-01",
        "end_date": None,
        "milestones": [{"name": "Beta", "date": "2026-06-01"}],
        "team": [{"name": "Bob", "role": "Dev"}],
        "risks": ["Timeline risk"],
        "success_criteria": None,
    }

    # ── 13. Extract from .md file ─────────────────────────────────────────
    def test_extract_md_file(self):
        client = self._client()
        md_content = b"# Project Phoenix\n\nObjective: Build a great product\n"
        def run():
            return client.post(
                "/project/extract",
                files=[("files", ("project.md", md_content, "text/markdown"))],
            )
        resp = self._mock_extraction(self._SAMPLE_EXTRACTED, run)
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted"]["project_name"] == "Test Extraction"
        assert "project.md" in data["files_processed"]
        assert data["found_count"] > 0

    # ── 14. Extract from .txt file ────────────────────────────────────────
    def test_extract_txt_file(self):
        client = self._client()
        txt_content = b"Project Name: Alpha\nBudget: $50,000\n"
        def run():
            return client.post(
                "/project/extract",
                files=[("files", ("brief.txt", txt_content, "text/plain"))],
            )
        resp = self._mock_extraction(self._SAMPLE_EXTRACTED, run)
        assert resp.status_code == 200
        data = resp.json()
        assert data["extracted"]["project_name"] == "Test Extraction"
        assert "brief.txt" in data["files_processed"]

    # ── 15. Unsupported file type returns 400 ─────────────────────────────
    def test_extract_unsupported_type(self):
        client = self._client()
        resp = client.post(
            "/project/extract",
            files=[("files", ("data.xlsx", b"\x00\x01\x02", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))],
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["error"]

    # ── 16. Empty file returns 400 ────────────────────────────────────────
    def test_extract_empty_file(self):
        client = self._client()
        resp = client.post(
            "/project/extract",
            files=[("files", ("empty.md", b"", "text/markdown"))],
        )
        assert resp.status_code == 400
        assert "No readable content" in resp.json()["error"]

    # ── 17. Missing fields listed correctly ───────────────────────────────
    def test_extract_missing_fields(self):
        client = self._client()
        md_content = b"# Some project docs\n"
        def run():
            return client.post(
                "/project/extract",
                files=[("files", ("doc.md", md_content, "text/markdown"))],
            )
        resp = self._mock_extraction(self._SAMPLE_EXTRACTED, run)
        assert resp.status_code == 200
        data = resp.json()
        missing = data["missing_fields"]
        # These are null/empty in _SAMPLE_EXTRACTED
        assert "cpmai_phase" in missing
        assert "budget_remaining" in missing
        assert "end_date" in missing
        assert "success_criteria" in missing
        assert "out_of_scope" in missing
        assert data["missing_count"] == len(missing)
        assert data["found_count"] + data["missing_count"] == 16  # total fields
