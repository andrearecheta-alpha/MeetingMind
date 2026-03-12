"""
tests/test_kb_seed.py
---------------------
Tests for POST /knowledge/seed and GET /knowledge/seed/status endpoints.

Uses FastAPI TestClient — no hardware, no real Whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TestKBSeed:
    """POST /knowledge/seed and GET /knowledge/seed/status endpoint tests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_seed_status(self):
        import meetingmind.main as m
        m._seed_status.update({"facts_count": 0, "last_seeded": None, "entities": {}})

    def test_seed_happy_path(self):
        """POST with valid content returns 200 with facts_stored."""
        self._reset_seed_status()
        client = self._client()
        resp = client.post("/knowledge/seed", json={
            "content": "Budget over $10,000 needs HH approval.\n\nSarah handles legal reviews.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "facts_stored" in data
        assert data["facts_stored"] >= 1
        assert "entities_extracted" in data
        self._reset_seed_status()

    def test_seed_empty_content_400(self):
        """POST with empty content returns 400."""
        client = self._client()
        resp = client.post("/knowledge/seed", json={"content": ""})
        assert resp.status_code == 400

    def test_seed_whitespace_only_400(self):
        """POST with whitespace-only content returns 400."""
        client = self._client()
        resp = client.post("/knowledge/seed", json={"content": "   \n\n   "})
        assert resp.status_code == 400

    def test_seed_custom_label(self):
        """POST with custom label returns 200."""
        self._reset_seed_status()
        client = self._client()
        resp = client.post("/knowledge/seed", json={
            "content": "ACOS above 6% is a red flag.",
            "label": "metrics_brief",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["facts_stored"] >= 1
        self._reset_seed_status()

    def test_seed_extracts_entities(self):
        """POST with rich content extracts entities (people, orgs, money, percentages)."""
        self._reset_seed_status()
        client = self._client()
        resp = client.post("/knowledge/seed", json={
            "content": (
                "Sarah handles legal reviews for Acme Corp. "
                "Budget is $50,000 and ACOS must stay below 6%."
            ),
        })
        assert resp.status_code == 200
        data = resp.json()
        entities = data["entities_extracted"]
        # Should have at least some entity categories populated
        has_entities = any(
            isinstance(v, list) and len(v) > 0
            for v in entities.values()
        )
        assert has_entities, f"Expected entities, got: {entities}"
        self._reset_seed_status()

    def test_status_returns_expected_shape(self):
        """GET /knowledge/seed/status returns facts_count, last_seeded, entities."""
        self._reset_seed_status()
        client = self._client()
        resp = client.get("/knowledge/seed/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "facts_count" in data
        assert "last_seeded" in data
        assert "entities" in data
        self._reset_seed_status()

    def test_status_after_seed_reflects_seed(self):
        """GET status after a POST seed reflects the new facts."""
        self._reset_seed_status()
        client = self._client()

        # Seed first
        client.post("/knowledge/seed", json={
            "content": "Ahmed manages PPC campaigns.\n\nInventory below 15 days is critical.",
        })

        # Check status
        resp = client.get("/knowledge/seed/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["facts_count"] >= 2
        assert data["last_seeded"] is not None
        self._reset_seed_status()
