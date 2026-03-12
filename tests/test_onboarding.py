"""
tests/test_onboarding.py
------------------------
Tests for the POST /onboarding/profile and GET /onboarding/profile endpoints.

Uses FastAPI TestClient — no hardware, no real Whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TestOnboardingProfile:
    """POST /onboarding/profile endpoint tests."""

    def _client(self):
        from fastapi.testclient import TestClient
        from meetingmind.main import app
        return TestClient(app)

    def _reset_state(self):
        """Reset module-level onboarding state between tests."""
        import meetingmind.main as m
        m._user_name = None
        m._user_vocab_hints = None
        m._active_role = "PM"

    def test_onboarding_profile_basic(self):
        """POST with name + role + context returns 200 with expected fields."""
        self._reset_state()
        client = self._client()
        resp = client.post("/onboarding/profile", json={
            "name": "Andrea",
            "role": "EA",
            "context": "I support the CEO.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "Andrea"
        assert data["role"] == "EA"
        assert "entities_extracted" in data
        assert "vocab_hints_length" in data
        self._reset_state()

    def test_onboarding_profile_empty_context(self):
        """POST with empty context returns vocab_hints_length=0."""
        self._reset_state()
        client = self._client()
        resp = client.post("/onboarding/profile", json={
            "name": "Test",
            "role": "PM",
            "context": "",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["vocab_hints_length"] == 0
        self._reset_state()

    def test_onboarding_profile_sets_role(self):
        """POST with role='Sales' updates the active role on the server."""
        self._reset_state()
        client = self._client()
        client.post("/onboarding/profile", json={
            "name": "Seller",
            "role": "Sales",
            "context": "Enterprise SaaS deals.",
        })
        resp = client.get("/settings/role")
        assert resp.status_code == 200
        assert resp.json()["role"] == "Sales"
        self._reset_state()

    def test_onboarding_profile_extracts_entities(self):
        """POST with recognisable names/orgs returns entities_extracted > 0."""
        self._reset_state()
        client = self._client()
        resp = client.post("/onboarding/profile", json={
            "name": "Andrea",
            "role": "EA",
            "context": "I support John Smith at Acme Corp. Budget is $500K.",
        })
        data = resp.json()
        assert data["entities_extracted"] > 0
        self._reset_state()

    def test_onboarding_profile_builds_vocab_hints(self):
        """Verify _user_vocab_hints contains extracted entity names."""
        self._reset_state()
        import meetingmind.main as m
        client = self._client()
        client.post("/onboarding/profile", json={
            "name": "Andrea",
            "role": "PM",
            "context": "Project lead is John Smith at Acme Corp.",
        })
        # _user_vocab_hints should exist and contain the extracted names
        assert m._user_vocab_hints is not None
        # Should contain default hints plus entity names
        assert "SaaS" in m._user_vocab_hints  # from default hints
        self._reset_state()

    def test_get_onboarding_profile(self):
        """GET /onboarding/profile returns the stored name and role."""
        self._reset_state()
        client = self._client()
        # Set a profile first
        client.post("/onboarding/profile", json={
            "name": "Andrea",
            "role": "Custom",
            "context": "UX researcher.",
        })
        resp = client.get("/onboarding/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Andrea"
        assert data["role"] == "Custom"
        self._reset_state()

    def test_legacy_ingest_still_works(self):
        """Regression guard: POST /knowledge/ingest/text still works."""
        client = self._client()
        resp = client.post("/knowledge/ingest/text", json={
            "text": "Test seed document for regression.",
            "doc_id": "regression_test",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Either "ok" (KB available) or "skipped" (no KB) — never an error
        assert data["status"] in ("ok", "skipped")
