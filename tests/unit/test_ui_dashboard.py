"""Unit test for Inference 2.0 Web Dashboard & Content Negotiation."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ui_dashboard_direct_endpoint():
    """Verify /ui endpoint serves the single-page web dashboard HTML."""
    res = client.get("/ui")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in res.text
    assert "INFERENCE" in res.text
    assert "Multi-Agent Consultation" in res.text


def test_root_browser_content_negotiation():
    """Verify root / serves HTML when accessed from a web browser."""
    res = client.get("/", headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in res.text
    assert "Inference — Multi-Agent Intelligence System" in res.text


def test_root_api_json_content_negotiation():
    """Verify root / serves JSON when accessed programmatically without browser HTML accept."""
    res = client.get("/", headers={"Accept": "application/json"})
    assert res.status_code == 200
    assert "application/json" in res.headers.get("content-type", "")
    data = res.json()
    assert data["name"] == "Inference"
    assert data["status"] == "online"
    assert data["version"] == "2.0.0"
