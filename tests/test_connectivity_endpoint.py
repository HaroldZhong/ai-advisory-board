"""GET /api/config/connectivity: network reachability + authenticated key check (audit §3, §4.3)."""
import httpx
import pytest
from fastapi.testclient import TestClient
from backend import main as main_module
from backend import openrouter_client


@pytest.fixture
def client():
    return TestClient(main_module.app)


def _stub_probe(monkeypatch, payload):
    async def fake_probe():
        return payload
    monkeypatch.setattr(openrouter_client, "check_connectivity", fake_probe)


def test_connectivity_ok_with_valid_key(client, monkeypatch):
    _stub_probe(monkeypatch, {"reachable": True, "key_valid": True, "error_kind": None, "detail": "ok"})
    body = client.get("/api/config/connectivity").json()
    assert body["reachable"] is True
    assert body["key_valid"] is True


def test_connectivity_ok_but_bad_key(client, monkeypatch):
    _stub_probe(monkeypatch, {
        "reachable": True, "key_valid": False, "error_kind": "auth",
        "detail": "Reached openrouter.ai but the API key was rejected. Check your key.",
    })
    body = client.get("/api/config/connectivity").json()
    assert body["reachable"] is True
    assert body["key_valid"] is False
    assert body["error_kind"] == "auth"


def test_connectivity_network_blocked(client, monkeypatch):
    _stub_probe(monkeypatch, {
        "reachable": False, "key_valid": None, "error_kind": "network",
        "detail": "Could not reach openrouter.ai. If you are behind a firewall "
                  "or in a region where openrouter.ai is blocked, set HTTPS_PROXY "
                  "or OPENROUTER_BASE_URL.",
    })
    body = client.get("/api/config/connectivity").json()
    assert body["reachable"] is False
    assert body["key_valid"] is None
    assert body["error_kind"] == "network"
    assert "HTTPS_PROXY" in body["detail"]


# Direct tests of the real probe's stage logic (transport mocked):

@pytest.mark.asyncio
async def test_probe_reports_bad_key_via_key_endpoint(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        if request.url.path.endswith("/key"):
            return httpx.Response(401, json={"error": {"code": 401}})
        return httpx.Response(404)

    monkeypatch.setattr(openrouter_client, "_probe_transport", httpx.MockTransport(handler))
    monkeypatch.setattr("backend.config.get_openrouter_api_key", lambda: "sk-or-bad")
    result = await openrouter_client.check_connectivity()
    assert result == {
        "reachable": True, "key_valid": False, "error_kind": "auth",
        "detail": "Reached openrouter.ai but the API key was rejected. Check your key.",
    }


@pytest.mark.asyncio
async def test_probe_tolerates_relay_without_key_endpoint(monkeypatch):
    """Relays set via OPENROUTER_BASE_URL often lack /key: reachable, key unknown, no error."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    monkeypatch.setattr(openrouter_client, "_probe_transport", httpx.MockTransport(handler))
    monkeypatch.setattr("backend.config.get_openrouter_api_key", lambda: "sk-or-anything")
    result = await openrouter_client.check_connectivity()
    assert result["reachable"] is True
    assert result["key_valid"] is None
    assert result["error_kind"] is None
    assert "key status unknown" in result["detail"]


@pytest.mark.asyncio
async def test_probe_treats_auth_gated_models_endpoint_as_reachable(monkeypatch):
    """Relays that require auth on /models still prove reachability; key check proceeds."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(401, json={"error": "auth required"})
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"label": "ok"}})
        return httpx.Response(404)

    monkeypatch.setattr(openrouter_client, "_probe_transport", httpx.MockTransport(handler))
    monkeypatch.setattr("backend.config.get_openrouter_api_key", lambda: "sk-or-good")
    result = await openrouter_client.check_connectivity()
    assert result["reachable"] is True
    assert result["key_valid"] is True
