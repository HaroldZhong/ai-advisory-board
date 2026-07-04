"""Generic OpenAI-compatible provider support (audit §8, owner decision #4).

Covers provider-kind resolution, request/catalog degradation off-OpenRouter,
ZDR gating, and the default OpenRouter path staying byte-identical.
"""
import importlib

import httpx
import pytest
from fastapi import HTTPException


def _reload_provider_modules(monkeypatch):
    """Reload config + consumers with dotenv disabled, so a developer's local
    .env cannot leak into the test (mirrors tests/test_openrouter_base_url.py)."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from backend import config, openrouter, openrouter_client
    importlib.reload(config)
    importlib.reload(openrouter)
    importlib.reload(openrouter_client)
    return config, openrouter, openrouter_client


def _restore(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    _reload_provider_modules(monkeypatch)


# ---------------------------------------------------------------------------
# Provider kind resolution
# ---------------------------------------------------------------------------

def test_default_provider_kind_is_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    assert config.provider_is_openrouter() is True
    _restore(monkeypatch)


def test_explicit_openai_compatible_kind(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openai-compatible"
    assert config.provider_is_openrouter() is False
    _restore(monkeypatch)


def test_kind_inferred_from_non_openrouter_base_url(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openai-compatible"
    _restore(monkeypatch)


def test_kind_not_inferred_when_base_url_is_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    _restore(monkeypatch)


def test_invalid_kind_value_falls_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "totally-bogus")
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    _restore(monkeypatch)


def test_explicit_kind_wins_over_inference(monkeypatch):
    """An explicit LLM_PROVIDER_KIND=openrouter overrides base-url inference."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openrouter")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    config, _openrouter, _client = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    _restore(monkeypatch)


# ---------------------------------------------------------------------------
# Request degradation (backend/openrouter.py)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_openai_compatible_never_sends_provider_field_even_with_zdr(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client = _reload_provider_modules(monkeypatch)

    captured_payloads = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_payloads.append(dict(kwargs["json"]))
            return _FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    result = await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert result["content"] == "ok"
    assert "provider" not in captured_payloads[0]
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_openrouter_default_still_sends_provider_field_when_zdr(monkeypatch):
    """Regression: the default OpenRouter path must stay byte-identical."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client = _reload_provider_modules(monkeypatch)

    captured_payloads = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_payloads.append(dict(kwargs["json"]))
            return _FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert captured_payloads[0]["provider"] == {"zdr": True}
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_400_with_reasoning_retried_once_without_it(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client = _reload_provider_modules(monkeypatch)

    # Registry model with supports_reasoning=True so `reasoning` gets added.
    reasoning_model = next(
        m["id"] for m in _config.CURATED_MODELS if m.get("supports_reasoning")
    )

    captured_payloads = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_payloads.append(dict(kwargs["json"]))
            if len(captured_payloads) == 1:
                return _FakeResponse(status_code=400)
            return _FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    result = await openrouter.query_model(
        reasoning_model,
        [{"role": "user", "content": "hi"}],
        thinking_effort="high",
    )

    assert len(captured_payloads) == 2
    assert "reasoning" in captured_payloads[0]
    assert "reasoning" not in captured_payloads[1]
    assert result["content"] == "ok"
    _restore(monkeypatch)


# ---------------------------------------------------------------------------
# Catalog enrichment degradation (backend/openrouter_client.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_compatible_skips_zdr_endpoint_fetch(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    _config, _openrouter, client = _reload_provider_modules(monkeypatch)

    async def fail_if_called():
        raise AssertionError("ZDR endpoint fetch must be skipped off-OpenRouter")

    async def fake_fetch_models():
        return [{"id": "some/model", "name": "Some Model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]

    monkeypatch.setattr(client, "fetch_openrouter_zdr_model_ids", fail_if_called)
    monkeypatch.setattr(client, "fetch_openrouter_models", fake_fetch_models)

    result = await client.get_openrouter_models_cached()

    assert result["some/model"]["supports_zdr"] is False
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_pricing_fallback_keeps_curated_pricing_when_live_lacks_it(monkeypatch):
    _config, _openrouter, client = _reload_provider_modules(monkeypatch)
    client.clear_cache()

    async def fake_fetch_models():
        # Live entry has no pricing info at all -> parse_openrouter_model gives 0/0.
        return [{"id": "curated/model", "name": "Curated Model"}]

    async def fake_fetch_zdr():
        return set()

    monkeypatch.setattr(client, "fetch_openrouter_models", fake_fetch_models)
    monkeypatch.setattr(client, "fetch_openrouter_zdr_model_ids", fake_fetch_zdr)

    curated_models = [
        {
            "id": "curated/model",
            "name": "Curated Model",
            "pricing": {"input": 3.5, "output": 14.0},
            "capabilities": [],
            "type": "council",
        }
    ]

    enriched = await client.get_enriched_models(curated_models)

    assert enriched[0]["pricing"] == {"input": 3.5, "output": 14.0}
    client.clear_cache()
    _restore(monkeypatch)


# ---------------------------------------------------------------------------
# ZDR gating (backend/main.py)
# ---------------------------------------------------------------------------

def _import_main(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from backend import config, main
    importlib.reload(config)
    importlib.reload(main)
    return main


@pytest.mark.asyncio
async def test_create_conversation_rejects_zdr_when_openai_compatible(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(topic="Test", zdr_enabled=True)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert exc.value.detail == "ZDR requires OpenRouter"
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_update_conversation_rejects_zdr_when_openai_compatible(monkeypatch, tmp_path):
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)
    conversation = await main.create_conversation(main.CreateConversationRequest(topic="Test"))

    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    importlib.reload(main.config)

    with pytest.raises(HTTPException) as exc:
        await main.update_conversation(conversation["id"], main.ConversationUpdate(zdr_enabled=True))

    assert exc.value.status_code == 400
    assert exc.value.detail == "ZDR requires OpenRouter"
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_config_status_includes_provider_kind(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)

    status = await main.get_config_status()

    assert status["provider_kind"] == "openai-compatible"
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_config_status_reports_openrouter_by_default(monkeypatch):
    main = _import_main(monkeypatch)

    status = await main.get_config_status()

    assert status["provider_kind"] == "openrouter"
    _restore(monkeypatch)
    importlib.reload(main)


# ---------------------------------------------------------------------------
# Connectivity probe: /key 404 -> key_valid null
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_key_probe_404_reports_unknown_key_status(monkeypatch):
    _config, _openrouter, client = _reload_provider_modules(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        if request.url.path.endswith("/key"):
            return httpx.Response(404)
        return httpx.Response(404)

    monkeypatch.setattr(client, "_probe_transport", httpx.MockTransport(handler))
    monkeypatch.setattr("backend.config.get_openrouter_api_key", lambda: "sk-or-anything")

    result = await client.check_connectivity()

    assert result["reachable"] is True
    assert result["key_valid"] is None
    assert result["error_kind"] is None
    assert "key status unknown" in result["detail"].lower()
    _restore(monkeypatch)


# ---------------------------------------------------------------------------
# End-to-end: a full non-stream chat turn against a mock openai-compatible
# transport succeeds.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_chat_turn_against_openai_compatible_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client = _reload_provider_modules(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello from a local model"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)
    real_async_client = openrouter.httpx.AsyncClient

    class PatchedAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", PatchedAsyncClient)

    result = await openrouter.query_model(
        "openai/gpt-4o-mini",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert result["content"] == "Hello from a local model"
    _restore(monkeypatch)
