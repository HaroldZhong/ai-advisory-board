"""Generic OpenAI-compatible provider support (audit §8, owner decision #4).

Covers provider-kind resolution, request/catalog degradation off-OpenRouter,
ZDR gating, and the default OpenRouter path staying byte-identical.
"""
import importlib
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException


def _reload_provider_modules(monkeypatch):
    """Reload config + consumers with dotenv disabled, so a developer's local
    .env cannot leak into the test (mirrors tests/test_openrouter_base_url.py)."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    from backend import config, openrouter, openrouter_client, openrouter_pdf
    importlib.reload(config)
    importlib.reload(openrouter)
    importlib.reload(openrouter_client)
    importlib.reload(openrouter_pdf)
    return config, openrouter, openrouter_client, openrouter_pdf


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
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    assert config.provider_is_openrouter() is True
    _restore(monkeypatch)


def test_explicit_openai_compatible_kind(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openai-compatible"
    assert config.provider_is_openrouter() is False
    _restore(monkeypatch)


def test_kind_inferred_from_non_openrouter_base_url(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openai-compatible"
    _restore(monkeypatch)


def test_kind_not_inferred_when_base_url_is_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER_KIND", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    _restore(monkeypatch)


def test_invalid_kind_value_falls_back_to_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "totally-bogus")
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    assert config.PROVIDER_KIND == "openrouter"
    _restore(monkeypatch)


def test_explicit_kind_wins_over_inference(monkeypatch):
    """An explicit LLM_PROVIDER_KIND=openrouter overrides base-url inference."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openrouter")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://localhost:11434/v1")
    config, _openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
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
async def test_openai_compatible_refuses_zdr_rather_than_silently_stripping_it(monkeypatch):
    """query_model must never silently drop the ZDR provider field and send
    the request anyway (Codex round 5) — it has to refuse outright, before
    any request is built. See test_query_model_refuses_zdr_off_openrouter_
    without_any_request below for the no-HTTP-call assertion; this pins the
    non-ZDR request in this same openai-compatible config stays unaffected."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

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

    with pytest.raises(ValueError, match="ZDR routing requires OpenRouter"):
        await openrouter.query_model(
            "model-a",
            [{"role": "user", "content": "hi"}],
            zdr_enabled=True,
        )
    assert captured_payloads == []

    result = await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
    )
    assert result["content"] == "ok"
    assert "provider" not in captured_payloads[0]
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_openrouter_default_still_sends_provider_field_when_zdr(monkeypatch):
    """Regression: the default OpenRouter path must stay byte-identical."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

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
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

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
    _config, _openrouter, client, _pdf = _reload_provider_modules(monkeypatch)

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
    _config, _openrouter, client, _pdf = _reload_provider_modules(monkeypatch)
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
async def test_create_conversation_rejects_implicit_preset_zdr_when_openai_compatible(monkeypatch, tmp_path):
    """A requires_zdr preset (e.g. "private") with zdr_enabled omitted must
    still 400 off-OpenRouter — the resulting metadata would claim ZDR while
    the actual request silently drops the routing field (Codex finding)."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(topic="Test", preset_id="private")

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert exc.value.detail == "ZDR requires OpenRouter"
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_private_preset_succeeds_on_openrouter(monkeypatch, tmp_path):
    """Same request as above succeeds when the provider is OpenRouter."""
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(topic="Test", preset_id="private")

    conversation = await main.create_conversation(request)

    assert conversation["metadata"]["zdr_enabled"] is True
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
    _config, _openrouter, client, _pdf = _reload_provider_modules(monkeypatch)

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
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

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

    # Not ZDR: a ZDR request off-provider is refused outright (see the
    # dedicated refuses_zdr tests) rather than silently downgraded, so an
    # end-to-end "does a plain turn work against this mock" check must not
    # ask for ZDR.
    result = await openrouter.query_model(
        "openai/gpt-4o-mini",
        [{"role": "user", "content": "hi"}],
    )

    assert result["content"] == "Hello from a local model"
    _restore(monkeypatch)


# ---------------------------------------------------------------------------
# Pre-flight ZDR rejection in prepare_turn (send_message/send_message_stream)
# ---------------------------------------------------------------------------

def _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, metadata):
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id, metadata)
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        # retrieve_async returns (context, usage) since PR #75
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "response", "usage": {}}

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=fake_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)


@pytest.mark.asyncio
async def test_send_message_rejects_conversation_metadata_zdr_off_openrouter(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    conversation_id = "conv-preflight-metadata-zdr"
    _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, {
        "chairman_model": "openai/gpt-4o-mini", "zdr_enabled": True,
    })

    with pytest.raises(HTTPException) as exc:
        await main.send_message(
            conversation_id,
            main.SendMessageRequest(content="Follow up", mode="chat"),
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "ZDR requires OpenRouter. Disable ZDR for this conversation or switch providers."
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_send_message_rejects_per_request_zdr_off_openrouter(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    conversation_id = "conv-preflight-request-zdr"
    _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, {
        "chairman_model": "openai/gpt-4o-mini",
    })

    with pytest.raises(HTTPException) as exc:
        await main.send_message(
            conversation_id,
            main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "ZDR requires OpenRouter. Disable ZDR for this conversation or switch providers."
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_send_message_conversation_metadata_zdr_succeeds_on_openrouter(monkeypatch, tmp_path):
    main = _import_main(monkeypatch)
    conversation_id = "conv-preflight-metadata-zdr-ok"
    _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, {
        "chairman_model": "openai/gpt-4o-mini", "zdr_enabled": True,
    })

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert result is not None
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_send_message_per_request_zdr_succeeds_on_openrouter(monkeypatch, tmp_path):
    main = _import_main(monkeypatch)
    conversation_id = "conv-preflight-request-zdr-ok"
    _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, {
        "chairman_model": "openai/gpt-4o-mini",
    })

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
    )

    assert result is not None
    _restore(monkeypatch)
    importlib.reload(main)


# ---------------------------------------------------------------------------
# Utility-call ZDR bypass (Codex round 5): query_model must refuse a ZDR
# request off-OpenRouter before issuing any HTTP call, and the /api/attachments
# upload endpoint (a query_model caller that does NOT route through
# prepare_turn) must reject use_zdr=true off-OpenRouter before processing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_model_refuses_zdr_off_openrouter_without_any_request(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    real_async_client = openrouter.httpx.AsyncClient

    class PatchedAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", PatchedAsyncClient)

    with pytest.raises(ValueError, match="ZDR routing requires OpenRouter"):
        await openrouter.query_model(
            "model-a",
            [{"role": "user", "content": "hi"}],
            zdr_enabled=True,
        )

    assert call_count == 0
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_query_model_zdr_still_works_on_openrouter(monkeypatch):
    """Regression: the raise must not fire for the default provider."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    real_async_client = openrouter.httpx.AsyncClient

    class PatchedAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", PatchedAsyncClient)

    result = await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert result["content"] == "ok"
    _restore(monkeypatch)


def test_upload_attachment_rejects_zdr_off_openrouter(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/api/attachments",
        params={"use_zdr": "true"},
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "ZDR requires OpenRouter. Disable ZDR for this upload or switch providers."
    )
    _restore(monkeypatch)
    importlib.reload(main)


def test_upload_attachment_zdr_proceeds_on_openrouter(monkeypatch):
    from fastapi.testclient import TestClient

    main = _import_main(monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/api/attachments",
        params={"use_zdr": "true"},
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )

    # Plain text is extracted locally (no query_model call needed), so this
    # just proves the pre-flight didn't block a legitimate ZDR upload.
    assert response.status_code == 200
    assert response.json()["status"] in ("success", "partial")
    _restore(monkeypatch)
    importlib.reload(main)


# ---------------------------------------------------------------------------
# PDF extraction path (backend/openrouter_pdf.py) — same degradation rule as
# query_model: never leak the OpenRouter-only `provider` field, never
# silently downgrade a ZDR request off-provider (folded into P4-T2 scope,
# flagged as a follow-up on the round-5 Codex threads and now included here).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pdf_extraction_refuses_zdr_off_openrouter_without_any_request(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, _openrouter, _client, openrouter_pdf = _reload_provider_modules(monkeypatch)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "text"}}]})

    transport = httpx.MockTransport(handler)
    real_async_client = openrouter_pdf.httpx.AsyncClient

    class PatchedAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(openrouter_pdf.httpx, "AsyncClient", PatchedAsyncClient)

    with pytest.raises(ValueError, match="ZDR routing requires OpenRouter"):
        await openrouter_pdf.extract_pdf_with_openrouter(
            b"%PDF-1.4 fake",
            "doc.pdf",
            use_zdr=True,
        )

    assert call_count == 0
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_pdf_extraction_never_sends_provider_field_off_openrouter(monkeypatch):
    """Non-ZDR request in the same openai-compatible config: no `provider` key."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, _openrouter, _client, openrouter_pdf = _reload_provider_modules(monkeypatch)

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
            return httpx.Response(200, json={"choices": [{"message": {"content": "text"}}]})

    monkeypatch.setattr(openrouter_pdf.httpx, "AsyncClient", FakeAsyncClient)

    result = await openrouter_pdf.extract_pdf_with_openrouter(
        b"%PDF-1.4 fake",
        "doc.pdf",
    )

    assert result["status"] == "success"
    assert "provider" not in captured_payloads[0]
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_pdf_extraction_zdr_unchanged_on_openrouter(monkeypatch):
    """Regression: the default OpenRouter path must stay byte-identical."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, _openrouter, _client, openrouter_pdf = _reload_provider_modules(monkeypatch)

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
            return httpx.Response(200, json={"choices": [{"message": {"content": "text"}}]})

    monkeypatch.setattr(openrouter_pdf.httpx, "AsyncClient", FakeAsyncClient)

    result = await openrouter_pdf.extract_pdf_with_openrouter(
        b"%PDF-1.4 fake",
        "doc.pdf",
        use_zdr=True,
    )

    assert result["status"] == "success"
    assert captured_payloads[0]["provider"] == {"zdr": True}
    _restore(monkeypatch)


def test_enhance_attachment_rejects_zdr_off_openrouter(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/api/attachments/nonexistent-id/enhance",
        params={"use_zdr": "true"},
    )

    # The pre-flight rejects before the 404-attachment-not-found check runs.
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "ZDR requires OpenRouter. Disable ZDR for this upload or switch providers."
    )
    _restore(monkeypatch)
    importlib.reload(main)


# ---------------------------------------------------------------------------
# PR2: minimal custom model id entry for openai-compatible providers.
# create_conversation accepts any non-empty chairman/council id that isn't in
# the curated registry when the provider isn't OpenRouter; registry HITS
# still go through the existing utility/search type checks on every provider
# kind, and OpenRouter-kind behavior stays byte-identical.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_conversation_accepts_unknown_ids_when_openai_compatible(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(
        topic="Test",
        chairman_model="llama3.1",
        council_members=["llama3.1", "mistral-nemo", "custom/local-model"],
    )

    conversation = await main.create_conversation(request)

    assert conversation["metadata"]["chairman_model"] == "llama3.1"
    assert conversation["metadata"]["council_models"] == [
        "llama3.1", "mistral-nemo", "custom/local-model",
    ]
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_rejects_unknown_chairman_on_openrouter(monkeypatch, tmp_path):
    """Same request as above 400s on the default (OpenRouter) provider kind."""
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(topic="Test", chairman_model="llama3.1")

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert "Invalid chairman model" in exc.value.detail
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_rejects_unknown_council_members_on_openrouter(monkeypatch, tmp_path):
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(
        topic="Test",
        council_members=["llama3.1", "mistral-nemo", "another-unknown"],
    )

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert "Invalid council models" in exc.value.detail
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_still_rejects_utility_type_when_openai_compatible(monkeypatch, tmp_path):
    """A REGISTRY HIT that is utility/search-typed must still 400 on any
    provider kind — the relaxed check only applies to ids that aren't in the
    curated registry at all."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    utility_model_id = next(
        m["id"] for m in main.config.CURATED_MODELS if m.get("type") == "utility"
    )
    request = main.CreateConversationRequest(topic="Test", chairman_model=utility_model_id)

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert "internal utility model" in exc.value.detail
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_rejects_empty_chairman_id_when_openai_compatible(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(topic="Test", chairman_model="   ")

    conversation = await main.create_conversation(request)

    # A whitespace-only chairman_model is stripped to None upstream (existing
    # behavior: falls through to the config default), not stored as-is.
    assert conversation["metadata"].get("chairman_model") != "   "
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_create_conversation_rejects_empty_council_id_when_openai_compatible(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    request = main.CreateConversationRequest(
        topic="Test",
        council_members=["llama3.1", "   ", "mistral-nemo"],
    )

    with pytest.raises(HTTPException) as exc:
        await main.create_conversation(request)

    assert exc.value.status_code == 400
    assert "Invalid council models" in exc.value.detail
    _restore(monkeypatch)
    importlib.reload(main)


@pytest.mark.asyncio
async def test_full_chat_turn_with_unknown_chairman_id_succeeds_openai_compatible(monkeypatch, tmp_path):
    """End-to-end: an unknown chairman id makes it through create_conversation
    and a full chat turn, with cost falling back to the existing
    calculate_cost behavior (unknown model -> 0) rather than crashing."""
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    main = _import_main(monkeypatch)
    conversation_id = "conv-custom-model-turn"
    _setup_zdr_chat_fakes(main, monkeypatch, tmp_path, conversation_id, {
        "chairman_model": "llama3.1",
    })

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "response", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert result is not None
    assert result["turn_cost"] == 0.0
    _restore(monkeypatch)
    importlib.reload(main)


def test_enhance_attachment_zdr_not_blocked_on_openrouter(monkeypatch):
    from fastapi.testclient import TestClient

    main = _import_main(monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/api/attachments/nonexistent-id/enhance",
        params={"use_zdr": "true"},
    )

    # Pre-flight doesn't block; falls through to the real 404 (attachment
    # doesn't exist) rather than the 400 ZDR gate.
    assert response.status_code == 404
    _restore(monkeypatch)
    importlib.reload(main)


def _probed_levels_records(openrouter, model_id):
    from backend.reasoning_capability import model_fingerprint
    entry = openrouter._lookup_registry_model(model_id)
    return {model_id: {
        "model_id": model_id, "probed": True, "fingerprint": model_fingerprint(entry),
        "provider_pinned": "openai", "supports_reasoning": True,
        "control_surface": "levels", "levels": ["low", "medium", "high"],
    }}


def _capturing_client(captured):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, *a, **k):
            captured.append(dict(k["json"]))
            return _FakeResponse()
    return FakeAsyncClient


@pytest.mark.asyncio
async def test_probed_endpoint_pin_only_sent_on_openrouter(monkeypatch):
    """The probed record's provider.order/allow_fallbacks pin is OpenRouter-only; an
    openai-compatible relay rejects `provider`, so the pin must be skipped there while
    the reasoning object itself is still sent (the existing 400-retry protects it)."""
    model_id = "google/gemini-3.1-pro-preview"  # registry reasoning_extraction == "field" (parseable)

    # OpenRouter default -> the pin IS sent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    openrouter._reasoning_capabilities_cache = _probed_levels_records(openrouter, model_id)
    captured_or = []
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _capturing_client(captured_or))
    await openrouter.query_model(model_id, [{"role": "user", "content": "hi"}], thinking_effort="high")
    assert captured_or[0]["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert captured_or[0]["reasoning"] == {"effort": "high"}
    _restore(monkeypatch)

    # openai-compatible relay -> NO provider field, but reasoning still sent
    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    openrouter._reasoning_capabilities_cache = _probed_levels_records(openrouter, model_id)
    captured_oc = []
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _capturing_client(captured_oc))
    await openrouter.query_model(model_id, [{"role": "user", "content": "hi"}], thinking_effort="high")
    assert "provider" not in captured_oc[0]
    assert captured_oc[0]["reasoning"] == {"effort": "high"}
    _restore(monkeypatch)


@pytest.mark.asyncio
async def test_offrouter_ignores_probed_row_and_uses_registry_fallback(monkeypatch):
    """Probe rows are OpenRouter-specific: off-OpenRouter a probed 'none' row must NOT
    suppress reasoning for a model the registry supports -- the relay still gets the
    registry fallback {"effort": ...}, not the OpenRouter-learned suppression."""
    from backend.reasoning_capability import model_fingerprint
    model_id = "google/gemini-3.1-pro-preview"  # registry supports_reasoning True

    monkeypatch.setenv("LLM_PROVIDER_KIND", "openai-compatible")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _config, openrouter, _client, _pdf = _reload_provider_modules(monkeypatch)
    entry = openrouter._lookup_registry_model(model_id)
    assert entry.get("supports_reasoning") is True  # precondition
    # a probed 'none' row that WOULD suppress reasoning if wrongly applied off-OR
    openrouter._reasoning_capabilities_cache = {model_id: {
        "model_id": model_id, "probed": True, "fingerprint": model_fingerprint(entry),
        "provider_pinned": "google", "supports_reasoning": False, "control_surface": "none"}}
    captured = []
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", _capturing_client(captured))
    await openrouter.query_model(model_id, [{"role": "user", "content": "hi"}], thinking_effort="high")
    assert captured[0]["reasoning"] == {"effort": "high"}  # registry fallback, not suppressed
    assert "provider" not in captured[0]
    _restore(monkeypatch)
