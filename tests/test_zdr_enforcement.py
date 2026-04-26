import importlib
from types import SimpleNamespace

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_query_model_adds_provider_zdr_when_enabled(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_payloads.append(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )
    await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
    )

    assert captured_payloads[0]["provider"] == {"zdr": True}
    assert "provider" not in captured_payloads[1]


@pytest.mark.asyncio
async def test_parallel_model_queries_propagate_zdr(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured = []

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False):
        captured.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
            }
        )
        return {"content": f"response from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    result = await openrouter.query_models_parallel(
        ["model-a", "model-b"],
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert set(result.keys()) == {"model-a", "model-b"}
    assert [call["zdr_enabled"] for call in captured] == [True, True]


@pytest.mark.asyncio
async def test_council_stage_calls_pass_zdr_to_openrouter(monkeypatch):
    council = import_module_with_api_key(monkeypatch, "backend.council")
    parallel_calls = []
    single_calls = []

    async def fake_query_models_parallel(models, messages, zdr_enabled=False):
        parallel_calls.append(
            {"models": models, "messages": messages, "zdr_enabled": zdr_enabled}
        )
        return {
            model: {"content": f"response from {model}", "usage": {}}
            for model in models
        }

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False):
        single_calls.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
            }
        )
        return {"content": "chairman response", "usage": {}}

    monkeypatch.setattr(council, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council, "query_model", fake_query_model)

    stage1_results = await council.stage1_collect_responses(
        "Question?",
        models=["model-a", "model-b"],
        zdr_enabled=True,
    )
    stage2_results, label_to_model = await council.stage2_collect_rankings(
        "Question?",
        stage1_results,
        models=["model-a", "model-b"],
        zdr_enabled=True,
    )
    await council.stage3_synthesize_final(
        "Question?",
        stage1_results,
        stage2_results,
        label_to_model,
        {"model-a": {"consensus_score": 1.0, "avg_rank": 1.0}},
        chairman_model="chair-model",
        zdr_enabled=True,
    )
    await council.run_tool_steward_phase(
        "Question?",
        run_id="run-1",
        chairman_model="chair-model",
        zdr_enabled=True,
    )

    assert [call["zdr_enabled"] for call in parallel_calls] == [True, True]
    assert [call["zdr_enabled"] for call in single_calls] == [True, True]


@pytest.mark.asyncio
async def test_sync_chat_passes_zdr_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-sync-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
    )

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_stream_chat_passes_zdr_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
    )

    async for _chunk in response.body_iterator:
        pass

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_web_search_passes_zdr_to_openrouter(monkeypatch):
    web_search = import_module_with_api_key(monkeypatch, "backend.web_search")
    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Search result https://example.com", "usage": {}}

    monkeypatch.setattr(web_search, "query_model", fake_query_model)

    await web_search.web_search_stage0("current facts", zdr_enabled=True)

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_reasoning_rag_passes_zdr_to_openrouter(monkeypatch, tmp_path):
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")
    rag = rag_module.CouncilRAG(persist_path=str(tmp_path))
    captured_kwargs = {}

    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": "Useful prior memory"}],
        },
    }

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Relevant prior memory"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    result = await rag.retrieve_async("current question", "current", zdr_enabled=True)

    assert result == "Relevant prior memory"
    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_image_file_processing_passes_zdr_to_openrouter(monkeypatch):
    file_processing = import_module_with_api_key(monkeypatch, "backend.file_processing")
    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Image description"}

    monkeypatch.setattr(file_processing, "query_model", fake_query_model)

    result = await file_processing.process_file(
        b"fake-image-bytes",
        "diagram.png",
        "image/png",
        zdr_enabled=True,
    )

    assert result.status == "success"
    assert captured_kwargs["zdr_enabled"] is True
