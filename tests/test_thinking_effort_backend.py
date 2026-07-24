import importlib
from types import SimpleNamespace

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_create_conversation_stores_preset_thinking_effort(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    conversation = await main.create_conversation(
        main.CreateConversationRequest(topic="Research", preset_id="research")
    )

    assert conversation["metadata"]["preset_id"] == "research"
    assert conversation["metadata"]["thinking_effort"] == "high"


@pytest.mark.asyncio
async def test_create_conversation_honors_budget_thinking_effort_uncapped(monkeypatch, tmp_path):
    # v1.3.0 B3: a preset is a SUGGESTION, never a cap. A Budget-preset user who
    # explicitly picks X-High keeps X-High -- the old silent clamp to medium is gone.
    main = import_module_with_api_key(monkeypatch, "backend.main")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    conversation = await main.create_conversation(
        main.CreateConversationRequest(
            topic="Budget",
            preset_id="budget",
            thinking_effort="xhigh",
        )
    )

    assert conversation["metadata"]["preset_id"] == "budget"
    assert conversation["metadata"]["thinking_effort"] == "xhigh"


def test_resolve_effective_thinking_effort_precedence(monkeypatch):
    main = import_module_with_api_key(monkeypatch, "backend.main")

    conversation = {"metadata": {"preset_id": "research", "thinking_effort": "low"}}

    assert main.resolve_effective_thinking_effort(
        conversation,
        main.SendMessageRequest(content="Hi", thinking_effort="xhigh"),
    ) == "xhigh"
    assert main.resolve_effective_thinking_effort(
        conversation,
        main.SendMessageRequest(content="Hi"),
    ) == "low"
    assert main.resolve_effective_thinking_effort(
        {"metadata": {"preset_id": "research"}},
        main.SendMessageRequest(content="Hi"),
    ) == "high"
    assert main.resolve_effective_thinking_effort(
        {"metadata": {}},
        main.SendMessageRequest(content="Hi"),
    ) == "medium"
    # v1.3.0 B3: the Budget preset no longer caps effort -- a stored or requested
    # level above the (former) medium cap is honored unchanged.
    assert main.resolve_effective_thinking_effort(
        {"metadata": {"preset_id": "budget", "thinking_effort": "high"}},
        main.SendMessageRequest(content="Hi"),
    ) == "high"
    assert main.resolve_effective_thinking_effort(
        {"metadata": {"preset_id": "budget", "thinking_effort": "low"}},
        main.SendMessageRequest(content="Hi", thinking_effort="xhigh"),
    ) == "xhigh"


@pytest.mark.asyncio
async def test_update_conversation_honors_budget_thinking_effort_uncapped(monkeypatch, tmp_path):
    # v1.3.0 B3: updating a Budget-preset conversation to High keeps High (no clamp).
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-budget-effort-uncapped"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"preset_id": "budget", "thinking_effort": "low"},
    )

    updated = await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(thinking_effort="high"),
    )

    assert updated["metadata"]["thinking_effort"] == "high"


@pytest.mark.asyncio
async def test_query_model_sends_reasoning_effort_only_for_supported_models(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {},
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
        "anthropic/claude-opus-4.7",
        [{"role": "user", "content": "hi"}],
        thinking_effort="high",
    )
    await openrouter.query_model(
        "openai/gpt-4o-mini",
        [{"role": "user", "content": "hi"}],
        thinking_effort="high",
    )

    assert captured_payloads[0]["reasoning"] == {"effort": "high"}
    assert "reasoning" not in captured_payloads[1]


@pytest.mark.asyncio
async def test_parallel_model_queries_propagate_thinking_effort(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured = []

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False, thinking_effort=None):
        captured.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
                "thinking_effort": thinking_effort,
            }
        )
        return {"content": f"response from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    result = await openrouter.query_models_parallel(
        ["model-a", "model-b"],
        [{"role": "user", "content": "hi"}],
        thinking_effort="high",
    )

    assert set(result.keys()) == {"model-a", "model-b"}
    assert [call["thinking_effort"] for call in captured] == ["high", "high"]


@pytest.mark.asyncio
async def test_council_stage_calls_pass_thinking_effort(monkeypatch):
    council = import_module_with_api_key(monkeypatch, "backend.council")
    parallel_calls = []
    single_calls = []

    async def fake_query_models_parallel(models, messages, zdr_enabled=False, thinking_effort=None):
        parallel_calls.append(
            {
                "models": models,
                "messages": messages,
                "zdr_enabled": zdr_enabled,
                "thinking_effort": thinking_effort,
            }
        )
        return {
            model: {"content": f"response from {model}", "usage": {}}
            for model in models
        }

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False, thinking_effort=None):
        single_calls.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
                "thinking_effort": thinking_effort,
            }
        )
        return {"content": "chairman response", "usage": {}}

    monkeypatch.setattr(council, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council, "query_model", fake_query_model)

    stage1_results = await council.stage1_collect_responses(
        "Question?",
        models=["model-a", "model-b"],
        thinking_effort="low",
    )
    stage2_results, label_to_model = await council.stage2_collect_rankings(
        "Question?",
        stage1_results,
        models=["model-a", "model-b"],
        thinking_effort="low",
    )
    await council.stage3_synthesize_final(
        "Question?",
        stage1_results,
        stage2_results,
        label_to_model,
        {"model-a": {"consensus_score": 1.0, "avg_rank": 1.0}},
        chairman_model="chair-model",
        thinking_effort="low",
    )

    assert [call["thinking_effort"] for call in parallel_calls] == ["low", "low"]
    # v1.3.0 B3: Stage-3 no longer floors effort to medium -- the requested "low"
    # reaches the chairman unchanged (user owns their effort).
    assert [call["thinking_effort"] for call in single_calls] == ["low"]


@pytest.mark.asyncio
async def test_stage3_passes_requested_effort_uncapped_after_b3(monkeypatch):
    # v1.3.0 B3: the Stage-3 per-model effort cap is retired. High effort now reaches
    # EVERY chairman -- including the formerly-capped lightweight flash-lite -- unchanged.
    council = import_module_with_api_key(monkeypatch, "backend.council")
    captured = []

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False, thinking_effort=None):
        captured.append(
            {
                "model": model,
                "thinking_effort": thinking_effort,
            }
        )
        return {"content": "chairman response", "usage": {}}

    monkeypatch.setattr(council, "query_model", fake_query_model)

    stage1_results = [{"model": "model-a", "response": "First answer"}]
    stage2_results = [{"model": "model-a", "ranking": "A"}]
    quality_metrics = {"model-a": {"consensus_score": 1.0, "avg_rank": 1.0}}

    await council.stage3_synthesize_final(
        "Question?",
        stage1_results,
        stage2_results,
        {"A": "model-a"},
        quality_metrics,
        chairman_model="google/gemini-2.5-flash-lite",
        thinking_effort="high",
    )
    await council.stage3_synthesize_final(
        "Question?",
        stage1_results,
        stage2_results,
        {"A": "model-a"},
        quality_metrics,
        chairman_model="anthropic/claude-opus-4.7",
        thinking_effort="high",
    )

    assert captured == [
        {"model": "google/gemini-2.5-flash-lite", "thinking_effort": "high"},
        {"model": "anthropic/claude-opus-4.7", "thinking_effort": "high"},
    ]


@pytest.mark.asyncio
async def test_sync_chat_passes_resolved_thinking_effort_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-sync-thinking"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.7", "thinking_effort": "high"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Thinking response", "usage": {}}

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

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert captured_kwargs["thinking_effort"] == "high"


@pytest.mark.asyncio
async def test_stream_chat_passes_request_thinking_effort_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-thinking"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.7", "thinking_effort": "low"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Thinking response", "usage": {}}

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

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", thinking_effort="xhigh"),
    )

    async for _chunk in response.body_iterator:
        pass

    assert captured_kwargs["thinking_effort"] == "xhigh"
