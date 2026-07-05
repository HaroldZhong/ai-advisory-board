import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


def stub_sync_council_dependencies(main, monkeypatch, captured=None):
    """Stub the council pipeline at its stage seams (turn_pipeline.run_turn
    dispatches through backend.main, so patching main covers both endpoints)."""
    from backend.tools.types import EvidencePack

    async def fake_generate_conversation_title(*args, **kwargs):
        return "Council title"

    async def fake_run_tool_steward_phase(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="Run the council"), None

    # stage1_collect_responses_progressive is the pipeline seam (P3-T6): an
    # async generator yielding ("model_complete", index, result) per model
    # then ("complete", stage1_results, None) with the full list.
    async def fake_stage1_progressive(*args, **kwargs):
        result = {"model": "model-a", "response": "Answer A", "usage": {}}
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{"model": "model-a", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "model-a"},
        )

    async def fake_stage3(*args, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        return {"model": "chair", "response": "Final answer", "usage": {}}

    async def fake_extract_topics(*args, **kwargs):
        return (["planning"], {})

    monkeypatch.setattr(main, "generate_conversation_title", fake_generate_conversation_title)
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_run_tool_steward_phase)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr("backend.council.calculate_quality_metrics", Mock(return_value={"model-a": {}}))

    async def fake_index_session(*args, **kwargs):
        return None

    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(index_session=fake_index_session, refresh_hybrid_index=Mock()),
    )


def test_run_plan_applies_advanced_overrides(monkeypatch):
    budget_router = import_module_with_api_key(monkeypatch, "backend.budget_router")

    monkeypatch.setattr(budget_router, "get_budget_spent_percentage", lambda *_: None)
    monkeypatch.setattr(
        budget_router,
        "get_session_policy",
        lambda *_: {"budget_usd": None},
    )

    run_plan = budget_router.create_run_plan(
        query="summarize this",
        conversation_id="conv-advanced-plan",
        chairman_model="anthropic/claude-opus-4.5",
        execution_mode="research",
        rag_preset="max",
        model_tier="budget",
    )

    assert run_plan.mode == "research"
    assert run_plan.rag_preset == "max"
    assert run_plan.rag_max_tokens == 32000
    assert run_plan.model_tier == "budget"
    assert run_plan.chairman_model == "google/gemini-2.5-flash-lite"
    assert run_plan.policy_reason == "advanced_override"


@pytest.mark.asyncio
async def test_send_message_rejects_invalid_advanced_settings(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-invalid-advanced"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)

    with pytest.raises(main.HTTPException) as exc:
        await main.send_message(
            conversation_id,
            main.SendMessageRequest(
                content="Hello",
                mode="chat",
                execution_mode="turbo",
            ),
        )

    assert exc.value.status_code == 400
    assert "execution_mode" in exc.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_mode", "research"),
        ("rag_preset", "max"),
    ],
)
async def test_sync_council_rejects_chat_only_advanced_settings(
    monkeypatch,
    tmp_path,
    field,
    value,
):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = f"conv-sync-council-{field}"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")
    stub_sync_council_dependencies(main, monkeypatch)

    request_kwargs = {field: value}
    with pytest.raises(main.HTTPException) as exc:
        await main.send_message(
            conversation_id,
            main.SendMessageRequest(
                content="Run the council",
                mode="council",
                **request_kwargs,
            ),
        )

    assert exc.value.status_code == 400
    assert field in exc.value.detail
    assert "chat mode" in exc.value.detail


@pytest.mark.asyncio
async def test_stream_council_rejects_chat_only_advanced_settings(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-council-advanced"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)

    with pytest.raises(main.HTTPException) as exc:
        await main.send_message_stream(
            conversation_id,
            main.SendMessageRequest(
                content="Run the council",
                mode="council",
                execution_mode="research",
            ),
        )

    assert exc.value.status_code == 400
    assert "execution_mode" in exc.value.detail
    assert "chat mode" in exc.value.detail


@pytest.mark.asyncio
async def test_sync_council_allows_model_tier_override_for_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-council-model-tier"
    captured = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.5"},
    )
    stub_sync_council_dependencies(main, monkeypatch, captured)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(
            content="Run the council",
            mode="council",
            model_tier="budget",
        ),
    )

    assert result["type"] == "council"
    assert captured["chairman_model"] == "google/gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_sync_chat_uses_advanced_settings_for_rag_and_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-sync-advanced"
    captured_rag = {}
    captured_chairman = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.5"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        captured_rag.update(kwargs)
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_chairman.update(kwargs)
        return {"content": "Advanced response", "usage": {}}

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

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(
            content="Follow up",
            mode="chat",
            execution_mode="research",
            rag_preset="max",
            model_tier="budget",
        ),
    )

    assert captured_rag["max_tokens"] == 32000
    assert captured_chairman["chairman_model"] == "google/gemini-2.5-flash-lite"
    assert result["run_plan"]["mode"] == "research"
    assert result["run_plan"]["rag_preset"] == "max"
    assert result["run_plan"]["model_tier"] == "budget"


@pytest.mark.asyncio
async def test_stream_chat_uses_advanced_settings_for_rag_and_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-advanced"
    captured_rag = {}
    captured_chairman = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.5"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        captured_rag.update(kwargs)
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_chairman.update(kwargs)
        return {"content": "Advanced response", "usage": {}}

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
        main.SendMessageRequest(
            content="Follow up",
            mode="chat",
            execution_mode="research",
            rag_preset="max",
            model_tier="budget",
        ),
    )

    async for _chunk in response.body_iterator:
        pass

    assert captured_rag["max_tokens"] == 32000
    assert captured_chairman["chairman_model"] == "google/gemini-2.5-flash-lite"
