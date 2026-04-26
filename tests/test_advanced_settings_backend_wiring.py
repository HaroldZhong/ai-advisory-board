import importlib
from types import SimpleNamespace

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


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
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_chairman.update(kwargs)
        return {"content": "Advanced response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
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
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_chairman.update(kwargs)
        return {"content": "Advanced response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
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
