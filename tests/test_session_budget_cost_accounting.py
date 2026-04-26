import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def test_calculate_turn_cost_counts_stage0_steward_and_council_stages(monkeypatch):
    main = import_main(monkeypatch)
    million_token_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    total = main.calculate_turn_cost(
        mode="council",
        stage1_results=[
            {"model": "openai/gpt-4o-mini", "usage": million_token_usage},
        ],
        stage2_results=[
            {"model": "openai/gpt-4o-mini", "usage": million_token_usage},
        ],
        stage3_result={"model": "openai/gpt-4o-mini", "usage": million_token_usage},
        extra_usage_records=[
            {"model": "openai/gpt-4o-mini", "usage": million_token_usage},
            {"model": "perplexity/sonar", "usage": million_token_usage},
        ],
    )

    assert total == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_session_policy_endpoints_persist_budget(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-budget-policy"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)

    default_state = await main.get_session_policy_endpoint(conversation_id)

    assert default_state["policy"]["budget_usd"] is None
    assert default_state["usage"]["spent_usd"] == 0.0
    assert default_state["budget_spent_pct"] is None

    updated_state = await main.update_session_policy_endpoint(
        conversation_id,
        main.SessionPolicyUpdate(budget_usd=2.0),
    )

    assert updated_state["policy"]["budget_usd"] == 2.0
    assert updated_state["policy"]["notify_thresholds"] == [0.70, 0.85, 1.00]
    assert updated_state["usage"]["spent_usd"] == 0.0

    stored = main.storage.get_session_policy(conversation_id)
    assert stored["budget_usd"] == 2.0


@pytest.mark.asyncio
async def test_session_policy_endpoint_rejects_invalid_budget(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-invalid-budget"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)

    with pytest.raises(main.HTTPException) as exc:
        await main.update_session_policy_endpoint(
            conversation_id,
            main.SessionPolicyUpdate(budget_usd=0),
        )

    assert exc.value.status_code == 400


def test_record_session_usage_warns_after_updated_turn_cost(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-budget-warning"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.set_session_policy(
        conversation_id,
        {
            "budget_usd": 1.0,
            "notify_thresholds": [0.70, 0.85, 1.00],
            "mode": "auto",
            "allow_overage": True,
        },
    )

    first = main.storage.record_session_usage(conversation_id, 0.72)
    second = main.storage.record_session_usage(conversation_id, 0.20)
    third = main.storage.record_session_usage(conversation_id, 0.10)

    assert first["warning_level"] == 0.70
    assert second["warning_level"] == 0.85
    assert third["warning_level"] == 1.00
    assert third["budget_spent_pct"] == pytest.approx(1.02)


@pytest.mark.asyncio
async def test_sync_chat_updates_total_and_session_cost(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-sync-cost"
    million_token_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_chat_with_chairman(*args, **kwargs):
        return {
            "content": "Budget-aware response",
            "usage": million_token_usage,
        }

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    fake_rag = SimpleNamespace(retrieve_async=fake_retrieve_async)

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    conversation = main.storage.get_conversation(conversation_id)
    usage = main.storage.get_session_usage(conversation_id)

    assert result["turn_cost"] == pytest.approx(0.75)
    assert result["total_cost"] == pytest.approx(0.75)
    assert conversation["total_cost"] == pytest.approx(0.75)
    assert conversation["messages"][-1]["running_cost"] == pytest.approx(0.75)
    assert usage["spent_usd"] == pytest.approx(0.75)
    assert usage["messages"] == 1
