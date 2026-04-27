import importlib

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def create_budgeted_conversation(main, conversation_id, *, allow_overage):
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.set_session_policy(
        conversation_id,
        {
            "budget_usd": 1.0,
            "notify_thresholds": [0.75, 0.85, 1.0],
            "mode": "auto",
            "allow_overage": allow_overage,
        },
    )
    main.storage.record_session_usage(conversation_id, 1.0)


def test_budget_guard_rejects_new_turn_after_enforced_cap(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-cap-reached"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    create_budgeted_conversation(main, conversation_id, allow_overage=False)

    with pytest.raises(main.HTTPException) as exc:
        main.ensure_budget_allows_new_turn(conversation_id)

    assert exc.value.status_code == 409
    assert exc.value.detail == "Session budget reached. Raise the cap before sending another message."


def test_budget_guard_preserves_legacy_overage_semantics(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-legacy-overage"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    create_budgeted_conversation(main, conversation_id, allow_overage=True)

    main.ensure_budget_allows_new_turn(conversation_id)


@pytest.mark.asyncio
async def test_budget_update_preserves_existing_overage_policy(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-preserve-overage-policy"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.set_session_policy(
        conversation_id,
        {
            "budget_usd": 1.0,
            "notify_thresholds": [0.75, 0.85, 1.0],
            "mode": "auto",
            "allow_overage": False,
        },
    )

    state = await main.update_session_policy_endpoint(
        conversation_id,
        main.SessionPolicyUpdate(budget_usd=2.0),
    )

    assert state["policy"]["budget_usd"] == 2.0
    assert state["policy"]["allow_overage"] is False


@pytest.mark.asyncio
async def test_sync_send_rejects_over_cap_before_adding_user_message(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-sync-cap"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    create_budgeted_conversation(main, conversation_id, allow_overage=False)

    with pytest.raises(main.HTTPException) as exc:
        await main.send_message(
            conversation_id,
            main.SendMessageRequest(content="This should be blocked", mode="chat"),
        )

    conversation = main.storage.get_conversation(conversation_id)
    assert exc.value.status_code == 409
    assert conversation["messages"] == []


@pytest.mark.asyncio
async def test_stream_send_rejects_over_cap_with_http_409(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-stream-cap"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    create_budgeted_conversation(main, conversation_id, allow_overage=False)

    with pytest.raises(main.HTTPException) as exc:
        await main.send_message_stream(
            conversation_id,
            main.SendMessageRequest(content="This should be blocked", mode="chat"),
        )

    conversation = main.storage.get_conversation(conversation_id)
    assert exc.value.status_code == 409
    assert conversation["messages"] == []
