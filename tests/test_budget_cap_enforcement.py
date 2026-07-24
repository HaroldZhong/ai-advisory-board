import importlib
import inspect

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


@pytest.mark.asyncio
async def test_new_budgeted_conversation_defaults_to_allow_overage(monkeypatch, tmp_path):
    """v1.3.0 D3 (correction #10): a NEW conversation created with a budget but no
    explicit overage choice now defaults to allow-overage -- the single 409 hard
    cap is opt-in, so over-budget turns warn rather than block."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    conv = await main.create_conversation(main.CreateConversationRequest(budget_usd=1.0))
    conv_id = conv["id"]

    assert main.storage.get_session_policy(conv_id)["allow_overage"] is True
    main.storage.record_session_usage(conv_id, 1.0)  # at the cap
    main.ensure_budget_allows_new_turn(conv_id)  # must NOT raise


@pytest.mark.asyncio
async def test_new_budgeted_conversation_explicit_hard_cap_still_blocks(monkeypatch, tmp_path):
    """The hard cap remains available as an explicit opt-in even after the default
    flip."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    conv = await main.create_conversation(
        main.CreateConversationRequest(budget_usd=1.0, budget_allow_overage=False)
    )
    conv_id = conv["id"]

    assert main.storage.get_session_policy(conv_id)["allow_overage"] is False
    main.storage.record_session_usage(conv_id, 1.0)
    with pytest.raises(main.HTTPException) as exc:
        main.ensure_budget_allows_new_turn(conv_id)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_default_flip_does_not_migrate_existing_conversations(monkeypatch, tmp_path):
    """Correction #10: existing stored allow_overage values are never bulk-migrated
    -- a conversation persisted with allow_overage=False keeps hard-capping."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    create_budgeted_conversation(main, "conv-existing-hardcap", allow_overage=False)
    assert main.storage.get_session_policy("conv-existing-hardcap")["allow_overage"] is False
    with pytest.raises(main.HTTPException) as exc:
        main.ensure_budget_allows_new_turn("conv-existing-hardcap")
    assert exc.value.status_code == 409


def test_budget_path_raises_409_exactly_once_in_source(monkeypatch):
    """D3 load-bearing SOURCE-LEVEL guard (plan §D3 Tests): after the overage
    default flip, the hard 409 cap must remain a SINGLE opt-in enforcement point.

    Behavioral tests all exercise the ONE current raise site, so a re-introduced
    second enforcement branch (a new `raise HTTPException(status_code=409, ...)`)
    would silently pass them. A source-level count catches that regression: at HEAD
    the only 409 in backend/main.py is the budget cap (BUDGET_CAP_REACHED_DETAIL)."""
    main = import_main(monkeypatch)
    source = inspect.getsource(main)
    count = source.count("status_code=409")
    assert count == 1, (
        f"expected exactly one status_code=409 (the opt-in budget cap) in backend/main.py, "
        f"found {count} -- a re-introduced enforcement branch would revert the D3 default flip"
    )
