import ast
import importlib
import inspect

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def _status_expr_is_409(node):
    """True if an AST expression denotes HTTP 409 -- a literal 409 or any
    *HTTP_409_CONFLICT constant (e.g. status.HTTP_409_CONFLICT)."""
    if isinstance(node, ast.Constant):
        return node.value == 409
    if isinstance(node, ast.Name):
        return node.id.endswith("HTTP_409_CONFLICT")
    if isinstance(node, ast.Attribute):
        return node.attr.endswith("HTTP_409_CONFLICT")
    return False


def _http_exception_409_raises(fn):
    """Count HTTPException(...) calls in ``fn``'s source whose status is 409, in ANY
    spelling: ``status_code=409``, ``status_code = 409``, a positional first arg, or
    an HTTP_409_CONFLICT constant. Syntax-aware (ast) so formatting cannot hide a
    re-introduced duplicate the way a raw substring count could (Codex #109 R4)."""
    tree = ast.parse(inspect.getsource(fn))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if callee != "HTTPException":
            continue
        if any(kw.arg == "status_code" and _status_expr_is_409(kw.value) for kw in node.keywords):
            count += 1
        elif node.args and _status_expr_is_409(node.args[0]):  # positional status_code
            count += 1
    return count


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


def test_budget_path_single_enforcement_point_respects_allow_overage(monkeypatch, tmp_path):
    """D3 load-bearing guard (plan §D3 Tests): the budget path must have a SINGLE
    opt-in 409 enforcement point that honors allow_overage.

    Exercises the real shared pre-flight -- prepare_turn, which BOTH send endpoints
    call and which invokes ensure_budget_allows_new_turn (main.py:953) -- for an
    OVER-BUDGET allow_overage=True conversation, and asserts it does NOT 409. A
    hard-cap branch re-introduced anywhere in the budget path (the helper OR
    prepare_turn) would block this allow-overage turn; the allow_overage=False reject
    tests (test_sync_send_rejects_*, test_stream_send_rejects_*) pin the ONE opt-in
    raise from the other side.

    This proves the flip WORKS (opt-in is respected end-to-end); its companion
    test_budget_cap_has_a_single_409_enforcement_point proves the SINGLE-point
    invariant from the source side. Together they cover both properties."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-allow-overage-preflight"
    create_budgeted_conversation(main, conversation_id, allow_overage=True)  # spent 1.0 / 1.0

    # prepare_turn is the sync pre-flight both endpoints run; over-budget +
    # allow_overage must reach the end without raising the opt-in 409.
    _conversation, mode, _zdr, _effort, _is_first = main.prepare_turn(
        conversation_id,
        main.SendMessageRequest(content="over budget but allowed", mode="chat"),
    )
    assert mode == "chat"


def test_budget_cap_has_a_single_409_enforcement_point(monkeypatch):
    """D3 load-bearing SOURCE-LEVEL guard (plan §D3 Tests): exactly one HTTPException
    409 raise across the send budget path -- the opt-in cap is a SINGLE enforcement
    point.

    Parses (syntax-aware, Codex #109 R4) each budget-path function --
    ensure_budget_allows_new_turn (raises the cap), prepare_turn (the shared
    pre-flight that calls it), and both send endpoints -- and counts HTTPException
    raises whose status is 409 in ANY spelling (status_code=409, spaced, positional,
    or an HTTP_409_CONFLICT constant), so formatting cannot hide a duplicate. A
    re-introduced hard-cap branch anywhere in the path -- even a `not allow_overage`
    opt-in duplicate the allow-overage behavioral test can't see (R3) -- adds a
    second 409 and trips this. Scoped to the budget path so an unrelated non-budget
    409 on another endpoint never false-trips it (R1); spanning prepare_turn + the
    endpoints, not just the helper (R2)."""
    main = import_main(monkeypatch)
    total = sum(
        _http_exception_409_raises(fn)
        for fn in (
            main.ensure_budget_allows_new_turn,
            main.prepare_turn,
            main.send_message,
            main.send_message_stream,
        )
    )
    assert total == 1, (
        f"expected exactly one HTTPException(409) across the send budget path "
        f"(ensure_budget_allows_new_turn / prepare_turn / send_message / "
        f"send_message_stream), found {total} -- a re-introduced enforcement branch "
        f"would revert the D3 default flip"
    )
