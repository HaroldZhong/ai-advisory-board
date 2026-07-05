import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def parse_sse_events(chunks):
    events = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events


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
    assert updated_state["policy"]["notify_thresholds"] == [0.75, 0.85, 1.00]
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
            "notify_thresholds": [0.75, 0.85, 1.00],
            "mode": "auto",
            "allow_overage": True,
        },
    )

    first = main.storage.record_session_usage(conversation_id, 0.76)
    second = main.storage.record_session_usage(conversation_id, 0.10)
    third = main.storage.record_session_usage(conversation_id, 0.14)

    assert first["warning_level"] == 0.75
    assert second["warning_level"] == 0.85
    assert third["warning_level"] == 1.00
    assert third["budget_spent_pct"] == pytest.approx(1.00)


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
        return "", {}

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    fake_rag = SimpleNamespace(
        retrieve_async=fake_retrieve_async,
        index_chat_turn=fake_index_chat_turn,
        refresh_hybrid_index=lambda *a, **k: None,
        store={},
    )

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
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


@pytest.mark.asyncio
async def test_sync_chat_counts_rag_extraction_usage_in_turn_cost(monkeypatch, tmp_path):
    """audit §12: the RAG extraction call (UTILITY_MODEL, inside
    retrieve_with_stats_async) burns tokens on every chat turn but its cost
    was previously invisible -- never counted in turn_cost or session budget.
    retrieve_async now returns (context, usage); turn_pipeline must append
    that usage to extra_usage_records so calculate_turn_cost counts it
    alongside the chairman's own usage."""
    main = import_main(monkeypatch)
    conversation_id = "conv-rag-extraction-cost"

    # Small, exact numbers so the registry-pricing arithmetic is easy to
    # hand-verify: google/gemini-2.5-flash (UTILITY_MODEL) pricing is
    # input=$0.3/M, output=$2.5/M tokens (backend/model_registry.json).
    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    rag_extraction_usage = {"prompt_tokens": 100_000, "completion_tokens": 10_000}

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
            "usage": chairman_usage,
        }

    async def fake_retrieve_async(*args, **kwargs):
        return "captured RAG context", rag_extraction_usage

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    fake_rag = SimpleNamespace(
        retrieve_async=fake_retrieve_async,
        index_chat_turn=fake_index_chat_turn,
        refresh_hybrid_index=lambda *a, **k: None,
        store={},
    )

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    # openai/gpt-4o-mini pricing (backend/model_registry.json): input=$0.15/M, output=$0.6/M
    chairman_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    rag_cost = (100_000 / 1_000_000) * 0.3 + (10_000 / 1_000_000) * 2.5
    expected_total = chairman_cost + rag_cost

    assert result["turn_cost"] == pytest.approx(expected_total)
    assert result["total_cost"] == pytest.approx(expected_total)


@pytest.mark.asyncio
async def test_sync_chat_skips_rag_cost_when_no_extraction_ran(monkeypatch, tmp_path):
    """Control: when retrieve_async returns an empty usage dict (nothing to
    retrieve, or the extraction call failed), no RAG cost record is added --
    only the chairman's own usage is counted."""
    main = import_main(monkeypatch)
    conversation_id = "conv-rag-no-extraction-cost"
    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

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
            "usage": chairman_usage,
        }

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    fake_rag = SimpleNamespace(
        retrieve_async=fake_retrieve_async,
        index_chat_turn=fake_index_chat_turn,
        refresh_hybrid_index=lambda *a, **k: None,
        store={},
    )

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    chairman_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    assert result["turn_cost"] == pytest.approx(chairman_cost)


@pytest.mark.asyncio
async def test_stream_chat_updates_total_and_session_cost(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-stream-cost"
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
            "content": "Streaming budget-aware response",
            "usage": million_token_usage,
        }

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    fake_rag = SimpleNamespace(
        retrieve_async=fake_retrieve_async,
        index_chat_turn=fake_index_chat_turn,
        refresh_hybrid_index=lambda *a, **k: None,
        store={},
    )

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    conversation = main.storage.get_conversation(conversation_id)
    usage = main.storage.get_session_usage(conversation_id)
    complete_events = [chunk for chunk in chunks if '"type": "complete"' in chunk]

    assert complete_events
    assert conversation["total_cost"] == pytest.approx(0.75)
    assert conversation["messages"][-1]["running_cost"] == pytest.approx(0.75)
    assert usage["spent_usd"] == pytest.approx(0.75)
    assert usage["messages"] == 1


@pytest.mark.asyncio
async def test_budget_warning_survives_a_billed_delta_after_the_base_crossing(monkeypatch, tmp_path):
    """Codex round 12 P2, superseded by round 23: the BASE
    record_session_usage call (recorded immediately after the chat message
    is persisted, round 10) can be the one that actually crosses a budget
    threshold. _get_new_warning_level only reports a threshold the FIRST
    time it's crossed (comparing against the already-persisted
    last_warning_level), so the SECOND, incremental delta call
    (topics/compression usage discovered during indexing) returns
    warning_level=None for the SAME turn.

    Round 12 fixed this by remembering the base call's warning_level and
    merging it back into the tail's emission if the delta call didn't
    report one of its own. Round 23 changed WHERE the base crossing is
    emitted: synchronously, in the same block that recorded it -- not
    merged into the tail after the indexing awaits. Round 25 tightened
    this further: round 23 put the emission right AFTER chat_response,
    leaving one yield (chat_response itself) between the record call and
    the warning reaching the stream -- a client disconnecting right after
    receiving chat_response would still lose the warning. Round 25 moved
    the emission BEFORE chat_response, so there are zero yields (and zero
    awaits) between record_session_usage persisting last_warning_level and
    the warning event being handed to the stream. This test pins the
    warning event's POSITION (before chat_response, among the first
    events) to prove it, not just its count/threshold.

    Fails pre-fix (pre-round-25): the warning event would still arrive
    AFTER chat_response."""
    main = import_main(monkeypatch)
    conversation_id = "conv-warning-survives-delta"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")
    main.storage.set_session_policy(
        conversation_id,
        {
            "budget_usd": 1.0,
            "notify_thresholds": [0.75, 0.85, 1.00],
            "mode": "auto",
            "allow_overage": True,
        },
    )

    # Base chairman usage alone crosses the 0.75 threshold: openai/gpt-4o-mini
    # pricing is input=$0.15/M, output=$0.6/M -- 1M/1M tokens costs $0.75.
    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    # A small, nonzero delta from compression usage: google/gemini-2.5-flash
    # (UTILITY_MODEL) pricing is input=$0.3/M, output=$2.5/M.
    compression_usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000}

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "Budget-aware response", "usage": chairman_usage}

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_extract_topics(*args, **kwargs):
        return ["topic"], {}

    async def fake_index_chat_turn(*args, **kwargs):
        return compression_usage  # nonzero delta

    fake_rag = SimpleNamespace(
        retrieve_async=fake_retrieve_async,
        index_chat_turn=fake_index_chat_turn,
        refresh_hybrid_index=lambda *a, **k: None,
        store={},
    )

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    events = parse_sse_events(chunks)

    warning_events = [e for e in events if e["type"] == "budget_warning"]
    complete_events = [e for e in events if e["type"] == "complete"]

    chairman_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    compression_cost = (10_000 / 1_000_000) * 0.3 + (1_000 / 1_000_000) * 2.5
    expected_total = chairman_cost + compression_cost

    assert len(warning_events) == 1, "the base call's 0.75 crossing must survive the delta call"
    assert warning_events[0]["data"]["threshold"] == 0.75
    assert len(complete_events) == 1
    # Completion payload carries the POST-delta totals (base + delta), not
    # just the base amount.
    assert complete_events[0]["data"]["turn_cost"] == pytest.approx(expected_total)
    assert complete_events[0]["data"]["total_cost"] == pytest.approx(expected_total)

    # Codex round 25: the warning is emitted right BEFORE chat_response, in
    # the same synchronous block that recorded the base crossing -- well
    # before the indexing-derived complete event, not merged in at the end,
    # and with zero yields between the record call and this event.
    event_types = [e["type"] for e in events]
    assert event_types.index("budget_warning") < event_types.index("complete")
    assert event_types.index("budget_warning") == event_types.index("chat_response") - 1
