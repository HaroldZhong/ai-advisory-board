import importlib
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "reasoning_usage_fixture.json")


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


# --- D1 reasoning-token accounting gate (v1.3.0 plan Phase D) ------------------
# NOTE (correction #4): the authoritative TURN TOTAL is usage.cost (see the
# "usage.cost is the authoritative turn total" tests above). The inside-vs-separate
# classification below is REGRESSION + DISPLAY evidence -- "how much of the billed
# cost was reasoning" -- NOT a recompute of the billed total. It never overrides
# usage.cost.
#
# The load-bearing gate is test_reasoning_usage_fixture_billing_relationship: it
# decides inside-vs-separate billing from a REAL captured usage payload, using
# OpenRouter's billed `cost` as the authority (token containment can't tell them
# apart). The decision is PROVIDER-SPECIFIC, so the gate also requires the fixture
# to record the route it is valid for.
#
# The classifier is single-sourced from scripts/capture_reasoning_usage_fixture.py
# (`classify_billing`) -- capture and gate share one implementation of the
# thresholds rather than each carrying a copy that can drift.
#
# The two characterization tests below describe CURRENT calculate_cost behaviour;
# they do NOT assert the billing relationship from memory. The synthetic-case row
# proves the gate is load-bearing without needing a live capture.


def test_calculate_cost_ignores_reasoning_token_field_today(monkeypatch):
    """Characterizes CURRENT behaviour: calculate_cost reads only
    completion_tokens and ignores any completion_tokens_details.reasoning_tokens
    field, so adding that field must not ACCIDENTALLY change the cost. This pins
    the current arithmetic so any D1 change is deliberate. Whether the final
    total SHOULD gain a separate reasoning line is decided by the captured
    fixture (test_reasoning_usage_fixture_billing_relationship), not asserted
    here."""
    main = import_main(monkeypatch)
    model = "openai/gpt-4o-mini"  # input $0.15/M, output $0.6/M
    base = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    with_reasoning = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "completion_tokens_details": {"reasoning_tokens": 400_000},
    }
    assert main.calculate_cost(with_reasoning, model) == pytest.approx(
        main.calculate_cost(base, model)
    )
    # Today's cost is the completion-token cost only.
    assert main.calculate_cost(with_reasoning, model) == pytest.approx(0.15 + 0.6)


def test_turn_cost_ignores_reasoning_token_field_today(monkeypatch):
    """Same current-behaviour characterization at the turn level: a Stage-3
    result carrying a reasoning-token field yields the same turn total as one
    without, under today's calculate_cost. Not a claim about how reasoning is
    billed -- that is the fixture's job."""
    main = import_main(monkeypatch)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    usage_with_reasoning = {
        **usage,
        "completion_tokens_details": {"reasoning_tokens": 750_000},
    }
    without = main.calculate_turn_cost(
        mode="council",
        stage3_result={"model": "openai/gpt-4o-mini", "usage": usage},
    )
    with_reasoning = main.calculate_turn_cost(
        mode="council",
        stage3_result={"model": "openai/gpt-4o-mini", "usage": usage_with_reasoning},
    )
    assert with_reasoning == pytest.approx(without)


# --- D1: usage.cost is the authoritative turn total (correction #4) ------------
# OpenRouter returns the billed per-request cost in usage.cost on every response.
# calculate_cost prefers it, so the turn total is the amount actually charged;
# registry token pricing is used ONLY when usage.cost is absent.


def test_calculate_cost_prefers_billed_usage_cost(monkeypatch):
    """When usage.cost is present it is authoritative -- NOT recomputed from
    registry token pricing, even if the two disagree."""
    main = import_main(monkeypatch)
    # Registry would price this at 0.15 + 0.6 = 0.75; the billed cost differs.
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cost": 0.884}
    assert main.calculate_cost(usage, "openai/gpt-4o-mini") == pytest.approx(0.884)


def test_calculate_cost_falls_back_to_registry_without_billed_cost(monkeypatch):
    """No usage.cost -> registry token pricing (unchanged legacy behaviour)."""
    main = import_main(monkeypatch)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert main.calculate_cost(usage, "openai/gpt-4o-mini") == pytest.approx(0.15 + 0.6)


def test_calculate_cost_billed_authoritative_without_registry_model(monkeypatch):
    """usage.cost stands on its own -- a model absent from the registry (no
    pricing) still yields the billed cost, not 0."""
    main = import_main(monkeypatch)
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.0123}
    assert main.calculate_cost(usage, "provider/not-in-registry") == pytest.approx(0.0123)


def test_calculate_cost_zero_billed_cost_is_authoritative(monkeypatch):
    """A genuinely free call reports cost 0.0; that is authoritative, not a
    trigger to fall back to registry pricing."""
    main = import_main(monkeypatch)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cost": 0.0}
    assert main.calculate_cost(usage, "openai/gpt-4o-mini") == pytest.approx(0.0)


def test_calculate_cost_ignores_anomalous_billed_cost(monkeypatch):
    """A non-numeric or negative cost is anomalous, not a valid billed total, so
    it falls back to registry pricing rather than corrupting the meter."""
    main = import_main(monkeypatch)
    base = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    for bad in (-1.0, "0.5", None, True):
        assert main.calculate_cost({**base, "cost": bad}, "openai/gpt-4o-mini") == pytest.approx(0.75)


def test_turn_cost_sums_authoritative_billed_costs(monkeypatch):
    """A council turn whose every call reports usage.cost totals the sum of those
    billed costs -- the authoritative turn total -- not the registry recompute."""
    main = import_main(monkeypatch)
    total = main.calculate_turn_cost(
        mode="council",
        stage1_results=[{"model": "openai/gpt-4o-mini",
                         "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.10}}],
        stage2_results=[{"model": "openai/gpt-4o-mini",
                         "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.20}}],
        stage3_result={"model": "openai/gpt-4o-mini",
                       "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.30}},
        extra_usage_records=[{"model": "perplexity/sonar",
                              "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.05}}],
    )
    assert total == pytest.approx(0.10 + 0.20 + 0.30 + 0.05)


def test_turn_cost_mixes_billed_and_registry_fallback(monkeypatch):
    """Per-record: authoritative where usage.cost is present, registry fallback
    where it is absent, summed into one turn total."""
    main = import_main(monkeypatch)
    total = main.calculate_turn_cost(
        mode="council",
        stage1_results=[{"model": "openai/gpt-4o-mini",
                         "usage": {"prompt_tokens": 5, "completion_tokens": 5, "cost": 0.40}}],
        # no usage.cost -> registry: 1M/1M @ gpt-4o-mini = 0.75
        stage3_result={"model": "openai/gpt-4o-mini",
                       "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}},
    )
    assert total == pytest.approx(0.40 + 0.75)


def test_sum_usage_preserves_billed_cost_through_merge(monkeypatch):
    """Codex PR#90 P2: when one index call triggers both compaction tiers, the two
    utility usage records are merged via rag._sum_usage before a single
    calculate_cost. Both legs carry OpenRouter's billed usage.cost, so the merged
    record must keep the SUMMED billed cost -- otherwise the already-billed calls
    fall back to registry pricing and diverge from real charges."""
    main = import_main(monkeypatch)
    from backend.rag import _sum_usage

    a = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.011}
    b = {"prompt_tokens": 20, "completion_tokens": 8, "cost": 0.022}
    merged = _sum_usage(a, b)
    assert merged["prompt_tokens"] == 30 and merged["completion_tokens"] == 13
    assert merged["cost"] == pytest.approx(0.033)
    # The billed total survives the single calculate_cost the caller makes.
    assert main.calculate_cost(merged, "google/gemini-2.5-flash") == pytest.approx(0.033)


def test_sum_usage_omits_cost_when_a_leg_is_unbilled(monkeypatch):
    """If a leg lacks a billed cost, the merge omits cost so calculate_cost
    registry-prices the summed tokens rather than under-counting the billed leg."""
    from backend.rag import _sum_usage

    merged = _sum_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.011},
        {"prompt_tokens": 20, "completion_tokens": 8},  # no billed cost
    )
    assert "cost" not in merged
    assert merged["prompt_tokens"] == 30 and merged["completion_tokens"] == 13


def _billing_classifier():
    """The single source of the inside/separate decision, imported from the
    capture harness so gate and capture can never drift apart."""
    return importlib.import_module("scripts.capture_reasoning_usage_fixture")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _routing_metadata(provider, *, selected=True):
    """An openrouter_metadata block in the documented shape: endpoints.available[]
    entries carry provider/model/selected (docs/guides/features/router-metadata)."""
    return {
        "requested": "openai/gpt-4o-mini",
        "summary": f"available=1, selected={provider}",
        "endpoints": {
            "total": 1,
            "available": [
                {"provider": provider, "model": "openai/gpt-4o-mini", "selected": selected}
            ],
        },
    }


def _endpoints_payload():
    """Mirrors the real GET /api/v1/models/{model}/endpoints response, verified
    live 2026-07 against openai/gpt-4o-mini: TWO providers, and TWO tags under one
    provider ("azure" vs "azure/swedencentral", both provider_name "Azure") whose
    completion rates genuinely differ. Billing is per endpoint, not per provider."""
    return {"data": {"id": "openai/gpt-4o-mini", "endpoints": [
        {"name": "Azure | openai/gpt-4o-mini", "tag": "azure", "provider_name": "Azure",
         "model_id": "openai/gpt-4o-mini", "model_name": "GPT-4o-mini",
         "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
        {"name": "OpenAI | openai/gpt-4o-mini", "tag": "openai", "provider_name": "OpenAI",
         "model_id": "openai/gpt-4o-mini", "model_name": "GPT-4o-mini",
         "pricing": {"prompt": "0.0000002", "completion": "0.0000009"}},
        {"name": "Azure (Sweden Central) | openai/gpt-4o-mini", "tag": "azure/swedencentral",
         "provider_name": "Azure", "model_id": "openai/gpt-4o-mini", "model_name": "GPT-4o-mini",
         "pricing": {"prompt": "0.000000165", "completion": "0.00000066",
                     "internal_reasoning": "0.000001"}},
    ]}}


def _endpoints_getter(payload=None):
    def _get(url, timeout=None):
        assert url.endswith("/endpoints"), "pricing must come from the per-endpoint route"
        return _FakeResponse(payload if payload is not None else _endpoints_payload())
    return _get


def test_endpoint_pricing_differs_between_providers_for_one_model():
    """Two providers serving the same model price differently, so a model-wide
    rate cannot stand in for a pinned route."""
    capture = _billing_classifier()
    azure = capture.fetch_endpoint_pricing("openai/gpt-4o-mini", "azure", get=_endpoints_getter())
    openai = capture.fetch_endpoint_pricing("openai/gpt-4o-mini", "openai", get=_endpoints_getter())

    assert azure["completion_per_token"] == 6e-7
    assert openai["completion_per_token"] == 9e-7
    assert azure["provider_name"] == "Azure" and openai["provider_name"] == "OpenAI"


def test_endpoint_pricing_differs_between_tags_of_one_provider():
    """The trap this defect was about: two tags of the SAME provider price
    differently, so matching on provider_name would pick the wrong rate."""
    capture = _billing_classifier()
    base = capture.fetch_endpoint_pricing("openai/gpt-4o-mini", "azure", get=_endpoints_getter())
    variant = capture.fetch_endpoint_pricing(
        "openai/gpt-4o-mini", "azure/swedencentral", get=_endpoints_getter()
    )

    assert base["provider_name"] == variant["provider_name"] == "Azure"
    assert base["completion_per_token"] == 6e-7
    assert variant["completion_per_token"] == 6.6e-7
    assert base["completion_per_token"] != variant["completion_per_token"]


def test_endpoint_pricing_variant_tag_records_full_identity():
    """The exact variant tag is selectable and its identity + own
    internal_reasoning price are recorded."""
    capture = _billing_classifier()
    price = capture.fetch_endpoint_pricing(
        "openai/gpt-4o-mini", "azure/swedencentral", get=_endpoints_getter()
    )

    assert price["tag"] == "azure/swedencentral"
    assert price["provider_name"] == "Azure"
    assert price["endpoint_name"] == "Azure (Sweden Central) | openai/gpt-4o-mini"
    assert price["endpoint_model_id"] == "openai/gpt-4o-mini"
    assert price["internal_reasoning_per_token"] == 1e-6
    assert price["source"].endswith("/models/openai/gpt-4o-mini/endpoints")
    assert price["fetched_at"]


def test_endpoint_pricing_rejects_missing_tag():
    """No exact tag -> fail (and say which tags exist) rather than fall back to a
    provider-name or model-wide rate."""
    capture = _billing_classifier()
    with pytest.raises(capture.CaptureError, match="no endpoint on .* has tag"):
        capture.fetch_endpoint_pricing("openai/gpt-4o-mini", "azur", get=_endpoints_getter())


def test_endpoint_pricing_rejects_ambiguous_tag():
    """Duplicate tags cannot be priced unambiguously."""
    capture = _billing_classifier()
    payload = _endpoints_payload()
    payload["data"]["endpoints"].append(dict(payload["data"]["endpoints"][0]))

    with pytest.raises(capture.CaptureError, match="ambiguous"):
        capture.fetch_endpoint_pricing(
            "openai/gpt-4o-mini", "azure", get=_endpoints_getter(payload)
        )


def test_routing_metadata_matching_display_name_is_accepted():
    """The router reports a display name; it is verified against the pinned
    ENDPOINT's provider_name."""
    capture = _billing_classifier()
    served = capture.select_provider_from_metadata(
        {"openrouter_metadata": _routing_metadata("OpenAI")}, "OpenAI"
    )
    assert served == "OpenAI"


def test_routing_metadata_accepts_variant_tag_served_under_provider_name():
    """Pinning `azure/swedencentral` is served as "Azure". Verifying against the
    endpoint's provider_name accepts it; verifying against the raw tag would have
    wrongly rejected a correct route."""
    capture = _billing_classifier()
    price = capture.fetch_endpoint_pricing(
        "openai/gpt-4o-mini", "azure/swedencentral", get=_endpoints_getter()
    )
    served = capture.select_provider_from_metadata(
        {"openrouter_metadata": _routing_metadata("Azure")}, price["provider_name"]
    )
    assert served == "Azure"


def test_routing_metadata_missing_is_rejected():
    """Without openrouter_metadata the served route is unverified, so the capture
    is not provider evidence."""
    capture = _billing_classifier()
    with pytest.raises(capture.CaptureError, match="openrouter_metadata"):
        capture.select_provider_from_metadata({"choices": []}, "OpenAI")


def test_routing_metadata_without_selected_endpoint_is_rejected():
    capture = _billing_classifier()
    with pytest.raises(capture.CaptureError, match="no selected endpoint"):
        capture.select_provider_from_metadata(
            {"openrouter_metadata": _routing_metadata("OpenAI", selected=False)}, "OpenAI"
        )


def test_routing_metadata_mismatched_provider_is_rejected():
    """A genuinely different provider still fails, even though variant tags of the
    same provider are tolerated."""
    capture = _billing_classifier()
    with pytest.raises(capture.CaptureError, match="routing mismatch"):
        capture.select_provider_from_metadata(
            {"openrouter_metadata": _routing_metadata("Azure")}, "OpenAI"
        )


def test_capture_records_verified_route_and_price_provenance():
    """End-to-end with mocked transports: a good capture records the verified
    route and the exact rates + provenance the gate will later re-derive from."""
    capture = _billing_classifier()

    def fake_post(url, headers=None, json=None, timeout=None):
        assert headers["X-OpenRouter-Metadata"] == "enabled"
        assert json["provider"] == {"order": ["openai"], "allow_fallbacks": False}
        assert "usage" not in json, "usage:{include:true} is deprecated and must not be sent"
        return _FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
            "openrouter_metadata": _routing_metadata("OpenAI"),
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
                "completion_tokens_details": {"reasoning_tokens": 400_000},
                "cost": 0.75,
                "cost_details": {"upstream_inference_cost": 0.70},
            },
        })

    fixture = capture.capture("openai/gpt-4o-mini", "openai", "key",
                              post=fake_post, get=_endpoints_getter())

    assert fixture["provider_selected"] == "OpenAI"
    price = fixture["price_authority"]
    assert price["source"].endswith("/models/openai/gpt-4o-mini/endpoints")
    assert price["fetched_at"]
    assert price["tag"] == "openai"
    assert price["provider_name"] == "OpenAI"
    # The OpenAI endpoint's own rate, not Azure's and not a model-wide rate.
    assert price["completion_per_token"] == 9e-7
    assert fixture["upstream_inference_cost"] == 0.70
    assert fixture["billing_relationship"] == "inside"
    # The gate must accept a genuine capture end-to-end.
    assert _assert_fixture_proves_inside_billing(fixture) == "inside"


def test_classification_follows_the_pinned_endpoint_rate():
    """Proof the verdict is derived from the PINNED endpoint's rate: one identical
    billed cost classifies differently depending on which tag was pinned, because
    the two endpoints price differently. A model-wide rate could not do this."""
    capture = _billing_classifier()
    usage_tokens = 1_000_000
    reasoning = 400_000
    billed = 0.99

    def verdict(tag):
        price = capture.fetch_endpoint_pricing(
            "openai/gpt-4o-mini", tag, get=_endpoints_getter()
        )
        completion_priced = (
            usage_tokens * price["prompt_per_token"]
            + usage_tokens * price["completion_per_token"]
        )
        relationship, _ = capture.classify_billing(
            reasoning_tokens=reasoning,
            reported_cost=billed,
            completion_priced_cost=completion_priced,
            completion_rate_per_token=price["completion_per_token"],
            internal_reasoning_rate_per_token=price["internal_reasoning_per_token"],
        )
        return relationship

    # azure prices completion at 6e-7 -> $0.75 priced, so $0.99 billed is a
    # surcharge matching 400k reasoning tokens: SEPARATE.
    assert verdict("azure") == "separate"
    # openai prices it at 9e-7 -> $1.10 priced, so the same $0.99 carries no
    # surcharge at all: INSIDE.
    assert verdict("openai") == "inside"


def test_failed_capture_leaves_fixture_untouched(monkeypatch, tmp_path):
    """A capture that cannot verify its route must not overwrite the placeholder
    (or a good prior capture) with junk."""
    capture = _billing_classifier()
    fixture_file = tmp_path / "reasoning_usage_fixture.json"
    original = '{"_placeholder": true}'
    fixture_file.write_text(original, encoding="utf-8")

    monkeypatch.setattr(capture, "FIXTURE", fixture_file)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", ["capture", "openai/gpt-4o-mini", "openai"])

    def exploding_capture(*args, **kwargs):
        raise capture.CaptureError("routing mismatch: simulated")

    monkeypatch.setattr(capture, "capture", exploding_capture)

    assert capture.main() == 1
    assert fixture_file.read_text(encoding="utf-8") == original


def _synthetic_fixture(*, reasoning, cost, label, internal_reasoning=None):
    """A fixture shaped like a real capture, carrying its own captured rates.

    Rates are USD PER TOKEN (OpenRouter's unit), unlike the curated registry's
    per-million: 1.5e-7 prompt + 6e-7 completion over 1M/1M tokens is $0.75."""
    return {
        "_placeholder": False,
        "model": "openai/gpt-4o-mini",
        "provider_requested": "synthetic-provider",
        "provider_selected": "Synthetic-Provider",  # display-name form, must still match
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        "cost": cost,
        "price_authority": {
            "source": "https://openrouter.ai/api/v1/models/openai/gpt-4o-mini/endpoints",
            "fetched_at": "2026-07-23T00:00:00+00:00",
            "units": "USD per token",
            "tag": "synthetic-provider",
            "provider_name": "Synthetic-Provider",
            "endpoint_name": "Synthetic-Provider | openai/gpt-4o-mini",
            "prompt_per_token": 1.5e-7,
            "completion_per_token": 6e-7,
            "internal_reasoning_per_token": internal_reasoning,
        },
        "billing_relationship": label,
    }


def _assert_fixture_proves_inside_billing(fixture):
    """Shared D1 gate policy applied to a fixture dict.

    The verdict is re-derived from the fixture's OWN stored rates, never from
    today's mutable registry -- a capture stays reproducible even after registry
    prices move (drift is reported separately, it must not silently reclassify).

    Raises AssertionError unless the fixture is provider-verified, cost-bearing,
    priced with recorded provenance, and shows a clear INSIDE relationship."""
    capture = _billing_classifier()
    usage = fixture["usage"]
    reported_cost = fixture.get("cost")
    recorded = fixture.get("billing_relationship")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

    requested = fixture.get("provider_requested")
    selected = fixture.get("provider_selected")
    if not requested or not selected:
        raise AssertionError(
            "fixture must record both provider_requested and the VERIFIED provider_selected "
            "(from openrouter_metadata); a pin alone does not prove which route served the call"
        )
    if reported_cost is None:
        raise AssertionError("capture must record the billed `usage.cost` -- it is the authority")

    price = fixture.get("price_authority") or {}
    for field in ("source", "fetched_at", "tag", "provider_name",
                  "prompt_per_token", "completion_per_token"):
        if price.get(field) is None:
            raise AssertionError(
                f"price_authority.{field} missing -- the verdict must be reproducible from the "
                "EXACT pinned endpoint's rates recorded AT CAPTURE TIME, with provenance"
            )

    # Pricing keys on the exact endpoint tag (variant tags of one provider price
    # differently), while the router's display name is verified against that
    # endpoint's provider_name -- so `azure/swedencentral` served as "Azure" passes.
    if price["tag"] != requested:
        raise AssertionError(
            f"priced endpoint tag {price['tag']!r} is not the pinned tag {requested!r} -- "
            "billing is per endpoint, so the rate must come from the tag that was pinned"
        )
    if capture._normalize_provider(selected) != capture._normalize_provider(price["provider_name"]):
        raise AssertionError(
            f"routing mismatch in fixture: pinned endpoint belongs to "
            f"{price['provider_name']!r} but the router served {selected!r} -- re-capture"
        )

    # Price from the STORED rates (USD per token), not the registry.
    completion_priced = (
        usage.get("prompt_tokens", 0) * price["prompt_per_token"]
        + usage.get("completion_tokens", 0) * price["completion_per_token"]
    )

    relationship, detail = capture.classify_billing(
        reasoning_tokens=reasoning,
        reported_cost=reported_cost,
        completion_priced_cost=completion_priced,
        completion_rate_per_token=price["completion_per_token"],
        internal_reasoning_rate_per_token=price.get("internal_reasoning_per_token"),
    )

    # The classifier is the authority; a recorded label that disagrees means the
    # fixture was hand-edited or captured under different thresholds.
    if recorded != relationship:
        raise AssertionError(
            f"fixture billing_relationship={recorded!r} disagrees with the classifier's "
            f"{relationship!r} (surcharge={detail['surcharge']}, "
            f"expected_separate={detail['expected_separate_surcharge']}, "
            f"noise={detail['noise']}) -- re-capture rather than hand-editing the label"
        )
    if relationship != "inside":
        raise AssertionError(
            f"D1 gate not satisfied: billing_relationship={relationship!r}. "
            f"{capture.CONCLUSIONS[relationship]}"
        )
    return relationship


@pytest.mark.parametrize(
    "case, reasoning, cost, label, internal_reasoning, should_pass",
    [
        # cost 0.75 == completion-priced -> no surcharge, signal large enough
        ("inside/large", 400_000, 0.75, "inside", None, True),
        # cost 0.99 == completion-priced + 400k*6e-7 -> clear separate charge
        ("separate/large", 400_000, 0.99, "separate", None, False),
        # a REAL separate charge too small to distinguish from noise: must be
        # ambiguous, never a false 'inside' (the false-pass hole)
        ("separate/small", 10_000, 0.756, "ambiguous", None, False),
        # genuinely inside but signal too small to prove it -> ambiguous
        ("inside/small", 10_000, 0.75, "ambiguous", None, False),
        # label claims inside while cost shows separate -> mismatch rejected
        ("mislabeled-inside", 400_000, 0.99, "inside", None, False),
        # an explicit internal_reasoning price is a named separate charge, even
        # though the billed cost alone would have read as 'inside'
        ("explicit-internal-reasoning", 400_000, 0.75, "separate", 6e-7, False),
    ],
)
def test_d1_billing_gate_synthetic_cases(case, reasoning, cost, label, internal_reasoning, should_pass):
    """Proves the D1 gate is load-bearing WITHOUT a live capture, preserving the
    cases as runnable tests (they were previously only demonstrated ad-hoc by
    swapping the fixture file in and out).

    Only a clear, provider-verified INSIDE reading may pass. Separate billing, an
    indistinguishable surcharge, a hand-edited label, and an explicitly priced
    internal_reasoning rate must all be rejected."""
    fixture = _synthetic_fixture(
        reasoning=reasoning, cost=cost, label=label, internal_reasoning=internal_reasoning
    )

    if should_pass:
        assert _assert_fixture_proves_inside_billing(fixture) == "inside"
    else:
        with pytest.raises(AssertionError):
            _assert_fixture_proves_inside_billing(fixture)


def test_d1_billing_gate_requires_verified_route():
    """A pin alone is not evidence: without the VERIFIED provider_selected from
    routing metadata the reading cannot be attributed to a route."""
    fixture = _synthetic_fixture(reasoning=400_000, cost=0.75, label="inside")
    fixture["provider_selected"] = None

    with pytest.raises(AssertionError, match="provider_selected"):
        _assert_fixture_proves_inside_billing(fixture)


def test_d1_billing_gate_rejects_routing_mismatch():
    """A capture served by a provider other than the pinned one is not evidence
    for the pinned provider's billing."""
    fixture = _synthetic_fixture(reasoning=400_000, cost=0.75, label="inside")
    fixture["provider_selected"] = "SomeOtherProvider"

    with pytest.raises(AssertionError, match="routing mismatch"):
        _assert_fixture_proves_inside_billing(fixture)


def test_d1_billing_gate_requires_recorded_price_provenance():
    """The verdict must be reproducible from rates recorded at capture time, so a
    fixture missing its provenance is rejected rather than re-priced from today's
    registry."""
    fixture = _synthetic_fixture(reasoning=400_000, cost=0.75, label="inside")
    fixture["price_authority"]["fetched_at"] = None

    with pytest.raises(AssertionError, match="fetched_at"):
        _assert_fixture_proves_inside_billing(fixture)


def test_reasoning_usage_fixture_billing_relationship():
    """D1 honesty gate (load-bearing once captured). Whether reasoning tokens are
    billed INSIDE completion_tokens or SEPARATELY must come from a REAL captured
    payload on a VERIFIED route, decided by the billed `usage.cost` against the
    rates recorded at capture time. Capture with:

        OPENROUTER_API_KEY=... uv run python scripts/capture_reasoning_usage_fixture.py <model_id> <provider_slug>

    Applies the same policy as test_d1_billing_gate_synthetic_cases, so the live
    reading is judged by exactly the rules those cases prove. Skips until the
    placeholder fixture is overwritten."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        fixture = json.load(fh)
    if fixture.get("_placeholder"):
        pytest.skip(
            "real usage fixture not captured yet -- run "
            "scripts/capture_reasoning_usage_fixture.py <model_id> <provider_slug> with a live key"
        )

    usage = fixture["usage"]
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    assert reasoning > 0, "fixture must come from a real reasoning turn"

    assert _assert_fixture_proves_inside_billing(fixture) == "inside"


def test_captured_rates_match_current_registry_or_report_drift(monkeypatch):
    """Registry prices are mutable; a captured verdict is not. This surfaces drift
    between the rates stored at capture time and today's curated registry WITHOUT
    letting that drift silently reclassify the billing verdict (the verdict is
    always re-derived from the stored rates). Skips until a real capture exists."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        fixture = json.load(fh)
    if fixture.get("_placeholder"):
        pytest.skip("no captured fixture yet -- nothing to compare against the registry")

    main = import_main(monkeypatch)
    model_meta = main.get_model_by_id(fixture["model"])
    if not model_meta or not (model_meta.get("pricing") or {}).get("output"):
        pytest.skip(f"{fixture['model']} is not in the curated registry -- no drift baseline")

    price = fixture["price_authority"]
    # Registry rates are per MILLION tokens; captured rates are per token.
    registry_completion_per_token = model_meta["pricing"]["output"] / 1_000_000
    captured = price["completion_per_token"]
    drift = abs(registry_completion_per_token - captured)

    assert drift <= max(1e-12, 0.05 * captured), (
        f"captured completion rate {captured}/tok (from {price['source']} at "
        f"{price['fetched_at']}) has drifted from the registry's "
        f"{registry_completion_per_token}/tok. The D1 verdict still stands on the captured "
        "rates by design -- re-capture the fixture or refresh the registry, but do NOT "
        "silently reclassify."
    )
