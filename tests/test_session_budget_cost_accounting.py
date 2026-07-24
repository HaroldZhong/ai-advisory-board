import importlib
import json
import os
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


def _billing_classifier():
    """The single source of the inside/separate decision, imported from the
    capture harness so gate and capture can never drift apart."""
    return importlib.import_module("scripts.capture_reasoning_usage_fixture")


def _synthetic_fixture(*, reasoning, cost, label, model="openai/gpt-4o-mini"):
    """A fixture shaped like a real capture. openai/gpt-4o-mini is priced
    input=$0.15/M, output=$0.6/M -> 1M/1M tokens is $0.75 completion-priced."""
    return {
        "_placeholder": False,
        "model": model,
        "provider_requested": "synthetic-provider",
        "provider_served": "synthetic-provider",
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        "cost": cost,
        "billing_relationship": label,
    }


def _assert_fixture_proves_inside_billing(main, fixture):
    """Shared D1 gate policy applied to a fixture dict.

    Raises AssertionError unless the fixture is provider-attributed, cost-bearing,
    and shows a clear INSIDE billing relationship per the single-sourced
    classifier. Returns the relationship on success."""
    capture = _billing_classifier()
    model = fixture["model"]
    usage = fixture["usage"]
    reported_cost = fixture.get("cost")
    recorded = fixture.get("billing_relationship")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    completion = usage.get("completion_tokens", 0)

    # The D1 decision is provider-specific; a fixture that cannot name its route
    # is not evidence for it.
    if not (fixture.get("provider_requested") or fixture.get("provider_served")):
        raise AssertionError(
            "fixture records no provider route (provider_requested/provider_served); the "
            "billing decision is provider-specific -- re-capture with a pinned provider"
        )
    if reported_cost is None:
        raise AssertionError(
            "capture must record OpenRouter's billed cost (usage.include=true) -- it is the authority"
        )

    model_meta = main.get_model_by_id(model)
    if not model_meta or (model_meta.get("pricing") or {}).get("output") is None:
        raise AssertionError(
            f"{model} is not registry-priced; capture with a registry-priced model so "
            "billed cost can be compared"
        )
    output_rate = model_meta["pricing"]["output"]
    completion_priced = main.calculate_cost(usage, model)

    relationship, detail = capture.classify_billing(
        reasoning_tokens=reasoning,
        completion_tokens=completion,
        reported_cost=reported_cost,
        completion_priced_cost=completion_priced,
        output_rate=output_rate,
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
    "case, reasoning, cost, label, should_pass",
    [
        # cost 0.75 == completion-priced -> no surcharge, signal large enough
        ("inside/large", 400_000, 0.75, "inside", True),
        # cost 0.99 == completion-priced + 400k*0.6/M -> clear separate charge
        ("separate/large", 400_000, 0.99, "separate", False),
        # a REAL separate charge too small to distinguish from noise: must be
        # ambiguous, never a false 'inside' (the false-pass hole)
        ("separate/small", 10_000, 0.756, "ambiguous", False),
        # genuinely inside but signal too small to prove it -> ambiguous
        ("inside/small", 10_000, 0.75, "ambiguous", False),
        # label claims inside while cost shows separate -> mismatch rejected
        ("mislabeled-inside", 400_000, 0.99, "inside", False),
    ],
)
def test_d1_billing_gate_synthetic_cases(monkeypatch, case, reasoning, cost, label, should_pass):
    """Proves the D1 gate is load-bearing WITHOUT a live capture, preserving the
    five cases as runnable tests (they were previously only demonstrated ad-hoc by
    swapping the fixture file in and out).

    Only a clear, provider-attributed INSIDE reading may pass. Separate billing,
    an indistinguishable surcharge, and a hand-edited label must all be rejected."""
    main = import_main(monkeypatch)
    fixture = _synthetic_fixture(reasoning=reasoning, cost=cost, label=label)

    if should_pass:
        assert _assert_fixture_proves_inside_billing(main, fixture) == "inside"
    else:
        with pytest.raises(AssertionError):
            _assert_fixture_proves_inside_billing(main, fixture)


def test_d1_billing_gate_requires_provider_evidence(monkeypatch):
    """A fixture with no recorded route cannot support a provider-specific
    billing decision, even when the numbers themselves read as 'inside'."""
    main = import_main(monkeypatch)
    fixture = _synthetic_fixture(reasoning=400_000, cost=0.75, label="inside")
    fixture["provider_requested"] = None
    fixture["provider_served"] = None

    with pytest.raises(AssertionError, match="provider"):
        _assert_fixture_proves_inside_billing(main, fixture)


def test_reasoning_usage_fixture_billing_relationship(monkeypatch):
    """D1 honesty gate (load-bearing once captured). Whether reasoning tokens
    are billed INSIDE completion_tokens or SEPARATELY must come from a REAL
    captured payload, decided by OpenRouter's billed `cost` -- NOT token
    containment (reasoning<=completion holds either way). Passes only when the
    fixture shows INSIDE billing (meter truthful as-is); FAILS on SEPARATE
    billing, because current calculate_cost then under-charges and D1 must add a
    provider-specific reasoning line. Capture with:

        OPENROUTER_API_KEY=... uv run python scripts/capture_reasoning_usage_fixture.py <model_id>

    Applies the same policy as test_d1_billing_gate_synthetic_cases, so the live
    reading is judged by exactly the rules those cases prove. Skips until the
    placeholder fixture is overwritten."""
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        fixture = json.load(fh)
    if fixture.get("_placeholder"):
        pytest.skip(
            "real usage fixture not captured yet -- run "
            "scripts/capture_reasoning_usage_fixture.py <model_id> [provider_slug] with a live key"
        )

    main = import_main(monkeypatch)
    usage = fixture["usage"]
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    assert reasoning > 0, "fixture must come from a real reasoning turn"

    assert _assert_fixture_proves_inside_billing(main, fixture) == "inside"
