"""query_models_as_completed must yield faster models before slower ones (P3-T6)."""
import asyncio

import pytest

from backend import council, openrouter


@pytest.mark.asyncio
async def test_query_models_as_completed_yields_fastest_first(monkeypatch):
    async def fake_query_model(model, messages, **kwargs):
        if model == "model-a":
            await asyncio.sleep(0.05)
        elif model == "model-b":
            await asyncio.sleep(0.5)
        return {"content": f"answer from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    order = []
    async for model, response in openrouter.query_models_as_completed(
        ["model-b", "model-a"], [{"role": "user", "content": "hi"}]
    ):
        order.append(model)
        assert response["content"] == f"answer from {model}"

    assert order == ["model-a", "model-b"], "fastest model must be yielded first"


@pytest.mark.asyncio
async def test_query_models_as_completed_yields_none_on_failure(monkeypatch):
    async def fake_query_model(model, messages, **kwargs):
        return None

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    results = {}
    async for model, response in openrouter.query_models_as_completed(
        ["model-a"], [{"role": "user", "content": "hi"}]
    ):
        results[model] = response

    assert results == {"model-a": None}


@pytest.mark.asyncio
async def test_stage1_collect_responses_progressive_matches_aggregate_shape(monkeypatch):
    """The progressive generator's per-model and final results must have the
    same shape as stage1_collect_responses (both use build_stage1_result)."""

    async def fake_query_model(model, messages, **kwargs):
        if model == "model-a":
            await asyncio.sleep(0.02)
        return {"content": f"answer from {model}", "reasoning_details": None, "usage": {"prompt_tokens": 1}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    events = []
    async for event in council.stage1_collect_responses_progressive(
        "question", models=["model-a", "model-b"]
    ):
        events.append(event)

    model_complete_events = [e for e in events if e[0] == "model_complete"]
    complete_events = [e for e in events if e[0] == "complete"]

    assert len(model_complete_events) == 2
    assert [idx for _, idx, _ in model_complete_events] == [0, 1]
    assert len(complete_events) == 1

    _, stage1_results, _ = complete_events[0]
    aggregate = await council.stage1_collect_responses("question", models=["model-a", "model-b"])

    assert stage1_results == aggregate


@pytest.mark.asyncio
async def test_stage1_model_complete_index_matches_final_aggregate_position(monkeypatch):
    """Codex round 3, PR #69: progressive events append in COMPLETION order
    but the aggregate is in CONFIGURED-model order. If the yielded index
    were raw completion order, a later-configured model finishing first
    would get index 0, then get bumped to a different slot when the
    aggregate replaces the list — reordering cards under the user's cursor.
    model-b (configured position 1) finishes before model-a (position 0);
    its yielded index must already be 1, matching its final position."""

    async def fake_query_model(model, messages, **kwargs):
        if model == "model-a":
            await asyncio.sleep(0.05)
        return {"content": f"answer from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    events = []
    async for event in council.stage1_collect_responses_progressive(
        "question", models=["model-a", "model-b"]
    ):
        events.append(event)

    model_complete_events = [e for e in events if e[0] == "model_complete"]
    complete_events = [e for e in events if e[0] == "complete"]
    _, stage1_results, _ = complete_events[0]

    for _, index, result in model_complete_events:
        assert stage1_results[index] == result, (
            f"model_complete index {index} for {result['model']} does not "
            "match its position in the final aggregate"
        )

    assert [r["model"] for r in stage1_results] == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_stage1_model_complete_index_stable_when_earlier_model_fails(monkeypatch):
    """A failed model is dropped from the aggregate (build_stage1_result
    returns None), so a later model's final index shifts down. The
    progressive generator must buffer until the failure resolves before
    assigning the later model's index, not yield-then-correct."""

    async def fake_query_model(model, messages, **kwargs):
        if model == "model-a":
            await asyncio.sleep(0.05)
            return None  # fails
        return {"content": f"answer from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    events = []
    async for event in council.stage1_collect_responses_progressive(
        "question", models=["model-a", "model-b"]
    ):
        events.append(event)

    model_complete_events = [e for e in events if e[0] == "model_complete"]
    complete_events = [e for e in events if e[0] == "complete"]
    _, stage1_results, _ = complete_events[0]

    assert [r["model"] for r in stage1_results] == ["model-b"]
    assert len(model_complete_events) == 1
    _, index, result = model_complete_events[0]
    assert index == 0
    assert result["model"] == "model-b"


@pytest.mark.asyncio
async def test_query_models_as_completed_cancels_pending_on_early_close(monkeypatch):
    """Closing the generator mid-stream (browser closes/navigates) must cancel
    in-flight model calls instead of letting them run to their full HTTP
    timeout for nothing (Codex P2, PR #69 round 2)."""
    cancelled = []

    async def fake_query_model(model, messages, **kwargs):
        if model == "model-fast":
            return {"content": "fast answer", "usage": {}}
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(model)
            raise
        return {"content": "slow answer", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    gen = openrouter.query_models_as_completed(
        ["model-slow", "model-fast"], [{"role": "user", "content": "hi"}]
    )
    first_model, _ = await gen.__anext__()
    assert first_model == "model-fast"

    await gen.aclose()
    # Let the cancelled task's except-block actually run.
    await asyncio.sleep(0)

    assert cancelled == ["model-slow"]
