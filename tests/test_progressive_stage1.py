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
