import importlib

import pytest


@pytest.mark.asyncio
async def test_stage2_preserves_unavailable_evaluator_slots(monkeypatch):
    council = importlib.import_module("backend.council")

    async def fake_query_models_parallel(models, messages, **kwargs):
        return {
            "model-a": {
                "content": "FINAL RANKING:\n1. Response A\n2. Response B",
                "usage": {"total_tokens": 10},
            },
            "model-b": None,
        }

    monkeypatch.setattr(council, "query_models_parallel", fake_query_models_parallel)

    stage2, label_to_model = await council.stage2_collect_rankings(
        "Question",
        [
            {"model": "model-a", "response": "Answer A"},
            {"model": "model-b", "response": "Answer B"},
        ],
        models=["model-a", "model-b"],
    )

    assert label_to_model == {
        "Response A": "model-a",
        "Response B": "model-b",
    }
    assert [result["model"] for result in stage2] == ["model-a", "model-b"]
    assert stage2[0]["parsed_ranking"] == ["Response A", "Response B"]
    assert stage2[1]["status"] == "unavailable"
    assert stage2[1]["parsed_ranking"] == []
    assert "model-b did not return a Stage 2 evaluation" in stage2[1]["ranking"]
