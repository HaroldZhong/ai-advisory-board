"""Characterization contracts for the unified turn pipeline (audit §5.1, P2-T5).

These tests pin the non-stream /message response shapes. They were written and
run GREEN against the pre-unification implementation, then kept green after
send_message became a collector over turn_pipeline.run_turn.

Model-call seams are patched in BOTH backend.main and backend.council
namespaces so the tests are agnostic to which module dispatches the calls
(pre-unification the council path went through council.run_full_council;
post-unification it goes through the stage functions bound on main).
"""
import importlib
from types import SimpleNamespace

import pytest


COUNCIL_KEYS = {
    "type", "stage1", "stage2", "stage3", "metadata", "evidence",
    "turn_cost", "total_cost", "session_usage", "budget_spent_pct",
}
CHAT_KEYS = {
    "type", "content", "reasoning", "turn_cost", "total_cost",
    "session_usage", "budget_spent_pct", "run_plan",
}


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def _patch_both(monkeypatch, main, name, value):
    """Patch a callable on backend.main AND backend.council namespaces."""
    council = importlib.import_module("backend.council")
    if hasattr(main, name):
        monkeypatch.setattr(main, name, value)
    if hasattr(council, name):
        monkeypatch.setattr(council, name, value)


def _setup_council_fakes(monkeypatch, main, stage1_calls=None):
    from backend.tools.types import EvidencePack

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="q"), {"prompt_tokens": 10, "completion_tokens": 5}

    # stage1_collect_responses_progressive is the pipeline seam (P3-T6): an
    # async generator yielding ("model_complete", index, result) per model
    # then ("complete", stage1_results, None) with the full list.
    async def fake_stage1_progressive(content, *args, **kwargs):
        if stage1_calls is not None:
            stage1_calls.append(content)
        result = {"model": "model-a", "response": "Answer A", "usage": {}}
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{"model": "model-a", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "model-a"},
        )

    async def fake_stage3(*args, **kwargs):
        return {"model": "chair", "response": "Final answer", "usage": {}}

    async def fake_title(*args, **kwargs):
        return "Test title"

    async def fake_topics(*args, **kwargs):
        return ["topic"]

    _patch_both(monkeypatch, main, "run_tool_steward_phase", fake_steward)
    _patch_both(monkeypatch, main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    _patch_both(monkeypatch, main, "stage2_collect_rankings", fake_stage2)
    _patch_both(monkeypatch, main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics", fake_topics)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            index_session=lambda *a, **k: None,
            refresh_hybrid_index=lambda *a, **k: None,
            index_document=lambda *a, **k: None,
        ),
    )


def _setup_chat_fakes(monkeypatch, main):
    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "Chat reply", "usage": {}, "reasoning": None}

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))


@pytest.mark.asyncio
async def test_non_stream_council_response_contract(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-contract-council")
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-contract-council",
        main.SendMessageRequest(content="What should we do?", mode="council"),
    )

    assert set(result.keys()) == COUNCIL_KEYS
    assert result["type"] == "council"
    assert result["stage1"][0]["response"] == "Answer A"
    assert result["stage3"]["response"] == "Final answer"
    assert result["metadata"]["label_to_model"] == {"Response A": "model-a"}
    assert result["evidence"] is not None
    assert isinstance(result["turn_cost"], float)


@pytest.mark.asyncio
async def test_non_stream_chat_response_contract(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-contract-chat")
    main.storage.add_user_message("conv-contract-chat", "Earlier question")
    main.storage.add_chat_message("conv-contract-chat", "Earlier answer")
    _setup_chat_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-contract-chat",
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert set(result.keys()) == CHAT_KEYS
    assert result["type"] == "chat"
    assert result["content"] == "Chat reply"
    assert isinstance(result["run_plan"], dict)


@pytest.mark.asyncio
async def test_non_stream_council_custom_instructions_reach_stage1(monkeypatch, tmp_path):
    """Drift proof (audit §5.1): pre-unification the non-stream endpoint silently
    dropped custom_instructions; the unified pipeline must feed them to Stage 1."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-drift-proof")
    stage1_calls = []
    _setup_council_fakes(monkeypatch, main, stage1_calls=stage1_calls)

    await main.send_message(
        "conv-drift-proof",
        main.SendMessageRequest(
            content="What should we do?",
            mode="council",
            custom_instructions="Answer like a pirate.",
        ),
    )

    assert len(stage1_calls) == 1
    assert "Answer like a pirate." in stage1_calls[0]
    assert "What should we do?" in stage1_calls[0]
