"""Ask the council on any turn (P3-T4, master plan P3-W2, owner decision #3).

Mid-conversation council runs must resolve the follow-up into ONE
self-contained, rewritten query and hand that SAME text to all four
pipeline consumers (tool steward, Stage 1, Stage 2, Stage 3) — the stored
user message keeps the original text. First-turn council (no history) skips
the rewrite entirely: nothing to resolve.
"""
import importlib
from types import SimpleNamespace

import pytest


REWRITTEN = "REWRITTEN-QUERY"


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


def _setup_council_fakes(monkeypatch, main, captured):
    """Council fakes that capture the first positional (query) arg each
    consumer was called with, keyed by consumer name."""
    from backend.tools.types import EvidencePack

    async def fake_steward(content, *args, **kwargs):
        captured["steward"] = content
        return EvidencePack(run_id="run-1", query="q"), None

    async def fake_stage1_progressive(content, *args, **kwargs):
        captured["stage1"] = content
        result = {"model": "model-a", "response": "Answer A", "usage": {}}
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(content, *args, **kwargs):
        captured["stage2"] = content
        return (
            [{"model": "model-a", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "model-a"},
        )

    async def fake_stage3(content, *args, **kwargs):
        captured["stage3"] = content
        return {"model": "chair", "response": "Council answer", "usage": {}}

    async def fake_title(*args, **kwargs):
        return "Title"

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


def _populated_conversation(main, conversation_id):
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "First question")
    main.storage.add_chat_message(conversation_id, "First answer")


@pytest.mark.asyncio
async def test_mid_conversation_council_rewrites_query_for_all_four_consumers(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    _populated_conversation(main, "conv-council-followup")

    captured = {}
    _setup_council_fakes(monkeypatch, main, captured)

    async def fake_rewrite_query(query, history, zdr_enabled=False):
        return REWRITTEN

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)

    result = await main.send_message(
        "conv-council-followup",
        main.SendMessageRequest(content="What about it?", mode="council"),
    )

    assert result["type"] == "council"
    assert captured["steward"] == REWRITTEN
    # Stage 1's content is composed (attachments/custom instructions/web
    # search preamble may wrap it) so assert containment, not equality.
    assert REWRITTEN in captured["stage1"]
    assert captured["stage2"] == REWRITTEN
    assert captured["stage3"] == REWRITTEN

    # The stored user message keeps the ORIGINAL text (owner decision #3).
    conversation = main.storage.get_conversation("conv-council-followup")
    user_messages = [m for m in conversation["messages"] if m["role"] == "user"]
    assert user_messages[-1]["content"] == "What about it?"


@pytest.mark.asyncio
async def test_first_turn_council_skips_rewrite_entirely(monkeypatch, tmp_path):
    """Nothing to resolve on turn 1 — identical to today's behavior."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-council-first")

    captured = {}
    _setup_council_fakes(monkeypatch, main, captured)

    async def exploding_rewrite_query(*args, **kwargs):
        raise AssertionError("rewrite_query must not be called on a first-turn council run")

    monkeypatch.setattr("backend.council.rewrite_query", exploding_rewrite_query)

    result = await main.send_message(
        "conv-council-first",
        main.SendMessageRequest(content="What should we do?", mode="council"),
    )

    assert result["type"] == "council"
    assert captured["steward"] == "What should we do?"
    assert "What should we do?" in captured["stage1"]
    assert captured["stage2"] == "What should we do?"
    assert captured["stage3"] == "What should we do?"
