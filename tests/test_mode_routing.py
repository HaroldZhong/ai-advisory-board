"""Backend-owned auto-mode resolution must account for edit truncation (P3-T1).

The frontend always sends mode="auto"; the backend resolves council-vs-chat
from the EFFECTIVE message count (pending edit_index truncation included).
Pre-fix, auto resolved before truncation, so an edit-back-to-message-0 send
would wrongly run chat — which is why the frontend used to compute the mode
itself.
"""
import importlib
from types import SimpleNamespace

import pytest


def import_main(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module("backend.main")


def _setup_council_fakes(monkeypatch, main):
    from backend.tools.types import EvidencePack

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="q"), None

    async def fake_stage1(*args, **kwargs):
        return [{"model": "model-a", "response": "Answer A", "usage": {}}]

    async def fake_stage2(*args, **kwargs):
        return (
            [{"model": "model-a", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "model-a"},
        )

    async def fake_stage3(*args, **kwargs):
        return {"model": "chair", "response": "Council answer", "usage": {}}

    async def fake_title(*args, **kwargs):
        return "Title"

    async def fake_topics(*args, **kwargs):
        return (["topic"], {})

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses", fake_stage1)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)

    async def fake_index_session(*args, **kwargs):
        return None

    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            index_session=fake_index_session,
            refresh_hybrid_index=lambda *a, **k: None,
            index_document=lambda *a, **k: None,
        ),
    )


def _setup_chat_fakes(monkeypatch, main):
    async def fake_title(*args, **kwargs):
        return "Title"

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_chat_with_chairman(*args, **kwargs):
        return {"content": "Chat answer", "usage": {}}

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    # Chat-first turns start a title task now — never let tests hit the network.
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=fake_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )


def _populated_conversation(main, conversation_id):
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "First question")
    main.storage.add_chat_message(conversation_id, "First answer")


@pytest.mark.asyncio
async def test_auto_with_edit_to_first_message_runs_council(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    _populated_conversation(main, "conv-edit-zero")
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-edit-zero",
        main.SendMessageRequest(content="Edited first question", mode="auto", edit_index=0),
    )

    assert result["type"] == "council"


@pytest.mark.asyncio
async def test_auto_mid_conversation_runs_chat(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    _populated_conversation(main, "conv-followup")
    _setup_chat_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-followup",
        main.SendMessageRequest(content="Follow up", mode="auto"),
    )

    assert result["type"] == "chat"


@pytest.mark.asyncio
async def test_auto_with_edit_mid_conversation_runs_chat(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    _populated_conversation(main, "conv-edit-two")
    _setup_chat_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-edit-two",
        main.SendMessageRequest(content="Edited follow up", mode="auto", edit_index=2),
    )

    assert result["type"] == "chat"

@pytest.mark.asyncio
async def test_auto_with_stale_edit_index_beyond_count_runs_council_when_empty(monkeypatch, tmp_path):
    """A stale client edit_index larger than the stored count clamps to the
    real count: an empty conversation routes to council regardless."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-stale-edit")
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-stale-edit",
        main.SendMessageRequest(content="Question", mode="auto", edit_index=2),
    )

    assert result["type"] == "council"


# --- default_mode routing (P3-T3): Chat-default / Council-explicit ---


@pytest.mark.asyncio
async def test_default_mode_chat_runs_chat_on_first_message(monkeypatch, tmp_path):
    """metadata.default_mode == 'chat' means EVERY turn runs chat, including the first."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-chat-default-first", {"default_mode": "chat"})
    _setup_chat_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-chat-default-first",
        main.SendMessageRequest(content="First question", mode="auto"),
    )

    assert result["type"] == "chat"


@pytest.mark.asyncio
async def test_default_mode_council_runs_council_on_first_message(monkeypatch, tmp_path):
    """metadata.default_mode == 'council' keeps today's auto behavior: council on the
    effectively-first turn."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-council-default-first", {"default_mode": "council"})
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-council-default-first",
        main.SendMessageRequest(content="First question", mode="auto"),
    )

    assert result["type"] == "council"


@pytest.mark.asyncio
async def test_default_mode_council_runs_chat_on_followup(monkeypatch, tmp_path):
    """metadata.default_mode == 'council' still runs chat after the first turn."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-council-default-followup", {"default_mode": "council"})
    main.storage.add_user_message("conv-council-default-followup", "First question")
    main.storage.add_chat_message("conv-council-default-followup", "First answer")
    _setup_chat_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-council-default-followup",
        main.SendMessageRequest(content="Follow up", mode="auto"),
    )

    assert result["type"] == "chat"


@pytest.mark.asyncio
async def test_explicit_request_mode_overrides_default_mode_chat(monkeypatch, tmp_path):
    """An explicit request.mode wins over metadata.default_mode."""
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-explicit-override", {"default_mode": "chat"})
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-explicit-override",
        main.SendMessageRequest(content="First question", mode="council"),
    )

    assert result["type"] == "council"
