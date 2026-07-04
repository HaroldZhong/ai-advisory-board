"""Chat-first conversations must get a title too (P3-T3, master plan P3-W2).

Before this change, title generation only ran in the council branch on the
first message — a conversation created with default_mode="chat" (which runs
chat on EVERY turn, including the first) never got a title. The fix moves
title generation out of the council-only branch and gates it on "no
assistant message yet" so both chat-first and council-first conversations
get exactly one generated title.
"""
import importlib
import json
from types import SimpleNamespace

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

    async def fake_topics(*args, **kwargs):
        return ["topic"]

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses", fake_stage1)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
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
        return {"content": "Chat answer", "usage": {}}

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))


async def fake_title(*args, **kwargs):
    return "Sentinel Title"


@pytest.mark.asyncio
async def test_chat_first_message_generates_title(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    main.storage.create_conversation("conv-chat-title", {"default_mode": "chat"})
    _setup_chat_fakes(monkeypatch, main)

    response = await main.send_message_stream(
        "conv-chat-title",
        main.SendMessageRequest(content="First question", mode="auto"),
    )
    chunks = [chunk async for chunk in response.body_iterator]
    events = parse_sse_events(chunks)

    assert {"type": "title_complete", "data": {"title": "Sentinel Title"}} in events
    conversation = main.storage.get_conversation("conv-chat-title")
    assert conversation["title"] == "Sentinel Title"


@pytest.mark.asyncio
async def test_council_first_message_still_generates_title(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    main.storage.create_conversation("conv-council-title")
    _setup_council_fakes(monkeypatch, main)

    result = await main.send_message(
        "conv-council-title",
        main.SendMessageRequest(content="First question", mode="auto"),
    )

    assert result["type"] == "council"
    conversation = main.storage.get_conversation("conv-council-title")
    assert conversation["title"] == "Sentinel Title"


@pytest.mark.asyncio
async def test_second_chat_message_does_not_regenerate_title(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation("conv-chat-title-2", {"default_mode": "chat"})
    main.storage.add_user_message("conv-chat-title-2", "First question")
    main.storage.add_chat_message("conv-chat-title-2", "First answer")
    main.storage.update_conversation_title("conv-chat-title-2", "Original Title")
    _setup_chat_fakes(monkeypatch, main)

    call_count = 0

    async def counting_title(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return "Should Not Be Used"

    monkeypatch.setattr(main, "generate_conversation_title", counting_title)

    result = await main.send_message(
        "conv-chat-title-2",
        main.SendMessageRequest(content="Follow up", mode="auto"),
    )

    assert result["type"] == "chat"
    assert call_count == 0
    conversation = main.storage.get_conversation("conv-chat-title-2")
    assert conversation["title"] == "Original Title"
