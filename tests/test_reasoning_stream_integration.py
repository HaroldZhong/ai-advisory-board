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


@pytest.mark.asyncio
async def test_stream_chat_emits_reasoning_and_content_events(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-stream-chat-reasoning"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.7"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        return {
            "content": "Visible response",
            "reasoning": "Reasoned through the answer",
            "usage": {},
        }

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    events = parse_sse_events(chunks)

    assert {
        "type": "reasoning_delta",
        "data": {
            "scope": "chat",
            "stage": "chat",
            "model": "anthropic/claude-opus-4.7",
            "text": "Reasoned through the answer",
            "detail_type": "reasoning.text",
        },
    } in events
    assert {
        "type": "content_delta",
        "data": {
            "scope": "chat",
            "stage": "chat",
            "model": "anthropic/claude-opus-4.7",
            "text": "Visible response",
        },
    } in events
    assert any(event["type"] == "chat_response" for event in events)
    stored = main.storage.get_conversation(conversation_id)
    assert stored["messages"][-1]["reasoning"] == "Reasoned through the answer"


@pytest.mark.asyncio
async def test_sync_chat_persists_reasoning(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-sync-chat-reasoning"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "anthropic/claude-opus-4.7"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        return {
            "content": "Visible sync response",
            "reasoning": "Stored sync reasoning",
            "usage": {},
        }

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert result["reasoning"] == "Stored sync reasoning"
    stored = main.storage.get_conversation(conversation_id)
    assert stored["messages"][-1]["content"] == "Visible sync response"
    assert stored["messages"][-1]["reasoning"] == "Stored sync reasoning"


@pytest.mark.asyncio
async def test_stream_chat_delta_events_use_default_chairman_model(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-stream-chat-default-chairman-reasoning"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return ""

    async def fake_chat_with_chairman(*args, **kwargs):
        return {
            "content": "Visible response",
            "reasoning": "Reasoned through the answer",
            "usage": {},
        }

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    events = parse_sse_events(chunks)
    delta_events = [
        event for event in events if event["type"] in {"reasoning_delta", "content_delta"}
    ]

    assert delta_events
    assert {event["data"]["model"] for event in delta_events} == {main.config.CHAIRMAN_MODEL}


@pytest.mark.asyncio
async def test_stream_council_emits_reasoning_events_for_each_stage(monkeypatch, tmp_path):
    main = import_main(monkeypatch)
    conversation_id = "conv-stream-council-reasoning"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {
            "chairman_model": "anthropic/claude-opus-4.7",
            "council_models": ["model-a"],
        },
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    evidence_pack = SimpleNamespace(model_dump=lambda: {"claims": []})

    async def fake_run_tool_steward_phase(*args, **kwargs):
        return evidence_pack, {}

    async def fake_stage1_collect_responses(*args, **kwargs):
        return [
            {
                "model": "model-a",
                "response": "Stage 1 visible",
                "reasoning": "Stage 1 reasoning",
                "usage": {},
            }
        ]

    async def fake_stage2_collect_rankings(*args, **kwargs):
        return [
            {
                "model": "model-a",
                "ranking": "FINAL RANKING:\n1. Response A",
                "parsed_ranking": ["Response A"],
                "reasoning": "Stage 2 reasoning",
                "usage": {},
            }
        ], {"Response A": "model-a"}

    async def fake_stage3_synthesize_final(*args, **kwargs):
        return {
            "model": "anthropic/claude-opus-4.7",
            "response": "Final visible",
            "reasoning": "Stage 3 reasoning",
            "usage": {},
            "confidence": "HIGH",
            "avg_consensus": 1.0,
            "quality_metrics": {},
        }

    async def fake_extract_topics(*args, **kwargs):
        return []

    fake_rag = SimpleNamespace(
        index_session=lambda *args, **kwargs: None,
        refresh_hybrid_index=lambda: None,
    )

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_run_tool_steward_phase)
    monkeypatch.setattr(main, "stage1_collect_responses", fake_stage1_collect_responses)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2_collect_rankings)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3_synthesize_final)
    monkeypatch.setattr("backend.council.extract_topics", fake_extract_topics)
    monkeypatch.setattr(main, "rag_system", fake_rag)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Run council", mode="council"),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    events = parse_sse_events(chunks)
    reasoning_events = [
        event for event in events if event["type"] == "reasoning_delta"
    ]

    assert [
        event["data"]["stage"] for event in reasoning_events
    ] == ["stage1", "stage2", "stage3"]
    assert [
        event["data"]["text"] for event in reasoning_events
    ] == ["Stage 1 reasoning", "Stage 2 reasoning", "Stage 3 reasoning"]
    assert any(event["type"] == "stage1_complete" for event in events)
    assert any(event["type"] == "stage2_complete" for event in events)
    assert any(event["type"] == "stage3_complete" for event in events)


@pytest.mark.asyncio
async def test_council_stage_functions_preserve_reasoning(monkeypatch):
    council = importlib.import_module("backend.council")

    async def fake_query_models_parallel(*args, **kwargs):
        return {
            "model-a": {
                "content": "Visible answer",
                "reasoning_details": "Model reasoning",
                "usage": {},
            }
        }

    async def fake_query_model(*args, **kwargs):
        return {
            "content": "Final visible",
            "reasoning_details": "Chairman reasoning",
            "usage": {},
        }

    monkeypatch.setattr(council, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council, "query_model", fake_query_model)

    stage1 = await council.stage1_collect_responses("Question", models=["model-a"])
    stage2, _label_to_model = await council.stage2_collect_rankings(
        "Question",
        [{"model": "model-a", "response": "Visible answer"}],
        models=["model-a"],
    )
    stage3 = await council.stage3_synthesize_final(
        "Question",
        [{"model": "model-a", "response": "Visible answer"}],
        [{"model": "model-a", "ranking": "FINAL RANKING:\n1. Response A"}],
        {"Response A": "model-a"},
        {"model-a": {"consensus_score": 1.0, "avg_rank": 1}},
        chairman_model="chairman",
    )

    assert stage1[0]["reasoning"] == "Model reasoning"
    assert stage2[0]["reasoning"] == "Model reasoning"
    assert stage3["reasoning"] == "Chairman reasoning"
