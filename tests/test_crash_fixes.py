import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_sync_council_indexes_the_conversation_after_the_assistant_message_is_saved(monkeypatch):
    """Codex round 6: index_session used to take a turn_index computed from
    get_turn_index(conversation) -- a THIRD get_conversation read staged
    between add_assistant_message and indexing. That parameter is gone now
    (index_session computes its own memory turn number internally, under its
    write lock, unified with index_chat_turn's scheme -- see
    tests/test_rag_persistence.py::test_index_session_and_index_chat_turn_share_one_monotonic_turn_sequence
    for the id-collision fix itself). What THIS test still verifies: the
    council branch calls index_session with the right conversation_id after
    the assistant message is saved, not before."""
    main = import_module_with_api_key(monkeypatch, "backend.main")

    initial_conversation = {
        "id": "conv-sync-crash",
        "messages": [],
        "metadata": {},
    }
    indexed_conversation = {
        "id": "conv-sync-crash",
        "messages": [
            {"role": "user", "content": "What should we do?"},
            {"role": "assistant", "stage3": {"response": "Do the thing."}},
        ],
        "metadata": {},
    }

    monkeypatch.setattr(
        main.storage,
        "get_conversation",
        # Four reads: endpoint pre-flight, the expected_anchor read right
        # after add_assistant_message (Codex round 17), the fresh-metadata
        # ZDR check guarding the indexing block (_zdr_flipped_on, Codex
        # round 14), and the completion-event total-cost read.
        Mock(side_effect=[initial_conversation, indexed_conversation, indexed_conversation, indexed_conversation]),
    )
    monkeypatch.setattr(main.storage, "add_user_message", Mock())
    monkeypatch.setattr(main.storage, "update_conversation_title", Mock())
    monkeypatch.setattr(main.storage, "add_assistant_message", Mock())
    monkeypatch.setattr(main.storage, "update_conversation_cost", Mock())
    monkeypatch.setattr(
        main.storage,
        "record_session_usage",
        Mock(return_value={"usage": {}, "warning_level": None, "budget_spent_pct": None}),
    )

    async def fake_generate_conversation_title(*args, **kwargs):
        return "Test title"

    monkeypatch.setattr(main, "generate_conversation_title", fake_generate_conversation_title)

    from backend.tools.types import EvidencePack

    async def fake_run_tool_steward_phase(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="What should we do?"), None

    # stage1_collect_responses_progressive is the pipeline seam (P3-T6): an
    # async generator yielding ("model_complete", index, result) per model
    # then ("complete", stage1_results, None) with the full list.
    async def fake_stage1_progressive(*args, **kwargs):
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

    async def fake_extract_topics(*args, **kwargs):
        return (["planning"], {})

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_run_tool_steward_phase)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr("backend.council.calculate_quality_metrics", Mock(return_value={"model-a": {}}))

    fake_rag = SimpleNamespace(
        index_session=AsyncMock(return_value=None),
        refresh_hybrid_index=Mock(),
    )
    monkeypatch.setattr(main, "rag_system", fake_rag)

    result = await main.send_message(
        "conv-sync-crash",
        main.SendMessageRequest(content="What should we do?", mode="council"),
    )

    assert result["type"] == "council"
    fake_rag.index_session.assert_called_once()
    assert fake_rag.index_session.call_args.args[0] == "conv-sync-crash"


def test_truncate_messages_does_not_crash_when_logging(monkeypatch, tmp_path):
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    conversation_id = "conv-edit-crash"

    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))

    storage.create_conversation(conversation_id)
    storage.add_user_message(conversation_id, "Original")
    storage.add_chat_message(conversation_id, "Response")

    truncated = storage.truncate_messages(conversation_id, 1)

    assert len(truncated["messages"]) == 1
    assert truncated["messages"][0]["content"] == "Original"
