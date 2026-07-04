"""run_full_council must return a 5-tuple even when every model fails (audit §4.2)."""
import importlib

import pytest
from backend import council
from backend.tools.types import EvidencePack, UsageLimits


@pytest.mark.asyncio
async def test_run_full_council_all_models_fail_returns_five_tuple(monkeypatch):
    async def all_fail_parallel(models, messages, **kwargs):
        return {m: None for m in models}

    async def fail_single(model, messages, **kwargs):
        return None

    monkeypatch.setattr(council, "query_models_parallel", all_fail_parallel)
    monkeypatch.setattr(council, "query_model", fail_single)

    result = await council.run_full_council("What should we do?")

    assert len(result) == 5, "must unpack cleanly at main.py:831"
    stage1, stage2, stage3, metadata, evidence_pack = result
    assert stage1 == []
    assert stage2 == []
    assert stage3["model"] == "error"
    assert "failed" in stage3["response"].lower()
    assert isinstance(metadata, dict)
    assert metadata["label_to_model"] == {}
    assert "steward_usage" in metadata, "steward cost must survive the all-fail path"
    assert isinstance(evidence_pack, EvidencePack)


@pytest.mark.asyncio
async def test_send_message_all_fail_skips_indexing_and_returns_error(monkeypatch, tmp_path):
    """The sync council endpoint must return the clean error result, not KeyError
    on metadata["label_to_model"], and must not RAG-index the failed turn."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()), None

    async def all_fail_stage1(*args, **kwargs):
        return []

    async def fake_title(*args, **kwargs):
        return "title"

    indexed = []
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses", all_fail_stage1)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(main.rag_system, "index_session", lambda *a, **k: indexed.append(a))

    result = await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    assert result["type"] == "council"
    assert result["stage3"]["model"] == "error"
    assert indexed == [], "failed turns must not be RAG-indexed"
