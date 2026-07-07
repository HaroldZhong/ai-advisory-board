"""run_full_council must return a 5-tuple even when every model fails (audit §4.2)."""
import importlib
import json
from types import SimpleNamespace

from fastapi import HTTPException
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
    """The sync council endpoint must surface all-fail as a retryable error,
    not persist/index a fake assistant turn."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()), None

    # stage1_collect_responses_progressive is the pipeline seam (P3-T6): an
    # async generator yielding ("model_complete", index, result) per model
    # then ("complete", stage1_results, None) with the full list. All models
    # failing means no model_complete events, just an empty aggregate.
    async def all_fail_stage1_progressive(*args, **kwargs):
        yield "complete", [], None

    async def fake_title(*args, **kwargs):
        return "title"

    indexed = []
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", all_fail_stage1_progressive)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(main.rag_system, "index_session", lambda *a, **k: indexed.append(a))

    with pytest.raises(HTTPException) as excinfo:
        await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    assert excinfo.value.status_code == 500
    assert "All models failed" in excinfo.value.detail
    assert indexed == [], "failed turns must not be RAG-indexed"

    saved = main.storage.get_conversation(conv_id)
    assert [message["role"] for message in saved["messages"]] == ["user"]
    assert "All models failed" not in json.dumps(saved)


@pytest.mark.asyncio
async def test_send_message_all_network_fail_names_network(monkeypatch, tmp_path):
    """All-network Stage 1 failures should point users at proxy/relay setup."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()), None

    async def network_fail_stage1_progressive(*args, **kwargs):
        yield "complete", [], {"failure_kinds": ["network", "network"]}

    async def fake_title(*args, **kwargs):
        return "title"

    indexed = []
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", network_fail_stage1_progressive)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(main.rag_system, "index_session", lambda *a, **k: indexed.append(a))

    with pytest.raises(HTTPException) as excinfo:
        await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    assert excinfo.value.status_code == 500
    assert "Could not reach openrouter.ai" in excinfo.value.detail
    assert "HTTPS_PROXY" in excinfo.value.detail
    assert indexed == []

    saved = main.storage.get_conversation(conv_id)
    assert [message["role"] for message in saved["messages"]] == ["user"]
    assert "Could not reach openrouter.ai" not in json.dumps(saved)


@pytest.mark.asyncio
async def test_send_message_mixed_all_fail_stays_generic(monkeypatch, tmp_path):
    """Mixed all-fail causes should not masquerade as a network block."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", tmp_path)

    conversation = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()), None

    async def mixed_fail_stage1_progressive(*args, **kwargs):
        yield "complete", [], {"failure_kinds": ["network", "timeout"]}

    async def fake_title(*args, **kwargs):
        return "title"

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", mixed_fail_stage1_progressive)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)

    with pytest.raises(HTTPException) as excinfo:
        await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    assert excinfo.value.status_code == 500
    assert "All models failed" in excinfo.value.detail
    assert "Could not reach openrouter.ai" not in excinfo.value.detail


@pytest.mark.asyncio
async def test_stage3_synthesis_failure_is_retryable_and_not_indexed(monkeypatch, tmp_path):
    """A Stage 3 chairman failure after successful stages 1/2 must not be
    persisted or indexed as a normal assistant answer."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    main = importlib.import_module("backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))

    conversation = await main.create_conversation(main.CreateConversationRequest())
    conv_id = conversation["id"]

    async def fake_steward(*args, **kwargs):
        return (
            EvidencePack(run_id="r", query="q", tools_used=[], key_facts=[], limits=UsageLimits()),
            {"prompt_tokens": 1000, "completion_tokens": 1000},
        )

    async def fake_stage1_progressive(*args, **kwargs):
        result = {
            "model": "openai/gpt-4o-mini",
            "response": "Answer A",
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
        }
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{
                "model": "openai/gpt-4o-mini",
                "ranking": "1. Response A",
                "parsed_ranking": ["Response A"],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
            }],
            {"Response A": "openai/gpt-4o-mini"},
        )

    async def fake_title(*args, **kwargs):
        return "title"

    attempts = []

    async def fail_stage3_query(*args, **kwargs):
        attempts.append(args)
        return None

    async def fake_index_session(*args, **kwargs):
        indexed.append(args)
        return None

    indexed = []
    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr(council, "query_model", fail_stage3_query)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            index_session=fake_index_session,
            refresh_hybrid_index=lambda *a, **k: None,
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        await main.send_message(conv_id, main.SendMessageRequest(content="hi", mode="council"))

    assert excinfo.value.status_code == 500
    assert "final synthesis" in excinfo.value.detail
    assert len(attempts) == 2
    assert indexed == [], "failed Stage 3 turns must not be RAG-indexed"

    saved = main.storage.get_conversation(conv_id)
    assert [message["role"] for message in saved["messages"]] == ["user"]
    assert "Unable to generate final synthesis" not in json.dumps(saved)
    assert saved["total_cost"] > 0
    assert saved["session_usage"]["spent_usd"] == pytest.approx(saved["total_cost"])
    assert saved["session_usage"]["messages"] == 1
