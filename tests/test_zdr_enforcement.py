import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


def _setup_council_fakes(monkeypatch, main, rag_system):
    """Mirrors tests/test_pipeline_unification.py's council fakes, but lets the
    caller supply the rag_system fake/spy so ZDR-guard behavior is observable."""
    from backend.tools.types import EvidencePack

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="q"), {"prompt_tokens": 10, "completion_tokens": 5}

    # stage1_collect_responses_progressive is the pipeline seam (P3-T6): an
    # async generator yielding ("model_complete", index, result) per model
    # then ("complete", stage1_results, None) with the full list.
    async def fake_stage1_progressive(content, *args, **kwargs):
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
        return (["topic"], {})

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "rag_system", rag_system)


@pytest.mark.asyncio
async def test_query_model_adds_provider_zdr_when_enabled(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            captured_payloads.append(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(openrouter.httpx, "AsyncClient", FakeAsyncClient)

    await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )
    await openrouter.query_model(
        "model-a",
        [{"role": "user", "content": "hi"}],
    )

    assert captured_payloads[0]["provider"] == {"zdr": True}
    assert "provider" not in captured_payloads[1]


@pytest.mark.asyncio
async def test_parallel_model_queries_propagate_zdr(monkeypatch):
    openrouter = import_module_with_api_key(monkeypatch, "backend.openrouter")
    captured = []

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False):
        captured.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
            }
        )
        return {"content": f"response from {model}", "usage": {}}

    monkeypatch.setattr(openrouter, "query_model", fake_query_model)

    result = await openrouter.query_models_parallel(
        ["model-a", "model-b"],
        [{"role": "user", "content": "hi"}],
        zdr_enabled=True,
    )

    assert set(result.keys()) == {"model-a", "model-b"}
    assert [call["zdr_enabled"] for call in captured] == [True, True]


@pytest.mark.asyncio
async def test_council_stage_calls_pass_zdr_to_openrouter(monkeypatch):
    council = import_module_with_api_key(monkeypatch, "backend.council")
    parallel_calls = []
    single_calls = []

    async def fake_query_models_parallel(models, messages, zdr_enabled=False):
        parallel_calls.append(
            {"models": models, "messages": messages, "zdr_enabled": zdr_enabled}
        )
        return {
            model: {"content": f"response from {model}", "usage": {}}
            for model in models
        }

    async def fake_query_model(model, messages, timeout=120.0, zdr_enabled=False):
        single_calls.append(
            {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                "zdr_enabled": zdr_enabled,
            }
        )
        return {"content": "chairman response", "usage": {}}

    monkeypatch.setattr(council, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council, "query_model", fake_query_model)

    stage1_results = await council.stage1_collect_responses(
        "Question?",
        models=["model-a", "model-b"],
        zdr_enabled=True,
    )
    stage2_results, label_to_model = await council.stage2_collect_rankings(
        "Question?",
        stage1_results,
        models=["model-a", "model-b"],
        zdr_enabled=True,
    )
    await council.stage3_synthesize_final(
        "Question?",
        stage1_results,
        stage2_results,
        label_to_model,
        {"model-a": {"consensus_score": 1.0, "avg_rank": 1.0}},
        chairman_model="chair-model",
        zdr_enabled=True,
    )
    await council.run_tool_steward_phase(
        "Question?",
        run_id="run-1",
        chairman_model="chair-model",
        zdr_enabled=True,
    )

    assert [call["zdr_enabled"] for call in parallel_calls] == [True, True]
    assert [call["zdr_enabled"] for call in single_calls] == [True, True]


@pytest.mark.asyncio
async def test_sync_chat_passes_zdr_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-sync-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
    )

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_sync_chat_uses_conversation_zdr_metadata(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-sync-metadata-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini", "zdr_enabled": True},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=False),
    )

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_stream_chat_passes_zdr_to_chairman(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=True),
    )

    async for _chunk in response.body_iterator:
        pass

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_stream_chat_uses_conversation_zdr_metadata(monkeypatch, tmp_path):
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-stream-metadata-zdr"
    captured_kwargs = {}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini", "zdr_enabled": True},
    )
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "ZDR response", "usage": {}}

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "rag_system", SimpleNamespace(retrieve_async=fake_retrieve_async))
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)

    response = await main.send_message_stream(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat", zdr_enabled=False),
    )

    async for _chunk in response.body_iterator:
        pass

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_web_search_passes_zdr_to_openrouter(monkeypatch):
    web_search = import_module_with_api_key(monkeypatch, "backend.web_search")
    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Search result https://example.com", "usage": {}}

    monkeypatch.setattr(web_search, "query_model", fake_query_model)

    await web_search.web_search_stage0("current facts", zdr_enabled=True)

    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_reasoning_rag_passes_zdr_to_openrouter(monkeypatch, tmp_path):
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")
    rag = rag_module.CouncilRAG(persist_path=str(tmp_path))
    captured_kwargs = {}

    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": "Useful prior memory", "message_anchor": 1000}],
        },
    }

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Relevant prior memory"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)
    # Codex round 10 read barrier: fake a valid non-ZDR source conversation
    # for "other" (this test isn't about the barrier itself). Codex round 26:
    # give it enough messages to satisfy the entry's message_anchor above.
    monkeypatch.setattr(rag_module, "get_conversation", lambda cid: {"metadata": {}, "messages": [{}] * 1000})

    context, usage = await rag.retrieve_async("current question", "current", zdr_enabled=True)

    assert context == "Relevant prior memory"
    assert captured_kwargs["zdr_enabled"] is True


@pytest.mark.asyncio
async def test_image_file_processing_passes_zdr_to_openrouter(monkeypatch):
    file_processing = import_module_with_api_key(monkeypatch, "backend.file_processing")
    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Image description"}

    monkeypatch.setattr(file_processing, "query_model", fake_query_model)

    result = await file_processing.process_file(
        b"fake-image-bytes",
        "diagram.png",
        "image/png",
        zdr_enabled=True,
    )

    assert result.status == "success"
    assert captured_kwargs["zdr_enabled"] is True


# --- P5-T2: ZDR memory boundary (index-time exclusion, Decision #5) ---


@pytest.mark.asyncio
async def test_council_turn_with_metadata_zdr_skips_memory_indexing(monkeypatch, tmp_path):
    """A conversation whose metadata already sets zdr_enabled=True must never
    have its turns written into the cross-conversation PageIndex memory store."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-zdr-metadata-council"
    main.storage.create_conversation(conversation_id, {"zdr_enabled": True})

    rag_system = SimpleNamespace(
        index_session=AsyncMock(return_value=None),
        index_document=AsyncMock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Sensitive question", mode="council"),
    )

    rag_system.index_session.assert_not_called()


@pytest.mark.asyncio
async def test_council_turn_with_per_message_zdr_skips_memory_indexing(monkeypatch, tmp_path):
    """A non-ZDR conversation with a single per-message zdr_enabled=True flag
    must also skip indexing that turn (effective ZDR = metadata OR per-message)."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-zdr-per-message-council"
    main.storage.create_conversation(conversation_id)  # no metadata ZDR

    rag_system = SimpleNamespace(
        index_session=AsyncMock(return_value=None),
        index_document=AsyncMock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Sensitive question", mode="council", zdr_enabled=True),
    )

    rag_system.index_session.assert_not_called()


@pytest.mark.asyncio
async def test_non_zdr_council_turn_still_indexes_memory(monkeypatch, tmp_path):
    """Control: a normal (non-ZDR) turn must still be indexed, proving the
    guard is ZDR-specific and not a blanket regression."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-non-zdr-council"
    main.storage.create_conversation(conversation_id)

    rag_system = SimpleNamespace(
        index_session=AsyncMock(return_value=None),
        index_document=AsyncMock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Normal question", mode="council"),
    )

    rag_system.index_session.assert_called_once()


@pytest.mark.asyncio
async def test_council_turn_skips_indexing_when_zdr_enabled_mid_turn(monkeypatch, tmp_path):
    """Codex P1 TOCTOU, closed at the root in round 6: a turn starts with the
    pre-flight zdr_enabled=False captured before the council ran. If the user
    flips ZDR on via PUT /api/conversations/{id} while the turn is still in
    flight, update_conversation purges this conversation's existing memories
    immediately. CouncilRAG.index_session's own ZDR write barrier re-checks
    CURRENT metadata synchronously right before it mutates the store, so it
    must not re-add memories after that purge -- this uses a REAL CouncilRAG
    (wrapped for call-observability) so the write barrier actually runs,
    rather than a bare mock that would bypass it entirely."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    rag_module = importlib.import_module("backend.rag")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))
    conversation_id = "conv-zdr-flipped-mid-turn"
    main.storage.create_conversation(conversation_id)  # starts non-ZDR
    monkeypatch.setattr(rag_module, "get_conversation", main.storage.get_conversation)
    real_rag = rag_module.CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    from backend.tools.types import EvidencePack

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="q"), {"prompt_tokens": 10, "completion_tokens": 5}

    async def fake_stage1_progressive(content, *args, **kwargs):
        result = {"model": "model-a", "response": "Answer A", "usage": {}}
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{"model": "model-a", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "model-a"},
        )

    async def fake_stage3(*args, **kwargs):
        # Simulate the user flipping ZDR on via PUT /api/conversations/{id}
        # while stage3 (or any earlier stage) was still running: real
        # storage metadata now says zdr_enabled=True, but this turn's
        # zdr_enabled local variable was already captured as False.
        main.storage.update_conversation_metadata(conversation_id, {"zdr_enabled": True})
        return {"model": "chair", "response": "Final answer", "usage": {}}

    async def fake_title(*args, **kwargs):
        return "Test title"

    async def fake_topics(*args, **kwargs):
        return (["topic"], {})

    rag_system = SimpleNamespace(
        index_session=Mock(wraps=real_rag.index_session),
        index_document=Mock(wraps=real_rag.index_document),
        refresh_hybrid_index=Mock(),
    )

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "rag_system", rag_system)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Sensitive question", mode="council"),
    )

    # Codex round 14: the pipeline's fresh-metadata guard (_zdr_flipped_on)
    # now fires for a mid-turn flip, so the whole indexing block -- including
    # the topics utility call -- is skipped before index_session is even
    # reached. (Previously index_session was called and its internal write
    # barrier refused the store write; the barrier still exists as the
    # authoritative last line of defense.)
    rag_system.index_session.assert_not_called()
    assert conversation_id not in real_rag.store


@pytest.mark.asyncio
async def test_council_turn_with_zdr_and_attachments_skips_document_indexing(monkeypatch, tmp_path):
    """Attachments attached to a ZDR turn must not be indexed into PageIndex
    either (backend/turn_pipeline.py's attachment loop calls index_document)."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-zdr-attachments"
    main.storage.create_conversation(conversation_id, {"zdr_enabled": True})

    rag_system = SimpleNamespace(
        index_session=AsyncMock(return_value=None),
        index_document=AsyncMock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)
    monkeypatch.setattr(main, "build_llm_context", lambda attachment_ids: "[Attachment] secret contents")
    monkeypatch.setattr(main, "get_attachment_text", lambda attachment_id: "secret contents")
    monkeypatch.setattr(
        main,
        "prepare_message_attachments",
        lambda conversation_id, attachment_ids: [{"attachment_id": aid} for aid in attachment_ids],
    )

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(
            content="Sensitive question",
            mode="council",
            attachment_ids=["att-1"],
        ),
    )

    rag_system.index_document.assert_not_called()


@pytest.mark.asyncio
async def test_council_turn_skips_document_indexing_when_zdr_enabled_before_attachment_loop(monkeypatch, tmp_path):
    """Codex P1, round 5/6 (symmetric to the mid-turn index_session case,
    closed at the root): the attachment-index loop is the FIRST thing
    run_turn does, so a ZDR flip landing between the request's pre-flight
    zdr_enabled capture and this loop's start must also be caught.
    build_llm_context is the call immediately before the loop; simulate the
    flip happening there via real storage.update_conversation_metadata.
    CouncilRAG.index_document's own ZDR write barrier re-checks CURRENT
    metadata synchronously right before it mutates the store, so this uses a
    REAL CouncilRAG (wrapped for call-observability) rather than a bare mock
    that would bypass the barrier entirely."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    rag_module = importlib.import_module("backend.rag")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))
    conversation_id = "conv-zdr-flipped-before-attachments"
    main.storage.create_conversation(conversation_id)  # starts non-ZDR
    monkeypatch.setattr(rag_module, "get_conversation", main.storage.get_conversation)
    real_rag = rag_module.CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    rag_system = SimpleNamespace(
        index_session=Mock(wraps=real_rag.index_session),
        index_document=Mock(wraps=real_rag.index_document),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    def flipping_build_llm_context(attachment_ids):
        # Simulate the user flipping ZDR on via PUT /api/conversations/{id}
        # right as this turn starts: real storage metadata now says
        # zdr_enabled=True, but this turn's zdr_enabled local variable was
        # already captured as False during pre-flight.
        main.storage.update_conversation_metadata(conversation_id, {"zdr_enabled": True})
        return "[Attachment] secret contents"

    monkeypatch.setattr(main, "build_llm_context", flipping_build_llm_context)
    monkeypatch.setattr(main, "get_attachment_text", lambda attachment_id: "secret contents")
    monkeypatch.setattr(
        main,
        "prepare_message_attachments",
        lambda conversation_id, attachment_ids: [{"attachment_id": aid} for aid in attachment_ids],
    )

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(
            content="Sensitive question",
            mode="council",
            attachment_ids=["att-1"],
        ),
    )

    rag_system.index_document.assert_called_once()  # the pipeline-level early skip doesn't fire (pre-flight was False)
    assert conversation_id not in real_rag.store  # but the write barrier inside index_document refused to store it


@pytest.mark.asyncio
async def test_startup_cleanup_removes_metadata_zdr_conversations(monkeypatch, tmp_path):
    """Decision #5's one-time startup sweep: any conversation already flagged
    zdr_enabled=True in metadata must have its memory-store entries purged when
    CouncilRAG initializes, while a normal conversation's entries survive."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-zdr", {"zdr_enabled": True})
    storage.create_conversation("conv-normal")
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-zdr": {"folder_id": "root", "turns": [{"turn": 0, "memory": "secret memory"}]},
        "conv-normal": {"folder_id": "root", "turns": [{"turn": 0, "memory": "normal memory"}]},
    }
    import json
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")
    # Marker already present: this store is post-upgrade steady state, not
    # the one-time-purge moment (that's covered separately in test_rag_persistence.py).
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-zdr" not in rag.store
    assert "conv-normal" in rag.store
    persisted = json.loads(index_file.read_text(encoding="utf-8"))
    assert "conv-zdr" not in persisted
    assert "conv-normal" in persisted


@pytest.mark.asyncio
async def test_startup_cleanup_removes_orphaned_entries_with_missing_conversation_file(monkeypatch, tmp_path):
    """Fail closed (Codex P2, plan revision): a missing conversation file means
    the conversation was deleted, so its store entries are pure orphaned,
    derived data. Unavailable metadata cannot prove an entry is safe to keep,
    and the delete-path has a crash window that could otherwise strand a ZDR
    conversation's memories forever if orphans were left alone. The sweep
    removes them rather than leaving them as-is."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-normal")
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-deleted": {"folder_id": "root", "turns": [{"turn": 0, "memory": "orphaned memory"}]},
        "conv-normal": {"folder_id": "root", "turns": [{"turn": 0, "memory": "normal memory"}]},
    }
    import json
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-deleted" not in rag.store
    assert "conv-normal" in rag.store


@pytest.mark.asyncio
async def test_startup_cleanup_removes_orphaned_entries_with_unreadable_conversation_file(monkeypatch, tmp_path):
    """Fail closed also covers a corrupt/unreadable conversation file (e.g.
    truncated by a crash mid-write): get_conversation raising is treated the
    same as a missing file, not left alone."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    conversations_dir.mkdir()
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    (conversations_dir / "conv-corrupt.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-corrupt": {"folder_id": "root", "turns": [{"turn": 0, "memory": "orphaned memory"}]},
    }
    import json
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-corrupt" not in rag.store


@pytest.mark.asyncio
async def test_startup_cleanup_removes_orphaned_entries_with_non_dict_conversation_record(monkeypatch, tmp_path):
    """Codex P2: get_conversation() can return valid JSON that isn't a dict
    (e.g. a conversation file holding "[]"). Calling .get() on that would
    raise OUTSIDE the try/except in cleanup_zdr_conversations, aborting
    CouncilRAG's constructor and backend startup. Must be treated as an
    orphan (fail closed) instead of crashing."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    conversations_dir.mkdir()
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    (conversations_dir / "conv-wrong-type.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-wrong-type": {"folder_id": "root", "turns": [{"turn": 0, "memory": "orphaned memory"}]},
    }
    import json
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-wrong-type" not in rag.store


@pytest.mark.asyncio
async def test_zdr_conversation_never_leaks_into_other_conversations_retrieval(monkeypatch, tmp_path):
    """End-to-end isolation: because a metadata-ZDR conversation's turns are
    never indexed (Stage: index-time exclusion) and any pre-existing entries
    are purged at startup (Stage: cleanup sweep), its content can never reach
    another conversation's retrieval/extraction prompt. This test drives both
    stages together rather than asserting a separate retrieval-time filter,
    which Decision #5 explicitly does not add."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    storage = main.storage
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-zdr", {"zdr_enabled": True})
    storage.create_conversation("conv-current")
    # Codex round 10 read barrier: retrieve_with_stats_async now checks each
    # source conversation's CURRENT metadata via get_conversation, so
    # "conv-other" (seeded below with memory but no real conversation file)
    # needs a real, non-ZDR conversation record too, or the barrier would
    # (correctly) exclude it as unreadable/orphaned -- same as a real
    # deployment would.
    storage.create_conversation("conv-other")

    pageindex_dir = tmp_path / "pageindex"
    monkeypatch.setattr("backend.rag.get_conversation", storage.get_conversation)
    real_rag = importlib.import_module("backend.rag").CouncilRAG(persist_path=str(pageindex_dir))
    # Seed a normal (non-ZDR) memory directly, bypassing the indexing guard,
    # to prove retrieval still works for legitimate cross-conversation memory.
    real_rag.store["conv-other"] = {
        "folder_id": "root",
        # message_anchor: 0 -- conv-other is a freshly created (0-message)
        # real conversation; round 26's anchor check needs this to satisfy
        # its CURRENT message count.
        "turns": [{"turn": 0, "memory": "ordinary memory should still retrieve", "message_anchor": 0}],
    }
    real_rag._save_store()
    monkeypatch.setattr(main, "rag_system", real_rag)

    # Attempt to index a ZDR turn's content into the now-running rag_system:
    # the turn_pipeline guard must prevent this from ever landing in the store.
    rag_system = SimpleNamespace(
        index_session=Mock(wraps=real_rag.index_session),
        index_document=AsyncMock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    await main.send_message(
        "conv-zdr",
        main.SendMessageRequest(content="SECRET_ZDR_CONTENT should never leak", mode="council"),
    )

    rag_system.index_session.assert_not_called()
    assert "conv-zdr" not in real_rag.store

    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    monkeypatch.setattr("backend.rag.query_model", fake_query_model)

    await real_rag.retrieve_async("current question", "conv-current", zdr_enabled=False)

    prompt = captured_kwargs["messages"][0]["content"]
    assert "SECRET_ZDR_CONTENT" not in prompt
    assert "ordinary memory should still retrieve" in prompt


@pytest.mark.asyncio
async def test_enabling_zdr_at_runtime_purges_existing_memories(monkeypatch, tmp_path):
    """Codex P1: the startup sweep (cleanup_zdr_conversations) only runs once
    per process, so a conversation that flips zdr_enabled=True at runtime (via
    PUT /api/conversations/{id}) would otherwise leave its already-indexed
    memories live and retrievable from other conversations until restart.
    update_conversation must purge them immediately, the same way conversation
    deletion already does via delete_conversation_memories."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-runtime-zdr-enable"
    private_preset = next(preset for preset in main.config.MODEL_PRESETS if preset["id"] == "private")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {
            "chairman_model": private_preset["chairman_model"],
            "council_models": private_preset["council_models"],
        },
    )

    fake_rag = SimpleNamespace(
        delete_conversation_memories=AsyncMock(),
        update_conversation_folder=Mock(),
    )
    monkeypatch.setattr(main, "rag_system", fake_rag)

    updated = await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(zdr_enabled=True),
    )

    assert updated["metadata"]["zdr_enabled"] is True
    fake_rag.delete_conversation_memories.assert_called_once_with(conversation_id)


@pytest.mark.asyncio
async def test_enabling_zdr_at_runtime_actually_removes_store_entries(monkeypatch, tmp_path):
    """End-to-end version of the above against the real CouncilRAG: seed the
    store with entries for a conversation, then flip zdr_enabled=True on it
    and assert the store no longer has that conversation's entries."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    rag_module = importlib.import_module("backend.rag")
    conversation_id = "conv-runtime-zdr-enable-real"
    private_preset = next(preset for preset in main.config.MODEL_PRESETS if preset["id"] == "private")

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))
    main.storage.create_conversation(
        conversation_id,
        {
            "chairman_model": private_preset["chairman_model"],
            "council_models": private_preset["council_models"],
        },
    )
    monkeypatch.setattr(rag_module, "get_conversation", main.storage.get_conversation)

    real_rag = rag_module.CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    real_rag.store[conversation_id] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "memory": "pre-existing memory before ZDR was enabled"}],
    }
    real_rag._save_store()
    monkeypatch.setattr(main, "rag_system", real_rag)

    await main.update_conversation(
        conversation_id,
        main.ConversationUpdate(zdr_enabled=True),
    )

    assert conversation_id not in real_rag.store

@pytest.mark.asyncio
async def test_startup_cleanup_removes_entries_with_malformed_metadata(monkeypatch, tmp_path):
    """Fail closed: a conversation record whose metadata is not a dict (e.g.
    "metadata": null from a partial migration) is removed during the startup
    sweep instead of crashing CouncilRAG construction (Codex round 7)."""
    import json
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-normal")
    conversation_path = Path(storage.get_conversation_path("conv-null-meta"))
    conversation_path.parent.mkdir(parents=True, exist_ok=True)
    conversation_path.write_text(
        json.dumps({"id": "conv-null-meta", "messages": [], "metadata": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-null-meta": {"folder_id": "root", "turns": [{"turn": 0, "memory": "suspect"}]},
        "conv-normal": {"folder_id": "root", "turns": [{"turn": 0, "memory": "normal"}]},
    }
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-null-meta" not in rag.store
    assert "conv-normal" in rag.store


@pytest.mark.asyncio
async def test_chat_turn_zdr_flip_mid_chairman_skips_topic_extraction(monkeypatch, tmp_path):
    """Codex round 14 (P1): a chat turn that starts non-ZDR must not send its
    question+answer to the topics utility call after a mid-turn ZDR flip —
    the store write was already refused by the write barrier, but the turn's
    CONTENT also has to stay out of the utility LLM call itself."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path / "conversations"))
    conversation_id = "conv-zdr-flip-chat-topics"
    main.storage.create_conversation(conversation_id)  # starts non-ZDR
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    topics_calls = []

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def fake_chat_with_chairman(*args, **kwargs):
        # Simulate the user flipping ZDR on via PUT while the chairman call
        # was still in flight.
        main.storage.update_conversation_metadata(conversation_id, {"zdr_enabled": True})
        return {"content": "Answer", "usage": {}}

    async def fake_topics(*args, **kwargs):
        topics_calls.append(args)
        return (["topic"], {})

    index_chat_turn = AsyncMock(return_value=None)
    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert result["content"] == "Answer"
    assert topics_calls == [], "topics utility call must be skipped after a mid-turn ZDR flip"
    index_chat_turn.assert_not_called()
