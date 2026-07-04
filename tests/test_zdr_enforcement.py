import importlib
from types import SimpleNamespace
from unittest.mock import Mock

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
        return ["topic"]

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics", fake_topics)
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
        return ""

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
        return ""

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
        return ""

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
        return ""

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
            "turns": [{"turn": 0, "memory": "Useful prior memory"}],
        },
    }

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {"content": "Relevant prior memory"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    result = await rag.retrieve_async("current question", "current", zdr_enabled=True)

    assert result == "Relevant prior memory"
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
        index_session=Mock(),
        index_document=Mock(),
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
        index_session=Mock(),
        index_document=Mock(),
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
        index_session=Mock(),
        index_document=Mock(),
        refresh_hybrid_index=Mock(),
    )
    _setup_council_fakes(monkeypatch, main, rag_system)

    await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Normal question", mode="council"),
    )

    rag_system.index_session.assert_called_once()


@pytest.mark.asyncio
async def test_council_turn_with_zdr_and_attachments_skips_document_indexing(monkeypatch, tmp_path):
    """Attachments attached to a ZDR turn must not be indexed into PageIndex
    either (backend/turn_pipeline.py's attachment loop calls index_document)."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    conversation_id = "conv-zdr-attachments"
    main.storage.create_conversation(conversation_id, {"zdr_enabled": True})

    rag_system = SimpleNamespace(
        index_session=Mock(),
        index_document=Mock(),
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

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-zdr" not in rag.store
    assert "conv-normal" in rag.store
    persisted = json.loads(index_file.read_text(encoding="utf-8"))
    assert "conv-zdr" not in persisted
    assert "conv-normal" in persisted


@pytest.mark.asyncio
async def test_startup_cleanup_leaves_missing_conversation_files_alone(monkeypatch, tmp_path):
    """Decision #5's documented boundary: conversations that no longer have a
    file on disk (already deleted) are left as-is by the sweep; deletion has
    its own purge path (delete_conversation_memories)."""
    storage = import_module_with_api_key(monkeypatch, "backend.storage")
    rag_module = import_module_with_api_key(monkeypatch, "backend.rag")

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    seeded_store = {
        "conv-deleted": {"folder_id": "root", "turns": [{"turn": 0, "memory": "orphaned memory"}]},
    }
    import json
    index_file.write_text(json.dumps(seeded_store), encoding="utf-8")

    rag = rag_module.CouncilRAG(persist_path=str(pageindex_dir))

    assert "conv-deleted" in rag.store


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

    pageindex_dir = tmp_path / "pageindex"
    monkeypatch.setattr("backend.rag.get_conversation", storage.get_conversation)
    real_rag = importlib.import_module("backend.rag").CouncilRAG(persist_path=str(pageindex_dir))
    # Seed a normal (non-ZDR) memory directly, bypassing the indexing guard,
    # to prove retrieval still works for legitimate cross-conversation memory.
    real_rag.store["conv-other"] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "memory": "ordinary memory should still retrieve"}],
    }
    real_rag._save_store()
    monkeypatch.setattr(main, "rag_system", real_rag)

    # Attempt to index a ZDR turn's content into the now-running rag_system:
    # the turn_pipeline guard must prevent this from ever landing in the store.
    rag_system = SimpleNamespace(
        index_session=Mock(wraps=real_rag.index_session),
        index_document=Mock(),
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
        delete_conversation_memories=Mock(),
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
