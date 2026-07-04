from pathlib import Path

import pytest

from backend.rag import CouncilRAG
from backend import rag as rag_module


def test_rag_preserves_corrupt_pageindex_file(tmp_path):
    index_file = tmp_path / "pageindex_memory.json"
    index_file.write_text("{not-json", encoding="utf-8")

    rag = CouncilRAG(persist_path=str(tmp_path))

    backups = list(tmp_path.glob("pageindex_memory.json.corrupt-*"))
    assert rag.enabled is True
    assert rag.store == {}
    assert backups
    assert backups[0].read_text(encoding="utf-8") == "{not-json"
    assert not index_file.exists()


def test_rag_disables_persistence_after_atomic_save_failure(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    def fail_write(path, content, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr("backend.rag.write_text_atomic", fail_write)

    rag.index_document("conv-1", "doc.txt", "hello")

    assert rag.enabled is False


def test_rag_save_store_skips_when_persistence_disabled(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))
    calls = []

    def fake_write(path, content, **kwargs):
        calls.append(Path(path))

    monkeypatch.setattr("backend.rag.write_text_atomic", fake_write)
    rag.enabled = False

    rag._save_store()

    assert calls == []


@pytest.mark.asyncio
async def test_retrieve_honors_max_tokens_budget(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    # One giant memory turn, comfortably larger than any cap we'll apply below.
    big_memory = "x" * 5000
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": big_memory}],
        },
    }

    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    max_tokens = 100  # cap = min(60_000, 100 * 4) = 400 chars
    await rag.retrieve_async("current question", "current", max_tokens=max_tokens)

    prompt = captured_kwargs["messages"][0]["content"]
    memory_section = prompt.split("USER MEMORY LOGS:\n", 1)[1]

    uncapped_memory_text = f"[Memory from Chat: other | Turn: 0]\n{big_memory}"
    cap = max_tokens * 4
    assert memory_section == uncapped_memory_text[-cap:]
    assert len(memory_section) == cap


@pytest.mark.asyncio
async def test_retrieve_defaults_to_legacy_60k_cap_when_max_tokens_none(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    big_memory = "y" * 70000
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": big_memory}],
        },
    }

    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    await rag.retrieve_async("current question", "current", max_tokens=None)

    prompt = captured_kwargs["messages"][0]["content"]
    memory_section = prompt.split("USER MEMORY LOGS:\n", 1)[1]

    uncapped_memory_text = f"[Memory from Chat: other | Turn: 0]\n{big_memory}"
    assert memory_section == uncapped_memory_text[-60000:]
    assert len(memory_section) == 60000
