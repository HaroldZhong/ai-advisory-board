from pathlib import Path

from backend.rag import CouncilRAG


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
