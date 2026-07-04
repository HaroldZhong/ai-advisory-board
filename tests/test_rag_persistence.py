from pathlib import Path

import pytest

from backend.rag import CouncilRAG, score_topic_overlap
from backend import rag as rag_module
from backend import storage


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


@pytest.mark.parametrize("bad_top_level_json", ["[]", '"a string"'])
def test_rag_resets_store_with_valid_json_but_wrong_top_level_type(tmp_path, caplog, bad_top_level_json):
    """Codex P2: json.load succeeds for e.g. `[]`, skipping the
    JSONDecodeError recovery path, so the store would not be a dict and the
    startup ZDR cleanup sweep's self.store.keys() would raise AttributeError,
    aborting backend startup. Must reset to {} the same way corrupt JSON does."""
    index_file = tmp_path / "pageindex_memory.json"
    index_file.write_text(bad_top_level_json, encoding="utf-8")

    with caplog.at_level("WARNING"):
        rag = CouncilRAG(persist_path=str(tmp_path))

    assert rag.enabled is True
    assert rag.store == {}
    assert any("invalid top-level type" in record.message for record in caplog.records)


def test_rag_disables_persistence_after_atomic_save_failure(tmp_path, monkeypatch):
    # index_document now has a ZDR write barrier (Codex round 6) that looks up
    # the conversation's current metadata before writing; a fabricated id with
    # no conversation file would fail closed and skip the write entirely
    # (never reaching write_text_atomic). Create a real, non-ZDR conversation
    # so the write barrier lets the index proceed and this test still
    # exercises the atomic-save-failure path.
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    def fail_write(path, content, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr("backend.rag.write_text_atomic", fail_write)

    rag.index_document("conv-1", "doc.txt", "hello")

    assert rag.enabled is False


def test_rag_write_barrier_blocks_index_session_and_index_document_for_zdr_conversation(tmp_path, monkeypatch):
    """Codex round 6: the ZDR write barrier lives INSIDE index_session and
    index_document themselves (not just the turn_pipeline call sites), so it
    closes the metadata-flip race regardless of how many awaits happen
    between a pipeline-level guard and the actual store mutation. Direct
    unit: a conversation whose metadata already says zdr_enabled=True must
    make both methods no-ops."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-zdr", {"zdr_enabled": True})

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    rag.index_session(
        "conv-zdr",
        0,
        "question",
        stage1_results=[],
        stage2_results=[],
        stage3_result={"model": "chair", "response": "answer"},
        topics=["topic"],
        quality_metrics={},
    )
    rag.index_document("conv-zdr", "doc.txt", "sensitive text")

    assert rag.store == {}


def test_rag_save_store_skips_when_persistence_disabled(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))
    calls = []

    def fake_write(path, content, **kwargs):
        calls.append(Path(path))

    monkeypatch.setattr("backend.rag.write_text_atomic", fake_write)
    rag.enabled = False

    rag._save_store()

    assert calls == []


async def _retrieve_and_capture_memory_section(rag, monkeypatch, max_tokens, query="current question"):
    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    await rag.retrieve_async(query, "current", max_tokens=max_tokens)

    prompt = captured_kwargs["messages"][0]["content"]
    return prompt.split("USER MEMORY LOGS:\n", 1)[1]


@pytest.mark.asyncio
async def test_retrieve_honors_max_tokens_budget_on_block_boundaries(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    # Several small blocks (not one oversized block) so the cap is applied by
    # dropping whole oldest blocks, not slicing raw characters.
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "memory": "oldest " + "a" * 100},
                {"turn": 1, "memory": "middle " + "b" * 100},
                {"turn": 2, "memory": "newest " + "c" * 100},
            ],
        },
    }

    max_tokens = 40  # cap = min(60_000, 40 * 4) = 160 chars: fits ~1 block, not all 3
    cap = max_tokens * 4
    memory_section = await _retrieve_and_capture_memory_section(rag, monkeypatch, max_tokens)

    # (a) every included block starts with its citation header intact.
    included_blocks = memory_section.split("\n\n")
    for block in included_blocks:
        assert block.startswith("[Memory from Chat: other | Turn: ")

    # (b) total length respects the cap.
    assert len(memory_section) <= cap

    # (c) newest blocks are preferred; the oldest (turn 0) block is dropped first.
    assert "Turn: 2" in memory_section
    assert "Turn: 0" not in memory_section


@pytest.mark.asyncio
async def test_retrieve_keeps_header_when_single_block_exceeds_cap(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    # A single memory turn whose block alone is larger than the cap.
    big_memory = "x" * 5000
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": big_memory}],
        },
    }

    max_tokens = 100  # cap = min(60_000, 100 * 4) = 400 chars, smaller than the block
    cap = max_tokens * 4
    memory_section = await _retrieve_and_capture_memory_section(rag, monkeypatch, max_tokens)

    header = "[Memory from Chat: other | Turn: 0]"
    assert memory_section.startswith(header)
    assert memory_section.endswith(big_memory[-1:])  # content tail preserved
    # Documented bound: cap plus the re-attached header line's overshoot.
    assert len(memory_section) <= cap + len(header) + 2


@pytest.mark.asyncio
async def test_retrieve_drops_tail_entirely_when_budget_leaves_no_room(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))

    big_memory = "x" * 5000
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [{"turn": 0, "memory": big_memory}],
        },
    }

    # cap = min(60_000, 1 * 4) = 4 chars: smaller than the header alone, so
    # tail_budget computes to 0. content[-0:] would (incorrectly) return the
    # WHOLE string; the guard must instead yield an empty tail.
    max_tokens = 1
    memory_section = await _retrieve_and_capture_memory_section(rag, monkeypatch, max_tokens)

    header = "[Memory from Chat: other | Turn: 0]"
    assert memory_section == header + "\n…"
    assert big_memory not in memory_section
    assert len(memory_section) <= len(header) + len("\n…")


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

    memory_section = await _retrieve_and_capture_memory_section(rag, monkeypatch, None)

    header = "[Memory from Chat: other | Turn: 0]"
    assert memory_section.startswith(header)
    assert memory_section.endswith(big_memory[-1:])
    assert len(memory_section) <= 60000 + len(header) + 2


# --- P5-T4: topic-overlap pre-filter (score_topic_overlap is a pure, module-level function) ---


def test_score_topic_overlap_counts_shared_tokens():
    assert score_topic_overlap("tell me about our budget", ["budget", "planning"]) == 1


def test_score_topic_overlap_is_case_insensitive():
    assert score_topic_overlap("what about RAG systems", ["rag", "retrieval"]) == 1


def test_score_topic_overlap_zero_when_topics_empty_or_missing():
    assert score_topic_overlap("anything", []) == 0
    assert score_topic_overlap("anything", None) == 0


def test_score_topic_overlap_zero_when_no_overlap():
    assert score_topic_overlap("cooking recipes", ["finance", "budget"]) == 0


@pytest.mark.asyncio
async def test_retrieve_filters_to_overlapping_turns_only(tmp_path, monkeypatch):
    """P5-T4: when the query overlaps at least one turn's stored topics,
    only the overlapping turn(s) should reach the extraction prompt."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "topics": ["cooking", "recipes"], "memory": "cooking memory"},
                {"turn": 1, "topics": ["budget", "finance"], "memory": "budget memory"},
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="what was our budget"
    )

    assert "Turn: 1" in memory_section
    assert "budget memory" in memory_section
    assert "Turn: 0" not in memory_section
    assert "cooking memory" not in memory_section


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_all_turns_when_nothing_overlaps(tmp_path, monkeypatch):
    """Quality floor: if no stored turn's topics overlap the query at all,
    the filter must not shrink the result below today's behavior -- fall
    back to every turn, same as if no filter existed."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "topics": ["cooking", "recipes"], "memory": "cooking memory"},
                {"turn": 1, "topics": ["budget", "finance"], "memory": "budget memory"},
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="completely unrelated astronomy question"
    )

    assert "Turn: 0" in memory_section
    assert "cooking memory" in memory_section
    assert "Turn: 1" in memory_section
    assert "budget memory" in memory_section


@pytest.mark.asyncio
async def test_retrieve_filter_composes_with_block_boundary_budget_cap(tmp_path, monkeypatch):
    """Ordering: the topic filter runs BEFORE the P5-T1 char-budget cap, so a
    tight budget applies only to the already-filtered (relevant) turns, not
    to the full unfiltered set."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "topics": ["budget"], "memory": "oldest " + "a" * 100},
                {"turn": 1, "topics": ["budget"], "memory": "middle " + "b" * 100},
                {"turn": 2, "topics": ["cooking"], "memory": "newest " + "c" * 100},
            ],
        },
    }

    max_tokens = 40  # cap = min(60_000, 40 * 4) = 160 chars: fits ~1 of the 2 filtered blocks
    cap = max_tokens * 4
    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, max_tokens, query="what was our budget plan"
    )

    # The cooking turn (no topic overlap) must never appear: filtered out
    # before the cap even runs.
    assert "Turn: 2" not in memory_section
    assert len(memory_section) <= cap
    # Of the two overlapping (budget) turns, the newest survives the cap.
    assert "Turn: 1" in memory_section
    assert "Turn: 0" not in memory_section


# --- P5-T4: store hygiene (warn, never auto-evict) ---


def test_save_store_warns_when_over_size_threshold(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(rag_module, "STORE_SIZE_WARNING_BYTES", 100)
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "conv-old": {"folder_id": "root", "turns": [{"turn": 0, "memory": "x" * 80}]},
        "conv-new": {"folder_id": "root", "turns": [{"turn": 0, "memory": "y" * 80}]},
    }

    with caplog.at_level("WARNING"):
        rag._save_store()

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("conv-old" in message for message in warnings)
    # No auto-eviction: both conversations must still be present afterward.
    assert set(rag.store.keys()) == {"conv-old", "conv-new"}


def test_save_store_does_not_warn_below_size_threshold(tmp_path, caplog):
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {"conv-small": {"folder_id": "root", "turns": [{"turn": 0, "memory": "small"}]}}

    with caplog.at_level("WARNING"):
        rag._save_store()

    assert not any("PageIndex store size" in r.message for r in caplog.records)
