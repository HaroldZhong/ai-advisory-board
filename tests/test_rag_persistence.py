import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.rag import CouncilRAG, score_topic_overlap
from backend import rag as rag_module
from backend import storage


def import_module_with_api_key(monkeypatch, module_name):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return importlib.import_module(module_name)


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


@pytest.mark.asyncio
async def test_rag_disables_persistence_after_atomic_save_failure(tmp_path, monkeypatch):
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

    await rag.index_document("conv-1", "doc.txt", "hello")

    assert rag.enabled is False


@pytest.mark.asyncio
async def test_rag_write_barrier_blocks_index_session_and_index_document_for_zdr_conversation(tmp_path, monkeypatch):
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

    await rag.index_session(
        "conv-zdr",
        "question",
        stage1_results=[],
        stage2_results=[],
        stage3_result={"model": "chair", "response": "answer"},
        topics=["topic"],
        quality_metrics={},
    )
    await rag.index_document("conv-zdr", "doc.txt", "sensitive text")

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
    # Codex round 10 read barrier: retrieve_with_stats_async now checks each
    # source conversation's CURRENT metadata via get_conversation before
    # including its memories. These pure topic-filter/budget-cap tests seed
    # rag.store directly without real conversation files, so fake a valid
    # non-ZDR conversation for any id -- the barrier itself is exercised
    # separately by test_retrieve_read_barrier_excludes_zdr_source_conversation.
    monkeypatch.setattr(rag_module, "get_conversation", lambda cid: {"metadata": {}})

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


@pytest.mark.asyncio
async def test_retrieve_always_includes_document_memories_regardless_of_topic_filter(tmp_path, monkeypatch):
    """Codex P2: document memories (index_document) store topics derived only
    from the filename (["document:{filename}"]), which can't be trusted as a
    relevance signal -- the extraction model inspects the document body
    itself instead. So when a session turn's topics DO overlap the query
    (triggering the filter), an unrelated-filename document must still ride
    along unconditionally, not get filtered out alongside non-overlapping
    turns."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "topics": ["budget"], "memory": "budget memory"},
                {
                    "turn": -1,
                    "topics": ["document:quarterly_report.pdf"],
                    "memory": "[Uploaded Document: quarterly_report.pdf]\ndocument body text",
                },
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="what was our budget"
    )

    assert "budget memory" in memory_section
    assert "document body text" in memory_section


@pytest.mark.asyncio
async def test_retrieve_keeps_document_memory_when_filename_does_not_match_query(tmp_path, monkeypatch):
    """Explicit fallback check for a document-only store: even though the
    general fallback (no turn overlaps) would already keep everything, this
    pins down that a document whose filename shares nothing with the query
    is still surfaced -- the extraction model, not the filter, judges its
    relevance."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {
                    "turn": -1,
                    "topics": ["document:random_notes.txt"],
                    "memory": "[Uploaded Document: random_notes.txt]\nunrelated filename document body",
                },
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="what was our budget plan"
    )

    assert "unrelated filename document body" in memory_section


@pytest.mark.asyncio
async def test_retrieve_preserves_original_order_so_cap_keeps_newer_matching_turn(tmp_path, monkeypatch):
    """Codex round 3: unioning document entries AFTER the filtered session
    turns made every document look newest to the P5-T1 char-budget cap
    (which walks reversed(memory_blocks) to keep the NEWEST blocks), so an
    older document could evict a newer, topic-matching turn that should have
    survived instead. The filter must preserve original relative order:
    document indexed FIRST (older), topic-matching turn indexed AFTER
    (newer); with a cap that only fits one block, the newer matching turn
    must survive and the older document must be dropped -- the reverse of
    what the pre-fix append-documents-last ordering produced."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {
                    "turn": -1,
                    "topics": ["document:old_notes.txt"],
                    "memory": "[Uploaded Document: old_notes.txt]\n" + "d" * 100,
                },
                {"turn": 0, "topics": ["budget"], "memory": "newer budget turn " + "t" * 100},
            ],
        },
    }

    max_tokens = 40  # cap = min(60_000, 40 * 4) = 160 chars: fits only one of the two blocks
    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, max_tokens, query="what was our budget"
    )

    assert "newer budget turn" in memory_section
    assert "old_notes.txt" not in memory_section


@pytest.mark.asyncio
async def test_retrieve_keeps_topicless_turns_eligible_alongside_a_matching_turn(tmp_path, monkeypatch):
    """Codex round 4: extract_topics can legitimately return [] on a timeout
    or error, so a session turn can have empty/missing topics through no
    fault of its own. Such a turn is unjudgeable, not irrelevant -- it must
    stay eligible (fail open, same rationale as documents) even when another
    turn DOES score an overlap and would otherwise trigger the filter."""
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {"turn": 0, "topics": [], "memory": "topicless memory"},
                {"turn": 1, "topics": ["budget"], "memory": "budget memory"},
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="what was our budget"
    )

    assert "topicless memory" in memory_section
    assert "budget memory" in memory_section


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


# --- P5-T5 feature 1: one-time memory purge on upgrade (owner decision) ---


def test_purge_resets_existing_store_and_writes_marker_when_marker_missing(tmp_path, caplog):
    """A store that predates the version marker must be dropped once, with a
    clear INFO log, and the marker must be written so it never happens again."""
    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    index_file.write_text(
        json.dumps({"conv-1": {"folder_id": "root", "turns": [{"turn": 0, "memory": "old memory"}]}}),
        encoding="utf-8",
    )

    with caplog.at_level("INFO"):
        rag = CouncilRAG(persist_path=str(pageindex_dir))

    assert rag.store == {}
    marker_path = pageindex_dir / "pageindex_memory.version"
    assert marker_path.read_text(encoding="utf-8").strip() == "2"
    # The store file on disk is also reset (not just the in-memory dict).
    assert json.loads(index_file.read_text(encoding="utf-8")) == {}
    assert any(
        "one-time memory reset" in r.message and "1 conversation" in r.message
        for r in caplog.records
    )


def test_purge_leaves_store_alone_when_marker_present(tmp_path, monkeypatch):
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    stored = {"conv-1": {"folder_id": "root", "turns": [{"turn": 0, "memory": "keep me"}]}}
    index_file.write_text(json.dumps(stored), encoding="utf-8")
    (pageindex_dir / "pageindex_memory.version").write_text("2", encoding="utf-8")

    rag = CouncilRAG(persist_path=str(pageindex_dir))

    assert rag.store == stored


def test_purge_fresh_install_writes_marker_without_reset_log(tmp_path, caplog):
    """No store file at all (fresh install): just write the marker, no purge
    log spam since there was nothing to purge."""
    pageindex_dir = tmp_path / "pageindex"

    with caplog.at_level("INFO"):
        rag = CouncilRAG(persist_path=str(pageindex_dir))

    assert rag.store == {}
    marker_path = pageindex_dir / "pageindex_memory.version"
    assert marker_path.read_text(encoding="utf-8").strip() == "2"
    assert not any("one-time memory reset" in r.message for r in caplog.records)


def test_purge_never_touches_conversations_or_attachments_dirs(tmp_path, monkeypatch):
    """Owner decision: the purge is scoped to the memory store file ONLY.
    Conversations and attachments must survive untouched even though a purge
    fires on this run."""
    conversations_dir = tmp_path / "data" / "conversations"
    attachments_dir = conversations_dir / "attachments"
    attachments_dir.mkdir(parents=True)
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-1")
    attachment_file = attachments_dir / "att-1.txt"
    attachment_file.write_text("attachment body", encoding="utf-8")

    pageindex_dir = tmp_path / "data"  # shares the parent "data" dir, like get_pageindex_dir()
    index_file = pageindex_dir / "pageindex_memory.json"
    index_file.write_text(json.dumps({"conv-1": {"folder_id": "root", "turns": [{"turn": 0, "memory": "x"}]}}), encoding="utf-8")

    CouncilRAG(persist_path=str(pageindex_dir))

    assert (conversations_dir / "conv-1.json").exists()
    assert attachment_file.read_text(encoding="utf-8") == "attachment body"


def test_purge_skips_marker_and_retries_when_save_fails(tmp_path, monkeypatch, caplog):
    """Codex round 5 P2: the marker must only be written once the reset
    actually landed on disk. If _save_store fails (disk full, permissions),
    writing the marker anyway would permanently mark this store "already
    reset" while the stale, un-purged store survives -- defeating the
    guarantee. On failure: no marker, an ERROR log, and the next startup
    must retry the reset."""
    pageindex_dir = tmp_path / "pageindex"
    pageindex_dir.mkdir()
    index_file = pageindex_dir / "pageindex_memory.json"
    index_file.write_text(
        json.dumps({"conv-1": {"folder_id": "root", "turns": [{"turn": 0, "memory": "old memory"}]}}),
        encoding="utf-8",
    )

    def fail_write(path, content, **kwargs):
        raise PermissionError("disk full")

    monkeypatch.setattr(rag_module, "write_text_atomic", fail_write)

    with caplog.at_level("ERROR"):
        rag = CouncilRAG(persist_path=str(pageindex_dir))

    marker_path = pageindex_dir / "pageindex_memory.version"
    assert not marker_path.exists()
    assert rag.enabled is False  # _save_store's failure path disables persistence
    assert any("one-time memory reset" in r.message.lower() and "retry" in r.message.lower() for r in caplog.records)

    # Next startup (write_text_atomic working again) must retry the reset:
    # the marker is still absent, so the same store is dropped again.
    monkeypatch.undo()
    with caplog.at_level("INFO"):
        rag2 = CouncilRAG(persist_path=str(pageindex_dir))

    assert rag2.store == {}
    assert marker_path.read_text(encoding="utf-8").strip() == "2"


def test_purge_tolerates_marker_write_failure_without_crashing_construction(tmp_path, monkeypatch, caplog):
    """Codex round 6 P2: the final write_text_atomic(marker) call on the
    fresh-install/already-empty-store path sat OUTSIDE any try/except.
    Disk-full or permissions there would raise straight out of
    _apply_one_time_memory_reset, aborting CouncilRAG's constructor and
    crashing backend startup entirely -- a much worse failure mode than a
    missing marker. Must log an ERROR and continue with a consistent
    (empty) in-memory store instead."""
    pageindex_dir = tmp_path / "pageindex"  # fresh install: no store file at all

    def fail_write(path, content, **kwargs):
        raise PermissionError("disk full")

    monkeypatch.setattr(rag_module, "write_text_atomic", fail_write)

    with caplog.at_level("ERROR"):
        rag = CouncilRAG(persist_path=str(pageindex_dir))

    # Construction must succeed (no raise) and leave a consistent state.
    assert rag.enabled is True
    assert rag.store == {}
    marker_path = pageindex_dir / "pageindex_memory.version"
    assert not marker_path.exists()
    assert any(
        "memory version marker could not be written" in r.message.lower()
        for r in caplog.records
    )


# --- P5-T5 feature 2: chat-turn indexing ---


@pytest.mark.asyncio
async def test_index_chat_turn_stores_entry_shape_compatible_with_retrieval(tmp_path, monkeypatch):
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    await rag.index_chat_turn("conv-1", "What is our budget?", "It is $10k.", ["budget"])

    turns = rag.store["conv-1"]["turns"]
    assert len(turns) == 1
    entry = turns[0]
    assert entry["turn"] == 0
    assert entry["topics"] == ["budget"]
    assert "What is our budget?" in entry["memory"]
    assert "It is $10k." in entry["memory"]


@pytest.mark.asyncio
async def test_index_chat_turn_write_barrier_blocks_zdr_conversation(tmp_path, monkeypatch):
    """Mirrors test_rag_write_barrier_blocks_index_session_and_index_document_for_zdr_conversation:
    the barrier must live inside index_chat_turn itself."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-zdr", {"zdr_enabled": True})

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    await rag.index_chat_turn("conv-zdr", "question", "answer", ["topic"])

    assert rag.store == {}


# --- P5-T5 feature 3: per-conversation summary tier ---


@pytest.mark.asyncio
async def test_summary_tier_compresses_oldest_half_above_threshold(tmp_path, monkeypatch):
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    async def fake_query_model(*args, **kwargs):
        return {"content": "Dense summary of earlier turns.", "usage": {"prompt_tokens": 5, "completion_tokens": 5}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    usage = await rag.index_chat_turn("conv-1", "new question", "new answer", ["newtopic"])

    turns = rag.store["conv-1"]["turns"]
    # Oldest half (of the post-append length) replaced by one summary entry.
    total_after_append = SUMMARY_COMPRESSION_THRESHOLD + 1
    half = (total_after_append + 1) // 2
    assert turns[0]["kind"] == "summary"
    assert turns[0]["turns_compressed"] == half
    assert turns[0]["summary"] == "Dense summary of earlier turns."
    assert set(turns[0]["topics"]) == {f"topic{i}" for i in range(half)}
    # Newest turns intact, including the just-added one.
    remaining_normal = [t for t in turns if t.get("kind") != "summary"]
    assert remaining_normal[-1]["memory"] == "Q: new question\nA: new answer"
    assert usage == {"prompt_tokens": 5, "completion_tokens": 5}


@pytest.mark.asyncio
async def test_summary_tier_skips_compression_on_llm_failure(tmp_path, monkeypatch, caplog):
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    async def fake_query_model(*args, **kwargs):
        return None

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    with caplog.at_level("WARNING"):
        usage = await rag.index_chat_turn("conv-1", "new question", "new answer", ["newtopic"])

    turns = rag.store["conv-1"]["turns"]
    # No data loss: all turns intact (original + the new one), no summary entry.
    assert len(turns) == SUMMARY_COMPRESSION_THRESHOLD + 1
    assert not any(t.get("kind") == "summary" for t in turns)
    assert usage is None
    assert any("summary compression" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_summary_tier_renders_as_a_retrieval_block_with_header(tmp_path, monkeypatch):
    rag = CouncilRAG(persist_path=str(tmp_path))
    rag.store = {
        "current": {"folder_id": "root", "turns": []},
        "other": {
            "folder_id": "root",
            "turns": [
                {
                    "kind": "summary",
                    "turn": 0,
                    "topics": ["budget"],
                    "summary": "Summary of 6 earlier turns about budget.",
                    "turns_compressed": 6,
                },
            ],
        },
    }

    memory_section = await _retrieve_and_capture_memory_section(rag, monkeypatch, None, query="what was our budget")

    assert "summary of 6 earlier turns" in memory_section.lower()
    assert "Summary of 6 earlier turns about budget." in memory_section


# --- Codex round on PR #80: error turns must not become memory ---


@pytest.mark.asyncio
async def test_chat_turn_where_chairman_raises_is_not_indexed(monkeypatch, tmp_path):
    """Codex P2: the except handler in turn_pipeline's chat branch fabricates
    an apology response_dict when chat_with_chairman raises. That fabricated
    text must never be written into cross-conversation memory -- mirrors the
    council branch's stage3_result.get("model") != "error" guard. Root fix is
    an explicit response_dict["error"] flag, not string-matching the apology."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-chairman-raises"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def raising_chat_with_chairman(*args, **kwargs):
        raise RuntimeError("chairman exploded")

    index_calls = []

    async def spy_index_chat_turn(*args, **kwargs):
        index_calls.append(args)
        return None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "chat_with_chairman", raising_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=spy_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    assert result["type"] == "chat"
    assert "I apologize" in result["content"]
    assert index_calls == []


@pytest.mark.asyncio
async def test_chat_turn_where_query_model_returns_none_is_not_indexed(monkeypatch, tmp_path):
    """Codex round 8: the OTHER leg of the round-2 fix. When query_model
    returns None (API failure/timeout) INSIDE chat_with_chairman itself
    (not via chat_with_chairman raising), council.chat_with_chairman builds
    its own fabricated apology dict -- previously WITHOUT the "error" flag,
    so turn_pipeline's `not response_dict.get("error")` guard passed and the
    apology got indexed as memory. Patches at the real seam (council's
    query_model import, not chat_with_chairman itself) so the REAL
    chat_with_chairman fallback branch runs and is what's under test."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-chairman-query-model-none"

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(conversation_id)
    main.storage.add_user_message(conversation_id, "Earlier question")
    main.storage.add_chat_message(conversation_id, "Earlier answer")

    async def fake_rewrite_query(*args, **kwargs):
        return "rewritten"

    async def fake_retrieve_async(*args, **kwargs):
        return "", {}

    async def query_model_returns_none(*args, **kwargs):
        return None

    index_calls = []

    async def spy_index_chat_turn(*args, **kwargs):
        index_calls.append(args)
        return None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.query_model", query_model_returns_none)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=spy_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    # Apology content still reaches the user (unchanged behavior)...
    assert result["type"] == "chat"
    assert "I apologize" in result["content"]
    # ...but the fallback must never become cross-conversation memory.
    assert index_calls == []


@pytest.mark.asyncio
async def test_summary_compression_usage_bills_totals_but_not_persisted_running_cost(monkeypatch, tmp_path):
    """Codex P2: summary-compression usage is discovered AFTER turn_cost is
    computed and the message is persisted (persistence-first: indexing must
    never risk losing the saved answer). The fix bills it as a delta added to
    the in-memory turn_cost, which reaches update_conversation_cost/
    record_session_usage/the completion event -- but the message's already
    -persisted running_cost snapshot predates the delta and must stay
    unchanged."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-compression-billing"

    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    compression_usage = {"prompt_tokens": 100_000, "completion_tokens": 10_000}

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
        return {"content": "Budget-aware response", "usage": chairman_usage}

    async def fake_extract_topics(*args, **kwargs):
        return (["topic"], {})

    async def fake_index_chat_turn(*args, **kwargs):
        return compression_usage

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=fake_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    # openai/gpt-4o-mini pricing: input=$0.15/M, output=$0.6/M
    chairman_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    # google/gemini-2.5-flash (UTILITY_MODEL) pricing: input=$0.3/M, output=$2.5/M
    compression_cost = (100_000 / 1_000_000) * 0.3 + (10_000 / 1_000_000) * 2.5
    expected_total = chairman_cost + compression_cost

    conversation = main.storage.get_conversation(conversation_id)

    # Completion payload / conversation total DO include the compression delta.
    assert result["turn_cost"] == pytest.approx(expected_total)
    assert result["total_cost"] == pytest.approx(expected_total)
    assert conversation["total_cost"] == pytest.approx(expected_total)
    # The already-persisted message running_cost snapshot excludes it.
    assert conversation["messages"][-1]["running_cost"] == pytest.approx(chairman_cost)


@pytest.mark.asyncio
async def test_chat_turn_bills_topic_extraction_usage(monkeypatch, tmp_path):
    """Codex round 6 P2: extract_topics_with_usage's own call burns
    UTILITY_MODEL tokens on every chat turn, but that usage used to be
    discarded entirely (plain extract_topics returns only the topic list).
    Now that compression usage is billed, this was the last remaining
    invisible-cost gap in the chat branch. Billed the same delta way as
    compression: turn_cost was already computed above this call, so the
    topics usage is added on top."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-topics-billing-chat"

    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    topics_usage = {"prompt_tokens": 50_000, "completion_tokens": 5_000}

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
        return {"content": "Budget-aware response", "usage": chairman_usage}

    async def fake_extract_topics_with_usage(*args, **kwargs):
        return ["topic"], topics_usage

    async def fake_index_chat_turn(*args, **kwargs):
        return None  # no compression this turn -- isolates the topics delta

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics_with_usage)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=fake_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )

    # openai/gpt-4o-mini pricing: input=$0.15/M, output=$0.6/M
    chairman_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    # google/gemini-2.5-flash (UTILITY_MODEL) pricing: input=$0.3/M, output=$2.5/M
    topics_cost = (50_000 / 1_000_000) * 0.3 + (5_000 / 1_000_000) * 2.5
    expected_total = chairman_cost + topics_cost

    conversation = main.storage.get_conversation(conversation_id)

    assert result["turn_cost"] == pytest.approx(expected_total)
    assert result["total_cost"] == pytest.approx(expected_total)
    assert conversation["total_cost"] == pytest.approx(expected_total)
    # Same delta convention as compression: persisted running_cost predates it.
    assert conversation["messages"][-1]["running_cost"] == pytest.approx(chairman_cost)


@pytest.mark.asyncio
async def test_council_turn_bills_topic_extraction_usage(monkeypatch, tmp_path):
    """Codex round 6 P2, council side: extract_topics_with_usage's call in
    the council branch burns UTILITY_MODEL tokens too and was equally
    unbilled. Same delta-billing convention."""
    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-topics-billing-council"

    stage_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    topics_usage = {"prompt_tokens": 50_000, "completion_tokens": 5_000}

    monkeypatch.setattr(main.storage, "DATA_DIR", str(tmp_path))
    main.storage.create_conversation(
        conversation_id,
        {"chairman_model": "openai/gpt-4o-mini"},
    )

    from backend.tools.types import EvidencePack

    async def fake_steward(*args, **kwargs):
        return EvidencePack(run_id="run-1", query="q"), {}

    async def fake_stage1_progressive(*args, **kwargs):
        result = {"model": "openai/gpt-4o-mini", "response": "Answer A", "usage": stage_usage}
        yield "model_complete", 0, result
        yield "complete", [result], None

    async def fake_stage2(*args, **kwargs):
        return (
            [{"model": "openai/gpt-4o-mini", "ranking": "1. Response A", "parsed_ranking": ["Response A"], "usage": {}}],
            {"Response A": "openai/gpt-4o-mini"},
        )

    async def fake_stage3(*args, **kwargs):
        return {"model": "openai/gpt-4o-mini", "response": "Final answer", "usage": {}}

    async def fake_title(*args, **kwargs):
        return "Test title"

    async def fake_extract_topics_with_usage(*args, **kwargs):
        return ["topic"], topics_usage

    async def fake_index_session(*args, **kwargs):
        return None  # no compression this turn -- isolates the topics delta

    async def fake_index_document(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "run_tool_steward_phase", fake_steward)
    monkeypatch.setattr(main, "stage1_collect_responses_progressive", fake_stage1_progressive)
    monkeypatch.setattr(main, "stage2_collect_rankings", fake_stage2)
    monkeypatch.setattr(main, "stage3_synthesize_final", fake_stage3)
    monkeypatch.setattr(main, "generate_conversation_title", fake_title)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics_with_usage)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            index_session=fake_index_session,
            refresh_hybrid_index=lambda *a, **k: None,
            index_document=fake_index_document,
        ),
    )

    result = await main.send_message(
        conversation_id,
        main.SendMessageRequest(content="What should we do?", mode="council"),
    )

    # openai/gpt-4o-mini pricing (all three stages use it here): input=$0.15/M, output=$0.6/M
    stage_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6
    # google/gemini-2.5-flash (UTILITY_MODEL) pricing: input=$0.3/M, output=$2.5/M
    topics_cost = (50_000 / 1_000_000) * 0.3 + (5_000 / 1_000_000) * 2.5
    expected_total = stage_cost + topics_cost

    conversation = main.storage.get_conversation(conversation_id)

    assert result["turn_cost"] == pytest.approx(expected_total)
    assert result["total_cost"] == pytest.approx(expected_total)
    assert conversation["total_cost"] == pytest.approx(expected_total)
    assert conversation["messages"][-1]["running_cost"] == pytest.approx(stage_cost)


# --- Codex round 3 on PR #80: per-conversation write serialization + monotonic turn numbers ---


@pytest.mark.asyncio
async def test_concurrent_index_chat_turn_calls_do_not_drop_a_write(tmp_path, monkeypatch):
    """Codex P2: _maybe_compress_oldest_half snapshots turns, awaits an LLM
    call, then replaces the whole list with [summary] + stale `rest`. Without
    a per-conversation lock around the whole append+compress body, a second
    concurrent index_chat_turn for the SAME conversation can append its turn
    during that await and have it silently dropped by the first call's stale
    write-back.

    Drives two REAL concurrent tasks for the same conversation (not
    sequential awaits -- the fixed code serializes them via the write lock,
    so awaiting the second call directly while the first still holds the
    lock would just deadlock the test, which is itself proof the lock works).
    The first call's compression LLM fake blocks on an asyncio.Event; the
    test asserts the second task is still pending -- provably blocked on the
    lock, not free to interleave -- before releasing the first. This fails
    pre-fix because there IS no lock to block on: the second call would run
    to completion immediately, appending during the first's stale-snapshot
    window, and then get silently overwritten by the first call's write-back.
    """
    import asyncio

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    release_compression = asyncio.Event()

    async def blocking_query_model(*args, **kwargs):
        # The FIRST index_chat_turn call (which triggers compression because
        # it pushes the store past the threshold) blocks here, still holding
        # the per-conversation write lock, until the test releases it.
        await release_compression.wait()
        return {"content": "Dense summary.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", blocking_query_model)

    first_task = asyncio.create_task(
        rag.index_chat_turn("conv-1", "first new question", "first new answer", ["a"])
    )
    await asyncio.sleep(0)  # let the first call append + hit the threshold + block on query_model

    second_task = asyncio.create_task(
        rag.index_chat_turn("conv-1", "second new question", "second new answer", ["b"])
    )
    await asyncio.sleep(0)

    # The lock must be blocking the second call right now -- it cannot have
    # appended yet, proving there is no window for its write to be dropped.
    assert not second_task.done()

    release_compression.set()
    await first_task
    await second_task

    turns = rag.store["conv-1"]["turns"]
    all_memories = [t.get("memory", "") for t in turns]
    assert any("first new question" in m and "first new answer" in m for m in all_memories)
    assert any("second new question" in m and "second new answer" in m for m in all_memories), (
        "the second concurrent write must survive, not be silently dropped by the first "
        "call's stale write-back"
    )


@pytest.mark.asyncio
async def test_chat_turn_numbering_is_monotonic_after_compaction(tmp_path, monkeypatch):
    """Codex P2: len(turns) reuses a number once compression shrinks the
    list (e.g. 12 turns -> compress oldest 6 into 1 summary -> len is now 7,
    same as the turn number the summary itself covers). The next indexed
    turn's number must be strictly greater than every existing turn number,
    regardless of compaction."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    async def fake_query_model(*args, **kwargs):
        return {"content": "Dense summary.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    # This call pushes the store past the threshold and triggers compaction
    # (oldest half replaced by one summary entry).
    await rag.index_chat_turn("conv-1", "triggers compaction", "answer", ["c"])
    turns_after_compaction = rag.store["conv-1"]["turns"]
    max_turn_after_compaction = max(t["turn"] for t in turns_after_compaction)

    # A naive len(turns) would now be len(turns_after_compaction), which can
    # collide with an existing turn number post-compaction (the bug: a
    # duplicate 7). The next real turn must be strictly greater than every
    # existing turn number.
    await rag.index_chat_turn("conv-1", "next turn after compaction", "answer", ["d"])
    new_entry = rag.store["conv-1"]["turns"][-1]

    assert new_entry["turn"] > max_turn_after_compaction
    all_turn_numbers = [t["turn"] for t in rag.store["conv-1"]["turns"]]
    assert len(all_turn_numbers) == len(set(all_turn_numbers)), "no duplicate turn numbers"


# --- Codex round 4 on PR #80: ZDR barrier must be re-checked UNDER the write lock ---


@pytest.mark.asyncio
async def test_zdr_flip_while_pending_on_write_lock_blocks_the_write(tmp_path, monkeypatch):
    """Codex P1: if the ZDR barrier runs BEFORE lock acquisition, a task can
    pass the (now-stale) barrier, suspend waiting for the lock while the user
    enables ZDR (which purges this conversation via update_conversation),
    then acquire the lock and write anyway -- re-indexing a now-ZDR
    conversation and undoing the purge. The fix checks the barrier UNDER the
    lock, immediately before the store mutation, so it always sees the
    CURRENT metadata.

    The test holds conv-1's write lock directly (no need to route a real
    call through it) while index_chat_turn -- which passes the barrier check
    the moment it's able to run, since it's the very first thing inside the
    lock -- is pending on that same lock. While it waits, flip the
    conversation's metadata to zdr_enabled=True and purge, simulating the
    user enabling ZDR mid-flight. Release the lock; index_chat_turn then
    acquires it, re-checks the barrier, sees ZDR, and must write nothing."""
    import asyncio

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")  # starts non-ZDR

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    rag.store["conv-1"] = {"folder_id": "root", "turns": [{"turn": 0, "topics": [], "memory": "Q0\nA0"}]}

    lock = rag._get_write_lock("conv-1")
    await lock.acquire()

    pending_task = asyncio.create_task(
        rag.index_chat_turn("conv-1", "pending question", "pending answer", ["b"])
    )
    await asyncio.sleep(0)

    # The pending call must be provably blocked on the lock -- it has not
    # re-checked ZDR or written anything yet.
    assert not pending_task.done()

    # Simulate the user enabling ZDR mid-flight: the barrier only reads
    # conversation metadata, so the metadata flip alone is what it must
    # react to -- the purge itself (delete_conversation_memories) now also
    # takes this same lock (Codex round 5) and is covered separately in
    # test_purge_waits_for_in_flight_compression_then_deletes_cleanly below.
    storage.update_conversation_metadata("conv-1", {"zdr_enabled": True})

    lock.release()
    await pending_task

    # The pending call must have written nothing: the barrier, re-checked
    # under the lock, saw the now-current ZDR metadata and refused. Only the
    # pre-seeded turn 0 remains -- the pending write's content never landed.
    turns = rag.store["conv-1"]["turns"]
    assert len(turns) == 1
    assert not any("pending question" in t.get("memory", "") for t in turns)


# --- Codex round 5 on PR #80: purges must honor the write lock too ---


@pytest.mark.asyncio
async def test_purge_waits_for_in_flight_compression_then_deletes_cleanly(tmp_path, monkeypatch):
    """Codex P2: delete_conversation_memories mutated self.store WITHOUT the
    per-conversation write lock, so a runtime ZDR purge (or conversation
    deletion) could interleave with an in-flight compression's
    snapshot-await-writeback window inside index_session/index_chat_turn --
    at best silently undone by the write-back, at worst a KeyError if the
    purge deleted the conversation entirely while compression's write-back
    tried to write self.store[conversation_id]["turns"] = ....

    Fixed by making delete_conversation_memories async and acquiring the same
    write lock. This drives a real concurrent purge while a real compression
    call is blocked mid-await: the purge must simply wait for the lock, not
    KeyError, and the in-flight turn must still complete (write-back
    succeeds against a store that still has the conversation, because the
    purge is queued behind the lock) before the purge finally runs and
    leaves the conversation removed."""
    import asyncio

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    release_compression = asyncio.Event()

    async def blocking_query_model(*args, **kwargs):
        # index_chat_turn is holding the write lock here, mid-compression.
        await release_compression.wait()
        return {"content": "Dense summary.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", blocking_query_model)

    index_task = asyncio.create_task(
        rag.index_chat_turn("conv-1", "triggers compaction", "answer", ["c"])
    )
    await asyncio.sleep(0)  # let it append + hit the threshold + block on query_model, holding the lock

    purge_task = asyncio.create_task(rag.delete_conversation_memories("conv-1"))
    await asyncio.sleep(0)

    # The purge must be provably blocked on the same lock -- it cannot have
    # deleted the conversation yet (that's what would have KeyError'd the
    # in-flight compression's write-back pre-fix).
    assert not purge_task.done()
    assert "conv-1" in rag.store

    release_compression.set()
    await index_task  # no KeyError: write-back completes against an intact store
    await purge_task  # then the purge runs and removes it

    assert "conv-1" not in rag.store


# --- Codex round 6 on PR #80: id collisions between council and chat memory ---


@pytest.mark.asyncio
async def test_index_session_and_index_chat_turn_share_one_monotonic_turn_sequence(tmp_path, monkeypatch):
    """Codex P2: index_session used to number its memory entries via
    get_turn_index(conversation), which counts only COUNCIL messages -- so a
    council turn and a chat turn in the SAME conversation could land in the
    memory store under the same "turn" number (e.g. the conversation's first
    council turn AND first chat turn both numbered 0). Fixed by having
    index_session compute its turn number the same way index_chat_turn
    already does: one past the max "turn" across ALL of this conversation's
    memory entries, council or chat. Interleave a chat turn, a council turn,
    then another chat turn for the same conversation and assert every stored
    turn number is unique and strictly increasing."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    await rag.index_chat_turn("conv-1", "chat question 1", "chat answer 1", ["a"])
    await rag.index_session(
        "conv-1",
        "council question 1",
        stage1_results=[],
        stage2_results=[],
        stage3_result={"model": "chair", "response": "council answer 1"},
        topics=["b"],
        quality_metrics={},
    )
    await rag.index_chat_turn("conv-1", "chat question 2", "chat answer 2", ["c"])

    turns = rag.store["conv-1"]["turns"]
    turn_numbers = [t["turn"] for t in turns]

    assert len(turn_numbers) == len(set(turn_numbers)), "no duplicate turn numbers between chat and council entries"
    assert turn_numbers == sorted(turn_numbers), "turn numbers strictly increase in insertion order"
    assert turn_numbers == [0, 1, 2]


# --- Codex round 7 on PR #80: index_document was the last unlocked store writer ---


@pytest.mark.asyncio
async def test_document_indexed_during_compression_survives_the_write_back(tmp_path, monkeypatch):
    """Codex P2: index_document appended to self.store[cid]["turns"] WITHOUT
    the per-conversation write lock -- so a document indexed while a
    concurrent index_chat_turn/index_session call was mid-compression
    (snapshotted turns, awaiting query_model) would get silently dropped by
    that compression's stale [summary_entry] + rest write-back, exactly like
    the round-3 concurrent-chat-turn bug this mirrors.

    Drives two real concurrent tasks for the same conversation: index_chat_turn
    triggers compression and blocks mid-await on an asyncio.Event;
    index_document is provably still pending on the same write lock before
    the event is released. Both must complete and the document memory must
    survive -- fails pre-fix because there was no lock to block on, so the
    document would land during the stale-snapshot window and then be
    overwritten by the write-back."""
    import asyncio

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )

    release_compression = asyncio.Event()

    async def blocking_query_model(*args, **kwargs):
        # index_chat_turn is holding the write lock here, mid-compression.
        await release_compression.wait()
        return {"content": "Dense summary.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", blocking_query_model)

    index_task = asyncio.create_task(
        rag.index_chat_turn("conv-1", "triggers compaction", "answer", ["c"])
    )
    await asyncio.sleep(0)  # let it append + hit the threshold + block on query_model, holding the lock

    document_task = asyncio.create_task(
        rag.index_document("conv-1", "report.pdf", "important document contents")
    )
    await asyncio.sleep(0)

    # The document indexing must be provably blocked on the same lock -- it
    # cannot have appended yet, proving there is no window for the
    # compression's write-back to silently drop it.
    assert not document_task.done()

    release_compression.set()
    await index_task
    await document_task

    turns = rag.store["conv-1"]["turns"]
    assert any(
        "important document contents" in t.get("memory", "") for t in turns
    ), "the document indexed during compression must survive the write-back"


# --- Codex round 9 on PR #80: documents/summaries must never be compressed ---


@pytest.mark.asyncio
async def test_compression_skips_document_entry_between_plain_turns(tmp_path, monkeypatch):
    """Codex P2: compression used to take turns[:half] blindly, which could
    catch a document entry (turn == -1) sitting among the oldest plain
    turns. Sweeping it into a summary destroys its sentinel -- the summary
    entry's "turn" is a normal number, so the document loses the
    always-eligible topic-filter bypass retrieve_with_stats_async grants it
    (checking turn == -1 directly) and its content could get filtered out
    of retrieval by topic mismatch post-compaction.

    Seed a conversation whose oldest half contains a document entry
    sandwiched between plain turns, trigger compression, and assert: the
    document survives verbatim with turn == -1 intact; the plain turns
    around it got compressed into one summary; and retrieval with a query
    that shares NO topic with the document's filename-derived topic still
    surfaces the document block (proving the bypass survived)."""
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    rag.store["conv-1"] = {"folder_id": "root", "turns": []}
    turns = rag.store["conv-1"]["turns"]
    # SUMMARY_COMPRESSION_THRESHOLD plain turns with a document sandwiched
    # in the middle of what will become the oldest (compressible) half.
    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        if i == SUMMARY_COMPRESSION_THRESHOLD // 4:
            turns.append({
                "turn": -1,
                "topics": ["document:report.pdf"],
                "memory": "[Uploaded Document: report.pdf]\nconfidential figures",
            })
        turns.append({"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"})

    async def fake_query_model(*args, **kwargs):
        return {"content": "Dense summary of earlier plain turns.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    # This call pushes past the threshold and triggers compaction.
    await rag.index_chat_turn("conv-1", "triggers compaction", "answer", ["z"])

    after = rag.store["conv-1"]["turns"]
    doc_entries = [t for t in after if t.get("turn") == -1]
    summary_entries = [t for t in after if t.get("kind") == "summary"]

    # The document survives verbatim, sentinel intact, not folded into the summary.
    assert len(doc_entries) == 1
    assert doc_entries[0]["memory"] == "[Uploaded Document: report.pdf]\nconfidential figures"
    assert doc_entries[0]["topics"] == ["document:report.pdf"]
    # Exactly one summary entry covering the compressed plain turns.
    assert len(summary_entries) == 1
    assert summary_entries[0]["summary"] == "Dense summary of earlier plain turns."
    # Order is sensible: summary appears before the document in the rebuilt
    # list, matching the original relative order (document was inserted
    # after some already-compressed plain turns but before the rest).
    doc_index = after.index(doc_entries[0])
    summary_index = after.index(summary_entries[0])
    assert summary_index < doc_index

    # Retrieval: a query sharing NO topic with the document's filename-
    # derived topic (or the summary's union topics) must still surface the
    # document block -- proving turn == -1's bypass survived compaction.
    memory_section = await _retrieve_and_capture_memory_section(
        rag, monkeypatch, None, query="completely unrelated astronomy question"
    )
    assert "confidential figures" in memory_section


@pytest.mark.asyncio
async def test_compression_does_not_recompress_an_existing_summary_entry(tmp_path, monkeypatch):
    """Codex P2: an existing summary entry (kind == "summary") must be
    excluded from the next compaction's "oldest half" too -- re-summarizing
    a summary would compound lossily (summary-of-summary drift) instead of
    compressing the newer plain turns that pushed the conversation over the
    threshold again."""
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    existing_summary = {
        "kind": "summary",
        "turn": 0,
        "topics": ["old"],
        "summary": "Original summary of the earliest turns.",
        "turns_compressed": 6,
    }
    rag.store["conv-1"] = {"folder_id": "root", "turns": [existing_summary]}
    turns = rag.store["conv-1"]["turns"]
    for i in range(1, SUMMARY_COMPRESSION_THRESHOLD):
        turns.append({"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"})

    seen_prompts = []

    async def fake_query_model(*args, **kwargs):
        seen_prompts.append(args[1][0]["content"])
        return {"content": "Second summary of newer plain turns.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    await rag.index_chat_turn("conv-1", "triggers second compaction", "answer", ["y"])

    after = rag.store["conv-1"]["turns"]
    summary_entries = [t for t in after if t.get("kind") == "summary"]

    # The original summary's text must never appear in the compression
    # prompt -- it was excluded from the compressible set entirely.
    assert not any("Original summary of the earliest turns." in p for p in seen_prompts)
    # The original summary entry survives unchanged, in place.
    assert existing_summary in after
    # A second, distinct summary entry now also exists for the newer turns.
    assert len(summary_entries) == 2
    assert any(s["summary"] == "Second summary of newer plain turns." for s in summary_entries)


# --- Codex round 10 on PR #80: retrieval read barrier ---


@pytest.mark.asyncio
async def test_retrieve_read_barrier_excludes_zdr_source_conversation(tmp_path, monkeypatch):
    """Codex P1: while a compression call holds a conversation's write lock,
    a ZDR flip or deletion's purge (delete_conversation_memories) is QUEUED
    behind that lock for as long as the LLM call takes -- but retrieval for
    a DIFFERENT conversation reads self.store directly, with no lock and no
    metadata check, so it could surface the source conversation's now-ZDR
    memory for that whole window. Simulates the pending-purge window
    directly (no purge call needed): seed memories for conv-a, flip its
    metadata to zdr_enabled=True in real storage, then retrieve for conv-b
    and assert conv-a's blocks are absent from the extraction prompt --
    fails pre-fix (the read barrier is the only thing that can catch this,
    since the store itself still has conv-a's entries)."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-a")
    storage.create_conversation("conv-b")
    storage.create_conversation("conv-c")  # legitimate other source, so retrieval still has something to build a prompt from

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    rag.store["conv-a"] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "topics": [], "memory": "SECRET_A_CONTENT should not leak"}],
    }
    rag.store["conv-c"] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "topics": [], "memory": "ordinary conv-c memory should still retrieve"}],
    }

    # Simulate the pending-purge window: metadata already flipped to ZDR
    # (as update_conversation would do synchronously) but no purge call has
    # run yet (as if it were still queued behind another writer's lock).
    storage.update_conversation_metadata("conv-a", {"zdr_enabled": True})
    assert "conv-a" in rag.store  # the store itself is untouched -- only the read barrier can catch this

    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    # Deliberately NOT using _retrieve_and_capture_memory_section's shared
    # get_conversation fake here -- this test needs the REAL storage-backed
    # get_conversation (already monkeypatched above) so the barrier actually
    # sees conv-a's just-flipped ZDR metadata.
    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    await rag.retrieve_async("current question", "conv-b", max_tokens=None)

    prompt = captured_kwargs["messages"][0]["content"]
    memory_section = prompt.split("USER MEMORY LOGS:\n", 1)[1]

    assert "SECRET_A_CONTENT" not in memory_section
    assert "conv-a" not in memory_section
    assert "ordinary conv-c memory should still retrieve" in memory_section


# --- Codex round 10 on PR #80: cancellation-safe two-step cost recording ---


@pytest.mark.asyncio
async def test_cancelling_generator_during_indexing_still_records_base_cost(monkeypatch, tmp_path):
    """Codex P2: a client disconnect during the post-response indexing
    awaits used to cancel run_turn's generator BEFORE
    update_conversation_cost/record_session_usage ran at all (they used to
    be a single shot at the very end) -- the saved chat message's cost never
    reached conversation total_cost or session usage. Fixed with two-step
    accounting: the BASE cost is recorded synchronously, immediately after
    the message is persisted, before any indexing awaits. This drives a
    real cancellation (gen.aclose()) while index_chat_turn is blocked on an
    event, and asserts the base cost already landed in both totals despite
    the cancellation -- fails pre-fix (both stay at zero)."""
    import asyncio

    main = import_module_with_api_key(monkeypatch, "backend.main")
    conversation_id = "conv-cancel-mid-indexing"

    chairman_usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}

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
        return {"content": "Budget-aware response", "usage": chairman_usage}

    async def fake_extract_topics_with_usage(*args, **kwargs):
        return ["topic"], {}

    index_started = asyncio.Event()
    never_releases = asyncio.Event()

    async def blocking_index_chat_turn(*args, **kwargs):
        index_started.set()
        await never_releases.wait()  # never set -- simulates the client disconnecting here
        return None

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics_with_usage", fake_extract_topics_with_usage)
    monkeypatch.setattr(main, "chat_with_chairman", fake_chat_with_chairman)
    monkeypatch.setattr(
        main,
        "rag_system",
        SimpleNamespace(
            retrieve_async=fake_retrieve_async,
            index_chat_turn=blocking_index_chat_turn,
            refresh_hybrid_index=lambda *a, **k: None,
            store={},
        ),
    )

    conversation, mode, zdr_enabled, thinking_effort, is_first_message = main.prepare_turn(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
    )
    gen = main.run_turn(
        conversation_id,
        main.SendMessageRequest(content="Follow up", mode="chat"),
        conversation=conversation,
        mode=mode,
        zdr_enabled=zdr_enabled,
        thinking_effort=thinking_effort,
        is_first_message=is_first_message,
    )

    # chat_response is yielded AFTER index_chat_turn returns, but
    # index_chat_turn never returns here -- so consuming the generator must
    # run as a background task, not inline, or this coroutine would block
    # forever waiting for an event that can't happen before the cancellation
    # this test is trying to trigger.
    async def consume():
        async for _event in gen:
            pass

    consumer_task = asyncio.create_task(consume())

    # index_chat_turn has started (and is blocked mid-await) -- the message
    # was already saved and the base cost already recorded by this point.
    # Cancel the CONSUMER task, exactly like Starlette does on a client
    # disconnect: the cancellation propagates into the generator at its
    # suspended await. (aclose() while the consumer is mid-anext would
    # raise "generator is already running".)
    await index_started.wait()
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    await gen.aclose()

    # openai/gpt-4o-mini pricing: input=$0.15/M, output=$0.6/M
    base_cost = (1_000_000 / 1_000_000) * 0.15 + (1_000_000 / 1_000_000) * 0.6

    conversation = main.storage.get_conversation(conversation_id)
    usage = main.storage.get_session_usage(conversation_id)

    assert conversation["total_cost"] == pytest.approx(base_cost)
    assert usage["spent_usd"] == pytest.approx(base_cost)
    assert usage["messages"] == 1


# --- Codex round 11 on PR #80: read barrier hardening + summary merge ---


@pytest.mark.asyncio
async def test_retrieve_read_barrier_skips_source_conversation_whose_get_conversation_raises(tmp_path, monkeypatch):
    """Codex P2: the round-10 read barrier calls get_conversation(cid) for
    every source conversation. That call can RAISE (corrupt JSON, file
    deleted mid-read) -- an unrelated source's read failure must not abort
    retrieval for every OTHER source. Fails pre-fix (the whole call raises,
    no context is returned at all)."""
    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    storage.create_conversation("conv-good")
    storage.create_conversation("conv-current")

    real_get_conversation = storage.get_conversation

    def flaky_get_conversation(cid):
        if cid == "conv-bad":
            raise OSError("simulated corrupt read for conv-bad")
        return real_get_conversation(cid)

    monkeypatch.setattr(rag_module, "get_conversation", flaky_get_conversation)

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    rag.store["conv-bad"] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "topics": [], "memory": "memory behind a raising read"}],
    }
    rag.store["conv-good"] = {
        "folder_id": "root",
        "turns": [{"turn": 0, "topics": [], "memory": "ordinary conv-good memory should still retrieve"}],
    }

    captured_kwargs = {}

    async def fake_query_model(*args, **kwargs):
        captured_kwargs["messages"] = args[1]
        return {"content": "irrelevant"}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    # Must not raise, despite conv-bad's get_conversation raising.
    result = await rag.retrieve_async("current question", "conv-current", max_tokens=None)

    assert result[0] != ""  # retrieval succeeded overall
    prompt = captured_kwargs["messages"][0]["content"]
    memory_section = prompt.split("USER MEMORY LOGS:\n", 1)[1]

    assert "ordinary conv-good memory should still retrieve" in memory_section
    assert "memory behind a raising read" not in memory_section


@pytest.mark.asyncio
async def test_summary_merge_combines_accumulated_summaries_at_first_position(tmp_path, monkeypatch):
    """Codex P2: summary entries are permanently exempt from plain-turn
    compression (round 9), so without a second-level cap they accumulate
    one per compaction cycle -- unbounded growth. Seed MAX_SUMMARY_ENTRIES+1
    summaries, plus a document and some plain turns, and trigger a
    compaction that also crosses the summary-merge threshold: the summaries
    merge into ONE at the position of the first summary (min turn, union
    topics, summed turns_compressed); plain turns and the document are
    untouched."""
    from backend.rag import MAX_SUMMARY_ENTRIES, SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    turns = []
    for i in range(MAX_SUMMARY_ENTRIES + 1):
        turns.append({
            "kind": "summary",
            "turn": i * 10,
            "topics": [f"topic{i}"],
            "summary": f"Summary number {i}.",
            "turns_compressed": 2 + i,
        })
    document_entry = {
        "turn": -1,
        "topics": ["document:report.pdf"],
        "memory": "[Uploaded Document: report.pdf]\nverbatim body",
    }
    turns.append(document_entry)
    # Enough plain turns to cross SUMMARY_COMPRESSION_THRESHOLD and trigger
    # the plain-turn compaction path too, so this exercises both tiers in
    # one call (the merge runs from inside that same call).
    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        turns.append({"turn": 1000 + i, "topics": [f"plain{i}"], "memory": f"Q{i}\nA{i}"})
    rag.store["conv-1"] = {"folder_id": "root", "turns": turns}

    responses = iter([
        {"content": "Compressed plain-turn summary.", "usage": {"prompt_tokens": 3, "completion_tokens": 3}},
        {"content": "Merged summary of everything.", "usage": {"prompt_tokens": 7, "completion_tokens": 7}},
    ])

    async def fake_query_model(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    usage = await rag.index_chat_turn("conv-1", "new question", "new answer", ["z"])

    after = rag.store["conv-1"]["turns"]
    summary_entries = [t for t in after if t.get("kind") == "summary"]
    doc_entries = [t for t in after if t.get("turn") == -1]

    # All original + new summaries merged into exactly one.
    assert len(summary_entries) == 1
    merged = summary_entries[0]
    assert merged["summary"] == "Merged summary of everything."
    # min turn across the pre-merge summaries (0, 10, 20, 30) is 0.
    assert merged["turn"] == 0
    assert set(merged["topics"]) >= {"topic0", "topic1", "topic2", "topic3"}
    # summed turns_compressed: (2+3+4+5) from the original 4 summaries, plus
    # whatever the plain-turn compaction's own new summary contributed.
    assert merged["turns_compressed"] >= (2 + 3 + 4 + 5)

    # Document survives verbatim, untouched.
    assert len(doc_entries) == 1
    assert doc_entries[0]["memory"] == "[Uploaded Document: report.pdf]\nverbatim body"

    # The merged entry sits where the FIRST original summary was (before
    # the document and the plain turns in original order).
    merged_index = after.index(merged)
    doc_index = after.index(doc_entries[0])
    assert merged_index < doc_index

    # Returned usage sums both LLM calls (compression + merge).
    assert usage == {"prompt_tokens": 10, "completion_tokens": 10}


@pytest.mark.asyncio
async def test_summary_merge_skips_on_llm_failure_leaving_all_summaries_intact(tmp_path, monkeypatch, caplog):
    """Codex P2: merge-LLM failure must be fail-open -- skip the merge
    entirely, no data loss. All summaries stay intact."""
    from backend.rag import MAX_SUMMARY_ENTRIES

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    turns = [
        {
            "kind": "summary",
            "turn": i * 10,
            "topics": [f"topic{i}"],
            "summary": f"Summary number {i}.",
            "turns_compressed": 2,
        }
        for i in range(MAX_SUMMARY_ENTRIES + 1)
    ]
    rag.store["conv-1"] = {"folder_id": "root", "turns": list(turns)}

    async def fake_query_model_none(*args, **kwargs):
        return None

    monkeypatch.setattr(rag_module, "query_model", fake_query_model_none)

    with caplog.at_level("WARNING"):
        merge_usage = await rag._maybe_merge_summaries("conv-1")

    assert merge_usage is None
    after = rag.store["conv-1"]["turns"]
    assert after == turns  # untouched
    summary_entries = [t for t in after if t.get("kind") == "summary"]
    assert len(summary_entries) == MAX_SUMMARY_ENTRIES + 1
    assert any("summary merge" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_repeated_compaction_cycles_keep_entry_count_bounded(tmp_path, monkeypatch):
    """Codex P2: growth-bound check. Loop many chat turns through a
    conversation with a mocked (always-succeeding) compression/merge LLM,
    and assert the total entry count never exceeds
    threshold + MAX_SUMMARY_ENTRIES + documents -- proving the second-level
    summary-merge tier actually keeps overall growth bounded over many
    cycles, not just one."""
    from backend.rag import MAX_SUMMARY_ENTRIES, SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))

    call_count = 0

    async def fake_query_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {"content": f"Summary #{call_count}.", "usage": {}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model)

    max_seen = 0
    for i in range(80):  # many cycles: several plain-turn compactions and summary merges
        await rag.index_chat_turn("conv-1", f"question {i}", f"answer {i}", [f"topic{i}"])
        max_seen = max(max_seen, len(rag.store["conv-1"]["turns"]))

    documents = 0  # this conversation never indexes a document
    bound = SUMMARY_COMPRESSION_THRESHOLD + MAX_SUMMARY_ENTRIES + documents
    assert max_seen <= bound, f"entry count grew to {max_seen}, exceeding the bound {bound}"


# --- Codex round 12 on PR #80: bill compression attempts even on blank content ---


@pytest.mark.asyncio
async def test_compression_bills_usage_even_when_response_content_is_blank(tmp_path, monkeypatch):
    """Codex P2: a compression attempt whose LLM response has real USAGE but
    blank content used to skip compression AND discard the usage entirely
    (returning None), silently eating the spent tokens. Fixed: bill
    whatever usage the call actually consumed regardless of whether the
    store was modified -- the store stays unchanged, but the returned usage
    must still reflect the blank-content response's real token spend."""
    from backend.rag import SUMMARY_COMPRESSION_THRESHOLD

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    for i in range(SUMMARY_COMPRESSION_THRESHOLD):
        rag.store.setdefault("conv-1", {"folder_id": "root", "turns": []})
        rag.store["conv-1"]["turns"].append(
            {"turn": i, "topics": [f"topic{i}"], "memory": f"Q{i}\nA{i}"}
        )
    turns_before = [dict(t) for t in rag.store["conv-1"]["turns"]]

    async def fake_query_model_blank_content(*args, **kwargs):
        return {"content": "", "usage": {"prompt_tokens": 42, "completion_tokens": 7}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model_blank_content)

    usage = await rag.index_chat_turn("conv-1", "new question", "new answer", ["newtopic"])

    turns_after = rag.store["conv-1"]["turns"]
    # Store unchanged by the compression attempt itself (the new turn from
    # this same index_chat_turn call is the only addition -- no summary).
    assert turns_after[:-1] == turns_before
    assert not any(t.get("kind") == "summary" for t in turns_after)
    # The blank-content response's real usage must still be billed.
    assert usage == {"prompt_tokens": 42, "completion_tokens": 7}


@pytest.mark.asyncio
async def test_summary_merge_bills_usage_even_when_response_content_is_blank(tmp_path, monkeypatch):
    """Codex P2: same fix, merge path. A merge attempt whose response has
    usage but blank content must still bill that usage; all summaries stay
    intact."""
    from backend.rag import MAX_SUMMARY_ENTRIES

    conversations_dir = tmp_path / "conversations"
    monkeypatch.setattr(storage, "DATA_DIR", str(conversations_dir))
    monkeypatch.setattr(rag_module, "get_conversation", storage.get_conversation)
    storage.create_conversation("conv-1")

    rag = CouncilRAG(persist_path=str(tmp_path / "pageindex"))
    turns = [
        {
            "kind": "summary",
            "turn": i * 10,
            "topics": [f"topic{i}"],
            "summary": f"Summary number {i}.",
            "turns_compressed": 2,
        }
        for i in range(MAX_SUMMARY_ENTRIES + 1)
    ]
    rag.store["conv-1"] = {"folder_id": "root", "turns": list(turns)}

    async def fake_query_model_blank_content(*args, **kwargs):
        return {"content": "   ", "usage": {"prompt_tokens": 11, "completion_tokens": 3}}

    monkeypatch.setattr(rag_module, "query_model", fake_query_model_blank_content)

    merge_usage = await rag._maybe_merge_summaries("conv-1")

    after = rag.store["conv-1"]["turns"]
    assert after == turns  # untouched
    assert merge_usage == {"prompt_tokens": 11, "completion_tokens": 3}
