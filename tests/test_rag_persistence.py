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
        return ["topic"]

    async def fake_index_chat_turn(*args, **kwargs):
        return compression_usage

    monkeypatch.setattr("backend.council.rewrite_query", fake_rewrite_query)
    monkeypatch.setattr("backend.council.extract_topics", fake_extract_topics)
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

    # Simulate the user enabling ZDR mid-flight: real metadata flip + the
    # same purge update_conversation performs.
    storage.update_conversation_metadata("conv-1", {"zdr_enabled": True})
    rag.delete_conversation_memories("conv-1")

    lock.release()
    await pending_task

    # The pending call must have written nothing: the barrier, re-checked
    # under the lock, saw the now-current ZDR metadata and refused.
    assert "conv-1" not in rag.store
