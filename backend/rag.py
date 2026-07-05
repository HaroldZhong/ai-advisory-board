import json
import os
import re
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from .logger import logger
from .config import RAG_SETTINGS, UTILITY_MODEL
from .openrouter import query_model
from .app_paths import get_pageindex_dir, write_text_atomic

# Store hygiene (P5-T4): warn, don't auto-evict, once the serialized store
# crosses this size. Destroying memory silently is an owner decision, not
# something this module should do on its own.
STORE_SIZE_WARNING_BYTES = 5 * 1024 * 1024  # 5 MB

# P5-T5 feature 1 (owner decision): bump this to force a one-time memory
# store reset on next startup (see _apply_one_time_memory_reset). Sidecar
# marker file, never a key inside the store dict.
MEMORY_STORE_VERSION = "2"

# P5-T5 feature 3: bounded per-conversation memory growth. Once a
# conversation's turns list exceeds this length, the oldest half is
# compressed into a single summary entry.
SUMMARY_COMPRESSION_THRESHOLD = 12

# Codex round 11: summary entries are permanently excluded from the plain-
# turn compression tier above (round 9), so without a second-level cap they
# accumulate one per compaction cycle -- unbounded growth in very long
# conversations, defeating the whole point of this tier. Once a
# conversation's summary-entry count exceeds this, merge them all into one.
MAX_SUMMARY_ENTRIES = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def _sum_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Add two usage dicts' prompt_tokens/completion_tokens together (Codex
    round 11: a single index_session/index_chat_turn call can trigger both
    plain-turn compression AND a summary merge in the same turn -- callers
    bill the returned usage with one calculate_cost() call, so both LLM
    calls' costs must land in one dict)."""
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
    }


def _tokenize(text: str) -> set:
    """Lowercase, alphanumeric-token split. Shared by the query and topics
    sides of the overlap score so matching is case-insensitive by construction."""
    return set(_WORD_RE.findall(text.lower()))


def score_topic_overlap(query: str, topics: List[str]) -> int:
    """Case-insensitive token-overlap score between a query and a turn's
    stored topics (P5-T4 retrieval pre-filter). Pure and testable in
    isolation: not a search engine, just a relevance pre-filter ahead of the
    existing budget cap. Returns the count of overlapping tokens (0 if
    `topics` is empty/missing or nothing overlaps)."""
    if not topics:
        return 0
    query_tokens = _tokenize(query)
    topic_tokens = _tokenize(" ".join(topics))
    return len(query_tokens & topic_tokens)


def get_conversation(conversation_id: str):
    """Lazy-imported indirection to backend.storage.get_conversation.

    Kept as a module-level seam (rather than importing storage at module
    load time) to avoid a storage<->rag import cycle, and so tests can
    monkeypatch it directly.
    """
    from .storage import get_conversation as _get_conversation
    return _get_conversation(conversation_id)


class CouncilRAG:
    """
    Reasoning-based RAG ("PageIndex") system.
    Replaces ChromaDB vector embeddings with LLM-reasoning cross-folder retrieval.
    """
    def __init__(self, persist_path: Optional[str] = None):
        self.enabled = False
        self.store = {}
        self.index_file = ""
        self.version_file = ""
        # ponytail: per-conversation lock, not a global one -- a global lock
        # would serialize unrelated conversations' writes for no reason. This
        # is a single-process desktop app, so an in-process asyncio.Lock per
        # conversation id is enough to make append+compress atomic against
        # itself; created on demand since conversation ids aren't known
        # up front. Upgrade to a real per-account/process lock if this ever
        # becomes multi-process.
        self._conversation_write_locks: Dict[str, asyncio.Lock] = {}
        try:
            if persist_path is None:
                persist_path = str(get_pageindex_dir())
            os.makedirs(persist_path, exist_ok=True)
            self.index_file = os.path.join(persist_path, "pageindex_memory.json")
            self.version_file = os.path.join(persist_path, "pageindex_memory.version")

            # Snapshot BEFORE loading: the corrupt-JSON recovery path below
            # renames self.index_file away, so checking existence afterward
            # would always say "no store file" even when one existed.
            store_file_existed = os.path.exists(self.index_file)

            # Load existing JSON index
            if os.path.exists(self.index_file):
                try:
                    with open(self.index_file, 'r', encoding='utf-8') as f:
                        self.store = json.load(f)
                except json.JSONDecodeError:
                    backup_path = self._backup_corrupt_index()
                    logger.exception(
                        "[RAG] Corrupt PageIndex store moved to %s; starting empty store",
                        backup_path,
                    )
                    self.store = {}
                if not isinstance(self.store, dict):
                    # Valid JSON but the wrong top-level type (e.g. "[]") skips
                    # the JSONDecodeError path above; memory is derived data,
                    # so resetting is the same safe recovery as the corrupt case.
                    logger.warning(
                        "[RAG] PageIndex store has invalid top-level type %s; resetting",
                        type(self.store).__name__,
                    )
                    self.store = {}
            else:
                self.store = {}

            self.enabled = True
            logger.info("[RAG] Initialized PageIndex Reasoning RAG successfully")
        except Exception as e:
            logger.exception("[RAG] WARNING: Failed to initialize: %s", e)
            self.enabled = False
            self.store = {}

        if self.enabled:
            self._apply_one_time_memory_reset(store_file_existed)
            self.cleanup_zdr_conversations()

    def _apply_one_time_memory_reset(self, store_file_existed: bool) -> None:
        """P5-T5 feature 1 (owner decision): guarantee historical ZDR cleanup
        regardless of the per-message-flag blind spot the startup sweep can't
        retroactively detect (cleanup_zdr_conversations' documented
        limitation). A sidecar version marker file -- deliberately NOT a key
        inside self.store, which would pollute conversation-id iteration --
        gates a one-time reset of the memory store to {}. Only the memory
        store is touched; conversations and attachments live in a completely
        separate directory tree and this method never looks at them.
        """
        try:
            with open(self.version_file, "r", encoding="utf-8") as f:
                current_version = f.read().strip()
        except (OSError, ValueError):
            current_version = None

        if current_version == MEMORY_STORE_VERSION:
            return

        conversations_dropped = len(self.store)
        if store_file_existed and conversations_dropped:
            self.store = {}
            # Codex round 5: the marker must only be written once the reset
            # actually landed on disk. _save_store can fail (disk full,
            # permissions, etc.) and swallow that into self.enabled = False;
            # writing the marker anyway would mark this store "already reset"
            # forever while the stale, un-purged store survives on disk --
            # defeating the whole guarantee. On failure: skip the marker
            # entirely and retry the reset on next startup.
            if not self._save_store():
                logger.error(
                    "[RAG] One-time memory reset for v1.2.0 failed to persist; "
                    "will retry on next startup (marker not written)"
                )
                return
            logger.info(
                "[RAG] Performing one-time memory reset for v1.2.0 ZDR guarantee, %d conversation(s) dropped",
                conversations_dropped,
            )
        # Fresh install (no prior store file) or an already-empty store: just
        # write the marker, no reset log spam since nothing was purged.
        # Codex round 6: this write can also fail (disk full, permissions)
        # and, unlike the reset-save path above, was never wrapped -- letting
        # it raise here would abort CouncilRAG's constructor and crash
        # backend startup entirely. Tolerate it the same way: log and
        # continue with an empty/unchanged in-memory store; the marker being
        # absent just means the next startup retries this whole method.
        try:
            write_text_atomic(Path(self.version_file), MEMORY_STORE_VERSION)
        except (OSError, RuntimeError):
            logger.error(
                "[RAG] Memory version marker could not be written; reset will retry next startup"
            )

    def cleanup_zdr_conversations(self) -> int:
        """One-time-per-process startup sweep (audit §12, Decision #5).

        Index-time exclusion (see turn_pipeline.run_turn) stops new ZDR turns
        from ever being written here, but this store may already contain
        memories from ZDR conversations indexed before that guard existed.
        For every conversation id in the store, look up its current metadata
        and remove the conversation's entries if either:
          (a) `metadata.zdr_enabled` is True, or
          (b) the conversation's file is missing or unreadable (orphaned
              entries). Unavailable metadata cannot prove an entry is safe to
              keep, so cleanup fails closed: a missing file means the
              conversation was deleted (its memories are pure orphaned,
              derived data) and a crash mid-deletion must not be able to
              permanently strand a ZDR conversation's memories.

        LIMITATION (documented, not fixed here): a turn's *effective* ZDR can
        also come from a per-message flag that was never persisted to
        conversation metadata. Pre-fix memories from such per-message-only
        ZDR turns cannot be identified retroactively and are NOT covered by
        this sweep. Going forward there is no gap: index-time exclusion checks
        the per-turn effective ZDR (metadata OR per-message) before writing,
        so no new per-message-only ZDR memory can ever land in the store.
        """
        zdr_removed = 0
        orphans_removed = 0
        for conversation_id in list(self.store.keys()):
            try:
                conversation = get_conversation(conversation_id)
            except Exception:
                logger.exception(
                    "[RAG] Failed to load conversation %s during ZDR cleanup sweep; removing as an orphan (fail closed)",
                    conversation_id,
                )
                del self.store[conversation_id]
                orphans_removed += 1
                continue
            if not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict):
                # None (missing file), a valid-JSON-but-wrong-type record
                # (e.g. a conversation file holding "[]"), or malformed
                # metadata (e.g. "metadata": null): no metadata to trust,
                # so fail closed rather than letting .get() raise below.
                del self.store[conversation_id]
                orphans_removed += 1
                continue
            if conversation["metadata"].get("zdr_enabled") is True:
                del self.store[conversation_id]
                zdr_removed += 1

        removed = zdr_removed + orphans_removed
        if removed:
            self._save_store()
            logger.info(
                "[RAG] ZDR cleanup sweep removed %d ZDR conversation(s) and %d orphaned conversation(s) from PageIndex memory",
                zdr_removed,
                orphans_removed,
            )
        return removed

    def _get_write_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get-or-create the per-conversation write lock (setdefault, not a
        defaultdict, so this stays a plain dict elsewhere).

        Codex round 7: the invariant is now total -- EVERY mutator of
        self.store[conversation_id] holds this lock for its entire body:
        index_session, index_chat_turn, index_document, and
        delete_conversation_memories. No per-conversation store write
        anywhere in this class happens outside this lock."""
        return self._conversation_write_locks.setdefault(conversation_id, asyncio.Lock())

    def _backup_corrupt_index(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        backup_path = f"{self.index_file}.corrupt-{timestamp}"
        os.replace(self.index_file, backup_path)
        return backup_path

    def _save_store(self) -> bool:
        """Persist self.store to disk. Returns True on a confirmed write,
        False when persistence is disabled or the write failed (callers that
        need to know whether the store actually landed on disk -- e.g. the
        one-time reset marker -- must check this instead of assuming)."""
        if not self.enabled:
            return False
        serialized = json.dumps(self.store, indent=2)
        self._warn_if_store_too_large(serialized)
        try:
            write_text_atomic(Path(self.index_file), serialized)
            return True
        except (OSError, RuntimeError) as e:
            logger.exception(
                "[RAG] Failed to persist PageIndex store; disabling persistence for this process: %s",
                e,
            )
            self.enabled = False
            return False

    def _warn_if_store_too_large(self, serialized: str) -> None:
        """Store hygiene (P5-T4): warn once the serialized store crosses
        STORE_SIZE_WARNING_BYTES. Deliberately NO auto-eviction here --
        destroying memory silently is an owner choice, not something this
        module should decide on its own; this only surfaces the problem.
        "Oldest" is approximated by dict insertion order (self.store isn't
        timestamped per-conversation), which is good enough for an operator
        to go look at growth over time.
        """
        size = len(serialized.encode("utf-8"))
        if size <= STORE_SIZE_WARNING_BYTES:
            return
        oldest_ids = list(self.store.keys())[:5]
        logger.warning(
            "[RAG] PageIndex store size is %d bytes (over the %d byte warning threshold); "
            "oldest conversation ids: %s. No automatic eviction is performed.",
            size,
            STORE_SIZE_WARNING_BYTES,
            oldest_ids,
        )

    def refresh_hybrid_index(self) -> None:
        """Legacy compatibility method. No longer needed for reasoning RAG."""
        pass

    async def delete_conversation_memories(self, conversation_id: str) -> bool:
        """
        Remove a conversation's memories from the global PageIndex store.

        Codex round 5: this mutates self.store, so it must honor the same
        per-conversation write lock as index_session/index_chat_turn --
        without it, a purge (runtime ZDR flip, conversation deletion) could
        interleave with an in-flight compression's snapshot-await-writeback
        window and KeyError, or race the compression's write-back and get
        silently undone. Serialized, the purge simply waits for any
        in-flight write to finish, then deletes -- correct either way.
        """
        async with self._get_write_lock(conversation_id):
            if conversation_id in self.store:
                del self.store[conversation_id]
                self._save_store()
                if self.enabled:
                    logger.info("[RAG] Purged PageIndex memories for conversation %s", conversation_id)
                return self.enabled
            return False

    async def purge_truncated_memories(self, conversation_id: str, edit_index: int) -> int:
        """Codex round 13: Edit & Regenerate truncates a conversation's
        messages to messages[:edit_index], but memory entries for the
        discarded turns previously survived indefinitely -- an edited-away
        answer stayed retrievable from OTHER conversations forever.

        Drops every session/chat/summary entry whose "message_anchor"
        exceeds edit_index (its source message(s) no longer exist after the
        truncation). A summary spanning any truncated turn is dropped too
        (message_anchor is the MAX of what it covers) -- lossy but safe:
        keeping a summary that partially describes deleted messages would
        leak edited-away content same as keeping the raw turn would.

        Document entries (turn == -1) are untouched -- they're
        attachment-lifecycle-managed (index_document), not tied to message
        history, and this purge only targets conversational turns.

        An entry with NO "message_anchor" at all (defensive: every entry
        written by this codebase carries one as of the v1.2.0 one-time
        reset) is treated as suspect and dropped too -- fail closed rather
        than trust an anchor-less entry's provenance.
        """
        async with self._get_write_lock(conversation_id):
            if conversation_id not in self.store:
                return 0

            turns = self.store[conversation_id]["turns"]
            kept = []
            dropped = 0
            for entry in turns:
                if entry.get("turn") == -1:
                    kept.append(entry)  # documents: untouched
                    continue
                anchor = entry.get("message_anchor")
                if anchor is None or anchor > edit_index:
                    dropped += 1
                    continue
                kept.append(entry)

            if dropped:
                self.store[conversation_id]["turns"] = kept
                self._save_store()
                if self.enabled:
                    logger.info(
                        "[RAG] Purged %d truncated-turn memor%s for conversation %s (edit_index=%d)",
                        dropped,
                        "y" if dropped == 1 else "ies",
                        conversation_id,
                        edit_index,
                    )
            return dropped

    def update_conversation_folder(self, conversation_id: str, new_folder_id: str):
        """
        Update the folder routing in the reasoning PageIndex for a conversation.
        """
        if conversation_id in self.store:
            self.store[conversation_id]["folder_id"] = new_folder_id
            self._save_store()
            if self.enabled:
                logger.info("[RAG] Updated PageIndex folder routing for conversation %s to %s", conversation_id, new_folder_id)

    async def index_session(
        self,
        conversation_id: str,
        user_question: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        stage3_result: Dict[str, Any],
        topics: List[str],
        quality_metrics: Dict[str, Dict[str, float]],
        expected_anchor: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Index one council session as chronological memory blocks.

        Turn number is computed here (not by the caller): Codex round 6 --
        the pipeline used to pass get_turn_index(conversation), which counts
        only COUNCIL messages, so a chat turn and a council turn in the same
        conversation could land in the store under the same memory turn
        number. Unified with index_chat_turn's scheme: one past the max
        "turn" seen across ALL entries in this conversation's memory (council
        or chat), computed under the same lock that guards the append.

        Returns the summary-compression LLM call's usage dict when the P5-T5
        summary tier compressed this conversation's oldest turns, else None.
        """
        if not self.enabled:
            return None

        stage3_model = stage3_result.get('model', 'unknown')
        final_text = stage3_result.get('response', '')
        if not final_text:
            return None

        # Codex round 3: append + compress must be atomic per conversation.
        # _maybe_compress_oldest_half awaits an LLM call after snapshotting
        # the turns list; without this lock, a concurrent index_session/
        # index_chat_turn call for the SAME conversation can append its own
        # turn during that await, and the stale snapshot's write-back would
        # silently drop it. A per-conversation lock (not global) serializes
        # writers to one conversation without blocking unrelated ones.
        async with self._get_write_lock(conversation_id):
            # ZDR write barrier: never index a conversation whose CURRENT
            # metadata says ZDR. Codex round 4 -- checked UNDER the
            # per-conversation write lock (not before acquiring it) so a task
            # that was suspended waiting for the lock while the user flipped
            # ZDR on and update_conversation purged this conversation cannot
            # then acquire the lock and undo that purge by writing anyway.
            # Per-message ZDR (request flag, not visible in metadata) is
            # enforced by the pipeline-level guards.
            conversation = get_conversation(conversation_id)
            if not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict) or conversation["metadata"].get("zdr_enabled"):
                logger.info("[RAG] Skipping index for %s (ZDR or unreadable)", conversation_id)
                return None

            # Codex round 17 (P1): with the answer delivered before indexing
            # (round 15), an edit/regenerate can truncate this conversation's
            # messages in the window between persistence and this call.
            # purge_truncated_memories runs against the history as it existed
            # THEN -- there's no entry to purge yet -- so a stale answer
            # indexed against an already-truncated history survives future
            # purges with a low, wrong anchor. expected_anchor is the message
            # count the caller observed immediately after persisting THIS
            # turn; if the current count is now lower, the turn was
            # truncated away underneath us -- skip indexing it entirely
            # (same fail-closed instinct as the ZDR barrier just above).
            current_messages = conversation.get("messages", [])
            if expected_anchor is not None and len(current_messages) < expected_anchor:
                logger.info(
                    "[RAG] Skipping index for %s: source turn truncated (expected_anchor=%d, current=%d)",
                    conversation_id, expected_anchor, len(current_messages),
                )
                return None
            # Same-length replacement race: an edit that removes N messages
            # and appends N new ones leaves the count unchanged but the
            # content at the anchor position no longer matches this turn's
            # answer. Cheap to check since the text is already in hand.
            # Codex round 18 (P2): checking only the assistant text missed a
            # narrower leg of the same race -- a replacement turn whose
            # answer happens to match (a generic apology, a short answer)
            # but whose USER PROMPT was replaced still passed, indexing the
            # stale question under a valid-looking anchor. run_turn always
            # persists exactly one add_user_message immediately followed by
            # one add_assistant_message for a turn (no other message lands
            # in between), so the paired user message for the entry at
            # expected_anchor sits one slot earlier, at expected_anchor - 2.
            if expected_anchor is not None and 0 < expected_anchor <= len(current_messages):
                anchor_message = current_messages[expected_anchor - 1]
                paired_user_message = (
                    current_messages[expected_anchor - 2] if expected_anchor >= 2 else None
                )
                if (
                    anchor_message.get("stage3", {}).get("response") != final_text
                    or (paired_user_message is not None and paired_user_message.get("content") != user_question)
                ):
                    logger.info(
                        "[RAG] Skipping index for %s: message at anchor %d no longer matches this turn's Q/A",
                        conversation_id, expected_anchor,
                    )
                    return None

            # Ensure conversation exists in store
            if conversation_id not in self.store:
                self.store[conversation_id] = {
                    "folder_id": "root",
                    "turns": []
                }

            turns = self.store[conversation_id]["turns"]
            turn_index = max((entry.get("turn", -1) for entry in turns), default=-1) + 1

            # Codex round 13: "message_anchor" ties this memory entry to the
            # conversation's message count AT INDEXING TIME. Codex round 17:
            # use the caller-supplied expected_anchor when given -- it was
            # read right after THIS turn's own persistence, before any of the
            # awaits above (topics extraction) that could race with a
            # concurrent edit/regenerate. Falling back to a fresh len() here
            # would re-read the CURRENT (already-truncation-checked-above)
            # count, which is equivalent when nothing raced, but
            # expected_anchor is the more precise, race-free source of truth.
            message_anchor = expected_anchor if expected_anchor is not None else len(current_messages)

            # Only index the user's question and the final synthesized answer to save context tokens for reasoning retrieve
            turn_memory = {
                "turn": turn_index,
                "topics": topics,
                "memory": f"Q: {user_question}\nA: {final_text}",
                "message_anchor": message_anchor,
            }
            turns.append(turn_memory)
            usage = await self._maybe_compress_oldest_half(conversation_id)
            self._save_store()
            if self.enabled:
                logger.info("[PHASE1] Indexed turn %d for conv=%s into PageIndex", turn_index, conversation_id)
            return usage

    async def index_chat_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        topics: List[str],
        expected_anchor: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Index one chat turn into cross-conversation memory. Chat turns
        previously left no memory (P5-T5 feature 2); the entry shape is
        IDENTICAL to a council turn's (question/answer/topics folded into the
        same "memory" text field) so retrieve_with_stats_async's block builder
        and the topic pre-filter need no special-casing.

        Turn number is computed here (not by the caller): Codex round 3 --
        the RAG summary tier can shrink self.store[cid]["turns"] via
        compaction, so a caller-side `len(turns)` reuses a number that
        compaction already freed up. The next number is one past the max
        "turn" seen across ALL entries (including a summary entry, which
        keeps the max turn number it covers), computed under the same lock
        that guards the append -- so two concurrent chat turns for the same
        conversation can never compute the same number.

        Returns the summary-compression LLM call's usage dict when the P5-T5
        summary tier compressed this conversation's oldest turns, else None.
        """
        if not self.enabled:
            return None

        if not answer:
            return None

        # Codex round 3: append + compress must be atomic per conversation,
        # same rationale as index_session's lock above.
        async with self._get_write_lock(conversation_id):
            # ZDR write barrier: identical to index_session's (copied, not
            # refactored into a shared helper here, to keep this diff minimal
            # and match the existing index_document barrier's own copy of the
            # same check). Codex round 4 -- checked UNDER the per-conversation
            # write lock so a purge triggered while waiting for the lock
            # cannot be undone.
            conversation = get_conversation(conversation_id)
            if not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict) or conversation["metadata"].get("zdr_enabled"):
                logger.info("[RAG] Skipping index for %s (ZDR or unreadable)", conversation_id)
                return None

            # Codex round 17 (P1): see index_session's identical comment --
            # with the answer delivered before indexing (round 15), an
            # edit/regenerate can truncate this conversation between
            # persistence and this call. Skip if the source turn is gone.
            current_messages = conversation.get("messages", [])
            if expected_anchor is not None and len(current_messages) < expected_anchor:
                logger.info(
                    "[RAG] Skipping index for %s: source turn truncated (expected_anchor=%d, current=%d)",
                    conversation_id, expected_anchor, len(current_messages),
                )
                return None
            # Same-length replacement race: see index_session's identical
            # comment. Codex round 18 (P2): also verify the paired user
            # message at expected_anchor - 2 (see index_session's identical
            # comment on the [.., user, assistant] layout run_turn always
            # persists) -- not just the assistant answer.
            if expected_anchor is not None and 0 < expected_anchor <= len(current_messages):
                anchor_message = current_messages[expected_anchor - 1]
                paired_user_message = (
                    current_messages[expected_anchor - 2] if expected_anchor >= 2 else None
                )
                if (
                    anchor_message.get("content") != answer
                    or (paired_user_message is not None and paired_user_message.get("content") != question)
                ):
                    logger.info(
                        "[RAG] Skipping index for %s: message at anchor %d no longer matches this turn's Q/A",
                        conversation_id, expected_anchor,
                    )
                    return None

            if conversation_id not in self.store:
                self.store[conversation_id] = {"folder_id": "root", "turns": []}

            turns = self.store[conversation_id]["turns"]
            turn_index = max((entry.get("turn", -1) for entry in turns), default=-1) + 1
            # Codex round 17: use the caller-supplied expected_anchor (read
            # right after this turn's own persistence, race-free) instead of
            # re-reading len() here -- see index_session's identical comment.
            message_anchor = expected_anchor if expected_anchor is not None else len(current_messages)

            turn_memory = {
                "turn": turn_index,
                "topics": topics,
                "memory": f"Q: {question}\nA: {answer}",
                "message_anchor": message_anchor,
            }
            turns.append(turn_memory)
            usage = await self._maybe_compress_oldest_half(conversation_id)
            self._save_store()
            if self.enabled:
                logger.info("[RAG] Indexed chat turn %d for conv=%s into PageIndex", turn_index, conversation_id)
            return usage

    @staticmethod
    def _conversation_zdr_or_unreadable(conversation_id: str) -> bool:
        """Codex round 20: fresh ZDR/unreadable check shared by the two
        compression-tier prompt builders (_maybe_compress_oldest_half,
        _maybe_merge_summaries). Both send stored conversation content to
        UTILITY_MODEL after at least one prior await in their call chain --
        a ZDR flip landing in that window isn't covered by the write
        barrier the caller (index_session/index_chat_turn) already checked
        before the append. Fail closed (True) on an unreadable/raising
        read, same convention as _zdr_flipped_on. Not used by the write
        barriers themselves -- those checks are the FIRST read in their
        call chain (no prior await to race), and are kept as their own
        inline copies (Codex round 4/17 comments explain why).
        """
        try:
            conversation = get_conversation(conversation_id)
        except Exception:
            return True
        return not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict) or bool(conversation["metadata"].get("zdr_enabled"))

    async def _maybe_compress_oldest_half(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """P5-T5 feature 3: bounded per-conversation memory growth.

        Once a conversation's turns list exceeds SUMMARY_COMPRESSION_THRESHOLD,
        compress the OLDEST half of the COMPRESSIBLE entries into a single
        dense summary entry via UTILITY_MODEL. On LLM failure (None), skip
        compression entirely -- no data loss, just unbounded growth for this
        conversation until the next successful attempt. Callers are
        responsible for _save_store()/logging the caller-specific context;
        this only mutates self.store in memory and returns the summarization
        call's usage (or None).

        Codex round 9: document entries (index_document, sentinel
        turn == -1) and existing summary entries (kind == "summary") are
        NON-COMPRESSIBLE and excluded from the "oldest half" -- a document
        swept into a summary loses its always-eligible topic-filter bypass
        (retrieve_with_stats_async checks turn == -1 directly), and
        re-summarizing an existing summary would drift/compound lossily.
        Both stay in place at their original positions; only the plain
        turns among them get compressed and collapsed into one summary
        entry at the position of the first compressed plain turn.
        """
        turns = self.store[conversation_id]["turns"]
        # Threshold counts ALL entries (not just compressible ones): simpler
        # than a second running count, and still bounds growth -- documents/
        # summaries are typically a small minority, so total length tracks
        # compressible length closely enough to trigger compaction at
        # roughly the same cadence.
        if len(turns) <= SUMMARY_COMPRESSION_THRESHOLD:
            return None

        compressible_indices = [
            i for i, entry in enumerate(turns)
            if entry.get("turn") != -1 and entry.get("kind") != "summary"
        ]
        half = (len(compressible_indices) + 1) // 2
        if half < 2:
            # Nothing meaningful to compress (0 or 1 compressible entries) --
            # e.g. an all-documents conversation. Skip; growth is bounded by
            # documents being finite in practice, not by this tier.
            return None
        oldest_indices = compressible_indices[:half]
        oldest_positions = set(oldest_indices)

        # Codex round 20 (P1): symmetric to _maybe_merge_summaries' round-17
        # fix -- this is the FIRST utility-model call in the compression
        # chain, but it still runs after the caller's write-barrier check
        # (index_session/index_chat_turn) and after this method's own
        # threshold/oldest-half computation above, both no-await work. The
        # write barrier's read is stale the moment ANY await happens after
        # it; re-check fresh, immediately before building this prompt, and
        # fail closed on an unreadable/raising read.
        # Codex round 20 (P1): symmetric to _maybe_merge_summaries' round-17
        # fix -- this is the FIRST utility-model call in the compression
        # chain, but it still runs after the caller's write-barrier check
        # (index_session/index_chat_turn) and after this method's own
        # threshold/oldest-half computation above, both no-await work. The
        # write barrier's read is stale the moment ANY await happens after
        # it; re-check fresh, immediately before building this prompt, and
        # fail closed on an unreadable/raising read.
        if self._conversation_zdr_or_unreadable(conversation_id):
            logger.info("[RAG] Skipping summary compression for %s (ZDR or unreadable)", conversation_id)
            return None

        oldest = [turns[i] for i in oldest_indices]
        compact_text = "\n\n".join(
            entry.get("summary", entry.get("memory", "")) for entry in oldest
        )
        prompt = (
            "Summarize the following conversation turns into a single dense, "
            "factual summary. Preserve concrete facts, decisions, and numbers. "
            "Be concise.\n\n" + compact_text
        )
        try:
            response = await query_model(UTILITY_MODEL, [{"role": "user", "content": prompt}], timeout=15.0)
        except Exception:
            logger.exception("[RAG] Summary compression call failed for conv=%s; skipping compression", conversation_id)
            return None

        if not response or not (response.get("content") or "").strip():
            # Codex round 12 (P2): bill whatever tokens the call actually
            # spent even though the store isn't modified -- a blank-content
            # response can still carry real usage, and discarding it here
            # silently ate that cost. Same convention as the exception path
            # above (which never spent tokens, so None is correct there).
            logger.warning("[RAG] Summary compression returned no content for conv=%s; skipping compression", conversation_id)
            return response.get("usage") or {} if response else None

        # Codex round 5 belt: delete_conversation_memories now honors this
        # same write lock, so it can no longer interleave with this await --
        # but check anyway, cheaply, to future-proof against any mutator that
        # ever forgets to take the lock. If the conversation vanished from
        # the store while this await was in flight, there is nothing left to
        # write back into; usage was already spent regardless.
        if conversation_id not in self.store:
            logger.warning(
                "[RAG] Conversation %s removed from PageIndex store during summary compression; discarding the compressed result",
                conversation_id,
            )
            return response.get("usage") or {}

        union_topics = []
        for entry in oldest:
            for topic in entry.get("topics") or []:
                if topic not in union_topics:
                    union_topics.append(topic)

        summary_entry = {
            "kind": "summary",
            "turn": oldest[0]["turn"],
            "topics": union_topics,
            "summary": response["content"].strip(),
            "turns_compressed": half,
            # Codex round 13: the max of the compressed entries' anchors --
            # purge_truncated_memories drops this summary if ANY turn it
            # covers had its source messages truncated away (lossy but
            # safe: keeping a summary that partially describes deleted
            # messages would leak edited-away content).
            "message_anchor": max((entry.get("message_anchor", 0) for entry in oldest), default=0),
        }
        # Codex round 9: rebuild in place, like the retrieval topic filter
        # does -- non-compressible entries (documents, prior summaries) stay
        # exactly where they were; the summary entry replaces the compressed
        # span at the position of its first compressed entry, and later
        # compressed entries are dropped (not re-emitted). Codex round 3: no
        # other writer can have appended to `turns` during the query_model
        # await above (this conversation's write lock is held for this
        # method's entire caller body), so this rebuild still sees an
        # accurate snapshot. The kept entries' own "turn" numbers are
        # untouched, so the next index_chat_turn's max()+1 scan still yields
        # a correct, monotonic number regardless of what number this summary
        # entry carries.
        rebuilt = []
        summary_inserted = False
        for i, entry in enumerate(turns):
            if i not in oldest_positions:
                rebuilt.append(entry)
                continue
            if not summary_inserted:
                rebuilt.append(summary_entry)
                summary_inserted = True
        self.store[conversation_id]["turns"] = rebuilt

        usage = response.get("usage") or {}
        # Codex round 11: second-level compaction -- merge accumulated
        # summaries so they don't grow unbounded. Runs under the same write
        # lock the caller already holds, right after this compaction's own
        # write-back, so it sees the just-rebuilt list.
        merge_usage = await self._maybe_merge_summaries(conversation_id)
        if merge_usage:
            usage = _sum_usage(usage, merge_usage)
        return usage

    async def _maybe_merge_summaries(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Codex round 11: second-level compaction. Document entries
        (turn == -1) are permanently exempt -- bounded by upload count in
        practice, and their bodies must stay verbatim (retrieval's topic
        filter bypass depends on turn == -1, round 9). Summary entries are
        NOT permanently exempt: once their count exceeds MAX_SUMMARY_ENTRIES,
        merge ALL of them into one via UTILITY_MODEL, replacing them at the
        position of the FIRST summary. On LLM failure: skip entirely, no
        data loss -- same fail-open convention as the plain-turn tier.
        """
        turns = self.store[conversation_id]["turns"]
        summary_positions = [i for i, entry in enumerate(turns) if entry.get("kind") == "summary"]
        if len(summary_positions) <= MAX_SUMMARY_ENTRIES:
            return None

        # Codex round 17 (P1): this method is reached via
        # _maybe_compress_oldest_half, which is itself called AFTER that
        # method's own compression call already awaited once. A ZDR flip
        # landing in that window isn't covered by any earlier check in this
        # call chain -- re-read metadata fresh, immediately before building
        # THIS method's own utility-model prompt, and fail closed (skip) on
        # an unreadable/raising read, same convention as _zdr_flipped_on.
        # Codex round 20: extracted into _conversation_zdr_or_unreadable,
        # shared with _maybe_compress_oldest_half's identical fresh check.
        if self._conversation_zdr_or_unreadable(conversation_id):
            logger.info("[RAG] Skipping summary merge for %s (ZDR or unreadable)", conversation_id)
            return None

        summaries = [turns[i] for i in summary_positions]
        merge_text = "\n\n".join(s["summary"] for s in summaries)
        prompt = (
            "Combine the following summaries into a single dense, factual "
            "summary. Preserve concrete facts, decisions, and numbers. "
            "Be concise.\n\n" + merge_text
        )
        try:
            response = await query_model(UTILITY_MODEL, [{"role": "user", "content": prompt}], timeout=15.0)
        except Exception:
            logger.exception("[RAG] Summary merge call failed for conv=%s; skipping merge", conversation_id)
            return None

        if not response or not (response.get("content") or "").strip():
            # Codex round 12 (P2): same fix as the plain-turn compression
            # path -- bill the usage a blank-content response still carries,
            # even though the merge doesn't apply.
            logger.warning("[RAG] Summary merge returned no content for conv=%s; skipping merge", conversation_id)
            return response.get("usage") or {} if response else None

        if conversation_id not in self.store:
            logger.warning(
                "[RAG] Conversation %s removed from PageIndex store during summary merge; discarding the merged result",
                conversation_id,
            )
            return response.get("usage") or {}

        union_topics = []
        for s in summaries:
            for topic in s.get("topics") or []:
                if topic not in union_topics:
                    union_topics.append(topic)

        merged_entry = {
            "kind": "summary",
            "turn": min(s["turn"] for s in summaries),
            "topics": union_topics,
            "summary": response["content"].strip(),
            "turns_compressed": sum(s.get("turns_compressed", 0) for s in summaries),
            # Codex round 13: merging summaries is still a max -- the merged
            # entry is only as safe to keep as its riskiest ancestor.
            "message_anchor": max((s.get("message_anchor", 0) for s in summaries), default=0),
        }
        summary_positions_set = set(summary_positions)
        rebuilt = []
        merged_inserted = False
        for i, entry in enumerate(turns):
            if i not in summary_positions_set:
                rebuilt.append(entry)
                continue
            if not merged_inserted:
                rebuilt.append(merged_entry)
                merged_inserted = True
        self.store[conversation_id]["turns"] = rebuilt
        return response.get("usage") or {}

    async def index_document(self, conversation_id: str, filename: str, text: str, max_chars: int = 8000):
        """
        Index extracted document text into the PageIndex store.
        Truncates to max_chars to keep the reasoning store manageable.

        Codex round 7: this is the last per-conversation store mutator that
        didn't hold the write lock -- a document appended here during an
        in-flight compression's snapshot-await-writeback window (inside
        index_session/index_chat_turn) would get silently dropped by that
        compression's stale write-back. Locked the same way as the other
        three mutators (index_session, index_chat_turn,
        delete_conversation_memories) to close it.
        """
        if not self.enabled or not text.strip():
            return

        async with self._get_write_lock(conversation_id):
            # ZDR write barrier: never index a conversation whose CURRENT
            # metadata says ZDR. Checked UNDER the per-conversation write
            # lock (mirrors index_session/index_chat_turn, Codex round 4) so
            # a purge triggered while waiting for the lock cannot be undone.
            conversation = get_conversation(conversation_id)
            if not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict) or conversation["metadata"].get("zdr_enabled"):
                logger.info("[RAG] Skipping index for %s (ZDR or unreadable)", conversation_id)
                return

            if conversation_id not in self.store:
                self.store[conversation_id] = {"folder_id": "root", "turns": []}

            truncated = text[:max_chars]
            doc_memory = {
                "turn": -1,  # Sentinel: document, not a turn
                # run_turn passes the attachment id as `filename`; stored
                # explicitly so attachment deletion (truncation cleanup or
                # the DELETE endpoint) can purge this memory precisely.
                "attachment_id": filename,
                "topics": [f"document:{filename}"],
                "memory": f"[Uploaded Document: {filename}]\n{truncated}"
            }
            self.store[conversation_id]["turns"].append(doc_memory)
            self._save_store()
            if self.enabled:
                logger.info("[RAG] Indexed document '%s' (%d chars) for conv=%s", filename, len(truncated), conversation_id)

    async def purge_document_memories(self, attachment_ids, conversation_id=None):
        """Drop document memories whose source attachments were deleted.

        conversation_id given: purge that conversation only (edit/regenerate
        truncation cleanup knows the conversation). None: sweep every stored
        conversation (the DELETE /api/attachments endpoint has no conversation
        context). Each conversation is mutated under its own write lock, same
        as every other store mutator.
        """
        if not self.enabled or not attachment_ids:
            return 0
        wanted = set(attachment_ids)
        target_ids = [conversation_id] if conversation_id is not None else list(self.store.keys())
        removed = 0
        for cid in target_ids:
            async with self._get_write_lock(cid):
                data = self.store.get(cid)
                if not data:
                    continue
                turns = data.get("turns", [])
                kept = [
                    entry for entry in turns
                    if not (entry.get("turn") == -1 and entry.get("attachment_id") in wanted)
                ]
                if len(kept) != len(turns):
                    removed += len(turns) - len(kept)
                    data["turns"] = kept
                    self._save_store()
        if removed:
            logger.info("[RAG] Purged %d document memor%s for deleted attachments", removed, "y" if removed == 1 else "ies")
        return removed

    def retrieve(
        self,
        query: str,
        conversation_id: str,
        max_tokens: int = None,
        zdr_enabled: bool = False,
    ) -> str:
        """Backward compatibility wrapper. Should ideally use retrieve_async."""
        logger.warning("Synchronous retrieve called on PageIndex. May block.")
        # We can't easily wait async in potentially sync contexts without event loops,
        # but in FastAPI, most callers are async. For Phase1 eval which might be sync,
        # we try to get the existing loop or run new.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return "" # Can't use sync retrieve in running loop easily
            else:
                result = loop.run_until_complete(
                    self.retrieve_with_stats_async(
                        query,
                        conversation_id,
                        max_tokens,
                        zdr_enabled=zdr_enabled,
                    )
                )
                return result["context"]
        except Exception:
            return ""

    async def retrieve_async(
        self,
        query: str,
        conversation_id: str,
        max_tokens: int = None,
        zdr_enabled: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        """Async version of retrieve to support the litellm calls properly.

        Returns (context, usage): usage is the extraction call's token usage
        (audit §12) so callers can account for its cost in turn_cost/session
        budget, which was previously invisible. Empty dict when no extraction
        call ran (e.g. nothing to retrieve) or it failed.
        """
        result = await self.retrieve_with_stats_async(
            query,
            conversation_id,
            max_tokens,
            zdr_enabled=zdr_enabled,
        )
        return result["context"], result.get("usage", {})
        
    def retrieve_with_stats(self, query: str, conversation_id: str, max_tokens: int = None) -> Dict[str, Any]:
        """Backward compat, returns empty since reasoning implies async LLM calls"""
        logger.warning("[RAG] Synch retrieve_with_stats called on PageIndex.")
        return {"context": "", "used_tokens": 0, "pieces": 0}

    async def retrieve_with_stats_async(
        self,
        query: str,
        conversation_id: str,
        max_tokens: int = None,
        zdr_enabled: bool = False,
    ) -> Dict[str, Any]:
        """
        Reasoning-based RAG Retrieval.
        Instead of chunk similarity, we pass the memory logs of OTHER conversations to a fast LLM 
        and ask it to extract relevant context facts for the current query.
        """
        if not self.enabled or not self.store:
            return {"context": "", "used_tokens": 0, "pieces": 0, "usage": {}}

        # 1. Gather all memories EXCEPT the current conversation
        # This prevents the RAG from repeating the short-term history which the chat already has.
        other_convs = {
            cid: data for cid, data in self.store.items()
            if cid != conversation_id
        }

        # Codex round 10 (P1): read barrier -- mirrors the write barrier. A
        # ZDR flip or conversation deletion's purge (delete_conversation_memories)
        # can be QUEUED behind another writer's write lock (e.g. an in-flight
        # compression) for as long as that writer's LLM call takes. Without
        # this check, retrieval for a DIFFERENT conversation reads self.store
        # directly with no lock and no metadata check, so it could surface a
        # source conversation's now-ZDR/deleted memory for that whole window
        # -- a purge waiting on the write lock cannot be undone by a write
        # (round 4), but it also must not leak through a read that runs
        # before the purge gets its turn. Fail closed: skip a source
        # conversation whose CURRENT state is missing, malformed, or ZDR.
        # The purge itself still completes after the in-flight write
        # finishes (that delay is fine); this only closes the READ leak.
        filtered_convs = {}
        for cid, data in other_convs.items():
            # Codex round 11 (P2): get_conversation can RAISE (corrupt JSON,
            # file deleted mid-read) -- an unrelated source conversation's
            # read failure must not abort retrieval for every OTHER source.
            # Same fail-closed treatment as an unreadable/malformed result:
            # skip just this one source.
            try:
                source_conversation = get_conversation(cid)
            except Exception:
                logger.info("[RAG] Skipping unreadable source conversation %s during retrieval", cid)
                continue
            if not isinstance(source_conversation, dict) or not isinstance(source_conversation.get("metadata"), dict) or source_conversation["metadata"].get("zdr_enabled"):
                continue
            filtered_convs[cid] = data
        other_convs = filtered_convs

        if not other_convs:
            return {"context": "", "used_tokens": 0, "pieces": 0, "usage": {}}

        # 1b. Topic pre-filter (P5-T4, audit §12): scoring every stored SESSION
        # turn against the query's tokens and keeping only the ones that
        # overlap cuts down how much irrelevant history gets stuffed into the
        # extraction prompt. This is a pre-filter, not a search engine.
        #
        # Three classes of entry are ALWAYS eligible regardless of scoring:
        #   1. Document memories (index_document, sentinel turn == -1): their
        #      stored "topics" is just [f"document:{filename}"], derived from
        #      the filename alone, not the document's actual content -- it
        #      can't be trusted as a relevance signal. The extraction model
        #      inspects the document body itself to decide relevance, the
        #      same as it always has.
        #   2. Topicless session turns (empty/missing "topics", e.g.
        #      extract_topics returned [] on a timeout or error): there is
        #      nothing to score them against, so they are unjudgeable rather
        #      than irrelevant -- same fail-open rationale as documents.
        #   3. Everything, when NO session turn scores an overlap at all
        #      (quality floor): the filter can never return less context
        #      than before it existed.
        #
        # Filtered in a single pass over the ORIGINAL order (not
        # partition-then-concatenate): the char-budget cap below walks
        # reversed(memory_blocks) to keep the newest blocks, so re-ordering
        # entries here (e.g. appending all documents last) would make an old
        # entry look newest and let it evict a genuinely newer, topic-
        # matching turn under a tight budget.
        all_entries = [(cid, turn) for cid, data in other_convs.items() for turn in data["turns"]]
        scores = [score_topic_overlap(query, turn.get("topics")) for _, turn in all_entries]
        any_session_turn_scores = any(
            score > 0 for (_, turn), score in zip(all_entries, scores) if turn.get("turn") != -1
        )
        turns_to_use = [
            (cid, turn) for (cid, turn), score in zip(all_entries, scores)
            if turn.get("turn") == -1
            or not turn.get("topics")
            or not any_session_turn_scores
            or score > 0
        ]

        # Flatten into a prompt-friendly string. Summary entries (P5-T5
        # feature 3) get their own citation header naming how many turns they
        # compress, instead of a normal turn body -- everything else is
        # unchanged from before the summary tier existed.
        memory_blocks = [
            (
                f"[Memory from Chat: {cid} | Turn: {turn['turn']}: summary of {turn['turns_compressed']} earlier turns]\n{turn['summary']}"
                if turn.get("kind") == "summary"
                else f"[Memory from Chat: {cid} | Turn: {turn['turn']}]\n{turn['memory']}"
            )
            for cid, turn in turns_to_use
        ]

        if not memory_blocks:
            return {"context": "", "used_tokens": 0, "pieces": 0, "usage": {}}

        # Limit the history string to roughly max_tokens or 60,000 chars to avoid overloading standard OpenRouter context.
        # chars≈tokens×4 heuristic; effective cap never exceeds the legacy 60k ceiling.
        # Trim on block boundaries (not raw chars) so every kept block keeps its
        # "[Memory from Chat: ...]" citation header intact and citable.
        cap = min(60000, max_tokens * 4) if max_tokens else 60000
        memory_text = "\n\n".join(memory_blocks)
        if len(memory_text) > cap:
            kept = []
            total = 0
            for block in reversed(memory_blocks):
                added = len(block) + (2 if kept else 0)  # account for "\n\n" join
                if total + added > cap:
                    break
                kept.append(block)
                total += added
            if not kept:
                # Single newest block alone exceeds cap: keep it but slice its
                # content tail and re-attach the header so it stays citable.
                # This can overshoot `cap` by the header's length; acceptable.
                header, _, content = memory_blocks[-1].partition("\n")
                header_line = header + "\n"
                tail_budget = max(cap - len(header_line) - 1, 0)
                # content[-0:] would slice the WHOLE string, defeating the cap; guard it.
                tail = content[-tail_budget:] if tail_budget > 0 else ""
                kept = [header_line + "…" + tail]
            memory_text = "\n\n".join(reversed(kept))
            
        # 2. Reasoning Extraction step
        no_context_token = "NO_RELEVANT_CONTEXT"
        extract_prompt = (
            "You are the memory retrieval engine (PageIndex) for an AI Advisory Board.\n"
            "Given a user query, extract all relevant insights, facts, or context from the provided User Memory Logs.\n"
            f'If nothing is relevant, say EXACTLY "{no_context_token}".\n'
            "When extracting, YOU MUST preserve the exact `[Memory from Chat: X]` citation prefixes and be extremely concise.\n\n"
            f"USER QUERY: {query}\n\n"
            f"USER MEMORY LOGS:\n{memory_text}"
        )

        try:
            logger.info("[RAG] Calling PageIndex reasoner (%s) for query=%r...", UTILITY_MODEL, query)
            response = await query_model(
                UTILITY_MODEL,
                [{"role": "user", "content": extract_prompt}],
                timeout=15.0,
                zdr_enabled=zdr_enabled,
            )

            # Surface the extraction call's usage regardless of outcome (audit
            # §12): this call burns UTILITY_MODEL tokens on every chat turn
            # and its cost was previously invisible to turn_cost/session
            # budget. Empty/None-safe: query_model always returns a dict with
            # a "usage" key, but tolerate a bare mock/None in tests too.
            usage = (response or {}).get("usage") or {}

            extracted_context = response.get("content", "").strip() if response else "NO_RELEVANT_CONTEXT"
            if "NO_RELEVANT_CONTEXT" in extracted_context or not extracted_context:
                logger.info("[RAG] PageIndex reasoned no context was relevant.")
                return {"context": "", "used_tokens": 0, "pieces": 0, "usage": usage}

            logger.info("[RAG] PageIndex retrieved context successfully.")
            return {
                "context": extracted_context,
                "used_tokens": int(len(extracted_context.split()) * 1.3),
                "pieces": extracted_context.count("[Memory from"),
                "usage": usage,
            }
        except Exception as e:
            logger.error("[RAG] Reasoning Engine Error: %s", e, exc_info=True)
            return {"context": "", "used_tokens": 0, "pieces": 0, "usage": {}}
