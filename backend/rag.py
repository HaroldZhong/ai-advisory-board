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

_WORD_RE = re.compile(r"[a-z0-9]+")


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

            # Ensure conversation exists in store
            if conversation_id not in self.store:
                self.store[conversation_id] = {
                    "folder_id": "root",
                    "turns": []
                }

            turns = self.store[conversation_id]["turns"]
            turn_index = max((entry.get("turn", -1) for entry in turns), default=-1) + 1

            # Only index the user's question and the final synthesized answer to save context tokens for reasoning retrieve
            turn_memory = {
                "turn": turn_index,
                "topics": topics,
                "memory": f"Q: {user_question}\nA: {final_text}"
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

            if conversation_id not in self.store:
                self.store[conversation_id] = {"folder_id": "root", "turns": []}

            turns = self.store[conversation_id]["turns"]
            turn_index = max((entry.get("turn", -1) for entry in turns), default=-1) + 1

            turn_memory = {
                "turn": turn_index,
                "topics": topics,
                "memory": f"Q: {question}\nA: {answer}",
            }
            turns.append(turn_memory)
            usage = await self._maybe_compress_oldest_half(conversation_id)
            self._save_store()
            if self.enabled:
                logger.info("[RAG] Indexed chat turn %d for conv=%s into PageIndex", turn_index, conversation_id)
            return usage

    async def _maybe_compress_oldest_half(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """P5-T5 feature 3: bounded per-conversation memory growth.

        Once a conversation's turns list exceeds SUMMARY_COMPRESSION_THRESHOLD,
        compress the OLDEST half into a single dense summary entry via
        UTILITY_MODEL. On LLM failure (None), skip compression entirely -- no
        data loss, just unbounded growth for this conversation until the next
        successful attempt. Callers are responsible for _save_store()/logging
        the caller-specific context; this only mutates self.store in memory
        and returns the summarization call's usage (or None).
        """
        turns = self.store[conversation_id]["turns"]
        if len(turns) <= SUMMARY_COMPRESSION_THRESHOLD:
            return None

        half = (len(turns) + 1) // 2
        oldest = turns[:half]
        rest = turns[half:]

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
            logger.warning("[RAG] Summary compression returned no content for conv=%s; skipping compression", conversation_id)
            return None

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
        }
        # Codex round 3: this write-back replaces the whole turns list with a
        # snapshot taken before the query_model await above. Callers
        # (index_session/index_chat_turn) hold this conversation's write lock
        # for their entire body, so no other writer can have appended to
        # `turns` during that await -- `rest` is still accurate. The kept
        # `rest` entries' own "turn" numbers are untouched, so the next
        # index_chat_turn's max()+1 scan still yields a correct, monotonic
        # number regardless of what number this summary entry carries.
        self.store[conversation_id]["turns"] = [summary_entry] + rest
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
                "topics": [f"document:{filename}"],
                "memory": f"[Uploaded Document: {filename}]\n{truncated}"
            }
            self.store[conversation_id]["turns"].append(doc_memory)
            self._save_store()
            if self.enabled:
                logger.info("[RAG] Indexed document '%s' (%d chars) for conv=%s", filename, len(truncated), conversation_id)

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
