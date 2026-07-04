import json
import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from .logger import logger
from .config import RAG_SETTINGS, UTILITY_MODEL
from .openrouter import query_model
from .app_paths import get_pageindex_dir, write_text_atomic


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
        try:
            if persist_path is None:
                persist_path = str(get_pageindex_dir())
            os.makedirs(persist_path, exist_ok=True)
            self.index_file = os.path.join(persist_path, "pageindex_memory.json")
            
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
            self.cleanup_zdr_conversations()

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

    def _backup_corrupt_index(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        backup_path = f"{self.index_file}.corrupt-{timestamp}"
        os.replace(self.index_file, backup_path)
        return backup_path

    def _save_store(self):
        if not self.enabled:
            return
        try:
            write_text_atomic(Path(self.index_file), json.dumps(self.store, indent=2))
        except (OSError, RuntimeError) as e:
            logger.exception(
                "[RAG] Failed to persist PageIndex store; disabling persistence for this process: %s",
                e,
            )
            self.enabled = False

    def refresh_hybrid_index(self) -> None:
        """Legacy compatibility method. No longer needed for reasoning RAG."""
        pass

    def delete_conversation_memories(self, conversation_id: str) -> bool:
        """
        Remove a conversation's memories from the global PageIndex store.
        """
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

    def index_session(
        self, 
        conversation_id: str, 
        turn_index: int, 
        user_question: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        stage3_result: Dict[str, Any],
        topics: List[str],
        quality_metrics: Dict[str, Dict[str, float]],
    ):
        """
        Index one council session as chronological memory blocks.
        """
        if not self.enabled:
            return

        # ZDR write barrier: never index a conversation whose CURRENT metadata
        # says ZDR. Synchronous check-then-write (no await window) closes the
        # runtime-flip race for good; flips after the write are handled by the
        # purge in update_conversation. Per-message ZDR (request flag, not
        # visible in metadata) is enforced by the pipeline-level guards.
        conversation = get_conversation(conversation_id)
        if not isinstance(conversation, dict) or not isinstance(conversation.get("metadata"), dict) or conversation["metadata"].get("zdr_enabled"):
            logger.info("[RAG] Skipping index for %s (ZDR or unreadable)", conversation_id)
            return

        # Ensure conversation exists in store
        if conversation_id not in self.store:
            self.store[conversation_id] = {
                "folder_id": "root", 
                "turns": []
            }
            
        stage3_model = stage3_result.get('model', 'unknown')
        final_text = stage3_result.get('response', '')
        
        # Only index the user's question and the final synthesized answer to save context tokens for reasoning retrieve
        if final_text:
            turn_memory = {
                "turn": turn_index,
                "topics": topics,
                "memory": f"Q: {user_question}\nA: {final_text}"
            }
            self.store[conversation_id]["turns"].append(turn_memory)
            self._save_store()
            if self.enabled:
                logger.info("[PHASE1] Indexed turn %d for conv=%s into PageIndex", turn_index, conversation_id)

    def index_document(self, conversation_id: str, filename: str, text: str, max_chars: int = 8000):
        """
        Index extracted document text into the PageIndex store.
        Truncates to max_chars to keep the reasoning store manageable.
        """
        if not self.enabled or not text.strip():
            return

        # ZDR write barrier: never index a conversation whose CURRENT metadata
        # says ZDR. Synchronous check-then-write (no await window) closes the
        # runtime-flip race for good; flips after the write are handled by the
        # purge in update_conversation. Per-message ZDR (request flag, not
        # visible in metadata) is enforced by the pipeline-level guards.
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

        # Flatten into a prompt-friendly string
        memory_blocks = []
        for cid, data in other_convs.items():
            for turn in data["turns"]:
                memory_blocks.append(f"[Memory from Chat: {cid} | Turn: {turn['turn']}]\n{turn['memory']}")

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
