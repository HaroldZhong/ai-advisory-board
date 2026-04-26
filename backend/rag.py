import json
import os
import asyncio
from typing import List, Dict, Any, Optional

from .logger import logger
from .config import RAG_SETTINGS
from .openrouter import query_model

class CouncilRAG:
    """
    Reasoning-based RAG ("PageIndex") system.
    Replaces ChromaDB vector embeddings with LLM-reasoning cross-folder retrieval.
    """
    def __init__(self, persist_path: str = "./data"):
        try:
            os.makedirs(persist_path, exist_ok=True)
            self.index_file = os.path.join(persist_path, "pageindex_memory.json")
            
            # Load existing JSON index
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.store = json.load(f)
            else:
                self.store = {}
                
            self.enabled = True
            logger.info("[RAG] Initialized PageIndex Reasoning RAG successfully")
        except Exception as e:
            logger.exception("[RAG] WARNING: Failed to initialize: %s", e)
            self.enabled = False
            self.store = {}

    def _save_store(self):
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.store, f, indent=2)
        except Exception as e:
            logger.error("[RAG] Failed to save PageIndex store: %s", e)

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
            logger.info("[RAG] Purged PageIndex memories for conversation %s", conversation_id)
            return True
        return False
        
    def update_conversation_folder(self, conversation_id: str, new_folder_id: str):
        """
        Update the folder routing in the reasoning PageIndex for a conversation.
        """
        if conversation_id in self.store:
            self.store[conversation_id]["folder_id"] = new_folder_id
            self._save_store()
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
            logger.info("[PHASE1] Indexed turn %d for conv=%s into PageIndex", turn_index, conversation_id)

    def index_document(self, conversation_id: str, filename: str, text: str, max_chars: int = 8000):
        """
        Index extracted document text into the PageIndex store.
        Truncates to max_chars to keep the reasoning store manageable.
        """
        if not self.enabled or not text.strip():
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
    ) -> str:
        """Async version of retrieve to support the litellm calls properly."""
        result = await self.retrieve_with_stats_async(
            query,
            conversation_id,
            max_tokens,
            zdr_enabled=zdr_enabled,
        )
        return result["context"]
        
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
            return {"context": "", "used_tokens": 0, "pieces": 0}
            
        # 1. Gather all memories EXCEPT the current conversation
        # This prevents the RAG from repeating the short-term history which the chat already has.
        other_convs = {
            cid: data for cid, data in self.store.items() 
            if cid != conversation_id
        }
        
        if not other_convs:
            return {"context": "", "used_tokens": 0, "pieces": 0}
            
        # Flatten into a prompt-friendly string
        memory_blocks = []
        for cid, data in other_convs.items():
            for turn in data["turns"]:
                memory_blocks.append(f"[Memory from Chat: {cid} | Turn: {turn['turn']}]\n{turn['memory']}")
                
        if not memory_blocks:
            return {"context": "", "used_tokens": 0, "pieces": 0}

        # Limit the history string generically to roughly max_tokens or 60,000 chars to avoid overloading standard OpenRouter context
        memory_text = "\n\n".join(memory_blocks)
        if len(memory_text) > 60000:
            memory_text = memory_text[-60000:]
            
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
            logger.info("[RAG] Calling PageIndex reasoner (gemini-2.5-flash) for query=%r...", query)
            response = await query_model(
                "google/gemini-2.5-flash",
                [{"role": "user", "content": extract_prompt}],
                timeout=15.0,
                zdr_enabled=zdr_enabled,
            )
            
            extracted_context = response.get("content", "").strip() if response else "NO_RELEVANT_CONTEXT"
            if "NO_RELEVANT_CONTEXT" in extracted_context or not extracted_context:
                logger.info("[RAG] PageIndex reasoned no context was relevant.")
                return {"context": "", "used_tokens": 0, "pieces": 0}
                
            logger.info("[RAG] PageIndex retrieved context successfully.")
            return {
                "context": extracted_context, 
                "used_tokens": int(len(extracted_context.split()) * 1.3), 
                "pieces": extracted_context.count("[Memory from")
            }
        except Exception as e:
            logger.error("[RAG] Reasoning Engine Error: %s", e, exc_info=True)
            return {"context": "", "used_tokens": 0, "pieces": 0}
