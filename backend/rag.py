import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os

from .hybrid_retrieval import HybridRetriever
from .logger import logger
from .config import RAG_SETTINGS

class CouncilRAG:
    def __init__(self, persist_path: str = "./data/chroma_db"):
        """
        Initialize the Council RAG system with ChromaDB.
        """
        try:
            # Ensure the directory exists
            os.makedirs(persist_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=persist_path)
            
            # Use a lightweight, local embedding model
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get the collection with cosine distance
            self.collection = self.client.get_or_create_collection(
                name="council_context",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(self.collection)
            
            self.enabled = True
            logger.info("[RAG] Initialized successfully with hybrid retrieval")
        except Exception as e:
            logger.exception("[RAG] WARNING: Failed to initialize: %s", e)
            logger.info("[RAG] RAG features will be disabled, but application will continue to work")
            self.enabled = False
            self.collection = None
            self.hybrid_retriever = None
    
    def refresh_hybrid_index(self) -> None:
        """
        Convenience wrapper for refreshing BM25 index.
        Call this after backfilling or after a batch of new sessions.
        """
        if self.enabled and self.hybrid_retriever:
            self.hybrid_retriever.refresh_index()

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
        Index one council session with enhanced metadata.
        
        Args:
            conversation_id: Unique conversation identifier
            turn_index: Turn number in conversation
            user_question: Original user question
            stage1_results: Individual model responses
            stage2_results: Model rankings (unused in indexing, but kept for API consistency)
            stage3_result: Final synthesis
            topics: List of extracted topics (required)
            quality_metrics: Per-model quality metrics (required)
        """
        # Early return if RAG is disabled
        if not self.enabled:
            return
        
        from datetime import datetime
        import json
        
        timestamp = datetime.utcnow().isoformat()
        
        ids = []
        documents = []
        metadatas = []

        # Helper to format text
        def format_text(text: str) -> str:
            return f"Q: {user_question}\n\nA: {text}"

        topics_str = json.dumps(topics or [])

        # Stage 1: Individual Opinions
        for idx, res in enumerate(stage1_results):
            model = res['model']
            text = res['response']
            quality = quality_metrics.get(model, {})
            
            doc_id = f"{conversation_id}:turn:{turn_index}:opinion:{idx}:{model}"
            ids.append(doc_id)
            documents.append(format_text(text))
            metadatas.append({
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "stage": "opinion",
                "model": model,
                "topics": topics_str,
                "avg_rank": quality.get("avg_rank", 999.0),
                "consensus_score": quality.get("consensus_score", 0.0),
                "timestamp": timestamp,
            })

        # Stage 3: Final Synthesis
        stage3_model = stage3_result.get('model', 'unknown')
        final_text = stage3_result.get('response', '')
        stage3_quality = quality_metrics.get(stage3_model, {})
        
        if final_text:
            doc_id = f"{conversation_id}:turn:{turn_index}:synthesis:{stage3_model}"
            ids.append(doc_id)
            documents.append(format_text(final_text))
            metadatas.append({
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "stage": "synthesis",
                "model": stage3_model,
                "topics": topics_str,
                "avg_rank": stage3_quality.get("avg_rank", 999.0),
                "consensus_score": stage3_quality.get("consensus_score", 0.0),
                "timestamp": timestamp,
            })

        # Upsert to ChromaDB
        if ids:
            logger.info(
                "[PHASE1] Indexing session conv=%s turn=%d docs=%d",
                conversation_id,
                turn_index,
                len(ids),
            )
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def retrieve(self, query: str, conversation_id: str, max_tokens: int = None) -> str:
        """
        Retrieve using hybrid BM25 plus dense approach.
        Returns formatted context string for Chairman.
        Backward compatible wrapper around retrieve_with_stats().
        """
        result = self.retrieve_with_stats(query, conversation_id, max_tokens)
        return result["context"]
    
    def retrieve_with_stats(self, query: str, conversation_id: str, max_tokens: int = None) -> Dict[str, Any]:
        """
        Retrieve with full stats for budget tracking.
        Returns {"context": str, "used_tokens": int, "pieces": int}
        """
        # Early return if RAG is disabled
        if not self.enabled:
            logger.info("[RAG] RAG disabled, returning empty context")
            return {"context": "", "used_tokens": 0, "pieces": 0}
        
        # Resolve budget from config
        if max_tokens is None:
            default_key = RAG_SETTINGS["default_preset"]
            requested = RAG_SETTINGS["presets"][default_key]["tokens"]
        else:
            requested = max_tokens
        
        actual_budget = min(requested, RAG_SETTINGS["absolute_max_tokens"])
        max_chunk_tokens = RAG_SETTINGS["max_chunk_tokens"]
        score_threshold = RAG_SETTINGS["score_threshold"]
        
        logger.info("[RAG] Budget: requested=%d, actual=%d", requested, actual_budget)
        logger.info("[RAG] Starting hybrid retrieval for query=%r conv=%s", query[:50], conversation_id)
        
        try:
            # Use hybrid retrieval
            logger.info("[RAG] Calling hybrid_retriever.retrieve()...")
            results = self.hybrid_retriever.retrieve(
                query=query,
                conversation_id=conversation_id,
                top_k=10,
            )
            logger.info("[RAG] Hybrid retriever returned %d results", len(results))

            logger.info(
                "[PHASE1] Hybrid RAG returned %d results for conv=%s",
                len(results),
                conversation_id,
            )

            # RRF scores are very small, so threshold needs to be low
            if results:
                scores = [float(r["score"]) for r in results]
                logger.debug("[RAG] RRF scores: min=%.6f, max=%.6f, scores=%s", 
                           min(scores), max(scores), [f"{s:.6f}" for s in scores[:5]])
            
            # Use threshold from config
            threshold = score_threshold
            
            filtered_chunks = []
            for res in results:
                score = float(res["score"])
                if score < threshold:
                    continue

                text = res.get("text") or ""
                meta = res.get("metadata") or {}

                filtered_chunks.append(
                    {
                        "id": res["id"],
                        "similarity": score,
                        "metadata": meta,
                        "text": text,
                    }
                )

            logger.info(
                "[PHASE1] Hybrid RAG chunks passing threshold=%d (threshold=%.4f)",
                len(filtered_chunks), threshold,
            )
            
            for i, chunk in enumerate(filtered_chunks[:3]):  # Log first 3 chunks for diagnostics
                text_preview = (chunk.get("text") or "")[:100]
                logger.debug(
                    "[RAG] Chunk %d: id=%s, score=%.4f, text_len=%d, preview=%r",
                    i, chunk.get("id", "?")[:50], chunk.get("similarity", 0),
                    len(chunk.get("text") or ""), text_preview
                )

            # Build context with token budget
            formatted_parts: List[str] = []
            used_tokens = 0

            for chunk in filtered_chunks:
                text = chunk["text"]
                # Skip empty chunks
                if not text or not text.strip():
                    logger.warning("[RAG] Skipping chunk with empty text: %s", chunk.get("id", "?")[:50])
                    continue
                
                # Crude token estimate (words * 1.3)
                words = text.split()
                est_tokens = int(len(words) * 1.3)
                
                # Truncate large chunks to fit within per-chunk limit
                if est_tokens > max_chunk_tokens:
                    # Calculate how many words we can keep
                    max_words = int(max_chunk_tokens / 1.3)
                    truncated_text = " ".join(words[:max_words]) + "\n\n...(truncated)"
                    chunk = {**chunk, "text": truncated_text}  # Create modified copy
                    est_tokens = max_chunk_tokens
                    logger.info("[RAG] Truncated chunk from %d to %d tokens", len(words), max_words)

                # Check if adding this chunk would exceed total budget
                # Quality floor: always include at least 1 chunk even if over budget
                if used_tokens + est_tokens > actual_budget and len(formatted_parts) >= 1:
                    logger.info("[RAG] Token budget reached (%d/%d), stopping", used_tokens, actual_budget)
                    break

                used_tokens += est_tokens
                formatted_parts.append(self._format_chunk(chunk))
                
                # After adding first chunk, if we're now over budget, stop
                if used_tokens > actual_budget:
                    logger.info("[RAG] Quality floor: included 1 chunk (%d tokens) despite budget", used_tokens)
                    break

            context = "\n\n".join(formatted_parts)
            logger.info(
                "[RAG] Budget: requested=%d, actual=%d, returned=%d, pieces=%d",
                requested, actual_budget, used_tokens, len(formatted_parts),
            )
            return {"context": context, "used_tokens": used_tokens, "pieces": len(formatted_parts)}
        except Exception as e:
            logger.error("[RAG] Error in retrieve: %s", e, exc_info=True)
            return {"context": "", "used_tokens": 0, "pieces": 0}

    def _format_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Format a single chunk for the LLM context.
        """
        meta = chunk['metadata']
        # Remove the "Q: ... A: " prefix for the context block to save tokens/reduce repetition if desired,
        # OR keep it. The plan said "Prepend user question to indexed text". 
        # The prompt will see "Q: ... A: ...". This is fine.
        
        return f"""[Turn {meta['turn_index']} | Stage {meta['stage']} | Model: {meta['model']}]
{chunk['text']}"""
