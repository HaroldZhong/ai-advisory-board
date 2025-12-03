import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os

from .hybrid_retrieval import HybridRetriever
from .logger import logger

# RAG Configuration
RAG_SIM_THRESHOLD = 0.15  # Lowered threshold for better recall (was 0.3)
RAG_MAX_TOKENS = 3000    # Increased context window

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

    def retrieve(self, query: str, conversation_id: str) -> str:
        """
        Retrieve using hybrid BM25 plus dense approach.
        Returns formatted context string for Chairman.
        """
        # Early return if RAG is disabled
        if not self.enabled:
            logger.info("[RAG] RAG disabled, returning empty context")
            return ""
        
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

            # RRF scores are small, so threshold is low and empirical
            threshold = 0.01
            
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
                "[PHASE1] Hybrid RAG chunks passing threshold=%d",
                len(filtered_chunks),
            )

            # Build context with token budget
            formatted_parts: List[str] = []
            used_tokens = 0

            for chunk in filtered_chunks:
                text = chunk["text"]
                # crude token estimate
                est_tokens = int(len(text.split()) * 1.3)

                if used_tokens + est_tokens > RAG_MAX_TOKENS:
                    break

                used_tokens += est_tokens
                formatted_parts.append(self._format_chunk(chunk))

            context = "\n\n".join(formatted_parts)
            logger.info(
                "[PHASE1] Hybrid RAG context tokens=%d, pieces=%d",
                used_tokens,
                len(formatted_parts),
            )
            logger.info("[RAG] Retrieve completed successfully, returning %d chars", len(context))
            return context
        except Exception as e:
            logger.error("[RAG] Error in retrieve: %s", e, exc_info=True)
            return ""

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
