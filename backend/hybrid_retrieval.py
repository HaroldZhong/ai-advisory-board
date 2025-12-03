# backend/hybrid_retrieval.py

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np

from .logger import logger  # adapt to your logger import


class HybridRetriever:
    """
    Combine BM25 (lexical) with Chroma dense retrieval using reciprocal rank fusion.
    BM25 index is built from the entire collection. At scoring time we filter
    results to the current conversation to prevent cross conversation bleed.
    """

    def __init__(self, chroma_collection):
        self.collection = chroma_collection

        self.bm25_index = None
        self.doc_ids: List[str] = []
        self.id_to_doc: Dict[str, str] = {}
        self.id_to_metadata: Dict[str, Dict[str, Any]] = {}

    def _rebuild_bm25_index(self) -> None:
        """
        Internal helper to rebuild BM25 index from the full collection.
        Called on first use or via refresh_index.
        """
        all_docs = self.collection.get(
            include=["documents", "metadatas"]  # IDs are always returned, don't include in list
        )

        ids = all_docs.get("ids") or []
        documents = all_docs.get("documents") or []
        metadatas = all_docs.get("metadatas") or []

        if not ids or not documents:
            logger.info("[PHASE1] BM25 index rebuild skipped, no documents")
            self.bm25_index = None
            self.doc_ids = []
            self.id_to_doc = {}
            self.id_to_metadata = {}
            return

        self.doc_ids = ids
        self.id_to_doc = {
            doc_id: doc for doc_id, doc in zip(ids, documents)
        }
        self.id_to_metadata = {
            doc_id: meta for doc_id, meta in zip(ids, metadatas)
        }

        tokenized = [doc.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)

        logger.info(
            "[PHASE1] BM25 index rebuilt, docs=%d", len(self.doc_ids)
        )

    def refresh_index(self) -> None:
        """
        Explicitly refresh BM25 index.
        Call this after a batch of new sessions is indexed.
        """
        self._rebuild_bm25_index()

    def _ensure_index(self) -> None:
        if self.bm25_index is None:
            self._rebuild_bm25_index()

    def retrieve(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 10,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using reciprocal rank fusion.

        Returns a list of dicts:
        - id
        - score (combined)
        - metadata
        - text
        """
        logger.info("[HybridRetriever] Starting retrieve for query=%r conv=%s", query[:50], conversation_id)
        
        self._ensure_index()
        logger.info("[HybridRetriever] Index ensured")

        if not self.doc_ids or self.bm25_index is None:
            logger.info("[PHASE1] Hybrid retrieval skipped, empty index")
            return []

        logger.info("[HybridRetriever] Tokenizing query...")
        query_tokens = query.lower().split()
        logger.info("[HybridRetriever] Query has %d tokens", len(query_tokens))

        # 1. BM25 retrieval over global index, then filter by conversation
        logger.info("[HybridRetriever] Getting BM25 scores...")
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        logger.info("[HybridRetriever] BM25 scores computed, getting top indices...")
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: top_k * 2]
        logger.info("[HybridRetriever] Found %d BM25 top indices", len(bm25_top_indices))

        bm25_results: Dict[str, Any] = {}
        for rank, idx in enumerate(bm25_top_indices):
            doc_id = self.doc_ids[idx]
            score = bm25_scores[idx]

            if score <= 0:
                continue

            meta = self.id_to_metadata.get(doc_id, {})
            if meta.get("conversation_id") != conversation_id:
                continue

            bm25_results[doc_id] = (rank + 1, float(score))

        # 2. Dense retrieval from Chroma, already filtered by conversation_id
        dense = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,
            where={"conversation_id": conversation_id},
        )

        dense_result_dict: Dict[str, Any] = {}
        ids_list = dense.get("ids") or [[]]
        distances_list = dense.get("distances") or [[]]

        if ids_list and ids_list[0]:
            for rank, (doc_id, distance) in enumerate(
                zip(ids_list[0], distances_list[0])
            ):
                # Chroma uses cosine distance, smaller is closer
                similarity = 1.0 - float(distance)
                dense_result_dict[doc_id] = (rank + 1, similarity)

        # 3. Reciprocal rank fusion
        k_const = 60.0
        combined_scores: Dict[str, float] = {}

        all_ids = set(bm25_results.keys()) | set(dense_result_dict.keys())
        for doc_id in all_ids:
            score = 0.0

            if doc_id in bm25_results:
                rank, _ = bm25_results[doc_id]
                score += bm25_weight / (k_const + rank)

            if doc_id in dense_result_dict:
                rank, _ = dense_result_dict[doc_id]
                score += dense_weight / (k_const + rank)

            combined_scores[doc_id] = score

        if not combined_scores:
            logger.info("[PHASE1] Hybrid retrieval found no candidates")
            return []

        # Sort by fused score and take top_k
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda d: combined_scores[d],
            reverse=True,
        )[:top_k]

        # 4. Fetch documents and metadata once, then map by id
        full = self.collection.get(
            ids=sorted_ids,
            include=["documents", "metadatas"],  # IDs already provided, don't include in list
        )

        fetched_ids = full.get("ids") or []
        fetched_docs = full.get("documents") or []
        fetched_metas = full.get("metadatas") or []

        doc_map: Dict[str, Any] = {}
        for doc_id, meta, doc in zip(
            fetched_ids, fetched_metas, fetched_docs
        ):
            doc_map[doc_id] = (meta or {}, doc or "")

        results: List[Dict[str, Any]] = []
        for doc_id in sorted_ids:
            meta, doc = doc_map.get(doc_id, ({}, ""))
            results.append(
                {
                    "id": doc_id,
                    "score": combined_scores[doc_id],
                    "metadata": meta,
                    "text": doc,
                }
            )

        logger.info(
            "[PHASE1] Hybrid retrieval, query=%r, conv=%s, candidates=%d, returned=%d",
            query,
            conversation_id,
            len(all_ids),
            len(results),
        )

        return results
