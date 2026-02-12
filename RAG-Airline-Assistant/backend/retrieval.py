"""
backend/retrieval.py
--------------------
Two-stage retrieval pipeline:
  Stage 1 — Dense vector search (ChromaDB + e5-base-v2)
  Stage 2 — Cross-encoder reranking (bge-reranker-base)

Filters out do_not_cite chunks from final results.
"""
from __future__ import annotations

from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import settings


class Retriever:
    """
    Loads models once at startup (lazy-loaded on first use).
    Designed to be instantiated once and reused across requests.
    """

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._embedder = None
        self._reranker = None

    # ------------------------------------------------------------------ #
    #  Lazy model loading
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.embed_model)
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.reranker_model)
        return self._reranker

    # ------------------------------------------------------------------ #
    #  Stage 1: Dense retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        airline_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed query and fetch top_k nearest chunks from ChromaDB.
        Optionally filter by airline name (case-insensitive match).
        """
        k = top_k or settings.retrieval_top_k

        # e5 models perform best with query prefix
        prefixed_query = f"query: {query}"
        q_emb = self.embedder.encode(
            [prefixed_query], normalize_embeddings=True
        ).tolist()[0]

        # Build optional where clause for airline filter
        # FIXED: Normalize to lowercase for case-insensitive matching
        where = None
        if airline_filter:
            airline_normalized = airline_filter.strip().lower()
            where = {"airline": {"$eq": airline_normalized}}

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],  # ids returned automatically
            where=where,
        )

        items: List[Dict[str, Any]] = []
        for i in range(len(res["ids"][0])):
            items.append(
                {
                    "id": res["ids"][0][i],
                    "doc": res["documents"][0][i],
                    "meta": res["metadatas"][0][i],
                    "distance": float(res["distances"][0][i]),
                }
            )
        return items

    # ------------------------------------------------------------------ #
    #  Stage 2: Cross-encoder reranking
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int | None = None,
        exclude_do_not_cite: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Score each candidate with the cross-encoder and return top_n.
        Chunks marked do_not_cite=True are excluded from final results
        (they may still inform internal logic but won't be shown to users).
        """
        n = top_n or settings.rerank_top_n

        if not candidates:
            return []

        # FIXED: Increased from 350 to 500 chars for better reranking context
        pairs = [(query, c["doc"][:500]) for c in candidates]
        scores = self.reranker.predict(pairs).tolist()

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Filter out internal/meta chunks not meant for citation
        if exclude_do_not_cite:
            candidates = [
                c for c in candidates
                if not c["meta"].get("do_not_cite", False)
            ]

        return candidates[:n]

    # ------------------------------------------------------------------ #
    #  Combined convenience method
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        airline_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: retrieve → rerank → return top results.
        This is what main.py should call.
        """
        candidates = self.retrieve(query, airline_filter=airline_filter)
        return self.rerank(query, candidates)


# REMOVED: Global variable instantiation (was causing confusion)
# retriever = Retriever()
# The retriever is instantiated in main.py where it's actually used