"""
backend/retrieval.py
--------------------
Two-stage retrieval pipeline:
  Stage 1 — Dense vector search (ChromaDB + e5-base-v2)
  Stage 2 — Cross-encoder reranking (bge-reranker-base)

Fixes (NO retrieval-quality change):
- Warmup to avoid first-user-query latency spikes
- Exact-match caches:
  * query embedding cache
  * reranker pair cache (query, doc[:500])
- Optional torch thread tuning via env TORCH_NUM_THREADS (speed only)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import threading
import os

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import settings


class _LRUCache:
    """Thread-safe LRU cache (exact-match only; safe for correctness)."""

    def __init__(self, maxsize: int = 2048):
        self.maxsize = maxsize
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


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

        self._emb_cache = _LRUCache(maxsize=2048)       # key: "query: <q>" -> embedding
        self._rerank_cache = _LRUCache(maxsize=10000)   # key: (q, doc500) -> score

        self._torch_threads = int(os.getenv("TORCH_NUM_THREADS", "0"))

    # -----------------------------
    # Lazy model loading
    # -----------------------------
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
            self._apply_torch_threads()
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.reranker_model)
            self._apply_torch_threads()
        return self._reranker

    def _apply_torch_threads(self) -> None:
        if self._torch_threads > 0:
            try:
                import torch  # type: ignore
                torch.set_num_threads(self._torch_threads)
            except Exception:
                pass

    # -----------------------------
    # Warmup (speed only)
    # -----------------------------
    def warmup(self) -> None:
        """
        Avoids first-query latency spikes.
        No effect on outputs.
        """
        try:
            _ = self._embed_query("warmup")
            _ = self.reranker.predict([("warmup", "warmup")])
        except Exception:
            pass

    # -----------------------------
    # Stage 1: Dense retrieval
    # -----------------------------
    def _embed_query(self, query: str) -> List[float]:
        prefixed_query = f"query: {query}"

        cached = self._emb_cache.get(prefixed_query)
        if cached is not None:
            return cached

        emb = self.embedder.encode([prefixed_query], normalize_embeddings=True).tolist()[0]
        self._emb_cache.set(prefixed_query, emb)
        return emb

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        airline_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or settings.retrieval_top_k
        q_emb = self._embed_query(query)

        where = None
        if airline_filter:
            airline_normalized = airline_filter.strip().lower()
            where = {"airline": {"$eq": airline_normalized}}

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
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

    # -----------------------------
    # Stage 2: Cross-encoder reranking
    # -----------------------------
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int | None = None,
        exclude_do_not_cite: bool = True,
    ) -> List[Dict[str, Any]]:
        n = top_n or settings.rerank_top_n
        if not candidates:
            return []

        # Keep same rerank context length (500) => no quality reduction
        doc500_list = [c["doc"][:500] for c in candidates]

        scores: List[Optional[float]] = [None] * len(candidates)
        uncached_pairs: List[Tuple[str, str]] = []
        uncached_idx: List[int] = []

        for i, doc500 in enumerate(doc500_list):
            key = (query, doc500)
            cached = self._rerank_cache.get(key)
            if cached is not None:
                scores[i] = float(cached)
            else:
                uncached_pairs.append((query, doc500))
                uncached_idx.append(i)

        if uncached_pairs:
            new_scores = self.reranker.predict(uncached_pairs).tolist()
            for idx, s in zip(uncached_idx, new_scores):
                scores[idx] = float(s)
                self._rerank_cache.set((query, doc500_list[idx]), float(s))

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s if s is not None else 0.0)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        if exclude_do_not_cite:
            candidates = [c for c in candidates if not c["meta"].get("do_not_cite", False)]

        return candidates[:n]

    # -----------------------------
    # Combined pipeline
    # -----------------------------
    def search(self, query: str, airline_filter: str | None = None) -> List[Dict[str, Any]]:
        candidates = self.retrieve(query, airline_filter=airline_filter)
        return self.rerank(query, candidates)