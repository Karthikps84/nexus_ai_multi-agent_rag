"""
==========================================================
Cache Manager — exact-string + semantic (cosine) layers
==========================================================
Architecture:
  Layer 0: embedding cache  – text → vector, TTL-based, exact-text key
  Layer 1: retrieval cache  – query → RetrievalArtifacts, exact then semantic
  Layer 2: response cache   – question → answer dict, exact then semantic

Semantic layer uses a lightweight in-process vector index (cosine similarity
over the currently live TTLCache keys) so paraphrased questions hit the cache
without an extra LLM call. Linear scan is fast enough for cache sizes in the
hundreds; anything >512 entries would need an ANN index instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from cachetools import TTLCache


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@dataclass(slots=True)
class CacheConfig:
    response_ttl_seconds: int
    retrieval_ttl_seconds: int
    embedding_ttl_seconds: int
    response_max_entries: int
    retrieval_max_entries: int
    embedding_max_entries: int
    # Cosine threshold above which a cached entry counts as a semantic match.
    # 0.93 catches paraphrases ("total revenue?" / "what was the revenue?")
    # without merging genuinely distinct questions.
    semantic_similarity_threshold: float = 0.93
    # Max entries to linear-scan per lookup. Cheap at hundreds of entries.
    semantic_max_scan: int = 512


class _SemanticIndex:
    """
    Parallel vector index over a TTLCache's keys. Self-cleaning: entries
    whose backing TTLCache key has expired or been evicted are dropped lazily
    on the next lookup, so this never drifts out of sync with the cache.
    """

    def __init__(self, threshold: float, max_scan: int):
        self._threshold = threshold
        self._max_scan = max_scan
        self._entries: list[dict] = []  # [{key, group, vector}]

    def add(self, key: str, group: str, vector: list[float]) -> None:
        self._entries.append({
            "key": key,
            "group": group,
            "vector": np.asarray(vector, dtype=np.float32),
        })
        if len(self._entries) > self._max_scan:
            self._entries = self._entries[-self._max_scan:]

    def find(
        self,
        group: str,
        vector: list[float],
        is_alive: Callable[[str], bool],
    ) -> Optional[tuple[str, float]]:
        # Lazy eviction of expired keys
        self._entries = [e for e in self._entries if is_alive(e["key"])]
        candidates = [e for e in self._entries if e["group"] == group]
        if not candidates:
            return None

        q = np.asarray(vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        best_key, best_score = None, -1.0
        for entry in candidates:
            v = entry["vector"]
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(q_norm, v_norm))
            if score > best_score:
                best_score, best_key = score, entry["key"]

        if best_key is not None and best_score >= self._threshold:
            return best_key, round(best_score, 4)
        return None


class CacheManager:
    def __init__(self, config: CacheConfig):
        self._config = config
        self.response_cache = TTLCache(
            maxsize=config.response_max_entries,
            ttl=config.response_ttl_seconds,
        )
        self.retrieval_cache = TTLCache(
            maxsize=config.retrieval_max_entries,
            ttl=config.retrieval_ttl_seconds,
        )
        self.embedding_cache = TTLCache(
            maxsize=config.embedding_max_entries,
            ttl=config.embedding_ttl_seconds,
        )
        self._response_idx = _SemanticIndex(
            config.semantic_similarity_threshold, config.semantic_max_scan
        )
        self._retrieval_idx = _SemanticIndex(
            config.semantic_similarity_threshold, config.semantic_max_scan
        )
        self._embed_fn: Optional[Callable[[str], list[float]]] = None

    def invalidate_documents(self) -> None:
        """Drop cached answers/retrievals; call when the document set changes."""
        self.response_cache.clear()
        self.retrieval_cache.clear()
        self._response_idx = _SemanticIndex(
            self._config.semantic_similarity_threshold, self._config.semantic_max_scan
        )
        self._retrieval_idx = _SemanticIndex(
            self._config.semantic_similarity_threshold, self._config.semantic_max_scan
        )

    def bind_embedder(self, embed_fn: Callable[[str], list[float]]) -> None:
        """
        Call once at startup with a (text → vector) function. After binding,
        all get_*/set_* calls also populate/consult the semantic index so
        paraphrased queries hit the cache without an LLM call. If never
        called, the cache behaves as a plain exact-string store.
        """
        self._embed_fn = embed_fn

    @property
    def semantic_enabled(self) -> bool:
        return self._embed_fn is not None

    # ── key builders (public, used by callers to pre-compute storage keys) ──

    @staticmethod
    def response_key(
        question: str,
        route: str | None,
        files: list[str] | None,
    ) -> str:
        return f"{route or 'auto'}|{','.join(sorted(files or []))}|{_normalize(question)}"

    @staticmethod
    def retrieval_key(
        question: str,
        source_filter: list[str] | None = None,
        top_k: int | None = None,
        final_k: int | None = None,
    ) -> str:
        return f"{_normalize(question)}|{','.join(sorted(source_filter or []))}|{top_k}|{final_k}"

    # ── grouping helpers (defines semantic search scope) ──────────────────

    @staticmethod
    def _response_group(route: str | None, files: list[str] | None) -> str:
        return f"{route or 'auto'}|{','.join(sorted(files or []))}"

    @staticmethod
    def _retrieval_group(
        source_filter: list[str] | None, top_k, final_k
    ) -> str:
        return f"{','.join(sorted(source_filter or []))}|{top_k}|{final_k}"

    # ── response cache ─────────────────────────────────────────────────────

    def get_response(self, key: str) -> Any:
        """
        Exact-match lookup first (O(1)). If that misses and the semantic
        index is enabled, tries a cosine-similarity lookup using the
        question text embedded in `key`.
        Returns the cached value or None.
        """
        exact = self.response_cache.get(key)
        if exact is not None:
            return exact

        if not self.semantic_enabled:
            return None

        # Extract the question portion from the key
        parts = key.split("|", 2)
        if len(parts) < 3:
            return None
        route_part = parts[0]
        files_part = parts[1]
        question_text = parts[2]

        group = f"{route_part}|{files_part}"
        vector = self._embed_fn(question_text)
        hit = self._response_idx.find(
            group, vector, is_alive=lambda k: k in self.response_cache
        )
        if hit is None:
            return None
        matched_key, _sim = hit
        return self.response_cache.get(matched_key)

    def set_response(self, key: str, value: Any) -> None:
        self.response_cache[key] = value
        if self.semantic_enabled:
            parts = key.split("|", 2)
            if len(parts) == 3:
                route_part, files_part, question_text = parts
                group = f"{route_part}|{files_part}"
                vector = self._embed_fn(question_text)
                self._response_idx.add(key, group, vector)

    # ── retrieval cache ────────────────────────────────────────────────────

    def get_retrieval(self, key: str) -> Any:
        exact = self.retrieval_cache.get(key)
        if exact is not None:
            return exact

        if not self.semantic_enabled:
            return None

        # key = "{question}|{files}|{top_k}|{final_k}"
        parts = key.split("|", 3)
        if len(parts) < 4:
            return None
        question_text = parts[0]
        files_part = parts[1]
        top_k = parts[2]
        final_k = parts[3]

        group = f"{files_part}|{top_k}|{final_k}"
        vector = self._embed_fn(question_text)
        hit = self._retrieval_idx.find(
            group, vector, is_alive=lambda k: k in self.retrieval_cache
        )
        if hit is None:
            return None
        matched_key, _sim = hit
        return self.retrieval_cache.get(matched_key)

    def set_retrieval(self, key: str, value: Any) -> None:
        self.retrieval_cache[key] = value
        if self.semantic_enabled:
            parts = key.split("|", 3)
            if len(parts) == 4:
                question_text = parts[0]
                files_part = parts[1]
                top_k = parts[2]
                final_k = parts[3]
                group = f"{files_part}|{top_k}|{final_k}"
                vector = self._embed_fn(question_text)
                self._retrieval_idx.add(key, group, vector)

    # ── embedding cache (plain exact-text, always) ─────────────────────────

    def get_embedding(self, key: str) -> Any:
        return self.embedding_cache.get(key)

    def set_embedding(self, key: str, value: Any) -> None:
        self.embedding_cache[key] = value


class CachedEmbeddings:
    """Transparent embedding cache wrapper around the underlying client."""

    def __init__(self, embeddings, cache_manager: CacheManager):
        self._embeddings = embeddings
        self._cache = cache_manager

    def embed_query(self, text: str):
        key = f"query::{_normalize(text)}"
        cached = self._cache.get_embedding(key)
        if cached is not None:
            return cached
        value = self._embeddings.embed_query(text)
        self._cache.set_embedding(key, value)
        return value

    def embed_documents(self, texts: list[str]):
        vectors: list[Any] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_idx: list[int] = []

        for idx, text in enumerate(texts):
            key = f"doc::{_normalize(text)}"
            cached = self._cache.get_embedding(key)
            if cached is None:
                missing_texts.append(text)
                missing_idx.append(idx)
            else:
                vectors[idx] = cached

        if missing_texts:
            computed = self._embeddings.embed_documents(missing_texts)
            for idx, text, value in zip(missing_idx, missing_texts, computed):
                key = f"doc::{_normalize(text)}"
                self._cache.set_embedding(key, value)
                vectors[idx] = value

        return vectors

    def __getattr__(self, item):
        return getattr(self._embeddings, item)
