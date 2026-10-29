from __future__ import annotations

import time
from dataclasses import dataclass

import config
from retriever.hybrid_search import (
    Reranker,
    build_bm25_index,
    mmr_filter,
    reciprocal_rank_fusion,
    rewrite_query,
)
from utils.logger import get_logger

logger = get_logger("retrieval_pipeline")


@dataclass(slots=True)
class RetrievalArtifacts:
    docs: list
    max_score: float
    query: str
    dense_latency_ms: float
    fusion_latency_ms: float
    rerank_latency_ms: float
    cache_hit: bool = False


class DenseRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int, source_filter: list | None = None):
        where = {"source_file": {"$in": source_filter}} if source_filter else None
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k, filter=where)


class SparseRetriever:
    def __init__(self):
        self._bm25 = None
        self._corpus = []

    def index(self, chunks: list):
        self._bm25, self._corpus = build_bm25_index(chunks)

    @property
    def available(self) -> bool:
        return self._bm25 is not None and bool(self._corpus)

    def search(self, query: str, top_k: int) -> list:
        if not self.available:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._corpus[i] for i in top_idx]


class HybridFusion:
    @staticmethod
    def merge(dense_scored: list, sparse_docs: list, top_k: int) -> list:
        if sparse_docs:
            return reciprocal_rank_fusion(dense_scored, sparse_docs)[:top_k]
        if config.USE_MMR:
            return mmr_filter(dense_scored, k=top_k)
        return [doc for doc, _ in dense_scored][:top_k]


class CrossEncoderReranker:
    def __init__(self):
        self._reranker = Reranker(enabled=config.ENABLE_RERANK)

    @property
    def available(self) -> bool:
        return self._reranker.available

    def rerank(self, query: str, docs: list) -> list:
        return self._reranker.rerank(query, docs)


class ContextBuilder:
    @staticmethod
    def build(docs: list) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
            for d in docs
        )


class ContextCompressor:
    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    def compress(self, docs: list) -> list:
        total = 0
        kept = []
        for doc in docs:
            size = len(doc.page_content)
            if kept and total + size > self.max_chars:
                break
            kept.append(doc)
            total += size
        return kept


class RetrievalPipeline:
    def __init__(self, vs_manager, llm=None, cache_manager=None):
        self.vs_manager = vs_manager
        self.llm = llm
        self.cache_manager = cache_manager
        self.dense = DenseRetriever(vs_manager.vectorstore)
        self.sparse = SparseRetriever()
        self.fusion = HybridFusion()
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder()
        self.context_compressor = ContextCompressor(max_chars=config.RETRIEVAL_CONTEXT_CHAR_BUDGET)

    def bind_vectorstore(self):
        self.dense = DenseRetriever(self.vs_manager.vectorstore)

    def index_chunks(self, chunks: list):
        self.sparse.index(chunks)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        final_k: int | None = None,
        source_filter: list | None = None,
        history: list | None = None,
        use_query_rewrite: bool = False,
    ) -> RetrievalArtifacts:
        top_k = top_k or config.SIMILARITY_TOP_K
        final_k = final_k or config.FINAL_TOP_K

        query = question
        if use_query_rewrite and self.llm is not None and history:
            query = rewrite_query(self.llm, question, history)

        cache_hit = False
        cache_key = None
        if self.cache_manager is not None:
            cache_key = self.cache_manager.retrieval_key(query, source_filter, top_k, final_k)
            cached = self.cache_manager.get_retrieval(cache_key)
            if cached is not None:
                cached.cache_hit = True
                return cached

        t0 = time.time()
        dense_scored = self.dense.search(query, top_k=top_k, source_filter=source_filter)
        dense_latency_ms = round((time.time() - t0) * 1000, 1)
        max_score = max((score for _, score in dense_scored), default=0.0)

        t0 = time.time()
        sparse_docs = []
        if self.sparse.available and config.ENABLE_HYBRID_SEARCH and not source_filter:
            sparse_docs = self.sparse.search(query, top_k=top_k)
        docs = self.fusion.merge(dense_scored, sparse_docs, top_k=top_k)
        fusion_latency_ms = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        if self.reranker.available and docs:
            docs = self.reranker.rerank(query, docs)
        rerank_latency_ms = round((time.time() - t0) * 1000, 1)

        docs = self.context_compressor.compress(docs[:final_k])
        artifacts = RetrievalArtifacts(
            docs=docs,
            max_score=max_score,
            query=query,
            dense_latency_ms=dense_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            cache_hit=cache_hit,
        )
        if self.cache_manager is not None and cache_key is not None:
            self.cache_manager.set_retrieval(cache_key, artifacts)

        logger.info(
            "retrieve query=%r dense=%sms fusion=%sms rerank=%sms docs=%s",
            query,
            dense_latency_ms,
            fusion_latency_ms,
            rerank_latency_ms,
            len(docs),
        )
        return artifacts

