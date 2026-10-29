"""
==========================================================
Hybrid retrieval — Dense (Chroma) + BM25 + RRF fusion + MMR
==========================================================
Extracted as standalone functions so the RetrievalAgent can
call them, and so they're independently testable/reusable.
"""

from __future__ import annotations

import config
from utils.helpers import content_key
from utils.logger import get_logger

logger = get_logger("hybrid_search")

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from flashrank import Ranker, RerankRequest
    _HAS_FLASHRANK = True
except ImportError:
    _HAS_FLASHRANK = False


def build_bm25_index(chunks: list):
    if not _HAS_BM25 or not chunks:
        return None, []
    tokenized = [c.page_content.lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def mmr_filter(scored_docs: list, k: int, lambda_mult: float = 0.5) -> list:
    """Lightweight local MMR -- avoids a second vector-store network call."""
    if not scored_docs:
        return []

    def _sim(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0

    selected, remaining = [], list(scored_docs)
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            texts = [d.page_content for d in selected]

            def _score(item, _texts=texts):
                doc, rel = item
                return lambda_mult * rel - (1 - lambda_mult) * max(_sim(doc.page_content, t) for t in _texts)

            best = max(remaining, key=_score)
        selected.append(best[0])
        remaining.remove(best)
    return selected


def reciprocal_rank_fusion(dense_scored: list, bm25_docs: list, k: int = 60) -> list:
    scores, doc_map = {}, {}
    for rank, (doc, _) in enumerate(dense_scored):
        key = content_key(doc.page_content)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = content_key(doc.page_content)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc
    return [doc_map[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]


class Reranker:
    """Local cross-encoder reranker -- no network hop."""
    def __init__(self, enabled: bool):
        self._ranker = None
        if enabled and _HAS_FLASHRANK:
            self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        elif enabled:
            logger.warning("flashrank not installed -- reranking disabled. `pip install flashrank`.")

    @property
    def available(self) -> bool:
        return self._ranker is not None

    def rerank(self, query: str, docs: list) -> list:
        if not self._ranker or not docs:
            return docs
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results]


def rewrite_query(llm, question: str, history: list) -> str:
    """
    Query rewriting: expand a short/context-dependent follow-up question
    ("what about the second one?") into a self-contained query using
    recent conversation history. Skipped when there's no history (nothing
    to rewrite against) -- avoids an unnecessary LLM call on turn one.
    """
    if not history:
        return question
    q = (question or "").strip()
    q_lower = q.lower()
    should_rewrite = (
        len(q) <= config.QUERY_REWRITE_MAX_QUESTION_CHARS
        and any(term in q_lower for term in config.QUERY_REWRITE_TRIGGER_TERMS)
    )
    if not config.ENABLE_QUERY_REWRITE or not should_rewrite:
        return question
    hist_str = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:])
    prompt = (
        "Given this conversation history and a follow-up question, rewrite the "
        "follow-up into a standalone question that makes sense without the "
        "history. If it's already standalone, return it unchanged. Return ONLY "
        "the rewritten question, nothing else.\n\n"
        f"History:\n{hist_str}\n\nFollow-up question: {question}\n\nStandalone question:"
    )
    try:
        rewritten = llm.invoke(prompt).content.strip()
        return rewritten or question
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original question: {e}")
        return question
