import time

from agents.base import BaseAgent
from retriever.pipeline import RetrievalPipeline
from vectorstore.chroma_store import VectorStoreManager
import config


class RetrievalAgent(BaseAgent):
    """
    Handles all retrieval: hybrid dense+BM25 search with RRF fusion, MMR
    diversification, cross-encoder reranking, metadata (source-file)
    filtering, and optional query rewriting for follow-up questions.
    """

    def __init__(self, vs_manager: VectorStoreManager, llm=None):
        super().__init__()
        self.vs_manager = vs_manager
        self.llm = llm  # used for query rewriting; optional
        self.pipeline = RetrievalPipeline(vs_manager, llm=llm)

    def index_chunks(self, chunks: list):
        self.pipeline.bind_vectorstore()
        self.pipeline.index_chunks(chunks)
        if self.pipeline.sparse.available:
            self.logger.info(f"BM25 index ready over {len(chunks)} chunks.")

    def retrieve(
        self, question: str, top_k: int = None, final_k: int = None,
        source_filter: list = None, history: list = None, use_query_rewrite: bool = False,
    ):
        t_start = time.time()
        artifacts = self.pipeline.retrieve(
            question=question,
            top_k=top_k or config.SIMILARITY_TOP_K,
            final_k=final_k or config.FINAL_TOP_K,
            source_filter=source_filter,
            history=history,
            use_query_rewrite=use_query_rewrite,
        )
        self.logger.info(
            "TOTAL: %.3fs | %s doc(s) returned | cache_hit=%s",
            time.time() - t_start,
            len(artifacts.docs),
            artifacts.cache_hit,
        )
        return artifacts.docs, artifacts.max_score

    def fetch_file_chunks(self, filename: str, max_chunks: int) -> list:
        """Cheap metadata-filtered fetch -- no embedding call. Used by
        summarize/compare/table agents that need whole-file context."""
        return self.vs_manager.fetch_file_chunks(filename, max_chunks)

    def list_known_files(self) -> list:
        return self.vs_manager.list_known_files()

    def resolve_target_files(self, question: str, explicit_files: list, min_files: int = 1) -> list:
        """
        Decide which file(s) a summarize/compare/table request applies to:
        1. Explicitly passed files (from the router/UI) win.
        2. Files named in the question text.
        3. Single-file corpus -> use it.
        4. Otherwise, a quick relevance search picks the most relevant files.
        """
        if explicit_files:
            return explicit_files

        matched = self._match_files_in_question(question)
        if matched:
            return matched

        known = self.list_known_files()
        if len(known) == 1:
            return known

        docs, _ = self.retrieve(question)
        seen = []
        for d in docs:
            f = d.metadata.get("source_file")
            if f and f not in seen:
                seen.append(f)
            if len(seen) >= max(min_files, 2):
                break
        return seen or known[:min_files]

    def _match_files_in_question(self, question: str) -> list:
        from pathlib import Path
        q = question.lower()
        matches = []
        for fname in self.list_known_files():
            stem = Path(fname).stem.lower()
            if stem in q or fname.lower() in q:
                matches.append(fname)
        return matches
