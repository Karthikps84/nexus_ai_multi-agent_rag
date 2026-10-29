import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from agents.retrieval import RetrievalAgent
from orchestration.model_router import RoutedChain, TaskModelRouter
from prompts.prompt_library import SUMMARY_PROMPT
import config


class SummarizerAgent(BaseAgent):
    """Generates executive / detailed / bullet summaries of one or more
    documents. Mode is inferred from the question if not given explicitly
    ("give me a quick overview" -> executive, "detailed summary" -> detailed,
    default -> bullet, which works well for most FAQ-style summarize asks)."""

    def __init__(self, fast_llm, primary_llm, model_router: TaskModelRouter, retrieval_agent: RetrievalAgent):
        super().__init__()
        chain_fast = SUMMARY_PROMPT | fast_llm | StrOutputParser()
        chain_primary = SUMMARY_PROMPT | primary_llm | StrOutputParser()
        self.chain = RoutedChain(model_router, "summarize", {"fast": chain_fast, "primary": chain_primary})
        self.retrieval_agent = retrieval_agent

    @staticmethod
    def _infer_mode(question: str) -> str:
        q = question.lower()
        if any(kw in q for kw in ("executive", "quick overview", "tl;dr", "tldr", "brief")):
            return "executive"
        if any(kw in q for kw in ("detailed", "in depth", "in-depth", "thorough", "comprehensive")):
            return "detailed"
        return "bullet"

    def summarize(self, question: str, target_files: list, mode: str = None, context_override: str = None) -> dict:
        mode = mode or self._infer_mode(question)

        if context_override:
            context = context_override
        else:
            max_per_file = config.SUMMARY_MAX_CHUNKS_PER_FILE
            parts = []
            for f in target_files:
                chunks = self.retrieval_agent.fetch_file_chunks(f, max_per_file)
                if chunks:
                    parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
            context = "\n\n".join(parts) if parts else "(no content found for the requested file(s))"

        start = time.time()
        answer = self.chain.invoke({"context": context, "question": question, "mode": mode})
        self.logger.info(f"Summarize ({mode}) generation: {time.time() - start:.2f}s")

        return {"answer": answer, "sources": target_files, "route": "summarize", "mode": mode, "context": context}

    def stream(self, question: str, target_files: list, mode: str = None):
        mode = mode or self._infer_mode(question)
        max_per_file = config.SUMMARY_MAX_CHUNKS_PER_FILE
        parts = []
        for f in target_files:
            chunks = self.retrieval_agent.fetch_file_chunks(f, max_per_file)
            if chunks:
                parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
        context = "\n\n".join(parts) if parts else "(no content found for the requested file(s))"
        yield from self.chain.stream({"context": context, "question": question, "mode": mode})
