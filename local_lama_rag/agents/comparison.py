import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from agents.retrieval import RetrievalAgent
from orchestration.model_router import RoutedChain, TaskModelRouter
from prompts.prompt_library import COMPARE_PROMPT
import config


class ComparisonAgent(BaseAgent):
    """Compares two or more documents (or tables/reports within them)."""

    def __init__(self, fast_llm, primary_llm, model_router: TaskModelRouter, retrieval_agent: RetrievalAgent):
        super().__init__()
        chain_fast = COMPARE_PROMPT | fast_llm | StrOutputParser()
        chain_primary = COMPARE_PROMPT | primary_llm | StrOutputParser()
        self.chain = RoutedChain(model_router, "compare", {"fast": chain_fast, "primary": chain_primary})
        self.retrieval_agent = retrieval_agent

    def compare(self, question: str, target_files: list) -> dict:
        max_per_file = config.COMPARE_MAX_CHUNKS_PER_FILE
        parts = []
        for f in target_files:
            chunks = self.retrieval_agent.fetch_file_chunks(f, max_per_file)
            if chunks:
                parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
        context = "\n\n".join(parts) if parts else "(not enough distinct files found to compare)"

        start = time.time()
        answer = self.chain.invoke({"context": context, "question": question})
        self.logger.info(f"Compare generation: {time.time() - start:.2f}s")

        return {"answer": answer, "sources": target_files, "route": "compare", "context": context}

    def stream(self, question: str, target_files: list):
        max_per_file = config.COMPARE_MAX_CHUNKS_PER_FILE
        parts = []
        for f in target_files:
            chunks = self.retrieval_agent.fetch_file_chunks(f, max_per_file)
            if chunks:
                parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
        context = "\n\n".join(parts) if parts else "(not enough distinct files found to compare)"
        yield from self.chain.stream({"context": context, "question": question})
