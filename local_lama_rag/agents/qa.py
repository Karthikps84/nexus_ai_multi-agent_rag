import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from orchestration.model_router import RoutedChain, TaskModelRouter
from prompts.prompt_library import QA_PROMPT, WEB_PROMPT
import config


class QAAgent(BaseAgent):
    """Answers questions grounded in retrieved context (local docs) or,
    when local relevance is too low, web search results."""

    def __init__(self, fast_llm, primary_llm, model_router: TaskModelRouter, web_search_fn, format_web_context_fn):
        super().__init__()
        chain_fast = QA_PROMPT | fast_llm | StrOutputParser()
        chain_primary = QA_PROMPT | primary_llm | StrOutputParser()
        web_chain_fast = WEB_PROMPT | fast_llm | StrOutputParser()
        web_chain_primary = WEB_PROMPT | primary_llm | StrOutputParser()
        self.chain = RoutedChain(model_router, "qa", {"fast": chain_fast, "primary": chain_primary})
        self.web_chain = RoutedChain(model_router, "qa", {"fast": web_chain_fast, "primary": web_chain_primary})
        self._web_search = web_search_fn
        self._format_web_context = format_web_context_fn

    def answer_local(self, question: str, docs: list, history_str: str) -> dict:
        context = "\n\n".join(
            f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
            for d in docs
        )
        start = time.time()
        answer = self.chain.invoke({"context": context, "question": question, "history": history_str})
        self.logger.info(f"Local QA generation: {time.time() - start:.2f}s")
        sources = sorted({d.metadata.get("source_file", "unknown") for d in docs})
        return {"answer": answer, "sources": sources, "route": "local", "context": context}

    def answer_web(self, question: str, history_str: str) -> dict:
        results = self._web_search(question, config.WEB_SEARCH_RESULTS)
        context = self._format_web_context(results)
        start = time.time()
        answer = self.web_chain.invoke({"context": context, "question": question, "history": history_str})
        self.logger.info(f"Web QA generation: {time.time() - start:.2f}s")
        sources = [f"{r['title']} ({r['link']})" for r in results]
        return {"answer": answer, "sources": sources, "route": "web", "context": context}

    def stream_local(self, question: str, docs: list, history_str: str):
        context = "\n\n".join(
            f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
            for d in docs
        )
        yield from self.chain.stream({"context": context, "question": question, "history": history_str})

    def stream_web(self, question: str, history_str: str):
        results = self._web_search(question, config.WEB_SEARCH_RESULTS)
        context = self._format_web_context(results)
        yield from self.web_chain.stream({"context": context, "question": question, "history": history_str})
