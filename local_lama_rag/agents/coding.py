import re
import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from prompts.prompt_library import (
    CODE_GENERATE_PROMPT, CODE_EXPLAIN_PROMPT, CODE_DEBUG_PROMPT, CODE_OPTIMIZE_PROMPT,
)


class CodingAgent(BaseAgent):
    """
    Generates, explains, debugs, and optimizes code. Uses the primary
    (larger) LLM rather than the fast one -- code correctness benefits
    more from stronger reasoning than FAQ-style lookups do, and code
    requests are typically one-off rather than high-frequency, so the
    latency tradeoff is worth it.
    """

    def __init__(self, llm):
        super().__init__()
        self.generate_chain = CODE_GENERATE_PROMPT | llm | StrOutputParser()
        self.explain_chain = CODE_EXPLAIN_PROMPT | llm | StrOutputParser()
        self.debug_chain = CODE_DEBUG_PROMPT | llm | StrOutputParser()
        self.optimize_chain = CODE_OPTIMIZE_PROMPT | llm | StrOutputParser()

    @staticmethod
    def _infer_action(question: str) -> str:
        q = question.lower()
        if any(kw in q for kw in ("debug", "fix this", "why does this fail", "error", "bug")):
            return "debug"
        if any(kw in q for kw in ("optimize", "speed up", "make this faster", "improve performance")):
            return "optimize"
        if any(kw in q for kw in ("explain this code", "what does this code do", "walk me through this code")):
            return "explain"
        return "generate"

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Pull the first fenced code block out of the user's message, if
        any -- for explain/debug/optimize requests where they pasted code
        inline."""
        m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def run(self, question: str, context: str = "", action: str = None) -> dict:
        action = action or self._infer_action(question)
        start = time.time()

        if action == "generate":
            answer = self.generate_chain.invoke({"question": question, "context": context})
        elif action == "explain":
            code = self._extract_code_block(question)
            answer = self.explain_chain.invoke({"code": code})
        elif action == "debug":
            code = self._extract_code_block(question)
            answer = self.debug_chain.invoke({"code": code, "error": ""})
        elif action == "optimize":
            code = self._extract_code_block(question)
            answer = self.optimize_chain.invoke({"code": code, "goal": "general (speed/readability)"})
        else:
            answer = self.generate_chain.invoke({"question": question, "context": context})

        self.logger.info(f"Code agent [{action}]: {time.time() - start:.2f}s")
        return {"answer": answer, "sources": [], "route": "code", "action": action}

    def stream(self, question: str, context: str = "", action: str = None):
        action = action or self._infer_action(question)
        if action == "generate":
            yield from self.generate_chain.stream({"question": question, "context": context})
        elif action == "explain":
            yield from self.explain_chain.stream({"code": self._extract_code_block(question)})
        elif action == "debug":
            yield from self.debug_chain.stream({"code": self._extract_code_block(question), "error": ""})
        elif action == "optimize":
            yield from self.optimize_chain.stream({"code": self._extract_code_block(question), "goal": "general"})
        else:
            yield from self.generate_chain.stream({"question": question, "context": context})
