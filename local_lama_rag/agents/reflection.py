import time

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
import config
from prompts.prompt_library import REFLECTION_PROMPT


class ReflectionAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.chain = REFLECTION_PROMPT | llm | StrOutputParser()

    def reflect(self, question: str, answer: str, context: str) -> dict:
        if not context.strip():
            return {"answer": answer, "changed": False, "notes": "No context available for reflection."}
        if not config.ENABLE_REFLECTION:
            return {"answer": answer, "changed": False, "notes": "Reflection disabled by config."}

        start = time.time()
        revised = self.chain.invoke({"question": question, "answer": answer, "context": context}).strip()
        self.logger.info("Reflection: %.2fs", time.time() - start)
        if not revised:
            return {"answer": answer, "changed": False, "notes": "Reflection returned empty output."}
        return {"answer": revised, "changed": revised != answer, "notes": "Answer reflected against context."}
