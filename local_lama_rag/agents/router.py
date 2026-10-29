from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from prompts.prompt_library import ROUTER_PROMPT

VALID_ROUTES = {"qa", "summarize", "compare", "table", "code", "image", "vision_analysis"}

_KEYWORDS = {
    "summarize": ("summarize", "summarise", "summary", "tl;dr", "tldr", "overview of", "recap"),
    "table": ("extract", "pull out", "list all", "table of", "tabulate"),
    "compare": ("compare", "comparison"),
    "code": ("write code", "generate code", "write a function", "write a script", "write python",
             "write a python", "code to ", "function to ", "script to ", "implement ",
             "debug this", "fix this code", "optimize this code",
             "explain this code", "refactor this"),
    # Checked before "image" -- specific detection/counting phrasing should
    # win over the broader VLM-description keywords when both could match
    # (e.g. "how many objects in this image").
    "vision_analysis": ("detect object", "object detection", "run detection", "how many objects",
                        "count the objects", "count objects", "bounding box", "bounding boxes",
                        "detect and count", "segment the image", "run yolo"),
    "image": ("figure", "chart", "diagram", "image", "graph shows", "picture", "screenshot"),
}


class RouterAgent(BaseAgent):
    """
    Detects intent and routes to the right specialized agent(s).

    Fast path: keyword matching, zero added latency, handles the large
    majority of requests correctly. Falls back to an LLM classification
    call only when no keyword matched -- ambiguous free-text questions
    that don't clearly signal a task type default to 'qa' via keywords
    already, so the LLM fallback mainly helps borderline cases where
    the keyword set genuinely doesn't apply.
    """

    def __init__(self, fast_llm, known_files_fn):
        super().__init__()
        self.chain = ROUTER_PROMPT | fast_llm | StrOutputParser()
        self._known_files_fn = known_files_fn

    def match_files_in_question(self, question: str) -> list:
        q = question.lower()
        return [f for f in self._known_files_fn() if Path(f).stem.lower() in q or f.lower() in q]

    def classify_keywords(self, question: str) -> str | None:
        q = question.lower()
        for route, kws in _KEYWORDS.items():
            if any(kw in q for kw in kws):
                return route
        return None

    def classify(self, question: str, use_llm_fallback: bool = True) -> dict:
        route = self.classify_keywords(question)
        method = "keyword"

        if route is None:
            if use_llm_fallback:
                raw = self.chain.invoke({"question": question}).strip().lower()
                route = raw if raw in VALID_ROUTES else "qa"
                method = "llm"
            else:
                route = "qa"
                method = "default"

        target_files = self.match_files_in_question(question)
        self.logger.info(f"Route: {route} (via {method}) | files: {target_files or 'none matched'}")
        return {"route": route, "target_files": target_files, "method": method}
