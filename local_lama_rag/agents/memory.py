from agents.base import BaseAgent
from utils.helpers import trim_history
import config


class MemoryAgent(BaseAgent):
    """
    Owns conversation history and the registry of files uploaded this
    session. Deliberately simple (in-process list, not a persisted store)
    -- the point is giving every other agent one place to read/write
    session state instead of each agent tracking its own copy.
    """

    def __init__(self, max_turns: int = None):
        super().__init__()
        self.max_turns = max_turns or config.MAX_HISTORY_TURNS
        self.history: list[tuple[str, str]] = []
        self._answer_cache: dict = {}

    def get_history(self) -> list:
        return self.history

    def format_history(self) -> str:
        if not self.history:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history)

    def add_turn(self, question: str, answer: str):
        self.history.append((question, answer))
        self.history = trim_history(self.history, self.max_turns)
        self.logger.info(f"Turn stored. History length: {len(self.history)}")

    def clear(self):
        self.history = []
        self._answer_cache = {}

    # ── simple answer cache, keyed by route+files+question ─────────────────
    def cache_get(self, key: str):
        return self._answer_cache.get(key)

    def cache_set(self, key: str, value):
        self._answer_cache[key] = value
