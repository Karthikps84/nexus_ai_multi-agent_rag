import uuid

from agents.memory import MemoryAgent


class SessionStore:
    """In-process session_id -> MemoryAgent map, so concurrent API callers
    get isolated conversation history instead of sharing one MemoryAgent."""

    def __init__(self):
        self._sessions: dict[str, MemoryAgent] = {}

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = MemoryAgent()
        return session_id

    def get(self, session_id: str) -> MemoryAgent:
        if session_id not in self._sessions:
            self._sessions[session_id] = MemoryAgent()
        return self._sessions[session_id]

    def clear(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            return True
        return False


sessions = SessionStore()
