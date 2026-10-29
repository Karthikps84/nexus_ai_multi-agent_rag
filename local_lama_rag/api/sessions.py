import uuid

from agents.memory import MemoryAgent


class SessionStore:
    """In-process (owner, session_id) -> MemoryAgent map. A session id from
    one account is never usable by another."""

    def __init__(self):
        self._sessions: dict[tuple[str, str], MemoryAgent] = {}

    def create(self, owner: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[(owner, session_id)] = MemoryAgent()
        return session_id

    def get(self, session_id: str, owner: str) -> MemoryAgent:
        key = (owner, session_id)
        if key not in self._sessions:
            self._sessions[key] = MemoryAgent()
        return self._sessions[key]

    def clear(self, session_id: str, owner: str) -> bool:
        key = (owner, session_id)
        if key in self._sessions:
            self._sessions[key].clear()
            return True
        return False


sessions = SessionStore()
