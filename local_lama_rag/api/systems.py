"""One MultiAgentSystem per user (own documents, own index), kept in a small LRU."""
from __future__ import annotations

import threading
from collections import OrderedDict

import config
from graph.workflow import MultiAgentSystem

from api.auth import User


class SystemManager:
    def __init__(self):
        self._systems: OrderedDict[str, MultiAgentSystem] = OrderedDict()
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def lock_for(self, user: User) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(user.id, threading.RLock())

    def get(self, user: User) -> MultiAgentSystem:
        with self.lock_for(user):
            with self._guard:
                if user.id in self._systems:
                    self._systems.move_to_end(user.id)
                    return self._systems[user.id]
            system = MultiAgentSystem(data_dir=user.data_dir, storage_dir=user.storage_dir)
            system.initialize()
            with self._guard:
                self._systems[user.id] = system
                while len(self._systems) > config.MAX_ACTIVE_USER_SYSTEMS:
                    self._systems.popitem(last=False)
            return system

    def clear(self) -> None:
        with self._guard:
            self._systems.clear()


systems = SystemManager()
