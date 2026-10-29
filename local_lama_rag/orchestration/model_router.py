from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelRegistry:
    primary: object
    fast: object
    vision: object
    verifier: object


class TaskModelRouter:
    def __init__(self, registry: ModelRegistry, routing: dict[str, str]):
        self.registry = registry
        self.routing = routing

    def resolve(self, task: str):
        model_role = self.routing.get(task, self.routing.get("default", "fast"))
        return getattr(self.registry, model_role)

    def resolve_name(self, task: str) -> str:
        model = self.resolve(task)
        return getattr(model, "model", task)

    def resolve_role(self, task: str) -> str:
        return self.routing.get(task, self.routing.get("default", "fast"))


class RoutedChain:
    """
    Wraps one pre-built chain per model role and picks the active one at
    call time via TaskModelRouter -- this is what actually makes
    config.MODEL_ROUTING drive model selection. Without this, agents were
    permanently bound to whichever model they got at construction, and
    editing MODEL_ROUTING had no effect at runtime.

    Chains are built once at construction (LangChain Runnables are cheap
    to construct; the expense is in .invoke()/.stream(), not assembly),
    so there's no added latency from having both available.
    """

    def __init__(self, model_router: TaskModelRouter, task: str, chains: dict):
        self._router = model_router
        self._task = task
        self._chains = chains

    def _active(self):
        role = self._router.resolve_role(self._task)
        return self._chains.get(role, self._chains.get("fast"))

    def invoke(self, *args, **kwargs):
        return self._active().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._active().stream(*args, **kwargs)

