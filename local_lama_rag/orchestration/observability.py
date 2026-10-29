from __future__ import annotations

import time
from dataclasses import asdict

from graph.state import AgentTrace


class ObservabilityTracker:
    def start(self, state, agent: str, model: str, detail: str = "") -> AgentTrace:
        trace = AgentTrace(agent=agent, model=model, detail=detail, started_at=time.time())
        state.trace.append(trace)
        return trace

    def finish(
        self,
        trace: AgentTrace,
        *,
        status: str = "completed",
        confidence: float | None = None,
        cache_hit: bool = False,
        metadata: dict | None = None,
    ):
        trace.ended_at = time.time()
        trace.status = status
        trace.confidence = confidence
        trace.cache_hit = cache_hit
        if metadata:
            trace.metadata.update(metadata)

    @staticmethod
    def to_payload(state) -> list[dict]:
        return [asdict(t) for t in state.trace]

