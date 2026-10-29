from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, TypedDict


StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(slots=True)
class ExecutionStep:
    id: str
    agent: str
    instruction: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    max_retries: int = 1
    attempts: int = 0
    optional: bool = False
    parallel_group: str = ""


@dataclass(slots=True)
class ExecutionPlan:
    steps: list[ExecutionStep] = field(default_factory=list)
    rationale: str = ""
    target_files: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "target_files": list(self.target_files),
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass(slots=True)
class AgentTrace:
    agent: str
    model: str
    detail: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    status: str = "pending"
    confidence: float | None = None
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SharedAgentState:
    question: str
    history: list = field(default_factory=list)
    intent: str = "qa"
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    retrieved_chunks: list = field(default_factory=list)
    retrieved_tables: list = field(default_factory=list)
    retrieved_images: list = field(default_factory=list)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    confidence: float | None = None
    execution_time: float = 0.0
    cache_hits: dict[str, bool] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    trace: list[AgentTrace] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    forced_route: Optional[str] = None
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    context: str = ""
    dataframe: object = None
    verification: Optional[dict] = None
    max_score: float = 0.0
    docs: list = field(default_factory=list)
    route: str = ""
    last_agent: str = ""
    resolved_query: str = ""

    def to_result(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "route": self.route,
            "context": self.context,
            "dataframe": self.dataframe,
            "verification": self.verification,
            "max_score": self.max_score,
            "docs": self.docs,
            "execution_plan": self.execution_plan.to_payload(),
            "citations": self.citations,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "cache_hits": dict(self.cache_hits),
            "token_usage": dict(self.token_usage),
            "trace": [asdict(item) for item in self.trace],
            "agent_outputs": self.agent_outputs,
        }


class AgentState(TypedDict):
    shared: SharedAgentState
