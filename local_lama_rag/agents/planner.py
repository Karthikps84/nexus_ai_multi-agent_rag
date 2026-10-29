import re

from langchain_core.output_parsers import StrOutputParser

from agents.base import BaseAgent
from graph.state import ExecutionPlan, ExecutionStep
from prompts.prompt_library import PLANNER_PROMPT


_SECOND_TASK_RE = re.compile(
    r" and (?:then |also )?(?:compare|summari[sz]e|explain|list|extract|write|show|give|tell|describe|"
    r"what|how|why|who|when|where|which|create|generate|find|calculate|translate)\b"
)


class PlannerAgent(BaseAgent):
    _LINE_RE = re.compile(
        r"^\s*([a-zA-Z0-9_\-]+)\s*\|\s*([a-z_]+)\s*\|\s*([^|]+)\|\s*(.+)$"
    )
    _VALID_AGENTS = {"retrieve", "qa", "summarize", "compare", "table", "code", "image", "vision_analysis", "verify"}

    def __init__(self, fast_llm):
        super().__init__()
        self.chain = PLANNER_PROMPT | fast_llm | StrOutputParser()

    def is_compound(self, question: str) -> bool:
        """Cheap heuristic gate before paying for an LLM planning call --
        only bother decomposing if the question plausibly has multiple asks."""
        if len(question) <= 40:
            return False
        q = question.lower()
        if any(j in q for j in (" then ", " also ", "; ")) or q.count("?") > 1:
            return True
        # A bare " and " usually joins two nouns ("pull requests and the changelog"),
        # so only treat it as compound when a second task follows it.
        return bool(_SECOND_TASK_RE.search(q))

    def _fallback_plan(self, question: str, route_hint: str) -> ExecutionPlan:
        steps: list[ExecutionStep] = []
        needs_retrieve = route_hint == "qa" or (
            route_hint == "code" and any(term in question.lower() for term in ("paper", "document", "based on"))
        )
        if needs_retrieve:
            steps.append(ExecutionStep(id="retrieve_1", agent="retrieve", instruction=question))
            depends = ["retrieve_1"]
        else:
            depends = []

        primary_id = f"{route_hint or 'qa'}_1"
        steps.append(
            ExecutionStep(
                id=primary_id,
                agent=route_hint or "qa",
                instruction=question,
                depends_on=depends,
            )
        )
        steps.append(
            ExecutionStep(
                id="verify_1",
                agent="verify",
                instruction="Verify and finalize the response.",
                depends_on=[primary_id],
            )
        )
        return ExecutionPlan(steps=steps, rationale="Heuristic single-request plan.")

    def plan(self, question: str, route_hint: str = "qa") -> ExecutionPlan:
        if not self.is_compound(question):
            return self._fallback_plan(question, route_hint)

        raw = self.chain.invoke({"question": question})
        steps: list[ExecutionStep] = []
        for line in raw.strip().splitlines():
            match = self._LINE_RE.match(line)
            if not match:
                continue
            step_id, agent, deps_raw, instruction = match.groups()
            agent = agent.strip().lower()
            if agent not in self._VALID_AGENTS:
                continue
            deps = [] if deps_raw.strip().lower() == "none" else [d.strip() for d in deps_raw.split(",") if d.strip()]
            steps.append(
                ExecutionStep(
                    id=step_id.strip(),
                    agent=agent,
                    instruction=instruction.strip(),
                    depends_on=deps,
                    parallel_group="analysis" if agent in {"summarize", "table", "image"} else "",
                )
            )

        if not steps:
            self.logger.info("Planner found no valid plan lines; using heuristic fallback.")
            return self._fallback_plan(question, route_hint)

        if steps[-1].agent != "verify":
            depended_on = {dep for step in steps for dep in step.depends_on}
            terminal_ids = [step.id for step in steps if step.id not in depended_on and step.agent != "verify"]
            steps.append(
                ExecutionStep(
                    id="verify_final",
                    agent="verify",
                    instruction="Verify and finalize the response.",
                    depends_on=terminal_ids or [steps[-1].id],
                )
            )

        return ExecutionPlan(steps=steps, rationale=raw.strip())
