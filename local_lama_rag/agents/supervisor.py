from __future__ import annotations

import asyncio
import time

import config
from agents.base import BaseAgent
from graph.state import ExecutionPlan, ExecutionStep, SharedAgentState


class SupervisorAgent(BaseAgent):
    def __init__(self, planner, router, verifier, reflector, model_router, observability):
        super().__init__()
        self.planner = planner
        self.router = router
        self.verifier = verifier
        self.reflector = reflector
        self.model_router = model_router
        self.observability = observability

    def analyze(self, state: SharedAgentState):
        if state.forced_route:
            state.intent = state.forced_route
            if not state.target_files:
                state.target_files = self.router.match_files_in_question(state.question)
            return
        result = self.router.classify(state.question)
        state.intent = result["route"]
        if not state.target_files:
            state.target_files = result["target_files"]

    def build_plan(self, state: SharedAgentState):
        state.execution_plan = self.planner.plan(state.question, route_hint=state.intent)
        if state.target_files:
            state.execution_plan.target_files = list(state.target_files)

    def _ready_steps(self, plan: ExecutionPlan) -> list[ExecutionStep]:
        completed = {step.id for step in plan.steps if step.status == "completed"}
        ready = []
        for step in plan.steps:
            if step.status != "pending":
                continue
            if all(dep in completed for dep in step.depends_on):
                ready.append(step)
        return ready

    async def _run_parallel(self, funcs: list):
        return await asyncio.gather(*[asyncio.to_thread(fn) for fn in funcs])

    @staticmethod
    def _supports_grounded_review(step: ExecutionStep, result: dict, state: SharedAgentState) -> bool:
        if step.agent in {"verify", "retrieve", "image"}:
            return False
        if not isinstance(result, dict):
            return False
        context = result.get("context", state.context)
        return bool((context or "").strip())

    def _maybe_improve_result(self, step: ExecutionStep, result: dict, state: SharedAgentState) -> tuple[dict, float | None]:
        context = result.get("context", state.context)
        verification = self.verifier.verify(result["answer"], context, result.get("sources", []))
        result["verification"] = verification
        confidence = verification.get("confidence")

        should_reflect = (
            config.ENABLE_REFLECTION
            and self._supports_grounded_review(step, result, state)
            and result.get("route", state.route) in config.REFLECTION_ROUTES
            and len(result.get("answer", "")) >= config.REFLECTION_MIN_ANSWER_CHARS
            and confidence is not None
            and confidence < config.REFLECTION_TRIGGER_CONFIDENCE
        )
        if should_reflect:
            reflected = self.reflector.reflect(
                question=step.instruction or state.question,
                answer=result["answer"],
                context=context,
            )
            result["answer"] = reflected["answer"]
            result["reflection"] = reflected
            verification = self.verifier.verify(
                result["answer"],
                context,
                result.get("sources", []),
                force_llm_check=(confidence is not None and confidence < config.FINAL_VERIFICATION_FORCE_LLM_THRESHOLD),
            )
            result["verification"] = verification
            confidence = verification.get("confidence")
        return result, confidence

    def execute(self, state: SharedAgentState, dispatch_step):
        plan = state.execution_plan
        if not plan.steps:
            return

        while True:
            ready = self._ready_steps(plan)
            if not ready:
                break

            parallel = [step for step in ready if step.parallel_group]
            sequential = [step for step in ready if not step.parallel_group]

            if parallel:
                funcs = [lambda s=step: self._execute_step(state, s, dispatch_step) for step in parallel]
                asyncio.run(self._run_parallel(funcs))
            for step in sequential:
                self._execute_step(state, step, dispatch_step)

    def _execute_step(self, state: SharedAgentState, step: ExecutionStep, dispatch_step):
        trace = self.observability.start(
            state,
            agent=step.agent,
            model=self.model_router.resolve_name(step.agent),
            detail=step.instruction,
        )
        started = time.time()
        step.status = "running"

        try:
            result = dispatch_step(step, state)
            confidence = None
            if isinstance(result, dict):
                confidence = (result.get("verification") or {}).get("confidence")
            if step.agent not in ("verify", "retrieve") and isinstance(result, dict) and result.get("answer"):
                result, confidence = self._maybe_improve_result(step, result, state)
                if confidence is not None and confidence < config.SUPERVISOR_MIN_CONFIDENCE and step.attempts < step.max_retries:
                    step.attempts += 1
                    self.logger.info("Retrying %s because confidence %.2f is below threshold", step.agent, confidence)
                    retry_result = dispatch_step(step, state, retry=True)
                    if retry_result.get("answer"):
                        retry_result, confidence = self._maybe_improve_result(step, retry_result, state)
                        retry_result["verification"] = self.verifier.verify(
                            retry_result["answer"],
                            retry_result.get("context", state.context),
                            retry_result.get("sources", []),
                            force_llm_check=True,
                        )
                        result = retry_result
                        confidence = retry_result["verification"].get("confidence")

            state.agent_outputs[step.id] = result
            state.last_agent = step.agent
            step.status = "completed"
            self.observability.finish(
                trace,
                status="completed",
                confidence=confidence,
                metadata={"latency_ms": round((time.time() - started) * 1000, 1), "attempts": step.attempts + 1},
            )
        except Exception as exc:
            step.status = "failed"
            self.observability.finish(
                trace,
                status="failed",
                metadata={"error": str(exc), "latency_ms": round((time.time() - started) * 1000, 1)},
            )
            if step.optional:
                self.logger.warning("Optional step %s failed: %s", step.agent, exc)
                return
            raise
