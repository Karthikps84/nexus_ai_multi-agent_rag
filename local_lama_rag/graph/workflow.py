"""
==========================================================
LangGraph Multi-Agent Workflow
==========================================================
Supervisor analyzes intent -> Planner produces an execution
plan -> Supervisor executes steps (with retries/parallelism)
-> final verifier packages the response + observability.
==========================================================
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langgraph.graph import END, StateGraph

# langchain_nvidia_ai_endpoints keeps a hardcoded list of "known" NVIDIA
# model types and warns when a model isn't on it -- cosmetic only, inference
# still works against any model NVIDIA's API actually serves. Silenced here
# rather than left to spam startup logs every time a model is swapped.
warnings.filterwarnings(
    "ignore",
    message=r"Found .* in available_models, but type is unknown",
    category=UserWarning,
)

import config
from agents.coding import CodingAgent
from agents.comparison import ComparisonAgent
from agents.image import ImageAgent
from agents.memory import MemoryAgent
from agents.planner import PlannerAgent
from agents.qa import QAAgent
from agents.reflection import ReflectionAgent
from agents.retrieval import RetrievalAgent
from agents.router import RouterAgent
from agents.summarizer import SummarizerAgent
from agents.supervisor import SupervisorAgent
from agents.table import TableAgent
from agents.verifier import VerifierAgent
from agents.vision_analysis import VisionAnalysisAgent
from graph.state import AgentState, SharedAgentState
from orchestration.cache import CacheConfig, CacheManager, CachedEmbeddings
from orchestration.model_router import ModelRegistry, TaskModelRouter
from orchestration.observability import ObservabilityTracker
from utils.langsmith_setup import setup_langsmith
from utils.logger import get_logger
from utils.web_search import format_results_as_context, web_search
from vectorstore.chroma_store import VectorStoreManager

load_dotenv()
logger = get_logger("workflow")

API_KEY = os.getenv("NVIDIA_API_KEY")


class MultiAgentSystem:
    def __init__(self, data_dir=config.DATA_DIR, storage_dir=config.STORAGE_DIR):
        setup_langsmith()
        if not API_KEY:
            raise EnvironmentError("NVIDIA_API_KEY must be set in your .env file (used for all LLM, vision and embedding models)")

        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.cache_manager = CacheManager(
            CacheConfig(
                response_ttl_seconds=config.RESPONSE_CACHE_TTL_SECONDS,
                retrieval_ttl_seconds=config.RETRIEVAL_CACHE_TTL_SECONDS,
                embedding_ttl_seconds=config.EMBEDDING_CACHE_TTL_SECONDS,
                response_max_entries=config.RESPONSE_CACHE_MAX_ENTRIES,
                retrieval_max_entries=config.RETRIEVAL_CACHE_MAX_ENTRIES,
                embedding_max_entries=config.EMBEDDING_CACHE_MAX_ENTRIES,
            )
        )

        primary_llm = ChatNVIDIA(
            model=config.LLM_MODEL,
            api_key=API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.1,
            timeout=config.NVIDIA_PRIMARY_TIMEOUT_SECONDS,
            max_completion_tokens=config.LLM_MAX_TOKENS,
        )
        fast_llm = ChatNVIDIA(
            model=config.FAST_LLM_MODEL,
            api_key=API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.1,
            timeout=config.NVIDIA_FAST_TIMEOUT_SECONDS,
            max_completion_tokens=config.LLM_MAX_TOKENS,
        )
        vision_llm = ChatNVIDIA(
            model=config.VISION_MODEL,
            api_key=API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            timeout=config.NVIDIA_VISION_TIMEOUT_SECONDS,
            max_completion_tokens=config.LLM_MAX_TOKENS,
        )
        embeddings = NVIDIAEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            truncate="END",
            max_batch_size=config.EMBED_BATCH_SIZE,
        )
        self.embeddings = CachedEmbeddings(embeddings, self.cache_manager)
        # Bind the (already text-cached) embed_query so semantic similarity
        # lookups on the response and retrieval caches reuse the same
        # embedding calls rather than firing fresh ones.
        self.cache_manager.bind_embedder(self.embeddings.embed_query)

        self.models = ModelRegistry(
            primary=primary_llm,
            fast=fast_llm,
            vision=vision_llm,
            verifier=fast_llm,
        )
        self.model_router = TaskModelRouter(self.models, config.MODEL_ROUTING)
        self.observability = ObservabilityTracker()

        self.vs_manager = VectorStoreManager(self.data_dir, self.storage_dir, self.embeddings)

        self.memory = MemoryAgent()
        self.retrieval = RetrievalAgent(self.vs_manager, llm=self.models.fast)
        self.retrieval.pipeline.cache_manager = self.cache_manager
        self.qa = QAAgent(self.models.fast, self.models.primary, self.model_router, web_search, format_results_as_context)
        self.summarizer = SummarizerAgent(self.models.fast, self.models.primary, self.model_router, self.retrieval)
        self.comparison = ComparisonAgent(self.models.fast, self.models.primary, self.model_router, self.retrieval)
        self.table = TableAgent(self.models.fast, self.models.primary, self.model_router, self.retrieval)
        self.coding = CodingAgent(self.models.primary)
        self.image = ImageAgent(self.models.vision, self.data_dir)
        self.vision_analysis = VisionAnalysisAgent(
            weights=config.VISION_ANALYSIS_WEIGHTS,
            confidence=config.VISION_ANALYSIS_CONFIDENCE,
            max_detections=config.VISION_ANALYSIS_MAX_DETECTIONS,
        ) if config.ENABLE_VISION_ANALYSIS else None
        self.verifier = VerifierAgent(self.models.verifier)
        self.reflector = ReflectionAgent(self.models.verifier)
        self.router = RouterAgent(self.models.fast, known_files_fn=self.retrieval.list_known_files)
        self.planner = PlannerAgent(self.models.fast)
        self.supervisor = SupervisorAgent(
            planner=self.planner,
            router=self.router,
            verifier=self.verifier,
            reflector=self.reflector,
            model_router=self.model_router,
            observability=self.observability,
        )

        self.graph = self._build_graph()

    def initialize(self, on_progress=None):
        chunks = self.vs_manager.initialize(on_progress=on_progress)
        self.retrieval.index_chunks(chunks)

    @staticmethod
    def _format_history(history: list) -> str:
        if not history:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in history)

    def _resolve_doc_context(self, question: str, state: SharedAgentState):
        history = state.history or []
        artifacts = self.retrieval.pipeline.retrieve(
            question=question,
            source_filter=state.target_files or None,
            history=history,
            use_query_rewrite=True,
        )
        state.docs = artifacts.docs
        state.retrieved_chunks = artifacts.docs
        state.max_score = artifacts.max_score
        state.context = self.retrieval.pipeline.context_builder.build(artifacts.docs)
        state.sources = sorted({d.metadata.get("source_file", "unknown") for d in artifacts.docs})
        state.cache_hits["retrieval"] = artifacts.cache_hit
        # artifacts.query is the history-resolved query ("explain more about
        # it" -> "explain more about cricket") used for the actual dense/
        # sparse search. Expose it so callers use the SAME resolved text for
        # web search and the LLM-facing question -- not the raw follow-up,
        # which the pipeline discarded after using it internally otherwise.
        state.resolved_query = artifacts.query
        return artifacts

    def _dispatch_step(self, step, state: SharedAgentState, retry: bool = False) -> dict:
        if step.agent == "retrieve":
            artifacts = self._resolve_doc_context(step.instruction, state)
            return {
                "answer": f"Retrieved {len(artifacts.docs)} supporting chunk(s).",
                "sources": state.sources,
                "route": "retrieve",
                "context": state.context,
                "retrieved_chunks": artifacts.docs,
            }

        if step.agent == "qa":
            artifacts = self._resolve_doc_context(step.instruction, state)
            history_str = self._format_history(state.history)
            # Use the history-resolved query (e.g. "explain more about it" ->
            # "explain more about cricket") for what the LLM/web search sees.
            # Retrieval already used this internally; without this, a
            # follow-up question that falls through to web search literally
            # searches the raw pronoun-only text with no topic in it.
            resolved_question = state.resolved_query or step.instruction
            if artifacts.max_score >= config.SIMILARITY_THRESHOLD:
                result = self.qa.answer_local(resolved_question, artifacts.docs, history_str)
            else:
                result = self.qa.answer_web(resolved_question, history_str)
            state.route = result["route"]
            return result

        if step.agent == "summarize":
            files = self.retrieval.resolve_target_files(step.instruction, state.target_files, min_files=1)
            return self.summarizer.summarize(step.instruction, files)

        if step.agent == "compare":
            files = self.retrieval.resolve_target_files(step.instruction, state.target_files, min_files=2)
            return self.comparison.compare(step.instruction, files)

        if step.agent == "table":
            files = self.retrieval.resolve_target_files(step.instruction, state.target_files, min_files=1)
            return self.table.extract(step.instruction, files)

        if step.agent == "code":
            context = state.context if any(s.agent == "retrieve" for s in state.execution_plan.steps) else ""
            if retry and state.agent_outputs:
                prior = next(
                    (out for out in reversed(list(state.agent_outputs.values())) if isinstance(out, dict) and out.get("answer")),
                    None,
                )
                if prior:
                    context = f"{context}\n\nPrevious attempt issues:\n{prior.get('verification', {}).get('unsupported_claims') or ''}"
            return self.coding.run(step.instruction, context=context)

        if step.agent == "image":
            files = state.target_files or self.retrieval.resolve_target_files(step.instruction, [], min_files=1)
            target = files[0] if files else None
            if not target:
                return {
                    "answer": "I need to know which document's image or figure you mean.",
                    "sources": [],
                    "route": "image",
                    "context": "",
                }
            return self.image.answer_question(target, step.instruction)

        if step.agent == "vision_analysis":
            if self.vision_analysis is None or not self.vision_analysis.available:
                return {
                    "answer": "Object detection isn't available (ultralytics not installed on this deployment).",
                    "sources": [], "route": "vision_analysis", "context": "", "dataframe": None,
                }
            files = state.target_files or self.retrieval.resolve_target_files(step.instruction, [], min_files=1)
            target = files[0] if files else None
            if not target:
                return {
                    "answer": "I need to know which image file to run detection on.",
                    "sources": [], "route": "vision_analysis", "context": "", "dataframe": None,
                }
            result = self.vision_analysis.analyze_file(target, self.data_dir)
            result["context"] = ""
            return result

        if step.agent == "verify":
            return self._finalize_from_outputs(state)

        raise ValueError(f"Unknown plan step agent: {step.agent}")

    def _finalize_from_outputs(self, state: SharedAgentState) -> dict:
        step_by_id = {step.id: step for step in state.execution_plan.steps}
        outputs = [
            result
            for step_id, result in state.agent_outputs.items()
            if step_by_id.get(step_id) and step_by_id[step_id].agent not in {"retrieve", "verify"} and isinstance(result, dict)
        ]
        if not outputs:
            state.answer = "I couldn't produce a supported answer for that request."
            state.sources = []
            state.route = state.intent
            state.context = ""
            state.verification = None
            return state.to_result()

        if len(outputs) == 1:
            result = outputs[0]
            state.answer = result.get("answer", "")
            state.sources = result.get("sources", [])
            state.route = result.get("route", state.intent)
            state.context = result.get("context", state.context)
            state.dataframe = result.get("dataframe")
            state.verification = result.get("verification")
        else:
            parts = []
            all_sources = []
            contexts = []
            confidences = []
            dataframe = None
            image_result = None
            vision_result = None
            for result in outputs:
                title = result.get("route", "result").capitalize()
                parts.append(f"### {title}\n{result.get('answer', '')}")
                all_sources.extend(result.get("sources", []))
                if result.get("context"):
                    contexts.append(result["context"])
                if result.get("dataframe") is not None:
                    dataframe = result["dataframe"]
                if result.get("route") == "image":
                    image_result = result
                if result.get("route") == "vision_analysis":
                    vision_result = result
                conf = (result.get("verification") or {}).get("confidence")
                if conf is not None:
                    confidences.append(conf)

            if image_result and vision_result:
                cross_check = self.verifier.cross_check_vision(image_result.get("answer", ""), vision_result)
                if cross_check:
                    parts.append(f"### Cross-check\n⚠️ {cross_check}")

            state.answer = "\n\n".join(parts)
            state.sources = sorted(set(all_sources))
            state.context = "\n\n".join(contexts)
            state.route = "planner"
            state.dataframe = dataframe
            avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
            force_llm_check = avg_confidence is not None and avg_confidence < config.FINAL_VERIFICATION_FORCE_LLM_THRESHOLD
            state.verification = self.verifier.verify(state.answer, state.context, state.sources, force_llm_check=force_llm_check)

        state.citations = list(state.sources)
        state.confidence = (state.verification or {}).get("confidence")
        return state.to_result()

    def _node_analyze(self, state: AgentState) -> AgentState:
        self.supervisor.analyze(state["shared"])
        return state

    def _node_plan(self, state: AgentState) -> AgentState:
        self.supervisor.build_plan(state["shared"])
        return state

    def _node_execute(self, state: AgentState) -> AgentState:
        start = time.time()
        self.supervisor.execute(state["shared"], self._dispatch_step)
        state["shared"].execution_time = round(time.time() - start, 3)
        return state

    def _node_finalize(self, state: AgentState) -> AgentState:
        shared = state["shared"]
        result = self._finalize_from_outputs(shared)
        shared.answer = result["answer"]
        shared.sources = result["sources"]
        shared.route = result["route"]
        shared.context = result["context"]
        shared.dataframe = result["dataframe"]
        shared.verification = result["verification"]
        shared.confidence = result["confidence"]
        return state

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("analyze", self._node_analyze)
        graph.add_node("plan", self._node_plan)
        graph.add_node("execute", self._node_execute)
        graph.add_node("finalize", self._node_finalize)
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def prepare_stream(self, question: str, target_files: list = None, forced_route: str = None, memory: MemoryAgent = None) -> dict:
        mem = memory or self.memory
        cache_key = self.cache_manager.response_key(question, forced_route, target_files)
        cached = self.cache_manager.get_response(cache_key) or mem.cache_get(cache_key)
        if cached:
            return {"streamable": False, "cached": True, "result": cached}

        route = forced_route
        if not route:
            classified = self.router.classify(question)
            route = classified["route"]
            if not target_files:
                target_files = classified["target_files"]

        if not forced_route and self.planner.is_compound(question):
            return {"streamable": False, "route": "planner"}

        history_str = self._format_history(mem.get_history())

        if route == "summarize":
            files = self.retrieval.resolve_target_files(question, target_files or [], min_files=1)
            max_per_file = config.SUMMARY_MAX_CHUNKS_PER_FILE
            parts = [
                f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks)
                for f in files
                if (chunks := self.retrieval.fetch_file_chunks(f, max_per_file))
            ]
            context = "\n\n".join(parts) if parts else "(no content found for the requested file(s))"
            mode = self.summarizer._infer_mode(question)
            return {
                "streamable": True,
                "route": "summarize",
                "sources": files,
                "context": context,
                "chain": self.summarizer.chain,
                "inputs": {"context": context, "question": question, "mode": mode},
            }

        if route == "compare":
            files = self.retrieval.resolve_target_files(question, target_files or [], min_files=2)
            max_per_file = config.COMPARE_MAX_CHUNKS_PER_FILE
            parts = [
                f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks)
                for f in files
                if (chunks := self.retrieval.fetch_file_chunks(f, max_per_file))
            ]
            context = "\n\n".join(parts) if parts else "(not enough distinct files found to compare)"
            return {
                "streamable": True,
                "route": "compare",
                "sources": files,
                "context": context,
                "chain": self.comparison.chain,
                "inputs": {"context": context, "question": question},
            }

        if route == "code":
            action = self.coding._infer_action(question)
            code = self.coding._extract_code_block(question)
            chain_map = {
                "generate": (self.coding.generate_chain, {"question": question, "context": ""}),
                "explain": (self.coding.explain_chain, {"code": code}),
                "debug": (self.coding.debug_chain, {"code": code, "error": ""}),
                "optimize": (self.coding.optimize_chain, {"code": code, "goal": "general"}),
            }
            chain, inputs = chain_map[action]
            return {"streamable": True, "route": "code", "sources": [], "context": "", "chain": chain, "inputs": inputs}

        if route in ("table", "image"):
            return {"streamable": False, "route": route}

        artifacts = self.retrieval.pipeline.retrieve(
            question=question,
            source_filter=target_files or None,
            history=mem.get_history(),
            use_query_rewrite=True,
        )
        resolved_question = artifacts.query or question
        if artifacts.max_score >= config.SIMILARITY_THRESHOLD:
            context = self.retrieval.pipeline.context_builder.build(artifacts.docs)
            sources = sorted({d.metadata.get("source_file", "unknown") for d in artifacts.docs})
            return {
                "streamable": True,
                "route": "local",
                "sources": sources,
                "context": context,
                "chain": self.qa.chain,
                "inputs": {"context": context, "question": resolved_question, "history": history_str},
            }

        # Same fix as the local branch: web search must use the
        # history-resolved query, not the raw follow-up text ("explain more
        # about it" has no topic in it on its own).
        results = web_search(resolved_question, config.WEB_SEARCH_RESULTS)
        context = format_results_as_context(results)
        sources = [f"{r['title']} ({r['link']})" for r in results]
        return {
            "streamable": True,
            "route": "web",
            "sources": sources,
            "context": context,
            "chain": self.qa.web_chain,
            "inputs": {"context": context, "question": resolved_question, "history": history_str},
        }

    def finalize(
        self,
        question: str,
        answer: str,
        route: str,
        sources: list,
        context: str = "",
        target_files: list = None,
        forced_route: str = None,
        memory: MemoryAgent = None,
    ) -> dict:
        mem = memory or self.memory
        verification = None
        if config.VERIFY_ANSWERS:
            verification = self.verifier.verify(answer, context, sources)
            if (
                verification.get("confidence") is not None
                and verification["confidence"] < config.FINAL_VERIFICATION_FORCE_LLM_THRESHOLD
            ):
                verification = self.verifier.verify(answer, context, sources, force_llm_check=True)
        result = {
            "answer": answer,
            "sources": sources,
            "route": route,
            "context": context,
            "verification": verification,
            "dataframe": None,
            "trace": [],
            "execution_plan": {"rationale": "streamed direct path", "target_files": target_files or [], "steps": []},
            "cache_hits": {},
            "execution_time": 0.0,
            "token_usage": {},
            "confidence": (verification or {}).get("confidence"),
        }
        mem.add_turn(question, answer)
        cache_key = self.cache_manager.response_key(question, forced_route, target_files)
        mem.cache_set(cache_key, result)
        self.cache_manager.set_response(cache_key, result)
        return result

    def answer_about_uploaded_image(self, image_bytes: bytes, question: str, mime: str = "image/png", memory: MemoryAgent = None) -> dict:
        mem = memory or self.memory
        start = time.time()
        result = self.image.answer_question_bytes(image_bytes, question, mime) if question and question.strip() else self.image.describe_bytes(image_bytes, mime)
        full_result = {
            "answer": result["answer"],
            "sources": result["sources"],
            "route": "image",
            "context": "",
            "verification": None,
            "dataframe": None,
            "trace": [
                {
                    "agent": "image",
                    "model": self.model_router.resolve_name("image"),
                    "detail": question or "Describe uploaded image",
                    "started_at": start,
                    "ended_at": time.time(),
                    "status": "completed",
                    "confidence": None,
                    "cache_hit": False,
                    "metadata": {},
                }
            ],
            "execution_plan": {"rationale": "ad-hoc image analysis", "target_files": [], "steps": []},
            "cache_hits": {},
            "execution_time": round(time.time() - start, 3),
            "token_usage": {},
            "confidence": None,
        }
        mem.add_turn(question or "[attached an image]", result["answer"])
        return full_result

    def analyze_uploaded_image(self, image_bytes: bytes, memory: MemoryAgent = None,
                                run_vlm_description: bool = True) -> dict:
        """
        Runs real object detection (VisionAnalysisAgent) on a chat-attached
        image, bypassing the graph entirely -- same bypass pattern as
        answer_about_uploaded_image(). Optionally also runs the VLM
        description and the cross-check between the two, so a single call
        can produce the full "GenAI description vs. real CV measurement"
        comparison that's the point of this feature.
        """
        mem = memory or self.memory
        start = time.time()

        if self.vision_analysis is None or not self.vision_analysis.available:
            result = {
                "answer": "Object detection isn't available (ultralytics not installed on this deployment).",
                "sources": [], "route": "vision_analysis", "context": "", "dataframe": None,
                "verification": None, "trace": [], "execution_plan": {}, "cache_hits": {},
                "execution_time": round(time.time() - start, 3), "token_usage": {}, "confidence": None,
            }
            return result

        detection = self.vision_analysis.analyze_bytes(image_bytes, source_label="attached image")
        answer_parts = [f"### Vision Analysis (YOLO)\n{detection['answer']}"]

        cross_check_note = None
        if run_vlm_description:
            try:
                vlm = self.image.describe_bytes(image_bytes, "image/png")
                answer_parts.append(f"### Vision-Language Description\n{vlm['answer']}")
                cross_check_note = self.verifier.cross_check_vision(vlm["answer"], detection)
                if cross_check_note:
                    answer_parts.append(f"### Cross-check\n⚠️ {cross_check_note}")
            except Exception as exc:
                logger.warning(f"VLM description failed; returning detection only: {exc}")
                answer_parts.append("### Vision-Language Description\n_Unavailable right now (vision model did not respond). Detection results above are still valid._")

        combined_answer = "\n\n".join(answer_parts)
        full_result = {
            "answer": combined_answer,
            "sources": ["attached image"],
            "route": "vision_analysis",
            "context": "",
            "verification": None,
            "dataframe": detection.get("dataframe"),
            "trace": [
                {
                    "agent": "vision_analysis", "model": detection.get("model", "yolo"),
                    "detail": "Real object detection", "started_at": start, "ended_at": time.time(),
                    "status": "completed", "confidence": None, "cache_hit": False,
                    "metadata": {"inference_seconds": detection.get("inference_seconds")},
                },
            ],
            "execution_plan": {"rationale": "ad-hoc vision analysis", "target_files": [], "steps": []},
            "cache_hits": {}, "execution_time": round(time.time() - start, 3),
            "token_usage": {}, "confidence": None,
        }
        mem.add_turn("[analyzed an attached image with object detection]", detection["answer"])
        return full_result

    def ask(self, question: str, target_files: list = None, forced_route: str = None, memory: MemoryAgent = None) -> dict:
        mem = memory or self.memory
        cache_key = self.cache_manager.response_key(question, forced_route, target_files)
        cached = self.cache_manager.get_response(cache_key) or mem.cache_get(cache_key)
        if cached:
            logger.info("Response cache HIT.")
            return cached

        shared = SharedAgentState(
            question=question,
            history=mem.get_history(),
            target_files=list(target_files or []),
            forced_route=forced_route,
        )
        start = time.time()
        max_attempts = max(1, int(config.NVIDIA_REQUEST_RETRIES) + 1)
        result_state = None
        for attempt in range(1, max_attempts + 1):
            try:
                result_state = self.graph.invoke({"shared": shared})
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as exc:
                if attempt == max_attempts:
                    logger.error("ask() timed out after %s attempts: %s", max_attempts, exc)
                    raise
                backoff = float(config.NVIDIA_RETRY_BACKOFF_SECONDS) * attempt
                logger.warning(
                    "ask() timeout on attempt %s/%s, retrying in %.1fs",
                    attempt,
                    max_attempts,
                    backoff,
                )
                time.sleep(backoff)

        shared = result_state["shared"]
        shared.execution_time = round(time.time() - start, 3)
        result = shared.to_result()

        mem.add_turn(question, result["answer"])
        mem.cache_set(cache_key, result)
        self.cache_manager.set_response(cache_key, result)
        logger.info("ask() route=%s total=%.3fs", result.get("route"), shared.execution_time)
        return result
