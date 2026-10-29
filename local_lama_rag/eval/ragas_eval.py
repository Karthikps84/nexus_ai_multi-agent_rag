"""
==========================================================
Evaluation Harness
==========================================================
Runs the refactored system against a small eval set and reports:
- RAGAS: faithfulness, answer relevancy, context precision, context recall
- DeepEval (optional): answer relevancy, faithfulness, contextual precision
- Runtime metrics: latency, confidence, citations

Usage:
    python eval/ragas_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.workflow import MultiAgentSystem
from utils.logger import get_logger

logger = get_logger("eval")
EVAL_SET_PATH = Path(__file__).with_name("eval_set.json")


def load_eval_set() -> list[dict]:
    if EVAL_SET_PATH.exists():
        return json.loads(EVAL_SET_PATH.read_text())
    return [
        {"question": "What is self attention?", "ground_truth": "Replace with a grounded answer from your corpus."},
        {"question": "Summarize the main contribution of the paper.", "ground_truth": "Replace with a grounded summary."},
    ]


def run_cases(system: MultiAgentSystem, eval_set: list[dict]) -> list[dict]:
    rows = []
    for item in eval_set:
        start = time.time()
        result = system.ask(item["question"])
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "context": result.get("context", ""),
                "latency_seconds": round(time.time() - start, 3),
                "confidence": (result.get("verification") or {}).get("confidence"),
                "sources": result.get("sources", []),
            }
        )
    return rows


def run_ragas(rows: list[dict]):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    system = MultiAgentSystem()
    system.initialize()
    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [[row["context"]] for row in rows],
            "ground_truth": [row["ground_truth"] for row in rows],
        }
    )
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(system.models.fast),
        embeddings=LangchainEmbeddingsWrapper(system.embeddings),
    )


def run_deepeval(rows: list[dict]):
    from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    metrics = {
        "answer_relevancy": AnswerRelevancyMetric(),
        "faithfulness": FaithfulnessMetric(),
        "contextual_precision": ContextualPrecisionMetric(),
    }
    aggregate = {name: [] for name in metrics}
    for row in rows:
        case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=[row["context"]],
            expected_output=row["ground_truth"],
        )
        for name, metric in metrics.items():
            metric.measure(case)
            aggregate[name].append(metric.score)
    return {name: round(sum(scores) / len(scores), 4) for name, scores in aggregate.items() if scores}


def summarize_runtime(rows: list[dict]) -> dict:
    confidences = [row["confidence"] for row in rows if row["confidence"] is not None]
    citation_hits = [1 for row in rows if row["sources"]]
    return {
        "avg_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / max(len(rows), 1), 3),
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "citation_coverage": round(sum(citation_hits) / max(len(rows), 1), 3),
    }


def main():
    eval_set = load_eval_set()
    system = MultiAgentSystem()
    system.initialize()
    rows = run_cases(system, eval_set)

    print("Runtime metrics")
    print(json.dumps(summarize_runtime(rows), indent=2))

    try:
        print("\nRAGAS")
        print(run_ragas(rows))
    except ImportError:
        logger.warning("ragas/datasets not installed; skipping RAGAS metrics.")

    try:
        print("\nDeepEval")
        print(json.dumps(run_deepeval(rows), indent=2))
    except ImportError:
        logger.warning("deepeval not installed; skipping DeepEval metrics.")


if __name__ == "__main__":
    main()
