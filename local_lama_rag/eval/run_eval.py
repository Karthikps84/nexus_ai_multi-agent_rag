"""
==========================================================
Full Evaluation Runner
==========================================================
Runs your indexed system against eval/eval_set.json and
reports all metrics. Three metric layers:

Layer 1 — Runtime (always, no extra deps):
  latency, confidence, citation coverage, answer length

Layer 2 — RAGAS (pip install ragas datasets):
  faithfulness, answer_relevancy, context_precision,
  context_recall

Layer 3 — DeepEval (pip install deepeval):
  answer_relevancy, faithfulness, contextual_precision,
  hallucination

Usage:
    python eval/run_eval.py                     # all layers
    python eval/run_eval.py --skip-ragas        # skip RAGAS
    python eval/run_eval.py --skip-deepeval     # skip DeepEval
    python eval/run_eval.py --filter easy       # only easy questions
    python eval/run_eval.py --out eval/results  # output directory
==========================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.memory import MemoryAgent
from graph.workflow import MultiAgentSystem
from utils.logger import get_logger

logger = get_logger("run_eval")

EVAL_SET_PATH = Path(__file__).with_name("eval_set.json")


def load_eval_set(filter_difficulty: str | None = None) -> list[dict]:
    if not EVAL_SET_PATH.exists():
        print("ERROR: eval/eval_set.json not found.")
        print("Run:  python eval/build_eval_set.py")
        sys.exit(1)

    rows = json.loads(EVAL_SET_PATH.read_text())
    if filter_difficulty:
        rows = [r for r in rows if r.get("difficulty") == filter_difficulty]

    if not rows:
        print(f"No eval pairs found (filter: {filter_difficulty or 'none'})")
        sys.exit(1)

    print(f"Loaded {len(rows)} eval pair(s)")
    return rows


def run_cases(system: MultiAgentSystem, eval_set: list[dict]) -> list[dict]:
    """Run each question through the full agent pipeline and collect outputs."""
    results = []
    for i, item in enumerate(eval_set, 1):
        # Fresh memory per question -- we're evaluating single-turn correctness,
        # not conversation continuity.
        mem = MemoryAgent()
        start = time.time()
        try:
            result = system.ask(item["question"], memory=mem)
            latency = round(time.time() - start, 3)
            results.append({
                "question":         item["question"],
                "ground_truth":     item["ground_truth"],
                "answer":           result["answer"],
                "context":          result.get("context", ""),
                "sources":          result.get("sources", []),
                "route":            result.get("route", ""),
                "confidence":       (result.get("verification") or {}).get("confidence"),
                "latency_seconds":  latency,
                "source_file":      item.get("source_file", ""),
                "difficulty":       item.get("difficulty", ""),
                "cache_hits":       result.get("cache_hits", {}),
            })
            print(f"  [{i}/{len(eval_set)}] {latency:.2f}s  route={result.get('route')}  "
                  f"conf={((result.get('verification') or {}).get('confidence') or 0):.2f}")
        except Exception as e:
            logger.error(f"Question {i} failed: {e}")
            results.append({
                "question":        item["question"],
                "ground_truth":    item["ground_truth"],
                "answer":          f"ERROR: {e}",
                "context":         "",
                "sources":         [],
                "route":           "error",
                "confidence":      None,
                "latency_seconds": round(time.time() - start, 3),
                "source_file":     item.get("source_file", ""),
                "difficulty":      item.get("difficulty", ""),
                "cache_hits":      {},
            })

    return results


def runtime_metrics(results: list[dict]) -> dict:
    """Compute metrics that need no extra libraries."""
    n = len(results)
    latencies = [r["latency_seconds"] for r in results]
    confidences = [r["confidence"] for r in results if r["confidence"] is not None]
    cited = [r for r in results if r["sources"]]
    routed_local = [r for r in results if r["route"] in ("local", "qa")]

    # Lexical overlap between answer and ground truth (proxy for correctness
    # without a judge LLM -- weak but free and reproducible).
    def lex_f1(pred: str, ref: str) -> float:
        pred_toks = set(pred.lower().split())
        ref_toks  = set(ref.lower().split())
        if not pred_toks or not ref_toks:
            return 0.0
        prec = len(pred_toks & ref_toks) / len(pred_toks)
        rec  = len(pred_toks & ref_toks) / len(ref_toks)
        if prec + rec == 0:
            return 0.0
        return round(2 * prec * rec / (prec + rec), 4)

    lex_scores = [lex_f1(r["answer"], r["ground_truth"]) for r in results]
    errors = [r for r in results if r["route"] == "error"]

    return {
        "total_questions":        n,
        "error_count":            len(errors),
        "avg_latency_seconds":    round(sum(latencies) / max(n, 1), 3),
        "p50_latency_seconds":    round(sorted(latencies)[n // 2], 3),
        "p95_latency_seconds":    round(sorted(latencies)[int(n * 0.95)], 3),
        "avg_confidence":         round(sum(confidences) / len(confidences), 3) if confidences else None,
        "citation_coverage":      round(len(cited) / max(n, 1), 3),
        "local_routing_rate":     round(len(routed_local) / max(n, 1), 3),
        "avg_lexical_f1":         round(sum(lex_scores) / max(n, 1), 4),
        # per-difficulty breakdown
        "by_difficulty": {
            d: {
                "count":      sum(1 for r in results if r["difficulty"] == d),
                "avg_lex_f1": round(sum(lex_f1(r["answer"], r["ground_truth"]) for r in results if r["difficulty"] == d)
                                    / max(sum(1 for r in results if r["difficulty"] == d), 1), 4),
            }
            for d in ("easy", "medium", "hard")
        },
    }


def run_ragas_metrics(results: list[dict], system: MultiAgentSystem) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy, context_precision, context_recall, faithfulness,
        )
    except ImportError:
        logger.warning("ragas/datasets not installed -- skipping. pip install ragas datasets")
        return None

    dataset = Dataset.from_dict({
        "question":     [r["question"] for r in results],
        "answer":       [r["answer"] for r in results],
        "contexts":     [[r["context"]] if r["context"] else [""] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    try:
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(system.models.fast),
            embeddings=LangchainEmbeddingsWrapper(system.embeddings),
        )
        return {k: round(float(v), 4) for k, v in score.items() if v is not None}
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}


def run_deepeval_metrics(results: list[dict]) -> dict | None:
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning("deepeval not installed -- skipping. pip install deepeval")
        return None

    metrics = {
        "answer_relevancy":    AnswerRelevancyMetric(threshold=0.5),
        "faithfulness":        FaithfulnessMetric(threshold=0.5),
        "contextual_precision": ContextualPrecisionMetric(threshold=0.5),
        "hallucination":       HallucinationMetric(threshold=0.5),
    }
    aggregate: dict[str, list[float]] = {k: [] for k in metrics}

    for r in results:
        if not r["context"]:
            continue  # DeepEval needs context -- skip web-fallback answers
        case = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            retrieval_context=[r["context"]],
            expected_output=r["ground_truth"],
        )
        for name, metric in metrics.items():
            try:
                metric.measure(case)
                aggregate[name].append(metric.score)
            except Exception as e:
                logger.warning(f"DeepEval metric {name} failed on one case: {e}")

    return {
        name: round(sum(scores) / len(scores), 4)
        for name, scores in aggregate.items()
        if scores
    }


def write_report(out_dir: Path, results: list[dict], runtime: dict,
                 ragas: dict | None, deepeval: dict | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Full results as JSON
    (out_dir / f"results_{timestamp}.json").write_text(
        json.dumps({"results": results, "runtime": runtime,
                    "ragas": ragas, "deepeval": deepeval}, indent=2)
    )

    # Human-readable report
    lines = [
        "# RAG Evaluation Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Questions: {runtime['total_questions']}",
        "",
        "## Runtime Metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg latency | {runtime['avg_latency_seconds']}s |",
        f"| p50 latency | {runtime['p50_latency_seconds']}s |",
        f"| p95 latency | {runtime['p95_latency_seconds']}s |",
        f"| Avg confidence | {runtime['avg_confidence'] or 'n/a'} |",
        f"| Citation coverage | {runtime['citation_coverage']:.0%} |",
        f"| Local routing rate | {runtime['local_routing_rate']:.0%} |",
        f"| Avg lexical F1 | {runtime['avg_lexical_f1']} |",
        f"| Errors | {runtime['error_count']} |",
        "",
        "### By difficulty",
        "| Difficulty | Count | Avg Lex F1 |",
        "|------------|-------|------------|",
    ]
    for d, s in runtime["by_difficulty"].items():
        lines.append(f"| {d} | {s['count']} | {s['avg_lex_f1']} |")

    if ragas:
        lines += ["", "## RAGAS Metrics",
                  "| Metric | Score |", "|--------|-------|"]
        for k, v in ragas.items():
            lines.append(f"| {k} | {v} |")

    if deepeval:
        lines += ["", "## DeepEval Metrics",
                  "| Metric | Score |", "|--------|-------|"]
        for k, v in deepeval.items():
            lines.append(f"| {k} | {v} |")

    lines += ["", "## Per-Question Results",
              "| # | Q | Route | Lex F1 | Conf | Latency |",
              "|---|---|-------|--------|------|---------|"]
    for i, r in enumerate(results, 1):
        q = r["question"][:50] + ("…" if len(r["question"]) > 50 else "")
        pred_toks = set(r["answer"].lower().split())
        ref_toks  = set(r["ground_truth"].lower().split())
        prec = len(pred_toks & ref_toks) / max(len(pred_toks), 1)
        rec  = len(pred_toks & ref_toks) / max(len(ref_toks), 1)
        f1   = round(2 * prec * rec / max(prec + rec, 1e-9), 3)
        conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "n/a"
        lines.append(f"| {i} | {q} | {r['route']} | {f1} | {conf} | {r['latency_seconds']}s |")

    report_path = out_dir / f"report_{timestamp}.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ragas",    action="store_true")
    parser.add_argument("--skip-deepeval", action="store_true")
    parser.add_argument("--filter", choices=["easy", "medium", "hard"],
                        help="Only run questions of this difficulty")
    parser.add_argument("--out", default="eval/results", help="Output directory")
    args = parser.parse_args()

    eval_set = load_eval_set(filter_difficulty=args.filter)

    print("Initializing system…")
    system = MultiAgentSystem()
    system.initialize()

    print(f"\nRunning {len(eval_set)} question(s)…")
    results = run_cases(system, eval_set)

    print("\nComputing runtime metrics…")
    runtime = runtime_metrics(results)
    print(json.dumps(runtime, indent=2))

    ragas = None
    if not args.skip_ragas:
        print("\nRunning RAGAS…")
        ragas = run_ragas_metrics(results, system)
        if ragas:
            print(json.dumps(ragas, indent=2))

    deepeval_scores = None
    if not args.skip_deepeval:
        print("\nRunning DeepEval…")
        deepeval_scores = run_deepeval_metrics(results)
        if deepeval_scores:
            print(json.dumps(deepeval_scores, indent=2))

    write_report(Path(args.out), results, runtime, ragas, deepeval_scores)


if __name__ == "__main__":
    main()
