"""Deterministic answer-quality gate for CI.

Builds a throwaway index from eval/ci_docs, asks every question in
eval/ci_eval_set.json through the real pipeline, and checks that the answer
contains the expected keywords and cites the expected source. Exits 1 when the
pass rate falls below --min-score, so a prompt/model/config change that
silently hurts quality turns the scheduled job red.

    python eval/ci_eval.py --min-score 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ATTEMPTS = 3  # per question, only retried when the call itself errors
QUESTION_TIMEOUT = 90  # seconds per attempt; a hung free-tier call must not hang CI
sys.path.insert(0, str(ROOT.parent))


def score_answer(item: dict, answer: str, sources: list[str]) -> tuple[bool, list[str]]:
    text = answer.lower()
    problems = []
    for group in item["must_contain"]:
        if not any(alt.lower() in text for alt in group):
            problems.append(f"missing {' / '.join(group)}")
    expected = item.get("expect_source")
    if expected and not any(expected in s for s in sources):
        problems.append(f"did not cite {expected}")
    return not problems, problems


def call_with_timeout(fn, timeout: float):
    """Run fn() in a daemon thread; raise TimeoutError if it takes longer than `timeout`.
    A stuck thread can't be killed, but being a daemon it never blocks interpreter exit."""
    box: dict = {}

    def runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--out", default="ci_eval_results.json")
    args = parser.parse_args()

    from graph.workflow import MultiAgentSystem

    items = json.loads((ROOT / "ci_eval_set.json").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="nexus-ci-eval-"))
    try:
        data_dir, storage_dir = tmp / "data", tmp / "storage"
        shutil.copytree(ROOT / "ci_docs", data_dir)
        system = MultiAgentSystem(data_dir=data_dir, storage_dir=storage_dir)
        system.initialize()

        results = []
        for item in items:
            t0 = time.time()
            answer, sources, ok, problems = "", [], False, []
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    out = call_with_timeout(lambda: system.ask(item["question"]), QUESTION_TIMEOUT)
                    answer, sources = str(out.get("answer", "")), list(out.get("sources", []))
                    ok, problems = score_answer(item, answer, sources)
                    break
                except Exception as exc:  # free NIM endpoints throw transient 404/429/5xx
                    problems = [f"error: {str(exc).splitlines()[0][:120]}"]
                    if attempt < ATTEMPTS:
                        time.sleep(3 * attempt)
            results.append(
                {"question": item["question"], "passed": ok, "problems": problems,
                 "seconds": round(time.time() - t0, 1), "answer": answer[:300]}
            )
            print(f"[{'PASS' if ok else 'FAIL'}] {item['question']}  ({results[-1]['seconds']}s)"
                  + ("" if ok else f"  -> {'; '.join(problems)}"), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(r["passed"] for r in results)
    rate = passed / len(results)
    print(f"\nScore: {passed}/{len(results)} = {rate:.0%} (minimum {args.min_score:.0%})")
    Path(args.out).write_text(json.dumps({"score": rate, "results": results}, indent=2))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        lines = [f"### Answer quality: {passed}/{len(results)} ({rate:.0%})", "", "| | Question | Note |", "|---|---|---|"]
        lines += [f"| {'✅' if r['passed'] else '❌'} | {r['question']} | {'; '.join(r['problems'])} |" for r in results]
        with open(summary, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if rate >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
