import importlib.util
import time
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("ci_eval", Path(__file__).resolve().parent.parent / "eval" / "ci_eval.py")
ci_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_eval)


ITEM = {"must_contain": [["bm25"], ["0.30", "0.3"]], "expect_source": "retrieval.md"}


def test_score_answer_passes_when_keywords_and_source_present():
    ok, problems = ci_eval.score_answer(ITEM, "It uses BM25 with threshold 0.3.", ["retrieval.md"])
    assert ok and problems == []


def test_score_answer_reports_missing_keyword_and_source():
    ok, problems = ci_eval.score_answer(ITEM, "It uses BM25.", ["web page"])
    assert not ok
    assert "missing 0.30 / 0.3" in problems
    assert "did not cite retrieval.md" in problems


def test_call_with_timeout_returns_value_and_propagates_errors():
    assert ci_eval.call_with_timeout(lambda: 42, 5) == 42
    with pytest.raises(ValueError):
        ci_eval.call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), 5)


def test_call_with_timeout_gives_up_on_a_hung_call():
    start = time.time()
    with pytest.raises(TimeoutError):
        ci_eval.call_with_timeout(lambda: time.sleep(30), 0.3)
    assert time.time() - start < 5
