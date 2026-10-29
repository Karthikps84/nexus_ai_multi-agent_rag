import pytest

from agents.planner import PlannerAgent


@pytest.fixture
def planner(fake_llm):
    return PlannerAgent(fake_llm)


@pytest.mark.parametrize(
    "q",
    [
        "Which tool creates release pull requests and the changelog?",
        "What are the pros and cons of hybrid search in this system?",
        "Short question and more",
    ],
)
def test_single_question_with_and_is_not_compound(planner, q):
    assert planner.is_compound(q) is False


@pytest.mark.parametrize(
    "q",
    [
        "Summarize the first paper and compare it with the second paper",
        "List the main agents and then explain how the verifier works",
        "What is BM25? What is a re-ranker in this pipeline?",
        "Extract the table of results; also tell me which model scored best",
    ],
)
def test_multi_task_questions_are_compound(planner, q):
    assert planner.is_compound(q) is True
