import pytest

from agents.router import RouterAgent


def _router(fake_llm, files=None):
    return RouterAgent(fake_llm, known_files_fn=lambda: files or [])


def test_classify_keywords_summarize(fake_llm):
    router = _router(fake_llm)
    assert router.classify_keywords("Give me a summary of this") == "summarize"


def test_classify_keywords_compare(fake_llm):
    router = _router(fake_llm)
    assert router.classify_keywords("Compare these two documents") == "compare"


def test_classify_keywords_vision_analysis_before_image(fake_llm):
    router = _router(fake_llm)
    # "how many objects" should route to vision_analysis, not the broader "image" bucket
    assert router.classify_keywords("How many objects are in this image?") == "vision_analysis"


def test_classify_keywords_code(fake_llm):
    router = _router(fake_llm)
    assert router.classify_keywords("Write a python function to sort a list") == "code"


def test_classify_keywords_no_match_returns_none(fake_llm):
    router = _router(fake_llm)
    assert router.classify_keywords("What is the capital of France?") is None


def test_classify_falls_back_to_qa_without_llm(fake_llm):
    router = _router(fake_llm)
    result = router.classify("What is the capital of France?", use_llm_fallback=False)
    assert result["route"] == "qa"
    assert result["method"] == "default"


def test_classify_uses_keyword_method_when_matched(fake_llm):
    router = _router(fake_llm)
    result = router.classify("Summarize this document")
    assert result["route"] == "summarize"
    assert result["method"] == "keyword"


def test_match_files_in_question(fake_llm):
    router = _router(fake_llm, files=["animals.pdf", "tech.pdf"])
    assert router.match_files_in_question("Tell me about animals.pdf") == ["animals.pdf"]
    assert router.match_files_in_question("Tell me something unrelated") == []


def test_docker_images_question_is_not_routed_to_image(fake_llm):
    router = _router(fake_llm, files=[])
    assert router.classify_keywords("Where are the Docker images published?") is None


@pytest.mark.parametrize(
    "q",
    ["Describe the image on page 2", "What does this chart show?", "Explain the diagram in the paper", "look at the attached image"],
)
def test_real_image_questions_still_route_to_image(fake_llm, q):
    assert _router(fake_llm, files=[]).classify_keywords(q) == "image"
