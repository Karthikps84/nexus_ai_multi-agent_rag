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
