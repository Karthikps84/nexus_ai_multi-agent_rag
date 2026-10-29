import os

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

os.environ.setdefault("NVIDIA_API_KEY", "test-key-not-real")


class _Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM(RunnableLambda):
    """Minimal stand-in for a LangChain chat model: a real Runnable (so it
    composes with `|` in LCEL chains like the router's prompt | llm | parser)
    whose .invoke() returns a canned response wrapped like a chat message."""

    def __init__(self, response: str = "fake response"):
        self.response = response
        self.calls: list = []
        super().__init__(self._call)

    def _call(self, prompt):
        self.calls.append(prompt)
        return _Msg(self.response)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Cats are small domesticated carnivorous mammals.", metadata={"source_file": "animals.pdf"}),
        Document(page_content="Dogs are loyal companions and have been domesticated for millennia.", metadata={"source_file": "animals.pdf"}),
        Document(page_content="Python is a high-level programming language known for readability.", metadata={"source_file": "tech.pdf"}),
        Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source_file": "travel.pdf"}),
    ]
