import pytest
from fastapi.testclient import TestClient

from api import main as api_main


class FakeSystem:
    def __init__(self):
        self.retrieval = type("R", (), {"list_known_files": staticmethod(lambda: ["a.pdf", "b.pdf"])})()

    def initialize(self, on_progress=None):
        pass

    def ask(self, question, target_files=None, forced_route=None, memory=None):
        return {
            "answer": f"answer to: {question}",
            "sources": ["a.pdf"],
            "route": "local",
            "context": "some context",
            "dataframe": None,
            "verification": {"confidence": 0.8},
            "max_score": 0.9,
            "docs": [],
            "execution_plan": {"rationale": "", "target_files": [], "steps": []},
            "citations": ["a.pdf"],
            "confidence": 0.8,
            "execution_time": 0.1,
            "cache_hits": {},
            "token_usage": {},
            "trace": [],
            "agent_outputs": {},
        }

    def answer_about_uploaded_image(self, image_bytes, question, mime="image/png", memory=None):
        base = self.ask(question or "[image]")
        base["route"] = "image"
        return base

    def analyze_uploaded_image(self, image_bytes, memory=None, run_vlm_description=True):
        base = self.ask("[vision analysis]")
        base["route"] = "vision_analysis"
        return base


@pytest.fixture
def client(monkeypatch):
    fake_system = FakeSystem()
    monkeypatch.setattr(api_main, "MultiAgentSystem", lambda: fake_system)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["initialized"] is True


def test_create_session_returns_id(client):
    resp = client.post("/sessions")
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_list_files(client):
    resp = client.get("/files")
    assert resp.status_code == 200
    assert resp.json()["files"] == ["a.pdf", "b.pdf"]


def test_ask_returns_answer(client):
    session_id = client.post("/sessions").json()["session_id"]
    resp = client.post("/ask", json={"session_id": session_id, "question": "What is X?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer to: What is X?"
    assert body["route"] == "local"


def test_ask_creates_session_on_the_fly_if_unknown(client):
    resp = client.post("/ask", json={"session_id": "does-not-exist-yet", "question": "hi"})
    assert resp.status_code == 200


def test_ask_image(client):
    session_id = client.post("/sessions").json()["session_id"]
    files = {"file": ("test.png", b"fake-bytes", "image/png")}
    resp = client.post("/ask/image", data={"session_id": session_id, "question": "what's this?"}, files=files)
    assert resp.status_code == 200
    assert resp.json()["route"] == "image"


def test_ask_vision_analysis(client):
    session_id = client.post("/sessions").json()["session_id"]
    files = {"file": ("test.png", b"fake-bytes", "image/png")}
    resp = client.post("/ask/vision-analysis", data={"session_id": session_id}, files=files)
    assert resp.status_code == 200
    assert resp.json()["route"] == "vision_analysis"


def test_upload_saves_supported_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.config, "DATA_DIR", tmp_path)
    files = {"files": ("note.txt", b"hello world", "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == ["note.txt"]
    assert body["skipped"] == []
    assert (tmp_path / "note.txt").read_bytes() == b"hello world"


def test_upload_skips_unsupported_extension(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.config, "DATA_DIR", tmp_path)
    files = {"files": ("script.exe", b"binary", "application/octet-stream")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == []
    assert body["skipped"] == ["script.exe"]
    assert not (tmp_path / "script.exe").exists()
