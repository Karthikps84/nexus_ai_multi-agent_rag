import json

import pytest
from fastapi.testclient import TestClient

import config
from api import main as api_main
from api.sessions import sessions
from api.systems import systems


class _Chain:
    def stream(self, inputs):
        yield from ["Hello ", "streamed ", "world"]


class FakeSystem:
    """Stand-in for MultiAgentSystem backed by a real per-user data dir."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self.removed = []
        self.initialized = 0
        self.vs_manager = type("VS", (), {"data_dir": data_dir})()
        self.retrieval = type(
            "R", (), {"list_known_files": staticmethod(lambda: sorted(p.name for p in data_dir.glob("*") if p.is_file()))}
        )()
        self.stream_mode = "stream"

    def initialize(self, on_progress=None):
        self.initialized += 1

    def remove_document(self, name):
        self.removed.append(name)

    def _result(self, question):
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

    def ask(self, question, target_files=None, forced_route=None, memory=None):
        return self._result(question)

    def prepare_stream(self, question, target_files=None, forced_route=None, memory=None):
        if self.stream_mode == "stream":
            return {"streamable": True, "route": "local", "sources": ["a.pdf"], "context": "ctx",
                    "chain": _Chain(), "inputs": {}, "docs": []}
        return {"streamable": False, "route": "planner"}

    def finalize(self, question, answer, route, sources, context="", target_files=None, forced_route=None, memory=None):
        r = self._result(question)
        r.update(answer=answer, route=route, sources=sources)
        return r

    def answer_about_uploaded_image(self, image_bytes, question, mime="image/png", memory=None):
        base = self._result(question or "[image]")
        base["route"] = "image"
        return base

    def analyze_uploaded_image(self, image_bytes, memory=None, run_vlm_description=True):
        base = self._result("[vision analysis]")
        base["route"] = "vision_analysis"
        return base

    vision_analysis = None


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USERS_ROOT", tmp_path / "users")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "shared_data")
    fakes = {}

    def fake_get(user):
        if user.id not in fakes:
            fakes[user.id] = FakeSystem(user.data_dir)
        return fakes[user.id]

    monkeypatch.setattr(systems, "get", fake_get)
    sessions._sessions.clear()
    with TestClient(api_main.app) as c:
        c.fakes = fakes
        yield c


def _register(client, name="alice", password="correct horse"):
    r = client.post("/auth/register", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health_ok(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "initialized": True}


def test_endpoints_require_login(client):
    assert client.get("/files").status_code == 401
    assert client.post("/sessions").status_code == 401
    assert client.post("/ask", json={"session_id": "x", "question": "hi"}).status_code == 401
    assert client.get("/files", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_register_login_and_me(client):
    headers = _register(client)
    assert client.get("/auth/me", headers=headers).json() == {"username": "alice"}
    ok = client.post("/auth/login", json={"username": "ALICE", "password": "correct horse"})
    assert ok.status_code == 200
    bad = client.post("/auth/login", json={"username": "alice", "password": "wrong password"})
    assert bad.status_code == 401


def test_register_validation(client):
    _register(client)
    assert client.post("/auth/register", json={"username": "alice", "password": "another pass"}).status_code == 400
    assert client.post("/auth/register", json={"username": "bob", "password": "short"}).status_code == 400
    assert client.post("/auth/register", json={"username": "a b!", "password": "long enough pw"}).status_code == 400


def test_register_seeds_sample_doc(client, tmp_path):
    (tmp_path / "shared_data").mkdir()
    (tmp_path / "shared_data" / "sample.md").write_text("# sample")
    _register(client, "carol")
    seeded = list((tmp_path / "users").glob("*/data/sample.md"))
    assert len(seeded) == 1


def test_expired_token_rejected(client):
    from api.auth import User, issue_token

    token = issue_token(User(id="u1", username="x"), ttl=-10)
    assert client.get("/files", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_list_files(client):
    headers = _register(client)
    client.get("/files", headers=headers)  # creates the fake system
    fake = next(iter(client.fakes.values()))
    (fake.data_dir / "a.pdf").write_bytes(b"x")
    (fake.data_dir / ".DS_Store").write_bytes(b"x")
    body = client.get("/files", headers=headers).json()
    assert body["files"] == ["a.pdf"]
    assert body["details"][0]["kind"] == "pdf"


def test_ask_returns_answer(client):
    headers = _register(client)
    sid = client.post("/sessions", headers=headers).json()["session_id"]
    body = client.post("/ask", json={"session_id": sid, "question": "What is X?"}, headers=headers).json()
    assert body["answer"] == "answer to: What is X?"
    assert body["route"] == "local"


def test_ask_creates_session_on_the_fly_if_unknown(client):
    headers = _register(client)
    resp = client.post("/ask", json={"session_id": "does-not-exist-yet", "question": "hi"}, headers=headers)
    assert resp.status_code == 200


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_ask_stream_sends_tokens_then_done(client):
    headers = _register(client)
    r = client.post("/ask/stream", json={"session_id": "s", "question": "hi"}, headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert "".join(d["t"] for e, d in events if e == "token") == "Hello streamed world"
    done = events[-1][1]
    assert done["answer"] == "Hello streamed world"
    assert done["route"] == "local"


def test_ask_stream_falls_back_to_done_only(client):
    headers = _register(client)
    client.get("/files", headers=headers)
    next(iter(client.fakes.values())).stream_mode = "none"
    events = _parse_sse(client.post("/ask/stream", json={"session_id": "s", "question": "hi"}, headers=headers).text)
    assert [e for e, _ in events] == ["done"]
    assert events[0][1]["answer"] == "answer to: hi"


def test_ask_image(client):
    headers = _register(client)
    sid = client.post("/sessions", headers=headers).json()["session_id"]
    files = {"file": ("test.png", b"fake-bytes", "image/png")}
    resp = client.post("/ask/image", data={"session_id": sid, "question": "what's this?"}, files=files, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["route"] == "image"


def test_ask_vision_analysis(client):
    headers = _register(client)
    sid = client.post("/sessions", headers=headers).json()["session_id"]
    files = {"file": ("test.png", b"fake-bytes", "image/png")}
    resp = client.post("/ask/vision-analysis", data={"session_id": sid}, files=files, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["route"] == "vision_analysis"


def test_upload_saves_supported_file_into_user_dir(client):
    headers = _register(client)
    files = {"files": ("note.txt", b"hello world", "text/plain")}
    body = client.post("/upload", files=files, headers=headers).json()
    assert body["saved"] == ["note.txt"] and body["skipped"] == []
    fake = next(iter(client.fakes.values()))
    assert (fake.data_dir / "note.txt").read_bytes() == b"hello world"
    assert fake.initialized == 1


def test_upload_skips_unsupported_extension(client):
    headers = _register(client)
    files = {"files": ("script.exe", b"binary", "application/octet-stream")}
    body = client.post("/upload", files=files, headers=headers).json()
    assert body["saved"] == [] and body["skipped"] == ["script.exe"]


def test_upload_cannot_escape_data_dir(client, tmp_path):
    headers = _register(client)
    files = {"files": ("../../evil.txt", b"pwn", "text/plain")}
    body = client.post("/upload", files=files, headers=headers).json()
    assert body["saved"] == ["evil.txt"]
    assert not (tmp_path / "evil.txt").exists()
    fake = next(iter(client.fakes.values()))
    assert (fake.data_dir / "evil.txt").exists()


def test_delete_file(client):
    headers = _register(client)
    client.post("/upload", files={"files": ("note.txt", b"hi", "text/plain")}, headers=headers)
    fake = next(iter(client.fakes.values()))
    resp = client.delete("/files/note.txt", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "note.txt", "files": []}
    assert fake.removed == ["note.txt"]
    assert not (fake.data_dir / "note.txt").exists()
    assert client.delete("/files/note.txt", headers=headers).status_code == 404
    assert client.delete("/files/..%2Fsecret.txt", headers=headers).status_code in (400, 404)


def test_users_cannot_see_each_others_files_or_sessions(client):
    alice = _register(client, "alice")
    bob = _register(client, "bobby")
    client.post("/upload", files={"files": ("alice-only.txt", b"x", "text/plain")}, headers=alice)
    assert client.get("/files", headers=bob).json()["files"] == []
    assert client.delete("/files/alice-only.txt", headers=bob).status_code == 404
    assert client.get("/files", headers=alice).json()["files"] == ["alice-only.txt"]
    sid = client.post("/sessions", headers=alice).json()["session_id"]
    owners = {k[0] for k in sessions._sessions if k[1] == sid}
    assert len(owners) == 1
