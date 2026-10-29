from __future__ import annotations

import base64
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
from utils.logger import get_logger
from utils.uploads import safe_filename, save_upload_bytes

from api.auth import AuthError, User, decode_token, issue_token, users
from api.schemas import (
    AskRequest,
    AskResponse,
    AuthRequest,
    AuthResponse,
    DeleteResponse,
    FileInfo,
    FilesResponse,
    HealthResponse,
    MeResponse,
    SessionCreateResponse,
    UploadResponse,
)
from api.sessions import sessions
from api.systems import systems

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("NVIDIA_API_KEY"):
        raise EnvironmentError("NVIDIA_API_KEY must be set in your .env file")
    yield
    systems.clear()


app = FastAPI(title="Nexus AI API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_token(creds.credentials)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _serialize_docs(docs: list) -> list[dict]:
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


def _serialize_dataframe(df) -> list[dict] | None:
    if df is None:
        return None
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return None


def _normalize_result(result: dict) -> dict:
    # langchain_core >=1.x can return `answer` as a TextAccessor (a str
    # subclass that's also callable) rather than a plain str; Pydantic's
    # serializer rejects it, so coerce explicitly.
    result = dict(result)
    if "answer" in result and result["answer"] is not None:
        result["answer"] = str(result["answer"])
    # agent_outputs is self-referential (see api/schemas.py); never serialized.
    result.pop("agent_outputs", None)
    return result


def _finish(result: dict, docs=None) -> dict:
    result = _normalize_result(result)
    result["docs"] = _serialize_docs(docs if docs is not None else (result.get("docs") or []))
    result["dataframe"] = _serialize_dataframe(result.get("dataframe"))
    return result


def _data_dir(system):
    return system.vs_manager.data_dir


# ── public ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", initialized=True)


@app.post("/auth/register", response_model=AuthResponse)
def register(body: AuthRequest):
    try:
        user = users.register(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(token=issue_token(user), username=user.username)


@app.post("/auth/login", response_model=AuthResponse)
def login(body: AuthRequest):
    try:
        user = users.login(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(token=issue_token(user), username=user.username)


# ── authenticated ─────────────────────────────────────────────────────────
@app.get("/auth/me", response_model=MeResponse)
def me(user: User = Depends(current_user)):
    return MeResponse(username=user.username)


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(user: User = Depends(current_user)):
    return SessionCreateResponse(session_id=sessions.create(user.id))


@app.get("/files", response_model=FilesResponse)
def list_files(user: User = Depends(current_user)):
    system = systems.get(user)
    names = [
        n for n in system.retrieval.list_known_files()
        if (_data_dir(system) / n).suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]
    details = []
    for n in names:
        p = _data_dir(system) / n
        ext = p.suffix.lower()
        kind = "image" if ext in config.IMAGE_EXTENSIONS else (ext.lstrip(".") or "file")
        details.append(FileInfo(name=n, size_bytes=p.stat().st_size if p.exists() else 0, kind=kind))
    return FilesResponse(files=names, details=details)


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...), user: User = Depends(current_user)):
    system = await run_in_threadpool(systems.get, user)
    saved, skipped = [], []
    contents = [(f.filename or "upload", await f.read()) for f in files]

    def work():
        with systems.lock_for(user):
            for name, content in contents:
                if save_upload_bytes(name, content, _data_dir(system)):
                    saved.append(safe_filename(name))
                else:
                    skipped.append(name)
            if saved:
                system.initialize()

    try:
        await run_in_threadpool(work)
    except Exception as exc:
        logger.exception("Reindex after upload failed")
        raise HTTPException(status_code=500, detail=f"Saved but reindex failed: {exc}") from exc

    return UploadResponse(saved=saved, skipped=skipped, files_indexed=len(system.retrieval.list_known_files()))


@app.delete("/files/{name}", response_model=DeleteResponse)
def delete_file(name: str, user: User = Depends(current_user)):
    if safe_filename(name) != name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    system = systems.get(user)
    path = _data_dir(system) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    with systems.lock_for(user):
        path.unlink()
        system.remove_document(name)
    remaining = [
        n for n in system.retrieval.list_known_files()
        if (_data_dir(system) / n).suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]
    return DeleteResponse(deleted=name, files=remaining)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, user: User = Depends(current_user)):
    system = systems.get(user)
    memory = sessions.get(request.session_id, user.id)
    try:
        result = system.ask(
            question=request.question,
            target_files=request.target_files,
            forced_route=request.forced_route,
            memory=memory,
        )
    except Exception as exc:
        logger.exception("ask() failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**_finish(result))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/ask/stream")
def ask_stream(request: AskRequest, user: User = Depends(current_user)):
    """Server-sent events: `meta`, many `token`, then `done` (full AskResponse).
    Routes that can't stream (planner, table, image, cache hits) send only `done`."""
    system = systems.get(user)
    memory = sessions.get(request.session_id, user.id)

    def events():
        t0 = time.time()
        try:
            prep = system.prepare_stream(
                request.question,
                target_files=request.target_files,
                forced_route=request.forced_route,
                memory=memory,
            )
            if prep.get("cached"):
                yield _sse("done", AskResponse(**_finish(prep["result"])).model_dump())
            elif prep.get("streamable"):
                yield _sse("meta", {"route": prep["route"], "sources": prep["sources"]})
                answer = ""
                for chunk in prep["chain"].stream(prep["inputs"]):
                    text = str(chunk)
                    answer += text
                    yield _sse("token", {"t": text})
                result = system.finalize(
                    request.question, answer, prep["route"], prep["sources"],
                    context=prep.get("context", ""),
                    target_files=request.target_files,
                    forced_route=request.forced_route,
                    memory=memory,
                )
                result = _finish(result, docs=prep.get("docs") or [])
                result["execution_time"] = round(time.time() - t0, 3)
                yield _sse("done", AskResponse(**result).model_dump())
            else:
                result = system.ask(
                    question=request.question,
                    target_files=request.target_files,
                    forced_route=request.forced_route,
                    memory=memory,
                )
                yield _sse("done", AskResponse(**_finish(result)).model_dump())
        except Exception as exc:
            logger.exception("ask/stream failed")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask/image", response_model=AskResponse)
async def ask_image(
    session_id: str = Form(...),
    question: str = Form(""),
    file: UploadFile = File(...),
    user: User = Depends(current_user),
):
    system = await run_in_threadpool(systems.get, user)
    memory = sessions.get(session_id, user.id)
    image_bytes = await file.read()
    try:
        result = await run_in_threadpool(
            system.answer_about_uploaded_image,
            image_bytes=image_bytes,
            question=question,
            mime=file.content_type or "image/png",
            memory=memory,
        )
    except Exception as exc:
        logger.exception("ask/image failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**_finish(result, docs=[]))


@app.post("/ask/vision-analysis", response_model=AskResponse)
async def ask_vision_analysis(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(current_user),
):
    system = await run_in_threadpool(systems.get, user)
    memory = sessions.get(session_id, user.id)
    image_bytes = await file.read()
    try:
        result = await run_in_threadpool(system.analyze_uploaded_image, image_bytes=image_bytes, memory=memory)
    except Exception as exc:
        logger.exception("ask/vision-analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = _finish(result, docs=[])
    try:
        annotated = system.vision_analysis.annotated_image_bytes(image_bytes) if system.vision_analysis else None
        if annotated:
            result["annotated_image"] = "data:image/png;base64," + base64.b64encode(annotated).decode()
    except Exception:
        logger.exception("annotated image failed")
    return AskResponse(**result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
