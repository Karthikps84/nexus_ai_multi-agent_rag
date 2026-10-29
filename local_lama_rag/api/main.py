from __future__ import annotations

from contextlib import asynccontextmanager

import base64
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from graph.workflow import MultiAgentSystem
from utils.logger import get_logger
from utils.uploads import save_upload_bytes

from api.schemas import (
    AskRequest,
    AskResponse,
    FileInfo,
    FilesResponse,
    HealthResponse,
    SessionCreateResponse,
    UploadResponse,
)
from api.sessions import sessions

logger = get_logger("api")

_state: dict[str, MultiAgentSystem | None] = {"system": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        system = MultiAgentSystem()
        system.initialize()
        _state["system"] = system
        logger.info("MultiAgentSystem initialized.")
    except EnvironmentError as exc:
        logger.error("Startup failed: %s", exc)
        raise
    yield
    _state["system"] = None


app = FastAPI(title="local_lama_rag API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_system() -> MultiAgentSystem:
    system = _state["system"]
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return system


def _serialize_docs(docs: list) -> list[dict]:
    return [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]


def _serialize_dataframe(df) -> list[dict] | None:
    if df is None:
        return None
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return None


def _normalize_result(result: dict) -> dict:
    # langchain_core >=1.x can return `answer` as a TextAccessor (a str
    # subclass that's also callable) rather than a plain str. Pydantic's
    # serializer chokes on it ("Circular reference detected") because it
    # looks like a complex object rather than a string. Coerce explicitly.
    result = dict(result)
    if "answer" in result and result["answer"] is not None:
        result["answer"] = str(result["answer"])
    # agent_outputs is self-referential (see api/schemas.py) and not part
    # of AskResponse -- drop it so it never reaches the Pydantic model.
    result.pop("agent_outputs", None)
    return result


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", initialized=_state["system"] is not None)


@app.post("/sessions", response_model=SessionCreateResponse)
def create_session():
    return SessionCreateResponse(session_id=sessions.create())


@app.get("/files", response_model=FilesResponse)
def list_files():
    system = get_system()
    names = [n for n in system.retrieval.list_known_files() if (config.DATA_DIR / n).suffix.lower() in config.SUPPORTED_EXTENSIONS]
    details = []
    for n in names:
        p = config.DATA_DIR / n
        ext = p.suffix.lower()
        kind = "image" if ext in config.IMAGE_EXTENSIONS else (ext.lstrip(".") or "file")
        details.append(FileInfo(name=n, size_bytes=p.stat().st_size if p.exists() else 0, kind=kind))
    return FilesResponse(files=names, details=details)


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    system = get_system()
    saved, skipped = [], []
    for f in files:
        content = await f.read()
        if save_upload_bytes(f.filename or "upload", content, config.DATA_DIR):
            saved.append(f.filename or "upload")
        else:
            skipped.append(f.filename or "upload")

    if saved:
        try:
            system.initialize()
        except Exception as exc:
            logger.exception("Reindex after upload failed")
            raise HTTPException(status_code=500, detail=f"Saved but reindex failed: {exc}") from exc

    return UploadResponse(
        saved=saved,
        skipped=skipped,
        files_indexed=len(system.retrieval.list_known_files()),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    system = get_system()
    memory = sessions.get(request.session_id)
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

    result = _normalize_result(result)
    result["docs"] = _serialize_docs(result.get("docs") or [])
    result["dataframe"] = _serialize_dataframe(result.get("dataframe"))
    return AskResponse(**result)


@app.post("/ask/image", response_model=AskResponse)
async def ask_image(
    session_id: str = Form(...),
    question: str = Form(""),
    file: UploadFile = File(...),
):
    system = get_system()
    memory = sessions.get(session_id)
    image_bytes = await file.read()
    try:
        result = system.answer_about_uploaded_image(
            image_bytes=image_bytes,
            question=question,
            mime=file.content_type or "image/png",
            memory=memory,
        )
    except Exception as exc:
        logger.exception("ask/image failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = _normalize_result(result)
    result["docs"] = []
    result["dataframe"] = _serialize_dataframe(result.get("dataframe"))
    return AskResponse(**result)


@app.post("/ask/vision-analysis", response_model=AskResponse)
async def ask_vision_analysis(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    system = get_system()
    memory = sessions.get(session_id)
    image_bytes = await file.read()
    try:
        result = system.analyze_uploaded_image(image_bytes=image_bytes, memory=memory)
    except Exception as exc:
        logger.exception("ask/vision-analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = _normalize_result(result)
    result["docs"] = []
    result["dataframe"] = _serialize_dataframe(result.get("dataframe"))
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
