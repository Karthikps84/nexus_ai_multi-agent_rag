from typing import Any, Optional

# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    session_id: str
    question: str
    target_files: Optional[list[str]] = None
    forced_route: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    route: str = ""
    context: str = ""
    dataframe: Any = None
    verification: Optional[dict] = None
    max_score: float = 0.0
    docs: list = Field(default_factory=list)
    execution_plan: dict = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    execution_time: float = 0.0
    cache_hits: dict = Field(default_factory=dict)
    token_usage: dict = Field(default_factory=dict)
    trace: list = Field(default_factory=list)
    annotated_image: Optional[str] = None  # data URL of detection boxes (vision-analysis only)
    # agent_outputs intentionally omitted: MultiAgentSystem.ask()'s result
    # dict is self-referential (the "verify" step's recorded output embeds
    # the full to_result() dict, which contains agent_outputs again), so it
    # isn't JSON-serializable and isn't useful to a frontend anyway.


class HealthResponse(BaseModel):
    status: str
    initialized: bool


class FileInfo(BaseModel):
    name: str
    size_bytes: int
    kind: str


class FilesResponse(BaseModel):
    files: list[str]
    details: list[FileInfo] = Field(default_factory=list)


class UploadResponse(BaseModel):
    saved: list[str]
    skipped: list[str]
    files_indexed: int
