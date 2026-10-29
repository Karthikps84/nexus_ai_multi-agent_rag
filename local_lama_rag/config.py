import os
from pathlib import Path

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")

# Per-user libraries live under USERS_ROOT/<user_id>/{data,storage}; the user
# database and signing secret live there too. Never commit this folder.
USERS_ROOT = Path(os.getenv("USERS_ROOT", "users"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ACTIVE_USER_SYSTEMS = int(os.getenv("MAX_ACTIVE_USER_SYSTEMS", "6"))

# pyrefly: ignore[missing-import]
from dotenv import load_dotenv
load_dotenv()

# Model defaults below were verified live against NVIDIA's /v1/models
# catalog and real chat/embeddings calls on 2026-10-20, after the
# previous defaults (meta/llama-3.1-70b-instruct, meta/llama-3.1-8b-instruct,
# nvidia/nemotron-nano-12b-v2-vl, nvidia/nv-embedqa-e5-v5) all returned
# HTTP 410 Gone (NIM catalog end-of-life, Aug 2026). Re-verify against
# https://integrate.api.nvidia.com/v1/models if these start failing too --
# NVIDIA retires NIM models on a rolling schedule with no deprecation
# window guarantee for the free tier.
#
# nvidia/nemotron-3.5-lightning-30b-a3b was tried first but is a reasoning
# model: it spends its whole token budget on hidden reasoning_content and
# can return an EMPTY content field under a short max_tokens budget
# (finish_reason="length" with content=""). openai/gpt-oss-20b has the same
# hidden-reasoning behavior but stayed stable (content populated) across
# repeated short-prompt calls through the real ChatNVIDIA client, matching
# how router/qa/etc. actually call it -- used as LLM_MODEL/FAST_LLM_MODEL
# for that reason.
#
# meta/llama-3.2-11b-vision-instruct is the ONLY multimodal model entitled
# on this account (confirmed via /v1/models + probing every other vision
# candidate -> all 404 not-entitled). It is also confirmed UNSTABLE here:
# repeated identical calls alternate between 200 and 451/timeout with no
# pattern. Still set as VISION_MODEL because there is no alternative, but
# image/vision_analysis features will intermittently fail until NVIDIA's
# entitlement/catalog changes.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Every model (LLM, vision, embedding) is served by NVIDIA NIM with
# NVIDIA_API_KEY; no other provider key is used.
#
# LLM_MODEL is the "primary" tier (routed to "code" tasks per MODEL_ROUTING
# below); FAST_LLM_MODEL serves most other routes. Measured live on NVIDIA:
# minimaxai/minimax-m3.1-flash gives good answers but takes 33-42s per call;
# meta/muse-glimmer-30b gave a 90-char answer in 52s (too weak);
# openai/gpt-oss-20b took 83s; nvidia/nemotron-3.5-lightning-30b-a3b runs
# 7-17s per call and is used for the fast tier.
FAST_LLM_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
LLM_MODEL = "minimaxai/minimax-m3.1-flash"
# z-ai/glm-5.3-flash timed out (>100s) on every call; this model answered in
# 1.4s in a live image test (llama-3.2-90b-vision: 1.7s, 11b: 9.4s).
VISION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
# Stored index is 2048-dim; switching to a different-dimension model (e.g.
# nemotron-3.5-embed-8b = 4096) needs a full re-index of storage/.
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"

# Several NVIDIA NIM models are "reasoning" models that spend tokens on a
# hidden reasoning_content field before (or instead of) the visible answer.
# Without a generous token budget, they hit finish_reason="length" and
# return an EMPTY content string -- confirmed live, twice, with two
# different models during this project's setup. This budget covers typical
# reasoning overhead (can run 1000+ tokens) plus a real answer.
LLM_MAX_TOKENS = 4096


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 48
EMBED_WORKERS = 5

SIMILARITY_TOP_K = 10
FINAL_TOP_K = 6
# Measured on the CI reference corpus: questions answered by the docs score
# 0.185-0.529, unrelated questions 0.023-0.055. 0.30 sat inside the in-doc
# range and wrongly sent answerable questions to web search.
SIMILARITY_THRESHOLD = 0.12
USE_MMR = True
ENABLE_HYBRID_SEARCH = True
ENABLE_RERANK = True
RETRIEVAL_CONTEXT_CHAR_BUDGET = 12000

WEB_SEARCH_RESULTS = 3

ENABLE_DOCLING_OCR = False   # heavy neural OCR fallback; off by default (see loaders module)
CAPTION_IMAGES = False       # set True to caption extracted images with VISION_MODEL (adds latency/cost)

# --- NVIDIA transport resilience ---
NVIDIA_PRIMARY_TIMEOUT_SECONDS = 180
NVIDIA_FAST_TIMEOUT_SECONDS = 90
NVIDIA_VISION_TIMEOUT_SECONDS = 90
NVIDIA_REQUEST_RETRIES = 2
NVIDIA_RETRY_BACKOFF_SECONDS = 2.0

MAX_HISTORY_TURNS = 6
SUMMARY_MAX_CHUNKS_PER_FILE = 6
COMPARE_MAX_CHUNKS_PER_FILE = 4
ENABLE_QUERY_REWRITE = True
QUERY_REWRITE_MAX_QUESTION_CHARS = 120
QUERY_REWRITE_TRIGGER_TERMS = (
    "it", "they", "them", "that", "those", "this", "these", "former", "latter",
    "second one", "first one", "above", "previous", "earlier",
)

# --- Verification ---
VERIFY_ANSWERS = True           # run the Citation & Verification agent after generation
VERIFY_MIN_OVERLAP = 0.15       # below this lexical-overlap ratio, a sentence is flagged unsupported
SUPERVISOR_MIN_CONFIDENCE = 0.35
FINAL_VERIFICATION_CONFIDENCE_THRESHOLD = 0.55
FINAL_VERIFICATION_FORCE_LLM_THRESHOLD = 0.35
ENABLE_REFLECTION = True
REFLECTION_MIN_ANSWER_CHARS = 240
REFLECTION_TRIGGER_CONFIDENCE = 0.55
REFLECTION_ROUTES = {"local"}

# --- Observability ---
LANGSMITH_ENABLED = True
LANGSMITH_PROJECT = "multiagent-rag"

# --- Caching ---
RESPONSE_CACHE_TTL_SECONDS = 1800
RETRIEVAL_CACHE_TTL_SECONDS = 1800
EMBEDDING_CACHE_TTL_SECONDS = 7200
RESPONSE_CACHE_MAX_ENTRIES = 256
RETRIEVAL_CACHE_MAX_ENTRIES = 256
EMBEDDING_CACHE_MAX_ENTRIES = 2048

# --- Model routing ---
MODEL_ROUTING = {
    "default": "fast",
    "retrieve": "fast",
    "qa": "fast",
    "summarize": "fast",
    "compare": "fast",
    "table": "fast",
    "planner": "fast",
    "supervisor": "fast",
    "code": "primary",
    "image": "vision",
    "vision_analysis": "fast",
    "verify": "verifier",
    "reflect": "verifier",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".txt", ".md"} | IMAGE_EXTENSIONS

# --- Vision Analysis (real CV inference, not a VLM caption) ---
ENABLE_VISION_ANALYSIS = True
# Weights string is passed straight to ultralytics.YOLO(...). If the file
# isn't already present locally, ultralytics downloads it automatically
# from the official ultralytics/assets GitHub Releases on first use --
# no manual download, no API key, no separate service. That IS the
# "get it from the GitHub repo" mechanism; there's no other API/MCP layer
# needed for this.
#
# To move to a newer model as Ultralytics ships one: change ONLY this
# string. Check current options at https://docs.ultralytics.com/models
# before bumping -- model family names change (yolo11 -> yolo26 happened
# Jan 2026), and this string isn't validated at import time.
#
# Scale suffix trades speed for accuracy: n (nano) < s < m < l < x (xlarge).
# 'n' is the right choice for a live demo on CPU; bump to 's' or 'm' only
# if you have GPU inference available, since larger scales get slow on CPU.
VISION_ANALYSIS_WEIGHTS = "yolo26n.pt"
VISION_ANALYSIS_CONFIDENCE = 0.25
VISION_ANALYSIS_MAX_DETECTIONS = 50
