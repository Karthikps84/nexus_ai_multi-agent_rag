"""
==========================================================
Multi-Agent RAG — Streamlit UI
==========================================================
Run:  streamlit run app.py
==========================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
import streamlit as st

from agents.memory import MemoryAgent
from graph.workflow import MultiAgentSystem
from utils.uploads import save_upload_bytes
import config

st.set_page_config(
    page_title="NexusAI — Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUPPORTED_EXT = config.SUPPORTED_EXTENSIONS
IMAGE_EXT = config.IMAGE_EXTENSIONS

SUGGESTIONS = {
    "Summarize": "Summarize the main document in bullet points.",
    "Compare": "Compare the two most relevant documents and highlight key differences.",
    "Implement": "Read the paper and write starter Python code for the core algorithm.",
    "Extract data": "Extract the main entities, values, and dates into a table.",
}

# ── Design tokens ─────────────────────────────────────────────────────────────
# Direction: deep navy + electric indigo accent, warm-white surface.
# Signature: a persistent glowing gradient "rail" on the left edge of
# the chat panel — a single strong mark that's absent from generic AI UIs.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --navy:       #0D1117;
  --navy-mid:   #161C26;
  --navy-card:  #1C2333;
  --navy-hover: #222B3E;
  --indigo:     #6366F1;
  --indigo-dim: rgba(99,102,241,0.15);
  --indigo-glow:rgba(99,102,241,0.30);
  --cyan:       #22D3EE;
  --emerald:    #34D399;
  --amber:      #FBBF24;
  --rose:       #F87171;
  --surf:       #F8F9FF;
  --surf-mid:   #EEF0F8;
  --border:     rgba(255,255,255,0.08);
  --border-hi:  rgba(99,102,241,0.40);
  --tx-hi:      #F0F3FF;
  --tx-mid:     #9BA3C2;
  --tx-lo:      #5B6480;
  --r:   14px;
  --rs:  8px;
  --sh: 0 8px 32px rgba(0,0,0,0.35);
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--navy) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--tx-hi) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--navy-mid) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--indigo-dim); border-radius: 4px; }

/* ── Wordmark ───────────────────────────────────────────────────────────── */
.wordmark {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0 28px;
}
.wordmark-gem {
  width: 32px; height: 32px; border-radius: 10px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--indigo), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; box-shadow: 0 4px 18px var(--indigo-glow);
}
.wordmark-name { font-size: 15px; font-weight: 700; letter-spacing: -.3px; color: var(--tx-hi); }
.wordmark-sub  { font-size: 11px; color: var(--tx-lo); letter-spacing: .3px; margin-top: 1px; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  font-size: 14px !important; color: var(--tx-mid) !important;
  background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--indigo) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background: var(--indigo) !important; height: 2px !important;
}
[data-testid="stTabPanel"] { padding-top: 18px !important; }

/* ── Generic card ───────────────────────────────────────────────────────── */
.card {
  background: var(--navy-card); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px 18px; margin-bottom: 10px;
  box-shadow: var(--sh); transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color: var(--border-hi); box-shadow: 0 8px 32px var(--indigo-glow); }

.row {
  display: flex; align-items: center; justify-content: space-between;
}
.lbl { font-size: 12px; color: var(--tx-mid); }
.val { font-size: 12px; font-weight: 600; color: var(--tx-hi);
  font-family: 'DM Mono', monospace; text-align: right; }

/* ── Pill badges ────────────────────────────────────────────────────────── */
.pill {
  display: inline-flex; align-items: center; gap: 4px; font-size: 11px;
  font-weight: 600; padding: 3px 10px; border-radius: 20px;
  font-family: 'Inter', sans-serif;
}
.pill-indigo  { background: var(--indigo-dim); color: #A5B4FC; border: 1px solid rgba(99,102,241,.25); }
.pill-cyan    { background: rgba(34,211,238,.10); color: #67E8F9; border: 1px solid rgba(34,211,238,.22); }
.pill-emerald { background: rgba(52,211,153,.10); color: #6EE7B7; border: 1px solid rgba(52,211,153,.22); }
.pill-amber   { background: rgba(251,191,36,.10); color: #FCD34D; border: 1px solid rgba(251,191,36,.22); }
.pill-rose    { background: rgba(248,113,113,.10); color: #FCA5A5; border: 1px solid rgba(248,113,113,.22); }
.pill-ghost   { background: rgba(255,255,255,.05); color: var(--tx-mid); border: 1px solid var(--border); }

/* ── Hero / page header ─────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, var(--navy-card) 0%, #12192E 100%);
  border: 1px solid var(--border); border-radius: 22px;
  padding: 30px 34px 26px; margin-bottom: 22px; position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--indigo), var(--cyan), transparent);
}
.hero::after {
  content: ''; position: absolute; top: -80px; right: -60px;
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(99,102,241,.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title {
  font-size: 28px; font-weight: 700; letter-spacing: -.5px;
  color: var(--tx-hi); margin: 0 0 6px;
}
.hero-sub { font-size: 14px; color: var(--tx-mid); margin: 0; max-width: 600px; }
.hero-chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }

/* ── Stat strip ─────────────────────────────────────────────────────────── */
.stats-strip {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 14px 0;
}
.stat-box {
  background: var(--navy-card); border: 1px solid var(--border);
  border-radius: var(--rs); padding: 12px 14px;
}
.stat-k { font-size: 11px; color: var(--tx-lo); text-transform: uppercase; letter-spacing: .5px; }
.stat-v { font-size: 20px; font-weight: 700; color: var(--tx-hi); margin-top: 4px; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  width: 100% !important;
  background: linear-gradient(135deg, var(--indigo), #4F46E5) !important;
  color: #fff !important; font-weight: 600 !important; border: none !important;
  border-radius: var(--rs) !important; font-size: 13px !important;
  padding: 9px 14px !important; letter-spacing: .1px !important;
  box-shadow: 0 4px 18px var(--indigo-glow) !important;
  transition: .15s ease !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px var(--indigo-glow) !important;
}
[data-testid="stButton"] > button:disabled {
  background: rgba(255,255,255,.06) !important; color: var(--tx-lo) !important;
  box-shadow: none !important; transform: none !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--indigo), var(--cyan)) !important;
  border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
  background: rgba(255,255,255,.05) !important; border-radius: 4px !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--navy-card) !important;
  border: 1.5px dashed rgba(99,102,241,.35) !important;
  border-radius: var(--r) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--indigo) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--tx-mid) !important; font-size: 13px !important; }

/* ── Chat messages ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  border-radius: var(--r) !important; margin-bottom: 12px !important;
  animation: slideUp 0.22s ease !important;
  transition: border-color .2s !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--navy-hover) !important;
  border: 1px solid rgba(255,255,255,.07) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: var(--navy-card) !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stChatMessage"]:hover { border-color: var(--border-hi) !important; }

/* Force text visibility inside chat messages -- covers every element type
   markdown rendering can produce, plus the container itself as a base
   case so anything not explicitly listed still inherits correctly. The
   .streamlit/config.toml theme (base=dark) is the primary fix; this is
   the safety net for markdown-generated content specifically. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  color: var(--tx-hi) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] div,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h5,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h6 {
  color: var(--tx-hi) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
  color: #A5B4FC !important;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
  background: var(--navy-card) !important;
  border: 1.5px solid rgba(99,102,241,.30) !important;
  border-radius: var(--r) !important;
}
[data-testid="stChatInputContainer"]:focus-within {
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 3px var(--indigo-glow) !important;
}
[data-testid="stChatInput"] textarea { color: var(--tx-hi) !important; }

/* ── Expanders ──────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--navy-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--rs) !important;
}
[data-testid="stExpander"] summary { color: var(--tx-mid) !important; font-size: 12px !important; }

/* ── DataFrame ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--rs) !important;
}

/* ── Multi-select tags ──────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  background: var(--indigo) !important;
}

/* ── Task form card ─────────────────────────────────────────────────────── */
.task-card {
  background: var(--navy-card); border: 1px solid var(--border-hi);
  border-radius: var(--r); padding: 20px 22px; margin-bottom: 16px;
  box-shadow: 0 0 0 1px var(--indigo-dim), var(--sh);
}
.task-title { font-size: 15px; font-weight: 600; color: var(--tx-hi); margin-bottom: 4px; }
.task-sub   { font-size: 12px; color: var(--tx-mid); margin-bottom: 14px; }

/* ── Empty state ────────────────────────────────────────────────────────── */
.empty-state {
  background: var(--navy-card); border: 1.5px dashed rgba(99,102,241,.25);
  border-radius: 20px; padding: 52px 32px; text-align: center; margin: 12px 0;
}

/* ── Action grid (empty-screen shortcuts) ───────────────────────────────── */
.action-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px;
}
.action-tile {
  background: var(--navy-card); border: 1px solid var(--border);
  border-radius: var(--r); padding: 16px 14px; cursor: pointer;
  transition: border-color .2s, box-shadow .2s, transform .15s;
}
.action-tile:hover {
  border-color: var(--border-hi); transform: translateY(-2px);
  box-shadow: 0 8px 24px var(--indigo-glow);
}
.action-tile-icon { font-size: 22px; margin-bottom: 8px; }
.action-tile-title { font-size: 13px; font-weight: 600; color: var(--tx-hi); margin-bottom: 4px; }
.action-tile-desc  { font-size: 11px; color: var(--tx-lo); line-height: 1.5; }

/* ── Meta row (route + confidence) ─────────────────────────────────────── */
.meta-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.meta-sep  { font-size: 11px; color: var(--tx-lo); }
.conf-track {
  background: rgba(255,255,255,.08); border-radius: 6px; height: 5px;
  width: 72px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 4px;
}
.conf-fill { height: 100%; border-radius: 6px; }

/* ── Dev panel ──────────────────────────────────────────────────────────── */
.trace-row {
  display: flex; align-items: center; gap: 8px; padding: 6px 0;
  border-bottom: 1px solid var(--border); font-size: 12px;
}
.trace-agent { font-family: 'DM Mono', monospace; color: var(--cyan); min-width: 90px; }
.trace-model { color: var(--tx-mid); min-width: 100px; }
.trace-stat  { color: var(--tx-lo); }

/* ── Workspace note ─────────────────────────────────────────────────────── */
.workspace-note {
  background: var(--navy-card); border: 1px solid var(--border);
  border-radius: var(--rs); padding: 10px 14px; margin-bottom: 14px;
  font-size: 13px; color: var(--tx-mid);
}

/* ── Sec label ──────────────────────────────────────────────────────────── */
.sec-label {
  font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
  color: var(--tx-lo); margin: 18px 0 8px;
}

code {
  font-family: 'DM Mono', monospace !important;
  background: rgba(99,102,241,.12) !important;
  border-radius: 4px !important; padding: 1px 5px !important;
  font-size: 12px !important; color: #A5B4FC !important;
}
hr { border-color: var(--border) !important; margin: 12px 0 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROUTE_LABEL = {
    "local": "📚 KB", "web": "🌐 Web", "summarize": "📝 Summary",
    "table": "🗂️ Table", "compare": "⚖️ Compare", "code": "💻 Code",
    "image": "🖼️ Image", "planner": "🧭 Plan", "retrieve": "🔎 Retrieval",
}
ROUTE_PILL = {
    "local": "emerald", "web": "amber", "summarize": "indigo", "table": "indigo",
    "compare": "amber", "code": "cyan", "image": "cyan", "planner": "indigo",
    "retrieve": "emerald",
}
ICON = {
    "pdf": "📕", "docx": "📘", "xlsx": "📗", "xls": "📗", "csv": "📊",
    "pptx": "📙", "txt": "📄", "md": "📝",
    "jpg": "🖼️", "jpeg": "🖼️", "png": "🖼️", "webp": "🖼️",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_uploaded_files(files, data_dir: Path) -> tuple[int, list[str]]:
    saved, skipped = 0, []
    for f in files:
        if save_upload_bytes(f.name, bytes(f.getbuffer()), data_dir):
            saved += 1
        else:
            skipped.append(f.name)
    return saved, skipped


@st.cache_resource(show_spinner=False)
def get_system() -> MultiAgentSystem:
    holder = st.empty()
    bar = holder.progress(0, text="Starting up…")

    def on_progress(pct: int, msg: str):
        bar.progress(min(pct, 100) / 100, text=msg)

    sys_ = MultiAgentSystem()
    sys_.initialize(on_progress=on_progress)
    holder.empty()
    return sys_


@st.cache_data(ttl=5, show_spinner=False)
def get_file_stats(data_dir: str) -> dict:
    base = Path(data_dir)
    files = sorted(f for f in base.glob("*") if f.is_file()) if base.exists() else []
    total_kb = round(sum(f.stat().st_size for f in files) / 1024, 1)
    return {"count": len(files), "total_kb": total_kb, "files": files}


def confidence_html(confidence) -> str:
    if confidence is None:
        return ""
    pct = int(confidence * 100)
    color = "#34D399" if confidence >= 0.6 else ("#FBBF24" if confidence >= 0.35 else "#F87171")
    return (
        f'<span class="meta-sep">confidence</span>'
        f'<span class="conf-track"><span class="conf-fill" style="width:{pct}%;background:{color};"></span></span>'
        f'<span class="meta-sep">{pct}%</span>'
    )


def render_meta(route: str, elapsed: float, verification: dict | None, cache_hit: bool = False):
    pill_cls = ROUTE_PILL.get(route, "indigo")
    label = ROUTE_LABEL.get(route, route)
    conf_html = confidence_html((verification or {}).get("confidence"))
    cache_tag = '<span class="pill pill-cyan">⚡ cached</span>' if cache_hit else ""
    st.markdown(
        f'<div class="meta-row">'
        f'<span class="pill pill-{pill_cls}">{label}</span>'
        f'{cache_tag}'
        f'<span class="meta-sep">{elapsed:.1f}s</span>'
        f'{conf_html}</div>',
        unsafe_allow_html=True,
    )
    if verification and verification.get("unsupported_claims"):
        st.warning(f"⚠️ {verification['unsupported_claims']}", icon=":material/warning:")


def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source(s)", expanded=False):
        for s in sources:
            st.markdown(f"- `{s}`")


def render_dev_panel(trace: list, plan: dict, cache_hits: dict, elapsed: float):
    with st.expander("◈ Developer panel", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Execution time", f"{elapsed:.3f}s")
        hits = [k for k, v in (cache_hits or {}).items() if v]
        col_b.metric("Cache hits", len(hits) if hits else "—")
        col_c.metric("Agents fired", len([t for t in (trace or []) if t.get("status") == "completed"]))

        if plan and plan.get("steps"):
            with st.expander("Execution plan", expanded=False):
                st.json(plan, expanded=False)

        if trace:
            st.markdown("**Trace**")
            for item in trace:
                latency = item.get("metadata", {}).get("latency_ms")
                latency_str = f"{latency} ms" if latency is not None else "n/a"
                conf = item.get("confidence")
                conf_str = f"{int(conf*100)}%" if conf is not None else "—"
                status_color = "#34D399" if item.get("status") == "completed" else "#F87171"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-agent">{item.get("agent", "?")}</span>'
                    f'<span class="trace-model">{item.get("model", "?")}</span>'
                    f'<span class="trace-stat">⏱ {latency_str}</span>'
                    f'<span class="trace-stat">conf {conf_str}</span>'
                    f'<span style="color:{status_color};font-size:11px;">{item.get("status","?")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_result(
    answer: str, route: str, sources: list, dataframe, verification: dict | None,
    elapsed: float, trace: list | None = None, plan: dict | None = None,
    cache_hits: dict | None = None, cache_hit: bool = False,
):
    if dataframe is not None:
        st.markdown(answer)
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
    else:
        st.markdown(answer)
    render_meta(route, elapsed, verification, cache_hit=cache_hit)
    render_sources(sources)
    render_dev_panel(trace or [], plan or {}, cache_hits or {}, elapsed)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("active_task", None), ("pending_image", None),
             ("prefilled_prompt", ""), ("pending_task_request", None)]:
    if k not in st.session_state:
        st.session_state[k] = v
st.session_state.setdefault("task_selected_files", [])
st.session_state.setdefault("task_extract_instruction", "")

if "memory" not in st.session_state:
    st.session_state.memory = MemoryAgent()


# ── Task execution ─────────────────────────────────────────────────────────────

def run_image_task(question: str, image_bytes: bytes, mime: str):
    label = question.strip() if question.strip() else "[attached an image]"
    st.session_state.messages.append({"role": "user", "content": label, "image": image_bytes})
    with st.chat_message("user"):
        st.image(image_bytes, width=260)
        if question.strip():
            st.markdown(question)

    elapsed = 0.0
    r = {}
    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            with st.spinner("Analysing image…"):
                r = system.answer_about_uploaded_image(
                    image_bytes, question, mime, memory=st.session_state.memory
                )
            elapsed = time.time() - t0
            render_result(
                r["answer"], r["route"], r.get("sources", []), None,
                r.get("verification"), elapsed,
                trace=r.get("trace"), plan=r.get("execution_plan"),
                cache_hits=r.get("cache_hits"),
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            elapsed = time.time() - t0
            r = {"answer": "Image analysis timed out. Try a narrower question or retry.",
                 "route": "image", "sources": [], "verification": None}
            st.error(r["answer"], icon=":material/schedule:")

    st.session_state.messages.append({
        "role": "assistant", "content": r.get("answer", ""),
        "sources": r.get("sources", []), "dataframe": None,
        "verification": r.get("verification"),
        "meta": {"route": r.get("route", "image"), "elapsed": elapsed},
        "trace": r.get("trace", []), "execution_plan": r.get("execution_plan", {}),
        "cache_hits": r.get("cache_hits", {}),
    })


def run_vision_analysis_task(image_bytes: bytes, with_vlm_comparison: bool, show_labels: bool = True):
    label = "Run object detection" + (" + compare against vision-language description" if with_vlm_comparison else "")
    st.session_state.messages.append({"role": "user", "content": label, "image": image_bytes})
    with st.chat_message("user"):
        st.image(image_bytes, width=260)
        st.markdown(label)

    elapsed = 0.0
    r = {}
    annotated = None
    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            with st.spinner("Running YOLO inference…"):
                r = system.analyze_uploaded_image(
                    image_bytes, memory=st.session_state.memory, run_vlm_description=with_vlm_comparison,
                )
            elapsed = time.time() - t0

            # Show the annotated (boxed) image alongside the answer, when available
            if system.vision_analysis is not None and system.vision_analysis.available:
                annotated = system.vision_analysis.annotated_image_bytes(
                    image_bytes, show_labels=show_labels, show_conf=show_labels
                )
                if annotated:
                    st.image(annotated, width=320, caption="Detected objects")

            render_result(
                r["answer"], r["route"], r.get("sources", []), r.get("dataframe"),
                r.get("verification"), elapsed,
                trace=r.get("trace"), plan=r.get("execution_plan"),
                cache_hits=r.get("cache_hits"),
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            elapsed = time.time() - t0
            r = {"answer": "Vision analysis timed out. Try again.",
                 "route": "vision_analysis", "sources": [], "verification": None}
            st.error(r["answer"], icon=":material/schedule:")

    st.session_state.messages.append({
        "role": "assistant",
        "content": r.get("answer", ""),
        "image": annotated,
        "sources": r.get("sources", []),
        "dataframe": r.get("dataframe"),
        "verification": r.get("verification"),
        "meta": {"route": r.get("route", "vision_analysis"), "elapsed": elapsed},
        "trace": r.get("trace", []),
        "execution_plan": r.get("execution_plan", {}),
        "cache_hits": r.get("cache_hits", {}),
    })


def run_task(question: str, forced_route: str | None = None, target_files: list | None = None):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    elapsed = 0.0
    answer, route, sources, dataframe, verification = "", forced_route or "local", [], None, None
    response_payload: dict = {}

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            prep = system.prepare_stream(
                question, target_files=target_files, forced_route=forced_route,
                memory=st.session_state.memory,
            )

            if prep.get("cached"):
                r = prep["result"]
                response_payload = r
                answer = r["answer"]
                route = r.get("route", "local")
                sources = r.get("sources", [])
                dataframe = r.get("dataframe")
                verification = r.get("verification")
                elapsed = time.time() - t0
                render_result(
                    answer, route, sources, dataframe, verification, elapsed,
                    trace=r.get("trace"), plan=r.get("execution_plan"),
                    cache_hits=r.get("cache_hits"), cache_hit=True,
                )

            elif prep.get("streamable"):
                placeholder = st.empty()
                for chunk in prep["chain"].stream(prep["inputs"]):
                    answer += chunk
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)
                route = prep["route"]
                sources = prep["sources"]
                elapsed = time.time() - t0
                result = system.finalize(
                    question, answer, route, sources,
                    context=prep.get("context", ""),
                    target_files=target_files,
                    forced_route=forced_route,
                    memory=st.session_state.memory,
                )
                response_payload = result
                verification = result.get("verification")
                render_meta(route, elapsed, verification)
                render_sources(sources)
                render_dev_panel(
                    result.get("trace", []), result.get("execution_plan", {}),
                    result.get("cache_hits", {}), elapsed,
                )

            else:
                with st.spinner("Working…"):
                    r = system.ask(
                        question, target_files=target_files, forced_route=forced_route,
                        memory=st.session_state.memory,
                    )
                response_payload = r
                answer = r["answer"]
                route = r.get("route", forced_route or "local")
                sources = r.get("sources", [])
                dataframe = r.get("dataframe")
                verification = r.get("verification")
                elapsed = time.time() - t0
                render_result(
                    answer, route, sources, dataframe, verification, elapsed,
                    trace=r.get("trace"), plan=r.get("execution_plan"),
                    cache_hits=r.get("cache_hits"),
                )

        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            elapsed = time.time() - t0
            answer = "The NVIDIA endpoint timed out. Please retry or simplify your question."
            st.error(answer, icon=":material/schedule:")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources, "dataframe": dataframe,
        "verification": verification,
        "meta": {"route": route, "elapsed": elapsed},
        "trace": response_payload.get("trace", []),
        "execution_plan": response_payload.get("execution_plan", {}),
        "cache_hits": response_payload.get("cache_hits", {}),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-title">◆ NexusAI — Document Intelligence</div>
  <p class="hero-sub">Ask naturally. The system routes your question through retrieval, reasoning, and verification automatically.</p>
  <div class="hero-chips">
    <span class="pill pill-emerald">📚 QA</span>
    <span class="pill pill-indigo">📝 Summarize</span>
    <span class="pill pill-indigo">🗂️ Extract</span>
    <span class="pill pill-amber">⚖️ Compare</span>
    <span class="pill pill-cyan">💻 Code</span>
    <span class="pill pill-cyan">🖼️ Vision</span>
    <span class="pill pill-ghost">🧭 Multi-agent plan</span>
    <span class="pill pill-ghost">✅ Verified</span>
  </div>
</div>
""", unsafe_allow_html=True)

system_ok, system = False, None
try:
    system = get_system()
    system_ok = True
except FileNotFoundError:
    pass
except Exception as e:
    st.error(f"System startup failed: {e}")

tab_docs, tab_chat = st.tabs(["📁 Documents", "💬 Chat"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="sec-label">Upload</div>', unsafe_allow_html=True)
        with st.form("upload_form", border=False):
            uploaded_files = st.file_uploader(
                "Drag & drop or click to browse",
                type=[e.lstrip(".") for e in sorted(SUPPORTED_EXT)],
                accept_multiple_files=True, key="doc_uploader",
                label_visibility="collapsed",
            )
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected")
            submitted = st.form_submit_button(
                ":material/upload_file: Index documents",
                disabled=not uploaded_files, use_container_width=True,
            )
        if submitted and uploaded_files:
            saved, skipped = save_uploaded_files(uploaded_files, Path(config.DATA_DIR))
            if skipped:
                st.warning(f"Skipped: {', '.join(skipped)}", icon=":material/warning:")
            if saved:
                get_file_stats.clear()
                st.cache_resource.clear()
                st.toast("Documents indexed ✓", icon=":material/check_circle:")
                st.rerun()
            else:
                st.error("No supported files saved.", icon=":material/error:")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Models</div>', unsafe_allow_html=True)
        for label, val in [
            ("Primary LLM", config.LLM_MODEL.split("/")[-1]),
            ("Fast LLM",    config.FAST_LLM_MODEL.split("/")[-1]),
            ("Vision",      config.VISION_MODEL.split("/")[-1]),
            ("Embeddings",  config.EMBEDDING_MODEL.split("/")[-1]),
        ]:
            st.markdown(
                f'<div class="card"><div class="row">'
                f'<span class="lbl">{label}</span><span class="val">{val}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="sec-label">Corpus</div>', unsafe_allow_html=True)
        stats = get_file_stats(str(config.DATA_DIR))
        existing = stats["files"]

        if not existing:
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:36px;margin-bottom:12px;">📂</div>
              <div style="font-size:16px;font-weight:600;color:var(--tx-hi);margin-bottom:6px;">No documents yet</div>
              <div style="font-size:13px;color:var(--tx-mid);">Upload files on the left to get started.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div class="stats-strip">
                  <div class="stat-box"><div class="stat-k">Files</div><div class="stat-v">{stats["count"]}</div></div>
                  <div class="stat-box"><div class="stat-k">Corpus size</div><div class="stat-v">{stats["total_kb"]:.1f} KB</div></div>
                  <div class="stat-box"><div class="stat-k">Retrieval</div><div class="stat-v">Hybrid</div></div>
                </div>""",
                unsafe_allow_html=True,
            )
            for fp in existing:
                icon = ICON.get(fp.suffix.lower().lstrip("."), "📄")
                size_kb = fp.stat().st_size / 1024
                st.markdown(
                    f'<div class="card"><div class="row">'
                    f'<span class="lbl">{icon} {fp.name}</span>'
                    f'<span class="val">{size_kb:.1f} KB</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("<hr/>", unsafe_allow_html=True)
            if st.button(":material/delete: Clear entire index", use_container_width=True):
                import shutil
                storage = Path(config.STORAGE_DIR)
                if storage.exists():
                    shutil.rmtree(storage)
                for fp in existing:
                    fp.unlink(missing_ok=True)
                get_file_stats.clear()
                st.cache_resource.clear()
                st.success("Index cleared.", icon=":material/delete:")
                time.sleep(0.6)
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not system_ok:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:36px;margin-bottom:12px;">◆</div>
          <div style="font-size:16px;font-weight:600;color:var(--tx-hi);margin-bottom:6px;">No documents indexed</div>
          <div style="font-size:13px;color:var(--tx-mid);">Go to the <strong>📁 Documents</strong> tab and upload files first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        known_files = system.retrieval.list_known_files()
        stats = get_file_stats(str(config.DATA_DIR))

        # ── Toolbar ────────────────────────────────────────────────────────────
        hdr_l, hdr_r = st.columns([5, 1])
        with hdr_l:
            st.markdown(
                f'<div class="workspace-note">'
                f'<strong>{stats["count"]}</strong> file(s) indexed &nbsp;·&nbsp; '
                f'Use the action buttons for guided tasks, or just type naturally.'
                f'</div>',
                unsafe_allow_html=True,
            )
        with hdr_r:
            if st.button(":material/delete: Clear", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.session_state.memory = MemoryAgent()
                st.session_state.pending_image = None
                st.session_state.prefilled_prompt = ""
                st.session_state.pending_task_request = None
                st.rerun()

        # ── Flush deferred task ────────────────────────────────────────────────
        if st.session_state.pending_task_request is not None:
            req = st.session_state.pending_task_request
            st.session_state.pending_task_request = None
            st.session_state.active_task = None
            st.session_state.task_selected_files = []
            st.session_state.task_extract_instruction = ""
            run_task(req["question"], forced_route=req["forced_route"], target_files=req["target_files"])
            st.rerun()

        # ── Empty-state action tiles ───────────────────────────────────────────
        if not st.session_state.messages:
            st.markdown(
                """<div class="action-grid">
                  <div class="action-tile">
                    <div class="action-tile-icon">📝</div>
                    <div class="action-tile-title">Summarize</div>
                    <div class="action-tile-desc">Executive, bullet, or detailed summary of any file.</div>
                  </div>
                  <div class="action-tile">
                    <div class="action-tile-icon">⚖️</div>
                    <div class="action-tile-title">Compare</div>
                    <div class="action-tile-desc">Side-by-side diff between two or more documents.</div>
                  </div>
                  <div class="action-tile">
                    <div class="action-tile-icon">💻</div>
                    <div class="action-tile-title">Generate code</div>
                    <div class="action-tile-desc">Turn a paper or spec into working Python.</div>
                  </div>
                  <div class="action-tile">
                    <div class="action-tile-icon">🗂️</div>
                    <div class="action-tile-title">Extract data</div>
                    <div class="action-tile-desc">Pull structured information into a clean table.</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            selected_chip = st.pills(
                "Quick starts",
                options=list(SUGGESTIONS.keys()),
                label_visibility="collapsed",
                selection_mode="single",
                key="starter_chips",
            )
            if selected_chip:
                st.session_state.prefilled_prompt = SUGGESTIONS[selected_chip]
                st.rerun()

        # ── Chat history ───────────────────────────────────────────────────────
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("image") is not None:
                    st.image(msg["image"], width=260)
                if msg.get("dataframe") is not None:
                    st.markdown(msg["content"])
                    st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)
                else:
                    st.markdown(msg["content"])

                if msg.get("meta"):
                    m = msg["meta"]
                    render_meta(m["route"], m["elapsed"], msg.get("verification"))

                render_sources(msg.get("sources", []))

                if msg.get("trace") or msg.get("execution_plan"):
                    render_dev_panel(
                        msg.get("trace", []),
                        msg.get("execution_plan", {}),
                        msg.get("cache_hits", {}),
                        msg.get("meta", {}).get("elapsed", 0.0),
                    )

        # ── Action buttons ─────────────────────────────────────────────────────
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("📝 Summarize", use_container_width=True, key="btn_summarize"):
                st.session_state.active_task = "summarize"
                st.session_state.task_selected_files = []
        with a2:
            if st.button("🗂️ Extract table", use_container_width=True, key="btn_table"):
                st.session_state.active_task = "table"
                st.session_state.task_selected_files = []
        with a3:
            if st.button("⚖️ Compare", use_container_width=True, key="btn_compare"):
                st.session_state.active_task = "compare"
                st.session_state.task_selected_files = []

        # ── Task form ──────────────────────────────────────────────────────────
        task = st.session_state.active_task
        if task:
            meta = {
                "summarize": ("📝 Summarize", "Choose one or more documents to summarize."),
                "table": ("🗂️ Extract table", "Choose documents and describe what to extract."),
                "compare": ("⚖️ Compare", "Choose two or more documents to compare."),
            }
            title, sub = meta[task]
            st.markdown(
                f'<div class="task-card">'
                f'<div class="task-title">{title}</div>'
                f'<div class="task-sub">{sub}</div>',
                unsafe_allow_html=True,
            )
            with st.form(f"task_{task}", border=False):
                min_files = 2 if task == "compare" else 1
                selected = st.multiselect(
                    "Files", options=known_files, key="task_selected_files",
                    label_visibility="collapsed",
                )
                extra = ""
                if task == "table":
                    extra = st.text_input(
                        "What to extract?",
                        placeholder="e.g. name, date, and total for each entry",
                        key="task_extract_instruction",
                    )
                ok_disabled = len(selected) < min_files or (task == "table" and not extra.strip())
                fc1, fc2 = st.columns([1, 1])
                with fc1:
                    run_ok = st.form_submit_button("Run", disabled=ok_disabled, use_container_width=True)
                with fc2:
                    cancel = st.form_submit_button("Cancel", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if cancel:
                st.session_state.active_task = None
                st.rerun()

            if run_ok:
                if task == "summarize":
                    q = f"Summarize these files: {', '.join(selected)}"
                elif task == "table":
                    q = f"{extra.strip()} from: {', '.join(selected)}"
                else:
                    q = f"Compare these files: {', '.join(selected)}"
                st.session_state.pending_task_request = {
                    "question": q, "forced_route": task, "target_files": selected,
                }
                st.rerun()

        # ── Image attach ───────────────────────────────────────────────────────
        with st.expander("📎 Attach image", expanded=False):
            attached = st.file_uploader(
                "Image to ask about",
                type=[e.lstrip(".") for e in sorted(IMAGE_EXT)],
                key="chat_image_attach", label_visibility="collapsed",
            )
            if attached:
                st.session_state.pending_image = {
                    "bytes": attached.getvalue(), "mime": attached.type or "image/png",
                }
                st.caption(f"✓ {attached.name} — ask below, send blank to describe it, "
                           f"or run real object detection with the buttons below.")

                if system.vision_analysis is not None and system.vision_analysis.available:
                    show_labels = st.checkbox("Show labels & confidence on image", value=True, key="yolo_show_labels")
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        if st.button("🎯 Detect objects (YOLO)", use_container_width=True, key="btn_detect_only"):
                            img_bytes = st.session_state.pending_image["bytes"]
                            st.session_state.pending_image = None
                            run_vision_analysis_task(img_bytes, with_vlm_comparison=False, show_labels=show_labels)
                            st.rerun()
                    with vc2:
                        if st.button("🔍 Detect + compare vs. VLM", use_container_width=True, key="btn_detect_compare"):
                            img_bytes = st.session_state.pending_image["bytes"]
                            st.session_state.pending_image = None
                            run_vision_analysis_task(img_bytes, with_vlm_comparison=True, show_labels=show_labels)
                            st.rerun()
                else:
                    st.caption("⚠️ Real object detection unavailable (ultralytics not installed).")

        # ── Chat input ─────────────────────────────────────────────────────────
        placeholder = st.session_state.prefilled_prompt or "Ask anything…"
        question = st.chat_input(placeholder, submit_mode="disable")

        if question:
            st.session_state.prefilled_prompt = ""
            pending = st.session_state.pending_image
            if pending:
                st.session_state.pending_image = None
                run_image_task(question, pending["bytes"], pending["mime"])
            else:
                run_task(question)
            st.rerun()
