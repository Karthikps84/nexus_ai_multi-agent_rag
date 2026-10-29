"""
==========================================================
Eval Set Builder
==========================================================
Run this FIRST before evaluation.

It reads every file in your data/ directory, asks the LLM
to generate grounded question-answer pairs from actual content,
and writes them to eval/eval_set.json.

Each entry has:
  question      - a realistic user question grounded in the doc
  ground_truth  - a short, verifiable correct answer from the doc
  source_file   - which file it came from
  page          - page number (if available)
  difficulty    - easy / medium / hard (for stratified analysis)

Usage:
    python eval/build_eval_set.py
    python eval/build_eval_set.py --files resume.pdf paper.pdf
    python eval/build_eval_set.py --per-file 5 --include-hard

The script generates PAIRS_PER_FILE pairs per document. Review
and correct any wrong ground_truths before running the full eval.
==========================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

import config
from vectorstore.chroma_store import VectorStoreManager
from orchestration.cache import CacheConfig, CacheManager, CachedEmbeddings
from utils.logger import get_logger

load_dotenv()
logger = get_logger("build_eval_set")

PAIR_GENERATION_PROMPT = """You are a QA dataset builder. Given the text excerpt below, generate {n} realistic question-answer pairs that someone would actually ask while studying this document.

Rules:
1. Questions must be ANSWERABLE from the text alone — no outside knowledge.
2. Ground truths must be SHORT (1-3 sentences max) and factually correct.
3. Each pair must have a difficulty: easy (fact lookup), medium (needs understanding), or hard (needs reasoning/synthesis).
4. Return a valid JSON array only, no other text:

[
  {{"question": "...", "ground_truth": "...", "difficulty": "easy|medium|hard"}},
  ...
]

Text excerpt:
{text}
"""


def build_pairs(llm, text: str, source_file: str, page: int, n: int = 3) -> list[dict]:
    prompt = PAIR_GENERATION_PROMPT.format(text=text[:3000], n=n)
    try:
        raw = llm.invoke(prompt).content.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        pairs = json.loads(raw)
        for p in pairs:
            p["source_file"] = source_file
            p["page"] = page
        return pairs
    except Exception as e:
        logger.warning(f"Failed to generate pairs for {source_file} p{page}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Build an eval set from your indexed documents.")
    parser.add_argument("--files", nargs="*", help="Specific files to include (default: all)")
    parser.add_argument("--per-file", type=int, default=5, help="Q&A pairs per file (default: 5)")
    parser.add_argument("--include-hard", action="store_true", help="Include hard reasoning questions")
    parser.add_argument("--out", default="eval/eval_set.json", help="Output path")
    args = parser.parse_args()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY not set in .env")
        sys.exit(1)

    llm = ChatNVIDIA(
        model=config.FAST_LLM_MODEL,
        api_key=api_key,
        base_url=config.NVIDIA_BASE_URL,
        temperature=0.2,
        timeout=120,
        max_completion_tokens=config.LLM_MAX_TOKENS,
    )
    embeddings_raw = NVIDIAEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=api_key,
        base_url=config.NVIDIA_BASE_URL,
        truncate="END",
        max_batch_size=config.EMBED_BATCH_SIZE,
    )
    cache_manager = CacheManager(CacheConfig(60, 60, 3600, 64, 64, 512))
    embeddings = CachedEmbeddings(embeddings_raw, cache_manager)

    vs = VectorStoreManager(config.DATA_DIR, config.STORAGE_DIR, embeddings)
    vs.initialize()

    # Fetch a sample of chunks per file
    known_files = sorted(p.name for p in config.DATA_DIR.glob("*") if p.is_file())
    target_files = args.files if args.files else known_files

    all_pairs = []
    for fname in target_files:
        if fname not in known_files:
            logger.warning(f"File not indexed: {fname}, skipping")
            continue

        chunks = vs.fetch_file_chunks(fname, max_chunks=args.per_file * 2)
        if not chunks:
            logger.warning(f"No chunks found for {fname}")
            continue

        # Spread over pages — pick chunks that cover different pages
        seen_pages: set[int] = set()
        selected_chunks = []
        for c in chunks:
            page = c.metadata.get("page", 0)
            if page not in seen_pages:
                seen_pages.add(page)
                selected_chunks.append(c)
            if len(selected_chunks) >= max(2, args.per_file // 2):
                break

        pairs_wanted = args.per_file
        per_chunk = max(1, pairs_wanted // max(len(selected_chunks), 1))

        for chunk in selected_chunks:
            page = chunk.metadata.get("page", 0)
            pairs = build_pairs(llm, chunk.page_content, fname, page, n=per_chunk)
            if not args.include_hard:
                pairs = [p for p in pairs if p.get("difficulty") != "hard"]
            all_pairs.extend(pairs)
            logger.info(f"Generated {len(pairs)} pair(s) from {fname} p{page}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_pairs, indent=2))
    print(f"\nWrote {len(all_pairs)} Q&A pairs to {out_path}")
    print("\nIMPORTANT: Review eval_set.json and correct any wrong ground_truths before running eval.")
    print("  Difficulty breakdown: ", end="")
    for d in ("easy", "medium", "hard"):
        count = sum(1 for p in all_pairs if p.get("difficulty") == d)
        print(f"{d}={count}", end="  ")
    print()


if __name__ == "__main__":
    main()
