"""
==========================================================
Chroma vector store — build/load with resilient embedding
==========================================================
"""

from __future__ import annotations

import json
import time
import uuid
import concurrent.futures
from pathlib import Path
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDoc

import config
from utils.logger import get_logger
from utils.helpers import token_count, compute_dataset_hash
from loaders.document_loader import load_documents

logger = get_logger("vectorstore")

ProgressFn = Optional[Callable[[int, str], None]]


class VectorStoreManager:
    def __init__(self, data_dir: Path, storage_dir: Path, embeddings: NVIDIAEmbeddings):
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.vectorstore: Optional[Chroma] = None

    # ── dataset change detection ────────────────────────────────────────────
    def _hash_file(self) -> Path:
        return self.storage_dir / "dataset.json"

    def _current_manifest(self) -> dict:
        supported = {e.lower() for e in config.SUPPORTED_EXTENSIONS}
        out = {}
        for p in sorted(self.data_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in supported:
                st = p.stat()
                out[p.name] = [st.st_size, int(st.st_mtime)]
        return out

    def _load_manifest(self) -> Optional[dict]:
        try:
            with open(self._hash_file()) as f:
                return json.load(f).get("files")
        except Exception:
            return None

    def _save_hash(self):
        with open(self._hash_file(), "w") as f:
            json.dump(
                {"dataset_hash": compute_dataset_hash(self.data_dir), "files": self._current_manifest()},
                f, indent=4,
            )

    def _dataset_changed(self) -> bool:
        hash_file = self._hash_file()
        if not hash_file.exists():
            return True
        with open(hash_file) as f:
            old_hash = json.load(f)["dataset_hash"]
        return old_hash != compute_dataset_hash(self.data_dir)

    # ── sanitization ─────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_chunks(chunks: list) -> list:
        import re
        import unicodedata

        control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        pua_re = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

        clean, dropped = [], 0
        for c in chunks:
            text = unicodedata.normalize("NFKC", c.page_content or "")
            text = control_re.sub("", text)
            text = pua_re.sub(" ", text).strip()
            if not text:
                dropped += 1
                continue
            printable = sum(1 for ch in text if ch.isprintable())
            if printable / max(len(text), 1) < 0.7:
                dropped += 1
                continue
            c.page_content = text
            clean.append(c)
        if dropped:
            logger.info(f"Sanitization dropped {dropped} empty/garbled chunk(s)")
        return clean

    # ── build ────────────────────────────────────────────────────────────────
    def build(self, on_progress: ProgressFn = None):
        def report(pct, msg):
            logger.info(msg)
            if on_progress:
                on_progress(pct, msg)

        report(2, "Loading documents...")
        documents = load_documents(self.data_dir)
        total_tokens = sum(token_count(d.page_content) for d in documents)
        report(8, f"Loaded {len(documents)} document(s), {total_tokens:,} tokens")

        report(12, "Splitting into chunks...")
        to_split = [d for d in documents if not d.metadata.get("structured")]
        keep_whole = [d for d in documents if d.metadata.get("structured")]
        chunks = self.splitter.split_documents(to_split) + keep_whole
        chunks = self._sanitize_chunks(chunks)
        report(18, f"Created {len(chunks)} chunks")

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        start = time.time()
        self._embed_and_insert(chunks, on_progress=on_progress)
        report(92, f"Vector store built in {time.time() - start:.2f}s")

        self._save_hash()
        report(100, "Index ready")
        return chunks  # caller (RetrievalAgent) uses this to build the BM25 index

    def _embed_and_insert(self, chunks: list, batch_size: int = None, on_progress: ProgressFn = None):
        batch_size = batch_size or config.EMBED_BATCH_SIZE
        embed_workers = config.EMBED_WORKERS
        batches = [chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Embedding {len(chunks)} chunks in {len(batches)} batches "
                    f"(batch_size={batch_size}, workers={embed_workers})...")

        def _embed_batch_safe(batch: list, size_hint: int):
            current_size = size_hint
            while current_size >= 1:
                sub_batches = [batch[j: j + current_size] for j in range(0, len(batch), current_size)]
                try:
                    results = []
                    for sb in sub_batches:
                        vectors = self.embeddings.embed_documents([d.page_content for d in sb])
                        results.append((sb, vectors))
                    return results
                except Exception as e:
                    if current_size <= 1:
                        logger.error(f"Skipping unembeddable chunk: {e}")
                        return None
                    logger.warning(f"Embed batch failed (size={current_size}): {e}. Retrying smaller.")
                    current_size = max(1, current_size // 2)
                    time.sleep(1)
            return None

        batch_results: list = [None] * len(batches)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=embed_workers) as pool:
            future_to_idx = {pool.submit(_embed_batch_safe, b, batch_size): i for i, b in enumerate(batches)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch {idx} embed worker raised: {e}")
                completed += 1
                if on_progress and batches:
                    pct = 20 + int(70 * completed / len(batches))
                    on_progress(pct, f"Embedding batch {completed}/{len(batches)}...")

        for result in batch_results:
            if result is None:
                continue
            for sub_batch, vectors in result:
                ids = [str(uuid.uuid4()) for _ in sub_batch]
                try:
                    self.vectorstore._collection.add(
                        ids=ids, embeddings=vectors,
                        documents=[d.page_content for d in sub_batch],
                        metadatas=[d.metadata for d in sub_batch],
                    )
                except Exception:
                    try:
                        self.vectorstore.add_documents(sub_batch, ids=ids)
                    except Exception as e2:
                        logger.error(f"Insert fallback also failed: {e2}")

    # ── load ─────────────────────────────────────────────────────────────────
    def load_existing(self):
        logger.info("Loading existing Chroma vector store...")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        return [LCDoc(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]

    def initialize(self, on_progress: ProgressFn = None):
        """Returns the full chunk list either way, so the caller can build BM25 from it."""
        self.storage_dir.mkdir(exist_ok=True)
        self._validate_data_folder()
        has_existing = self.storage_dir.exists() and any(self.storage_dir.iterdir())

        if has_existing and not self._dataset_changed():
            logger.info("Dataset unchanged. Loading existing index.")
            if on_progress:
                on_progress(30, "Loading existing index...")
            chunks = self.load_existing()
            if on_progress:
                on_progress(100, "Index ready")
            return chunks

        if has_existing:
            try:
                return self._sync_incremental(on_progress=on_progress)
            except Exception as e:
                logger.warning(f"Incremental sync failed ({e}); falling back to full rebuild.")

        logger.info("No index found (or sync failed). Rebuilding.")
        return self.build(on_progress=on_progress)

    def _sync_incremental(self, on_progress: ProgressFn = None):
        """Embed only added/changed files; drop chunks of removed/changed ones."""
        existing_chunks = self.load_existing()
        current = self._current_manifest()
        manifest = self._load_manifest()
        if manifest is None:
            # Legacy dataset.json: treat files already present in the index as unchanged.
            indexed = {c.metadata.get("source_file") for c in existing_chunks}
            manifest = {n: v for n, v in current.items() if n in indexed}

        removed = [n for n in manifest if n not in current]
        changed = [n for n in current if n in manifest and list(manifest[n]) != list(current[n])]
        added = [n for n in current if n not in manifest]
        logger.info(f"Incremental sync: +{len(added)} new, ~{len(changed)} changed, -{len(removed)} removed")

        for name in removed + changed:
            self.vectorstore._collection.delete(where={"source_file": name})

        to_index = added + changed
        if to_index:
            if on_progress:
                on_progress(10, f"Loading {len(to_index)} new/changed file(s)...")
            documents = load_documents(self.data_dir, only=set(to_index))
            to_split = [d for d in documents if not d.metadata.get("structured")]
            keep_whole = [d for d in documents if d.metadata.get("structured")]
            chunks = self._sanitize_chunks(self.splitter.split_documents(to_split) + keep_whole)
            self._embed_and_insert(chunks, on_progress=on_progress)

        self._save_hash()
        if on_progress:
            on_progress(100, "Index ready")
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        return [LCDoc(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]

    def _validate_data_folder(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}.")
        files = [p for p in self.data_dir.iterdir() if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No files found in {self.data_dir}.")
        supported = [p for p in files if p.suffix.lower() in config.SUPPORTED_EXTENSIONS]
        if not supported:
            raise FileNotFoundError(f"No supported documents in {self.data_dir}.")

    def fetch_file_chunks(self, filename: str, max_chunks: int) -> list:
        try:
            raw = self.vectorstore.get(where={"source_file": filename}, include=["documents", "metadatas"])
            docs = [LCDoc(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]
            docs.sort(key=lambda d: d.metadata.get("page", 0) if isinstance(d.metadata.get("page"), int) else 0)
            return docs[:max_chunks]
        except Exception as e:
            logger.warning(f"Could not fetch chunks for {filename}: {e}")
            return []

    def list_known_files(self) -> list:
        if not self.data_dir.exists():
            return []
        return sorted(p.name for p in self.data_dir.glob("*") if p.is_file())
