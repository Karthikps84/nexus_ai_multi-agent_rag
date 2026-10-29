import hashlib
from pathlib import Path

import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_encoding.encode(text))


def compute_dataset_hash(data_dir: Path) -> str:
    sha = hashlib.sha256()
    for file in sorted(Path(data_dir).glob("*")):
        stat = file.stat()
        sha.update(file.name.encode())
        sha.update(str(stat.st_size).encode())
        sha.update(str(int(stat.st_mtime)).encode())
    return sha.hexdigest()


def trim_history(history: list, max_turns: int) -> list:
    return history[-max_turns:] if max_turns > 0 else []


def content_key(text: str) -> str:
    """Stable dedup key -- full-content hash, not a truncated prefix."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
