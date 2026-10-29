from pathlib import Path

import config


def safe_filename(filename: str) -> str:
    """Strip any directory part so an upload can never escape data_dir."""
    return Path((filename or "").replace("\\", "/")).name


def save_upload_bytes(filename: str, content: bytes, data_dir: Path) -> bool:
    """Write one uploaded file's bytes to data_dir if its extension is supported.
    Returns True if saved, False if skipped (unsupported extension, bad name or too large)."""
    name = safe_filename(filename)
    if not name or name.startswith("."):
        return False
    if Path(name).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return False
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / name).write_bytes(content)
    return True
