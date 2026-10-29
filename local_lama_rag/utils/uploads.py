from pathlib import Path

import config


def save_upload_bytes(filename: str, content: bytes, data_dir: Path) -> bool:
    """Write one uploaded file's bytes to data_dir if its extension is supported.
    Returns True if saved, False if skipped (unsupported extension)."""
    if Path(filename).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / filename).write_bytes(content)
    return True
