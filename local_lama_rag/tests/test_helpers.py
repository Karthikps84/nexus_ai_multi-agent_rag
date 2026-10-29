from pathlib import Path

from utils.helpers import compute_dataset_hash, content_key, token_count, trim_history


def test_token_count_counts_tokens():
    assert token_count("") == 0
    assert token_count("hello world") > 0
    assert token_count("hello world") == token_count("hello world")


def test_trim_history_keeps_last_n():
    history = [(f"q{i}", f"a{i}") for i in range(10)]
    assert trim_history(history, 3) == history[-3:]
    assert trim_history(history, 0) == []
    assert trim_history([], 5) == []


def test_content_key_stable_and_sensitive_to_content():
    a = content_key("hello world")
    b = content_key("hello world")
    c = content_key("hello world!")
    assert a == b
    assert a != c


def test_compute_dataset_hash_changes_with_content(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha")
    h1 = compute_dataset_hash(tmp_path)

    (tmp_path / "b.txt").write_text("beta")
    h2 = compute_dataset_hash(tmp_path)

    assert h1 != h2
    assert compute_dataset_hash(tmp_path) == h2
