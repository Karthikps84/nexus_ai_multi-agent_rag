from langchain_core.embeddings import DeterministicFakeEmbedding

from vectorstore.chroma_store import VectorStoreManager


def _manager(tmp_path):
    return VectorStoreManager(tmp_path / "data", tmp_path / "storage", DeterministicFakeEmbedding(size=16))


def _sources(chunks):
    return sorted({c.metadata["source_file"] for c in chunks})


def test_empty_library_starts_with_empty_index(tmp_path):
    assert _manager(tmp_path).initialize() == []


def test_add_change_remove_files_incrementally(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("alpha " * 50)
    assert _sources(_manager(tmp_path).initialize()) == ["a.txt"]

    (tmp_path / "data" / "b.txt").write_text("bravo " * 50)
    assert _sources(_manager(tmp_path).initialize()) == ["a.txt", "b.txt"]

    (tmp_path / "data" / "a.txt").unlink()
    assert _sources(_manager(tmp_path).initialize()) == ["b.txt"]


def test_remove_file_drops_chunks_and_manifest(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("alpha " * 50)
    (tmp_path / "data" / "b.txt").write_text("bravo " * 50)
    vm = _manager(tmp_path)
    vm.initialize()
    vm.remove_file("a.txt")
    assert _sources(vm.all_chunks()) == ["b.txt"]
    assert "a.txt" not in vm._load_manifest()
    (tmp_path / "data" / "a.txt").unlink()
    assert _sources(_manager(tmp_path).initialize()) == ["b.txt"]


def test_deleting_every_file_leaves_a_working_empty_index(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("alpha " * 50)
    _manager(tmp_path).initialize()
    (tmp_path / "data" / "a.txt").unlink()
    assert _manager(tmp_path).initialize() == []
