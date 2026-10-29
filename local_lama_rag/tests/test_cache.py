import pytest

from orchestration.cache import CacheConfig, CacheManager, CachedEmbeddings


@pytest.fixture
def cache_manager():
    config = CacheConfig(
        response_ttl_seconds=60,
        retrieval_ttl_seconds=60,
        embedding_ttl_seconds=60,
        response_max_entries=10,
        retrieval_max_entries=10,
        embedding_max_entries=10,
    )
    return CacheManager(config)


def test_response_key_is_deterministic_and_order_insensitive():
    k1 = CacheManager.response_key("What is X?", "qa", ["b.pdf", "a.pdf"])
    k2 = CacheManager.response_key("What is X?", "qa", ["a.pdf", "b.pdf"])
    assert k1 == k2


def test_response_key_differs_by_route():
    k1 = CacheManager.response_key("What is X?", "qa", [])
    k2 = CacheManager.response_key("What is X?", "summarize", [])
    assert k1 != k2


def test_set_and_get_response_exact_match(cache_manager):
    key = CacheManager.response_key("hello", "qa", [])
    assert cache_manager.get_response(key) is None
    cache_manager.set_response(key, {"answer": "world"})
    assert cache_manager.get_response(key) == {"answer": "world"}


def test_set_and_get_retrieval_exact_match(cache_manager):
    key = CacheManager.retrieval_key("hello", [], 8, 4)
    assert cache_manager.get_retrieval(key) is None
    cache_manager.set_retrieval(key, ["doc1"])
    assert cache_manager.get_retrieval(key) == ["doc1"]


def test_embedding_cache_respects_max_entries():
    config = CacheConfig(
        response_ttl_seconds=60, retrieval_ttl_seconds=60, embedding_ttl_seconds=60,
        response_max_entries=10, retrieval_max_entries=10, embedding_max_entries=2,
    )
    manager = CacheManager(config)
    manager.set_embedding("a", [1.0])
    manager.set_embedding("b", [2.0])
    manager.set_embedding("c", [3.0])  # evicts the oldest (TTLCache is LRU-like under maxsize)
    assert len(manager.embedding_cache) == 2


def test_cached_embeddings_caches_query(cache_manager):
    calls = []

    class FakeEmbeddings:
        def embed_query(self, text):
            calls.append(text)
            return [0.1, 0.2]

    cached = CachedEmbeddings(FakeEmbeddings(), cache_manager)
    v1 = cached.embed_query("hello")
    v2 = cached.embed_query("hello")
    assert v1 == v2 == [0.1, 0.2]
    assert len(calls) == 1  # second call hit the cache, no underlying call


def test_cached_embeddings_documents_partial_cache_hit(cache_manager):
    calls = []

    class FakeEmbeddings:
        def embed_documents(self, texts):
            calls.append(list(texts))
            return [[float(len(t))] for t in texts]

    cached = CachedEmbeddings(FakeEmbeddings(), cache_manager)
    cached.embed_documents(["alpha"])
    result = cached.embed_documents(["alpha", "beta"])

    assert len(result) == 2
    assert calls[-1] == ["beta"]  # only the uncached text was re-embedded
