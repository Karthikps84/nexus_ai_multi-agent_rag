from retriever.hybrid_search import build_bm25_index, mmr_filter, reciprocal_rank_fusion


def test_build_bm25_index_empty_chunks_returns_none():
    index, chunks = build_bm25_index([])
    assert index is None
    assert chunks == []


def test_build_bm25_index_returns_index_for_chunks(sample_docs):
    index, chunks = build_bm25_index(sample_docs)
    assert chunks == sample_docs
    if index is not None:  # BM25Okapi present
        assert len(index.doc_freqs) == len(sample_docs)


def test_mmr_filter_empty_input_returns_empty():
    assert mmr_filter([], k=3) == []


def test_mmr_filter_respects_k(sample_docs):
    scored = [(d, 1.0 - i * 0.1) for i, d in enumerate(sample_docs)]
    selected = mmr_filter(scored, k=2)
    assert len(selected) == 2
    assert selected[0] is sample_docs[0]  # highest-relevance doc picked first


def test_mmr_filter_k_larger_than_pool_returns_all(sample_docs):
    scored = [(d, 1.0) for d in sample_docs]
    selected = mmr_filter(scored, k=100)
    assert len(selected) == len(sample_docs)


def test_reciprocal_rank_fusion_merges_dense_and_sparse(sample_docs):
    dense_scored = [(sample_docs[0], 0.9), (sample_docs[1], 0.5)]
    bm25_docs = [sample_docs[1], sample_docs[2]]

    fused = reciprocal_rank_fusion(dense_scored, bm25_docs)

    fused_contents = {d.page_content for d in fused}
    assert fused_contents == {sample_docs[0].page_content, sample_docs[1].page_content, sample_docs[2].page_content}
    # doc appearing in both lists (rank 1 in dense, rank 0 in sparse) should outrank
    # a doc appearing only once at a worse rank
    assert fused[0].page_content == sample_docs[1].page_content


def test_reciprocal_rank_fusion_no_duplicates(sample_docs):
    dense_scored = [(sample_docs[0], 0.9)]
    bm25_docs = [sample_docs[0]]
    fused = reciprocal_rank_fusion(dense_scored, bm25_docs)
    assert len(fused) == 1
