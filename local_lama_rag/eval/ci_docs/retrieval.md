# Nexus AI retrieval

Retrieval is hybrid. Dense vector search in Chroma is combined with BM25 keyword search, the two
result lists are fused, and a FlashRank re-ranker orders the final passages. Maximal marginal
relevance (MMR) removes near-duplicate passages.

Documents are split into chunks of 900 characters with an overlap of 150 characters.
In the reference configuration the dense search returns the top 10 candidates, the final answer
context uses the top 6 passages, and the similarity threshold is 0.30. When no passage reaches the
threshold, the system falls back to web search instead of guessing.

Answers cite their sources inline in the form [file page], for example [report.pdf p.3].
When the library is empty, the index starts empty and the first uploaded file builds it.
