-- Phase 2: build the similarity-search index after data is loaded.
-- HNSW with cosine distance; embeddings are pre-normalized so this is
-- equivalent to dot-product / cosine-similarity ranking.
CREATE INDEX IF NOT EXISTS problems_embedding_hnsw
    ON problems USING hnsw (embedding vector_cosine_ops);

ANALYZE problems;
