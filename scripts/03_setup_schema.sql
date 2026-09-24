-- Phase 2: schema for problems + their embeddings.
-- bge-base-en-v1.5 produces 768-dim vectors (upgraded from bge-small/384 --
-- see lib/retrieval.py docstring).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS problems (
    problem_id          TEXT PRIMARY KEY,
    slug                TEXT NOT NULL,
    title               TEXT NOT NULL,
    difficulty          TEXT NOT NULL,
    tags                TEXT[] NOT NULL,
    problem_statement   TEXT NOT NULL,
    approach_notes      TEXT NOT NULL,
    estimated_date      TEXT,
    source              TEXT NOT NULL,
    source_split        TEXT NOT NULL,
    embedding            vector(768)
);

CREATE INDEX IF NOT EXISTS problems_tags_gin ON problems USING GIN (tags);

-- HNSW index for cosine-distance similarity search.
-- Built after data + embeddings are loaded (see 05_build_index.sql).
