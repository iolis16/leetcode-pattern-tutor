"""
Shared retrieval logic (Phase 2), importable from later phases (generation,
API server) without depending on the numbered pipeline-script filenames.

BGE's asymmetric convention: queries get a retrieval-instruction prefix,
documents (already embedded in 04_embed_and_load.py) do not. Mismatching
this measurably hurts retrieval quality for BGE models.

Upgraded from bge-small (384-dim) to bge-base (768-dim) -- see README
"Is there a way for RAG to beat baseline?" / lever #2. Manual tracing of
Phase 5's RAG failures found the true bottleneck was retrieval RECALL: the
correct pattern's tags weren't present anywhere in bge-small's top-15
neighbors for the failing cases, at any rank -- re-ranking (lever #1)
can't fix an answer that was never retrieved. A stronger embedding model
was the direct next lever to test. `problems.embedding` is `vector(768)`
to match; a full re-embed of the corpus is required after this change
(old 384-dim vectors are not compatible/comparable with new 768-dim ones).
"""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str):
    model = get_model()
    return model.encode(QUERY_PREFIX + text, normalize_embeddings=True)


def retrieve_similar(problem_statement: str, k: int = 5, exclude_problem_id: str | None = None):
    """Return the top-k most similar problems to `problem_statement`."""
    query_vec = embed_query(problem_statement)

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT problem_id, slug, title, difficulty, tags,
                       problem_statement, approach_notes,
                       1 - (embedding <=> %s) AS similarity
                FROM problems
                WHERE %s::text IS NULL OR problem_id != %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, exclude_problem_id, exclude_problem_id, query_vec, k),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
