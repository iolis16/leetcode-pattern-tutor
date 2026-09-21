"""
Shared retrieval logic (Phase 2), importable from later phases (generation,
API server) without depending on the numbered pipeline-script filenames.

BGE's asymmetric convention: queries get a retrieval-instruction prefix,
documents (already embedded in 04_embed_and_load.py) do not. Mismatching
this measurably hurts retrieval quality for BGE models.
"""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
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
