"""
Phase 2: embed every normalized problem with bge-small-en-v1.5 and load
problem + embedding into Postgres (pgvector).

BGE models are trained with an asymmetric convention: documents are embedded
as-is, but queries should be prefixed with an instruction. We follow that
here so retrieval quality matches how the model was trained -- see
`query_prefix` in 06_retrieve.py for the query-side counterpart.
"""
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def load_rows():
    path = PROCESSED_DIR / "problems.jsonl"
    return [json.loads(line) for line in path.open()]


def embed_text(row: dict) -> str:
    # Tags carry pattern signal that plain statement text often doesn't
    # spell out explicitly (e.g. "Sliding Window" never appears in the
    # problem text itself) -- folding them in helps retrieval match on
    # pattern, not just surface topic (e.g. "array").
    tag_str = ", ".join(row["tags"])
    return f"{row['title']}\nTags: {tag_str}\n\n{row['problem_statement']}"


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} problems to embed.")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [embed_text(r) for r in rows]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # so pgvector cosine distance == dot product
    )

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for row, emb in tqdm(list(zip(rows, embeddings)), desc="loading"):
                cur.execute(
                    """
                    INSERT INTO problems (
                        problem_id, slug, title, difficulty, tags,
                        problem_statement, approach_notes, estimated_date,
                        source, source_split, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (problem_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        row["problem_id"], row["slug"], row["title"],
                        row["difficulty"], row["tags"],
                        row["problem_statement"], row["approach_notes"],
                        row["estimated_date"], row["source"],
                        row["source_split"], emb,
                    ),
                )
        conn.commit()

    print("Done loading problems + embeddings.")


if __name__ == "__main__":
    main()
