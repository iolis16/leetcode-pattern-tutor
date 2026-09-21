# LeetCode Pattern Tutor

A RAG-based tool that identifies the algorithmic pattern(s) behind a new
problem by retrieving similar problems and explaining the connection —
a tutor, not a solution generator.

## Status

- [x] Phase 1 — Data pipeline
- [x] Phase 2 — Embedding + retrieval
- [x] Phase 3 — Generation
- [x] Phase 4 — API + frontend
- [ ] Phase 5 — Evaluation
- [ ] Phase 6 — Deploy + polish

## Phase 1: Data pipeline

**Source:** [`newfacade/LeetCodeDataset`](https://huggingface.co/datasets/newfacade/LeetCodeDataset)
(Apache 2.0), built for the paper *LeetCodeDataset: A Temporal Dataset for
Robust Evaluation and Efficient Training of Code LLMs* (arXiv:2504.14655).

**Pipeline:**
1. `scripts/01_download_and_normalize.py` — pulls both splits from Hugging
   Face, normalizes into our own flat schema, de-dupes on `problem_id`,
   writes `data/processed/problems.jsonl`.
2. `scripts/02_profile_data.py` — sanity-checks tag/statement coverage and
   distribution.

**Normalized schema** (one JSON object per line):

| field | notes |
|---|---|
| `problem_id` | LeetCode question id |
| `slug` | e.g. `two-sum` |
| `title` | derived from slug (dataset has no title field) |
| `difficulty` | Easy / Medium / Hard |
| `tags` | list of pattern/topic tags, e.g. `["Array", "Hash Table"]` |
| `problem_statement` | full problem text |
| `approach_notes` | LLM-generated approach explanation + solution from the source dataset — **contains code**, so Phase 3's retrieval-context prompt must be careful not to just relay it verbatim |
| `estimated_date` | when the problem was likely added to LeetCode |
| `source`, `source_split` | provenance |

**Result:** 2,869 problems, 63 unique tags, 0 missing statements, 11 missing
tags (0.4%). Difficulty is reasonably balanced (686 Easy / 1,498 Medium /
685 Hard). Core patterns are well represented: Dynamic Programming (563),
Two Pointers (203), Sliding Window (144), Backtracking (101), Union Find
(84), Monotonic Stack (62), Topological Sort (34).

**Known caveat:** `approach_notes` is an LLM-generated explanation from the
source dataset's own paper (not an official LeetCode editorial), and it
includes full code. That's fine as retrieval context, but Phase 3's prompt
needs explicit instructions not to leak that code into the tutor's output.

## Phase 2: Embedding + retrieval

**Embedding model:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
via `sentence-transformers`, run locally on CPU (384-dim vectors, no API
cost). BGE uses an asymmetric convention: documents are embedded as-is,
queries get a `"Represent this sentence for searching relevant passages: "`
prefix — both sides are implemented consistently across the load and
retrieve scripts.

**Vector store:** Postgres + pgvector, **running locally for now** instead
of Supabase — same interface (it's literally the `vector` extension
Supabase also uses), just not yet on a hosted instance. Reasoning: getting
the pipeline correct didn't need an external account, and this environment
already had a different Postgres@14 service running for another project, so
a fresh, isolated instance was safer than touching it. Migrating to
Supabase in Phase 6 is a `DATABASE_URL` swap, no code changes.

- Dedicated Postgres 17 instance (pgvector's bottled build targets 17/18,
  not 14/16) lives at `.pgdata/`, port `5433` — isolated from any other
  Postgres on this machine. Manage it with `scripts/db.sh {start|stop|status|psql}`.
- `scripts/03_setup_schema.sql` — `problems` table (`vector(384)` column +
  GIN index on `tags`).
- `scripts/04_embed_and_load.py` — embeds `title + tags + problem_statement`
  per problem (tags are folded into the embedded text because a pattern
  like "Sliding Window" often never appears in the problem's own wording)
  and upserts into Postgres.
- `scripts/05_build_index.sql` — HNSW index (`vector_cosine_ops`), built
  after data load per pgvector's own guidance.
- `scripts/06_retrieve.py` — `retrieve_similar(problem_statement, k=5)`:
  embeds the query with the BGE query prefix, returns top-k by cosine
  similarity with tags/difficulty/approach notes attached.

**Sanity check:** ran retrieval on hand-written (non-corpus) phrasings of
House Robber (DP), Number of Islands (graph/BFS/DFS), and a sorted-array
two-sum variant (two pointers) — top-4 results were on-pattern in all three
cases, e.g. the islands query surfaced Number of Islands, Count Sub
Islands, Number of Closed Islands, Shortest Bridge, all tagged
DFS/BFS/Union-Find/Matrix.

**Known caveat:** `approach_notes` (returned alongside each retrieved
problem) still contains full solution code — carried over from the Phase 1
caveat. Phase 3's generation prompt must not relay it verbatim.

## Phase 3: Generation

**Model:** local LLM via [Ollama](https://ollama.com), `qwen3:latest` (8.2B,
Q4_K_M, already present on this machine) — **$0 cost, no API key**. This was
a deliberate tradeoff over the Claude API: genuinely free beats "cheap"
for a project with no revenue, at the cost of somewhat weaker
instruction-following/reasoning than a hosted frontier model. If a stronger
tutor voice matters more than $0 cost later, swapping `MODEL` in
`scripts/lib/generation.py` to the Anthropic SDK is a small, contained
change (the retrieval/prompt-building code is provider-agnostic).

- `scripts/lib/generation.py` — `generate_pattern_analysis(problem_statement, k=5)`:
  retrieves top-k similar problems, builds a prompt with only their
  title/difficulty/tags/statement (never `approach_notes`, which contains
  code — see Phase 1/2 caveats), and calls Ollama with a Pydantic schema
  (`PatternAnalysis`: `patterns`, `confidence`, `reasoning`,
  `cited_problem_titles`) passed as `format=` so **grammar-constrained
  decoding enforces the schema** — the model cannot return malformed output.
- System prompt explicitly forbids code/pseudocode and requires citing
  retrieved problems by title as evidence.
- `scripts/07_generate_analysis.py` — demo CLI.

**Sanity check:** ran on a verbatim Climbing Stairs restatement (correctly
identified DP/Memoization, cited "Climbing Stairs" by name, high
confidence) and a from-scratch phrasing of the meeting-rooms scheduling
problem (correctly identified Greedy + Heap, cited "Meeting Rooms II"
specifically, and explicitly explained why other retrieved-but-irrelevant
results like Walls and Gates didn't apply). No code or pseudocode appeared
in either output.

**Known caveat resolved:** the Phase 1/2 caveat about `approach_notes`
containing code is now moot for generation — that field is structurally
excluded from the prompt, not just discouraged by instruction.

## Phase 4: API + frontend

**Backend:** FastAPI (`app/main.py`), one real endpoint:

- `POST /api/analyze` — `{problem_statement, k}` → `{patterns, confidence,
  reasoning, cited_problem_titles, retrieved}`. Thin wrapper around
  `generate_pattern_analysis()` from Phase 3; validates input
  (`problem_statement` 20-8000 chars, `k` 1-10) and maps a down Postgres or
  Ollama to a `503` with a specific hint instead of a raw 500.
- `GET /api/health` — liveness check.
- The embedding model is preloaded at startup (FastAPI `lifespan`) instead
  of on first request, so the first `/api/analyze` call isn't slower than
  the rest.

**Frontend:** plain HTML/CSS/JS (`static/`) — no build step, no framework.
Mounted as static files on the same FastAPI app at `/`, so the frontend and
API share one origin and one process; no CORS configuration needed. This
was the deliberate "fastest" choice from the original spec over a React
app, appropriate for a single-page tool with one form and one result view.
Paste a problem, hit Analyze (or Cmd/Ctrl+Enter), see pattern chips,
confidence, reasoning, and the retrieved problems with the cited ones
visually marked. Each retrieved problem's title links out to its LeetCode
page (`https://leetcode.com/problems/<slug>/`, built from the `slug` field
already in the DB from Phase 1 — no extra lookup needed).

**Verified:** ran the full stack (Postgres + Ollama + FastAPI) and hit
`/api/analyze` with a from-scratch phrasing of the "insert interval into a
sorted, non-overlapping list, merging overlaps" problem — correctly
retrieved Insert Interval / Merge Intervals / Non Overlapping Intervals,
identified Greedy + Two Pointers, and cited three retrieved problems by
name in the reasoning. Confirmed `/` serves the frontend (200) and
`/api/health` responds.

**Known limitation:** a single `/api/analyze` call took roughly 60-90s in
testing — Qwen3's "thinking" mode is slow on this hardware (M2 Pro, 16GB).
That's the real cost of the $0-generation tradeoff from Phase 3. Options if
this matters more than cost later: disable `think` in
`lib/generation.py`, switch to a smaller/faster local model, or move
generation to a hosted API (Claude) as noted in the Phase 3 section.

**Not yet tested in an actual browser** — no browser automation tool was
available in this environment, so the UI was verified by reading the
HTML/CSS/JS and exercising the API directly with curl, not by clicking
through it. The page was opened in the system browser
(`open http://127.0.0.1:8000`) for manual visual confirmation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # DATABASE_URL for local Postgres
scripts/db.sh start
python3 scripts/01_download_and_normalize.py
python3 scripts/02_profile_data.py
psql -h 127.0.0.1 -p 5433 -U postgres -d leetcode_tutor -f scripts/03_setup_schema.sql
python3 scripts/04_embed_and_load.py
psql -h 127.0.0.1 -p 5433 -U postgres -d leetcode_tutor -f scripts/05_build_index.sql
python3 scripts/06_retrieve.py           # retrieval demo
ollama pull qwen3:latest                 # if not already present
python3 scripts/07_generate_analysis.py  # full pipeline demo

uvicorn app.main:app --reload            # http://127.0.0.1:8000
```
