"""
Phase 3: given a new problem + retrieved similar problems, ask a local LLM
(via Ollama) to identify the likely pattern(s) and explain why, citing the
retrieved examples as evidence -- a tutor, not a solution generator.

Runs entirely locally (no API key, no cost) using qwen3:latest, already
pulled on this machine. Structured output is enforced via Ollama's
grammar-constrained decoding (`format=<json schema>`), so the model cannot
emit a response that violates PatternAnalysis's schema.

Design note: the retrieved context sent to the model deliberately excludes
`approach_notes` (title/tags/difficulty/statement only). approach_notes
contains full solution code (see Phase 1/2 caveats in the README) -- rather
than relying on a prompt instruction to stop the model from relaying that
code, we just never put it in the context at all. Structurally safer than
policing it after the fact.

Phase 5 addition: an optional `allowed_patterns` vocabulary constrains the
`patterns` field via a JSON-schema enum (still grammar-enforced, so the
model literally cannot emit a tag outside the list) -- used by the eval
script for clean, unambiguous scoring against ground-truth LeetCode tags.
It's off by default so Phase 4's live UI keeps natural, unconstrained
pattern names. `generate_pattern_analysis_baseline` is the no-retrieval
comparison point for the same eval.

Known limitation, empirically verified (not assumed): Ollama's
grammar-constrained decoding enforces `type`/`maxItems`/`minItems`/`enum`
(these compile cleanly into a generation grammar -- each token choice only
needs local context) but does NOT enforce `uniqueItems` (this requires
tracking generation history across the whole array, which a
context-free grammar can't express). Declaring `uniqueItems: true` in the
schema is harmless but was confirmed to do nothing on its own -- re-ran the
exact problem that produced 5x "Dynamic Programming" and got
`["Dynamic Programming", "Breadth-First Search", "Dynamic Programming",
"Dynamic Programming", "Dynamic Programming"]`, still repeating. The actual
fix is `_dedupe_patterns()` below: deterministic post-processing, since
distinctness can't be guaranteed at the decoding layer with this setup.
"""
from typing import Literal

import ollama
from pydantic import BaseModel, Field

from .retrieval import retrieve_similar

MODEL = "qwen3:latest"

RAG_SYSTEM_PROMPT = """\
You are a tutor helping a student recognize algorithmic patterns in coding \
interview problems (e.g. sliding window, two pointers, dynamic programming, \
graph traversal, binary search, backtracking, union find).

You will be given a new problem and a set of similar problems retrieved by \
embedding similarity, each with its title, difficulty, and known pattern \
tags. Your job:

1. Identify which pattern(s) most likely apply to the new problem.
2. Explain WHY, explicitly referencing the retrieved similar problems by \
title as evidence -- e.g. "like 'Longest Substring Without Repeating \
Characters', this problem asks for a contiguous run satisfying a \
constraint, which is the signature of Sliding Window."
3. Rate your confidence.

Hard rule: you are a tutor, not a solution generator. Never output code, \
pseudocode, or a step-by-step algorithm. Only identify the pattern and \
explain the reasoning at a conceptual level. If the student wants the \
actual solution, that's not your job.

If the retrieved problems don't clearly support any pattern, say so plainly \
in your reasoning and lower your confidence -- do not force a match.\
"""

BASELINE_SYSTEM_PROMPT = """\
You are a tutor helping a student recognize algorithmic patterns in coding \
interview problems (e.g. sliding window, two pointers, dynamic programming, \
graph traversal, binary search, backtracking, union find).

You will be given a new problem with no other context. Your job:

1. Identify which pattern(s) most likely apply to the new problem.
2. Explain your reasoning conceptually.
3. Rate your confidence.

Hard rule: you are a tutor, not a solution generator. Never output code, \
pseudocode, or a step-by-step algorithm. Only identify the pattern and \
explain the reasoning at a conceptual level.\
"""


class PatternAnalysis(BaseModel):
    patterns: list[str] = Field(
        description="Likely pattern(s), most likely first, e.g. ['Sliding Window', 'Hash Table']"
    )
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(
        description="Explanation that references retrieved problems by title as evidence"
    )
    cited_problem_titles: list[str] = Field(
        default_factory=list,
        description="Titles of retrieved problems actually used as evidence, subset of what was retrieved",
    )


def build_output_schema(allowed_patterns: list[str] | None = None) -> dict:
    schema = PatternAnalysis.model_json_schema()
    # Cap array length -- an enum constrains which strings are legal but not
    # how many, and a local model with no retrieved evidence to ground it
    # can degenerate into cycling through the whole vocabulary with
    # duplicates (observed: 31 items, repeats, 12 minutes to emit on this
    # hardware). Real patterns lists are short and ranked; 5 is generous.
    schema["properties"]["patterns"]["maxItems"] = 5
    schema["properties"]["patterns"]["minItems"] = 1
    # Declared for documentation/future-compatibility, but Ollama's grammar
    # decoder does not actually enforce this -- see module docstring.
    # Real dedup happens in _dedupe_patterns() after parsing.
    schema["properties"]["patterns"]["uniqueItems"] = True
    if allowed_patterns:
        schema["properties"]["patterns"]["items"] = {"type": "string", "enum": allowed_patterns}
    return schema


def build_user_message(problem_statement: str, retrieved: list[dict]) -> str:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"- \"{r['title']}\" ({r['difficulty']}, tags: {', '.join(r['tags'])})\n"
            f"  {r['problem_statement'][:400].strip()}"
        )
    context = "\n".join(context_blocks)

    return (
        f"New problem:\n{problem_statement}\n\n"
        f"Retrieved similar problems (by embedding similarity):\n{context}"
    )


def _vocabulary_note(allowed_patterns: list[str] | None) -> str:
    if not allowed_patterns:
        return ""
    return (
        "\n\nYou must choose `patterns` only from this exact vocabulary "
        f"(pick the ones that apply, most likely first): {', '.join(allowed_patterns)}. "
        "List each pattern at most once -- do not repeat a pattern to fill space."
    )


def _dedupe_patterns(analysis: PatternAnalysis) -> PatternAnalysis:
    """The real uniqueness enforcement -- see module docstring. Preserves the
    model's ranking (first occurrence order), just drops repeats."""
    seen = []
    for p in analysis.patterns:
        if p not in seen:
            seen.append(p)
    return analysis.model_copy(update={"patterns": seen})


def generate_pattern_analysis(
    problem_statement: str, k: int = 5, allowed_patterns: list[str] | None = None
) -> tuple[PatternAnalysis, list[dict]]:
    retrieved = retrieve_similar(problem_statement, k=k)
    # Only title/difficulty/tags/statement reach the prompt -- see module docstring.
    user_message = build_user_message(problem_statement, retrieved)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT + _vocabulary_note(allowed_patterns)},
            {"role": "user", "content": user_message},
        ],
        format=build_output_schema(allowed_patterns),
        think=True,
        options={"temperature": 0.2},
    )
    analysis = _dedupe_patterns(PatternAnalysis.model_validate_json(response.message.content))
    return analysis, retrieved


def generate_pattern_analysis_baseline(
    problem_statement: str, allowed_patterns: list[str] | None = None
) -> PatternAnalysis:
    """Phase 5 comparison point: same model, same output schema, no retrieval context."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT + _vocabulary_note(allowed_patterns)},
            {"role": "user", "content": f"New problem:\n{problem_statement}"},
        ],
        format=build_output_schema(allowed_patterns),
        think=True,
        options={"temperature": 0.2},
    )
    return _dedupe_patterns(PatternAnalysis.model_validate_json(response.message.content))
