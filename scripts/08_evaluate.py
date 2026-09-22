"""
Phase 5: evaluate the RAG pipeline (retrieval + generation) against a
no-retrieval baseline (same model, same problem, no retrieved context) on
a genuine held-out test set.

Held-out set: the source dataset's own 228-problem `test` split, which was
deliberately deleted from the Postgres corpus before this script exists
(see README Phase 5) -- so retrieval cannot find these problems or their
exact duplicates. This is the difference between "does retrieval help" and
"did the model memorize the test set."

Ground truth / scoring vocabulary: PATTERN_TAGS below is a curated subset
of the dataset's 63 tags -- algorithmic techniques (Dynamic Programming,
Sliding Window, Union Find, ...) with generic input-shape tags (Array,
String, Matrix, Tree, Graph) and narrow domain tags (Math, Geometry, Number
Theory, Game Theory, Brainteaser, Concurrency) excluded. Rationale: this
project is about recognizing a *technique*, not describing what type the
input is -- "Array" appears on 1,769/2,869 problems and would make the eval
trivially easy without signaling anything about pattern recognition.

Both the RAG and baseline model calls are constrained (via JSON-schema enum,
grammar-enforced by Ollama) to only output tags from PATTERN_TAGS, which
makes scoring exact-string-match rather than fuzzy/subjective.

Metrics, computed per problem against its ground-truth tags filtered to
PATTERN_TAGS:
  - top1: is the model's first-ranked pattern in the ground truth set?
  - any: is ANY of the model's predicted patterns in the ground truth set?
(Ground truth is often multi-label, e.g. ["Greedy", "Sorting"], so `any` is
the more forgiving metric and `top1` the stricter one.)

Usage: python3 scripts/08_evaluate.py [--n N] [--seed SEED]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.generation import generate_pattern_analysis, generate_pattern_analysis_baseline

PATTERN_TAGS = [
    "Two Pointers", "Sliding Window", "Binary Search", "Depth-First Search",
    "Breadth-First Search", "Dynamic Programming", "Greedy", "Backtracking",
    "Union Find", "Topological Sort", "Trie", "Monotonic Stack",
    "Monotonic Queue", "Divide and Conquer", "Memoization", "Hash Table",
    "Heap (Priority Queue)", "Prefix Sum", "Bit Manipulation", "Sorting",
    "Shortest Path", "Recursion",
]

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
EVAL_DIR = Path(__file__).parent.parent / "data" / "eval"


def load_test_set() -> list[dict]:
    rows = [json.loads(line) for line in (PROCESSED_DIR / "problems.jsonl").open()]
    test_rows = [r for r in rows if r["source_split"] == "test"]
    return [r for r in test_rows if set(r["tags"]) & set(PATTERN_TAGS)]


def score(predicted: list[str], ground_truth: set[str]) -> dict:
    top1 = bool(predicted) and predicted[0] in ground_truth
    any_match = any(p in ground_truth for p in predicted)
    return {"top1": top1, "any": any_match}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="sample size from the held-out test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    usable = load_test_set()
    random.Random(args.seed).shuffle(usable)
    sample = usable[: args.n]

    print(f"Held-out usable problems: {len(usable)}, evaluating on {len(sample)} (seed={args.seed})")

    results = []
    out_path = EVAL_DIR / "results.jsonl"
    with out_path.open("w") as out_f:
        for i, problem in enumerate(sample, 1):
            ground_truth = set(problem["tags"]) & set(PATTERN_TAGS)
            t0 = time.time()

            rag_analysis, retrieved = generate_pattern_analysis(
                problem["problem_statement"], k=5, allowed_patterns=PATTERN_TAGS
            )
            rag_score = score(rag_analysis.patterns, ground_truth)

            baseline_analysis = generate_pattern_analysis_baseline(
                problem["problem_statement"], allowed_patterns=PATTERN_TAGS
            )
            baseline_score = score(baseline_analysis.patterns, ground_truth)

            elapsed = time.time() - t0
            record = {
                "problem_id": problem["problem_id"],
                "title": problem["title"],
                "ground_truth": sorted(ground_truth),
                "rag_patterns": rag_analysis.patterns,
                "rag_score": rag_score,
                "baseline_patterns": baseline_analysis.patterns,
                "baseline_score": baseline_score,
                "elapsed_s": round(elapsed, 1),
            }
            results.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(
                f"[{i}/{len(sample)}] {problem['title']} "
                f"(gt={sorted(ground_truth)}) "
                f"RAG={'hit' if rag_score['any'] else 'miss'} "
                f"baseline={'hit' if baseline_score['any'] else 'miss'} "
                f"({elapsed:.0f}s)"
            )

    n = len(results)
    rag_top1 = sum(r["rag_score"]["top1"] for r in results) / n
    rag_any = sum(r["rag_score"]["any"] for r in results) / n
    baseline_top1 = sum(r["baseline_score"]["top1"] for r in results) / n
    baseline_any = sum(r["baseline_score"]["any"] for r in results) / n

    summary = {
        "n": n,
        "seed": args.seed,
        "rag_top1_accuracy": round(rag_top1, 3),
        "rag_any_accuracy": round(rag_any, 3),
        "baseline_top1_accuracy": round(baseline_top1, 3),
        "baseline_any_accuracy": round(baseline_any, 3),
    }
    (EVAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"n = {n}")
    print(f"RAG      top1={rag_top1:.1%}  any={rag_any:.1%}")
    print(f"Baseline top1={baseline_top1:.1%}  any={baseline_any:.1%}")


if __name__ == "__main__":
    main()
