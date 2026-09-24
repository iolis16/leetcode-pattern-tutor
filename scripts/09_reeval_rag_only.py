"""
Phase 5 follow-up: re-run RAG only (not baseline) on the same n=10 held-out
sample from data/eval/results.jsonl, using the current generation code
(dedup fix + lever #1 anti-anchoring tally + lever #2 bge-base embeddings).

Baseline is deliberately NOT re-run: it never retrieves anything, so it's
unaffected by every change made since the original eval (verified: 0/10
baseline records had duplicate-tag output, so even the dedup fix is a
no-op for baseline). Re-running it would just burn time reproducing
identical results. This keeps existing baseline_patterns/baseline_score
from the original run and only updates the RAG side, then recomputes
rag_score for a clean before/after comparison on the same problems.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.generation import generate_pattern_analysis

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


def score(predicted: list[str], ground_truth: set[str]) -> dict:
    top1 = bool(predicted) and predicted[0] in ground_truth
    any_match = any(p in ground_truth for p in predicted)
    return {"top1": top1, "any": any_match}


def main():
    old_results = [json.loads(l) for l in (EVAL_DIR / "results.jsonl").open()]
    rows = [json.loads(l) for l in (PROCESSED_DIR / "problems.jsonl").open()]
    by_id = {r["problem_id"]: r for r in rows}

    new_results = []
    out_path = EVAL_DIR / "results_v2.jsonl"
    with out_path.open("w") as out_f:
        for i, old in enumerate(old_results, 1):
            problem = by_id[old["problem_id"]]
            ground_truth = set(old["ground_truth"])
            t0 = time.time()

            rag_analysis, _ = generate_pattern_analysis(
                problem["problem_statement"], k=5, allowed_patterns=PATTERN_TAGS
            )
            rag_score = score(rag_analysis.patterns, ground_truth)
            elapsed = time.time() - t0

            record = {
                "problem_id": old["problem_id"],
                "title": old["title"],
                "ground_truth": old["ground_truth"],
                "rag_patterns_old": old["rag_patterns"],
                "rag_score_old": old["rag_score"],
                "rag_patterns": rag_analysis.patterns,
                "rag_score": rag_score,
                "baseline_patterns": old["baseline_patterns"],  # unchanged, reused
                "baseline_score": old["baseline_score"],  # unchanged, reused
                "elapsed_s": round(elapsed, 1),
            }
            new_results.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            changed = "CHANGED" if rag_score != old["rag_score"] else "same"
            print(
                f"[{i}/{len(old_results)}] {old['title']} "
                f"old_rag={'hit' if old['rag_score']['any'] else 'miss'} "
                f"new_rag={'hit' if rag_score['any'] else 'miss'} ({changed}) "
                f"({elapsed:.0f}s)"
            )

    n = len(new_results)
    rag_top1 = sum(r["rag_score"]["top1"] for r in new_results) / n
    rag_any = sum(r["rag_score"]["any"] for r in new_results) / n
    baseline_top1 = sum(r["baseline_score"]["top1"] for r in new_results) / n
    baseline_any = sum(r["baseline_score"]["any"] for r in new_results) / n

    summary = {
        "n": n,
        "seed": 42,
        "note": "RAG re-run with dedup + lever#1 (anti-anchoring tally) + lever#2 (bge-base embeddings); baseline reused unchanged from original n=10 run.",
        "rag_top1_accuracy": round(rag_top1, 3),
        "rag_any_accuracy": round(rag_any, 3),
        "baseline_top1_accuracy": round(baseline_top1, 3),
        "baseline_any_accuracy": round(baseline_any, 3),
    }
    (EVAL_DIR / "summary_v2.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary (v2: dedup + lever#1 + lever#2) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
