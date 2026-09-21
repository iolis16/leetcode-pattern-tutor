"""
Phase 1 QA: sanity-check the normalized dataset before building on top of it.
"""
import json
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def main():
    path = PROCESSED_DIR / "problems.jsonl"
    rows = [json.loads(line) for line in path.open()]

    print(f"Total problems: {len(rows)}")

    no_tags = sum(1 for r in rows if not r["tags"])
    no_statement = sum(1 for r in rows if not r["problem_statement"])
    print(f"Missing tags: {no_tags}")
    print(f"Missing problem statement: {no_statement}")

    diff_counts = Counter(r["difficulty"] for r in rows)
    print(f"Difficulty distribution: {dict(diff_counts)}")

    tag_counts = Counter(tag for r in rows for tag in r["tags"])
    print(f"Unique tags: {len(tag_counts)}")
    print("Top 20 tags:")
    for tag, count in tag_counts.most_common(20):
        print(f"  {tag}: {count}")

    tags_per_problem = [len(r["tags"]) for r in rows]
    avg_tags = sum(tags_per_problem) / len(tags_per_problem)
    print(f"Avg tags per problem: {avg_tags:.2f}")

    lengths = [len(r["problem_statement"]) for r in rows]
    print(f"Problem statement length (chars): min={min(lengths)}, "
          f"max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
