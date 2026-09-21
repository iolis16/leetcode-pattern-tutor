"""
Phase 1: Download newfacade/LeetCodeDataset from Hugging Face and normalize
it into a flat schema we control, independent of the source dataset's fields.

Output: data/processed/problems.jsonl (one normalized problem per line)
"""
import ast
import json
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def slug_to_title(slug: str) -> str:
    return " ".join(word.capitalize() for word in slug.split("-"))


def parse_tags(raw_tags) -> list[str]:
    if isinstance(raw_tags, list):
        return raw_tags
    if isinstance(raw_tags, str):
        try:
            parsed = ast.literal_eval(raw_tags)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except (ValueError, SyntaxError):
            pass
    return []


def normalize_row(row: dict, split: str) -> dict:
    return {
        "problem_id": row["question_id"],
        "slug": row["task_id"],
        "title": slug_to_title(row["task_id"]),
        "difficulty": row["difficulty"],
        "tags": parse_tags(row["tags"]),
        "problem_statement": row["problem_description"].strip(),
        "approach_notes": row["response"].strip(),
        "estimated_date": str(row["estimated_date"]),
        "source": "newfacade/LeetCodeDataset",
        "source_split": split,
    }


def main():
    print("Downloading newfacade/LeetCodeDataset from Hugging Face...")
    ds = load_dataset("newfacade/LeetCodeDataset")

    normalized = []
    for split in ds:
        for row in ds[split]:
            normalized.append(normalize_row(row, split))

    # de-dupe on problem_id, in case of overlap between splits
    seen = {}
    for item in normalized:
        seen[item["problem_id"]] = item
    normalized = list(seen.values())
    normalized.sort(key=lambda x: int(x["problem_id"]))

    out_path = PROCESSED_DIR / "problems.jsonl"
    with out_path.open("w") as f:
        for item in normalized:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(normalized)} normalized problems to {out_path}")


if __name__ == "__main__":
    main()
