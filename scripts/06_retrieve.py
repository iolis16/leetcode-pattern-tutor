"""
Phase 2 demo: given a new problem statement, print the top-k most similar
problems with their tags/patterns. Logic lives in lib/retrieval.py (project
root) so later phases (generation, API server) can import it directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.retrieval import retrieve_similar

if __name__ == "__main__":
    demo_problem = (
        "Given a string s, find the length of the longest substring "
        "without repeating characters."
    )
    results = retrieve_similar(demo_problem, k=5)
    print(f"Query: {demo_problem}\n")
    for r in results:
        print(f"[{r['similarity']:.3f}] {r['title']} ({r['difficulty']}) - {r['tags']}")
