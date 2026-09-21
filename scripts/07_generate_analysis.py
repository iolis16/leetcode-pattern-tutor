"""
Phase 3 demo: given a new problem statement, retrieve similar problems and
ask Claude to identify the likely pattern(s), citing evidence, with no code.

Requires ANTHROPIC_API_KEY in .env.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.generation import generate_pattern_analysis

if __name__ == "__main__":
    demo_problem = (
        "You are climbing a staircase. It takes n steps to reach the top. "
        "Each time you can either climb 1 or 2 steps. In how many distinct "
        "ways can you climb to the top?"
    )
    analysis, retrieved = generate_pattern_analysis(demo_problem)

    print(f"Problem: {demo_problem}\n")
    print("Retrieved context:")
    for r in retrieved:
        print(f"  - {r['title']} ({r['difficulty']}) - {r['tags']}")
    print()
    print(f"Patterns: {analysis.patterns}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Cited: {analysis.cited_problem_titles}")
    print(f"Reasoning:\n{analysis.reasoning}")
