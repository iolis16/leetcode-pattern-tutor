"""
Phase 4: FastAPI backend wrapping the retrieval + generation pipeline, and
serving the static frontend from the same process (single origin, no CORS
needed, single deployable unit for Phase 6).
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import ollama
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.schemas import AnalyzeRequest, AnalyzeResponse, RetrievedProblem
from lib.generation import generate_pattern_analysis
from lib.retrieval import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once at startup instead of on first request.
    get_model()
    yield


app = FastAPI(title="LeetCode Pattern Tutor", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        analysis, retrieved = generate_pattern_analysis(req.problem_statement, k=req.k)
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable. Is scripts/db.sh running?")
    except (ollama.ResponseError, ConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Local LLM unavailable: {e}. Is `ollama serve` running?")

    return AnalyzeResponse(
        patterns=analysis.patterns,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
        cited_problem_titles=analysis.cited_problem_titles,
        retrieved=[
            RetrievedProblem(
                problem_id=r["problem_id"],
                slug=r["slug"],
                title=r["title"],
                difficulty=r["difficulty"],
                tags=r["tags"],
                similarity=r["similarity"],
            )
            for r in retrieved
        ],
    )


app.mount("/", StaticFiles(directory=ROOT_DIR / "static", html=True), name="static")
