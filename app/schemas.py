from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    problem_statement: str = Field(min_length=20, max_length=8000)
    k: int = Field(default=5, ge=1, le=10)


class RetrievedProblem(BaseModel):
    problem_id: str
    slug: str
    title: str
    difficulty: str
    tags: list[str]
    similarity: float


class AnalyzeResponse(BaseModel):
    patterns: list[str]
    confidence: Literal["low", "medium", "high"]
    reasoning: str
    cited_problem_titles: list[str]
    retrieved: list[RetrievedProblem]
