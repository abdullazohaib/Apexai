from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field, model_validator


class AIResponses(BaseModel):
    chatgpt: str = Field(..., min_length=1, description="ChatGPT response text")
    gemini: str = Field(..., min_length=1, description="Gemini response text")
    claude: str = Field(..., min_length=1, description="Claude response text")

    @model_validator(mode="after")
    def strip_responses(self) -> "AIResponses":
        self.chatgpt = self.chatgpt.strip()
        self.gemini = self.gemini.strip()
        self.claude = self.claude.strip()
        return self


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User's question")
    responses: AIResponses
    file_context: Optional[str] = Field(None, description="Optional text from uploaded file")


class ScoreBreakdownOut(BaseModel):
    keyword_relevance: float
    length_score: float
    readability_score: float
    cosine_similarity: float
    factual_score: float
    total: float


class CompareResponse(BaseModel):
    question: str
    scores: Dict[str, float]
    breakdowns: Dict[str, ScoreBreakdownOut]
    winner: str
    ranked: list[str]
    final_answer: str
    from_cache: bool
    log_id: int


class QueryLogOut(BaseModel):
    id: int
    question: str
    winner: Optional[str]
    score_chatgpt: Optional[float]
    score_gemini: Optional[float]
    score_claude: Optional[float]
    final_answer: Optional[str]
    from_cache: bool
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_queries: int
    cache_hits: int
    winner_distribution: Dict[str, int]
    avg_scores: Dict[str, float]
    recent_queries: list[QueryLogOut]
    cache_stats: Dict[str, int]