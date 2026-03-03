from __future__ import annotations

import io
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    CompareRequest,
    CompareResponse,
    AIResponses,
    DashboardStats,
    QueryLogOut,
    ScoreBreakdownOut,
)
from app.core.cache import cache, make_cache_key
from app.core.logger import logger
from app.core.scorer import score_responses
from app.core.synthesizer import synthesize_with_question
from app.db.models import QueryLog, get_db

router = APIRouter()


# ── /ask  (with optional file upload) ─────────────────────────────────────────

@router.post("/ask", response_model=CompareResponse, summary="Submit question + file + AI responses")
async def ask(
    question: str = Form(...),
    chatgpt: str = Form(...),
    gemini: str = Form(...),
    claude_resp: str = Form(..., alias="claude"),
    file: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
):
    file_context: str | None = None
    if file and file.filename:
        raw = await file.read()
        try:
            file_context = raw.decode("utf-8", errors="replace")[:5000]  # cap at 5 KB
        except Exception:
            file_context = None

    request = CompareRequest(
        question=question,
        responses=AIResponses(chatgpt=chatgpt, gemini=gemini, claude=claude_resp),
        file_context=file_context,
    )
    return await _process(request, db)


# ── /compare  (JSON endpoint) ─────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse, summary="JSON compare endpoint")
async def compare(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
):
    return await _process(request, db)


# ── /dashboard  (stats) ────────────────────────────────────────────────────────

@router.get("/dashboard/stats", response_model=DashboardStats, summary="Analytics dashboard data")
async def dashboard_stats(db: AsyncSession = Depends(get_db)):
    total_q = await db.scalar(select(func.count()).select_from(QueryLog))
    cache_hits = await db.scalar(
        select(func.count()).select_from(QueryLog).where(QueryLog.from_cache == 1)
    )

    # Winner distribution
    winner_rows = await db.execute(
        select(QueryLog.winner, func.count().label("cnt"))
        .group_by(QueryLog.winner)
        .where(QueryLog.winner.isnot(None))
    )
    winner_dist = {row.winner: row.cnt for row in winner_rows}

    # Avg scores
    avg_rows = await db.execute(
        select(
            func.avg(QueryLog.score_chatgpt).label("chatgpt"),
            func.avg(QueryLog.score_gemini).label("gemini"),
            func.avg(QueryLog.score_claude).label("claude"),
        )
    )
    avg = avg_rows.one()
    avg_scores = {
        "chatgpt": round(avg.chatgpt or 0, 4),
        "gemini": round(avg.gemini or 0, 4),
        "claude": round(avg.claude or 0, 4),
    }

    # Recent 10
    recent_rows = await db.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(10)
    )
    recent = [
        QueryLogOut(
            id=r.id,
            question=r.question,
            winner=r.winner,
            score_chatgpt=r.score_chatgpt,
            score_gemini=r.score_gemini,
            score_claude=r.score_claude,
            final_answer=r.final_answer,
            from_cache=bool(r.from_cache),
            created_at=r.created_at,
        )
        for r in recent_rows.scalars()
    ]

    return DashboardStats(
        total_queries=total_q or 0,
        cache_hits=cache_hits or 0,
        winner_distribution=winner_dist,
        avg_scores=avg_scores,
        recent_queries=recent,
        cache_stats=cache.stats(),
    )


@router.get("/history", response_model=List[QueryLogOut], summary="Full query history")
async def history(limit: int = 50, db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
    )
    return [
        QueryLogOut(
            id=r.id,
            question=r.question,
            winner=r.winner,
            score_chatgpt=r.score_chatgpt,
            score_gemini=r.score_gemini,
            score_claude=r.score_claude,
            final_answer=r.final_answer,
            from_cache=bool(r.from_cache),
            created_at=r.created_at,
        )
        for r in rows.scalars()
    ]


@router.delete("/cache", summary="Clear in-memory cache")
async def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}


# ── Internal processing ────────────────────────────────────────────────────────

async def _process(request: CompareRequest, db: AsyncSession) -> CompareResponse:
    responses_dict = {
        "chatgpt": request.responses.chatgpt,
        "gemini": request.responses.gemini,
        "claude": request.responses.claude,
    }

    cache_key = make_cache_key(request.question, responses_dict)
    cached = cache.get(cache_key)

    if cached:
        logger.info(f"Returning cached result for question: {request.question[:60]}")
        # Still log the cached hit
        await _save_log(db, request, cached, from_cache=True)
        return cached

    # Score
    scoring_result = score_responses(request.question, responses_dict)

    # Synthesize — pass the question for relevance-guided sentence selection
    final_answer = synthesize_with_question(request.question, responses_dict, scoring_result)

    breakdowns_out = {
        model: ScoreBreakdownOut(**bd.to_dict())
        for model, bd in scoring_result.breakdowns.items()
    }

    log_entry = await _save_log(
        db,
        request,
        None,
        scoring_result=scoring_result,
        final_answer=final_answer,
        breakdowns_out=breakdowns_out,
        from_cache=False,
    )

    response = CompareResponse(
        question=request.question,
        scores=scoring_result.scores,
        breakdowns=breakdowns_out,
        winner=scoring_result.winner,
        ranked=scoring_result.ranked,
        final_answer=final_answer,
        from_cache=False,
        log_id=log_entry.id,
    )

    cache.set(cache_key, response)
    return response


async def _save_log(
    db: AsyncSession,
    request: CompareRequest,
    cached_response: CompareResponse | None,
    scoring_result=None,
    final_answer: str = "",
    breakdowns_out: dict | None = None,
    from_cache: bool = False,
) -> QueryLog:
    if cached_response:
        entry = QueryLog(
            question=request.question,
            file_context=request.file_context,
            response_chatgpt=request.responses.chatgpt,
            response_gemini=request.responses.gemini,
            response_claude=request.responses.claude,
            score_chatgpt=cached_response.scores.get("chatgpt"),
            score_gemini=cached_response.scores.get("gemini"),
            score_claude=cached_response.scores.get("claude"),
            breakdown_chatgpt=cached_response.breakdowns.get("chatgpt").model_dump() if cached_response.breakdowns.get("chatgpt") else None,
            breakdown_gemini=cached_response.breakdowns.get("gemini").model_dump() if cached_response.breakdowns.get("gemini") else None,
            breakdown_claude=cached_response.breakdowns.get("claude").model_dump() if cached_response.breakdowns.get("claude") else None,
            winner=cached_response.winner,
            final_answer=cached_response.final_answer,
            from_cache=1,
        )
    else:
        entry = QueryLog(
            question=request.question,
            file_context=request.file_context,
            response_chatgpt=request.responses.chatgpt,
            response_gemini=request.responses.gemini,
            response_claude=request.responses.claude,
            score_chatgpt=scoring_result.scores.get("chatgpt"),
            score_gemini=scoring_result.scores.get("gemini"),
            score_claude=scoring_result.scores.get("claude"),
            breakdown_chatgpt=breakdowns_out["chatgpt"].model_dump() if breakdowns_out else None,
            breakdown_gemini=breakdowns_out["gemini"].model_dump() if breakdowns_out else None,
            breakdown_claude=breakdowns_out["claude"].model_dump() if breakdowns_out else None,
            winner=scoring_result.winner,
            final_answer=final_answer,
            from_cache=0,
        )

    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    logger.info(f"Logged query id={entry.id} from_cache={from_cache}")
    return entry