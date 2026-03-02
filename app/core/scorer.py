"""
Rule-based scoring engine.

Each response is scored on 5 dimensions:
  1. keyword_relevance  – TF overlap between question and response
  2. length_score       – Gaussian penalty for too-short / too-long responses
  3. readability_score  – Normalised Flesch Reading Ease (0-100 → 0-1)
  4. cosine_similarity  – Average cosine sim vs the other two responses
  5. factual_score      – Jaccard similarity against combined peer text
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.core.logger import logger

# ── Stop words (lightweight, no NLTK dependency for this list) ───────────────
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "as", "by", "from", "that", "this", "was",
    "are", "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "i", "we", "you", "he", "she",
    "they", "what", "how", "why", "when", "where", "who", "which",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    keyword_relevance: float = 0.0
    length_score: float = 0.0
    readability_score: float = 0.0
    cosine_similarity: float = 0.0
    factual_score: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class ScoringResult:
    scores: Dict[str, float] = field(default_factory=dict)
    breakdowns: Dict[str, ScoreBreakdown] = field(default_factory=dict)
    winner: str = ""
    ranked: List[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return {t for t in tokens if t not in STOP_WORDS}


def _word_count(text: str) -> int:
    return len(text.split())


def _keyword_relevance(question: str, response: str) -> float:
    q_tokens = _tokenize(question)
    r_tokens = _tokenize(response)
    if not q_tokens:
        return 0.5
    overlap = q_tokens & r_tokens
    return len(overlap) / len(q_tokens)


def _length_score(response: str) -> float:
    """Gaussian peak at ideal midpoint, decays outside [IDEAL_MIN, IDEAL_MAX]."""
    wc = _word_count(response)
    low, high = settings.IDEAL_MIN_WORDS, settings.IDEAL_MAX_WORDS
    mid = (low + high) / 2
    sigma = (high - low) / 4  # 2-sigma covers the ideal range
    if sigma == 0:
        return 1.0
    score = math.exp(-0.5 * ((wc - mid) / sigma) ** 2)
    return max(0.0, min(1.0, score))


def _readability_score(text: str) -> float:
    """Flesch Reading Ease: 0–100 → normalise to 0–1; higher = better."""
    try:
        fre = textstat.flesch_reading_ease(text)
        # Clamp to 0–100 then normalise
        return max(0.0, min(1.0, fre / 100.0))
    except Exception:
        return 0.5


def _cosine_scores(texts: List[str]) -> List[float]:
    """
    For each text, compute its average cosine similarity to all OTHER texts.
    Returns scores in the same order as `texts`.
    """
    if len(texts) < 2:
        return [1.0] * len(texts)
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        n = len(texts)
        scores = []
        for i in range(n):
            others = [sim_matrix[i, j] for j in range(n) if j != i]
            scores.append(float(np.mean(others)))
        return scores
    except Exception as exc:
        logger.warning(f"Cosine similarity failed: {exc}")
        return [0.5] * len(texts)


def _factual_score(response: str, peers: List[str]) -> float:
    """Jaccard similarity between response tokens and the union of peer tokens."""
    r_tokens = _tokenize(response)
    peer_tokens: set[str] = set()
    for p in peers:
        peer_tokens |= _tokenize(p)
    if not peer_tokens:
        return 0.5
    intersection = r_tokens & peer_tokens
    union = r_tokens | peer_tokens
    return len(intersection) / len(union) if union else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def score_responses(
    question: str,
    responses: Dict[str, str],
) -> ScoringResult:
    """
    Score a dict of {model_name: response_text} against the user question.
    Returns a ScoringResult with per-model breakdowns and an overall winner.
    """
    model_names = list(responses.keys())
    texts = [responses[m] for m in model_names]

    # Pre-compute cosine scores (needs all texts simultaneously)
    cosine_vals = _cosine_scores(texts)

    result = ScoringResult()

    weights = {
        "keyword": settings.WEIGHT_KEYWORD,
        "length": settings.WEIGHT_LENGTH,
        "readability": settings.WEIGHT_READABILITY,
        "cosine": settings.WEIGHT_COSINE,
        "factual": settings.WEIGHT_FACTUAL,
    }

    for idx, model in enumerate(model_names):
        text = texts[idx]
        peers = [texts[j] for j in range(len(texts)) if j != idx]

        kw = _keyword_relevance(question, text)
        ln = _length_score(text)
        rd = _readability_score(text)
        cs = cosine_vals[idx]
        fc = _factual_score(text, peers)

        total = (
            kw * weights["keyword"]
            + ln * weights["length"]
            + rd * weights["readability"]
            + cs * weights["cosine"]
            + fc * weights["factual"]
        )

        bd = ScoreBreakdown(
            keyword_relevance=round(kw, 4),
            length_score=round(ln, 4),
            readability_score=round(rd, 4),
            cosine_similarity=round(cs, 4),
            factual_score=round(fc, 4),
            total=round(total, 4),
        )

        result.scores[model] = round(total, 4)
        result.breakdowns[model] = bd

        logger.debug(f"Scored [{model}]: {bd}")

    # Rank models highest → lowest
    result.ranked = sorted(model_names, key=lambda m: result.scores[m], reverse=True)
    result.winner = result.ranked[0]

    logger.info(
        f"Scoring complete | winner={result.winner} | scores={result.scores}"
    )
    return result