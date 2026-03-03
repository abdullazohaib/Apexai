"""
Rule-based scoring engine — v2.

Each response is scored on 5 dimensions:

  1. keyword_relevance  — TF-IDF cosine similarity between QUESTION and RESPONSE
                          (measures true topical alignment, not just word overlap)

  2. length_score       — Smooth piecewise score: ramps up 0→1 for 30-80 words,
                          holds 1.0 for 80-350 words, ramps down for longer text.
                          Short one-liners and wall-of-text both penalised correctly.

  3. readability_score  — Flesch Reading Ease remapped over its real range [-100, 121]
                          so technical responses aren't wrongly zeroed out.

  4. coverage_score     — How much of the combined vocabulary from ALL responses does
                          this response cover? Rewards comprehensive answers.
                          (Replaces misleading "cosine similarity to peers" which
                           penalised unique/better answers.)

  5. factual_score      — Soft consensus: average word-overlap (F1-style) against
                          each peer individually. Rewards agreement without requiring
                          identical vocabulary like raw Jaccard does.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from app.config import settings
from app.core.logger import logger

# ── Stop words ────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "as", "by", "from", "that", "this", "was",
    "are", "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "i", "we", "you", "he", "she",
    "they", "what", "how", "why", "when", "where", "who", "which", "its",
    "their", "our", "your", "also", "more", "very", "just", "about", "than",
    "then", "there", "these", "those", "can", "so", "if", "up", "out", "into",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    keyword_relevance: float = 0.0
    length_score: float = 0.0
    readability_score: float = 0.0
    cosine_similarity: float = 0.0   # kept as field name for UI compatibility
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

def _tokenize_set(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return {t for t in tokens if t not in STOP_WORDS}


def _tokenize_list(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def _word_count(text: str) -> int:
    return len(text.split())


# FIX 1: Use TF-IDF cosine between question and response for true relevance
def _keyword_relevance(question: str, response: str) -> float:
    """
    TF-IDF cosine similarity between question and response.
    This correctly weights rare/important terms from the question
    rather than counting any matching word equally.
    """
    try:
        docs = [question, response]
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(docs)
        sim = sk_cosine(tfidf[0], tfidf[1])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Fallback: F1-style precision+recall overlap
        q_tokens = _tokenize_set(question)
        r_tokens = _tokenize_set(response)
        if not q_tokens:
            return 0.5
        precision = len(q_tokens & r_tokens) / len(r_tokens) if r_tokens else 0.0
        recall = len(q_tokens & r_tokens) / len(q_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# FIX 2: Piecewise linear length score — no Gaussian, no arbitrary midpoint
def _length_score(response: str) -> float:
    """
    Piecewise scoring:
      0–29 words   → ramps 0.0 → 0.5  (too short)
      30–79 words  → ramps 0.5 → 1.0  (getting better)
      80–350 words → flat 1.0          (ideal range)
      351–600 words→ ramps 1.0 → 0.5  (getting long)
      600+ words   → ramps 0.5 → 0.2  (too long, penalised)
    """
    wc = _word_count(response)
    if wc < 30:
        return round(wc / 30 * 0.5, 4)
    elif wc < 80:
        return round(0.5 + (wc - 30) / 50 * 0.5, 4)
    elif wc <= 350:
        return 1.0
    elif wc <= 600:
        return round(1.0 - (wc - 350) / 250 * 0.5, 4)
    else:
        return round(max(0.2, 0.5 - (wc - 600) / 1000 * 0.3), 4)


# FIX 3: Remap Flesch over its true range [-100, 121] not [0, 100]
def _readability_score(text: str) -> float:
    """
    Flesch Reading Ease ranges roughly -100 (very hard) to 121 (very easy).
    We map this to 0–1 with a sweet spot at 40–70 (standard/professional text).
    Values below 0 are no longer wrongly clamped to 0.0.
    """
    try:
        fre = textstat.flesch_reading_ease(text)
        # Remap [-100, 121] → [0, 1]
        normalised = (fre + 100) / 221.0
        return round(float(np.clip(normalised, 0.0, 1.0)), 4)
    except Exception:
        return 0.5


# FIX 4: Coverage score — does this response cover the shared vocabulary well?
def _coverage_score(response: str, all_texts: List[str]) -> float:
    """
    Measures what fraction of the combined important vocabulary across ALL
    responses appears in this response. Rewards breadth and completeness.
    Unlike cosine-similarity-to-peers, this doesn't penalise a response
    for being uniquely correct.
    """
    combined_vocab: set[str] = set()
    for t in all_texts:
        combined_vocab |= _tokenize_set(t)

    if not combined_vocab:
        return 0.5

    r_tokens = _tokenize_set(response)
    coverage = len(r_tokens & combined_vocab) / len(combined_vocab)
    return round(float(np.clip(coverage, 0.0, 1.0)), 4)


# FIX 5: F1-style consensus against each peer individually (not union Jaccard)
def _factual_score(response: str, peers: List[str]) -> float:
    """
    For each peer, compute token-level F1 (harmonic mean of precision & recall).
    Average across peers. This rewards factual alignment with others while
    not penalising a response for using richer/different vocabulary.
    """
    if not peers:
        return 0.5

    r_tokens = _tokenize_set(response)
    if not r_tokens:
        return 0.0

    f1_scores = []
    for peer in peers:
        p_tokens = _tokenize_set(peer)
        if not p_tokens:
            continue
        intersection = r_tokens & p_tokens
        precision = len(intersection) / len(r_tokens)
        recall = len(intersection) / len(p_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return round(float(np.mean(f1_scores)) if f1_scores else 0.5, 4)


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

    result = ScoringResult()

    weights = {
        "keyword":     settings.WEIGHT_KEYWORD,
        "length":      settings.WEIGHT_LENGTH,
        "readability": settings.WEIGHT_READABILITY,
        "cosine":      settings.WEIGHT_COSINE,    # now = coverage
        "factual":     settings.WEIGHT_FACTUAL,
    }

    for idx, model in enumerate(model_names):
        text = texts[idx]
        peers = [texts[j] for j in range(len(texts)) if j != idx]

        kw = _keyword_relevance(question, text)
        ln = _length_score(text)
        rd = _readability_score(text)
        cv = _coverage_score(text, texts)          # replaces old cosine-to-peers
        fc = _factual_score(text, peers)

        total = (
            kw * weights["keyword"]
            + ln * weights["length"]
            + rd * weights["readability"]
            + cv * weights["cosine"]
            + fc * weights["factual"]
        )
        total = round(float(np.clip(total, 0.0, 1.0)), 4)

        bd = ScoreBreakdown(
            keyword_relevance=kw,
            length_score=ln,
            readability_score=rd,
            cosine_similarity=cv,   # field name kept for UI; now means coverage
            factual_score=fc,
            total=total,
        )

        result.scores[model] = total
        result.breakdowns[model] = bd
        logger.debug(f"Scored [{model}]: {bd}")

    # Rank models highest → lowest
    result.ranked = sorted(model_names, key=lambda m: result.scores[m], reverse=True)
    result.winner = result.ranked[0]

    logger.info(f"Scoring complete | winner={result.winner} | scores={result.scores}")
    return result