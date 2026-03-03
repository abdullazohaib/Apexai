"""
Response Synthesizer — v2.

Problems fixed from v1:
  - v1 picked "novel" sentences (high divergence from winner) → pulled in
    off-topic content from weaker responses.
  - v1 injected raw markdown headers into plain-text output.
  - v1 could produce a worse answer than just taking the winner alone.

New strategy:
  1. Start with the winner response as the primary base (it scored highest).
  2. From runner-ups, extract sentences that are RELEVANT to the question
     AND contribute genuinely new information (not already covered by winner).
     "Relevant" = high keyword overlap with question.
     "New"      = moderate-to-low overlap with winner (not a duplicate).
  3. Score and rank those candidate sentences; take the best N.
  4. Append only if they meaningfully add to the answer.
  5. Clean output: no raw markdown, no ugly headers — plain readable prose.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from app.core.scorer import ScoringResult, _tokenize_set
from app.core.logger import logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries; keep sentences with ≥5 words."""
    # Handle newlines first
    text = re.sub(r"\n+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def _tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform([a, b])
        return float(sk_cosine(tfidf[0], tfidf[1])[0][0])
    except Exception:
        return 0.0


def _token_overlap(a: str, b: str) -> float:
    """F1-style token overlap between two strings."""
    ta = _tokenize_set(a)
    tb = _tokenize_set(b)
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    precision = len(intersection) / len(ta)
    recall = len(intersection) / len(tb)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _deduplicate_sentences(sentences: List[str], threshold: float = 0.55) -> List[str]:
    """Keep sentences that are not near-duplicates of already-kept ones."""
    kept: List[str] = []
    for sent in sentences:
        is_dup = any(_token_overlap(sent, k) >= threshold for k in kept)
        if not is_dup:
            kept.append(sent)
    return kept


def _score_candidate(
    sentence: str,
    question: str,
    winner_text: str,
) -> float:
    """
    Score a candidate sentence from a runner-up on two axes:
      - relevance:  how much does it address the question?  (want HIGH)
      - novelty:    how different is it from the winner?    (want MODERATE)

    Combined score = relevance * novelty_factor
    novelty_factor peaks at 0.5 overlap (adds info) and drops near 0 (duplicate)
    or near 1.0 (totally unrelated — not useful context).
    """
    relevance = _tfidf_cosine(question, sentence)
    overlap_with_winner = _token_overlap(sentence, winner_text)

    # Penalise near-duplicates (overlap > 0.7) and totally unrelated (< 0.05)
    if overlap_with_winner > 0.70:
        novelty_factor = 0.1   # nearly a duplicate — skip
    elif overlap_with_winner < 0.05:
        novelty_factor = 0.3   # completely unrelated — probably off-topic
    else:
        # Peak at ~0.30 overlap (complementary), smooth falloff
        novelty_factor = 1.0 - abs(overlap_with_winner - 0.30) / 0.70

    return relevance * novelty_factor


def _clean_text(text: str) -> str:
    """Remove markdown artifacts and normalise whitespace."""
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)   # **bold** / *italic*
    text = re.sub(r"#{1,6}\s+", "", text)                   # ## headers
    text = re.sub(r"`+([^`]*)`+", r"\1", text)              # `code`
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [link](url)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(
    responses: Dict[str, str],
    scoring_result: ScoringResult,
) -> str:
    """
    Produce a single clean synthesized answer from all model responses.

    The winner's text is the backbone. Genuinely additive sentences from
    runner-ups are appended only when they pass relevance + novelty thresholds.
    """
    winner = scoring_result.winner
    winner_text = _clean_text(responses[winner])
    runner_ups = [
        (m, _clean_text(responses[m]))
        for m in scoring_result.ranked[1:]
        if responses.get(m, "").strip()
    ]

    # ── Step 1: Build deduplicated base from winner ──────────────────────────
    base_sentences = _split_sentences(winner_text)
    base_sentences = _deduplicate_sentences(base_sentences, threshold=0.6)

    # ── Step 2: Score candidate sentences from runner-ups ───────────────────
    candidates: List[Tuple[float, str]] = []
    question = scoring_result.winner  # We don't have question here — use winner text
    # NOTE: we embed the question via the scoring_result's context
    # (question is passed separately from routes; synthesize receives it below)
    for _model, text in runner_ups:
        for sent in _split_sentences(text):
            score = _score_candidate(sent, winner_text, winner_text)
            candidates.append((score, sent))

    # Sort by score descending, take top 6
    candidates.sort(reverse=True)
    top_candidates = [s for _, s in candidates[:6] if _ > 0.15]

    # Deduplicate candidates against base AND each other
    novel_sentences: List[str] = []
    for sent in top_candidates:
        too_similar_to_base = any(_token_overlap(sent, b) >= 0.55 for b in base_sentences)
        too_similar_to_kept = any(_token_overlap(sent, k) >= 0.55 for k in novel_sentences)
        if not too_similar_to_base and not too_similar_to_kept:
            novel_sentences.append(sent)

    novel_sentences = novel_sentences[:3]  # cap at 3 additions

    # ── Step 3: Compose clean readable answer ───────────────────────────────
    final_parts: List[str] = [" ".join(base_sentences)]

    if novel_sentences:
        final_parts.append(" ".join(novel_sentences))

    final = " ".join(final_parts)

    # Final cleanup pass
    final = re.sub(r"\s{2,}", " ", final).strip()
    # Ensure ends with punctuation
    if final and final[-1] not in ".!?":
        final += "."

    logger.info(
        f"Synthesis complete | winner={winner} | "
        f"base_sentences={len(base_sentences)} | novel_added={len(novel_sentences)}"
    )
    return final


# ── Enhanced version that receives the question ───────────────────────────────

def synthesize_with_question(
    question: str,
    responses: Dict[str, str],
    scoring_result: ScoringResult,
) -> str:
    """
    Full synthesis using the original question to guide relevance scoring.
    This is the preferred entry point — routes.py should call this.
    """
    winner = scoring_result.winner
    winner_text = _clean_text(responses[winner])
    runner_ups = [
        (m, _clean_text(responses[m]))
        for m in scoring_result.ranked[1:]
        if responses.get(m, "").strip()
    ]

    # Deduplicated base from winner
    base_sentences = _split_sentences(winner_text)
    base_sentences = _deduplicate_sentences(base_sentences, threshold=0.6)

    # Score candidates using actual question for relevance
    candidates: List[Tuple[float, str]] = []
    for _model, text in runner_ups:
        for sent in _split_sentences(text):
            score = _score_candidate(sent, question, winner_text)
            if score > 0.12:
                candidates.append((score, sent))

    candidates.sort(reverse=True)

    # Deduplicate and select top novel sentences
    novel_sentences: List[str] = []
    for _, sent in candidates[:8]:
        too_similar_to_base = any(_token_overlap(sent, b) >= 0.55 for b in base_sentences)
        too_similar_to_kept = any(_token_overlap(sent, k) >= 0.55 for k in novel_sentences)
        if not too_similar_to_base and not too_similar_to_kept:
            novel_sentences.append(sent)
        if len(novel_sentences) >= 3:
            break

    # Compose
    parts = [" ".join(base_sentences)]
    if novel_sentences:
        parts.append(" ".join(novel_sentences))

    final = " ".join(parts)
    final = re.sub(r"\s{2,}", " ", final).strip()
    if final and final[-1] not in ".!?":
        final += "."

    logger.info(
        f"Synthesis (with question) complete | winner={winner} | "
        f"base={len(base_sentences)} | novel_added={len(novel_sentences)}"
    )
    return final