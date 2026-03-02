"""
Response Synthesizer.

Strategy:
  1. Start with the highest-scored (winner) response as the base.
  2. Extract unique informational keywords from runner-up responses that
     are NOT already present in the winner text.
  3. Build supplementary sentences from runner-ups containing those keywords.
  4. Append a deduplicated "Additional Insights" paragraph if meaningful
     extra content exists.
  5. Lightly clean the final text (remove duplicate sentences, normalise whitespace).
"""

from __future__ import annotations

import re
from typing import Dict, List

from app.core.scorer import ScoringResult, _tokenize
from app.core.logger import logger

# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter that handles common abbreviations."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_token_sets(sentences: List[str]) -> List[set]:
    return [_tokenize(s) for s in sentences]


def _deduplicate_sentences(sentences: List[str], threshold: float = 0.6) -> List[str]:
    """
    Remove sentences that are highly similar to an already-kept sentence
    (Jaccard similarity ≥ threshold).
    """
    kept: List[str] = []
    kept_tokens: List[set] = []

    for sent in sentences:
        tokens = _tokenize(sent)
        if not tokens:
            continue
        duplicate = False
        for kt in kept_tokens:
            union = tokens | kt
            intersection = tokens & kt
            if union and (len(intersection) / len(union)) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(sent)
            kept_tokens.append(tokens)

    return kept


def _extract_novel_sentences(
    winner_text: str,
    other_texts: List[str],
    top_n: int = 3,
) -> List[str]:
    """
    From each runner-up, pick sentences whose tokens have LOW overlap
    with the winner text – i.e. genuinely new information.
    """
    winner_tokens = _tokenize(winner_text)
    novel: List[str] = []

    for text in other_texts:
        sentences = _split_sentences(text)
        # Score each sentence by novelty (inverse Jaccard vs winner)
        scored = []
        for sent in sentences:
            stokens = _tokenize(sent)
            if len(stokens) < 4:
                continue
            union = stokens | winner_tokens
            intersection = stokens & winner_tokens
            novelty = 1.0 - (len(intersection) / len(union)) if union else 0.0
            scored.append((novelty, sent))

        scored.sort(reverse=True)
        novel.extend(s for _, s in scored[:top_n])

    return novel


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(
    responses: Dict[str, str],
    scoring_result: ScoringResult,
) -> str:
    """
    Produce a single synthesized answer from all model responses.
    """
    winner = scoring_result.winner
    winner_text = responses[winner]

    others = [
        responses[m] for m in scoring_result.ranked[1:]
        if responses.get(m, "").strip()
    ]

    # Base: sentences from the winner
    base_sentences = _split_sentences(winner_text)
    base_sentences = _deduplicate_sentences(base_sentences)

    # Novel sentences from runner-ups
    novel_sentences = _extract_novel_sentences(winner_text, others)
    novel_sentences = _deduplicate_sentences(novel_sentences)

    # Remove novels that are too similar to any base sentence
    base_tokens_list = _sentence_token_sets(base_sentences)
    filtered_novel: List[str] = []
    for sent in novel_sentences:
        stokens = _tokenize(sent)
        too_similar = False
        for bt in base_tokens_list:
            union = stokens | bt
            intersection = stokens & bt
            if union and (len(intersection) / len(union)) >= 0.55:
                too_similar = True
                break
        if not too_similar:
            filtered_novel.append(sent)

    # Compose final answer
    parts: List[str] = []

    # Winner intro tag
    model_label = winner.upper()
    parts.append(f"[Synthesized from {model_label} + cross-model insights]\n")

    # Base paragraph
    parts.append(" ".join(base_sentences))

    # Additional insights paragraph (if meaningful)
    if filtered_novel:
        insight_text = " ".join(filtered_novel[:5])  # cap at 5 extra sentences
        parts.append(f"\n\n**Additional Insights:** {insight_text}")

    final = "\n".join(parts)

    logger.info(
        f"Synthesis complete | winner={winner} | "
        f"base_sentences={len(base_sentences)} | "
        f"novel_added={len(filtered_novel)}"
    )

    return final.strip()