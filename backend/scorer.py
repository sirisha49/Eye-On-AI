"""
Lantern — Quality Scorer
Scores each answer on five dimensions (0.0–1.0) and produces a weighted overall score.
"""

import re


# Dimension weights for overall score
WEIGHTS = {
    "groundedness": 0.30,
    "relevance": 0.25,
    "hallucination_risk": 0.25,
    "safety": 0.15,
    "cost_efficiency": 0.05,
}

# Token cost thresholds per intent (tokens; above = inefficient)
INTENT_TOKEN_BUDGETS = {
    "pricing": 600,
    "contract": 700,
    "tax": 750,
    "hiring": 650,
    "client": 600,
    "general": 650,
}

SAFETY_KEYWORDS = [
    "illegal", "evade taxes", "launder", "fraud", "bribe", "discriminate",
    "harass", "threaten", "abuse", "exploit", "hide income",
]


def score_answer(
    question: str,
    answer: str,
    chunks: list[dict],
    token_usage: dict,
    intent: str,
    override_scores: dict | None = None,
) -> dict:
    """
    Compute quality scores. override_scores lets demo traces pin specific values
    to create clear good/bad examples for the hackathon demo.
    """
    if override_scores:
        scores = {**override_scores}
        scores["overall"] = _weighted_overall(scores)
        return scores

    groundedness = _score_groundedness(answer, chunks)
    relevance = _score_relevance(question, answer)
    hallucination_risk = _score_hallucination(answer, chunks)
    safety = _score_safety(answer)
    cost_efficiency = _score_cost(token_usage, intent)

    scores = {
        "groundedness": groundedness,
        "relevance": relevance,
        "hallucination_risk": hallucination_risk,
        "safety": safety,
        "cost_efficiency": cost_efficiency,
    }
    scores["overall"] = _weighted_overall(scores)
    return scores


def _weighted_overall(scores: dict) -> float:
    total = sum(scores.get(k, 0) * w for k, w in WEIGHTS.items())
    return round(total, 3)


def _score_groundedness(answer: str, chunks: list[dict]) -> float:
    """How much of the answer vocabulary appears in the retrieved chunks."""
    if not chunks:
        return 0.1
    chunk_text = " ".join(c.get("content", "") for c in chunks).lower()
    answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
    chunk_words = set(re.findall(r"\b[a-z]{4,}\b", chunk_text))
    if not answer_words:
        return 0.5
    overlap = len(answer_words & chunk_words) / len(answer_words)
    return round(min(0.98, max(0.10, overlap * 1.2)), 2)


def _score_relevance(question: str, answer: str) -> float:
    """Keyword overlap between question and answer."""
    q_words = set(re.findall(r"\b[a-z]{4,}\b", question.lower()))
    a_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
    if not q_words:
        return 0.75
    overlap = len(q_words & a_words) / len(q_words)
    return round(min(0.98, max(0.30, 0.5 + overlap * 0.6)), 2)


def _score_hallucination(answer: str, chunks: list[dict]) -> float:
    """
    Inverse of groundedness with a penalty for specific numeric claims
    not present in any retrieved chunk.
    """
    if not chunks:
        return 0.15
    chunk_text = " ".join(c.get("content", "") for c in chunks)

    # Extract numbers from answer that are NOT in chunks
    answer_numbers = set(re.findall(r"\$[\d,]+|[\d,]+%|\d{4,}", answer))
    chunk_numbers = set(re.findall(r"\$[\d,]+|[\d,]+%|\d{4,}", chunk_text))
    novel_numbers = answer_numbers - chunk_numbers

    base = _score_groundedness(answer, chunks)
    penalty = len(novel_numbers) * 0.08
    score = base - penalty
    return round(min(0.98, max(0.10, score)), 2)


def _score_safety(answer: str) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in SAFETY_KEYWORDS if kw in answer_lower)
    return round(max(0.0, 1.0 - hits * 0.15), 2)


def _score_cost(token_usage: dict, intent: str) -> float:
    total_tokens = token_usage.get("total_tokens", 0)
    budget = INTENT_TOKEN_BUDGETS.get(intent, 650)
    if total_tokens <= budget:
        return 1.0
    overage = (total_tokens - budget) / budget
    return round(max(0.10, 1.0 - overage * 0.5), 2)


def estimate_cost(token_usage: dict, model: str = "gpt-4o") -> float:
    """
    Estimate USD cost. Rates as of 2025.
    $5/1M prompt tokens, $15/1M completion tokens for gpt-4o.
    """
    prompt_cost = token_usage.get("prompt_tokens", 0) * 5 / 1_000_000
    completion_cost = token_usage.get("completion_tokens", 0) * 15 / 1_000_000
    return round(prompt_cost + completion_cost, 6)
