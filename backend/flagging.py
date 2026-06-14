"""
Lantern — Flagging Engine
Rules-based engine that inspects a completed trace and returns flag strings.
"""
import re

HALLUCINATION_THRESHOLD = 0.50
SAFETY_THRESHOLD        = 0.50
CITATION_THRESHOLD      = 0.50
TOKEN_COST_MULTIPLIER   = 2.0

INTENT_AVG_TOKENS = {
    "science": 500, "health": 480, "finance": 520, "legal": 540,
    "technology": 480, "career": 460, "psychology": 470, "history": 520,
    "business": 500, "pricing": 480, "contract": 520, "tax": 550,
    "hiring": 490, "client": 470, "general": 500,
}


def flag_trace(trace: dict) -> list[str]:
    flags = []
    scores    = trace.get("rules_scores") or trace.get("quality_scores", {})
    evaluator = trace.get("evaluator_scores", {})
    answer    = trace.get("answer", "")
    intent    = trace.get("intent", "general")
    spans     = {s["name"]: s for s in trace.get("spans", [])}

    # ── Rule 1: Hallucination risk ─────────────────────────────────
    hr = scores.get("hallucination_risk", 1.0)
    if hr < HALLUCINATION_THRESHOLD:
        flags.append(
            f"hallucination_risk: Answer may cite facts not found in retrieved evidence "
            f"(rules score {hr:.2f}, threshold {HALLUCINATION_THRESHOLD})"
        )

    # ── Rule 2: Citation integrity ─────────────────────────────────
    ci = scores.get("citation_integrity", 1.0)
    if ci < CITATION_THRESHOLD:
        flags.append(
            "citation_integrity: Answer uses citation language ('according to', 'studies show', etc.) "
            "but no verifiable source was found in retrieved evidence chunks."
        )

    # ── Rule 3: No chunks but answer makes specific claims ─────────
    chunks_span = spans.get("CHUNKS", {})
    chunks = chunks_span.get("data", {}).get("chunks", [])
    if not chunks and _has_specific_claims(answer):
        flags.append(
            "no_evidence_with_claims: Answer makes specific factual claims but no "
            "knowledge chunks were retrieved to ground them."
        )

    # ── Rule 4: Excessive token cost ───────────────────────────────
    total_tokens = trace.get("token_usage", {}).get("total_tokens", 0)
    avg = INTENT_AVG_TOKENS.get(intent, 500)
    if total_tokens > avg * TOKEN_COST_MULTIPLIER:
        flags.append(
            f"excessive_token_cost: {total_tokens} tokens is "
            f"{round(total_tokens/avg, 1)}x the {intent} average of {avg}."
        )

    # ── Rule 5: Evaluator verdict FAIL ─────────────────────────────
    verdict = evaluator.get("verdict", "")
    if verdict == "FAIL":
        flags.append(
            f"evaluator_verdict: Evaluator agent returned FAIL verdict — "
            f"{evaluator.get('summary', 'see evaluator report')[:120]}"
        )

    # ── Rule 6: Evaluator safety below threshold ───────────────────
    eval_safety = evaluator.get("safety_score", 1.0)
    if eval_safety < SAFETY_THRESHOLD:
        flags.append(
            f"evaluator_safety: Evaluator scored safety {eval_safety:.2f} — "
            f"{evaluator.get('safety_reason', 'potential safety concern')}"
        )

    # ── Rule 7: User thumbs-down ────────────────────────────────────
    if trace.get("feedback") == "down":
        flags.append("user_negative_feedback: User rated this answer as unhelpful.")

    return flags


def _has_specific_claims(answer: str) -> bool:
    return bool(re.search(r"\$[\d,]+|[\d.]+%|\b\d{3,}\b", answer))
