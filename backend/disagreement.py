"""
Lantern — Disagreement Engine
Compares Layer 1 (Rules Engine) vs Layer 2 (Evaluator Agent) on two axes:
  1. Hallucination confidence: (1 - hallucination_risk) vs safety_score
  2. Overall quality:          avg(groundedness, 1-hallucination) vs avg(rel, safety, clarity)

Also provides root-cause analysis for each significant disagreement.
"""

CAUSE_LABELS = {
    "fluency_vs_grounding_gap":     "Fluency vs Grounding Gap",
    "evaluator_overcaution":        "Evaluator Overcaution",
    "scope_mismatch":               "Scope Mismatch",
    "format_vs_substance":          "Format vs Substance",
    "missing_attribution_phrases":  "Missing Attribution Phrases",
    "length_vs_substance":          "Length vs Substance",
}


def analyze_disagreement(
    dimension: str,
    rules_score: float,
    rules_method: str,
    evaluator_score: float,
    evaluator_reason: str,
    sentence_analysis: list[dict],
    chunks: list[dict],
) -> dict:
    """Root-cause explanation for why two independent methods scored a dimension differently."""
    direction = "evaluator_higher" if evaluator_score > rules_score else "rules_higher"
    analysis: dict = {
        "dimension":        dimension,
        "rules_score":      round(rules_score, 3),
        "evaluator_score":  round(evaluator_score, 3),
        "delta":            round(abs(evaluator_score - rules_score), 3),
        "direction":        direction,
        "likely_cause":     None,
        "explanation":      "",
        "evidence":         [],
    }

    if dimension in ("hallucination", "groundedness"):
        unverified = [
            s for s in sentence_analysis
            if s.get("verdict") in ("UNVERIFIED", "CONTRADICTION")
        ]
        if direction == "evaluator_higher" and unverified:
            analysis["likely_cause"] = "fluency_vs_grounding_gap"
            analysis["explanation"] = (
                f"The evaluator judges the answer as well-written and plausible "
                f"(scoring {evaluator_score:.0%}), likely because the claims sound "
                f"authoritative and internally consistent. However, the rules engine "
                f"found {len(unverified)} specific claim(s) with no supporting evidence "
                f"in the retrieved chunks (scoring {rules_score:.0%}). This is a classic "
                f"'fluent but unverified' pattern — the answer reads as correct but "
                f"contains claims that cannot be traced to source material."
            )
            analysis["evidence"] = [
                {
                    "sentence":   s["sentence"],
                    "entailment": s.get("entailment_score", 0),
                    "claims":     s.get("numeric_claims", []),
                }
                for s in unverified
            ]
        elif direction == "evaluator_higher":
            analysis["likely_cause"] = "scope_mismatch"
            analysis["explanation"] = (
                f"The evaluator's safety score ({evaluator_score:.0%}) is higher than "
                f"the rules engine's hallucination confidence ({rules_score:.0%}). "
                f"All sentences passed NLI verification — this likely reflects a gap in "
                f"chunk retrieval coverage rather than actual hallucination in the answer. "
                f"Evaluator's note: '{evaluator_reason}'"
            )
        else:
            analysis["likely_cause"] = "evaluator_overcaution"
            analysis["explanation"] = (
                f"The rules engine found strong evidence support ({rules_score:.0%}) "
                f"via direct NLI entailment matching against retrieved chunks, but the "
                f"evaluator was more conservative ({evaluator_score:.0%}). This may "
                f"indicate the evaluator is applying a stricter or more contextual "
                f"standard — e.g. technically supported but missing important caveats. "
                f"Evaluator's note: '{evaluator_reason}'"
            )

    elif dimension == "citation_integrity":
        if direction == "evaluator_higher":
            analysis["likely_cause"] = "format_vs_substance"
            analysis["explanation"] = (
                f"The rules engine checks for citation PATTERNS (phrases like "
                f"'according to X' matched against available sources) and scored "
                f"{rules_score:.0%}. The evaluator assesses whether citations are "
                f"SUBSTANTIVELY accurate and scored {evaluator_score:.0%}. "
                f"Evaluator's note: '{evaluator_reason}'"
            )
        else:
            analysis["likely_cause"] = "missing_attribution_phrases"
            analysis["explanation"] = (
                f"The rules engine found citation-style phrases without matching "
                f"sources (scoring {rules_score:.0%}), but the evaluator judged "
                f"the overall attribution as reasonable in context "
                f"(scoring {evaluator_score:.0%}). "
                f"Evaluator's note: '{evaluator_reason}'"
            )

    elif dimension == "completeness":
        analysis["likely_cause"] = "length_vs_substance"
        if direction == "evaluator_higher":
            analysis["explanation"] = (
                f"The rules engine measures completeness via word count relative to "
                f"question complexity (scoring {rules_score:.0%}). The evaluator "
                f"assesses whether the CONTENT fully addresses the question regardless "
                f"of length (scoring {evaluator_score:.0%}). "
                f"Evaluator's note: '{evaluator_reason}'"
            )
        else:
            analysis["explanation"] = (
                f"The answer meets expected length for this question type "
                f"(rules score: {rules_score:.0%}), but the evaluator identified "
                f"content gaps despite adequate length "
                f"(evaluator score: {evaluator_score:.0%}). "
                f"Evaluator's note: '{evaluator_reason}'"
            )

    elif dimension == "overall_quality":
        if direction == "evaluator_higher":
            analysis["likely_cause"] = "fluency_vs_grounding_gap"
            analysis["explanation"] = (
                f"The evaluator rates overall quality higher ({evaluator_score:.0%}) "
                f"than the rules engine ({rules_score:.0%}). The evaluator perceives "
                f"the answer as well-structured and relevant, while the rules engine's "
                f"lower score reflects gaps in evidence grounding (NLI entailment) "
                f"and/or unverified numeric claims. The answer may be fluent but "
                f"lacks full evidential support from the retrieved documents."
            )
        else:
            analysis["likely_cause"] = "evaluator_overcaution"
            analysis["explanation"] = (
                f"The rules engine rates overall quality higher ({rules_score:.0%}) "
                f"than the evaluator ({evaluator_score:.0%}). Strong evidence grounding "
                f"was found via NLI, but the evaluator may have flagged tone, caveat "
                f"quality, or framing issues that deterministic rules cannot detect. "
                f"Evaluator's note: '{evaluator_reason}'"
            )

    return analysis


def compute_disagreement(rules: dict, evaluator: dict) -> dict:
    rules_no_halluc = round(max(0.0, 1.0 - rules.get("hallucination_risk", 0)), 3)
    eval_halluc     = round(evaluator.get("safety_score", 0.5), 3)

    rules_quality   = round((rules.get("groundedness", 0) + rules_no_halluc) / 2, 3)
    eval_quality    = round(
        (evaluator.get("relevance_score", 0)
         + evaluator.get("safety_score", 0)
         + evaluator.get("clarity_score", 0)) / 3, 3
    )

    quality_delta   = round(abs(rules_quality - eval_quality), 3)
    halluc_delta    = round(abs(rules_no_halluc - eval_halluc), 3)
    significant     = quality_delta > 0.25 or halluc_delta > 0.25
    level           = "LOW" if significant else "HIGH"

    if not significant:
        interp = "Both scoring layers agree on answer quality — high signal reliability."
    elif rules_quality > eval_quality:
        interp = (
            f"Rules engine rates quality higher ({rules_quality:.0%}) than the evaluator "
            f"({eval_quality:.0%}) — the evaluator may have detected a framing, caveat, "
            f"or tone issue the deterministic rules engine cannot catch."
        )
    else:
        interp = (
            f"Evaluator rates quality higher ({eval_quality:.0%}) than the rules engine "
            f"({rules_quality:.0%}) — the answer is well-written but may lack direct "
            f"evidence grounding in the retrieved chunks."
        )

    return {
        "rules_implied_quality":     rules_quality,
        "evaluator_implied_quality": eval_quality,
        "quality_delta":             quality_delta,
        "delta":                     quality_delta,
        "rules_no_halluc":           rules_no_halluc,
        "eval_halluc_confidence":    eval_halluc,
        "halluc_delta":              halluc_delta,
        "significant_disagreement":  significant,
        "level":                     level,
        "interpretation":            interp,
        "disagreement_analyses":     [],   # populated by agent.py after sentence_analysis is ready
    }
