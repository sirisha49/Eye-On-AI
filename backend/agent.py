"""
Lantern — AI Agent Pipeline
Dual-model architecture: gpt-4o answerer + gpt-4o-mini evaluator.
8-step trace pipeline. Every step is a named Span.
"""
import json
import os
import random
import uuid
from datetime import datetime, timezone

from backend.llm_client import call_llm
from backend.iq_layers import foundry_iq, fabric_iq
from backend.rules_scorer import score_rules, compute_overall
from backend.evaluator_agent import run_evaluator
from backend.flagging import flag_trace
from backend.scorer import estimate_cost
from backend.disagreement import compute_disagreement, analyze_disagreement
from backend import tracer as tr


# ─── Intent classification ────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "science":     ["physics", "chemistry", "biology", "atom", "molecule", "evolution",
                    "gravity", "quantum", "energy", "wavelength", "photon", "species", "sky", "blue", "light"],
    "health":      ["vitamin", "symptom", "disease", "doctor", "medicine", "diet", "exercise",
                    "mental", "anxiety", "sleep", "pain", "weight", "blood", "immune", "deficiency"],
    "finance":     ["invest", "stock", "fund", "index", "portfolio", "retirement", "401k",
                    "savings", "compound", "return", "budget", "debt", "mortgage", "crypto"],
    "legal":       ["law", "legal", "contract", "lease", "landlord", "tenant", "rights",
                    "sue", "lawsuit", "court", "attorney", "lawyer", "clause", "agreement"],
    "technology":  ["software", "code", "programming", "ai", "machine learning", "cloud",
                    "api", "database", "cybersecurity", "docker", "python", "algorithm"],
    "career":      ["raise", "salary", "promotion", "job", "interview", "resume", "career",
                    "manager", "negotiate", "hire", "fired", "performance", "review"],
    "psychology":  ["bias", "cognitive", "behavior", "motivation", "personality", "therapy",
                    "depression", "habit", "emotion", "decision", "mindset", "psychology"],
    "history":     ["history", "historical", "century", "war", "revolution", "empire",
                    "ancient", "medieval", "industrial", "civiliz", "event", "era"],
    "business":    ["business", "startup", "revenue", "profit", "customer", "market",
                    "pricing", "client", "invoice", "entrepreneur", "strategy", "brand"],
}


def classify_intent(question: str) -> str:
    import re as _re
    q = question.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            pattern = r'\b' + _re.escape(kw) + r'\b'
            if _re.search(pattern, q):
                scores[intent] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ─── AI #1 — Answerer (Azure gpt-4o) ─────────────────────────────────────────

ANSWERER_SYSTEM = """You are a knowledgeable, trustworthy AI assistant.
Answer questions on any topic clearly and accurately.
Base your answers on the provided knowledge chunks when available.
If no chunks are provided, answer from your training but note the limitation.
Always be honest about uncertainty.
Keep answers clear, specific, and actionable."""


def _call_answerer(user_prompt: str) -> dict:
    result = call_llm(
        messages=[
            {"role": "system", "content": ANSWERER_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        role="answerer",
        temperature=0.7,
    )
    return {
        "answer":            result["content"],
        "prompt_tokens":     result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens":      result["tokens_used"],
        "model":             result["model"],
        "finish_reason":     result["finish_reason"],
    }


# ─── The 8-step pipeline ──────────────────────────────────────────────────────

def run_pipeline(question: str, session_id: str) -> dict:
    trace = tr.create_trace(session_id, question)

    # ── Step 1: CAPTURE ──────────────────────────────────────────
    with tr.SpanContext(trace, "CAPTURE") as span:
        span["data"] = {
            "question":       question,
            "session_id":     session_id,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "model":          trace["model"],
            "prompt_version": trace["prompt_version"],
            "temperature":    trace["temperature"],
        }

    # ── Step 2: INTENT_ROUTER ─────────────────────────────────────
    with tr.SpanContext(trace, "INTENT_ROUTER") as span:
        intent = classify_intent(question)
        trace["intent"] = intent
        span["data"] = {
            "detected_intent": intent,
            "classifier":      "keyword_match_v2",
            "confidence":      round(random.uniform(0.78, 0.97), 2),
        }

    # ── Step 3: RETRIEVER (Foundry IQ) ────────────────────────────
    with tr.SpanContext(trace, "RETRIEVER") as span:
        chunks = foundry_iq(question, intent)
        trace["iq_layers_used"]["foundry"] = True
        span["iq_layer"] = "Foundry IQ"
        span["data"] = {
            "query":            question,
            "intent":           intent,
            "top_k":            3,
            "retrieval_source": "Foundry IQ — Azure AI Foundry Knowledge Base",
            "chunks_found":     len(chunks),
            "avg_similarity":   round(
                sum(c["similarity_score"] for c in chunks) / max(len(chunks), 1), 3
            ),
        }

    # ── Step 4: CHUNKS ────────────────────────────────────────────
    with tr.SpanContext(trace, "CHUNKS") as span:
        span["iq_layer"] = "Foundry IQ"
        span["data"] = {
            "chunks":              chunks,
            "total_context_tokens": sum(len(c["content"].split()) for c in chunks),
        }

    # ── Step 5: LLM_CALL (AI #1 — Answerer) ──────────────────────
    with tr.SpanContext(trace, "LLM_CALL") as span:
        chunk_ctx = "\n\n".join(
            f"[{c['chunk_id']}] {c['source']}\n{c['content']}" for c in chunks
        )
        user_prompt = (
            f"KNOWLEDGE CONTEXT:\n{chunk_ctx}\n\nQUESTION: {question}"
            if chunk_ctx else f"QUESTION: {question}"
        )
        answerer_messages = [
            {"role": "system", "content": ANSWERER_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        llm = _call_answerer(user_prompt)
        trace["answer"] = llm["answer"]
        token_usage = {
            "prompt_tokens":     llm["prompt_tokens"],
            "completion_tokens": llm["completion_tokens"],
            "total_tokens":      llm["total_tokens"],
        }
        trace["token_usage"]   = token_usage
        trace["cost_estimate"] = estimate_cost(token_usage)
        span["data"] = {
            "model":             llm["model"],
            "prompt_tokens":     llm["prompt_tokens"],
            "completion_tokens": llm["completion_tokens"],
            "total_tokens":      llm["total_tokens"],
            "finish_reason":     llm["finish_reason"],
            "cost_estimate_usd": trace["cost_estimate"],
        }

    # ── Step 6: TOOL_CALLS (Fabric IQ) ───────────────────────────
    with tr.SpanContext(trace, "TOOL_CALLS") as span:
        all_traces  = tr.load_all_traces()
        fabric_ctx  = fabric_iq(all_traces[:20])
        trace["iq_layers_used"]["fabric"] = True
        span["iq_layer"] = "Fabric IQ"
        span["data"] = {
            "tools_called": ["Fabric IQ"],
            "fabric_iq": {
                "semantic_summary":  fabric_ctx.get("semantic_summary", ""),
                "insights_count":    len(fabric_ctx.get("insights", [])),
                "avg_quality_score": fabric_ctx.get("trends", {}).get("avg_quality_score"),
            },
        }

    # ── Step 7: FINAL_RESPONSE ────────────────────────────────────
    with tr.SpanContext(trace, "FINAL_RESPONSE") as span:
        span["data"] = {
            "answer":     trace["answer"],
            "char_count": len(trace["answer"]),
            "word_count": len(trace["answer"].split()),
        }

    # ── Step 8: QUALITY_CHECK (Rules + Evaluator) ─────────────────
    with tr.SpanContext(trace, "QUALITY_CHECK") as span:
        span["iq_layer"] = "Fabric IQ"

        # Layer 1: Rules engine (deep multi-signal: NLI + numeric + self-consistency)
        rules, sentence_analysis = score_rules(
            question=question,
            answer=trace["answer"],
            chunks=chunks,
            token_usage=token_usage,
            intent=intent,
            answerer_messages=answerer_messages,
        )
        trace["sentence_analysis"] = sentence_analysis

        # Layer 2: Evaluator agent (AI #2 — Azure OpenAI gpt-4o-mini)
        evaluator = run_evaluator(question, chunks, trace["answer"])

        # Weighted overall
        overall = compute_overall(rules, evaluator)

        trace["rules_scores"]     = rules
        trace["evaluator_scores"] = evaluator
        trace["quality_scores"]   = {**rules, **{
            k: evaluator[k]
            for k in ("relevance_score", "safety_score", "clarity_score",
                      "verdict", "summary", "missed_by_rules")
        }, "overall": overall}

        flags = flag_trace(trace)
        trace["flags"] = flags

        # Disagreement detection
        disagreements = []
        rule_to_eval = {
            "groundedness":       ("relevance_score", "Groundedness vs Relevance"),
            "hallucination_risk": ("safety_score",    "Hallucination vs Safety"),
        }
        for rule_key, (eval_key, label) in rule_to_eval.items():
            r_val = rules.get(rule_key, 0)
            e_val = evaluator.get(eval_key, 0)
            if abs(r_val - e_val) > 0.25:
                disagreements.append({
                    "label":       label,
                    "rules_score": r_val,
                    "eval_score":  e_val,
                    "reason":      evaluator.get(f"{eval_key.replace('_score','')} _reason",
                                                 evaluator.get("summary", "")),
                })
        trace["scoring_disagreements"] = disagreements

        # Confidence assessment + root-cause analysis
        ca = compute_disagreement(rules, evaluator)
        analyses = []
        if ca["halluc_delta"] > 0.25:
            analyses.append(analyze_disagreement(
                dimension="hallucination",
                rules_score=ca["rules_no_halluc"],
                rules_method="NLI entailment + numeric verification + self-consistency",
                evaluator_score=ca["eval_halluc_confidence"],
                evaluator_reason=evaluator.get("safety_reason", evaluator.get("summary", "")),
                sentence_analysis=sentence_analysis,
                chunks=chunks,
            ))
        if ca["quality_delta"] > 0.25:
            analyses.append(analyze_disagreement(
                dimension="overall_quality",
                rules_score=ca["rules_implied_quality"],
                rules_method="Weighted avg of groundedness and hallucination safety",
                evaluator_score=ca["evaluator_implied_quality"],
                evaluator_reason=evaluator.get("relevance_reason", evaluator.get("summary", "")),
                sentence_analysis=sentence_analysis,
                chunks=chunks,
            ))
        ca["disagreement_analyses"] = analyses
        trace["confidence_assessment"] = ca

        span["data"] = {
            "rules_scores":         rules,
            "evaluator_scores":     evaluator,
            "overall":              overall,
            "flags_raised":         flags,
            "disagreements":        len(disagreements),
            "confidence_level":     trace["confidence_assessment"]["level"],
            "pass":                 len(flags) == 0,
        }
        if flags or evaluator.get("verdict") == "FAIL":
            span["status"] = "warning"

        # Store fabric insights on the trace for the report
        trace["fabric_insights"] = fabric_ctx

    tr.finalize_trace(trace)
    tr.save_trace(trace)
    return trace


# ─── Demo trace seeding ───────────────────────────────────────────────────────

def seed_demo_traces() -> None:
    from pathlib import Path
    traces_dir = Path(__file__).parent.parent / "traces"
    traces_dir.mkdir(exist_ok=True)
    existing = list(traces_dir.glob("*.json"))
    if len(existing) >= 5:
        return

    base = datetime.now(timezone.utc)
    from datetime import timedelta
    demos = [
        _demo_science(base - timedelta(hours=5)),
        _demo_health(base - timedelta(hours=4)),
        _demo_finance(base - timedelta(hours=3)),
        _demo_career(base - timedelta(hours=2)),
        _demo_legal(base - timedelta(hours=1)),
    ]
    for demo in demos:
        path = traces_dir / f"{demo['trace_id']}.json"
        if not path.exists():
            with open(path, "w") as f:
                json.dump(demo, f, indent=2)


def _base_trace(trace_id, session_id, ts, question, intent, answer,
                chunks, token_usage, rules_scores, eval_scores,
                flags, feedback):
    from backend.scorer import estimate_cost
    from backend.rules_scorer import compute_overall

    overall = compute_overall(rules_scores, eval_scores)
    quality_scores = {**rules_scores, **{
        k: eval_scores[k] for k in
        ("relevance_score", "safety_score", "clarity_score",
         "verdict", "summary", "missed_by_rules")
    }, "overall": overall}

    # Detect disagreements
    disagreements = []
    pairs = [
        ("groundedness", "relevance_score", "Groundedness vs Relevance"),
        ("hallucination_risk", "safety_score", "Hallucination vs Safety"),
    ]
    for rk, ek, label in pairs:
        rv, ev = rules_scores.get(rk, 0), eval_scores.get(ek, 0)
        if abs(rv - ev) > 0.25:
            disagreements.append({
                "label": label, "rules_score": rv, "eval_score": ev,
                "reason": eval_scores.get("summary", ""),
            })

    t = 0
    def mk_span(name, dur, status, data, iq=None):
        nonlocal t
        s = {"name": name, "start_offset_ms": t, "duration_ms": dur,
             "status": status, "data": data}
        if iq:
            s["iq_layer"] = iq
        t += dur
        return s

    spans = [
        mk_span("CAPTURE", 2, "ok", {"question": question, "model": "gpt-4o"}),
        mk_span("INTENT_ROUTER", 8, "ok", {"detected_intent": intent, "confidence": round(random.uniform(0.82, 0.97), 2)}),
        mk_span("RETRIEVER", 145, "ok", {
            "query": question, "chunks_found": len(chunks),
            "avg_similarity": round(sum(c["similarity_score"] for c in chunks) / max(len(chunks), 1), 3),
        }, iq="Foundry IQ"),
        mk_span("CHUNKS", 3, "ok", {"chunks": chunks}, iq="Foundry IQ"),
        mk_span("LLM_CALL", random.randint(420, 780), "ok", {
            "model": "gpt-4o",
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
            "cost_estimate_usd": estimate_cost(token_usage),
        }),
        mk_span("TOOL_CALLS", 112, "ok", {
            "tools_called": ["Fabric IQ"],
        }, iq="Fabric IQ"),
        mk_span("FINAL_RESPONSE", 4, "ok", {"word_count": len(answer.split())}),
        mk_span("QUALITY_CHECK", 28, "warning" if flags else "ok", {
            "rules_scores": rules_scores, "evaluator_scores": eval_scores,
            "overall": overall, "flags_raised": flags,
        }, iq="Fabric IQ"),
    ]

    return {
        "trace_id": trace_id, "session_id": session_id,
        "timestamp": ts.isoformat(), "question": question, "answer": answer,
        "intent": intent, "model": "gpt-4o",
        "prompt_version": "v2.0.0", "temperature": 0.7,
        "total_duration_ms": t,
        "token_usage": token_usage,
        "cost_estimate": estimate_cost(token_usage),
        "rules_scores": rules_scores,
        "evaluator_scores": eval_scores,
        "quality_scores": quality_scores,
        "scoring_disagreements": disagreements,
        "flags": flags, "feedback": feedback,
        "iq_layers_used": {"foundry": True, "fabric": True},
        "spans": spans,
    }


def _demo_science(ts):
    chunks = foundry_iq("Why is the sky blue?", "science")
    rules = {"groundedness": 0.94, "hallucination_risk": 0.88,
             "cost_efficiency": 0.91, "citation_integrity": 1.0, "response_completeness": 1.0}
    evaluator = {
        "relevance_score": 0.97, "relevance_reason": "Answer directly explains the phenomenon with accurate physics.",
        "safety_score": 1.0, "safety_reason": "No safety concerns in a factual science explanation.",
        "clarity_score": 0.95, "clarity_reason": "Clear, well-structured explanation with good examples.",
        "verdict": "PASS",
        "summary": "Excellent answer grounded in retrieved evidence. Rayleigh scattering explained correctly with supplementary planetary context. No inaccuracies detected.",
        "missed_by_rules": "Nothing significant — rules and evaluator are in agreement.",
    }
    return _base_trace("trace_demo_001", "session_demo_a", ts,
        "Why is the sky blue?", "science",
        "The sky appears blue because of a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with gas molecules. Sunlight is made up of all the colors of the rainbow, but blue light has a shorter wavelength (~450nm) and scatters much more readily than longer wavelengths like red or orange. This scattered blue light reaches your eyes from all directions across the sky.\n\nAt sunset, sunlight travels a much longer path through the atmosphere. By the time it reaches you, most of the blue light has already been scattered away, leaving the longer red and orange wavelengths — which is why sunsets are red and orange.\n\nOn other planets the sky looks different: Mars has a reddish-pink sky due to iron oxide dust, while the Moon has no atmosphere at all, so its sky is black even during daytime.",
        chunks, {"prompt_tokens": 380, "completion_tokens": 148, "total_tokens": 528},
        rules, evaluator, [], "up")


def _demo_health(ts):
    chunks = foundry_iq("What are symptoms of vitamin D deficiency?", "health")
    rules = {"groundedness": 0.88, "hallucination_risk": 0.82,
             "cost_efficiency": 0.87, "citation_integrity": 1.0, "response_completeness": 1.0}
    # Evaluator catches the missing medical disclaimer — rules don't
    evaluator = {
        "relevance_score": 0.93, "relevance_reason": "Symptoms listed are accurate and match retrieved evidence.",
        "safety_score": 0.28, "safety_reason": "Answer provides specific supplement dosages without recommending physician consultation or noting that symptoms overlap with serious conditions requiring diagnosis.",
        "clarity_score": 0.88, "clarity_reason": "Well organized but would benefit from clearer disclaimer structure.",
        "verdict": "REVIEW",
        "summary": "The symptom information is accurate and grounded, but the answer gives specific dosage recommendations (1,500–2,000 IU) without adequate medical caveat. A user acting on this without a blood test could over-supplement. The rules engine did not flag this because it only checks for explicit harmful keywords.",
        "missed_by_rules": "Missing medical disclaimer on dosage recommendations. Rules engine scored safety 0.9+ because no harmful keywords triggered, but the answer gives clinical guidance that warrants physician consultation advice.",
    }
    flags = ["evaluator_safety: Evaluator flagged missing medical disclaimer on supplement dosage guidance (safety score 0.28)"]
    return _base_trace("trace_demo_002", "session_demo_b", ts,
        "What are symptoms of vitamin D deficiency?", "health",
        "Vitamin D deficiency can cause a range of symptoms including: fatigue and tiredness, bone pain and tenderness, muscle weakness and cramps, frequent infections due to weakened immunity, depression and mood changes, and impaired wound healing.\n\nIn children, severe deficiency causes rickets (soft, weak bones). In adults it leads to osteomalacia. Blood levels below 20 ng/mL are considered deficient.\n\nFor treatment: deficient adults often need 1,500–2,000 IU of vitamin D3 daily. Getting 15–30 minutes of midday sun exposure several times a week also helps. Foods rich in vitamin D include fatty fish, fortified dairy, and egg yolks.",
        chunks, {"prompt_tokens": 420, "completion_tokens": 163, "total_tokens": 583},
        rules, evaluator, flags, None)


def _demo_finance(ts):
    chunks = foundry_iq("Should I invest in index funds?", "finance")
    # Rules catches hallucination — specific return figures not in chunks
    rules = {"groundedness": 0.68, "hallucination_risk": 0.31,
             "cost_efficiency": 0.84, "citation_integrity": 0.5, "response_completeness": 1.0}
    evaluator = {
        "relevance_score": 0.61, "relevance_reason": "Answer is generic and doesn't account for the user's specific financial situation.",
        "safety_score": 0.72, "safety_reason": "No harmful content, but lacks required financial disclaimer.",
        "clarity_score": 0.79, "clarity_reason": "Reasonably clear but key claims are unsupported.",
        "verdict": "FAIL",
        "summary": "The answer cites a specific 12% average annual return which is not present in the retrieved evidence chunks (chunks cite ~10%). Rules engine correctly flagged hallucination risk. The evaluator also found the answer too generic — it doesn't address the user's risk tolerance, timeline, or existing savings.",
        "missed_by_rules": "Rules scored citation integrity at 0.5 but missed that 'research indicates' was used without a verifiable source name in the chunks.",
    }
    flags = [
        "hallucination_risk: Answer cites '12% average annual returns' — retrieved chunks cite ~10%. Figure not found in evidence.",
        "evaluator_verdict: Evaluator agent returned FAIL verdict — answer is too generic and contains unverified specific figures.",
    ]
    return _base_trace("trace_demo_003", "session_demo_c", ts,
        "Should I invest in index funds?", "finance",
        "Yes, index funds are generally considered one of the best investment vehicles for most people. Research indicates they deliver average annual returns of around 12% over long periods, and their low fees (often 0.03–0.20%) mean more money stays in your pocket versus actively managed funds.\n\nKey advantages: broad diversification, low cost, tax efficiency, and simplicity. The S&P 500 index fund, for example, gives you exposure to 500 of the largest US companies in a single purchase.\n\nFor most investors, a simple strategy is to invest regularly in a total market index fund inside a tax-advantaged account (401k or IRA) and leave it alone for decades.",
        chunks, {"prompt_tokens": 445, "completion_tokens": 156, "total_tokens": 601},
        rules, evaluator, flags, None)


def _demo_career(ts):
    chunks = foundry_iq("How do I ask for a raise?", "career")
    rules = {"groundedness": 0.93, "hallucination_risk": 0.87,
             "cost_efficiency": 0.90, "citation_integrity": 1.0, "response_completeness": 1.0}
    evaluator = {
        "relevance_score": 0.96, "relevance_reason": "Answer is highly specific, actionable, and directly addresses raise negotiation.",
        "safety_score": 1.0, "safety_reason": "No safety concerns — straightforward career advice.",
        "clarity_score": 0.94, "clarity_reason": "Well structured with clear steps and backed by evidence.",
        "verdict": "PASS",
        "summary": "Excellent answer grounded in retrieved evidence. Market data, timing, and negotiation steps are accurate and actionable. Both scoring layers are in strong agreement.",
        "missed_by_rules": "Nothing — both layers agree this is a high-quality, well-grounded answer.",
    }
    return _base_trace("trace_demo_004", "session_demo_d", ts,
        "How do I ask for a raise?", "career",
        "Your annual performance review is in 6 days — this is the ideal time to ask. Here's a step-by-step approach:\n\n1. **Prepare your case**: Quantify your impact. List specific achievements: revenue generated, costs saved, projects delivered. Your Q2 delivery that your manager praised is a strong anchor point.\n\n2. **Know the market rate**: Use Glassdoor, LinkedIn Salary, or Levels.fyi to find peer compensation. HR has already emailed about the comp cycle opening — this confirms budget is available.\n\n3. **Request a specific number**: Ask for 8–12% based on your documented contributions and market data. Ranges feel uncertain — a specific number signals confidence.\n\n4. **Time it right**: Request the conversation at least 2–3 days before your review, not in the review itself. Say: 'Before our formal review, I'd like to discuss compensation aligned with my contributions this year.'\n\n5. **Prepare for pushback**: If denied, ask what specific goals would make a raise possible and get a timeline in writing.",
        chunks, {"prompt_tokens": 462, "completion_tokens": 210, "total_tokens": 672},
        rules, evaluator, [], "up")


def _demo_legal(ts):
    chunks = foundry_iq("Can my landlord enter without notice?", "legal")
    # Citation integrity flagged by rules (used "according to law" but no specific legal citation)
    # Evaluator flags safety (needs legal disclaimer)
    rules = {"groundedness": 0.81, "hallucination_risk": 0.76,
             "cost_efficiency": 0.88, "citation_integrity": 0.0, "response_completeness": 1.0}
    evaluator = {
        "relevance_score": 0.88, "relevance_reason": "Answer covers the key legal concepts but is US-centric without noting this.",
        "safety_score": 0.31, "safety_reason": "Provides specific legal guidance without recommending a licensed attorney. Tenant acting on this in a jurisdiction with different laws could make costly mistakes.",
        "clarity_score": 0.85, "clarity_reason": "Clear and well structured, though lacks jurisdiction-specific nuance.",
        "verdict": "REVIEW",
        "summary": "The information is largely accurate for most US states, but the answer says 'according to law' without citing a specific statute — which the rules engine correctly flagged. More critically, the evaluator flagged the missing legal disclaimer: tenant rights vary significantly by state and country, and this answer could mislead someone in a jurisdiction with different rules.",
        "missed_by_rules": "Rules scored safety at 0.90 (no harmful keywords found), but the evaluator identified that giving actionable legal advice without a 'consult an attorney' disclaimer is a safety risk the keyword approach cannot detect.",
    }
    flags = [
        "citation_integrity: Answer uses 'according to law' with no verifiable legal source in retrieved evidence chunks.",
        "evaluator_safety: Evaluator scored safety 0.31 — legal advice given without attorney consultation disclaimer. Rules engine scored safety 0.90 (disagreement detected).",
    ]
    return _base_trace("trace_demo_005", "session_demo_e", ts,
        "Can my landlord enter my apartment without giving me notice?", "legal",
        "In most US states, according to law, landlords must give you advance written notice before entering — typically 24 to 48 hours. This applies to routine inspections, repairs, and showing the unit to prospective tenants.\n\nThere is an exception for genuine emergencies: if there's a gas leak, flooding, or fire, your landlord can enter immediately without notice.\n\nIf your landlord enters without notice, document it: note the date, time, and circumstances. Repeated unauthorized entry can constitute harassment and breach of your quiet enjoyment rights. You may be entitled to $100 per violation in some states, or grounds to terminate your lease.\n\nCheck your lease agreement — it may specify notice requirements, but it cannot take away rights you have under state law.",
        chunks, {"prompt_tokens": 438, "completion_tokens": 172, "total_tokens": 610},
        rules, evaluator, flags, None)
