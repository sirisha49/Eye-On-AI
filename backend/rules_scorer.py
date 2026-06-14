"""
Lantern — Rules-Based Scoring Engine
Deep multi-signal hallucination detection:
  Signal 1 — NLI entailment (cross-encoder/nli-deberta-v3-small)
  Signal 2 — Numeric/entity verification (regex)
  Signal 3 — Self-consistency sampling (all-MiniLM-L6-v2 + 2 extra LLM calls)
"""
import re

COST_BASELINES = {
    "pricing": 400, "contract": 600, "tax": 500, "hiring": 450,
    "health": 400, "science": 500, "finance": 450, "legal": 550,
    "technology": 450, "career": 400, "history": 500,
    "psychology": 450, "client": 400, "general": 350,
}

CITATION_PHRASES = [
    "according to", "studies show", "research indicates",
    "experts say", "data shows", "evidence suggests",
    "reports indicate", "statistics show",
]

# ─── Lazy model loading ───────────────────────────────────────────────────────

_nli_pipeline = None
_st_model = None


def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline as _hf_pipeline
        _nli_pipeline = _hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )
    return _nli_pipeline


def _get_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        s.strip() for s in sentences
        if len(s.split()) >= 5 and '\n' not in s
    ]


def _clean_for_nli(text: str) -> str:
    """Strip Markdown and leading label-phrases so NLI sees plain text."""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)  # bold/italic
    text = re.sub(r'#{1,6}\s*', '', text)                 # headings
    text = re.sub(r'`(.+?)`', r'\1', text)               # inline code
    # Remove "Label: " prefixes like "Long-term returns: " or "Step 1: "
    text = re.sub(r'^[A-Za-z0-9][^:]{0,50}:\s+', '', text)
    return text.strip()


# ─── Signal 1: NLI Entailment ─────────────────────────────────────────────────

def _nli_entailment(answer: str, chunks: list[dict]) -> tuple[float, list[dict]]:
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.5, []
    if not chunks:
        return 0.0, [
            {"sentence": s, "entailment_score": 0.0, "contradicted": False, "supporting_chunk": None}
            for s in sentences
        ]

    try:
        nli = _get_nli()

        # Split chunks into individual sentences for fine-grained NLI
        # (paragraph-level NLI fails when a chunk contains both supporting
        #  and nuancing sentences that together downgrade entailment scores)
        chunk_sentence_pairs: list[tuple[str, str]] = []  # (chunk_id, sentence_text)
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            for cs in _split_sentences(chunk.get("content", "")):
                chunk_sentence_pairs.append((chunk_id, cs))

        if not chunk_sentence_pairs:
            return _groundedness_fallback(answer, chunks), []

        # One batched NLI call: all (chunk_sentence, answer_sentence) pairs
        n_cs = len(chunk_sentence_pairs)
        all_pairs = [
            {"text": cs_text, "text_pair": _clean_for_nli(sent)}
            for sent in sentences
            for _, cs_text in chunk_sentence_pairs
        ]

        all_results = nli(all_pairs, top_k=None)

        sentence_results = []
        for s_idx, sentence in enumerate(sentences):
            best_ent   = 0.0
            best_chunk = None
            is_contra  = False

            for cs_idx, (chunk_id, _) in enumerate(chunk_sentence_pairs):
                pair_result = all_results[s_idx * n_cs + cs_idx]
                score_map   = {r["label"].upper(): r["score"] for r in pair_result}
                ent    = score_map.get("ENTAILMENT",   0.0)
                contra = score_map.get("CONTRADICTION", 0.0)

                if ent > best_ent:
                    best_ent   = ent
                    best_chunk = chunk_id
                # Only flag contradiction when it clearly dominates entailment on the same pair
                if contra > 0.80 and contra > ent:
                    is_contra = True

            sentence_results.append({
                "sentence":          sentence,
                "entailment_score":  round(best_ent, 3),
                "contradicted":      is_contra,
                "supporting_chunk":  best_chunk,
            })

        mean_score = sum(r["entailment_score"] for r in sentence_results) / len(sentence_results)
        return round(mean_score, 3), sentence_results

    except Exception:
        score = _groundedness_fallback(answer, chunks)
        return score, []


def _groundedness_fallback(answer: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    chunk_text  = " ".join(c.get("content", "") for c in chunks).lower()
    chunk_words = set(re.findall(r"\b[a-z]{4,}\b", chunk_text))
    sentences   = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.5
    matched = 0
    for sent in sentences:
        words = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
        if not words or len(words & chunk_words) / len(words) >= 0.25:
            matched += 1
    return round(min(0.98, matched / len(sentences)), 2)


# ─── Signal 2: Numeric Verification ──────────────────────────────────────────

# Ordered most-specific → least-specific so findall doesn't re-match substrings
_NUMERIC_RE = re.compile(
    r'\$[\d,]+'                                                          # $1,000
    r'|\d+(?:\.\d+)?\s*(?:times|×)\s*(?:more|less|greater|smaller)?'   # 5.5 times more
    r'|~\s*\d+(?:\.\d+)?\s*(?:nm|cm|km|kg|hz|ghz|ms|%)?'              # ~700nm, ~5.5%
    r'|\d+(?:\.\d+)?\s*(?:nm|cm|km|kg|hz|ghz|ms|ev)\b'                # 450nm, 700nm
    r'|\d+(?:\.\d+)?%'                                                  # 5.5% or 99%
    r'|\b\d{3,}\b'                                                      # 1000, 2024
    r'|\b\d+\.\d+\b',                                                   # 5.5, 4.7
    re.IGNORECASE,
)


def _claim_in_chunk(claim: str, chunk_text: str) -> bool:
    if claim in chunk_text:
        return True
    # Strip tilde and whitespace for fuzzy match (e.g. "~450nm" vs "450nm")
    norm = re.sub(r'[~\s]', '', claim.lower())
    norm_chunk = re.sub(r'\s', '', chunk_text.lower())
    return bool(norm) and norm in norm_chunk


def _numeric_verification(answer: str, chunks: list[dict]) -> tuple[float, list[dict]]:
    raw = _NUMERIC_RE.findall(answer)
    # Deduplicate while preserving order
    seen: set[str] = set()
    figures = [f.strip() for f in raw if f.strip() and not (f.strip() in seen or seen.add(f.strip()))]

    if not figures:
        return 0.85, []
    if not chunks:
        return 0.0, [{"claim": f, "verified": False} for f in figures]

    chunk_text = " ".join(c.get("content", "") for c in chunks)
    claims = [{"claim": f, "verified": _claim_in_chunk(f, chunk_text)} for f in figures]
    verified = sum(1 for c in claims if c["verified"])
    return round(min(0.98, verified / len(figures)), 2), claims


# ─── Signal 3: Self-Consistency ───────────────────────────────────────────────

def _self_consistency(answer: str, answerer_messages: list[dict] | None) -> tuple[float, list[float]]:
    sentences = _split_sentences(answer)
    if not sentences or not answerer_messages:
        return 0.75, [0.75] * len(sentences)

    # Two extra LLM samples at higher temperature
    samples = []
    for _ in range(2):
        try:
            import backend.llm_client as _llmc
            r = _llmc.call_llm(answerer_messages, role="answerer", temperature=0.7)
            samples.append(r["content"])
        except Exception:
            pass

    if not samples:
        return 0.75, [0.75] * len(sentences)

    try:
        import numpy as np
        st = _get_st()

        orig_embeds = st.encode(sentences, show_progress_bar=False)

        per_sentence = []
        for orig_e in orig_embeds:
            sims = []
            for sample in samples:
                s_sents    = _split_sentences(sample) or [sample[:500]]
                s_embeds   = st.encode(s_sents, show_progress_bar=False)
                dots       = s_embeds @ orig_e
                norm_orig  = float(np.linalg.norm(orig_e)) + 1e-8
                norm_s     = np.linalg.norm(s_embeds, axis=1)
                cs         = dots / (norm_s * norm_orig + 1e-8)
                sims.append(float(np.max(cs)))
            per_sentence.append(round(float(np.mean(sims)), 3))

        return round(float(np.mean(per_sentence)), 3), per_sentence

    except Exception:
        return 0.75, [0.75] * len(sentences)


# ─── Main scoring function ────────────────────────────────────────────────────

def score_rules(
    question: str,
    answer: str,
    chunks: list[dict],
    token_usage: dict,
    intent: str,
    answerer_messages: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """Returns (scores_dict, sentence_analysis)."""
    groundedness_score, nli_results       = _nli_entailment(answer, chunks)
    numeric_score,      numeric_claims    = _numeric_verification(answer, chunks)
    consistency_score,  per_sent_cons     = _self_consistency(answer, answerer_messages)

    hallucination_raw = (
        0.40 * groundedness_score
        + 0.30 * numeric_score
        + 0.30 * consistency_score
    )
    # Only apply the contradiction cap when a sentence has NO competing strong support
    any_contra = any(
        r.get("contradicted", False) and r.get("entailment_score", 1.0) < 0.5
        for r in nli_results
    )
    hallucination_risk = min(0.30, hallucination_raw) if any_contra else hallucination_raw

    sentences = _split_sentences(answer)
    sentence_analysis = []
    for i, sent in enumerate(sentences):
        nli_d = nli_results[i] if i < len(nli_results) else {}
        cons  = per_sent_cons[i] if i < len(per_sent_cons) else 0.75
        ent   = nli_d.get("entailment_score", 0.5)
        contra = nli_d.get("contradicted", False)

        sent_claims = [c for c in numeric_claims if c["claim"] in sent]

        # CONTRADICTION requires both a detected contradiction AND no competing strong support
        if contra and ent < 0.5:
            verdict = "CONTRADICTION"
        elif ent >= 0.7:
            verdict = "SUPPORTED"
        elif ent >= 0.4:
            verdict = "PARTIAL"
        else:
            verdict = "UNVERIFIED"

        sentence_analysis.append({
            "sentence":         sent,
            "entailment_score": ent,
            "contradicted":     contra,
            "supporting_chunk": nli_d.get("supporting_chunk"),
            "numeric_claims":   sent_claims,
            "numeric_verified": all(c["verified"] for c in sent_claims) if sent_claims else None,
            "consistency_score": cons,
            "verdict":           verdict,
        })

    scores = {
        "groundedness":          round(min(0.98, groundedness_score), 2),
        "hallucination_risk":    round(min(0.98, hallucination_risk), 2),
        "cost_efficiency":       _cost_efficiency(token_usage, intent),
        "citation_integrity":    _citation_integrity(answer, chunks),
        "response_completeness": _response_completeness(question, answer),
    }
    return scores, sentence_analysis


# ─── Supporting scorers (unchanged) ──────────────────────────────────────────

def _cost_efficiency(token_usage: dict, intent: str) -> float:
    actual   = token_usage.get("total_tokens", 0)
    baseline = COST_BASELINES.get(intent, 350)
    overage  = max(0, actual - baseline) / baseline
    return round(max(0.0, 1.0 - overage), 2)


def _citation_integrity(answer: str, chunks: list[dict]) -> float:
    lower  = answer.lower()
    claims = sum(1 for phrase in CITATION_PHRASES if phrase in lower)
    if claims == 0:
        return 1.0
    if not chunks:
        return 0.0
    verified = min(claims, len(chunks))
    return round(min(1.0, verified / claims), 2)


def _response_completeness(question: str, answer: str) -> float:
    words   = len(answer.split())
    trivial = len(question.split()) <= 4
    if trivial and words >= 20:
        return 1.0
    if words < 20:
        return 0.2
    if 50 <= words <= 400:
        return 1.0
    if words > 600:
        return 0.7
    return round(0.2 + (words - 20) / 30 * 0.8, 2)


def compute_overall(rules: dict, evaluator: dict) -> float:
    weights = {
        "groundedness":          0.20,
        "hallucination_risk":    0.20,
        "relevance_score":       0.15,
        "safety_score":          0.15,
        "clarity_score":         0.10,
        "cost_efficiency":       0.10,
        "citation_integrity":    0.05,
        "response_completeness": 0.05,
    }
    combined = {**rules, **{k: evaluator.get(k, 0.5) for k in ("relevance_score", "safety_score", "clarity_score")}}
    total    = sum(combined.get(k, 0) * w for k, w in weights.items())
    return round(min(0.99, total), 3)
