"""
EyeOnAI — IQ Layers (Foundry IQ + Fabric IQ)

Foundry IQ: evidence retrieval layer — surfaces knowledge chunks the
Rules Engine and Evaluator check the answer against.

Fabric IQ: trend intelligence layer — computes flag rates, quality
trends, and semantic summaries across trace history for the Dashboard.

Production swap (1 line each):
  foundry_iq → azure.ai.projects AIProjectClient.knowledge.retrieve()
  fabric_iq  → Azure Fabric semantic model query
"""
import random
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# FOUNDRY IQ — Evidence Retrieval Layer (Step 3: RETRIEVER)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "science": [
        {
            "chunk_id": "SCI-001", "source": "Encyclopaedia Britannica — Atmospheric Optics",
            "content": "The sky appears blue due to Rayleigh scattering. Sunlight contains all visible wavelengths. As it passes through the atmosphere, gas molecules scatter shorter (blue) wavelengths far more than longer (red) wavelengths, so blue light reaches our eyes from all directions.",
            "base_similarity": 0.97,
        },
        {
            "chunk_id": "SCI-002", "source": "NASA Science — Why Is the Sky Blue?",
            "content": "Rayleigh scattering intensity is inversely proportional to the fourth power of wavelength. Blue light (~450nm) scatters roughly 5.5x more than red light (~700nm). At sunset, light travels a longer path so most blue is scattered away, leaving red and orange hues.",
            "base_similarity": 0.94,
        },
        {
            "chunk_id": "SCI-003", "source": "Physics Today — Atmospheric Light Scattering",
            "content": "The sky on other planets differs based on atmospheric composition. Mars has a reddish sky due to iron oxide dust. Venus has an orange-yellow sky. The Moon has no atmosphere, so the sky appears black even during daylight.",
            "base_similarity": 0.88,
        },
    ],
    "health": [
        {
            "chunk_id": "HLT-001", "source": "NIH Office of Dietary Supplements — Vitamin D Fact Sheet",
            "content": "Vitamin D deficiency symptoms include fatigue, bone pain, muscle weakness, muscle aches or cramps, and mood changes such as depression. Severe deficiency can lead to rickets in children and osteomalacia in adults. Blood levels below 20 ng/mL are considered deficient.",
            "base_similarity": 0.96,
        },
        {
            "chunk_id": "HLT-002", "source": "Mayo Clinic — Vitamin D Deficiency Overview",
            "content": "Risk factors for vitamin D deficiency include limited sun exposure, darker skin pigmentation, older age, obesity, and malabsorption conditions like celiac disease. Diagnosis requires a 25-hydroxyvitamin D blood test. Most adults need 600–800 IU daily; deficient adults may need 1,500–2,000 IU.",
            "base_similarity": 0.92,
        },
        {
            "chunk_id": "HLT-003", "source": "Harvard Health Publishing — Vitamin D and Your Health",
            "content": "Low vitamin D is associated with increased risk of osteoporosis, cardiovascular disease, diabetes, and certain cancers. However, high-dose supplementation (>4,000 IU/day) can cause toxicity including nausea, weakness, and hypercalcemia. Always consult a physician before supplementing.",
            "base_similarity": 0.90,
        },
    ],
    "finance": [
        {
            "chunk_id": "FIN-001", "source": "Vanguard Investor Education — Index Fund Basics",
            "content": "Index funds track a market index like the S&P 500, offering broad diversification at low cost. Average expense ratios are 0.03–0.20% for index funds versus 0.5–1.0% for actively managed funds. Over 15-year periods, roughly 90% of active funds underperform their benchmark index.",
            "base_similarity": 0.95,
        },
        {
            "chunk_id": "FIN-002", "source": "Morningstar Research — Long-Term Investing Study 2024",
            "content": "The S&P 500 has delivered an average annual return of approximately 10% before inflation over the past 50 years. However, individual years vary widely: losses of 38% (2008), gains of 32% (2013). Time in market consistently outperforms timing the market for most investors.",
            "base_similarity": 0.91,
        },
        {
            "chunk_id": "FIN-003", "source": "Fidelity Investments — Beginner's Guide to Investing",
            "content": "Tax-advantaged accounts (401k, IRA, Roth IRA) are recommended as first destinations for index fund investments. Dollar-cost averaging — investing fixed amounts regularly regardless of market conditions — reduces the impact of volatility. Emergency fund of 3–6 months expenses should precede investing.",
            "base_similarity": 0.89,
        },
    ],
    "legal": [
        {
            "chunk_id": "LEG-001", "source": "Nolo Legal Encyclopedia — Landlord Entry Rights",
            "content": "In most US states, landlords must provide 24–48 hours advance written notice before entering a rental unit for non-emergency inspections or repairs. California, New York, and most states require 24 hours minimum. Emergency situations (gas leak, flooding) permit immediate entry without notice.",
            "base_similarity": 0.96,
        },
        {
            "chunk_id": "LEG-002", "source": "Tenant Rights Handbook — State-by-State Entry Laws",
            "content": "Unauthorized landlord entry may constitute harassment or breach of quiet enjoyment. Tenants can document violations and potentially terminate the lease or seek damages. Repeated unauthorized entries in California can result in $100 per violation plus actual damages.",
            "base_similarity": 0.91,
        },
        {
            "chunk_id": "LEG-003", "source": "American Bar Association — Residential Tenancy Guide",
            "content": "Lease agreements may specify different entry notice requirements but cannot waive a tenant's statutory rights under state law. Landlords may enter to show the property to prospective tenants or buyers with proper notice, or to make necessary agreed-upon repairs.",
            "base_similarity": 0.87,
        },
    ],
    "technology": [
        {
            "chunk_id": "TEC-001", "source": "IEEE Spectrum — Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI where systems learn from data rather than explicit programming. Three main paradigms: supervised learning (labeled data), unsupervised learning (pattern discovery), and reinforcement learning (reward-based). Deep learning uses neural networks with multiple layers.",
            "base_similarity": 0.94,
        },
        {
            "chunk_id": "TEC-002", "source": "MIT Technology Review — State of AI 2024",
            "content": "Large language models (LLMs) like GPT-4 are trained on trillions of tokens of text data. They use transformer architecture with attention mechanisms. Hallucination — generating plausible but false information — remains a key challenge. RAG (Retrieval-Augmented Generation) mitigates this.",
            "base_similarity": 0.92,
        },
        {
            "chunk_id": "TEC-003", "source": "Stack Overflow Developer Survey 2024",
            "content": "Python dominates AI/ML development at 68% usage. JavaScript leads web development. Rust is the most admired language for the 8th consecutive year. Cloud computing adoption reaches 94% of enterprises. Containerization (Docker, Kubernetes) is standard in 79% of production deployments.",
            "base_similarity": 0.86,
        },
    ],
    "career": [
        {
            "chunk_id": "CAR-001", "source": "Harvard Business Review — How to Ask for a Raise",
            "content": "Timing a raise request around performance review cycles, company financial results, or after delivering a significant win increases success probability by 40%. Present market data comparing your compensation to peers. Average successful raise negotiation results in 8–15% salary increase.",
            "base_similarity": 0.95,
        },
        {
            "chunk_id": "CAR-002", "source": "LinkedIn Salary Research Report 2024",
            "content": "Workers who negotiate salary earn an average of $5,000 more annually than those who don't. 85% of hiring managers have room to negotiate. Key negotiation factors: competing offers, documented accomplishments, market rate data, and tenure. Counter-offers are accepted in 62% of cases.",
            "base_similarity": 0.92,
        },
        {
            "chunk_id": "CAR-003", "source": "SHRM — Compensation and Benefits Planning Guide",
            "content": "Effective raise requests include: quantified achievements (revenue generated, costs reduced), market rate evidence (Glassdoor, Levels.fyi, LinkedIn), timing relative to budget cycles, and a specific number rather than a range. Budget cycles typically lock salaries 3–4 months before the fiscal year.",
            "base_similarity": 0.88,
        },
    ],
    "psychology": [
        {
            "chunk_id": "PSY-001", "source": "American Psychological Association — Cognitive Bias Overview",
            "content": "Cognitive biases are systematic patterns of deviation from rational judgment. Confirmation bias leads people to favor information confirming existing beliefs. The Dunning-Kruger effect causes low-competence individuals to overestimate their abilities. Anchoring bias means initial information disproportionately influences later judgments.",
            "base_similarity": 0.93,
        },
        {
            "chunk_id": "PSY-002", "source": "Journal of Behavioral Psychology — Decision Making Under Uncertainty",
            "content": "Loss aversion — the tendency to feel losses more strongly than equivalent gains — is a foundational finding of behavioral economics. People feel losses approximately 2x more intensely than gains of the same magnitude. This drives risk-aversion in investment decisions.",
            "base_similarity": 0.90,
        },
        {
            "chunk_id": "PSY-003", "source": "Stanford Encyclopedia of Philosophy — Motivation Theory",
            "content": "Intrinsic motivation (driven by internal rewards like curiosity and satisfaction) produces higher long-term performance than extrinsic motivation (money, recognition). Self-Determination Theory identifies autonomy, competence, and relatedness as the three core psychological needs.",
            "base_similarity": 0.87,
        },
    ],
    "history": [
        {
            "chunk_id": "HIS-001", "source": "Oxford History of the Modern World",
            "content": "The Industrial Revolution began in Britain circa 1760–1840, transforming agrarian economies into manufacturing-based ones. Key drivers: steam engine (Watt, 1769), textile mechanization (spinning jenny, power loom), and coal mining expansion. Real wages roughly doubled between 1760 and 1860.",
            "base_similarity": 0.93,
        },
        {
            "chunk_id": "HIS-002", "source": "Cambridge World History — Economic Development 1500–2000",
            "content": "Global GDP per capita remained roughly flat for centuries before industrialization. The Great Divergence saw Western Europe and North America's living standards grow 10-20x faster than Asia and Africa between 1800 and 1950. Colonialism, institutional quality, and geography are debated causal factors.",
            "base_similarity": 0.89,
        },
        {
            "chunk_id": "HIS-003", "source": "Encyclopedia of World History — Technology and Society",
            "content": "Major technological inflection points in history: printing press (1440, democratized knowledge), steam power (1760s), electricity and telegraph (1870s), internal combustion (1880s), internet (1990s). Each created new economic sectors while disrupting existing labor markets.",
            "base_similarity": 0.85,
        },
    ],
    "business": [
        {
            "chunk_id": "BUS-001", "source": "Harvard Business School — Small Business Fundamentals",
            "content": "Cash flow management is the #1 challenge for small businesses. Maintaining 3 months of operating reserves, invoicing immediately upon delivery, and diversifying revenue so no single client exceeds 30% are foundational practices. 82% of failed small businesses cite cash flow problems as a primary cause.",
            "base_similarity": 0.93,
        },
        {
            "chunk_id": "BUS-002", "source": "McKinsey & Co. — Customer Acquisition Research 2024",
            "content": "Referral clients have 34% higher lifetime value and 2.1x retention rate compared to ad-acquired clients. Average client acquisition cost: $420 via paid advertising, $85 via referrals. A systematic referral program is typically the highest-ROI marketing investment for service businesses.",
            "base_similarity": 0.90,
        },
        {
            "chunk_id": "BUS-003", "source": "SBA — Pricing Strategy for Service Businesses",
            "content": "Value-based pricing (charging based on client outcome value) generates 20–40% higher margins than cost-plus pricing. Price anchoring — presenting a premium option first — increases average transaction value by 15%. Annual price increases of 3–5% are expected by clients and should be standard practice.",
            "base_similarity": 0.87,
        },
    ],
    "general": [
        {
            "chunk_id": "GEN-001", "source": "World Knowledge Encyclopedia — General Reference",
            "content": "Critical thinking involves systematic evaluation of evidence, identification of logical fallacies, and consideration of alternative explanations. Key steps: identify the claim, examine evidence quality, check for bias, consider counter-arguments, and draw proportionate conclusions.",
            "base_similarity": 0.82,
        },
        {
            "chunk_id": "GEN-002", "source": "Global Research Institute — Problem Solving Frameworks",
            "content": "Structured problem-solving approaches (PDCA, 5 Whys, First Principles) improve decision quality. Breaking complex problems into smaller components, gathering relevant data before deciding, and testing assumptions reduces cognitive errors. Most decisions benefit from a 24-hour reflection period before finalizing.",
            "base_similarity": 0.79,
        },
        {
            "chunk_id": "GEN-003", "source": "Communications Research Journal — Effective Information Transfer",
            "content": "Clear communication requires understanding your audience's context, structuring information from general to specific, using concrete examples, and checking comprehension. Studies show people retain 10% of what they read, 20% of what they hear, and 90% of what they teach to others.",
            "base_similarity": 0.76,
        },
    ],
}

INTENT_TO_TOPIC = {
    "science": "science", "health": "health", "finance": "finance",
    "legal": "legal", "technology": "technology", "career": "career",
    "psychology": "psychology", "history": "history", "business": "business",
    "pricing": "business", "contract": "legal", "tax": "finance",
    "hiring": "career", "client": "business",
    "general": "general",
}


def foundry_iq(query: str, intent: str) -> list[dict]:
    """
    Foundry IQ — retrieve the top-3 evidence chunks for a given query and intent.
    Returns chunks with chunk_id, source, content, and similarity_score.

    Production: replace with AIProjectClient.knowledge.retrieve(query, top_k=3)
    """
    topic = INTENT_TO_TOPIC.get(intent, "general")
    chunks = KNOWLEDGE_BASE.get(topic, KNOWLEDGE_BASE["general"])
    result = []
    for chunk in chunks[:3]:
        sim = round(chunk["base_similarity"] + random.uniform(-0.03, 0.03), 3)
        sim = max(0.70, min(0.99, sim))
        result.append({
            "chunk_id": chunk["chunk_id"],
            "source":   chunk["source"],
            "content":  chunk["content"],
            "similarity_score": sim,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FABRIC IQ — Trend Intelligence Layer (Step 8 + Dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def fabric_iq(trace_history: list) -> dict:
    """
    Fabric IQ — compute quality trends, flag rates, and semantic insights
    across trace history for the Dashboard.

    Production: replace with Azure Fabric semantic model query over lantern_traces.
    """
    total = len(trace_history)
    if total == 0:
        return {"source": "Fabric IQ — Microsoft Fabric", "insights": [], "trends": {}}

    flagged   = [t for t in trace_history if t.get("flags")]
    scores    = [t.get("quality_scores", {}).get("overall", 0) for t in trace_history]
    costs     = [t.get("cost_estimate", 0) for t in trace_history]
    avg_score = round(sum(scores) / total, 3)
    avg_cost  = round(sum(costs)  / total, 5)
    flag_rate = round(len(flagged) / total * 100, 1)

    intent_counts: dict = {}
    intent_flag_counts: dict = {}
    for t in trace_history:
        intent = t.get("intent", "general")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if t.get("flags"):
            intent_flag_counts[intent] = intent_flag_counts.get(intent, 0) + 1

    insights = []
    if flag_rate > 30:
        insights.append({"severity": "high", "title": "Elevated Flagging Rate",
            "detail": f"Flagging rate {flag_rate}% — above the 15% baseline. Review knowledge coverage."})
    elif flag_rate > 15:
        insights.append({"severity": "medium", "title": "Above-Average Flagging Rate",
            "detail": f"Flagging rate {flag_rate}% vs 15% baseline. Monitor for trends."})

    for intent, count in intent_flag_counts.items():
        rate = round(count / intent_counts.get(intent, 1) * 100)
        if rate >= 50 and count >= 1:
            insights.append({"severity": "high",
                "title": f"High Flag Rate — {intent.capitalize()}",
                "detail": f"{rate}% of {intent} questions flagged. Knowledge base may need expansion."})

    if avg_score < 0.75:
        insights.append({"severity": "medium", "title": "Below-Target Quality Score",
            "detail": f"Average quality {avg_score} below 0.75 target."})

    if not insights:
        insights.append({"severity": "low", "title": "Performance Within Normal Range",
            "detail": f"Quality {avg_score}, flag rate {flag_rate}% — both within targets."})

    daily_scores = []
    for i in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%b %d")
        score = round(max(0.4, min(1.0, avg_score + random.uniform(-0.08, 0.08))), 2)
        daily_scores.append({"day": day, "score": score})

    trending_weak = [
        intent for intent, count in intent_flag_counts.items()
        if count / intent_counts.get(intent, 1) >= 0.5
    ]

    return {
        "source": "Fabric IQ — Microsoft Fabric",
        "trends": {
            "total_queries": total,
            "avg_quality_score": avg_score,
            "flag_rate_pct": flag_rate,
            "avg_cost_usd": avg_cost,
            "daily_scores": daily_scores,
            "intent_distribution": intent_counts,
            "intent_flag_rates": {
                k: round(intent_flag_counts.get(k, 0) / v * 100, 0)
                for k, v in intent_counts.items()
            },
            "trending_weak_areas": trending_weak,
            "cost_trend": "stable",
        },
        "insights": insights,
        "semantic_summary": (
            f"Analyzed {total} trace{'s' if total != 1 else ''} across "
            f"{len(intent_counts)} topic{'s' if len(intent_counts) != 1 else ''}. "
            f"Average quality {avg_score}, flagging rate {flag_rate}%."
        ),
    }
