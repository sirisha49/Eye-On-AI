"""
EyeOnAI — Flask Application Entry Point
Run with: python app.py
"""

import os
import uuid

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

load_dotenv()

from backend.agent import run_pipeline, seed_demo_traces
from backend.tracer import load_all_traces, load_trace, save_feedback
from backend.iq_layers import fabric_iq

app = Flask(__name__, static_folder="frontend", static_url_path="")

# Seed demo data on startup
seed_demo_traces()


# ─── Frontend ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ─── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    session_id = data.get("session_id") or str(uuid.uuid4())
    trace = run_pipeline(question, session_id)
    return jsonify(trace)


@app.route("/api/traces", methods=["GET"])
def get_traces():
    traces = load_all_traces()
    # Strip verbose spans for the list view
    summary = []
    for t in traces:
        summary.append({
            "trace_id": t["trace_id"],
            "timestamp": t["timestamp"],
            "question": t["question"],
            "answer": t["answer"],
            "intent": t["intent"],
            "model": t["model"],
            "total_duration_ms": t["total_duration_ms"],
            "token_usage": t["token_usage"],
            "cost_estimate": t["cost_estimate"],
            "quality_scores": t["quality_scores"],
            "flags": t["flags"],
            "feedback": t["feedback"],
            "iq_layers_used": t.get("iq_layers_used", {}),
        })
    return jsonify(summary)


@app.route("/api/traces/<trace_id>", methods=["GET"])
def get_trace(trace_id):
    trace = load_trace(trace_id)
    if not trace:
        return jsonify({"error": "Trace not found"}), 404
    return jsonify(trace)


@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    traces = load_all_traces()
    flagged = [t for t in traces if t.get("flags")]
    total = len(traces)
    scores = [t.get("quality_scores", {}).get("overall", 0) for t in traces]
    costs = [t.get("cost_estimate", 0) for t in traces]

    stats = {
        "total": total,
        "flagged_count": len(flagged),
        "flagged_pct": round(len(flagged) / max(total, 1) * 100, 1),
        "avg_score": round(sum(scores) / max(len(scores), 1), 3),
        "avg_cost_usd": round(sum(costs) / max(len(costs), 1), 6),
    }

    flagged_summary = []
    for t in flagged:
        flagged_summary.append({
            "trace_id": t["trace_id"],
            "timestamp": t["timestamp"],
            "question": t["question"],
            "intent": t["intent"],
            "quality_scores": t["quality_scores"],
            "flags": t["flags"],
            "model": t["model"],
            "cost_estimate": t["cost_estimate"],
            "total_duration_ms": t["total_duration_ms"],
            "feedback": t["feedback"],
            "iq_layers_used": t.get("iq_layers_used", {}),
        })

    return jsonify({"stats": stats, "flagged": flagged_summary})


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    trace_id = data.get("trace_id")
    sentiment = data.get("sentiment")
    if not trace_id or sentiment not in ("up", "down"):
        return jsonify({"error": "Invalid request"}), 400
    ok = save_feedback(trace_id, sentiment)
    return jsonify({"ok": ok})


@app.route("/api/fabric-insights", methods=["GET"])
def fabric_insights():
    traces = load_all_traces()
    insights = fabric_iq(traces)
    return jsonify(insights)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("ENVIRONMENT", "demo") == "demo"
    print(f"\n👁  EyeOnAI is running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
