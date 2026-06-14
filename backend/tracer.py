"""
Lantern — Trace builder and persistence layer.
Each user request becomes a Trace containing ordered Spans.
Traces are stored as JSON files under ./traces/.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACES_DIR = Path(__file__).parent.parent / "traces"


class SpanContext:
    """Context manager that records start/end time for a span."""

    def __init__(self, trace: dict, name: str):
        self.trace = trace
        self.name = name
        self.span: dict = {}

    def __enter__(self):
        self.span = {
            "name": self.name,
            "start_offset_ms": int((time.perf_counter() - self.trace["_t0"]) * 1000),
            "duration_ms": 0,
            "status": "ok",
            "data": {},
        }
        self.trace["spans"].append(self.span)
        self._t_start = time.perf_counter()
        return self.span

    def __exit__(self, exc_type, *_):
        self.span["duration_ms"] = int((time.perf_counter() - self._t_start) * 1000)
        if exc_type:
            self.span["status"] = "error"
            self.span["error"] = str(exc_type)


def create_trace(session_id: str, question: str) -> dict:
    trace = {
        "trace_id": f"tr_{uuid.uuid4().hex[:12]}",
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": "",
        "intent": "general",
        "model": "gpt-4o",
        "prompt_version": "v1.2.0",
        "temperature": 0.7,
        "total_duration_ms": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost_estimate": 0.0,
        "quality_scores": {},
        "flags": [],
        "feedback": None,
        "iq_layers_used": {"foundry": False, "fabric": False},
        "spans": [],
        "_t0": time.perf_counter(),
    }
    return trace


def finalize_trace(trace: dict) -> dict:
    """Strip internal timing marker and record total duration."""
    t0 = trace.pop("_t0", None)
    if t0 is not None:
        trace["total_duration_ms"] = int((time.perf_counter() - t0) * 1000)
    return trace


def save_trace(trace: dict) -> None:
    TRACES_DIR.mkdir(exist_ok=True)
    path = TRACES_DIR / f"{trace['trace_id']}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)


def load_trace(trace_id: str) -> dict | None:
    path = TRACES_DIR / f"{trace_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_traces() -> list[dict]:
    if not TRACES_DIR.exists():
        return []
    traces = []
    for path in sorted(TRACES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                traces.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return traces


def save_feedback(trace_id: str, sentiment: str) -> bool:
    """Update feedback field and re-flag if thumbs-down."""
    trace = load_trace(trace_id)
    if not trace:
        return False
    trace["feedback"] = sentiment
    if sentiment == "down":
        flag = "user_negative_feedback: User rated this answer as unhelpful"
        if flag not in trace.get("flags", []):
            trace.setdefault("flags", []).append(flag)
    elif sentiment == "up":
        # Remove thumbs-down flag if present
        trace["flags"] = [
            f for f in trace.get("flags", []) if "user_negative_feedback" not in f
        ]
    save_trace(trace)
    return True
