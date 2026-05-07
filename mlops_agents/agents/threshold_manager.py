import json
import logging
import os
from datetime import datetime, timezone
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from mlops_agents.rag.store import RAGStore

logger = logging.getLogger(__name__)

THRESHOLD_LIMITS = {
    "accuracy_major": (0.6, 0.85),
    "accuracy_critical": (0.5, 0.75),
    "drift_major": (0.1, 0.7),
    "drift_critical": (0.3, 0.9),
    "latency_major_ms": (500, 3000),
    "latency_critical_ms": (1000, 5000),
    "error_rate_major": (0.01, 0.2),
    "error_rate_critical": (0.05, 0.3),
}

MAX_THRESHOLD_DELTA = 0.02

def _clamp_thresholds(thresholds: dict) -> dict:
    for key, (low, high) in THRESHOLD_LIMITS.items():
        if key in thresholds:
            thresholds[key] = max(low, min(high, thresholds[key]))
    return thresholds

def _compute_drift_trend(trend: list[dict]) -> str:
    if len(trend) < 3: return "flat"
    vals = [m.get("drift_score", 0.0) for m in trend[:5] if m.get("drift_score") is not None]
    if len(vals) < 3: return "flat"
    if vals[0] > vals[-1] * 1.1: return "increasing"
    if vals[0] < vals[-1] * 0.9: return "decreasing"
    return "flat"

def _llm_threshold_advisor(state: AgentState, thresholds: dict, historical_stats: dict, trend: list[dict]) -> dict:
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    llm = ChatOllama(model=model_name, temperature=0)
    
    prompt = f"""
    Recommend SMALL monitoring threshold adjustments (+/-0.02 max). 
    Current: {json.dumps(thresholds)}
    Incident: {state.get('diagnosis')}
    History: {json.dumps(historical_stats)}
    Return ONLY JSON with keys: should_update, confidence, reasoning, adjustments.
    """
    try:
        response = llm.invoke([
            SystemMessage(content="You are an adaptive threshold tuning agent."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if "```" in raw: raw = raw.split("```")[1].replace("json", "", 1).strip()
        return json.loads(raw)
    except Exception as exc:
        logger.error(f"LLM Advisor failed: {exc}")
        return {"should_update": False, "adjustments": {}}

def run_threshold_update(state: AgentState, rag: RAGStore) -> None:
    """Main entry point for the threshold learning subsystem."""
    metrics = state.get("metrics") or {}
    model_id, env = metrics.get("model_id", "unknown"), metrics.get("environment", "production")

    # Cooldown Check
    existing = rag.get_dynamic_thresholds(model_id=model_id)
    now = datetime.now(timezone.utc)
    if existing and (now - datetime.fromisoformat(existing["updated_at"])).total_seconds() < 1800:
        return

    # Data Gathering
    thresholds = existing["thresholds"].copy() if existing else (state.get("thresholds") or {})
    if not thresholds: return
    
    hist_stats = rag.get_incident_stats(model_id=model_id, environment=env)
    trend = rag.query_recent_metrics(model_id=model_id, environment=env)
    
    # Logic: Learning Rate & Advisor
    lr = min(0.1, 1 / max(hist_stats.get("total", 1), 1))
    proposal = _llm_threshold_advisor(state, thresholds, hist_stats, trend)

    if not proposal.get("should_update"): return

    # Apply Adjustments
    conf = max(0.0, min(1.0, float(proposal.get("confidence", 0.0))))
    applied = {}
    for key, delta in proposal.get("adjustments", {}).items():
        if key in thresholds:
            eff_delta = float(delta) * conf * lr
            thresholds[key] += max(-MAX_THRESHOLD_DELTA, min(MAX_THRESHOLD_DELTA, eff_delta))
            applied[key] = eff_delta

    # Trend awareness override
    if _compute_drift_trend(trend) == "increasing":
        thresholds["drift_major"] -= 0.01 * lr

    # Persist
    rag.save_dynamic_thresholds(model_id=model_id, thresholds={
        "model_id": model_id, "environment": env, "updated_at": now.isoformat(),
        "thresholds": _clamp_thresholds(thresholds), "applied_adjustments": applied
    })