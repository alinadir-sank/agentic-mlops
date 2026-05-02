"""
agents/monitor_agent.py

Monitor Agent — polls real-time model metrics, classifies severity,
and saves a snapshot to the RAG metrics history collection.

Severity thresholds are controlled entirely by environment variables
so they can be tuned per deployment without code changes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from mlops_agents.rag.store import RAGStore
from tools.metrics_source import fetch_model_metrics, MetricsSourceError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity thresholds (overridable via env)
# ---------------------------------------------------------------------------

def _threshold(env_key: str, default: float) -> float:
    return float(os.getenv(env_key, str(default)))


THRESHOLDS = {
    "accuracy_critical":   lambda: _threshold("THRESHOLD_ACCURACY_CRITICAL",  0.65),
    "accuracy_major":      lambda: _threshold("THRESHOLD_ACCURACY_MAJOR",     0.72),
    "accuracy_minor":      lambda: _threshold("THRESHOLD_ACCURACY_MINOR",     0.80),
    "drift_critical":      lambda: _threshold("THRESHOLD_DRIFT_CRITICAL",     0.60),
    "drift_major":         lambda: _threshold("THRESHOLD_DRIFT_MAJOR",        0.35),
    "drift_minor":         lambda: _threshold("THRESHOLD_DRIFT_MINOR",        0.20),
    "latency_critical_ms": lambda: _threshold("THRESHOLD_LATENCY_CRITICAL_MS", 2000),
    "latency_major_ms":    lambda: _threshold("THRESHOLD_LATENCY_MAJOR_MS",   1000),
    "error_rate_critical": lambda: _threshold("THRESHOLD_ERROR_RATE_CRITICAL", 0.10),
    "error_rate_major":    lambda: _threshold("THRESHOLD_ERROR_RATE_MAJOR",    0.05),
}


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

def _rule_based_severity(metrics: dict) -> str | None:
    """
    Apply deterministic threshold rules.
    Returns a severity string or None if the case is ambiguous (grey-zone).
    """
    accuracy = metrics.get("accuracy")
    drift = metrics.get("drift_score")
    latency = metrics.get("latency_p99_ms")
    error_rate = metrics.get("error_rate")

    # Critical — any single metric breaches the critical threshold
    if (
        (accuracy is not None and accuracy < THRESHOLDS["accuracy_critical"]())
        or (drift is not None and drift > THRESHOLDS["drift_critical"]())
        or (latency is not None and latency > THRESHOLDS["latency_critical_ms"]())
        or (error_rate is not None and error_rate > THRESHOLDS["error_rate_critical"]())
    ):
        return "critical"

    # Healthy — all available metrics are within minor thresholds
    all_ok = all([
        accuracy is None or accuracy >= THRESHOLDS["accuracy_minor"](),
        drift is None or drift <= THRESHOLDS["drift_minor"](),
        latency is None or latency <= THRESHOLDS["latency_major_ms"](),
        error_rate is None or error_rate <= THRESHOLDS["error_rate_major"](),
    ])
    if all_ok:
        return "none"

    # Clear major
    if (
        (accuracy is not None and accuracy < THRESHOLDS["accuracy_major"]())
        or (drift is not None and drift > THRESHOLDS["drift_major"]())
    ):
        return "major"

    # Ambiguous — let the LLM decide
    return None


def _llm_severity(metrics: dict, trend: list[dict]) -> str:
    """
    Call the local Ollama LLM to classify the grey-zone severity.
    Falls back to "minor" if the LLM response cannot be parsed.
    """
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=0)

    trend_summary = ""
    if trend:
        recent = trend[:5]
        trend_summary = "\n".join(
            f"  - accuracy={m.get('accuracy','N/A')} drift={m.get('drift_score','N/A')} "
            f"latency={m.get('latency_p99_ms','N/A')}ms  [{m.get('sampled_at','')}]"
            for m in recent
        )

    prompt = f"""You are an ML reliability expert. Classify the severity of the following model monitoring alert.

Current metrics:
{json.dumps(metrics, indent=2)}

Recent trend (last 5 snapshots, newest first):
{trend_summary or "  No history available."}

Severity definitions:
- none    : all metrics healthy, no action needed
- minor   : slight degradation, monitor closely
- major   : significant degradation, human review required before remediation
- critical: severe degradation, immediate automated remediation required

Reply with ONLY one word: none, minor, major, or critical."""

    try:
        response = llm.invoke(
            [SystemMessage(content="You classify ML model alert severity."),
             HumanMessage(content=prompt)]
        )
        word = response.content.strip().lower().split()[0]
        if word in ("none", "minor", "major", "critical"):
            return word
        logger.warning("LLM returned unexpected severity '%s', defaulting to minor", word)
        return "minor"
    except Exception as exc:
        logger.error("LLM severity classification failed: %s — defaulting to minor", exc)
        return "minor"


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def monitor_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Monitor Agent.

    1. Fetches real-time metrics from the configured data source.
    2. Queries RAG for recent trend history.
    3. Classifies severity (rules first, LLM for grey zones).
    4. Saves a metrics snapshot to RAG.
    5. Returns the updated state.
    """
    model_id = state.get("model_id", os.getenv("DEFAULT_MODEL_ID", ""))
    environment = state.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    window_minutes = int(os.getenv("METRICS_WINDOW_MINUTES", "15"))

    if not model_id:
        raise ValueError(
            "state['model_id'] is required but not set. "
            "Pass model_id when invoking the graph."
        )

    logger.info("Monitor Agent: fetching metrics for %s (%s)", model_id, environment)

    # ── 1. Fetch real-time metrics ──────────────────────────────────────────
    try:
        metrics = fetch_model_metrics(
            model_id=model_id,
            environment=environment,
            window_minutes=window_minutes,
        )
    except MetricsSourceError as exc:
        logger.error("Metrics fetch failed: %s", exc)
        # Surface the error gracefully — treat as minor so the graph continues
        metrics = {
            "model_id": model_id,
            "model_version": "unknown",
            "environment": environment,
            "sampled_at": "",
            "fetch_error": str(exc),
        }

    logger.info(
        "Metrics: accuracy=%.3f drift=%.3f latency_p99=%.1fms error_rate=%.4f",
        metrics.get("accuracy") or 0.0,
        metrics.get("drift_score") or 0.0,
        metrics.get("latency_p99_ms") or 0.0,
        metrics.get("error_rate") or 0.0,
    )

    # ── 2. Query RAG trend window ───────────────────────────────────────────
    trend = rag.query_recent_metrics(
        model_id=model_id,
        n_results=int(os.getenv("TREND_WINDOW_SIZE", "20")),
        environment=environment,
    )

    # ── 3. Severity classification ──────────────────────────────────────────
    severity = _rule_based_severity(metrics)
    classification_method = "rule-based"

    if severity is None:
        logger.info("Grey-zone — delegating severity classification to LLM")
        severity = _llm_severity(metrics, trend)
        classification_method = "llm"

    logger.info(
        "Severity: %s (method=%s)", severity, classification_method
    )

    # ── 4. Save snapshot to RAG ─────────────────────────────────────────────
    rag.save_metrics_snapshot(metrics=metrics, severity=severity)

    # ── 5. Update state ─────────────────────────────────────────────────────
    return {
        **state,
        "metrics": metrics,
        "severity": severity,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=(
                    f"[Monitor] severity={severity} "
                    f"accuracy={metrics.get('accuracy')} "
                    f"drift={metrics.get('drift_score')} "
                    f"latency_p99={metrics.get('latency_p99_ms')}ms "
                    f"(classified by {classification_method})"
                )
            )
        ],
    }
