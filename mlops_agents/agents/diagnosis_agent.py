"""
agents/diagnosis_agent.py

Diagnosis Agent — LLM-powered root cause analysis enriched with:
  - Similar past incidents from the incidents RAG collection
  - Relevant runbooks and post-mortems from the runbooks collection
  - Recent metrics trend from the metrics_history collection

Returns a structured JSON diagnosis with root cause, evidence,
recommended action, and confidence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from mlops_agents.rag.store import RAGStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"retrain", "rollback", "scale", "investigate"}

SYSTEM_PROMPT = """You are a senior MLOps engineer specialising in ML model reliability.
Your job is to diagnose the root cause of a model degradation incident and recommend a remediation action.

You will be given:
1. Current model metrics
2. Similar past incidents from the institutional knowledge base
3. Relevant runbooks and post-mortems
4. Recent metrics trend

You MUST respond with ONLY a valid JSON object — no preamble, no explanation, no markdown.

JSON schema:
{
  "root_cause": "<concise one-sentence root cause>",
  "root_cause_category": "<one of: concept_drift | data_drift | model_staleness | infrastructure | data_quality | unknown>",
  "evidence": ["<evidence point 1>", "<evidence point 2>"],
  "recommended_action": "<one of: retrain | rollback | scale | investigate>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence reasoning chain>",
  "retrain_prescription": {
    "data_strategy": "<one of: recent_window | full_history | weighted_recent | drift_period_only>",
    "window_days": <integer — how many days of data to train on>,
    "drift_period_weight": <float — upsampling weight for drift period data, 1.0 = no upsampling>,
    "exclude_before": "<ISO-8601 date or empty string>",
    "refit_preprocessors": <boolean>,
    "drifted_features": ["<feature names that drifted>"],
    "optimize_for": "<one of: f2_score | roc_auc | recall | precision>",
    "target_recall": <float 0.0-1.0>,
    "target_roc_auc": <float 0.0-1.0>,
    "deployment_strategy": "<one of: canary | blue_green | immediate>",
    "canary_traffic_pct": <integer 1-50>,
    "shadow_period_hours": <integer>
  }
}

Rules for retrain_prescription:
- Only populate if recommended_action is "retrain". Set to null otherwise.
- window_days: derive from when drift started. If drift onset was recent (< 7 days), use 14. If gradual (weeks), use 30-60.
- drift_period_weight: set to 2.0 if drift_score > 0.4, 1.5 if 0.2-0.4, 1.0 otherwise.
- optimize_for: use recall for fraud/safety models, roc_auc for general classifiers.
- deployment_strategy: use canary for critical severity, immediate for minor.
- target_recall: set 0.05-0.10 above the current degraded recall value.

Root cause category rules:
- drift_score high + accuracy gradual decline → data_drift (covariate shift)
- drift_score moderate + recall collapse sudden → concept_drift (label relationship changed)
- drift_score high + recall collapse + error_rate rising → both (severe, use drift_period_only strategy)
- latency spike + accuracy stable → infrastructure (scale, not retrain)"""


# ---------------------------------------------------------------------------
# RAG context builders
# ---------------------------------------------------------------------------

def _format_similar_incidents(incidents: list[dict]) -> str:
    if not incidents:
        return "No similar past incidents found."

    lines = []
    for i, inc in enumerate(incidents, 1):
        meta = inc.get("metadata", {})
        payload = meta.get("raw_payload", {})
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}

        lines.append(
            f"Incident {i} (similarity distance: {inc.get('distance', 'N/A'):.3f}):\n"
            f"  Severity: {meta.get('severity', 'N/A')}\n"
            f"  Action taken: {meta.get('recommended_action', 'N/A')}\n"
            f"  Outcome: {meta.get('remediation_status', 'N/A')}\n"
            f"  Accuracy: {meta.get('accuracy', 'N/A')}, "
            f"Drift: {meta.get('drift_score', 'N/A')}\n"
            f"  Diagnosis: {payload.get('diagnosis', 'N/A')}"
        )
    return "\n\n".join(lines)


def _format_runbooks(runbooks: list[dict]) -> str:
    if not runbooks:
        return "No relevant runbooks found."

    lines = []
    for rb in runbooks:
        meta = rb.get("metadata", {})
        lines.append(
            f"[{meta.get('doc_type', 'doc').upper()}] {meta.get('title', 'Untitled')}\n"
            f"{rb.get('document', '')[:600]}"  # first 600 chars
        )
    return "\n\n---\n\n".join(lines)


def _format_trend(trend: list[dict]) -> str:
    if not trend:
        return "No recent trend data."

    rows = []
    for snap in trend[:8]:  # last 8 snapshots
        rows.append(
            f"  {snap.get('sampled_at', 'N/A')[:19]}  "
            f"acc={snap.get('accuracy', 'N/A')}  "
            f"drift={snap.get('drift_score', 'N/A')}  "
            f"lat={snap.get('latency_p99_ms', 'N/A')}ms  "
            f"err={snap.get('error_rate', 'N/A')}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Response parsing & normalisation
# ---------------------------------------------------------------------------

def _parse_diagnosis(raw: str) -> dict[str, Any]:
    """
    Parse the LLM output into a diagnosis dict.

    Handles:
      - Clean JSON response
      - JSON wrapped in markdown code fences
      - Extra text before/after the JSON object
      - evidence as list-of-dicts (small model quirk)
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", raw).strip()

    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {raw[:300]}")

    parsed = json.loads(match.group())

    # Normalise evidence list — small models sometimes return list-of-dicts
    evidence = parsed.get("evidence", [])
    normalised: list[str] = []
    for item in evidence:
        if isinstance(item, str):
            normalised.append(item)
        elif isinstance(item, dict):
            # e.g. {"point": "..."} or {"text": "..."}
            normalised.append(
                item.get("point") or item.get("text") or item.get("description")
                or json.dumps(item)
            )
        else:
            normalised.append(str(item))
    parsed["evidence"] = normalised

    # Validate recommended_action
    action = parsed.get("recommended_action", "investigate").lower()
    if action not in VALID_ACTIONS:
        logger.warning(
            "LLM returned invalid action '%s', defaulting to 'investigate'", action
        )
        action = "investigate"
    parsed["recommended_action"] = action

    # Clamp confidence
    confidence = float(parsed.get("confidence", 0.5))
    parsed["confidence"] = max(0.0, min(1.0, confidence))

    # Extract and validate retrain prescription
    prescription = parsed.get("retrain_prescription")
    if parsed.get("recommended_action") == "retrain" and prescription:
        # clamp numeric fields to sane ranges
        prescription["window_days"]         = max(7, min(int(prescription.get("window_days", 30)), 180))
        prescription["drift_period_weight"] = max(1.0, min(float(prescription.get("drift_period_weight", 1.0)), 5.0))
        prescription["target_recall"]       = max(0.0, min(float(prescription.get("target_recall", 0.80)), 1.0))
        prescription["target_roc_auc"]      = max(0.0, min(float(prescription.get("target_roc_auc", 0.88)), 1.0))
        prescription["canary_traffic_pct"]  = max(1, min(int(prescription.get("canary_traffic_pct", 10)), 50))
        prescription["shadow_period_hours"] = max(0, int(prescription.get("shadow_period_hours", 2)))

        if prescription.get("deployment_strategy") not in ("canary", "blue_green", "immediate"):
            prescription["deployment_strategy"] = "canary"
        if prescription.get("optimize_for") not in ("f2_score", "roc_auc", "recall", "precision"):
            prescription["optimize_for"] = "recall"
        if prescription.get("data_strategy") not in ("recent_window", "full_history", "weighted_recent", "drift_period_only"):
            prescription["data_strategy"] = "recent_window"

        parsed["retrain_prescription"] = prescription
    else:
        parsed["retrain_prescription"] = None

    return parsed


def _fallback_diagnosis(metrics: dict, severity: str) -> dict[str, Any]:
    """
    Rule-based fallback diagnosis when the LLM fails entirely.
    """
    accuracy = metrics.get("accuracy")
    drift = metrics.get("drift_score")
    latency = metrics.get("latency_p99_ms")
    error_rate = metrics.get("error_rate")

    if drift is not None and drift > 0.4:
        return {
            "root_cause": "Significant data distribution shift detected.",
            "evidence": [f"Drift score {drift:.3f} exceeds threshold 0.40."],
            "recommended_action": "retrain",
            "confidence": 0.5,
            "reasoning": "High drift score strongly suggests training data distribution mismatch.",
        }
    if accuracy is not None and accuracy < 0.70:
        return {
            "root_cause": "Model accuracy has degraded significantly.",
            "root_cause_category": "model_staleness",
            "evidence": [f"Accuracy {accuracy:.3f} is below acceptable threshold."],
            "recommended_action": "retrain",
            "confidence": 0.4,
            "reasoning": "Low accuracy with no other signals suggests model staleness.",
            "retrain_prescription": {
                "data_strategy":       "recent_window",
                "window_days":         30,
                "drift_period_weight": 1.5,
                "exclude_before":      "",
                "refit_preprocessors": True,
                "drifted_features":    [],
                "optimize_for":        "recall",
                "target_recall":       0.80,
                "target_roc_auc":      0.88,
                "deployment_strategy": "canary",
                "canary_traffic_pct":  10,
                "shadow_period_hours": 2,
            },
        }
    if latency is not None and latency > 1500:
        return {
            "root_cause": "Serving latency is critically high.",
            "evidence": [f"p99 latency {latency:.0f}ms exceeds 1500ms threshold."],
            "recommended_action": "scale",
            "confidence": 0.5,
            "reasoning": "Latency spike without accuracy degradation suggests capacity issue.",
        }
    return {
        "root_cause": "Ambiguous degradation — manual investigation required.",
        "evidence": [
            f"accuracy={accuracy}", f"drift={drift}",
            f"latency_p99={latency}ms", f"error_rate={error_rate}"
        ],
        "recommended_action": "investigate",
        "confidence": 0.3,
        "reasoning": "No single metric clearly identifies the root cause.",
    }


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def diagnosis_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Diagnosis Agent.

    1. Builds a rich context string from RAG:
         - Top similar past incidents
         - Relevant runbooks
         - Recent metrics trend
    2. Invokes the LLM with the context + current metrics.
    3. Parses the structured JSON response.
    4. Falls back to rule-based diagnosis on LLM failure.
    5. Updates state with diagnosis and recommended_action.
    """
    metrics: dict = state.get("metrics") or {}
    severity: str = state.get("severity", "minor")
    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    logger.info("Diagnosis Agent: analysing %s (%s) severity=%s", model_id, environment, severity)

    # ── 1. Build incident query text ────────────────────────────────────────
    query_text = (
        f"Model {model_id} in {environment}. Severity: {severity}. "
        f"Accuracy: {metrics.get('accuracy')}. "
        f"Drift score: {metrics.get('drift_score')}. "
        f"p99 latency: {metrics.get('latency_p99_ms')} ms. "
        f"Error rate: {metrics.get('error_rate')}."
    )

    # ── 2. RAG retrieval ────────────────────────────────────────────────────
    n_incidents = int(os.getenv("RAG_SIMILAR_INCIDENTS", "5"))
    n_runbooks = int(os.getenv("RAG_SIMILAR_RUNBOOKS", "3"))
    n_trend = int(os.getenv("RAG_TREND_SNAPSHOTS", "10"))

    similar_incidents = rag.query_similar_incidents(
        query_text=query_text,
        n_results=n_incidents,
        where={"environment": environment} if environment != "unknown" else None,
    )

    relevant_runbooks = rag.query_runbooks(
        query_text=query_text,
        n_results=n_runbooks,
    )

    trend = rag.query_recent_metrics(
        model_id=model_id,
        n_results=n_trend,
        environment=environment,
    )

    logger.info(
        "RAG context: %d similar incidents, %d runbooks, %d trend snapshots",
        len(similar_incidents), len(relevant_runbooks), len(trend),
    )

    # ── 3. Build LLM prompt ─────────────────────────────────────────────────
    prompt = f"""## Current Incident

Model: {model_id}  
Environment: {environment}  
Severity: {severity}

### Current Metrics
{json.dumps(metrics, indent=2)}

### Similar Past Incidents (from institutional memory)
{_format_similar_incidents(similar_incidents)}

### Relevant Runbooks & Post-Mortems
{_format_runbooks(relevant_runbooks)}

### Recent Metrics Trend (newest first)
{_format_trend(trend)}

Diagnose the root cause and recommend an action.
Remember: reply with ONLY the JSON object."""

    # ── 4. LLM call ─────────────────────────────────────────────────────────
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=0)

    diagnosis_json: dict
    diagnosis_text: str

    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        diagnosis_json = _parse_diagnosis(response.content)
        diagnosis_text = diagnosis_json.get("root_cause", "")
        logger.info(
            "Diagnosis: '%s' → action=%s confidence=%.2f",
            diagnosis_text,
            diagnosis_json.get("recommended_action"),
            diagnosis_json.get("confidence", 0.0),
        )
    except Exception as exc:
        logger.error("LLM diagnosis failed: %s — using rule-based fallback", exc)
        diagnosis_json = _fallback_diagnosis(metrics, severity)
        diagnosis_text = diagnosis_json["root_cause"]

    # ── 5. Update state ─────────────────────────────────────────────────────
    prescription = diagnosis_json.get("retrain_prescription")
    drifted      = prescription.get("drifted_features", []) if prescription else []

    return {
        **state,
        "diagnosis":             diagnosis_text,
        "diagnosis_json":        diagnosis_json,
        "recommended_action":    diagnosis_json["recommended_action"],
        "retrain_prescription":  prescription,
        "drifted_features":      drifted,
        "similar_incidents":     similar_incidents,
        "relevant_runbooks":     relevant_runbooks,
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=(
                    f"[Diagnosis] root_cause='{diagnosis_text}' "
                    f"action={diagnosis_json['recommended_action']} "
                    f"confidence={diagnosis_json.get('confidence', 0.0):.2f} "
                    f"prescription={'yes' if prescription else 'none'}"
                )
            )
        ],
    }
