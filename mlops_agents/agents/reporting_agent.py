"""
agents/reporting_agent.py

Reporting Agent — generates a structured markdown incident report,
saves the full incident to the RAG incidents collection, dispatches
Slack/email notifications, and performs adaptive threshold learning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from mlops_agents.rag.store import RAGStore
from tools.mcp_tools import send_slack_notification, send_email_alert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold learning config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(state: AgentState, historical_stats: dict) -> str:
    """Generate a structured markdown incident report."""

    metrics: dict = state.get("metrics") or {}
    diag_json: dict = state.get("diagnosis_json") or {}
    similar = state.get("similar_incidents") or []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_id = metrics.get("model_id", "unknown")
    environment = metrics.get("environment", "production")
    severity = state.get("severity", "none")

    report_lines = [
        f"# Incident Report — {model_id}",
        "",
        f"**Date:** {now}  ",
        f"**Model:** `{model_id}` v{metrics.get('model_version', 'unknown')}  ",
        f"**Environment:** {environment}  ",
        f"**Severity:** {severity.upper()}  ",
        f"**Incident ID:** `{state.get('incident_id', 'pending')}`",
        "",
        "---",
        "",
        "## Metrics at Detection",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {metrics.get('accuracy', 'N/A')} |",
        f"| Drift Score | {metrics.get('drift_score', 'N/A')} |",
        f"| p99 Latency | {metrics.get('latency_p99_ms', 'N/A')} ms |",
        f"| p95 Latency | {metrics.get('latency_p95_ms', 'N/A')} ms |",
        f"| Error Rate | {metrics.get('error_rate', 'N/A')} |",
        f"| Predictions (window) | {metrics.get('prediction_count', 'N/A')} |",
        "",
        "---",
        "",
        "## Root Cause Analysis",
        "",
        f"**Root Cause:** {state.get('diagnosis', 'N/A')}  ",
        f"**Confidence:** {diag_json.get('confidence', 'N/A')}  ",
        "",
        "**Evidence:**",
    ]

    for ev in diag_json.get("evidence", []):
        report_lines.append(f"- {ev}")

    report_lines += [
        "",
        f"**Reasoning:** {diag_json.get('reasoning', 'N/A')}",
        "",
        "---",
        "",
        "## Remediation",
        "",
        f"**Action:** `{state.get('remediation_action', 'N/A')}`  ",
        f"**Status:** {state.get('remediation_status', 'N/A')}  ",
        f"**Detail:** {state.get('remediation_detail', 'N/A')}  ",
        f"**Human Approved:** {state.get('human_approved', False)}",
        "",
        "---",
        "",
        "## Historical Context",
        "",
        f"**Total past incidents:** {historical_stats.get('total', 0)}  ",
    ]

    sev_dist = historical_stats.get("severity_distribution", {})
    if sev_dist:
        report_lines.append(f"**Severity distribution:** {sev_dist}  ")

    act_dist = historical_stats.get("action_distribution", {})
    if act_dist:
        report_lines.append(f"**Most common actions:** {act_dist}  ")

    success_rate = historical_stats.get("remediation_success_rate")
    if success_rate is not None:
        report_lines.append(
            f"**Historical remediation success rate:** {success_rate:.1%}"
        )

    if similar:
        report_lines += [
            "",
            "### Most Similar Past Incidents",
        ]

        for i, inc in enumerate(similar[:3], 1):
            meta = inc.get("metadata", {})

            report_lines.append(
                f"{i}. **{meta.get('severity', 'N/A').upper()}** — "
                f"{meta.get('recommended_action', 'N/A')} → "
                f"{meta.get('remediation_status', 'N/A')} "
                f"(similarity: {inc.get('distance', 0):.3f})"
            )

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _llm_executive_summary(state: AgentState, report: str) -> str:
    """
    Ask the LLM to write a concise executive summary for Slack.
    """

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You write concise executive summaries of MLOps incidents."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Write a concise 3-sentence summary:\n\n{report[:1500]}"
                    )
                ),
            ]
        )

        return response.content.strip()

    except Exception as exc:
        logger.warning("LLM summary failed: %s — using template", exc)

        metrics: dict = state.get("metrics") or {}

        return (
            f"Model `{metrics.get('model_id', 'unknown')}` "
            f"({state.get('severity', '?').upper()}) "
            f"in {metrics.get('environment', 'production')} degraded. "
            f"Root cause: {state.get('diagnosis', 'unknown')}. "
            f"Remediation status: {state.get('remediation_status', 'N/A')}."
        )


# ---------------------------------------------------------------------------
# Threshold learning helpers
# ---------------------------------------------------------------------------

def _compute_drift_trend(trend: list[dict]) -> str:
    """
    Simple drift trend detector.
    """

    if len(trend) < 3:
        return "flat"

    recent = trend[:5]

    vals = [
        m.get("drift_score", 0.0)
        for m in recent
        if m.get("drift_score") is not None
    ]

    if len(vals) < 3:
        return "flat"

    if vals[0] > vals[-1] * 1.1:
        return "increasing"

    if vals[0] < vals[-1] * 0.9:
        return "decreasing"

    return "flat"


def _clamp_thresholds(thresholds: dict) -> dict:
    """
    Clamp thresholds to safe ranges.
    """

    for key, (low, high) in THRESHOLD_LIMITS.items():
        if key in thresholds:
            thresholds[key] = max(low, min(high, thresholds[key]))

    return thresholds


def _llm_threshold_advisor(
    state: AgentState,
    thresholds: dict,
    historical_stats: dict,
    trend: list[dict],
) -> dict:
    """
    Ask the LLM to propose SMALL threshold adjustments.
    """

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    )

    metrics = state.get("metrics") or {}

    prompt = f"""
You are an adaptive ML reliability agent.

Your task:
Recommend SMALL monitoring threshold adjustments.

Goals:
- reduce false positives
- detect degradation earlier
- avoid unnecessary retraining

Rules:
- only propose small deltas
- max delta per threshold is +/-0.02
- return ONLY valid JSON

Current thresholds:
{json.dumps(thresholds, indent=2)}

Current incident:
{json.dumps({
        "severity": state.get("severity"),
        "metrics": metrics,
        "diagnosis": state.get("diagnosis"),
        "remediation_status": state.get("remediation_status"),
        "human_approved": state.get("human_approved"),
    }, indent=2)}

Historical stats:
{json.dumps(historical_stats, indent=2)}

Recent trend:
{json.dumps(trend[:5], indent=2)}

Return JSON:
{{
  "should_update": true,
  "confidence": 0.0,
  "reasoning": "...",
  "adjustments": {{
      "accuracy_major": -0.01,
      "drift_major": 0.01
  }}
}}
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an adaptive threshold tuning agent."
                ),
                HumanMessage(content=prompt),
            ]
        )

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw.replace("json", "", 1).strip()

        return json.loads(raw)

    except Exception as exc:
        logger.error("Threshold advisor failed: %s", exc)

        return {
            "should_update": False,
            "confidence": 0.0,
            "reasoning": str(exc),
            "adjustments": {},
        }


def _apply_threshold_adjustments(
    thresholds: dict,
    proposal: dict,
    learning_rate: float,
) -> tuple[dict, dict]:
    """
    Safely apply threshold deltas.
    """

    updated = thresholds.copy()

    confidence = float(proposal.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    adjustments = proposal.get("adjustments", {}) or {}

    applied = {}

    for key, delta in adjustments.items():

        if key not in updated:
            continue

        try:
            delta = float(delta)
        except Exception:
            continue

        delta = max(
            -MAX_THRESHOLD_DELTA,
            min(MAX_THRESHOLD_DELTA, delta),
        )

        effective_delta = delta * confidence * learning_rate

        updated[key] += effective_delta

        applied[key] = {
            "requested_delta": delta,
            "effective_delta": effective_delta,
        }

    updated = _clamp_thresholds(updated)

    return updated, applied


# ---------------------------------------------------------------------------
# Threshold learning
# ---------------------------------------------------------------------------

def threshold_learning(state: AgentState, rag: RAGStore) -> None:
    """
    Adaptive threshold learning agent.
    """

    metrics: dict = state.get("metrics") or {}

    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    existing = rag.get_dynamic_thresholds(model_id=model_id)

    now = datetime.now(timezone.utc)

    if existing and "updated_at" in existing:
        last_update = datetime.fromisoformat(existing["updated_at"])

        if (now - last_update).total_seconds() < 1800:
            logger.info("Threshold learning skipped (cooldown active)")
            return

    # ------------------------------------------------------------------
    # Base thresholds
    # ------------------------------------------------------------------

    if existing:
        thresholds = existing["thresholds"].copy()
        version = existing.get("version", 1) + 1
    else:
        thresholds = state.get("thresholds") or {}
        version = 1

    if not thresholds:
        logger.warning("No thresholds available — skipping learning")
        return

    # ------------------------------------------------------------------
    # Historical context
    # ------------------------------------------------------------------

    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    trend = rag.query_recent_metrics(
        model_id=model_id,
        environment=environment,
    )

    drift_trend = _compute_drift_trend(trend)

    # ------------------------------------------------------------------
    # Learning rate
    # ------------------------------------------------------------------

    num_incidents = max(historical_stats.get("total", 1), 1)

    learning_rate = min(0.1, 1 / num_incidents)

    logger.info(
        "Threshold learning rate: %.4f",
        learning_rate,
    )

    # ------------------------------------------------------------------
    # LLM advisor
    # ------------------------------------------------------------------

    proposal = _llm_threshold_advisor(
        state=state,
        thresholds=thresholds,
        historical_stats=historical_stats,
        trend=trend,
    )

    logger.info(
        "Threshold advisor proposal: %s",
        proposal,
    )

    if not proposal.get("should_update", False):
        logger.info("Advisor recommended no update")
        return

    # ------------------------------------------------------------------
    # Apply validated adjustments
    # ------------------------------------------------------------------

    thresholds, applied_adjustments = _apply_threshold_adjustments(
        thresholds=thresholds,
        proposal=proposal,
        learning_rate=learning_rate,
    )

    # ------------------------------------------------------------------
    # Trend awareness
    # ------------------------------------------------------------------

    if drift_trend == "increasing":
        logger.info(
            "Increasing drift trend detected — tightening drift threshold"
        )

        thresholds["drift_major"] -= 0.01 * learning_rate

    thresholds = _clamp_thresholds(thresholds)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    payload = {
        "model_id": model_id,
        "environment": environment,
        "version": version,
        "updated_at": now.isoformat(),
        "thresholds": thresholds,
        "advisor_reasoning": proposal.get("reasoning"),
        "advisor_confidence": proposal.get("confidence"),
        "applied_adjustments": applied_adjustments,
    }

    rag.save_dynamic_thresholds(
        model_id=model_id,
        thresholds=payload,
    )

    logger.info(
        "Updated dynamic thresholds v%s: %s",
        version,
        thresholds,
    )


# ---------------------------------------------------------------------------
# Reporting Agent
# ---------------------------------------------------------------------------

def reporting_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Reporting Agent.
    """

    metrics: dict = state.get("metrics") or {}

    severity: str = state.get("severity", "none")
    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    logger.info(
        "Reporting Agent: generating report for %s (%s)",
        model_id,
        environment,
    )

    # ------------------------------------------------------------------
    # Historical stats
    # ------------------------------------------------------------------

    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------

    report = _build_report(state, historical_stats)

    # ------------------------------------------------------------------
    # Save incident
    # ------------------------------------------------------------------

    state_with_report = {
        **state,
        "report": report,
    }

    incident_id = rag.save_incident(state_with_report)

    logger.info("Saved incident to RAG: %s", incident_id)

    report = report.replace("pending", incident_id, 1)

    # ------------------------------------------------------------------
    # Slack
    # ------------------------------------------------------------------

    notifications: list[str] = []

    slack_enabled = (
        os.getenv("SLACK_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    )

    if slack_enabled:

        summary = _llm_executive_summary(state, report)

        slack_result = send_slack_notification(
            message=summary,
            severity=severity,
            incident_id=incident_id,
        )

        if slack_result.get("status") == "success":
            notifications.append("slack")
        else:
            logger.warning(
                "Slack notification failed: %s",
                slack_result.get("detail"),
            )

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    email_min_severity = os.getenv("EMAIL_MIN_SEVERITY", "major")

    severity_rank = {
        "none": 0,
        "minor": 1,
        "major": 2,
        "critical": 3,
    }

    should_email = (
        severity_rank.get(severity, 0)
        >= severity_rank.get(email_min_severity, 2)
    )

    email_enabled = (
        os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    )

    if should_email and email_enabled:

        email_result = send_email_alert(
            subject=(
                f"[MLOps {severity.upper()}] "
                f"{model_id} degradation — {incident_id}"
            ),
            body=report,
            severity=severity,
            incident_id=incident_id,
        )

        if email_result.get("status") == "success":
            notifications.append("email")
        else:
            logger.warning(
                "Email alert failed: %s",
                email_result.get("detail"),
            )

    logger.info(
        "Reporting complete. incident_id=%s notifications=%s",
        incident_id,
        notifications,
    )

    # ------------------------------------------------------------------
    # Threshold learning
    # ------------------------------------------------------------------

    try:
        threshold_learning(state, rag)
    except Exception as exc:
        logger.error(
            "Threshold learning failed: %s",
            exc,
        )

    return {
        **state,
        "report": report,
        "incident_id": incident_id,
        "notifications_sent": notifications,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=(
                    f"[Reporting] incident_id={incident_id} "
                    f"notifications={notifications}"
                )
            )
        ],
    }
