"""
agents/reporting_agent.py

Reporting Agent — generates a structured markdown incident report,
saves the full incident to the RAG incidents collection, and dispatches
Slack and email notifications.
"""

from __future__ import annotations

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
        f"",
        f"**Date:** {now}  ",
        f"**Model:** `{model_id}` v{metrics.get('model_version', 'unknown')}  ",
        f"**Environment:** {environment}  ",
        f"**Severity:** {severity.upper()}  ",
        f"**Incident ID:** `{state.get('incident_id', 'pending')}`",
        f"",
        f"---",
        f"",
        f"## Metrics at Detection",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {metrics.get('accuracy', 'N/A')} |",
        f"| Drift Score | {metrics.get('drift_score', 'N/A')} |",
        f"| p99 Latency | {metrics.get('latency_p99_ms', 'N/A')} ms |",
        f"| p95 Latency | {metrics.get('latency_p95_ms', 'N/A')} ms |",
        f"| Error Rate | {metrics.get('error_rate', 'N/A')} |",
        f"| Predictions (window) | {metrics.get('prediction_count', 'N/A')} |",
        f"",
        f"---",
        f"",
        f"## Root Cause Analysis",
        f"",
        f"**Root Cause:** {state.get('diagnosis', 'N/A')}  ",
        f"**Confidence:** {diag_json.get('confidence', 'N/A')}  ",
        f"",
        f"**Evidence:**",
    ]
    for ev in diag_json.get("evidence", []):
        report_lines.append(f"- {ev}")

    report_lines += [
        f"",
        f"**Reasoning:** {diag_json.get('reasoning', 'N/A')}",
        f"",
        f"---",
        f"",
        f"## Remediation",
        f"",
        f"**Action:** `{state.get('remediation_action', 'N/A')}`  ",
        f"**Status:** {state.get('remediation_status', 'N/A')}  ",
        f"**Detail:** {state.get('remediation_detail', 'N/A')}  ",
        f"**Human Approved:** {state.get('human_approved', False)}",
        f"",
        f"---",
        f"",
        f"## Historical Context",
        f"",
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
            f"",
            f"### Most Similar Past Incidents",
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


def _llm_executive_summary(state: AgentState, report: str) -> str:
    """
    Ask the LLM to write a 3-sentence executive summary for the Slack notification.
    Falls back to a templated summary on failure.
    """
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=ollama_url, temperature=0)

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You write concise 3-sentence executive summaries of MLOps incidents for Slack."
                ),
                HumanMessage(
                    content=(
                        f"Write a 3-sentence executive summary of this incident report:\n\n{report[:1500]}"
                    )
                ),
            ]
        )
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM summary failed: %s — using template", exc)
        metrics: dict = state.get("metrics") or {}
        return (
            f"Model `{metrics.get('model_id', 'unknown')}` ({state.get('severity','?').upper()}) "
            f"in {metrics.get('environment', 'production')} experienced degradation. "
            f"Root cause: {state.get('diagnosis', 'unknown')}. "
            f"Action `{state.get('remediation_action', 'N/A')}` completed with "
            f"status: {state.get('remediation_status', 'N/A')}."
        )
    
def _compute_drift_trend(trend: list[dict]) -> str:
    """
    Simple drift trend detector using last 5 points.
    """
    if len(trend) < 3:
        return "flat"

    recent = trend[:5]
    drift_vals = [m.get("drift_score", 0.0) for m in recent if m.get("drift_score") is not None]

    if len(drift_vals) < 3:
        return "flat"

    # crude slope
    if drift_vals[0] > drift_vals[-1] * 1.1:
        return "increasing"
    if drift_vals[0] < drift_vals[-1] * 0.9:
        return "decreasing"

    return "flat"


def _clamp_thresholds(t: dict) -> dict:
    """Clamp thresholds to safe bounds."""
    t["accuracy_major"] = max(0.6, min(0.85, t["accuracy_major"]))
    t["accuracy_critical"] = max(0.5, min(0.75, t["accuracy_critical"]))

    t["drift_major"] = max(0.1, min(0.7, t["drift_major"]))
    t["drift_critical"] = max(0.3, min(0.9, t["drift_critical"]))

    t["latency_major_ms"] = max(500, min(3000, t["latency_major_ms"]))
    t["latency_critical_ms"] = max(1000, min(5000, t["latency_critical_ms"]))

    t["error_rate_major"] = max(0.01, min(0.2, t["error_rate_major"]))
    t["error_rate_critical"] = max(0.05, min(0.3, t["error_rate_critical"]))

    return t


def threshold_learning(state: AgentState, rag: RAGStore) -> None:
    """
    Outcome-based threshold tuning.

    Uses:
    - Incident outcome
    - Historical stats
    - Trend awareness
    - Learning rate
    - Cooldown
    """
    metrics: dict = state.get("metrics") or {}
    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    # ── Cooldown ─────────────────────────────────────────────────────
    existing = rag.get_dynamic_thresholds(model_id=model_id)

    now = datetime.now(timezone.utc)
    if existing and "updated_at" in existing:
        last_update = datetime.fromisoformat(existing["updated_at"])
        if (now - last_update).total_seconds() < 1800:  # 30 minutes
            logger.info("Threshold learning skipped (cooldown active)")
            return

    # ── Base thresholds ──────────────────────────────────────────────
    if existing:
        thresholds = existing["thresholds"].copy()
        version = existing.get("version", 1) + 1
    else:
        # fallback to defaults via monitor logic
        thresholds = state.get("thresholds") or {}
        version = 1

    if not thresholds:
        logger.warning("No thresholds found — skipping learning")
        return

    # ── Learning rate ────────────────────────────────────────────────
    stats = rag.get_incident_stats(model_id=model_id, environment=environment)
    num_incidents = max(stats.get("total", 1), 1)
    learning_rate = min(0.1, 1 / num_incidents)

    logger.info("Threshold learning rate: %.4f (incidents=%d)", learning_rate, num_incidents)

    severity = state.get("severity", "none")
    remediation_status = state.get("remediation_status", "")
    human_approved = state.get("human_approved", False)

    # ── Outcome-based adjustment ─────────────────────────────────────

    # Case 1: false positive → too sensitive
    if severity in ("major", "critical") and (
        remediation_status == "skipped" or not human_approved
    ):
        logger.info("Detected false positive → relaxing thresholds")
        thresholds["accuracy_major"] += 0.01 * learning_rate
        thresholds["drift_major"] += 0.02 * learning_rate

    # Case 2: successful remediation → reinforce
    elif remediation_status == "success":
        logger.info("Successful remediation → reinforcing sensitivity")
        thresholds["accuracy_major"] -= 0.005 * learning_rate
        thresholds["drift_major"] -= 0.01 * learning_rate

    # Case 3: missed detection (minor but bad metrics)
    elif severity in ("none", "minor"):
        acc = metrics.get("accuracy", 1.0)
        drift = metrics.get("drift_score", 0.0)

        if acc < thresholds["accuracy_major"] or drift > thresholds["drift_major"]:
            logger.info("Missed detection → tightening thresholds")
            thresholds["accuracy_major"] -= 0.01 * learning_rate
            thresholds["drift_major"] -= 0.02 * learning_rate

    # ── Trend awareness ──────────────────────────────────────────────
    trend = rag.query_recent_metrics(model_id=model_id, environment=environment)
    drift_trend = _compute_drift_trend(trend)

    if drift_trend == "increasing":
        logger.info("Drift trend increasing → preemptively tightening drift threshold")
        thresholds["drift_major"] -= 0.02 * learning_rate

    # ── Clamp ────────────────────────────────────────────────────────
    thresholds = _clamp_thresholds(thresholds)

    # ── Save ─────────────────────────────────────────────────────────
    payload = {
        "model_id": model_id,
        "environment": environment,
        "version": version,
        "updated_at": now.isoformat(),
        "thresholds": thresholds,
    }

    rag.save_dynamic_thresholds(model_id=model_id, thresholds=payload)

    logger.info("Updated dynamic thresholds v%s: %s", version, thresholds)


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def reporting_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Reporting Agent.

    1. Queries RAG for historical incident statistics.
    2. Generates a full markdown incident report.
    3. Saves the incident to the RAG incidents collection.
    4. Sends Slack notification (executive summary).
    5. Sends email alert for major/critical incidents.
    6. Returns updated state.
    """
    metrics: dict = state.get("metrics") or {}
    severity: str = state.get("severity", "none")
    model_id: str = metrics.get("model_id", "unknown")
    environment: str = metrics.get("environment", "production")

    logger.info("Reporting Agent: generating report for %s (%s)", model_id, environment)

    # ── 1. Historical stats from RAG ────────────────────────────────────────
    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    # ── 2. Build report ─────────────────────────────────────────────────────
    report = _build_report(state, historical_stats)

    # ── 3. Save incident to RAG ─────────────────────────────────────────────
    # Temporarily set report in state so save_incident captures it
    state_with_report = {**state, "report": report}
    incident_id = rag.save_incident(state_with_report)
    logger.info("Saved incident to RAG: %s", incident_id)

    # Update report with the real incident_id
    report = report.replace("pending", incident_id, 1)

    # ── 4. Slack notification ────────────────────────────────────────────────
    notifications: list[str] = []

    slack_enabled = os.getenv("SLACK_NOTIFICATIONS_ENABLED", "true").lower() == "true"
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
            logger.warning("Slack notification failed: %s", slack_result.get("detail"))

    # ── 5. Email alert (major/critical only by default) ──────────────────────
    email_min_severity = os.getenv("EMAIL_MIN_SEVERITY", "major")
    severity_rank = {"none": 0, "minor": 1, "major": 2, "critical": 3}
    should_email = (
        severity_rank.get(severity, 0) >= severity_rank.get(email_min_severity, 2)
    )

    email_enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
    if should_email and email_enabled:
        email_result = send_email_alert(
            subject=f"[MLOps {severity.upper()}] {model_id} degradation — {incident_id}",
            body=report,
            severity=severity,
            incident_id=incident_id,
        )
        if email_result.get("status") == "success":
            notifications.append("email")
        else:
            logger.warning("Email alert failed: %s", email_result.get("detail"))

    logger.info(
        "Reporting complete. incident_id=%s notifications=%s",
        incident_id, notifications,
    )

    # ── 6. Threshold learning ───────────────────────────────────────────
    try:
        threshold_learning(state, rag)
    except Exception as e:
        logger.error("Threshold learning failed: %s", e)

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
