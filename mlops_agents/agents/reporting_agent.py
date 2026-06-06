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

from mlops_agents.llm_manager import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState
from mlops_agents.rag.store import RAGStore
from mlops_agents.tools.mcp_tools import send_slack_notification, send_email_alert
from mlops_agents.agents.threshold_manager import run_threshold_update

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(state: AgentState, historical_stats: dict) -> str:
    """Generate a structured markdown incident report."""

    metrics: dict = state.get("metrics") or {}
    diag_json: dict = state.get("diagnosis_json") or {}
    similar = state.get("similar_incidents") or []
    metadata: dict = metrics.get("metadata") or {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # Monitor stores model_id/environment at top-level state and inside
    # metrics["metadata"]; they are NOT keys of `metrics` itself.
    model_id = state.get("model_id") or metadata.get("model_name", "unknown")
    environment = state.get("environment") or "production"
    model_version = state.get("model_version") or metadata.get("model_version", "unknown")
    severity = state.get("severity", "none")

    report_lines = [
        f"# Incident Report — {model_id}",
        "",
        f"**Date:** {now}  ",
        f"**Model:** `{model_id}` v{model_version}  ",
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

    llm = get_llm(temperature=0)

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
# Reporting Agent
# ---------------------------------------------------------------------------

def reporting_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Reporting Agent.
    Aggregates diagnosis metrics, strips telemetry histogram payloads for vector safety,
    persists history snapshots to RAG memory, and triggers threshold optimizations.
    """
    metrics: dict = state.get("metrics") or {}
    severity: str = state.get("severity", "none")
    
    # Extract structural names using our updated metadata object patterns
    metadata = metrics.get("metadata") or {}
    model_id: str = state.get("model_id") or metadata.get("model_name", "unknown")
    environment: str = state.get("environment", "production")

    logger.info("Reporting Agent: generating incident report for %s (%s)", model_id, environment)

    # 1. Historical Stats retrieval from vector layer
    historical_stats = rag.get_incident_stats(
        model_id=model_id,
        environment=environment,
    )

    # 2. Build report string context payload layout
    report = _build_report(state, historical_stats)

    # 3. CRITICAL STRUCTURAL ADJUSTMENT: Prepare a vector-optimized state copy.
    # We strip out the heavy histogram arrays so we don't bloat the vector database metadata attributes.
    sanitized_metrics = {
        k: v for k, v in metrics.items() 
        if k not in ["reference_histograms", "production_histograms"]
    }
    # Track metadata summary stats instead of heavy matrices
    sanitized_metrics["had_reference_histograms"] = "reference_histograms" in metrics
    sanitized_metrics["had_production_histograms"] = "production_histograms" in metrics

    vector_safe_state = {
        **state,
        "metrics": sanitized_metrics,
        "report": report
    }

    # Save clean, optimized document footprint to RAG
    incident_id = rag.save_incident(vector_safe_state)
    logger.info("Saved compact incident snapshot to RAG database. Assigned ID: %s", incident_id)

    # Replace temp tokens with valid incident identifiers
    report = report.replace("pending", incident_id, 1)

    # 4. Slack Channels Notification Handlers (Disabled via Env Flags based on your context)
    notifications: list[str] = []
    slack_enabled = os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true"

    if slack_enabled:
        try:
            summary = _llm_executive_summary(state, report)
            slack_result = send_slack_notification(
                message=summary,
                severity=severity,
                incident_id=incident_id,
            )
            if slack_result.get("status") == "success":
                notifications.append("slack")
        except Exception as err:
            logger.warning("Slack reporting interface bypassed or error encountered: %s", err)

    # 5. Email Alert Dispatch Handling (Disabled via Env Flags based on your context)
    email_min_severity = os.getenv("EMAIL_MIN_SEVERITY", "major")
    severity_rank = {"none": 0, "minor": 1, "major": 2, "critical": 3}
    
    should_email = severity_rank.get(severity, 0) >= severity_rank.get(email_min_severity, 2)
    email_enabled = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true"

    if should_email and email_enabled:
        try:
            email_result = send_email_alert(
                subject=f"[MLOps {severity.upper()}] {model_id} degradation — {incident_id}",
                body=report,
                severity=severity,
                incident_id=incident_id,
            )
            if email_result.get("status") == "success":
                notifications.append("email")
        except Exception as err:
            logger.warning("Email reporting interface bypassed or error encountered: %s", err)

    logger.info("Reporting sequence complete. incident_id=%s notifications=%s", incident_id, notifications)

    # 6. Threshold learning (Feeds tuned boundary rules back into RAG for Monitor Node ingestion)
    print(f"[Reporting Agent] Initiating threshold optimization for model {model_id} in environment {environment}")
    try:
        run_threshold_update(state, rag)
    except Exception as exc:
        logger.error("Dynamic threshold tuning optimizer failed: %s", exc)

    return {
        **state,
        "report": report,
        "incident_id": incident_id,
        "notifications_sent": notifications,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"[Reporting] Compiled incident_id={incident_id}. Actions completed.")
        ]
    }
