"""
agents/remediation_agent.py

Remediation Agent — dispatches the appropriate tool based on the
recommended_action from the Diagnosis Agent.

No LLM calls here — this is a deterministic tool dispatcher.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage

from state import AgentState
from tools.mcp_tools import (
    trigger_retraining_pipeline,
    rollback_deployment,
    scale_deployment,
    open_github_issue,
)

logger = logging.getLogger(__name__)


def remediation_agent(state: AgentState) -> AgentState:
    """
    LangGraph node — Remediation Agent.

    Executes the recommended_action using the real MCP tools.
    Records the outcome in state as remediation_action + remediation_status.
    """
    metrics: dict = state.get("metrics") or {}
    model_id: str = metrics.get("model_id", os.getenv("DEFAULT_MODEL_ID", "unknown"))
    environment: str = metrics.get("environment", os.getenv("DEFAULT_ENVIRONMENT", "production"))
    action: str = state.get("recommended_action", "investigate")
    diagnosis: str = state.get("diagnosis", "")
    severity: str = state.get("severity", "minor")

    logger.info(
        "Remediation Agent: executing action='%s' for %s (%s)",
        action, model_id, environment,
    )

    if action == "retrain":
        result = trigger_retraining_pipeline(
            model_id=model_id,
            environment=environment,
            reason=diagnosis,
        )

    elif action == "rollback":
        result = rollback_deployment(
            model_id=model_id,
            environment=environment,
            reason=diagnosis,
        )

    elif action == "scale":
        result = scale_deployment(
            model_id=model_id,
            environment=environment,
        )

    elif action == "investigate":
        result = open_github_issue(
            model_id=model_id,
            environment=environment,
            diagnosis=diagnosis,
            severity=severity,
            metrics=metrics,
        )

    else:
        result = {
            "status": "failed",
            "detail": f"Unknown action '{action}'. No tool executed.",
        }
        logger.error("Unknown remediation action '%s'", action)

    status: str = result.get("status", "failed")
    detail: str = result.get("detail", "")

    logger.info("Remediation outcome: status=%s detail=%s", status, detail)

    return {
        **state,
        "remediation_action": action,
        "remediation_status": status,
        "remediation_detail": detail,
        "messages": state.get("messages", [])
        + [
            HumanMessage(
                content=(
                    f"[Remediation] action={action} status={status} — {detail}"
                )
            )
        ],
    }
