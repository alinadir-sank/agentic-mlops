"""
agents/diagnosis_agent.py

Diagnosis Agent — LLM-powered root cause analysis enriched with:
  - Similar past incidents from the incidents RAG collection
  - Relevant runbooks and post-mortems from the runbooks collection
  - Recent metrics trend from the metrics_history collection

LLM output is constrained via Pydantic structured output + with_structured_output()
+ with_retry() — identical pattern to the threshold advisor and monitor agent —
so small 1B models cannot produce unparseable JSON.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

from state import AgentState
from mlops_agents.rag.store import RAGStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid enum literals
# ---------------------------------------------------------------------------

RecommendedAction  = Literal["retrain", "rollback", "scale", "investigate"]
RootCauseCategory  = Literal[
    "concept_drift", "data_drift", "model_staleness",
    "infrastructure", "data_quality", "unknown"
]
DataStrategy       = Literal["recent_window", "full_history", "weighted_recent", "drift_period_only"]
OptimizeFor        = Literal["f2_score", "roc_auc", "recall", "precision"]
DeploymentStrategy = Literal["canary", "blue_green", "immediate"]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RetrainPrescription(BaseModel):
    """
    Structured retraining prescription produced when recommended_action == 'retrain'.
    Every numeric field is clamped at the validator layer so the LLM cannot
    produce out-of-range values.
    """

    data_strategy: DataStrategy = Field(
        default="recent_window",
        description="How to select training data for the retrain run.",
    )
    window_days: int = Field(
        default=30,
        description="How many calendar days of data to include.",
    )
    drift_period_weight: float = Field(
        default=1.5,
        description="Upsampling multiplier for drift-period records (1.0 = no upsampling).",
    )
    exclude_before: str = Field(
        default="",
        description="ISO-8601 date; exclude data before this date. Empty string = no exclusion.",
    )
    refit_preprocessors: bool = Field(
        default=True,
        description="Whether to refit scalers/encoders on the new training window.",
    )
    drifted_features: list[str] = Field(
        default_factory=list,
        description="Feature names that showed statistically significant drift.",
    )
    optimize_for: OptimizeFor = Field(
        default="recall",
        description="Primary metric to optimise during hyperparameter search.",
    )
    target_recall: float = Field(
        default=0.80,
        description="Minimum acceptable recall on the validation set.",
    )
    target_roc_auc: float = Field(
        default=0.88,
        description="Minimum acceptable ROC-AUC on the validation set.",
    )
    deployment_strategy: DeploymentStrategy = Field(
        default="canary",
        description="How to promote the retrained model to production.",
    )
    canary_traffic_pct: int = Field(
        default=10,
        description="Percentage of traffic routed to the canary model (1–50).",
    )
    shadow_period_hours: int = Field(
        default=2,
        description="Hours to run the new model in shadow mode before promotion.",
    )

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("window_days", mode="before")
    @classmethod
    def clamp_window_days(cls, v: Any) -> int:
        try:
            return max(7, min(180, int(v)))
        except (TypeError, ValueError):
            return 30

    @field_validator("drift_period_weight", mode="before")
    @classmethod
    def clamp_drift_weight(cls, v: Any) -> float:
        try:
            return max(1.0, min(5.0, float(v)))
        except (TypeError, ValueError):
            return 1.5

    @field_validator("target_recall", "target_roc_auc", mode="before")
    @classmethod
    def clamp_metric_targets(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.80

    @field_validator("canary_traffic_pct", mode="before")
    @classmethod
    def clamp_canary_pct(cls, v: Any) -> int:
        try:
            return max(1, min(50, int(v)))
        except (TypeError, ValueError):
            return 10

    @field_validator("shadow_period_hours", mode="before")
    @classmethod
    def clamp_shadow_hours(cls, v: Any) -> int:
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 2

    @field_validator("data_strategy", mode="before")
    @classmethod
    def normalise_data_strategy(cls, v: Any) -> str:
        valid = ("recent_window", "full_history", "weighted_recent", "drift_period_only")
        if isinstance(v, str) and v.strip().lower() in valid:
            return v.strip().lower()
        return "recent_window"

    @field_validator("optimize_for", mode="before")
    @classmethod
    def normalise_optimize_for(cls, v: Any) -> str:
        valid = ("f2_score", "roc_auc", "recall", "precision")
        if isinstance(v, str) and v.strip().lower() in valid:
            return v.strip().lower()
        return "recall"

    @field_validator("deployment_strategy", mode="before")
    @classmethod
    def normalise_deployment_strategy(cls, v: Any) -> str:
        valid = ("canary", "blue_green", "immediate")
        if isinstance(v, str) and v.strip().lower() in valid:
            return v.strip().lower()
        return "canary"

    @field_validator("drifted_features", mode="before")
    @classmethod
    def normalise_features(cls, v: Any) -> list[str]:
        """Accept list-of-dicts or list-of-strings from small models."""
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(
                    item.get("name") or item.get("feature") or item.get("text")
                    or json.dumps(item)
                )
            else:
                result.append(str(item))
        return result


class DiagnosisOutput(BaseModel):
    """
    Full structured output schema for the Diagnosis Agent LLM call.

    retrain_prescription is Optional — the LLM should set it only when
    recommended_action == 'retrain'. We enforce this in the model validator.
    """

    root_cause: str = Field(
        description="Concise one-sentence root cause of the degradation.",
    )
    root_cause_category: RootCauseCategory = Field(
        default="unknown",
        description=(
            "Category of root cause: concept_drift | data_drift | model_staleness "
            "| infrastructure | data_quality | unknown"
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="List of evidence points supporting the root cause.",
    )
    recommended_action: RecommendedAction = Field(
        description="Remediation action: retrain | rollback | scale | investigate",
    )
    confidence: float = Field(
        description="Confidence score for this diagnosis, between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="2-3 sentence reasoning chain connecting evidence to recommendation.",
    )
    retrain_prescription: Optional[RetrainPrescription] = Field(
        default=None,
        description="Retraining parameters — only populated when recommended_action is 'retrain'.",
    )

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("recommended_action", mode="before")
    @classmethod
    def normalise_action(cls, v: Any) -> str:
        valid = ("retrain", "rollback", "scale", "investigate")
        if isinstance(v, str) and v.strip().lower() in valid:
            return v.strip().lower()
        logger.warning("Invalid recommended_action '%s' — defaulting to 'investigate'", v)
        return "investigate"

    @field_validator("root_cause_category", mode="before")
    @classmethod
    def normalise_category(cls, v: Any) -> str:
        valid = ("concept_drift", "data_drift", "model_staleness",
                 "infrastructure", "data_quality", "unknown")
        if isinstance(v, str) and v.strip().lower() in valid:
            return v.strip().lower()
        return "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("evidence", mode="before")
    @classmethod
    def normalise_evidence(cls, v: Any) -> list[str]:
        """
        Small models sometimes return evidence as list-of-dicts instead of strings.
        Flatten to strings here rather than in a separate post-processing step.
        """
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(
                    item.get("point") or item.get("text") or item.get("description")
                    or json.dumps(item)
                )
            else:
                result.append(str(item))
        return result

    @model_validator(mode="after")
    def clear_prescription_when_not_retraining(self) -> "DiagnosisOutput":
        """Ensure prescription is None when action is not retrain."""
        if self.recommended_action != "retrain":
            self.retrain_prescription = None
        return self


# ---------------------------------------------------------------------------
# RAG context builders
# ---------------------------------------------------------------------------

def _format_similar_incidents(incidents: list[dict]) -> str:
    if not incidents:
        return "No similar past incidents found."

    lines = []
    for i, inc in enumerate(incidents, 1):
        meta    = inc.get("metadata", {})
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
            f"{rb.get('document', '')[:600]}"
        )
    return "\n\n---\n\n".join(lines)


def _format_trend(trend: list[dict]) -> str:
    if not trend:
        return "No recent trend data."

    rows = []
    for snap in trend[:8]:
        rows.append(
            f"  {snap.get('sampled_at', 'N/A')[:19]}  "
            f"acc={snap.get('accuracy', 'N/A')}  "
            f"drift={snap.get('drift_score', 'N/A')}  "
            f"lat={snap.get('latency_p99_ms', 'N/A')}ms  "
            f"err={snap.get('error_rate', 'N/A')}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Fallback rule-based diagnosis (fires only when all LLM retries fail)
# ---------------------------------------------------------------------------

def _fallback_diagnosis(metrics: dict, severity: str) -> DiagnosisOutput:
    """
    Returns a DiagnosisOutput constructed from deterministic rules.
    This is the last resort — only reached if all 3 LLM retries are exhausted.
    """
    accuracy   = metrics.get("accuracy")
    drift      = metrics.get("drift_score")
    latency    = metrics.get("latency_p99_ms")
    error_rate = metrics.get("error_rate")

    if drift is not None and drift > 0.4:
        return DiagnosisOutput(
            root_cause="Significant data distribution shift detected.",
            root_cause_category="data_drift",
            evidence=[f"Drift score {drift:.3f} exceeds threshold 0.40."],
            recommended_action="retrain",
            confidence=0.5,
            reasoning="High drift score strongly suggests training data distribution mismatch.",
            retrain_prescription=RetrainPrescription(
                data_strategy="weighted_recent",
                window_days=30,
                drift_period_weight=2.0,
                refit_preprocessors=True,
                optimize_for="recall",
                target_recall=0.80,
                target_roc_auc=0.88,
                deployment_strategy="canary",
                canary_traffic_pct=10,
                shadow_period_hours=2,
            ),
        )

    if accuracy is not None and accuracy < 0.70:
        return DiagnosisOutput(
            root_cause="Model accuracy has degraded significantly.",
            root_cause_category="model_staleness",
            evidence=[f"Accuracy {accuracy:.3f} is below acceptable threshold."],
            recommended_action="retrain",
            confidence=0.4,
            reasoning="Low accuracy with no other signals suggests model staleness.",
            retrain_prescription=RetrainPrescription(
                data_strategy="recent_window",
                window_days=30,
                drift_period_weight=1.5,
                refit_preprocessors=True,
                optimize_for="recall",
                target_recall=0.80,
                target_roc_auc=0.88,
                deployment_strategy="canary",
                canary_traffic_pct=10,
                shadow_period_hours=2,
            ),
        )

    if latency is not None and latency > 1500:
        return DiagnosisOutput(
            root_cause="Serving latency is critically high.",
            root_cause_category="infrastructure",
            evidence=[f"p99 latency {latency:.0f}ms exceeds 1500ms threshold."],
            recommended_action="scale",
            confidence=0.5,
            reasoning="Latency spike without accuracy degradation suggests capacity issue.",
        )

    return DiagnosisOutput(
        root_cause="Ambiguous degradation — manual investigation required.",
        root_cause_category="unknown",
        evidence=[
            f"accuracy={accuracy}",
            f"drift={drift}",
            f"latency_p99={latency}ms",
            f"error_rate={error_rate}",
        ],
        recommended_action="investigate",
        confidence=0.3,
        reasoning="No single metric clearly identifies the root cause.",
    )


# ---------------------------------------------------------------------------
# LLM setup (mirrors threshold agent + monitor agent pattern exactly)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior MLOps engineer specialising in ML model reliability.
Diagnose the root cause of a model degradation incident and recommend a remediation action.

You will be given:
1. Current model metrics
2. Similar past incidents from the institutional knowledge base
3. Relevant runbooks and post-mortems
4. Recent metrics trend

Return a JSON object with exactly these fields:
  root_cause              — concise one-sentence root cause
  root_cause_category     — one of: concept_drift | data_drift | model_staleness | infrastructure | data_quality | unknown
  evidence                — list of strings (evidence points)
  recommended_action      — one of: retrain | rollback | scale | investigate
  confidence              — float 0.0–1.0
  reasoning               — 2-3 sentence reasoning chain
  retrain_prescription    — object (only when recommended_action is retrain, else null)

retrain_prescription fields (all required when present):
  data_strategy           — one of: recent_window | full_history | weighted_recent | drift_period_only
  window_days             — integer 7–180
  drift_period_weight     — float 1.0–5.0
  exclude_before          — ISO-8601 date string or empty string
  refit_preprocessors     — boolean
  drifted_features        — list of feature name strings
  optimize_for            — one of: f2_score | roc_auc | recall | precision
  target_recall           — float 0.0–1.0
  target_roc_auc          — float 0.0–1.0
  deployment_strategy     — one of: canary | blue_green | immediate
  canary_traffic_pct      — integer 1–50
  shadow_period_hours     — integer >= 0

Root cause category rules:
  drift_score high + accuracy gradual decline   → data_drift
  drift_score moderate + recall sudden collapse → concept_drift
  drift_score high + recall collapse + errors   → data_drift (use drift_period_only strategy)
  latency spike + accuracy stable               → infrastructure (scale, not retrain)"""


def _build_diagnosis_llm() -> Any:
    """
    ChatOllama bound to DiagnosisOutput schema with exponential-backoff retry.
    Identical construction pattern to _llm_threshold_advisor() and _build_severity_llm().
    """
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0,
    ).with_structured_output(DiagnosisOutput)

    llm = llm.with_retry(
        retry_exception_types=(ValidationError, Exception),
        max_attempt_number=3,
        wait_exponential_jitter=True,
    )
    return llm


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def diagnosis_agent(state: AgentState, rag: RAGStore) -> AgentState:
    """
    LangGraph node — Diagnosis Agent.

    1. Retrieves RAG context: similar incidents, runbooks, metrics trend.
    2. Invokes the structured LLM (DiagnosisOutput schema).
    3. Falls back to rule-based DiagnosisOutput on total LLM failure.
    4. Updates state with typed diagnosis fields.
    """
    metrics:     dict = state.get("metrics") or {}
    severity:    str  = state.get("severity", "minor")
    model_id:    str  = metrics.get("model_id", "unknown")
    environment: str  = metrics.get("environment", "production")

    logger.info(
        "Diagnosis Agent: analysing %s (%s) severity=%s", model_id, environment, severity
    )

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
    n_runbooks  = int(os.getenv("RAG_SIMILAR_RUNBOOKS",  "3"))
    n_trend     = int(os.getenv("RAG_TREND_SNAPSHOTS",   "10"))

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
Return only the JSON object — no preamble, no markdown."""

    # ── 4. Structured LLM call ──────────────────────────────────────────────
    llm = _build_diagnosis_llm()
    result: DiagnosisOutput

    try:
        result = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        logger.info(
            "Diagnosis: '%s' [%s] → action=%s confidence=%.2f prescription=%s",
            result.root_cause,
            result.root_cause_category,
            result.recommended_action,
            result.confidence,
            "yes" if result.retrain_prescription else "none",
        )
    except Exception as exc:
        logger.error(
            "All LLM diagnosis retries exhausted: %s — using rule-based fallback", exc
        )
        result = _fallback_diagnosis(metrics, severity)

    # ── 5. Materialise prescription and drifted features ────────────────────
    prescription = (
        result.retrain_prescription.model_dump()
        if result.retrain_prescription else None
    )
    drifted = prescription.get("drifted_features", []) if prescription else []

    # ── 6. Update state ─────────────────────────────────────────────────────
    # Store both the full Pydantic-validated dict and a plain diagnosis_json dict
    # so downstream agents (reporting, remediation) have the same interface as before.
    diagnosis_json = result.model_dump()

    return {
        **state,
        "diagnosis":            result.root_cause,
        "diagnosis_json":       diagnosis_json,
        "recommended_action":   result.recommended_action,
        "retrain_prescription": prescription,
        "drifted_features":     drifted,
        "similar_incidents":    similar_incidents,
        "relevant_runbooks":    relevant_runbooks,
        "messages": state.get("messages", []) + [
            HumanMessage(
                content=(
                    f"[Diagnosis] root_cause='{result.root_cause}' "
                    f"category={result.root_cause_category} "
                    f"action={result.recommended_action} "
                    f"confidence={result.confidence:.2f} "
                    f"prescription={'yes' if prescription else 'none'}"
                )
            )
        ],
    }