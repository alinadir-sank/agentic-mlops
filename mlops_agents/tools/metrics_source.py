"""
tools/metrics_source.py

Production metrics data source adapter.

Assumes the following integrations are configured via environment variables:

  Azure Monitor   — AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
                    AZURE_MONITOR_WORKSPACE_ID, AZURE_TENANT_ID,
                    AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
  Prometheus      — PROMETHEUS_URL
  MLflow          — MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

All public functions raise MetricsSourceError on unrecoverable failure so
that the Monitor Agent can surface a structured error to the graph rather
than crashing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MetricsSourceError(Exception):
    """Raised when a metrics data source cannot be reached or returns bad data."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """
    Normalised metrics snapshot for a single deployed model.

    All fields that could not be fetched are set to None so downstream
    code can distinguish "missing" from "zero".
    """

    model_id: str
    model_version: str
    environment: str
    sampled_at: str                      # ISO-8601 UTC

    # Performance
    accuracy: float | None = None        # 0–1
    f1_score: float | None = None        # 0–1
    precision: float | None = None       # 0–1
    recall: float | None = None          # 0–1
    auc_roc: float | None = None         # 0–1

    # Drift
    drift_score: float | None = None     # PSI or JS-divergence, 0–1 scale
    feature_drift: dict | None = None    # per-feature drift scores
    label_drift: float | None = None     # prediction distribution shift

    # Serving / infra
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    error_rate: float | None = None      # 0–1
    prediction_count: int | None = None  # in the last monitoring window
    throughput_rps: float | None = None  # requests per second

    # Model registry metadata
    training_data_hash: str | None = None
    last_retrain_date: str | None = None  # ISO-8601
    deployed_at: str | None = None        # ISO-8601

    # Raw extras (anything platform-specific)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------

class AzureMonitorSource:
    """
    Pulls model performance metrics from Azure Monitor / Azure ML.

    Required env vars:
        AZURE_SUBSCRIPTION_ID
        AZURE_RESOURCE_GROUP
        AZURE_MONITOR_WORKSPACE_ID
        AZURE_TENANT_ID
        AZURE_CLIENT_ID
        AZURE_CLIENT_SECRET
        AZURE_ML_WORKSPACE_NAME  (for MLflow registry integration)
    """

    def __init__(self) -> None:
        try:
            from azure.identity import ClientSecretCredential
            from azure.monitor.query import MetricsQueryClient
        except ImportError as e:
            raise MetricsSourceError(
                "azure-monitor-query and azure-identity packages are required. "
                "pip install azure-monitor-query azure-identity"
            ) from e

        self._credential = ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )
        self._metrics_client = MetricsQueryClient(self._credential)
        self._subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
        self._resource_group = os.environ["AZURE_RESOURCE_GROUP"]
        self._workspace_id = os.environ["AZURE_MONITOR_WORKSPACE_ID"]

    def fetch(
        self,
        model_id: str,
        environment: str,
        window_minutes: int = 15,
    ) -> ModelMetrics:
        from azure.monitor.query import MetricAggregationType

        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=window_minutes)

        resource_uri = (
            f"/subscriptions/{self._subscription_id}"
            f"/resourceGroups/{self._resource_group}"
            f"/providers/Microsoft.MachineLearningServices"
            f"/workspaces/{self._workspace_id}"
            f"/onlineEndpoints/{model_id}"
        )

        try:
            result = self._metrics_client.query_resource(
                resource_uri=resource_uri,
                metric_names=[
                    "RequestsPerMinute",
                    "RequestLatency_P99",
                    "RequestLatency_P95",
                    "RequestLatency_P50",
                    "ModelErrorRate",
                ],
                timespan=(start, end),
                granularity=timedelta(minutes=window_minutes),
                aggregations=[MetricAggregationType.AVERAGE],
            )
        except Exception as exc:
            raise MetricsSourceError(
                f"Azure Monitor query failed for {model_id}: {exc}"
            ) from exc

        parsed: dict[str, float] = {}
        for metric in result.metrics:
            for ts in metric.timeseries:
                for dp in ts.data:
                    if dp.average is not None:
                        parsed[metric.name] = dp.average

        # Fetch accuracy / drift from Azure ML Data Drift monitor
        accuracy, drift_score = self._fetch_ml_monitor_metrics(
            model_id, environment, start, end
        )

        return ModelMetrics(
            model_id=model_id,
            model_version=self._get_model_version(model_id),
            environment=environment,
            sampled_at=end.isoformat(),
            latency_p50_ms=parsed.get("RequestLatency_P50"),
            latency_p95_ms=parsed.get("RequestLatency_P95"),
            latency_p99_ms=parsed.get("RequestLatency_P99"),
            error_rate=parsed.get("ModelErrorRate"),
            throughput_rps=parsed.get("RequestsPerMinute"),
            accuracy=accuracy,
            drift_score=drift_score,
        )

    def _fetch_ml_monitor_metrics(
        self,
        model_id: str,
        environment: str,
        start: datetime,
        end: datetime,
    ) -> tuple[float | None, float | None]:
        """Query Azure ML data drift and data quality monitors."""
        try:
            from azure.ai.ml import MLClient

            ml_client = MLClient(
                credential=self._credential,
                subscription_id=self._subscription_id,
                resource_group_name=self._resource_group,
                workspace_name=os.environ.get("AZURE_ML_WORKSPACE_NAME", ""),
            )
            monitor_name = f"{model_id}-{environment}-monitor"
            signals = ml_client.model_monitors.get(monitor_name)
            accuracy = getattr(signals, "accuracy", None)
            drift_score = getattr(signals, "data_drift_score", None)
            return accuracy, drift_score
        except Exception as exc:
            logger.warning("Could not fetch Azure ML monitor metrics: %s", exc)
            return None, None

    def _get_model_version(self, model_id: str) -> str:
        try:
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import Model

            ml_client = MLClient(
                credential=self._credential,
                subscription_id=self._subscription_id,
                resource_group_name=self._resource_group,
                workspace_name=os.environ.get("AZURE_ML_WORKSPACE_NAME", ""),
            )
            models: list[Model] = list(
                ml_client.models.list(name=model_id, latest_version_only=True)
            )
            return str(models[0].version) if models else "unknown"
        except Exception:
            return "unknown"


class PrometheusSource:
    """
    Pulls metrics from a Prometheus / Grafana endpoint.

    Expected env vars:
        PROMETHEUS_URL              — e.g. http://prometheus.svc:9090
        PROMETHEUS_BEARER_TOKEN     — optional
        PROMETHEUS_MODEL_LABEL      — label name used to identify the model,
                                      defaults to "model_id"

    Expected Prometheus metrics (configure in your ML serving layer):
        mlops_model_accuracy{model_id="...", environment="..."}
        mlops_model_drift_score{...}
        mlops_request_latency_seconds{quantile="0.99", ...}
        mlops_request_latency_seconds{quantile="0.95", ...}
        mlops_request_latency_seconds{quantile="0.50", ...}
        mlops_request_errors_total{...}
        mlops_predictions_total{...}
    """

    def __init__(self) -> None:
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise MetricsSourceError(
                "requests package required: pip install requests"
            ) from e

        self._base_url = os.environ["PROMETHEUS_URL"].rstrip("/")
        self._token = os.getenv("PROMETHEUS_BEARER_TOKEN", "")
        self._model_label = os.getenv("PROMETHEUS_MODEL_LABEL", "model_id")

    def _query(self, promql: str) -> float | None:
        import requests

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            resp = requests.get(
                f"{self._base_url}/api/v1/query",
                params={"query": promql},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
            return None
        except Exception as exc:
            logger.warning("Prometheus query failed ('%s'): %s", promql, exc)
            return None

    def fetch(
        self,
        model_id: str,
        environment: str,
        window_minutes: int = 15,
    ) -> ModelMetrics:
        lbl = self._model_label
        env_filter = f'{lbl}="{model_id}",environment="{environment}"'
        window = f"{window_minutes}m"

        accuracy = self._query(
            f'avg_over_time(mlops_model_accuracy{{{env_filter}}}[{window}])'
        )
        drift = self._query(
            f'avg_over_time(mlops_model_drift_score{{{env_filter}}}[{window}])'
        )
        lat_p99 = self._query(
            f'histogram_quantile(0.99, rate(mlops_request_latency_seconds_bucket{{{env_filter}}}[{window}])) * 1000'
        )
        lat_p95 = self._query(
            f'histogram_quantile(0.95, rate(mlops_request_latency_seconds_bucket{{{env_filter}}}[{window}])) * 1000'
        )
        lat_p50 = self._query(
            f'histogram_quantile(0.50, rate(mlops_request_latency_seconds_bucket{{{env_filter}}}[{window}])) * 1000'
        )
        errors = self._query(
            f'rate(mlops_request_errors_total{{{env_filter}}}[{window}])'
        )
        preds = self._query(
            f'increase(mlops_predictions_total{{{env_filter}}}[{window}])'
        )

        return ModelMetrics(
            model_id=model_id,
            model_version=os.getenv("MODEL_VERSION", "unknown"),
            environment=environment,
            sampled_at=datetime.now(timezone.utc).isoformat(),
            accuracy=accuracy,
            drift_score=drift,
            latency_p50_ms=lat_p50,
            latency_p95_ms=lat_p95,
            latency_p99_ms=lat_p99,
            error_rate=errors,
            prediction_count=int(preds) if preds is not None else None,
        )


class MLflowSource:
    """
    Pulls evaluation metrics from an MLflow Tracking Server.

    Expected env vars:
        MLFLOW_TRACKING_URI      — e.g. http://mlflow.svc:5000
        MLFLOW_EXPERIMENT_NAME   — experiment to query
        MLFLOW_BEARER_TOKEN      — optional, for managed MLflow
    """

    def __init__(self) -> None:
        try:
            import mlflow  # noqa: F401
        except ImportError as e:
            raise MetricsSourceError(
                "mlflow package required: pip install mlflow"
            ) from e

        import mlflow

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        token = os.getenv("MLFLOW_BEARER_TOKEN", "")
        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.tracking.MlflowClient()
        self._experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

    def fetch(
        self,
        model_id: str,
        environment: str,
        window_minutes: int = 15,  # noqa: ARG002 — not used for MLflow batch evals
    ) -> ModelMetrics:
        import mlflow

        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if not experiment:
            raise MetricsSourceError(
                f"MLflow experiment '{self._experiment_name}' not found."
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_id = '{model_id}' and tags.environment = '{environment}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            raise MetricsSourceError(
                f"No MLflow runs found for model_id={model_id}, env={environment}."
            )

        run = runs.iloc[0]

        def _get(col: str) -> float | None:
            v = run.get(f"metrics.{col}")
            return float(v) if v is not None else None

        return ModelMetrics(
            model_id=model_id,
            model_version=run.get("tags.mlflow.source.git.commit", "unknown"),
            environment=environment,
            sampled_at=datetime.now(timezone.utc).isoformat(),
            accuracy=_get("accuracy"),
            f1_score=_get("f1_score"),
            precision=_get("precision"),
            recall=_get("recall"),
            auc_roc=_get("auc_roc"),
            drift_score=_get("drift_score"),
        )


# ---------------------------------------------------------------------------
# Factory — selects data source from env
# ---------------------------------------------------------------------------

SOURCES: dict[str, type] = {
    "azure": AzureMonitorSource,
    "prometheus": PrometheusSource,
    "mlflow": MLflowSource,
}


def get_metrics_source() -> AzureMonitorSource | PrometheusSource | MLflowSource:
    """
    Instantiate and return the configured metrics data source.

    Controlled by METRICS_SOURCE env var (default: prometheus).
    """
    source_name = os.getenv("METRICS_SOURCE", "prometheus").lower()
    cls = SOURCES.get(source_name)
    if cls is None:
        raise MetricsSourceError(
            f"Unknown METRICS_SOURCE='{source_name}'. "
            f"Valid options: {list(SOURCES.keys())}"
        )
    return cls()


def fetch_model_metrics(
    model_id: str,
    environment: str,
    window_minutes: int = 15,
) -> dict[str, Any]:
    """
    Top-level function called by the Monitor Agent.

    Returns a plain dict (ModelMetrics.to_dict()) so it integrates cleanly
    with the LangGraph state without importing dataclass types in agent files.

    Raises MetricsSourceError on failure.
    """
    source = get_metrics_source()
    snapshot = source.fetch(
        model_id=model_id,
        environment=environment,
        window_minutes=window_minutes,
    )
    return snapshot.to_dict()
