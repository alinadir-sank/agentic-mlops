"""
api/main.py

FastAPI backend — wraps the LangGraph CLI pipeline with HTTP endpoints
so the Streamlit dashboard can trigger runs, poll status, and handle
human-in-the-loop approvals without touching the terminal.

This is a new file. It sits alongside the existing CLI (main.py) and
shares the same underlying graph/workflow.py and agent code.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /runs                      trigger a new pipeline run
    GET  /runs                      list all runs + status
    GET  /runs/{thread_id}          single run detail
    POST /runs/{thread_id}/approve  resume paused major-severity thread
    POST /runs/{thread_id}/reject   reject + end paused thread
    GET  /metrics/current           latest metrics snapshot from model server
    GET  /incidents                 ChromaDB incident history
    GET  /health                    Ollama + ChromaDB + model server reachability
    POST /drift/inject              forward drift config to model server
    POST /drift/reset               clear active drift on model server
    GET  /drift/status              current drift config from model server
"""

from __future__ import annotations

from cProfile import run
from contextlib import asynccontextmanager
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests as http
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
import time

from api.run_store import save_run, get_run, list_all_runs

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")

# ── config ────────────────────────────────────────────────────────────────────
MODEL_SERVER_URL = os.getenv("FRAUD_MODEL_MCP_URL", "http://localhost:8080")
OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@asynccontextmanager
async def lifespan(app: FastAPI):
    from mlops_agents.rag.init_collections import init
    # This runs EVERY time the server starts or reloads
    init()
    yield
    # Clean up code here if needed

app = FastAPI(
    title="MLOps Agent API",
    description="HTTP wrapper around the LangGraph MLOps pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# ── pydantic models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    model_id:    str = os.getenv("DEFAULT_MODEL_ID", "fraud-classifier-v1")
    environment: str = "production"

class DriftInjectRequest(BaseModel):
    type:          str         = "none"
    features:      list[str]   = []
    magnitude:     float       = 1.0
    swap_features: list[str]   = []
    description:   str         = ""

# ── pipeline execution helpers ────────────────────────────────────────────────

def _build_graph():
    """Import and build the LangGraph app. Isolated here so imports are lazy."""
    from mlops_agents.rag.store import RAGStore
    from mlops_agents.graph.workflow import build_graph
    rag = RAGStore()
    return build_graph(rag=rag)


async def _execute_pipeline(thread_id: str, model_id: str, environment: str):
    """
    Background task — runs the full pipeline and updates the runs registry.
    Catches GraphInterrupt to surface the human approval pause.
    """

    run = get_run(thread_id)
    run["status"]       = "running"
    run["started_at"]   = datetime.now(timezone.utc).isoformat()
    run["current_agent"] = "monitor"
    save_run(thread_id, run)

    try:
        app_graph = _build_graph()
        config    = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "model_id":    model_id,
            "environment": environment,
            "messages":    [],
        }

        # stream events so we can update current_agent in real time
        for event in app_graph.stream(initial_state, config=config):
            node_name = list(event.keys())[0]
            node_out  = event[node_name] or {}

            run["current_agent"] = node_name

            if "severity" in node_out:
                run["severity"] = node_out["severity"]
            if "diagnosis" in node_out:
                run["diagnosis"] = node_out["diagnosis"]
            if "recommended_action" in node_out:
                run["recommended_action"] = node_out["recommended_action"]
            if "remediation_status" in node_out:
                run["remediation_status"] = node_out["remediation_status"]
            if "incident_id" in node_out:
                run["incident_id"] = node_out["incident_id"]
            if "report" in node_out:
                run["report"] = node_out["report"]
            if "remediation_action" in node_out:
                run["remediation_action"] = node_out["remediation_action"]
            if "remediation_detail" in node_out:
                run["remediation_detail"] = node_out["remediation_detail"]
            if "diagnosis_json" in node_out:
                run["diagnosis_json"] = node_out["diagnosis_json"]
            if "retrain_prescription" in node_out:
                run["retrain_prescription"] = node_out["retrain_prescription"]
            if "drifted_features" in node_out:
                run["drifted_features"] = node_out["drifted_features"]
            if "similar_incidents" in node_out:
                run["similar_incidents"] = node_out["similar_incidents"]
            if "relevant_runbooks" in node_out:
                run["relevant_runbooks"] = node_out["relevant_runbooks"]
            if "notifications_sent" in node_out:
                run["notifications_sent"] = node_out["notifications_sent"]
            if "messages" in node_out:
                run["messages"] = [
                    m.content if hasattr(m, "content") else str(m)
                    for m in node_out["messages"]
                ]

        run["status"]       = "completed"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:
        exc_name = type(exc).__name__
        if "GraphInterrupt" in exc_name or "interrupt" in str(exc).lower():
            run["status"] = "awaiting_approval"
            logger.info("Pipeline paused at human approval — thread %s", thread_id)
        else:
            run["status"] = "failed"
            run["error"]  = str(exc)
            logger.error("Pipeline failed for thread %s: %s", thread_id, exc)
    finally:
        save_run(thread_id, run)


async def _resume_pipeline(thread_id: str, approved: bool):
    """Background task — resumes a paused pipeline after human decision."""

    run = get_run(thread_id)
    run["status"]       = "running"
    run["current_agent"] = "remediation"
    save_run(thread_id, run)

    try:
        app_graph    = _build_graph()
        config       = {"configurable": {"thread_id": thread_id}}
        resume_state = {"human_approved": approved}

        for event in app_graph.stream(resume_state, config=config):
            node_name = list(event.keys())[0]
            node_out  = event[node_name] or {}
            run["current_agent"] = node_name

            if "remediation_status" in node_out:
                run["remediation_status"] = node_out["remediation_status"]
            if "incident_id" in node_out:
                run["incident_id"] = node_out["incident_id"]
            if "report" in node_out:
                run["report"] = node_out["report"]
            if "remediation_action" in node_out:
                run["remediation_action"] = node_out["remediation_action"]
            if "remediation_detail" in node_out:
                run["remediation_detail"] = node_out["remediation_detail"]
            if "diagnosis_json" in node_out:
                run["diagnosis_json"] = node_out["diagnosis_json"]
            if "retrain_prescription" in node_out:
                run["retrain_prescription"] = node_out["retrain_prescription"]
            if "drifted_features" in node_out:
                run["drifted_features"] = node_out["drifted_features"]
            if "similar_incidents" in node_out:
                run["similar_incidents"] = node_out["similar_incidents"]
            if "relevant_runbooks" in node_out:
                run["relevant_runbooks"] = node_out["relevant_runbooks"]
            if "notifications_sent" in node_out:
                run["notifications_sent"] = node_out["notifications_sent"]
            if "messages" in node_out:
                run["messages"] = [
                    m.content if hasattr(m, "content") else str(m)
                    for m in node_out["messages"]
                ]

        run["status"]       = "completed"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        run["human_approved"] = approved

    except Exception as exc:
        run["status"] = "failed"
        run["error"]  = str(exc)
        logger.error("Resume failed for thread %s: %s", thread_id, exc)
    finally:
        save_run(thread_id, run)

# ── pipeline run endpoints ────────────────────────────────────────────────────

@app.post("/runs", status_code=202)
async def trigger_run(req: RunRequest, background_tasks: BackgroundTasks):
    """Trigger a new monitoring pipeline run. Returns immediately with thread_id."""
    thread_id = str(uuid.uuid4())

    run_data = {
        "thread_id":           thread_id,
        "model_id":            req.model_id,
        "environment":         req.environment,
        "status":              "queued",
        "severity":            None,
        "current_agent":       None,
        "diagnosis":           None,
        "recommended_action":  None,
        "remediation_status":  None,
        "incident_id":         None,
        "report":              None,
        "human_approved":      None,
        "error":               None,
        "created_at":          datetime.now(timezone.utc).isoformat(),
        "created_at_ts":       time.time(),
        "started_at":          None,
        "completed_at":        None,
        "remediation_action":   None,
        "remediation_detail":   None,
        "diagnosis_json":       None,
        "retrain_prescription": None,
        "drifted_features":     [],
        "similar_incidents":    [],
        "relevant_runbooks":    [],
        "notifications_sent":   [],
        "messages":             [],
    }

    save_run(thread_id, run_data)

    background_tasks.add_task(
        _execute_pipeline, thread_id, req.model_id, req.environment
    )

    logger.info("Queued run %s for model=%s env=%s", thread_id, req.model_id, req.environment)
    return {"thread_id": thread_id, "status": "queued"}


@app.get("/runs")
async def list_runs():
    """List all runs sorted newest first."""
    return list_all_runs()


@app.get("/runs/{thread_id}")
async def get_run(thread_id: str):
    """Get the current state of a single pipeline run."""
    run = get_run(thread_id)
    if not run:
        raise HTTPException(404, f"Thread {thread_id} not found")
    return run


@app.post("/runs/{thread_id}/approve", status_code=202)
async def approve_run(thread_id: str, background_tasks: BackgroundTasks):
    """Approve a pipeline paused at the human approval checkpoint."""
    run = get_run(thread_id)
    if not run:
        raise HTTPException(404, f"Thread {thread_id} not found")
    if run["status"] != "awaiting_approval":
        raise HTTPException(400, f"Run is not awaiting approval — status: {run['status']}")

    background_tasks.add_task(_resume_pipeline, thread_id, True)
    return {"thread_id": thread_id, "status": "resuming", "approved": True}


@app.post("/runs/{thread_id}/reject", status_code=202)
async def reject_run(thread_id: str, background_tasks: BackgroundTasks):
    """Reject a pipeline paused at the human approval checkpoint."""
    run = get_run(thread_id)
    if not run:
        raise HTTPException(404, f"Thread {thread_id} not found")
    if run["status"] != "awaiting_approval":
        raise HTTPException(400, f"Run is not awaiting approval — status: {run['status']}")

    background_tasks.add_task(_resume_pipeline, thread_id, False)
    return {"thread_id": thread_id, "status": "resuming", "approved": False}

# ── metrics endpoint ──────────────────────────────────────────────────────────

@app.get("/metrics/current")
async def current_metrics(model_id: str = None):
    """
    Fetch live metrics from the model server via MCP.
    The model server logs a snapshot to MLflow tagged run_type=metrics_snapshot.
    """
    try:
        r = http.post(
            f"{MODEL_SERVER_URL}/mcp/call",
            json={"tool": "get_current_metrics", "params": {}},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Could not reach model server: {e}")

# ── incidents endpoint ────────────────────────────────────────────────────────

@app.get("/incidents")
async def list_incidents(limit: int = 50, severity: str = None):
    """
    Query ChromaDB for recent incidents.
    Optionally filter by severity: none | minor | major | critical
    """
    try:
        from mlops_agents.rag.store import RAGStore
        rag    = RAGStore()
        where  = {"severity": severity} if severity else None

        # use get() with a limit rather than a semantic query
        results = rag._incidents.get(
            where=where,
            include=["metadatas"],
            limit=limit,
        )
        metas = results.get("metadatas") or []
        return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)

    except Exception as e:
        raise HTTPException(500, f"ChromaDB query failed: {e}")

# ── drift endpoints ───────────────────────────────────────────────────────────

@app.post("/drift/inject")
async def inject_drift(req: DriftInjectRequest):
    """Forward a structured drift config to the model server."""
    try:
        r = http.post(
            f"{MODEL_SERVER_URL}/mcp/call",
            json={
                "tool": "inject_drift",
                "params": {
                    "type":          req.type,
                    "features":      req.features,
                    "magnitude":     req.magnitude,
                    "swap_features": req.swap_features,
                    "description":   req.description,
                },
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Drift injection failed: {e}")


@app.post("/drift/reset")
async def reset_drift():
    """Clear all active drift on the model server."""
    try:
        r = http.post(
            f"{MODEL_SERVER_URL}/mcp/call",
            json={"tool": "inject_drift", "params": {"type": "none"}},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Drift reset failed: {e}")


@app.get("/drift/status")
async def drift_status():
    """Get current drift config from the model server."""
    try:
        r = http.post(
            f"{MODEL_SERVER_URL}/mcp/call",
            json={"tool": "get_drift_status", "params": {}},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Could not reach model server: {e}")

# ── health endpoint ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check reachability of all dependent services."""
    results = {}

    # Ollama
    try:
        r = http.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        results["ollama"] = r.ok
    except Exception:
        results["ollama"] = False

    # Model server
    try:
        r = http.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        results["model_server"] = r.ok
    except Exception:
        results["model_server"] = False

    # ChromaDB
    try:
        from mlops_agents.rag.store import RAGStore
        rag = RAGStore()
        results["chromadb"] = rag._incidents.count() >= 0
    except Exception:
        results["chromadb"] = False

    # MLflow (via model server watcher — if model server is up, MLflow auth is ok)
    results["mlflow"] = results["model_server"]

    overall = all(results.values())
    return {
        "status":    "ok" if overall else "degraded",
        "services":  results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )