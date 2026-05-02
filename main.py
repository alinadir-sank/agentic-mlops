"""
main.py

Production entry point for the MLOps Multi-Agent pipeline.

Usage:
    # Run a single pipeline cycle for a given model
    python main.py --model-id fraud-classifier-v2 --environment production

    # Run with a specific thread ID (for resuming interrupted graphs)
    python main.py --model-id fraud-classifier-v2 --thread-id abc-123

    # Resume a graph that was interrupted for human approval
    python main.py --resume --thread-id abc-123 --approve

Environment:
    See .env.example for all required/optional variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")


def _check_connectivity() -> None:
    """Quick connectivity checks before running the pipeline."""
    import requests

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info("Ollama reachable. Available models: %s", models)
    except Exception as exc:
        logger.error("Cannot reach Ollama at %s: %s", ollama_url, exc)
        sys.exit(1)


def run_pipeline(
    model_id: str,
    environment: str,
    thread_id: str | None = None,
) -> dict:
    """
    Execute one full pipeline cycle.

    Args:
        model_id:    The deployed model to monitor.
        environment: Deployment environment (production/staging/canary).
        thread_id:   LangGraph thread ID for checkpointing. Auto-generated if None.

    Returns:
        Final AgentState dict.
    """
    from mlops_agents.rag.store import RAGStore
    from mlops_agents.graph.workflow import build_graph

    thread_id = thread_id or str(uuid.uuid4())
    logger.info(
        "Starting pipeline | model=%s env=%s thread=%s",
        model_id, environment, thread_id,
    )

    rag = RAGStore()
    app = build_graph(rag=rag)

    initial_state = {
        "model_id": model_id,
        "environment": environment,
        "messages": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = app.invoke(initial_state, config=config)
    except Exception as exc:
        # Check if this is a GraphInterrupt (human approval checkpoint)
        if "GraphInterrupt" in type(exc).__name__ or "interrupt" in str(exc).lower():
            logger.info(
                "Pipeline paused at human approval checkpoint. "
                "Thread ID: %s  Resume with: python main.py --resume --thread-id %s --approve",
                thread_id, thread_id,
            )
            return {"status": "interrupted", "thread_id": thread_id}
        raise

    severity = final_state.get("severity", "none")
    incident_id = final_state.get("incident_id")

    logger.info(
        "Pipeline complete | severity=%s remediation=%s incident_id=%s",
        severity,
        final_state.get("remediation_status", "N/A"),
        incident_id or "N/A",
    )

    if final_state.get("report"):
        print("\n" + "=" * 70)
        print(final_state["report"])
        print("=" * 70 + "\n")

    return final_state


def resume_pipeline(thread_id: str, approve: bool) -> dict:
    """
    Resume a pipeline that was interrupted at the human approval checkpoint.

    Args:
        thread_id: The thread ID of the interrupted run.
        approve:   True to approve remediation, False to reject.
    """
    from mlops_agents.rag.store import RAGStore
    from mlops_agents.graph.workflow import build_graph

    logger.info(
        "Resuming pipeline | thread=%s approved=%s", thread_id, approve
    )

    rag = RAGStore()
    app = build_graph(rag=rag)

    config = {"configurable": {"thread_id": thread_id}}
    resume_state = {"human_approved": approve}

    final_state = app.invoke(resume_state, config=config)

    logger.info(
        "Resumed pipeline complete | remediation=%s incident_id=%s",
        final_state.get("remediation_status", "N/A"),
        final_state.get("incident_id", "N/A"),
    )

    if final_state.get("report"):
        print("\n" + "=" * 70)
        print(final_state["report"])
        print("=" * 70 + "\n")

    return final_state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLOps Multi-Agent Monitoring & Remediation Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the monitoring pipeline")
    run_parser.add_argument(
        "--model-id",
        required=False,
        default=os.getenv("DEFAULT_MODEL_ID", ""),
        help="Model ID to monitor (or set DEFAULT_MODEL_ID env var)",
    )
    run_parser.add_argument(
        "--environment",
        default=os.getenv("DEFAULT_ENVIRONMENT", "production"),
        choices=["production", "staging", "canary"],
    )
    run_parser.add_argument("--thread-id", default=None)
    run_parser.add_argument(
        "--skip-connectivity-check", action="store_true"
    )

    # Resume command (after human approval)
    resume_parser = subparsers.add_parser(
        "resume", help="Resume an interrupted pipeline"
    )
    resume_parser.add_argument("--thread-id", required=True)
    resume_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the remediation action",
    )
    resume_parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject the remediation action",
    )

    # Init command
    subparsers.add_parser("init", help="Initialise ChromaDB collections")

    args = parser.parse_args()

    if args.command == "init":
        from mlops_agents.rag.init_collections import init_collections
        init_collections()
        print("✅  ChromaDB collections initialised.")
        return

    if args.command == "resume":
        if not args.approve and not args.reject:
            parser.error("Specify --approve or --reject when resuming.")
        resume_pipeline(thread_id=args.thread_id, approve=args.approve)
        return

    # Default: run
    if not args.model_id:
        parser.error(
            "--model-id is required (or set DEFAULT_MODEL_ID environment variable)."
        )

    if not getattr(args, "skip_connectivity_check", False):
        _check_connectivity()

    run_pipeline(
        model_id=args.model_id,
        environment=args.environment,
        thread_id=args.thread_id,
    )


if __name__ == "__main__":
    main()
