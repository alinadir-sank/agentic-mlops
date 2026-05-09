"""
rag/init_collections.py

ChromaDB collection schema initialisation script.
Run once before first use, or safely re-run (idempotent).

Usage:
    python -m rag.init_collections
    # or directly:
    python rag/init_collections.py
"""

import logging
import os
import sys
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("rag.init_collections")

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Collection definitions
# ---------------------------------------------------------------------------

COLLECTIONS: list[dict] = [
    {
        "name": "incidents",
        "description": (
            "Full incident records — one document per resolved incident. "
            "Each document contains a serialised JSON payload with metrics, "
            "diagnosis, remediation outcome, and a human-readable report. "
            "Used by the Diagnosis Agent to retrieve the top-k most similar "
            "past incidents for in-context learning."
        ),
        "metadata_schema": {
            # Fields stored in the ChromaDB metadata dict for each document.
            # ChromaDB metadata values must be str | int | float | bool.
            "incident_id":          "str   — UUID v4",
            "model_id":             "str   — deployed model identifier",
            "model_version":        "str   — semver or tag",
            "environment":          "str   — production | staging | canary",
            "severity":             "str   — none | minor | major | critical",
            "recommended_action":   "str   — retrain | rollback | scale | investigate",
            "remediation_status":   "str   — success | failed | skipped",
            "human_approved":       "bool  — whether human approval was required",
            "accuracy":             "float — model accuracy at incident time",
            "drift_score":          "float — data drift score at incident time",
            "latency_p99_ms":       "float — p99 latency in ms at incident time",
            "error_rate":           "float — error rate fraction (0–1)",
            "created_at":           "str   — ISO-8601 UTC timestamp",
            "resolved_at":          "str   — ISO-8601 UTC timestamp or empty string",
        },
    },
    {
        "name": "metrics_history",
        "description": (
            "Lightweight metric snapshots saved after every monitor cycle — "
            "including healthy runs. Used by the Monitor Agent to retrieve "
            "recent trend windows for anomaly context, and by the Reporting "
            "Agent to generate aggregate statistics."
        ),
        "metadata_schema": {
            "snapshot_id":      "str   — UUID v4",
            "model_id":         "str   — deployed model identifier",
            "model_version":    "str   — semver or tag",
            "environment":      "str   — production | staging | canary",
            "severity":         "str   — none | minor | major | critical",
            "accuracy":         "float — model accuracy",
            "drift_score":      "float — data drift score",
            "latency_p99_ms":   "float — p99 latency in ms",
            "error_rate":       "float — error rate fraction (0–1)",
            "prediction_count": "int   — number of predictions in window",
            "sampled_at":       "str   — ISO-8601 UTC timestamp",
        },
    },
    {
        "name": "runbooks",
        "description": (
            "Institutional knowledge base — runbooks, post-mortems, "
            "remediation playbooks, and engineering notes ingested "
            "offline. Queried by the Diagnosis Agent to ground its "
            "recommendations in org-specific procedures."
        ),
        "metadata_schema": {
            "doc_id":       "str — UUID v4 or slug",
            "title":        "str — human-readable document title",
            "doc_type":     "str — runbook | post_mortem | playbook | note",
            "tags":         "str — comma-separated keyword tags",
            "author":       "str — author name or team slug",
            "source_url":   "str — Confluence / GitHub URL or empty string",
            "created_at":   "str — ISO-8601 date",
            "updated_at":   "str — ISO-8601 date",
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.ClientAPI:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./rag_data")
    host = os.getenv("CHROMA_HOST", "")
    port = int(os.getenv("CHROMA_PORT", "8000"))

    if host:
        logger.info("Connecting to remote ChromaDB at %s:%s", host, port)
        return chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )

    logger.info("Using local PersistentClient at '%s'", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_embedding_function() -> embedding_functions.EmbeddingFunction:
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info("Embedding model: %s  (Ollama @ %s)", embed_model, ollama_url)
    return embedding_functions.OllamaEmbeddingFunction(
        model_name=embed_model,
        url=f"{ollama_url}/api/embeddings",
    )


# ---------------------------------------------------------------------------
# Main init
# ---------------------------------------------------------------------------

def init_collections(reset: bool = False) -> dict[str, chromadb.Collection]:
    """
    Create (or verify) all ChromaDB collections.

    Args:
        reset: If True, delete and recreate every collection.
               USE WITH CAUTION — destroys all stored data.

    Returns:
        Dict mapping collection name → Collection object.
    """
    client = _get_chroma_client()
    embed_fn = _get_embedding_function()
    created: dict[str, chromadb.Collection] = {}

    for spec in COLLECTIONS:
        name = spec["name"]

        if reset:
            try:
                client.delete_collection(name)
                logger.warning("Deleted collection '%s' (reset=True)", name)
            except Exception:
                pass  # did not exist yet

        try:
            col = client.get_collection(name=name, embedding_function=embed_fn)
            count = col.count()
            logger.info(
                "Collection '%s' already exists — %d documents. Skipping creation.",
                name,
                count,
            )
        except Exception:
            col = client.create_collection(
                name=name,
                embedding_function=embed_fn,
                metadata={
                    "description": spec["description"],
                    "schema_version": "1.0",
                    "created_at": datetime.utcnow().isoformat(),
                    "hnsw:space": "cosine",          # cosine similarity for RAG
                },
            )
            logger.info("Created collection '%s'.", name)

        created[name] = col

    logger.info("Initialisation complete. Collections: %s", list(created.keys()))
    return created


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialise ChromaDB collections for the MLOps agent system."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all collections. Destroys all stored data.",
    )
    args = parser.parse_args()

    if args.reset:
        confirm = input(
            "\n⚠️  --reset will DELETE all collection data. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    init_collections(reset=args.reset)
    print("\n✅  ChromaDB collections ready.")
