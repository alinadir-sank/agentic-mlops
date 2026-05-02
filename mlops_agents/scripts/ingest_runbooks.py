"""
scripts/ingest_runbooks.py

Offline utility to bulk-ingest runbooks, post-mortems, and playbooks
into the ChromaDB runbooks collection.

Supports:
  - Plain .txt and .md files from a directory
  - A JSON manifest file (see examples below)

Usage:
    # Ingest all .md/.txt files from a directory
    python scripts/ingest_runbooks.py --dir ./docs/runbooks

    # Ingest from a JSON manifest
    python scripts/ingest_runbooks.py --manifest ./docs/runbooks.json

    # Ingest a single file
    python scripts/ingest_runbooks.py --file ./docs/retrain-runbook.md \\
        --title "Retraining Runbook" --doc-type runbook --tags "retrain,drift"

JSON manifest format:
[
  {
    "title": "Model Retraining Runbook",
    "doc_type": "runbook",
    "tags": "retrain,drift,accuracy",
    "author": "ml-platform-team",
    "source_url": "https://confluence.example.com/display/ML/Retrain",
    "file": "./docs/retrain-runbook.md"
  },
  {
    "title": "2024-03 Production Outage Post-Mortem",
    "doc_type": "post_mortem",
    "tags": "outage,latency,rollback",
    "author": "oncall",
    "content": "Inline content can also be placed directly here..."
  }
]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from mlops_agents.rag.store import RAGStore

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_runbooks")


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def ingest_from_directory(rag: RAGStore, directory: str, doc_type: str = "runbook") -> int:
    count = 0
    for filepath in Path(directory).glob("**/*"):
        if filepath.suffix not in (".md", ".txt"):
            continue
        content = _read_file(str(filepath))
        doc_id = rag.ingest_runbook(
            {
                "title": filepath.stem.replace("-", " ").replace("_", " ").title(),
                "content": content,
                "doc_type": doc_type,
                "tags": "",
                "source_url": str(filepath),
            }
        )
        logger.info("Ingested '%s' → %s", filepath.name, doc_id)
        count += 1
    return count


def ingest_from_manifest(rag: RAGStore, manifest_path: str) -> int:
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)

    count = 0
    for entry in entries:
        if "file" in entry:
            content = _read_file(entry["file"])
        elif "content" in entry:
            content = entry["content"]
        else:
            logger.warning("Entry '%s' has no 'file' or 'content' key — skipping.", entry.get("title"))
            continue

        doc_id = rag.ingest_runbook(
            {
                "title": entry.get("title", "Untitled"),
                "content": content,
                "doc_type": entry.get("doc_type", "note"),
                "tags": entry.get("tags", ""),
                "author": entry.get("author", ""),
                "source_url": entry.get("source_url", ""),
                "doc_id": entry.get("doc_id"),
            }
        )
        logger.info("Ingested '%s' → %s", entry.get("title"), doc_id)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest runbooks into ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory of .md/.txt files to ingest")
    group.add_argument("--manifest", help="JSON manifest file")
    group.add_argument("--file", help="Single file to ingest")

    parser.add_argument("--title", default="", help="Title for --file mode")
    parser.add_argument(
        "--doc-type",
        default="runbook",
        choices=["runbook", "post_mortem", "playbook", "note"],
    )
    parser.add_argument("--tags", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--source-url", default="")

    args = parser.parse_args()
    rag = RAGStore()

    if args.dir:
        count = ingest_from_directory(rag, args.dir, doc_type=args.doc_type)
    elif args.manifest:
        count = ingest_from_manifest(rag, args.manifest)
    else:
        content = _read_file(args.file)
        doc_id = rag.ingest_runbook(
            {
                "title": args.title or Path(args.file).stem,
                "content": content,
                "doc_type": args.doc_type,
                "tags": args.tags,
                "author": args.author,
                "source_url": args.source_url or args.file,
            }
        )
        count = 1
        logger.info("Ingested '%s' → %s", args.title or args.file, doc_id)

    print(f"\n✅  Ingested {count} document(s) into the runbooks collection.")


if __name__ == "__main__":
    main()
