"""Ingest all files from data/raw into the vector store."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.preprocessor import preprocess
from app.observability.logger import get_logger
from app.retrieval import vector_store

log = get_logger("ingest")

RAW_DIR = Path("data/raw")


def ingest_all() -> None:
    if not RAW_DIR.exists():
        log.error("raw_dir_missing", path=str(RAW_DIR))
        print(f"No data found at {RAW_DIR}. Run 'python scripts/scrape_docs.py' first.")
        return

    files = sorted(RAW_DIR.iterdir())
    if not files:
        log.warning("no_files_found", path=str(RAW_DIR))
        print("No files to ingest.")
        return

    all_chunks = []
    for fpath in files:
        if fpath.is_dir():
            continue
        log.info("loading", file=str(fpath))
        try:
            docs = load_documents(str(fpath))
            docs = preprocess(docs)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            log.info("chunked", file=str(fpath), n_chunks=len(chunks))
        except Exception as exc:
            log.error("load_failed", file=str(fpath), error=str(exc))

    if all_chunks:
        log.info("embedding_start", total_chunks=len(all_chunks))
        vector_store.add_documents(all_chunks)
        log.info("ingestion_complete", total_chunks=len(all_chunks))
        print(f"Ingested {len(all_chunks)} chunks from {len(files)} files.")
    else:
        print("No chunks produced.")


if __name__ == "__main__":
    ingest_all()
