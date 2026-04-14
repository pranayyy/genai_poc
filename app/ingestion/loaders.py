"""Document loaders for various source types."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document


def load_documents(source: str, source_type: str | None = None) -> list[Document]:
    """Load documents from *source* (file path or URL).

    Parameters
    ----------
    source:
        A file path or URL.
    source_type:
        One of ``"pdf"``, ``"html"``, ``"url"``, ``"csv"``, ``"json"``.
        If *None* it is inferred from the file extension / scheme.
    """
    if source_type is None:
        source_type = _infer_type(source)

    loader_fn = _LOADERS.get(source_type)
    if loader_fn is None:
        raise ValueError(f"Unsupported source_type={source_type!r}")

    docs = loader_fn(source)

    # Attach common metadata
    now = datetime.now(timezone.utc).isoformat()
    for doc in docs:
        doc.metadata.setdefault("source", source)
        doc.metadata.setdefault("source_type", source_type)
        doc.metadata["ingested_at"] = now

    return docs


# ── Private loaders ─────────────────────────────────────────


def _load_pdf(source: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(source).load()


def _load_html(source: str) -> list[Document]:
    from langchain_community.document_loaders import BSHTMLLoader

    return BSHTMLLoader(source, open_encoding="utf-8").load()


def _load_url(source: str) -> list[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    return WebBaseLoader(source).load()


def _load_csv(source: str) -> list[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader

    return CSVLoader(source).load()


def _load_json(source: str) -> list[Document]:
    """Load a JSON file.  Supports a list of objects or a single object."""
    path = Path(source)
    raw = json.loads(path.read_text(encoding="utf-8"))

    items: Sequence[dict] = raw if isinstance(raw, list) else [raw]
    docs: list[Document] = []
    for idx, item in enumerate(items):
        text = json.dumps(item, ensure_ascii=False, indent=2)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": source, "row_index": idx},
            )
        )
    return docs


_LOADERS = {
    "pdf": _load_pdf,
    "html": _load_html,
    "url": _load_url,
    "csv": _load_csv,
    "json": _load_json,
}


def _infer_type(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return "url"
    suffix = Path(source).suffix.lower().lstrip(".")
    mapping = {"pdf": "pdf", "html": "html", "htm": "html", "csv": "csv", "json": "json"}
    if suffix in mapping:
        return mapping[suffix]
    raise ValueError(f"Cannot infer source_type from {source!r}; pass source_type explicitly.")
