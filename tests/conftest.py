"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield Path(d)


@pytest.fixture
def sample_pdf_path(tmp_dir: Path) -> Path:
    """Create a minimal PDF for testing (uses pypdf to write)."""
    # We'll just test with a text file renamed — loaders test separately
    p = tmp_dir / "sample.txt"
    p.write_text("This is a test document with sample content for testing.", encoding="utf-8")
    return p


@pytest.fixture
def sample_json_path(tmp_dir: Path) -> Path:
    p = tmp_dir / "data.json"
    p.write_text(
        '[{"name": "Alice", "role": "Engineer"}, {"name": "Bob", "role": "Designer"}]',
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_csv_path(tmp_dir: Path) -> Path:
    p = tmp_dir / "data.csv"
    p.write_text("name,role\nAlice,Engineer\nBob,Designer\n", encoding="utf-8")
    return p
