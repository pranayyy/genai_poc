"""Scrape LangChain and Python documentation for the demo dataset."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from bs4 import BeautifulSoup

from app.observability.logger import get_logger

log = get_logger("scrape")

RAW_DIR = Path("data/raw")

# ── URLs to scrape ──────────────────────────────────────────
LANGCHAIN_URLS = [
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/chains/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/output_parsers/",
    "https://python.langchain.com/docs/concepts/document_loaders/",
    "https://python.langchain.com/docs/concepts/chat_models/",
]

PYTHON_URLS = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/modules.html",
    "https://docs.python.org/3/tutorial/inputoutput.html",
    "https://docs.python.org/3/tutorial/errors.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/stdlib.html",
    "https://docs.python.org/3/tutorial/venv.html",
]


def scrape_url(url: str) -> str | None:
    """Fetch a URL and extract the main text content."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("scrape_failed", url=url, error=str(exc))
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove script / style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to find main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""

    return text if len(text) > 100 else None


def scrape_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_urls = [("langchain", u) for u in LANGCHAIN_URLS] + [("python", u) for u in PYTHON_URLS]

    total = 0
    for category, url in all_urls:
        log.info("scraping", url=url)
        text = scrape_url(url)
        if text is None:
            continue

        # Create a filename from the URL
        slug = url.rstrip("/").split("/")[-1] or "index"
        filename = f"{category}_{slug}.html"
        out_path = RAW_DIR / filename
        out_path.write_text(text, encoding="utf-8")
        total += 1
        log.info("scraped", file=str(out_path), chars=len(text))

    log.info("scrape_complete", total_files=total)


if __name__ == "__main__":
    scrape_all()
