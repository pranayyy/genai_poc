# GenAI FAQ System

A production-ready **Retrieval-Augmented Generation (RAG)** FAQ system built with LangGraph, Groq (free LLM), local HuggingFace embeddings, ChromaDB, FastAPI, and Streamlit.

---

## What This Project Does

This system answers natural-language questions by:
1. Searching a local vector database of ingested documents (ChromaDB)
2. Re-ranking the top retrieved chunks using a cross-encoder model
3. Generating a cited answer via a Large Language Model (Groq / OpenAI)
4. Applying input and output guardrails (injection detection, PII scan, faithfulness check)
5. Returning the answer with source citations, confidence score, and a full pipeline trace

Everything runs **locally with no mandatory paid API** — Groq provides a free tier (14,400 req/day), and all embeddings run on-device using HuggingFace sentence-transformers.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph `StateGraph` (7-node pipeline) |
| LLM | Groq `llama-3.1-8b-instant` (free) or OpenAI GPT-4o |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector Store | ChromaDB (persistent, local) |
| Backend API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Observability | structlog (JSON) + per-request trace IDs |
| Evaluation | RAGAS metrics + custom golden Q/A dataset |
| Python | 3.11 (required) |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                        │
│                                                             │
│  ┌────────────┐   ┌──────────┐   ┌─────────┐              │
│  │ Input Guard│──▶│ Retrieve │──▶│ Rerank  │              │
│  │            │   │(ChromaDB)│   │(cross-  │              │
│  │ • injection│   │          │   │ encoder)│              │
│  │ • off-topic│   │ top-20   │   │ top-5   │              │
│  └────────────┘   └──────────┘   └────┬────┘              │
│       │ blocked                        │                    │
│       ▼                                ▼                    │
│  blocked_response            ┌──────────────────┐          │
│                              │    Generate       │          │
│                              │  (Groq / OpenAI) │          │
│                              └────────┬─────────┘          │
│                                       │                    │
│                              ┌────────▼─────────┐          │
│                              │  Output Guard    │          │
│                              │  • PII scan      │          │
│                              │  • faithfulness  │          │
│                              └────────┬─────────┘          │
└───────────────────────────────────────┼────────────────────┘
                                        │
                                        ▼
                              Answer + Sources + Trace
```

All state is passed through a typed `PipelineState` dict. Trace context uses Python `contextvars` (not state keys) because LangGraph strips undeclared TypedDict fields between nodes.

---

## New Joiner Setup Guide

### Prerequisites

- Python **3.11** (3.12/3.13 have dependency conflicts — use 3.11 exactly)
- Git
- A free [Groq API key](https://console.groq.com) (takes ~1 minute to create)

### Step-by-step setup

**1. Clone and enter the project**
```bash
git clone <repo-url>
cd genai_poc
```

**2. Create a virtual environment with Python 3.11**
```bash
py -3.11 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**3. Install all dependencies**
```bash
pip install -e ".[dev]"
```

**4. Configure environment variables**
```bash
cp .env.example .env
```
Open `.env` and fill in:
```
LLM_PROVIDER=groq
GROQ_API_KEY=<your key from console.groq.com>
GROQ_MODEL=llama-3.1-8b-instant
```
> Leave `OPENAI_API_KEY` blank unless you want to switch to GPT-4o.

**5. Scrape sample documentation into `data/raw/`**
```bash
python scripts/scrape_docs.py
```
This downloads LangChain and Python docs pages as HTML files.

**6. Ingest documents into ChromaDB**
```bash
python scripts/ingest.py
```
This chunks, embeds (using local HuggingFace model), and stores ~1,000 chunks in ChromaDB at `data/processed/chroma_db/`.  
The embedding model (`all-MiniLM-L6-v2`, ~91 MB) auto-downloads from HuggingFace on first run.

**7. Start the API server**
```bash
uvicorn app.main:app --reload --port 8000
```

**8. Start the Streamlit UI** (new terminal)
```bash
streamlit run ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser and ask questions like:
- *"What is LangChain?"*
- *"How do I use a vector store?"*
- *"What are Python decorators?"*

### Run tests
```bash
pytest tests/ -v
```
Expected: **29 passed, 3 skipped** (the 3 skipped require a live API key).

---

## Project Structure

```
genai_poc/
├── app/
│   ├── config.py            # All settings via pydantic-settings + .env
│   ├── llm_factory.py       # Provider-agnostic LLM + embeddings factory
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   └── routes.py        # REST endpoints
│   ├── ingestion/
│   │   ├── loaders.py       # File/URL document loaders
│   │   ├── preprocessor.py  # Text cleaning
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter
│   │   └── embedder.py      # Delegates to llm_factory.get_embeddings()
│   ├── retrieval/
│   │   ├── vector_store.py  # ChromaDB wrapper
│   │   ├── retriever.py     # Dense retrieval (top-K)
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── generation/
│   │   ├── prompts.py       # System + user prompt templates
│   │   └── generator.py     # LangChain messages → LLM → cited answer
│   ├── guardrails/
│   │   ├── input_guard.py   # Injection detection, off-topic filter
│   │   └── output_guard.py  # PII scan, faithfulness check
│   ├── pipeline/
│   │   └── graph.py         # LangGraph StateGraph (7 nodes)
│   ├── observability/
│   │   ├── logger.py        # structlog JSON logger
│   │   └── tracing.py       # TraceContext + stage timing
│   └── evaluation/
│       ├── evaluator.py     # RAGAS metrics runner
│       └── datasets.py      # Golden Q/A dataset loader
├── ui/
│   └── app.py               # Streamlit chat + admin + eval tabs
├── scripts/
│   ├── scrape_docs.py       # Scrape LangChain/Python docs to data/raw/
│   ├── ingest.py            # Chunk + embed + load into ChromaDB
│   └── run_eval.py          # Run evaluation against golden dataset
├── tests/                   # Pytest suite (29 tests)
├── eval/
│   └── golden_qa.json       # 20 hand-crafted Q/A pairs for evaluation
├── data/
│   ├── raw/                 # Scraped HTML files (git-ignored)
│   └── processed/chroma_db/ # Persisted ChromaDB (git-ignored)
├── .env                     # Your local secrets (git-ignored)
├── .env.example             # Template — copy to .env
├── pyproject.toml           # Dependencies + build config
└── Makefile                 # Shortcut commands
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Ask a question — runs full RAG pipeline |
| `POST` | `/api/ingest/file` | Upload a PDF/txt/HTML file to ingest |
| `POST` | `/api/ingest/url` | Ingest content from a public URL |
| `GET` | `/api/health` | Health check + ChromaDB collection stats |
| `POST` | `/api/evaluate` | Run evaluation against golden Q/A dataset |

Example query:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a vector store?"}'
```

---

## Switching LLM Provider

Edit `.env` to toggle between Groq (free) and OpenAI:

```bash
# Use Groq (free, default)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Use OpenAI (paid)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

Restart uvicorn after changing `.env` (settings are loaded at startup).

---

## Evaluation

```bash
python scripts/run_eval.py
```

Runs the pipeline against 20 golden Q/A pairs in `eval/golden_qa.json` and reports:

| Metric | Description |
|--------|-------------|
| Faithfulness | Answer claims supported by retrieved context |
| Answer Relevance | Answer addresses the question |
| Context Precision | Retrieved chunks are on-topic |
| Context Recall | Relevant information was retrieved |
| Citation Accuracy | Sources correctly identified |
| Refusal Accuracy | Off-topic questions correctly declined |

---

## Makefile Shortcuts

```bash
make install     # pip install -e ".[dev]"
make scrape      # python scripts/scrape_docs.py
make ingest      # python scripts/ingest.py
make serve       # uvicorn app.main:app --reload --port 8000
make ui          # streamlit run ui/app.py --server.port 8501
make test        # pytest tests/ -v
make eval        # python scripts/run_eval.py
```

---

## Known Issues / Notes

- **`embeddings.position_ids UNEXPECTED`** warning from HuggingFace on model load — harmless, can be ignored
- **HuggingFaceEmbeddings deprecation warning** — upgrade path: `pip install langchain-huggingface` and update the import in `app/llm_factory.py`
- **First ingest is slow** — the embedding model (~91 MB) downloads once then caches locally
- **Groq model names** — Groq periodically decommissions models; check [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) if you get a `model_decommissioned` error and update `GROQ_MODEL` in `.env`
