.PHONY: install serve ui ingest test eval lint clean

install:
	pip install -e ".[dev]"

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

ingest:
	python scripts/ingest.py

scrape:
	python scripts/scrape_docs.py

test:
	pytest tests/ -v --tb=short

eval:
	python scripts/run_eval.py

lint:
	ruff check app/ tests/ scripts/ ui/
	ruff format --check app/ tests/ scripts/ ui/

format:
	ruff check --fix app/ tests/ scripts/ ui/
	ruff format app/ tests/ scripts/ ui/

clean:
	rm -rf data/processed/chroma_db
	rm -rf eval/results/*
	rm -rf .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
