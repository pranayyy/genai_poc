"""Streamlit FAQ chat interface."""

from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="GenAI FAQ", page_icon="🔍", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.title("GenAI FAQ System")
    tab = st.radio("Navigation", ["💬 Chat", "📂 Admin", "📊 Evaluation"], label_visibility="collapsed")

    st.divider()
    # Collection stats
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5).json()
        st.metric("Documents indexed", health.get("document_count", 0))
    except Exception:
        st.warning("Backend not reachable")


# ── Chat ────────────────────────────────────────────────────
if tab == "💬 Chat":
    st.header("Ask a question")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "traces" not in st.session_state:
        st.session_state.traces = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/query",
                        json={"query": prompt},
                        timeout=60,
                    ).json()

                    answer = resp.get("answer", "No answer returned.")
                    sources = resp.get("sources", [])
                    warnings = resp.get("warnings", [])
                    trace = resp.get("trace", {})
                    confidence = resp.get("confidence", 0)

                    # Show warnings
                    if warnings:
                        for w in warnings:
                            st.warning(w)

                    # Show answer
                    st.markdown(answer)

                    # Confidence badge
                    if confidence < 0.5:
                        st.caption("⚠️ Low confidence answer")

                    # Sources expander
                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)})"):
                            for s in sources:
                                page_info = f", page {s['page']}" if s.get("page") else ""
                                score = f"{s.get('relevance_score', 0):.3f}"
                                st.markdown(f"- **{s['document']}**{page_info} (score: {score})")

                    # Trace expander
                    if trace and trace.get("stages"):
                        with st.expander("🔍 Pipeline trace"):
                            st.caption(f"Trace ID: `{trace.get('trace_id', 'N/A')}`")
                            st.caption(f"Total: {trace.get('total_ms', 0):.0f} ms")
                            for stg in trace["stages"]:
                                col1, col2 = st.columns([3, 1])
                                col1.text(stg["name"])
                                col2.text(f"{stg['duration_ms']:.0f} ms")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.traces.append(trace)

                except Exception as exc:
                    st.error(f"Error: {exc}")


# ── Admin ───────────────────────────────────────────────────
elif tab == "📂 Admin":
    st.header("Data Ingestion")

    ingest_tab1, ingest_tab2 = st.tabs(["Upload File", "Ingest URL"])

    with ingest_tab1:
        uploaded = st.file_uploader(
            "Upload a document (PDF, HTML, CSV, JSON)",
            type=["pdf", "html", "htm", "csv", "json"],
        )
        if uploaded and st.button("Ingest file", key="ingest_file_btn"):
            with st.spinner("Ingesting…"):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = httpx.post(f"{API_BASE}/ingest/file", files=files, timeout=120).json()
                    st.success(f"{resp.get('message')} — {resp.get('chunks_added', 0)} chunks added")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with ingest_tab2:
        url = st.text_input("Document URL")
        if url and st.button("Ingest URL", key="ingest_url_btn"):
            with st.spinner("Ingesting…"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/ingest/url", json={"url": url}, timeout=120
                    ).json()
                    st.success(f"{resp.get('message')} — {resp.get('chunks_added', 0)} chunks added")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Collection Info")
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5).json()
        st.json(health)
    except Exception:
        st.warning("Backend not reachable")


# ── Evaluation ──────────────────────────────────────────────
elif tab == "📊 Evaluation":
    st.header("Evaluation Dashboard")

    if st.button("Run evaluation"):
        with st.spinner("Running evaluation suite…"):
            try:
                resp = httpx.post(f"{API_BASE}/evaluate", timeout=1200).json()
                metrics = resp.get("metrics", {})

                st.subheader("Metrics")
                cols = st.columns(len(metrics) if metrics else 1)
                for col, (name, value) in zip(cols, metrics.items()):
                    col.metric(name, f"{value:.3f}")

                if resp.get("details"):
                    with st.expander("Detailed results"):
                        st.json(resp["details"])
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    st.info("Evaluation runs the golden Q/A dataset through the pipeline and measures faithfulness, relevance, context precision, and context recall using RAGAS.")
