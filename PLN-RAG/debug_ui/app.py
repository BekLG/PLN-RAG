import json
import httpx
import streamlit as st

st.set_page_config(page_title="PLN-RAG Debug UI", layout="wide")

# Initialize session state for ingest results
if "ingest_results" not in st.session_state:
    st.session_state.ingest_results = None
if "query_results" not in st.session_state:
    st.session_state.query_results = None
if "qdrant_results" not in st.session_state:
    st.session_state.qdrant_results = None

st.title("PLN-RAG Debug UI")

col1, col2 = st.columns([3, 1])
with col1:
    api_base = st.text_input("API base URL", value="http://localhost:8000")
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("Clear Database", type="secondary"):
        try:
            resp = httpx.delete(
                f"{api_base}/reset",
                timeout=30,
            )
            resp.raise_for_status()
            st.success("Database cleared successfully!")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to clear database: {exc}")

def _split_texts(raw_text: str) -> list[str]:
    parts = [part.strip() for part in raw_text.split("\n\n") if part.strip()]
    return parts


def _all_pln_statements(payload: dict) -> list[str]:
    statements: list[str] = []
    seen: set[str] = set()
    for item in payload.get("results", []):
        for chunk in item.get("chunks", []):
            for stmt in chunk.get("pln_canonicalized", []):
                if stmt in seen:
                    continue
                seen.add(stmt)
                statements.append(stmt)
    return statements

st.header("Ingest (Debug)")
raw_texts = st.text_area(
    "Texts (separate with a blank line)",
    height=180,
    placeholder="Paste one or more paragraphs here...",
    key="ingest_text_area"
)

if st.button("Run debug ingest", key="run_ingest_btn"):
    texts = _split_texts(raw_texts)
    if not texts:
        st.warning("Please provide at least one text block.")
    else:
        try:
            resp = httpx.post(
                f"{api_base}/debug/ingest",
                json={"texts": texts},
                timeout=420,  # Long LangExtract/Gemini calls can exceed 3 minutes.
            )
            resp.raise_for_status()
            payload = resp.json()
            st.success(f"Processed {payload.get('processed_count', 0)} texts")
            # Store in session state
            st.session_state.ingest_results = payload
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.session_state.ingest_results = None

# Display stored ingest results
if st.session_state.ingest_results is not None:
    payload = st.session_state.ingest_results
    for item in payload.get("results", []):
        label = f"Text: {item.get('text', '')[:80]}"
        with st.expander(label, expanded=False):
            if item.get("status") != "success":
                st.error(item.get("error", "Unknown error"))
                continue
            for idx, chunk in enumerate(item.get("chunks", []), start=1):
                st.subheader(f"Chunk {idx}")
                st.code(chunk.get("chunk", ""))

                left, right = st.columns(2)
                with left:
                    st.markdown("**LangExtract result**")
                    st.json(chunk.get("langextract_postprocessed", {}), expanded=False)
                with right:
                    st.markdown("**Final PLN statements**")
                    st.code(
                        "\n".join(chunk.get("pln_canonicalized", [])) or "(none)",
                        language="lisp",
                    )

                with st.expander("PeTTaChainer atoms added", expanded=False):
                    st.code(
                        "\n".join(chunk.get("atomspace_added", [])) or "(none)",
                        language="lisp",
                    )
                with st.expander("Schema alignment", expanded=False):
                    st.json(chunk.get("schema_alignment", []), expanded=False)
                with st.expander("Predicate registry", expanded=False):
                    st.json(chunk.get("predicate_registry", []), expanded=False)
                with st.expander("Qdrant/parser context", expanded=False):
                    st.json(chunk.get("context", []), expanded=False)

    st.subheader("Whole PLN Statement Set")
    st.code("\n".join(_all_pln_statements(payload)) or "(none)", language="lisp")

st.header("Query (Debug)")
question = st.text_input("Question", key="query_input")

col1, col2 = st.columns([1, 5])
with col1:
    run_query = st.button("Run debug query", key="run_query_btn")

if run_query:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            resp = httpx.post(
                f"{api_base}/debug/query",
                json={"question": question},
                timeout=240,  # Complex parsing/reasoning can be slow on dense text.
            )
            resp.raise_for_status()
            payload = resp.json()
            # Store in session state
            st.session_state.query_results = payload
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.session_state.query_results = None

# Display stored query results (persists even when typing in query box)
if st.session_state.query_results is not None:
    payload = st.session_state.query_results

    st.subheader("Answer")
    st.write(payload.get("answer", ""))

    st.subheader("Proof")
    st.code(payload.get("proof", ""))

    st.subheader("Parser output")

    # Single dropdown for query debug details - use stable key
    query_debug_view = st.selectbox(
        "View query details",
        ["Summary", "LangExtract", "PLN", "PeTTaChainer", "Qdrant"],
        key="query_debug_view"
    )

    if query_debug_view == "Summary":
        st.json(
            {
                "executed_query": payload.get("executed_query", ""),
                "query_source": payload.get("query_source", ""),
                "fallback_used": payload.get("fallback_used", False),
                "query_status": payload.get("query_status", ""),
                "execution_candidates": payload.get("execution_candidates", []),
                "qdrant_aligned_queries": payload.get("qdrant_aligned_queries", []),
                "parser_candidate_queries": payload.get("pln_canonicalized_queries", []),
                "supporting_statements": payload.get("supporting_statements", []),
                "qdrant_context_atom_count": len(payload.get("context", [])),
            },
            expanded=False,
        )
    elif query_debug_view == "LangExtract":
        st.markdown("**Post-processed**")
        st.json(payload.get("langextract_postprocessed", {}))
    elif query_debug_view == "PLN":
        st.markdown("**Canonicalized queries**")
        st.code(
            "\n".join(payload.get("pln_canonicalized_queries", [])) or "(none)",
            language="lisp",
        )
        st.markdown("**Supporting statements**")
        st.code(
            "\n".join(payload.get("supporting_statements", [])) or "(none)",
            language="lisp",
        )
    elif query_debug_view == "PeTTaChainer":
        st.markdown("**Execution**")
        st.json(
            {
                "executed_query": payload.get("executed_query", ""),
                "query_source": payload.get("query_source", ""),
                "fallback_used": payload.get("fallback_used", False),
                "query_status": payload.get("query_status", ""),
                "execution_candidates": payload.get("execution_candidates", []),
                "sources": payload.get("sources", []),
            }
        )
    elif query_debug_view == "Qdrant":
        st.markdown("**Qdrant/context atoms**")
        st.markdown("**Context**")
        st.json(payload.get("context", []))
        st.markdown("**Qdrant matches**")
        st.json(payload.get("qdrant_matches", []), expanded=False)
        st.markdown("**Qdrant aligned answer candidates**")
        st.code(
            "\n".join(payload.get("qdrant_aligned_queries", [])) or "(none)",
            language="lisp",
        )
        st.caption(
            "Beki-style mode: Qdrant retrieves chunk PLN as parser context. "
            "The final executable target is filtered by the question intent."
        )

st.header("Qdrant Store")
qdrant_limit = st.number_input(
    "Payload limit",
    min_value=1,
    max_value=200,
    value=50,
    step=10,
)

if st.button("Refresh Qdrant contents", key="refresh_qdrant_btn"):
    try:
        resp = httpx.get(
            f"{api_base}/debug/qdrant",
            params={"limit": int(qdrant_limit)},
            timeout=30,
        )
        resp.raise_for_status()
        st.session_state.qdrant_results = resp.json()
    except Exception as exc:
        st.error(f"Failed to read Qdrant: {exc}")
        st.session_state.qdrant_results = None

if st.session_state.qdrant_results is not None:
    payload = st.session_state.qdrant_results
    st.caption(
        f"Qdrant enabled: {payload.get('enabled')} | "
        f"chunk points: {payload.get('count', 0)} | "
        f"predicate cards: {payload.get('predicate_count', 0)}"
    )
    for idx, point in enumerate(payload.get("points", []), start=1):
        point_payload = point.get("payload", {})
        with st.expander(
            f"Point {idx}: {point_payload.get('nl', '')[:90]}",
            expanded=False,
        ):
            st.markdown("**Natural language**")
            st.write(point_payload.get("nl", ""))
            st.markdown("**Stored PLN atoms**")
            st.code(
                "\n".join(point_payload.get("pln", [])) or "(none)",
                language="lisp",
            )
            st.markdown("**Query targets**")
            st.code(
                "\n".join(point_payload.get("query_targets", [])) or "(none)",
                language="lisp",
            )
            st.markdown("**Metadata**")
            st.json(point_payload.get("metadata", {}), expanded=False)
            st.markdown("**Raw point**")
            st.json(point, expanded=False)

    st.subheader("Predicate Card Store")
    for idx, point in enumerate(payload.get("predicate_points", []), start=1):
        point_payload = point.get("payload", {})
        label = point_payload.get("predicate", "unknown")
        arity = point_payload.get("arity", "?")
        with st.expander(f"Predicate {idx}: {label}/{arity}", expanded=False):
            st.json(point_payload, expanded=False)
