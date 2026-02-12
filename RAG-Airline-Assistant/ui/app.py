"""
ui/app.py
---------
Streamlit chat interface for the Airline Dispute RAG Assistant.
Run with:  streamlit run ui/app.py
"""
import streamlit as st
import requests
import re

import re

def clean_text(s: str) -> str:
    s = s or ""

    # Remove OpenAI-style cite markers: [oaicite:9]{index=9}
    s = re.sub(r"\[oaicite:\d+\]\{index=\d+\}", "", s)

    # Remove any leftover bracket cite markers if present
    s = re.sub(r"\[oaicite:\d+\]", "", s)
    s = re.sub(r"\{index=\d+\}", "", s)

    # Remove contentReference variants (full or partial fragments)
    s = re.sub(r"[:.]?\s*contentReference\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[:.]?\s*conte\b", "", s, flags=re.IGNORECASE)  # catches ".conte" tail

    # Cleanup excessive whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()



API_URL = "http://127.0.0.1:8000"

# ------------------------------------------------------------------ #
#  Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Airline Dispute RAG Assistant",
    page_icon="✈️",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("✈️ Airline RAG Assistant")
    st.caption("Powered by ChromaDB + bge-reranker + FastAPI")
    st.divider()

    st.subheader("🔧 Controls")

    if st.button("📥 Ingest / Refresh Policy Docs", use_container_width=True):
        with st.spinner("Ingesting documents into vector store..."):
            try:
                r = requests.post(f"{API_URL}/ingest", timeout=120)
                result = r.json()
                st.success(
                    f"✅ {result.get('ingested_files', 0)} files | "
                    f"{result.get('ingested_chunks', 0)} chunks"
                )
            except Exception as e:
                st.error(f"❌ Ingest failed: {e}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("💬 Try asking:")
    st.markdown("""
- *"My Delta flight was canceled — can I get a refund?"*
- *"American Airlines lost my bag, what are my rights?"*
- *"United delayed my bag by 2 days, can I get reimbursed?"*
- *"I have a non-refundable ticket and want to cancel"*
- *"What does DOT say about refunds?"*
""")

    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("🟢 Backend connected")
        st.caption(f"Model: {health.get('embed_model', '?')}")
    except Exception:
        st.error("🔴 Backend not running — start with:\n`uvicorn backend.main:app --reload`")


# ------------------------------------------------------------------ #
#  Session state
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []   # User messages only, for context window


# ------------------------------------------------------------------ #
#  Main chat area
# ------------------------------------------------------------------ #
st.title("✈️ Airline Dispute RAG Assistant")
st.caption("Ask about refunds, disruptions, or baggage issues — American, Delta, United + DOT regulations")

# Display chat history
for role, content in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(content)

    elif role == "bot":
        with st.chat_message("assistant"):
            # Mode badge
            mode = content.get("mode", "answer")
            if mode == "clarify":
                st.info("🔍 " + clean_text(content.get("answer", "")))
            else:
                st.markdown(clean_text(content.get("answer", "")))

            # Evidence + debug panel
            debug = content.get("debug", {})
            citations = content.get("citations", [])

            with st.expander("🔍 Evidence & Debug", expanded=False):
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    score = debug.get("top_score")
                    st.metric("Top Rerank Score", f"{score:.3f}" if score is not None else "—")
                with col2:
                    st.metric("Confidence", debug.get("confidence", "—").upper() if debug.get("confidence") else "—")
                with col3:
                    st.metric("Chunks Used", debug.get("chunks_retrieved", len(citations)))

                # Query used for retrieval
                if debug.get("query_used"):
                    st.markdown("**Retrieval query built from slots:**")
                    st.code(debug["query_used"], language=None)

                # Slots
                st.markdown("**Extracted slots:**")
                slots = debug.get("slots", {})
                slot_cols = st.columns(3)
                slot_items = list(slots.items())
                for idx, (k, v) in enumerate(slot_items):
                    with slot_cols[idx % 3]:
                        color = "🟢" if v not in (None, "unknown", "") else "🟡"
                        st.markdown(f"{color} **{k}**: `{v}`")

                # Missing slots (if clarify mode)
                if debug.get("missing_slots"):
                    st.warning(f"Missing slots: {', '.join(debug['missing_slots'])}")

                # Citations
                if citations:
                    st.markdown("**Retrieved policy chunks:**")
                    for i, cite in enumerate(citations, 1):
                        authority_icon = "⚖️" if cite.get("authority") == "REGULATOR" else "🏢"
                        st.markdown(
                            f"{authority_icon} **[{i}]** `{cite.get('airline', '?')}` — "
                            f"{cite.get('domain', '?')} "
                            f"*(score: {cite.get('rerank_score', 0):.3f})*"
                        )
                        if cite.get("url"):
                            st.caption(f"🔗 {cite['url']}")
                        st.markdown(f"**Excerpt {i}:**")
                        excerpt = clean_text(cite.get("excerpt", ""))
                        st.code(excerpt, language="text")
                        st.divider()



# ------------------------------------------------------------------ #
#  Chat input
# ------------------------------------------------------------------ #
user_input = st.chat_input("Ask about a refund, disruption, or baggage issue...")

if user_input:
    # Add user message to display
    st.session_state.messages.append(("user", user_input))

    # Build conversation history for context
    st.session_state.history.append(user_input)
    history_window = st.session_state.history[-5:]  # Last 5 turns

    with st.spinner("Searching policies..."):
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={
                    "message": user_input,
                    "conversation_history": history_window,
                },
                timeout=30,
            ).json()
        except requests.exceptions.ConnectionError:
            resp = {
                "mode": "clarify",
                "answer": "❌ Cannot connect to backend. Make sure `uvicorn backend.main:app --reload` is running.",
                "citations": [],
                "debug": {},
            }
        except Exception as e:
            resp = {
                "mode": "clarify",
                "answer": f"❌ Error: {str(e)}",
                "citations": [],
                "debug": {},
            }

    st.session_state.messages.append(("bot", resp))
    st.rerun()