"""
ui/app.py - Streamlit chat interface
Run with: streamlit run ui/app.py
"""
import streamlit as st
import requests
import re

def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\[oaicite:\d+\]\{index=\d+\}", "", s)
    s = re.sub(r"\[oaicite:\d+\]", "", s)
    s = re.sub(r"\{index=\d+\}", "", s)
    s = re.sub(r"[:.]?\s*contentReference\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[:.]?\s*conte\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Airline Dispute RAG Assistant", page_icon="✈️", layout="wide")

with st.sidebar:
    st.title("✈️ Airline RAG Assistant")
    st.caption("ChromaDB · bge-reranker · FastAPI · Decision Engine")
    st.divider()
    st.subheader("🔧 Controls")
    if st.button("📥 Ingest / Refresh Policy Docs", use_container_width=True):
        with st.spinner("Ingesting documents into vector store..."):
            try:
                r = requests.post(f"{API_URL}/ingest", timeout=120)
                result = r.json()
                st.success(f"✅ {result.get('ingested_files',0)} files · {result.get('ingested_chunks',0)} chunks")
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
- *"What does DOT say about significant schedule changes?"*
""")
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("🟢 Backend connected")
        st.caption(f"Embed: {health.get('embed_model','?')}")
    except Exception:
        st.error("🔴 Backend not running\n`uvicorn backend.main:app --reload`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

st.title("✈️ Airline Dispute RAG Assistant")
st.caption("Ask about refunds, disruptions, or baggage — American · Delta · United · DOT")

for role, content in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(content)
    elif role == "bot":
        with st.chat_message("assistant"):
            mode = content.get("mode", "answer")
            answer_text = clean_text(content.get("answer", ""))
            if mode == "clarify":
                st.info(f"🔍 {answer_text}")
            elif mode == "escalate":
                st.warning(f"🔺 {answer_text}")
            else:
                st.markdown(answer_text)

            debug = content.get("debug", {})
            citations = content.get("citations", [])
            decision = debug.get("decision") or {}

            # Decision Engine Panel — always expanded so users see guidance
            if decision and decision.get("action") == "answer":
                with st.expander("📋 What This Means For You", expanded=True):
                    st.markdown(f"**{decision.get('recommended_action','')}**")
                    options = decision.get("options", [])
                    if options:
                        st.markdown("**Your options:**")
                        for opt in options:
                            st.markdown(f"- {clean_text(opt)}")
                    escalate_if = decision.get("escalate_if", [])
                    if escalate_if:
                        st.markdown("**🔺 Escalate if:**")
                        for esc in escalate_if:
                            st.markdown(f"  - {esc}")

            # Evidence & Debug Panel — collapsed by default
            with st.expander("🔍 Evidence & Debug", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    score = debug.get("top_score")
                    st.metric("Top Rerank Score", f"{score:.3f}" if score is not None else "—")
                with col2:
                    conf = debug.get("confidence") or "—"
                    st.metric("Confidence", conf.upper())
                with col3:
                    st.metric("Chunks Used", debug.get("chunks_retrieved", len(citations)))

                if debug.get("query_used"):
                    st.markdown("**Retrieval query:**")
                    st.code(debug["query_used"], language=None)

                st.markdown("**Extracted slots:**")
                slots = debug.get("slots", {})
                slot_cols = st.columns(3)
                for idx, (k, v) in enumerate(slots.items()):
                    with slot_cols[idx % 3]:
                        color = "🟢" if v not in (None, "unknown", "") else "🟡"
                        st.markdown(f"{color} **{k}**: `{v}`")

                if debug.get("missing_slots"):
                    st.warning(f"Missing: {', '.join(debug['missing_slots'])}")

                if decision:
                    st.markdown("**Decision engine:**")
                    st.caption(f"Action: `{decision.get('action')}` | {decision.get('reason','')}")

                if citations:
                    st.markdown("**Retrieved policy chunks:**")
                    for i, cite in enumerate(citations, 1):
                        authority_icon = "⚖️" if cite.get("authority") == "REGULATOR" else "🏢"
                        st.markdown(
                            f"{authority_icon} **[{i}]** `{cite.get('airline','?')}` — "
                            f"{cite.get('domain','?')} *(score: {cite.get('rerank_score',0):.3f})*"
                        )
                        if cite.get("url"):
                            st.caption(f"🔗 {cite['url']}")
                        excerpt = clean_text(cite.get("excerpt", ""))
                        # NOTE: No nested expander here — Streamlit forbids expanders inside expanders
                        st.code(excerpt, language="text")

user_input = st.chat_input("Ask about a refund, disruption, or baggage issue...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    st.session_state.history.append(user_input)
    history_window = st.session_state.history[-5:]
    with st.spinner("Searching policies..."):
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"message": user_input, "conversation_history": history_window},
                timeout=30,
            ).json()
        except requests.exceptions.ConnectionError:
            resp = {"mode":"clarify","answer":"❌ Cannot connect to backend. Make sure `uvicorn backend.main:app --reload` is running.","citations":[],"debug":{}}
        except Exception as e:
            resp = {"mode":"clarify","answer":f"❌ Error: {str(e)}","citations":[],"debug":{}}
    st.session_state.messages.append(("bot", resp))
    st.rerun()