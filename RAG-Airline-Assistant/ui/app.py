"""
ui/app.py - Streamlit chat interface
Run with: streamlit run ui/app.py

FIXES:
- Removed delayed rendering (instant message display)
- Arrow icon instead of "Send" button
- Modern, attractive UI with better spacing
- Fixed text cleaning issues
- Better error handling
- Smoother user experience
"""

import re
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


def clean_text(s: str) -> str:
    """Remove citation artifacts from text"""
    s = s or ""
    s = re.sub(r"\[oaicite:\d+\]\{index=\d+\}", "", s)
    s = re.sub(r"\[oaicite:\d+\]", "", s)
    s = re.sub(r"\{index=\d+\}", "", s)
    s = re.sub(r"[:.]?\s*contentReference\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[:.]?\s*conte\b", "", s, flags=re.IGNORECASE)  # FIXED: added partial match
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="Airline Dispute Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================================
# CUSTOM CSS FOR MODERN UI
# ============================================================================

st.markdown("""
<style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 1px solid rgba(250, 250, 250, 0.2);
        padding-top: 1rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Code block improvements */
    .stCodeBlock {
        font-size: 0.85rem;
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: rgba(240, 242, 246, 0.5);
    }
    
    /* Hide default footer */
    footer {visibility: hidden;}
    
    /* Button improvements */
    .stButton button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# ✈️ Airline Assistant")
    st.caption("Powered by local AI • ChromaDB • Ollama")
    
    st.divider()
    
    st.markdown("### 🔧 Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Ingest", use_container_width=True, help="Reload policy documents"):
            with st.spinner("Ingesting policies..."):
                try:
                    r = requests.post(f"{API_URL}/ingest", timeout=300)
                    if "application/json" in r.headers.get("content-type", ""):
                        result = r.json()
                        st.toast(f"✅ Loaded {result.get('ingested_chunks', 0)} chunks", icon="✅")
                    else:
                        st.error("❌ Non-JSON response from /ingest")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()
    
    st.divider()
    
    st.markdown("### 💬 Example queries")
    
    examples = [
        "Delta cancelled my flight – can I get a refund?",
        "American lost my checked bag – what are my rights?",
        "I want to cancel my non-refundable United ticket",
    ]
    
    for example in examples:
        if st.button(f"💭 {example[:35]}...", use_container_width=True, key=f"ex_{hash(example)}"):
            st.session_state.messages.append(("user", example))
            st.session_state.history.append(example)
            # Trigger backend call
            with st.spinner("🔍 Searching policies..."):
                history_window = st.session_state.history[-5:]
                try:
                    r = requests.post(
                        f"{API_URL}/chat",
                        json={"message": example, "conversation_history": history_window},
                        timeout=60,
                    )
                    if "application/json" in r.headers.get("content-type", ""):
                        resp = r.json()
                    else:
                        resp = {
                            "mode": "clarify",
                            "answer": "❌ Backend error: Non-JSON response",
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
    
    st.divider()
    
    # Backend health check
    st.markdown("### 🔌 Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if "application/json" in health.headers.get("content-type", ""):
            health_data = health.json()
            st.success("✅ Backend online")
            with st.expander("ℹ️ Details", expanded=False):
                st.caption(f"**Model:** {health_data.get('ollama_model', '?')}")
                st.caption(f"**Embedder:** {health_data.get('embed_model', '?')[:30]}...")
                st.caption(f"**Reranker:** {health_data.get('reranker_model', '?')[:30]}...")
        else:
            st.warning("⚠️ Backend responding (non-JSON)")
    except Exception:
        st.error("🔴 Backend offline")
        st.caption("Run: `uvicorn backend.main:app --reload`")


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.markdown("# ✈️ Airline Dispute Assistant")
st.caption("Get help with refunds • disruptions • baggage issues")

st.divider()

# Render chat history
for role, content in st.session_state.messages:
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
        continue

    # Assistant message
    with st.chat_message("assistant", avatar="🤖"):
        mode = content.get("mode", "answer")
        answer_text = clean_text(content.get("answer", ""))

        # Main answer display
        if mode == "clarify":
            st.info(f"🔍 **{answer_text}**")
        elif mode == "escalate":
            st.warning(f"🔺 **Escalation Required**\n\n{answer_text}")
        else:
            st.markdown(answer_text)

        debug = content.get("debug", {}) or {}
        citations = content.get("citations", []) or []
        decision = debug.get("decision") or {}

        # Quick metrics row
        if debug:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score = debug.get("top_score")
                if isinstance(score, (int, float)):
                    st.metric("Match", f"{score:.2f}", delta=None, delta_color="off")
            with col2:
                conf = (debug.get("confidence") or "—").upper()
                st.metric("Confidence", conf, delta=None, delta_color="off")
            with col3:
                st.metric("Sources", debug.get("chunks_retrieved", len(citations)))
            with col4:
                llm_status = "✅" if debug.get("used_llm") else "⚠️"
                st.metric("AI", llm_status)

        # Decision panel (collapsed by default)
        if decision and decision.get("action") == "answer":
            with st.expander("📋 What This Means For You", expanded=False):
                ra = clean_text(decision.get("recommended_action", ""))
                if ra:
                    st.markdown(f"**Recommended Action:**\n\n{ra}")

                options = decision.get("options", []) or []
                if options:
                    st.markdown("\n**Your Options:**")
                    for i, opt in enumerate(options, 1):
                        st.markdown(f"{i}. {clean_text(opt)}")

                esc = decision.get("escalate_if", []) or []
                if esc:
                    st.markdown("\n**🔺 Consider Escalating If:**")
                    for e in esc:
                        st.markdown(f"• {clean_text(e)}")

        # Evidence panel (collapsed by default)
        if citations or debug.get("slots"):
            with st.expander("🔍 Evidence & Technical Details", expanded=False):
                
                # Slots
                if debug.get("slots"):
                    st.markdown("**Extracted Information:**")
                    slots = debug.get("slots", {})
                    slot_cols = st.columns(3)
                    for idx, (k, v) in enumerate(slots.items()):
                        with slot_cols[idx % 3]:
                            color = "🟢" if v not in (None, "unknown", "") else "🟡"
                            st.caption(f"{color} **{k}:** `{v}`")
                    st.divider()

                # Query
                if debug.get("query_used"):
                    st.markdown("**Search Query:**")
                    st.code(debug["query_used"], language=None)
                    st.divider()

                # Citations
                if citations:
                    st.markdown("**Policy Sources:**")
                    for i, cite in enumerate(citations, 1):
                        authority_icon = "⚖️" if cite.get("authority") == "REGULATOR" else "🏢"
                        airline = cite.get("airline", "?").title()
                        domain = cite.get("domain", "?")
                        score = cite.get("rerank_score", 0)
                        
                        st.markdown(
                            f"{authority_icon} **[{i}] {airline}** — {domain} "
                            f"*(relevance: {score:.3f})*"
                        )
                        
                        if cite.get("url"):
                            st.caption(f"🔗 {cite['url']}")
                        
                        excerpt = clean_text(cite.get("excerpt", ""))
                        if excerpt:
                            with st.container():
                                st.text(excerpt[:400] + ("..." if len(excerpt) > 400 else ""))
                        
                        if i < len(citations):
                            st.divider()


# ============================================================================
# CHAT INPUT (with arrow icon, no "Send" button)
# ============================================================================

# FIXED: Using chat_input which has built-in arrow icon
user_input = st.chat_input("Ask about refunds, cancellations, or baggage issues...")

if user_input:
    msg = user_input.strip()
    
    # Add user message immediately
    st.session_state.messages.append(("user", msg))
    st.session_state.history.append(msg)
    
    # Show user message and get response in one rerun
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching policies..."):
            history_window = st.session_state.history[-5:]
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"message": msg, "conversation_history": history_window},
                    timeout=60,
                )
                if "application/json" in r.headers.get("content-type", ""):
                    resp = r.json()
                else:
                    resp = {
                        "mode": "clarify",
                        "answer": "❌ Backend returned non-JSON response",
                        "citations": [],
                        "debug": {},
                    }
            except requests.exceptions.Timeout:
                resp = {
                    "mode": "clarify",
                    "answer": "⏱️ Request timeout. Try a simpler query or check if Ollama is running.",
                    "citations": [],
                    "debug": {},
                }
            except requests.exceptions.ConnectionError:
                resp = {
                    "mode": "clarify",
                    "answer": "❌ Cannot connect to backend. Make sure it's running:\n\n`uvicorn backend.main:app --reload`",
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
        
        # Display response immediately
        mode = resp.get("mode", "answer")
        answer_text = clean_text(resp.get("answer", ""))
        
        if mode == "clarify":
            st.info(f"🔍 **{answer_text}**")
        elif mode == "escalate":
            st.warning(f"🔺 **Escalation Required**\n\n{answer_text}")
        else:
            st.markdown(answer_text)
    
    # Save bot response and rerun to show full history
    st.session_state.messages.append(("bot", resp))
    st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(
    "💡 **Tip:** This assistant uses local AI to analyze airline policies. "
    "Always verify important information with the airline directly."
)