"""
ui/app.py - Streamlit chat UI (FULL FILE)

Fixes:
1) User message appears immediately
2) Spinner stays visible during entire streaming
3) Streaming feels smoother (throttled UI updates)
4) If backend errors (404/500), show it immediately (no silent hang)
"""

import re
import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"


# -----------------------------------------------------------------------------
# Text cleaning
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Conversation history -> role-based turns for backend
# -----------------------------------------------------------------------------
def build_conversation_history(messages, max_turns: int = 12):
    history = []
    for role, content in messages[-max_turns:]:
        if role == "user":
            history.append({"role": "user", "content": content})
        else:
            ans = ""
            if isinstance(content, dict):
                ans = content.get("answer", "") or ""
            history.append({"role": "assistant", "content": clean_text(ans)})
    return history


# -----------------------------------------------------------------------------
# Backend readiness
# -----------------------------------------------------------------------------
def backend_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if "application/json" in r.headers.get("content-type", ""):
            return r.json()
    except Exception:
        return None
    return None


def backend_ready() -> bool:
    h = backend_health()
    return bool(h and h.get("ready", False))


# -----------------------------------------------------------------------------
# Backend call with SAFE error handling (no silent hangs)
# -----------------------------------------------------------------------------
def call_backend_stream(message: str):
    """
    Calls /chat_stream.

    Returns:
      (json_obj, None)   if backend replies application/json
      (None, response)   if backend replies text/plain stream
      (error_json, None) on any exception or HTTP error
    """
    conversation_history = build_conversation_history(st.session_state.messages, max_turns=12)

    # Longer timeout on cold start
    timeout_s = 600 if not backend_ready() else 300

    try:
        r = requests.post(
            f"{API_URL}/chat_stream",
            json={"message": message, "conversation_history": conversation_history},
            timeout=timeout_s,
            stream=True,
        )

        # ✅ CRITICAL: if backend returns error, don't "stream" it silently
        if r.status_code >= 400:
            try:
                body = r.text[:2000]
            except Exception:
                body = "<unable to read error body>"
            return {
                "mode": "clarify",
                "answer": f"❌ Backend error {r.status_code}\n\n{body}",
                "citations": [],
                "debug": {"status_code": r.status_code},
            }, None

        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                return r.json(), None
            finally:
                try:
                    r.close()
                except Exception:
                    pass

        return None, r

    except requests.exceptions.Timeout:
        return {
            "mode": "clarify",
            "answer": "⏱️ Stream timeout. Backend is taking too long. Try again.",
            "citations": [],
            "debug": {},
        }, None
    except requests.exceptions.ConnectionError:
        return {
            "mode": "clarify",
            "answer": "❌ Cannot connect to backend. Start it with:\n\n`python -m uvicorn backend.main:app`",
            "citations": [],
            "debug": {},
        }, None
    except Exception as e:
        return {"mode": "clarify", "answer": f"❌ Stream error: {type(e).__name__}: {e}", "citations": [], "debug": {}}, None


# -----------------------------------------------------------------------------
# UI setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Airline Dispute Assistant", page_icon="✈️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("# ✈️ Airline Dispute Assistant")
st.caption("Get help with refunds • disruptions • baggage issues")
st.divider()

with st.sidebar:
    st.markdown("### 🔌 Status")
    h = backend_health()
    if h is None:
        st.error("🔴 Backend offline")
        st.caption("Run: `python -m uvicorn backend.main:app`")
    else:
        if h.get("ready"):
            st.success("✅ Backend warmed up")
        else:
            st.warning("⏳ Backend starting (warming models)")
        st.caption(f"**Model:** {h.get('ollama_model', '?')}")
        st.caption(f"**Embedder:** {h.get('embed_model', '?')}")
        st.caption(f"**Reranker:** {h.get('reranker_model', '?')}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Render completed history
# -----------------------------------------------------------------------------
for role, content in st.session_state.messages:
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            mode = content.get("mode", "answer")
            answer_text = clean_text(content.get("answer", ""))
            if mode == "clarify":
                st.info(f"🔍 **{answer_text}**")
            elif mode == "escalate":
                st.warning(f"🔺 **Escalation Required**\n\n{answer_text}")
            else:
                st.markdown(answer_text)


# -----------------------------------------------------------------------------
# Input + immediate render + streaming response
# -----------------------------------------------------------------------------
user_input = st.chat_input("Ask about refunds, cancellations, or baggage issues...")

if user_input:
    msg = user_input.strip()

    # ✅ Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg)
    st.session_state.messages.append(("user", msg))

    # Assistant response
    with st.chat_message("assistant", avatar="🤖"):
        # ✅ Spinner stays visible during entire streaming duration
        with st.spinner("🔍 Searching policies..."):
            final_json, stream_resp = call_backend_stream(msg)

            # JSON response (clarify/escalate/answer as JSON)
            if final_json is not None:
                resp = final_json
                mode = resp.get("mode", "answer")
                answer_text = clean_text(resp.get("answer", ""))

                if mode == "clarify":
                    st.info(f"🔍 **{answer_text}**")
                elif mode == "escalate":
                    st.warning(f"🔺 **Escalation Required**\n\n{answer_text}")
                else:
                    st.markdown(answer_text)

                st.session_state.messages.append(("bot", resp))
                st.rerun()

            # Stream response
            if stream_resp is None:
                resp = {"mode": "clarify", "answer": "❌ Streaming failed (no response).", "citations": [], "debug": {}}
                st.info(f"🔍 **{resp['answer']}**")
                st.session_state.messages.append(("bot", resp))
                st.rerun()

            placeholder = st.empty()
            accum = ""
            last_render_t = time.time()
            last_render_len = 0

            try:
                for chunk in stream_resp.iter_content(chunk_size=4096, decode_unicode=True):
                    if not chunk:
                        continue
                    accum += chunk

                    now = time.time()
                    # ✅ smoother: update ~2 times/sec OR every ~400 chars
                    if (now - last_render_t) >= 0.50 or (len(accum) - last_render_len) >= 400:
                        placeholder.markdown(clean_text(accum))
                        last_render_t = now
                        last_render_len = len(accum)

                placeholder.markdown(clean_text(accum))
            finally:
                try:
                    stream_resp.close()
                except Exception:
                    pass

    st.session_state.messages.append(("bot", {"mode": "answer", "answer": accum, "citations": [], "debug": {"streamed": True}}))
    st.rerun()