"""
backend/main.py

Fixes added (NO retrieval-quality change):
1) Reduce prompt size (transcript 12->6, evidence snippet 250->120) to reduce TTFT
2) /chat_stream streaming is failure-safe and prints timings
3) num_predict lowered to 350 for faster generation (still enough for your 4–6 sentence format)
4) warmup retriever (embedding + reranker) so first user query doesn't spike
   + airline filter fallback (prevents wrong-filter false negatives)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import settings
from .decisionengine import DecisionEngine
from .ingestion import ingest_policies
from .retrieval import Retriever
from .ollama_client import (
    OLLAMA_MODEL as ACTIVE_OLLAMA_MODEL,
    generate as ollama_generate,
    generate_stream,
)
from .slots import (
    detect_case,
    extract_slots,
    missing_slots,
    clarifying_question,
    build_retrieval_query,
    detect_airline,
)

app = FastAPI(title="Airline Dispute RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
decision_engine = DecisionEngine()
READY = False


# -----------------------------
# Startup warmup
# -----------------------------
@app.on_event("startup")
def warmup():
    global READY
    try:
        _ = retriever.collection
        _ = retriever.embedder
        _ = retriever.reranker

        # FIX 4: warmup to avoid first-query latency spikes
        if hasattr(retriever, "warmup"):
            retriever.warmup()

        READY = True
        print("✅ Warmup complete", flush=True)
        print(f"✅ Active Ollama model: {ACTIVE_OLLAMA_MODEL}", flush=True)
    except Exception as e:
        READY = False
        print(f"❌ Warmup failed: {type(e).__name__}: {e}", flush=True)


# -----------------------------
# Schemas
# -----------------------------
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatTurn] = []


# -----------------------------
# Helpers
# -----------------------------
def _safe_turns(turns: Optional[List[ChatTurn]]) -> List[ChatTurn]:
    return [t for t in (turns or []) if t.role and t.content]


def _user_only_text(turns: List[ChatTurn]) -> List[str]:
    return [t.content for t in turns if t.role == "user"]


def _is_new_issue(current_msg: str, user_history: List[str]) -> bool:
    if not user_history:
        return True

    msg_lower = current_msg.lower().strip()
    markers = [
        "new issue",
        "new problem",
        "different problem",
        "different issue",
        "change topic",
        "another question",
        "separate issue",
        "unrelated",
        "by the way",
        "also i have",
        "now about",
    ]
    if any(m in msg_lower for m in markers):
        return True

    current_airline = detect_airline(current_msg)
    if current_airline:
        history_airline = detect_airline(" ".join(user_history))
        if history_airline and current_airline.lower() != history_airline.lower():
            return True

    return False


def _get_relevant_context(current_msg: str, user_history: List[str], max_tokens: int = 2000) -> str:
    relevant_history = user_history[-2:] if _is_new_issue(current_msg, user_history) else user_history
    full = " ".join(relevant_history + [current_msg])

    max_chars = max_tokens * 4
    if len(full) > max_chars:
        full = "..." + full[-max_chars:]
    return full.strip()


def _flatten_transcript(turns: List[ChatTurn], max_turns: int = 12) -> str:
    turns = turns[-max_turns:]
    lines = []
    for t in turns:
        role = "User" if t.role == "user" else "Assistant"
        lines.append(f"{role}: {t.content}")
    return "\n".join(lines).strip()


def _build_citations(top_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in top_chunks:
        out.append(
            {
                "source": c["meta"].get("source", ""),
                "airline": c["meta"].get("airline", ""),
                "authority": c["meta"].get("authority", ""),
                "domain": c["meta"].get("domain", ""),
                "url": c["meta"].get("url", ""),
                "chunk_id": c["id"],
                "rerank_score": round(float(c.get("rerank_score", 0.0)), 4),
                "excerpt": (c["doc"][:600] or ""),
            }
        )
    return out


def _fallback_answer(top_chunks: List[Dict[str, Any]], confidence: str) -> str:
    header = (
        "⚠️ **Low confidence** — I found relevant sections, but please verify with the airline:\n\n"
        if confidence == "low"
        else "Here's what I found in the policy evidence:\n\n"
    )
    parts = []
    for i, c in enumerate(top_chunks, 1):
        airline = c["meta"].get("airline", "Unknown")
        domain = c["meta"].get("domain", "")
        parts.append(f"**[{i}] {airline} — {domain}**\n{c['doc'].strip()}")
    return header + "\n\n---\n\n".join(parts)


def _build_prompt(
    user_msg: str,
    decision: Any,
    top_chunks: List[Dict[str, Any]],
    confidence: str,
    slots: Dict[str, Any],
    turns: List[ChatTurn],
) -> str:
    evidence_lines = []
    for i, c in enumerate(top_chunks[:4], 1):
        meta = c["meta"]
        airline_display = (
            meta.get("airline", "").title()
            if meta.get("airline") not in ("dot", "internal")
            else meta.get("airline", "").upper()
        )

        # FIX 1: smaller evidence snippets (speed; no retrieval quality change)
        snippet = (c["doc"] or "").strip()[:120]
        evidence_lines.append(f"[{i}] {airline_display} {meta.get('domain','')}:\n{snippet}")

    # FIX 1: fewer turns (speed; preserves context)
    transcript = _flatten_transcript(turns, max_turns=6)

    situation = getattr(decision, "recommended_action", "")

    return f"""You are an expert airline dispute assistant. Interpret policies and explain what they mean for this passenger.

CURRENT QUESTION: "{user_msg}"

CONVERSATION:
{transcript}

SITUATION ANALYSIS:
{situation}

POLICY EVIDENCE:
{chr(10).join(evidence_lines)}

INSTRUCTIONS:
- Give 2–3 concrete next steps and why
- Cite evidence like [1][2]
- Be conversational (4–6 sentences)
- Don't ask for info already known

Answer:""".strip()


def _clarify_json(case: str, slots: Dict[str, Any], missing: List[str]) -> Dict[str, Any]:
    return {
        "mode": "clarify",
        "answer": clarifying_question(case, missing, slots),
        "citations": [],
        "debug": {"case": case, "slots": slots, "missing_slots": missing, "used_llm": False},
    }


def _low_match_json(case: str, slots: Dict[str, Any], query: str, top_score: float) -> Dict[str, Any]:
    return {
        "mode": "clarify",
        "answer": "I couldn't find a strong policy match. Which airline is this for and what exactly happened?",
        "citations": [],
        "debug": {"case": case, "slots": slots, "query_used": query, "top_score": round(top_score, 4), "used_llm": False},
    }


def _search_with_airline_fallback(query: str, airline_filter: str | None) -> tuple[list[Dict[str, Any]], float, str]:
    used_filter = airline_filter.lower().strip() if airline_filter else None

    chunks = retriever.search(query, airline_filter=used_filter)
    score = float(chunks[0]["rerank_score"]) if chunks else 0.0

    # If filter was applied but score is low, retry once without filter and keep the better result
    if used_filter and score < 0.15:
        alt_chunks = retriever.search(query, airline_filter=None)
        alt_score = float(alt_chunks[0]["rerank_score"]) if alt_chunks else 0.0
        if alt_score > score:
            return alt_chunks, alt_score, "none"
        return chunks, score, used_filter

    return chunks, score, used_filter or "none"


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": READY,
        "collection": settings.collection_name,
        "embed_model": settings.embed_model,
        "reranker_model": settings.reranker_model,
        "ollama_model": ACTIVE_OLLAMA_MODEL,
    }


@app.post("/ingest")
def ingest():
    result = ingest_policies()
    return {"status": "ok", **result}


@app.post("/chat")
def chat(req: ChatRequest):
    user_msg = (req.message or "").strip()
    turns = _safe_turns(req.conversation_history)

    user_history = _user_only_text(turns)
    context = _get_relevant_context(user_msg, user_history, max_tokens=2000)

    case = detect_case(context)
    slots = extract_slots(context, case)

    miss = missing_slots(slots)
    if miss:
        return _clarify_json(case, slots, miss)

    query = build_retrieval_query(user_msg, slots)

    airline_filter = (slots.get("airline") or "").strip()
    top_chunks, top_score, used_filter = _search_with_airline_fallback(query, airline_filter)

    if top_score < 0.15:
        return _low_match_json(case, slots, query, top_score)

    confidence = "high" if top_score >= 0.4 else "medium" if top_score >= 0.2 else "low"
    decision = decision_engine.evaluate(case=case, slots=slots, confidence=confidence, top_score=top_score)

    try:
        prompt = _build_prompt(user_msg, decision, top_chunks, confidence, slots, turns)
        # FIX 3: num_predict lowered
        answer = ollama_generate(prompt, timeout_s=240, num_predict=350, temperature=0.4)
        if not answer.strip():
            raise RuntimeError("empty answer")
        used_llm = True
        llm_error = None
    except Exception as e:
        answer = _fallback_answer(top_chunks, confidence)
        used_llm = False
        llm_error = f"{type(e).__name__}: {e}"

    return {
        "mode": "answer",
        "answer": answer,
        "citations": _build_citations(top_chunks),
        "debug": {
            "case": case,
            "slots": slots,
            "query_used": query,
            "airline_filter_used": used_filter,
            "top_score": round(top_score, 4),
            "confidence": confidence,
            "chunks_retrieved": len(top_chunks),
            "used_llm": used_llm,
            "llm_error": llm_error,
            "decision": decision.to_dict(),
        },
    }


@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    t0 = time.time()
    print("[TIMING] /chat_stream called", flush=True)

    user_msg = (req.message or "").strip()
    turns = _safe_turns(req.conversation_history)

    user_history = _user_only_text(turns)
    context = _get_relevant_context(user_msg, user_history, max_tokens=2000)

    case = detect_case(context)
    slots = extract_slots(context, case)

    miss = missing_slots(slots)
    if miss:
        print("[TIMING] early_return: missing slots -> clarify json", flush=True)
        return _clarify_json(case, slots, miss)

    query = build_retrieval_query(user_msg, slots)

    airline_filter = (slots.get("airline") or "").strip()
    top_chunks, top_score, used_filter = _search_with_airline_fallback(query, airline_filter)

    t_retrieval = time.time()
    print(
        f"[TIMING] retrieval+rerank: {t_retrieval - t0:.2f}s | top_score={top_score:.3f} | filter={used_filter}",
        flush=True,
    )

    if top_score < 0.15:
        print("[TIMING] early_return: low_score -> clarify json", flush=True)
        return _low_match_json(case, slots, query, top_score)

    confidence = "high" if top_score >= 0.4 else "medium" if top_score >= 0.2 else "low"
    decision = decision_engine.evaluate(case=case, slots=slots, confidence=confidence, top_score=top_score)

    prompt = _build_prompt(user_msg, decision, top_chunks, confidence, slots, turns)
    print(f"[TIMING] prompt_built: {time.time() - t0:.2f}s | starting stream…", flush=True)

    def event_stream():
        first = True
        try:
            # FIX 3: num_predict lowered
            for chunk in generate_stream(prompt, timeout_s=240, num_predict=350, temperature=0.4):
                if first:
                    first = False
                    print(f"[TIMING] first_token: {time.time() - t0:.2f}s", flush=True)
                yield chunk
            print(f"[TIMING] done_total: {time.time() - t0:.2f}s", flush=True)
        except Exception as e:
            msg = f"\n\n❌ Streaming failed: {type(e).__name__}: {e}\n"
            print(f"[ERROR] streaming failed: {type(e).__name__}: {e}", flush=True)
            yield msg

    return StreamingResponse(event_stream(), media_type="text/plain")