"""
backend/main.py
---------------
FastAPI application — the main RAG pipeline lives here.

Endpoints:
  GET  /health       — health check
  POST /ingest       — (re)ingest policy documents
  POST /chat         — main chat endpoint
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .ingestion import ingest_policies
from .retrieval import Retriever
from .slots import (
    build_retrieval_query,
    clarifying_question,
    detect_case,
    extract_slots,
    missing_slots,
)

# ------------------------------------------------------------------ #
#  App setup
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Airline Dispute RAG Assistant",
    description="Grounded policy retrieval for refunds, disruptions, and baggage issues.",
    version="1.0.0",
)

# Allow Streamlit (port 8501) to call FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared retriever — models loaded once at startup
retriever = Retriever()


# ------------------------------------------------------------------ #
#  Request / Response schemas
# ------------------------------------------------------------------ #
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[str] = []   # Last N user messages for context


class CitationItem(BaseModel):
    source: str
    airline: str
    authority: str
    domain: str
    url: str
    chunk_id: str
    rerank_score: float
    excerpt: str


class ChatResponse(BaseModel):
    mode: str                           # "clarify" | "answer"
    answer: str
    citations: List[CitationItem]
    debug: Dict[str, Any]


# ------------------------------------------------------------------ #
#  Helper: build formatted answer from retrieved chunks
# ------------------------------------------------------------------ #
def _build_answer(top_chunks: List[Dict[str, Any]], confidence: str) -> str:
    if confidence == "low":
        header = (
            "⚠️ **Low confidence** — I found some relevant policy sections, "
            "but please verify directly with the airline:\n\n"
        )
    else:
        header = "Based on the airline's policy and DOT regulations:\n\n"

    lines = []
    for i, c in enumerate(top_chunks, 1):
        airline = c["meta"].get("airline", "Unknown")
        domain = c["meta"].get("domain", "")
        excerpt = c["doc"].strip()
        lines.append(f"**[{i}] {airline} — {domain}**\n{excerpt}")

    return header + "\n\n---\n\n".join(lines)


def _build_citations(top_chunks: List[Dict[str, Any]]) -> List[CitationItem]:
    citations = []
    for c in top_chunks:
        meta = c["meta"]
        citations.append(
            CitationItem(
                source=meta.get("source", ""),
                airline=meta.get("airline", ""),
                authority=meta.get("authority", ""),
                domain=meta.get("domain", ""),
                url=meta.get("url", ""),
                chunk_id=c["id"],
                rerank_score=round(c["rerank_score"], 4),
                excerpt=c["doc"][:200],
            )
        )
    return citations


# ------------------------------------------------------------------ #
#  Endpoints
# ------------------------------------------------------------------ #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "embed_model": settings.embed_model,
        "reranker_model": settings.reranker_model,
    }

@app.post("/debug-retrieval")
def debug_retrieval(req: ChatRequest):
    """Temporary: see raw retrieval scores without gating"""
    user_msg = req.message.strip()
    full_context = " ".join(req.conversation_history[-3:] + [user_msg])
    
    case = detect_case(full_context)
    slots = extract_slots(full_context, case)
    query = build_retrieval_query(user_msg, slots)
    
    # Raw candidates BEFORE reranking
    candidates = retriever.retrieve(query, airline_filter=None)
    
    # Rerank ALL of them (no top_n limit)
    pairs = [(query, c["doc"]) for c in candidates]
    scores = retriever.reranker.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return {
        "case": case,
        "slots": slots,
        "query_used": query,
        "results": [
            {
                "rank": i + 1,
                "score": c["rerank_score"],
                "airline": c["meta"].get("airline"),
                "domain": c["meta"].get("domain"),
                "doc_preview": c["doc"][:120],
            }
            for i, c in enumerate(candidates)
        ],
    }

@app.post("/ingest")
def ingest():
    """Re-ingest all policy documents. Clears and rebuilds the vector store."""
    result = ingest_policies()
    return {"status": "ok", **result}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full RAG pipeline:
    1. Merge conversation history for context
    2. Detect case (refund / baggage)
    3. Extract slots
    4. Ask clarifying question if required slots missing
    5. Build enriched retrieval query from slots
    6. Retrieve + rerank
    7. Evidence gate (confidence threshold)
    8. Return grounded answer with citations
    """
    user_msg = req.message.strip()

    # --- 1. Context window: last 3 user turns + current message ---
    full_context = " ".join(req.conversation_history[-3:] + [user_msg])

    # --- 2. Detect case ---
    case = detect_case(full_context)

    # --- 3. Extract slots ---
    slots = extract_slots(full_context, case)

    # --- 4. Clarify if required slots missing ---
    missing = missing_slots(slots)
    if missing:
        question = clarifying_question(case, missing)
        return ChatResponse(
            mode="clarify",
            answer=question,
            citations=[],
            debug={
                "case": case,
                "slots": slots,
                "missing_slots": missing,
                "top_score": None,
                "confidence": None,
            },
        )

    # --- 5. Build enriched query ---
    query = build_retrieval_query(user_msg, slots)

    # --- 6. Retrieve + rerank (optionally filtered by airline) ---
    airline_filter = slots.get("airline")  # Re-enabled after name fix
    top_chunks = retriever.search(query, airline_filter=airline_filter)

    # --- 7. Evidence gate ---
    top_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0

    if top_score < settings.rerank_threshold_none:
        return ChatResponse(
            mode="clarify",
            answer=(
                "I couldn't find a clear policy match for your question. "
                "Could you clarify: (1) which airline, and (2) whether the airline "
                "canceled your flight or you want to cancel it yourself?"
            ),
            citations=[],
            debug={
                "case": case,
                "slots": slots,
                "query_used": query,
                "top_score": round(top_score, 4),
                "confidence": "none",
            },
        )

    confidence = "high" if top_score >= settings.rerank_threshold_low else "low"

    # --- 8. Build response ---
    answer = _build_answer(top_chunks, confidence)
    citations = _build_citations(top_chunks)

    return ChatResponse(
        mode="answer",
        answer=answer,
        citations=citations,
        debug={
            "case": case,
            "slots": slots,
            "query_used": query,
            "top_score": round(top_score, 4),
            "confidence": confidence,
            "chunks_retrieved": len(top_chunks),
        },
    )