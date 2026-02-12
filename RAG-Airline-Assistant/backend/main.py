"""
backend/main.py
"""
from __future__ import annotations
from typing import Any, Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .config import settings
from .decisionengine import DecisionEngine, DecisionResult
from .ingestion import ingest_policies
from .retrieval import Retriever
from .slots import build_retrieval_query, clarifying_question, detect_case, extract_slots, missing_slots

app = FastAPI(title="Airline Dispute RAG Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501","http://127.0.0.1:8501"], allow_methods=["*"], allow_headers=["*"])

retriever = Retriever()
decision_engine = DecisionEngine()

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[str] = []

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
    mode: str
    answer: str
    citations: List[CitationItem]
    debug: Dict[str, Any]

def _build_answer(top_chunks, decision, confidence):
    """Policy evidence only. Decision guidance is rendered by the UI panel separately."""
    parts = []
    if confidence == "low":
        parts.append("⚠️ **Low confidence** — relevant sections found but please verify with the airline.\n")
    else:
        parts.append("Based on the airline's policy and DOT regulations:\n")
    evidence_lines = []
    for i, c in enumerate(top_chunks, 1):
        airline = c["meta"].get("airline", "Unknown")
        domain = c["meta"].get("domain", "")
        evidence_lines.append(f"**[{i}] {airline} — {domain}**\n{c['doc'].strip()}")
    parts.append("\n\n---\n\n".join(evidence_lines))
    return "\n".join(parts)

def _build_citations(top_chunks):
    return [CitationItem(
        source=c["meta"].get("source",""), airline=c["meta"].get("airline",""),
        authority=c["meta"].get("authority",""), domain=c["meta"].get("domain",""),
        url=c["meta"].get("url",""), chunk_id=c["id"],
        rerank_score=round(c["rerank_score"],4), excerpt=c["doc"][:600],
    ) for c in top_chunks]

@app.get("/health")
def health():
    return {"status":"ok","collection":settings.collection_name,"embed_model":settings.embed_model,"reranker_model":settings.reranker_model}

@app.post("/debug-retrieval")
def debug_retrieval(req: ChatRequest):
    user_msg = req.message.strip()
    full_context = " ".join(req.conversation_history[-3:] + [user_msg])
    case = detect_case(full_context)
    slots = extract_slots(full_context, case)
    query = build_retrieval_query(user_msg, slots)
    candidates = retriever.retrieve(query, airline_filter=None)
    pairs = [(query, c["doc"]) for c in candidates]
    scores = retriever.reranker.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return {"case":case,"slots":slots,"query_used":query,"results":[{"rank":i+1,"score":c["rerank_score"],"airline":c["meta"].get("airline"),"domain":c["meta"].get("domain"),"doc_preview":c["doc"][:120]} for i,c in enumerate(candidates)]}

@app.post("/ingest")
def ingest():
    result = ingest_policies()
    return {"status":"ok",**result}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_msg = req.message.strip()
    full_context = " ".join(req.conversation_history[-3:] + [user_msg])
    case = detect_case(full_context)
    slots = extract_slots(full_context, case)
    missing = missing_slots(slots)
    if missing:
        question = clarifying_question(case, missing)
        return ChatResponse(mode="clarify", answer=question, citations=[], debug={"case":case,"slots":slots,"missing_slots":missing,"top_score":None,"confidence":None,"decision":None})
    query = build_retrieval_query(user_msg, slots)
    top_chunks = retriever.search(query, airline_filter=None)
    top_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0
    if top_score < settings.rerank_threshold_none:
        return ChatResponse(mode="clarify", answer="I couldn't find a clear policy match. Could you clarify: (1) which airline, and (2) whether the airline canceled your flight or you want to cancel it yourself?", citations=[], debug={"case":case,"slots":slots,"query_used":query,"top_score":round(top_score,4),"confidence":"none","decision":None})
    confidence = "high" if top_score >= settings.rerank_threshold_low else "low"
    decision = decision_engine.evaluate(case=case, slots=slots, confidence=confidence, top_score=top_score)
    answer = _build_answer(top_chunks, decision, confidence)
    citations = _build_citations(top_chunks)
    return ChatResponse(mode=decision.action, answer=answer, citations=citations, debug={"case":case,"slots":slots,"query_used":query,"top_score":round(top_score,4),"confidence":confidence,"chunks_retrieved":len(top_chunks),"decision":{"action":decision.action,"recommended_action":decision.recommended_action,"options":decision.options,"reason":decision.reason,"escalate_if":decision.escalate_if}})