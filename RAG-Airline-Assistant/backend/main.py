"""
backend/main.py
---------------
FastAPI application — RAG pipeline with FULL conversation memory.

KEY IMPROVEMENT: Uses entire conversation context, not just last N turns.
Context persists until user starts a completely new issue.
"""
from __future__ import annotations
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests  # ADDED: for LLM error handling

from .config import settings
from .ingestion import ingest_policies
from .retrieval import Retriever
from .decisionengine import DecisionEngine
from .ollama_client import generate as ollama_generate
from .slots import detect_case, extract_slots, missing_slots, clarifying_question, build_retrieval_query


app = FastAPI(
    title="Airline Dispute RAG Assistant",
    description="Grounded policy retrieval for refunds, disruptions, and baggage issues.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
decision_engine = DecisionEngine()


# ------------------------------------------------------------------ #
#  Schemas
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
#  Context Management
# ------------------------------------------------------------------ #
def _is_new_issue(current_msg: str, conversation_history: List[str]) -> bool:
    """
    Detect if user is starting a completely NEW issue vs continuing current one.
    
    Returns True if:
    - No history (first message)
    - User explicitly says "new issue" / "different problem" / "change topic"
    - User mentions a DIFFERENT airline than what's in history
    
    Returns False if:
    - Continuing discussion of same issue
    - Follow-up questions about same situation
    """
    if not conversation_history:
        return True  # First message is always a new issue
    
    msg_lower = current_msg.lower().strip()
    
    # Explicit new issue signals
    new_issue_phrases = [
        "new issue", "new problem", "different problem", "different issue",
        "change topic", "another question", "separate issue", "unrelated",
        "by the way", "also i have", "now about",
    ]
    
    if any(phrase in msg_lower for phrase in new_issue_phrases):
        return True
    
    # Check for airline switch (strong signal of new issue)
    from .slots import detect_airline
    
    current_airline = detect_airline(current_msg)
    if current_airline:
        # Check if different airline mentioned in history
        history_text = " ".join(conversation_history)
        history_airline = detect_airline(history_text)
        
        if history_airline and current_airline.lower() != history_airline.lower():
            return True  # Switched airlines = new issue
    
    # Default: assume continuation of current issue
    return False


def _get_relevant_context(
    current_msg: str, 
    conversation_history: List[str],
    max_tokens: int = 2000
) -> str:
    """
    Build conversation context intelligently:
    - Use FULL history if it's the same issue
    - Only use recent context if new issue detected
    - Respect token limits to avoid overwhelming the LLM
    
    Rough estimate: 1 token ≈ 4 characters
    """
    if _is_new_issue(current_msg, conversation_history):
        # New issue: only use last 2 turns for minimal context
        relevant_history = conversation_history[-2:] if conversation_history else []
    else:
        # Same issue: use ALL history (subject to token limit)
        relevant_history = conversation_history
    
    # Build context string
    full_context = " ".join(relevant_history + [current_msg])
    
    # Token limit check (rough: 1 token ≈ 4 chars)
    max_chars = max_tokens * 4
    if len(full_context) > max_chars:
        # Truncate from the BEGINNING (keep recent context)
        # But always keep at least the current message
        excess = len(full_context) - max_chars
        if excess < len(full_context) - len(current_msg):
            full_context = "..." + full_context[excess:]
        # Otherwise just use current message
        else:
            full_context = current_msg
    
    return full_context.strip()


# ------------------------------------------------------------------ #
#  Out-of-scope detection
# ------------------------------------------------------------------ #
_HUMAN_AGENT_PHRASES = [
    "human agent", "speak to agent", "talk to agent", "chat with agent",
    "speak to a human", "talk to a human", "chat with a human", "real person",
    "live agent", "live support", "customer service", "speak to someone",
    "talk to someone", "call the airline", "phone number", "contact number",
    "supervisor", "manager",
]

_GREETING_PHRASES = [
    "hello", "hi there", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "what can you do", "help me",
    "what do you do",
]

_UNRELATED_PHRASES = [
    "weather", "hotel", "restaurant", "car rental", "visa",
    "passport", "flight status", "track my flight", "gate number",
    "seat upgrade", "lounge access", "frequent flyer", "miles",
]


def _is_human_agent_request(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in _HUMAN_AGENT_PHRASES)


def _is_greeting(text: str) -> bool:
    t = text.lower().strip()
    dispute_keywords = [
        "refund", "cancel", "bag", "baggage", "luggage", "flight",
        "ticket", "airline", "american", "delta", "united", "dot",
        "compensation", "reimburs", "delay", "lost", "damaged",
        "frustrated", "angry", "refusing", "denied",
    ]
    if any(k in t for k in dispute_keywords):
        return False
    if len(t) > 60:
        return False
    return any(t.startswith(p) or t == p for p in _GREETING_PHRASES)


def _is_unrelated(text: str) -> bool:
    t = text.lower().strip()
    dispute_keywords = [
        "refund", "cancel", "bag", "baggage", "luggage", "flight",
        "ticket", "airline", "american", "delta", "united", "dot",
        "compensation", "reimburs", "delay", "lost", "damaged",
    ]
    if any(k in t for k in dispute_keywords):
        return False
    return any(p in t for p in _UNRELATED_PHRASES)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _build_answer(top_chunks: List[Dict[str, Any]], confidence: str) -> str:
    """Fallback answer builder when LLM is unavailable."""
    header = (
        "⚠️ **Low confidence** — I found relevant sections, but please verify with the airline:\n\n"
        if confidence == "low"
        else "Here's what I found in the policy evidence:\n\n"
    )
    lines = []
    for i, c in enumerate(top_chunks, 1):
        airline = c["meta"].get("airline", "Unknown")
        domain = c["meta"].get("domain", "")
        lines.append(f"**[{i}] {airline} — {domain}**\n{c['doc'].strip()}")
    return header + "\n\n---\n\n".join(lines)


def _build_citations(top_chunks: List[Dict[str, Any]]) -> List[CitationItem]:
    return [
        CitationItem(
            source=c["meta"].get("source", ""),
            airline=c["meta"].get("airline", ""),
            authority=c["meta"].get("authority", ""),
            domain=c["meta"].get("domain", ""),
            url=c["meta"].get("url", ""),
            chunk_id=c["id"],
            rerank_score=round(float(c["rerank_score"]), 4),
            excerpt=(c["doc"][:600] or ""),
        )
        for c in top_chunks
    ]


def _build_grounded_prompt(
    user_msg: str, 
    decision: Any, 
    top_chunks: List[Dict[str, Any]], 
    confidence: str,
    slots: Dict[str, Any],
    conversation_history: List[str]
) -> str:
    """
    Build a reasoning-focused prompt that makes LLM interpret and synthesize,
    not just regurgitate policies.
    """
    
    # ── EVIDENCE: Concise and focused ──────────────────────────────────
    evidence_lines = []
    for i, c in enumerate(top_chunks[:4], 1):  # Only top 4, not 5
        meta = c["meta"]
        airline_display = meta.get('airline', '').title() if meta.get('airline') not in ('dot', 'internal') else meta.get('airline', '').upper()
        
        # SHORTER excerpts (250 chars instead of 400)
        text = (c["doc"] or "").strip()[:250]
        
        # Cleaner format
        source_tag = f"{airline_display} {meta.get('domain', '')}"
        evidence_lines.append(f"[{i}] {source_tag}:\n{text}")
    
    evidence_block = "\n\n".join(evidence_lines)
    
    # ── CONVERSATION CONTEXT: Adaptive ─────────────────────────────────
    context_block = ""
    if conversation_history:
        if len(conversation_history) <= 3:
            # Short: show all
            context_block = "CONVERSATION:\n" + "\n".join([f"User: {msg}" for msg in conversation_history])
        else:
            # Long: summary format
            context_block = (
                f"CONVERSATION SUMMARY:\n"
                f"Started: {conversation_history[0][:60]}...\n"
                f"Recent: {conversation_history[-1][:80]}..."
            )
    
    # ── KNOWN FACTS: What we already know ──────────────────────────────
    facts = []
    if slots.get("airline"):
        facts.append(f"Airline: {slots['airline']}")
    if slots.get("baggage_status") and slots['baggage_status'] != "unknown":
        facts.append(f"Issue: {slots['baggage_status']} baggage")
    if slots.get("airline_cancelled") == "yes":
        facts.append(f"Airline cancelled flight")
    elif slots.get("airline_cancelled") == "no":
        facts.append(f"Passenger wants to cancel")
    
    known_facts = "KNOWN FACTS:\n" + "\n".join([f"• {f}" for f in facts]) if facts else ""
    
    # ── SITUATION ANALYSIS: From decision engine ───────────────────────
    situation = getattr(decision, "recommended_action", "")
    
    # ── BUILD REASONING-FOCUSED PROMPT ─────────────────────────────────
    return f"""You are an expert airline dispute assistant. Your job is to INTERPRET policies and REASON about what they mean for this specific passenger, not just quote them.

CURRENT QUESTION: "{user_msg}"

{context_block}

{known_facts}

SITUATION ANALYSIS:
{situation}

POLICY EVIDENCE:
{evidence_block}

YOUR TASK - CRITICAL INSTRUCTIONS:

1. INTERPRET, don't just quote:
   • Read the policies above
   • THINK about what they mean for THIS passenger's situation
   • Explain the implications in plain language
   
2. REASON about next steps:
   • Based on the policies, what should the passenger DO?
   • What are their 2-3 BEST options right now?
   • What happens if they take each action?

3. BE SPECIFIC and ACTIONABLE:
   • Give concrete steps, not generic advice
   • Explain WHY each step matters
   • Cite policies [1][2] to back up your reasoning

4. STAY CONVERSATIONAL:
   • 4-6 sentences max unless genuinely complex
   • Natural tone, not robotic
   • Acknowledge emotions if passenger is frustrated

5. DON'T ASK for info you already know (see KNOWN FACTS)

EXAMPLE OF GOOD RESPONSE:
"Based on United's policy [1] and DOT regulations [2], here's what this means for you: Since your bag has been delayed for 2 days, you're entitled to reimbursement for essential purchases like toiletries and clothing. Keep all receipts. Your best next steps are: (1) File a formal delayed baggage claim with United right away if you haven't - get a reference number, and (2) Start documenting what you've had to buy. If the bag isn't found within 5 days, it will likely be declared lost, which opens up additional compensation."

Now respond, focusing on INTERPRETATION and ACTIONABLE GUIDANCE:""".strip()


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
        "ollama_model": getattr(settings, "ollama_model", "local"),
    }


@app.post("/ingest")
def ingest():
    result = ingest_policies()
    return {"status": "ok", **result}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_msg = (req.message or "").strip()

    # ── STEP 0: Out-of-scope detection ────────────────────────────────
    if _is_human_agent_request(user_msg):
        return ChatResponse(
            mode="escalate",
            answer=(
                "I understand you'd like to speak with a human agent. "
                "Here's how to reach a live agent for each airline:\n\n"
                "- **American Airlines**: 1-800-433-7300\n"
                "- **Delta Air Lines**: 1-800-221-1212\n"
                "- **United Airlines**: 1-800-864-8331\n"
                "- **DOT Aviation Consumer Protection**: 202-366-2220\n\n"
                "You can also file a formal complaint at [transportation.gov/airconsumer]"
                "(https://www.transportation.gov/airconsumer)."
            ),
            citations=[],
            debug={"case": "escalate", "reason": "human_agent_request", "slots": {}},
        )

    if _is_greeting(user_msg):
        return ChatResponse(
            mode="clarify",
            answer=(
                "Hi! I'm the Airline Dispute Assistant. I can help you with:\n\n"
                "- ✈️ **Flight refunds** — if your flight was canceled or changed\n"
                "- 🧳 **Baggage issues** — lost, delayed, or damaged bags\n"
                "- 📋 **DOT regulations** — your passenger rights\n\n"
                "Which airline and what happened?"
            ),
            citations=[],
            debug={"case": "greeting", "reason": "greeting_detected", "slots": {}},
        )

    if _is_unrelated(user_msg):
        return ChatResponse(
            mode="clarify",
            answer=(
                "I'm specialized in airline refunds and baggage disputes. "
                "I may not have the right information for that question. "
                "Could you describe a refund, cancellation, or baggage issue I can help with?"
            ),
            citations=[],
            debug={"case": "out_of_scope", "reason": "unrelated_topic", "slots": {}},
        )

    # ── STEP 1: Build intelligent conversation context ────────────────
    # CRITICAL: Get relevant context (full conversation for same issue)
    full_context = _get_relevant_context(
        user_msg, 
        req.conversation_history or [],
        max_tokens=2000
    )
    
    # Detect case and extract slots from FULL relevant context
    case = detect_case(full_context)
    slots = extract_slots(full_context, case)
    
    # Track if this is a new issue
    is_new = _is_new_issue(user_msg, req.conversation_history or [])

    # ── STEP 2: Clarify ONLY if absolutely critical slots missing ─────
    missing = missing_slots(slots)
    if missing:
        q = clarifying_question(case, missing, slots)
        return ChatResponse(
            mode="clarify",
            answer=q,
            citations=[],
            debug={
                "case": case,
                "slots": slots,
                "missing_slots": missing,
                "top_score": None,
                "confidence": None,
                "chunks_retrieved": 0,
                "used_llm": False,
                "decision": None,
                "is_new_issue": is_new,
                "context_length": len(full_context),
            },
        )

    # ── STEP 3: Build retrieval query ─────────────────────────────────
    query = build_retrieval_query(user_msg, slots)

    # ── STEP 4: Retrieve + rerank ──────────────────────────────────────
    airline_filter = (slots.get("airline") or "").strip()
    if airline_filter:
        top_chunks = retriever.search(query, airline_filter=airline_filter.lower())
        
        # Fallback: try without filter if results are poor
        if not top_chunks or top_chunks[0]["rerank_score"] < 0.2:
            top_chunks_no_filter = retriever.search(query, airline_filter=None)
            if top_chunks_no_filter and top_chunks_no_filter[0]["rerank_score"] > (top_chunks[0]["rerank_score"] if top_chunks else 0) + 0.1:
                top_chunks = top_chunks_no_filter
    else:
        top_chunks = retriever.search(query, airline_filter=None)
    
    top_score = float(top_chunks[0]["rerank_score"]) if top_chunks else 0.0

    # ── STEP 5: Evidence gate (LOWERED THRESHOLD) ──────────────────────
    if top_score < 0.15:
        return ChatResponse(
            mode="clarify",
            answer=(
                "I couldn't find a strong policy match for that. "
                "Could you clarify which airline this is for and what specifically happened?"
            ),
            citations=[],
            debug={
                "case": case,
                "slots": slots,
                "query_used": query,
                "top_score": round(top_score, 4),
                "confidence": "none",
                "chunks_retrieved": len(top_chunks),
                "used_llm": False,
                "decision": None,
                "is_new_issue": is_new,
                "context_length": len(full_context),
            },
        )

    # Adjusted confidence levels
    if top_score >= 0.4:
        confidence = "high"
    elif top_score >= 0.2:
        confidence = "medium"
    else:
        confidence = "low"

    # ── STEP 6: Decision engine (provides context, not conversation) ───
    decision = decision_engine.evaluate(
        case=case, slots=slots, confidence=confidence, top_score=top_score
    )

    # ── STEP 7: LLM grounded generation with better error handling ─────
    used_llm = True
    llm_error_detail = None
    try:
        # Build prompt
        prompt = _build_grounded_prompt(
            user_msg, 
            decision, 
            top_chunks, 
            confidence, 
            slots,
            req.conversation_history or []
        )
        
        # CRITICAL: Check prompt length BEFORE calling LLM
        # Ollama llama3.1:8b has 4096 token context (roughly 16,384 chars)
        # Leave room for response (600 tokens = 2400 chars)
        MAX_PROMPT_CHARS = 14000
        
        if len(prompt) > MAX_PROMPT_CHARS:
            print(f"⚠️ Prompt too long ({len(prompt)} chars), truncating to {MAX_PROMPT_CHARS}")
            # Truncate from the middle (keep beginning context and recent evidence)
            keep_start = 2000  # Keep instructions
            keep_end = MAX_PROMPT_CHARS - keep_start - 100
            prompt = prompt[:keep_start] + f"\n\n[...truncated {len(prompt) - MAX_PROMPT_CHARS} chars...]\n\n" + prompt[-keep_end:]
        
        # Log for debugging
        print(f"🤖 Calling LLM | Prompt: {len(prompt)} chars | Timeout: 180s")
        
        # Call LLM with increased timeout and better parameters
        answer = ollama_generate(
            prompt, 
            timeout_s=180,      # Increased from 120 for complex queries
            num_predict=600,    # Increased from 500 for fuller responses
            temperature=0.4     # Slightly increased for more natural variety
        )
        
        # Validate response
        if not answer or not answer.strip():
            raise RuntimeError("LLM returned empty response")
        
        # Log success
        print(f"✅ LLM success | Response: {len(answer)} chars")
        
    except requests.exceptions.Timeout:
        used_llm = False
        llm_error_detail = "LLM request timed out (>180s)"
        print(f"❌ {llm_error_detail}")
        answer = _build_answer(top_chunks, confidence)
        
    except requests.exceptions.ConnectionError as e:
        used_llm = False  
        llm_error_detail = f"Cannot connect to Ollama: {e}"
        print(f"❌ {llm_error_detail}")
        answer = _build_answer(top_chunks, confidence)
        
    except Exception as e:
        used_llm = False
        llm_error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"❌ LLM error: {llm_error_detail}")
        
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        
        answer = _build_answer(top_chunks, confidence)

    citations = _build_citations(top_chunks)
    mode = "answer"

    return ChatResponse(
        mode=mode,
        answer=answer,
        citations=citations,
        debug={
            "case": case,
            "slots": slots,
            "query_used": query,
            "top_score": round(top_score, 4),
            "confidence": confidence,
            "chunks_retrieved": len(top_chunks),
            "used_llm": used_llm,
            "llm_error": llm_error_detail,  # ADDED: show LLM error if any
            "decision": decision.to_dict(),
            "is_new_issue": is_new,
            "context_length": len(full_context),
            "conversation_turns": len(req.conversation_history or []) + 1,
        },
    )