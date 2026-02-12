# 🛫 Airline Dispute RAG Assistant (Fully Local, Production-Style)

A fully local Retrieval-Augmented Generation (RAG) assistant designed to handle airline disputes intelligently.

This system:

- Extracts structured dispute details from user messages
- Retrieves grounded airline policy evidence
- Applies deterministic decision logic
- Generates citation-backed responses
- Asks clarifying questions when needed
- Escalates complex/legal cases to human agents

All components run locally. No external APIs. $0 cost.

---

# 🧠 Architecture Overview

User  
↓  
Slot Extraction  
↓  
Missing Info Detector  
↓  
Vector Retrieval (ChromaDB + e5 embeddings)  
↓  
Cross-Encoder Reranker  
↓  
Evidence Confidence Gate  
↓  
Decision Engine (Playbook Logic)  
↓  
Grounded LLM Response (Ollama)  
↓  
Citations + Debug Panel  

---

# 🔧 Tech Stack

## Core AI

- **LLM:** Ollama (mistral or llama3.1)
- **Embeddings:** sentence-transformers (`intfloat/e5-base-v2`)
- **Vector DB:** ChromaDB (local persistent store)
- **Reranker:** `BAAI/bge-reranker-base`

## Backend

- FastAPI
- Deterministic decision engine
- YAML-style playbook logic

## Frontend

- Streamlit
- Chat UI
- Evidence viewer
- Debug transparency panel

---

# 🎯 Key Capabilities

## 1️⃣ Intelligent Slot Extraction

From a message like:

> Delta cancelled my flight due to snow. I booked Basic Economy. Can I get a refund?

System extracts:

- case: refund
- airline: Delta
- airline_cancelled: yes
- weather_related: yes
- ticket_refundable: no

---

## 2️⃣ Clarifying Question System

If key details are missing:

User:  
> My flight was cancelled. I want refund.

System response:
- mode: clarify
- asks: “Which airline is this for?”

Prevents hallucination.

---

## 3️⃣ Evidence-Based Responses

LLM is constrained to:

- Use ONLY retrieved policy chunks
- Cite evidence like [1], [2]
- Avoid fabricating rules

All answers are grounded.

---

## 4️⃣ Decision Engine (Not Just Chat)

The system does not blindly answer.

It applies rule logic:

- Airline cancelled → Full refund eligible
- Voluntary cancel → Depends on fare
- Weather + waiver → Special handling
- Lost baggage → Compensation flow

Deterministic, not purely generative.

---

## 5️⃣ Escalation System

The assistant escalates when:

- Evidence confidence is too low
- Unsupported dispute type
- Decision engine flags escalation
- User expresses legal threat / regulatory complaint

Escalation produces:

- Structured agent summary
- Extracted slots
- Recommended next steps

---

# 📊 Modes

The system returns one of:

- `answer`
- `clarify`
- `escalate`

Cases (domain types):

- refund
- baggage
- disruption

Escalation is a response strategy — not a case.

---



