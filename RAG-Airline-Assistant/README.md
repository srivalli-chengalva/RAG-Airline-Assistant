# ✈️ Airline Dispute RAG Assistant

A fully local, $0-cost RAG (Retrieval-Augmented Generation) system for resolving airline disputes — refunds, disruptions, and baggage issues — grounded in real airline policies and DOT regulations.

---

## 🧠 What It Does

Users describe their airline dispute in plain language. The system:
1. **Detects** the issue type (refund/disruption or baggage)
2. **Extracts slots** (airline name, cancellation type, baggage status, etc.)
3. **Asks clarifying questions** if required information is missing
4. **Retrieves** the most relevant policy chunks from a local vector database
5. **Reranks** results using a cross-encoder for production-grade precision
6. **Returns** a grounded answer with citations and confidence scores

No OpenAI. No API costs. Runs entirely on your machine.

---

## 🏗️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM (Day 2) | Ollama — `llama3.1:8b` | Slot extraction, answer generation |
| Embeddings | `intfloat/e5-base-v2` | Semantic chunk encoding |
| Vector DB | ChromaDB (local) | Persistent similarity search |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder precision boost |
| Backend | FastAPI | REST API — `/chat`, `/ingest` |
| Frontend | Streamlit | Chat UI with evidence panel |
| Decision Logic | YAML Playbooks (Day 3) | Eligibility rules engine |

---

## 📁 Project Structure

```
airline-rag-assistant/
│
├── backend/
│   ├── __init__.py          # Package marker
│   ├── config.py            # All settings (models, paths, thresholds)
│   ├── ingestion.py         # Policy chunking + embedding (module version)
│   ├── retrieval.py         # Two-stage retrieval: dense search + reranker
│   ├── slots.py             # Slot extraction + missing-info detector
│   └── main.py              # FastAPI app — /health, /ingest, /chat
│
├── data/
│   └── policies/
│       ├── _meta/
│       │   └── authority_cross_reference_2026-02.txt   # Internal rules (DO_NOT_CITE)
│       ├── American_Airlines/
│       │   ├── american_checked_baggage_policy_2026-02.txt
│       │   ├── american_optional_service_fees_2026-02.txt
│       │   └── american_refund_policy_2026-02.txt
│       ├── Delta_Air_Lines/
│       │   ├── delta_baggage_2026-02.txt
│       │   └── delta_refund_policy_2026-02.txt
│       ├── United_Airlines/
│       │   ├── united_baggage_policy_2026-02.txt
│       │   └── united_refund_policy_2026-02.txt
│       └── DOT/
│           ├── dot_baggage_2026-02.txt
│           └── dot_refunds_2026-02.txt
│
├── playbooks/               # YAML decision rules (Day 3)
│
├── scripts/
│   ├── ingest_docs.py       # CLI: run once to populate vector store
│   └── check_store.py       # CLI: verify vector store contents
│
├── ui/
│   └── app.py               # Streamlit chat interface
│
├── vector_store/            # Auto-created by ChromaDB — DO NOT commit to Git
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running (for Day 2 LLM features)

```bash
ollama pull llama3.1:8b
```

### 2. Clone and set up environment

```bash
git clone <your-repo-url>
cd airline-rag-assistant

# Create virtual environment
python -m venv .venv

# Activate — Mac/Linux
source .venv/bin/activate

# Activate — Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Ingest policy documents

Run from the project root:

```bash
python scripts/ingest_docs.py
```

Expected output:
```
✅ Ingestion complete
   Files ingested:  10
   Chunks ingested: 118
   Vector store:    vector_store/
```

### 4. Verify vector store (optional)

```bash
python scripts/check_store.py
```

### 5. Start the backend API

```bash
uvicorn backend.main:app --reload
```

API docs available at: http://127.0.0.1:8000/docs

### 6. Start the Streamlit UI

Open a second terminal (with `.venv` activated):

```bash
streamlit run ui/app.py
```

UI opens at: http://localhost:8501

---

## 🔄 RAG Pipeline (How It Works)

```
User message
     │
     ▼
detect_case()          → "refund" or "baggage"
     │
     ▼
extract_slots()        → airline, cancellation type, baggage status, etc.
     │
     ▼
missing_slots()?       → ask clarifying question if required info missing
     │
     ▼
build_retrieval_query() → enrich query with slot context
     │
     ▼
retriever.retrieve()   → Stage 1: dense vector search (top 12 candidates)
     │
     ▼
retriever.rerank()     → Stage 2: cross-encoder reranking (top 5)
     │
     ▼
evidence_gate()        → check confidence threshold
     │
     ▼
build_answer()         → grounded response with citations
```

---

## 🗄️ Policy Document Format

Each policy file uses a structured format for reliable ingestion:

```
SOURCE: American Airlines
URL: https://www.aa.com/...
CAPTURED_ON: 2026-02-12
AUTHORITY: AIRLINE          ← or REGULATOR for DOT files
DOMAIN: BAGGAGE             ← or REFUND, BAGGAGE_FEES, etc.

SECTION: Delayed Baggage Procedures
[policy content here]

SECTION: Lost Baggage
[policy content here]
```

Files in `_meta/` with `DO_NOT_CITE: TRUE` are used for internal retrieval logic but are never shown to users.

---

## 🎛️ Configuration

All tunable settings live in `backend/config.py`:

| Setting | Default | Description |
|---|---|---|
| `embed_model` | `intfloat/e5-base-v2` | Embedding model |
| `reranker_model` | `BAAI/bge-reranker-base` | Cross-encoder reranker |
| `retrieval_top_k` | `12` | Candidates fetched from vector DB |
| `rerank_top_n` | `5` | Kept after reranking |
| `rerank_threshold_none` | `0.30` | Below → ask for clarification |
| `rerank_threshold_low` | `0.50` | Below → low confidence warning |
| `ollama_model` | `llama3.1:8b` | LLM for generation (Day 2) |

---

## 🧪 Example Queries

| Query | Expected Behavior |
|---|---|
| `"My Delta flight was canceled, can I get a refund?"` | Returns Delta + DOT refund policy with high confidence |
| `"American Airlines lost my bag"` | Returns AA + DOT lost baggage policy |
| `"I have a baggage issue"` | Asks: *"Is your baggage lost, delayed, or damaged?"* |
| `"United delayed my bag by 2 days, can I be reimbursed?"` | Returns United + DOT delayed baggage reimbursement policy |
| `"What does DOT say about significant schedule changes?"` | Returns DOT refund regulation sections |

---

## 🗺️ Roadmap

- [x] **Day 1** — Ingestion, retrieval, reranker, slot extraction, FastAPI, Streamlit UI
- [ ] **Day 2** — Ollama LLM generation (grounded answers, not just excerpts)
- [ ] **Day 3** — YAML playbook decision engine (eligibility rules)
- [ ] **Day 4** — Evaluation scripts (Recall@k, retrieval accuracy)
- [ ] **Day 5** — Conversation memory + multi-turn slot tracking

---

## ⚠️ Important Notes

- `vector_store/` is excluded from Git (see `.gitignore`). Run `ingest_docs.py` after cloning.
- Policy files are snapshots captured in February 2026. Always verify current policies directly with airlines or DOT.
- This system is for **informational purposes only** and does not constitute legal advice.

---

## 📄 License

MIT License — see `LICENSE` for details.