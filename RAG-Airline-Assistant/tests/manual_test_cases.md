# Manual Test Cases — Airline Dispute RAG Assistant

Backend:
uvicorn backend.main:app --reload --port 8000

UI:
streamlit run ui/app.py

Click “Ingest” once before testing.

---

# REFUND CASES

## 1) Airline Cancelled + Weather

Prompt:
Delta cancelled my flight due to a snowstorm. I booked Basic Economy. Can I get a full refund?

Expected:
- mode: answer
- slots.airline_cancelled = yes
- confidence high
- citations present

---

## 2) Voluntary Cancellation

Prompt:
I want to cancel my United Airlines flight next week. The airline did not cancel it. Is my ticket refundable?

Expected:
- mode: answer
- airline_cancelled = no
- decision recommends checking fare type
- citations present

---

## 3) Missing Airline

Prompt:
My flight was cancelled. I want a refund.

Expected:
- mode: clarify
- asks which airline
- no citations

---

# BAGGAGE CASES

## 4) Lost Baggage

Prompt:
My American Airlines checked bag never arrived. I filed a report. What compensation can I get?

Expected:
- mode: answer
- baggage_status = lost
- compensation guidance
- citations present

---

## 5) Missing Baggage Type

Prompt:
My Delta bag issue happened yesterday. What can I do?

Expected:
- mode: clarify
- asks lost/delayed/damaged

---

# ESCALATION CASES

## 6) Legal Threat

Prompt:
This is fraud. I will sue and file a DOT complaint.

Expected:
- mode: escalate
- structured agent summary in debug
- no citations

---

## 7) Unsupported Airline

Prompt:
Air Example Airlines denied my refund.

Expected:
- mode: escalate OR clarify
- low evidence trigger
- no hallucinated policy

---

# QUALITY CHECK

Every answer must:
- Use citations if mode=answer
- Not invent policy
- Ask only 1–2 clarifying questions
- Provide clear next steps
