from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionResult:
    action: str  # "answer" | "clarify" | "escalate"
    recommended_action: str  # Context for LLM, not user-facing template
    options: List[str]  # Key points to cover, not rigid bullet list
    clarifying_question: Optional[str] = None
    reason: str = ""
    escalate_if: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "recommended_action": self.recommended_action,
            "options": self.options,
            "clarifying_question": self.clarifying_question,
            "reason": self.reason,
            "escalate_if": self.escalate_if or [],
        }


class DecisionEngine:
    """
    FIXED: Decision engine now provides CONTEXT for the LLM, not rigid templates.
    
    The LLM handles the actual conversation. The decision engine just provides:
    - Situational context (what kind of case is this?)
    - Key policy points to cover
    - Escalation signals to watch for
    
    The LLM weaves these into natural conversation.
    """

    def evaluate(
        self,
        *,
        case: str,
        slots: Dict[str, Any],
        confidence: str,
        top_score: float,
    ) -> DecisionResult:
        
        airline = (slots.get("airline") or "").strip()

        # CRITICAL: Don't block if airline is missing - let LLM ask naturally
        if not airline:
            return DecisionResult(
                action="answer",  # Changed from "clarify" - let LLM handle it
                recommended_action="Ask which airline naturally if needed to give specific guidance.",
                options=[
                    "Provide general refund/baggage rights that apply across airlines",
                    "Mention that specific policies vary by carrier",
                ],
                reason="Missing airline — letting LLM ask naturally in conversation",
                escalate_if=[],
            )

        # ── REFUND CASE ────────────────────────────────────────────────
        if case == "refund":
            airline_cancelled = slots.get("airline_cancelled", "unknown")
            schedule_change = slots.get("schedule_change", "unknown")
            refund_denied = slots.get("refund_denied", "unknown")

            # ESCALATION: Refund actively denied
            if refund_denied == "yes" or slots.get("cash_refund_refused") == "yes":
                return DecisionResult(
                    action="answer",  # LLM will handle escalation naturally
                    recommended_action=(
                        "The passenger reports being denied a refund they're entitled to. "
                        "Escalation guidance should be provided."
                    ),
                    options=[
                        "Document the denial in writing (email/chat transcript)",
                        "Request supervisor escalation with reference to cancellation rights",
                        "File DOT complaint with proof of cancellation and payment",
                    ],
                    reason="Refund denial reported — escalation context provided",
                    escalate_if=[],
                )

            # Airline cancelled or significant schedule change
            if airline_cancelled == "yes" or schedule_change == "yes":
                return DecisionResult(
                    action="answer",
                    recommended_action=(
                        "Airline-initiated cancellation or significant change. "
                        "Passenger likely entitled to full cash refund per DOT rules. "
                        "Emphasize: request explicit refund, not just credit/voucher."
                    ),
                    options=[
                        "Request cash/card refund explicitly",
                        "Don't have to accept voucher/credit if they don't want it",
                        "Keep all documentation (booking confirmation, cancellation notice)",
                        "If travel waiver exists, rebooking may also be an option",
                    ],
                    reason=f"Airline-initiated disruption | score={top_score:.3f}",
                    escalate_if=[
                        "Airline denies refund despite confirmed cancellation",
                        "Airline only offers credit/voucher and refuses cash",
                    ],
                )

            # Voluntary cancellation
            if airline_cancelled == "no":
                return DecisionResult(
                    action="answer",
                    recommended_action=(
                        "Passenger-initiated cancellation. Refund depends on fare type. "
                        "Refundable fares get cash refund. Non-refundable typically become credit. "
                        "24-hour rule may apply if recent booking."
                    ),
                    options=[
                        "Check ticket confirmation for fare rules (refundable vs non-refundable)",
                        "24-hour cancellation rule if booked recently (full refund)",
                        "Non-refundable tickets usually become travel credit minus fees",
                        "Travel waiver (if active) might allow fee-free changes",
                    ],
                    reason=f"Voluntary cancellation | score={top_score:.3f}",
                    escalate_if=["Refundable fare purchased but refund denied"],
                )

            # Unknown cancellation type - provide general guidance
            return DecisionResult(
                action="answer",
                recommended_action=(
                    "Cancellation context unclear. Provide general refund guidance "
                    "and naturally ask whether airline cancelled or passenger wants to cancel."
                ),
                options=[
                    "Airline cancellation = likely entitled to full refund",
                    "Voluntary cancellation = depends on fare type",
                    "Check for active travel waivers (weather/disruption)",
                ],
                reason=f"Refund case — cancellation type unknown | score={top_score:.3f}",
                escalate_if=[],
            )

        # ── BAGGAGE CASE ───────────────────────────────────────────────
        if case == "baggage":
            status = slots.get("baggage_status", "unknown")
            
            # Provide general baggage guidance - LLM will ask for specifics naturally
            if status == "unknown":
                return DecisionResult(
                    action="answer",
                    recommended_action=(
                        "Baggage issue but specific status unclear. Provide general guidance "
                        "and naturally ask if delayed/lost/damaged."
                    ),
                    options=[
                        "File baggage claim immediately and get reference number",
                        "Keep receipts for any essential purchases",
                        "Report timing varies: damaged (24hr domestic, 7d intl), delayed (immediately)",
                        "Bag fees may be refundable if delay/loss meets DOT thresholds",
                    ],
                    reason=f"Baggage case — status unknown | score={top_score:.3f}",
                    escalate_if=["Airline refuses to accept claim or provide reference"],
                )

            # Status-specific guidance
            guidance_map = {
                "lost": (
                    "Baggage reported lost. File claim, get reference number, ask about liability caps. "
                    "Airlines search 5-21 days before declaring officially lost."
                ),
                "delayed": (
                    "Baggage delayed. File report (Property Irregularity Report/PIR). "
                    "Keep receipts for essentials. Reimbursement for reasonable expenses. "
                    "Bag fees refundable if delay exceeds thresholds (12hr domestic, 15-30hr intl)."
                ),
                "damaged": (
                    "Baggage damaged. Report before leaving airport if possible. "
                    "Take photos. File damage report within 24hr (domestic) or 7d (international). "
                    "Airline may repair, replace, or compensate."
                ),
            }

            if status in guidance_map:
                return DecisionResult(
                    action="answer",
                    recommended_action=guidance_map[status],
                    options=[
                        "File claim at baggage service desk or online",
                        "Keep all receipts and documentation",
                        "Get reference number (PIR for delayed/lost, damage report number)",
                        "Ask airline about their specific claim procedures and deadlines",
                    ],
                    reason=f"Baggage {status} | score={top_score:.3f}",
                    escalate_if=[
                        "High-value items involved",
                        "Airline refuses to process claim",
                    ],
                )

        # ── FALLBACK ───────────────────────────────────────────────────
        return DecisionResult(
            action="answer",
            recommended_action=(
                "Issue outside main refund/baggage categories. Provide general guidance "
                "and suggest direct airline contact or DOT complaint if needed."
            ),
            options=[
                "Contact airline customer relations directly",
                "File DOT complaint if consumer rights issue",
            ],
            reason=f"Unsupported case: {case}",
            escalate_if=[],
        )