from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionResult:
    # "answer" | "clarify" | "escalate"
    action: str

    # What we recommend the user do next (shown to user)
    recommended_action: str

    # Concrete steps the user can take (shown to user)
    options: List[str]

    # If we need to ask something before proceeding
    clarifying_question: Optional[str] = None

    # Why we made this decision (debug panel only)
    reason: str = ""

    # Conditions that should trigger escalation
    escalate_if: List[str] = field(default_factory=list)


class DecisionEngine:
    """
    Deterministic controller — decides what the system does next.
    Does NOT generate language. Does NOT retrieve. Does NOT interpret policy.

    Inputs:  case, slots, confidence, top_score
    Output:  DecisionResult (action + guidance for the user)

    Future: load YAML playbooks here for richer eligibility rules.
    """

    def evaluate(
        self,
        *,
        case: str,
        slots: Dict[str, Any],
        confidence: str,
        top_score: float,
    ) -> DecisionResult:

        # ── Guard: no evidence ────────────────────────────────────────
        if confidence == "none":
            return DecisionResult(
                action="clarify",
                recommended_action="I need a bit more detail to find the right policy.",
                options=[],
                clarifying_question=(
                    "Which airline is this for, and did the airline cancel "
                    "the flight or are you looking to cancel it yourself?"
                ),
                reason="Evidence gate failed — no strong policy match.",
                escalate_if=["User is unable to provide airline or cancellation type"],
            )

        # ── Read slots ─────────────────────────────────────────────────
        airline = (slots.get("airline") or "").strip()
        # FIX: correct slot name is airline_cancelled (not cancellation_type)
        airline_cancelled = (slots.get("airline_cancelled") or "unknown").strip()
        baggage_status = (slots.get("baggage_status") or "unknown").strip()
        weather_related = (slots.get("weather_related") or "no").strip()
        travel_waiver = (slots.get("travel_waiver_active") or "unknown").strip()
        schedule_change = (slots.get("schedule_change") or "unknown").strip()

        # ── Refund / Disruption ────────────────────────────────────────
        if case == "refund":
            return self._evaluate_refund(
                airline=airline,
                airline_cancelled=airline_cancelled,
                weather_related=weather_related,
                travel_waiver=travel_waiver,
                schedule_change=schedule_change,
                top_score=top_score,
            )

        # ── Baggage ────────────────────────────────────────────────────
        if case == "baggage":
            return self._evaluate_baggage(
                airline=airline,
                baggage_status=baggage_status,
                top_score=top_score,
            )

        # ── Fallback ───────────────────────────────────────────────────
        return DecisionResult(
            action="escalate",
            recommended_action=(
                "This dispute type isn't covered by the current policy set. "
                "I recommend contacting the airline directly or filing a DOT complaint."
            ),
            options=[
                "File a complaint at transportation.gov/airconsumer",
                "Contact the airline's customer relations team directly",
            ],
            reason=f"Unsupported case='{case}'.",
            escalate_if=[],
        )

    # ── Refund sub-evaluator ───────────────────────────────────────────
    def _evaluate_refund(
        self,
        *,
        airline: str,
        airline_cancelled: str,
        weather_related: str,
        travel_waiver: str,
        schedule_change: str,
        top_score: float,
    ) -> DecisionResult:

        # Airline canceled or significant schedule change → strong refund right
        if airline_cancelled == "yes" or schedule_change == "yes":
            rec = (
                "You are likely entitled to a full refund to your original payment method. "
                "Request a refund explicitly — do not accept a voucher or credit unless you prefer it."
            )
            opts = [
                "Request a cash/card refund, not just travel credit — DOT rules require this option.",
                "If the airline pushes back, cite the DOT significant change/cancellation rule.",
                "Keep all booking confirmation and cancellation emails as documentation.",
            ]
            if travel_waiver == "yes":
                opts.insert(0, "A travel waiver is active — you may also have fee-free rebooking as an option.")
            escalate = [
                "Airline denies refund despite confirmed cancellation",
                "Airline offers only credit/voucher and refuses cash refund",
            ]

        # Weather related but no waiver confirmed
        elif weather_related == "yes" and travel_waiver == "unknown":
            rec = (
                "Check if the airline has issued a travel waiver for your route and date. "
                "Waivers unlock fee-free rebooking and sometimes refunds."
            )
            opts = [
                "Visit the airline's website to check for active travel waivers.",
                "If a waiver exists, rebook within the waiver window to avoid fees.",
                "If no waiver, refund eligibility depends on your fare type.",
            ]
            escalate = ["No waiver found but flight significantly disrupted by weather"]

        # Voluntary cancellation
        elif airline_cancelled == "no":
            rec = (
                "For voluntary cancellations, your refund depends on your fare type. "
                "Refundable fares get a full refund. Non-refundable fares typically get travel credit minus fees."
            )
            opts = [
                "Check your ticket confirmation for fare type (refundable vs non-refundable).",
                "Cancel before departure to preserve at least travel credit value.",
                "If within 24 hours of booking, you are entitled to a full refund under DOT rules.",
            ]
            escalate = ["User purchased refundable fare but airline denies refund"]

        # Unknown cancellation type — guidance without blocking
        else:
            rec = (
                "To give you the most accurate guidance, it helps to know: "
                "did the airline cancel your flight, or are you looking to cancel yourself?"
            )
            opts = [
                "If airline canceled: you are likely entitled to a full refund.",
                "If you want to cancel: refund depends on your fare type and timing.",
                "Either way, check for active travel waivers before taking any action.",
            ]
            escalate = []

        return DecisionResult(
            action="answer",
            recommended_action=rec,
            options=opts,
            reason=f"Refund flow | airline_cancelled={airline_cancelled} | score={top_score:.3f}",
            escalate_if=escalate,
        )

    # ── Baggage sub-evaluator ──────────────────────────────────────────
    def _evaluate_baggage(
        self,
        *,
        airline: str,
        baggage_status: str,
        top_score: float,
    ) -> DecisionResult:

        if baggage_status == "lost":
            rec = (
                "File a lost baggage claim immediately if you haven't already. "
                "Airlines typically search for 5–21 days before declaring a bag officially lost."
            )
            opts = [
                "File a claim at the baggage service desk or online — get a reference number.",
                "Keep receipts for any essential purchases made while waiting.",
                "Ask the airline for their liability cap — DOT limits apply for international flights.",
                "If the bag isn't found within 21 days, request compensation under the Montreal Convention.",
            ]
            escalate = [
                "High-value items were in the bag (electronics, jewelry, medications)",
                "Airline refuses to open a claim or provide a reference number",
            ]

        elif baggage_status == "delayed":
            rec = (
                "Your bag is delayed — the airline is responsible for locating and delivering it. "
                "You may claim reimbursement for reasonable essential purchases in the meantime."
            )
            opts = [
                "Keep all receipts for essentials (toiletries, clothing, medications).",
                "Report the delay at the baggage service office and get a Property Irregularity Report (PIR).",
                "Ask the airline for their delayed baggage reimbursement policy and daily cap.",
                "Bag fees may be refundable if the delay exceeds DOT/airline thresholds.",
            ]
            escalate = [
                "Delay exceeds 12 hours domestic / 15–30 hours international",
                "Airline refuses reimbursement for documented essential purchases",
            ]

        elif baggage_status == "damaged":
            rec = (
                "Report damaged baggage before leaving the airport if possible. "
                "Most airlines require claims within 24 hours (domestic) or 7 days (international)."
            )
            opts = [
                "File a damage report at the airport baggage service office immediately.",
                "Take photos of the damage before leaving the airport.",
                "Keep your baggage tags and boarding passes as proof.",
                "Ask whether the airline will repair, replace, or compensate for the bag.",
            ]
            escalate = [
                "Damage was not caused by passenger — airline disputes liability",
                "Airline refuses to process claim within required time window",
            ]

        elif baggage_status == "pilfered":
            rec = (
                "File a claim for missing items as soon as possible. "
                "Note that airlines typically exclude valuables like electronics, jewelry, and cash."
            )
            opts = [
                "File a missing items report at the baggage office or online.",
                "Provide a detailed list of missing items with estimated values.",
                "Note that valuable items (electronics, jewelry, cash) are often excluded from coverage.",
                "Keep a copy of your original packing list if you have one.",
            ]
            escalate = [
                "High-value items missing and airline denies claim",
            ]

        else:
            # baggage_status unknown — give general guidance
            rec = "To give you accurate guidance, could you confirm: is your baggage lost, delayed, damaged, or do you have items missing from inside?"
            opts = [
                "Lost: file a claim immediately, airlines search 5–21 days.",
                "Delayed: keep receipts for essentials, file a delay report.",
                "Damaged: report before leaving the airport.",
                "Missing items: file a pilferage claim within 24 hours.",
            ]
            escalate = []

        return DecisionResult(
            action="answer",
            recommended_action=rec,
            options=opts,
            reason=f"Baggage flow | baggage_status={baggage_status} | score={top_score:.3f}",
            escalate_if=escalate,
        )