"""
backend/slots.py
----------------
Slot extraction + missing-info detection.
Rule-based now (fast + deterministic).
Later you can swap extract_slots() with Ollama structured JSON.
"""

from __future__ import annotations
from typing import Any, Dict, List
import re


AIRLINE_KEYWORDS = {
    "american": "American Airlines",
    "aa": "American Airlines",
    "delta": "Delta Airlines",
    "dl": "Delta Airlines",
    "united": "United Airlines",
    "ua": "United Airlines",
}


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _has_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def detect_airline(text: str) -> str:
    t = _norm(text)
    for k, v in AIRLINE_KEYWORDS.items():
        # word-boundary match for short codes like aa/dl/ua
        if len(k) <= 2:
            if re.search(rf"\b{k}\b", t):
                return v
        else:
            if k in t:
                return v
    return ""


def detect_case(full_context: str) -> str:
    t = _norm(full_context)

    baggage = ["baggage", "bag", "luggage", "suitcase", "checked bag", "lost bag", "delayed bag", "damaged bag"]
    refund = ["refund", "cancel", "canceled", "cancelled", "rebook", "schedule change", "credit", "voucher"]

    if _has_any(t, baggage):
        return "baggage"
    if _has_any(t, refund):
        return "refund"
    # default for demo
    return "refund"


def post_process_slots(full_context: str, slots: Dict[str, Any], case: str) -> Dict[str, Any]:
    """
    Deterministic slot overrides from obvious keywords/phrases.
    Goal: reduce unnecessary clarifying questions for clear user statements.
    """
    text = _norm(full_context)
    out = dict(slots or {})

    # -------------------------
    # Refund / disruption slots
    # -------------------------
    if case == "refund":
        # Airline cancelled?
        if out.get("airline_cancelled", "unknown") == "unknown":
            yes_cancel = [
                # Passive voice
                "my flight was cancelled", "my flight was canceled",
                "flight was cancelled", "flight was canceled",
                "flight got cancelled", "flight got canceled",
                # Active voice — airline as subject (covers "Delta cancelled my flight")
                "cancelled my flight", "canceled my flight",
                "cancelled the flight", "canceled the flight",
                # Generic
                "airline cancelled", "airline canceled",
                "they cancelled", "they canceled",
                "cancelled by", "canceled by",
                "carrier cancelled", "carrier canceled",
                # Refusing refund implies airline-initiated event
                "refusing to refund", "refused my refund", "denying my refund",
                "won't refund", "will not refund",
            ]
            no_cancel = [
                "i cancelled", "i canceled", "i want to cancel", "i plan to cancel",
                "i'm cancelling", "i am cancelling", "i need to cancel",
            ]
            if _has_any(text, yes_cancel):
                out["airline_cancelled"] = "yes"
            elif _has_any(text, no_cancel):
                out["airline_cancelled"] = "no"

        # Significant schedule change?
        if out.get("schedule_change", "unknown") == "unknown":
            yes_sched = [
                "schedule change", "changed my flight time", "changed my flight",
                "time changed", "moved my flight", "rescheduled", "rerouted",
                "changed the itinerary", "connection changed",
            ]
            no_sched = ["no schedule change", "no change in schedule", "schedule unchanged"]
            if _has_any(text, yes_sched) and not _has_any(text, no_sched):
                out["schedule_change"] = "yes"
            elif _has_any(text, no_sched):
                out["schedule_change"] = "no"

        # Weather-related?
        if out.get("weather_related", "unknown") == "unknown":
            yes_weather = [
                "snow", "snowstorm", "storm", "hurricane", "cyclone", "typhoon",
                "thunderstorm", "blizzard", "ice", "icy", "fog", "heavy rain", "weather",
            ]
            no_weather = ["not weather", "not due to weather", "weather is fine"]
            if _has_any(text, yes_weather) and not _has_any(text, no_weather):
                out["weather_related"] = "yes"
            elif _has_any(text, no_weather):
                out["weather_related"] = "no"

        # Travel waiver active?
        if out.get("travel_waiver_active", "unknown") == "unknown":
            yes_waiver = ["travel waiver", "waiver", "travel advisory", "travel alert", "weather waiver"]
            no_waiver = ["no waiver", "waiver not", "no travel waiver"]
            if _has_any(text, yes_waiver) and not _has_any(text, no_waiver):
                out["travel_waiver_active"] = "yes"
            elif _has_any(text, no_waiver):
                out["travel_waiver_active"] = "no"

        # Ticket refundable?
        if out.get("ticket_refundable", "unknown") == "unknown":
            yes_ref = ["refundable ticket", "fully refundable", "refundable fare"]
            no_ref = ["nonrefundable", "non-refundable", "basic economy", "no refund"]
            if _has_any(text, yes_ref):
                out["ticket_refundable"] = "yes"
            elif _has_any(text, no_ref):
                out["ticket_refundable"] = "no"

        # --- NEW: escalation signals ---
        if out.get("refund_denied", "unknown") == "unknown":
            if _has_any(text, [
                "denied my refund", "refund denied", "refused refund", "refusing refund",
                "won't refund", "will not refund", "not giving me a refund",
            ]):
                out["refund_denied"] = "yes"

        if out.get("cash_refund_refused", "unknown") == "unknown":
            if _has_any(text, [
                "only offering credit", "only credit", "only voucher", "only travel credit",
                "refusing cash refund", "won't give cash", "will not give cash",
                "no cash refund", "cash refund refused",
            ]):
                out["cash_refund_refused"] = "yes"

    # -------------------------
    # Baggage slots
    # -------------------------
    if case == "baggage":
        if out.get("baggage_status", "unknown") in ("unknown", "", None):
            delayed = ["baggage delayed", "bag delayed", "didn't arrive", "not arrived", "still not here", "missed bag"]
            lost = ["baggage lost", "bag lost", "never arrived", "missing bag", "lost luggage"]
            damaged = ["baggage damaged", "bag damaged", "broken suitcase", "damaged luggage", "wheel broke", "handle broke", "torn"]
            if _has_any(text, lost):
                out["baggage_status"] = "lost"
            elif _has_any(text, damaged):
                out["baggage_status"] = "damaged"
            elif _has_any(text, delayed):
                out["baggage_status"] = "delayed"

        if out.get("baggage_report_filed", "unknown") == "unknown":
            yes_report = ["filed a report", "filed report", "filed a claim", "submitted a claim", "reported it", "pir", "property irregularity report"]
            no_report = ["haven't reported", "have not reported", "didn't report", "not reported yet"]
            if _has_any(text, yes_report):
                out["baggage_report_filed"] = "yes"
            elif _has_any(text, no_report):
                out["baggage_report_filed"] = "no"

    return out


def extract_slots(full_context: str, case: str) -> Dict[str, Any]:
    """
    Basic rule-based slot extraction.
    """
    airline = detect_airline(full_context)

    if case == "refund":
        slots = {
            "case": "refund",
            "airline": airline,
            "airline_cancelled": "unknown",
            "travel_waiver_active": "unknown",
            "ticket_refundable": "unknown",
            "weather_related": "unknown",
            "schedule_change": "unknown",
            # NEW escalation signals
            "refund_denied": "unknown",
            "cash_refund_refused": "unknown",
        }
        return post_process_slots(full_context, slots, case)

    if case == "baggage":
        slots = {
            "case": "baggage",
            "airline": airline,
            "baggage_status": "unknown",
            "baggage_report_filed": "unknown",
            "bag_fee_refund": "unknown",
        }
        return post_process_slots(full_context, slots, case)

    return {"case": case, "airline": airline}


def missing_slots(slots: Dict[str, Any]) -> List[str]:
    """
    Only block retrieval when we have truly zero information.
    
    KEY PRINCIPLE: Let the LLM ask clarifying questions naturally.
    Only use hard template blocks for cases where retrieval is impossible.
    
    - airline: required (can't filter policies without it)
    - airline_cancelled: NOT required — LLM can handle ambiguity naturally
    - baggage_status: NOT required — LLM can ask naturally
    """
    # Only block if we have no airline at all — we can't retrieve any policy
    if not (slots.get("airline") or "").strip():
        return ["airline"]

    # Everything else: let retrieval + LLM handle it naturally
    return []


def clarifying_question(case: str, missing: List[str], slots: Dict[str, Any] | None = None) -> str:
    slots = slots or {}
    if "airline" in missing:
        return "Which airline is this for?"

    if case == "refund" and "airline_cancelled" in missing:
        return "Did the airline cancel or significantly change your flight, or are you looking to cancel it yourself?"

    if case == "baggage" and "baggage_status" in missing:
        return "Is your baggage delayed, lost, or damaged?"

    return "Can you share a bit more detail so I can check the right policy?"


def build_retrieval_query(user_msg: str, slots: Dict[str, Any]) -> str:
    """
    Build enriched retrieval query using known slots.
    """
    airline = (slots.get("airline") or "").strip()
    case = slots.get("case") or "refund"

    parts = [user_msg]

    if case == "refund":
        parts += ["refund", "cancellation", "refund policy"]
        if airline:
            parts.append(airline)
        if slots.get("airline_cancelled") == "yes":
            parts += ["airline cancelled", "involuntary cancellation", "cash refund"]
        if slots.get("airline_cancelled") == "no":
            parts += ["voluntary cancellation", "travel credit", "non-refundable"]
        if slots.get("schedule_change") == "yes":
            parts += ["significant schedule change", "delay", "reroute"]
        if slots.get("weather_related") == "yes":
            parts += ["weather", "travel waiver"]
    elif case == "baggage":
        parts += ["baggage", "lost", "delayed", "damaged", "compensation", "claim"]
        if airline:
            parts.append(airline)

    return " ".join([p for p in parts if p]).strip()