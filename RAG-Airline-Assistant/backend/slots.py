"""
backend/slots.py
----------------
Slot extraction and missing-info detection.
Rule-based for Day 1 (fast, deterministic).
Will be upgraded to Ollama structured JSON on Day 2.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ------------------------------------------------------------------ #
#  Airline detection
# ------------------------------------------------------------------ #
AIRLINE_KEYWORDS = {
    "american": "American Airlines",
    "aa": "American Airlines",
    "delta": "Delta Airlines",
    "dl": "Delta Airlines",
    "united": "United Airlines",
    "ua": "United Airlines",
}

def detect_airline(text: str) -> str | None:
    """Return normalized airline name or None if not mentioned."""
    t = text.lower()
    for keyword, name in AIRLINE_KEYWORDS.items():
        if keyword in t:
            return name
    return None


# ------------------------------------------------------------------ #
#  Case / intent detection
# ------------------------------------------------------------------ #
BAGGAGE_KEYWORDS = [
    "bag", "baggage", "luggage", "suitcase",
    "lost", "delayed bag", "missing bag", "damaged bag",
    "bag fee", "checked bag",
]

REFUND_KEYWORDS = [
    "refund", "cancel", "cancellation", "money back",
    "reimbur", "credit", "voucher", "schedule change",
    "delay", "disruption", "waiver",
]

def detect_case(text: str) -> str:
    """Detect whether the issue is baggage or refund/disruption."""
    t = text.lower()
    bag_hits = sum(1 for k in BAGGAGE_KEYWORDS if k in t)
    ref_hits = sum(1 for k in REFUND_KEYWORDS if k in t)
    return "baggage" if bag_hits > ref_hits else "refund"


# ------------------------------------------------------------------ #
#  Slot extraction
# ------------------------------------------------------------------ #
def extract_slots(text: str, case: str) -> Dict[str, Any]:
    """Extract structured slots from user message."""
    t = text.lower()
    slots: Dict[str, Any] = {
        "case": case,
        "airline": detect_airline(t),
    }

    if case == "refund":
        # Was the flight canceled by the airline?
        if any(p in t for p in ["airline cancel", "flight cancel", "they cancel", "was cancel", "got cancel"]):
            slots["airline_cancelled"] = "yes"
        elif any(p in t for p in ["i want to cancel", "i need to cancel", "i cancel", "voluntar"]):
            slots["airline_cancelled"] = "no"
        else:
            slots["airline_cancelled"] = "unknown"

        # Travel waiver
        slots["travel_waiver_active"] = "yes" if "waiver" in t else "unknown"

        # Ticket type
        if "refundable" in t and "non" not in t:
            slots["ticket_refundable"] = "yes"
        elif "non-refundable" in t or "nonrefundable" in t:
            slots["ticket_refundable"] = "no"
        else:
            slots["ticket_refundable"] = "unknown"

        # Weather
        slots["weather_related"] = (
            "yes" if any(k in t for k in ["storm", "weather", "hurricane", "snow", "flood"]) else "no"
        )

        # Significant schedule change
        slots["schedule_change"] = "yes" if any(k in t for k in ["schedule change", "time change", "reroute", "rerouted"]) else "unknown"

    if case == "baggage":
        # Baggage status
        if "lost" in t:
            slots["baggage_status"] = "lost"
        elif any(k in t for k in ["damage", "broken", "crack", "torn", "wheel"]):
            slots["baggage_status"] = "damaged"
        elif any(k in t for k in ["delay", "late", "didn't arrive", "not arrived"]):
            slots["baggage_status"] = "delayed"
        elif any(k in t for k in ["missing item", "pilfered", "stolen from bag"]):
            slots["baggage_status"] = "pilfered"
        else:
            slots["baggage_status"] = "unknown"

        # Claim filed
        slots["claim_filed"] = "yes" if any(k in t for k in ["filed", "claim", "reported"]) else "unknown"

        # Bag fee refund
        slots["bag_fee_refund"] = "yes" if any(k in t for k in ["fee", "charge", "paid for bag"]) else "unknown"

    return slots


# ------------------------------------------------------------------ #
#  Missing-info detection
# ------------------------------------------------------------------ #
def required_slots(case: str) -> List[str]:
    """Which slots must be known before we can search policy?"""
    if case == "refund":
        return ["airline_cancelled"]
    if case == "baggage":
        return ["baggage_status"]
    return []


def missing_slots(slots: Dict[str, Any]) -> List[str]:
    """Return list of required slots that are still unknown."""
    req = required_slots(slots["case"])
    return [r for r in req if slots.get(r, "unknown") in (None, "unknown", "")]


# ------------------------------------------------------------------ #
#  Clarifying questions
# ------------------------------------------------------------------ #
CLARIFYING_QUESTIONS: Dict[str, Dict[str, str]] = {
    "refund": {
        "airline_cancelled": (
            "Did the airline cancel or significantly change your flight, "
            "or are you looking to cancel it yourself?"
        ),
        "travel_waiver_active": (
            "Has the airline issued a travel waiver for your route or date "
            "(often due to weather)?"
        ),
    },
    "baggage": {
        "baggage_status": (
            "Is your baggage lost, delayed, damaged, or do you have missing items from inside?"
        ),
    },
}

def clarifying_question(case: str, missing: List[str]) -> str:
    """Generate a single clear clarifying question for the first missing slot."""
    if not missing:
        return ""
    first_missing = missing[0]
    q = CLARIFYING_QUESTIONS.get(case, {}).get(
        first_missing,
        "Could you provide more detail so I can find the right policy?",
    )
    return q


# ------------------------------------------------------------------ #
#  Query builder — turns slots into a richer retrieval query
# ------------------------------------------------------------------ #
def build_retrieval_query(user_text: str, slots: Dict[str, Any]) -> str:
    """
    Augment the raw user message with slot context for better retrieval.
    E.g. "my bag is lost" → "lost baggage claim compensation Delta Air Lines"
    """
    parts = [user_text]

    case = slots.get("case", "")
    airline = slots.get("airline")

    if case == "baggage":
        status = slots.get("baggage_status", "")
        if status and status != "unknown":
            parts.append(f"{status} baggage")
        parts.append("baggage claim reimbursement")

    if case == "refund":
        if slots.get("airline_cancelled") == "yes":
            parts.append("airline cancelled flight refund")
        elif slots.get("airline_cancelled") == "no":
            parts.append("voluntary cancellation non-refundable ticket")
        if slots.get("schedule_change") == "yes":
            parts.append("significant schedule change refund")
        parts.append("refund policy")

    if airline:
        parts.append(airline)

    return " ".join(parts)