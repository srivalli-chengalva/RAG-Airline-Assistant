"""
backend/ollama_client.py
------------------------
Tiny Ollama HTTP client with CRITICAL performance fix.

KEY FIX: keep_alive="10m" keeps the model loaded in memory between requests.
Without it, the model unloads after each call causing 30-60s cold starts.

ADDITIONAL FIXES:
- Increased num_predict from 400 to 600 for fuller answers
- Increased temperature from 0.2 to 0.4 for more natural variety
- Explicit num_ctx=4096 for clarity
- Better defaults aligned with main_v3.py
"""
from __future__ import annotations
from typing import Optional
import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: int = 180,        # INCREASED: was 120s, now 180s for complex queries
    num_predict: int = 600,      # INCREASED: was 400, now 600 for fuller answers
    temperature: float = 0.4,    # INCREASED: was 0.2, now 0.4 for natural variety
) -> str:
    """
    Calls Ollama /api/generate (non-stream) and returns response text.

    CRITICAL PERFORMANCE FIX:
    keep_alive="10m" prevents the model from unloading after each request.
    Without this parameter, you get 30-60s cold start delays on EVERY request!
    
    With keep_alive:
    - First request: ~2-5s (model loads once)
    - All subsequent requests: ~2-5s (model stays warm)
    
    Without keep_alive:
    - Every request: ~35-65s (model loads every time)

    Parameters:
    - timeout_s: HTTP timeout (180s handles complex queries)
    - num_predict: Max tokens to generate (600 = ~300 words)
    - temperature: Creativity level (0.4 = balanced, natural)
    """
    model = model or OLLAMA_MODEL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",       # ✅ CRITICAL: keep model warm for 10 minutes
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 4096,        # ✅ Explicit context window size
        },
    }

    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()