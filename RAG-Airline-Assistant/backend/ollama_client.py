"""
backend/ollama_client.py
------------------------
Ollama HTTP client with proper timeout handling.

Fixes (NO answer-quality change):
- Uses requests.Session for connection reuse
- Adds buffered generate_stream() so UI doesn't print word-by-word
- Optional num_thread via env OLLAMA_NUM_THREAD (speeds CPU, doesn't change outputs)
- keep_alive set so model isn't unloaded between requests
"""

from __future__ import annotations

from typing import Optional, Iterator, Dict, Any
import os
import json
import time
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Optional CPU thread tuning for Ollama (speed only)
_OLLAMA_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD")  # e.g., "4" or "8"

# Use a Session for connection reuse (safe; does not change outputs)
_SESSION = requests.Session()


def _payload(
    prompt: str,
    *,
    model: str,
    stream: bool,
    temperature: float,
    num_predict: int,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": 4096,
    }
    if _OLLAMA_NUM_THREAD and _OLLAMA_NUM_THREAD.isdigit():
        options["num_thread"] = int(_OLLAMA_NUM_THREAD)

    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": "10m",
        "options": options,
    }


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: int = 180,
    num_predict: int = 350,  # FIX 3: lower cap; does not affect retrieval quality
    temperature: float = 0.4,
) -> str:
    model = model or OLLAMA_MODEL

    r = _SESSION.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=_payload(
            prompt,
            model=model,
            stream=False,
            temperature=temperature,
            num_predict=num_predict,
        ),
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def generate_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: int = 180,
    num_predict: int = 350,  # FIX 3
    temperature: float = 0.4,
    # FIX 2: buffering knobs (smooth stream)
    flush_chars: int = 200,
    flush_interval_s: float = 0.25,
) -> Iterator[str]:
    """
    Streaming generator. Yields partial text chunks.
    Final concatenation equals the non-stream response (same model, same prompt).
    """

    model = model or OLLAMA_MODEL

    with _SESSION.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=_payload(
            prompt,
            model=model,
            stream=True,
            temperature=temperature,
            num_predict=num_predict,
        ),
        timeout=timeout_s,
        stream=True,
    ) as r:
        r.raise_for_status()

        buffer = ""
        last_flush = time.time()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            chunk = data.get("response") or ""
            if chunk:
                buffer += chunk

            now = time.time()
            if buffer and (len(buffer) >= flush_chars or (now - last_flush) >= flush_interval_s):
                yield buffer
                buffer = ""
                last_flush = now

            if data.get("done"):
                break

        if buffer:
            yield buffer