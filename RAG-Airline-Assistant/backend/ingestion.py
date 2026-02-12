"""
backend/ingestion.py
--------------------
Module version of ingestion — called by FastAPI's /ingest endpoint.
The standalone scripts/ingest_docs.py is for one-off CLI runs.
Both share the same core logic but this one is importable.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .config import settings


# ------------------------------------------------------------------ #
#  Chunking
# ------------------------------------------------------------------ #
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    SECTION-aware chunking:
    - Strips front-matter header block (SOURCE / URL / CAPTURED_ON lines)
    - Keeps SECTION blocks together when possible
    - Falls back to sentence-boundary splitting for long sections
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Strip front-matter: skip lines until first blank line OR first SECTION:
    lines = text.splitlines()
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "" or line.strip().startswith("SECTION:"):
            content_start = i
            break
    text = "\n".join(lines[content_start:]).strip()

    # Split on SECTION markers if present
    sections = re.split(r"\n(?=SECTION:)", text) if "SECTION:" in text else [text]

    chunks: List[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
            continue

        # Long section → split with overlap
        i = 0
        while i < len(section):
            chunk = section[i : i + max_chars]

            # Try to end at a sentence boundary
            if i + max_chars < len(section):
                last_period = chunk.rfind(". ")
                if last_period > int(max_chars * 0.7):
                    chunk = chunk[: last_period + 1]

            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            i += max(len(chunk) - overlap, 1)

    return chunks


# ------------------------------------------------------------------ #
#  Metadata helpers
# ------------------------------------------------------------------ #
def parse_front_matter(text: str) -> Dict[str, str]:
    """Read KEY: VALUE lines from the top of a policy file."""
    meta: Dict[str, str] = {}
    for line in text.splitlines()[:40]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k:
            meta[k] = v
    return meta


def infer_path_metadata(file_path: Path, policies_root: Path) -> Dict[str, object]:
    """Infer airline / authority from folder structure."""
    rel = file_path.relative_to(policies_root)
    top = rel.parts[0] if rel.parts else ""
    top_lower = top.lower()

    md: Dict[str, object] = {}
    if top_lower == "_meta":
        md["airline"] = "INTERNAL"
        md["authority"] = "INTERNAL_META"
        md["domain"] = "META_POLICY"
        md["do_not_cite"] = True
    elif "dot" in top_lower:
        md["airline"] = "DOT"
        md["authority"] = "REGULATOR"
        md["do_not_cite"] = False
    else:
        md["airline"] = top.replace("_", " ").strip()
        md["authority"] = "AIRLINE"
        md["do_not_cite"] = False

    return md


def normalize_bool(val: str) -> bool:
    return str(val).strip().lower() in {"true", "yes", "1"}


# ------------------------------------------------------------------ #
#  Main ingestion function
# ------------------------------------------------------------------ #
def ingest_policies(
    policies_dir: str | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    embed_model: str | None = None,
    max_chars: int = 900,
    overlap: int = 150,
) -> Dict[str, int]:
    """
    Ingest all .txt policy files into ChromaDB.
    Uses settings defaults when parameters are not provided.
    Returns {"ingested_files": N, "ingested_chunks": N}
    """
    policies_root = Path(policies_dir or settings.data_dir)
    persist_path = str(persist_dir or settings.chroma_dir)
    col_name = collection_name or settings.collection_name
    model_name = embed_model or settings.embed_model

    if not policies_root.exists():
        raise FileNotFoundError(
            f"Policies directory not found: {policies_root.resolve()}"
        )

    os.makedirs(persist_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Delete and recreate collection for a clean re-ingest
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(model_name)

    files_ingested = 0
    chunks_ingested = 0

    for file_path in sorted(policies_root.rglob("*.txt")):
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue

        front = parse_front_matter(raw)
        path_md = infer_path_metadata(file_path, policies_root)

        do_not_cite = normalize_bool(front.get("do_not_cite", "")) or bool(
            path_md.get("do_not_cite", False)
        )

        # FIXED: Normalize airline to lowercase for consistent filtering
        # This ensures "Delta Airlines" from user matches "delta airlines" in DB
        airline_raw = front.get("airline") or str(path_md.get("airline") or "")
        airline_normalized = airline_raw.strip().lower()

        metadata_base = {
            "source_file": str(file_path).replace("\\", "/"),
            "source": front.get("source") or "",
            "url": front.get("url") or "",
            "captured_on": front.get("captured_on") or "",
            "authority": front.get("authority") or str(path_md.get("authority") or ""),
            "airline": airline_normalized,  # FIXED: was inline .lower() that wasn't clear
            "domain": front.get("domain") or str(path_md.get("domain") or ""),
            "do_not_cite": do_not_cite,
        }

        chunks = chunk_text(raw, max_chars=max_chars, overlap=overlap)
        if not chunks:
            continue

        passages = [f"passage: {c}" for c in chunks]
        embeddings = embedder.encode(passages, normalize_embeddings=True).tolist()

        ids, metadatas, documents = [], [], []
        for i, (chunk, _emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{file_path.stem}__{i}__{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            md = dict(metadata_base)
            md["chunk_index"] = i
            metadatas.append(md)
            documents.append(chunk)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        files_ingested += 1
        chunks_ingested += len(chunks)

    return {"ingested_files": files_ingested, "ingested_chunks": chunks_ingested}