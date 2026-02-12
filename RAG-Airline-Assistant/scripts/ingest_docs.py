import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ----------------------------
# Chunking (SECTION-aware)
# ----------------------------
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    SECTION-aware chunking:
    - Strips front-matter header block (SOURCE / URL / CAPTURED_ON lines)
    - Keeps SECTION blocks together when possible.
    - Falls back to sentence-ish splitting for long sections.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # FIXED: Strip front-matter before chunking (same as backend)
    lines = text.splitlines()
    content_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "" or line.strip().startswith("SECTION:"):
            content_start = i
            break
    text = "\n".join(lines[content_start:]).strip()

    # Split into sections by "SECTION:" markers if present
    # Otherwise treat whole document as one section
    sections = re.split(r"\n(?=SECTION:)", text) if "SECTION:" in text else [text]

    chunks: List[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
            continue

        # Long section -> split with overlap
        i = 0
        while i < len(section):
            chunk = section[i : i + max_chars]

            # try to end at a sentence boundary
            if i + max_chars < len(section):
                last_period = chunk.rfind(". ")
                if last_period > int(max_chars * 0.7):
                    chunk = chunk[: last_period + 1]

            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            i += max(len(chunk) - overlap, 1)

    return chunks


# ----------------------------
# Metadata parsing
# ----------------------------
def parse_front_matter(text: str) -> Dict[str, str]:
    """
    Reads KEY: VALUE lines near the top of snapshot files.
    Example:
      SOURCE: ...
      URL: ...
      CAPTURED_ON: ...
      AUTHORITY: ...
      DOMAIN: ...
      DO_NOT_CITE: TRUE
    """
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


# ----------------------------
# Ingestion
# ----------------------------
def ingest_policies(
    policies_dir: str = "data/policies",
    persist_dir: str = "vector_store",
    collection_name: str = "policies",
    embed_model: str = "intfloat/e5-base-v2",
    max_chars: int = 900,
    overlap: int = 150,
) -> Tuple[int, int]:
    policies_root = Path(policies_dir)
    if not policies_root.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_root.resolve()}")

    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # FIXED: Delete and recreate collection for clean re-ingest
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(embed_model)

    files_ingested = 0
    chunks_ingested = 0

    # Recursively read .txt files (sorted for consistent order)
    for file_path in sorted(policies_root.rglob("*.txt")):
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue

        front = parse_front_matter(raw)
        path_md = infer_path_metadata(file_path, policies_root)

        # Merge metadata (front-matter overrides path inference)
        do_not_cite = normalize_bool(front.get("do_not_cite", "")) or bool(path_md.get("do_not_cite", False))

        # FIXED: Normalize airline to lowercase for case-insensitive filtering
        airline_raw = front.get("airline") or str(path_md.get("airline") or "")
        airline_normalized = airline_raw.strip().lower()

        metadata_base = {
            "source_file": str(file_path).replace("\\", "/"),
            "source": front.get("source") or "",
            "url": front.get("url") or "",
            "captured_on": front.get("captured_on") or "",
            "authority": front.get("authority") or str(path_md.get("authority") or ""),
            "airline": airline_normalized,  # FIXED: was not normalized
            "domain": front.get("domain") or str(path_md.get("domain") or ""),
            "do_not_cite": do_not_cite,
        }

        chunks = chunk_text(raw, max_chars=max_chars, overlap=overlap)
        if not chunks:
            continue

        # e5 requires prefixes for best retrieval quality
        passages = [f"passage: {c}" for c in chunks]
        embeddings = embedder.encode(passages, normalize_embeddings=True).tolist()

        ids = []
        metadatas = []
        documents = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
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

    return files_ingested, chunks_ingested


if __name__ == "__main__":
    files, chunks = ingest_policies()
    print(f"\n✅ Ingestion complete")
    print(f"   Files ingested:  {files}")
    print(f"   Chunks ingested: {chunks}")
    print(f"   Vector store:    vector_store/ (local only; should NOT be pushed to GitHub)\n")