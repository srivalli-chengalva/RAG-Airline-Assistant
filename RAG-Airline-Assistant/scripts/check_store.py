import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="vector_store",
    settings=Settings(anonymized_telemetry=False),
)

col = client.get_or_create_collection("policies")
count = col.count()
print(f"\n✅ Total chunks in vector store: {count}")

# Peek at first 3 chunks
results = col.peek(limit=3)
print(f"\n📄 Sample chunks:\n")
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
    print(f"--- Chunk {i} ---")
    print(f"Source:    {meta.get('source_file', 'unknown')}")
    print(f"Airline:   {meta.get('airline', 'unknown')}")
    print(f"Authority: {meta.get('authority', 'unknown')}")
    print(f"Text:      {doc[:120]}...")
    print()