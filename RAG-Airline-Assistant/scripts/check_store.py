import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings

# Added error handling
try:
    client = chromadb.PersistentClient(
        path="vector_store",
        settings=Settings(anonymized_telemetry=False),
    )

    col = client.get_or_create_collection("policies")
    count = col.count()
    
    print(f"\n{'='*60}")
    print(f"  VECTOR STORE STATUS")
    print(f"{'='*60}")
    print(f"\n✅ Total chunks in vector store: {count}")
    
    if count == 0:
        print("\n⚠️  WARNING: Vector store is empty!")
        print("   Run: python scripts/ingest_docs.py")
        print(f"{'='*60}\n")
        sys.exit(1)

    # Peek at first 3 chunks
    results = col.peek(limit=min(3, count))
    print(f"\n📄 Sample chunks:\n")
    print(f"{'-'*60}\n")
    
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        print(f"Chunk {i}:")
        print(f"  📁 Source:    {meta.get('source_file', 'unknown')}")
        
        # Show if airline is properly normalized (lowercase)
        airline = meta.get('airline', 'unknown')
        airline_status = "✅" if airline.islower() or airline in ("DOT", "INTERNAL") else "⚠️"
        print(f"  {airline_status} Airline:   {airline}")
        
        authority = meta.get('authority', 'unknown')
        authority_icon = "⚖️" if authority == "REGULATOR" else "🏢"
        print(f"  {authority_icon} Authority: {authority}")
        
        domain = meta.get('domain', 'unknown')
        print(f"  📋 Domain:    {domain}")
        
        do_not_cite = meta.get('do_not_cite', False)
        cite_status = "🚫" if do_not_cite else "✅"
        print(f"  {cite_status} Citable:  {not do_not_cite}")
        
        print(f"  📝 Preview:   {doc[:100]}...")
        print(f"{'-'*60}\n")

    # Validation checks
    print("🔍 Validation Checks:\n")
    
    # Check 1: Airline normalization
    all_airlines = set()
    for meta in results["metadatas"]:
        all_airlines.add(meta.get('airline', ''))
    
    non_normalized = [a for a in all_airlines if a and not (a.islower() or a in ("DOT", "INTERNAL"))]
    if non_normalized:
        print(f"  ⚠️  WARNING: Found non-lowercase airlines: {non_normalized}")
        print(f"     This will break filtering! Re-ingest with fixed script.")
    else:
        print(f"  ✅ Airline normalization: OK")
    
    # Check 2: Citation filtering
    citable_count = sum(1 for meta in results["metadatas"] if not meta.get('do_not_cite', False))
    print(f"  ✅ Citable chunks: {citable_count}/{len(results['metadatas'])} in sample")
    
    # Check 3: Authority distribution
    authorities = {}
    for meta in results["metadatas"]:
        auth = meta.get('authority', 'unknown')
        authorities[auth] = authorities.get(auth, 0) + 1
    
    print(f"  ℹ️  Authority breakdown in sample:")
    for auth, cnt in sorted(authorities.items()):
        icon = "⚖️" if auth == "REGULATOR" else "🏢" if auth == "AIRLINE" else "📋"
        print(f"     {icon} {auth}: {cnt}")
    
    print(f"\n{'='*60}\n")

except FileNotFoundError:
    print("\n❌ ERROR: Vector store not found at 'vector_store/'")
    print("   Run: python scripts/ingest_docs.py")
    print()
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)