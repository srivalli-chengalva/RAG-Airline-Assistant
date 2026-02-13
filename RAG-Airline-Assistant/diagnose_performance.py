#!/usr/bin/env python3
"""
diagnose_performance.py
-----------------------
Diagnostic script to identify performance bottlenecks in RAG-Airline-Assistant.

Usage:
    python diagnose_performance.py
"""

import time
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_ollama_connection():
    """Test Ollama connectivity and response time."""
    print_header("1️⃣  Testing Ollama Connection")
    
    try:
        import requests
        
        # Test connection
        print("Testing /api/tags endpoint...")
        start = time.time()
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        elapsed = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            print(f"✅ Connection successful ({elapsed:.2f}s)")
            print(f"   Available models: {len(models)}")
            for m in models:
                print(f"     • {m.get('name')}")
        else:
            print(f"⚠️  HTTP {r.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Fix: Start Ollama with 'ollama serve'")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test generation speed
    print("\nTesting generation speed...")
    try:
        start = time.time()
        payload = {
            "model": "llama3.1:8b",
            "prompt": "Say hello in one word.",
            "stream": False,
            "keep_alive": "10m"
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        elapsed = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            response = data.get("response", "")
            print(f"✅ Generation successful ({elapsed:.2f}s)")
            print(f"   Response: {response[:100]}")
            
            if elapsed > 20:
                print("⚠️  Warning: Generation is slow (>20s)")
                print("   Possible causes:")
                print("     • Model not loaded (first request)")
                print("     • CPU bottleneck (no GPU)")
                print("     • Ollama overloaded")
        else:
            print(f"❌ HTTP {r.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out after 30s")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def test_model_loading():
    """Test sentence-transformers and cross-encoder loading times."""
    print_header("2️⃣  Testing Model Loading Times")
    
    try:
        # Test embedder
        print("Loading sentence-transformers embedder...")
        start = time.time()
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("intfloat/e5-base-v2")
        elapsed = time.time() - start
        print(f"✅ Embedder loaded ({elapsed:.2f}s)")
        
        if elapsed > 30:
            print("⚠️  Warning: Embedder loading is slow (>30s)")
            print("   First load is expected to be slow (downloading)")
            print("   Subsequent loads should be <5s")
        
        # Test embedding
        print("\nTesting embedding speed...")
        start = time.time()
        emb = embedder.encode(["test query"], normalize_embeddings=True)
        elapsed = time.time() - start
        print(f"✅ Embedding generated ({elapsed:.3f}s)")
        
        if elapsed > 1:
            print("⚠️  Warning: Embedding is slow (>1s)")
        
    except Exception as e:
        print(f"❌ Error loading embedder: {e}")
        return False
    
    try:
        # Test reranker
        print("\nLoading cross-encoder reranker...")
        start = time.time()
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        elapsed = time.time() - start
        print(f"✅ Reranker loaded ({elapsed:.2f}s)")
        
        if elapsed > 30:
            print("⚠️  Warning: Reranker loading is slow (>30s)")
        
        # Test reranking
        print("\nTesting reranking speed...")
        start = time.time()
        pairs = [("query", "document text here")] * 10
        scores = reranker.predict(pairs)
        elapsed = time.time() - start
        print(f"✅ Reranking complete ({elapsed:.3f}s for 10 pairs)")
        
        if elapsed > 2:
            print("⚠️  Warning: Reranking is slow (>2s for 10 pairs)")
        
    except Exception as e:
        print(f"❌ Error loading reranker: {e}")
        return False
    
    return True


def test_chromadb():
    """Test ChromaDB access and query speed."""
    print_header("3️⃣  Testing ChromaDB")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        print("Connecting to vector store...")
        client = chromadb.PersistentClient(
            path="vector_store",
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection("policies")
        count = collection.count()
        print(f"✅ Connected to ChromaDB")
        print(f"   Total chunks: {count}")
        
        if count == 0:
            print("⚠️  Warning: Vector store is empty!")
            print("   Fix: Run 'python scripts/ingest_docs.py'")
            return False
        
        # Test query speed
        print("\nTesting query speed...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("intfloat/e5-base-v2")
        
        start = time.time()
        query_emb = embedder.encode(["refund policy"], normalize_embeddings=True).tolist()[0]
        res = collection.query(query_embeddings=[query_emb], n_results=10)
        elapsed = time.time() - start
        
        print(f"✅ Query complete ({elapsed:.3f}s)")
        print(f"   Results: {len(res['ids'][0])}")
        
        if elapsed > 1:
            print("⚠️  Warning: Query is slow (>1s)")
        
    except FileNotFoundError:
        print("❌ Vector store not found at 'vector_store/'")
        print("   Fix: Run 'python scripts/ingest_docs.py'")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def test_backend_startup():
    """Test if backend can start and respond."""
    print_header("4️⃣  Testing Backend Startup")
    
    try:
        import requests
        
        print("Testing /health endpoint...")
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            
            if r.status_code == 200:
                data = r.json()
                print(f"✅ Backend is running")
                print(f"   Status: {data.get('status')}")
                print(f"   Ollama: {data.get('ollama_model', '?')}")
            else:
                print(f"⚠️  Backend returned HTTP {r.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("⚠️  Backend is not running")
            print("   Start with: uvicorn backend.main:app --reload")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def check_system_resources():
    """Check CPU, memory, disk usage."""
    print_header("5️⃣  System Resources")
    
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")
        if cpu_percent > 80:
            print("⚠️  High CPU usage")
        
        # Memory
        mem = psutil.virtual_memory()
        print(f"Memory: {mem.percent}% used ({mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB)")
        if mem.percent > 85:
            print("⚠️  High memory usage")
        
        # Disk
        disk = psutil.disk_usage('.')
        print(f"Disk: {disk.percent}% used ({disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB)")
        if disk.percent > 90:
            print("⚠️  Low disk space")
        
    except ImportError:
        print("ℹ️  Install psutil for resource monitoring: pip install psutil")
    except Exception as e:
        print(f"Could not check resources: {e}")


def main():
    print("\n🔬 RAG-Airline-Assistant Performance Diagnostics")
    print("=" * 60)
    
    results = {
        "ollama": test_ollama_connection(),
        "models": test_model_loading(),
        "chromadb": test_chromadb(),
        "backend": test_backend_startup(),
    }
    
    check_system_resources()
    
    # Summary
    print_header("📊 Diagnostic Summary")
    
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.title()}: {'OK' if status else 'FAILED'}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✅ All systems operational!")
        print("   If you're still experiencing slowness:")
        print("   1. Check that models are loaded at startup (not per-request)")
        print("   2. Monitor request timing logs")
        print("   3. Consider using GPU acceleration")
    else:
        print("\n⚠️  Some components failed. Fix the issues above.")
        print("   Common fixes:")
        print("   • Start Ollama: ollama serve")
        print("   • Ingest policies: python scripts/ingest_docs.py")
        print("   • Start backend: uvicorn backend.main:app --reload")
    
    print()


if __name__ == "__main__":
    main()
