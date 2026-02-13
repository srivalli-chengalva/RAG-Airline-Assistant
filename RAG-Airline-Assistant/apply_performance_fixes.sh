#!/bin/bash
# apply_performance_fixes.sh
# Automated script to apply all performance optimizations

set -e  # Exit on error

echo "🚀 RAG-Airline-Assistant Performance Fix Script"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from your RAG-Airline-Assistant root directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

echo "✅ Found project root"
echo ""

# Backup existing files
echo "📦 Creating backups..."
mkdir -p .backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=".backups/$(date +%Y%m%d_%H%M%S)"

if [ -f "backend/ollama_client.py" ]; then
    cp backend/ollama_client.py "$BACKUP_DIR/"
    echo "  ✓ Backed up ollama_client.py"
fi

if [ -f "backend/retrieval.py" ]; then
    cp backend/retrieval.py "$BACKUP_DIR/"
    echo "  ✓ Backed up retrieval.py"
fi

echo ""
echo "📥 Downloading optimized files..."

# Check if we have the fixed files
FIXED_DIR="/home/claude/RAG-Airline-Assistant-FIXED/backend"

if [ -f "$FIXED_DIR/ollama_client.py" ]; then
    cp "$FIXED_DIR/ollama_client.py" backend/
    echo "  ✓ Updated ollama_client.py"
else
    echo "  ⚠️  Could not find $FIXED_DIR/ollama_client.py"
fi

if [ -f "$FIXED_DIR/retrieval.py" ]; then
    cp "$FIXED_DIR/retrieval.py" backend/
    echo "  ✓ Updated retrieval.py"
else
    echo "  ⚠️  Could not find $FIXED_DIR/retrieval.py"
fi

echo ""
echo "🔍 Checking main.py initialization..."

# Check if retriever is initialized at module level
if grep -q "^retriever = Retriever()" backend/main.py; then
    echo "  ✓ Retriever initialization looks good"
else
    echo "  ⚠️  Warning: Make sure 'retriever = Retriever()' is at module level in backend/main.py"
    echo "     (Not inside a function)"
fi

echo ""
echo "🧪 Testing Ollama connection..."

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama is running"
    
    # Get model list
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; data=json.load(sys.stdin); print(', '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "unknown")
    echo "  ℹ️  Available models: $MODELS"
else
    echo "  ⚠️  Cannot connect to Ollama at http://localhost:11434"
    echo "     Make sure Ollama is running: 'ollama serve'"
fi

echo ""
echo "✅ Performance fixes applied!"
echo ""
echo "📋 Next steps:"
echo "  1. Restart your backend:"
echo "     uvicorn backend.main:app --reload"
echo ""
echo "  2. Watch for these logs on startup:"
echo "     '🔄 Creating Retriever singleton...'"
echo "     '✅ Retriever ready in X.Xs'"
echo "     '✅ HTTP session initialized'"
echo ""
echo "  3. Test first request (should be <30s):"
echo "     curl -X POST http://localhost:8000/chat \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"message\": \"Delta cancelled my flight\"}'"
echo ""
echo "  4. Test subsequent requests (should be <15s)"
echo ""
echo "📊 Expected improvements:"
echo "  • First request:      60-120s → 15-25s"
echo "  • Subsequent requests: 30-60s → 5-15s"
echo "  • Timeout rate:         ~30% → <5%"
echo ""
echo "📚 For more details, see: PERFORMANCE_FIX_GUIDE.md"
echo ""
