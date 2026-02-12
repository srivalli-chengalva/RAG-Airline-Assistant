from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # --- Paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "policies"

    @property
    def chroma_dir(self) -> Path:
        return self.project_root / "vector_store"

    # --- ChromaDB ---
    collection_name: str = "policies"

    # --- Models ---
    embed_model: str = "intfloat/e5-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval tuning (FIXED: Better thresholds for real-world use) ---
    retrieval_top_k: int = 15        # Fetch more candidates for better coverage
    rerank_top_n: int = 6            # Keep more high-quality results
    rerank_threshold_none: float = 0.15   # LOWERED: Use evidence better (was 0.30)
    rerank_threshold_low: float = 0.40    # LOWERED: More forgiving (was 0.50)

    # --- Ollama (LLM) ---
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"


# Singleton settings object — import this everywhere
settings = Settings()