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

    # --- Retrieval tuning ---
    retrieval_top_k: int = 12        # How many candidates to fetch from vector DB
    rerank_top_n: int = 5            # How many to keep after reranking
    rerank_threshold_none: float = 0.30   # Below this → no answer, ask for clarification
    rerank_threshold_low: float = 0.50    # Below this → low confidence warning

    # --- Ollama (LLM - Day 2) ---
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"


# Singleton settings object — import this everywhere
settings = Settings()