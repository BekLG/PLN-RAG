from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_model: str = "openai/gpt-4o-mini"

    # Options: "nl2pln" | "canonical_pln" | "manhin"
    parser: str = "nl2pln"
    nl2pln_module_path: str = "data/simba_all.json"
    canonical_pln_nl2pln_module_path: str = "data/simba_canonical_pln.json"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pln_rag"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "nomic-embed-text"

    # Atomspace persistence
    atomspace_path: str = "data/atomspace/kb.metta"

    # FAISS predicate store (used by Manhin parser)
    faiss_path: str = "data/faiss"

    # Processing
    chunk_size: int = 512  # chars per chunk
    chunk_overlap: int = 64  # overlap between chunks
    context_top_k: int = 10  # atoms to retrieve as parser context

    # Reasoning
    chaining_timeout: int = 30  # seconds before proof search is killed
    chaining_max_steps: int = 100

    # Query execution
    query_fallback_enabled: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
