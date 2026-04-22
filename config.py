from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict


class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    openai_model: str = "openai/gpt-4o-mini"

    # Options: "nl2pln" | "canonical_pln" | "manhin"
    parser: str = "canonical_pln"
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

    # ConceptNet background knowledge
    conceptnet_enabled: bool = False
    conceptnet_autoload: bool = True
    conceptnet_input_file: str = "data/conceptnet/conceptnet-assertions-5.7.0.csv.gz"
    conceptnet_atomspace_path: str = "data/conceptnet/conceptnet_background.metta"
    conceptnet_vector_payload_path: str = (
        "data/conceptnet/conceptnet_background.jsonl"
    )
    conceptnet_manifest_path: str = "data/conceptnet/conceptnet_manifest.json"
    conceptnet_index_on_startup: bool = True
    conceptnet_min_weight: float = 2.0
    conceptnet_coverage_percent: float = 100.0
    conceptnet_sample_seed: int = 42
    conceptnet_auto_rebuild_on_change: bool = True
    conceptnet_reindex_on_reset: bool = True
    conceptnet_startup_fail_open: bool = True

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
