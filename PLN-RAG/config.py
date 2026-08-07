from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict, field_validator
from typing import Optional
from pathlib import Path

# Root .env is two levels up from this file (PLN-RAG/config.py -> root)
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_LOCAL_ENV = Path(__file__).resolve().parent / ".env"

# Use root .env if it exists, otherwise fall back to PLN-RAG/.env
_ENV_FILE = str(_ROOT_ENV) if _ROOT_ENV.exists() else str(_LOCAL_ENV)

# Provider prefixes some SDKs accept but LangExtract's router does not. Its
# patterns are anchored, so "^gemini" happens to match "gemini/gemini-2.5-flash"
# while "^gpt-4" cannot match "openai/gpt-4o-mini" — the same prefix style works
# for one provider and fails for the other. Normalize centrally instead.
PROVIDER_PREFIXES = frozenset({"gemini", "google", "openai", "azure", "ollama"})


def normalize_model_id(model: str | None) -> str | None:
    """
    Strip a leading `<provider>/` prefix when the provider is one we recognize.

    Only allow-listed prefixes are stripped, so legitimate ids that contain a
    slash are left intact: Ollama tags (`gemma2:2b`), HuggingFace repositories
    (`Qwen/Qwen2.5-7B`, `meta-llama/Llama-3-8B`), and anything else.
    """
    if model is None:
        return None
    value = str(model).strip()
    if "/" not in value:
        return value
    prefix, remainder = value.split("/", 1)
    if prefix.lower() in PROVIDER_PREFIXES and remainder.strip():
        return remainder.strip()
    return value


class Settings(BaseSettings):
    # LLM — provide either OpenAI or Gemini credentials
    openai_api_key: Optional[str] = None
    openai_model: str = "openai/gpt-4o-mini"

    # Gemini (uses Google's OpenAI-compatible endpoint)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini/gemini-2.5-flash"

    # LangExtract parser (NL -> LangExtract objects -> canonical PLN)
    langextract_api_key: Optional[str] = None
    langextract_model_id: str = "gemini-2.5-flash"
    langextract_model_url: Optional[str] = None
    langextract_examples_path: str = "data/langextract_examples.json"
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 1
    langextract_cache_enabled: bool = True
    langextract_cache_max_entries: int = 128
    langextract_skip_fuzzy: bool = True
    langextract_chunk_size: Optional[int] = 2000
    mention_prepass_enabled: bool = True

    # Optional document-level neural coreference. Disabled by default because
    # LingMess is a large model and proof extraction must fail open.
    coreference_enabled: bool = False
    coreference_backend: str = "lingmess"
    coreference_model: str = "biu-nlp/lingmess-coref"
    coreference_device: str = "auto"
    coreference_mode: str = "hint_only"
    coreference_fail_open: bool = True
    coreference_max_tokens_in_batch: int = 10000
    coreference_include_pair_logits: bool = False

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pln_rag_evidence_v2"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "nomic-embed-text"
    use_vector_store: bool = True
    qdrant_hybrid_enabled: bool = True

    # Authoritative evidence and claim ledger. Qdrant and Atomspace are
    # rebuildable projections of this local database.
    evidence_ledger_enabled: bool = True
    evidence_ledger_path: str = "data/evidence/evidence.db"
    evidence_outbox_batch_size: int = 256

    # Atomspace persistence
    atomspace_path: str = "data/atomspace/kb.metta"

    # FAISS predicate store (no longer used)
    faiss_path: str = "data/faiss"

    # Processing
    chunk_size: int = 2000  # chars per chunk
    chunk_overlap: int = 64  # overlap between chunks
    context_top_k: int = 10  # atoms to retrieve as parser context

    # Reasoning
    chaining_timeout: int = 30  # seconds before proof search is killed
    chaining_max_steps: int = 100
    strict_proof_validation_enabled: bool = True

    # Query execution
    query_fallback_enabled: bool = True
    query_alignment_enabled: bool = True
    query_alignment_top_k: int = 8
    query_alignment_min_score: float = 0.55
    query_alignment_hybrid_min_score: float = 0.30
    query_candidate_max_tries: int = 5
    answer_generation_enabled: bool = True
    source_lookup_max_atoms: int = 0

    # Semantic target gate. Replaces hardcoded term matching when deciding which
    # proof targets a question asks about. Disabling it admits every structurally
    # valid candidate, which is looser, not stricter.
    query_target_gate_enabled: bool = True
    query_target_gate_top_k: int = 5
    query_target_gate_timeout: int = 12
    query_target_gate_min_confidence: float = 0.7
    # Higher bar for asks_negation: that verdict inverts the answer, so a bad
    # call yields a confident wrong result instead of an honest unknown.
    query_target_gate_negation_min_confidence: float = 0.85
    # Keep only candidates sharing the top candidate's head and args. Left off:
    # it discarded correct targets whenever a negation-lexicalized variant
    # happened to rank first, and made query_candidate_max_tries inert.
    query_proof_equivalent_only: bool = False

    # Predicate registry / mapping graph
    predicate_registry_enabled: bool = True
    predicate_registry_path: str = "data/predicate_registry.json"
    predicate_mapping_enabled: bool = True
    predicate_mapping_llm_enabled: bool = True
    # Ask the LLM classifier about predicate pairs at ingest time, and turn the
    # approved mappings into bridge rules. Both were off, which left intra-document
    # predicate drift unrepaired: a rule concluding `BecomesUnbalanced` could not
    # satisfy another rule's `Unbalanced` premise, severing the chain inside the KB.
    # Bridges carry the classifier's confidence as their STV strength, so any proof
    # that traverses one is reported as support_kind="probabilistic", never
    # "entailed". An explicit negative fact still vetoes a bridge.
    predicate_mapping_online_enabled: bool = True
    predicate_mapping_emit_bridges: bool = True
    predicate_mapping_collection: str = "pln_rag_predicates"
    predicate_mapping_top_k: int = 6
    predicate_mapping_max_candidates: int = 6
    predicate_mapping_min_score: float = 0.62
    predicate_mapping_proof_threshold: float = 0.82
    predicate_mapping_timeout: int = 12
    predicate_mapping_total_timeout: int = 30

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("openai_model", "gemini_model", "langextract_model_id")
    @classmethod
    def _normalize_model_id(cls, value):
        return normalize_model_id(value)

    @field_validator("coreference_backend")
    @classmethod
    def _validate_coreference_backend(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"lingmess"}:
            raise ValueError("COREFERENCE_BACKEND must be 'lingmess'")
        return normalized

    @field_validator("coreference_device")
    @classmethod
    def _validate_coreference_device(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "cpu", "cuda", "cuda:0"}:
            raise ValueError("COREFERENCE_DEVICE must be auto, cpu, cuda, or cuda:0")
        return normalized

    @field_validator("coreference_mode")
    @classmethod
    def _validate_coreference_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized != "hint_only":
            raise ValueError("COREFERENCE_MODE currently supports only hint_only")
        return normalized


@lru_cache
def get_settings() -> Settings:
    return Settings()
