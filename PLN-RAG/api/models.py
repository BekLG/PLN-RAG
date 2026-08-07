from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal


#  Ingest 

class IngestRequest(BaseModel):
    texts: List[str]


class IngestItemResult(BaseModel):
    text: str
    atoms: List[str] = []
    status: Literal["success", "failed"]
    error: Optional[str] = None
    chunk_count: int = 0
    batch_count: int = 0
    batch_sizes: List[int] = []
    parser_calls: int = 0
    rejected_count: int = 0
    rejected_samples: List[str] = []


class IngestResponse(BaseModel):
    processed_count: int
    results: List[IngestItemResult]


#  Query 

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    pln_query: str
    original_query: str
    executed_query: str
    query_source: Literal["qdrant_alignment", "parser", "deterministic", "none"] = "none"
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    raw_proof: str
    sources: List[str]       # NL sentences that contributed to the proof
    answer: str
    qdrant_aligned_queries: List[str] = Field(default_factory=list)
    execution_candidates: List[str] = Field(default_factory=list)
    candidate_count: Optional[int] = None
    candidate_count_tried: Optional[int] = None
    executed_candidate_index: Optional[int] = None
    retry_used: Optional[bool] = None
    context_retrieval_seconds: Optional[float] = None
    parse_query_seconds: Optional[float] = None
    reasoning_seconds: Optional[float] = None
    source_lookup_seconds: Optional[float] = None
    answer_generation_seconds: Optional[float] = None
    intent_mode: Literal["boolean", "open", "factors", "explanation", "sufficiency"] = "boolean"
    proof_status: Literal["positive", "negative", "both", "unknown", "unanswered"] = "unknown"
    negative_query: str = ""
    positive_proof: List[str] = Field(default_factory=list)
    negative_proof: List[str] = Field(default_factory=list)
    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    canonical_proposition: str = ""
    target_alignment: Literal["exact", "compatible", "rejected", "none"] = "none"
    proof_validated: bool = False
    support_kind: Literal[
        "entailed", "probabilistic", "explicit_negative", "conflict", "unknown"
    ] = "unknown"
    unresolved_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    # Proved values for an open (which/what/who) question. Empty for boolean
    # questions, which are answered by proof_status instead.
    bindings: List[Dict[str, Any]] = Field(default_factory=list)
    normalization_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)


#  Reset 

class ResetRequest(BaseModel):
    scope: Literal["all", "vectordb", "atomspace"] = "all"


class ResetResponse(BaseModel):
    status: Literal["ok"]
    scope: str


class RebuildResponse(BaseModel):
    status: Literal["ok"]
    atomspace_count: int = 0
    qdrant_indexed_count: int = 0
    pending_index_count: int = 0


#  Health 

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    parser: str
    atomspace_size: int
    vectordb_count: int
    evidence_document_count: int = 0
    evidence_claim_count: int = 0
    pending_index_count: int = 0
    uptime_seconds: float


#  Debug

class LangExtractPostprocessed(BaseModel):
    statements: List[str] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)
    canonicalization_context: Dict[str, Any] = Field(default_factory=dict)
    mention_prepass: Dict[str, Any] = Field(default_factory=dict)
    mention_prompt_hint: str = ""
    statement_sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Set when the LangExtract provider call failed. Empty output plus an empty
    # provider_error means the model genuinely extracted nothing.
    provider_error: str = ""


class LangExtractQueryPostprocessed(BaseModel):
    queries: List[str] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)
    canonicalization_context: Dict[str, Any] = Field(default_factory=dict)
    mention_prepass: Dict[str, Any] = Field(default_factory=dict)
    mention_prompt_hint: str = ""
    query_sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provider_error: str = ""


class DebugIngestRequest(BaseModel):
    texts: List[str]


class DebugIngestChunkResult(BaseModel):
    chunk: str
    chunk_start: int = 0
    chunk_end: int = 0
    coreference: Dict[str, Any] = Field(default_factory=dict)
    context: List[str]
    langextract_postprocessed: LangExtractPostprocessed
    pln_canonicalized: List[str] = Field(default_factory=list)
    atomspace_added: List[str] = Field(default_factory=list)
    evidence_records: List[Dict[str, Any]] = Field(default_factory=list)
    schema_alignment: List[Dict[str, Any]] = Field(default_factory=list)
    predicate_registry: List[Dict[str, Any]] = Field(default_factory=list)


class DebugIngestItemResult(BaseModel):
    text: str
    chunks: List[DebugIngestChunkResult] = Field(default_factory=list)
    coreference: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "failed"]
    error: Optional[str] = None


class DebugIngestResponse(BaseModel):
    processed_count: int
    results: List[DebugIngestItemResult]


class DebugQueryRequest(BaseModel):
    question: str


class DebugQueryResponse(BaseModel):
    question: str
    context: List[str] = Field(default_factory=list)
    qdrant_matches: List[Dict[str, Any]] = Field(default_factory=list)
    qdrant_aligned_queries: List[str] = Field(default_factory=list)
    execution_candidates: List[str] = Field(default_factory=list)
    langextract_postprocessed: LangExtractQueryPostprocessed
    pln_canonicalized_queries: List[str] = Field(default_factory=list)
    supporting_statements: List[str] = Field(default_factory=list)
    executed_query: str
    query_source: Literal["qdrant_alignment", "parser", "deterministic", "none"] = "none"
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    proof: str
    sources: List[str]
    answer: str
    intent_mode: Literal["boolean", "open", "factors", "explanation", "sufficiency"] = "boolean"
    proof_status: Literal["positive", "negative", "both", "unknown", "unanswered"] = "unknown"
    negative_query: str = ""
    positive_proof: List[str] = Field(default_factory=list)
    negative_proof: List[str] = Field(default_factory=list)
    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    canonical_proposition: str = ""
    target_alignment: Literal["exact", "compatible", "rejected", "none"] = "none"
    proof_validated: bool = False
    support_kind: Literal[
        "entailed", "probabilistic", "explicit_negative", "conflict", "unknown"
    ] = "unknown"
    unresolved_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    # Proved values for an open (which/what/who) question. Empty for boolean
    # questions, which are answered by proof_status instead.
    bindings: List[Dict[str, Any]] = Field(default_factory=list)
    normalization_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    # One entry per discarded query candidate: query, source, gate stage, detail.
    candidate_rejections: List[Dict[str, Any]] = Field(default_factory=list)


class DebugQdrantPoint(BaseModel):
    id: str | int
    payload: Dict[str, Any] = Field(default_factory=dict)


class DebugQdrantResponse(BaseModel):
    enabled: bool
    count: int
    points: List[DebugQdrantPoint] = Field(default_factory=list)
    predicate_count: int = 0
    predicate_points: List[DebugQdrantPoint] = Field(default_factory=list)
    evidence_document_count: int = 0
    evidence_claim_count: int = 0
    pending_index_count: int = 0
