from pydantic import BaseModel
from typing import List, Optional, Literal


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
    fallback_used: bool
    query_status: Literal["well_aligned", "weakly_aligned", "malformed", "no_query"]
    raw_proof: str
    sources: List[str]       # NL sentences that contributed to the proof
    answer: str


#  Reset 

class ResetRequest(BaseModel):
    scope: Literal["all", "vectordb", "atomspace"] = "all"


class ResetResponse(BaseModel):
    status: Literal["ok"]
    scope: str


#  Health 

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    parser: str
    atomspace_size: int
    background_atomspace_size: int
    vectordb_count: int
    conceptnet_enabled: bool
    conceptnet_indexing: bool
    conceptnet_vectors_indexed: int
    conceptnet_vectors_expected: int
    conceptnet_last_error: str
    uptime_seconds: float
