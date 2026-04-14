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


class IngestResponse(BaseModel):
    processed_count: int
    results: List[IngestItemResult]


#  Query 

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    pln_query: str
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
    vectordb_count: int
    uptime_seconds: float
