import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    ResetRequest, ResetResponse,
    RebuildResponse,
    HealthResponse,
    DebugIngestRequest, DebugIngestResponse,
    DebugQueryRequest, DebugQueryResponse,
    DebugQdrantResponse,
)
from core.orchestration.service import PLNRAGService
from parsers import LangExtractPLNParser

_start_time = time.time()
_service: PLNRAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    print("[Startup] Initializing LangExtract parser...")
    parser = LangExtractPLNParser()
    _service = PLNRAGService(parser)
    print("[Startup] Service ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title="PLN-RAG API",
    description="Probabilistic Logic Network RAG service",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service() -> PLNRAGService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _service


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Ingest a batch of texts into the knowledge base.
    Texts are chunked, parsed into PLN atoms, added to the
    atomspace, and indexed in the vector store.
    Processing is sequential — each text sees all previous atoms.
    """
    svc = get_service()
    results = await svc.ingest_batch(req.texts)
    return IngestResponse(
        processed_count=len(results),
        results=results
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Ask a question against the knowledge base.
    The question is parsed into a PLN query, reasoned over via
    PeTTaChainer, and the proof trace is translated to natural language.
    """
    svc = get_service()
    return await svc.query(req.question)


@app.delete("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = ResetRequest()):
    """
    Clear the knowledge base.
    scope='all': clears atomspace + vector DB
    scope='atomspace': clears only the PLN atomspace
    scope='vectordb': clears only Qdrant
    """
    svc = get_service()
    svc.reset(req.scope)
    return ResetResponse(status="ok", scope=req.scope)


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild():
    """Rebuild Atomspace and the Qdrant evidence index from SQLite."""
    return RebuildResponse(status="ok", **get_service().rebuild_indexes())


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check — returns component status and sizes."""
    svc = get_service()
    info = svc.health()
    return HealthResponse(
        status="ok",
        parser=info["parser"],
        atomspace_size=info["atomspace_size"],
        vectordb_count=info["vectordb_count"],
        evidence_document_count=info["evidence_document_count"],
        evidence_claim_count=info["evidence_claim_count"],
        pending_index_count=info["pending_index_count"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/debug/ingest", response_model=DebugIngestResponse)
async def debug_ingest(req: DebugIngestRequest):
    """
    Debug ingest path: returns step-by-step outputs per chunk
    (LangExtract post-processed, canonicalized PLN, atoms added).
    """
    svc = get_service()
    results = await svc.debug_ingest_batch(req.texts)
    return DebugIngestResponse(
        processed_count=len(results),
        results=results,
    )


@app.post("/debug/query", response_model=DebugQueryResponse)
async def debug_query(req: DebugQueryRequest):
    """
    Debug query path: returns parser outputs plus answer and proof.
    """
    svc = get_service()
    return await svc.debug_query(req.question)


@app.get("/debug/qdrant", response_model=DebugQdrantResponse)
async def debug_qdrant(limit: int = 50):
    """
    Inspect raw Qdrant payloads stored during ingest.
    """
    svc = get_service()
    return svc.debug_qdrant(limit=limit)
