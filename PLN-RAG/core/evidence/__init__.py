"""Evidence-linked claim records used by ingestion and retrieval."""

from core.evidence.records import (
    ClaimEvidenceRecord,
    build_claim_evidence_record,
    normalize_query_target,
)

__all__ = [
    "ClaimEvidenceRecord",
    "build_claim_evidence_record",
    "normalize_query_target",
]
