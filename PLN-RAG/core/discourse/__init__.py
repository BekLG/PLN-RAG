"""Lightweight discourse helpers used before semantic extraction."""

from core.discourse.coreference import (
    ChunkCorefResult,
    CorefCluster,
    CorefMention,
    CoreferenceResolver,
    DocumentCorefResult,
    NullCoreferenceResolver,
    project_coref_to_chunk,
)
from core.discourse.mention_prepass import (
    Mention,
    MentionPrepass,
    MentionPrepassResult,
)

__all__ = [
    "ChunkCorefResult",
    "CorefCluster",
    "CorefMention",
    "CoreferenceResolver",
    "DocumentCorefResult",
    "Mention",
    "MentionPrepass",
    "MentionPrepassResult",
    "NullCoreferenceResolver",
    "project_coref_to_chunk",
]
