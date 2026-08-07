from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol


PRONOUNS = {
    "he",
    "her",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "it",
    "its",
    "itself",
    "she",
    "their",
    "theirs",
    "them",
    "themselves",
    "they",
}
NAME_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "sir", "madam"}


@dataclass
class CorefMention:
    text: str
    start: int
    end: int
    mention_type: str | None = None
    is_pronoun: bool = False
    pair_logit: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorefCluster:
    cluster_id: str
    mentions: list[CorefMention] = field(default_factory=list)
    canonical_mention: CorefMention | None = None
    ambiguous: bool = False

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "mentions": [mention.to_dict() for mention in self.mentions],
            "canonical_mention": (
                self.canonical_mention.to_dict()
                if self.canonical_mention is not None
                else None
            ),
            "ambiguous": self.ambiguous,
        }


@dataclass
class DocumentCorefResult:
    text: str
    clusters: list[CorefCluster] = field(default_factory=list)
    backend: str = "none"
    model_name: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text_length": len(self.text),
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "backend": self.backend,
            "model_name": self.model_name,
            "error": self.error,
        }


@dataclass
class ChunkCorefResult:
    text: str
    chunk_start: int
    chunk_end: int
    clusters: list[CorefCluster] = field(default_factory=list)
    backend: str = "none"
    model_name: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text_length": len(self.text),
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "backend": self.backend,
            "model_name": self.model_name,
            "error": self.error,
        }


class CoreferenceResolver(Protocol):
    def resolve_document(self, text: str) -> DocumentCorefResult:
        ...


class NullCoreferenceResolver:
    backend = "none"
    model_name = ""

    def resolve_document(self, text: str) -> DocumentCorefResult:
        return DocumentCorefResult(text=text, backend=self.backend, model_name="")


def build_cluster(
    *,
    cluster_id: str,
    text: str,
    spans: list[tuple[int, int]],
    pair_logits: dict[tuple[int, int], float] | None = None,
) -> CorefCluster:
    mentions: list[CorefMention] = []
    pair_logits = pair_logits or {}
    for index, (start, end) in enumerate(spans):
        if start < 0 or end < start or end > len(text):
            continue
        mention_text = text[start:end]
        mentions.append(
            CorefMention(
                text=mention_text,
                start=start,
                end=end,
                mention_type=_mention_type(mention_text),
                is_pronoun=_is_pronoun(mention_text),
                pair_logit=pair_logits.get((index, 0)),
            )
        )
    return finalize_cluster(CorefCluster(cluster_id=cluster_id, mentions=mentions))


def finalize_cluster(cluster: CorefCluster) -> CorefCluster:
    canonical, ambiguous = select_canonical_mention(cluster.mentions)
    cluster.canonical_mention = canonical
    cluster.ambiguous = bool(cluster.ambiguous or ambiguous)
    return cluster


def select_canonical_mention(
    mentions: list[CorefMention],
) -> tuple[CorefMention | None, bool]:
    non_pronouns = [mention for mention in mentions if not mention.is_pronoun]
    if not non_pronouns:
        return None, False

    named = [mention for mention in non_pronouns if _is_proper_name_like(mention.text)]
    if named:
        if _has_conflicting_names(named):
            return None, True
        return _longest_mention(named), False

    return _longest_mention(non_pronouns), False


def project_coref_to_chunk(
    result: DocumentCorefResult,
    chunk_start: int,
    chunk_end: int,
    chunk_text: str,
) -> ChunkCorefResult:
    clusters: list[CorefCluster] = []
    for cluster in result.clusters:
        mentions = [
            mention
            for mention in cluster.mentions
            if mention.start >= chunk_start and mention.end <= chunk_end
        ]
        if not mentions:
            continue
        clusters.append(
            CorefCluster(
                cluster_id=cluster.cluster_id,
                mentions=list(mentions),
                canonical_mention=cluster.canonical_mention,
                ambiguous=cluster.ambiguous,
            )
        )
    return ChunkCorefResult(
        text=chunk_text,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        clusters=clusters,
        backend=result.backend,
        model_name=result.model_name,
        error=result.error,
    )


def _mention_type(text: str) -> str:
    if _is_pronoun(text):
        return "pronoun"
    if _is_proper_name_like(text):
        return "name"
    return "nominal"


def _is_pronoun(text: str) -> bool:
    return _normalized_text(text) in PRONOUNS


def _is_proper_name_like(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z.'-]*", text)
    if not tokens:
        return False
    content = [token for token in tokens if token.rstrip(".").lower() not in NAME_TITLES]
    return bool(content) and all(token[:1].isupper() for token in content)


def _has_conflicting_names(mentions: list[CorefMention]) -> bool:
    normalized = [_normalized_name(mention.text) for mention in mentions]
    normalized = [item for item in normalized if item]
    for i, left in enumerate(normalized):
        for right in normalized[i + 1 :]:
            if not _names_are_aliases(left, right):
                return True
    return False


def _names_are_aliases(left: str, right: str) -> bool:
    if left == right:
        return True
    left_tokens = left.split()
    right_tokens = right.split()
    return (
        len(left_tokens) > 0
        and len(right_tokens) > 0
        and (
            _subsequence(left_tokens, right_tokens)
            or _subsequence(right_tokens, left_tokens)
        )
    )


def _subsequence(shorter: list[str], longer: list[str]) -> bool:
    if len(shorter) > len(longer):
        return False
    for start in range(0, len(longer) - len(shorter) + 1):
        if longer[start : start + len(shorter)] == shorter:
            return True
    return False


def _longest_mention(mentions: list[CorefMention]) -> CorefMention:
    return sorted(
        mentions,
        key=lambda mention: (
            -len(_normalized_text(mention.text)),
            mention.start,
            mention.end,
        ),
    )[0]


def _normalized_name(text: str) -> str:
    tokens = [
        token.rstrip(".").lower()
        for token in re.findall(r"[A-Za-z][A-Za-z.'-]*", text)
        if token.rstrip(".").lower() not in NAME_TITLES
    ]
    return " ".join(tokens)


def _normalized_text(text: str) -> str:
    return " ".join(str(text).lower().split())
