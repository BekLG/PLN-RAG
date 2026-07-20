from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from core.discourse.coreference import ChunkCorefResult, CorefCluster, CorefMention


PRONOUNS = {
    "he",
    "her",
    "him",
    "his",
    "it",
    "its",
    "she",
    "their",
    "them",
    "they",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
STOPWORDS = {
    "and",
    "are",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "near",
    "not",
    "of",
    "on",
    "or",
    "over",
    "than",
    "to",
    "was",
    "were",
    "who",
    "with",
}


@dataclass
class Mention:
    id: str
    text: str
    canonical: str
    kind: str
    sentence_index: int
    start: int
    end: int
    candidates: list[str] = field(default_factory=list)
    cluster_id: str | None = None
    canonical_mention: str | None = None
    resolved_to: str | None = None
    resolution_source: str | None = None
    ambiguous: bool = False
    global_start: int | None = None
    global_end: int | None = None
    pair_logit: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MentionPrepassResult:
    mentions: list[Mention] = field(default_factory=list)
    coreference: dict = field(default_factory=dict)

    @property
    def ambiguous_pronouns(self) -> list[Mention]:
        return [
            mention
            for mention in self.mentions
            if mention.kind == "pronoun"
            and (len(mention.candidates) > 1 or mention.ambiguous)
        ]

    @property
    def resolved_pronouns(self) -> list[Mention]:
        return [
            mention
            for mention in self.mentions
            if mention.kind == "pronoun"
            and mention.resolved_to
            and not mention.ambiguous
        ]

    @property
    def unresolved_pronouns(self) -> list[Mention]:
        return [
            mention
            for mention in self.mentions
            if mention.kind == "pronoun"
            and not mention.resolved_to
            and not mention.candidates
            and mention.resolution_source == "unresolved"
        ]

    def to_dict(self) -> dict:
        return {
            "mentions": [mention.to_dict() for mention in self.mentions],
            "ambiguous_pronouns": [
                mention.to_dict() for mention in self.ambiguous_pronouns
            ],
            "resolved_pronouns": [
                mention.to_dict() for mention in self.resolved_pronouns
            ],
            "unresolved_pronouns": [
                mention.to_dict() for mention in self.unresolved_pronouns
            ],
            "coreference": self.coreference,
        }

    def prompt_hint(self, *, max_mentions: int = 16) -> str:
        ambiguous = self.ambiguous_pronouns
        resolved = self.resolved_pronouns
        unresolved = self.unresolved_pronouns
        if not ambiguous and not resolved and not unresolved:
            return ""

        relevant_ids = {
            candidate
            for mention in ambiguous
            for candidate in mention.candidates
        }
        relevant_ids.update(mention.id for mention in ambiguous)
        relevant_ids.update(mention.id for mention in resolved)
        relevant_ids.update(mention.id for mention in unresolved)
        relevant = [
            mention for mention in self.mentions if mention.id in relevant_ids
        ][:max_mentions]

        lines = [
            "",
            "",
            "Mention prepass hints:",
            "- Mentions below are deterministic anchors, not facts.",
            "- If a pronoun has multiple candidates, do not replace it with one candidate.",
            "- For ambiguous pronouns, preserve the pronoun mention id as the argument "
            "(for example it_m4) instead of guessing camera or phone.",
            "- For unambiguous resolved pronouns, use the canonical entity only as the "
            "semantic argument and preserve the original source mention.",
        ]
        for mention in relevant:
            candidate_text = ""
            if mention.candidates:
                candidate_text = (
                    " candidates="
                    + ",".join(self._candidate_labels(mention.candidates))
                )
            resolution_text = ""
            if mention.resolved_to:
                resolution_text = (
                    f" resolved_to={mention.resolved_to!r}"
                    f" source={mention.resolution_source!r}"
                )
            elif mention.ambiguous:
                resolution_text = " unresolved=ambiguous"
            elif mention.resolution_source == "unresolved":
                resolution_text = " unresolved=no_canonical_mention"
            lines.append(
                f"- {mention.id}: {mention.kind} text={mention.text!r} "
                f"canonical={mention.canonical!r}{candidate_text}{resolution_text}"
            )
        for mention in resolved[:max_mentions]:
            lines.append(
                f"- The mention {mention.text!r} at characters "
                f"{mention.global_start if mention.global_start is not None else mention.start}-"
                f"{mention.global_end if mention.global_end is not None else mention.end} "
                f"belongs to the same entity cluster as {mention.resolved_to!r}. "
                f"Use {mention.resolved_to!r} as the semantic subject/object when "
                "extracting propositions; preserve the original mention text."
            )
        for mention in ambiguous[:max_mentions]:
            lines.append(
                f"- The mention {mention.text!r} has multiple possible named "
                "antecedents. Do not guess its referent."
            )
        return "\n".join(lines)

    def _candidate_labels(self, ids: Iterable[str]) -> list[str]:
        by_id = {mention.id: mention for mention in self.mentions}
        return [
            f"{candidate_id}:{by_id[candidate_id].canonical}"
            for candidate_id in ids
            if candidate_id in by_id
        ]


class MentionPrepass:
    """Small deterministic mention scan used to preserve ambiguity for LangExtract."""

    def build(
        self,
        text: str,
        *,
        coref_result: ChunkCorefResult | None = None,
        global_offset: int = 0,
    ) -> MentionPrepassResult:
        mentions: list[Mention] = []
        prior_nominals: list[Mention] = []
        sentence_spans = list(self._sentence_spans(text))
        mention_number = 1

        for sentence_index, start, end in sentence_spans:
            sentence = text[start:end]
            tokens = list(self._tokens(sentence, offset=start))
            token_mentions: list[Mention] = []

            for idx, (token, token_start, token_end) in enumerate(tokens):
                lower = token.lower()
                if lower in PRONOUNS:
                    candidates = [
                        mention.id
                        for mention in prior_nominals[-6:]
                        if self._pronoun_compatible(lower, mention)
                    ]
                    mention = Mention(
                        id=f"{lower}_m{mention_number}",
                        text=token,
                        canonical=lower,
                        kind="pronoun",
                        sentence_index=sentence_index,
                        start=token_start,
                        end=token_end,
                        candidates=candidates,
                    )
                    mention_number += 1
                    mentions.append(mention)
                    token_mentions.append(mention)
                    continue

                if token[:1].isupper() and not self._is_sentence_initial(
                    token_start,
                    start,
                    text,
                ):
                    mention = self._new_nominal(
                        mention_number,
                        token,
                        "name",
                        sentence_index,
                        token_start,
                        token_end,
                    )
                    mention_number += 1
                    mentions.append(mention)
                    token_mentions.append(mention)
                    continue

                if lower in DETERMINERS and idx + 1 < len(tokens):
                    noun, noun_start, noun_end = tokens[idx + 1]
                    noun_lower = noun.lower()
                    if self._is_nominal_token(noun_lower):
                        mention = self._new_nominal(
                            mention_number,
                            noun,
                            "definite" if lower == "the" else "nominal",
                            sentence_index,
                            noun_start,
                            noun_end,
                        )
                        mention_number += 1
                    mentions.append(mention)
                    token_mentions.append(mention)

            prior_nominals.extend(
                mention for mention in token_mentions if mention.kind != "pronoun"
            )

        for mention in mentions:
            mention.global_start = global_offset + mention.start
            mention.global_end = global_offset + mention.end
            mention.resolution_source = mention.resolution_source or "deterministic"

        if coref_result is not None:
            mention_number = self._merge_coreference(
                mentions,
                mention_number,
                text,
                coref_result,
            )

        coreference_metadata = coref_result.to_dict() if coref_result else {}
        return MentionPrepassResult(
            mentions=mentions,
            coreference=coreference_metadata,
        )

    def _new_nominal(
        self,
        number: int,
        text: str,
        kind: str,
        sentence_index: int,
        start: int,
        end: int,
    ) -> Mention:
        canonical = self._canonical(text)
        return Mention(
            id=f"{canonical}_m{number}",
            text=text,
            canonical=canonical,
            kind=kind,
            sentence_index=sentence_index,
            start=start,
            end=end,
        )

    def _sentence_spans(self, text: str) -> Iterable[tuple[int, int, int]]:
        start = 0
        sentence_index = 0
        for match in re.finditer(r"[.!?](?:\s+|$)", text):
            end = match.end()
            if text[start:end].strip():
                yield sentence_index, start, end
                sentence_index += 1
            start = end
        if text[start:].strip():
            yield sentence_index, start, len(text)

    def _tokens(self, text: str, *, offset: int) -> Iterable[tuple[str, int, int]]:
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_-]*\b", text):
            yield match.group(0), offset + match.start(), offset + match.end()

    def _canonical(self, value: str) -> str:
        value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
        value = value.replace("-", "_")
        value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_").lower()
        return self._singularize(value)

    def _singularize(self, value: str) -> str:
        if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us")):
            return value[:-1]
        return value

    def _is_nominal_token(self, token: str) -> bool:
        return (
            len(token) > 2
            and token not in STOPWORDS
            and token not in PRONOUNS
            and token not in DETERMINERS
        )

    def _is_sentence_initial(self, token_start: int, sentence_start: int, text: str) -> bool:
        return text[sentence_start:token_start].strip() == ""

    def _pronoun_compatible(self, pronoun: str, mention: Mention) -> bool:
        if pronoun in {"it", "its"}:
            return mention.kind in {"definite", "nominal"}
        if pronoun in {"he", "her", "him", "his", "she"}:
            return mention.kind == "name"
        return mention.kind in {"name", "nominal"}

    def _merge_coreference(
        self,
        mentions: list[Mention],
        mention_number: int,
        text: str,
        coref_result: ChunkCorefResult,
    ) -> int:
        by_span = {
            (mention.global_start, mention.global_end): mention
            for mention in mentions
        }
        used_ids = {mention.id for mention in mentions}
        for cluster in coref_result.clusters:
            for coref_mention in cluster.mentions:
                if coref_mention.start < coref_result.chunk_start:
                    continue
                if coref_mention.end > coref_result.chunk_end:
                    continue
                local_start = coref_mention.start - coref_result.chunk_start
                local_end = coref_mention.end - coref_result.chunk_start
                if local_start < 0 or local_end > len(text):
                    continue
                key = (coref_mention.start, coref_mention.end)
                mention = by_span.get(key)
                if mention is None:
                    mention = self._mention_from_coref(
                        coref_mention,
                        cluster,
                        mention_number,
                        local_start,
                        local_end,
                        used_ids,
                    )
                    mention_number += 1
                    mentions.append(mention)
                    by_span[key] = mention
                    used_ids.add(mention.id)
                self._apply_coref_metadata(mention, coref_mention, cluster)
        mentions.sort(key=lambda mention: (mention.start, mention.end, mention.id))
        return mention_number

    def _mention_from_coref(
        self,
        coref_mention: CorefMention,
        cluster: CorefCluster,
        number: int,
        local_start: int,
        local_end: int,
        used_ids: set[str],
    ) -> Mention:
        canonical = self._canonical(coref_mention.text)
        kind = "pronoun" if coref_mention.is_pronoun else (
            coref_mention.mention_type or "nominal"
        )
        mention_id = f"{canonical or 'mention'}_m{number}"
        while mention_id in used_ids:
            number += 1
            mention_id = f"{canonical or 'mention'}_m{number}"
        return Mention(
            id=mention_id,
            text=coref_mention.text,
            canonical=canonical,
            kind=kind,
            sentence_index=-1,
            start=local_start,
            end=local_end,
            cluster_id=cluster.cluster_id,
            global_start=coref_mention.start,
            global_end=coref_mention.end,
            pair_logit=coref_mention.pair_logit,
        )

    def _apply_coref_metadata(
        self,
        mention: Mention,
        coref_mention: CorefMention,
        cluster: CorefCluster,
    ) -> None:
        mention.cluster_id = cluster.cluster_id
        mention.global_start = coref_mention.start
        mention.global_end = coref_mention.end
        mention.pair_logit = coref_mention.pair_logit
        mention.ambiguous = bool(cluster.ambiguous)
        canonical = cluster.canonical_mention
        mention.canonical_mention = canonical.text if canonical else None
        if mention.kind == "pronoun":
            if cluster.ambiguous:
                mention.resolved_to = None
                mention.resolution_source = "unresolved"
            elif canonical is not None:
                mention.resolved_to = canonical.text
                mention.canonical = self._canonical(canonical.text)
                mention.resolution_source = "lingmess"
            else:
                mention.resolution_source = "unresolved"
