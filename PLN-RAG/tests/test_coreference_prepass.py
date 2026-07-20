from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace

from core.discourse import (
    DocumentCorefResult,
    MentionPrepass,
    project_coref_to_chunk,
)
from core.discourse.coreference import build_cluster
from core.extraction.langextract_chunker import (
    TextChunk,
    split_langextract_text_with_spans,
)


def span(text: str, needle: str, start: int = 0) -> tuple[int, int]:
    offset = text.index(needle, start)
    return offset, offset + len(needle)


def document_result(text: str, spans: list[tuple[int, int]], *, ambiguous=False):
    cluster = build_cluster(cluster_id="fake_c1", text=text, spans=spans)
    cluster.ambiguous = ambiguous or cluster.ambiguous
    if cluster.ambiguous:
        cluster.canonical_mention = None
    return DocumentCorefResult(
        text=text,
        clusters=[cluster],
        backend="fake",
        model_name="fake-coref",
    )


class CoreferenceMentionPrepassTests(unittest.TestCase):
    def test_basic_resolution_maps_pronoun_to_canonical_name(self):
        text = (
            "John met Alex after work. Alex explained the project. "
            "He later gave John the files."
        )
        alex = span(text, "Alex explained")
        he = span(text, "He later")
        doc = document_result(text, [span(text, "Alex", alex[0]), (he[0], he[0] + 2)])

        chunk_coref = project_coref_to_chunk(doc, 0, len(text), text)
        result = MentionPrepass().build(text, coref_result=chunk_coref)

        pronoun = next(mention for mention in result.mentions if mention.text == "He")
        self.assertEqual("Alex", pronoun.resolved_to)
        self.assertEqual("lingmess", pronoun.resolution_source)
        self.assertEqual(he[0], pronoun.global_start)
        self.assertIn("Use 'Alex' as the semantic", result.prompt_hint())

    def test_ambiguous_model_cluster_does_not_force_resolution(self):
        text = "John called Alex because he was worried."
        doc = document_result(
            text,
            [span(text, "John"), span(text, "Alex"), span(text, "he")],
            ambiguous=True,
        )

        result = MentionPrepass().build(
            text,
            coref_result=project_coref_to_chunk(doc, 0, len(text), text),
        )

        pronoun = next(mention for mention in result.mentions if mention.text == "he")
        self.assertIsNone(pronoun.resolved_to)
        self.assertTrue(pronoun.ambiguous)
        self.assertIn("Do not guess", result.prompt_hint())

    def test_cross_chunk_resolution_uses_document_level_antecedent(self):
        text = "Alex explained the project in detail.\n\nHe later gave John the files."
        chunks = split_langextract_text_with_spans(text, chunk_size=45)
        self.assertEqual(2, len(chunks))
        doc = document_result(text, [span(text, "Alex"), span(text, "He")])

        second = chunks[1]
        chunk_coref = project_coref_to_chunk(doc, second.start, second.end, second.text)
        result = MentionPrepass().build(
            second.text,
            coref_result=chunk_coref,
            global_offset=second.start,
        )

        pronoun = next(mention for mention in result.mentions if mention.text == "He")
        self.assertEqual("Alex", pronoun.resolved_to)
        self.assertEqual(second.start, pronoun.global_start)

    def test_possessive_pronoun_preserves_source_mention(self):
        text = "Alex submitted his report."
        doc = document_result(text, [span(text, "Alex"), span(text, "his")])

        result = MentionPrepass().build(
            text,
            coref_result=project_coref_to_chunk(doc, 0, len(text), text),
        )

        pronoun = next(mention for mention in result.mentions if mention.text == "his")
        self.assertEqual("Alex", pronoun.resolved_to)
        self.assertEqual("his", pronoun.text)

    def test_plural_mentions_can_resolve_to_group_nominal(self):
        text = "The engineers reviewed the design. They approved their plan."
        doc = document_result(
            text,
            [span(text, "engineers"), span(text, "They"), span(text, "their")],
        )

        result = MentionPrepass().build(
            text,
            coref_result=project_coref_to_chunk(doc, 0, len(text), text),
        )

        resolved = {
            mention.text: mention.resolved_to
            for mention in result.mentions
            if mention.text in {"They", "their"}
        }
        self.assertEqual({"They": "engineers", "their": "engineers"}, resolved)

    def test_pronoun_only_cluster_remains_unresolved(self):
        text = "He said his report was ready."
        doc = document_result(text, [span(text, "He"), span(text, "his")])

        result = MentionPrepass().build(
            text,
            coref_result=project_coref_to_chunk(doc, 0, len(text), text),
        )

        self.assertTrue(result.unresolved_pronouns)
        self.assertTrue(
            all(mention.resolved_to is None for mention in result.unresolved_pronouns)
        )

    def test_repeated_text_offsets_are_cursor_based(self):
        text = "Same sentence. Same sentence.\n\nSame sentence."
        chunks = split_langextract_text_with_spans(text, chunk_size=20)

        starts = [chunk.start for chunk in chunks if chunk.text == "Same sentence."]
        self.assertEqual([0, 15, 31], starts)
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.start : chunk.end])

    def test_query_without_antecedent_remains_unresolved(self):
        result = MentionPrepass().build("Did he submit the report?")

        self.assertEqual([], result.resolved_pronouns)
        self.assertEqual("", result.prompt_hint())


class ServiceCoreferenceLifecycleTests(unittest.TestCase):
    def setUp(self):
        if "pettachainer.pettachainer" not in sys.modules:
            package = ModuleType("pettachainer")
            module = ModuleType("pettachainer.pettachainer")
            module.PeTTaChainer = object
            package.pettachainer = module
            sys.modules["pettachainer"] = package
            sys.modules["pettachainer.pettachainer"] = module

    def test_resolver_runs_once_per_document_not_once_per_chunk(self):
        from core.orchestration.service import PLNRAGService

        text = "Alex explained the project.\n\nHe submitted the report."
        resolver = CountingResolver(text)
        parser = FakeParser()
        service = PLNRAGService.__new__(PLNRAGService)
        service._parser = parser
        service._chunker = TwoChunker()
        service._reasoner = FakeReasoner()
        service._vector_store = None
        service._context_top_k = 1
        service._coreference_enabled = True
        service._coreference_fail_open = True
        service._coreference_resolver = resolver

        result = service._ingest_single(text)

        self.assertEqual("success", result.status)
        self.assertEqual(1, resolver.calls)
        self.assertEqual(2, len(parser.prepasses))
        self.assertTrue(parser.prepasses[1].resolved_pronouns)

    def test_coreference_failure_fails_open_to_deterministic_prepass(self):
        from core.orchestration.service import PLNRAGService

        text = "Alex bought the camera."
        parser = FakeParser()
        service = PLNRAGService.__new__(PLNRAGService)
        service._parser = parser
        service._chunker = OneChunker()
        service._reasoner = FakeReasoner()
        service._vector_store = None
        service._context_top_k = 1
        service._coreference_enabled = True
        service._coreference_fail_open = True
        service._coreference_resolver = FailingResolver()

        result = service._ingest_single(text)

        self.assertEqual("success", result.status)
        self.assertEqual(1, len(parser.prepasses))
        self.assertEqual("boom", parser.prepasses[0].coreference.get("error"))


class CountingResolver:
    backend = "fake"
    model_name = "fake"

    def __init__(self, text: str):
        self.calls = 0
        self._text = text

    def resolve_document(self, text: str):
        self.calls += 1
        return document_result(text, [span(text, "Alex"), span(text, "He")])


class FailingResolver:
    backend = "fake"
    model_name = "fake"

    def resolve_document(self, text: str):
        raise RuntimeError("boom")


class FakeParser:
    def __init__(self):
        self.prepasses = []

    def build_mention_prepass(self, text, *, coref_result=None, global_offset=0):
        return MentionPrepass().build(
            text,
            coref_result=coref_result,
            global_offset=global_offset,
        )

    def parse(self, text, context, mention_prepass=None):
        self.prepasses.append(mention_prepass)
        return SimpleNamespace(
            statements=[
                f"(: fake_fact_{len(self.prepasses)} "
                "(FakeStatement test) (STV 1.0 1.0))"
            ],
            metadata={
                "statement_to_source": {},
                "mention_prepass": mention_prepass.to_dict(),
            },
        )


class FakeReasoner:
    def add_statements(self, statements, provenance=None):
        return statements


class OneChunker:
    def chunk_with_spans(self, text):
        return [TextChunk(index=0, text=text, start=0, end=len(text))]


class TwoChunker:
    def chunk_with_spans(self, text):
        split = text.index("\n\n") + 2
        return [
            TextChunk(index=0, text=text[: split - 2], start=0, end=split - 2),
            TextChunk(index=1, text=text[split:], start=split, end=len(text)),
        ]


if __name__ == "__main__":
    unittest.main()
