from __future__ import annotations

import unittest
import sys
from types import ModuleType

if "pettachainer.pettachainer" not in sys.modules:
    package = ModuleType("pettachainer")
    module = ModuleType("pettachainer.pettachainer")
    module.PeTTaChainer = object
    package.pettachainer = module
    sys.modules["pettachainer"] = package
    sys.modules["pettachainer.pettachainer"] = module

from core.orchestration.service import PLNRAGService


class QueryOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.service = PLNRAGService.__new__(PLNRAGService)

    def test_evidence_linked_qdrant_target_is_executable(self):
        candidates = self.service._query_candidates(
            "Is Abebe consuming excessive carbohydrates?",
            ["(: $prf (ConsumesHigherCarbohydrates abebe) $tv)"],
            [],
        )

        self.assertEqual(
            [
                (
                    "(: $prf (ConsumesHigherCarbohydrates abebe) $tv)",
                    "qdrant_alignment",
                )
            ],
            candidates,
        )

    def test_vector_store_outage_fails_open(self):
        class FailingVectorStore:
            def retrieve_context(self, _text, top_k):
                raise RuntimeError("offline")

            def search(self, _text, top_k):
                raise RuntimeError("offline")

        self.service._vector_store = FailingVectorStore()
        self.service._context_top_k = 3

        self.assertEqual(([], []), self.service._retrieve_ingest_context("text"))
        self.assertEqual(
            ([], []),
            self.service._qdrant_context_with_matches("question", top_k=3),
        )

    def test_qdrant_target_with_wrong_question_intent_is_rejected(self):
        candidates = self.service._query_candidates(
            "Is Abebe currently obese?",
            ["(: $prf (HasIncreasedWeight abebe) $tv)"],
            [],
        )

        self.assertEqual([], candidates)

    def test_alignment_rejects_legacy_chunk_payload_when_ledger_is_enabled(self):
        self.service._evidence_ledger = object()
        matches = self.service._alignment_matches(
            [
                {
                    "score": 0.99,
                    "validation_state": "",
                    "claim_evidence_id": "",
                },
                {
                    "score": 0.9,
                    "validation_state": "accepted",
                    "claim_evidence_id": "linked-record",
                },
            ]
        )

        self.assertEqual(1, len(matches))
        self.assertEqual("linked-record", matches[0]["claim_evidence_id"])

    def test_alignment_uses_hybrid_score_threshold(self):
        self.service._evidence_ledger = object()
        matches = self.service._alignment_matches(
            [
                {
                    "score": 0.4,
                    "score_kind": "hybrid_rrf",
                    "validation_state": "accepted",
                    "claim_evidence_id": "hybrid-record",
                },
                {
                    "score": 0.4,
                    "score_kind": "dense_cosine",
                    "validation_state": "accepted",
                    "claim_evidence_id": "dense-record",
                },
            ]
        )

        self.assertEqual(
            ["hybrid-record"],
            [item["claim_evidence_id"] for item in matches],
        )

    def test_candidate_with_wrong_known_arity_is_rejected(self):
        class FakeReasoner:
            def predicate_arities(self):
                return {"QualifiesForMeritScholarship": {1}}

        self.service._reasoner = FakeReasoner()

        candidates = self.service._query_candidates(
            "Does Lena qualify for a merit scholarship?",
            [],
            [
                "(: $prf (QualifiesForMeritScholarship lena merit) $tv)",
                "(: $prf (QualifiesForMeritScholarship lena) $tv)",
            ],
        )

        self.assertEqual(
            [("(: $prf (QualifiesForMeritScholarship lena) $tv)", "parser")],
            candidates,
        )

    def test_matching_parser_target_is_executable(self):
        candidates = self.service._query_candidates(
            "Is Abebe consuming excessive carbohydrates?",
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
            ["(: $prf (ConsumesHigherCarbohydrates abebe) $tv)"],
        )

        self.assertEqual(
            [("(: $prf (ConsumesHigherCarbohydrates abebe) $tv)", "parser")],
            candidates,
        )

    def test_rejected_parser_target_is_not_restored(self):
        candidates = self.service._query_candidates(
            "Is Abebe currently obese?",
            [],
            ["(: $prf (HasIncreasedWeight abebe) $tv)"],
        )

        self.assertEqual([], candidates)

    def test_sufficiency_never_executes_supporting_fact(self):
        candidates = self.service._query_candidates(
            "Is Abebe's pasta consumption alone sufficient to cause diabetes?",
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
        )

        self.assertEqual([], candidates)

    def test_trusted_atomspace_conclusion_recovers_missing_parser_query(self):
        class FakeReasoner:
            def proposition_signatures(self):
                return [
                    {
                        "head": "ClassifiedAsPolluted",
                        "args": ["$lake"],
                        "arity": 1,
                        "variables": ["$lake"],
                        "negated": False,
                        "role": "conclusion",
                    },
                    {
                        "head": "IsA",
                        "args": ["lake_aster", "lake"],
                        "arity": 2,
                        "variables": [],
                        "negated": False,
                        "role": "fact",
                    },
                ]

            def predicate_arities(self):
                return {"ClassifiedAsPolluted": {1}}

        self.service._reasoner = FakeReasoner()
        trusted = self.service._deterministic_query_candidates(
            "Is Lake Aster classified as polluted?"
        )
        candidates = self.service._query_candidates(
            "Is Lake Aster classified as polluted?",
            [],
            [],
            trusted,
        )

        self.assertEqual(
            [
                (
                    "(: $prf (ClassifiedAsPolluted lake_aster) $tv)",
                    "deterministic",
                )
            ],
            candidates,
        )


if __name__ == "__main__":
    unittest.main()
