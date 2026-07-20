from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.evidence.records import build_claim_evidence_record
from core.query.alignment import build_aligned_queries
from storage.evidence_ledger import EvidenceLedger


class EvidenceRecordTests(unittest.TestCase):
    def test_exact_source_builds_retrievable_claim(self):
        text = "M7 exceeded 90C. It shut down automatically."
        source_text = "It shut down automatically."
        start = text.index(source_text)
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom=(
                "(: m7_shutdown (ShutDownAutomatically m7) "
                "(STV 1.0 1.0))"
            ),
            chunk_text=text,
            chunk_index=0,
            chunk_start=100,
            source={
                "text": source_text,
                "char_interval": {
                    "start_pos": start,
                    "end_pos": start + len(source_text),
                },
                "class": "fact",
            },
            mention_prepass={
                "mentions": [
                    {
                        "text": "It",
                        "start": start,
                        "end": start + 2,
                        "resolved_to": "M7",
                        "resolution_source": "lingmess",
                        "ambiguous": False,
                    }
                ]
            },
        )

        self.assertEqual("accepted", record.validation_state)
        self.assertEqual(["(ShutDownAutomatically m7)"], record.query_targets)
        self.assertIn("It refers to M7", record.retrieval_text)
        self.assertEqual(100 + start, record.global_source_start)
        self.assertEqual(1, len(record.qdrant_records()))

    def test_explicit_negative_is_indexed_as_positive_polarity_query(self):
        text = "M7 did not shut down automatically."
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom=(
                "(: m7_shutdown_neg (Not (ShutDownAutomatically m7)) "
                "(STV 1.0 1.0))"
            ),
            chunk_text=text,
            chunk_index=0,
            chunk_start=0,
            source={
                "text": text,
                "char_interval": {"start_pos": 0, "end_pos": len(text)},
                "class": "negative_fact",
            },
        )

        self.assertEqual(["(ShutDownAutomatically m7)"], record.query_targets)
        self.assertEqual("negative", record.targets[0].polarity)

    def test_invalid_source_is_quarantined(self):
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom="(: false_fact (Obese abebe) (STV 1.0 1.0))",
            chunk_text="Abebe gained some weight.",
            chunk_index=0,
            chunk_start=0,
            source={
                "text": "Abebe is obese.",
                "char_interval": {"start_pos": 0, "end_pos": 16},
                "class": "fact",
            },
        )

        self.assertEqual("quarantined", record.validation_state)
        self.assertFalse(record.is_reasoner_eligible)
        self.assertEqual([], record.qdrant_records())

    def test_introduced_argument_is_quarantined(self):
        text = "M7 shut down automatically."
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom=(
                "(: wrong_target (Activate shutdown_controller) "
                "(STV 1.0 1.0))"
            ),
            chunk_text=text,
            chunk_index=0,
            chunk_start=0,
            source={
                "text": text,
                "char_interval": {"start_pos": 0, "end_pos": len(text)},
                "class": "fact",
            },
        )

        self.assertEqual("quarantined", record.validation_state)
        self.assertIn(
            "ungrounded claim argument: shutdown_controller",
            record.validation_errors,
        )

    def test_negative_target_without_source_negation_is_quarantined(self):
        text = "M7 shut down automatically."
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom=(
                "(: unsupported_neg (Not (ShutDownAutomatically m7)) "
                "(STV 1.0 1.0))"
            ),
            chunk_text=text,
            chunk_index=0,
            chunk_start=0,
            source={
                "text": text,
                "char_interval": {"start_pos": 0, "end_pos": len(text)},
                "class": "fact",
            },
        )

        self.assertEqual("quarantined", record.validation_state)
        self.assertIn(
            "negative claim lacks explicit source negation",
            record.validation_errors,
        )

    def test_normalizer_generated_claim_is_not_retrieval_authority(self):
        record = build_claim_evidence_record(
            document_id="doc",
            evidence_id="evidence",
            atom="(: abebe_type (IsA abebe person) (STV 1.0 1.0))",
            chunk_text="Abebe eats pasta.",
            chunk_index=0,
            chunk_start=0,
            source=None,
        )

        self.assertEqual("derived", record.validation_state)
        self.assertTrue(record.is_reasoner_eligible)
        self.assertEqual([], record.qdrant_records())


class EvidenceLedgerTests(unittest.TestCase):
    def test_ledger_records_claim_and_rebuild_material(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = EvidenceLedger(str(Path(temp_dir) / "evidence.db"))
            text = "Lena qualifies for the scholarship."
            document_id = ledger.record_document(text)
            evidence_id = ledger.record_evidence(
                document_id=document_id,
                chunk_index=0,
                start_pos=0,
                end_pos=len(text),
                text=text,
            )
            record = build_claim_evidence_record(
                document_id=document_id,
                evidence_id=evidence_id,
                atom=(
                    "(: lena_qualifies (QualifiesForScholarship lena) "
                    "(STV 1.0 1.0))"
                ),
                chunk_text=text,
                chunk_index=0,
                chunk_start=0,
                source={
                    "text": text,
                    "char_interval": {"start_pos": 0, "end_pos": len(text)},
                    "class": "fact",
                },
            )
            ledger.record_claims([record])

            self.assertEqual(1, ledger.document_count)
            self.assertEqual(1, ledger.evidence_count)
            self.assertEqual(1, ledger.claim_count)
            self.assertEqual(1, ledger.pending_count)
            self.assertEqual(record.atom, ledger.reasoner_records()[0]["atom"])
            pending = ledger.pending_index_records()
            self.assertEqual(record.claim_evidence_id, pending[0]["claim_evidence_id"])
            ledger.mark_indexed([pending[0]["outbox_id"]])
            self.assertEqual(0, ledger.pending_count)


class EvidenceQueryAlignmentTests(unittest.TestCase):
    def test_negative_evidence_builds_positive_query_target(self):
        result = build_aligned_queries(
            "Did M7 shut down automatically?",
            [
                {
                    "score": 0.9,
                    "nl": "M7 did not shut down automatically.",
                    "pln": [
                        "(: shutdown_neg (Not (ShutDownAutomatically m7)) "
                        "(STV 1.0 1.0))"
                    ],
                    "query_targets": ["(ShutDownAutomatically m7)"],
                }
            ],
        )

        self.assertEqual(
            ["(: $prf (ShutDownAutomatically m7) $tv)"],
            result.queries,
        )


if __name__ == "__main__":
    unittest.main()
