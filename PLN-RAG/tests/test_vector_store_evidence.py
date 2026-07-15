from __future__ import annotations

import unittest

from storage.vector_store import VectorStore


class _Response:
    def raise_for_status(self):
        return None


class _Client:
    def __init__(self):
        self.requests = []

    def put(self, url, json):
        self.requests.append((url, json))
        return _Response()


class EvidenceVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore.__new__(VectorStore)
        self.store._collection = "evidence_v2"
        self.store._qdrant = "http://qdrant:6333"
        self.store._client = _Client()
        self.store._ensure_evidence_collection = lambda _size: None

    def test_upsert_uses_named_evidence_vectors_and_linked_payload(self):
        record = {
            "index_id": "index-1",
            "claim_id": "claim-1",
            "evidence_id": "evidence-1",
            "claim_evidence_id": "claim-evidence-1",
            "retrieval_text": "M7 shut down automatically.",
            "nl": "It shut down automatically.",
            "pln": ["(: shutdown (ShutDownAutomatically m7) (STV 1.0 1.0))"],
            "query_targets": ["(ShutDownAutomatically m7)"],
            "validation_state": "accepted",
            "outbox_id": 42,
        }

        self.store._upsert_evidence_records([record], [[0.1, 0.2]])

        _url, body = self.store._client.requests[0]
        point = body["points"][0]
        self.assertEqual([0.1, 0.2], point["vector"]["dense"])
        self.assertIn("lexical", point["vector"])
        self.assertEqual("claim-1", point["payload"]["claim_id"])
        self.assertEqual(
            ["(ShutDownAutomatically m7)"],
            point["payload"]["query_targets"],
        )
        self.assertNotIn("outbox_id", point["payload"])

    def test_sparse_vector_is_deterministic_and_sorted(self):
        first = self.store._sparse_vector("Atlas alert Atlas")
        second = self.store._sparse_vector("Atlas alert Atlas")

        self.assertEqual(first, second)
        self.assertEqual(sorted(first["indices"]), first["indices"])
        self.assertEqual(2, len(first["indices"]))


if __name__ == "__main__":
    unittest.main()
