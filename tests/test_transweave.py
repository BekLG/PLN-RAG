import unittest

from core.senf import SENF, SENFEntity
from core.transweave import TransWeaveAligner


class TransWeaveAlignerTests(unittest.TestCase):
    def setUp(self):
        self.aligner = TransWeaveAligner()

    def test_same_kind_is_not_identity(self):
        left = SENF(entities={"soccer": SENFEntity("soccer", "sport")})
        right = SENF(entities={"sport": SENFEntity("sport", "sport")})
        self.assertEqual(self.aligner.build_weaves(left, right), [])

    def test_unrelated_concepts_are_not_identity(self):
        left = SENF(entities={"sport": SENFEntity("sport", "concept")})
        right = SENF(entities={"healthy": SENFEntity("healthy", "concept")})
        self.assertEqual(self.aligner.build_weaves(left, right), [])

    def test_literal_identity_is_not_persisted(self):
        left = SENF(entities={"soccer": SENFEntity("soccer", "sport")})
        right = SENF(entities={"soccer": SENFEntity("soccer", "activity")})
        self.assertEqual(self.aligner.build_weaves(left, right), [])

    def test_declared_equivalence_is_mapped(self):
        left = SENF(entities={"soccer": SENFEntity("soccer", "sport")})
        right = SENF(entities={"football": SENFEntity("football", "activity")})
        weave = self.aligner.build_weaves(left, right)[0]
        self.assertEqual(weave.entity_map, {"soccer": "football"})


if __name__ == "__main__":
    unittest.main()
