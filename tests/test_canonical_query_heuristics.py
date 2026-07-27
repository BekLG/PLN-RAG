import sys
import types
import unittest

sys.modules.setdefault("dspy", types.SimpleNamespace())

from parsers.canonical_pln_parser import CanonicalPLNParser


class CanonicalQueryHeuristicTests(unittest.TestCase):
    def setUp(self):
        self.parser = CanonicalPLNParser.__new__(CanonicalPLNParser)

    def test_single_word_subject(self):
        queries = self.parser._build_heuristic_question_queries(
            "is physician educated"
        )
        self.assertIn("(: $prf (IsA physician educated) $tv)", queries)
        self.assertIn("(: $prf (Educated physician) $tv)", queries)

    def test_multiword_subject(self):
        queries = self.parser._build_heuristic_question_queries(
            "is mobile phone electronic"
        )
        self.assertIn(
            "(: $prf (IsA mobile_phone electronic) $tv)",
            queries,
        )
        self.assertIn(
            "(: $prf (Electronic mobile_phone) $tv)",
            queries,
        )

    def test_article_is_removed_from_subject(self):
        queries = self.parser._build_heuristic_question_queries(
            "is a bicycle ecofriendly"
        )
        self.assertIn("(: $prf (IsA bicycle ecofriendly) $tv)", queries)

    def test_simple_copular_fact_is_materialized(self):
        facts = self.parser._materialize_simple_copular_facts(
            ["cat is animal"],
            [],
        )
        self.assertEqual(
            facts,
            [
                "(: canonical_cat_animal_fact "
                "(IsA cat animal) (STV 1.0 1.0))"
            ],
        )

    def test_existing_copular_fact_is_not_duplicated(self):
        statements = [
            "(: cat_fact (IsA cat animal) (STV 1.0 1.0))",
        ]
        facts = self.parser._materialize_simple_copular_facts(
            ["cat is animal"],
            statements,
        )
        self.assertEqual(facts, [])

    def test_complex_sentence_is_not_materialized(self):
        facts = self.parser._materialize_simple_copular_facts(
            ["people who eat fish are smart"],
            [],
        )
        self.assertEqual(facts, [])


if __name__ == "__main__":
    unittest.main()
