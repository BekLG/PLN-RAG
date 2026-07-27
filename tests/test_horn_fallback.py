import unittest

from core.horn_fallback import HornFallback


class HornFallbackTests(unittest.TestCase):
    def test_direct_fact(self):
        statements = ["(: fact (Eats kebede fish) (STV 1.0 1.0))"]
        self.assertTrue(HornFallback(statements).prove("(Eats kebede fish)"))

    def test_one_hop_rule(self):
        statements = [
            "(: fact (IsA soccer sport) (STV 1.0 1.0))",
            "(: rule (Implication (Premises (IsA $x sport)) (Conclusions (Healthy $x))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(Healthy soccer)"))

    def test_empty_premise_rule(self):
        statements = [
            "(: fact_rule (Implication (Premises) (Conclusions (IsA soccer sport))) (STV 1.0 1.0))",
            "(: healthy_rule (Implication (Premises (IsA $x sport)) (Conclusions (Healthy $x))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(Healthy soccer)"))

    def test_multi_hop_rule(self):
        statements = [
            "(: fact (A item) (STV 1.0 1.0))",
            "(: first (Implication (Premises (A $x)) (Conclusions (B $x))) (STV 1.0 1.0))",
            "(: second (Implication (Premises (B $x)) (Conclusions (C $x))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(C item)"))

    def test_multiple_premises(self):
        statements = [
            "(: first_fact (A item) (STV 1.0 1.0))",
            "(: second_fact (B item) (STV 1.0 1.0))",
            "(: rule (Implication (Premises (A $x) (B $x)) (Conclusions (C $x))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(C item)"))

    def test_multiple_premises_backtrack(self):
        statements = [
            "(: first_wrong (A wrong) (STV 1.0 1.0))",
            "(: first_right (A right) (STV 1.0 1.0))",
            "(: second_right (B right) (STV 1.0 1.0))",
            "(: rule (Implication (Premises (A $x) (B $x)) (Conclusions (C))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(C)"))

    def test_soccer_football_equivalence(self):
        statements = [
            "(: fact_rule (Implication (Premises) (Conclusions (IsA soccer sport))) (STV 1.0 1.0))",
            "(: healthy_rule (Implication (Premises (IsA $x sport)) (Conclusions (Healthy $x))) (STV 1.0 1.0))",
        ]
        self.assertTrue(HornFallback(statements).prove("(Healthy football)"))

    def test_declared_equivalences(self):
        pairs = [
            ("aircraft", "plane"),
            ("automobile", "car"),
            ("bicycle", "bike"),
            ("canine", "dog"),
            ("couch", "sofa"),
            ("feline", "cat"),
            ("infant", "baby"),
            ("mobile_phone", "cellphone"),
            ("physician", "doctor"),
        ]
        for query_symbol, fact_symbol in pairs:
            with self.subTest(query_symbol=query_symbol):
                statements = [
                    f"(: fact (Known {fact_symbol}) (STV 1.0 1.0))",
                ]
                self.assertTrue(
                    HornFallback(statements).prove(f"(Known {query_symbol})")
                )

    def test_unrelated_entity_is_not_proven(self):
        statements = [
            "(: fact (IsA soccer sport) (STV 1.0 1.0))",
            "(: rule (Implication (Premises (IsA $x sport)) (Conclusions (Healthy $x))) (STV 1.0 1.0))",
        ]
        self.assertFalse(HornFallback(statements).prove("(Healthy basketball)"))


if __name__ == "__main__":
    unittest.main()
