from __future__ import annotations

import unittest

from core.pln.postprocessor import PLNPostprocessor


class QueryPlanningTests(unittest.TestCase):
    def setUp(self):
        self.postprocessor = PLNPostprocessor()

    def test_should_question_falls_back_to_relevant_rule_conclusion(self):
        result = self.postprocessor.process(
            text="Should Nia be flagged for urgent respiratory review?",
            statements=[],
            queries=[],
            context=[
                "(: urgent_rule (Implication (Premises "
                "(HasShortnessOfBreath $p) "
                "(HasOxygenSaturationBelow $p 92percent)) "
                "(Conclusions (FlaggedForUrgentRespiratoryReview $p))) "
                "(STV 1.0 1.0))"
            ],
            plan_queries=True,
        )

        self.assertIn(
            "(: $prf (FlaggedForUrgentRespiratoryReview nia) $tv)",
            result.queries,
        )

    def test_negative_fact_can_supply_boolean_query_target(self):
        result = self.postprocessor.process(
            text="Did Lena lead a recognized club?",
            statements=[],
            queries=[],
            context=[
                "(: club_neg (Not (LedRecognizedClub lena)) (STV 1.0 1.0))"
            ],
            plan_queries=True,
        )

        self.assertIn(
            "(: $prf (LedRecognizedClub lena) $tv)",
            result.queries,
        )


if __name__ == "__main__":
    unittest.main()
