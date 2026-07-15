from __future__ import annotations

import unittest

from core.pln.postprocessor import PLNPostprocessor


class TypeInferenceTests(unittest.TestCase):
    def test_extract_proper_name_map_tracks_capitalized_names(self):
        postprocessor = PLNPostprocessor()

        proper_name_map = postprocessor.extract_proper_name_map(
            "Abebe eats pasta almost every day, often in large portions."
        )

        self.assertEqual("abebe", proper_name_map["abebe"])

    def test_infer_entity_types_adds_person_for_subject_names(self):
        postprocessor = PLNPostprocessor()
        statements = [
            "(: abebe_pasta_eats_frequently_fact "
            "(EatsFrequently abebe pasta) (STV 1.0 1.0))",
            "(: abebe_eats_large_portions_fact "
            "(EatsLargePortions abebe) (STV 1.0 1.0))",
            "(: pasta_is_refined_fact (IsRefined pasta) (STV 1.0 1.0))",
        ]

        inferred = postprocessor.infer_entity_types(
            statements,
            {"abebe": "abebe", "pasta": "pasta"},
        )

        self.assertTrue(any("IsA abebe person" in item for item in inferred))
        self.assertFalse(any("IsA pasta person" in item for item in inferred))

    def test_full_process_pipeline_adds_required_person_type(self):
        postprocessor = PLNPostprocessor()
        result = postprocessor.process(
            text=(
                "Abebe eats pasta almost every day, often in large portions, "
                "and prefers refined pasta over whole grain."
            ),
            statements=[
                "(: abebe_pasta_eats_frequently_fact "
                "(EatsFrequently abebe pasta) (STV 1.0 1.0))",
                "(: abebe_eats_large_portions_fact "
                "(EatsLargePortions abebe) (STV 1.0 1.0))",
                "(: pasta_is_refined_fact (IsRefined pasta) (STV 1.0 1.0))",
                "(: pasta_is_whole_grain_neg "
                "(Not (IsWholeGrain pasta)) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertTrue(
            any("IsA abebe person" in item for item in result.statements)
        )
        self.assertTrue(
            any("EatsLargePortions abebe pasta" in item for item in result.statements)
        )

    def test_rule_requiring_person_type_gets_supporting_facts(self):
        postprocessor = PLNPostprocessor()
        result = postprocessor.process(
            text=(
                "People who frequently eat pasta tend to consume higher "
                "amounts of carbohydrates. Abebe eats pasta almost every day."
            ),
            statements=[
                "(: x_consumes_high_carbs_rule "
                "(Implication (Premises (IsA $x person) "
                "(EatsFrequently $x $food) (IsA $food pasta)) "
                "(Conclusions (ConsumesHighCarbs $x))) (STV 1.0 1.0))",
                "(: abebe_pasta_eats_frequently_fact "
                "(EatsFrequently abebe pasta) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertTrue(
            any("IsA abebe person" in item for item in result.statements)
        )
        self.assertFalse(
            any("IsA pasta pasta" in item for item in result.statements)
        )
        self.assertTrue(
            any("EatsFrequently abebe pasta" in item for item in result.statements)
        )


if __name__ == "__main__":
    unittest.main()
