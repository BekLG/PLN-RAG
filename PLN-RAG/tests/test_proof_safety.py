from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.extraction.langextract_pln import translate_extractions_to_pln
from core.pln.postprocessor import PLNPostprocessor


def extraction(kind: str, text: str, **attributes):
    return SimpleNamespace(
        extraction_class=kind,
        extraction_text=text,
        attributes=attributes,
    )


class ProofSafetyTests(unittest.TestCase):
    def test_epistemic_absence_is_not_direct_property_negation(self):
        translated = translate_extractions_to_pln(
            [
                extraction(
                    "negation",
                    "has not been clinically classified as obese",
                    predicate="obese",
                    subject="Abebe",
                )
            ],
            source_text="Abebe has not been clinically classified as obese.",
        )

        self.assertEqual([], translated.statements)
        self.assertIn("epistemic", translated.rejected[0].reason)

    def test_status_negation_is_allowed(self):
        translated = translate_extractions_to_pln(
            [
                extraction(
                    "negation",
                    "has not been clinically classified as obese",
                    predicate="clinically-classified-as-obese",
                    subject="Abebe",
                )
            ],
            source_text="Abebe has not been clinically classified as obese.",
        )

        self.assertEqual(1, len(translated.statements))
        self.assertIn("Not (ClinicallyClassifiedAsObese abebe)", translated.statements[0])

    def test_hedged_rule_is_not_absolute(self):
        translated = translate_extractions_to_pln(
            [
                extraction(
                    "rule",
                    "People who eat pasta tend to consume more carbohydrates",
                    head_predicate="consumes-higher-carbohydrates",
                    head_args="$x",
                    body="(eats $x pasta)",
                )
            ]
        )

        self.assertIn("(STV 0.70 0.80)", translated.statements[0])

    def test_missing_rule_premises_are_never_invented(self):
        result = PLNPostprocessor().process(
            text="People who eat fish are smart.",
            statements=[
                "(: smart_rule (Implication (Premises (Eats $x fish)) "
                "(Conclusions (Smart $x))) (STV 1.0 1.0))"
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertFalse(any("materialized_" in item for item in result.statements))
        self.assertFalse(any("(Eats " in item and "Implication" not in item for item in result.statements))

    def test_portion_repair_uses_one_stable_arity(self):
        result = PLNPostprocessor().process(
            text="Abebe eats pasta frequently in large portions.",
            statements=[
                "(: eats_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))",
                "(: portions_fact (EatsLargePortions abebe) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertTrue(any("(EatsLargePortions abebe pasta)" in item for item in result.statements))
        self.assertFalse(any("(EatsLargePortions abebe) " in item for item in result.statements))

    def test_established_predicate_arity_rejects_conflicting_fact(self):
        result = PLNPostprocessor().process(
            text="Dana owns a laptop.",
            statements=[
                "(: bad_owns_fact (Owns dana) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[
                "(: owns_fact (Owns alex laptop) (STV 1.0 1.0))",
            ],
            plan_queries=False,
        )

        self.assertFalse(any("(Owns dana)" in item for item in result.statements))
        self.assertTrue(
            any(
                item.get("reason") == "predicate_arity_conflict"
                for item in result.registry_decisions
            )
        )

    def test_missing_universal_variable_in_person_rule_is_repaired(self):
        post = PLNPostprocessor()
        statement = (
            "(: risk_rule (Implication (Premises (IsA person) (Obese) "
            "(Or (Sedentary) (HasFamilyHistoryOfDiabetes))) "
            "(Conclusions (AtElevatedDiabetesRisk))) (STV 1.0 1.0))"
        )

        repaired = post.repair_missing_universal_variable(statement)

        self.assertIn("(IsA $x person)", repaired)
        self.assertIn("(Obese $x)", repaired)
        self.assertIn("(Sedentary $x)", repaired)
        self.assertIn("(HasFamilyHistoryOfDiabetes $x)", repaired)
        self.assertIn("(AtElevatedDiabetesRisk $x)", repaired)

    def test_semantic_mapping_bridges_are_suppressed_by_default(self):
        class Registry:
            def align_statements(self, **kwargs):
                return kwargs["statements"], []

            def build_validated_bridges(self, statements, context):
                return [
                    "(: unsafe (Implication (Premises (CanIncreaseTemperature $x)) "
                    "(Conclusions (Overheating $x))) (STV 0.9 0.8))"
                ], []

        result = PLNPostprocessor(predicate_registry=Registry()).process(
            text="A blocked filter can increase temperature.",
            statements=[
                "(: filter_fact (CanIncreaseTemperature blocked_filter) "
                "(STV 1.0 1.0))"
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertFalse(any("(Overheating $x)" in item for item in result.statements))
        self.assertTrue(
            any(
                item.get("action") == "semantic_bridges_suppressed"
                for item in result.alignment_decisions
            )
        )

    def test_unencoded_modal_rule_is_rejected(self):
        translated = translate_extractions_to_pln(
            [
                extraction(
                    "rule",
                    "Overheating can trigger automatic shutdown.",
                    head_predicate="triggers-automatic-shutdown",
                    head_args="$x",
                    body="(overheating $x)",
                )
            ]
        )

        self.assertEqual([], translated.statements)
        self.assertIn("modal", translated.rejected[0].reason)

    def test_modal_rule_is_allowed_when_predicate_encodes_modality(self):
        translated = translate_extractions_to_pln(
            [
                extraction(
                    "rule",
                    "Overheating can trigger automatic shutdown.",
                    head_predicate="can-trigger-automatic-shutdown",
                    head_args="$x",
                    body="(overheating $x)",
                )
            ]
        )

        self.assertEqual(1, len(translated.statements))
        self.assertIn("CanTriggerAutomaticShutdown", translated.statements[0])


if __name__ == "__main__":
    unittest.main()
