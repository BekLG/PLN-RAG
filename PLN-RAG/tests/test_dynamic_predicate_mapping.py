from __future__ import annotations

import unittest

from core.pln.postprocessor import PLNPostprocessor
from core.pln.predicate_mapping import (
    PredicateMappingEngine,
    RelationProposal,
)
from core.pln.predicate_registry import PredicateRegistry
from core.pln.schema_alignment import PLNSchemaAligner


class FakeClassifier:
    def __init__(self, relations):
        self.relations = relations

    def classify(self, source, target):
        relation, confidence, argument_mapping = self.relations.get(
            (source.predicate, target.predicate),
            ("unrelated", 0.99, list(range(source.arity))),
        )
        return RelationProposal(
            relation=relation,
            confidence=confidence,
            reason="fake_classifier_result",
            classifier="fake",
            argument_mapping=argument_mapping,
        )


def registry_with(relations, *, threshold=0.82):
    aligner = PLNSchemaAligner(PLNPostprocessor.STRUCTURAL_HEADS)
    engine = PredicateMappingEngine(
        classifier=FakeClassifier(relations),
        proof_threshold=threshold,
        retrieval_min_score=0.62,
        retrieval_top_k=8,
        max_candidates=16,
    )
    return PredicateRegistry(
        schema_aligner=aligner,
        mapping_engine=engine,
        autosave=False,
    )


class DynamicPredicateMappingTests(unittest.TestCase):
    def test_carbohydrate_mapping_creates_validated_bridge(self):
        registry = registry_with(
            {
                (
                    "ConsumesHigherCarbohydrates",
                    "HasHighCarbohydrateIntake",
                ): ("source_implies_target", 0.91, [0])
            }
        )
        statements = [
            "(: carb_rule (Implication (Premises (IsA $x person) "
            "(EatsFrequently $x pasta)) (Conclusions "
            "(ConsumesHigherCarbohydrates $x))) (STV 1.0 1.0))",
            "(: obesity_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x) "
            "(TotalCalorieIntakeExceedsEnergyExpenditure $x)) "
            "(Conclusions (LeadsToObesity $x))) (STV 1.0 1.0))",
        ]

        _, decisions = registry.align_statements(
            statements=statements,
            context=[],
            source_text="People who eat pasta consume more carbohydrates.",
        )
        bridges, bridge_decisions = registry.build_validated_bridges(
            statements,
            [],
        )

        self.assertTrue(
            any(item.get("action") == "mapping_approved" for item in decisions)
        )
        self.assertEqual(1, len(bridges))
        self.assertIn(
            "(Premises (ConsumesHigherCarbohydrates $x))",
            bridges[0],
        )
        self.assertIn(
            "(Conclusions (HasHighCarbohydrateIntake $x))",
            bridges[0],
        )
        self.assertTrue(bridge_decisions[0]["proof_safe"])

    def test_related_mapping_never_becomes_bridge(self):
        registry = registry_with(
            {
                ("ConsumesSugar", "HasHighCarbohydrateIntake"): (
                    "related",
                    0.99,
                    [0],
                )
            }
        )
        statements = [
            "(: sugar_fact (ConsumesSugar abebe) (STV 1.0 1.0))",
            "(: carb_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x)) "
            "(Conclusions (AtRiskOfObesity $x))) (STV 1.0 1.0))",
        ]

        registry.align_statements(
            statements=statements,
            context=[],
            source_text="Abebe consumes sugar.",
        )
        bridges, decisions = registry.build_validated_bridges(statements, [])

        self.assertEqual([], bridges)
        self.assertFalse(
            any(item.get("action") == "bridge_added" for item in decisions)
        )

    def test_low_confidence_mapping_is_recorded_but_not_proof_safe(self):
        registry = registry_with(
            {
                ("ConsumesHigherCarbohydrates", "HasHighCarbohydrateIntake"): (
                    "source_implies_target",
                    0.61,
                    [0],
                )
            }
        )
        statements = [
            "(: carb_fact (ConsumesHigherCarbohydrates abebe) (STV 1.0 1.0))",
            "(: obesity_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x)) "
            "(Conclusions (AtRiskOfObesity $x))) (STV 1.0 1.0))",
        ]

        _, decisions = registry.align_statements(
            statements=statements,
            context=[],
            source_text="Abebe consumes higher carbohydrates.",
        )
        bridges, _ = registry.build_validated_bridges(statements, [])

        self.assertEqual([], bridges)
        self.assertTrue(
            any(
                item.get("action") == "mapping_recorded_not_proof_safe"
                and "confidence_below_threshold" in item.get("reason", "")
                for item in decisions
            )
        )

    def test_argument_order_mismatch_rejects_bridge(self):
        registry = registry_with(
            {
                ("HasOwner", "OwnerHas"): (
                    "source_implies_target",
                    0.95,
                    [1, 0],
                )
            }
        )
        statements = [
            "(: owner_fact (HasOwner camera abebe) (STV 1.0 1.0))",
            "(: owner_rule (Implication (Premises "
            "(OwnerHas $person $thing)) "
            "(Conclusions (CanUse $person $thing))) (STV 1.0 1.0))",
        ]

        _, decisions = registry.align_statements(
            statements=statements,
            context=[],
            source_text="Abebe owns a camera.",
        )
        bridges, _ = registry.build_validated_bridges(statements, [])

        self.assertEqual([], bridges)
        self.assertTrue(
            any(
                item.get("action") == "mapping_recorded_not_proof_safe"
                and "argument_order_not_identity" in item.get("reason", "")
                for item in decisions
            )
        )


if __name__ == "__main__":
    unittest.main()
