from __future__ import annotations

import sys
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

if "pettachainer.pettachainer" not in sys.modules:
    package = ModuleType("pettachainer")
    module = ModuleType("pettachainer.pettachainer")
    module.PeTTaChainer = object
    package.pettachainer = module
    sys.modules["pettachainer"] = package
    sys.modules["pettachainer.pettachainer"] = module

from core.reasoning.reasoner import Reasoner


class ReasonerPolarityTests(unittest.TestCase):
    def setUp(self):
        self.reasoner = Reasoner.__new__(Reasoner)

    def test_explicit_negative_is_a_negative_answer(self):
        self.reasoner.query = lambda query, seed_terms=None: (
            ["negative-proof"] if "(Not " in query else []
        )

        outcome = self.reasoner.query_polarity(
            "(: $prf (Authorized alice) $tv)"
        )

        self.assertEqual("negative", outcome.status)
        self.assertEqual(["negative-proof"], outcome.negative_proof)

    def test_add_statements_is_idempotent(self):
        class FakeHandler:
            def __init__(self):
                self.atoms = []

            def add_atom(self, atom):
                self.atoms.append(atom)

        with tempfile.TemporaryDirectory() as directory:
            reasoner = Reasoner.__new__(Reasoner)
            reasoner._atomspace_path = str(Path(directory) / "kb.metta")
            reasoner._provenance_path = f"{reasoner._atomspace_path}.provenance.jsonl"
            reasoner._lock = threading.Lock()
            reasoner._handler = FakeHandler()
            reasoner._atom_keys = set()
            atom = "(: online (Online camera_2) (STV 1.0 1.0))"

            first = reasoner.add_statements([atom])
            second = reasoner.add_statements([atom])

            self.assertEqual([atom], first)
            self.assertEqual([], second)
            self.assertEqual([atom], reasoner._handler.atoms)
            self.assertEqual(
                [atom],
                Path(reasoner._atomspace_path).read_text(encoding="utf-8").splitlines(),
            )

    def test_contradiction_is_reported_as_both(self):
        self.reasoner.query = lambda query, seed_terms=None: ["proof"]

        outcome = self.reasoner.query_polarity(
            "(: $prf (Online camera_2) $tv)"
        )

        self.assertEqual("both", outcome.status)

    def test_synthetic_zero_truth_negation_is_not_a_real_contradiction(self):
        with tempfile.TemporaryDirectory() as directory:
            atomspace = Path(directory) / "atomspace.metta"
            atomspace.write_text(
                "(: active_fact (HasActiveEmploymentStatus omar) (STV 1.0 1.0))\n",
                encoding="utf-8",
            )
            self.reasoner._atomspace_path = str(atomspace)
            self.reasoner._background_files = set()
            self.reasoner.query = lambda query, seed_terms=None: (
                ["(: false_neg (Not (HasActiveEmploymentStatus omar)) (STV 0.0 1.0))"]
                if "(Not " in query
                else ["positive-proof"]
            )

            outcome = self.reasoner.query_polarity(
                "(: $prf (HasActiveEmploymentStatus omar) $tv)"
            )

        self.assertEqual("positive", outcome.status)
        self.assertEqual([], outcome.negative_proof)

    def test_no_positive_or_negative_proof_is_unknown(self):
        self.reasoner.query = lambda query, seed_terms=None: []

        outcome = self.reasoner.query_polarity(
            "(: $prf (Polluted lake_2) $tv)"
        )

        self.assertEqual("unknown", outcome.status)

    def test_rule_conclusion_requires_exact_arity_match(self):
        bindings = self.reasoner._match_conclusion(
            "(: $prf (QualifiesForMeritScholarship lena merit) $tv)",
            ["QualifiesForMeritScholarship", "$student"],
        )

        self.assertIsNone(bindings)

    def test_factor_explanation_exposes_grounded_rule_requirements(self):
        self.reasoner.get_atoms = lambda: [
            "(: risk_rule (Implication (Premises (Obese $x) "
            "(Or (Sedentary $x) (FamilyHistory $x))) "
            "(Conclusions (IncreasesDiabetesRisk $x))) (STV 1.0 1.0))"
        ]
        self.reasoner.query_polarity = lambda query, seed_terms=None: SimpleNamespace(
            status="unknown"
        )
        self.reasoner.query = lambda query, seed_terms=None: []

        explanations = self.reasoner.explain_requirements(
            {"increase", "diabetes", "risk"},
            entity="abebe",
            direction="increase",
        )

        self.assertEqual(1, len(explanations))
        rendered = str(explanations[0]["requirements"])
        self.assertIn("(Obese abebe)", rendered)
        self.assertIn("(Sedentary abebe)", rendered)
        self.assertIn("(FamilyHistory abebe)", rendered)

    def test_or_premise_succeeds_when_one_alternative_is_proved(self):
        self.reasoner._prove_grounded_premise = lambda premise, seen: (
            ["b-proof"] if premise == ["B", "x"] else []
        )

        states = self.reasoner._prove_premise(
            ["Or", ["A", "x"], ["B", "x"]],
            {},
            set(),
        )

        self.assertEqual([({}, ["b-proof"])], states)

    def test_negative_rule_premise_requires_explicit_negative_fact(self):
        self.reasoner._query_exact_fact = lambda query: []
        self.reasoner._backward_chain = lambda query, seen=None: []
        self.reasoner._handler = SimpleNamespace(
            query=lambda *args, **kwargs: ["synthetic-negation-proof"]
        )
        self.reasoner._query_timeout = 1
        self.reasoner._max_steps = 1

        proof = self.reasoner._prove_grounded_premise(
            ["Not", ["ProvidesWrittenNotice", "bright_ware"]],
            set(),
        )

        self.assertEqual([], proof)

    def test_explicit_negative_fact_can_satisfy_negative_rule_premise(self):
        explicit = (
            "(: no_notice (Not (ProvidesWrittenNotice bright_ware)) "
            "(STV 1.0 1.0))"
        )
        self.reasoner._query_exact_fact = lambda query: (
            [explicit] if "(Not (ProvidesWrittenNotice bright_ware))" in query else []
        )

        proof = self.reasoner._prove_grounded_premise(
            ["Not", ["ProvidesWrittenNotice", "bright_ware"]],
            set(),
        )

        self.assertEqual([explicit], proof)

    def test_proof_sources_use_exact_atom_provenance(self):
        atom = "(: online_fact (Online camera_2) (STV 1.0 1.0))"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "atom": atom,
                        "source": {"text": "Camera-2 is online."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.reasoner._provenance_path = str(path)

            sources = self.reasoner.sources_for_proof([atom])

        self.assertEqual(["Camera-2 is online."], sources)


if __name__ == "__main__":
    unittest.main()
