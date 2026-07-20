from __future__ import annotations

import json
import unittest
from pathlib import Path


EXAMPLES_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "langextract_examples.json"
)


class CompactLangExtractExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

    def test_example_budget_stays_compact(self):
        self.assertLessEqual(len(self.payload["statement_examples"]), 12)
        self.assertLessEqual(len(self.payload["query_examples"]), 10)
        self.assertLess(EXAMPLES_PATH.stat().st_size, 20_000)

    def test_statement_examples_keep_required_extraction_classes(self):
        classes = {
            extraction["class"]
            for example in self.payload["statement_examples"]
            for extraction in example.get("extractions", [])
        }
        self.assertTrue(
            {"type_decl", "rule", "fact", "inheritance", "property", "negation"}
            <= classes
        )

    def test_rule_examples_keep_logical_structure_coverage(self):
        bodies = [
            extraction.get("attributes", {}).get("body", "").lower()
            for example in self.payload["statement_examples"]
            for extraction in example.get("extractions", [])
            if extraction.get("class") == "rule"
        ]
        self.assertTrue(any("(and " in body for body in bodies))
        self.assertTrue(any("(not " in body for body in bodies))
        self.assertTrue(any("(or " in body for body in bodies))
        self.assertTrue(any(body.count("$") >= 3 for body in bodies))

    def test_queries_keep_grounded_and_open_argument_shapes(self):
        queries = [
            extraction.get("attributes", {})
            for example in self.payload["query_examples"]
            for extraction in example.get("extractions", [])
        ]
        arguments = [query.get("arguments", []) for query in queries]
        self.assertTrue(
            any(not any(str(arg).startswith("$") for arg in args) for args in arguments)
        )
        self.assertTrue(any(args and str(args[0]).startswith("$") for args in arguments))
        self.assertTrue(
            any(
                any(str(arg).startswith("$") for arg in args[1:])
                for args in arguments
            )
        )
        self.assertTrue(any(len(args) >= 3 for args in arguments))


if __name__ == "__main__":
    unittest.main()
