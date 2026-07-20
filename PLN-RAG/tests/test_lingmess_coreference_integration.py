from __future__ import annotations

import os
import unittest


@unittest.skipUnless(
    os.getenv("RUN_COREFERENCE_INTEGRATION_TESTS", "").lower() == "true",
    "set RUN_COREFERENCE_INTEGRATION_TESTS=true to run LingMess integration",
)
class LingMessCoreferenceIntegrationTests(unittest.TestCase):
    def test_lingmess_resolves_simple_document_when_available(self):
        try:
            from core.discourse.lingmess_coreference import (
                LingMessCoreferenceResolver,
            )
        except Exception as exc:
            self.skipTest(f"LingMess backend unavailable: {exc}")

        resolver = LingMessCoreferenceResolver(device="cpu", max_tokens_in_batch=500)
        try:
            result = resolver.resolve_document(
                "Alex explained the project. He submitted the report."
            )
        except Exception as exc:
            self.skipTest(f"LingMess model unavailable: {exc}")

        self.assertEqual("lingmess", result.backend)
        self.assertTrue(result.clusters)


if __name__ == "__main__":
    unittest.main()
