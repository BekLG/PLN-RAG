from __future__ import annotations

import unittest

from parsers.langextract_pln_parser import _remap_metadata


class ProvenanceLineageTests(unittest.TestCase):
    def test_reordered_outputs_are_mapped_by_structure_not_position(self):
        first = "(: first (Likes lena tea) (STV 1.0 1.0))"
        second = "(: second (Visits omar addis) (STV 1.0 1.0))"
        metadata = {
            first: {"text": "Lena likes tea."},
            second: {"text": "Omar visits Addis."},
        }

        remapped = _remap_metadata(
            [first, second],
            [second, first],
            metadata,
        )

        self.assertEqual("Omar visits Addis.", remapped[second]["text"])
        self.assertEqual("Lena likes tea.", remapped[first]["text"])

    def test_unrelated_generated_atom_does_not_borrow_source(self):
        source = "(: source (HasFever nia) (STV 1.0 1.0))"
        generated = "(: inferred (IsA nia person) (STV 1.0 1.0))"

        remapped = _remap_metadata(
            [source],
            [generated],
            {source: {"text": "Nia has a fever."}},
        )

        self.assertNotIn(generated, remapped)


if __name__ == "__main__":
    unittest.main()
