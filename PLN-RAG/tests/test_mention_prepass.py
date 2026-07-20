from __future__ import annotations

import unittest

from core.discourse import MentionPrepass


class MentionPrepassTests(unittest.TestCase):
    def test_ambiguous_pronoun_keeps_multiple_candidates(self):
        result = MentionPrepass().build(
            "Alex put the camera near the phone. It was broken."
        )

        nominals = {mention.canonical: mention for mention in result.mentions}
        self.assertIn("camera", nominals)
        self.assertIn("phone", nominals)
        self.assertEqual(1, len(result.ambiguous_pronouns))

        pronoun = result.ambiguous_pronouns[0]
        candidate_labels = {
            mention.canonical
            for mention in result.mentions
            if mention.id in pronoun.candidates
        }
        self.assertEqual({"camera", "phone"}, candidate_labels)

    def test_prompt_hint_tells_langextract_to_preserve_ambiguity(self):
        result = MentionPrepass().build(
            "Alex put the camera near the phone. It was broken."
        )
        hint = result.prompt_hint()

        self.assertIn("Mention prepass hints", hint)
        self.assertIn("do not replace it with one candidate", hint)
        self.assertIn("it_m", hint)
        self.assertIn("camera", hint)
        self.assertIn("phone", hint)

    def test_non_ambiguous_text_still_records_mentions(self):
        result = MentionPrepass().build("Alex bought the camera.")

        self.assertEqual([], result.ambiguous_pronouns)
        self.assertTrue(
            any(mention.canonical == "camera" for mention in result.mentions)
        )

    def test_simple_human_pronoun_does_not_inject_a_mention_id(self):
        result = MentionPrepass().build(
            "Hana has a valid passport. She completed online check-in."
        )

        self.assertEqual([], result.ambiguous_pronouns)
        self.assertEqual("", result.prompt_hint())


if __name__ == "__main__":
    unittest.main()
