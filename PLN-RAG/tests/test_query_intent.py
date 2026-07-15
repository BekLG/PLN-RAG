from __future__ import annotations

import unittest

from core.query.intent import (
    QuestionMode,
    parse_question_intent,
    query_intent_score,
    query_matches_intent,
)


class QueryIntentTests(unittest.TestCase):
    def test_sufficiency_is_not_reduced_to_supporting_fact(self):
        intent = parse_question_intent(
            "Is Abebe's pasta consumption alone sufficient to cause diabetes?"
        )

        self.assertEqual(QuestionMode.SUFFICIENCY, intent.mode)
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (EatsFrequently abebe pasta) $tv)",
            )
        )

    def test_factor_question_has_non_atomic_answer_mode(self):
        intent = parse_question_intent("What factors increase Abebe's diabetes risk?")

        self.assertEqual(QuestionMode.FACTORS, intent.mode)
        self.assertEqual("increase", intent.direction)
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (AtElevatedDiabetesRisk abebe) $tv)",
            )
        )

    def test_should_question_is_boolean(self):
        intent = parse_question_intent(
            "Should Nia be flagged for urgent respiratory review?"
        )

        self.assertEqual(QuestionMode.BOOLEAN, intent.mode)
        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (FlaggedForUrgentRespiratoryReview nia) $tv)",
            )
        )

    def test_cross_domain_boolean_targets_require_relation_match(self):
        cases = [
            (
                "Is Motor-1 overheating?",
                "(: $prf (Overheating motor_1) $tv)",
                "(: $prf (Running motor_1) $tv)",
            ),
            (
                "Is Alice authorized?",
                "(: $prf (Authorized alice) $tv)",
                "(: $prf (Employee alice) $tv)",
            ),
            (
                "Is Lake-2 polluted?",
                "(: $prf (Polluted lake_2) $tv)",
                "(: $prf (NearFactory lake_2) $tv)",
            ),
        ]
        for question, expected, unrelated in cases:
            with self.subTest(question=question):
                intent = parse_question_intent(question)
                self.assertTrue(query_matches_intent(intent, expected))
                self.assertFalse(query_matches_intent(intent, unrelated))

    def test_entity_renaming_does_not_change_intent_shape(self):
        first = parse_question_intent("Is Abebe consuming higher carbohydrates?")
        second = parse_question_intent("Is Selam consuming higher carbohydrates?")

        self.assertEqual(first.mode, second.mode)
        self.assertEqual(first.terms - set(first.entities), second.terms - set(second.entities))

    def test_property_value_can_carry_question_semantics(self):
        intent = parse_question_intent("Is Tom white?")

        self.assertTrue(
            query_matches_intent(intent, "(: $prf (Has tom color white) $tv)")
        )

    def test_status_predicate_does_not_answer_plain_state_question(self):
        intent = parse_question_intent("Is Abebe currently obese?")

        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (ClinicallyClassifiedAsObese abebe) $tv)",
            )
        )
        self.assertTrue(
            query_matches_intent(intent, "(: $prf (Obese abebe) $tv)")
        )

    def test_more_specific_risk_candidate_scores_higher(self):
        intent = parse_question_intent("Is Abebe at elevated risk of diabetes?")

        self.assertGreater(
            query_intent_score(
                intent,
                "(: $prf (AtElevatedDiabetesRisk abebe) $tv)",
            ),
            query_intent_score(
                intent,
                "(: $prf (PlaysARoleInDiabetesRisk abebe) $tv)",
            ),
        )

    def test_denial_predicate_cannot_answer_positive_access_question(self):
        intent = parse_question_intent("May Omar access the secure archive?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (AccessesSecureArchive omar) $tv)",
            )
        )
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (IsDeniedArchiveAccess omar) $tv)",
            )
        )

    def test_entity_identifier_can_match_compound_kb_entity(self):
        intent = parse_question_intent("Must batch Q4 be rejected?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (Rejected batch_q4) $tv)",
            )
        )

    def test_generic_type_fact_cannot_answer_event_question(self):
        intent = parse_question_intent("Was the malware alert triggered on Atlas?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (TriggeredMalwareAlert atlas) $tv)",
            )
        )
        self.assertFalse(
            query_matches_intent(intent, "(: $prf (IsA atlas server) $tv)")
        )

    def test_subtype_words_are_required_for_query_target(self):
        intent = parse_question_intent(
            "Does Lena qualify for a leadership scholarship?"
        )

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (QualifiesForLeadershipScholarship lena) $tv)",
            )
        )
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (QualifiesForMeritScholarship lena) $tv)",
            )
        )

    def test_action_word_is_required_when_related_fact_mentions_same_topic(self):
        intent = parse_question_intent("Did the customer waive the delivery deadline?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (WaivedDeliveryDeadline customer) $tv)",
            )
        )
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (MissesDeliveryDeadline customer) $tv)",
            )
        )

    def test_property_question_requires_property_terms(self):
        intent = parse_question_intent("Is batch Q4 missing a packaging date?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (MissingPackagingDate batch_q4) $tv)",
            )
        )
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (ContaminationDetected batch_q4) $tv)",
            )
        )

    def test_hyphenated_question_terms_match_compound_predicate(self):
        intent = parse_question_intent("Has Hana completed online check-in?")

        self.assertTrue(
            query_matches_intent(
                intent,
                "(: $prf (CompletedOnlineCheckIn hana) $tv)",
            )
        )


if __name__ == "__main__":
    unittest.main()
