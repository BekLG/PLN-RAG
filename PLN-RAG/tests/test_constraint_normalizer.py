from __future__ import annotations

import unittest

from core.pln.constraint_normalizer import PLNConstraintNormalizer
from core.pln.postprocessor import PLNPostprocessor


class ConstraintNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = PLNConstraintNormalizer(PLNPostprocessor.STRUCTURAL_HEADS)

    def test_numeric_thresholds_are_materialized_when_values_satisfy_rule(self):
        statements = [
            "(: merit_rule (Implication (Premises (IsA $student student) "
            "(HasGpaAtLeast $student 38) "
            "(HasCompletedVolunteerHoursAtLeast $student 40)) "
            "(Conclusions (QualifiesForMeritScholarship $student))) (STV 1.0 1.0))",
            "(: gpa_fact (HasGpa lena 39) (STV 1.0 1.0))",
            "(: hours_fact (CompletedVolunteerHours lena 45) (STV 1.0 1.0))",
        ]

        normalized, decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(HasGpaAtLeast lena 38)", rendered)
        self.assertIn("(HasCompletedVolunteerHoursAtLeast lena 40)", rendered)
        self.assertIn("(IsA lena student)", rendered)
        self.assertTrue(decisions)

    def test_numeric_thresholds_respect_units_and_direction(self):
        statements = [
            "(: overheating_rule (Implication (Premises "
            "(InternalTemperatureStaysAbove $x 90 celsius) "
            "(DurationMoreThan $x 10 minute)) "
            "(Conclusions (Overheating $x))) (STV 1.0 1.0))",
            "(: temp_fact (InternalTemperature m7 94celsius) (STV 1.0 1.0))",
            "(: duration_fact (DurationAtTemperature m7 12minute) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(InternalTemperatureStaysAbove m7 90 celsius)", rendered)
        self.assertIn("(DurationMoreThan m7 10 minute)", rendered)

    def test_below_threshold_and_role_type_are_materialized(self):
        statements = [
            "(: urgent_rule (Implication (Premises (IsA $p patient) "
            "(HasShortnessOfBreath $p) "
            "(HasOxygenSaturationBelow $p 92percent)) "
            "(Conclusions (FlaggedForUrgentRespiratoryReview $p))) (STV 1.0 1.0))",
            "(: breath_fact (HasShortnessOfBreath nia) (STV 1.0 1.0))",
            "(: oxygen_fact (HasOxygenSaturationAt nia 90percent) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(HasOxygenSaturationBelow nia 92percent)", rendered)
        self.assertIn("(IsA nia patient)", rendered)

    def test_frequency_wording_can_satisfy_frequent_rule_premise(self):
        statements = [
            "(: carbs_rule (Implication (Premises "
            "(EatsFrequently $x refined_pasta) "
            "(EatsLargePortions $x refined_pasta)) "
            "(Conclusions (ConsumesHigherCarbohydrates $x))) (STV 1.0 1.0))",
            "(: eating_fact (EatsFiveTimesAWeek mira refined_pasta) (STV 1.0 1.0))",
            "(: portion_fact (EatsLargePortions mira refined_pasta) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)

        self.assertTrue(
            any("(EatsFrequently mira refined_pasta)" in item for item in normalized)
        )

    def test_unsatisfied_threshold_is_not_materialized(self):
        statements = [
            "(: urgent_rule (Implication (Premises "
            "(HasOxygenSaturationBelow $p 92percent)) "
            "(Conclusions (FlaggedForUrgentRespiratoryReview $p))) (STV 1.0 1.0))",
            "(: oxygen_fact (HasOxygenSaturationAt nia 96percent) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)

        self.assertFalse(
            any("(HasOxygenSaturationBelow nia 92percent)" in item for item in normalized)
        )

    def test_generic_has_properties_satisfy_document_local_thresholds(self):
        statements = [
            "(: merit_rule (Implication (Premises (IsA $student student) "
            "(HasGpaAtLeast38 $student) "
            "(HasCompletedAtLeast40VolunteerHours $student)) "
            "(Conclusions (QualifiesForMeritScholarship $student))) (STV 1.0 1.0))",
            "(: gpa_fact (Has lena gpa 39) (STV 1.0 1.0))",
            "(: hours_fact (Has lena volunteer_hour 45) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(
            statements,
            source_text=(
                "A student qualifies with a GPA of at least 3.8 and at least "
                "40 volunteer hours. Lena has a GPA of 3.9 and 45 volunteer hours."
            ),
        )
        rendered = "\n".join(normalized)

        self.assertIn("(HasGpaAtLeast38 lena)", rendered)
        self.assertIn("(HasCompletedAtLeast40VolunteerHours lena)", rendered)
        self.assertIn("(IsA lena student)", rendered)

    def test_unattached_measurement_uses_unique_rule_entity(self):
        statements = [
            "(: fire_rule (Implication (Premises (IsA $r region) "
            "(VegetationIsDry $r) (WindSpeedExceeds $r 30kilometer_per_hour)) "
            "(Conclusions (AtHighWildfireRisk $r))) (STV 1.0 1.0))",
            "(: dry_fact (HasDryVegetation pinevale) (STV 1.0 1.0))",
            "(: wind_fact (Has wind_speed value 35kilometer_per_hour) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(VegetationIsDry pinevale)", rendered)
        self.assertIn("(WindSpeedExceeds pinevale 30kilometer_per_hour)", rendered)
        self.assertIn("(IsA pinevale region)", rendered)

    def test_duration_encoded_in_threshold_requires_duration_evidence(self):
        statements = [
            "(: hot_rule (Implication (Premises "
            "(InternalTemperatureStaysAbove $x 90 celsius 10 minute)) "
            "(Conclusions (Overheating $x))) (STV 1.0 1.0))",
            "(: temp_fact (Has m7 internal_temperature 94degree_celsius) (STV 1.0 1.0))",
            "(: duration_fact (Has m7 temperature_duration twelve_minute) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)

        self.assertTrue(
            any(
                "(InternalTemperatureStaysAbove m7 90 celsius 10 minute)" in item
                for item in normalized
            )
        )

    def test_generic_has_measurement_satisfies_variable_comparison_rule(self):
        statements = [
            "(: overheating_rule (Implication (Premises (IsA $m machine) "
            "(HasInternalTemperature $m $temp) "
            "(GreaterThan $temp 90celsius) "
            "(HasTemperatureDuration $m $duration) "
            "(GreaterThan $duration 10minute)) "
            "(Conclusions (IsOverheating $m))) (STV 1.0 1.0))",
            "(: temp_fact (Has machine_m7 internal_temperature 94celsius) "
            "(STV 1.0 1.0))",
            "(: duration_fact (Has machine_m7 temperature_duration 12minute) "
            "(STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(HasInternalTemperature machine_m7 94celsius)", rendered)
        self.assertIn("(GreaterThan 94celsius 90celsius)", rendered)
        self.assertIn("(HasTemperatureDuration machine_m7 12minute)", rendered)
        self.assertIn("(GreaterThan 12minute 10minute)", rendered)
        self.assertIn("(IsA machine_m7 machine)", rendered)

    def test_embedded_threshold_without_separator_is_split(self):
        statements = [
            "(: irrigation_rule (Implication (Premises (IsA $f field) "
            "(SoilMoistureIsBelow20Percent $f)) "
            "(Conclusions (NeedsIrrigation $f))) (STV 1.0 1.0))",
            "(: moisture_fact (Has field_delta soil_moisture 18percent) "
            "(STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)
        rendered = "\n".join(normalized)

        self.assertIn("(SoilMoistureIsBelow20Percent field_delta)", rendered)
        self.assertIn("(IsA field_delta field)", rendered)

    def test_source_text_can_support_rule_required_negative_premise(self):
        statements = [
            "(: irrigation_rule (Implication (Premises (IsA $f field) "
            "(SoilMoistureBelow $f 20percent) "
            "(Not (RainForecastWithinTwoDays $f))) "
            "(Conclusions (NeedsIrrigation $f))) (STV 1.0 1.0))",
            "(: moisture_fact (HasSoilMoisture field_delta 18percent) "
            "(STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(
            statements,
            source_text=(
                "Field Delta has soil moisture of 18 percent. "
                "No rain is forecast for the next two days."
            ),
        )
        rendered = "\n".join(normalized)

        self.assertIn("(SoilMoistureBelow field_delta 20percent)", rendered)
        self.assertIn("(Not (RainForecastWithinTwoDays field_delta))", rendered)
        self.assertIn("(IsA field_delta field)", rendered)

    def test_explicit_negative_is_rebound_only_to_negative_rule_premise(self):
        statements = [
            "(: delay_rule (Implication (Premises (IsA $x shipment) "
            "(MissesScheduledDeparture $x) "
            "(Not (HasConfirmedReplacementRoute $x))) "
            "(Conclusions (IsDelayed $x))) (STV 1.0 1.0))",
            "(: shipment_type (IsA s22 shipment) (STV 1.0 1.0))",
            "(: missed_fact (MissedScheduledDeparture s22) (STV 1.0 1.0))",
            "(: no_route (Not (HasBeenConfirmed replacement_route)) (STV 1.0 1.0))",
        ]

        normalized, _decisions = self.normalizer.normalize(statements)

        self.assertTrue(
            any(
                "(Not (HasConfirmedReplacementRoute s22))" in item
                for item in normalized
            )
        )


if __name__ == "__main__":
    unittest.main()
