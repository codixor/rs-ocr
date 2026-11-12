"""Behavioral controls for the offline evaluation's measurements and failures."""

import hashlib
import unittest

from benchmark_selection import (
    BASELINE, best_thresholds, canonical_target, metrics, paired_f1_interval, selected,
    validate_predictions, verify_gain,
)
from public_samples import validate_info, verify_bytes


def tile(index, score, margin):
    return {"index": index, "target_score": score, "margin": margin,
            "selected": score >= 0.5 and margin >= 0.1 and margin > 0}


class Measurements(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"id": "first", "rows": 2, "columns": 2, "expected": [0, 2]},
            {"id": "second", "rows": 2, "columns": 2, "expected": []},
        ]
        self.records = [
            {"id": "first", "tiles": [tile(0, 0.8, 0.4), tile(1, 0.7, 0.3), tile(2, 0.3, 0.2), tile(3, 0.1, -0.2)]},
            {"id": "second", "tiles": [tile(i, 0.1, -0.2) for i in range(4)]},
        ]
        self.predictions = validate_predictions(self.samples, self.records)

    def test_confusion_counts_and_exact_grids_are_independently_known(self):
        value = metrics(self.samples, self.predictions, BASELINE)
        self.assertEqual([value[key] for key in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 5])
        self.assertEqual(value["precision"], 0.5)
        self.assertEqual(value["recall"], 0.5)
        self.assertEqual(value["f1"], 0.5)
        self.assertEqual(value["tile_accuracy"], 0.75)
        self.assertEqual(value["exact_grids"], 1)
        self.assertEqual(value["exact_grid_rate"], 0.5)
        low, high = value["exact_grid_wilson_95"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_no_predicted_positives_do_not_report_perfect_precision(self):
        value = metrics(self.samples, self.predictions, {"min_score": 1, "min_margin": 1})
        self.assertIsNone(value["precision"])
        self.assertEqual(value["recall"], 0)
        self.assertEqual(value["f1"], 0)
        value = metrics(self.samples[1:], self.predictions, BASELINE)
        self.assertIsNone(value["precision"])
        self.assertIsNone(value["recall"])
        self.assertIsNone(value["f1"])
        self.assertEqual(value["tile_accuracy"], 1)

    def test_empty_benchmark_cannot_succeed(self):
        with self.assertRaises(ValueError):
            metrics([], {}, BASELINE)

    def test_ties_and_losing_targets_fail_even_with_zero_thresholds(self):
        self.assertEqual(selected([tile(0, 0.8, 0), tile(1, 0.8, -0.1)], {"min_score": 0, "min_margin": 0}), [])

    def test_decimal_thresholds_use_rust_f32_boundary_semantics(self):
        self.assertEqual(selected([tile(0, 0.699999988079071, 0.5)], {"min_score": 0.7, "min_margin": 0}), [0])
        self.assertEqual(selected([tile(0, 0.6999999284744263, 0.5)], {"min_score": 0.7, "min_margin": 0}), [])

    def test_calibration_finds_lower_threshold_without_changing_strict_winner(self):
        samples = [{"id": "dev", "rows": 1, "columns": 3, "expected": [0]}]
        values = {"dev": {"tiles": [tile(0, 0.3, 0.15), tile(1, 0.2, 0.1), tile(2, 0.8, -0.1)]}}
        thresholds = best_thresholds(samples, values)
        self.assertEqual(metrics(samples, values, BASELINE)["f1"], 0)
        self.assertEqual(metrics(samples, values, thresholds)["f1"], 1)
        self.assertEqual(selected(values["dev"]["tiles"], thresholds), [0])

    def test_missing_duplicate_or_unexpected_records_are_rejected(self):
        for records in [self.records[:1], self.records + self.records[:1], [{"id": "other", "tiles": []}]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_predictions(self.samples, records)

    def test_missing_or_reordered_tiles_are_rejected(self):
        for tiles in [self.records[0]["tiles"][:3], list(reversed(self.records[0]["tiles"]))]:
            with self.assertRaises(ValueError):
                validate_predictions(self.samples, [{"id": "first", "tiles": tiles}, self.records[1]])

    def test_non_finite_scores_and_rust_decision_mismatches_are_rejected(self):
        for key, value in [
            ("index", False), ("target_score", True), ("target_score", float("nan")),
            ("margin", float("inf")), ("target_score", 1.1), ("selected", False),
        ]:
            records = [{"id": "first", "tiles": [dict(item) for item in self.records[0]["tiles"]]}, self.records[1]]
            records[0]["tiles"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_predictions(self.samples, records)

    def test_paired_resampling_preserves_a_known_unchanged_result(self):
        interval = paired_f1_interval(self.samples[:1], self.predictions, BASELINE, repetitions=100)
        self.assertEqual(interval["f1_delta_95"], [0, 0])
        self.assertEqual(interval["valid_repetitions"], 100)
        with self.assertRaises(ValueError):
            paired_f1_interval(self.samples[1:], self.predictions, BASELINE, repetitions=10)

    def test_improvement_gate_rejects_unchanged_or_worse_results(self):
        baseline = {"f1": 0.5, "tile_accuracy": 0.75}
        for calibrated in [baseline, {"f1": 0.4, "tile_accuracy": 0.8}, {"f1": 0.6, "tile_accuracy": 0.7}]:
            with self.assertRaises(ValueError):
                verify_gain({"held_out": {"all": {"baseline": baseline, "calibrated": calibrated}}})
        verify_gain({"held_out": {"all": {"baseline": baseline, "calibrated": {"f1": 0.6, "tile_accuracy": 0.8}}}})


class SourceValidation(unittest.TestCase):
    def test_invalid_annotations_do_not_become_negative_examples(self):
        for answers in [[True], [-1], [9], [0, 0], "0"]:
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                validate_info({"instruction": "select", "keyword": "cars", "correct_answers": answers}, 9)
        validate_info({"instruction": "select", "keyword": "cars", "correct_answers": []}, 9)

    def test_source_integrity_positive_and_corrupted_controls(self):
        data = b"known bytes"
        sha = hashlib.sha1(b"blob 11\0" + data).hexdigest()
        entry = {"path": "image.png", "size": len(data), "git_blob": sha}
        verify_bytes(data, entry)
        for corrupted in [b"known bytES", data + b"x"]:
            with self.assertRaises(ValueError):
                verify_bytes(corrupted, entry)
        with self.assertRaises(ValueError):
            verify_bytes(data, dict(entry, sha256="0" * 64))

    def test_only_documented_keyword_aliases_are_normalized(self):
        self.assertEqual(canonical_target("bus"), "buses")
        self.assertEqual(canonical_target("a fire hydrant"), "fire hydrants")
        with self.assertRaises(ValueError):
            canonical_target("unmodeled category")


if __name__ == "__main__":
    unittest.main()
