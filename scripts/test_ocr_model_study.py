"""Controls for development-only OCR model selection."""
import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import ocr_model_study as s


class OcrModelStudyTests(unittest.TestCase):
    def test_literal_targets_ignore_segmentation_without_repairing_ocr(self):
        sample = {"decimal_separator": "dot", "date_order": "dmy"}
        self.assertTrue(s.available("merchant", "EXAMPLE STORE", sample, {"spans": [{"text": "EXAMPLE"}, {"text": "STORE"}]}))
        self.assertFalse(s.available("merchant", "EXAMPLE STORE", sample, {"spans": [{"text": "EXAMPLE ST0RE"}]}))
        self.assertTrue(s.available("date", "03/04/2018", sample, {"spans": [{"text": "03-04-2018 12:00"}]}))
        self.assertFalse(s.available("date", "03/04/2018", sample, {"spans": [{"text": "03-04-2019"}]}))

    def test_ocr_recovery_and_extracted_field_accuracy_are_distinct(self):
        sample = {"id": "a", "decimal_separator": "dot", "date_order": "dmy", "expected": {"total": "10.00"}}
        rows = {"a": {"spans": [{"text": "10.00"}], "fields": {"total": {"value": None}}, "elapsed_ms": 3.0}}
        m = s.metrics([sample], rows)
        self.assertEqual(m["target_availability"]["present"], 1)
        self.assertEqual(m["extraction"]["aggregate"]["correct"], 0)
        self.assertEqual(m["observed_document_ms"]["median"], 3)

    def test_selection_prioritizes_recovery_then_extraction_then_memory(self):
        candidates = {name: {"files": [{"size": size}]} for name, size in [("tiny", 1), ("small", 2), ("medium", 3)]}
        results = {name: {"target_availability": {"present": 5}, "extraction": {"aggregate": {"correct": 3}}} for name in candidates}
        self.assertEqual(s.choose(candidates, results), "tiny")
        results["small"]["extraction"]["aggregate"]["correct"] = 4
        self.assertEqual(s.choose(candidates, results), "small")
        results["medium"]["target_availability"]["present"] = 6
        self.assertEqual(s.choose(candidates, results), "medium")

    def test_new_samples_are_disjoint_and_data_validation_detects_leakage(self):
        manifest = s.f.read(s.SAMPLES)
        s.f.validate_manifest(manifest)
        broken = copy.deepcopy(manifest)
        dev = next(v for v in broken["samples"] if v["split"] == "dev" and v["domain"] == "invoice")
        test = next(v for v in broken["samples"] if v["split"] == "test" and v["domain"] == "invoice")
        test["group"] = dev["group"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            s.f.validate_manifest(broken)

    def test_selection_cannot_be_refrozen_after_test_predictions_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "test-tiny.jsonl").write_text("{}\n")
            with patch.object(s, "DATA", data):
                with self.assertRaisesRegex(ValueError, "held-out"):
                    s.freeze()


if __name__ == "__main__":
    unittest.main()
