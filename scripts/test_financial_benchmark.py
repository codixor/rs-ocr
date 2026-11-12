"""Behavioral controls for the public financial-document measurement oracle."""
import copy
import unittest

import financial_benchmark as b


class FinancialBenchmarkTests(unittest.TestCase):
    def test_amount_labels_are_exact_not_ocr_repairs(self):
        self.assertEqual(b.amount_label("RM31.60"), 3160)
        self.assertEqual(b.amount_label("24,37"), 2437)
        self.assertEqual(b.amount_label("$1,234.56"), 123456)
        for bad in ("1O.00", "1.234", "NaN", "Infinity", ""):
            with self.assertRaises(ValueError):
                b.amount_label(bad)

    def test_dates_use_declared_order_and_no_fuzzy_spelling(self):
        self.assertEqual(b.date_key("03/04/18", "dmy"), "2018-04-03")
        self.assertEqual(b.date_key("03/04/18", "mdy"), "2018-03-04")
        self.assertEqual(b.date_key("8 May 99", "dmy"), "1999-05-08")
        self.assertEqual(b.date_key("March 23, 1966", "mdy"), "1966-03-23")
        self.assertNotEqual(b.text_key("J0SE"), b.text_key("JOSE"))
        self.assertEqual(b.text_key("Store  Name"), b.text_key("STORENAME"))

    def test_missing_wrong_ambiguous_and_negative_controls_count_as_failures(self):
        sample = {"id": "a", "date_order": "dmy", "decimal_separator": "dot", "expected": {"total": "10.00", "merchant": "Acme", "date": "01/02/2026", "tax": None}}
        row = {"id": "a", "spans": [{"text": "ACME"}], "fields": {"total": {"value": 1100}, "merchant": {"value": None, "candidates": ["A", "B"]}, "tax": {"value": 10}}}
        result = b.score([sample], {"a": row})
        self.assertEqual(result["aggregate"]["correct"], 0)
        self.assertEqual(result["fields"]["merchant"]["ambiguous"], 1)
        self.assertEqual(result["fields"]["tax"]["false_positive"], 1)
        self.assertEqual(result["fields"]["merchant"]["target_in_ocr"], 1)
        row["fields"] = {"total": {"value": 1000}, "merchant": {"value": "ACME"}, "date": {"value": "1/2/2026"}}
        result = b.score([sample], {"a": row})
        self.assertEqual(result["aggregate"]["accuracy"], 1.0)
        self.assertEqual(result["exact_documents"], 1)

    def test_money_presence_is_literal_and_does_not_repair_ocr(self):
        sample = {"decimal_separator": "dot"}
        self.assertTrue(b.target_in_ocr("amount", "500.00", sample, {"spans": [{"text": "PAYEE500.00"}]}))
        self.assertFalse(b.target_in_ocr("amount", "500.00", sample, {"spans": [{"text": "5OO.00"}]}))
        self.assertFalse(b.target_in_ocr("amount", "500.00", sample, {"spans": [{"text": "1500.00"}]}))

    def test_actual_manifest_has_no_image_or_issuer_leakage(self):
        manifest = b.read(b.MANIFEST)
        b.validate_manifest(manifest)
        duplicate = copy.deepcopy(manifest)
        duplicate["samples"].append(duplicate["samples"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            b.validate_manifest(duplicate)
        overlap = copy.deepcopy(manifest)
        dev = next(s for s in overlap["samples"] if s["domain"] == "receipt" and s["split"] == "dev")
        test = next(s for s in overlap["samples"] if s["domain"] == "receipt" and s["split"] == "test")
        test["group"] = dev["group"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            b.validate_manifest(overlap)

    def test_artifact_path_escape_is_rejected(self):
        manifest = copy.deepcopy(b.read(b.MANIFEST))
        manifest["samples"][0]["artifacts"][0]["path"] = "../escaped.json"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            b.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
