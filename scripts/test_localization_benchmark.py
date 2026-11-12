"""Controls for freshness, spatial decisions, fallback and honest accuracy claims."""

from copy import deepcopy
import json
import unittest

import localization_benchmark as study


def record(target_class=3):
    value = {"id": "sample", "width": 400, "height": 400, "target_class": target_class,
             "objects": [{"class_id": 3, "confidence": study.f32(0.8), "bounds": [95., 95., 205., 205.]}],
             "tiles": [{"index": i, "target_score": study.f32(0.1), "margin": -study.f32(0.5), "selected": False} for i in range(16)],
             "spatial_options": {"min_confidence": 0.25, "min_coverage": study.f32(0.01)}}
    value["spatial_selected"] = study.spatial_selected(value, value["spatial_options"])
    return value


class SpatialEvaluation(unittest.TestCase):
    def test_boxes_select_fragments_and_respect_strict_positive_overlap(self):
        value = record()
        self.assertEqual(value["spatial_selected"], [1, 4, 5, 6, 9])
        value["objects"][0]["bounds"] = [100., 100., 200., 200.]
        self.assertEqual(study.spatial_selected(value, {"min_confidence": 0.25, "min_coverage": 0}), [5])
        self.assertEqual(study.spatial_selected(value, {"min_confidence": 0.9, "min_coverage": 0}), [])

    def test_valid_records_pass_and_bad_outputs_cannot_be_dropped(self):
        sample = {"id": "sample", "rows": 4, "columns": 4, "target": "motorcycles", "expected": [5]}
        self.assertIn("sample", study.validate_records([sample], [record()]))
        for mutation in ["score", "bounds", "class", "decision", "dimensions"]:
            value = record()
            if mutation == "score": value["objects"][0]["confidence"] = float("nan")
            elif mutation == "bounds": value["objects"][0]["bounds"][0] = -1
            elif mutation == "class": value["target_class"] = 1
            elif mutation == "decision": value["spatial_selected"] = []
            else: value["width"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.validate_records([sample], [value])
        with self.assertRaises(ValueError):
            study.validate_records([sample], [])

    def test_unsupported_class_falls_back_to_existing_clip_decisions(self):
        sample = {"id": "sample", "rows": 4, "columns": 4, "target": "crosswalks", "expected": [0]}
        value = record(None)
        value["tiles"][0].update(target_score=study.f32(0.9), margin=study.f32(0.8), selected=True)
        values = study.validate_records([sample], [value])
        baseline, hybrid = study.selections([sample], values, {"min_confidence": 0.9, "min_coverage": 0.5})
        self.assertEqual(baseline, {"sample": [0]})
        self.assertEqual(hybrid, baseline)

    def test_development_fitting_uses_the_declared_objective(self):
        sample = {"id": "sample", "rows": 4, "columns": 4, "target": "motorcycles", "expected": [0]}
        value = record()
        value["objects"] = [
            {"class_id": 3, "confidence": study.f32(0.2), "bounds": [0., 0., 100., 100.]},
            {"class_id": 3, "confidence": study.f32(0.1), "bounds": [100., 0., 200., 100.]},
        ]
        chosen = study.fit([sample], {"sample": value})
        self.assertEqual(chosen, {"min_confidence": 0.2, "min_coverage": 0.2})
        self.assertEqual(study.spatial_selected(value, chosen), [0])

    def test_unseen_categories_keep_grids_and_use_untuned_api_defaults(self):
        sample = {"id": "sample", "rows": 4, "columns": 4, "target": "taxis", "expected": [0]}
        value = record(None)
        value["tiles"][0].update(target_score=study.f32(0.4), margin=study.f32(0.2), selected=False)
        values = study.validate_records([sample], [value])
        baseline, hybrid = study.selections([sample], values, {"min_confidence": 0.1, "min_coverage": 0.1})
        self.assertEqual(baseline, {"sample": []})
        self.assertEqual(hybrid, baseline)

    def test_improvement_oracle_requires_both_f1_and_exact_grid_gain(self):
        baseline = {"f1": 0.5, "exact_grids": 1}
        for candidate in [baseline, {"f1": 0.6, "exact_grids": 1}, {"f1": 0.4, "exact_grids": 2}]:
            with self.assertRaises(ValueError):
                study.verify_improvement({"held_out": {"all": {"clip_baseline": baseline, "spatial_hybrid": candidate}}})
        study.verify_improvement({"held_out": {"all": {"clip_baseline": baseline, "spatial_hybrid": {"f1": 0.6, "exact_grids": 2}}}})


class Freshness(unittest.TestCase):
    def test_fresh_manifest_passes_and_known_id_and_image_leaks_fail(self):
        previous = json.loads(study.PREVIOUS.read_text())
        manifest = json.loads(study.MANIFEST.read_text())
        study.check_freshness(manifest, previous)
        for mutation in ["id", "old_blob", "new_blob"]:
            changed = deepcopy(manifest)
            if mutation == "id": changed["samples"][0]["id"] = previous["samples"][0]["id"]
            elif mutation == "old_blob": changed["samples"][0]["tile_blobs"][0] = previous["samples"][0]["tile_blobs"][0]
            else: changed["samples"][0]["tile_blobs"][0] = changed["samples"][-1]["tile_blobs"][0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.check_freshness(changed, previous)


if __name__ == "__main__":
    unittest.main()
