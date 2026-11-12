#!/usr/bin/env python3
"""Offline Rust inference, development-only calibration and held-out evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile

from public_samples import CRATE, DEFAULT_DATA, MANIFEST, digest, validate_info, verify

ROOT = CRATE.parent
MODEL = CRATE / "models/clip-vit-b32"
PROFILE = CRATE / "model-manifests/public-selection-profile.json"
REPORT = CRATE / "docs/public-selection-results.json"
LABELS = [
    "bicycles", "buses", "cars", "crosswalks", "fire hydrants", "motorcycles",
    "stairs", "traffic lights", "bridges", "chimneys", "palm trees", "tractors",
    "buildings", "trees", "an empty road", "a sidewalk",
]
BASELINE = {"min_score": 0.5, "min_margin": 0.1}


def canonical_target(keyword):
    target = {"bus": "buses", "a fire hydrant": "fire hydrants"}.get(keyword, keyword)
    if target not in LABELS:
        raise ValueError(f"unsupported target: {keyword!r}")
    return target


def load_samples(manifest, directory, split):
    samples = []
    for entry in manifest["samples"]:
        if entry["split"] != split:
            continue
        sample = dict(entry)
        folder = directory / "dataset" / sample["id"]
        info = json.loads((folder / "info.json").read_text())
        validate_info(info, sample["rows"] * sample["columns"])
        sample.update(target=canonical_target(info["keyword"]), expected=sorted(info["correct_answers"]))
        sample["image"] = str(folder / "full_images.png")
        samples.append(sample)
    return samples


def selected(tiles, thresholds):
    # Match the public Rust f32 thresholds, including decimal rounding at a
    # boundary such as 0.7 -> 0.699999988079071.
    min_score, min_margin = (
        struct.unpack("f", struct.pack("f", thresholds[key]))[0]
        for key in ["min_score", "min_margin"]
    )
    return [tile["index"] for tile in tiles if tile["margin"] > 0
            and tile["target_score"] >= min_score
            and tile["margin"] >= min_margin]


def validate_predictions(samples, records):
    expected = {sample["id"]: sample for sample in samples}
    found = {}
    for record in records:
        key = record["id"]
        if key not in expected or key in found:
            raise ValueError("duplicate or unexpected prediction id")
        tiles = record["tiles"]
        count = expected[key]["rows"] * expected[key]["columns"]
        if (len(tiles) != count or any(type(tile["index"]) is not int for tile in tiles)
                or [tile["index"] for tile in tiles] != list(range(count))):
            raise ValueError("incomplete or misordered tile predictions")
        for tile in tiles:
            score, margin = tile["target_score"], tile["margin"]
            if (type(score) not in (int, float) or type(margin) not in (int, float)
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or not math.isfinite(margin) or not -1 <= margin <= 1):
                raise ValueError("invalid score/margin")
            baseline_selected = tile["margin"] > 0 and score >= 0.5 and margin >= 0.1
            if tile["selected"] is not baseline_selected:
                raise ValueError("Python and Rust baseline decisions differ")
        found[key] = record
    if set(found) != set(expected):
        raise ValueError("missing predictions; failed samples cannot be dropped")
    return found


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(samples, predictions, thresholds):
    return metrics_from_selected(samples, {
        sample["id"]: selected(predictions[sample["id"]]["tiles"], thresholds)
        for sample in samples
    })


def metrics_from_selected(samples, selections):
    """Measure decisions directly without inventing classification probabilities."""
    if not samples:
        raise ValueError("cannot measure an empty sample")
    tp = fp = fn = tn = exact = 0
    for sample in samples:
        truth = set(sample["expected"])
        prediction = set(selections[sample["id"]])
        count = sample["rows"] * sample["columns"]
        if any(type(index) is not int or not 0 <= index < count for index in prediction):
            raise ValueError("invalid selected index")
        tp += len(truth & prediction)
        fp += len(prediction - truth)
        fn += len(truth - prediction)
        tn += count - len(truth | prediction)
        exact += truth == prediction
    tiles = tp + fp + fn + tn
    # Wilson interval for grids; tiles from one grid are correlated.
    p = exact / len(samples)
    z = 1.959963984540054
    denominator = 1 + z * z / len(samples)
    centre = (p + z * z / (2 * len(samples))) / denominator
    radius = z * math.sqrt(p * (1 - p) / len(samples) + z * z / (4 * len(samples) ** 2)) / denominator
    return {
        "grids": len(samples), "tiles": tiles, "positives": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "tile_accuracy": (tp + tn) / tiles,
        "exact_grids": exact, "exact_grid_rate": p,
        "exact_grid_wilson_95": [max(0, centre - radius), min(1, centre + radius)],
    }


def best_thresholds(samples, predictions):
    # A small prespecified search; no test examples or labels enter selection.
    choices = []
    for score in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for margin in [0.0, 0.05, 0.1, 0.2]:
            thresholds = {"min_score": score, "min_margin": margin}
            measurement = metrics(samples, predictions, thresholds)
            key = (measurement["f1"] or 0, measurement["exact_grid_rate"],
                   measurement["precision"] or 0, score, margin)
            choices.append((key, thresholds))
    return max(choices, key=lambda item: item[0])[1]


def paired_f1_interval(samples, values, thresholds, repetitions=2000):
    """Paired resampling by grid, retaining within-grid tile dependence."""
    counts = []
    for sample in samples:
        pair = []
        for setting in [BASELINE, thresholds]:
            value = metrics([sample], values, setting)
            pair.append((value["tp"], value["fp"], value["fn"]))
        counts.append(pair)
    generator = random.Random("rs-ocr-f1-bootstrap-v1")
    deltas = []
    for _ in range(repetitions):
        sums = [[0, 0, 0], [0, 0, 0]]
        for _ in samples:
            for setting, row in enumerate(counts[generator.randrange(len(counts))]):
                for column, count in enumerate(row):
                    sums[setting][column] += count
        scores = [ratio(2 * tp, 2 * tp + fp + fn) for tp, fp, fn in sums]
        if all(score is not None for score in scores):
            deltas.append(scores[1] - scores[0])
    if not deltas:
        raise ValueError("F1 difference is undefined for every bootstrap replicate")
    deltas.sort()

    def quantile(probability):
        position = (len(deltas) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        return deltas[low] + (deltas[high] - deltas[low]) * (position - low)

    return {"method": "paired grid bootstrap, percentile interval", "seed": "rs-ocr-f1-bootstrap-v1",
            "repetitions": repetitions, "valid_repetitions": len(deltas),
            "f1_delta_95": [quantile(0.025), quantile(0.975)]}


def verify_gain(report):
    measurement = report["held_out"]["all"]
    baseline, calibrated = measurement["baseline"], measurement["calibrated"]
    if calibrated["f1"] <= baseline["f1"] or calibrated["tile_accuracy"] <= baseline["tile_accuracy"]:
        raise ValueError("the profile did not improve held-out tile F1 and accuracy")


def model_digest():
    expected = json.loads((CRATE / "model-manifests/clip-vit-b32.json").read_text())
    # Hash actual assets, including the tokenizer, instead of trusting names.
    for entry in expected["files"]:
        path = MODEL / entry["name"]
        with path.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != entry["size"] or sha != entry["sha256"]:
            raise ValueError(f"benchmark asset differs from pinned model: {path.name}")
    return next(entry["sha256"] for entry in expected["files"] if entry["name"] == "model.onnx")


def prediction_path(directory, split):
    return directory / f"{split}-predictions.jsonl"


def infer(samples, directory, split):
    path = prediction_path(directory, split)
    # Only complete successful subprocess output replaces an existing run.
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        requests = Path(temporary) / "requests.jsonl"
        output = Path(temporary) / "predictions.jsonl"
        with requests.open("w") as handle:
            for sample in samples:
                request = {key: sample[key] for key in ["id", "image", "rows", "columns", "target"]}
                request["alternatives"] = [label for label in LABELS if label != sample["target"]]
                handle.write(json.dumps(request) + "\n")
        command = [
            "cargo", "run", "-p", "rs-ocr", "--release", "--locked", "--features",
            "image-selection", "--example", "selection_batch", "--", str(MODEL),
        ]
        with requests.open("r") as stdin, output.open("w") as stdout:
            subprocess.run(command, cwd=ROOT, stdin=stdin, stdout=stdout, check=True)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        validate_predictions(samples, records)
        output.replace(path)
    print(f"saved {split}: {len(samples)} complete Rust predictions", flush=True)


def predictions(samples, directory, split):
    records = [json.loads(line) for line in prediction_path(directory, split).read_text().splitlines()]
    return validate_predictions(samples, records)


def calibrate(manifest, directory):
    samples = load_samples(manifest, directory, "development")
    values = predictions(samples, directory, "development")
    thresholds = best_thresholds(samples, values)
    profile = {
        "schema_version": 1,
        "dataset_revision": manifest["revision"],
        "sample_manifest_sha256": digest(MANIFEST.read_bytes()),
        "model_sha256": model_digest(),
        "development_predictions_sha256": digest(prediction_path(directory, "development").read_bytes()),
        "labels": LABELS, "baseline": BASELINE, "thresholds": thresholds,
        "selection_rule": (
            "maximum development tile F1, then exact grids, precision and stricter thresholds; "
            "10 score x 4 margin settings"
        ),
        "development_baseline": metrics(samples, values, BASELINE),
        "development_calibrated": metrics(samples, values, thresholds),
    }
    PROFILE.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


def make_report(manifest, directory):
    profile = json.loads(PROFILE.read_text())
    if profile["sample_manifest_sha256"] != digest(MANIFEST.read_bytes()) or profile["model_sha256"] != model_digest():
        raise ValueError("profile does not match dataset/model")
    development = load_samples(manifest, directory, "development")
    dev_predictions = predictions(development, directory, "development")
    if profile["development_predictions_sha256"] != digest(prediction_path(directory, "development").read_bytes()):
        raise ValueError("development predictions changed after calibration")
    if profile["thresholds"] != best_thresholds(development, dev_predictions):
        raise ValueError("profile was not selected by the development-only rule")
    test = load_samples(manifest, directory, "test")
    values = predictions(test, directory, "test")
    groups = {"all": test, "3x3": [s for s in test if s["rows"] == 3], "4x4": [s for s in test if s["rows"] == 4]}
    groups.update({
        target: [s for s in test if s["target"] == target]
        for target in sorted({s["target"] for s in test})
    })
    measurements = {
        key: {"baseline": metrics(samples, values, BASELINE),
              "calibrated": metrics(samples, values, profile["thresholds"])}
        for key, samples in groups.items()
    }
    details = []
    for sample in test:
        tiles = values[sample["id"]]["tiles"]
        details.append({"id": sample["id"], "target": sample["target"], "rows": sample["rows"],
                        "expected": sample["expected"], "baseline": selected(tiles, BASELINE),
                        "calibrated": selected(tiles, profile["thresholds"])})
    return {
        "schema_version": 1,
        "source": manifest["source"], "dataset_revision": manifest["revision"],
        "dataset_license": manifest["license"], "attribution": manifest["attribution"],
        "sample_manifest_sha256": digest(MANIFEST.read_bytes()),
        "profile_sha256": digest(PROFILE.read_bytes()),
        "model_sha256": profile["model_sha256"],
        "test_predictions_sha256": digest(prediction_path(directory, "test").read_bytes()),
        "protocol": (
            "120 deterministically selected grids; 60 development/60 test; "
            "no identical source tile/full-image blobs cross splits; F1 calibration uses development only; "
            "test assessed after profile freeze; no failed samples excluded"
        ),
        "labels": LABELS, "baseline": BASELINE, "calibrated": profile["thresholds"],
        "held_out": measurements, "test_grids": details,
        "paired_uncertainty": paired_f1_interval(test, values, profile["thresholds"]),
        "limitations": [
            "One public dataset, one annotator, small sample; no claim of general challenge success",
            "Repeated or near-duplicate scenes can remain despite exact source-byte grouping",
            "Correlated tiles; exact-grid Wilson interval assumes independent grids and may be optimistic",
            "Whole-tile CLIP can miss small/partial/secondary objects in 4x4 grids",
            "Relative scores depend on this exact competing label set; thresholds are not universal",
            "No training, dynamic-refresh loop, live-service interaction or token verification",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "infer-development", "calibrate", "infer-test", "report", "verify-report", "verify-profile",
    ])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    args.data = args.data.resolve()
    manifest = json.loads(MANIFEST.read_text())
    verify(manifest, args.data)
    if args.command.startswith("infer-"):
        if args.command == "infer-test" and not PROFILE.exists():
            raise ValueError("freeze a development-selected profile before test inference")
        split = "development" if args.command == "infer-development" else "test"
        model_digest()
        infer(load_samples(manifest, args.data, split), args.data, split)
    elif args.command == "calibrate":
        if prediction_path(args.data, "test").exists():
            raise ValueError("test already evaluated: use a new untouched split before recalibrating")
        calibrate(manifest, args.data)
    else:
        report = make_report(manifest, args.data)
        if args.command == "report":
            REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: report["held_out"][key] for key in ["all", "3x3", "4x4"]}, indent=2))
        elif json.loads(REPORT.read_text()) != report:
            raise ValueError("report differs from measured predictions")
        elif args.command == "verify-profile":
            verify_gain(report)
            sample = next(item for item in report["test_grids"] if item["calibrated"] != item["baseline"])
            command = [
                "cargo", "run", "-p", "rs-ocr", "--release", "--locked", "--features", "image-selection",
                "--example", "image_selection", "--", "--profile", str(PROFILE), str(MODEL),
                str(args.data / "dataset" / sample["id"] / "full_images.png"),
                str(sample["rows"]), str(sample["rows"]), sample["target"],
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            line = next(line for line in result.stdout.splitlines() if line.startswith("selected: "))
            if json.loads(line.removeprefix("selected: ")) != sample["calibrated"]:
                raise ValueError("Rust profile CLI differs from the measured calibrated decisions")
            print("PROFILE_INFERENCE_VERIFIED", sample["id"], line)
        else:
            verify_gain(report)
            print("ACCURACY_REPORT_VERIFIED")


if __name__ == "__main__":
    main()
