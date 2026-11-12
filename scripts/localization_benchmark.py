#!/usr/bin/env python3
"""Fresh offline 4x4 study: full-image Rust detection versus frozen CLIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile

import benchmark_selection as clip
from public_samples import CRATE, REVISION, digest, fetch, validate_info, verify_bytes

MANIFEST = CRATE / "model-manifests/localization-samples.json"
PROFILE = CRATE / "model-manifests/localization-profile.json"
REPORT = CRATE / "docs/localization-results.json"
DATA = CRATE / "models/localization-samples"
DETECTOR = CRATE / "models/yolo11s/model.onnx"
MODEL_MANIFEST = CRATE / "model-manifests/yolo11s.json"
PREVIOUS = CRATE / "model-manifests/public-selection-samples.json"
CLIP_PROFILE = CRATE / "model-manifests/public-selection-profile.json"
SEED = "rs-ocr-fresh-localization-v1"
CLASS_IDS = {"bicycles": 1, "cars": 2, "motorcycles": 3, "buses": 5,
             "traffic lights": 9, "fire hydrants": 10}


def source_blobs(sample):
    return set(sample["tile_blobs"]) | {
        entry["git_blob"] for entry in sample["files"] if entry["path"].endswith("full_images.png")
    }


def fresh_manifest(tree, previous):
    if tree["truncated"]:
        raise ValueError("incomplete source index")
    files = {}
    for entry in tree["tree"]:
        parts = entry["path"].split("/")
        if len(parts) == 3 and parts[0] == "dataset" and entry["type"] == "blob":
            files.setdefault(parts[1], {})[parts[2]] = entry
    excluded_ids = {sample["id"] for sample in previous["samples"]}
    seen = set().union(*(source_blobs(sample) for sample in previous["samples"]))
    samples = []
    for key in sorted(files, key=lambda key: digest(f"{SEED}:{key}".encode())):
        group = files[key]
        if key in excluded_ids or sum(name.startswith("tile_") for name in group) != 16:
            continue
        tiles = [group[f"tile_{i}.png"]["sha"] for i in range(16)]
        blobs = set(tiles) | {group["full_images.png"]["sha"]}
        if blobs & seen:
            continue
        seen.update(blobs)
        samples.append({"id": key, "split": "development" if len(samples) < 50 else "test",
                        "rows": 4, "columns": 4, "tile_blobs": tiles,
                        "files": [{"path": group[name]["path"], "size": group[name]["size"],
                                   "git_blob": group[name]["sha"]} for name in ["full_images.png", "info.json"]]})
        if len(samples) == 100:
            break
    if len(samples) != 100:
        raise ValueError("not enough fresh 4x4 grids")
    return {"schema_version": 1, "source": previous["source"], "revision": REVISION,
            "license": previous["license"], "attribution": previous["attribution"], "seed": SEED,
            "previous_manifest_sha256": digest(PREVIOUS.read_bytes()),
            "selection": "SHA256(seed:id) order; 100 fresh 4x4 grids; first 50 development, next 50 test; exclude all previous IDs/source blobs and shared blobs among selected grids",
            "samples": sorted(samples, key=lambda sample: (sample["split"], sample["id"]))}


def check_freshness(manifest, previous):
    old_ids = {sample["id"] for sample in previous["samples"]}
    old_blobs = set().union(*(source_blobs(sample) for sample in previous["samples"]))
    seen_ids, seen_blobs = set(), set()
    counts = {"development": 0, "test": 0}
    for sample in manifest["samples"]:
        if sample["rows"] != 4 or sample["columns"] != 4 or len(sample["tile_blobs"]) != 16:
            raise ValueError("expected 4x4 samples")
        blobs = source_blobs(sample)
        if sample["id"] in old_ids | seen_ids or blobs & (old_blobs | seen_blobs):
            raise ValueError("sample/image leakage against earlier or new study")
        seen_ids.add(sample["id"])
        seen_blobs.update(blobs)
        counts[sample["split"]] += 1
    if counts != {"development": 50, "test": 50}:
        raise ValueError("expected 50 development and 50 test samples")


def verify_data(manifest, directory):
    if manifest["revision"] != REVISION or manifest["previous_manifest_sha256"] != digest(PREVIOUS.read_bytes()):
        raise ValueError("source or previous study changed")
    check_freshness(manifest, json.loads(PREVIOUS.read_text()))
    for sample in manifest["samples"]:
        for entry in sample["files"]:
            verify_bytes((directory / entry["path"]).read_bytes(), entry)
        info = json.loads((directory / "dataset" / sample["id"] / "info.json").read_text())
        validate_info(info, 16)
    print("FRESH_LOCALIZATION_DATA_VERIFIED", flush=True)


def load_samples(manifest, directory, split):
    samples = []
    for entry in manifest["samples"]:
        if entry["split"] != split:
            continue
        sample = dict(entry)
        folder = directory / "dataset" / sample["id"]
        info = json.loads((folder / "info.json").read_text())
        validate_info(info, 16)
        keyword = info["keyword"].strip().lower()
        target = {"bus": "buses", "a fire hydrant": "fire hydrants"}.get(keyword, keyword)
        sample.update(target=target, expected=sorted(info["correct_answers"]), image=str(folder / "full_images.png"))
        samples.append(sample)
    return samples


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def spatial_selected(record, thresholds):
    if record["target_class"] is None:
        return None
    minimum = f32(thresholds["min_confidence"])
    coverage = f32(thresholds["min_coverage"])
    indices = []
    for index in range(16):
        row, column = divmod(index, 4)
        left, right = column * record["width"] // 4, (column + 1) * record["width"] // 4
        top, bottom = row * record["height"] // 4, (row + 1) * record["height"] // 4
        maximum = 0.0
        for obj in record["objects"]:
            if obj["class_id"] != record["target_class"] or obj["confidence"] < minimum:
                continue
            x1, y1, x2, y2 = obj["bounds"]
            width = max(0.0, f32(min(x2, right) - max(x1, left)))
            height = max(0.0, f32(min(y2, bottom) - max(y1, top)))
            maximum = max(maximum, f32(f32(width * height) / ((right - left) * (bottom - top))))
        if maximum > 0 and maximum >= coverage:
            indices.append(index)
    return indices


def validate_records(samples, records):
    values = clip.validate_predictions(samples, records)
    for sample in samples:
        record = values[sample["id"]]
        expected_class = CLASS_IDS.get(sample["target"])
        if record["target_class"] != expected_class or (expected_class is not None and type(record["target_class"]) is not int):
            raise ValueError("Rust/Python class mapping differs")
        for key in ["width", "height"]:
            if type(record[key]) is not int or not 4 <= record[key] <= 8192:
                raise ValueError("invalid source dimensions")
        if len(record["objects"]) > 100:
            raise ValueError("too many detection records")
        for obj in record["objects"]:
            if type(obj["class_id"]) is not int or not 0 <= obj["class_id"] < 80:
                raise ValueError("invalid object class")
            score = obj["confidence"]
            bounds = obj["bounds"]
            if (type(score) not in (int, float) or not math.isfinite(score) or not 0.05 <= score <= 1
                    or len(bounds) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)):
                raise ValueError("invalid object confidence/coordinates")
            x1, y1, x2, y2 = bounds
            if not 0 <= x1 < x2 <= record["width"] or not 0 <= y1 < y2 <= record["height"]:
                raise ValueError("out-of-image object bounds")
        setting = record["spatial_options"]
        if any(type(setting[key]) not in (int, float) or not math.isfinite(setting[key])
               or not 0 <= setting[key] <= 1 for key in ["min_confidence", "min_coverage"]):
            raise ValueError("invalid recorded spatial thresholds")
        if spatial_selected(record, setting) != record["spatial_selected"]:
            raise ValueError("Rust/Python spatial selections differ")
    return values


def selections(samples, values, thresholds):
    baseline_settings = json.loads(CLIP_PROFILE.read_text())["thresholds"]
    # Fresh data can introduce categories outside the old calibration vocabulary.
    # Keep these examples using the established API defaults, with no retuning.
    baseline = {s["id"]: clip.selected(values[s["id"]]["tiles"],
                                      baseline_settings if s["target"] in clip.LABELS else clip.BASELINE)
                for s in samples}
    hybrid = {s["id"]: (spatial_selected(values[s["id"]], thresholds)
                         if values[s["id"]]["target_class"] is not None else baseline[s["id"]]) for s in samples}
    return baseline, hybrid


def fit(samples, values):
    choices = []
    for confidence in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for coverage in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]:
            thresholds = {"min_confidence": confidence, "min_coverage": coverage}
            _, hybrid = selections(samples, values, thresholds)
            metric = clip.metrics_from_selected(samples, hybrid)
            key = (metric["exact_grid_rate"], metric["f1"] or 0, metric["precision"] or 0, confidence, coverage)
            choices.append((key, thresholds))
    return max(choices, key=lambda pair: pair[0])[1]


def model_digest():
    clip.model_digest()
    manifest = json.loads(MODEL_MANIFEST.read_text())
    entry = manifest["files"][0]
    with DETECTOR.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if DETECTOR.stat().st_size != entry["size"] or sha != entry["sha256"]:
        raise ValueError("detector differs from pinned asset")
    return sha


def predictions(directory, samples, split):
    path = directory / f"{split}-predictions.jsonl"
    return validate_records(samples, [json.loads(line) for line in path.read_text().splitlines()])


def infer(directory, samples, split):
    model_digest()
    command = ["cargo", "run", "-p", "rs-ocr", "--release", "--locked", "--features", "object-detection",
               "--example", "spatial_batch", "--", str(clip.MODEL), str(DETECTOR)]
    if split == "test":
        if not PROFILE.exists():
            raise ValueError("freeze development settings before test inference")
        command.append(str(PROFILE))
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        requests, output = Path(temporary) / "requests.jsonl", Path(temporary) / "predictions.jsonl"
        with requests.open("w") as handle:
            for sample in samples:
                request = {key: sample[key] for key in ["id", "image", "rows", "columns", "target"]}
                request["alternatives"] = [label for label in clip.LABELS if label != sample["target"]]
                handle.write(json.dumps(request) + "\n")
        with requests.open() as stdin, output.open("w") as stdout:
            subprocess.run(command, cwd=CRATE.parent, stdin=stdin, stdout=stdout, check=True)
        validate_records(samples, [json.loads(line) for line in output.read_text().splitlines()])
        output.replace(directory / f"{split}-predictions.jsonl")
    print("RUST_LOCALIZATION_INFERENCE_VERIFIED", split, len(samples))


def calibrate(manifest, directory):
    if (directory / "test-predictions.jsonl").exists():
        raise ValueError("test predictions already exist; do not retune this study")
    samples = load_samples(manifest, directory, "development")
    values = predictions(directory, samples, "development")
    thresholds = fit(samples, values)
    baseline, hybrid = selections(samples, values, thresholds)
    profile = {"schema_version": 1, "dataset_revision": REVISION,
               "manifest_sha256": digest(MANIFEST.read_bytes()), "previous_clip_profile_sha256": digest(CLIP_PROFILE.read_bytes()),
               "detector_sha256": model_digest(), "detector_manifest_sha256": digest(MODEL_MANIFEST.read_bytes()),
               "development_predictions_sha256": digest((directory / "development-predictions.jsonl").read_bytes()),
               "objective": "development exact grids, then tile F1, precision, stricter thresholds; 5 confidence x 6 coverage settings",
               "thresholds": thresholds, "nms_iou": 0.45, "inference_score_floor": 0.05,
               "max_detections": 100, "unsupported_policy": "previous CLIP profile for known labels; API defaults for unseen labels",
               "development_baseline": clip.metrics_from_selected(samples, baseline),
               "development_hybrid": clip.metrics_from_selected(samples, hybrid)}
    PROFILE.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


def measured_report(manifest, directory):
    profile = json.loads(PROFILE.read_text())
    bindings = {"manifest_sha256": digest(MANIFEST.read_bytes()),
                "previous_clip_profile_sha256": digest(CLIP_PROFILE.read_bytes()),
                "detector_sha256": model_digest(), "detector_manifest_sha256": digest(MODEL_MANIFEST.read_bytes()),
                "development_predictions_sha256": digest((directory / "development-predictions.jsonl").read_bytes())}
    if any(profile[key] != value for key, value in bindings.items()):
        raise ValueError("profile provenance changed")
    development = load_samples(manifest, directory, "development")
    if fit(development, predictions(directory, development, "development")) != profile["thresholds"]:
        raise ValueError("profile is not the development-selected choice")
    samples = load_samples(manifest, directory, "test")
    values = predictions(directory, samples, "test")
    if any(record["spatial_options"] != {key: f32(value) for key, value in profile["thresholds"].items()} for record in values.values()):
        raise ValueError("held-out Rust run did not use the frozen profile")
    baseline, hybrid = selections(samples, values, profile["thresholds"])
    groups = {"all": samples,
              "detector_supported": [s for s in samples if values[s["id"]]["target_class"] is not None],
              "clip_fallback": [s for s in samples if values[s["id"]]["target_class"] is None]}
    groups["unseen_label_defaults"] = [s for s in samples if s["target"] not in clip.LABELS]
    groups.update({label: [s for s in samples if s["target"] == label] for label in sorted({s["target"] for s in samples})})
    measurements = {key: {"clip_baseline": clip.metrics_from_selected(group, baseline),
                          "spatial_hybrid": clip.metrics_from_selected(group, hybrid)}
                    for key, group in groups.items() if group}
    details = [{"id": s["id"], "target": s["target"], "expected": s["expected"],
                "method": "detector" if values[s["id"]]["target_class"] is not None else "clip_fallback",
                "clip_settings": "previous_profile" if s["target"] in clip.LABELS else "api_defaults_unseen_label",
                "clip": baseline[s["id"]], "hybrid": hybrid[s["id"]]} for s in samples]
    return {"schema_version": 1, "source": manifest["source"], "dataset_revision": REVISION,
            "dataset_license": manifest["license"], "attribution": manifest["attribution"],
            "manifest_sha256": digest(MANIFEST.read_bytes()), "profile_sha256": digest(PROFILE.read_bytes()),
            "test_predictions_sha256": digest((directory / "test-predictions.jsonl").read_bytes()),
            "protocol": "100 fresh 4x4 grids, 50 development/50 test; no old/new shared source image blobs; selection chosen on development only; baseline is prior frozen CLIP profile for known labels, API defaults for unseen labels; every test detection selection executed in Rust",
            "thresholds": profile["thresholds"], "held_out": measurements, "test_grids": details,
            "limitations": ["One public dataset/annotator; near-duplicate scenes may remain",
                            "Boxes include background and miss small objects; this is detection, not segmentation",
                            "Unsupported classes use explicitly reported CLIP fallback",
                            "No live acceptance or dynamic-refresh evaluation"]}


def verify_improvement(report):
    baseline = report["held_out"]["all"]["clip_baseline"]
    hybrid = report["held_out"]["all"]["spatial_hybrid"]
    if hybrid["f1"] <= baseline["f1"] or hybrid["exact_grids"] <= baseline["exact_grids"]:
        raise ValueError("held-out tile F1 and exact grids did not both improve")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create-manifest", "download", "verify-data", "infer-development",
                                           "calibrate", "infer-test", "report", "verify-report"])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--tree", type=Path)
    args = parser.parse_args()
    args.data = args.data.resolve()
    if args.command == "create-manifest":
        if args.tree is None:
            parser.error("create-manifest needs --tree")
        manifest = fresh_manifest(json.loads(args.tree.read_text()), json.loads(PREVIOUS.read_text()))
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print("created fresh sample manifest", len(manifest["samples"]))
        return
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "download":
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda entry: fetch(entry, args.data), [e for s in manifest["samples"] for e in s["files"]]))
    verify_data(manifest, args.data)
    if args.command.startswith("infer-"):
        split = "development" if args.command == "infer-development" else "test"
        infer(args.data, load_samples(manifest, args.data, split), split)
    elif args.command == "calibrate":
        calibrate(manifest, args.data)
    elif args.command in ["report", "verify-report"]:
        report = measured_report(manifest, args.data)
        if args.command == "report":
            REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: report["held_out"][key] for key in ["all", "detector_supported", "clip_fallback"] if key in report["held_out"]}, indent=2))
        else:
            if json.loads(REPORT.read_text()) != report:
                raise ValueError("report differs from raw predictions")
            verify_improvement(report)
            print("LOCALIZATION_REPORT_VERIFIED")


if __name__ == "__main__":
    main()
