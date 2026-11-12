# Full-image detection for contiguous-photo grids

The optional `object-detection` feature adds `rs_ocr::detection`, using the
existing pure Rust `RTen` 0.27.0 runtime. `ObjectDetector` runs an entire image
through a pretrained COCO detector, preserving context that is lost when CLIP
classifies small 4×4 fragments independently. `DetectionResult::select_grid`
maps object boxes back to the original pixel grid. `HybridSelector` uses this
method for supported targets and returns an explicit CLIP fallback otherwise.

This implementation uses **object detection, not segmentation**. Box coverage
can include background. The published results apply to contiguous photographs;
independent-image 3×3 montages should continue using `ImageSelector`.

## Model and processing

The pinned FP32 **YOLO11s** ONNX export is from
[unileon-robotics/YOLO11-ONNX](https://huggingface.co/unileon-robotics/YOLO11-ONNX),
revision `0e2a3f195ae62a25bc038ee9665b8e4ac5b98793`, file
`dynamic/yolo11s.onnx`. It is 38,168,496 bytes, SHA-256
`f750b6dbdfb7346ce2f109246c9d1950d030904daab3d49bf2942d4772ce69df`.
The model card declares **AGPL-3.0**. Weights have their own license and remain
separate from the crate; review that license for your intended use. The
[manifest](../model-manifests/yolo11s.json) records source, size and digest.

```sh
python scripts/download_detector.py
```

The helper reuses the existing atomic, SHA-256-verified downloader. Inference
does not download anything or use Python. RGB inputs use `/255` normalization,
bilinear aspect-preserving resize and centered padding of 114. Dynamic input
dimensions default to 640×640; fixed model dimensions are respected, with a
supported range of 32..=1280. Raw outputs must be `[1,84,anchors]` float32: xywh
plus 80 COCO probabilities. Non-finite values, invalid scores and incompatible outputs fail.

The decoder restores original-image coordinates and performs class-aware NMS.
The benchmark retains scores ≥0.05, uses NMS `IoU` 0.45, and keeps at most 100
objects. A cell is selected when a matching object meets the score threshold
and its box covers the configured fraction of that cell. Coverage uses the
largest individual matching-box intersection, not a sum of overlaps. A zero
coverage threshold still requires positive overlap; touching edges do not count.
Odd image dimensions use the same proportional boundaries as `Grid`.

`CocoClass::from_label` recognizes canonical COCO names and documented common
aliases, including bicycles, cars, motorcycles, buses, traffic lights and fire
hydrants. It does not map taxis to all cars or pretend that crosswalks/stairs
are COCO classes. The hybrid fallback is explicit; detector failures remain errors.

## Rust API

```rust,no_run
use rs_ocr::{
    detection::{HybridSelector, HybridSelection, SpatialOptions},
    images::selection::{Grid, SelectionOptions},
};

let selector = HybridSelector::from_model_dirs(
    "models/yolo11s", "models/clip-vit-b32",
)?;
let classifier_options = SelectionOptions::new(
    "bicycles", vec!["cars".into(), "an empty road".into()],
);
let result = selector.select_file(
    "single-photo-grid.png", Grid::new(4, 4)?, &classifier_options,
    SpatialOptions { min_confidence: 0.1, min_coverage: 0.1 },
)?;
println!("{:?}", result.selected_indices());
match result {
    HybridSelection::Objects { detections, selection, .. } => {
        println!("{} objects; {:?}", detections.objects.len(), selection.tile_coverage);
    }
    HybridSelection::Classification(result) => println!("CLIP fallback: {} tiles", result.tiles.len()),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Enable `object-detection` on the dependency. Both file and encoded-byte methods
reuse loaded sessions. This feature includes `image-selection`. The example's
CLIP alternatives illustrate the interface; benchmark parity on fallback classes
requires the complete earlier profile vocabulary and its thresholds. Spatial
options above are a measured research preset for the pinned detector, not a
universal setting. Bare `SpatialOptions::default()` remains 0.25/0.01.

For object-only use, construct `ObjectDetector`, call `detect_file`/`detect_bytes`,
then `select_grid`. If changing selection confidence later, retain detections
down to the lowest threshold you intend to use. The full COCO vocabulary is
available through validated `CocoClass` indices and canonical names.

## Fresh evaluation

Evaluated on **2026-10-29**, Intel macOS, Rust 1.95.0, `RTen` 0.27.0. Source data:
[Suphannee Sivakorn's reCAPTCHA Study](https://github.com/ssivakorn/reCAPTCHA-study),
revision `efb3595c33780abf9d326c729973e72d165366c2`, **CC BY-NC 4.0**.
The source-derived [sample manifest](../model-manifests/localization-samples.json)
retains the dataset attribution/license. Images and annotations are downloaded
into ignored `models/localization-samples/` and are not redistributed with code.

Before inference, SHA-256 ordering of a new seed and challenge ID selected 100
fresh 4×4 grids, excluding every prior-study ID and shared source image blob.
The first 50 were development; the other 50 were held out. Selected grids also
share no identical source tile/full-image blobs with each other. This excludes
exact duplicates, not all near-duplicate scenes.
Freshness is relative to the earlier local study; overlap with pretrained-model
training images has not been audited.

The baseline uses the earlier fixed CLIP profile, score 0.3/margin 0.1, with its
complete 16-label vocabulary. Fresh data also contains taxis, an unseen label.
Those grids were retained using the existing API defaults 0.5/0.1 and the same
fixed alternatives plus the new target. This loader clarification occurred
before any test inference; it introduced no fitted settings or taxi-to-car
remapping. Unsupported target categories use these same baseline decisions in
the hybrid method.

Development-only search considered five score thresholds × six cell-coverage
thresholds, maximizing exact grids, then tile F1, precision and stricter settings
for ties. It selected confidence **0.1** and coverage **0.1**, increasing development
exact grids from 3/50 to 15/50. The profile was frozen before test inference.
Calibration refuses to run once test predictions exist. No test images were
discarded and no inference failures were excluded from metrics.

### Held-out 50 grids / 800 tiles

| Metric | CLIP baseline | Spatial hybrid |
|---|---:|---:|
| Tile accuracy | 79.13% | **88.00%** |
| Tile precision | **92.90%** | 88.93% |
| Tile recall | 48.00% | **77.67%** |
| Tile F1 | 63.30% | **82.92%** |
| Exact grids | 7/50 (14%) | **15/50 (30%)** |
| TP / FP / FN / TN | 144 / 11 / 156 / 489 | 233 / 29 / 67 / 471 |

Of the 50 grids, 45 used detection: their F1 rose from 60.91% to 83.43%, with
exact matches rising from 7/45 to 15/45. Five used CLIP fallback: two crosswalk,
one stairs and two taxi grids; their outputs remain unchanged. Two taxi grids
are separately identified as using uncalibrated API defaults. Eleven previously
incorrect grids become exact matches; three previous exact matches are lost.
The improvement has a precision cost and is not uniform: bicycle exact matches
fall from 2/7 to 1/7 even though bicycle tile F1 improves.

The spatial exact-grid Wilson 95% interval is 19.10–43.75%, assuming independent
grids. The [machine-readable report](localization-results.json) includes
per-category metrics, every expected/predicted index set, fallback method,
confidence intervals and hashes binding source/profile/prediction assets. Each
test selection was executed in Rust; the Python evaluator independently checks
the spatial arithmetic using float32 boundary semantics.

This sample is small, from one public dataset and one annotator. Box localization
can miss fragments or select background; the remaining 35 failed grids are real
limitations. Results do not establish live-service acceptance or dynamic-refresh
success. Further model tuning needs a new untouched test split.

## Reproduce and verify

from the repository root, with Python 3.11+ and the configured mise toolchain:

```sh
python scripts/download_detector.py
python scripts/localization_benchmark.py download
python scripts/localization_benchmark.py infer-development
python scripts/localization_benchmark.py calibrate
python scripts/localization_benchmark.py infer-test
python scripts/localization_benchmark.py report
python scripts/localization_benchmark.py verify-report
cargo test --all-features --release --locked --test detection_smoke -- --ignored
```

Download CLIP/model reference photos using the earlier setup instructions first.
For an already-completed run, use `verify-data` and `verify-report`. To reproduce
from scratch without overwriting observations, both studies support a fresh
`--data PATH`; keep the earlier raw predictions for comparison. Predictions and
weights remain ignored. The `spatial_batch` example loads both models once and
emits complete JSONL records, aborting on failed input rather than dropping it.
