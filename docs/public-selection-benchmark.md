# Public image-selection evaluation

This preserves the original CLIP-only study. The later
[fresh 4×4 localization study](localization-benchmark.md) evaluates full-image
object detection on different samples without retuning this completed test set.

Measured on 2026-10-29 using the pinned FP32 CLIP ViT-B/32 model, RTen 0.27.0,
Rust 1.95.0, and macOS on an Intel MacBook Pro. All image inference ran in Rust.
The Python standard-library scripts download data, orchestrate the Rust example,
calibrate thresholds and compute metrics; they do not run a Python ML model.

## Data and protocol

Source: [Suphannee Sivakorn's reCAPTCHA Study](https://github.com/ssivakorn/reCAPTCHA-study),
revision `efb3595c33780abf9d326c729973e72d165366c2`. Its human-annotated data is
published under [CC BY-NC 4.0](https://github.com/ssivakorn/reCAPTCHA-study/blob/efb3595c33780abf9d326c729973e72d165366c2/LICENSE).
The benchmark uses existing public offline images. The images and annotations
are downloaded separately into ignored `models/public-selection-samples/`;
they are not included in the crate's distributable files. The source-derived
sample manifest retains its attribution and dataset license independently of
the crate's code license.

The upstream dataset has 1,000 grids. Before inference, the sampler groups
challenges sharing identical source tile or full-image Git blobs, keeps one
representative from each group, then orders IDs by SHA-256 of a fixed seed and
ID. It selects 80 independent-image 3×3 grids and 40 single-photo 4×4 grids.
The first half of each size goes to development and the second half to testing:

| Split | 3×3 grids | 4×4 grids | Tiles | Positive tiles |
|---|---:|---:|---:|---:|
| Development | 40 | 20 | 680 | 228 |
| Held-out test | 40 | 20 | 680 | 239 |

The [sample manifest](../model-manifests/public-selection-samples.json) pins
every downloaded file's size and Git blob digest. Verification rejects corrupt
downloads, duplicate IDs, malformed annotations and exact source-image overlap
between splits. Near-duplicate scenes can still remain.

Both runs use the same fixed 16-label vocabulary, including competing objects,
buildings, trees, empty road and sidewalk. The instruction aliases `bus` and
`a fire hydrant` map to `buses` and `fire hydrants`. No labels or samples are
silently dropped. Inputs are the supplied `full_images.png` grids without UI
text. Their square tile geometry does not justify changing the existing center
crop. The difficult development examples instead show small, noisy objects and
partial objects that span several tiles.

The baseline uses the API defaults: score ≥0.5, margin ≥0.1, strict target win.
Calibration searches a prespecified 10 score thresholds ×4 margin thresholds
on development data, maximizing tile F1, then exact grid rate, precision, and
stricter thresholds for ties. It selected **score ≥0.3, margin ≥0.1**. This profile
was written before test inference. Calibration refuses to overwrite it after
test predictions exist. No weights, prompts or preprocessing were tuned.

## Held-out results

| Metric | Default thresholds | Calibrated profile |
|---|---:|---:|
| Tile accuracy | 83.53% | **84.71%** |
| Tile precision | **95.68%** | 92.45% |
| Tile recall | 55.65% | **61.51%** |
| Tile F1 | 70.37% | **73.87%** |
| Exact grids | 20/60 (33.33%) | 20/60 (33.33%) |
| True positives / false positives | 133 / 6 | 147 / 12 |
| False negatives / true negatives | 106 / 435 | 92 / 429 |

This is a measured recall/F1 improvement with a precision cost. It does not
improve exact grid success on this sample. Thirteen grids change selections;
none become fully correct or lose an existing exact match.

| Grid type | Baseline tile F1 | Calibrated tile F1 | Calibrated exact grids |
|---|---:|---:|---:|
| 3×3 independent images | 78.82% | 82.95% | 19/40 |
| 4×4 single photo | 60.57% | 62.98% | 1/20 |

The paired grid bootstrap (2,000 replicates, fixed seed) gives a 95% percentile
interval of **+1.28 to +6.42 percentage points** for the tile F1 difference,
assuming independent grids. The exact-grid rate's Wilson interval is 22.73–45.94%
for either setting. These intervals describe this sample and its assumptions,
not a guarantee for other datasets or a live service.

The [machine-readable report](public-selection-results.json) contains confusion
counts, per-category results, uncertainty, every test grid's expected and
predicted indices, and hashes binding the profile, data manifest and predictions.
For example, cars have only 54.90% calibrated F1 and no exact grid matches in
11 test grids; motorcycles have 39.22% F1 and no exact matches in six grids.
Traffic-light F1 decreases slightly, from 70.00% to 69.77%. The aggregate gain
does not apply uniformly to every category.

## Use the optional profile

The library's conservative defaults stay at 0.5/0.1. The example can load the
[frozen profile](../model-manifests/public-selection-profile.json), which supplies
both thresholds and the entire calibrated competing-label set:

```sh
cargo run --release --features image-selection --example image_selection -- --profile model-manifests/public-selection-profile.json models/clip-vit-b32 local-grid.png 3 3 cars
```

Do not append alternative labels when using a profile. Unknown targets, duplicate
labels, unsupported schemas and invalid thresholds are rejected. Use the pinned
model assets verified by the download helper. A different model, vocabulary,
prompt, or input distribution needs its own calibration and fresh test split.

Rust applications can use the same `SelectionOptions` API with the profile's
target, all other profile labels as alternatives, and `min_score = 0.3`,
`min_margin = 0.1`. Dropping competing labels changes relative scores and invalidates
this calibration.

## Reproduce

from the repository root, with Python 3.11 or later and the configured mise Rust
toolchain. Model setup is documented in [image-selection.md](image-selection.md).
On a fresh data directory:

```sh
python scripts/public_samples.py download
python scripts/benchmark_selection.py infer-development
python scripts/benchmark_selection.py calibrate
python scripts/benchmark_selection.py infer-test
python scripts/benchmark_selection.py report
python scripts/benchmark_selection.py verify-report
python scripts/benchmark_selection.py verify-profile
python -m unittest discover -s scripts -p 'test_*.py'
```

For an existing completed run, use `verify-report` and `verify-profile` without
recalibrating. Both scripts accept `--data PATH`. A fresh directory permits a
new reproduction; retain the original predictions for comparison. The JSONL
`selection_batch` example reuses one model across all requests, caps inference
at four tiles per batch, and aborts on any failed sample. Reproduction on other
hardware/runtime versions may produce different floating-point outputs.

## Remaining accuracy limits

CLIP classifies each tile independently. It can miss a small car behind noise,
a wheel fragment, or a rider spanning a 4×4 grid. A lower threshold cannot recover
a target that loses to another label. The inspected development scooter example
`2026-05-02-17-04-28` still misses seven annotated tiles after calibration.

These are one annotator's offline labels, not responses verified against a live
challenge service. The sample has only 60 test grids, no test examples of stairs,
and no dynamic refresh sequences. Exact byte grouping does not remove all shared
scenes. Improving 4×4 performance calls for evaluating full-image localization or
segmentation on a fresh held-out set; further threshold tuning on this test set
would invalidate its role as an independent check.
