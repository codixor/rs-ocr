# Local image selection with CLIP

The optional `image-selection` feature provides `rs_ocr::images::selection`.
It uses the pretrained **CLIP ViT-B/32** image/text model through a full-precision
ONNX export and the same pure Rust **`RTen` 0.27.0** runtime already used by `RustO`.
No model training or Python inference service is needed.

CLIP was chosen because labels can be supplied at runtime. A fixed COCO object
detector would restrict the supported label vocabulary; CLIP can compare images
with different textual categories without retraining. This module performs
whole-tile classification, not pixel segmentation or object localization.

## Pinned model setup

from the repository root:

```sh
python scripts/download_clip.py --test-images
```

This Python standard-library helper is only for downloading files. It verifies
every file against the checked-in SHA-256 and size manifest, including cache
hits. Downloads are written to temporary files before replacing a destination.
Use `--output PATH` to choose another model directory. Omit `--test-images` when
the reference photos are not needed.

The combined ONNX model is about 606 MB plus a 2.2 MB tokenizer. The selected
export is `Xenova/clip-vit-base-patch32` at revision
`6b7c523703ae26e39be4b03774966f8571c7f523`; its model card declares MIT licensing.
The download helper saves the original FP32 export as `model.onnx`.
Weights and photos remain under the ignored `models/` directory.

The quantized combined export failed the known-photo classification check in
`RTen` 0.27.0 on the tested Intel Mac, while ONNX Runtime 1.23.2 recognized the
same photo correctly. This implementation therefore uses the tested FP32 export.
Its tokenizer loader also supplies the historical BPE `ignore_merges=false`
default in memory when older JSON omits it, without altering downloaded files.

## Rust API

Enable `image-selection` on your `rs-ocr` dependency. Create one selector and
reuse its model and tokenizer across requests:

```rust,no_run
use rs_ocr::images::selection::{Grid, ImageSelector, SelectionOptions};

let selector = ImageSelector::from_model_dir("models/clip-vit-b32")?;
let grid = Grid::new(3, 3)?;
let options = SelectionOptions::new(
    "buses",
    vec!["cars".into(), "bicycles".into(), "buildings".into(), "empty road".into()],
);
let result = selector.select_file("local-grid.png", grid, &options)?;
println!("selected tiles: {:?}", result.selected_indices());

let bytes = std::fs::read("another-grid.png")?;
let result = selector.select_bytes(&bytes, grid, &options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The grid must describe just the tile area, without instruction text or UI
borders. Tiles are split proportionally, so non-divisible image dimensions do
not lose pixels. Indices, rows and columns are **zero-based, in row-major order**.
A 3×3 grid therefore has indices 0 through 8.

`SelectionResult` includes every tile, original pixel rectangle, all label scores,
target score, margin and selection flag. `selected_indices()` returns only tiles
whose target label wins strictly, meets `min_score` (default 0.5), and beats the
strongest alternative by `min_margin` (default 0.1). Ties never select a tile.

Scores are softmax values relative to the supplied labels, not calibrated object
presence probabilities. At least one distinct competing label is required;
otherwise a one-label softmax would always score 1. Supply plausible competing
and background categories. A tile can contain several objects; a single winning
class may miss secondary or partially visible objects. No challenge-specific
accuracy claim follows from the reference-photo tests.

The separate [public-sample benchmark](public-selection-benchmark.md) now measures
120 labeled offline grids with a 60-grid held-out test split. It includes a
reproducible evaluation and optional fixed-vocabulary threshold profile. That
profile improves tile F1 on this sample while reducing precision; exact grid
success remains 20/60. It does not change this API's default thresholds.

For single photographs split into 4×4 cells, the later
[full-image detection study](localization-benchmark.md) tests a spatial alternative
on a fresh, disjoint sample and provides `HybridSelector` with explicit fallback.

## Input and model contract

- Encoded PNG/JPEG inputs, 1 byte to 32 MiB; decoded dimensions at most 8192 per
  side and decoder allocation limit 64 MiB.
- Positive grid dimensions with at most 64 tiles; every tile needs positive area.
- Between 2 and 32 distinct non-empty labels, ignoring outer whitespace and case.
- Labels become `a photo of {label}` captions. Captions over CLIP's 77-token
  context, including start/end tokens, are rejected without silent truncation.
- Tiles use aspect-preserving bicubic resizing of the shortest side to 224,
  followed by a center crop, RGB rescaling and the pinned CLIP channel normalization.
  Extremely thin tiles requiring a resized edge over 8192 pixels are rejected.
- CPU inference runs in batches of at most four tiles. It is synchronous.

This is a local image classifier. It supplies no browser interaction, challenge
state handling or reCAPTCHA verification tokens.

## CLI and real-model verification

```sh
cargo run --release --features image-selection --example image_selection -- models/clip-vit-b32 local-grid.png 3 3 buses cars bicycles buildings "empty road"
cargo test --all-features --release --locked --test image_selection_smoke -- --ignored
```

The model test uses the downloaded reference photos, builds a known 3×3 montage,
and checks positive/negative classification, indices across batch boundaries,
file/byte equivalence, reversed labels, and invalid input. Set `RS_OCR_CLIP_DIR`
to use another directory containing the same model, tokenizer and reference photos.
Ordinary unit tests do not need weights or network access.

## Sources and reference-image provenance

- [CLIP research and original implementation](https://github.com/openai/CLIP)
- [Selected ONNX export and license](https://huggingface.co/Xenova/clip-vit-base-patch32)
- [`RTen` CLIP example](https://github.com/robertknight/rten/blob/v0.27.0/rten-examples/src/clip.rs)
- [Model and input checksums](../model-manifests/clip-vit-b32.json)
- [`RTen` reference-photo notes](https://github.com/robertknight/rten/blob/v0.27.0/tools/test-images/README.md):
  `horses.jpeg` is from the author's photo library; `sofa-cats.jpg` comes from
  COCO validation image `000000039769.jpg`. The photos are downloaded separately
  for local testing and are not redistributed in this crate.
