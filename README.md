# rs-ocr

Local OCR for Rust applications. It reads text CAPTCHAs, printed cheques,
invoices and receipts, and Arabic, French and English documents. Inference
runs on [`RustO`](https://github.com/byrizki/rusto-rs) 0.3.2 and its pure-Rust
`RTen` engine. An optional Burn backend is also available. You don't need
Python, `OpenCV`, a C++ OCR runtime or a remote service, and loading a model
never downloads anything.

## Modules

| Module | Feature | Purpose |
| --- | --- | --- |
| `captch` | default | Decode text CAPTCHAs with alphabet and length checks |
| `documents` | default | `DocumentReader`: page OCR with spans, boxes and scores |
| `cheques` | default | Printed-cheque fields with exact decimal amounts |
| `bills` | default | Invoice and receipt fields with retained evidence |
| `bidi` | default | Visual-to-logical reordering for Arabic lines |
| `documents::layout` | `document-layout` | PP-DocLayout-S region detection (23 classes) |
| `images::selection` | `image-selection` | CLIP image-grid tile classifier |
| `detection` | `object-detection` | `ObjectDetector` and `HybridSelector` for photo grids |
| `engines::burn` | `burn` | PP-OCRv6 Tiny on Burn's CPU backend (`burn-vulkan` adds a GPU device) |

## Further reading

- [Financial document APIs](docs/financial-documents.md) and the
  [original financial study](docs/financial-benchmark.md)
- [OCR model comparison](docs/ocr-model-study.md)
- [Layout study](docs/layout-benchmark.md) and
  [document layout API, model preparation and evaluation](docs/document-layout.md)
- [Image selection setup and API](docs/image-selection.md),
  [public-sample benchmark](docs/public-selection-benchmark.md) and
  [4×4 localization evaluation](docs/localization-benchmark.md)

## Install

The crate is not on crates.io yet, so add it as a git dependency:

```toml
[dependencies]
rs-ocr = { git = "https://github.com/codixor/rs-ocr" }
# optional modules:
# rs-ocr = { git = "https://github.com/codixor/rs-ocr", features = ["document-layout", "image-selection"] }
```

It needs Rust 1.99.1 or newer.

## Model setup

Model weights are kept out of the crate. For CAPTCHAs and Latin-script
documents, download the versioned upstream
[PP-OCRv6 Tiny archive](https://github.com/byrizki/rusto-rs/releases/download/models-rten-v0.3.2/RustO-Models-PPOCRv6-Tiny.zip)
and put `det.rten`, `rec.rten` and `dict.txt` together in one directory:

```sh
mkdir -p models/ppocrv6-tiny
curl --fail --location --output ppocrv6-tiny.zip \
  https://github.com/byrizki/rusto-rs/releases/download/models-rten-v0.3.2/RustO-Models-PPOCRv6-Tiny.zip
unzip -j ppocrv6-tiny.zip -d models/ppocrv6-tiny
```

Git ignores the `models/` directory. Another model family needs its matching
`RustO` preset and dictionary. The Tiny pack does not recognize Arabic.

### Arabic pack (PP-OCRv6)

For Arabic and mixed Arabic/Latin pages, use `models/ppocrv6-arabic-small`.
It combines the Tiny `det.rten` with the community Arabic recognizer
[`medyas/arabic_PP-OCRv6_small_rec`](https://huggingface.co/medyas/arabic_PP-OCRv6_small_rec)
(Apache-2.0), converted with `paddle2onnx` and `rten-convert`. It uses the
749-entry `ppocrv5_arabic_dict.txt` dictionary, saved as `dict.txt`. Load it
with `DocumentReader::from_model_dir`, which applies the v6 preset.

```sh
mkdir -p models/ppocrv6-arabic-small
cp models/ppocrv6-tiny/det.rten models/ppocrv6-arabic-small/
# rec.rten: convert inference.json + inference.pdiparams from the Hugging Face
# repository with paddle2onnx (opset 14), then: rten-convert rec.onnx rec.rten
```

The recognizer emits Arabic in visual (left-to-right image) order. Every reader
passes span text through `rs_ocr::bidi::visual_to_logical`, which restores
reading order and keeps numbers, e-mail addresses and Latin words
left-to-right. Lines without Arabic letters are left unchanged.

Character error rate on rendered test pages (`tests/arabic_smoke.rs`):

| Page | v6 Tiny | v5 Arabic | v6 Arabic |
| --- | --- | --- | --- |
| `arabic-invoice` | 83.6% | 25.1% | 0.0% |
| `arabic-notice` | 82.2% | 17.3% | 1.4% |
| `mixed-invoice` (EN/FR/AR) | 15.0% | 1.2% | 2.3% |

Keep the Tiny pack for CAPTCHAs and decimal amounts. The Arabic pack lowercases
some Latin capitals (`A7B9C2` becomes `A7B9c2`) and reads decimal points as
commas (`115.50` becomes `115,50`).

## Library usage

```rust,no_run
use rs_ocr::captch::{CaptchaDecoder, DecodeOptions};

let mut decoder = CaptchaDecoder::from_model_dir("models/ppocrv6-tiny")?;
let options = DecodeOptions {
    allowed_chars: Some("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".into()),
    expected_length: Some(6),
    min_confidence: 0.5,
};

let result = decoder.decode_file("captcha.png", &options)?;
println!("{} ({:.3})", result.text, result.confidence);

// Reuse the same model sessions for later images, including encoded bytes.
let bytes = std::fs::read("another-captcha.png")?;
let another = decoder.decode_bytes(&bytes, &options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For explicit model paths or another family:

```rust,no_run
use rs_ocr::{InitializeConfig, captch::CaptchaDecoder};

let config = InitializeConfig::ppv6("det.rten", "rec.rten", "dict.txt");
let mut decoder = CaptchaDecoder::new(config)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Inputs are encoded PNG or JPEG images. The decoder strips Unicode whitespace
from the candidate and keeps letter case. `raw_text` and the original segments
stay available with their bounding polygons and recognition scores.

Alphabet and length checks run after recognition. They reject unexpected
output; they never delete characters, change case, or choose between `O`/`0`
and `I`/`1`. Confidence is the lowest score among the segments that make up the
candidate. Scores are not calibrated probabilities of being correct.

Inference is synchronous and takes `&mut self`. In an async application, keep
the decoder on a worker thread and reuse it.

## Burn backend

The `burn` feature generates Burn code from the PP-OCRv6 Tiny ONNX models at
build time. Put `det.onnx` and `rec.onnx` in `models/onnx/ppocrv6-tiny/`, or
set `RS_OCR_ONNX_DIR` to their directory. `RTen` stays the default engine.

```sh
RS_OCR_ONNX_DIR=/path/to/onnx cargo test --release --features burn --test burn_parity -- --ignored
```

## Try a local image

```sh
cargo run --release --example captch -- models/ppocrv6-tiny tests/fixtures/plain.png 6
```

The example prints the candidate on stdout and its confidence on stderr. The
final length argument is optional, and the bundled synthetic fixture reads
`A7B9C2`. This is a clean-image baseline only; accuracy on distorted or noisy
CAPTCHAs has not been measured yet.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo clippy --features image-selection,object-detection,document-layout --all-targets --locked -- -D warnings
cargo test --locked
cargo deny check
# model tests (need the weights above):
cargo test --release --locked --test rusto_smoke -- --ignored
cargo test --release --locked --test arabic_smoke -- --ignored
```

Ordinary tests do not need weights. `rusto_smoke` loads `models/ppocrv6-tiny`
(override with `RS_OCR_MODEL_DIR`). `arabic_smoke` loads
`models/ppocrv6-arabic-small` (override with `RS_OCR_ARABIC_MODEL_DIR`).

## Licence

Licensed under either the [Apache License 2.0](LICENSE-APACHE) or the
[MIT licence](LICENSE-MIT), at your option. Model weights carry their own
upstream licences.
