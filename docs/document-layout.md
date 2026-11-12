# Optional PP-DocLayout-S CPU companion

Enable `document-layout` to add `rs_ocr::documents::layout`. Tiny remains the page
text reader, running through `RustO` 0.3.2 / `RTen` 0.27.0. This feature adds an
independent PP-DocLayout-S region detector and explicit region association; it
does not enable a handwriting model, table-structure decoder or automatic
financial-field understanding.

## API

```rust,no_run
use rs_ocr::{
    documents::layout::{LayoutDetector, LayoutOptions, LayoutReader},
    documents::{DocumentReader, ExtractionOptions},
};

let model = "models/pp-doclayout-s/pp_doclayout_s.onnx";
let detector = LayoutDetector::new(model)?;
let layout = detector.detect_file("invoice.jpg", LayoutOptions::default())?;
for region in &layout.regions {
    println!("{} {:?} score={}", region.class.name(), region.bounds, region.confidence);
}

let tiny = DocumentReader::from_model_dir("models/ppocrv6-tiny")?;
let mut reader = LayoutReader::new(model, tiny)?;
let page = reader.read_file("invoice.jpg", LayoutOptions::default())?;
let bill = page.extract_bill(ExtractionOptions::default())?;
println!("total: {:?}, issues: {:?}", bill.total.unique(), bill.issues);
let cheque = page.extract_cheque(ExtractionOptions::default())?;
println!("payee: {:?}", cheque.payee.unique());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Both readers also accept image bytes. Sessions are synchronous and reused;
async applications should run them on a blocking worker. `LayoutDocument::new`
can associate an existing validated `Document` with a `PageLayout` without running
OCR again. Its read-only accessors expose the original document, regions and
source-span memberships.

A span belongs to a region when at least 80% of its axis-aligned area lies inside
it. Overlapping memberships are retained. Original OCR text, order, coordinates
and scores are unchanged. Image/chart/seal regions remain visible in layout
results but are not treated as field text. All 23 classes follow the official
label order; `LayoutClass::new` rejects unknown codes.

`extract_bill` and `extract_cheque` first use the original full-page extractors.
Only fields with **no original candidates** may receive candidates from explicit
labels inside a model region. Original conflicts are not resolved. Region-local
header/date/currency guesses cannot backfill fields. Conflicting region values
stay ambiguous; duplicate evidence does not increase confidence. A malformed
final payable total cannot be replaced by an intermediate regional total.
Every added field emits `ReviewIssue::InferredField`, retains original page span
indices and uses the weakest OCR score, not the region score. Original invalid
amount evidence remains reviewable; arithmetic checks run after backfilling.

## Model provenance and contract

The prepared FP32 graph is **4,917,812 bytes**. Source files and preparation
checksums live in `model-manifests/doc-layout-model.json`; weights stay ignored
under `models/pp-doclayout-s`.

The model starts from the community
[ONNX export](https://huggingface.co/stefanj0/PP-DocLayout-S-ONNX/tree/7e851b9e5ad66613d7318a3213ab3336ebe0a514)
of [Paddle's model](https://huggingface.co/PaddlePaddle/PP-DocLayout-S/tree/8ac289e66575bb9bba6e15c53719d8b15cc9b3b2),
under Apache-2.0. Its embedded NMS used an intersection-over-union threshold of
0.6 and a score threshold of 0.025, differing from the official `inference.yml`.
The deterministic preparation script changes only those two constants to 0.5
and 0.3, respectively; initializer weights are checked unchanged. The original
source graph remains preserved as `source.onnx`. We have not independently
compared this export with Paddle native inference.

The adapter follows official 480×480 resize **without keeping aspect ratio**,
RGB `ImageNet` normalization, and `scale_factor=[480/original_height,
480/original_width]`. Official `interp: 2` means cubic interpolation; the pure-Rust
implementation uses the `OpenCV` cubic coefficient -0.75. It samples source pixels
without expanding an entire grayscale scan to RGB. `OpenCV` integer rounding may
differ by one RGB unit. The graph outputs original-image coordinates; the adapter
does not scale them twice. Clipped extents and raw model boxes are both retained.

Inputs are PNG/JPEG, at most 32 MiB encoded, 8192 pixels per edge and 128 MiB
decoder allocation. These are decoder limits, not a claim about total process
RAM. Counts, class IDs, scores, tensor ranks and finite geometry are validated.
The graph performs class-aware NMS and keeps at most 100 regions. Post-filter
options accept score 0.3..=1 and a result limit 1..=100; default score is 0.5.

## Setup and independent reference

Rust inference has no Python service or ONNX Runtime dependency. An isolated
Python environment is used only to prepare the community export and independently
check CPU predictions:

```sh
python -m venv models/pp-doclayout-s/reference-env
models/pp-doclayout-s/reference-env/bin/python -m pip install --no-cache-dir -r scripts/doc-layout-reference-requirements.txt
python scripts/doc_layout_study.py download-model
python scripts/doc_layout_study.py download-data
python scripts/doc_layout_study.py verify-model
cargo run --features document-layout --release --locked --example doc_layout_batch -- models/pp-doclayout-s/pp_doclayout_s.onnx models/ppocrv6-tiny
```

The batch example reads JSONL jobs with `id`, `kind` (`bill` or `cheque`), `image`
and an explicit decimal separator. It records both extraction versions from
identical page spans, regions, memberships and separate layout/OCR times.
Supplying `spans` reuses cached text for development; fresh test runs use actual
Tiny OCR. The batch CLI expects JSONL on stdin.

Parity covers four public images: three positive region cases, one empty
negative control, and portrait plus landscape pages. Independent `OpenCV`
preprocessing and ONNX Runtime 1.23.2 CPU outputs must match native Rust within
0.002 score and 0.5 original-image pixels, with identical region counts/classes.
This is compatibility evidence for that model/version/domain, not equivalence
across all inputs. `docs/doc-layout-parity.json` records measured errors and
reference versions. Unit tests also compare cubic resize pixels with independent
`OpenCV` fixtures.

## Extraction evaluation and limits

The new study retires all 58 previously observed layout-study documents to
development, using cached original Tiny spans. Eight fresh documents have source
IDs and image hashes absent from all earlier studies: four SROIE receipts, two
Zenodo invoices with unused issuers, and two public-domain Wikimedia archival
cheques. Exact published labels remain unchanged. Legible cheque labels were
manually transcribed before inference by the implementer, not a blinded annotator;
uncertain fields were omitted. Source URLs, revisions, licenses/attribution and
checksums are in `model-manifests/doc-layout-samples.json`. Images and raw OCR
remain ignored; public availability is not a blanket redistribution license.

The model, class map, thresholds, sources, labels and development evidence are
frozen before fresh test inference. No tuning or promotion uses that test set.
Original financial/model/layout reports are unchanged and verify against their
original frozen source bytes. Historical verification needs those local caches;
downloading images alone does not restore original predictions.

Machine-readable extraction results are in `docs/doc-layout-results.json`.
Model regions are generic categories; a table box is not table-cell structure,
and a text box does not identify a total or payee. This optional model must earn
any accuracy claim on independent data. It cannot recover characters Tiny failed
to recognize.


## Measured outcome

| Split/domain | Documents | Baseline correct fields | Assisted correct fields |
|---|---:|---:|---:|
| Development receipts | 36 | 82/107 | 82/107 |
| Development invoices | 12 | 21/58 | 21/58 |
| Development cheques | 10 | 9/35 | 9/35 |
| Fresh receipts | 4 | 6/12 | 6/12 |
| Fresh invoices | 2 | 4/10 | 4/10 |
| Fresh cheques | 2 | 0/6 | 0/6 |

Development has 39 regions on 29/58 pages; the fresh set has three regions on
3/8 pages. Neither split receives a field backfill. All 66 document workflows
complete without errors. Fresh complete-document extraction is 1/8 for both
versions. Development counts include six blank controls, five correctly omitted;
the fresh six cheque fields are populated labels, not blank-field credit.

The additional layout step has an observed median of **115.5 ms** on development
and **150.3 ms** on fresh images on this Intel Mac. These single-run observations
include file reads, decoding, preprocessing and detection; they exclude initial
layout-model loading. They are not a controlled latency benchmark or a guarantee
for other CPUs. Development OCR is cached, while all eight fresh documents run
actual Tiny OCR. Detector/recognizer weights and decimal settings are unchanged.

On the four reference images, maximum observed differences are **0.000698** in
score and **0.104 original-image pixels** in coordinates. Both are within the
declared 0.002 / 0.5 tolerances; empty/nonempty predictions and class counts
agree. These measurements prove bounded compatibility on the tested domain,
not improved financial recognition or all-input numerical equivalence.

**Keep the companion optional.** Its region API is usable, but this experiment
does not justify enabling the extra CPU work for default bill/cheque extraction.
Generic text/image/table classes do not identify merchant, payee or total fields,
and this model does not improve handwritten character recognition. No thresholds
or parser rules were retuned from the fresh results.

Verification:

```sh
cargo test --all-features --locked
python scripts/doc_layout_study.py verify-model
cargo test --all-features --release --locked -- --include-ignored
python scripts/doc_layout_study.py verify-report
python -m unittest discover -s scripts -p 'test_*.py'
python scripts/doc_layout_study.py verify-code
```

Local original prediction caches remain necessary for report verification.
Model/data downloads restore inputs; they do not reconstruct historical evidence.
The runner refuses to overwrite test results or refreeze after test evaluation.
