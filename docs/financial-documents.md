# Printed cheques, invoices and receipts

`rs_ocr::cheques` and `rs_ocr::bills` provide initial structured document OCR.
They are available with default features and reuse `RustO` 0.3.2 / PP-OCRv6
through `documents::DocumentReader`. There is no Python inference service.

The extractors support **explicit English labels**, several Portuguese invoice
labels, and reviewable fallbacks for merchant/bank headers, printed dates,
standalone currency amounts and cheque numbers in the upper right. They preserve
identifiers and date order. The fallbacks are heuristics and emit
`ReviewIssue::InferredField`; recognition scores are not extraction probabilities.
Handwriting, signatures, country-specific MICR, complex/multi-column layouts and
invoice line-item tables remain incomplete. See the measured
[public-sample study](financial-benchmark.md) before assessing coverage.
The default Chinese/Latin OCR pack does not recognize Arabic; use the matching
regional `RustO` configuration when appropriate. Another recognition pack does not
automatically add field labels in its language.

## Available fields

| Module | Fields |
|---|---|
| `cheques` | payee, date, cheque/check number, bank, account number, numeric amount, amount words, raw labeled MICR |
| `bills` | merchant/vendor/seller/store, invoice/receipt number, date, explicit currency, subtotal, tax/VAT, total/amount due |

Each field retains **all candidates**, raw field text, original span indices and
the weakest contributing OCR score. `unique()` returns a value only if all parsed
candidates agree. Conflicting totals or payees return no unique value; they are
not resolved by picking the most confident candidate. Original document spans
retain unmodified text and coordinates. Known labels that overlap, such as
`Amount in words` and `Amount`, or `Total tax` and `Total`, are kept distinct.

Missing required fields, conflicting candidates, low confidence and invalid
amounts appear in `ReviewIssue`. Low-confidence candidates remain available.
Explicit merchant/bank labels take precedence over header fallbacks. Date
recognition separates common numeric or day-month-name forms from attached
timestamps, and accepts month-name-first dates, without choosing numeric day/month
order or a century. Other labeled date
text is preserved. Repeated differently formatted dates can remain ambiguous.
For bills, subtotal plus tax is checked against total using checked integer
arithmetic; discounts, shipping or multiple tax components may require review.
Cheques flag zero/negative numeric amounts. Amount-in-words agreement, bank
account correctness, MICR validity, signatures and authenticity are not verified.
An empty issue list means only that the configured checks passed; it does not
approve a document for payment.

## Rust API

```rust,no_run
use rs_ocr::{bills::BillReader, cheques::ChequeReader, documents::ExtractionOptions};

let options = ExtractionOptions::default();
let mut bills = BillReader::from_model_dir("models/ppocrv6-tiny")?;
let report = bills.read_file("invoice.png", options)?;
if let Some(total) = report.total.unique() {
    println!("{} minor units at scale {}", total.value.minor_units, total.value.fraction_digits);
}
println!("review issues: {:?}", report.issues);

let mut cheques = ChequeReader::from_model_dir("models/ppocrv6-tiny")?;
let bytes = std::fs::read("cheque.png")?;
let report = cheques.read_bytes(&bytes, options)?;
println!("{:?}", report.payee.unique().map(|candidate| &candidate.value));
# Ok::<(), Box<dyn std::error::Error>>(())
```

Both readers accept `InitializeConfig` through `new` for a matching model pack.
They are synchronous and reused with `&mut self`. Async applications can keep a
reader on a blocking worker. Encoded image reads are capped at 32 MiB plus the
overflow-detection byte. Documents accept at most 4,096 spans and 1 MiB of text,
and reject invalid confidence/geometry. Supported OCR inputs are PNG/JPEG images;
this does not add a PDF renderer.

To reuse another OCR engine or an existing transcription:

```rust
use rs_ocr::{bills, documents::{Document, ExtractionOptions}};
let document = Document::from_text(
    "Merchant: Example Store\nSubtotal: 110.00\nVAT: 5.50\nTotal: 115.50",
)?;
let bill = bills::extract(document, ExtractionOptions::default())?;
assert_eq!(bill.total.unique().unwrap().value.minor_units, 11_550);
assert!(bill.issues.is_empty());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Transcriptions have `None` confidence/geometry; no fake OCR scores are assigned.
`Document::new` accepts validated `TextSpan` records from another engine. Explicit
same-row label/value fragments are joined geometrically for simple layouts.
Separate labels on the same row bound their own values. A label may use one
immediate, nearby value on the next geometric row; several possible values or
missing geometry prevent this association. Overlapping text in the same column
is kept on separate rows. Wrapped merchant names can join up to three nearby,
aligned lines ending in a legal company suffix; these remain reviewable hints.

Bills prefer explicitly final/payable labels such as `Nett Total`, `Final Total`,
`Grand Total`, or `Total a pagar` over ordinary intermediate totals. Multiple
conflicting final values still return no unique total. This does not implement
discount reconciliation. Recognized `(RM)`-style currency qualifiers and cheque
protection stars/ellipsis are removed only for field amount parsing; original
candidate text remains available. A decimal point attached to digits is never
discarded to salvage a malformed amount such as `.1.40`.

## Caller-selected single-line regions

`RegionReader` reuses the existing Tiny recognizer directly through `RustO` and
`RTen`, bypassing page detection for a caller-specified field location:

```rust,no_run
use rs_ocr::documents::{ImageRegion, RegionReader};
let mut reader = RegionReader::from_model_dir("models/ppocrv6-tiny")?;
let region = ImageRegion { x: 50, y: 420, width: 700, height: 70 };
let evidence = reader.read_file("tests/fixtures/cheque.png", region)?;
println!("{:?}", evidence.spans());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Coordinates refer to the original image; the returned box is the **supplied
region extent**, not a model-detected text box. This reader preserves the raw
recognition score and never overwrites a page field. Callers must review and
reconcile conflicting results themselves. Inputs are limited to 32 MiB encoded,
8192 pixels per edge, 64 MiB decoder allocation, positive in-bounds rectangles,
and a maximum width/height ratio of 64 to bound recognition resizing. File and
byte APIs use the same path. Tiny remains a printed-text model: the new public
region diagnostics did **not** recover the annotated amounts. See the
[layout study](layout-benchmark.md) for the measured result and limits.

## Exact amounts and conventions

`Amount` stores signed `i64` minor units and an explicit fractional scale. It
never uses binary floating point for money, repairs `O` into `0`, rounds excess
digits, or infers a currency from `$`. Common surrounding currency tokens can be
parsed but remain in raw evidence; currency is a separate explicit field.

```rust
use rs_ocr::documents::{Amount, AmountFormat, DecimalSeparator};
let format = AmountFormat { decimal_separator: DecimalSeparator::Comma, fraction_digits: 2 };
assert_eq!(Amount::parse("1.234,56", format)?.minor_units, 123_456);
assert!(Amount::parse("1,23.45", AmountFormat::default()).is_err());
assert!(Amount::parse("1O.00", AmountFormat::default()).is_err());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Default format is decimal dot, optional three-digit comma grouping and two
fractional digits. Comma-decimal formatting uses dot grouping. Precision 0..=6
is caller-selected; ambiguous separators, malformed grouping and overflow are
errors. Day/month order, currency scale and country-specific banking formats
must be supplied/handled by the caller rather than inferred from a scan.

## CLI and verification

```sh
cargo run --release --example documents -- bill models/ppocrv6-tiny tests/fixtures/invoice.png
cargo run --release --example documents -- cheque models/ppocrv6-tiny tests/fixtures/cheque.png
cargo test --release --locked --test document_smoke -- --ignored
```

Synthetic printed fixtures contain fictional data and a cheque marked not valid
for payment. Real OCR tests verify invoice total **115.50**, reference **0007**,
cheque amount **1234.56**, number **000042**, file/byte equivalence, session reuse
and malformed-image handling. Unit tests cover locale/grouping/overflow,
conflicting values, missing/low-confidence evidence, label collisions, attached
timestamps, inference review flags and geometry. A separate ignored integration
test re-runs image OCR and field extraction on one held-out public receipt,
invoice and cheque. Public-study measurements and remaining gaps are recorded
in [financial-benchmark.md](financial-benchmark.md).
