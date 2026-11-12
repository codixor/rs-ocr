# Public cheque, invoice and receipt study

This study evaluates **60 public documents**, with **30 development and 30 held-out
documents**. Inference uses local `RustO` 0.3.2 / PP-OCRv6 Tiny in Rust. Python only
downloads data, scores predictions and checks evidence; it provides no inference
service. No model weights were trained or changed.

## Sources and protocol

- **40 real Malaysian receipts:** [SROIE](https://arxiv.org/abs/2103.10213), from
  [the corrected public mirror](https://github.com/zzzDavid/ICDAR-2019-SROIE/tree/27be4271b251c256f695acbade9a801bffe85994/data).
  A deterministic hash selects 40 of 626 IDs. Exact company groups are assigned
  to disjoint development/test sets of 20 each. These are local study splits of
  the mirror, **not the official SROIE test leaderboard**. Repository software
  licensing does not establish a separate dataset redistribution license; images
  and original annotations stay local and ignored.
- **12 real invoices/receipts:** Francisco Cruz and Mauro Castelli's
  [author-published dataset](https://zenodo.org/records/7213544), DOI
  `10.5281/zenodo.7213544`, CC BY 4.0. A deterministic hash selects 12 annotation
  IDs, six per split. Issuer groups do not overlap. This source contains
  Portuguese documents, challenging the initially English-label parser.
- **8 public cheque scans/facsimiles:** individual Wikimedia source URLs,
  attribution, image timestamps and public-domain statements are in the
  [sample manifest](../model-manifests/financial-samples.json). Four per split;
  one held-out form is blank. Most are historical and handwritten. This is a
  small purposive diagnostic set, not representative of modern banking.

Cheque annotations were manually transcribed from legible image fields and
publisher descriptions **before extraction changes**. Uncertain fields are
unscored; `null` explicitly denotes a visibly blank or redacted field. The same
author annotated the cheque images and improved the parser, so this is not a
blinded experiment. The historical cheque captions can provide context beyond
what an OCR model can read.

The initial image-to-field baseline was saved before modifying the parser.
Development failures guided changes. Parser source hashes were then frozen in
the [profile](../model-manifests/financial-profile.json) before evaluating the
held-out split. Both versions use **identical cached OCR spans**, including
confidence and geometry. Every file has a SHA-256 checksum. Duplicate image
content and cross-split issuer groups are rejected.

Field accuracy requires a unique value. Missing, ambiguous and wrong values fail.
Text comparison ignores case and whitespace only; punctuation, spelling and OCR
character errors still matter. Money compares exact integer cents. Date scoring
normalizes equivalent printed forms using declared dataset date order and a
documented two-digit-year pivot of 70; the Rust API retains raw date strings.
Blank controls score correct only when the parser abstains. Coverage and wrong
values are reported alongside accuracy in the JSON report.

Published bill labels remain unchanged. A development receipt has an empty total
annotation and is unscored for that field. One development invoice's published
total is **10.49**, while its printed discounted payable amount is **8.39**; this
discrepancy is retained. Portuguese jobs use preset comma-decimal conventions in
both versions; one development invoice prints decimal dots and remains outside
that preset. Neither labels nor caller locale were retroactively repaired to
raise accuracy.

## Held-out results

Counts are exact field matches. Negative controls are listed separately for
cheque amounts/dates/payees below.

| Domain and field | Baseline | Improved |
|---|---:|---:|
| Receipt merchant | 0/20 | 9/20 |
| Receipt date | 0/20 | 13/20 |
| Receipt total | 4/20 | 6/20 |
| **Receipt fields combined** | **4/60 (6.7%)** | **28/60 (46.7%)** |
| Invoice merchant | 0/6 | 3/6 |
| Invoice date | 0/6 | 5/6 |
| Invoice reference | 0/6 | 0/6 |
| Invoice tax | 0/6 | 0/6 |
| Invoice total | 1/6 | 1/6 |
| **Invoice fields combined** | **1/30 (3.3%)** | **9/30 (30.0%)** |
| Cheque number | 0/3 | 1/3 |
| Cheque bank | 0/3 | 0/3 |
| Cheque amount, populated fields | 0/3 | 0/3 |
| Cheque date, populated fields | 0/1 | 0/1 |
| Cheque payee, populated fields | 0/1 | 0/1 |
| Cheque blank controls: amount/date/payee | 3/3 | 3/3 |

Whole-document exact matches: receipts **0/20 → 2/20**, invoices **0/6 → 0/6**,
cheques **0/4 → 0/4**. Every selected document produced OCR output; there were no
image decoding/model execution failures. This does not imply correct extraction.

Development combined field accuracy: receipts **6/59 → 48/59**, invoices
**3/28 → 9/28**, cheques **1/13 → 2/13** (the cheque count includes one redacted
payee control). These development results are not held-out performance.

## What improved and what remains

Changes add reviewable header and upper-right cheque-number fallbacks, separate
printed dates from adjacent timestamps, recognize additional invoice/receipt
labels, handle full-width colons, support `RM`/`MYR` amount prefixes and preserve
conflicting values instead of choosing whichever scores highest. Exact decimal
parsing still rejects OCR substitutions such as `O` for `0`.

The report also checks whether target text is present anywhere in concatenated
OCR output, ignoring segmentation. For money this diagnostic requires a literal
two-decimal token using the declared convention. This is an availability hint,
not an OCR accuracy score or proof that the value belongs to the correct field.

- All **20/20 receipt totals** are present as literal values somewhere in OCR,
  but only **6/20** are uniquely extracted. Qualified labels, column grouping
  and competing totals remain substantial parser gaps.
- Exact merchant text occurs in **14/20** receipt OCR outputs and is extracted
  in **9/20**. Logos, multi-line legal names and alternative shop/legal names
  remain difficult.
- Invoice total text occurs in **4/6** OCR outputs, but extraction matches
  **1/6**. Reference numbers and tax tables remain unsupported on these layouts.
- Only **1/3 populated cheque amounts** appears as a literal decimal in OCR;
  that printed amount is attached to payee text and lacks a recognized currency
  marker, so the conservative extractor abstains. The other cheque amounts are
  not recognized reliably. A handwriting-capable recognizer and explicit field
  localization require another evaluation with **fresh held-out samples**.

The sample is too small and selective to estimate production accuracy. High
recognition confidence can coexist with wrong text. Heuristic fields require
review, and these results establish neither MICR correctness nor authenticity.

## Running the study

```sh
python scripts/financial_benchmark.py download
python scripts/financial_benchmark.py verify-data
python scripts/financial_benchmark.py verify-report
cargo test --release --locked --test financial_public_smoke -- --ignored
```

The local ignored directory `models/financial-public` contains the downloaded
images, original annotations and baseline/improved JSON-lines evidence. The
download command restores pinned **source data**, not archived predictions.
Report verification and the public smoke test require the original local
prediction cache from this study. A fresh checkout can evaluate the current API
using the `financial_batch` example, but cannot regenerate the historical
baseline from current parser source alone. Raw financial OCR outputs are not
published with this aggregate report.

`financial_batch` accepts JSON lines with `id`, `kind` (`bill` or `cheque`),
`image`, and optional `decimal_separator` (`dot` or `comma`). It emits spans,
field candidates and review issues. Supplying `spans` instead of `image` replays
existing OCR through the public extractors. `run-dev` and `run-test` replay the
local baseline cache with the frozen implementation; `verify-report` independently
recomputes every metric from that cache and checks candidate source indices,
weakest contributing confidence, identical OCR spans, data/model/source hashes
and prediction checksums. Refreezing after a held-out result exists is rejected.

Machine-readable evidence: [aggregate results](financial-results.json),
[source manifest](../model-manifests/financial-samples.json),
[frozen implementation](../model-manifests/financial-profile.json).
