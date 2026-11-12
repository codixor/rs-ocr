# Tiny financial layout study

Tiny FP32 remains the default pack, using `RustO` 0.3.2 and `RTen` 0.27.0. This
pass improves field association and adds caller-selected single-line region OCR.
**It did not improve aggregate extraction on the fresh test set. Handwriting
remains unreliable.** No model weights, dictionary, runtime or decimal convention
were changed. Python downloads, checks and scores evidence; inference and field
extraction run in Rust.

## Protocol and provenance

The 48 development images are the prior model study's 30 development and 18
previously observed test images. That old test split is explicitly retired to
development here. Ten test documents are new: six SROIE receipts, two Zenodo
invoices, and two Wikimedia archival cheque forms/scans. Test IDs and image
hashes do not occur in either earlier study; issuer groups do not overlap this
development split. The cheque groups represent individual archival forms, not a
representative bank population. Image and label manifests include SHA-256 hashes,
source URLs or archive members and attribution. A fixed hashed ordering selected
unused receipt/invoice labels before inference, rejecting development issuers.

Receipts use the [SROIE mirror](https://github.com/zzzDavid/ICDAR-2019-SROIE) at
revision `27be4271b251c256f695acbade9a801bffe85994`; invoices use the authors'
[190-document dataset](https://zenodo.org/records/7213544). New cheque sources are
the [Bank of Reubens blank form](https://commons.wikimedia.org/wiki/File:Bank_of_Reubens_Check,_circa_1920s_-_Reubens,_Idaho_(51077339463).jpg)
(CC BY 2.0, attribution in manifest) and an
[1881 Columbus Savings Bank cheque](https://commons.wikimedia.org/wiki/File:1881_Columbus_Savings_Bank_check_-_DPLA_-_4dd0c4680df1fad28d77c5aa02775ea7.jpg)
(public domain). Public availability is not a blanket redistribution license;
source images, annotations and raw OCR stay in ignored `models/layout-public`.

Published invoice/receipt labels remain unchanged, including known mismatches
between pre-discount totals and printed payable amounts. Cheque fields and two
region rectangles were manually annotated before inference; uncertain fields
were omitted and only visibly blank fields were labeled null. The annotator is
the implementer, not blinded. Numeric dates use declared dataset order; text is
case/whitespace normalized only, with no fuzzy spelling credit. Amounts compare
exact minor units. Blank controls are reported separately from populated fields.

The baseline uses the previous frozen parser and identical Tiny OCR spans.
After development-only changes, source/model/manifest/evidence hashes were
frozen **before inspecting test predictions**. The changed parser then consumed
the same spans, isolating extraction from recognition. The old reports were not
rewritten. Their original source files are retained in a checksummed local
`models/financial-public/frozen-sources/` archive; historical report verification
may use these original bytes. Executing those old studies with new extraction
code is rejected. Reproducing a historical verification requires its original
local evidence and frozen sources, not only downloading images.

## Results

Correct fields / scored fields, including blank controls where shown:

| Split/domain | Documents | Baseline | New | Whole documents before → after |
|---|---:|---:|---:|---:|
| Development receipts | 30 | 67/89 | 72/89 | 12 → 15 |
| Development invoices | 10 | 15/48 | 17/48 | 0 → 0 |
| Development cheques | 8 | 4/27 | 5/27 | 0 → 0 |
| Fresh receipts | 6 | 10/18 | 10/18 | 2 → 2 |
| Fresh invoices | 2 | 4/10 | 4/10 | 0 → 0 |
| Fresh cheques | 2 | 4/8 | 4/8 | 0 → 0 |

Development gains are eight fields: five receipt totals, one invoice date, one
invoice total and one cheque amount. Development receipt totals improve from
19/29 to 24/29; cheque populated numeric amounts from 0/8 to 1/8. This is tuning
performance, not evidence of generalization. One published development total
still disagrees with the printed discounted payable amount; it remains wrong
under the original annotation rather than relabeling it for credit.

On fresh data, populated fields are **14/31 for both parsers**. Five blank cheque
controls contribute four correct omissions and one false positive for both.
Thus the combined result is **18/36 for both**. Fresh receipts have merchant
3/6, date 5/6 and total 2/6; fresh invoices have merchant/date 2/2 each, but
reference/tax/total 0/2 each. Neither populated cheque bank nor the populated
handwritten amount is correct (0/3 positive cheque fields). The cheque-number
hint can mistake a printed year for a number on a form with no number filled
in. These scans do not support a production cheque-accuracy claim.

All 58 baseline page OCR runs and both parser comparisons completed without
errors. The JSON report records every failed field by ID/status and fingerprints
of original evidence. Field candidates are independently checked against source
span indices and the weakest contributing recognition score.

## Region OCR and remaining work

`documents::RegionReader` uses the same Tiny recognizer directly, bypassing page
text detection for a supplied single-line location. It returns raw text/score and
the caller's original-coordinate rectangle. It never replaces a page field or
selects between conflicting amounts. The API bounds image decoding, encoded
reads, region arithmetic and aspect ratio, and reuses its recognizer session.

Two rectangles were frozen before region inference: a printed protected amount
on a development cheque and the handwritten numeric amount on the fresh archival
cheque. Both requests completed, but **neither crop contained the correct literal
amount in OCR**. The printed development amount was already present in full-page
OCR, while the handwritten amount was absent in both paths. Manual region
locations are oracle hints, not automatic localization. Cropping can remove
context or hurt Tiny; there is no evidence to enable automatic crop retries.
A native synthetic positive control correctly reads `AMOUNT: 1234.56`, checks
file/byte equality, retained confidence/coordinates, invalid inputs and session
reuse after a failure.

Next accuracy work needs an independently labeled validation set covering
unseen receipt layouts, invoice table tax/reference association, and handwriting
with separate character-level transcription. General column/table reasoning,
multilingual labels, MICR, signatures and authenticity remain unsupported.

## Verification

```sh
python scripts/layout_benchmark.py download-data
python scripts/layout_benchmark.py verify-data
python scripts/layout_benchmark.py verify-report
python scripts/layout_benchmark.py verify-history
python -m unittest discover -s scripts -p 'test_*.py'
cargo test --all-features --release --locked -- --include-ignored
python scripts/layout_benchmark.py verify-code
```

Reports and manifests are tracked source artifacts. Raw prediction caches are
local study evidence, required for report verification; downloads fetch only
public images/annotations. `run-test`, `regions` and `freeze` refuse to overwrite
or silently refreeze existing test evidence. New parser/model tuning requires a
new study and a new held-out split. See `layout-results.json`,
`model-manifests/layout-samples.json` and `model-manifests/layout-profile.json`.
