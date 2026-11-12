# OCR model selection with RTen

This study holds **RTen 0.27.0**, **RustO 0.3.2**, preprocessing presets and the
existing financial field parser fixed while comparing different model weights.
Inference runs in Rust on the Intel Mac CPU; Python downloads artifacts and
calculates measurements. There is no runtime migration or Python OCR service.

## Candidates

All weights come from the pinned
[official RustO model release](https://github.com/byrizki/rusto-rs/releases/tag/models-rten-v0.3.2).
Archive and extracted-file SHA-256 checksums are recorded in the
[candidate manifest](../model-manifests/ocr-model-candidates.json).

| Candidate | Detector | Recognizer | Uncompressed weights and dictionary |
|---|---|---|---:|
| Tiny | PP-OCRv6 Tiny | PP-OCRv6 Tiny | 6,386,492 bytes |
| Small | PP-OCRv6 Small | PP-OCRv6 Small | 31,257,307 bytes |
| Medium | PP-OCRv6 Medium | PP-OCRv6 Medium | 138,857,755 bytes |
| Latin | PP-OCRv6 Small | PP-OCRv5 Latin | 17,933,250 bytes |

The Latin release contains only recognition weights and its dictionary. It is
paired explicitly with the Small detector and the shared `[3,48,320]` recognition
input convention, using the PPv6 detection preset. It is a measured combination,
not an official standalone full-pipeline Latin package. All candidates use FP32.

## Protocol and interpretation

The 30 prior **development** images are used for selection: 20 receipts, six
Portuguese invoices and four archival cheques. The previous held-out split and
published results remain untouched. A new split adds **18 fresh public images**:
ten SROIE receipts, four author-published Zenodo invoices and four archival cheque
scans. No fresh ID or image hash occurs in the prior study; issuer groups are
disjoint from development. The new cheque set includes US Treasury, Shasta Dam,
Commercial State Bank and a published typewritten cheque scan. Each image's
source URL, attribution, license and checksum is recorded in the
[sample manifest](../model-manifests/ocr-model-samples.json). Images stay local
and ignored. Cheque gold fields were manually transcribed before test inference;
the annotator and implementer are the same, so the study is not blinded.

Selection is defined before reading candidate scores: maximize exact positive
target-text availability in OCR, then correct extracted fields, then prefer the
smaller pack. A separate choice is made per document domain. The choice is
frozen before running any fresh test inference. Both Tiny and the selected pack
are evaluated on each domain. The test split does not select or retune models.

Target availability measures whether the annotated value can be recovered
literally from OCR, ignoring segmentation and case/whitespace. Date forms are
normalized using the declared dataset order. Monetary targets require a literal
decimal token in the declared convention. No spelling or digit repairs receive
credit. This diagnostic can find a matching value in another field; it measures
text recovery, not field understanding, calibrated confidence or full OCR CER.
Extraction accuracy separately requires a unique correct field value using the
existing financial scoring rules. Blank controls are excluded from positive
target recall and remain included in extraction accuracy.

Observed times include image OCR and field extraction, and the first document
also includes model loading. These measurements are hardware-specific and are
not a controlled performance benchmark. “Selected” means best among these four
compatible packs on this development set, not a universal best OCR model.

## Reproduction

```sh
python scripts/ocr_model_study.py download-models
python scripts/ocr_model_study.py download-data
python scripts/ocr_model_study.py verify-data
python scripts/ocr_model_study.py run --model medium --split dev
python scripts/ocr_model_study.py verify-report
```

Run each development candidate once before freezing a new study. The frozen
selection records the development prediction digests and the Rust batch source.
The runner refuses to refreeze after test evidence exists and refuses to
overwrite test predictions. Local raw evidence remains in the ignored directory
`models/ocr-model-study`. Source downloads restore the inputs, not the historical
prediction cache. Reproducing the archived baseline report from the previous
study still requires that study's local cache.

The existing APIs already accept compatible packs without changing RTen:

```rust,no_run
use rs_ocr::bills::BillReader;
// Supply the directory for the selected PPv6-compatible pack.
let mut reader = BillReader::from_model_dir("models/ocr-model-study/packs/medium")?;
let report = reader.read_file("invoice.jpg", Default::default())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Measured outcome and current choice

All four candidates completed **30 development images each** without inference
errors. The development-frozen selection was **Latin for receipts** and **Small
for invoices and cheques**. Latin tied Medium's development receipt recovery and
extraction counts, with a much smaller pack. Small recovered more invoice target
text and tied Medium's cheque target recovery with better extracted-field counts.

| Development domain | Tiny: targets / extracted fields | Small | Medium | Latin |
|---|---:|---:|---:|---:|
| Receipts | 56/59 / 48/59 | 57/59 / 49/59 | 57/59 / 51/59 | 57/59 / 51/59 |
| Invoices | 21/28 / 9/28 | 24/28 / 12/28 | 23/28 / 10/28 | 22/28 / 10/28 |
| Cheques | 4/12 / 2/13 | 6/12 / 3/13 | 6/12 / 2/13 | 5/12 / 3/13 |

The cheque development extraction denominator includes one redacted-field
control; positive target recovery excludes it.

| Fresh test domain | Tiny target recovery | Selected target recovery | Tiny extraction | Selected extraction |
|---|---:|---:|---:|---:|
| 10 receipts, Latin selected | 29/30 | 29/30 | 19/30 | 19/30 |
| 4 invoices, Small selected | 18/20 | 19/20 | 6/20 | 6/20 |
| 4 cheques, Small selected | 3/14 | 4/14 | 2/14 | 2/14 |
| **Combined** | **50/64 (78.1%)** | **52/64 (81.3%)** | **27/64 (42.2%)** | **27/64 (42.2%)** |

All 36 fresh image/model runs completed without errors (18 Tiny baseline,
10 Latin receipts, eight Small invoices/cheques). Fresh cheque numeric amount
extraction remains **0/4** for both Tiny and Small. The invoice total improved
from **0/4 to 1/4**, while date extraction fell from **4/4 to 3/4**. These offset,
so there is no aggregate field-accuracy gain. Original held-out documents were
not reused as fresh test images, and neither model selection nor field rules
were retuned from these test outcomes.

**Keep Tiny as the default.** The comparison supports Small as an optional
choice for somewhat better raw invoice/cheque text recovery, not as a proven
improvement to complete financial extraction. There is no fresh receipt gain
that justifies promoting Latin, and no general advantage demonstrated for Medium
that justifies its CPU cost here. This is a deployment judgment after validation;
the original development-frozen selection remains recorded without alteration.

Model upgrades alone have not closed the gaps. Further work needs a
handwriting-capable recognition/localization pipeline for cheques and better
field/layout interpretation for bills. Any subsequent model choice or parser
tuning needs another fresh held-out set. Freshness here means unseen by our local
study; upstream pretraining datasets were not audited for overlap.

Machine-readable records: [results](ocr-model-results.json),
[frozen selection](../model-manifests/ocr-model-selection.json),
[candidate sources](../model-manifests/ocr-model-candidates.json),
[samples and provenance](../model-manifests/ocr-model-samples.json).
