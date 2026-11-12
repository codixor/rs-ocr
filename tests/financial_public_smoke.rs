//! Repeat actual image OCR on one held-out document per domain, then compare
//! public API extraction with the measured run. Downloaded data stays ignored.
use rs_ocr::{
    bills, cheques,
    documents::{DecimalSeparator, DocumentReader, ExtractionOptions},
};
use serde_json::Value;
use std::{error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires pinned public financial samples, local benchmark outputs and PP-OCRv6 models"]
fn fresh_public_images_reproduce_cached_fields() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest: Value = serde_json::from_str(&fs::read_to_string(
        root.join("model-manifests/layout-samples.json"),
    )?)?;
    let evidence: Vec<Value> =
        fs::read_to_string(root.join("models/layout-public/improved-test.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
    let mut reader = DocumentReader::from_model_dir(root.join("models/ppocrv6-tiny"))?;
    for domain in ["receipt", "invoice", "cheque"] {
        let sample = manifest["samples"]
            .as_array()
            .ok_or("missing samples")?
            .iter()
            .find(|s| s["domain"] == domain && s["split"] == "test")
            .ok_or("missing domain")?;
        let cached = evidence
            .iter()
            .find(|v| v["id"] == sample["id"])
            .ok_or("missing evidence")?;
        let image = root
            .join("models/layout-public")
            .join(sample["image"].as_str().ok_or("missing image")?);
        let doc = reader.read_file(image)?;
        let cached_spans = cached["spans"].as_array().ok_or("missing spans")?;
        assert_eq!(doc.spans().len(), cached_spans.len());
        for (actual, expected) in doc.spans().iter().zip(cached_spans) {
            assert_eq!(
                actual.text,
                expected["text"].as_str().ok_or("missing text")?
            );
            let score = expected["confidence"]
                .as_f64()
                .ok_or("missing confidence")?;
            assert!((f64::from(actual.confidence.ok_or("missing score")?) - score).abs() < 0.0001);
        }
        let mut options = ExtractionOptions::default();
        if sample["decimal_separator"] == "comma" {
            options.amount_format.decimal_separator = DecimalSeparator::Comma;
        }
        if domain == "cheque" {
            let report = cheques::extract(doc, options)?;
            assert_eq!(
                report.amount.unique().map(|c| c.value.minor_units),
                cached["fields"]["amount"]["value"].as_i64()
            );
            assert_eq!(
                report.cheque_number.unique().map(|c| c.value.as_str()),
                cached["fields"]["cheque_number"]["value"].as_str()
            );
        } else {
            let report = bills::extract(doc, options)?;
            assert_eq!(
                report.total.unique().map(|c| c.value.minor_units),
                cached["fields"]["total"]["value"].as_i64()
            );
            assert_eq!(
                report.merchant.unique().map(|c| c.value.as_str()),
                cached["fields"]["merchant"]["value"].as_str()
            );
            assert_eq!(
                report.date.unique().map(|c| c.value.as_str()),
                cached["fields"]["date"]["value"].as_str()
            );
        }
    }
    Ok(())
}
