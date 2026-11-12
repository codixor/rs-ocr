use rs_ocr::{bills::BillReader, cheques::ChequeReader, documents::ExtractionOptions};
use std::{env, fs, path::PathBuf};

#[test]
#[ignore = "requires local PP-OCRv6 models; uses synthetic printed financial documents"]
fn reads_printed_invoice_and_cheque_through_real_ocr_without_amount_rounding() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = env::var_os("RS_OCR_MODEL_DIR")
        .map_or_else(|| root.join("models/ppocrv6-tiny"), PathBuf::from);
    let mut bills = BillReader::from_model_dir(&models).unwrap();
    let invoice = root.join("tests/fixtures/invoice.png");
    let result = bills
        .read_file(&invoice, ExtractionOptions::default())
        .unwrap();
    assert_eq!(
        result.total.unique().map(|field| field.value.minor_units),
        Some(11_550),
        "{result:?}"
    );
    assert_eq!(
        result
            .reference_number
            .unique()
            .map(|field| field.value.as_str()),
        Some("0007"),
        "{result:?}"
    );
    assert_eq!(
        result,
        bills
            .read_bytes(&fs::read(invoice).unwrap(), ExtractionOptions::default())
            .unwrap()
    );
    let mut cheques = ChequeReader::from_model_dir(models).unwrap();
    let result = cheques
        .read_file(
            root.join("tests/fixtures/cheque.png"),
            ExtractionOptions::default(),
        )
        .unwrap();
    assert_eq!(
        result.amount.unique().map(|field| field.value.minor_units),
        Some(123_456),
        "{result:?}"
    );
    assert_eq!(
        result
            .cheque_number
            .unique()
            .map(|field| field.value.as_str()),
        Some("000042"),
        "{result:?}"
    );
    assert!(
        bills
            .read_bytes(b"not an image", ExtractionOptions::default())
            .is_err()
    );
    assert!(
        cheques
            .read_bytes(&[], ExtractionOptions::default())
            .is_err()
    );
}
