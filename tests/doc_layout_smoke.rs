//! Real PP-DocLayout-S CPU inference and preserved Tiny page evidence.
use rs_ocr::{
    documents::layout::{LayoutDetector, LayoutError, LayoutOptions, LayoutReader},
    documents::{DocumentReader, ExtractionOptions},
};
use serde_json::Value;
use std::{error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires pinned PP-DocLayout-S/Tiny models and public document study samples"]
fn real_layout_cpu_file_bytes_reference_and_tiny_companion_workflows() -> Result<(), Box<dyn Error>>
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = root.join("models/pp-doclayout-s/pp_doclayout_s.onnx");
    let image = root.join("models/doc-layout-public/invoice-20210509_001704.jpg");
    let bytes = fs::read(&image)?;
    let detector = LayoutDetector::new(&model)?;
    let layout = detector.detect_file(&image, LayoutOptions::default())?;
    assert_eq!(
        layout,
        detector.detect_bytes(&bytes, LayoutOptions::default())?
    );
    assert_eq!((layout.width, layout.height), (3000, 4000));
    assert_ne!(
        layout.regions,
        [] as [rs_ocr::documents::layout::LayoutRegion; 0]
    );
    let reference: Value = serde_json::from_str(&fs::read_to_string(
        root.join("models/pp-doclayout-s/reference.json"),
    )?)?;
    let expected = reference["predictions"]
        .as_array()
        .ok_or("missing reference")?
        .iter()
        .find(|r| r["id"] == "invoice-20210509_001704")
        .ok_or("missing image reference")?;
    assert_eq!(
        layout.regions.len(),
        expected["regions"]
            .as_array()
            .ok_or("missing regions")?
            .len()
    );
    for region in &layout.regions {
        let expected = expected["regions"]
            .as_array()
            .ok_or("missing regions")?
            .iter()
            .find(|r| r["class_id"].as_u64() == Some(u64::from(region.class.code())))
            .ok_or("missing class")?;
        assert!(
            (f64::from(region.confidence)
                - expected["confidence"].as_f64().ok_or("missing score")?)
            .abs()
                < 0.002
        );
        for (actual, expected) in region
            .raw_bounds
            .iter()
            .zip(expected["raw_bounds"].as_array().ok_or("missing bounds")?)
        {
            assert!((f64::from(*actual) - expected.as_f64().ok_or("bad bound")?).abs() < 0.5);
        }
    }
    assert_eq!(
        detector
            .detect_file(
                root.join("models/doc-layout-public/sroie-087.jpg"),
                LayoutOptions::default()
            )?
            .regions,
        [] as [rs_ocr::documents::layout::LayoutRegion; 0]
    );
    assert!(matches!(
        detector.detect_file(
            "not-present.png",
            LayoutOptions {
                min_confidence: f32::NAN,
                ..LayoutOptions::default()
            }
        ),
        Err(LayoutError::InvalidInput(_))
    ));
    assert!(
        detector
            .detect_bytes(b"invalid image", LayoutOptions::default())
            .is_err()
    );
    assert!(
        detector
            .detect_bytes(&[], LayoutOptions::default())
            .is_err()
    );
    assert!(LayoutDetector::new(root.join("models/ppocrv6-tiny/rec.rten")).is_err());
    let reader = DocumentReader::from_model_dir(root.join("models/ppocrv6-tiny"))?;
    let mut reader = LayoutReader::new(&model, reader)?;
    let page = reader.read_file(&image, LayoutOptions::default())?;
    let via_bytes = reader.read_bytes(&bytes, LayoutOptions::default())?;
    assert_eq!(page, via_bytes);
    assert_eq!(page.layout(), &layout);
    let plain = rs_ocr::bills::extract(page.document().clone(), ExtractionOptions::default())?;
    let assisted = page.extract_bill(ExtractionOptions::default())?;
    assert_eq!(assisted.document, *page.document());
    if !plain.total.candidates.is_empty() {
        assert_eq!(plain.total, assisted.total);
    }
    for members in page.region_spans() {
        assert!(members.iter().all(|&i| i < page.document().spans().len()));
    }
    assert!(reader.read_bytes(b"bad", LayoutOptions::default()).is_err());
    assert_eq!(reader.read_bytes(&bytes, LayoutOptions::default())?, page);
    Ok(())
}
