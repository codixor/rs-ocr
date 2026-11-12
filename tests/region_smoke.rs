//! Native Tiny single-line OCR, preserving caller-region evidence.
use rs_ocr::documents::{DocumentError, ImageRegion, RegionReader};
use std::{error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires local Tiny PP-OCRv6 recognizer and dictionary"]
fn tiny_region_reader_preserves_scores_coordinates_and_file_byte_equivalence()
-> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut reader = RegionReader::from_model_dir(root.join("models/ppocrv6-tiny"))?;
    let path = root.join("tests/fixtures/cheque.png");
    let region = ImageRegion {
        x: 50,
        y: 420,
        width: 700,
        height: 70,
    };
    let doc = reader.read_file(&path, region)?;
    assert_eq!(doc, reader.read_bytes(&fs::read(&path)?, region)?);
    assert_eq!(doc.spans().len(), 1);
    assert_eq!(doc.raw_text().replace(' ', ""), "AMOUNT:1234.56");
    assert!(doc.spans()[0].confidence.is_some_and(|s| s > 0.5));
    assert_eq!(
        doc.spans()[0].box_points,
        Some([[50.0, 420.0], [750.0, 420.0], [750.0, 490.0], [50.0, 490.0]])
    );
    assert!(matches!(
        reader.read_file(
            &path,
            ImageRegion {
                x: u32::MAX,
                ..region
            }
        ),
        Err(DocumentError::InvalidInput(_))
    ));
    assert!(reader.read_bytes(b"not an image", region).is_err());
    assert!(reader.read_bytes(&[], region).is_err());
    assert!(
        reader
            .read_file(root.join("missing-image.png"), region)
            .is_err()
    );
    // A failed request leaves the reused recognizer usable.
    assert_eq!(doc, reader.read_file(path, region)?);
    Ok(())
}
