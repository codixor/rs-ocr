//! Explicit real-model checks; ordinary tests do not download models.

use std::path::PathBuf;

use rs_ocr::captch::{CaptchaDecoder, CaptchaError, DecodeOptions};

#[test]
#[ignore = "requires RustO models; see README.md for setup"]
fn decodes_known_text_from_files_and_bytes_and_rejects_invalid_images() {
    let model_dir = std::env::var_os("RS_OCR_MODEL_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ppocrv6-tiny"),
        PathBuf::from,
    );
    let mut decoder = CaptchaDecoder::from_model_dir(&model_dir)
        .expect("load real models from RS_OCR_MODEL_DIR or models/ppocrv6-tiny");
    let options = DecodeOptions {
        allowed_chars: Some("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".into()),
        expected_length: Some(6),
        ..DecodeOptions::default()
    };
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plain.png");
    let from_file = decoder.decode_file(&fixture, &options).unwrap();
    let from_bytes = decoder
        .decode_bytes(include_bytes!("fixtures/plain.png"), &options)
        .unwrap();
    assert_eq!(from_file.text, "A7B9C2");
    assert_eq!(from_bytes.text, "A7B9C2");
    assert_eq!(from_file, from_bytes);
    assert_ne!(
        from_file.segments,
        [] as [rs_ocr::captch::CaptchaSegment; 0]
    );
    assert!(from_file.confidence >= options.min_confidence);
    assert!(matches!(
        decoder.decode_bytes(b"this is not a PNG", &options),
        Err(CaptchaError::Backend(_))
    ));
    assert!(matches!(
        decoder.decode_bytes(&[], &options),
        Err(CaptchaError::EmptyImage)
    ));
    assert!(matches!(
        decoder.decode_file(fixture.with_file_name("missing.png"), &options),
        Err(CaptchaError::ImageIo { .. })
    ));
}
