//! The Burn backend must read the same text as the `RTen` backend from the same models.

use std::path::PathBuf;

use rs_ocr::{
    captch::{CaptchaDecoder, DecodeOptions},
    engines::burn::{self as burn_engine, BurnDevice},
};

#[test]
#[ignore = "requires RustO models; see README.md for setup"]
fn burn_cpu_decodes_the_same_captcha_as_rten() {
    let model_dir = std::env::var_os("RS_OCR_MODEL_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ppocrv6-tiny"),
        PathBuf::from,
    );
    let options = DecodeOptions {
        allowed_chars: Some("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".into()),
        expected_length: Some(6),
        ..DecodeOptions::default()
    };
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plain.png");

    let rten = CaptchaDecoder::from_model_dir(&model_dir)
        .expect("load RTen models")
        .decode_file(&fixture, &options)
        .unwrap();
    let burn = CaptchaDecoder::from_engine(
        burn_engine::ppv6_tiny(&model_dir, BurnDevice::Cpu).expect("load Burn engine"),
    )
    .decode_file(&fixture, &options)
    .unwrap();

    assert_eq!(rten.text, "A7B9C2");
    assert_eq!(burn.text, rten.text);
    assert!(
        (burn.confidence - rten.confidence).abs() < 1e-3,
        "confidence drifted: burn {} vs rten {}",
        burn.confidence,
        rten.confidence
    );
}
