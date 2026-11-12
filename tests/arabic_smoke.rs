//! Arabic pages through the PP-OCRv6 Arabic pack (v6 Tiny detector + Arabic v6 small
//! recognizer), checked against the exact expected lines in reading order.

use std::{env, fs, path::PathBuf};

use rs_ocr::documents::DocumentReader;

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

fn words(text: &str) -> Vec<char> {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .collect()
}

#[test]
#[ignore = "requires local Arabic PP-OCRv6 models; see README.md for setup"]
fn reads_arabic_pages_in_reading_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = env::var_os("RS_OCR_ARABIC_MODEL_DIR")
        .map_or_else(|| root.join("models/ppocrv6-arabic-small"), PathBuf::from);
    let mut reader = DocumentReader::from_model_dir(models).expect("load Arabic models");
    for page in ["arabic-invoice", "arabic-notice"] {
        let fixtures = root.join("tests/fixtures");
        let expected = fs::read_to_string(fixtures.join(format!("{page}.txt"))).unwrap();
        let document = reader
            .read_file(fixtures.join(format!("{page}.png")))
            .unwrap();
        let got: Vec<&str> = document.spans().iter().map(|s| s.text.as_str()).collect();
        let (got, expected) = (words(&got.join("\n")), words(&expected));
        let errors = levenshtein(&got, &expected);
        // Measured 0/207 and 3/208 on 2026-11-11; allow small drift across platforms.
        assert!(
            errors * 100 <= expected.len() * 3,
            "{page}: {errors} character errors in {} characters\n{}",
            expected.len(),
            got.iter().collect::<String>()
        );
    }
}
