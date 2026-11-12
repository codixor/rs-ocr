//! Local CAPTCHA image decoding using reusable `RustO` model sessions.
//!
//! Whitespace is removed from the candidate; letter case is preserved.
//! Alphabet and length constraints validate the candidate after recognition.
//! They do not modify model logits or substitute ambiguous characters.

use std::path::{Path, PathBuf};

use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};

/// Per-image constraints applied after OCR recognition.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOptions {
    /// Allowed Unicode characters, or `None` to accept any non-whitespace text.
    /// An unexpected character rejects the candidate rather than being deleted.
    pub allowed_chars: Option<String>,
    /// Expected number of Unicode scalar values after removing whitespace.
    pub expected_length: Option<usize>,
    /// Minimum confidence of every non-blank OCR segment, in `0.0..=1.0`.
    /// This is a model score, not a calibrated probability of correctness.
    pub min_confidence: f32,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            allowed_chars: None,
            expected_length: None,
            min_confidence: 0.5,
        }
    }
}

impl DecodeOptions {
    /// Validate constraints without loading a model.
    ///
    /// # Errors
    /// Returns [`CaptchaError::InvalidOptions`] for a non-finite or out-of-range
    /// confidence, zero expected length, or an empty/whitespace-containing alphabet.
    pub fn validate(&self) -> Result<(), CaptchaError> {
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(CaptchaError::InvalidOptions(
                "min_confidence must be finite and in 0.0..=1.0",
            ));
        }
        if self.expected_length == Some(0) {
            return Err(CaptchaError::InvalidOptions(
                "expected_length must be positive",
            ));
        }
        if self.allowed_chars.as_ref().is_some_and(|alphabet| {
            alphabet.is_empty() || alphabet.chars().any(char::is_whitespace)
        }) {
            return Err(CaptchaError::InvalidOptions(
                "allowed_chars must be non-empty and contain no whitespace",
            ));
        }
        Ok(())
    }
}

/// A recognized text segment with its original image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptchaSegment {
    /// Unmodified text produced by the recognition model.
    pub text: String,
    /// Recognition score in `0.0..=1.0`.
    pub confidence: f32,
    /// Quadrilateral in pixels: top-left, top-right, bottom-right, bottom-left.
    pub box_points: [[f32; 2]; 4],
}

/// A candidate that passed the requested confidence, alphabet and length checks.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptchaResult {
    /// Text in backend reading order with all Unicode whitespace removed.
    pub text: String,
    /// Original segment texts joined by newlines, before normalization.
    pub raw_text: String,
    /// Lowest confidence among the non-blank segments contributing to `text`.
    pub confidence: f32,
    /// Original segments, including their scores and bounding polygons.
    pub segments: Vec<CaptchaSegment>,
}

/// Input, inference and candidate-validation failures.
#[derive(Debug, thiserror::Error)]
pub enum CaptchaError {
    /// Invalid per-image constraints.
    #[error("invalid CAPTCHA options: {0}")]
    InvalidOptions(&'static str),
    /// The image file could not be read.
    #[error("could not read image {path}: {source}")]
    ImageIo {
        /// Image path supplied by the caller.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// No encoded image data was supplied.
    #[error("image data is empty")]
    EmptyImage,
    /// Model loading, image decoding or inference failed.
    #[error("RustO OCR failed: {0}")]
    Backend(#[from] rusto::EngineError),
    /// The backend did not return the requested structured result.
    #[error("RustO returned spatial text instead of structured segments")]
    UnexpectedOutput,
    /// No non-whitespace text was recognized.
    #[error("no text was recognized")]
    NoText,
    /// A backend segment contained an invalid score.
    #[error("RustO returned a non-finite or out-of-range confidence")]
    InvalidConfidence,
    /// At least one contributing segment was below the requested threshold.
    #[error("candidate {text:?} has confidence {actual}, below {minimum}")]
    LowConfidence {
        /// Normalized candidate, available for diagnostics.
        text: String,
        /// Lowest contributing segment score.
        actual: f32,
        /// Requested threshold.
        minimum: f32,
    },
    /// The candidate contains a character outside the configured alphabet.
    #[error("candidate {text:?} contains disallowed character {character:?}")]
    UnexpectedCharacter {
        /// Normalized candidate, without deleting the unexpected character.
        text: String,
        /// First character outside the alphabet.
        character: char,
    },
    /// The candidate has a different length from the requested value.
    #[error("candidate {text:?} has {actual} characters, expected {expected}")]
    LengthMismatch {
        /// Normalized candidate.
        text: String,
        /// Number of recognized Unicode scalar values.
        actual: usize,
        /// Requested length.
        expected: usize,
    },
}

/// Loaded `RustO` model sessions, reused for successive local image requests.
///
/// Inference is synchronous and takes `&mut self`. Async applications should run
/// it on a blocking worker and reuse that worker's decoder.
pub struct CaptchaDecoder {
    engine: RustO,
}

impl CaptchaDecoder {
    /// Load detector, recognizer and dictionary using a `RustO` configuration.
    /// Use the v6 Tiny pack: the Arabic pack lowercases some Latin capitals.
    ///
    /// # Errors
    /// Returns [`CaptchaError::Backend`] if the configuration or models cannot load.
    pub fn new(config: InitializeConfig) -> Result<Self, CaptchaError> {
        Ok(Self {
            engine: RustO::initialize(config)?,
        })
    }

    /// Wrap an already built engine, for example one from `engines::burn::ppv6_tiny`.
    #[must_use]
    pub fn from_engine(engine: RustO) -> Self {
        Self { engine }
    }

    /// Load `det.rten`, `rec.rten`, and `dict.txt` using the PP-OCRv6 preset.
    /// Files are loaded locally; this method does not download weights.
    ///
    /// # Errors
    /// Returns [`CaptchaError::Backend`] if a required model cannot load.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self, CaptchaError> {
        let directory = directory.as_ref();
        Self::new(InitializeConfig::ppv6(
            directory.join("det.rten"),
            directory.join("rec.rten"),
            directory.join("dict.txt"),
        ))
    }

    /// Decode an encoded PNG or JPEG file, including non-UTF-8 filesystem paths.
    ///
    /// # Errors
    /// Returns an input, backend, or candidate-validation [`CaptchaError`].
    pub fn decode_file(
        &mut self,
        path: impl AsRef<Path>,
        options: &DecodeOptions,
    ) -> Result<CaptchaResult, CaptchaError> {
        options.validate()?;
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| CaptchaError::ImageIo {
            path: path.to_path_buf(),
            source,
        })?;
        self.decode_owned(bytes, options)
    }

    /// Decode encoded image bytes. The caller retains ownership of the input.
    ///
    /// # Errors
    /// Returns an input, backend, or candidate-validation [`CaptchaError`].
    pub fn decode_bytes(
        &mut self,
        bytes: &[u8],
        options: &DecodeOptions,
    ) -> Result<CaptchaResult, CaptchaError> {
        options.validate()?;
        self.decode_owned(bytes.to_vec(), options)
    }

    fn decode_owned(
        &mut self,
        bytes: Vec<u8>,
        options: &DecodeOptions,
    ) -> Result<CaptchaResult, CaptchaError> {
        if bytes.is_empty() {
            return Err(CaptchaError::EmptyImage);
        }
        // Retain low-scoring segments: dropping them here could make a truncated
        // candidate incorrectly pass its confidence or length constraints.
        let run_options = OcrRunOptions {
            text_score: Some(0.0),
            ..OcrRunOptions::default()
        };
        let DetectTextResult::Structured(segments) = self
            .engine
            .detect_text(&ImageSource::Bytes(bytes), &run_options)?
        else {
            return Err(CaptchaError::UnexpectedOutput);
        };
        let segments = segments
            .into_iter()
            .map(|segment| CaptchaSegment {
                text: crate::bidi::visual_to_logical(&segment.text),
                confidence: segment.score,
                box_points: segment.box_points.map(|(x, y)| [x, y]),
            })
            .collect();
        assemble_result(segments, options)
    }
}

fn assemble_result(
    segments: Vec<CaptchaSegment>,
    options: &DecodeOptions,
) -> Result<CaptchaResult, CaptchaError> {
    let mut text = String::new();
    let mut confidence = 1.0_f32;
    for segment in &segments {
        if !segment.confidence.is_finite() || !(0.0..=1.0).contains(&segment.confidence) {
            return Err(CaptchaError::InvalidConfidence);
        }
        if segment
            .text
            .chars()
            .any(|character| !character.is_whitespace())
        {
            confidence = confidence.min(segment.confidence);
            text.extend(
                segment
                    .text
                    .chars()
                    .filter(|character| !character.is_whitespace()),
            );
        }
    }
    if text.is_empty() {
        return Err(CaptchaError::NoText);
    }
    if confidence < options.min_confidence {
        return Err(CaptchaError::LowConfidence {
            text,
            actual: confidence,
            minimum: options.min_confidence,
        });
    }
    if let Some(alphabet) = &options.allowed_chars
        && let Some(character) = text
            .chars()
            .find(|character| !alphabet.contains(*character))
    {
        return Err(CaptchaError::UnexpectedCharacter { text, character });
    }
    if let Some(expected) = options.expected_length {
        let actual = text.chars().count();
        if actual != expected {
            return Err(CaptchaError::LengthMismatch {
                text,
                actual,
                expected,
            });
        }
    }
    let raw_text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(CaptchaResult {
        text,
        raw_text,
        confidence,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str, confidence: f32) -> CaptchaSegment {
        CaptchaSegment {
            text: text.into(),
            confidence,
            box_points: [[0.0, 0.0]; 4],
        }
    }

    #[test]
    fn joins_segments_without_whitespace_and_preserves_case_and_raw_text() {
        let result = assemble_result(
            vec![segment(" Ab ", 0.9), segment("3 4", 0.6)],
            &DecodeOptions::default(),
        )
        .unwrap();
        assert_eq!(result.text, "Ab34");
        assert_eq!(result.raw_text, " Ab \n3 4");
        assert_eq!(result.confidence.to_bits(), 0.6_f32.to_bits());
        assert_eq!(result.segments.len(), 2);
    }

    #[test]
    fn rejects_characters_outside_the_alphabet_instead_of_deleting_them() {
        let options = DecodeOptions {
            allowed_chars: Some("A1234".into()),
            ..DecodeOptions::default()
        };
        assert!(matches!(
            assemble_result(vec![segment("A$4", 0.9)], &options),
            Err(CaptchaError::UnexpectedCharacter { character: '$', .. })
        ));
    }

    #[test]
    fn measures_expected_length_in_characters() {
        let options = DecodeOptions {
            expected_length: Some(2),
            ..DecodeOptions::default()
        };
        assert_eq!(
            assemble_result(vec![segment("اب", 0.9)], &options)
                .unwrap()
                .text,
            "اب"
        );
        assert!(matches!(
            assemble_result(vec![segment("ABC", 0.9)], &options),
            Err(CaptchaError::LengthMismatch {
                expected: 2,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn a_weak_segment_is_not_hidden_by_a_strong_segment() {
        assert!(matches!(
            assemble_result(
                vec![segment("ABC", 0.99), segment("2", 0.1)],
                &DecodeOptions::default(),
            ),
            Err(CaptchaError::LowConfidence { .. })
        ));
    }

    #[test]
    fn blank_output_is_not_a_successful_decode() {
        for segments in [vec![], vec![segment(" \t\n", 0.99)]] {
            assert!(matches!(
                assemble_result(segments, &DecodeOptions::default()),
                Err(CaptchaError::NoText)
            ));
        }
    }

    #[test]
    fn blank_segments_do_not_lower_the_text_confidence() {
        let result = assemble_result(
            vec![segment(" ", 0.0), segment("ABC", 0.9)],
            &DecodeOptions::default(),
        )
        .unwrap();
        assert_eq!(result.text, "ABC");
        assert_eq!(result.confidence.to_bits(), 0.9_f32.to_bits());
    }

    #[test]
    fn invalid_backend_confidence_is_rejected() {
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(matches!(
                assemble_result(vec![segment("ABC", value)], &DecodeOptions::default()),
                Err(CaptchaError::InvalidConfidence)
            ));
        }
    }

    #[test]
    fn invalid_constraints_are_rejected_before_inference() {
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(
                DecodeOptions {
                    min_confidence: value,
                    ..DecodeOptions::default()
                }
                .validate()
                .is_err()
            );
        }
        for alphabet in ["", "ABC ", "\t"] {
            assert!(
                DecodeOptions {
                    allowed_chars: Some(alphabet.into()),
                    ..DecodeOptions::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            DecodeOptions {
                expected_length: Some(0),
                ..DecodeOptions::default()
            }
            .validate()
            .is_err()
        );
        for value in [0.0, 1.0] {
            assert!(
                DecodeOptions {
                    min_confidence: value,
                    ..DecodeOptions::default()
                }
                .validate()
                .is_ok()
            );
        }
    }

    #[test]
    fn missing_model_files_return_an_error() {
        assert!(CaptchaDecoder::from_model_dir("/path/that/does/not/exist/rs-ocr-models").is_err());
    }
}
