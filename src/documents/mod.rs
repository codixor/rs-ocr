#![doc = include_str!("../../docs/financial-documents.md")]
//! Shared document OCR, extraction evidence and exact decimal amounts.
//!
//! Text is retained with whitespace and coordinates. Missing/ambiguous fields
//! remain explicit; OCR recognition does not establish document authenticity.

mod extraction;
#[cfg(feature = "document-layout")]
pub mod layout;
mod money;
mod regions;

pub(crate) use extraction::{
    amount_field, cheque_number, currency_amount, date_field, field, header, review_field,
};
pub use money::{Amount, AmountFormat, DecimalSeparator, ParseAmountError};
pub use regions::{ImageRegion, RegionReader};

use rusto::{DetectTextResult, ImageSource, InitializeConfig, OcrRunOptions, RustO};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// Recognized text with original evidence; manually supplied text can lack scores/geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpan {
    /// Unmodified text.
    pub text: String,
    /// Recognition score, if produced by an OCR model.
    pub confidence: Option<f32>,
    /// Original quadrilateral in pixels, if available.
    pub box_points: Option<[[f32; 2]; 4]>,
}

/// Validated document text, reusable by multiple domain extractors.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    spans: Vec<TextSpan>,
}

impl Document {
    /// Validate OCR or caller-supplied spans without changing text.
    ///
    /// # Errors
    /// Rejects empty text, excessive size, non-finite geometry and invalid scores.
    pub fn new(spans: Vec<TextSpan>) -> Result<Self, DocumentError> {
        if spans.len() > 4096
            || spans
                .iter()
                .try_fold(0_usize, |size, span| size.checked_add(span.text.len()))
                .is_none_or(|size| size > 1024 * 1024)
        {
            return Err(DocumentError::InvalidInput(
                "document exceeds span/text limits",
            ));
        }
        for span in &spans {
            if span
                .confidence
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
                || span.box_points.is_some_and(|points| {
                    points.iter().flatten().any(|v| !v.is_finite() || *v < 0.0)
                })
            {
                return Err(DocumentError::InvalidInput(
                    "invalid recognition score or geometry",
                ));
            }
        }
        if !spans.iter().any(|s| !s.text.trim().is_empty()) {
            return Err(DocumentError::NoText);
        }
        Ok(Self { spans })
    }

    /// Use an existing transcription without inventing OCR confidence or geometry.
    ///
    /// # Errors
    /// Returns the same document validation errors as [`Self::new`].
    pub fn from_text(text: &str) -> Result<Self, DocumentError> {
        if text.len() > 1024 * 1024 {
            return Err(DocumentError::InvalidInput("document exceeds text limit"));
        }
        Self::new(
            text.lines()
                .take(4097)
                .map(|line| TextSpan {
                    text: line.to_owned(),
                    confidence: None,
                    box_points: None,
                })
                .collect(),
        )
    }

    /// Original evidence in backend/caller order.
    #[must_use]
    pub fn spans(&self) -> &[TextSpan] {
        &self.spans
    }

    /// Unmodified span texts joined by newlines.
    #[must_use]
    pub fn raw_text(&self) -> String {
        self.spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Candidate value tied to the source spans that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldCandidate<T> {
    /// Parsed value, without repairing ambiguous OCR characters.
    pub value: T,
    /// Recognized field value before parsing.
    pub raw_text: String,
    /// Lowest contributing OCR score; `None` for unscored transcription.
    pub confidence: Option<f32>,
    /// Indices into [`Document::spans`].
    pub span_indices: Vec<usize>,
}

/// All candidates for a field; conflicting values are never silently resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct Field<T> {
    /// Candidates in source reading order.
    pub candidates: Vec<FieldCandidate<T>>,
}

impl<T: PartialEq> Field<T> {
    /// A value only when every candidate agrees. Repeated equal candidates use
    /// the weakest recognition score; an unscored candidate remains unscored.
    #[must_use]
    pub fn unique(&self) -> Option<&FieldCandidate<T>> {
        let first = self.candidates.first()?;
        if self
            .candidates
            .iter()
            .any(|candidate| candidate.value != first.value)
        {
            return None;
        }
        self.candidates
            .iter()
            .min_by(|a, b| match (a.confidence, b.confidence) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, _) => std::cmp::Ordering::Less,
                (_, None) => std::cmp::Ordering::Greater,
                (Some(a), Some(b)) => a.total_cmp(&b),
            })
    }
}

/// Extraction settings; date strings and currency are deliberately not inferred.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractionOptions {
    /// Values below this recognition score produce a review issue, not deletion.
    pub min_confidence: f32,
    /// Explicit decimal convention and fractional precision for monetary fields.
    pub amount_format: AmountFormat,
}

impl Default for ExtractionOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            amount_format: AmountFormat::default(),
        }
    }
}

impl ExtractionOptions {
    /// Validate settings before OCR or extraction.
    ///
    /// # Errors
    /// Rejects invalid confidence or amount formats.
    pub fn validate(self) -> Result<(), DocumentError> {
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(DocumentError::InvalidInput(
                "confidence must be finite and in 0..=1",
            ));
        }
        self.amount_format
            .validate()
            .map_err(DocumentError::Amount)?;
        Ok(())
    }
}

/// Reviewable missing, conflicting or inconsistent extraction evidence.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReviewIssue {
    /// A layout/text heuristic supplied a value without an explicit field label.
    InferredField {
        /// Field requiring review.
        field: &'static str,
        /// Method used; this is not a calibrated confidence score.
        method: &'static str,
    },
    /// Required field had no candidate.
    MissingField(&'static str),
    /// Different source values were found for one field.
    AmbiguousField(&'static str),
    /// A contributing score is below the caller's threshold.
    LowConfidence {
        /// Field name.
        field: &'static str,
        /// Lowest score.
        score: f32,
    },
    /// A monetary value could not be parsed exactly.
    InvalidAmount {
        /// Field name.
        field: &'static str,
        /// Original recognized value.
        raw: String,
    },
    /// Subtotal plus tax differs from total; discounts/shipping need review too.
    TotalsMismatch,
    /// Arithmetic could not be checked without overflow.
    AmountOverflow,
    /// Cheque amount is zero or negative.
    NonPositiveChequeAmount,
}

/// OCR, image I/O or structured-input validation failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DocumentError {
    /// Invalid input/options/recognition evidence.
    #[error("invalid document input: {0}")]
    InvalidInput(&'static str),
    /// Local image could not be read.
    #[error("could not read document {path}: {source}")]
    ImageIo {
        /// Caller-supplied path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// OCR engine/model/image decoding failure.
    #[error("document OCR failed: {0}")]
    Backend(#[from] rusto::EngineError),
    /// Bounded image decoding failed before region recognition.
    #[error("document image decoding failed: {0}")]
    Image(#[from] image::ImageError),
    /// No nonblank text recognized/supplied.
    #[error("document contains no text")]
    NoText,
    /// Expected structured OCR output was absent.
    #[error("expected structured document OCR output")]
    UnexpectedOutput,
    /// Invalid decimal configuration.
    #[error("invalid amount format: {0}")]
    Amount(#[from] ParseAmountError),
}

/// Reusable `RustO` page OCR preserving text, scores and original coordinates.
pub struct DocumentReader {
    engine: RustO,
}

impl DocumentReader {
    /// Load a matching `RustO` detector/recognizer/dictionary configuration.
    ///
    /// # Errors
    /// Returns model/configuration loading errors.
    pub fn new(config: InitializeConfig) -> Result<Self, DocumentError> {
        Ok(Self {
            engine: RustO::initialize(config)?,
        })
    }

    /// Wrap an already built engine, for example one from `engines::burn::ppv6_tiny`.
    #[must_use]
    pub fn from_engine(engine: RustO) -> Self {
        Self { engine }
    }

    /// Load the existing PP-OCRv6 model pack from a local directory.
    ///
    /// # Errors
    /// Returns missing or incompatible model errors.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self, DocumentError> {
        let dir = directory.as_ref();
        Self::new(InitializeConfig::ppv6(
            dir.join("det.rten"),
            dir.join("rec.rten"),
            dir.join("dict.txt"),
        ))
    }

    /// OCR PNG/JPEG bytes, retaining low-scoring spans for review.
    ///
    /// # Errors
    /// Rejects empty/oversized inputs, OCR errors and malformed evidence.
    pub fn read_bytes(&mut self, bytes: &[u8]) -> Result<Document, DocumentError> {
        if bytes.is_empty() || bytes.len() > 32 * 1024 * 1024 {
            return Err(DocumentError::InvalidInput(
                "encoded image must be 1 byte..=32 MiB",
            ));
        }
        let DetectTextResult::Structured(spans) = self.engine.detect_text(
            &ImageSource::Bytes(bytes.to_vec()),
            &OcrRunOptions {
                text_score: Some(0.0),
                ..OcrRunOptions::default()
            },
        )?
        else {
            return Err(DocumentError::UnexpectedOutput);
        };
        Document::new(
            spans
                .into_iter()
                .map(|s| TextSpan {
                    text: crate::bidi::visual_to_logical(&s.text),
                    confidence: Some(s.score),
                    box_points: Some(s.box_points.map(|(x, y)| [x, y])),
                })
                .collect(),
        )
    }

    /// OCR a local file with bounded encoded reads.
    ///
    /// # Errors
    /// Returns I/O, encoded input, OCR or evidence-validation errors.
    pub fn read_file(&mut self, path: impl AsRef<Path>) -> Result<Document, DocumentError> {
        let path = path.as_ref();
        let read = || -> std::io::Result<Vec<u8>> {
            let mut bytes = Vec::new();
            File::open(path)?
                .take(32 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            Ok(bytes)
        };
        let bytes = read().map_err(|source| DocumentError::ImageIo {
            path: path.to_path_buf(),
            source,
        })?;
        self.read_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_empty_oversized_or_invalid_document_evidence() {
        assert!(matches!(
            Document::from_text(" \n\t"),
            Err(DocumentError::NoText)
        ));
        assert!(Document::from_text(&"x".repeat(1024 * 1024 + 1)).is_err());
        assert!(Document::from_text(&"x\n".repeat(4097)).is_err());
        for confidence in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(
                Document::new(vec![TextSpan {
                    text: "TOTAL: 1".into(),
                    confidence: Some(confidence),
                    box_points: None
                }])
                .is_err()
            );
        }
        assert!(
            Document::new(vec![TextSpan {
                text: "TOTAL: 1".into(),
                confidence: None,
                box_points: Some([[f32::NAN, 0.0]; 4])
            }])
            .is_err()
        );
    }
}
