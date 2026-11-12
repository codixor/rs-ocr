#![doc = include_str!("../../../docs/document-layout.md")]
//! Optional PP-DocLayout-S regions alongside Tiny OCR.

mod companion;
mod model;
mod preprocess;

pub use companion::{LayoutDocument, LayoutReader};
pub use model::{LayoutDetector, LayoutOptions};

/// Official PP-DocLayout-S labels, in model output order.
pub const CLASS_NAMES: [&str; 23] = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
];

/// Validated identifier into the official 23-class label table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutClass(u8);

impl LayoutClass {
    /// Text block class.
    pub const TEXT: Self = Self(2);
    /// Table region class.
    pub const TABLE: Self = Self(8);
    /// Main document title class.
    pub const DOCUMENT_TITLE: Self = Self(11);
    /// Validate an official class code.
    #[must_use]
    pub fn new(code: u8) -> Option<Self> {
        (usize::from(code) < CLASS_NAMES.len()).then_some(Self(code))
    }
    /// Numeric model class code.
    #[must_use]
    pub fn code(self) -> u8 {
        self.0
    }
    /// Official class name.
    #[must_use]
    pub fn name(self) -> &'static str {
        CLASS_NAMES[usize::from(self.0)]
    }
}

/// Axis-aligned extent in original image pixels; right/bottom are exclusive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutBox {
    /// Left edge.
    pub left: f32,
    /// Top edge.
    pub top: f32,
    /// Right edge.
    pub right: f32,
    /// Bottom edge.
    pub bottom: f32,
}

impl LayoutBox {
    pub(crate) fn area(self) -> f32 {
        (self.right - self.left).max(0.0) * (self.bottom - self.top).max(0.0)
    }
    pub(crate) fn valid(self) -> bool {
        [self.left, self.top, self.right, self.bottom]
            .iter()
            .all(|v| v.is_finite())
            && self.left >= 0.0
            && self.top >= 0.0
            && self.area() > 0.0
    }
    pub(crate) fn intersection(self, other: Self) -> f32 {
        (self.right.min(other.right) - self.left.max(other.left)).max(0.0)
            * (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0)
    }
}

/// A model-predicted region, not a financial-field classification.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutRegion {
    /// Official region category.
    pub class: LayoutClass,
    /// Unmodified model score.
    pub confidence: f32,
    /// Original-frame extent clipped to the image.
    pub bounds: LayoutBox,
    /// Original model coordinates before clipping: left, top, right, bottom.
    pub raw_bounds: [f32; 4],
}

/// Page layout independent of its OCR text.
#[derive(Debug, Clone, PartialEq)]
pub struct PageLayout {
    /// Original width.
    pub width: u32,
    /// Original height.
    pub height: u32,
    /// Highest scores first; coordinates always refer to the original page.
    pub regions: Vec<LayoutRegion>,
}

/// Input, model-interface, inference or OCR failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// Invalid options/image/geometry.
    #[error("invalid layout input: {0}")]
    InvalidInput(&'static str),
    /// Incompatible model signature.
    #[error("unsupported layout model interface: {0}")]
    ModelInterface(&'static str),
    /// Invalid model output.
    #[error("invalid layout output: {0}")]
    InvalidOutput(&'static str),
    /// Model loading failed.
    #[error("layout model loading failed: {0}")]
    Load(#[from] rten::LoadError),
    /// CPU inference failed.
    #[error("layout inference failed: {0}")]
    Run(#[from] rten::RunError),
    /// Bounded image decoding failed.
    #[error("layout image decoding failed: {0}")]
    Image(#[from] image::ImageError),
    /// File I/O failed.
    #[error("layout image I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Tiny OCR or extraction failed.
    #[error("layout companion OCR failed: {0}")]
    Document(#[from] crate::documents::DocumentError),
}
