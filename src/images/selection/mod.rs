#![doc = include_str!("../../../docs/image-selection.md")]

mod grid;
mod model;
pub(crate) mod preprocess;
mod scoring;
mod tokenizer;

pub use grid::{Grid, TileRegion};
pub use model::ImageSelector;
pub use scoring::{LabelScore, SelectionOptions, SelectionResult, TilePrediction};

/// Configuration, image, model and inference failures.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SelectionError {
    /// Grid dimensions must be positive and contain at most 64 tiles.
    #[error("invalid image grid: {0}")]
    InvalidGrid(&'static str),
    /// Invalid labels, thresholds or label count.
    #[error("invalid selection options: {0}")]
    InvalidOptions(&'static str),
    /// Encoded data or dimensions cannot be processed.
    #[error("invalid image: {0}")]
    InvalidImage(&'static str),
    /// Local image file could not be read.
    #[error("could not read image {path}: {source}")]
    ImageIo {
        /// Supplied image path.
        path: std::path::PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// Image decoding failed.
    #[error("image decoding failed: {0}")]
    Image(#[from] image::ImageError),
    /// ONNX model could not be loaded by `RTen`.
    #[error("CLIP model loading failed: {0}")]
    ModelLoad(#[from] rten::LoadError),
    /// Tokenizer configuration could not be loaded.
    #[error("CLIP tokenizer loading failed: {0}")]
    TokenizerLoad(#[from] rten_text::tokenizer::FromJsonError),
    /// Text tokenization failed.
    #[error("CLIP tokenization failed: {0}")]
    Tokenization(#[from] rten_text::TokenizerError),
    /// A caption exceeds the CLIP context size, including its boundary tokens.
    #[error("caption for {label:?} has {tokens} tokens; maximum is 77")]
    LabelTooLong {
        /// Supplied label.
        label: String,
        /// Token count including start/end tokens and the prompt template.
        tokens: usize,
    },
    /// Required CLIP inputs/outputs are absent or incompatible.
    #[error("incompatible CLIP model: {0}")]
    ModelInterface(&'static str),
    /// Inference failed.
    #[error("CLIP inference failed: {0}")]
    Inference(#[from] rten::RunError),
    /// Model logits had unexpected dimensions, types or non-finite values.
    #[error("invalid CLIP output: {0}")]
    InvalidOutput(&'static str),
}
