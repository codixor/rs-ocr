#![doc = include_str!("../../docs/localization-benchmark.md")]
//! Full-image YOLO detection and spatial selection of image-grid cells.
//!
//! Enable `object-detection`. The pinned FP32 YOLO11s model uses the existing
//! pure Rust `RTen` runtime. Weights are loaded locally and licensed separately.

mod geometry;
mod hybrid;
mod model;

pub use geometry::{
    BoundingBox, CocoClass, Detection, DetectionResult, GridObjectSelection, SpatialOptions,
};
pub use hybrid::{HybridSelection, HybridSelector};
pub use model::{DetectionOptions, ObjectDetector};

/// Invalid configuration, input, model or model output.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DetectionError {
    /// Invalid confidence, overlap or result geometry.
    #[error("invalid detection configuration or geometry: {0}")]
    InvalidInput(&'static str),
    /// Image reading, decoding or grid validation failed.
    #[error("image/grid or classifier failed: {0}")]
    Image(#[from] crate::images::selection::SelectionError),
    /// The local detector could not load.
    #[error("detector loading failed: {0}")]
    ModelLoad(#[from] rten::LoadError),
    /// Required detector nodes or dimensions are missing.
    #[error("incompatible detector: {0}")]
    ModelInterface(&'static str),
    /// The runtime failed to execute the model.
    #[error("detector inference failed: {0}")]
    Inference(#[from] rten::RunError),
    /// The model did not produce the documented finite COCO tensor.
    #[error("invalid detector output: {0}")]
    InvalidOutput(&'static str),
}
