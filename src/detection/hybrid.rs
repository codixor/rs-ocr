use super::{
    CocoClass, DetectionError, DetectionOptions, DetectionResult, GridObjectSelection,
    ObjectDetector, SpatialOptions,
};
use crate::images::selection::{
    Grid, ImageSelector, SelectionOptions, SelectionResult, preprocess::read_image_file,
};
use std::path::Path;

/// Explicit method and evidence used for a grid selection.
#[derive(Debug, Clone, PartialEq)]
pub enum HybridSelection {
    /// Full-image detection for a supported class.
    Objects {
        /// Selected COCO class.
        class: CocoClass,
        /// Full-image object scores and bounds.
        detections: DetectionResult,
        /// Cell selections and per-cell overlap.
        selection: GridObjectSelection,
    },
    /// Explicit classifier fallback for a class absent from COCO.
    Classification(SelectionResult),
}

impl HybridSelection {
    /// Zero-based row-major selected cell indices for either method.
    #[must_use]
    pub fn selected_indices(&self) -> Vec<usize> {
        match self {
            Self::Objects { selection, .. } => selection.selected_indices.clone(),
            Self::Classification(result) => result.selected_indices(),
        }
    }
}

/// Reuses both models, selecting object detection for supported classes and
/// explicit CLIP classification otherwise. Spatial selection assumes cells cut
/// from one contiguous photograph; independent-image montages should use CLIP.
pub struct HybridSelector {
    detector: ObjectDetector,
    classifier: ImageSelector,
}

impl HybridSelector {
    /// Use already-loaded models without downloading or reloading weights.
    #[must_use]
    pub fn new(detector: ObjectDetector, classifier: ImageSelector) -> Self {
        Self {
            detector,
            classifier,
        }
    }

    /// Load local detector and classifier directories, each containing `model.onnx`.
    /// The classifier also needs its matching `tokenizer.json`.
    ///
    /// # Errors
    /// Returns either model's initialization errors.
    pub fn from_model_dirs(
        detector: impl AsRef<Path>,
        classifier: impl AsRef<Path>,
    ) -> Result<Self, DetectionError> {
        Ok(Self::new(
            ObjectDetector::new(detector.as_ref().join("model.onnx"))?,
            ImageSelector::from_model_dir(classifier)?,
        ))
    }

    /// Classify a contiguous-photo grid using full context when the target is
    /// supported. The supplied CLIP options govern only the fallback.
    ///
    /// # Errors
    /// Rejects invalid options/images/grids or failed inference. A detector failure
    /// is returned as an error; it never silently becomes a fallback or no-hit.
    pub fn select_bytes(
        &self,
        bytes: &[u8],
        grid: Grid,
        classifier_options: &SelectionOptions,
        spatial_options: SpatialOptions,
    ) -> Result<HybridSelection, DetectionError> {
        classifier_options.validate()?;
        spatial_options.validate()?;
        if let Some(class) = CocoClass::from_label(&classifier_options.target) {
            let detections = self.detector.detect_bytes(
                bytes,
                DetectionOptions {
                    min_confidence: spatial_options.min_confidence,
                    ..DetectionOptions::default()
                },
            )?;
            let selection = detections.select_grid(grid, class, spatial_options)?;
            Ok(HybridSelection::Objects {
                class,
                detections,
                selection,
            })
        } else {
            Ok(HybridSelection::Classification(
                self.classifier
                    .select_bytes(bytes, grid, classifier_options)?,
            ))
        }
    }

    /// Select cells from a local contiguous-photo grid, using bounded file reads.
    ///
    /// # Errors
    /// Returns configuration, file/image, grid or inference errors.
    pub fn select_file(
        &self,
        path: impl AsRef<Path>,
        grid: Grid,
        classifier_options: &SelectionOptions,
        spatial_options: SpatialOptions,
    ) -> Result<HybridSelection, DetectionError> {
        classifier_options.validate()?;
        spatial_options.validate()?;
        self.select_bytes(
            &read_image_file(path.as_ref())?,
            grid,
            classifier_options,
            spatial_options,
        )
    }
}
