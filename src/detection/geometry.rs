use crate::images::selection::Grid;

use super::DetectionError;

fn pixel(value: u32) -> Result<f32, DetectionError> {
    Ok(f32::from(u16::try_from(value).map_err(|_| {
        DetectionError::InvalidInput("pixel exceeds supported range")
    })?))
}

const LABELS: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Validated zero-based class index in the standard 80-class COCO vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CocoClass(u8);

impl CocoClass {
    /// Construct a class, rejecting indices outside 0..80.
    #[must_use]
    pub fn new(index: u8) -> Option<Self> {
        (index < 80).then_some(Self(index))
    }

    /// Index in the detector's class dimension.
    #[must_use]
    pub fn index(self) -> u8 {
        self.0
    }

    /// Canonical English class name.
    #[must_use]
    pub fn label(self) -> &'static str {
        LABELS[usize::from(self.0)]
    }

    /// Match a canonical class name or a documented common plural alias.
    /// Unsupported classes such as crosswalks return `None`, never an empty hit.
    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        let label = label.trim().to_lowercase();
        let canonical = match label.as_str() {
            "bicycles" => "bicycle",
            "cars" => "car",
            "motorcycles" => "motorcycle",
            "buses" => "bus",
            "traffic lights" => "traffic light",
            "a fire hydrant" | "fire hydrants" => "fire hydrant",
            "cats" => "cat",
            "horses" => "horse",
            other => other,
        };
        LABELS
            .iter()
            .position(|value| *value == canonical)
            .and_then(|index| u8::try_from(index).ok())
            .map(Self)
    }
}

/// Half-open original-image coordinates; units are pixels, without rounding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Left edge, inclusive.
    pub left: f32,
    /// Top edge, inclusive.
    pub top: f32,
    /// Right edge, exclusive.
    pub right: f32,
    /// Bottom edge, exclusive.
    pub bottom: f32,
}

impl BoundingBox {
    pub(super) fn valid(self) -> bool {
        [self.left, self.top, self.right, self.bottom]
            .iter()
            .all(|v| v.is_finite())
            && self.left < self.right
            && self.top < self.bottom
    }
    fn area(self) -> f32 {
        (self.right - self.left) * (self.bottom - self.top)
    }
    pub(super) fn intersection(self, other: Self) -> f32 {
        (self.right.min(other.right) - self.left.max(other.left)).max(0.0)
            * (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0)
    }
    pub(super) fn iou(self, other: Self) -> f32 {
        let intersection = self.intersection(other);
        intersection / (self.area() + other.area() - intersection)
    }
}

/// One object after class-aware non-maximum suppression.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// COCO class.
    pub class: CocoClass,
    /// Detector score; not a calibrated correctness probability.
    pub confidence: f32,
    /// Box in original-image coordinates, clipped to image boundaries.
    pub bounds: BoundingBox,
}

/// Full-image detections retained for reuse at different spatial thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// Original image width.
    pub width: u32,
    /// Original image height.
    pub height: u32,
    /// Objects sorted by decreasing confidence.
    pub objects: Vec<Detection>,
}

/// Selection thresholds applied to original-image object boxes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialOptions {
    /// Minimum object confidence. Inference must retain objects down to this value.
    pub min_confidence: f32,
    /// Minimum fraction of a tile covered by any single matching object box.
    /// A threshold of zero still requires strictly positive overlap.
    pub min_coverage: f32,
}

impl Default for SpatialOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.25,
            min_coverage: 0.01,
        }
    }
}

impl SpatialOptions {
    /// Validate both thresholds.
    ///
    /// # Errors
    /// Returns an error for non-finite values or values outside 0..=1.
    pub fn validate(self) -> Result<(), DetectionError> {
        if [self.min_confidence, self.min_coverage]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(DetectionError::InvalidInput(
                "thresholds must be finite and in 0..=1",
            ));
        }
        Ok(())
    }
}

/// Object-based selection with auditable per-tile overlap.
#[derive(Debug, Clone, PartialEq)]
pub struct GridObjectSelection {
    /// Zero-based row-major selected indices.
    pub selected_indices: Vec<usize>,
    /// Largest matching-box intersection divided by tile area, in grid order.
    pub tile_coverage: Vec<f32>,
}

impl DetectionResult {
    /// Map full-image object boxes to grid cells without cropping away context.
    ///
    /// # Errors
    /// Rejects invalid thresholds, dimensions, scores, or out-of-image geometry.
    pub fn select_grid(
        &self,
        grid: Grid,
        class: CocoClass,
        options: SpatialOptions,
    ) -> Result<GridObjectSelection, DetectionError> {
        options.validate()?;
        if self.width == 0
            || self.height == 0
            || self.width > 8192
            || self.height > 8192
            || self.objects.len() > 1000
        {
            return Err(DetectionError::InvalidInput(
                "invalid image dimensions or object count",
            ));
        }
        for object in &self.objects {
            let b = object.bounds;
            if !object.confidence.is_finite()
                || !(0.0..=1.0).contains(&object.confidence)
                || !b.valid()
                || b.left < 0.0
                || b.top < 0.0
                || b.right > pixel(self.width)?
                || b.bottom > pixel(self.height)?
            {
                return Err(DetectionError::InvalidInput(
                    "invalid object score or bounds",
                ));
            }
        }
        let mut selected_indices = Vec::new();
        let mut tile_coverage = Vec::with_capacity(grid.tile_count());
        for (index, tile) in grid.regions(self.width, self.height)?.iter().enumerate() {
            let bounds = BoundingBox {
                left: pixel(tile.x)?,
                top: pixel(tile.y)?,
                right: pixel(tile.x + tile.width)?,
                bottom: pixel(tile.y + tile.height)?,
            };
            let coverage = self
                .objects
                .iter()
                .filter(|object| {
                    object.class == class && object.confidence >= options.min_confidence
                })
                .map(|object| object.bounds.intersection(bounds) / bounds.area())
                .fold(0.0_f32, f32::max);
            if coverage > 0.0 && coverage >= options.min_coverage {
                selected_indices.push(index);
            }
            tile_coverage.push(coverage);
        }
        Ok(GridObjectSelection {
            selected_indices,
            tile_coverage,
        })
    }
}

pub(super) fn nms(mut objects: Vec<Detection>, threshold: f32, limit: usize) -> Vec<Detection> {
    objects.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::new();
    for object in objects {
        if kept
            .iter()
            .any(|other| other.class == object.class && object.bounds.iou(other.bounds) > threshold)
        {
            continue;
        }
        kept.push(object);
        if kept.len() == limit {
            break;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    fn object(class: u8, confidence: f32, bounds: [f32; 4]) -> Detection {
        Detection {
            class: CocoClass::new(class).unwrap(),
            confidence,
            bounds: BoundingBox {
                left: bounds[0],
                top: bounds[1],
                right: bounds[2],
                bottom: bounds[3],
            },
        }
    }
    #[test]
    fn spatial_overlap_covers_fragments_and_excludes_touching_edges() {
        let result = DetectionResult {
            width: 450,
            height: 450,
            objects: vec![object(1, 0.9, [112.0, 112.0, 225.0, 225.0])],
        };
        let grid = Grid::new(4, 4).unwrap();
        assert_eq!(
            result
                .select_grid(
                    grid,
                    CocoClass::new(1).unwrap(),
                    SpatialOptions {
                        min_coverage: 0.0,
                        ..SpatialOptions::default()
                    }
                )
                .unwrap()
                .selected_indices,
            [5]
        );
        assert!(
            result
                .select_grid(grid, CocoClass::new(2).unwrap(), SpatialOptions::default())
                .unwrap()
                .selected_indices
                .is_empty()
        );
    }
    #[test]
    fn one_object_can_select_several_cells_and_thresholds_are_enforced() {
        let result = DetectionResult {
            width: 400,
            height: 400,
            objects: vec![object(3, 0.8, [95.0, 95.0, 205.0, 205.0])],
        };
        let class = CocoClass::new(3).unwrap();
        let grid = Grid::new(4, 4).unwrap();
        let selected = result
            .select_grid(
                grid,
                class,
                SpatialOptions {
                    min_confidence: 0.8,
                    min_coverage: 0.01,
                },
            )
            .unwrap();
        assert_eq!(selected.selected_indices, [1, 4, 5, 6, 9]);
        assert!(
            result
                .select_grid(
                    grid,
                    class,
                    SpatialOptions {
                        min_confidence: 0.9,
                        min_coverage: 0.0
                    }
                )
                .unwrap()
                .selected_indices
                .is_empty()
        );
    }
    #[test]
    fn class_aware_nms_retains_different_classes_and_distant_objects() {
        let result = nms(
            vec![
                object(1, 0.9, [0., 0., 10., 10.]),
                object(1, 0.8, [1., 1., 11., 11.]),
                object(2, 0.7, [0., 0., 10., 10.]),
                object(1, 0.6, [20., 20., 30., 30.]),
            ],
            0.45,
            100,
        );
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].confidence.to_bits(), 0.9_f32.to_bits());
        assert_eq!(nms(result, 0.45, 1).len(), 1);
    }
    #[test]
    fn unsupported_labels_and_malformed_detections_fail_explicitly() {
        assert_eq!(CocoClass::from_label("buses").unwrap().index(), 5);
        assert!(CocoClass::from_label("crosswalks").is_none());
        assert!(CocoClass::new(80).is_none());
        let class = CocoClass::new(1).unwrap();
        for object in [
            object(1, f32::NAN, [0., 0., 10., 10.]),
            object(1, 0.9, [-1., 0., 10., 10.]),
            object(1, 0.9, [10., 0., 0., 10.]),
        ] {
            assert!(
                DetectionResult {
                    width: 20,
                    height: 20,
                    objects: vec![object]
                }
                .select_grid(Grid::new(4, 4).unwrap(), class, SpatialOptions::default())
                .is_err()
            );
        }
        assert!(
            SpatialOptions {
                min_confidence: f32::INFINITY,
                min_coverage: 0.0
            }
            .validate()
            .is_err()
        );
    }
}
