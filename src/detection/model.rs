use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use rten::{Dimension, Model, NodeId};
use rten_tensor::{NdTensor, prelude::*};

use crate::images::selection::preprocess::{decode_image, read_image_file};

use super::{BoundingBox, CocoClass, Detection, DetectionError, DetectionResult, geometry::nms};

/// Detection score threshold and class-aware NMS configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionOptions {
    /// Minimum score retained before NMS.
    pub min_confidence: f32,
    /// Maximum intersection-over-union before suppressing a same-class box.
    pub nms_iou: f32,
    /// Maximum returned objects, 1..=1000.
    pub max_detections: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.25,
            nms_iou: 0.45,
            max_detections: 100,
        }
    }
}

impl DetectionOptions {
    /// Check bounds before image loading/inference.
    ///
    /// # Errors
    /// Rejects non-finite thresholds, values outside 0..=1 and invalid limits.
    pub fn validate(self) -> Result<(), DetectionError> {
        if [self.min_confidence, self.nms_iou]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !(1..=1000).contains(&self.max_detections)
        {
            return Err(DetectionError::InvalidInput(
                "invalid detector thresholds or object limit",
            ));
        }
        Ok(())
    }
}

struct Letterbox {
    input_width: u16,
    input_height: u16,
    resized_width: u16,
    resized_height: u16,
    left: u16,
    top: u16,
    width: u16,
    height: u16,
}

impl Letterbox {
    fn new(
        width: u32,
        height: u32,
        input_width: usize,
        input_height: usize,
    ) -> Result<Self, DetectionError> {
        if width == 0 || height == 0 {
            return Err(DetectionError::InvalidInput("zero-area image"));
        }
        let pixel = |value| {
            u16::try_from(value).map_err(|_| {
                DetectionError::InvalidInput("pixel dimension exceeds supported range")
            })
        };
        let width = pixel(u64::from(width))?;
        let height = pixel(u64::from(height))?;
        let input_width = u16::try_from(input_width)
            .map_err(|_| DetectionError::InvalidInput("invalid detector width"))?;
        let input_height = u16::try_from(input_height)
            .map_err(|_| DetectionError::InvalidInput("invalid detector height"))?;
        if width > 8192
            || height > 8192
            || !(32..=1280).contains(&input_width)
            || !(32..=1280).contains(&input_height)
        {
            return Err(DetectionError::InvalidInput(
                "unsupported image/model dimensions",
            ));
        }
        // Exact positive rational rounding avoids unchecked float-to-pixel casts.
        let (resized_width, resized_height) = if u32::from(width) * u32::from(input_height)
            > u32::from(height) * u32::from(input_width)
        {
            (
                input_width,
                pixel(
                    (u64::from(height) * u64::from(input_width) + u64::from(width) / 2)
                        / u64::from(width),
                )?
                .max(1),
            )
        } else {
            (
                pixel(
                    (u64::from(width) * u64::from(input_height) + u64::from(height) / 2)
                        / u64::from(height),
                )?
                .max(1),
                input_height,
            )
        };
        Ok(Self {
            input_width,
            input_height,
            resized_width,
            resized_height,
            left: (input_width - resized_width) / 2,
            top: (input_height - resized_height) / 2,
            width,
            height,
        })
    }

    fn pixels(&self, image: &DynamicImage) -> NdTensor<f32, 4> {
        let image = image
            .resize_exact(
                u32::from(self.resized_width),
                u32::from(self.resized_height),
                FilterType::Triangle,
            )
            .to_rgb8();
        let mut canvas = RgbImage::from_pixel(
            u32::from(self.input_width),
            u32::from(self.input_height),
            Rgb([114, 114, 114]),
        );
        image::imageops::replace(
            &mut canvas,
            &image,
            i64::from(self.left),
            i64::from(self.top),
        );
        NdTensor::from_fn(
            [
                1,
                3,
                usize::from(self.input_height),
                usize::from(self.input_width),
            ],
            |[_, channel, y, x]| {
                f32::from(canvas.as_raw()[(y * usize::from(self.input_width) + x) * 3 + channel])
                    / 255.0
            },
        )
    }

    fn unmap(&self, xywh: [f32; 4]) -> Option<BoundingBox> {
        let [cx, cy, width, height] = xywh;
        if width <= 0.0 || height <= 0.0 {
            return None;
        }
        let scale_x = f32::from(self.width) / f32::from(self.resized_width);
        let scale_y = f32::from(self.height) / f32::from(self.resized_height);
        let bounds = BoundingBox {
            left: ((cx - width * 0.5 - f32::from(self.left)) * scale_x)
                .clamp(0.0, f32::from(self.width)),
            right: ((cx + width * 0.5 - f32::from(self.left)) * scale_x)
                .clamp(0.0, f32::from(self.width)),
            top: ((cy - height * 0.5 - f32::from(self.top)) * scale_y)
                .clamp(0.0, f32::from(self.height)),
            bottom: ((cy + height * 0.5 - f32::from(self.top)) * scale_y)
                .clamp(0.0, f32::from(self.height)),
        };
        bounds.valid().then_some(bounds)
    }
}

/// Reusable local YOLOv8/YOLO11 COCO detector with the documented raw ONNX output.
pub struct ObjectDetector {
    model: Model,
    input: NodeId,
    output: NodeId,
    input_width: usize,
    input_height: usize,
}

impl ObjectDetector {
    /// Load a local raw FP32 model with `images` input and `output0` output.
    /// Dynamic image dimensions use 640×640; fixed dimensions are respected.
    ///
    /// # Errors
    /// Returns an error for missing/incompatible files or unsupported input size.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, DetectionError> {
        let model = Model::load_file(path)?;
        let input = model.node_id("images")?;
        let output = model.node_id("output0")?;
        if model.input_ids() != [input] {
            return Err(DetectionError::ModelInterface(
                "expected exactly one images input",
            ));
        }
        let shape = model
            .input_shape(0)
            .ok_or(DetectionError::ModelInterface("missing input shape"))?;
        if shape.len() != 4
            || matches!(shape[0], Dimension::Fixed(value) if value != 1)
            || matches!(shape[1], Dimension::Fixed(value) if value != 3)
        {
            return Err(DetectionError::ModelInterface("expected NCHW RGB input"));
        }
        let dimension = |index| match shape[index] {
            Dimension::Fixed(value) => value,
            Dimension::Symbolic(_) => 640,
        };
        let (input_height, input_width) = (dimension(2), dimension(3));
        if !(32..=1280).contains(&input_width) || !(32..=1280).contains(&input_height) {
            return Err(DetectionError::ModelInterface(
                "expected spatial input dimensions in 32..=1280",
            ));
        }
        Ok(Self {
            model,
            input,
            output,
            input_width,
            input_height,
        })
    }

    /// Detect objects in local PNG/JPEG bytes.
    ///
    /// # Errors
    /// Rejects invalid options/images, inference errors and malformed outputs.
    pub fn detect_bytes(
        &self,
        bytes: &[u8],
        options: DetectionOptions,
    ) -> Result<DetectionResult, DetectionError> {
        options.validate()?;
        let image = decode_image(bytes)?;
        let transform = Letterbox::new(
            image.width(),
            image.height(),
            self.input_width,
            self.input_height,
        )?;
        let pixels = transform.pixels(&image);
        let [output] = self.model.run_n(
            vec![(self.input, pixels.view().into())],
            [self.output],
            None,
        )?;
        let output: NdTensor<f32, 3> = output
            .try_into()
            .map_err(|_| DetectionError::InvalidOutput("expected float32 rank-3 output"))?;
        let [batch, attributes, anchors] = output.shape();
        if batch != 1 || attributes != 84 || anchors == 0 || anchors > 40_000 {
            return Err(DetectionError::InvalidOutput(
                "expected [1,84,anchors] COCO output",
            ));
        }
        let mut objects = Vec::new();
        for anchor in 0..anchors {
            let xywh = std::array::from_fn(|index| output[[0, index, anchor]]);
            if xywh.iter().any(|value| !value.is_finite()) {
                return Err(DetectionError::InvalidOutput(
                    "non-finite object coordinates",
                ));
            }
            let mut best_class = 0_u8;
            let mut confidence = 0.0_f32;
            for class in 0_u8..80 {
                let score = output[[0, usize::from(class) + 4, anchor]];
                if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                    return Err(DetectionError::InvalidOutput(
                        "expected finite class probabilities",
                    ));
                }
                if score > confidence {
                    confidence = score;
                    best_class = class;
                }
            }
            if confidence < options.min_confidence {
                continue;
            }
            if let Some(bounds) = transform.unmap(xywh) {
                let class = CocoClass::new(best_class)
                    .ok_or(DetectionError::InvalidOutput("invalid COCO class"))?;
                objects.push(Detection {
                    class,
                    confidence,
                    bounds,
                });
            }
        }
        Ok(DetectionResult {
            width: image.width(),
            height: image.height(),
            objects: nms(objects, options.nms_iou, options.max_detections),
        })
    }

    /// Detect objects in a local encoded image, reading at most 32 MiB plus one byte.
    ///
    /// # Errors
    /// Returns image I/O, validation, inference or output errors.
    pub fn detect_file(
        &self,
        path: impl AsRef<Path>,
        options: DetectionOptions,
    ) -> Result<DetectionResult, DetectionError> {
        options.validate()?;
        let path = path.as_ref();
        let bytes = read_image_file(path)?;
        self.detect_bytes(&bytes, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn letterbox_preserves_aspect_ratio_and_restores_original_coordinates() {
        let transform = Letterbox::new(400, 200, 640, 640).unwrap();
        assert_eq!(
            (
                transform.resized_width,
                transform.resized_height,
                transform.top
            ),
            (640, 320, 160)
        );
        let bounds = transform.unmap([320., 320., 320., 160.]).unwrap();
        assert_eq!(
            bounds,
            BoundingBox {
                left: 100.,
                top: 50.,
                right: 300.,
                bottom: 150.
            }
        );
        assert!(transform.unmap([0., 0., 10., 10.]).is_none());
    }
    #[test]
    fn invalid_options_and_zero_dimensions_are_rejected() {
        assert!(
            DetectionOptions {
                nms_iou: f32::NAN,
                ..DetectionOptions::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            DetectionOptions {
                max_detections: 0,
                ..DetectionOptions::default()
            }
            .validate()
            .is_err()
        );
        assert!(Letterbox::new(0, 4, 640, 640).is_err());
    }
}
