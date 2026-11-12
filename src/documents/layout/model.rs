use super::{LayoutBox, LayoutClass, LayoutError, LayoutRegion, PageLayout, preprocess};
use rten::{Dimension, Model, NodeId};
use rten_tensor::{NdTensor, prelude::*};
use std::path::Path;

/// Post-filtering for the pinned graph's class-aware NMS (`IoU` 0.5, score 0.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutOptions {
    /// Retained region score, 0.3..=1. The graph cannot recover lower scores.
    pub min_confidence: f32,
    /// Maximum returned regions, 1..=100 (the graph's top-k limit).
    pub max_regions: usize,
}
impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_regions: 100,
        }
    }
}
impl LayoutOptions {
    /// Validate thresholds before reading images or running inference.
    /// # Errors
    /// Rejects non-finite scores, scores below the graph floor and invalid counts.
    pub fn validate(self) -> Result<(), LayoutError> {
        if !self.min_confidence.is_finite()
            || !(0.3..=1.0).contains(&self.min_confidence)
            || !(1..=100).contains(&self.max_regions)
        {
            return Err(LayoutError::InvalidInput(
                "invalid region threshold or limit",
            ));
        }
        Ok(())
    }
}

/// Reusable PP-DocLayout-S CPU session for the pinned Paddle-to-ONNX export.
/// This adapter handles its two inputs and all 23 official output classes.
pub struct LayoutDetector {
    model: Model,
    image: NodeId,
    scale_factor: NodeId,
    detections: NodeId,
    count: NodeId,
}

impl LayoutDetector {
    /// Load the pinned FP32 ONNX graph with image/scale-factor inputs and NMS.
    /// # Errors
    /// Returns loading/operator errors or an incompatible model signature.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, LayoutError> {
        let model = Model::load_file(path)?;
        let image = model.node_id("image")?;
        let scale_factor = model.node_id("scale_factor")?;
        let detections = model.node_id("fetch_name_0")?;
        let count = model.node_id("fetch_name_1")?;
        if model.input_ids().len() != 2
            || !model.input_ids().contains(&image)
            || !model.input_ids().contains(&scale_factor)
            || model.output_ids() != [detections, count]
        {
            return Err(LayoutError::ModelInterface(
                "expected pinned two-input/two-output graph",
            ));
        }
        let expected = [vec![0, 3, 480, 480], vec![0, 2]];
        for (index, id) in model.input_ids().iter().enumerate() {
            let shape = model
                .input_shape(index)
                .ok_or(LayoutError::ModelInterface("missing input shape"))?;
            let want = &expected[usize::from(*id != image)];
            if shape.len() != want.len()
                || shape
                    .iter()
                    .zip(want)
                    .enumerate()
                    .any(|(axis, (actual, &n))| match actual {
                        Dimension::Fixed(v) => *v != if axis == 0 { 1 } else { n },
                        Dimension::Symbolic(_) => axis != 0,
                    })
            {
                return Err(LayoutError::ModelInterface(
                    "expected [1,3,480,480] and [1,2]",
                ));
            }
        }
        Ok(Self {
            model,
            image,
            scale_factor,
            detections,
            count,
        })
    }

    /// Detect regions from bounded PNG/JPEG bytes, without running OCR.
    /// # Errors
    /// Returns invalid options, image limits/decoding, inference or output errors.
    pub fn detect_bytes(
        &self,
        bytes: &[u8],
        options: LayoutOptions,
    ) -> Result<PageLayout, LayoutError> {
        options.validate()?;
        let image = preprocess::decode(bytes)?;
        let pixels = preprocess::pixels(&image)?;
        let width = u16::try_from(image.width())
            .map_err(|_| LayoutError::InvalidInput("image width exceeds limits"))?;
        let height = u16::try_from(image.height())
            .map_err(|_| LayoutError::InvalidInput("image height exceeds limits"))?;
        let scale = NdTensor::from_data(
            [1, 2],
            vec![
                f32::from(preprocess::SIDE) / f32::from(height),
                f32::from(preprocess::SIDE) / f32::from(width),
            ],
        );
        let [rows, count] = self.model.run_n(
            vec![
                (self.image, pixels.view().into()),
                (self.scale_factor, scale.view().into()),
            ],
            [self.detections, self.count],
            None,
        )?;
        let rows: NdTensor<f32, 2> = rows
            .try_into()
            .map_err(|_| LayoutError::InvalidOutput("expected float32 [N,6]"))?;
        let count: NdTensor<i32, 1> = count
            .try_into()
            .map_err(|_| LayoutError::InvalidOutput("expected int32 [1] count"))?;
        if rows.shape()[1] != 6 || rows.shape()[0] > 2300 || count.shape() != [1] {
            return Err(LayoutError::InvalidOutput("invalid detection/count shape"));
        }
        let count = usize::try_from(count[[0]])
            .map_err(|_| LayoutError::InvalidOutput("negative region count"))?;
        if count > rows.shape()[0] || count > 100 {
            return Err(LayoutError::InvalidOutput(
                "region count exceeds tensor or graph limit",
            ));
        }
        let rows = (0..count)
            .map(|i| std::array::from_fn(|j| rows[[i, j]]))
            .collect::<Vec<_>>();
        decode_rows(&rows, width, height, options)
    }

    /// Read at most 32 MiB plus one overflow byte, then detect page regions.
    /// # Errors
    /// Returns I/O and the same validation/inference errors as [`Self::detect_bytes`].
    pub fn detect_file(
        &self,
        path: impl AsRef<Path>,
        options: LayoutOptions,
    ) -> Result<PageLayout, LayoutError> {
        options.validate()?;
        self.detect_bytes(&preprocess::read_file(path.as_ref())?, options)
    }
}

fn decode_rows(
    rows: &[[f32; 6]],
    width: u16,
    height: u16,
    options: LayoutOptions,
) -> Result<PageLayout, LayoutError> {
    let mut regions = Vec::new();
    for &[class, score, left, top, right, bottom] in rows {
        if [class, score, left, top, right, bottom]
            .iter()
            .any(|v| !v.is_finite())
            || !(0.0..=1.0).contains(&score)
            || right <= left
            || bottom <= top
        {
            return Err(LayoutError::InvalidOutput(
                "invalid region score or coordinates",
            ));
        }
        let class = (0_u8..23)
            .find(|&id| (class - f32::from(id)).abs() < f32::EPSILON)
            .and_then(LayoutClass::new)
            .ok_or(LayoutError::InvalidOutput(
                "non-integer or unknown layout class",
            ))?;
        if score < options.min_confidence {
            continue;
        }
        let bounds = LayoutBox {
            left: left.clamp(0.0, f32::from(width)),
            top: top.clamp(0.0, f32::from(height)),
            right: right.clamp(0.0, f32::from(width)),
            bottom: bottom.clamp(0.0, f32::from(height)),
        };
        if bounds.valid() {
            regions.push(LayoutRegion {
                class,
                confidence: score,
                bounds,
                raw_bounds: [left, top, right, bottom],
            });
        }
    }
    regions.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    regions.truncate(options.max_regions);
    Ok(PageLayout {
        width: u32::from(width),
        height: u32::from(height),
        regions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_all_official_classes_and_keeps_original_pixel_coordinates() {
        for (id, name) in (0_u8..23).zip(super::super::CLASS_NAMES) {
            assert_eq!(LayoutClass::new(id).unwrap().name(), name);
        }
        let spec: serde_json::Value = serde_json::from_str(include_str!(
            "../../../model-manifests/doc-layout-model.json"
        ))
        .unwrap();
        assert_eq!(
            spec["classes"],
            serde_json::json!(super::super::CLASS_NAMES)
        );
        assert!(LayoutClass::new(23).is_none());
        let result = decode_rows(
            &[
                [8.0, 0.7, -2.0, 20.0, 500.0, 800.0],
                [2.0, 0.9, 30.0, 100.0, 400.0, 700.0],
            ],
            600,
            900,
            LayoutOptions::default(),
        )
        .unwrap();
        assert_eq!(result.regions[0].class, LayoutClass::TEXT);
        assert!((result.regions[0].bounds.bottom - 700.0).abs() < f32::EPSILON);
        assert!(result.regions[1].bounds.left.abs() < f32::EPSILON);
        assert!(
            result.regions[1]
                .raw_bounds
                .iter()
                .zip([-2.0, 20.0, 500.0, 800.0])
                .all(|(a, b)| (a - b).abs() < f32::EPSILON)
        );
    }
    #[test]
    fn rejects_malformed_values_and_options_without_silently_skipping_them() {
        for class in [23.0, 2.1, -1.0, f32::NAN] {
            assert!(
                decode_rows(
                    &[[class, 0.8, 0.0, 0.0, 10.0, 20.0]],
                    100,
                    200,
                    LayoutOptions::default()
                )
                .is_err()
            );
        }
        for row in [
            [2.0, f32::NAN, 0.0, 0.0, 10.0, 20.0],
            [2.0, 1.1, 0.0, 0.0, 10.0, 20.0],
            [2.0, 0.8, 0.0, 0.0, -1.0, 20.0],
            [2.0, 0.8, 0.0, 0.0, f32::INFINITY, 20.0],
        ] {
            assert!(decode_rows(&[row], 100, 200, LayoutOptions::default()).is_err());
        }
        for min_confidence in [f32::NAN, 0.2, 1.1] {
            assert!(
                LayoutOptions {
                    min_confidence,
                    ..LayoutOptions::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            LayoutOptions {
                max_regions: 0,
                ..LayoutOptions::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            LayoutOptions {
                max_regions: 101,
                ..LayoutOptions::default()
            }
            .validate()
            .is_err()
        );
    }
}
