use std::path::Path;

use rten::{Model, NodeId};
use rten_tensor::{NdTensor, prelude::*};
use rten_text::Tokenizer;

use super::preprocess::{IMAGE_SIZE, decode_image, preprocess, read_image_file};
use super::scoring::score_tiles;
use super::tokenizer::load_tokenizer;
use super::{Grid, SelectionError, SelectionOptions, SelectionResult};

struct ModelNodes {
    input_ids: NodeId,
    pixels: NodeId,
    attention_mask: Option<NodeId>,
    logits: NodeId,
}

/// Reusable CLIP ViT-B/32 sessions and tokenizer for local image classification.
///
/// Supports the documented combined ONNX export, with 224×224 RGB images and
/// a 77-token context. No files are downloaded during construction or inference.
pub struct ImageSelector {
    model: Model,
    tokenizer: Tokenizer,
    nodes: ModelNodes,
    start_token: i32,
    end_token: i32,
}

impl ImageSelector {
    /// Load a combined CLIP ONNX model and its matching tokenizer.
    ///
    /// # Errors
    /// Returns [`SelectionError`] for model/tokenizer loading failures or an
    /// incompatible model interface.
    pub fn new(
        model: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
    ) -> Result<Self, SelectionError> {
        let model = Model::load_file(model)?;
        let tokenizer = load_tokenizer(tokenizer.as_ref())?;
        let input_ids = model.node_id("input_ids")?;
        let pixels = model.node_id("pixel_values")?;
        if ![input_ids, pixels]
            .iter()
            .all(|id| model.input_ids().contains(id))
        {
            return Err(SelectionError::ModelInterface(
                "expected input_ids and pixel_values inputs",
            ));
        }
        let nodes = ModelNodes {
            input_ids,
            pixels,
            attention_mask: model
                .find_node("attention_mask")
                .filter(|id| model.input_ids().contains(id)),
            logits: model.node_id("logits_per_image")?,
        };
        let token = |text| {
            i32::try_from(tokenizer.get_token_id(text)?)
                .map_err(|_| SelectionError::ModelInterface("token id does not fit i32"))
        };
        let start_token = token("<|startoftext|>")?;
        let end_token = token("<|endoftext|>")?;
        Ok(Self {
            model,
            tokenizer,
            nodes,
            start_token,
            end_token,
        })
    }

    /// Load `model.onnx` and `tokenizer.json` from a local directory.
    ///
    /// # Errors
    /// Returns [`SelectionError`] if either required file cannot load.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self, SelectionError> {
        Self::new(
            directory.as_ref().join("model.onnx"),
            directory.as_ref().join("tokenizer.json"),
        )
    }

    /// Classify a PNG/JPEG grid from a local file.
    /// Tile indices are zero-based and ordered left-to-right, top-to-bottom.
    ///
    /// # Errors
    /// Returns [`SelectionError`] for invalid options, file/image errors, unsupported
    /// dimensions or captions, or failed inference.
    pub fn select_file(
        &self,
        path: impl AsRef<Path>,
        grid: Grid,
        options: &SelectionOptions,
    ) -> Result<SelectionResult, SelectionError> {
        options.validate()?;
        let path = path.as_ref();
        let bytes = read_image_file(path)?;
        self.select_bytes(&bytes, grid, options)
    }

    /// Classify an encoded PNG/JPEG grid retained by the caller.
    ///
    /// # Errors
    /// Returns [`SelectionError`] for invalid labels/options/images, oversized
    /// captions, or failed/incompatible inference.
    pub fn select_bytes(
        &self,
        bytes: &[u8],
        grid: Grid,
        options: &SelectionOptions,
    ) -> Result<SelectionResult, SelectionError> {
        options.validate()?;
        let image = decode_image(bytes)?;
        let regions = grid.regions(image.width(), image.height())?;
        let (input_ids, mask) = self.encode_labels(&options.labels())?;
        let mut all_logits = Vec::with_capacity(regions.len());
        // Cap image batches to bound intermediate activations on CPU.
        for chunk in regions.chunks(4) {
            let mut pixels = Vec::with_capacity(chunk.len() * 3 * IMAGE_SIZE * IMAGE_SIZE);
            for region in chunk {
                let tile = image.crop_imm(region.x, region.y, region.width, region.height);
                pixels.extend(preprocess(&tile)?);
            }
            let pixels = NdTensor::from_data([chunk.len(), 3, IMAGE_SIZE, IMAGE_SIZE], pixels);
            let mut inputs = vec![
                (self.nodes.input_ids, input_ids.view().into()),
                (self.nodes.pixels, pixels.view().into()),
            ];
            if let Some(node) = self.nodes.attention_mask {
                inputs.push((node, mask.view().into()));
            }
            let [output] = self.model.run_n(inputs, [self.nodes.logits], None)?;
            let logits: NdTensor<f32, 2> = output
                .try_into()
                .map_err(|_| SelectionError::InvalidOutput("expected float32 rank-2 logits"))?;
            if logits.shape() != [chunk.len(), options.alternatives.len() + 1] {
                return Err(SelectionError::InvalidOutput(
                    "logits dimensions differ from image/label count",
                ));
            }
            for row in 0..chunk.len() {
                all_logits.push(
                    (0..logits.size(1))
                        .map(|column| logits[[row, column]])
                        .collect(),
                );
            }
        }
        score_tiles(&all_logits, grid, regions, options)
    }

    fn encode_labels(
        &self,
        labels: &[&str],
    ) -> Result<(NdTensor<i32, 2>, NdTensor<i32, 2>), SelectionError> {
        let mut sequences = Vec::with_capacity(labels.len());
        for label in labels {
            let caption = format!("a photo of {label}");
            let tokens = self
                .tokenizer
                .encode(caption.as_str(), None)?
                .into_token_ids();
            if tokens.len() + 2 > 77 {
                return Err(SelectionError::LabelTooLong {
                    label: (*label).into(),
                    tokens: tokens.len() + 2,
                });
            }
            let mut sequence = vec![self.start_token];
            for token in tokens {
                sequence
                    .push(i32::try_from(token).map_err(|_| {
                        SelectionError::ModelInterface("token id does not fit i32")
                    })?);
            }
            sequence.push(self.end_token);
            sequences.push(sequence);
        }
        let width = sequences
            .iter()
            .map(Vec::len)
            .max()
            .ok_or(SelectionError::InvalidOptions("no labels"))?;
        let ids = NdTensor::from_fn([sequences.len(), width], |[row, column]| {
            sequences[row]
                .get(column)
                .copied()
                .unwrap_or(self.end_token)
        });
        let mask = NdTensor::from_fn([sequences.len(), width], |[row, column]| {
            i32::from(column < sequences[row].len())
        });
        Ok((ids, mask))
    }
}
