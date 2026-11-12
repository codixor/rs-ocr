//! Optional Burn backend for the PP-OCRv6 Tiny detector and recognizer.
//!
//! The ONNX models are converted to Rust at build time by burn-onnx and their weights are
//! embedded in the binary, so at run time only `dict.txt` is read from the model directory.
//! Pre/post-processing (resize, DB box extraction, CTC decoding) stays in `RustO`; only the
//! two network forward passes run on Burn, so results match the `RTen` backend.
//!
//! On the CPU, `RTen` is faster than Burn for these small models (measured on an `i9-9820X`:
//! detector 55 ms vs 165 ms, recognizer 4 ms vs 16 ms per line). Use this backend for the
//! optional GPU device (`burn-vulkan`) or to compare backends. A Vulkan device compiles its
//! shaders on first use (tens of seconds on an RX 580) and again for every new text-line
//! width, so it only pays off in a long-running process.

use std::{collections::HashMap, path::Path};

use ::burn::prelude::*;
use ::burn::tensor::TensorData;
use burn_store::{BurnpackStore, ModuleSnapshot};
use ndarray::{ArrayD, IxDyn};
use rusto::{EngineError, InferenceSession, InitializeConfig, RustO};

#[allow(clippy::all, clippy::pedantic, unused, unused_qualifications)]
mod det_model {
    include!(concat!(env!("OUT_DIR"), "/burn/det/det.rs"));
}

#[allow(clippy::all, clippy::pedantic, unused, unused_qualifications)]
mod rec_model {
    include!(concat!(env!("OUT_DIR"), "/burn/rec/rec.rs"));
}

static DET_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/burn/det/det.bpk"));
static REC_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/burn/rec/rec.bpk"));

/// Where the Burn networks run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BurnDevice {
    /// Burn's pure-Rust CPU backend (Flex).
    #[default]
    Cpu,
    /// A discrete GPU through Vulkan; `gpu` is its index among discrete GPUs (0 = first).
    #[cfg(feature = "burn-vulkan")]
    Vulkan {
        /// Index of the discrete GPU.
        gpu: usize,
    },
}

impl BurnDevice {
    fn device(self) -> Device {
        match self {
            Self::Cpu => Device::flex(),
            #[cfg(feature = "burn-vulkan")]
            Self::Vulkan { gpu } => Device::vulkan(burn::tensor::DeviceKind::DiscreteGpu(gpu)),
        }
    }
}

/// Build a PP-OCRv6 Tiny `RustO` engine whose networks run on Burn.
///
/// `directory` must contain `dict.txt`; the `.rten` files are not read.
///
/// # Errors
/// Returns [`EngineError`] if the embedded weights or the dictionary cannot load.
pub fn ppv6_tiny(directory: impl AsRef<Path>, device: BurnDevice) -> Result<RustO, EngineError> {
    let dir = directory.as_ref();
    // Model paths are part of the preset but unused: custom sessions replace RTen loading.
    let config = InitializeConfig::ppv6(
        dir.join("det.rten"),
        dir.join("rec.rten"),
        dir.join("dict.txt"),
    );
    let device = device.device();
    let mut det = det_model::Model::new(&device);
    det.load_from(&mut BurnpackStore::from_static(DET_WEIGHTS))
        .map_err(|e| EngineError::Engine(format!("Burn detector weights: {e}")))?;
    let mut rec = rec_model::Model::new(&device);
    rec.load_from(&mut BurnpackStore::from_static(REC_WEIGHTS))
        .map_err(|e| EngineError::Engine(format!("Burn recognizer weights: {e}")))?;
    RustO::with_custom_engines(
        config,
        DetSession {
            model: det,
            device: device.clone(),
        },
        RecSession { model: rec, device },
    )
}

fn to_tensor(input: &ArrayD<f32>, device: &Device) -> Result<Tensor<4>, EngineError> {
    let shape: [usize; 4] = input
        .shape()
        .try_into()
        .map_err(|_| EngineError::InvalidInputShape)?;
    // `iter` walks logical (row-major) order whatever the array's memory layout.
    let data: Vec<f32> = input.iter().copied().collect();
    Ok(Tensor::from_data(TensorData::new(data, shape), device))
}

fn to_array<const D: usize>(output: Tensor<D>) -> Result<ArrayD<f32>, EngineError> {
    let dims = output.dims().to_vec();
    let values = output
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| EngineError::OutputError(format!("Burn output: {e:?}")))?;
    Ok(ArrayD::from_shape_vec(IxDyn(&dims), values)?)
}

macro_rules! session {
    ($name:ident, $model:path) => {
        struct $name {
            model: $model,
            device: Device,
        }

        impl InferenceSession for $name {
            fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
                to_array(self.model.forward(to_tensor(&input, &self.device)?))
            }

            fn run_all(
                &mut self,
                input: ArrayD<f32>,
            ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
                Ok(HashMap::from([("output".to_owned(), self.run(input)?)]))
            }

            fn run_with_inputs(
                &mut self,
                inputs: HashMap<String, ArrayD<f32>>,
            ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
                let mut values = inputs.into_values();
                match (values.next(), values.next()) {
                    (Some(input), None) => self.run_all(input),
                    _ => Err(EngineError::Engine(
                        "expected exactly one input tensor".into(),
                    )),
                }
            }
        }
    };
}

session!(DetSession, det_model::Model);
session!(RecSession, rec_model::Model);
