#![doc = include_str!("../README.md")]

pub mod bidi;
pub mod captch;

pub mod bills;
pub mod cheques;
pub mod documents;

/// Inference backends other than the default `RTen` one.
pub mod engines;

/// Image classification helpers.
pub mod images;

#[cfg(feature = "object-detection")]
pub mod detection;

/// `RustO` model configuration, including presets for different model families.
pub use rusto::InitializeConfig;
