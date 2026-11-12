//! Optional inference engines. `RTen` (through `RustO`) is always available; the
//! modules here add alternatives behind Cargo features.

#[cfg(feature = "burn")]
pub mod burn;
