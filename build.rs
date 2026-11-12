//! With the `burn` feature, converts the PP-OCRv6 Tiny ONNX models to Burn code with
//! burn-onnx. Without it, this script does nothing and `RTen` remains the only backend.

fn main() {
    #[cfg(feature = "burn")]
    burn_models::generate();
}

#[cfg(feature = "burn")]
mod burn_models {
    use std::path::PathBuf;

    use burn_onnx::ModelGen;

    /// Files are fetched once from Hugging Face `byrizki/rusto-models` (see README).
    const MODELS: [(&str, &str); 2] = [("det.onnx", "det"), ("rec.onnx", "rec")];

    pub fn generate() {
        println!("cargo:rerun-if-env-changed=RS_OCR_ONNX_DIR");
        let dir = std::env::var_os("RS_OCR_ONNX_DIR").map_or_else(
            || {
                PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"))
                    .join("models/onnx/ppocrv6-tiny")
            },
            PathBuf::from,
        );
        for (file, name) in MODELS {
            let input = dir.join(file);
            assert!(
                input.is_file(),
                "the `burn` feature needs {} (PP-OCRv6 Tiny ONNX). Download it from \
                 https://huggingface.co/byrizki/rusto-models/resolve/main/ppocrv6-tiny/{file} \
                 or set RS_OCR_ONNX_DIR",
                input.display()
            );
            println!("cargo:rerun-if-changed={}", input.display());
            ModelGen::new()
                .input(input.to_str().expect("UTF-8 ONNX path"))
                .out_dir(&format!("burn/{name}/"))
                .run_from_script();
        }
    }
}
