//! Decode a local image using externally supplied `RustO` Tiny models.

use std::{env, error::Error, path::PathBuf};

use rs_ocr::captch::{CaptchaDecoder, DecodeOptions};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let Some(models) = arguments.next() else {
        return Err("usage: captch <model-directory> <image.png> [expected-length]".into());
    };
    let image = arguments.next().ok_or("missing image path")?;
    let expected_length = arguments
        .next()
        .map(|value| {
            value
                .to_str()
                .ok_or("expected-length must be UTF-8")?
                .parse::<usize>()
                .map_err(|_| "expected-length must be a positive integer")
        })
        .transpose()?;
    if arguments.next().is_some() {
        return Err("too many arguments".into());
    }
    let options = DecodeOptions {
        expected_length,
        ..DecodeOptions::default()
    };
    options.validate()?;
    let mut decoder = CaptchaDecoder::from_model_dir(PathBuf::from(models))?;
    let result = decoder.decode_file(PathBuf::from(image), &options)?;
    println!("{}", result.text);
    eprintln!("confidence: {:.4}", result.confidence);
    Ok(())
}
