//! Print every recognized line of a page, in reading order, one per line.
//!
//! Usage: `read_page <model-dir> <image>`; the model directory holds `det.rten`,
//! `rec.rten` and `dict.txt`. Timing goes to stderr.

use std::{env, error::Error, path::PathBuf, time::Instant};

use rs_ocr::documents::DocumentReader;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: read_page <model-dir> <image>";
    let models = PathBuf::from(args.next().ok_or(usage)?);
    let image = PathBuf::from(args.next().ok_or(usage)?);
    let mut reader = DocumentReader::from_model_dir(models)?;
    let start = Instant::now();
    let document = reader.read_file(image)?;
    eprintln!("read in {:?}", start.elapsed());
    for span in document.spans() {
        println!("{}", span.text);
    }
    Ok(())
}
