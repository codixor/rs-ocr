//! Read one page with the `RTen` backend and with the Burn backend, and compare text and time.
//!
//! Usage: `burn_compare <model-dir> <image> [runs] [cpu|vulkan:N]`

use std::{env, error::Error, path::PathBuf, time::Instant};

use rs_ocr::{
    documents::{Document, DocumentReader},
    engines::burn::{self as burn_engine, BurnDevice},
};

fn timed(
    reader: &mut DocumentReader,
    image: &PathBuf,
    runs: usize,
) -> Result<Document, Box<dyn Error>> {
    let mut last = None;
    for run in 0..runs {
        let start = Instant::now();
        last = Some(reader.read_file(image)?);
        eprintln!("  run {run}: {:?}", start.elapsed());
    }
    last.ok_or_else(|| "runs must be at least 1".into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: burn_compare <model-dir> <image> [runs] [cpu|vulkan:N]";
    let models = PathBuf::from(args.next().ok_or(usage)?);
    let image = PathBuf::from(args.next().ok_or(usage)?);
    let runs: usize = args.next().map_or(Ok(3), |r| r.to_string_lossy().parse())?;
    let device = match args
        .next()
        .map(|d| d.to_string_lossy().into_owned())
        .as_deref()
    {
        None | Some("cpu") => BurnDevice::Cpu,
        #[cfg(feature = "burn-vulkan")]
        Some(d) if d.starts_with("vulkan") => BurnDevice::Vulkan {
            gpu: d.strip_prefix("vulkan:").map_or(Ok(0), str::parse)?,
        },
        Some(other) => return Err(format!("unknown device {other}; {usage}").into()),
    };

    eprintln!("RTen (CPU):");
    let start = Instant::now();
    let mut rten = DocumentReader::from_model_dir(&models)?;
    eprintln!("  load: {:?}", start.elapsed());
    let rten_doc = timed(&mut rten, &image, runs)?;

    eprintln!("Burn ({device:?}):");
    let start = Instant::now();
    let mut burn = DocumentReader::from_engine(burn_engine::ppv6_tiny(&models, device)?);
    eprintln!("  load: {:?}", start.elapsed());
    let burn_doc = timed(&mut burn, &image, runs)?;

    println!(
        "--- RTen ({} spans)\n{}",
        rten_doc.spans().len(),
        rten_doc.raw_text()
    );
    println!(
        "--- Burn ({} spans)\n{}",
        burn_doc.spans().len(),
        burn_doc.raw_text()
    );
    println!(
        "--- text identical: {}",
        rten_doc.raw_text() == burn_doc.raw_text()
    );
    Ok(())
}
