//! OCR a printed invoice/receipt or cheque and inspect structured evidence.
use rs_ocr::{bills::BillReader, cheques::ChequeReader, documents::ExtractionOptions};
use std::{env, error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let mode = args
        .next()
        .ok_or("usage: documents <bill|cheque> <model-dir> <image>")?;
    let models = PathBuf::from(args.next().ok_or("missing model directory")?);
    let image = PathBuf::from(args.next().ok_or("missing image")?);
    if mode == "bill" {
        println!(
            "{:#?}",
            BillReader::from_model_dir(models)?.read_file(image, ExtractionOptions::default())?
        );
    } else if mode == "cheque" {
        println!(
            "{:#?}",
            ChequeReader::from_model_dir(models)?.read_file(image, ExtractionOptions::default())?
        );
    } else {
        return Err("expected bill or cheque".into());
    }
    Ok(())
}
