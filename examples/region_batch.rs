//! Caller-selected single-line Tiny OCR diagnostics, separate from page extraction.
use rs_ocr::documents::{ImageRegion, RegionReader};
use serde_json::{Value, json};
use std::{
    env,
    error::Error,
    io::{self, BufRead},
};

fn process(job: &Value, reader: &mut RegionReader) -> Result<Value, Box<dyn Error>> {
    let region = job["region"].as_array().ok_or("missing region")?;
    if region.len() != 4 {
        return Err("expected x,y,width,height".into());
    }
    let values = region
        .iter()
        .map(|v| {
            u32::try_from(v.as_u64().ok_or("invalid coordinate")?)
                .map_err(|_| "coordinate overflow")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let region = ImageRegion {
        x: values[0],
        y: values[1],
        width: values[2],
        height: values[3],
    };
    let doc = reader.read_file(job["image"].as_str().ok_or("missing image")?, region)?;
    let spans: Vec<_> = doc
        .spans()
        .iter()
        .map(|s| json!({"text":s.text,"confidence":s.confidence,"box_points":s.box_points}))
        .collect();
    Ok(json!({"id":job["id"],"region":job["region"],"spans":spans}))
}

fn main() -> Result<(), Box<dyn Error>> {
    let directory = env::args()
        .nth(1)
        .ok_or("usage: region_batch MODEL_DIR < jobs.jsonl")?;
    let mut reader = RegionReader::from_model_dir(directory)?;
    for line in io::stdin().lock().lines() {
        let job: Value = serde_json::from_str(&line?)?;
        let result = process(&job, &mut reader)
            .unwrap_or_else(|e| json!({"id":job["id"],"error":e.to_string()}));
        println!("{result}");
    }
    Ok(())
}
