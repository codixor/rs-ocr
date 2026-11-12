//! JSONL full-image detector and CLIP comparison, reusing both Rust models.

use rs_ocr::{
    detection::{CocoClass, DetectionOptions, ObjectDetector, SpatialOptions},
    images::selection::{Grid, ImageSelector, SelectionOptions},
};
use serde_json::{Value, json};
use std::{
    env,
    error::Error,
    io::{self, BufRead},
    path::PathBuf,
};

fn text<'a>(request: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    request[key]
        .as_str()
        .ok_or_else(|| format!("missing string: {key}").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let clip =
        ImageSelector::from_model_dir(PathBuf::from(args.next().ok_or("missing CLIP directory")?))?;
    let detector = ObjectDetector::new(PathBuf::from(
        args.next().ok_or("missing detector ONNX file")?,
    ))?;
    let spatial_options = if let Some(path) = args.next() {
        let profile: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        SpatialOptions {
            min_confidence: serde_json::from_value(
                profile["thresholds"]["min_confidence"].clone(),
            )?,
            min_coverage: serde_json::from_value(profile["thresholds"]["min_coverage"].clone())?,
        }
    } else {
        SpatialOptions::default()
    };
    spatial_options.validate()?;
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let request: Value = serde_json::from_str(&line?)?;
        let rows = u32::try_from(request["rows"].as_u64().ok_or("invalid rows")?)?;
        let columns = u32::try_from(request["columns"].as_u64().ok_or("invalid columns")?)?;
        let alternatives = request["alternatives"]
            .as_array()
            .ok_or("missing alternatives")?
            .iter()
            .map(|value| value.as_str().map(str::to_owned).ok_or("invalid label"))
            .collect::<Result<Vec<_>, _>>()?;
        let options = SelectionOptions::new(text(&request, "target")?, alternatives);
        let result = clip.select_file(
            text(&request, "image")?,
            Grid::new(rows, columns)?,
            &options,
        )?;
        let objects = detector.detect_file(
            text(&request, "image")?,
            DetectionOptions {
                min_confidence: 0.05,
                ..DetectionOptions::default()
            },
        )?;
        let tiles: Vec<_> = result
            .tiles
            .iter()
            .map(|tile| {
                json!({"index": tile.index,
            "target_score": tile.target_score, "margin": tile.margin, "selected": tile.selected})
            })
            .collect();
        let detections: Vec<_> = objects.objects.iter().map(|object| json!({"class_id": object.class.index(),
            "confidence": object.confidence, "bounds": [object.bounds.left, object.bounds.top, object.bounds.right, object.bounds.bottom]})).collect();
        let spatial = CocoClass::from_label(&options.target)
            .map(|class| objects.select_grid(Grid::new(rows, columns)?, class, spatial_options))
            .transpose()?;
        println!(
            "{}",
            json!({"id": text(&request,"id")?, "tiles": tiles,
            "width": objects.width, "height": objects.height, "objects": detections,
            "spatial_selected": spatial.as_ref().map(|s| &s.selected_indices),
            "spatial_options": {"min_confidence": spatial_options.min_confidence, "min_coverage": spatial_options.min_coverage},
            "target_class": CocoClass::from_label(&options.target).map(CocoClass::index)})
        );
        if (index + 1) % 10 == 0 {
            eprintln!("evaluated {} spatial grids", index + 1);
        }
    }
    Ok(())
}
