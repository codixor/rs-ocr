//! Reuse one Rust model for JSONL offline evaluation requests on standard input.

use std::{env, error::Error, io, io::BufRead, path::PathBuf};

use rs_ocr::images::selection::{Grid, ImageSelector, SelectionOptions};
use serde_json::{Value, json};

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("missing string: {key}").into())
}

fn dimension(value: &Value, key: &str) -> Result<u32, Box<dyn Error>> {
    Ok(u32::try_from(
        value[key].as_u64().ok_or("missing positive dimension")?,
    )?)
}

fn evaluate(selector: &ImageSelector, request: &Value) -> Result<Value, Box<dyn Error>> {
    let grid = Grid::new(dimension(request, "rows")?, dimension(request, "columns")?)?;
    let alternatives = request["alternatives"]
        .as_array()
        .ok_or("missing alternative labels")?
        .iter()
        .map(|label| label.as_str().map(str::to_owned).ok_or("invalid label"))
        .collect::<Result<Vec<_>, _>>()?;
    let options = SelectionOptions::new(text(request, "target")?, alternatives);
    let result = selector.select_file(text(request, "image")?, grid, &options)?;
    let tiles: Vec<_> = result
        .tiles
        .iter()
        .map(|tile| {
            json!({
                "index": tile.index,
                "target_score": tile.target_score,
                "margin": tile.margin,
                "selected": tile.selected,
                "best_label": tile.scores[0].label,
            })
        })
        .collect();
    Ok(json!({"id": text(request, "id")?, "tiles": tiles}))
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_dir = env::args_os()
        .nth(1)
        .ok_or("usage: selection_batch <model-dir> < requests.jsonl > predictions.jsonl")?;
    let selector = ImageSelector::from_model_dir(PathBuf::from(model_dir))?;
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let request: Value = serde_json::from_str(&line?)?;
        // Fail closed instead of silently dropping failed examples from metrics.
        let result = evaluate(&selector, &request)?;
        println!("{result}");
        if (index + 1) % 10 == 0 {
            eprintln!("evaluated {} grids", index + 1);
        }
    }
    Ok(())
}
