//! Classify a local image grid with CLIP and print zero-based selected indices.

use std::{env, error::Error, path::PathBuf};

use rs_ocr::images::selection::{Grid, ImageSelector, SelectionOptions};
use serde_json::Value;

fn profile_options(profile: &Value, target: &str) -> Result<SelectionOptions, Box<dyn Error>> {
    if profile["schema_version"].as_u64() != Some(1) {
        return Err("unsupported profile schema".into());
    }
    let labels = profile["labels"]
        .as_array()
        .ok_or("profile needs a fixed label set")?
        .iter()
        .map(|label| label.as_str().ok_or("invalid profile label"))
        .collect::<Result<Vec<_>, _>>()?;
    let target_index = labels
        .iter()
        .position(|label| label.trim().eq_ignore_ascii_case(target.trim()))
        .ok_or("target is outside the calibrated label set")?;
    let mut options = SelectionOptions::new(
        labels[target_index],
        labels
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != target_index)
            .map(|(_, label)| (*label).to_owned())
            .collect(),
    );
    options.min_score = serde_json::from_value(profile["thresholds"]["min_score"].clone())?;
    options.min_margin = serde_json::from_value(profile["thresholds"]["min_margin"].clone())?;
    options.validate()?;
    Ok(options)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1).peekable();
    let profile = if args.peek().is_some_and(|argument| argument == "--profile") {
        args.next();
        Some(args.next().ok_or("missing profile path")?)
    } else {
        None
    };
    let model_dir = args.next().ok_or(
        "usage: image_selection [--profile PATH] <model-dir> <image> <rows> <columns> <target> [<alternative>...]",
    )?;
    let image = args.next().ok_or("missing image path")?;
    let mut next_text = |name| {
        args.next()
            .ok_or(name)?
            .into_string()
            .map_err(|_| "arguments must be valid UTF-8")
    };
    let rows: u32 = next_text("missing row count")?.parse()?;
    let columns: u32 = next_text("missing column count")?.parse()?;
    let target = next_text("missing target label")?;
    let alternatives = args
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "labels must be valid UTF-8")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let grid = Grid::new(rows, columns)?;
    let options = if let Some(path) = profile {
        if !alternatives.is_empty() {
            return Err("profile fixes competing labels; do not supply alternatives".into());
        }
        profile_options(&serde_json::from_slice(&std::fs::read(path)?)?, &target)?
    } else {
        SelectionOptions::new(target, alternatives)
    };
    options.validate()?;
    let selector = ImageSelector::from_model_dir(PathBuf::from(model_dir))?;
    let result = selector.select_file(PathBuf::from(image), grid, &options)?;
    println!("selected: {:?}", result.selected_indices());
    for tile in result.tiles {
        println!(
            "tile={} row={} column={} target_score={:.4} margin={:.4} selected={} best={:?}",
            tile.index,
            tile.row,
            tile.column,
            tile.target_score,
            tile.margin,
            tile.selected,
            tile.scores[0].label
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn profile_keeps_the_calibrated_label_set_and_thresholds() {
        let profile = json!({"schema_version": 1, "labels": ["cars", "buses", "trees"],
                             "thresholds": {"min_score": 0.3, "min_margin": 0.1}});
        let options = profile_options(&profile, " BUSES ").unwrap();
        assert_eq!(options.target, "buses");
        assert_eq!(options.alternatives, ["cars", "trees"]);
        assert_eq!(options.min_score, 0.3);
        assert_eq!(options.min_margin, 0.1);
        assert!(profile_options(&profile, "horses").is_err());
    }

    #[test]
    fn invalid_profiles_are_rejected() {
        for profile in [
            json!({}),
            json!({"schema_version": 2, "labels": ["cars", "buses"], "thresholds": {"min_score": 0.3, "min_margin": 0.1}}),
            json!({"schema_version": 1, "labels": ["cars", " Cars "], "thresholds": {"min_score": 0.3, "min_margin": 0.1}}),
            json!({"schema_version": 1, "labels": ["cars", "buses"], "thresholds": {"min_score": 1.1, "min_margin": 0.1}}),
        ] {
            assert!(profile_options(&profile, "cars").is_err());
        }
    }
}
