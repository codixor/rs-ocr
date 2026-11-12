use std::collections::HashSet;

use super::{Grid, SelectionError, TileRegion};

/// A target label, competing labels, and selection thresholds.
///
/// Scores are relative to these labels. Include meaningful alternatives and a
/// background category; a confident winner is not proof of object presence.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionOptions {
    /// Object label to select, for example `horses`.
    pub target: String,
    /// One or more competing object/background labels; 32 total labels maximum.
    pub alternatives: Vec<String>,
    /// Minimum target softmax score, in `0.0..=1.0`.
    pub min_score: f32,
    /// Minimum difference between target score and strongest alternative.
    pub min_margin: f32,
}

impl SelectionOptions {
    /// Construct options with score threshold 0.5 and margin 0.1.
    /// Options are validated before image loading or inference.
    #[must_use]
    pub fn new(target: impl Into<String>, alternatives: Vec<String>) -> Self {
        Self {
            target: target.into(),
            alternatives,
            min_score: 0.5,
            min_margin: 0.1,
        }
    }

    /// Check labels and thresholds without loading a model.
    ///
    /// # Errors
    /// Returns [`SelectionError::InvalidOptions`] for empty/duplicate labels,
    /// fewer than two or more than 32 labels, or invalid thresholds.
    pub fn validate(&self) -> Result<(), SelectionError> {
        if self.alternatives.is_empty() || self.alternatives.len() >= 32 {
            return Err(SelectionError::InvalidOptions("supply 2..=32 total labels"));
        }
        for value in [self.min_score, self.min_margin] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(SelectionError::InvalidOptions(
                    "thresholds must be finite and in 0.0..=1.0",
                ));
            }
        }
        let mut seen = HashSet::new();
        for label in self.labels() {
            if label.is_empty() || !seen.insert(label.to_lowercase()) {
                return Err(SelectionError::InvalidOptions(
                    "labels must be non-empty and distinct ignoring case/outer whitespace",
                ));
            }
            if label.len() > 4096 {
                return Err(SelectionError::InvalidOptions(
                    "labels must not exceed 4096 UTF-8 bytes",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn labels(&self) -> Vec<&str> {
        std::iter::once(self.target.trim())
            .chain(self.alternatives.iter().map(|label| label.trim()))
            .collect()
    }
}

/// A label's score relative to all supplied labels for the same tile.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelScore {
    /// Supplied label, with outer whitespace removed.
    pub label: String,
    /// Softmax score; not a calibrated correctness probability.
    pub score: f32,
}

/// Classification and selection decision for one tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TilePrediction {
    /// Zero-based row-major tile index.
    pub index: usize,
    /// Zero-based row.
    pub row: usize,
    /// Zero-based column.
    pub column: usize,
    /// Rectangle in the original image.
    pub region: TileRegion,
    /// All label scores sorted highest first; ties preserve input label order.
    pub scores: Vec<LabelScore>,
    /// Target label's relative score.
    pub target_score: f32,
    /// Target score minus the highest alternative score; may be negative.
    pub margin: f32,
    /// Target wins strictly and meets both configured thresholds.
    pub selected: bool,
}

/// Complete classification of a local image grid.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionResult {
    /// Target label.
    pub target: String,
    /// Grid used for this request.
    pub grid: Grid,
    /// Predictions in zero-based row-major order.
    pub tiles: Vec<TilePrediction>,
}

impl SelectionResult {
    /// Indices of selected tiles in zero-based row-major order.
    #[must_use]
    pub fn selected_indices(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .filter(|tile| tile.selected)
            .map(|tile| tile.index)
            .collect()
    }
}

pub(super) fn softmax(logits: &[f32]) -> Result<Vec<f32>, SelectionError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(SelectionError::InvalidOutput(
            "expected finite non-empty logits",
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut scores: Vec<_> = logits.iter().map(|value| (*value - max).exp()).collect();
    let total: f32 = scores.iter().sum();
    for value in &mut scores {
        *value /= total;
    }
    Ok(scores)
}

pub(super) fn score_tiles(
    logits: &[Vec<f32>],
    grid: Grid,
    regions: Vec<TileRegion>,
    options: &SelectionOptions,
) -> Result<SelectionResult, SelectionError> {
    options.validate()?;
    let labels = options.labels();
    if logits.len() != grid.tile_count() || regions.len() != grid.tile_count() {
        return Err(SelectionError::InvalidOutput(
            "tile count differs from grid",
        ));
    }
    let columns = usize::try_from(grid.columns())
        .map_err(|_| SelectionError::InvalidGrid("invalid column count"))?;
    let mut tiles = Vec::with_capacity(logits.len());
    for (index, (values, region)) in logits.iter().zip(regions).enumerate() {
        if values.len() != labels.len() {
            return Err(SelectionError::InvalidOutput(
                "label count differs from logits",
            ));
        }
        let probabilities = softmax(values)?;
        let target_score = probabilities[0];
        let alternative = probabilities[1..].iter().copied().fold(0.0, f32::max);
        let margin = target_score - alternative;
        let selected = target_score > alternative
            && target_score >= options.min_score
            && margin >= options.min_margin;
        let mut scores: Vec<_> = labels
            .iter()
            .zip(probabilities)
            .map(|(label, score)| LabelScore {
                label: (*label).into(),
                score,
            })
            .collect();
        scores.sort_by(|left, right| right.score.total_cmp(&left.score));
        tiles.push(TilePrediction {
            index,
            row: index / columns,
            column: index % columns,
            region,
            scores,
            target_score,
            margin,
            selected,
        });
    }
    Ok(SelectionResult {
        target: options.target.trim().into(),
        grid,
        tiles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> SelectionOptions {
        SelectionOptions::new("horses", vec!["cats".into()])
    }

    #[test]
    fn scores_are_per_tile_and_preserve_row_major_indices() {
        let regions = Grid::new(1, 2).unwrap().regions(4, 2).unwrap();
        let result = score_tiles(
            &[vec![3.0, 0.0], vec![0.0, 3.0]],
            Grid::new(1, 2).unwrap(),
            regions,
            &options(),
        )
        .unwrap();
        assert_eq!(result.selected_indices(), vec![0]);
        assert_eq!(result.tiles[0].scores[0].label, "horses");
        assert_eq!(result.tiles[1].scores[0].label, "cats");
        assert!((result.tiles[0].target_score - 0.952_574_13).abs() < 0.000_001);
        assert_eq!(result.tiles[1].column, 1);
    }

    #[test]
    fn tied_or_insufficiently_separated_labels_do_not_select_a_tile() {
        let grid = Grid::new(1, 1).unwrap();
        for logits in [vec![1.0, 1.0], vec![1.01, 1.0]] {
            let result =
                score_tiles(&[logits], grid, grid.regions(2, 2).unwrap(), &options()).unwrap();
            assert!(result.selected_indices().is_empty());
        }
        let mut zero_margin = options();
        zero_margin.min_margin = 0.0;
        zero_margin.min_score = 0.0;
        assert!(
            score_tiles(
                &[vec![1.0, 1.0]],
                grid,
                grid.regions(2, 2).unwrap(),
                &zero_margin
            )
            .unwrap()
            .selected_indices()
            .is_empty()
        );
    }

    #[test]
    fn the_score_threshold_is_enforced_even_when_the_target_wins() {
        let mut options = options();
        options.min_score = 0.99;
        let grid = Grid::new(1, 1).unwrap();
        assert!(
            score_tiles(
                &[vec![3.0, 0.0]],
                grid,
                grid.regions(2, 2).unwrap(),
                &options
            )
            .unwrap()
            .selected_indices()
            .is_empty()
        );
    }

    #[test]
    fn softmax_is_stable_for_large_logits_and_rejects_non_finite_values() {
        let scores = softmax(&[10_000.0, 10_000.0]).unwrap();
        assert_eq!(scores, vec![0.5, 0.5]);
        for values in [
            vec![],
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 0.0],
            vec![-f32::INFINITY, 0.0],
        ] {
            assert!(softmax(&values).is_err());
        }
    }

    #[test]
    fn label_and_grid_output_shapes_are_checked() {
        let grid = Grid::new(1, 1).unwrap();
        assert!(score_tiles(&[], grid, grid.regions(2, 2).unwrap(), &options()).is_err());
        assert!(score_tiles(&[vec![1.0]], grid, grid.regions(2, 2).unwrap(), &options()).is_err());
        assert!(score_tiles(&[vec![1.0, 0.0]], grid, vec![], &options()).is_err());
    }

    #[test]
    fn a_target_without_alternatives_is_not_accepted_as_certain() {
        assert!(SelectionOptions::new("horses", vec![]).validate().is_err());
        assert!(
            SelectionOptions::new("horses", vec![" Horses ".into()])
                .validate()
                .is_err()
        );
        assert!(
            SelectionOptions::new(" ", vec!["cats".into()])
                .validate()
                .is_err()
        );
        assert!(
            SelectionOptions::new("horses", vec![String::new()])
                .validate()
                .is_err()
        );
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let mut invalid = options();
            invalid.min_score = value;
            assert!(invalid.validate().is_err());
            invalid.min_score = 0.5;
            invalid.min_margin = value;
            assert!(invalid.validate().is_err());
        }
        assert!(options().validate().is_ok());
    }
}
