use super::SelectionError;

/// Validated row/column layout with at most 64 tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    rows: u32,
    columns: u32,
    tile_count: usize,
}

/// A tile's rectangle in original image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRegion {
    /// Left edge, inclusive.
    pub x: u32,
    /// Top edge, inclusive.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Grid {
    /// Construct a positive grid containing no more than 64 tiles.
    ///
    /// # Errors
    /// Returns [`SelectionError::InvalidGrid`] for zero, excessive or overflowing dimensions.
    pub fn new(rows: u32, columns: u32) -> Result<Self, SelectionError> {
        let count = rows
            .checked_mul(columns)
            .filter(|count| (1..=64).contains(count))
            .ok_or(SelectionError::InvalidGrid("expected 1..=64 tiles"))?;
        let tile_count = usize::try_from(count)
            .map_err(|_| SelectionError::InvalidGrid("grid cannot fit this platform"))?;
        Ok(Self {
            rows,
            columns,
            tile_count,
        })
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(self) -> u32 {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub fn columns(self) -> u32 {
        self.columns
    }

    /// Number of tiles.
    #[must_use]
    pub fn tile_count(self) -> usize {
        self.tile_count
    }

    /// Split an image into row-major rectangles, covering every pixel once.
    /// Remainder pixels are distributed by proportional integer boundaries.
    ///
    /// # Errors
    /// Returns [`SelectionError::InvalidGrid`] if a tile would have zero area.
    pub fn regions(self, width: u32, height: u32) -> Result<Vec<TileRegion>, SelectionError> {
        if width < self.columns || height < self.rows {
            return Err(SelectionError::InvalidGrid(
                "image is smaller than the grid",
            ));
        }
        let boundary = |index: u32, pixels: u32, divisions: u32| {
            u32::try_from(u64::from(index) * u64::from(pixels) / u64::from(divisions))
                .map_err(|_| SelectionError::InvalidGrid("pixel boundary overflow"))
        };
        let mut regions = Vec::with_capacity(self.tile_count);
        for row in 0..self.rows {
            let y = boundary(row, height, self.rows)?;
            let bottom = boundary(row + 1, height, self.rows)?;
            for column in 0..self.columns {
                let x = boundary(column, width, self.columns)?;
                let right = boundary(column + 1, width, self.columns)?;
                regions.push(TileRegion {
                    x,
                    y,
                    width: right - x,
                    height: bottom - y,
                });
            }
        }
        Ok(regions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_sized_images_have_no_dropped_or_overlapping_pixels() {
        let regions = Grid::new(2, 3).unwrap().regions(7, 5).unwrap();
        assert_eq!(
            regions,
            vec![
                TileRegion {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2
                },
                TileRegion {
                    x: 2,
                    y: 0,
                    width: 2,
                    height: 2
                },
                TileRegion {
                    x: 4,
                    y: 0,
                    width: 3,
                    height: 2
                },
                TileRegion {
                    x: 0,
                    y: 2,
                    width: 2,
                    height: 3
                },
                TileRegion {
                    x: 2,
                    y: 2,
                    width: 2,
                    height: 3
                },
                TileRegion {
                    x: 4,
                    y: 2,
                    width: 3,
                    height: 3
                },
            ]
        );
        let mut coverage = [0_u8; 35];
        for region in regions {
            for y in region.y..region.y + region.height {
                for x in region.x..region.x + region.width {
                    coverage[(y * 7 + x) as usize] += 1;
                }
            }
        }
        assert!(coverage.into_iter().all(|count| count == 1));
    }

    #[test]
    fn zero_overflow_and_excessive_grids_are_rejected() {
        for (rows, columns) in [(0, 3), (3, 0), (u32::MAX, 2), (9, 8)] {
            assert!(Grid::new(rows, columns).is_err());
        }
        assert!(Grid::new(8, 8).is_ok());
    }

    #[test]
    fn an_image_must_have_at_least_one_pixel_per_tile() {
        let grid = Grid::new(3, 3).unwrap();
        assert!(grid.regions(2, 3).is_err());
        assert!(grid.regions(3, 2).is_err());
        assert_eq!(grid.regions(3, 3).unwrap().len(), 9);
    }
}
