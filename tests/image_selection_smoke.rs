//! Explicit real-model inference; weights and reference photos remain external.

use std::{io::Cursor, path::PathBuf};

use image::{
    DynamicImage, ImageFormat, RgbImage,
    imageops::{FilterType, replace},
};
use rs_ocr::images::selection::{Grid, ImageSelector, SelectionError, SelectionOptions};

#[test]
#[ignore = "requires pinned CLIP files and reference photos; see docs/image-selection.md"]
fn classifies_known_photos_and_selects_row_major_tiles_from_files_and_bytes() {
    let directory = std::env::var_os("RS_OCR_CLIP_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/clip-vit-b32"),
        PathBuf::from,
    );
    let selector =
        ImageSelector::from_model_dir(&directory).expect("load pinned CLIP model and tokenizer");
    let horses = directory.join("test-images/horses.jpeg");
    let cats = directory.join("test-images/sofa-cats.jpg");
    let options = SelectionOptions::new("horses", vec!["cats".into()]);
    let single = Grid::new(1, 1).unwrap();
    assert_eq!(
        selector
            .select_file(&horses, single, &options)
            .unwrap()
            .selected_indices(),
        vec![0]
    );
    assert_eq!(
        selector
            .select_file(&cats, single, &options)
            .unwrap()
            .selected_indices(),
        [] as [usize; 0]
    );

    let horses = image::open(horses)
        .unwrap()
        .resize_to_fill(224, 224, FilterType::CatmullRom)
        .to_rgb8();
    let cats = image::open(cats)
        .unwrap()
        .resize_to_fill(224, 224, FilterType::CatmullRom)
        .to_rgb8();
    let mut grid_image = RgbImage::new(672, 672);
    for index in 0..9 {
        replace(
            &mut grid_image,
            if index % 2 == 0 { &horses } else { &cats },
            (index % 3) * 224,
            (index / 3) * 224,
        );
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(grid_image)
        .write_to(&mut bytes, ImageFormat::Png)
        .unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), bytes.get_ref()).unwrap();
    let grid = Grid::new(3, 3).unwrap();
    let from_file = selector.select_file(file.path(), grid, &options).unwrap();
    let from_bytes = selector
        .select_bytes(bytes.get_ref(), grid, &options)
        .unwrap();
    assert_eq!(from_file.selected_indices(), vec![0, 2, 4, 6, 8]);
    assert_eq!(from_bytes.selected_indices(), vec![0, 2, 4, 6, 8]);
    assert_eq!(from_file, from_bytes);
    assert_eq!(from_file.tiles.len(), 9);
    for tile in &from_file.tiles {
        assert!((tile.scores.iter().map(|score| score.score).sum::<f32>() - 1.0).abs() < 0.00001);
    }
    let inverse = SelectionOptions::new("cats", vec!["horses".into()]);
    assert_eq!(
        selector
            .select_bytes(bytes.get_ref(), grid, &inverse)
            .unwrap()
            .selected_indices(),
        vec![1, 3, 5, 7]
    );
    assert!(selector.select_bytes(&[], grid, &options).is_err());
    assert!(
        selector
            .select_bytes(b"not an image", grid, &options)
            .is_err()
    );
    let long = SelectionOptions::new("horse ".repeat(100), vec!["cats".into()]);
    assert!(matches!(
        selector.select_bytes(bytes.get_ref(), grid, &long),
        Err(SelectionError::LabelTooLong { .. })
    ));
}
