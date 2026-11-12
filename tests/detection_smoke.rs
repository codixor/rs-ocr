use rs_ocr::{
    detection::{
        CocoClass, DetectionOptions, HybridSelection, HybridSelector, ObjectDetector,
        SpatialOptions,
    },
    images::selection::{Grid, ImageSelector, SelectionOptions},
};
use std::{env, fs, path::PathBuf};

#[test]
#[ignore = "requires pinned detector and CLIP reference photos; see docs/localization-benchmark.md"]
fn detects_known_cats_and_restores_grid_coordinates_from_files_and_bytes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = env::var_os("RS_OCR_DETECTOR_DIR")
        .map_or_else(|| root.join("models/yolo11s"), PathBuf::from);
    let detector = ObjectDetector::new(model.join("model.onnx")).unwrap();
    let photo = root.join("models/clip-vit-b32/test-images/sofa-cats.jpg");
    let result = detector
        .detect_file(&photo, DetectionOptions::default())
        .unwrap();
    let cat = CocoClass::from_label("cat").unwrap();
    let cats: Vec<_> = result
        .objects
        .iter()
        .filter(|object| object.class == cat)
        .collect();
    assert_eq!(cats.len(), 2, "objects: {:?}", result.objects);
    assert!(cats.iter().all(
        |object| f64::from(object.bounds.right) <= f64::from(result.width)
            && f64::from(object.bounds.bottom) <= f64::from(result.height)
    ));
    assert_eq!(
        result,
        detector
            .detect_bytes(&fs::read(photo).unwrap(), DetectionOptions::default())
            .unwrap()
    );
    assert_eq!(
        result
            .select_grid(Grid::new(1, 1).unwrap(), cat, SpatialOptions::default())
            .unwrap()
            .selected_indices,
        [0]
    );
    assert!(
        detector
            .detect_bytes(b"not an image", DetectionOptions::default())
            .is_err()
    );
    let classifier = ImageSelector::from_model_dir(root.join("models/clip-vit-b32")).unwrap();
    let selector = HybridSelector::new(detector, classifier);
    let photo = root.join("models/clip-vit-b32/test-images/sofa-cats.jpg");
    let result = selector
        .select_file(
            &photo,
            Grid::new(1, 1).unwrap(),
            &SelectionOptions::new("cats", vec!["roads".into()]),
            SpatialOptions::default(),
        )
        .unwrap();
    assert!(matches!(result, HybridSelection::Objects { .. }));
    assert_eq!(result.selected_indices(), [0]);
    let fallback = selector
        .select_bytes(
            &fs::read(photo).unwrap(),
            Grid::new(1, 1).unwrap(),
            &SelectionOptions::new("crosswalks", vec!["cats".into()]),
            SpatialOptions::default(),
        )
        .unwrap();
    assert!(matches!(fallback, HybridSelection::Classification(_)));
    assert_eq!(fallback.selected_indices(), [] as [usize; 0]);
}
