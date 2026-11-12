use super::LayoutError;
use image::{DynamicImage, GenericImageView, ImageReader, Limits};
use rten_tensor::NdTensor;
use std::{
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

pub(super) const SIDE: u16 = 480;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn read_file(path: &Path) -> Result<Vec<u8>, LayoutError> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(super) fn decode(bytes: &[u8]) -> Result<DynamicImage, LayoutError> {
    if bytes.is_empty() || bytes.len() > 32 * 1024 * 1024 {
        return Err(LayoutError::InvalidInput(
            "encoded image must be 1 byte..=32 MiB",
        ));
    }
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
}

fn cubic(distance: f32) -> f32 {
    // OpenCV INTER_CUBIC uses a=-0.75. This differs from image's CatmullRom.
    let x = distance.abs();
    if x <= 1.0 {
        (1.25 * x - 2.25) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0
    } else {
        0.0
    }
}

#[allow(clippy::cast_possible_truncation)]
fn coordinate(value: f32) -> i16 {
    // Only half-pixel source positions for edges <=8192 reach this helper.
    value.floor() as i16
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn channel(value: f32) -> u8 {
    // Cubic overshoot is saturated before rounding to the encoded RGB range.
    value.clamp(0.0, 255.0).round() as u8
}

fn axis(size: u16) -> Result<Vec<[(u32, f32); 4]>, LayoutError> {
    let scale = f32::from(size) / f32::from(SIDE);
    (0..SIDE)
        .map(|out| {
            let position = (f32::from(out) + 0.5) * scale - 0.5;
            let base = coordinate(position);
            let fraction = position - f32::from(base);
            let mut taps = [(0, 0.0); 4];
            for (tap, offset) in taps.iter_mut().zip(-1_i16..=2) {
                let index = (i32::from(base) + i32::from(offset)).clamp(0, i32::from(size) - 1);
                *tap = (
                    u32::try_from(index)
                        .map_err(|_| LayoutError::InvalidInput("invalid resize index"))?,
                    cubic(fraction - f32::from(offset)),
                );
            }
            Ok(taps)
        })
        .collect()
}

pub(super) fn pixels(image: &DynamicImage) -> Result<NdTensor<f32, 4>, LayoutError> {
    let (width, height) = (u16::try_from(image.width()), u16::try_from(image.height()));
    let (Ok(width), Ok(height)) = (width, height) else {
        return Err(LayoutError::InvalidInput("unsupported image dimensions"));
    };
    if width == 0 || height == 0 || width > 8192 || height > 8192 {
        return Err(LayoutError::InvalidInput(
            "image dimensions must be 1..=8192",
        ));
    }
    let (xs, ys) = (axis(width)?, axis(height)?);
    // Read RGB components on demand. Expanding a full 48 MP grayscale scan to
    // RGB would allocate another 144 MB just to create a 480x480 model input.
    Ok(NdTensor::from_fn(
        [1, 3, usize::from(SIDE), usize::from(SIDE)],
        |[_, c, y, x]| {
            let mut value = 0.0;
            for &(sy, wy) in &ys[y] {
                for &(sx, wx) in &xs[x] {
                    value += f32::from(image.get_pixel(sx, sy)[c]) * wx * wy;
                }
            }
            (f32::from(channel(value)) / 255.0 - MEAN[c]) / STD[c]
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rten_tensor::prelude::*;
    #[test]
    fn rgb_normalization_and_non_square_resize_preserve_channels() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            20,
            60,
            image::Rgb([255, 128, 0]),
        ));
        let input = pixels(&image).unwrap();
        assert_eq!(input.shape(), [1, 3, 480, 480]);
        for (c, expected) in [
            (0, (1.0 - 0.485) / 0.229),
            (1, (128.0 / 255.0 - 0.456) / 0.224),
            (2, -0.406 / 0.225),
        ] {
            assert!((input[[0, c, 0, 0]] - expected).abs() < 1e-6);
            assert!((input[[0, c, 479, 479]] - expected).abs() < 1e-6);
        }
        assert!(decode(b"invalid PNG").is_err());
        assert!(decode(&[]).is_err());
        assert!(pixels(&DynamicImage::new_rgb8(8193, 1)).is_err());
    }
    #[test]
    fn cubic_resize_agrees_with_independent_opencv_reference_pixels() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/doc_layout_cubic.json"
        ))
        .unwrap();
        let image = image::RgbImage::from_fn(997, 541, |x, y| {
            image::Rgb([
                u8::try_from((13 * x + 7 * y) % 256).unwrap(),
                u8::try_from((3 * x + 17 * y) % 256).unwrap(),
                u8::try_from((29 * x + 5 * y) % 256).unwrap(),
            ])
        });
        let result = pixels(&DynamicImage::ImageRgb8(image)).unwrap();
        for point in fixture["points"].as_array().unwrap() {
            let x = usize::try_from(point["x"].as_u64().unwrap()).unwrap();
            let y = usize::try_from(point["y"].as_u64().unwrap()).unwrap();
            for c in 0..3 {
                let expected = f32::from(u8::try_from(point["rgb"][c].as_u64().unwrap()).unwrap());
                let actual = (result[[0, c, y, x]] * STD[c] + MEAN[c]) * 255.0;
                // OpenCV integer coefficient rounding can differ by one RGB unit.
                assert!(
                    (actual - expected).abs() < 1.01,
                    "channel {c} at {x},{y}: {actual} vs {expected}"
                );
            }
        }
    }
}
