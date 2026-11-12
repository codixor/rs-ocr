use std::{
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

use image::{DynamicImage, ImageReader, imageops::FilterType};

use super::SelectionError;

pub(super) const IMAGE_SIZE: usize = 224;
const EDGE: u32 = 224;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_72];
const STD: [f32; 3] = [0.268_629_55, 0.261_302_6, 0.275_777_1];

pub(crate) fn read_image_file(path: &Path) -> Result<Vec<u8>, SelectionError> {
    let read = || -> std::io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(32 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    };
    read().map_err(|source| SelectionError::ImageIo {
        path: path.to_path_buf(),
        source,
    })
}

pub(crate) fn decode_image(bytes: &[u8]) -> Result<DynamicImage, SelectionError> {
    if bytes.is_empty() || bytes.len() > 32 * 1024 * 1024 {
        return Err(SelectionError::InvalidImage(
            "encoded image must contain 1 byte..=32 MiB",
        ));
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| SelectionError::InvalidImage("could not identify image format"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode()?)
}

// Match the pinned model's preprocessing: preserve aspect ratio while resizing
// the shortest side to 224, center-crop, rescale to [0,1], then normalize CHW.
pub(super) fn preprocess(image: &DynamicImage) -> Result<Vec<f32>, SelectionError> {
    let (width, height) = (image.width(), image.height());
    let shortest = width.min(height);
    if shortest == 0 {
        return Err(SelectionError::InvalidImage("image has zero area"));
    }
    let scaled = |value: u32| {
        u32::try_from(u64::from(value) * u64::from(EDGE) / u64::from(shortest))
            .map_err(|_| SelectionError::InvalidImage("resized image dimension overflow"))
    };
    let (scaled_width, scaled_height) = (scaled(width)?, scaled(height)?);
    if scaled_width > 8192 || scaled_height > 8192 {
        return Err(SelectionError::InvalidImage(
            "tile aspect ratio is too extreme",
        ));
    }
    let resized = image.resize_exact(scaled_width, scaled_height, FilterType::CatmullRom);
    let cropped = resized
        .crop_imm(
            (scaled_width - EDGE) / 2,
            (scaled_height - EDGE) / 2,
            EDGE,
            EDGE,
        )
        .to_rgb8();
    let mut pixels = Vec::with_capacity(3 * IMAGE_SIZE * IMAGE_SIZE);
    for channel in 0..3 {
        pixels.extend(
            cropped
                .pixels()
                .map(|pixel| (f32::from(pixel[channel]) / 255.0 - MEAN[channel]) / STD[channel]),
        );
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};

    #[test]
    fn rgb_channels_are_normalized_in_chw_order() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(224, 224, Rgb([255, 0, 0])));
        let pixels = preprocess(&image).unwrap();
        let plane = 224 * 224;
        assert_eq!(pixels.len(), 3 * plane);
        assert!((pixels[0] - 1.930_336_2).abs() < 0.00001);
        assert!((pixels[plane] + 1.752_097_1).abs() < 0.00001);
        assert!((pixels[plane * 2] + 1.480_219_8).abs() < 0.00001);
    }

    #[test]
    fn center_crop_does_not_squash_a_rectangular_image() {
        let mut image = RgbImage::from_pixel(448, 224, Rgb([0, 0, 0]));
        for y in 0..224 {
            for x in 112..336 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let pixels = preprocess(&DynamicImage::ImageRgb8(image)).unwrap();
        assert!(
            pixels[..224 * 224]
                .iter()
                .all(|value| (*value - 1.930_336_2).abs() < 0.00001)
        );
    }

    #[test]
    fn empty_or_extreme_aspect_ratios_return_errors() {
        for (width, height) in [(0, 1), (1, 0), (1, 10000)] {
            let image = DynamicImage::ImageRgb8(RgbImage::new(width, height));
            assert!(preprocess(&image).is_err());
        }
    }
}
