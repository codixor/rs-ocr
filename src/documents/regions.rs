use super::{Document, DocumentError, TextSpan};
use image::{ImageReader, Limits};
use rusto::{InitializeConfig, RecConfig, TextRecognizer, image_impl::Mat};
use std::{
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

/// Caller-selected rectangle in original image pixels, with exclusive right/bottom.
/// This is a field-location hint, not an automatically detected handwriting box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRegion {
    /// Left coordinate.
    pub x: u32,
    /// Top coordinate.
    pub y: u32,
    /// Positive width.
    pub width: u32,
    /// Positive height.
    pub height: u32,
}

impl ImageRegion {
    fn validate(self, width: u32, height: u32) -> Result<(), DocumentError> {
        if self.width == 0
            || self.height == 0
            || u64::from(self.width) > u64::from(self.height) * 64
            || self.x.checked_add(self.width).is_none_or(|x| x > width)
            || self.y.checked_add(self.height).is_none_or(|y| y > height)
        {
            return Err(DocumentError::InvalidInput(
                "empty or out-of-bounds image region",
            ));
        }
        Ok(())
    }
}

fn crop(bytes: &[u8], region: ImageRegion) -> Result<image::DynamicImage, DocumentError> {
    if bytes.is_empty() || bytes.len() > 32 * 1024 * 1024 {
        return Err(DocumentError::InvalidInput(
            "encoded image must be 1 byte..=32 MiB",
        ));
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode()?;
    region.validate(image.width(), image.height())?;
    Ok(image.crop_imm(region.x, region.y, region.width, region.height))
}

/// Reusable single-line recognizer using the same `RustO`/`RTen` weights.
///
/// A caller-supplied crop bypasses page text detection. It can expose a missed
/// field, but Tiny remains a printed-text model. It does not solve handwriting.
/// Results are independent evidence and never overwrite a page extraction.
pub struct RegionReader {
    recognizer: TextRecognizer,
}

impl RegionReader {
    /// Load a recognizer and its matching dictionary/shape configuration.
    ///
    /// # Errors
    /// Returns model/configuration loading errors.
    pub fn new(config: RecConfig) -> Result<Self, DocumentError> {
        Ok(Self {
            recognizer: TextRecognizer::new(config)?,
        })
    }

    /// Use the existing PP-OCRv6 recognizer/dictionary; no detector is loaded.
    ///
    /// # Errors
    /// Returns model/configuration errors.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self, DocumentError> {
        let dir = directory.as_ref();
        Self::new(
            InitializeConfig::ppv6(
                dir.join("det.rten"),
                dir.join("rec.rten"),
                dir.join("dict.txt"),
            )
            .rec,
        )
    }

    /// Recognize one caller-selected line in bounded PNG/JPEG bytes.
    ///
    /// The returned span box is the **supplied region extent**, not a model
    /// localization. Scores and text are unmodified, including weak results.
    ///
    /// # Errors
    /// Rejects invalid images, over 32 MiB encoded, over 8192 pixels per edge,
    /// over 64 MiB decoding allocation, regions wider than 64 times their height,
    /// invalid regions, or empty OCR output.
    pub fn read_bytes(
        &mut self,
        bytes: &[u8],
        region: ImageRegion,
    ) -> Result<Document, DocumentError> {
        let image = crop(bytes, region)?;
        let output = self.recognizer.run(&[Mat::new(image)], false)?;
        let text = crate::bidi::visual_to_logical(
            output.txts.first().ok_or(DocumentError::UnexpectedOutput)?,
        );
        let confidence = *output
            .scores
            .first()
            .ok_or(DocumentError::UnexpectedOutput)?;
        // Decoding and bounds validation cap every coordinate at 8192, exactly
        // representable in f32. Conversion via u16 also checks that invariant.
        let px = |v| {
            u16::try_from(v)
                .map(f32::from)
                .map_err(|_| DocumentError::InvalidInput("region coordinate exceeds image limits"))
        };
        let (x, y, right, bottom) = (
            px(region.x)?,
            px(region.y)?,
            px(region.x + region.width)?,
            px(region.y + region.height)?,
        );
        Document::new(vec![TextSpan {
            text,
            confidence: Some(confidence),
            box_points: Some([[x, y], [right, y], [right, bottom], [x, bottom]]),
        }])
    }

    /// Read a bounded local file and recognize one line.
    ///
    /// # Errors
    /// Returns file I/O and the same validation/OCR errors as [`Self::read_bytes`].
    pub fn read_file(
        &mut self,
        path: impl AsRef<Path>,
        region: ImageRegion,
    ) -> Result<Document, DocumentError> {
        let path = path.as_ref();
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|file| file.take(32 * 1024 * 1024 + 1).read_to_end(&mut bytes))
            .map_err(|source| DocumentError::ImageIo {
                path: path.to_path_buf(),
                source,
            })?;
        self.read_bytes(&bytes, region)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_region_overflow_and_bounded_image_decoding() {
        let good = ImageRegion {
            x: 2,
            y: 3,
            width: 4,
            height: 5,
        };
        assert!(good.validate(6, 8).is_ok());
        for bad in [
            ImageRegion { width: 0, ..good },
            ImageRegion { height: 0, ..good },
            ImageRegion {
                x: u32::MAX,
                ..good
            },
            ImageRegion {
                y: u32::MAX,
                ..good
            },
            ImageRegion { width: 5, ..good },
            ImageRegion { height: 6, ..good },
        ] {
            assert!(bad.validate(6, 8).is_err());
        }
        assert!(crop(&[], good).is_err());
        assert!(crop(b"not an image", good).is_err());
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(8193, 1)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        assert!(crop(bytes.get_ref(), good).is_err());
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(10, 10)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        assert_eq!(crop(bytes.get_ref(), good).unwrap().width(), 4);
        assert!(crop(bytes.get_ref(), ImageRegion { x: 9, ..good }).is_err());
    }
}
