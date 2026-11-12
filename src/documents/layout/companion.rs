use super::{LayoutBox, LayoutDetector, LayoutError, LayoutOptions, PageLayout, preprocess};
use crate::{
    bills, cheques,
    documents::{
        Document, DocumentReader, ExtractionOptions, Field, FieldCandidate, ReviewIssue,
        review_field,
    },
};
use std::path::Path;

/// Original page OCR plus model regions and their associated source span indices.
/// Layout scores are never substituted for OCR recognition scores.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutDocument {
    /// Unmodified page text, coordinates and OCR scores.
    document: Document,
    /// Region predictions in original image coordinates.
    layout: PageLayout,
    /// One source-span list per layout region, including overlapping memberships.
    region_spans: Vec<Vec<usize>>,
}

fn span_bounds(points: [[f32; 2]; 4]) -> LayoutBox {
    LayoutBox {
        left: points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
        top: points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
        right: points.iter().map(|p| p[0]).fold(0.0, f32::max),
        bottom: points.iter().map(|p| p[1]).fold(0.0, f32::max),
    }
}

fn append<T: Clone>(
    field: &Field<T>,
    map: &[usize],
    issues: &[ReviewIssue],
    name: &str,
    out: &mut Vec<FieldCandidate<T>>,
) -> Result<(), LayoutError> {
    // Region-local header/date/currency fallbacks are not financial semantics.
    if issues
        .iter()
        .any(|i| matches!(i,ReviewIssue::InferredField{field,..} if *field==name))
    {
        return Ok(());
    }
    for c in &field.candidates {
        let indices = c
            .span_indices
            .iter()
            .map(|&i| map.get(i).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or(LayoutError::InvalidOutput("invalid local field evidence"))?;
        out.push(FieldCandidate {
            value: c.value.clone(),
            raw_text: c.raw_text.clone(),
            confidence: c.confidence,
            span_indices: indices,
        });
    }
    Ok(())
}

fn fill<T: PartialEq>(
    field: &mut Field<T>,
    additions: Vec<FieldCandidate<T>>,
    name: &'static str,
    options: ExtractionOptions,
    issues: &mut Vec<ReviewIssue>,
) {
    if !field.candidates.is_empty() || additions.is_empty() {
        return;
    }
    for candidate in additions {
        if !field
            .candidates
            .iter()
            .any(|c| c.value == candidate.value && c.span_indices == candidate.span_indices)
        {
            field.candidates.push(candidate);
        }
    }
    issues.retain(|i| !matches!(i,ReviewIssue::MissingField(f) if *f==name));
    issues.push(ReviewIssue::InferredField {
        field: name,
        method: "PP-DocLayout-S region association; explicit local label",
    });
    review_field(field, name, false, options, issues);
}

macro_rules! backfill {
    ($report:ident,$locals:ident,$options:ident,$($member:ident),+)=>{$(
        let mut additions=Vec::new();
        for (local,map) in &$locals {
            append(&local.$member,map,&local.issues,stringify!($member),&mut additions)?;
        }
        fill(&mut $report.$member,additions,stringify!($member),$options,&mut $report.issues);
    )+};
}

impl LayoutDocument {
    /// Original, unmodified OCR evidence.
    #[must_use]
    pub fn document(&self) -> &Document {
        &self.document
    }
    /// Predicted page regions.
    #[must_use]
    pub fn layout(&self) -> &PageLayout {
        &self.layout
    }
    /// Source span memberships, indexed by layout region.
    #[must_use]
    pub fn region_spans(&self) -> &[Vec<usize>] {
        &self.region_spans
    }
    /// Associate at least 80%-contained OCR spans with each region.
    /// No text is dropped, reordered, repaired or re-recognized.
    /// # Errors
    /// Rejects invalid page dimensions, region scores, limits or geometry.
    pub fn new(document: Document, layout: PageLayout) -> Result<Self, LayoutError> {
        let width = u16::try_from(layout.width)
            .map_err(|_| LayoutError::InvalidInput("invalid layout width"))?;
        let height = u16::try_from(layout.height)
            .map_err(|_| LayoutError::InvalidInput("invalid layout height"))?;
        if width == 0 || height == 0 || width > 8192 || height > 8192 || layout.regions.len() > 100
        {
            return Err(LayoutError::InvalidInput(
                "invalid layout page dimensions or region count",
            ));
        }
        let mut region_spans = Vec::new();
        for region in &layout.regions {
            if !region.bounds.valid()
                || region.bounds.right > f32::from(width)
                || region.bounds.bottom > f32::from(height)
                || !region.confidence.is_finite()
                || !(0.0..=1.0).contains(&region.confidence)
                || region.raw_bounds.iter().any(|v| !v.is_finite())
            {
                return Err(LayoutError::InvalidInput(
                    "invalid region geometry or score",
                ));
            }
            let indices = document
                .spans()
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    let bounds = span_bounds(s.box_points?);
                    (bounds.valid() && region.bounds.intersection(bounds) / bounds.area() >= 0.8)
                        .then_some(i)
                })
                .collect();
            region_spans.push(indices);
        }
        Ok(Self {
            document,
            layout,
            region_spans,
        })
    }

    fn documents(&self) -> Result<Vec<(Document, Vec<usize>)>, LayoutError> {
        self.region_spans
            .iter()
            .zip(&self.layout.regions)
            .filter(|(indices, region)| {
                indices
                    .iter()
                    .any(|&i| !self.document.spans()[i].text.trim().is_empty())
                    && ![1, 16, 18, 20, 21].contains(&region.class.code())
            })
            .map(|(indices, _)| {
                Ok((
                    Document::new(
                        indices
                            .iter()
                            .map(|&i| self.document.spans()[i].clone())
                            .collect(),
                    )?,
                    indices.clone(),
                ))
            })
            .collect()
    }

    /// Backfill missing bill candidates from explicitly labeled region-local text.
    /// Existing page candidates always remain authoritative, including conflicts.
    /// New candidates retain original span indices/OCR scores and require review.
    /// # Errors
    /// Returns invalid extraction options/evidence.
    pub fn extract_bill(
        &self,
        options: ExtractionOptions,
    ) -> Result<bills::BillReport, LayoutError> {
        let mut report = bills::extract(self.document.clone(), options)?;
        let final_total = !crate::documents::field(&self.document, bills::PAYABLE_LABELS)
            .candidates
            .is_empty();
        let locals = self
            .documents()?
            .into_iter()
            .map(|(doc, map)| {
                let local_final = !crate::documents::field(&doc, bills::PAYABLE_LABELS)
                    .candidates
                    .is_empty();
                let mut local = bills::extract(doc, options)?;
                // A malformed final payable total must never be replaced with
                // a valid intermediate amount from a different region.
                if final_total && !local_final {
                    local.total.candidates.clear();
                }
                Ok((local, map))
            })
            .collect::<Result<Vec<_>, LayoutError>>()?;
        backfill!(
            report,
            locals,
            options,
            merchant,
            reference_number,
            date,
            currency,
            subtotal,
            tax,
            total
        );
        if let (Some(a), Some(b), Some(c)) = (
            report.subtotal.unique(),
            report.tax.unique(),
            report.total.unique(),
        ) {
            match a.value.minor_units.checked_add(b.value.minor_units) {
                Some(sum)
                    if sum != c.value.minor_units
                        && !report.issues.contains(&ReviewIssue::TotalsMismatch) =>
                {
                    report.issues.push(ReviewIssue::TotalsMismatch);
                }
                None if !report.issues.contains(&ReviewIssue::AmountOverflow) => {
                    report.issues.push(ReviewIssue::AmountOverflow);
                }
                _ => {}
            }
        }
        Ok(report)
    }

    /// Backfill missing explicitly labeled cheque candidates; this does not add
    /// handwriting recognition, signatures, MICR validation or field semantics.
    /// # Errors
    /// Returns invalid extraction options/evidence.
    pub fn extract_cheque(
        &self,
        options: ExtractionOptions,
    ) -> Result<cheques::ChequeReport, LayoutError> {
        let mut report = cheques::extract(self.document.clone(), options)?;
        let locals = self
            .documents()?
            .into_iter()
            .map(|(doc, map)| Ok((cheques::extract(doc, options)?, map)))
            .collect::<Result<Vec<_>, LayoutError>>()?;
        backfill!(
            report,
            locals,
            options,
            payee,
            date,
            cheque_number,
            bank,
            account_number,
            amount,
            amount_words,
            micr
        );
        if report
            .amount
            .unique()
            .is_some_and(|c| c.value.minor_units <= 0)
            && !report
                .issues
                .contains(&ReviewIssue::NonPositiveChequeAmount)
        {
            report.issues.push(ReviewIssue::NonPositiveChequeAmount);
        }
        Ok(report)
    }
}

/// Reused layout and Tiny OCR sessions. Layout is an explicit opt-in companion.
pub struct LayoutReader {
    detector: LayoutDetector,
    reader: DocumentReader,
}
impl LayoutReader {
    /// Load a layout model and reuse the caller's configured page OCR reader.
    /// # Errors
    /// Returns model signature/loading errors.
    pub fn new(model: impl AsRef<Path>, reader: DocumentReader) -> Result<Self, LayoutError> {
        Ok(Self {
            detector: LayoutDetector::new(model)?,
            reader,
        })
    }
    /// Read original page OCR and layout regions from bounded image bytes.
    /// # Errors
    /// Returns layout/image/OCR validation or inference errors.
    pub fn read_bytes(
        &mut self,
        bytes: &[u8],
        options: LayoutOptions,
    ) -> Result<LayoutDocument, LayoutError> {
        let layout = self.detector.detect_bytes(bytes, options)?;
        LayoutDocument::new(self.reader.read_bytes(bytes)?, layout)
    }
    /// Read at most 32 MiB plus an overflow byte, then run both sessions.
    /// # Errors
    /// Returns file I/O, validation and model inference errors.
    pub fn read_file(
        &mut self,
        path: impl AsRef<Path>,
        options: LayoutOptions,
    ) -> Result<LayoutDocument, LayoutError> {
        options.validate()?;
        self.read_bytes(&preprocess::read_file(path.as_ref())?, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        documents::TextSpan,
        documents::layout::{LayoutClass, LayoutRegion},
    };
    fn span(text: &str, x: f32) -> TextSpan {
        TextSpan {
            text: text.into(),
            confidence: Some(0.4),
            box_points: Some([[x, 10.0], [x + 50.0, 10.0], [x + 50.0, 30.0], [x, 30.0]]),
        }
    }
    fn layout() -> PageLayout {
        PageLayout {
            width: 800,
            height: 100,
            regions: vec![LayoutRegion {
                class: LayoutClass::TEXT,
                confidence: 0.99,
                bounds: LayoutBox {
                    left: 0.0,
                    top: 0.0,
                    right: 200.0,
                    bottom: 60.0,
                },
                raw_bounds: [0.0, 0.0, 200.0, 60.0],
            }],
        }
    }
    #[test]
    fn local_labeled_backfill_retains_original_evidence_and_review_flags() {
        let doc = Document::new(vec![
            span("Total:", 0.0),
            span("10.00", 100.0),
            span("Unrelated", 500.0),
        ])
        .unwrap();
        let original = bills::extract(doc.clone(), ExtractionOptions::default()).unwrap();
        assert!(original.total.candidates.is_empty());
        let page = LayoutDocument::new(doc.clone(), layout()).unwrap();
        let report = page.extract_bill(ExtractionOptions::default()).unwrap();
        assert_eq!(report.document, doc);
        let c = report.total.unique().unwrap();
        assert_eq!(c.value.minor_units, 1000);
        assert_eq!(c.span_indices, [0, 1]);
        assert_eq!(c.confidence, Some(0.4));
        assert!(
            report
                .issues
                .iter()
                .any(|i| matches!(i, ReviewIssue::InferredField { field: "total", .. }))
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| matches!(i, ReviewIssue::LowConfidence { field: "total", .. }))
        );
        assert!(!report.issues.contains(&ReviewIssue::MissingField("total")));
    }
    #[test]
    fn page_conflicts_and_unlabeled_text_are_never_resolved_by_region_scores() {
        let doc = Document::from_text("Total: 10.00\nTotal: 11.00").unwrap();
        let before = bills::extract(doc.clone(), ExtractionOptions::default()).unwrap();
        let page = LayoutDocument::new(doc, layout()).unwrap();
        let after = page.extract_bill(ExtractionOptions::default()).unwrap();
        assert_eq!(before, after);
        assert!(after.total.unique().is_none());
        let page = LayoutDocument::new(Document::new(vec![span("10.00", 0.0)]).unwrap(), layout())
            .unwrap();
        assert!(
            page.extract_bill(ExtractionOptions::default())
                .unwrap()
                .total
                .candidates
                .is_empty()
        );
    }
    #[test]
    fn association_requires_eighty_percent_overlap_and_valid_layout() {
        let page = LayoutDocument::new(
            Document::new(vec![span("Near", 190.0), span("Inside", 10.0)]).unwrap(),
            layout(),
        )
        .unwrap();
        assert_eq!(page.region_spans, [vec![1]]);
        let mut bad = layout();
        bad.regions[0].confidence = f32::NAN;
        assert!(LayoutDocument::new(Document::from_text("Example").unwrap(), bad).is_err());
    }
    #[test]
    fn conflicting_region_backfills_and_invalid_final_totals_require_review() {
        let mut spans = vec![
            span("Total:", 0.0),
            span("10.00", 100.0),
            span("Unrelated", 500.0),
        ];
        let mut second = vec![
            span("Total:", 0.0),
            span("11.00", 100.0),
            span("Unrelated", 500.0),
        ];
        for s in &mut second {
            for p in s.box_points.as_mut().unwrap() {
                p[1] += 40.0;
            }
        }
        spans.extend(second);
        let mut hints = layout();
        let mut second = hints.regions[0].clone();
        hints.regions[0].bounds.bottom = 40.0;
        second.bounds.top = 45.0;
        second.bounds.bottom = 90.0;
        hints.regions.push(second);
        let page = LayoutDocument::new(Document::new(spans).unwrap(), hints).unwrap();
        let r = page.extract_bill(ExtractionOptions::default()).unwrap();
        assert!(r.total.unique().is_none());
        assert_eq!(r.total.candidates.len(), 2);
        assert!(r.issues.contains(&ReviewIssue::AmbiguousField("total")));
        let doc = Document::new(vec![
            span("Total:", 0.0),
            span("10.00", 100.0),
            span("Nett Total: 1O.01", 400.0),
        ])
        .unwrap();
        let page = LayoutDocument::new(doc, layout()).unwrap();
        assert!(
            page.extract_bill(ExtractionOptions::default())
                .unwrap()
                .total
                .candidates
                .is_empty()
        );
        let mut blank = span(" ", 0.0);
        blank
            .box_points
            .as_mut()
            .unwrap()
            .iter_mut()
            .for_each(|p| p[1] += 40.0);
        let doc = Document::new(vec![span("Total: 10.00", 500.0), blank]).unwrap();
        assert!(
            LayoutDocument::new(doc, layout())
                .unwrap()
                .extract_bill(ExtractionOptions::default())
                .is_ok()
        );
    }
}
