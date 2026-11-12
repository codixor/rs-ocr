//! Field association controls independent of any benchmark label strings.
use rs_ocr::{
    bills, cheques,
    documents::{Document, ExtractionOptions, ReviewIssue, TextSpan},
};

fn span(text: &str, x: f32, y: f32, width: f32, confidence: Option<f32>) -> TextSpan {
    TextSpan {
        text: text.into(),
        confidence,
        box_points: Some([[x, y], [x + width, y], [x + width, y + 20.0], [x, y + 20.0]]),
    }
}

#[test]
fn column_labels_preserve_only_their_own_evidence_and_weakest_score() {
    let doc = Document::new(vec![
        span("Invoice No:", 0.0, 0.0, 90.0, Some(0.9)),
        span("0007", 100.0, 0.0, 40.0, Some(0.3)),
        span("Date:", 220.0, 0.0, 40.0, Some(0.8)),
        span("2026-10-29", 270.0, 0.0, 100.0, None),
        span("Total:", 0.0, 40.0, 50.0, Some(0.9)),
        span("$12.00", 100.0, 40.0, 50.0, Some(0.8)),
    ])
    .unwrap();
    let report = bills::extract(doc, ExtractionOptions::default()).unwrap();
    let reference = report.reference_number.unique().unwrap();
    assert_eq!(reference.value, "0007");
    assert_eq!(reference.span_indices, [0, 1]);
    assert_eq!(reference.confidence, Some(0.3));
    assert!(report.date.unique().unwrap().confidence.is_none());
    assert_eq!(report.date.unique().unwrap().span_indices, [2, 3]);
    assert_eq!(report.total.unique().unwrap().value.minor_units, 1200);
    assert!(report.issues.iter().any(|i| matches!(
        i,
        ReviewIssue::LowConfidence {
            field: "reference_number",
            ..
        }
    )));
}

#[test]
fn immediate_below_value_needs_geometry_and_one_unambiguous_neighbor() {
    let make = |next: Vec<TextSpan>| {
        let mut spans = vec![span("Total:", 0.0, 0.0, 60.0, Some(0.9))];
        spans.extend(next);
        bills::extract(Document::new(spans).unwrap(), ExtractionOptions::default()).unwrap()
    };
    let report = make(vec![span("8.25", 150.0, 30.0, 50.0, Some(0.2))]);
    let total = report.total.unique().unwrap();
    assert_eq!(total.value.minor_units, 825);
    assert_eq!(total.span_indices, [0, 1]);
    assert_eq!(total.confidence, Some(0.2));
    for next in [
        vec![span("8.25", 150.0, 100.0, 50.0, Some(0.9))],
        vec![span("Tax: 8.25", 0.0, 30.0, 80.0, Some(0.9))],
        vec![
            span("8.25", 100.0, 30.0, 50.0, Some(0.9)),
            span("9.25", 200.0, 30.0, 50.0, Some(0.9)),
        ],
    ] {
        assert_eq!(
            make(next).total.candidates,
            [] as [rs_ocr::documents::FieldCandidate<rs_ocr::documents::Amount>; 0]
        );
    }
    assert_eq!(
        bills::extract(
            Document::from_text("Total:\n8.25").unwrap(),
            ExtractionOptions::default()
        )
        .unwrap()
        .total
        .candidates,
        [] as [rs_ocr::documents::FieldCandidate<rs_ocr::documents::Amount>; 0]
    );
}

#[test]
fn final_payable_role_precedes_intermediate_totals_but_preserves_conflicts() {
    for text in [
        "Total Amount: 8.48\nNett Total: 8.50",
        "Total: 11.00\nGrand Total: 11.01",
    ] {
        let r = bills::extract(
            Document::from_text(text).unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert!(r.total.unique().is_some());
        assert!(r.document.raw_text().contains(text));
    }
    let r = bills::extract(
        Document::from_text("Total: 10.00\nNett Total: 10.01\nFinal Total: 10.02").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert!(r.total.unique().is_none());
    assert_eq!(r.total.candidates.len(), 2);
    let r = bills::extract(
        Document::from_text("Total: 10.00\nNett Total: 1O.01").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert!(r.total.unique().is_none());
    assert!(
        r.issues
            .iter()
            .any(|i| matches!(i,ReviewIssue::InvalidAmount{raw,..} if raw=="1O.01"))
    );
    let r = bills::extract(
        Document::from_text("Total (RM): 8.25\nTotal GST: 0.50\nTotal Qty: 2").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.total.unique().unwrap().value.minor_units, 825);
}

#[test]
fn wrapped_legal_header_keeps_all_evidence_without_joining_address() {
    let r = bills::extract(
        Document::new(vec![
            span("EXAMPLE HARDWARE", 0.0, 0.0, 180.0, Some(0.9)),
            span("ENTERPRISE SDN BHD", 0.0, 30.0, 180.0, Some(0.4)),
            span("No. 7, Main Street", 0.0, 60.0, 180.0, Some(0.9)),
        ])
        .unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        r.merchant.unique().unwrap().value,
        "EXAMPLE HARDWARE ENTERPRISE SDN BHD"
    );
    assert_eq!(r.merchant.unique().unwrap().span_indices, [0, 1]);
    assert!(r.issues.iter().any(|i| matches!(
        i,
        ReviewIssue::InferredField {
            field: "merchant",
            ..
        }
    )));
}

#[test]
fn cheque_stacked_label_star_protection_month_first_dates_and_conflicts() {
    let r = cheques::extract(
        Document::new(vec![
            span("PAY TO THE", 0.0, 0.0, 80.0, Some(0.9)),
            span("ORDER OF", 0.0, 30.0, 80.0, Some(0.9)),
            span("Alice Example", 100.0, 30.0, 120.0, Some(0.8)),
            span("$", 250.0, 30.0, 10.0, Some(0.7)),
            span("***5,000.00", 270.0, 30.0, 100.0, Some(0.3)),
            span("Date: March 23, 1966", 0.0, 80.0, 200.0, Some(0.9)),
        ])
        .unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.payee.unique().unwrap().value, "Alice Example");
    assert_eq!(r.payee.unique().unwrap().span_indices, [1, 2]);
    let amount = r.amount.unique().unwrap();
    assert_eq!(amount.value.minor_units, 500_000);
    assert_eq!(amount.raw_text, "$ ***5,000.00");
    assert_eq!(amount.span_indices, [3, 4]);
    assert_eq!(amount.confidence, Some(0.3));
    assert_eq!(r.date.unique().unwrap().value, "March 23, 1966");
    for bad in ["$***5,OOO.00", "$5,000.000", "$***5,000.00 account 123"] {
        assert_eq!(
            cheques::extract(
                Document::from_text(bad).unwrap(),
                ExtractionOptions::default()
            )
            .unwrap()
            .amount
            .candidates,
            [] as [rs_ocr::documents::FieldCandidate<rs_ocr::documents::Amount>; 0]
        );
    }
    let r = cheques::extract(
        Document::from_text("$***5,000.00\n$***6,000.00").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert!(r.amount.unique().is_none());
    let r = cheques::extract(
        Document::from_text("PAY TO THE\nORDER OF\n$\nNo.").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        r.payee.candidates,
        [] as [rs_ocr::documents::FieldCandidate<std::string::String>; 0]
    );
    assert_eq!(
        r.amount.candidates,
        [] as [rs_ocr::documents::FieldCandidate<rs_ocr::documents::Amount>; 0]
    );
    assert_eq!(
        r.cheque_number.candidates,
        [] as [rs_ocr::documents::FieldCandidate<std::string::String>; 0]
    );
}

#[test]
fn overlapping_text_rows_do_not_merge_and_numeric_punctuation_is_not_repaired() {
    let r = bills::extract(
        Document::new(vec![
            span("EXAMPLE SDN BHD", 0.0, 0.0, 180.0, Some(0.9)),
            span("2026-10-29", 40.0, 12.0, 100.0, Some(0.9)),
        ])
        .unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.merchant.unique().unwrap().value, "EXAMPLE SDN BHD");
    assert_eq!(r.date.unique().unwrap().value, "2026-10-29");
    for text in ["Total .1.40", "Total: .1.40", "Total 1O.00"] {
        let r = bills::extract(
            Document::from_text(text).unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert!(r.total.unique().is_none());
        assert!(
            r.issues
                .iter()
                .any(|i| matches!(i, ReviewIssue::InvalidAmount { .. }))
        );
    }
    let r = bills::extract(
        Document::from_text("Total. 1.40").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.total.unique().unwrap().value.minor_units, 140);
    let r = cheques::extract(
        Document::from_text("BANK OF EXAMPLE").unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.bank.unique().unwrap().value, "BANK OF EXAMPLE");
    let r = cheques::extract(
        Document::new(vec![
            span("PAY TO THE ORDER OF", 0.0, 0.0, 180.0, Some(0.9)),
            span("$", 200.0, 30.0, 10.0, Some(0.9)),
        ])
        .unwrap(),
        ExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        r.payee.candidates,
        [] as [rs_ocr::documents::FieldCandidate<std::string::String>; 0]
    );
}
