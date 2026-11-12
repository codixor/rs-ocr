use super::{Amount, Document, ExtractionOptions, Field, FieldCandidate, ReviewIssue};
use regex::Regex;
use std::sync::OnceLock;

struct Row {
    text: String,
    indices: Vec<usize>,
}

fn candidate(document: &Document, value: String, indices: Vec<usize>) -> FieldCandidate<String> {
    let confidence = indices.iter().try_fold(1.0_f32, |lowest, &index| {
        document.spans[index].confidence.map(|v| lowest.min(v))
    });
    FieldCandidate {
        raw_text: value.clone(),
        value,
        confidence,
        span_indices: indices,
    }
}

fn normalized(text: &str) -> String {
    text.replace(['：', '﹕'], ":")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn label_tail<'a>(text: &'a str, aliases: &[&str]) -> Option<&'a str> {
    let lower = text.to_ascii_lowercase();
    aliases.iter().find_map(|alias| {
        lower
            .strip_prefix(alias)
            .filter(|tail| {
                alias.ends_with('#')
                    || tail.is_empty()
                    || tail.starts_with(|c: char| {
                        c.is_whitespace() || matches!(c, ':' | '#' | '(' | '.')
                    })
            })
            .map(|_| {
                let tail = text[alias.len()..].trim();
                // A punctuation delimiter can follow a label. A decimal point
                // attached to digits is part of the value, including bad OCR.
                let without_dots = tail.trim_start_matches('.');
                let tail = if without_dots.is_empty()
                    || without_dots.starts_with(|c: char| c.is_whitespace() || c == ':')
                {
                    without_dots
                } else {
                    tail
                };
                tail.trim().trim_start_matches([':', '#']).trim()
            })
    })
}

fn rows(document: &Document) -> Vec<Row> {
    let mut indices: Vec<_> = (0..document.spans.len()).collect();
    let positioned = document.spans.iter().all(|span| span.box_points.is_some());
    let bounds = |index: usize| {
        let points = document.spans[index].box_points.unwrap_or([[0.0; 2]; 4]);
        let top = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let bottom = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let left = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        (left, top + (bottom - top) * 0.5, (bottom - top).max(1.0))
    };
    let horizontal = |index: usize| {
        let points = document.spans[index].box_points.unwrap_or([[0.0; 2]; 4]);
        (
            points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
            points
                .iter()
                .map(|p| p[0])
                .fold(f32::NEG_INFINITY, f32::max),
        )
    };
    if positioned {
        indices.sort_by(|a, b| {
            bounds(*a)
                .1
                .total_cmp(&bounds(*b).1)
                .then_with(|| bounds(*a).0.total_cmp(&bounds(*b).0))
        });
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for index in indices {
        if positioned && let Some(group) = groups.last_mut() {
            let (_, centre, height) = bounds(group[0]);
            let (_, y, h) = bounds(index);
            let (left, right) = horizontal(index);
            let separate_columns = group.iter().all(|&other| {
                let (x, end) = horizontal(other);
                right.min(end) - left.max(x) <= (right - left).min(end - x) * 0.25
            });
            if (y - centre).abs() <= height.min(h) * 0.65 && separate_columns {
                group.push(index);
                continue;
            }
        }
        groups.push(vec![index]);
    }
    groups
        .into_iter()
        .map(|mut indices| {
            if positioned {
                indices.sort_by(|a, b| bounds(*a).0.total_cmp(&bounds(*b).0));
            }
            let text = indices
                .iter()
                .map(|index| document.spans[*index].text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            Row { text, indices }
        })
        .collect()
}

// Used to separate columns and to prevent a bare label consuming another label.
const LABELS: &[&str] = &[
    "merchant",
    "vendor",
    "seller",
    "store",
    "invoice number",
    "invoice no",
    "receipt no",
    "date",
    "data",
    "time",
    "hora",
    "cashier",
    "currency",
    "moeda",
    "subtotal",
    "sub total",
    "tax",
    "vat",
    "total",
    "amount",
    "payee",
    "pay to",
    "bank",
    "account",
    "micr",
    "cheque",
    "check",
    "no.",
];

fn next_label(text: &str) -> bool {
    label_tail(text, LABELS).is_some()
}

fn reserved(text: &str, aliases: &[&str]) -> bool {
    [
        "amount in words",
        "amount words",
        "amount due",
        "total tax",
        "total due",
        "total qty",
        "total quantity",
        "total excluding",
        "total (excluding",
        "total iva",
        "total poupanca",
        "total poupança",
        "total gst",
        "total sales (excluding gst)",
        "total-excl-gst",
        "total s/",
        "total de descontos",
    ]
    .iter()
    .any(|specific| !aliases.contains(specific) && label_tail(text, &[*specific]).is_some())
}

fn row_bounds(document: &Document, row: &Row) -> Option<(f32, f32, f32, f32)> {
    let mut bounds = (f32::INFINITY, f32::INFINITY, 0.0_f32, 0.0_f32);
    for &index in &row.indices {
        for [x, y] in document.spans[index].box_points? {
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);
        }
    }
    Some(bounds)
}

fn below_value(
    document: &Document,
    row: &Row,
    next: Option<&Row>,
) -> Option<FieldCandidate<String>> {
    let next = next?;
    let (left, top, right, bottom) = row_bounds(document, row)?;
    let (x, y, end, _) = row_bounds(document, next)?;
    // Only an immediate, nearby, single value. A whole table row or another
    // label is never guessed to be the value of a heading above it.
    if next.indices.len() != 1
        || y < bottom
        || y - bottom > (bottom - top).max(1.0) * 2.0
        || !(x <= right && end >= left || x >= right)
        || next_label(&normalized(&next.text))
    {
        return None;
    }
    let mut indices = row.indices.clone();
    indices.extend(&next.indices);
    Some(candidate(document, normalized(&next.text), indices))
}

pub(crate) fn field(document: &Document, aliases: &[&str]) -> Field<String> {
    let mut result = Field {
        candidates: Vec::new(),
    };
    let mut aliases = aliases.to_vec();
    aliases.sort_by_key(|alias| std::cmp::Reverse(alias.len()));
    let lines = rows(document);
    for (line_index, row) in lines.iter().enumerate() {
        for start in 0..row.indices.len() {
            let first = normalized(&document.spans[row.indices[start]].text);
            if label_tail(&first, &aliases).is_none() || reserved(&first, &aliases) {
                continue;
            }
            let end = (start + 1..row.indices.len())
                .find(|&i| {
                    let text = normalized(&document.spans[row.indices[i]].text);
                    next_label(&text)
                        || aliases
                            .iter()
                            .any(|s| s.starts_with("pay") || *s == "order of")
                            && text.starts_with(['$', '€', '£', '₹'])
                })
                .unwrap_or(row.indices.len());
            let indices = row.indices[start..end].to_vec();
            let text = normalized(
                &indices
                    .iter()
                    .map(|&i| document.spans[i].text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            let Some(value) = label_tail(&text, &aliases) else {
                continue;
            };
            // Second explicit label inside one OCR span also ends a value.
            let lower = value.to_ascii_lowercase();
            let stop = LABELS
                .iter()
                .filter_map(|label| lower.find(&format!(" {label}:")))
                .min()
                .unwrap_or(value.len());
            let value = value[..stop].trim();
            if !value.is_empty() {
                result
                    .candidates
                    .push(candidate(document, value.into(), indices));
            } else if start == 0
                && end == row.indices.len()
                && let Some(value) = below_value(document, row, lines.get(line_index + 1))
            {
                result.candidates.push(value);
            }
        }
    }
    result
}

/// Prefer explicit dates; otherwise retain every syntactically valid printed date.
/// No day/month ordering or two-digit century is inferred.
pub(crate) fn date_field(
    document: &Document,
    aliases: &[&str],
    issues: &mut Vec<ReviewIssue>,
) -> Field<String> {
    static DATES: OnceLock<Regex> = OnceLock::new();
    let dates = DATES.get_or_init(|| Regex::new(concat!(
        r"(?i)(?:^|[^0-9])(?P<date>",
        r"[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|",
        r"[0-9]{1,2}/[0-9]{1,2}/(?:[0-9]{4}|[0-9]{2})|",
        r"[0-9]{1,2}-[0-9]{1,2}-(?:[0-9]{4}|[0-9]{2})|",
        r"[0-9]{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:[0-9]{4}|[0-9]{2})",
        r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+[0-9]{1,2},?\s+[0-9]{4}",
        r")(?:$|[^0-9]|[0-9]{1,2}:)"
    )).expect("constant date regex"));
    let mut labeled = field(document, aliases);
    if !labeled.candidates.is_empty() {
        for c in &mut labeled.candidates {
            if let Some(m) = dates.captures(&c.value).and_then(|m| m.name("date")) {
                c.value = m.as_str().into();
            }
        }
        return labeled;
    }
    let mut result = Field {
        candidates: Vec::new(),
    };
    for row in rows(document) {
        for m in dates.captures_iter(&row.text) {
            let Some(value) = m.name("date").map(|v| v.as_str()) else {
                continue;
            };
            let parts: Vec<_> = value.split(['/', '-']).collect();
            if parts.len() == 3 {
                let nums: Vec<u32> = parts.iter().filter_map(|n| n.parse().ok()).collect();
                if nums.len() != 3 || nums.contains(&0) {
                    continue;
                }
                let (a, b) = if parts[0].len() == 4 {
                    (nums[1], nums[2])
                } else {
                    (nums[0], nums[1])
                };
                if a > 31 || b > 31 || (a > 12 && b > 12) {
                    continue;
                }
            }
            result
                .candidates
                .push(candidate(document, value.into(), row.indices.clone()));
        }
    }
    mark_inferred(&result, "date", "unlabeled printed date", issues);
    result
}

fn mark_inferred<T>(
    field: &Field<T>,
    name: &'static str,
    method: &'static str,
    issues: &mut Vec<ReviewIssue>,
) {
    if !field.candidates.is_empty() {
        issues.push(ReviewIssue::InferredField {
            field: name,
            method,
        });
    }
}

fn wrapped_header(
    document: &Document,
    lines: &[Row],
    start: usize,
) -> Option<FieldCandidate<String>> {
    static LEGAL: OnceLock<Regex> = OnceLock::new();
    let legal = LEGAL.get_or_init(|| {
        Regex::new(r"(?i)(?:sdn\.?\s+bhd\.?|s/b|ltd\.?|limited|lda\.?)\s*$")
            .expect("constant company suffix")
    });
    let first = &lines[start];
    if legal.is_match(&first.text) {
        return None;
    }
    let (left, top, right, bottom) = row_bounds(document, first)?;
    let height = (bottom - top).max(1.0);
    let mut indices = first.indices.clone();
    let mut texts = vec![normalized(&first.text)];
    let mut previous_bottom = bottom;
    for row in lines.iter().skip(start + 1).take(2) {
        let (x, y, end, foot) = row_bounds(document, row)?;
        let text = normalized(&row.text);
        if y < previous_bottom
            || y - previous_bottom > height * 1.5
            || !(height * 0.65..=height * 1.5).contains(&(foot - y))
            || (x - left).abs() > height && ((x + end) - (left + right)).abs() > height * 2.0
            || text.chars().any(|c| c.is_ascii_digit() || c == ':')
            || [
                "rua", "jalan", "address", "tel", "fax", "lot", "no.", "gst", "nif",
            ]
            .iter()
            .any(|p| text.to_lowercase().starts_with(p))
        {
            break;
        }
        texts.push(text);
        indices.extend(&row.indices);
        if legal.is_match(texts.last()?) {
            return Some(candidate(document, texts.join(" "), indices));
        }
        previous_bottom = foot;
    }
    None
}

/// Conservative header fallback. Unscored input remains unscored and inferred
/// values always require review; OCR errors are not repaired.
pub(crate) fn header(
    document: &Document,
    bank: bool,
    issues: &mut Vec<ReviewIssue>,
) -> Field<String> {
    static REGISTRATION: OnceLock<Regex> = OnceLock::new();
    let registration = REGISTRATION.get_or_init(|| {
        Regex::new(r"\s*[（(][0-9][0-9A-Za-z -]*[)）].*$").expect("constant registration regex")
    });
    let mut result = Field {
        candidates: Vec::new(),
    };
    let rows = rows(document);
    for (index, row) in rows
        .iter()
        .take(if bank { rows.len() } else { 10 })
        .enumerate()
    {
        let text = normalized(&row.text);
        let lower = text.to_lowercase();
        if bank {
            if !lower
                .split(|c: char| !c.is_alphabetic())
                .any(|word| word == "bank")
            {
                continue;
            }
        } else {
            let alphabetic = text.chars().filter(|c| c.is_alphabetic()).count();
            if alphabetic < 4
                || text.chars().filter(char::is_ascii_digit).count() > 3
                    && !registration.is_match(&text)
            {
                continue;
            }
            if [
                "invoice",
                "receipt",
                "tax invoice",
                "taxinvoice",
                "date",
                "data",
                "tel",
                "fax",
                "gst",
                "company no",
                "co-reg",
                "nif",
                "no.",
                "no ",
                "address",
                "subtotal",
                "total",
                "currency",
                "quant",
                "fatura",
                "factura",
                "contrib",
                "rua ",
                "av. ",
            ]
            .iter()
            .any(|prefix| lower.starts_with(prefix))
            {
                continue;
            }
        }
        let value = registration.replace(&text, "").trim().to_owned();
        if !bank && let Some(wrapped) = wrapped_header(document, &rows, index) {
            result.candidates.push(wrapped);
            break;
        }
        result
            .candidates
            .push(candidate(document, value, row.indices.clone()));
        if !bank {
            break;
        }
    }
    mark_inferred(
        &result,
        if bank { "bank" } else { "merchant" },
        "document header",
        issues,
    );
    result
}

/// Only currency-marked amounts, never a bare number or memo amount.
pub(crate) fn currency_amount(
    document: &Document,
    options: ExtractionOptions,
    issues: &mut Vec<ReviewIssue>,
) -> Field<Amount> {
    let mut result = Field {
        candidates: Vec::new(),
    };
    for row in rows(document) {
        for (start, &index) in row.indices.iter().enumerate() {
            let raw = document.spans[index].text.trim();
            if ![
                "AED", "USD", "EUR", "GBP", "SAR", "INR", "MYR", "RM", "$", "€", "£", "₹",
            ]
            .iter()
            .any(|token| raw.starts_with(token))
            {
                continue;
            }
            let end = (start + 1..row.indices.len())
                .find(|&i| next_label(&normalized(&document.spans[row.indices[i]].text)))
                .unwrap_or(row.indices.len());
            let indices = row.indices[start..end].to_vec();
            let raw = indices
                .iter()
                .map(|&i| document.spans[i].text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            if let Ok(value) = Amount::parse(protected_amount(&raw), options.amount_format) {
                let source = candidate(document, raw, indices);
                result.candidates.push(FieldCandidate {
                    value,
                    raw_text: source.raw_text,
                    confidence: source.confidence,
                    span_indices: source.span_indices,
                });
            }
        }
    }
    mark_inferred(&result, "amount", "standalone currency amount", issues);
    result
}

// Cheque protection stars/ellipsis and printed currency qualifiers are decorations,
// never character repairs. Only leading stars after a currency are removed.
fn protected_amount(raw: &str) -> &str {
    let raw = raw.trim();
    let value = [
        "AED", "USD", "EUR", "GBP", "SAR", "INR", "MYR", "RM", "$", "€", "£", "₹",
    ]
    .iter()
    .find_map(|currency| raw.strip_prefix(currency))
    .unwrap_or(raw)
    .trim();
    if value.starts_with(['*', '…']) {
        value.trim_start_matches(['*', '…']).trim()
    } else {
        raw
    }
}

fn amount_value(raw: &str) -> &str {
    let raw = raw.trim();
    if let Some(tail) = ["(RM)", "(MYR)", "(USD)", "(EUR)", "(AED)", "(GBP)"]
        .iter()
        .find_map(|prefix| raw.strip_prefix(prefix))
    {
        return tail.trim().trim_start_matches(':').trim();
    }
    raw
}

/// A standalone short integer in the upper right is a reviewable number hint.
/// Fractional routing numbers, money and bottom MICR lines are excluded.
pub(crate) fn cheque_number(document: &Document, issues: &mut Vec<ReviewIssue>) -> Field<String> {
    let mut result = field(
        document,
        &[
            "cheque number",
            "cheque no.",
            "cheque no",
            "check number",
            "check no.",
            "check no",
        ],
    );
    if !result.candidates.is_empty() {
        return result;
    }
    result = field(document, &["no.", "no"]);
    result
        .candidates
        .retain(|c| !c.value.is_empty() && c.value.bytes().all(|v| v.is_ascii_digit()));
    if !result.candidates.is_empty() {
        return result;
    }
    if document.spans.iter().any(|s| s.box_points.is_none()) {
        return result;
    }
    let right = document
        .spans
        .iter()
        .filter_map(|s| s.box_points)
        .flatten()
        .map(|p| p[0])
        .fold(0.0_f32, f32::max);
    let bottom = document
        .spans
        .iter()
        .filter_map(|s| s.box_points)
        .flatten()
        .map(|p| p[1])
        .fold(0.0_f32, f32::max);
    for (index, s) in document.spans.iter().enumerate() {
        let raw = s.text.trim();
        let Some(points) = s.box_points else {
            continue;
        };
        let left = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let y = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        if (3..=10).contains(&raw.len())
            && raw.bytes().all(|c| c.is_ascii_digit())
            && left >= right * 0.5
            && y <= bottom * 0.25
        {
            result
                .candidates
                .push(candidate(document, raw.into(), vec![index]));
        }
    }
    mark_inferred(
        &result,
        "cheque_number",
        "upper-right standalone identifier",
        issues,
    );
    result
}

pub(crate) fn amount_field(
    document: &Document,
    aliases: &[&str],
    name: &'static str,
    options: ExtractionOptions,
    issues: &mut Vec<ReviewIssue>,
) -> Field<Amount> {
    let mut candidates = Vec::new();
    for candidate in field(document, aliases).candidates {
        let value = amount_value(&candidate.value);
        let value = if name == "amount" {
            protected_amount(value)
        } else {
            value
        };
        match Amount::parse(value, options.amount_format) {
            Ok(value) => candidates.push(FieldCandidate {
                value,
                raw_text: candidate.raw_text,
                confidence: candidate.confidence,
                span_indices: candidate.span_indices,
            }),
            Err(_) => issues.push(ReviewIssue::InvalidAmount {
                field: name,
                raw: candidate.raw_text,
            }),
        }
    }
    Field { candidates }
}

pub(crate) fn review_field<T: PartialEq>(
    field: &Field<T>,
    name: &'static str,
    required: bool,
    options: ExtractionOptions,
    issues: &mut Vec<ReviewIssue>,
) {
    if field.candidates.is_empty() {
        if required {
            issues.push(ReviewIssue::MissingField(name));
        }
    } else if field.unique().is_none() {
        issues.push(ReviewIssue::AmbiguousField(name));
    }
    if let Some(score) = field
        .candidates
        .iter()
        .filter_map(|candidate| candidate.confidence)
        .min_by(f32::total_cmp)
        && score < options.min_confidence
    {
        issues.push(ReviewIssue::LowConfidence { field: name, score });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::TextSpan;
    fn span(text: &str, score: f32, x: f32, y: f32) -> TextSpan {
        TextSpan {
            text: text.into(),
            confidence: Some(score),
            box_points: Some([[x, y], [x + 40.0, y], [x + 40.0, y + 20.0], [x, y + 20.0]]),
        }
    }
    #[test]
    fn joins_geometric_label_value_evidence_and_preserves_weak_scores() {
        let document = Document::new(vec![
            span("115.50", 0.3, 200.0, 10.0),
            span("TOTAL:", 0.9, 10.0, 10.0),
        ])
        .unwrap();
        let extracted = field(&document, &["total"]);
        assert_eq!(extracted.unique().unwrap().value, "115.50");
        assert_eq!(extracted.unique().unwrap().span_indices, [1, 0]);
        let mut issues = Vec::new();
        review_field(
            &extracted,
            "total",
            true,
            ExtractionOptions::default(),
            &mut issues,
        );
        assert_eq!(
            issues,
            [ReviewIssue::LowConfidence {
                field: "total",
                score: 0.3
            }]
        );
    }
    #[test]
    fn conflicting_candidates_are_not_resolved_and_labels_match_boundaries() {
        let document = Document::from_text("TOTAL: 10.00\nTOTAL: 11.00\nTOTALLY: 99.00").unwrap();
        let extracted = field(&document, &["total"]);
        assert_eq!(extracted.candidates.len(), 2);
        assert!(extracted.unique().is_none());
        assert!(document.spans()[0].confidence.is_none());
        assert_eq!(
            document.raw_text(),
            "TOTAL: 10.00\nTOTAL: 11.00\nTOTALLY: 99.00"
        );
    }
    #[test]
    fn unscored_or_weak_duplicate_values_do_not_become_high_confidence() {
        let mut document = Document::new(vec![
            span("TOTAL: 10", 0.9, 0.0, 0.0),
            span("TOTAL: 10", 0.1, 0.0, 50.0),
        ])
        .unwrap();
        assert_eq!(
            field(&document, &["total"]).unique().unwrap().confidence,
            Some(0.1)
        );
        document.spans[1].confidence = None;
        assert_eq!(
            field(&document, &["total"]).unique().unwrap().confidence,
            None
        );
    }

    #[test]
    fn dates_can_touch_times_without_swallowing_them_or_mixed_separators() {
        for (text, expected) in [
            ("Date: 26-05-201815:16:13", "26-05-2018"),
            ("21/05/2018Time: 11:12", "21/05/2018"),
            ("ORD#15 -REG #19-02/03/2018 12:59", "02/03/2018"),
            ("22 Jun 201818:07", "22 Jun 2018"),
        ] {
            let doc = Document::from_text(text).unwrap();
            let mut issues = vec![];
            let dates = date_field(&doc, &["date"], &mut issues);
            assert_eq!(dates.unique().unwrap().value, expected);
            assert!(dates.unique().unwrap().confidence.is_none());
        }
        let mut issues = vec![];
        assert!(
            date_field(
                &Document::from_text("32/50/2026 12345678901").unwrap(),
                &["date"],
                &mut issues
            )
            .candidates
            .is_empty()
        );
        let dates = date_field(
            &Document::from_text("01/02/2026\n02/02/2026").unwrap(),
            &["date"],
            &mut issues,
        );
        assert!(dates.unique().is_none());
    }

    #[test]
    fn inferred_amounts_keep_conflicts_and_reject_identifier_or_ocr_repairs() {
        let doc = Document::from_text("$12.34\nFor $9.00\n123456789\n$1O.00\n$15.00").unwrap();
        let mut issues = vec![];
        let amount = currency_amount(&doc, ExtractionOptions::default(), &mut issues);
        assert_eq!(amount.candidates.len(), 2);
        assert!(amount.unique().is_none());
        assert!(issues.iter().any(|i| matches!(
            i,
            ReviewIssue::InferredField {
                field: "amount",
                ..
            }
        )));
        assert_eq!(amount.candidates[0].raw_text, "$12.34");
        assert_eq!(amount.candidates[0].span_indices, [0]);
        assert!(amount.candidates[0].confidence.is_none());
    }

    #[test]
    fn cheque_number_uses_geometry_without_consuming_amount_or_micr() {
        let mut issues = vec![];
        let doc = Document::new(vec![
            span("000042", 0.2, 500.0, 0.0),
            span("$120.00", 0.9, 500.0, 100.0),
            span("123456789", 0.9, 0.0, 300.0),
        ])
        .unwrap();
        let number = cheque_number(&doc, &mut issues);
        assert_eq!(number.unique().unwrap().value, "000042");
        assert_eq!(number.unique().unwrap().confidence, Some(0.2));
        assert!(!issues.is_empty());
        assert!(
            cheque_number(
                &Document::from_text("000042\n123456789").unwrap(),
                &mut issues
            )
            .candidates
            .is_empty()
        );
    }

    #[test]
    fn header_fallback_is_reviewable_and_explicit_values_take_precedence() {
        let report = crate::bills::extract(
            Document::from_text("TAX INVOICE\nEXAMPLE STORE\nTotal: RM12.30").unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert_eq!(report.merchant.unique().unwrap().value, "EXAMPLE STORE");
        assert_eq!(report.total.unique().unwrap().value.minor_units, 1230);
        assert!(report.issues.iter().any(|i| matches!(
            i,
            ReviewIssue::InferredField {
                field: "merchant",
                ..
            }
        )));
        let report = crate::bills::extract(
            Document::from_text("OTHER HEADER\nMerchant: Example\nTotal: 1.00").unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert_eq!(report.merchant.unique().unwrap().value, "Example");
        assert!(report.issues.is_empty());
    }
}
