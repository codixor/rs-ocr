//! Initial printed-cheque field extraction with retained OCR evidence.
//!
//! Supports explicit English labels and caller-selected amount conventions.
//! Dates, account identifiers, amount words and MICR are preserved as text.
//! This does not authenticate a cheque, recognize signatures, or validate MICR.

use crate::{
    InitializeConfig,
    documents::{
        Amount, Document, DocumentError, DocumentReader, ExtractionOptions, Field, ReviewIssue,
        amount_field, cheque_number, currency_amount, date_field, field, header, review_field,
    },
};
use std::path::Path;

/// Extracted cheque values and the underlying document evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ChequeReport {
    /// Full recognized document.
    pub document: Document,
    /// Explicit payee label.
    pub payee: Field<String>,
    /// Uninterpreted date; no day/month guessing.
    pub date: Field<String>,
    /// Cheque/check number, preserving leading zeros.
    pub cheque_number: Field<String>,
    /// Explicit bank label, when present.
    pub bank: Field<String>,
    /// Explicit account label; no substitution of O/0 or other characters.
    pub account_number: Field<String>,
    /// Exact numeric amount.
    pub amount: Field<Amount>,
    /// Recognized amount in words, without claiming numeric agreement.
    pub amount_words: Field<String>,
    /// Raw explicitly labeled MICR text, without a country-specific decoder.
    pub micr: Field<String>,
    /// Missing, conflicting, low-confidence or invalid values.
    pub issues: Vec<ReviewIssue>,
}

fn payee_field(document: &Document) -> Field<String> {
    let mut payee = field(document, &["pay to the order of", "payee", "pay to"]);
    payee.candidates.retain(|c| {
        ![
            "the",
            "the order",
            "the order of",
            "order of",
            "$",
            "€",
            "£",
            "₹",
            "dollars",
        ]
        .iter()
        .any(|fragment| c.value.eq_ignore_ascii_case(fragment))
    });
    if payee.candidates.is_empty()
        && document
            .spans()
            .iter()
            .any(|s| s.text.to_ascii_lowercase().trim().starts_with("pay"))
    {
        // Common cheque label wraps onto a second line. An isolated fragment
        // "PAY TO THE" is a label, never the beneficiary of a blank cheque.
        payee = field(document, &["order of"]);
    }
    payee.candidates.retain(|c| {
        !["$", "€", "£", "₹", "dollars"]
            .iter()
            .any(|v| c.value.eq_ignore_ascii_case(v))
    });
    payee
}

fn bank_field(document: &Document, issues: &mut Vec<ReviewIssue>) -> Field<String> {
    let mut bank = field(document, &["bank name"]);
    let mut short_bank = field(document, &["bank"]);
    short_bank.candidates.retain(|c| {
        c.span_indices.first().is_some_and(|&i| {
            let label = document.spans()[i].text.trim().to_ascii_lowercase();
            label == "bank" || label.starts_with("bank:") || label.starts_with("bank :")
        })
    });
    bank.candidates.extend(short_bank.candidates);
    if bank.candidates.is_empty() {
        bank = header(document, true, issues);
    }
    bank
}

/// Extract fields from OCR or an existing transcription.
///
/// # Errors
/// Returns an error for invalid extraction options. Missing fields are review issues.
pub fn extract(
    document: Document,
    options: ExtractionOptions,
) -> Result<ChequeReport, DocumentError> {
    options.validate()?;
    let mut issues = Vec::new();
    let payee = payee_field(&document);
    let date = date_field(&document, &["date"], &mut issues);
    let cheque_number = cheque_number(&document, &mut issues);
    let bank = bank_field(&document, &mut issues);
    let account_number = field(&document, &["account number", "account no.", "account no"]);
    let amount_words = field(&document, &["amount in words", "amount words"]);
    let micr = field(&document, &["micr"]);
    let mut amount = amount_field(
        &document,
        &["numeric amount", "amount"],
        "amount",
        options,
        &mut issues,
    );
    if amount.candidates.is_empty() {
        amount = currency_amount(&document, options, &mut issues);
    }
    for (value, name, required) in [
        (&payee, "payee", true),
        (&date, "date", true),
        (&cheque_number, "cheque_number", true),
        (&bank, "bank", false),
        (&account_number, "account_number", false),
        (&amount_words, "amount_words", false),
        (&micr, "micr", false),
    ] {
        review_field(value, name, required, options, &mut issues);
    }
    review_field(&amount, "amount", true, options, &mut issues);
    if amount
        .unique()
        .is_some_and(|candidate| candidate.value.minor_units <= 0)
    {
        issues.push(ReviewIssue::NonPositiveChequeAmount);
    }
    Ok(ChequeReport {
        document,
        payee,
        date,
        cheque_number,
        bank,
        account_number,
        amount,
        amount_words,
        micr,
        issues,
    })
}

/// Reusable cheque OCR sessions, sharing the standard document reader.
pub struct ChequeReader {
    reader: DocumentReader,
}

impl ChequeReader {
    /// Reuse a document reader, for example one built on the Burn backend.
    #[must_use]
    pub fn from_document_reader(reader: DocumentReader) -> Self {
        Self { reader }
    }

    /// Load a matching regional OCR model configuration.
    ///
    /// # Errors
    /// Returns OCR model loading errors.
    pub fn new(config: InitializeConfig) -> Result<Self, DocumentError> {
        Ok(Self {
            reader: DocumentReader::new(config)?,
        })
    }
    /// Load the local PP-OCRv6 pack.
    ///
    /// # Errors
    /// Returns model/configuration failures.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self, DocumentError> {
        Ok(Self {
            reader: DocumentReader::from_model_dir(directory)?,
        })
    }
    /// OCR and extract a printed cheque from bytes.
    ///
    /// # Errors
    /// Returns invalid options/input, OCR or evidence validation errors.
    pub fn read_bytes(
        &mut self,
        bytes: &[u8],
        options: ExtractionOptions,
    ) -> Result<ChequeReport, DocumentError> {
        options.validate()?;
        extract(self.reader.read_bytes(bytes)?, options)
    }
    /// OCR and extract a local printed cheque image.
    ///
    /// # Errors
    /// Returns invalid options, file I/O, OCR or evidence validation errors.
    pub fn read_file(
        &mut self,
        path: impl AsRef<Path>,
        options: ExtractionOptions,
    ) -> Result<ChequeReport, DocumentError> {
        options.validate()?;
        extract(self.reader.read_file(path)?, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_labeled_cheque_without_losing_identifiers_or_guessing_dates() {
        let doc = Document::from_text("Pay to the order of: Alice Example\nDate: 03/04/2026\nCheque No: 000042\nAccount No: 00O123\nAmount: AED 1,234.56\nMICR: raw 00123").unwrap();
        let report = extract(doc, ExtractionOptions::default()).unwrap();
        assert_eq!(report.payee.unique().unwrap().value, "Alice Example");
        assert_eq!(report.cheque_number.unique().unwrap().value, "000042");
        assert_eq!(report.account_number.unique().unwrap().value, "00O123");
        assert_eq!(report.date.unique().unwrap().value, "03/04/2026");
        assert_eq!(report.amount.unique().unwrap().value.minor_units, 123_456);
        assert!(report.issues.is_empty());
    }
    #[test]
    fn missing_invalid_and_conflicting_values_remain_reviewable() {
        let report = extract(
            Document::from_text("Payee: Alice\nPayee: Bob\nAmount: 1O.00").unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert!(report.payee.unique().is_none());
        assert!(
            report
                .issues
                .contains(&ReviewIssue::AmbiguousField("payee"))
        );
        assert!(report.issues.contains(&ReviewIssue::MissingField("amount")));
        assert!(report.issues.iter().any(
            |issue| matches!(issue, ReviewIssue::InvalidAmount { raw, .. } if raw == "1O.00")
        ));
    }

    #[test]
    fn numeric_amount_does_not_consume_the_amount_in_words() {
        let report = extract(Document::from_text("Payee: Alice\nDate: 2026-10-29\nCheque No: 42\nAmount: 12.00\nAmount in words: Twelve only").unwrap(), ExtractionOptions::default()).unwrap();
        assert_eq!(report.amount.unique().unwrap().value.minor_units, 1200);
        assert_eq!(report.amount_words.unique().unwrap().value, "Twelve only");
        assert!(report.issues.is_empty());
    }
}
