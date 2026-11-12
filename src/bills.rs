//! Invoice/receipt extraction with labeled values and reviewable header hints.
//!
//! Exact decimal amounts and original evidence are retained. This initial module
//! does not parse line-item tables or guess currency.

use crate::{
    InitializeConfig,
    documents::{
        Amount, Document, DocumentError, DocumentReader, ExtractionOptions, Field, ReviewIssue,
        amount_field, date_field, field, header, review_field,
    },
};
use std::path::Path;

const TOTAL_LABELS: &[&str] = &[
    "total payable",
    "total payment",
    "total do documento",
    "total (eat in)",
    "total inc. gst",
    "total inc。 gst",
    "total 6% supplies (inc. gst)",
    "grand total",
    "total due",
    "amount due",
    "nett total",
    "net total",
    "total amount",
    "totalamount",
    "total rounded",
    "total sales incl gst",
    "total after adj incl gst",
    "total sales",
    "total (inclusive of gst)",
    "total (including gst)",
    "total incl. gst@6%",
    "total incl. gst@0%",
    "total includes gst 6%",
    "totalrm including gst 6%",
    "total.rm including gst 6%",
    "total incl. gst",
    "total (eur)",
    "total a pagar",
    "total pago",
    "total da venda",
    "total",
];

// These explicitly describe the settled/payable amount after rounding or
// adjustments. Ordinary totals remain evidence in the document; they describe
// a different stage of the bill. Conflicting payable values still conflict.
pub(crate) const PAYABLE_LABELS: &[&str] = &[
    "nett total",
    "net total",
    "final total",
    "total rounded",
    "total payable",
    "amount due",
    "total due",
    "grand total",
    "total a pagar",
    "total pago",
    "total do documento",
];

/// Extracted invoice/receipt values and their original evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct BillReport {
    /// Complete recognized document.
    pub document: Document,
    /// Merchant label, or a reviewable document header fallback.
    pub merchant: Field<String>,
    /// Invoice or receipt number; leading zeros preserved.
    pub reference_number: Field<String>,
    /// Uninterpreted printed date.
    pub date: Field<String>,
    /// Explicit printed currency, without mapping ambiguous symbols to codes.
    pub currency: Field<String>,
    /// Exact subtotal.
    pub subtotal: Field<Amount>,
    /// Exact explicitly labeled tax/VAT, without combining conflicting values.
    pub tax: Field<Amount>,
    /// Exact grand total or amount due.
    pub total: Field<Amount>,
    /// Missing, conflicting, invalid or inconsistent values needing review.
    pub issues: Vec<ReviewIssue>,
}

/// Extract invoice/receipt fields from OCR or a transcription.
///
/// # Errors
/// Returns invalid extraction options. Missing or inconsistent fields are issues.
pub fn extract(
    document: Document,
    options: ExtractionOptions,
) -> Result<BillReport, DocumentError> {
    options.validate()?;
    let mut issues = Vec::new();
    let mut merchant = field(&document, &["merchant", "vendor", "seller", "store"]);
    if merchant.candidates.is_empty() {
        merchant = header(&document, false, &mut issues);
    }
    let reference_number = field(
        &document,
        &[
            "invoice number",
            "invoice no.",
            "invoice no",
            "receipt number",
            "receipt no.",
            "receipt no",
            "invoice #",
            "invoice#",
            "inv no.",
            "inv no",
            "invno",
            "inv#",
        ],
    );
    let date = date_field(
        &document,
        &["invoice date", "receipt date", "date", "data"],
        &mut issues,
    );
    let currency = field(&document, &["currency", "moeda"]);
    let subtotal = amount_field(
        &document,
        &[
            "subtotal",
            "sub total",
            "sub-total",
            "total (excluding gst)",
        ],
        "subtotal",
        options,
        &mut issues,
    );
    let tax = amount_field(
        &document,
        &["total tax", "tax", "vat", "total iva"],
        "tax",
        options,
        &mut issues,
    );
    let payable = field(&document, PAYABLE_LABELS);
    let labels = if payable.candidates.is_empty() {
        TOTAL_LABELS
    } else {
        PAYABLE_LABELS
    };
    let total = amount_field(&document, labels, "total", options, &mut issues);
    for (value, name, required) in [
        (&merchant, "merchant", true),
        (&reference_number, "reference_number", false),
        (&date, "date", false),
        (&currency, "currency", false),
    ] {
        review_field(value, name, required, options, &mut issues);
    }
    review_field(&subtotal, "subtotal", false, options, &mut issues);
    review_field(&tax, "tax", false, options, &mut issues);
    review_field(&total, "total", true, options, &mut issues);
    if let (Some(subtotal), Some(tax), Some(total)) =
        (subtotal.unique(), tax.unique(), total.unique())
    {
        match subtotal
            .value
            .minor_units
            .checked_add(tax.value.minor_units)
        {
            Some(sum) if sum != total.value.minor_units => issues.push(ReviewIssue::TotalsMismatch),
            None => issues.push(ReviewIssue::AmountOverflow),
            _ => {}
        }
    }
    Ok(BillReport {
        document,
        merchant,
        reference_number,
        date,
        currency,
        subtotal,
        tax,
        total,
        issues,
    })
}

/// Reusable document OCR sessions for invoices and receipts.
pub struct BillReader {
    reader: DocumentReader,
}

impl BillReader {
    /// Reuse a document reader, for example one built on the Burn backend.
    #[must_use]
    pub fn from_document_reader(reader: DocumentReader) -> Self {
        Self { reader }
    }

    /// Load a matching regional OCR configuration.
    ///
    /// # Errors
    /// Returns model loading/configuration failures.
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
    /// OCR bytes and extract an invoice/receipt.
    ///
    /// # Errors
    /// Returns invalid options/input, OCR or evidence validation errors.
    pub fn read_bytes(
        &mut self,
        bytes: &[u8],
        options: ExtractionOptions,
    ) -> Result<BillReport, DocumentError> {
        options.validate()?;
        extract(self.reader.read_bytes(bytes)?, options)
    }
    /// OCR a local invoice/receipt image and extract its fields.
    ///
    /// # Errors
    /// Returns invalid options, file I/O, OCR or evidence validation errors.
    pub fn read_file(
        &mut self,
        path: impl AsRef<Path>,
        options: ExtractionOptions,
    ) -> Result<BillReport, DocumentError> {
        options.validate()?;
        extract(self.reader.read_file(path)?, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_invoice_and_checks_exact_totals() {
        let document = Document::from_text("Merchant: Example Store\nInvoice No: 0007\nCurrency: AED\nSubtotal: 110.00\nVAT: 5.50\nTotal: 115.50").unwrap();
        let report = extract(document, ExtractionOptions::default()).unwrap();
        assert_eq!(report.merchant.unique().unwrap().value, "Example Store");
        assert_eq!(report.reference_number.unique().unwrap().value, "0007");
        assert_eq!(report.total.unique().unwrap().value.minor_units, 11550);
        assert!(report.issues.is_empty());
    }
    #[test]
    fn inconsistent_totals_and_duplicate_totals_are_not_silently_accepted() {
        let report = extract(
            Document::from_text("Merchant: Example\nSubtotal: 10.00\nTax: 1.00\nTotal: 12.00")
                .unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert!(report.issues.contains(&ReviewIssue::TotalsMismatch));
        let report = extract(
            Document::from_text("Merchant: Example\nTotal: 10.00\nTotal: 20.00").unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert!(report.total.unique().is_none());
        assert!(
            report
                .issues
                .contains(&ReviewIssue::AmbiguousField("total"))
        );
    }

    #[test]
    fn total_tax_does_not_become_a_grand_total_candidate() {
        let report = extract(
            Document::from_text("Merchant: Example\nSubtotal: 100\nTotal Tax: 5\nTotal: 105")
                .unwrap(),
            ExtractionOptions::default(),
        )
        .unwrap();
        assert_eq!(report.total.unique().unwrap().value.minor_units, 10_500);
        assert_eq!(report.tax.unique().unwrap().value.minor_units, 500);
        assert!(report.issues.is_empty());
    }
}
