/// Decimal separator chosen by the caller, not inferred from ambiguous values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimalSeparator {
    /// `1,234.56`.
    Dot,
    /// `1.234,56`.
    Comma,
}

/// Explicit decimal convention and fractional precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmountFormat {
    /// Decimal character; the other character may group thousands.
    pub decimal_separator: DecimalSeparator,
    /// Number of fractional digits, 0..=6; never rounds excess digits.
    pub fraction_digits: u8,
}

impl Default for AmountFormat {
    fn default() -> Self {
        Self {
            decimal_separator: DecimalSeparator::Dot,
            fraction_digits: 2,
        }
    }
}

impl AmountFormat {
    /// Check the supported fractional precision.
    ///
    /// # Errors
    /// Rejects precision greater than six.
    pub fn validate(self) -> Result<(), ParseAmountError> {
        if self.fraction_digits > 6 {
            return Err(ParseAmountError::InvalidFormat);
        }
        Ok(())
    }
}

/// Signed exact amount; currency is retained separately by the domain report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Amount {
    /// Integer count of units at `fraction_digits` precision.
    pub minor_units: i64,
    /// Decimal scale, for example two for cents/fils.
    pub fraction_digits: u8,
}

/// Monetary parsing errors without OCR substitutions or rounding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseAmountError {
    /// Unsupported precision.
    #[error("fractional precision must be 0..=6")]
    InvalidFormat,
    /// Invalid characters, decimal convention, grouping or precision.
    #[error("invalid or ambiguous decimal amount")]
    InvalidNumber,
    /// Exact scaled amount cannot fit i64.
    #[error("amount overflow")]
    Overflow,
}

impl Amount {
    /// Parse an exact amount with explicit decimal convention. Common surrounding
    /// currency tokens are accepted but never mapped to an inferred currency.
    /// Excess fractional digits and malformed thousands groups are rejected.
    ///
    /// # Errors
    /// Returns invalid format/number or overflow errors.
    pub fn parse(text: &str, format: AmountFormat) -> Result<Self, ParseAmountError> {
        format.validate()?;
        let mut text = text.trim();
        for token in [
            "AED", "USD", "EUR", "GBP", "SAR", "INR", "MYR", "RM", "$", "€", "£", "₹",
        ] {
            if let Some(value) = text.strip_prefix(token) {
                text = value.trim();
                break;
            }
            if let Some(value) = text.strip_suffix(token) {
                text = value.trim();
                break;
            }
        }
        let negative = text.starts_with('-');
        if negative || text.starts_with('+') {
            text = &text[1..];
        }
        let (decimal, grouping) = match format.decimal_separator {
            DecimalSeparator::Dot => ('.', ','),
            DecimalSeparator::Comma => (',', '.'),
        };
        let parts = text.split(decimal).collect::<Vec<_>>();
        if parts.len() > 2 || parts[0].is_empty() {
            return Err(ParseAmountError::InvalidNumber);
        }
        let groups = parts[0].split(grouping).collect::<Vec<_>>();
        if groups
            .iter()
            .any(|g| g.is_empty() || !g.bytes().all(|b| b.is_ascii_digit()))
            || (groups.len() > 1
                && (groups[0].len() > 3 || groups[1..].iter().any(|g| g.len() != 3)))
        {
            return Err(ParseAmountError::InvalidNumber);
        }
        let fraction = parts.get(1).copied().unwrap_or("");
        if (parts.len() == 2 && fraction.is_empty())
            || fraction.len() > usize::from(format.fraction_digits)
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(ParseAmountError::InvalidNumber);
        }
        let factor = 10_i128.pow(u32::from(format.fraction_digits));
        let integer: i128 = groups
            .concat()
            .parse()
            .map_err(|_| ParseAmountError::Overflow)?;
        let fraction_value: i128 = if fraction.is_empty() {
            0
        } else {
            fraction.parse().map_err(|_| ParseAmountError::Overflow)?
        };
        let missing = u32::from(format.fraction_digits)
            - u32::try_from(fraction.len()).map_err(|_| ParseAmountError::InvalidNumber)?;
        let value = integer
            .checked_mul(factor)
            .and_then(|v| v.checked_add(fraction_value * 10_i128.pow(missing)))
            .and_then(|v| if negative { v.checked_neg() } else { Some(v) })
            .ok_or(ParseAmountError::Overflow)?;
        Ok(Self {
            minor_units: i64::try_from(value).map_err(|_| ParseAmountError::Overflow)?,
            fraction_digits: format.fraction_digits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_decimals_grouping_sign_and_caller_locale() {
        for (text, expected) in [
            ("1,234.56", 123_456),
            ("AED 12.3", 1230),
            ("-0.01", -1),
            ("12", 1200),
        ] {
            assert_eq!(
                Amount::parse(text, AmountFormat::default())
                    .unwrap()
                    .minor_units,
                expected
            );
        }
        assert_eq!(
            Amount::parse(
                "1.234,56",
                AmountFormat {
                    decimal_separator: DecimalSeparator::Comma,
                    fraction_digits: 2
                }
            )
            .unwrap()
            .minor_units,
            123_456
        );
        assert_eq!(
            Amount::parse("-92233720368547758.08", AmountFormat::default())
                .unwrap()
                .minor_units,
            i64::MIN
        );
    }
    #[test]
    fn rejects_bad_ocr_ambiguous_grouping_precision_and_overflow() {
        for text in [
            "1O.00",
            "1,23.45",
            "1.234",
            "12.",
            ".50",
            "",
            "NaN",
            "1 234.56",
            "1.234,56",
            "92233720368547758.08",
        ] {
            assert!(
                Amount::parse(text, AmountFormat::default()).is_err(),
                "{text}"
            );
        }
        assert!(
            Amount::parse(
                "1",
                AmountFormat {
                    fraction_digits: 7,
                    ..AmountFormat::default()
                }
            )
            .is_err()
        );
    }
}
