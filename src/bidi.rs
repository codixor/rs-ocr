//! Reading order for right-to-left recognizer output.
//!
//! `PaddleOCR` Arabic recognizers (PP-OCRv5 Arabic and the PP-OCRv6 Arabic fine-tune) emit a
//! line in *visual* order: characters as they appear from left to right on the image.
//! Logical (reading) order is the reverse, except that left-to-right runs inside the line
//! (numbers, Latin words) keep their own direction.

/// Arabic-script letters, presentation forms and Arabic-Indic digits.
fn is_arabic(c: char) -> bool {
    matches!(c,
        '\u{0600}'..='\u{06FF}'
        | '\u{0750}'..='\u{077F}'
        | '\u{08A0}'..='\u{08FF}'
        | '\u{FB50}'..='\u{FDFF}'
        | '\u{FE70}'..='\u{FEFF}')
}

/// Characters that start or continue a left-to-right run (Latin letters, ASCII digits).
fn is_ltr(c: char) -> bool {
    c.is_ascii_alphanumeric() || (c.is_alphabetic() && !is_arabic(c))
}

/// Separators kept inside a number (`25.50`, `11:30`, `1,250`). In Arabic text a `-` or a
/// space splits numbers into separate runs that are laid out right to left, so
/// `2026-118` is displayed as `118-2026`.
fn joins_digits(c: char) -> bool {
    matches!(c, '.' | ',' | ':' | '/')
}

fn mirror(c: char) -> char {
    match c {
        '(' => ')',
        ')' => '(',
        '[' => ']',
        ']' => '[',
        '{' => '}',
        '}' => '{',
        '<' => '>',
        '>' => '<',
        '«' => '»',
        '»' => '«',
        other => other,
    }
}

/// Convert one visually ordered recognized line to reading order.
///
/// Lines without Arabic letters are returned unchanged, so Latin output is never touched.
#[must_use]
pub fn visual_to_logical(line: &str) -> String {
    if !line.chars().any(is_arabic) {
        return line.to_owned();
    }
    let reversed: Vec<char> = line.chars().rev().map(mirror).collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < reversed.len() {
        if is_arabic(reversed[i]) || !is_ltr(reversed[i]) {
            out.push(reversed[i]);
            i += 1;
            continue;
        }
        // A stretch of non-Arabic text, from this letter/digit to the last letter/digit
        // before the next Arabic character.
        let mut stretch_end = i;
        let mut j = i;
        while j < reversed.len() && !is_arabic(reversed[j]) {
            if is_ltr(reversed[j]) {
                stretch_end = j + 1;
            }
            j += 1;
        }
        let stretch = &reversed[i..stretch_end];
        if stretch.iter().any(|&c| c.is_alphabetic()) {
            // Latin words (with any digits and separators) form one left-to-right run.
            push_run(&mut out, stretch);
        } else {
            // Numbers only: each number is its own run; separators between them keep
            // their right-to-left position.
            let mut k = 0;
            while k < stretch.len() {
                if !stretch[k].is_ascii_digit() {
                    out.push(stretch[k]);
                    k += 1;
                    continue;
                }
                let mut e = k + 1;
                while e < stretch.len() {
                    if stretch[e].is_ascii_digit() {
                        e += 1;
                    } else if joins_digits(stretch[e])
                        && stretch.get(e + 1).is_some_and(char::is_ascii_digit)
                    {
                        e += 2;
                    } else {
                        break;
                    }
                }
                push_run(&mut out, &stretch[k..e]);
                k = e;
            }
        }
        i = stretch_end;
    }
    out
}

/// Append a left-to-right run, undoing the line reversal and bracket mirroring.
fn push_run(out: &mut String, run: &[char]) {
    out.extend(run.iter().rev().map(|&c| mirror(c)));
}

#[cfg(test)]
mod tests {
    use super::visual_to_logical;

    #[test]
    fn latin_lines_are_unchanged() {
        for line in ["Invoice No. 2026-117", "A7B9C2", "Conditions générales", ""] {
            assert_eq!(visual_to_logical(line), line);
        }
    }

    #[test]
    fn arabic_words_are_reversed_into_reading_order() {
        assert_eq!(visual_to_logical("ايلوج راد"), "دار جوليا");
    }

    #[test]
    fn numbers_and_latin_runs_keep_their_direction() {
        // On the image, numbers and Latin words read left to right, so the recognizer
        // emits them as-is inside the mirrored Arabic text.
        assert_eq!(
            visual_to_logical("ربمفون 11 خيراتب 117 مقر ةروتافلا"),
            "الفاتورة رقم 117 بتاريخ 11 نوفمبر"
        );
        assert_eq!(
            visual_to_logical("يسنوت رانيد TND 873.50 غلبملا"),
            "المبلغ TND 873.50 دينار تونسي"
        );
        assert_eq!(visual_to_logical("خيراتلا 11-11-2026"), "2026-11-11 التاريخ");
        assert_eq!(visual_to_logical("Hotel Julia راد"), "دار Hotel Julia");
        assert_eq!(
            visual_to_logical("contact@darjulia.tn :ينورتكلإلا ديربلا"),
            "البريد الإلكتروني: contact@darjulia.tn"
        );
    }

    #[test]
    fn separate_numbers_are_laid_out_right_to_left() {
        // As browsers display them: `2026-118` -> `118-2026`, `72 280 113` -> `113 280 72`.
        assert_eq!(visual_to_logical("118-2026 مقر"), "رقم 2026-118");
        assert_eq!(visual_to_logical("113 280 72 مقرلاب"), "بالرقم 72 280 113");
    }

    #[test]
    fn brackets_are_mirrored() {
        // RTL text shows the opening bracket at the right with a mirrored glyph.
        assert_eq!(visual_to_logical("(ةظحالم)"), "(ملاحظة)");
    }
}
