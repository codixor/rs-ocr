use std::path::Path;

use rten_text::{Tokenizer, tokenizer::FromJsonError};

use super::SelectionError;

pub(super) fn load_tokenizer(path: &Path) -> Result<Tokenizer, SelectionError> {
    let source = std::fs::read_to_string(path).map_err(FromJsonError::IoError)?;
    from_json(&source)
}

fn from_json(source: &str) -> Result<Tokenizer, SelectionError> {
    let mut configuration: serde_json::Value =
        serde_json::from_str(source).map_err(FromJsonError::JsonError)?;
    if let Some(model) = configuration
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
        && model.get("type").and_then(serde_json::Value::as_str) == Some("BPE")
    {
        // Older Hugging Face exports omit this field. Their BPE builder defaults
        // it to false, while rten-text 0.27 requires it explicitly in JSON.
        // Normalize only in memory; preserve explicit values and downloaded bytes.
        model
            .entry("ignore_merges")
            .or_insert(serde_json::Value::Bool(false));
    }
    let normalized = serde_json::to_string(&configuration).map_err(FromJsonError::JsonError)?;
    Ok(Tokenizer::from_json(&normalized)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        let mut vocabulary = serde_json::Map::new();
        let mut non_printable = 0;
        // A valid BPE file needs the full GPT-2 byte alphabet, even for this
        // three-letter fixture. Expected merge outputs below are hand-specified.
        for byte in 0..=255_u8 {
            let character = char::from(byte);
            let encoded =
                if character.is_control() || character.is_whitespace() || character == '\u{ad}' {
                    let encoded = char::from_u32(256 + non_printable).unwrap();
                    non_printable += 1;
                    encoded
                } else {
                    character
                };
            vocabulary.insert(encoded.to_string(), serde_json::json!(byte));
        }
        for (text, id) in [("ab", 256), ("bc", 257), ("abc", 258)] {
            vocabulary.insert(text.into(), serde_json::json!(id));
        }
        serde_json::json!({"model":{"type":"BPE","vocab":vocabulary,"merges":["a b","b c","a bc"]},"added_tokens":[]})
    }

    #[test]
    fn older_bpe_files_use_the_huggingface_default_for_ignore_merges() {
        let tokenizer = from_json(&fixture().to_string()).unwrap();
        assert_eq!(
            tokenizer.encode("abc", None).unwrap().token_ids(),
            &[256, 99]
        );
    }

    #[test]
    fn an_explicit_ignore_merges_setting_is_preserved() {
        let mut source = fixture();
        source["model"]["ignore_merges"] = true.into();
        let tokenizer = from_json(&source.to_string()).unwrap();
        assert_eq!(tokenizer.encode("abc", None).unwrap().token_ids(), &[258]);
    }
}
