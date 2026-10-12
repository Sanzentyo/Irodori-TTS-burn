use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::{IrodoriError, ModelConfig, Result};

#[derive(Clone)]
pub(super) struct TextFrontend {
    tokenizer: Tokenizer,
    bos: Option<u32>,
    pad: u32,
    max_text: usize,
    max_caption: usize,
}

/// CPU-only tokenizer snapshot that can answer queries while the GPU is busy.
#[derive(Clone)]
pub struct TokenCounter(pub(super) TextFrontend);

impl TokenCounter {
    pub fn count(&self, text: &str) -> Result<usize> {
        self.0.count(crate::normalize_text(text).trim())
    }
}

pub(super) struct Tokens {
    pub ids: Tensor<2, Int>,
    pub mask: Tensor<2, Bool>,
    pub count: usize,
}

impl TextFrontend {
    pub fn load(path: &Path, config: &ModelConfig) -> Result<Self> {
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        let bos = if config.text_add_bos {
            Some(
                ["<s>", "<bos>", "[CLS]"]
                    .into_iter()
                    .find_map(|name| tokenizer.token_to_id(name))
                    .ok_or_else(|| {
                        IrodoriError::Tokenizer("checkpoint requires a BOS token".into())
                    })?,
            )
        } else {
            None
        };
        let pad = ["<pad>", "[PAD]", "</s>"]
            .into_iter()
            .find_map(|name| tokenizer.token_to_id(name))
            .ok_or_else(|| IrodoriError::Tokenizer("tokenizer has no padding token".into()))?;
        Ok(Self {
            tokenizer,
            bos,
            pad,
            max_text: config.max_text_len.unwrap_or(256),
            max_caption: config.max_caption_len.unwrap_or(512),
        })
    }

    fn ids(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        Ok(self
            .bos
            .into_iter()
            .chain(encoding.get_ids().iter().copied())
            .collect())
    }

    pub fn count(&self, text: &str) -> Result<usize> {
        Ok(self.ids(text)?.len())
    }

    pub fn encode(&self, text: &str, caption: bool, device: &Device) -> Result<Tokens> {
        let mut ids = self.ids(text)?;
        let limit = if caption {
            self.max_caption
        } else {
            self.max_text
        };
        if ids.len() > limit {
            return Err(IrodoriError::Config(format!(
                "{} exceeds {limit} tokens; split the input",
                if caption { "instructions" } else { "text" }
            )));
        }
        let count = if text.is_empty() { 0 } else { ids.len() };
        if ids.is_empty() {
            ids.push(self.pad);
        }
        let width = ids.len();
        let ids = ids
            .into_iter()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IrodoriError::Tokenizer(e.to_string()))?;
        Ok(Tokens {
            ids: Tensor::from_data(TensorData::new(ids, [1, width]), device),
            mask: Tensor::from_data(TensorData::new(vec![count > 0; width], [1, width]), device),
            count,
        })
    }

    pub fn split(&self, text: &str) -> Result<Vec<String>> {
        let normalized = crate::normalize_text(text);
        if normalized.trim().is_empty() {
            return Err(IrodoriError::Config(
                "text is empty after normalization".into(),
            ));
        }
        if self.count(normalized.trim())? <= self.max_text {
            return Ok(vec![normalized.trim().to_owned()]);
        }
        let mut output = Vec::new();
        for sentence in sentences(&normalized) {
            let mut pending = vec![sentence.to_owned()];
            while let Some(piece) = pending.pop() {
                if self.count(&piece)? <= self.max_text {
                    output.push(piece);
                } else {
                    let (first, second) = split_piece(&piece)?;
                    pending.push(second.to_owned());
                    pending.push(first.to_owned());
                }
            }
        }
        Ok(output)
    }
}

fn sentences(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for (offset, ch) in text.char_indices() {
        if matches!(ch, '。' | '!' | '?' | '\n') {
            let end = offset + ch.len_utf8();
            let part = text[start..end].trim();
            if !part.is_empty() {
                pieces.push(part);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        pieces.push(tail);
    }
    pieces
}

/// Prefer a clause boundary near the middle; never drop a Unicode scalar.
pub(super) fn split_piece(text: &str) -> Result<(&str, &str)> {
    let boundaries = text
        .char_indices()
        .map(|(i, c)| (i + c.len_utf8(), c))
        .filter(|(i, _)| *i < text.len())
        .collect::<Vec<_>>();
    let middle = text.len() / 2;
    let cut = boundaries
        .iter()
        .filter(|(_, c)| matches!(c, '。' | '!' | '?' | '、' | ',' | ' ' | '\n'))
        .min_by_key(|(i, _)| i.abs_diff(middle))
        .or_else(|| boundaries.iter().min_by_key(|(i, _)| i.abs_diff(middle)))
        .map(|(i, _)| *i)
        .ok_or_else(|| {
            IrodoriError::Config("one character exceeds the model's request limit".into())
        })?;
    Ok((&text[..cut], &text[cut..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_normalization_corpus() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/frontend.json")).unwrap();
        for case in cases.as_array().unwrap() {
            assert_eq!(
                crate::normalize_text(case["input"].as_str().unwrap()).trim(),
                case["normalized"].as_str().unwrap(),
                "input: {}",
                case["input"]
            );
        }
    }

    #[test]
    #[ignore = "requires IRODORI_TOKENIZER pointing to the released v4.1 tokenizer.json"]
    fn released_tokenizer_matches_official_corpus_and_splits_without_loss() {
        let path = std::env::var_os("IRODORI_TOKENIZER").expect("set IRODORI_TOKENIZER");
        let config = ModelConfig {
            text_add_bos: true,
            max_text_len: Some(256),
            max_caption_len: Some(512),
            ..Default::default()
        };
        let frontend = TextFrontend::load(Path::new(&path), &config).unwrap();
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/frontend.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let normalized = case["normalized"].as_str().unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(
                if normalized.is_empty() {
                    vec![]
                } else {
                    frontend.ids(normalized).unwrap()
                },
                expected
            );
        }
        let text = "今日は東京の天気をお伝えします。明日は雨になるでしょう。".repeat(50);
        let pieces = frontend.split(&text).unwrap();
        assert!(pieces.len() > 1);
        assert!(
            pieces
                .iter()
                .all(|piece| frontend.count(piece).unwrap() <= 256)
        );
        assert_eq!(pieces.concat(), crate::normalize_text(&text));
        let caption = "落ち着いた声で話してください。".repeat(1000);
        assert!(frontend.count(&caption).unwrap() > 512);
        assert!(frontend.encode(&caption, true, &Device::default()).is_err());
    }

    #[test]
    fn long_text_split_preserves_japanese_emoji_and_every_character() {
        for text in [
            "今日は晴れ、明日は雨です😮‍💨",
            "境界のない日本語",
            "a😀bc",
            "hello world",
        ] {
            let (left, right) = split_piece(text).unwrap();
            assert!(!left.is_empty() && !right.is_empty());
            assert_eq!(format!("{left}{right}"), text);
        }
        assert!(split_piece("あ").is_err());
        assert_eq!(
            sentences("こんにちは。次です!\n最後"),
            ["こんにちは。", "次です!", "最後"]
        );
    }
}
