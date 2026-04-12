//! Text normalization for Japanese TTS input.
//!
//! Mirrors the Python `irodori_tts.text_normalization` module with identical
//! replacement rules and Unicode normalisation (NFKC via the
//! `unicode-normalization` crate).

use once_cell::sync::Lazy;
use regex::Regex;
use unicode_normalization::UnicodeNormalization as _;

/// Simple character-level substitutions applied before regex passes.
const SIMPLE_REPLACE: &[(&str, &str)] = &[
    ("\t", ""),
    ("[n]", ""),
    (r"\[n\]", ""),
    // Japanese fullwidth space → remove
    ("\u{3000}", ""),
    ("？", "?"),
    ("！", "!"),
    ("♥", "♡"),
    ("●", "○"),
    ("◯", "○"),
    ("〇", "○"),
];

/// Compiled regex replacements applied after [`SIMPLE_REPLACE`].
static REGEX_REPLACE: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    vec![
        // Strip specific control / decorative symbols
        (Regex::new(r"[;▼♀♂《》≪≫①②③④⑤⑥]").unwrap(), ""),
        // Various dash-like and line-drawing characters → remove
        (
            Regex::new(r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]")
                .unwrap(),
            "",
        ),
        // Fullwidth tilde / wave-dash → long vowel mark
        (Regex::new(r"[\uff5e\u301C]").unwrap(), "ー"),
        // Three-or-more ellipsis dots → double ellipsis
        (Regex::new(r"…{3,}").unwrap(), "……"),
    ]
});

/// Bracket pairs whose outer layer is stripped by [`strip_outer_brackets`].
const BRACKET_PAIRS: &[(char, char)] = &[
    ('「', '」'),
    ('『', '』'),
    ('（', '）'),
    ('【', '】'),
    ('(', ')'),
];

/// Remove one layer of surrounding brackets if the outermost pair encloses
/// the entire string.  Repeats until no enclosing pair remains.
pub fn strip_outer_brackets(mut text: &str) -> &str {
    'outer: loop {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 2 {
            break;
        }
        let start = chars[0];
        let end = *chars.last().unwrap();

        // Find which bracket pair this is, if any.
        let Some(&(open, close)) = BRACKET_PAIRS.iter().find(|&&(o, c)| o == start && c == end)
        else {
            break;
        };

        // Walk through to verify the opening bracket at index 0 is still open
        // when we reach the last character — i.e. it truly encloses everything.
        let mut depth: i32 = 0;
        for (i, &ch) in chars.iter().enumerate() {
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
            }
            if depth == 0 && i < chars.len() - 1 {
                // The opening bracket closed before the end → not enclosing all.
                break 'outer;
            }
        }

        // depth == 0 at the last character → strip outer pair and repeat.
        let byte_start = start.len_utf8();
        let byte_end = text.len() - end.len_utf8();
        text = &text[byte_start..byte_end];
    }
    text
}

/// Normalise Japanese / mixed-script text for TTS.
///
/// Steps (matching Python implementation order):
/// 1. Simple character-level replacements
/// 2. Regex-based replacements
/// 3. Strip enclosing brackets
/// 4. Unicode NFKC normalisation
/// 5. Collapse ASCII `..` / `...` to `…`
pub fn normalize_text(text: &str) -> String {
    // 1. Simple replacements
    let mut s = text.to_owned();
    for &(old, new) in SIMPLE_REPLACE {
        s = s.replace(old, new);
    }

    // 2. Regex replacements
    for (re, replacement) in REGEX_REPLACE.iter() {
        s = re.replace_all(&s, *replacement).into_owned();
    }

    // 3. Strip outer brackets (operates on &str slices, no allocation)
    let stripped = strip_outer_brackets(&s);
    let mut s = stripped.to_owned();

    // 4. NFKC normalisation
    s = s.nfkc().collect::<String>();

    // 5. Collapse ASCII dots
    s = s.replace("...", "…");
    s = s.replace("..", "…");

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_is_removed() {
        assert_eq!(normalize_text("hello\tworld"), "helloworld");
    }

    #[test]
    fn fullwidth_space_is_removed() {
        assert_eq!(normalize_text("こんにちは\u{3000}世界"), "こんにちは世界");
    }

    #[test]
    fn question_mark_normalized() {
        assert_eq!(normalize_text("本当？"), "本当?");
    }

    #[test]
    fn exclamation_normalized() {
        assert_eq!(normalize_text("すごい！"), "すごい!");
    }

    #[test]
    fn wave_dash_to_katakana_hyphen() {
        assert_eq!(normalize_text("あ〜い"), "あーい");
        assert_eq!(normalize_text("あ\u{ff5e}い"), "あーい");
    }

    #[test]
    fn ellipsis_collapsed() {
        assert_eq!(normalize_text("……………"), "……");
        assert_eq!(normalize_text("..."), "…");
        assert_eq!(normalize_text(".."), "…");
    }

    #[test]
    fn strip_outer_brackets_kagi() {
        assert_eq!(strip_outer_brackets("「こんにちは」"), "こんにちは");
    }

    #[test]
    fn strip_outer_brackets_not_enclosing() {
        // "「A」B」" — outer pair does not enclose all, so nothing stripped
        assert_eq!(strip_outer_brackets("「A」B」"), "「A」B」");
    }

    #[test]
    fn strip_outer_brackets_nested() {
        // Double-nested
        assert_eq!(strip_outer_brackets("「「hello」」"), "hello");
    }

    #[test]
    fn nfkc_applied() {
        // Fullwidth 'A' → ASCII 'A'
        assert_eq!(normalize_text("\u{ff21}"), "A");
    }
}
