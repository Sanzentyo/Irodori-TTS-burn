//! CPU text statistics and latent-tail detection used by synthesis frontends.

use crate::Result;

const ANNOTATION_EMOJIS: &[&str] = &[
    "😮\u{200d}💨",
    "⏱️",
    "⏩",
    "⏸️",
    "🌬️",
    "🍭",
    "🎛️",
    "🎭",
    "🎵",
    "🐢",
    "🐱",
    "👂",
    "👃",
    "👅",
    "👌",
    "👏",
    "💋",
    "💥",
    "💦",
    "💪",
    "📄",
    "📞",
    "📢",
    "📣",
    "😆",
    "😊",
    "😌",
    "😎",
    "😏",
    "😒",
    "😖",
    "😟",
    "😠",
    "😪",
    "😭",
    "😮",
    "😰",
    "😱",
    "😲",
    "😴",
    "🙄",
    "🙏",
    "🤐",
    "🤔",
    "🤢",
    "🤧",
    "🤭",
    "🥤",
    "🥱",
    "🥴",
    "🥵",
    "🥹",
    "🥺",
    "🫣",
    "🫶",
    "📖",
];

fn count_annotation_emojis(text: &str) -> usize {
    let mut count = 0;
    let mut offset = 0;
    while offset < text.len() {
        let rest = &text[offset..];
        if let Some(matched) = ANNOTATION_EMOJIS
            .iter()
            .filter(|emoji| rest.starts_with(**emoji))
            .max_by_key(|emoji| emoji.chars().count())
        {
            count += 1;
            offset += matched.len();
        } else {
            let Some(character) = rest.chars().next() else {
                break;
            };
            offset += character.len_utf8();
        }
    }
    count
}

fn log1p_cap(value: usize, cap: usize) -> f32 {
    (value.min(cap) as f32).ln_1p() / (cap as f32).ln_1p()
}

/// The released v4/v4.1 predictor's fourteen normalized text statistics.
pub fn build_duration_features(
    text: &str,
    token_count: usize,
    max_text_len: usize,
    has_speaker: bool,
    aux_dim: usize,
) -> Result<Vec<f32>> {
    if aux_dim != 14 || max_text_len == 0 {
        return Err(crate::IrodoriError::Config(
            "duration features require width 14 and a positive text limit".into(),
        ));
    }
    let char_count = text.chars().count().max(1);
    let kana_count = text
        .chars()
        .filter(|character| matches!(*character as u32, 0x3040..=0x30ff))
        .count();
    let kanji_count = text
        .chars()
        .filter(|character| {
            matches!(
                *character as u32,
                0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2fa1f
            )
        })
        .count();
    let alnum_count = text
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .count();
    let count = |needle: &str| text.matches(needle).count();

    Ok(vec![
        token_count.min(max_text_len) as f32 / max_text_len as f32,
        (char_count.min(512) as f32).ln_1p() / 512.0_f32.ln_1p(),
        token_count as f32 / char_count as f32,
        log1p_cap(count("。") + count("."), 8),
        log1p_cap(count("、") + count(","), 16),
        log1p_cap(count("ー"), 8),
        log1p_cap(count("…"), 8),
        log1p_cap(count("！") + count("!"), 8),
        log1p_cap(count("？") + count("?"), 8),
        log1p_cap(count_annotation_emojis(text), 8),
        kana_count as f32 / char_count as f32,
        kanji_count as f32 / char_count as f32,
        alnum_count as f32 / char_count as f32,
        if has_speaker { 1.0 } else { 0.0 },
    ])
}

/// First near-flat, near-zero latent window, with zero padding at the tail.
/// A zero result means that the caller should preserve the full latent.
///
/// # Panics
/// Panics if the supplied flat data does not have shape `[total_t, latent_dim]`
/// or if the window geometry overflows the host's index type.
pub fn find_flattening_point(
    latent_data: &[f32],
    total_t: usize,
    latent_dim: usize,
    window_size: usize,
    std_threshold: f32,
    mean_threshold: f32,
) -> usize {
    assert_eq!(
        total_t.checked_mul(latent_dim),
        Some(latent_data.len()),
        "latent shape mismatch"
    );
    assert!(
        total_t
            .checked_add(window_size)
            .and_then(|t| t.checked_mul(latent_dim))
            .is_some(),
        "latent window overflow"
    );
    if total_t == 0 || window_size == 0 {
        return total_t;
    }
    // Pad with zeros so the window always has `window_size` frames.
    let padded_t = total_t + window_size;

    for i in 0..(padded_t - window_size) {
        let w_start = i * latent_dim;
        let w_end = (i + window_size) * latent_dim;

        let window: &[f32] = if w_end <= latent_data.len() {
            &latent_data[w_start..w_end]
        } else {
            // Window extends into the zero-padding — compute manually.
            let avail = latent_data.len().saturating_sub(w_start);
            // Mean and std over the available elements + zeros for the rest.
            let n = (window_size * latent_dim) as f32;
            let sum: f32 = latent_data[w_start..w_start + avail].iter().sum();
            let mean = sum / n;
            let sq_sum: f32 = latent_data[w_start..w_start + avail]
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                + (n as usize - avail) as f32 * mean.powi(2); // zero elements
            let std = (sq_sum / n).sqrt();
            if std < std_threshold && mean.abs() < mean_threshold {
                return i;
            }
            continue;
        };

        let n = window.len() as f32;
        let mean: f32 = window.iter().sum::<f32>() / n;
        let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        if std < std_threshold && mean.abs() < mean_threshold {
            return i;
        }
    }
    total_t
}
