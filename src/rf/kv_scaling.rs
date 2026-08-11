//! Speaker KV-cache scaling for force-speaker guidance.

use burn::tensor::{Tensor, backend::Backend};

use crate::model::attention::{CondKvCache, SpeakerKvRange};

/// Scale the speaker range in each packed K/V cache.
///
/// The canonical context order is `[text | speaker? | caption?]`. Prefix and
/// suffix slices are copied unchanged while only the recorded speaker slice is
/// multiplied. This keeps caption conditioning intact without retaining split
/// context tensors alongside the packed cache.
pub fn scale_speaker_kv_cache<B: Backend>(
    caches: Vec<CondKvCache<B>>,
    scale: f32,
    max_layers: Option<usize>,
) -> Vec<CondKvCache<B>> {
    let n = max_layers.map_or(caches.len(), |m| m.min(caches.len()));
    caches
        .into_iter()
        .enumerate()
        .map(|(i, cache)| {
            if i >= n || scale == 1.0 {
                return cache;
            }
            let Some(speaker_range) = cache.speaker_range else {
                return cache;
            };

            let CondKvCache {
                ctx_k,
                ctx_v,
                ctx_mask,
                joint_mask,
                speaker_range: _,
                packed_ctx_kv_wgsl: _,
                joint_mask_wgsl: _,
            } = cache;

            CondKvCache {
                ctx_k: scale_packed_speaker(ctx_k, speaker_range, scale),
                ctx_v: scale_packed_speaker(ctx_v, speaker_range, scale),
                ctx_mask,
                joint_mask,
                speaker_range: Some(speaker_range),
                // Scaling replaces both source tensors, so a previously
                // packed WGPU view and its host-derived mask proof would be
                // stale and must be rebuilt.
                packed_ctx_kv_wgsl: None,
                joint_mask_wgsl: None,
            }
        })
        .collect()
}

fn scale_packed_speaker<B: Backend>(
    packed: Tensor<B, 4>,
    range: SpeakerKvRange,
    scale: f32,
) -> Tensor<B, 4> {
    let seq = packed.dims()[1];
    assert!(
        range.end() <= seq,
        "speaker KV range end {} exceeds packed sequence length {seq}",
        range.end()
    );
    if range.len() == 0 || scale == 1.0 {
        return packed;
    }

    let mut parts = Vec::with_capacity(3);
    if range.start() > 0 {
        parts.push(packed.clone().narrow(1, 0, range.start()));
    }
    parts.push(packed.clone().narrow(1, range.start(), range.len()) * scale);
    if range.end() < seq {
        parts.push(packed.narrow(1, range.end(), seq - range.end()));
    }

    if parts.len() == 1 {
        parts.pop().expect("speaker slice was inserted")
    } else {
        Tensor::cat(parts, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::attention::WgslJointMask;
    use burn::backend::NdArray;
    use burn::tensor::{Bool, Tensor};

    type B = NdArray<f32>;

    fn make_unit_cache(
        seq_text: usize,
        seq_speaker: usize,
        seq_caption: usize,
        heads: usize,
        head_dim: usize,
        device: &<B as Backend>::Device,
    ) -> CondKvCache<B> {
        let text_k = Tensor::<B, 4>::ones([1, seq_text, heads, head_dim], device);
        let text_v = text_k.clone();
        let speaker_k = Tensor::<B, 4>::ones([1, seq_speaker, heads, head_dim], device);
        let speaker_v = speaker_k.clone();

        let mut keys = vec![text_k, speaker_k];
        let mut values = vec![text_v, speaker_v];
        if seq_caption > 0 {
            keys.push(Tensor::<B, 4>::ones([1, seq_caption, heads, head_dim], device) * 4.0);
            values.push(Tensor::<B, 4>::ones([1, seq_caption, heads, head_dim], device) * 4.0);
        }

        CondKvCache {
            ctx_k: Tensor::cat(keys, 1),
            ctx_v: Tensor::cat(values, 1),
            ctx_mask: Tensor::<B, 2, Bool>::full(
                [1, seq_text + seq_speaker + seq_caption],
                true,
                device,
            ),
            joint_mask: None,
            speaker_range: Some(SpeakerKvRange::from_start_len(seq_text, seq_speaker)),
            packed_ctx_kv_wgsl: None,
            joint_mask_wgsl: None,
        }
    }

    fn packed_values(cache: &CondKvCache<B>) -> Vec<f32> {
        cache.ctx_k.clone().into_data().to_vec::<f32>().unwrap()
    }

    #[test]
    fn scale_speaker_kv_cache_scales_only_packed_speaker_range() {
        let device = <B as Backend>::Device::default();
        let (seq_text, seq_speaker, heads, head_dim) = (4, 3, 2, 8);
        let cache = make_unit_cache(seq_text, seq_speaker, 0, heads, head_dim, &device);
        let scaled = scale_speaker_kv_cache(vec![cache], 2.0, None);
        let values = packed_values(&scaled[0]);
        let text_end = seq_text * heads * head_dim;

        assert!(values[..text_end].iter().all(|&value| value == 1.0));
        assert!(values[text_end..].iter().all(|&value| value == 2.0));
        assert_eq!(
            scaled[0].speaker_range,
            Some(SpeakerKvRange::from_start_len(seq_text, seq_speaker))
        );
    }

    #[test]
    fn scale_speaker_kv_cache_respects_max_layers() {
        let device = <B as Backend>::Device::default();
        let (seq_text, seq_speaker, heads, head_dim) = (2, 2, 1, 4);
        let caches = vec![
            make_unit_cache(seq_text, seq_speaker, 0, heads, head_dim, &device),
            make_unit_cache(seq_text, seq_speaker, 0, heads, head_dim, &device),
        ];
        let scaled = scale_speaker_kv_cache(caches, 3.0, Some(1));
        let speaker_offset = seq_text * heads * head_dim;

        assert!(
            packed_values(&scaled[0])[speaker_offset..]
                .iter()
                .all(|&value| value == 3.0)
        );
        assert!(
            packed_values(&scaled[1])[speaker_offset..]
                .iter()
                .all(|&value| value == 1.0)
        );
    }

    #[test]
    fn scale_speaker_kv_cache_preserves_caption_values_and_order() {
        let device = <B as Backend>::Device::default();
        let (seq_text, seq_speaker, seq_caption, heads, head_dim) = (2, 3, 4, 1, 2);
        let cache = make_unit_cache(seq_text, seq_speaker, seq_caption, heads, head_dim, &device);
        let scaled = scale_speaker_kv_cache(vec![cache], 2.0, None);
        let values = packed_values(&scaled[0]);
        let text_end = seq_text * heads * head_dim;
        let speaker_end = text_end + seq_speaker * heads * head_dim;

        assert!(values[..text_end].iter().all(|&value| value == 1.0));
        assert!(
            values[text_end..speaker_end]
                .iter()
                .all(|&value| value == 2.0)
        );
        assert!(values[speaker_end..].iter().all(|&value| value == 4.0));
    }

    #[test]
    fn inverse_scaling_restores_packed_kv() {
        let device = <B as Backend>::Device::default();
        let cache = make_unit_cache(2, 3, 4, 1, 2, &device);
        let original_k = cache.ctx_k.clone();
        let original_v = cache.ctx_v.clone();

        let scaled = scale_speaker_kv_cache(vec![cache], 2.0, None);
        let restored = scale_speaker_kv_cache(scaled, 0.5, None)
            .pop()
            .expect("one cache");

        let k_diff: f32 = (original_k - restored.ctx_k).abs().max().into_scalar();
        let v_diff: f32 = (original_v - restored.ctx_v).abs().max().into_scalar();
        assert_eq!(k_diff, 0.0);
        assert_eq!(v_diff, 0.0);
    }

    #[test]
    fn cache_without_speaker_is_unchanged() {
        let device = <B as Backend>::Device::default();
        let ctx_k = Tensor::<B, 4>::ones([1, 3, 1, 2], &device);
        let cache = CondKvCache {
            ctx_v: ctx_k.clone(),
            ctx_k: ctx_k.clone(),
            ctx_mask: Tensor::<B, 2, Bool>::full([1, 3], true, &device),
            joint_mask: None,
            speaker_range: None,
            packed_ctx_kv_wgsl: None,
            joint_mask_wgsl: None,
        };

        let unchanged = scale_speaker_kv_cache(vec![cache], 9.0, None)
            .pop()
            .expect("one cache");
        let max_diff: f32 = (ctx_k - unchanged.ctx_k).abs().max().into_scalar();
        assert_eq!(max_diff, 0.0);
    }

    #[test]
    fn speaker_scaling_invalidates_wgsl_derived_state() {
        let device = <B as Backend>::Device::default();
        let mut cache = make_unit_cache(2, 2, 0, 1, 2, &device);
        cache.packed_ctx_kv_wgsl = Some(Tensor::<B, 4>::stack::<5>(
            vec![cache.ctx_k.clone(), cache.ctx_v.clone()],
            0,
        ));
        cache.joint_mask_wgsl = Some(WgslJointMask::AllValid);

        let scaled = scale_speaker_kv_cache(vec![cache], 2.0, None)
            .pop()
            .expect("one cache");
        assert!(scaled.packed_ctx_kv_wgsl.is_none());
        assert!(scaled.joint_mask_wgsl.is_none());
    }
}
