//! Speaker KV-cache scaling for force-speaker guidance.

use burn::tensor::{Tensor, backend::Backend};

use crate::model::attention::CondKvCache;

/// Scale the speaker (aux) K/V tensors in a precomputed KV cache.
///
/// Returns a new `Vec` — burn tensors are immutable.
pub fn scale_speaker_kv_cache<B: Backend>(
    caches: Vec<CondKvCache<B>>,
    scale: f32,
    max_layers: Option<usize>,
) -> Vec<CondKvCache<B>> {
    let n = max_layers.map_or(caches.len(), |m| m.min(caches.len()));
    caches
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            if i < n {
                let CondKvCache {
                    text_k,
                    text_v,
                    aux_k,
                    aux_v,
                    ctx_mask,
                    joint_mask,
                    ctx_k: _,
                    ctx_v: _,
                } = c;

                let new_aux_k = aux_k.map(|k| k * scale);
                let new_aux_v = aux_v.map(|v| v * scale);

                // Recompute pre-concatenated K/V with the scaled aux portion.
                let new_ctx_k = match &new_aux_k {
                    Some(ak) => Tensor::cat(vec![text_k.clone(), ak.clone()], 1),
                    None => text_k.clone(),
                };
                let new_ctx_v = match &new_aux_v {
                    Some(av) => Tensor::cat(vec![text_v.clone(), av.clone()], 1),
                    None => text_v.clone(),
                };

                CondKvCache {
                    text_k,
                    text_v,
                    aux_k: new_aux_k,
                    aux_v: new_aux_v,
                    ctx_k: new_ctx_k,
                    ctx_v: new_ctx_v,
                    ctx_mask,
                    joint_mask,
                }
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray<f32>;

    fn make_unit_cache(
        seq_text: usize,
        seq_aux: usize,
        heads: usize,
        head_dim: usize,
        device: &<B as burn::tensor::backend::Backend>::Device,
    ) -> CondKvCache<B> {
        let text_k = Tensor::<B, 4>::ones([1, seq_text, heads, head_dim], device);
        let text_v = Tensor::<B, 4>::ones([1, seq_text, heads, head_dim], device);
        let aux_k = Tensor::<B, 4>::ones([1, seq_aux, heads, head_dim], device);
        let aux_v = Tensor::<B, 4>::ones([1, seq_aux, heads, head_dim], device);
        let ctx_k = Tensor::cat(vec![text_k.clone(), aux_k.clone()], 1);
        let ctx_v = Tensor::cat(vec![text_v.clone(), aux_v.clone()], 1);
        let ctx_mask =
            Tensor::<B, 2, burn::tensor::Bool>::full([1, seq_text + seq_aux], true, device);
        CondKvCache {
            text_k,
            text_v,
            aux_k: Some(aux_k),
            aux_v: Some(aux_v),
            ctx_k,
            ctx_v,
            ctx_mask,
            joint_mask: None,
        }
    }

    /// Scaling by 2.0 must double the aux K/V values and rebuild ctx accordingly.
    #[test]
    fn scale_speaker_kv_cache_doubles_aux_and_rebuilds_ctx() {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let (seq_text, seq_aux, heads, head_dim) = (4, 3, 2, 8);
        let cache = make_unit_cache(seq_text, seq_aux, heads, head_dim, &device);
        let scaled = scale_speaker_kv_cache(vec![cache], 2.0, None);
        let c = &scaled[0];

        // aux K/V must be scaled by 2 (originally 1s → now 2s)
        let aux_k_max: f32 = c.aux_k.as_ref().unwrap().clone().max().into_scalar();
        let aux_k_min: f32 = c.aux_k.as_ref().unwrap().clone().min().into_scalar();
        assert!((aux_k_max - 2.0).abs() < 1e-6, "aux_k max should be 2.0");
        assert!((aux_k_min - 2.0).abs() < 1e-6, "aux_k min should be 2.0");

        // text K/V must be unchanged (originally 1s → still 1s)
        let text_k_max: f32 = c.text_k.clone().max().into_scalar();
        assert!((text_k_max - 1.0).abs() < 1e-6, "text_k must be unchanged");

        // ctx_k = [text(1s) | aux(2s)]: text portion = 1, aux portion = 2
        let ctx_k_data = c.ctx_k.clone().into_data();
        let vals = ctx_k_data.to_vec::<f32>().unwrap();
        let text_vals = &vals[..seq_text * heads * head_dim];
        let aux_vals = &vals[seq_text * heads * head_dim..];
        assert!(
            text_vals.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "ctx_k text portion unchanged"
        );
        assert!(
            aux_vals.iter().all(|&v| (v - 2.0).abs() < 1e-6),
            "ctx_k aux portion scaled"
        );
    }

    /// max_layers=1 must only scale the first cache; the second remains unchanged.
    #[test]
    fn scale_speaker_kv_cache_respects_max_layers() {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let (seq_text, seq_aux, heads, head_dim) = (2, 2, 1, 4);
        let caches = vec![
            make_unit_cache(seq_text, seq_aux, heads, head_dim, &device),
            make_unit_cache(seq_text, seq_aux, heads, head_dim, &device),
        ];
        let scaled = scale_speaker_kv_cache(caches, 3.0, Some(1));

        // Layer 0 — aux_k should be 3.0
        let aux_k0_max: f32 = scaled[0]
            .aux_k
            .as_ref()
            .unwrap()
            .clone()
            .max()
            .into_scalar();
        assert!(
            (aux_k0_max - 3.0).abs() < 1e-6,
            "layer 0 aux_k should be 3.0"
        );

        // Layer 1 — aux_k should still be 1.0 (not scaled)
        let aux_k1_max: f32 = scaled[1]
            .aux_k
            .as_ref()
            .unwrap()
            .clone()
            .max()
            .into_scalar();
        assert!(
            (aux_k1_max - 1.0).abs() < 1e-6,
            "layer 1 aux_k should be unchanged (1.0)"
        );
    }
}
