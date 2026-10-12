//! WGPU-only execution policy for the exact v4-Small ModernBERT encoder.

use burn::tensor::Device;
use burn::{
    backend::wgpu::{CubeTensor, WgpuRuntime},
    nn::{LayerNorm, Linear},
    tensor::{Bool, DType, Int, Tensor},
};

use crate::kernels::modern_bert_residual_layer_norm::{
    BATCH, SEQUENCE, V4_BOUNDARIES, WIDTH, modern_bert_residual_layer_norm_wgsl,
    supports_modern_bert_residual_layer_norm_device,
};

#[cfg(test)]
use super::ModernBertConfig;
#[cfg(test)]
use crate::kernels::modern_bert_residual_layer_norm::ELEMENTS;

use super::{
    ModernBertLayerType, ModernBertModel, PretrainedTextBackbone, SharedModernBertConditioner,
    bool_mask_to_float, full_attention_valid_mask, precompute_modern_bert_rope,
    sliding_attention_valid_mask,
};

const V4_LAYERS: usize = 25;
const V4_VOCAB_SIZE: usize = 102_400;
const V4_INTERMEDIATE_SIZE: usize = 3_072;
const V4_NUM_HEADS: usize = 12;
const V4_HEAD_DIM: usize = 64;
const V4_MAX_POSITIONS: usize = 8_192;
const V4_SLIDING_HALF_WINDOW: usize = 64;
const V4_FULL_ROPE_THETA: f64 = 160_000.0;
const V4_SLIDING_ROPE_THETA: f64 = 10_000.0;
const V4_NORM_EPSILON: f64 = 1.0e-5;

type WgpuDevice = Device;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WgslInputMetadata {
    ids_shape: [usize; 2],
    ids_strides: [usize; 2],
    mask_shape: [usize; 2],
    mask_strides: [usize; 2],
    ids_contiguous: bool,
    mask_contiguous: bool,
    ids_are_i32: bool,
    mask_is_bool: bool,
    same_device: bool,
}

impl WgslInputMetadata {
    fn supports_exact_v4(self) -> bool {
        self.ids_shape == [BATCH, SEQUENCE]
            && self.ids_strides == [SEQUENCE, 1]
            && self.mask_shape == [BATCH, SEQUENCE]
            && self.mask_strides == [SEQUENCE, 1]
            && self.ids_contiguous
            && self.mask_contiguous
            && self.ids_are_i32
            && self.mask_is_bool
            && self.same_device
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MlpResidualTarget {
    AttentionNorm(usize),
    FinalNorm,
}

const fn mlp_residual_target(layer_index: usize, layer_count: usize) -> MlpResidualTarget {
    if layer_index + 1 < layer_count {
        MlpResidualTarget::AttentionNorm(layer_index + 1)
    } else {
        MlpResidualTarget::FinalNorm
    }
}

fn input_metadata(
    input_ids: &Tensor<2, Int>,
    attention_mask: &Tensor<2, Bool>,
) -> Option<WgslInputMetadata> {
    let ids = input_ids
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let mask = attention_mask
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    if ids.meta.num_dims() != 2 || mask.meta.num_dims() != 2 {
        return None;
    }
    Some(WgslInputMetadata {
        ids_shape: ids.meta.shape().dims::<2>(),
        ids_strides: [ids.meta.strides()[0], ids.meta.strides()[1]],
        mask_shape: mask.meta.shape().dims::<2>(),
        mask_strides: [mask.meta.strides()[0], mask.meta.strides()[1]],
        ids_contiguous: ids.is_contiguous(),
        mask_contiguous: mask.is_contiguous(),
        ids_are_i32: ids.dtype == DType::I32,
        mask_is_bool: matches!(mask.dtype, DType::Bool(_)),
        same_device: ids.device == mask.device,
    })
}

fn linear_matches(linear: &Linear, input: usize, output: usize, device: &WgpuDevice) -> bool {
    let weight = linear.weight.val();
    if weight.dims() != [input, output]
        || weight.dtype() != DType::F32
        || weight.device() != device.clone()
        || linear.bias.is_some()
    {
        return false;
    }
    let weight = weight
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    weight.is_contiguous()
        && weight.meta.num_dims() == 2
        && &weight.meta.strides()[..] == [output, 1].as_slice()
}

fn norm_matches(norm: &LayerNorm) -> bool {
    let gamma = norm.gamma.val();
    if gamma.dims() != [WIDTH] || gamma.dtype() != DType::F32 || norm.beta.is_some() {
        return false;
    }
    let gamma = gamma
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    gamma.meta.num_dims() == 1
        && gamma.is_contiguous()
        && &gamma.meta.strides()[..] == [1].as_slice()
}

fn norm_supports_exact_launch(norm: &LayerNorm, device: &WgpuDevice) -> bool {
    if !norm_matches(norm) {
        return false;
    }
    let gamma = norm
        .gamma
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    Device::from(gamma.device.clone()) == device.clone()
        && supports_modern_bert_residual_layer_norm_device(&gamma)
}

fn rank2_exact_layout(
    tensor: &CubeTensor<WgpuRuntime>,
    shape: [usize; 2],
    strides: [usize; 2],
) -> bool {
    tensor.dtype == DType::F32
        && tensor.meta.num_dims() == 2
        && tensor.meta.shape().dims::<2>() == shape
        && tensor.is_contiguous()
        && &tensor.meta.strides()[..] == strides.as_slice()
}

fn embedding_matches_v4(model: &ModernBertModel, device: &WgpuDevice) -> bool {
    let weight = model.embeddings.tok_embeddings.weight.val();
    weight.device() == device.clone()
        && model.embeddings.norm.gamma.val().device() == device.clone()
        && rank2_exact_layout(
            &weight
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            [V4_VOCAB_SIZE, WIDTH],
            [WIDTH, 1],
        )
        && norm_matches(&model.embeddings.norm)
}

fn model_topology_matches_v4(model: &ModernBertModel, device: &WgpuDevice) -> bool {
    if model.layers.len() != V4_LAYERS
        || model.head_dim != V4_HEAD_DIM
        || model.max_position_embeddings != V4_MAX_POSITIONS
        || model.sliding_half_window != V4_SLIDING_HALF_WINDOW
        || model.full_rope_theta != V4_FULL_ROPE_THETA
        || model.sliding_rope_theta != V4_SLIDING_ROPE_THETA
        || model.norm_eps != V4_NORM_EPSILON
        || !embedding_matches_v4(model, device)
        || !norm_matches(&model.final_norm)
    {
        return false;
    }

    model.layers.iter().enumerate().all(|(index, layer)| {
        let expected_layer_type = if index.is_multiple_of(3) {
            ModernBertLayerType::Full
        } else {
            ModernBertLayerType::Sliding
        };
        let attention_norm_matches = if index == 0 {
            layer.attn_norm.is_none()
        } else {
            layer.attn_norm.as_ref().is_some_and(norm_matches)
        };
        layer.layer_type == expected_layer_type
            && attention_norm_matches
            && norm_matches(&layer.mlp_norm)
            && layer.attn.num_heads == V4_NUM_HEADS
            && layer.attn.head_dim == V4_HEAD_DIM
            && linear_matches(&layer.attn.wqkv, WIDTH, 3 * WIDTH, device)
            && linear_matches(&layer.attn.wo, WIDTH, WIDTH, device)
            && layer.mlp.intermediate_size == V4_INTERMEDIATE_SIZE
            && linear_matches(&layer.mlp.wi, WIDTH, 2 * V4_INTERMEDIATE_SIZE, device)
            && linear_matches(&layer.mlp.wo, V4_INTERMEDIATE_SIZE, WIDTH, device)
    })
}

fn boundary_norms(model: &ModernBertModel) -> impl Iterator<Item = &LayerNorm> {
    model
        .layers
        .iter()
        .map(|layer| &layer.mlp_norm)
        .chain(
            model
                .layers
                .iter()
                .skip(1)
                .filter_map(|layer| layer.attn_norm.as_ref()),
        )
        .chain(core::iter::once(&model.final_norm))
}

fn fused_boundary(
    residual: Tensor<3>,
    branch: Tensor<3>,
    norm: &LayerNorm,
) -> Option<(Tensor<3>, Tensor<3>)> {
    if norm.beta.is_some() {
        return None;
    }
    let (updated, normalized) = modern_bert_residual_layer_norm_wgsl(
        residual
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        branch
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        norm.gamma
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
    )
    .ok()?;
    Some((
        Tensor::from_primitive::<crate::WgpuRaw>(updated),
        Tensor::from_primitive::<crate::WgpuRaw>(normalized),
    ))
}

impl ModernBertModel {
    /// Fail-closed selector for the measured B1/S3/D768 f32 encoder path.
    pub(crate) fn supports_forward_wgsl(
        &self,
        input_ids: &Tensor<2, Int>,
        attention_mask: &Tensor<2, Bool>,
    ) -> bool {
        let Some(metadata) = input_metadata(input_ids, attention_mask) else {
            return false;
        };
        let input_device = input_ids.device();
        if !metadata.supports_exact_v4() || !model_topology_matches_v4(self, &input_device) {
            return false;
        }

        let mut boundary_count = 0;
        let supported = boundary_norms(self).all(|norm| {
            boundary_count += 1;
            norm_supports_exact_launch(norm, &input_device)
        });
        supported && boundary_count == V4_BOUNDARIES
    }

    /// Select the exact WGSL carry loop or fail safely to the generic forward.
    pub(crate) fn forward_wgsl(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        if !self.supports_forward_wgsl(&input_ids, &attention_mask) {
            return self.forward(input_ids, attention_mask);
        }

        let fallback_ids = input_ids.clone();
        let fallback_mask = attention_mask.clone();
        self.try_forward_wgsl_exact(input_ids, attention_mask)
            .unwrap_or_else(|| self.forward(fallback_ids, fallback_mask))
    }

    fn try_forward_wgsl_exact(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Bool>,
    ) -> Option<Tensor<3>> {
        let device = input_ids.device();
        let mut hidden_states = self.embeddings.forward(input_ids);
        // Layer zero has no attention norm; its embedding output is both the
        // residual owner and the first attention input.
        let mut attention_input = hidden_states.clone();
        let full_rope =
            precompute_modern_bert_rope(self.head_dim, SEQUENCE, self.full_rope_theta, &device);
        let sliding_rope =
            precompute_modern_bert_rope(self.head_dim, SEQUENCE, self.sliding_rope_theta, &device);
        let full_mask = full_attention_valid_mask(attention_mask.clone());
        let sliding_mask =
            sliding_attention_valid_mask(attention_mask.clone(), self.sliding_half_window, &device);

        for (index, layer) in self.layers.iter().enumerate() {
            let (rope, valid_mask) = match layer.layer_type {
                ModernBertLayerType::Full => (&full_rope, full_mask.clone()),
                ModernBertLayerType::Sliding => (&sliding_rope, sliding_mask.clone()),
            };
            let backend_mask = valid_mask.expand([BATCH, 1, SEQUENCE, SEQUENCE]).bool_not();
            let attention_branch = layer.attn.forward(attention_input, rope, backend_mask);
            let (attention_residual, mlp_input) =
                fused_boundary(hidden_states, attention_branch, &layer.mlp_norm)?;
            let mlp_branch = layer.mlp.forward(mlp_input);

            match mlp_residual_target(index, self.layers.len()) {
                MlpResidualTarget::AttentionNorm(next_index) => {
                    let next_norm = self.layers.get(next_index)?.attn_norm.as_ref()?;
                    (hidden_states, attention_input) =
                        fused_boundary(attention_residual, mlp_branch, next_norm)?;
                }
                MlpResidualTarget::FinalNorm => {
                    let (_, normalized) =
                        fused_boundary(attention_residual, mlp_branch, &self.final_norm)?;
                    return Some(normalized * bool_mask_to_float(attention_mask, &device));
                }
            }
        }
        None
    }
}

impl PretrainedTextBackbone {
    pub(crate) fn forward_wgsl(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        self.backbone.forward_wgsl(input_ids, attention_mask)
    }
}

impl SharedModernBertConditioner {
    pub(crate) fn encode_text_wgsl(
        &self,
        input_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let state = self
            .pretrained_text_backbone
            .forward_wgsl(input_ids, mask.clone());
        self.text_encoder.forward(state, mask)
    }
}

#[cfg(test)]
fn forward_carried_generic(
    model: &ModernBertModel,
    input_ids: Tensor<2, Int>,
    attention_mask: Tensor<2, Bool>,
) -> Tensor<3> {
    let [_, sequence] = input_ids.dims();
    let device = input_ids.device();
    let mut hidden_states = model.embeddings.forward(input_ids);
    let mut attention_input = hidden_states.clone();
    let full_rope =
        precompute_modern_bert_rope(model.head_dim, sequence, model.full_rope_theta, &device);
    let sliding_rope =
        precompute_modern_bert_rope(model.head_dim, sequence, model.sliding_rope_theta, &device);
    let full_mask = full_attention_valid_mask(attention_mask.clone());
    let sliding_mask =
        sliding_attention_valid_mask(attention_mask.clone(), model.sliding_half_window, &device);

    for (index, layer) in model.layers.iter().enumerate() {
        let (rope, valid_mask) = match layer.layer_type {
            ModernBertLayerType::Full => (&full_rope, full_mask.clone()),
            ModernBertLayerType::Sliding => (&sliding_rope, sliding_mask.clone()),
        };
        let backend_mask = valid_mask.expand([1, 1, sequence, sequence]).bool_not();
        let attention_branch = layer.attn.forward(attention_input, rope, backend_mask);
        let attention_residual = hidden_states + attention_branch;
        let mlp_input = layer.mlp_norm.forward(attention_residual.clone());
        let mlp_branch = layer.mlp.forward(mlp_input);
        hidden_states = attention_residual + mlp_branch;

        match mlp_residual_target(index, model.layers.len()) {
            MlpResidualTarget::AttentionNorm(next_index) => {
                attention_input = model.layers[next_index]
                    .attn_norm
                    .as_ref()
                    .expect("nonzero ModernBERT layers own attention norms")
                    .forward(hidden_states.clone());
            }
            MlpResidualTarget::FinalNorm => {
                let normalized = model.final_norm.forward(hidden_states);
                return normalized * bool_mask_to_float(attention_mask, &device);
            }
        }
    }
    model.final_norm.forward(hidden_states) * bool_mask_to_float(attention_mask, &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exact_input_metadata() -> WgslInputMetadata {
        WgslInputMetadata {
            ids_shape: [1, 3],
            ids_strides: [3, 1],
            mask_shape: [1, 3],
            mask_strides: [3, 1],
            ids_contiguous: true,
            mask_contiguous: true,
            ids_are_i32: true,
            mask_is_bool: true,
            same_device: true,
        }
    }

    #[test]
    fn exact_input_selector_fails_closed() {
        let exact = exact_input_metadata();
        assert!(exact.supports_exact_v4());
        assert!(
            !WgslInputMetadata {
                ids_shape: [2, 3],
                mask_shape: [2, 3],
                ..exact
            }
            .supports_exact_v4()
        );
        assert!(
            !WgslInputMetadata {
                ids_shape: [1, 4],
                mask_shape: [1, 4],
                ..exact
            }
            .supports_exact_v4()
        );
        assert!(
            !WgslInputMetadata {
                ids_strides: [1, 1],
                ids_contiguous: false,
                ..exact
            }
            .supports_exact_v4()
        );
        assert!(
            !WgslInputMetadata {
                ids_are_i32: false,
                ..exact
            }
            .supports_exact_v4()
        );
        assert!(
            !WgslInputMetadata {
                mask_is_bool: false,
                ..exact
            }
            .supports_exact_v4()
        );
        assert!(
            !WgslInputMetadata {
                same_device: false,
                ..exact
            }
            .supports_exact_v4()
        );
    }

    #[test]
    fn v4_boundary_plan_has_layer_zero_and_final_ownership() {
        let targets = (0..V4_LAYERS)
            .map(|index| mlp_residual_target(index, V4_LAYERS))
            .collect::<Vec<_>>();
        assert_eq!(targets.len(), V4_LAYERS);
        assert_eq!(targets[0], MlpResidualTarget::AttentionNorm(1));
        assert!(!targets.contains(&MlpResidualTarget::AttentionNorm(0)));
        assert_eq!(targets[V4_LAYERS - 2], MlpResidualTarget::AttentionNorm(24));
        assert_eq!(targets[V4_LAYERS - 1], MlpResidualTarget::FinalNorm);
        assert_eq!(
            V4_LAYERS
                + targets
                    .iter()
                    .filter(|target| matches!(target, MlpResidualTarget::AttentionNorm(_)))
                    .count()
                + targets
                    .iter()
                    .filter(|target| matches!(target, MlpResidualTarget::FinalNorm))
                    .count(),
            V4_BOUNDARIES
        );
    }

    #[test]
    fn tiny_generic_carry_matches_existing_forward() {
        let device = Default::default();
        let config = ModernBertConfig::tiny();
        let model = ModernBertModel::new(&config, &device);
        let input_ids = Tensor::<2, Int>::from_data([[2, 7, 3]], &device);
        let mask = Tensor::<2, Bool>::from_data([[true, true, false]], &device);
        let expected = model.forward(input_ids.clone(), mask.clone());
        let carried = forward_carried_generic(&model, input_ids, mask);
        let max_abs = (expected - carried).abs().max().into_scalar::<f32>();
        assert!(max_abs <= 1.0e-6, "generic carry max_abs={max_abs:e}");
    }

    #[test]
    fn exact_kernel_shape_accounting_is_consistent() {
        assert_eq!(ELEMENTS, BATCH * SEQUENCE * WIDTH);
        assert_eq!(V4_BOUNDARIES, 2 * V4_LAYERS);
    }
}
