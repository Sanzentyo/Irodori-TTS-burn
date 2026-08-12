//! Burn-native inference implementation of the ModernBERT backbone embedded in
//! `Aratako/Irodori-TTS-v4-Small`.
//!
//! The released v4 checkpoint embeds a fine-tuned
//! `sbintuitions/modernbert-ja-310m` at revision
//! `77675fc96a7e445e982e2ba90246b816efc74ec6`.  It is shared by the text and
//! caption paths; each path then owns a separate residual-MLP projector.
//!
//! This module deliberately does not reuse Irodori's scratch text attention:
//! ModernBERT uses non-interleaved (rotate-half) RoPE, bias-free LayerNorm,
//! alternating full/sliding bidirectional attention, and a GELU-gated MLP.

use burn::tensor::Device;
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{
        Bool, Int, Tensor, TensorData, activation::gelu, module::attention as burn_attention,
        ops::AttentionModuleOptions,
    },
};
use serde::Deserialize;

use super::norm::RmsNorm;

mod wgsl;

/// Attention topology of one ModernBERT encoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModernBertLayerType {
    Full,
    Sliding,
}

/// Exact structural configuration embedded in the v4-Small safetensors metadata.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModernBertConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) norm_eps: f64,
    pub(crate) pad_token_id: usize,
    /// Total local-attention width from the HF config.  The inclusive
    /// bidirectional half-window is `local_attention / 2`.
    pub(crate) local_attention: usize,
    pub(crate) full_rope_theta: f64,
    pub(crate) sliding_rope_theta: f64,
    pub(crate) layer_types: Vec<ModernBertLayerType>,
}

#[derive(Debug, Deserialize)]
struct ModernBertRopeEntry {
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct ModernBertRopeParameters {
    full_attention: ModernBertRopeEntry,
    sliding_attention: ModernBertRopeEntry,
}

/// Operational fields consumed by this implementation from the embedded
/// Hugging Face configuration. Unknown fields are intentionally ignored; the
/// fields below are precisely those that affect the Burn execution graph.
#[derive(Debug, Deserialize)]
struct ModernBertMetadata {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    norm_eps: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    local_attention: usize,
    global_attn_every_n_layers: usize,
    layer_types: Vec<String>,
    rope_parameters: ModernBertRopeParameters,
    attention_bias: bool,
    attention_dropout: f64,
    embedding_dropout: f64,
    mlp_bias: bool,
    mlp_dropout: f64,
    norm_bias: bool,
    hidden_activation: String,
    position_embedding_type: String,
}

impl Default for ModernBertConfig {
    fn default() -> Self {
        Self::v4_small()
    }
}

impl ModernBertConfig {
    /// Configuration serialized as `text_encoder_config_json` in v4-Small.
    pub(crate) fn v4_small() -> Self {
        let layer_types = (0..25)
            .map(|layer| {
                if layer % 3 == 0 {
                    ModernBertLayerType::Full
                } else {
                    ModernBertLayerType::Sliding
                }
            })
            .collect();

        Self {
            vocab_size: 102_400,
            hidden_size: 768,
            intermediate_size: 3_072,
            num_hidden_layers: 25,
            num_attention_heads: 12,
            max_position_embeddings: 8_192,
            norm_eps: 1e-5,
            pad_token_id: 3,
            local_attention: 128,
            full_rope_theta: 160_000.0,
            sliding_rope_theta: 10_000.0,
            layer_types,
        }
    }

    /// Validate the checkpoint's separately embedded ModernBERT metadata.
    ///
    /// Record shape checks alone cannot detect graph-changing differences such
    /// as RoPE theta, sliding-window width, or the full-attention layer pattern.
    /// The current implementation intentionally targets the released v4-Small
    /// backbone, so reject a shape-compatible but semantically different
    /// checkpoint instead of silently running the wrong graph.
    pub(crate) fn validate_v4_metadata(json: &str) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let metadata: ModernBertMetadata = serde_json::from_str(json)?;
        let layer_types = metadata
            .layer_types
            .iter()
            .map(|layer_type| match layer_type.as_str() {
                "full_attention" => Ok(ModernBertLayerType::Full),
                "sliding_attention" => Ok(ModernBertLayerType::Sliding),
                other => Err(IrodoriError::Config(format!(
                    "unsupported ModernBERT layer type {other:?}"
                ))),
            })
            .collect::<crate::error::Result<Vec<_>>>()?;
        let actual = Self {
            vocab_size: metadata.vocab_size,
            hidden_size: metadata.hidden_size,
            intermediate_size: metadata.intermediate_size,
            num_hidden_layers: metadata.num_hidden_layers,
            num_attention_heads: metadata.num_attention_heads,
            max_position_embeddings: metadata.max_position_embeddings,
            norm_eps: metadata.norm_eps,
            pad_token_id: metadata.pad_token_id,
            local_attention: metadata.local_attention,
            full_rope_theta: metadata.rope_parameters.full_attention.rope_theta,
            sliding_rope_theta: metadata.rope_parameters.sliding_attention.rope_theta,
            layer_types,
        };
        let expected = Self::v4_small();
        if actual != expected {
            return Err(IrodoriError::Config(format!(
                "checkpoint ModernBERT graph does not match released v4-Small: expected {expected:?}, got {actual:?}"
            )));
        }

        let zero_dropout = metadata.attention_dropout == 0.0
            && metadata.embedding_dropout == 0.0
            && metadata.mlp_dropout == 0.0;
        if metadata.layer_norm_eps != metadata.norm_eps
            || metadata.global_attn_every_n_layers != 3
            || metadata.attention_bias
            || metadata.mlp_bias
            || metadata.norm_bias
            || !zero_dropout
            || metadata.hidden_activation != "gelu"
            || metadata.position_embedding_type != "rope"
        {
            return Err(IrodoriError::Config(format!(
                "checkpoint ModernBERT operator settings are unsupported: {metadata:?}"
            )));
        }
        Ok(())
    }

    fn validate(&self) {
        assert!(
            self.hidden_size > 0,
            "ModernBERT hidden_size must be positive"
        );
        assert!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "ModernBERT hidden_size must be divisible by num_attention_heads"
        );
        assert!(
            self.head_dim().is_multiple_of(2),
            "ModernBERT head_dim must be even for RoPE"
        );
        assert_eq!(
            self.layer_types.len(),
            self.num_hidden_layers,
            "ModernBERT layer_types length must match num_hidden_layers"
        );
        assert!(
            self.local_attention.is_multiple_of(2),
            "ModernBERT local_attention must be even"
        );
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub(crate) fn sliding_half_window(&self) -> usize {
        self.local_attention / 2
    }

    #[cfg(test)]
    fn tiny() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            max_position_embeddings: 32,
            norm_eps: 1e-5,
            pad_token_id: 3,
            local_attention: 4,
            full_rope_theta: 160_000.0,
            sliding_rope_theta: 10_000.0,
            layer_types: vec![ModernBertLayerType::Full, ModernBertLayerType::Sliding],
        }
    }
}

/// Token embedding followed by bias-free LayerNorm.
///
/// Weight mapping:
/// - `tok_embeddings.weight` -> Burn `Embedding::weight` (no transpose)
/// - `norm.weight` -> Burn `LayerNorm::gamma`; `beta` is absent
#[derive(Module, Debug)]
pub(crate) struct ModernBertEmbeddings {
    pub(crate) tok_embeddings: Embedding,
    pub(crate) norm: LayerNorm,
}

impl ModernBertEmbeddings {
    fn new(config: &ModernBertConfig, device: &Device) -> Self {
        Self {
            tok_embeddings: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device),
            norm: layer_norm(config.hidden_size, config.norm_eps, device),
        }
    }

    fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        self.norm.forward(self.tok_embeddings.forward(input_ids))
    }
}

/// GELU-gated ModernBERT feed-forward block.
///
/// HF computes `Wo(gelu(input) * gate)` after splitting `Wi(x)` in half.
#[derive(Module, Debug)]
pub(crate) struct ModernBertMlp {
    pub(crate) wi: Linear,
    pub(crate) wo: Linear,
    intermediate_size: usize,
}

impl ModernBertMlp {
    fn new(config: &ModernBertConfig, device: &Device) -> Self {
        Self {
            wi: LinearConfig::new(config.hidden_size, config.intermediate_size * 2)
                .with_bias(false)
                .init(device),
            wo: LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            intermediate_size: config.intermediate_size,
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let projected = self.wi.forward(hidden_states);
        let input = projected.clone().narrow(2, 0, self.intermediate_size);
        let gate = projected.narrow(2, self.intermediate_size, self.intermediate_size);
        self.wo.forward(gelu(input) * gate)
    }
}

/// One ModernBERT multi-head self-attention block.
#[derive(Module, Debug)]
pub(crate) struct ModernBertAttention {
    pub(crate) wqkv: Linear,
    pub(crate) wo: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl ModernBertAttention {
    fn new(config: &ModernBertConfig, device: &Device) -> Self {
        Self {
            wqkv: LinearConfig::new(config.hidden_size, config.hidden_size * 3)
                .with_bias(false)
                .init(device),
            wo: LinearConfig::new(config.hidden_size, config.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<3>,
        position_embeddings: &ModernBertRope,
        backend_mask: Tensor<4, Bool>,
    ) -> Tensor<3> {
        let [batch, seq, hidden_size] = hidden_states.dims();
        let qkv = self.wqkv.forward(hidden_states);

        let q = qkv.clone().narrow(2, 0, hidden_size).reshape([
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ]);
        let k = qkv.clone().narrow(2, hidden_size, hidden_size).reshape([
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ]);
        let v = qkv.narrow(2, hidden_size * 2, hidden_size).reshape([
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ]);

        let q = apply_modern_bert_rope(q, position_embeddings);
        let k = apply_modern_bert_rope(k, position_embeddings);
        let output = native_attention(q, k, v, backend_mask).reshape([
            batch,
            seq,
            self.num_heads * self.head_dim,
        ]);
        self.wo.forward(output)
    }
}

/// One pre-norm ModernBERT encoder layer.
#[derive(Module, Debug)]
pub(crate) struct ModernBertEncoderLayer {
    /// Layer zero uses identity here and has no `attn_norm.weight` in the checkpoint.
    pub(crate) attn_norm: Option<LayerNorm>,
    pub(crate) attn: ModernBertAttention,
    pub(crate) mlp_norm: LayerNorm,
    pub(crate) mlp: ModernBertMlp,
    #[module(skip)]
    layer_type: ModernBertLayerType,
}

impl ModernBertEncoderLayer {
    fn new(
        config: &ModernBertConfig,
        layer_index: usize,
        layer_type: ModernBertLayerType,
        device: &Device,
    ) -> Self {
        Self {
            attn_norm: (layer_index != 0)
                .then(|| layer_norm(config.hidden_size, config.norm_eps, device)),
            attn: ModernBertAttention::new(config, device),
            mlp_norm: layer_norm(config.hidden_size, config.norm_eps, device),
            mlp: ModernBertMlp::new(config, device),
            layer_type,
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<3>,
        position_embeddings: &ModernBertRope,
        backend_mask: Tensor<4, Bool>,
    ) -> Tensor<3> {
        let attention_input = match &self.attn_norm {
            Some(norm) => norm.forward(hidden_states.clone()),
            None => hidden_states.clone(),
        };
        let hidden_states = hidden_states
            + self
                .attn
                .forward(attention_input, position_embeddings, backend_mask);
        let mlp_input = self.mlp_norm.forward(hidden_states.clone());
        hidden_states + self.mlp.forward(mlp_input)
    }
}

/// The 25-layer ModernBERT encoder embedded in v4-Small.
#[derive(Module, Debug)]
pub(crate) struct ModernBertModel {
    pub(crate) embeddings: ModernBertEmbeddings,
    pub(crate) layers: Vec<ModernBertEncoderLayer>,
    pub(crate) final_norm: LayerNorm,
    head_dim: usize,
    max_position_embeddings: usize,
    sliding_half_window: usize,
    full_rope_theta: f64,
    sliding_rope_theta: f64,
    norm_eps: f64,
}

impl ModernBertModel {
    pub(crate) fn new(config: &ModernBertConfig, device: &Device) -> Self {
        config.validate();
        let layers = config
            .layer_types
            .iter()
            .copied()
            .enumerate()
            .map(|(index, layer_type)| {
                ModernBertEncoderLayer::new(config, index, layer_type, device)
            })
            .collect();

        Self {
            embeddings: ModernBertEmbeddings::new(config, device),
            layers,
            final_norm: layer_norm(config.hidden_size, config.norm_eps, device),
            head_dim: config.head_dim(),
            max_position_embeddings: config.max_position_embeddings,
            sliding_half_window: config.sliding_half_window(),
            full_rope_theta: config.full_rope_theta,
            sliding_rope_theta: config.sliding_rope_theta,
            norm_eps: config.norm_eps,
        }
    }

    /// Encode right-padded token IDs. `attention_mask` uses `true = valid token`.
    ///
    /// As in HF ModernBERT, padding masks keys but not query rows inside the
    /// transformer.  The final result is hard-masked to zero, matching
    /// `PretrainedTextBackbone.forward` in official Irodori v4.
    pub(crate) fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let [batch, seq] = input_ids.dims();
        assert!(
            seq <= self.max_position_embeddings,
            "ModernBERT input length {seq} exceeds max_position_embeddings {}",
            self.max_position_embeddings
        );
        assert_eq!(
            attention_mask.dims(),
            [batch, seq],
            "ModernBERT attention mask must match input IDs"
        );

        let device = input_ids.device();
        let mut hidden_states = self.embeddings.forward(input_ids);
        let full_rope =
            precompute_modern_bert_rope(self.head_dim, seq, self.full_rope_theta, &device);
        let sliding_rope =
            precompute_modern_bert_rope(self.head_dim, seq, self.sliding_rope_theta, &device);
        let full_mask = full_attention_valid_mask(attention_mask.clone());
        let sliding_mask =
            sliding_attention_valid_mask(attention_mask.clone(), self.sliding_half_window, &device);

        for layer in &self.layers {
            let (rope, valid_mask) = match layer.layer_type {
                ModernBertLayerType::Full => (&full_rope, full_mask.clone()),
                ModernBertLayerType::Sliding => (&sliding_rope, sliding_mask.clone()),
            };
            // Burn uses `true = masked out`. Its tuned 0.22 WGPU attention
            // currently mutates/reuses mask workspace internally, so each
            // layer must own a freshly materialized query/key mask.
            let backend_mask = valid_mask.expand([batch, 1, seq, seq]).bool_not();
            hidden_states = layer.forward(hidden_states, rope, backend_mask);
        }

        let hidden_states = self.final_norm.forward(hidden_states);
        hidden_states * bool_mask_to_float(attention_mask, &device)
    }
}

/// Matches the Python wrapper and therefore the checkpoint nesting
/// `pretrained_text_backbone.backbone.*`.
#[derive(Module, Debug)]
pub(crate) struct PretrainedTextBackbone {
    pub(crate) backbone: ModernBertModel,
}

impl PretrainedTextBackbone {
    pub(crate) fn new(config: &ModernBertConfig, device: &Device) -> Self {
        Self {
            backbone: ModernBertModel::new(config, device),
        }
    }

    pub(crate) fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        self.backbone.forward(input_ids, attention_mask)
    }
}

/// Separate text/caption projection from the shared 768-wide backbone to the
/// 512-wide Irodori condition space.
#[derive(Module, Debug)]
pub(crate) struct PretrainedConditionProjector {
    pub(crate) projector: Linear,
    pub(crate) residual_norm: Option<RmsNorm>,
    pub(crate) residual_up: Option<Linear>,
    pub(crate) residual_down: Option<Linear>,
    pub(crate) residual_dropout: Option<Dropout>,
}

impl PretrainedConditionProjector {
    pub(crate) fn new(
        backbone_dim: usize,
        output_dim: usize,
        hidden_ratio: f64,
        dropout: f64,
        norm_eps: f64,
        device: &Device,
    ) -> Self {
        assert!(
            hidden_ratio > 0.0,
            "projector hidden_ratio must be positive"
        );
        let hidden_dim = ((output_dim as f64 * hidden_ratio).round() as usize).max(1);

        Self {
            projector: LinearConfig::new(backbone_dim, output_dim).init(device),
            residual_norm: Some(RmsNorm::new(backbone_dim, norm_eps, device)),
            residual_up: Some(LinearConfig::new(backbone_dim, hidden_dim).init(device)),
            residual_down: Some(LinearConfig::new(hidden_dim, output_dim).init(device)),
            residual_dropout: Some(DropoutConfig::new(dropout).init()),
        }
    }

    pub(crate) fn forward(&self, backbone_state: Tensor<3>, mask: Tensor<2, Bool>) -> Tensor<3> {
        let mut projected = self.projector.forward(backbone_state.clone());
        let norm = self
            .residual_norm
            .as_ref()
            .expect("residual projector must contain residual_norm");
        let up = self
            .residual_up
            .as_ref()
            .expect("residual projector must contain residual_up");
        let down = self
            .residual_down
            .as_ref()
            .expect("residual projector must contain residual_down");
        let dropout = self
            .residual_dropout
            .as_ref()
            .expect("residual projector must contain residual_dropout");

        let residual = burn::tensor::activation::silu(up.forward(norm.forward(backbone_state)));
        projected = projected + down.forward(dropout.forward(residual));

        let device = mask.device();
        projected * bool_mask_to_float(mask, &device)
    }
}

/// Convenience composition demonstrating that text and caption share one
/// backbone while retaining independent projector parameters.
#[derive(Module, Debug)]
pub struct SharedModernBertConditioner {
    pub(crate) pretrained_text_backbone: PretrainedTextBackbone,
    pub(crate) text_encoder: PretrainedConditionProjector,
    pub(crate) caption_encoder: PretrainedConditionProjector,
}

impl SharedModernBertConditioner {
    pub fn v4_small(device: &Device) -> Self {
        let config = ModernBertConfig::v4_small();
        Self::new(&config, 512, 2.0, device)
    }

    pub(crate) fn new(
        config: &ModernBertConfig,
        condition_dim: usize,
        projector_hidden_ratio: f64,
        device: &Device,
    ) -> Self {
        Self {
            pretrained_text_backbone: PretrainedTextBackbone::new(config, device),
            text_encoder: PretrainedConditionProjector::new(
                config.hidden_size,
                condition_dim,
                projector_hidden_ratio,
                0.0,
                config.norm_eps,
                device,
            ),
            caption_encoder: PretrainedConditionProjector::new(
                config.hidden_size,
                condition_dim,
                projector_hidden_ratio,
                0.0,
                config.norm_eps,
                device,
            ),
        }
    }

    pub(crate) fn encode_text(
        &self,
        input_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let state = self
            .pretrained_text_backbone
            .forward(input_ids, mask.clone());
        self.text_encoder.forward(state, mask)
    }

    pub(crate) fn encode_caption(
        &self,
        input_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let state = self
            .pretrained_text_backbone
            .forward(input_ids, mask.clone());
        self.caption_encoder.forward(state, mask)
    }

    /// Return the shared backbone state and both independently projected
    /// condition states in one pass. This is also the canonical parity surface
    /// for checking a converted v4 checkpoint against the PyTorch release.
    pub fn forward_all(
        &self,
        input_ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let backbone = self
            .pretrained_text_backbone
            .forward(input_ids, mask.clone());
        let text = self.text_encoder.forward(backbone.clone(), mask.clone());
        let caption = self.caption_encoder.forward(backbone.clone(), mask);
        (backbone, text, caption)
    }
}

#[derive(Clone, Debug)]
struct ModernBertRope {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

/// ModernBERT duplicates each frequency vector as `[freqs, freqs]`; this is
/// intentionally different from Irodori's adjacent-pair complex RoPE.
fn precompute_modern_bert_rope(
    head_dim: usize,
    seq_len: usize,
    theta: f64,
    device: &Device,
) -> ModernBertRope {
    assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    let inv_freq = (0..half)
        .map(|index| 1.0 / (theta as f32).powf((2 * index) as f32 / head_dim as f32))
        .collect::<Vec<_>>();

    let mut cos = vec![0.0_f32; seq_len * head_dim];
    let mut sin = vec![0.0_f32; seq_len * head_dim];
    for position in 0..seq_len {
        for (index, frequency) in inv_freq.iter().copied().enumerate() {
            let angle = position as f32 * frequency;
            let cos_value = angle.cos();
            let sin_value = angle.sin();
            cos[position * head_dim + index] = cos_value;
            cos[position * head_dim + half + index] = cos_value;
            sin[position * head_dim + index] = sin_value;
            sin[position * head_dim + half + index] = sin_value;
        }
    }

    ModernBertRope {
        cos: Tensor::from_floats(TensorData::new(cos, [seq_len, head_dim]), device),
        sin: Tensor::from_floats(TensorData::new(sin, [seq_len, head_dim]), device),
    }
}

fn apply_modern_bert_rope(tensor: Tensor<4>, rope: &ModernBertRope) -> Tensor<4> {
    let [batch, seq, heads, head_dim] = tensor.dims();
    let half = head_dim / 2;
    let first = tensor.clone().slice([0..batch, 0..seq, 0..heads, 0..half]);
    let second = tensor
        .clone()
        .slice([0..batch, 0..seq, 0..heads, half..head_dim]);
    let rotated = Tensor::cat(vec![-second, first], 3);
    let cos = rope.cos.clone().reshape([1, seq, 1, head_dim]);
    let sin = rope.sin.clone().reshape([1, seq, 1, head_dim]);
    // HF performs this arithmetic in f32 before converting back to Q/K dtype.
    // Burn's backend fixes the tensor element type; the f32 NdArray/WGPU path
    // therefore matches exactly, while reduced-precision backends round here.
    tensor * cos + rotated * sin
}

fn full_attention_valid_mask(key_mask: Tensor<2, Bool>) -> Tensor<4, Bool> {
    key_mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2)
}

fn sliding_attention_valid_mask(
    key_mask: Tensor<2, Bool>,
    half_window: usize,
    device: &Device,
) -> Tensor<4, Bool> {
    let [batch, seq] = key_mask.dims();
    let local = (0..seq)
        .flat_map(|query| (0..seq).map(move |key| query.abs_diff(key) <= half_window))
        .collect::<Vec<_>>();
    let local = Tensor::<2, Bool>::from_data(TensorData::new(local, [seq, seq]), device)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
        .expand([batch, 1, seq, seq]);
    let keys = full_attention_valid_mask(key_mask).expand([batch, 1, seq, seq]);
    local.bool_and(keys)
}

fn native_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    backend_mask: Tensor<4, Bool>,
) -> Tensor<4> {
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };
    // Query/key axes were materialized once by the model forward because Burn
    // 0.22.0-pre.2's tuned WGPU attention is incorrect with that broadcast
    // stride. The head axis remains the supported ordinary broadcast.
    burn_attention(q, k, v, Some(backend_mask), None, options).swap_dims(1, 2)
}

fn layer_norm(dim: usize, eps: f64, device: &Device) -> LayerNorm {
    LayerNormConfig::new(dim)
        .with_epsilon(eps)
        .with_bias(false)
        .init(device)
}

fn bool_mask_to_float(mask: Tensor<2, Bool>, device: &Device) -> Tensor<3> {
    let [batch, seq] = mask.dims();
    let ones = Tensor::<2>::ones([batch, seq], device);
    let zeros = Tensor::<2>::zeros([batch, seq], device);
    ones.mask_where(mask.bool_not(), zeros)
        .unsqueeze_dim::<3>(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Default::default()
    }

    #[test]
    fn v4_config_matches_embedded_metadata() {
        let config = ModernBertConfig::v4_small();
        assert_eq!(config.vocab_size, 102_400);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.intermediate_size, 3_072);
        assert_eq!(config.num_hidden_layers, 25);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.sliding_half_window(), 64);
        assert_eq!(config.layer_types[0], ModernBertLayerType::Full);
        assert_eq!(config.layer_types[1], ModernBertLayerType::Sliding);
        assert_eq!(config.layer_types[24], ModernBertLayerType::Full);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|kind| **kind == ModernBertLayerType::Full)
                .count(),
            9
        );
    }

    fn v4_metadata_json() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 102400,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 25,
            "num_attention_heads": 12,
            "max_position_embeddings": 8192,
            "norm_eps": 1e-5,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 3,
            "local_attention": 128,
            "global_attn_every_n_layers": 3,
            "layer_types": (0..25)
                .map(|layer| if layer % 3 == 0 { "full_attention" } else { "sliding_attention" })
                .collect::<Vec<_>>(),
            "rope_parameters": {
                "full_attention": { "rope_theta": 160000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            },
            "attention_bias": false,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "mlp_bias": false,
            "mlp_dropout": 0.0,
            "norm_bias": false,
            "hidden_activation": "gelu",
            "position_embedding_type": "rope"
        })
    }

    #[test]
    fn released_v4_metadata_is_accepted() {
        ModernBertConfig::validate_v4_metadata(&v4_metadata_json().to_string()).unwrap();
    }

    #[test]
    fn graph_changing_v4_metadata_is_rejected() {
        let mut metadata = v4_metadata_json();
        metadata["rope_parameters"]["sliding_attention"]["rope_theta"] = serde_json::json!(12345.0);
        let error = ModernBertConfig::validate_v4_metadata(&metadata.to_string()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match released v4-Small")
        );
    }

    #[test]
    fn unsupported_v4_operator_settings_are_rejected() {
        let mut metadata = v4_metadata_json();
        metadata["attention_bias"] = serde_json::json!(true);
        let error = ModernBertConfig::validate_v4_metadata(&metadata.to_string()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("operator settings are unsupported")
        );
    }

    #[test]
    fn rotate_half_rope_matches_known_values() {
        let device = device();
        let rope = precompute_modern_bert_rope(4, 2, 100.0, &device);
        let input =
            Tensor::<4>::from_floats([[[[0.0, 0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0, 4.0]]]], &device);
        let output = apply_modern_bert_rope(input, &rope);
        let values = output
            .slice([0..1, 1..2, 0..1, 0..4])
            .reshape([4])
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = [
            1.0_f32 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
            2.0_f32 * 0.1_f32.cos() - 4.0 * 0.1_f32.sin(),
            3.0_f32 * 1.0_f32.cos() + 1.0 * 1.0_f32.sin(),
            4.0_f32 * 0.1_f32.cos() + 2.0 * 0.1_f32.sin(),
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn sliding_mask_is_inclusive_and_respects_key_padding() {
        let device = device();
        let keys = Tensor::<2, Bool>::from_data([[true, true, false, true, true]], &device);
        let mask = sliding_attention_valid_mask(keys, 1, &device);
        assert_eq!(mask.dims(), [1, 1, 5, 5]);
        let values = mask.to_data().convert::<bool>().to_vec::<bool>().unwrap();
        assert_eq!(&values[0..5], &[true, true, false, false, false]);
        assert_eq!(&values[10..15], &[false, true, false, true, false]);
        assert_eq!(&values[20..25], &[false, false, false, true, true]);
    }

    #[test]
    fn tiny_backbone_preserves_shape_and_zeroes_masked_rows() {
        let device = device();
        let config = ModernBertConfig::tiny();
        let model = ModernBertModel::new(&config, &device);
        let ids = Tensor::<2, Int>::zeros([1, 4], &device);
        let mask = Tensor::<2, Bool>::from_data([[true, true, false, false]], &device);
        let output = model.forward(ids, mask);
        assert_eq!(output.dims(), [1, 4, 8]);
        let masked = output.slice([0..1, 2..4, 0..8]);
        let sum = masked.abs().sum().into_scalar::<f32>();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn all_false_mask_is_finite_and_exactly_zero() {
        let device = device();
        let config = ModernBertConfig::tiny();
        let model = ModernBertModel::new(&config, &device);
        let ids = Tensor::<2, Int>::zeros([1, 3], &device);
        let mask = Tensor::<2, Bool>::from_data([[false, false, false]], &device);
        let values = model.forward(ids, mask).to_data().to_vec::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn residual_projector_uses_rms_not_mean_centering() {
        let device = device();
        let norm = RmsNorm::new(2, 0.0, &device);
        let input = Tensor::<3>::from_floats([[[3.0, 4.0]]], &device);
        let values = norm.forward(input).to_data().to_vec::<f32>().unwrap();
        let rms = 12.5_f32.sqrt();

        assert!((values[0] - 3.0 / rms).abs() < 1e-6);
        assert!((values[1] - 4.0 / rms).abs() < 1e-6);
        assert!((values[0] + values[1]).abs() > 1e-3);
    }

    #[test]
    fn residual_projector_masks_bias_outputs() {
        let device = device();
        let projector = PretrainedConditionProjector::new(8, 4, 2.0, 0.0, 1e-5, &device);
        let state = Tensor::<3>::ones([1, 3, 8], &device);
        let mask = Tensor::<2, Bool>::from_data([[true, false, true]], &device);
        let output = projector.forward(state, mask);
        assert_eq!(output.dims(), [1, 3, 4]);
        let masked = output.slice([0..1, 1..2, 0..4]);
        assert_eq!(masked.abs().sum().into_scalar::<f32>(), 0.0);
    }

    #[test]
    fn shared_conditioner_uses_one_backbone_and_two_projectors() {
        let device = device();
        let config = ModernBertConfig::tiny();
        let conditioner = SharedModernBertConditioner::new(&config, 4, 2.0, &device);
        let ids = Tensor::<2, Int>::zeros([1, 3], &device);
        let mask = Tensor::<2, Bool>::ones([1, 3], &device);
        assert_eq!(
            conditioner.encode_text(ids.clone(), mask.clone()).dims(),
            [1, 3, 4]
        );
        assert_eq!(conditioner.encode_caption(ids, mask).dims(), [1, 3, 4]);
    }
}
