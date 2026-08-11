//! Weight loading for DACVAE from a pre-converted safetensors file.
//!
//! The weights file is produced by `scripts/convert_dacvae_weights.py`, which
//! resolves all `weight_norm` parametrisations into plain tensors.

use std::path::Path;

use burn::{
    module::{Param, ParamId},
    nn::conv::{Conv1d, ConvTranspose1d, ConvTranspose1dConfig},
    prelude::*,
};
use serde::Deserialize;

use crate::{error::IrodoriError, weights::TensorStore};

use super::{
    bottleneck::VaeBottleneck,
    decoder::{Decoder, DecoderBlock, WmHead},
    encoder::{Encoder, EncoderBlock},
    layers::{ResidualUnit, Snake1d, conv_transpose_pad, make_conv1d},
    model::{DACVAE_HOP_LENGTH, DACVAE_SAMPLE_RATE, DacVaeCodec},
};

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Build a `DacVaeCodec<B>` from a safetensors weights file.
pub fn load_codec<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<DacVaeCodec<B>, IrodoriError> {
    let store = TensorStore::load(path)?;
    build_codec(&store, device)
}

fn build_codec<B: Backend>(
    store: &TensorStore,
    device: &B::Device,
) -> Result<DacVaeCodec<B>, IrodoriError> {
    validate_codec_metadata(&store.config_json)?;
    Ok(DacVaeCodec {
        encoder: build_encoder(store, device)?,
        bottleneck: build_bottleneck(store, device)?,
        decoder: build_decoder(store, device)?,
        hop_length: DACVAE_HOP_LENGTH,
        sample_rate: DACVAE_SAMPLE_RATE,
    })
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CodecMetadata {
    encoder_dim: usize,
    encoder_rates: [usize; 4],
    latent_dim: usize,
    decoder_dim: usize,
    decoder_rates: [usize; 4],
    n_codebooks: usize,
    codebook_size: usize,
    codebook_dim: usize,
    quantizer_dropout: bool,
    sample_rate: usize,
}

const EXPECTED_CODEC_METADATA: CodecMetadata = CodecMetadata {
    encoder_dim: 64,
    encoder_rates: [2, 8, 10, 12],
    latent_dim: 1024,
    decoder_dim: 1536,
    decoder_rates: [12, 10, 8, 2],
    n_codebooks: 16,
    codebook_size: 1024,
    codebook_dim: 32,
    quantizer_dropout: false,
    sample_rate: 48_000,
};

fn validate_codec_metadata(config_json: &str) -> Result<(), IrodoriError> {
    let actual: CodecMetadata = serde_json::from_str(config_json)?;
    if actual != EXPECTED_CODEC_METADATA {
        return Err(IrodoriError::Config(format!(
            "codec metadata does not match the supported Semantic-DACVAE topology: expected {EXPECTED_CODEC_METADATA:?}, got {actual:?}"
        )));
    }
    Ok(())
}

// ─── Primitive helpers ───────────────────────────────────────────────────────

fn snake1d<B: Backend>(
    store: &TensorStore,
    key: &str,
    channels: usize,
    device: &B::Device,
) -> Result<Snake1d<B>, IrodoriError> {
    let alpha_key = format!("{key}.alpha");
    let alpha: Tensor<B, 3> = store.tensor(&alpha_key, device)?;
    ensure_tensor_shape(&alpha_key, alpha.dims(), [1, channels, 1])?;
    Ok(Snake1d::new(alpha))
}

/// Load an optional tensor without discarding errors for a present key.
///
/// Codec biases are legitimately absent in some converted checkpoints. A key
/// that exists, however, must still satisfy its requested rank and dtype.
fn optional_tensor<B: Backend, const D: usize>(
    store: &TensorStore,
    key: &str,
    device: &B::Device,
) -> Result<Option<Tensor<B, D>>, IrodoriError> {
    store
        .has(key)
        .then(|| store.tensor(key, device))
        .transpose()
}

fn ensure_tensor_shape<const D: usize>(
    key: &str,
    actual: [usize; D],
    expected: [usize; D],
) -> Result<(), IrodoriError> {
    if actual != expected {
        return Err(IrodoriError::Shape(format!(
            "tensor '{key}' must have shape {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv1d<B: Backend>(
    store: &TensorStore,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    device: &B::Device,
) -> Result<Conv1d<B>, IrodoriError> {
    let weight_key = format!("{prefix}.weight");
    let bias_key = format!("{prefix}.bias");
    let weight: Tensor<B, 3> = store.tensor(&weight_key, device)?;
    ensure_tensor_shape(&weight_key, weight.dims(), [out_ch, in_ch, kernel])?;
    let bias = optional_tensor(store, &bias_key, device)?;
    if let Some(bias) = &bias {
        ensure_tensor_shape(&bias_key, bias.dims(), [out_ch])?;
    }
    Ok(make_conv1d(
        in_ch, out_ch, kernel, stride, dilation, weight, bias, device,
    ))
}

fn conv_transpose1d<B: Backend>(
    store: &TensorStore,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    device: &B::Device,
) -> Result<ConvTranspose1d<B>, IrodoriError> {
    let kernel = 2 * stride;
    let (padding, padding_out) = conv_transpose_pad(stride);
    let weight_key = format!("{prefix}.weight");
    let bias_key = format!("{prefix}.bias");
    let weight: Tensor<B, 3> = store.tensor(&weight_key, device)?;
    ensure_tensor_shape(&weight_key, weight.dims(), [in_ch, out_ch, kernel])?;
    let bias = optional_tensor(store, &bias_key, device)?;
    if let Some(bias) = &bias {
        ensure_tensor_shape(&bias_key, bias.dims(), [out_ch])?;
    }

    let mut conv = ConvTranspose1dConfig::new([in_ch, out_ch], kernel)
        .with_stride(stride)
        .with_padding(padding)
        .with_padding_out(padding_out)
        .with_bias(bias.is_some())
        .init::<B>(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = bias.map(|b| Param::initialized(ParamId::new(), b));
    Ok(conv)
}

/// ResidualUnit with kernel=7 for the dilated conv, kernel=1 for the 1×1 conv.
fn residual_unit<B: Backend>(
    store: &TensorStore,
    prefix: &str,
    dim: usize,
    dilation: usize,
    device: &B::Device,
) -> Result<ResidualUnit<B>, IrodoriError> {
    // Python block layout: [Snake0, Conv(dil), Snake1, Conv(1x1)]
    let act0 = snake1d(store, &format!("{prefix}.block.0"), dim, device)?;
    let conv_dil = conv1d(
        store,
        &format!("{prefix}.block.1"),
        dim,
        dim,
        7,
        1,
        dilation,
        device,
    )?;
    let act1 = snake1d(store, &format!("{prefix}.block.2"), dim, device)?;
    let conv_1x1 = conv1d(
        store,
        &format!("{prefix}.block.3"),
        dim,
        dim,
        1,
        1,
        1,
        device,
    )?;
    Ok(ResidualUnit {
        act0,
        conv_dil,
        act1,
        conv_1x1,
        packed_conv_1x1_weight: None,
        packed_conv_dil_weight_vectors: None,
    })
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

fn encoder_block<B: Backend>(
    store: &TensorStore,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
    device: &B::Device,
) -> Result<EncoderBlock<B>, IrodoriError> {
    // Python: block.[0..2]=ResUnit×3, block.3=Snake, block.4=striding conv
    let res0 = residual_unit(store, &format!("{prefix}.block.0"), in_dim, 1, device)?;
    let res1 = residual_unit(store, &format!("{prefix}.block.1"), in_dim, 3, device)?;
    let res2 = residual_unit(store, &format!("{prefix}.block.2"), in_dim, 9, device)?;
    let tail_act = snake1d(store, &format!("{prefix}.block.3"), in_dim, device)?;
    let tail_conv = conv1d(
        store,
        &format!("{prefix}.block.4"),
        in_dim,
        out_dim,
        2 * stride,
        stride,
        1,
        device,
    )?;
    Ok(EncoderBlock {
        res0,
        res1,
        res2,
        tail_act,
        tail_conv,
    })
}

fn build_encoder<B: Backend>(
    store: &TensorStore,
    device: &B::Device,
) -> Result<Encoder<B>, IrodoriError> {
    // encoder_dim=64, rates=[2,8,10,12], latent_dim=1024
    let stem = conv1d(store, "encoder.block.0", 1, 64, 7, 1, 1, device)?;
    let block0 = encoder_block(store, "encoder.block.1", 64, 128, 2, device)?;
    let block1 = encoder_block(store, "encoder.block.2", 128, 256, 8, device)?;
    let block2 = encoder_block(store, "encoder.block.3", 256, 512, 10, device)?;
    let block3 = encoder_block(store, "encoder.block.4", 512, 1024, 12, device)?;
    let tail_act = snake1d(store, "encoder.block.5", 1024, device)?;
    let tail_conv = conv1d(store, "encoder.block.6", 1024, 1024, 3, 1, 1, device)?;
    Ok(Encoder {
        stem,
        block0,
        block1,
        block2,
        block3,
        tail_act,
        tail_conv,
    })
}

// ─── Bottleneck ──────────────────────────────────────────────────────────────

fn build_bottleneck<B: Backend>(
    store: &TensorStore,
    device: &B::Device,
) -> Result<VaeBottleneck<B>, IrodoriError> {
    // in_proj: Conv1d(1024→64, k=1)  out=64 → split → mean[0:32]
    // out_proj: Conv1d(32→1024, k=1)
    let in_proj = conv1d(store, "quantizer.in_proj", 1024, 64, 1, 1, 1, device)?;
    let out_proj = conv1d(store, "quantizer.out_proj", 32, 1024, 1, 1, 1, device)?;
    Ok(VaeBottleneck {
        in_proj,
        out_proj,
        codebook_dim: 32,
    })
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

fn decoder_block<B: Backend>(
    store: &TensorStore,
    prefix: &str,
    input_dim: usize,
    output_dim: usize,
    stride: usize,
    device: &B::Device,
) -> Result<DecoderBlock<B>, IrodoriError> {
    // Python block layout (main path only):
    //   block.0 = Snake(input_dim)
    //   block.1 = ConvTranspose1d(input_dim, output_dim)
    //   block.4 = ResUnit(output_dim, dil=1)
    //   block.5 = ResUnit(output_dim, dil=3)
    //   block.8 = ResUnit(output_dim, dil=9)
    let act = snake1d(store, &format!("{prefix}.block.0"), input_dim, device)?;
    let conv_t = conv_transpose1d(
        store,
        &format!("{prefix}.block.1"),
        input_dim,
        output_dim,
        stride,
        device,
    )?;
    let res0 = residual_unit(store, &format!("{prefix}.block.4"), output_dim, 1, device)?;
    let res1 = residual_unit(store, &format!("{prefix}.block.5"), output_dim, 3, device)?;
    let res2 = residual_unit(store, &format!("{prefix}.block.8"), output_dim, 9, device)?;
    Ok(DecoderBlock {
        act,
        conv_t,
        packed_conv_t_weight: None,
        res0,
        res1,
        res2,
    })
}

fn build_wm_head<B: Backend>(
    store: &TensorStore,
    device: &B::Device,
) -> Result<WmHead<B>, IrodoriError> {
    // forward_no_conv path: Snake(96) + Conv(96→1, k=7) + Tanh
    let act = snake1d(store, "decoder.wm_model.encoder_block.pre.0", 96, device)?;
    let conv = conv1d(
        store,
        "decoder.wm_model.encoder_block.pre.1",
        96,
        1,
        7,
        1,
        1,
        device,
    )?;
    Ok(WmHead { act, conv })
}

fn build_decoder<B: Backend>(
    store: &TensorStore,
    device: &B::Device,
) -> Result<Decoder<B>, IrodoriError> {
    // decoder_dim=1536, rates=[12,10,8,2]
    let stem = conv1d(store, "decoder.model.0", 1024, 1536, 7, 1, 1, device)?;
    let block0 = decoder_block(store, "decoder.model.1", 1536, 768, 12, device)?;
    let block1 = decoder_block(store, "decoder.model.2", 768, 384, 10, device)?;
    let block2 = decoder_block(store, "decoder.model.3", 384, 192, 8, device)?;
    let block3 = decoder_block(store, "decoder.model.4", 192, 96, 2, device)?;
    let wm_head = build_wm_head(store, device)?;
    Ok(Decoder {
        stem,
        block0,
        block1,
        block2,
        block3,
        wm_head,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use burn::backend::NdArray;
    use safetensors::{Dtype, tensor::TensorView};

    use super::*;

    type B = NdArray<f32>;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn tensor_store(tensors: &[(&str, Vec<u8>, Dtype, Vec<usize>)]) -> TensorStore {
        let views = tensors
            .iter()
            .map(|(name, bytes, dtype, shape)| {
                (
                    *name,
                    TensorView::new(*dtype, shape.clone(), bytes)
                        .expect("test tensor metadata must be valid"),
                )
            })
            .collect::<Vec<_>>();
        let metadata = HashMap::from([("config_json".to_string(), "{}".to_string())]);
        let encoded = safetensors::tensor::serialize(views, Some(metadata))
            .expect("test checkpoint serialization must succeed");
        let file = tempfile::NamedTempFile::new().expect("test checkpoint file must be created");
        std::fs::write(file.path(), encoded).expect("test checkpoint must be written");
        TensorStore::load(file.path()).expect("test checkpoint must load")
    }

    fn conv_weight() -> Vec<f32> {
        vec![0.0; 3 * 2]
    }

    fn official_codec_config_json() -> String {
        serde_json::to_string(&serde_json::json!({
            "encoder_dim": 64,
            "encoder_rates": [2, 8, 10, 12],
            "latent_dim": 1024,
            "decoder_dim": 1536,
            "decoder_rates": [12, 10, 8, 2],
            "n_codebooks": 16,
            "codebook_size": 1024,
            "codebook_dim": 32,
            "quantizer_dropout": false,
            "sample_rate": 48_000,
        }))
        .expect("test metadata must serialize")
    }

    #[test]
    fn official_codec_metadata_is_accepted() {
        validate_codec_metadata(&official_codec_config_json())
            .expect("the released codec topology must be accepted");
    }

    #[test]
    fn graph_changing_codec_metadata_is_rejected() {
        let mut metadata: serde_json::Value =
            serde_json::from_str(&official_codec_config_json()).expect("test metadata must parse");
        metadata["sample_rate"] = 44_100.into();
        assert!(matches!(
            validate_codec_metadata(&metadata.to_string()),
            Err(IrodoriError::Config(_))
        ));

        metadata["sample_rate"] = 48_000.into();
        metadata["encoder_rates"] = serde_json::json!([2, 8, 10, 10]);
        assert!(matches!(
            validate_codec_metadata(&metadata.to_string()),
            Err(IrodoriError::Config(_))
        ));
    }

    #[test]
    fn rank_correct_wrong_snake_shape_is_rejected() {
        let store = tensor_store(&[("act.alpha", f32_bytes(&[0.0; 4]), Dtype::F32, vec![1, 4, 1])]);
        assert!(matches!(
            snake1d::<B>(&store, "act", 3, &Default::default()),
            Err(IrodoriError::Shape(_))
        ));
    }

    #[test]
    fn missing_conv_bias_remains_optional() {
        let store = tensor_store(&[(
            "layer.weight",
            f32_bytes(&conv_weight()),
            Dtype::F32,
            vec![3, 2, 1],
        )]);

        let conv = conv1d::<B>(&store, "layer", 2, 3, 1, 1, 1, &Default::default())
            .expect("a missing optional bias must be accepted");
        assert!(conv.bias.is_none());
    }

    #[test]
    fn present_conv_bias_is_loaded() {
        let store = tensor_store(&[
            (
                "layer.weight",
                f32_bytes(&conv_weight()),
                Dtype::F32,
                vec![3, 2, 1],
            ),
            (
                "layer.bias",
                f32_bytes(&[1.0, 2.0, 3.0]),
                Dtype::F32,
                vec![3],
            ),
        ]);

        let conv = conv1d::<B>(&store, "layer", 2, 3, 1, 1, 1, &Default::default())
            .expect("a valid present bias must load");
        assert!(conv.bias.is_some());
    }

    #[test]
    fn rank_correct_wrong_conv_shapes_are_rejected() {
        let wrong_weight = tensor_store(&[(
            "layer.weight",
            f32_bytes(&[0.0; 3 * 3]),
            Dtype::F32,
            vec![3, 3, 1],
        )]);
        assert!(matches!(
            conv1d::<B>(&wrong_weight, "layer", 2, 3, 1, 1, 1, &Default::default()),
            Err(IrodoriError::Shape(_))
        ));

        let wrong_bias = tensor_store(&[
            (
                "layer.weight",
                f32_bytes(&conv_weight()),
                Dtype::F32,
                vec![3, 2, 1],
            ),
            ("layer.bias", f32_bytes(&[1.0, 2.0]), Dtype::F32, vec![2]),
        ]);
        assert!(matches!(
            conv1d::<B>(&wrong_bias, "layer", 2, 3, 1, 1, 1, &Default::default()),
            Err(IrodoriError::Shape(_))
        ));
    }

    #[test]
    fn malformed_conv_bias_is_not_treated_as_missing() {
        let store = tensor_store(&[
            (
                "layer.weight",
                f32_bytes(&conv_weight()),
                Dtype::F32,
                vec![3, 2, 1],
            ),
            (
                "layer.bias",
                f32_bytes(&[1.0, 2.0, 3.0]),
                Dtype::F32,
                vec![1, 3],
            ),
        ]);

        let result = conv1d::<B>(&store, "layer", 2, 3, 1, 1, 1, &Default::default());
        assert!(matches!(
            result,
            Err(IrodoriError::WrongDim(key, 1, 2)) if key == "layer.bias"
        ));
    }

    #[test]
    fn malformed_transposed_conv_bias_is_not_treated_as_missing() {
        let store = tensor_store(&[
            (
                "layer.weight",
                f32_bytes(&[0.0; 2 * 3 * 4]),
                Dtype::F32,
                vec![2, 3, 4],
            ),
            (
                "layer.bias",
                f32_bytes(&[1.0, 2.0, 3.0]),
                Dtype::F32,
                vec![1, 3],
            ),
        ]);

        let result = conv_transpose1d::<B>(&store, "layer", 2, 3, 2, &Default::default());
        assert!(matches!(
            result,
            Err(IrodoriError::WrongDim(key, 1, 2)) if key == "layer.bias"
        ));
    }

    #[test]
    fn rank_correct_wrong_transposed_conv_shape_is_rejected() {
        let store = tensor_store(&[(
            "layer.weight",
            f32_bytes(&[0.0; 2 * 2 * 4]),
            Dtype::F32,
            vec![2, 2, 4],
        )]);

        assert!(matches!(
            conv_transpose1d::<B>(&store, "layer", 2, 3, 2, &Default::default()),
            Err(IrodoriError::Shape(_))
        ));
    }
}
