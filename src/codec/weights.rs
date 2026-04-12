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

use crate::{error::IrodoriError, weights::TensorStore};

use super::{
    bottleneck::VaeBottleneck,
    decoder::{Decoder, DecoderBlock, WmHead},
    encoder::{Encoder, EncoderBlock},
    layers::{ResidualUnit, Snake1d, conv_transpose_pad, make_conv1d},
    model::DacVaeCodec,
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
    let hop_length: usize = 2 * 8 * 10 * 12; // 1920
    let sample_rate: usize = 48_000;

    Ok(DacVaeCodec {
        encoder: build_encoder(store, device)?,
        bottleneck: build_bottleneck(store, device)?,
        decoder: build_decoder(store, device)?,
        hop_length,
        sample_rate,
    })
}

// ─── Primitive helpers ───────────────────────────────────────────────────────

fn snake1d<B: Backend>(
    store: &TensorStore,
    key: &str,
    device: &B::Device,
) -> Result<Snake1d<B>, IrodoriError> {
    let alpha: Tensor<B, 3> = store.tensor(&format!("{key}.alpha"), device)?;
    Ok(Snake1d::new(alpha))
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
    let weight: Tensor<B, 3> = store.tensor(&format!("{prefix}.weight"), device)?;
    let bias: Option<Tensor<B, 1>> = store.tensor(&format!("{prefix}.bias"), device).ok();
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
    let weight: Tensor<B, 3> = store.tensor(&format!("{prefix}.weight"), device)?;
    let bias: Option<Tensor<B, 1>> = store.tensor(&format!("{prefix}.bias"), device).ok();

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
    let act0 = snake1d(store, &format!("{prefix}.block.0"), device)?;
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
    let act1 = snake1d(store, &format!("{prefix}.block.2"), device)?;
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
    let tail_act = snake1d(store, &format!("{prefix}.block.3"), device)?;
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
    let tail_act = snake1d(store, "encoder.block.5", device)?;
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
    let act = snake1d(store, &format!("{prefix}.block.0"), device)?;
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
    let act = snake1d(store, "decoder.wm_model.encoder_block.pre.0", device)?;
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
