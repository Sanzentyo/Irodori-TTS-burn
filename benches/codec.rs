//! Criterion benchmarks for the DACVAE codec.
//!
//! Requires the converted codec weights at `target/dacvae_weights.safetensors`.
//! Generate them first with `just codec-convert`, then run with:
//! ```sh
//! just bench-codec
//! ```
//!
//! Benchmarks:
//! - `codec_encode_1s`  — encode 1 second of silence (48 kHz, 48000 samples)
//! - `codec_encode_5s`  — encode 5 seconds of silence (240000 samples)
//! - `codec_decode_1s`  — decode 25 latent frames (~1s)
//! - `codec_decode_5s`  — decode 125 latent frames (~5s)

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use criterion::{Criterion, criterion_group, criterion_main};

use irodori_tts_burn::load_codec;

const WEIGHTS_PATH: &str = "target/dacvae_weights.safetensors";
const SAMPLE_RATE: usize = 48_000;
const HOP_LENGTH: usize = 1920;

type B = NdArray;

fn setup_codec() -> irodori_tts_burn::codec::DacVaeCodec<B> {
    let path = std::path::Path::new(WEIGHTS_PATH);
    if !path.exists() {
        panic!(
            "Codec weights not found at {WEIGHTS_PATH}. \
             Run `just codec-convert` first."
        );
    }
    load_codec::<B>(path, &Default::default()).expect("failed to load codec")
}

/// Build a latent tensor with `n_frames` of zeros.
fn zero_latent(n_frames: usize) -> Tensor<B, 3> {
    Tensor::<B, 3>::zeros([1, n_frames, 32], &Default::default())
}

/// Build a minimal noise tensor to avoid degenerate all-zero paths.
fn noise_audio(seconds: f32) -> Tensor<B, 3> {
    let n = (SAMPLE_RATE as f32 * seconds).round() as usize;
    // Deterministic "noise" via sin wave at 440 Hz, amplitude 0.01
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.01 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    Tensor::<B, 3>::from_data(TensorData::new(samples, [1, 1, n]), &Default::default())
}

fn bench_encode(c: &mut Criterion) {
    let codec = setup_codec();

    let mut group = c.benchmark_group("codec_encode");

    group.bench_function("1s_sine", |b| {
        let audio = noise_audio(1.0);
        b.iter(|| codec.encode(audio.clone()))
    });

    group.bench_function("5s_sine", |b| {
        let audio = noise_audio(5.0);
        b.iter(|| codec.encode(audio.clone()))
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let codec = setup_codec();

    // 1 s ≈ SAMPLE_RATE / HOP_LENGTH = 48000/1920 = 25 frames
    let frames_1s = SAMPLE_RATE / HOP_LENGTH;
    let frames_5s = frames_1s * 5;

    let mut group = c.benchmark_group("codec_decode");

    group.bench_function("1s_zero_latent", |b| {
        let latent = zero_latent(frames_1s);
        b.iter(|| codec.decode(latent.clone()))
    });

    group.bench_function("5s_zero_latent", |b| {
        let latent = zero_latent(frames_5s);
        b.iter(|| codec.decode(latent.clone()))
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let codec = setup_codec();

    let mut group = c.benchmark_group("codec_roundtrip");

    group.bench_function("1s_encode_decode", |b| {
        let audio = noise_audio(1.0);
        b.iter(|| {
            let latent = codec.encode(audio.clone());
            codec.decode(latent)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
