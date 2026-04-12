//! Simple wall-clock timing benchmark for the DACVAE codec.
//!
//! Works with any backend (NdArray, LibTorch, WGPU).  Mirrors the
//! `scripts/bench_codec_py.py` output format so results can be compared
//! directly.
//!
//! # Usage
//! ```sh
//! # NdArray (CPU, no libtorch dependency)
//! cargo run --release --bin bench_codec -- --weights target/dacvae_weights.safetensors
//!
//! # LibTorch (matches PyTorch performance)
//! just bench-codec-tch
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::{Tensor, TensorData};
use clap::Parser;

use irodori_tts_burn::codec::load_codec;

// ── Backend selection (mirrors pipeline.rs pattern) ─────────────────────────

#[cfg(any(
    all(feature = "backend_wgpu", feature = "backend_wgpu_f16"),
    all(feature = "backend_wgpu", feature = "backend_wgpu_bf16"),
    all(feature = "backend_wgpu_f16", feature = "backend_wgpu_bf16"),
    all(feature = "backend_wgpu", feature = "backend_tch"),
    all(feature = "backend_wgpu", feature = "backend_tch_bf16"),
    all(feature = "backend_wgpu_f16", feature = "backend_tch"),
    all(feature = "backend_wgpu_f16", feature = "backend_tch_bf16"),
    all(feature = "backend_wgpu_bf16", feature = "backend_tch"),
    all(feature = "backend_wgpu_bf16", feature = "backend_tch_bf16"),
    all(feature = "backend_tch", feature = "backend_tch_bf16"),
))]
compile_error!(
    "Enable at most one of: backend_wgpu, backend_wgpu_f16, backend_wgpu_bf16, backend_tch, backend_tch_bf16"
);

#[cfg(feature = "backend_wgpu")]
type B = burn::backend::Wgpu;
#[cfg(feature = "backend_wgpu")]
fn default_device() -> burn::backend::wgpu::WgpuDevice {
    Default::default()
}
#[cfg(feature = "backend_wgpu")]
fn backend_name() -> &'static str {
    "WGPU (f32)"
}

#[cfg(feature = "backend_tch")]
type B = burn::backend::LibTorch;
#[cfg(feature = "backend_tch")]
fn default_device() -> burn::backend::libtorch::LibTorchDevice {
    burn::backend::libtorch::LibTorchDevice::Cpu
}
#[cfg(feature = "backend_tch")]
fn backend_name() -> &'static str {
    "LibTorch (f32, CPU)"
}

#[cfg(feature = "backend_tch_bf16")]
type B = burn::backend::LibTorch<half::bf16>;
#[cfg(feature = "backend_tch_bf16")]
fn default_device() -> burn::backend::libtorch::LibTorchDevice {
    burn::backend::libtorch::LibTorchDevice::Cpu
}
#[cfg(feature = "backend_tch_bf16")]
fn backend_name() -> &'static str {
    "LibTorch (bf16, CPU)"
}

// Default: NdArray
#[cfg(not(any(
    feature = "backend_wgpu",
    feature = "backend_wgpu_f16",
    feature = "backend_wgpu_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
type B = burn::backend::NdArray;
#[cfg(not(any(
    feature = "backend_wgpu",
    feature = "backend_wgpu_f16",
    feature = "backend_wgpu_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
fn default_device() -> burn::backend::ndarray::NdArrayDevice {
    Default::default()
}
#[cfg(not(any(
    feature = "backend_wgpu",
    feature = "backend_wgpu_f16",
    feature = "backend_wgpu_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
fn backend_name() -> &'static str {
    "NdArray (CPU)"
}

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_codec",
    about = "Benchmark DACVAE codec encode/decode speed"
)]
struct Args {
    /// Path to the DACVAE safetensors weights.
    #[arg(long, default_value = "target/dacvae_weights.safetensors")]
    weights: PathBuf,

    /// Number of warmup iterations per benchmark.
    #[arg(long, default_value_t = 2)]
    n_warmup: usize,

    /// Number of timed runs per benchmark.
    #[arg(long, default_value_t = 5)]
    n_runs: usize,

    /// Only run 1-second benchmarks (faster smoke test).
    #[arg(long)]
    quick: bool,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const SAMPLE_RATE: usize = 48_000;
const HOP_LENGTH: usize = 1_920;
const LATENT_DIM: usize = 32;

fn sine_audio(seconds: f32) -> Tensor<B, 3> {
    let n = (SAMPLE_RATE as f32 * seconds).round() as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.01 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    Tensor::<B, 3>::from_data(TensorData::new(samples, [1, 1, n]), &default_device())
}

fn zero_latent(frames: usize) -> Tensor<B, 3> {
    Tensor::<B, 3>::zeros([1, frames, LATENT_DIM], &default_device())
}

/// Run `f` `n_warmup + n_runs` times, return sorted durations of the timed runs.
fn bench_fn<F: FnMut()>(label: &str, mut f: F, n_warmup: usize, n_runs: usize) {
    for _ in 0..n_warmup {
        f();
    }
    let mut durations: Vec<Duration> = (0..n_runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    durations.sort();

    let median = durations[durations.len() / 2];
    let mean: Duration = durations.iter().sum::<Duration>() / n_runs as u32;
    println!(
        "[bench] {label}: median={:.1}ms  mean={:.1}ms  runs={n_runs}",
        median.as_secs_f64() * 1000.0,
        mean.as_secs_f64() * 1000.0,
    );
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run(args: Args) -> Result<()> {
    println!(
        "\n=== Rust DACVAE codec benchmark (backend={}) ===\n",
        backend_name()
    );

    let codec = load_codec::<B>(&args.weights, &default_device())
        .context("Failed to load codec weights")?;

    let durations: &[f32] = if args.quick { &[1.0] } else { &[1.0, 5.0] };

    for &dur in durations {
        let audio = sine_audio(dur);
        let frames = (SAMPLE_RATE as f32 * dur / HOP_LENGTH as f32).round() as usize;
        let latent = zero_latent(frames);

        bench_fn(
            &format!("encode_{dur:.0}s_sine"),
            || {
                let _ = codec.encode(audio.clone());
            },
            args.n_warmup,
            args.n_runs,
        );
        bench_fn(
            &format!("decode_{dur:.0}s_zero_latent"),
            || {
                let _ = codec.decode(latent.clone());
            },
            args.n_warmup,
            args.n_runs,
        );
        bench_fn(
            &format!("roundtrip_{dur:.0}s"),
            || {
                let z = codec.encode(audio.clone());
                let _ = codec.decode(z);
            },
            args.n_warmup,
            args.n_runs,
        );
    }

    println!();
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
