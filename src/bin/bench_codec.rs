//! Simple wall-clock timing benchmark for the DACVAE codec.
//!
//! Works with any backend (LibTorch, WGPU, CUDA). Backend is selected at
//! runtime via `--backend`.
//!
//! # Usage
//! ```sh
//! just bench-codec-tch   # --backend libtorch
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use burn::tensor::{Tensor, TensorData};
use clap::Parser;

use irodori_tts_burn::backend_config::BackendConfig;
use irodori_tts_burn::load_codec;
use irodori_tts_burn::{InferenceBackendKind, dispatch_inference};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_codec",
    about = "Benchmark DACVAE codec encode/decode speed"
)]
struct Args {
    /// Inference backend to use.
    #[arg(long)]
    backend: InferenceBackendKind,

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

    /// GPU device index (0-based).
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const SAMPLE_RATE: usize = 48_000;
const HOP_LENGTH: usize = 1_920;
const LATENT_DIM: usize = 32;

fn sine_audio<B: BackendConfig>(seconds: f32, device: &B::Device) -> Tensor<B, 3> {
    let n = (SAMPLE_RATE as f32 * seconds).round() as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.01 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    Tensor::<B, 3>::from_data(TensorData::new(samples, [1, 1, n]), device)
}

fn zero_latent<B: BackendConfig>(frames: usize, device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::zeros([1, frames, LATENT_DIM], device)
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

fn main() {
    let args = Args::parse();
    let result = dispatch_inference!(args.backend, args.gpu_id, |B, device| run::<B>(
        args, device
    ));
    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

fn run<B: BackendConfig>(args: Args, device: B::Device) -> Result<()> {
    println!(
        "\n=== Rust DACVAE codec benchmark (backend={}) ===\n",
        B::backend_label()
    );

    let codec = load_codec::<B>(&args.weights, &device).context("Failed to load codec weights")?;

    let durations: &[f32] = if args.quick { &[1.0] } else { &[1.0, 5.0] };

    for &dur in durations {
        let audio = sine_audio::<B>(dur, &device);
        let frames = (SAMPLE_RATE as f32 * dur / HOP_LENGTH as f32).round() as usize;
        let latent = zero_latent::<B>(frames, &device);

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
