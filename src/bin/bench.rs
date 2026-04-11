//! Throughput benchmark for NdArray and WGPU backends.
//!
//! Uses the small validation model (`target/validate_weights.safetensors`).
//! Generate it first with `just validate-fixtures`.
//!
//! # Usage
//! ```sh
//! just bench                       # NdArray
//! just bench -- --backend wgpu     # WGPU (GPU or software fallback)
//! just bench -- --steps 5 --runs 3 # Custom steps/runs
//! ```

#![recursion_limit = "256"]

use std::time::{Duration, Instant};

use burn::{
    backend::NdArray,
    tensor::{Bool, Int, Tensor, backend::Backend},
};
use clap::{Parser, ValueEnum};
use safetensors::SafeTensors;

use irodori_tts_burn::{
    inference::InferenceBuilder,
    rf::{SamplerParams, SamplingRequest},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "bench", about = "Benchmark NdArray vs WGPU backend throughput")]
struct Args {
    /// Backend to use for benchmarking.
    #[arg(long, default_value = "ndarray")]
    backend: BackendChoice,

    /// Path to the safetensors weights file.
    #[arg(long, default_value = "target/validate_weights.safetensors")]
    weights: String,

    /// Number of diffusion steps per run.
    #[arg(long, default_value_t = 4)]
    steps: usize,

    /// Number of sampler runs (excluding warm-up).
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Number of warm-up runs (not included in timing).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Output sequence length in frames.
    #[arg(long, default_value_t = 16)]
    seq_len: usize,
}

#[derive(Debug, Clone, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Wgpu,
}

// ---------------------------------------------------------------------------
// Generic benchmark runner
// ---------------------------------------------------------------------------

fn run_bench<B: Backend>(device: B::Device, args: &Args) {
    println!("Loading model from '{}' …", args.weights);
    let engine = InferenceBuilder::<B, _>::new(device.clone())
        .load_weights(&args.weights)
        .expect("failed to load model weights")
        .with_sampling(SamplerParams {
            num_steps: args.steps,
            ..SamplerParams::default()
        })
        .build();

    // Build dummy text tokens from the validate weights file (its shape info
    // only — we just need a plausible integer sequence).
    let bytes = std::fs::read(&args.weights).expect("could not re-read weights for shape info");
    let st = SafeTensors::deserialize(&bytes).expect("deserialise");
    let text_vocab: usize = st
        .tensor("text_encoder.embedding.weight")
        .map(|v| v.shape()[0])
        .unwrap_or(256);
    drop(st);
    drop(bytes);

    let text_len: usize = 4;
    let text_ids = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![0_i32, 1, 2, 3], [1, text_len]),
        &device,
    )
    .clamp(0, text_vocab as i32 - 1);
    let text_mask: Tensor<B, 2, Bool> =
        Tensor::<B, 2>::ones([1, text_len], &device).greater_elem(0.0f32);

    let make_request = || SamplingRequest::<B> {
        text_ids: text_ids.clone(),
        text_mask: text_mask.clone(),
        ref_latent: None,
        ref_mask: None,
        sequence_length: args.seq_len,
        caption_ids: None,
        caption_mask: None,
        initial_noise: None,
    };

    // Warm-up
    print!("Warming up ({} run(s)) … ", args.warmup);
    for _ in 0..args.warmup {
        let _out = engine.sample(make_request());
    }
    println!("done.");

    // Timed runs
    let mut durations: Vec<Duration> = Vec::with_capacity(args.runs);
    for i in 0..args.runs {
        let t0 = Instant::now();
        let _out = engine.sample(make_request());
        let elapsed = t0.elapsed();
        durations.push(elapsed);
        println!(
            "  run {:>2}: {:.1} ms",
            i + 1,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    let mean_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / durations.len() as f64;
    let min_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let max_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\nResults ({} steps × {} runs):", args.steps, args.runs);
    println!("  mean: {mean_ms:.1} ms");
    println!("  min:  {min_ms:.1} ms");
    println!("  max:  {max_ms:.1} ms");
    println!("  per-step: {:.1} ms", mean_ms / args.steps as f64);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    println!("=== Irodori-TTS-burn benchmark ===");
    println!("backend  : {:?}", args.backend);
    println!("weights  : {}", args.weights);
    println!("steps    : {}", args.steps);
    println!("runs     : {}", args.runs);
    println!("seq_len  : {}", args.seq_len);
    println!();

    match args.backend {
        BackendChoice::Ndarray => {
            type B = NdArray<f32>;
            run_bench::<B>(Default::default(), &args);
        }
        BackendChoice::Wgpu => {
            use burn::backend::Wgpu;
            use burn::backend::wgpu::WgpuDevice;
            type B = Wgpu<f32, i32>;
            run_bench::<B>(WgpuDevice::DefaultDevice, &args);
        }
    }
}
