//! Measure rank-4 batching for the three LowRankAdaLN modulation branches.
//!
//! Run with:
//! `cargo run --release --bin bench_adaln_modulation -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Distribution, Tensor, activation::silu, backend::Backend},
};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const MODEL_DIM: usize = 1_280;
const RANK: usize = 192;
const BRANCHES: usize = 3;
const ADALN_MODULES: usize = 24;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const SEED: u64 = 0;

#[derive(Clone)]
struct OriginalWeights {
    down: [Tensor<B, 2>; BRANCHES],
    up: [Tensor<B, 2>; BRANCHES],
    bias: [Tensor<B, 1>; BRANCHES],
}

#[derive(Clone)]
struct PackedWeights {
    down: Tensor<B, 4>,
    up: Tensor<B, 4>,
    bias: Tensor<B, 4>,
}

fn original_weights(device: &<B as Backend>::Device) -> OriginalWeights {
    let down = std::array::from_fn(|_| {
        Tensor::random(
            [MODEL_DIM, RANK],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
    });
    let up = std::array::from_fn(|_| {
        Tensor::random(
            [RANK, MODEL_DIM],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
    });
    let bias = std::array::from_fn(|_| {
        Tensor::random([MODEL_DIM], Distribution::Uniform(-0.05, 0.05), device)
    });
    OriginalWeights { down, up, bias }
}

/// Stack the existing per-branch tensors without changing their storage layout.
fn pack_weights(weights: &OriginalWeights) -> PackedWeights {
    let down = Tensor::<B, 2>::stack::<3>(weights.down.to_vec(), 0).unsqueeze_dim::<4>(0);
    let up = Tensor::<B, 2>::stack::<3>(weights.up.to_vec(), 0).unsqueeze_dim::<4>(0);
    let bias =
        Tensor::<B, 1>::stack::<2>(weights.bias.to_vec(), 0).reshape([1, BRANCHES, 1, MODEL_DIM]);
    PackedWeights { down, up, bias }
}

/// Current LowRankAdaLN implementation: three independent SiLU/down/up/bias branches.
fn baseline_modulation(raw: Tensor<B, 4>, weights: &OriginalWeights) -> [Tensor<B, 3>; BRANCHES] {
    let batch = raw.dims()[0];
    std::array::from_fn(|branch| {
        let raw_branch = raw
            .clone()
            .narrow(1, branch, 1)
            .reshape([batch, 1, MODEL_DIM]);
        let refined = silu(raw_branch.clone())
            .matmul(weights.down[branch].clone().unsqueeze_dim::<3>(0))
            .matmul(weights.up[branch].clone().unsqueeze_dim::<3>(0))
            + weights.bias[branch].clone().reshape([1, 1, MODEL_DIM]);
        refined + raw_branch
    })
}

/// Candidate implementation: one rank-4 SiLU and two batched matmuls.
fn batched_modulation(raw: Tensor<B, 4>, weights: &PackedWeights) -> Tensor<B, 4> {
    let activated = silu(raw.clone());
    batched_modulation_from_activated(raw, activated, weights)
}

fn batched_modulation_from_activated(
    raw: Tensor<B, 4>,
    activated: Tensor<B, 4>,
    weights: &PackedWeights,
) -> Tensor<B, 4> {
    let refined = activated
        .matmul(weights.down.clone())
        .matmul(weights.up.clone())
        + weights.bias.clone();
    refined + raw
}

/// Execute the 24 AdaLN modules in one v4 DiT forward.
///
/// The weights are intentionally shared in this isolated benchmark: the
/// measured difference only depends on whether the identical SiLU input is
/// recomputed for every module. Production modules keep distinct weights.
fn modulation_group(
    raw: &Tensor<B, 4>,
    weights: &PackedWeights,
    reuse_activated: bool,
) -> Tensor<B, 4> {
    let activated = reuse_activated.then(|| silu(raw.clone()));
    let mut output = None;
    for _ in 0..ADALN_MODULES {
        output = Some(match &activated {
            Some(activated) => {
                batched_modulation_from_activated(raw.clone(), activated.clone(), weights)
            }
            None => batched_modulation(raw.clone(), weights),
        });
    }
    output.expect("ADALN_MODULES must be non-zero")
}

fn stack_outputs(outputs: [Tensor<B, 3>; BRANCHES]) -> Tensor<B, 4> {
    Tensor::<B, 3>::stack::<4>(outputs.into_iter().collect(), 1)
}

fn sync_3d(outputs: [Tensor<B, 3>; BRANCHES]) {
    // WGPU queue submission is ordered. Reading the last branch waits for all three branches.
    let [_, _, gate] = outputs;
    let _ = gate.slice([0..1, 0..1, 0..1]).into_data();
}

fn sync_4d(tensor: Tensor<B, 4>) {
    let _ = tensor.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

fn max_abs_diff(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn measure_baseline(raw: &Tensor<B, 4>, weights: &OriginalWeights) -> f64 {
    let mut warmup_output = None;
    for _ in 0..WARMUP {
        warmup_output = Some(baseline_modulation(raw.clone(), weights));
    }
    sync_3d(warmup_output.expect("WARMUP must be non-zero"));

    let started = Instant::now();
    let mut output = None;
    for _ in 0..ITERATIONS {
        output = Some(baseline_modulation(raw.clone(), weights));
    }
    sync_3d(output.expect("ITERATIONS must be non-zero"));
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn measure_batched(raw: &Tensor<B, 4>, weights: &PackedWeights) -> f64 {
    let mut warmup_output = None;
    for _ in 0..WARMUP {
        warmup_output = Some(batched_modulation(raw.clone(), weights));
    }
    sync_4d(warmup_output.expect("WARMUP must be non-zero"));

    let started = Instant::now();
    let mut output = None;
    for _ in 0..ITERATIONS {
        output = Some(batched_modulation(raw.clone(), weights));
    }
    sync_4d(output.expect("ITERATIONS must be non-zero"));
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn measure_modulation_group(
    raw: &Tensor<B, 4>,
    weights: &PackedWeights,
    reuse_activated: bool,
) -> f64 {
    let mut warmup_output = None;
    for _ in 0..WARMUP {
        warmup_output = Some(modulation_group(raw, weights, reuse_activated));
    }
    sync_4d(warmup_output.expect("WARMUP must be non-zero"));

    let started = Instant::now();
    let mut output = None;
    for _ in 0..ITERATIONS {
        output = Some(modulation_group(raw, weights, reuse_activated));
    }
    sync_4d(output.expect("ITERATIONS must be non-zero"));
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn measure_pack(weights: &OriginalWeights) -> (PackedWeights, f64) {
    let mut warmup_output = None;
    for _ in 0..WARMUP {
        warmup_output = Some(pack_weights(weights));
    }
    let warmup_output = warmup_output.expect("WARMUP must be non-zero");
    sync_4d(warmup_output.bias);

    let started = Instant::now();
    let mut output = None;
    for _ in 0..ITERATIONS {
        output = Some(pack_weights(weights));
    }
    let output = output.expect("ITERATIONS must be non-zero");
    sync_4d(output.bias.clone());
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
    (output, elapsed_us)
}

fn parse_adapter_index() -> Result<Option<usize>, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let adapter_index = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if let Some(extra) = args.next() {
        return Err(io::Error::other(format!(
            "unexpected argument {extra:?}; expected at most one WGPU adapter index"
        ))
        .into());
    }
    Ok(adapter_index)
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_index = parse_adapter_index()?;
    let device = adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);

    println!(
        "LowRankAdaLN modulation benchmark device={device:?} D={MODEL_DIM} R={RANK} \
         ({WARMUP} warmup, {ITERATIONS} measured, seed={SEED})"
    );

    let weights = original_weights(&device);
    let (packed, pack_us) = measure_pack(&weights);
    println!("weight pack (3 down + 3 up + 3 bias): {pack_us:.1} us/pack");

    for batch in [1, 2] {
        let raw = Tensor::<B, 4>::random(
            [batch, BRANCHES, 1, MODEL_DIM],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let expected = stack_outputs(baseline_modulation(raw.clone(), &weights));
        let actual = batched_modulation(raw.clone(), &packed);
        let max_abs = max_abs_diff(expected, actual)?;

        let baseline_us = measure_baseline(&raw, &weights);
        let batched_us = measure_batched(&raw, &packed);
        println!(
            "B={batch}: baseline={baseline_us:.1} us, batched={batched_us:.1} us, \
             speedup={:.2}x, max_abs={max_abs:.3e}",
            baseline_us / batched_us
        );

        let repeated = modulation_group(&raw, &packed, false);
        let reused = modulation_group(&raw, &packed, true);
        let reused_max_abs = max_abs_diff(repeated, reused)?;
        let repeated_us = measure_modulation_group(&raw, &packed, false);
        let reused_us = measure_modulation_group(&raw, &packed, true);
        println!(
            "B={batch}: {ADALN_MODULES} modules repeated_silu={repeated_us:.1} us, \
             shared_silu={reused_us:.1} us, speedup={:.3}x, saved={:.1} us/forward, \
             max_abs={reused_max_abs:.3e}",
            repeated_us / reused_us,
            repeated_us - reused_us,
        );
    }

    Ok(())
}
