//! Measure the exact production and alternative v4 DiT projection layouts.
//!
//! The benchmark covers the four large projections executed by every one of
//! the twelve diffusion blocks in each of the four Euler evaluations:
//! MLP `w1||w3`, MLP `w2`, JointAttention `QKV||gate`, and attention `wo`.
//! It accepts the runtime latent sequence length so long-sequence decisions are
//! not inferred from the historical `S=50` workload.
//!
//! The primary timer is pre-synchronized launch through device completion.
//! Full owned contiguous f32 CPU readback and all-element comparison are
//! performed separately before timing. Inputs and both physical weight layouts
//! remain GPU resident throughout each timing trial.
//!
//! Run with:
//! `cargo run --release --bin bench_dit_matmul_layout -- <adapter> <sequence>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{
        WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup, into_contiguous,
    },
    tensor::{Distribution, Tensor, TensorPrimitive, backend::Backend, module::linear},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const MODEL_DIM: usize = 1_280;
const MLP_HIDDEN: usize = 3_680;
const LAYERS: usize = 12;
const B1_CALLS: usize = 2 * LAYERS;
const B2_CALLS: usize = 2 * LAYERS;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const SEED: u64 = 0;
const MAX_ABS_TOLERANCE: f32 = 1.0e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightLayout {
    RowMajor,
    ColumnMajor,
}

impl WeightLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColumnMajor => "col",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputLayout {
    BatchedRank3,
    FlattenedRank2,
}

impl InputLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::BatchedRank3 => "rank3",
            Self::FlattenedRank2 => "flat",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Variant {
    input: InputLayout,
    weight: WeightLayout,
}

impl Variant {
    const fn label(self) -> &'static str {
        match (self.input, self.weight) {
            (InputLayout::BatchedRank3, WeightLayout::RowMajor) => "rank3-row",
            (InputLayout::BatchedRank3, WeightLayout::ColumnMajor) => "rank3-col",
            (InputLayout::FlattenedRank2, WeightLayout::RowMajor) => "flat-row",
            (InputLayout::FlattenedRank2, WeightLayout::ColumnMajor) => "flat-col",
        }
    }
}

const VARIANTS: [Variant; 4] = [
    Variant {
        input: InputLayout::BatchedRank3,
        weight: WeightLayout::RowMajor,
    },
    Variant {
        input: InputLayout::BatchedRank3,
        weight: WeightLayout::ColumnMajor,
    },
    Variant {
        input: InputLayout::FlattenedRank2,
        weight: WeightLayout::RowMajor,
    },
    Variant {
        input: InputLayout::FlattenedRank2,
        weight: WeightLayout::ColumnMajor,
    },
];

const FLAT_ROW: Variant = Variant {
    input: InputLayout::FlattenedRank2,
    weight: WeightLayout::RowMajor,
};
const FLAT_COLUMN: Variant = Variant {
    input: InputLayout::FlattenedRank2,
    weight: WeightLayout::ColumnMajor,
};

#[derive(Clone, Copy, Debug)]
enum ProductionPolicy {
    RowAll,
    RowB1ColumnB2,
}

impl ProductionPolicy {
    const fn for_batch(self, batch: usize) -> Variant {
        match (self, batch) {
            (Self::RowAll, _) | (Self::RowB1ColumnB2, 1) => FLAT_ROW,
            (Self::RowB1ColumnB2, 2) => FLAT_COLUMN,
            (Self::RowB1ColumnB2, _) => panic!("only B=1/B=2 are part of the replay"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ShapeCase {
    name: &'static str,
    k: usize,
    n: usize,
    production: ProductionPolicy,
}

const SHAPES: [ShapeCase; 4] = [
    ShapeCase {
        name: "mlp_expand_w1_w3",
        k: MODEL_DIM,
        n: 2 * MLP_HIDDEN,
        production: ProductionPolicy::RowAll,
    },
    ShapeCase {
        name: "mlp_contract_w2",
        k: MLP_HIDDEN,
        n: MODEL_DIM,
        production: ProductionPolicy::RowB1ColumnB2,
    },
    ShapeCase {
        name: "attention_qkv_gate",
        k: MODEL_DIM,
        n: 4 * MODEL_DIM,
        production: ProductionPolicy::RowAll,
    },
    ShapeCase {
        name: "attention_output_wo",
        k: MODEL_DIM,
        n: MODEL_DIM,
        production: ProductionPolicy::RowB1ColumnB2,
    },
];

#[derive(Clone, Debug)]
struct BatchResult {
    medians_us: [f64; VARIANTS.len()],
}

impl BatchResult {
    fn for_variant(&self, variant: Variant) -> f64 {
        self.medians_us[variant_index(variant)]
    }

    fn best(&self) -> (Variant, f64) {
        VARIANTS
            .into_iter()
            .zip(self.medians_us)
            .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
            .expect("at least one projection variant")
    }
}

#[derive(Clone, Debug)]
struct ShapeResult {
    case: ShapeCase,
    b1: BatchResult,
    b2: BatchResult,
    pack_us: f64,
    weight_bytes: usize,
}

fn variant_index(variant: Variant) -> usize {
    VARIANTS
        .iter()
        .position(|candidate| *candidate == variant)
        .expect("variant must be benchmarked")
}

fn parse_args() -> Result<(usize, usize), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let adapter = args
        .next()
        .ok_or_else(|| io::Error::other("missing required WGPU adapter index"))?
        .parse::<usize>()?;
    let sequence = args
        .next()
        .ok_or_else(|| io::Error::other("missing required latent sequence length"))?
        .parse::<usize>()?;
    if ![13, 25, 50, 100, 200].contains(&sequence) {
        return Err(io::Error::other(format!(
            "unsupported sequence {sequence}; expected one of 13,25,50,100,200"
        ))
        .into());
    }
    if let Some(extra) = args.next() {
        return Err(io::Error::other(format!(
            "unexpected argument {extra:?}; expected <adapter> <sequence>"
        ))
        .into());
    }
    Ok((adapter, sequence))
}

fn sync_device(device: &<B as Backend>::Device, label: &str) -> Result<(), Box<dyn Error>> {
    cubecl::future::block_on(WgpuRuntime::client(device).sync())
        .map_err(|error| io::Error::other(format!("{label} device sync failed: {error}")))?;
    Ok(())
}

fn pack_row_major(weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let packed = into_contiguous(weight.into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(packed))
}

fn project(
    input: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    variant: Variant,
    batch: usize,
    sequence: usize,
    case: ShapeCase,
) -> Tensor<B, 3> {
    match variant.input {
        InputLayout::BatchedRank3 => input.matmul(weight.unsqueeze::<3>()),
        InputLayout::FlattenedRank2 => {
            linear(input.reshape([batch * sequence, case.k]), weight, None)
                .reshape([batch, sequence, case.n])
        }
    }
}

fn full_comparison(
    reference: Tensor<B, 3>,
    candidate: Tensor<B, 3>,
) -> Result<(usize, f32, f64), Box<dyn Error>> {
    let expected = reference.into_data().to_vec::<f32>()?;
    let actual = candidate.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() || expected.is_empty() {
        return Err(io::Error::other("projection output length mismatch or empty output").into());
    }
    let element_count = expected.len();
    let mut bit_mismatch = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    for (lhs, rhs) in expected.into_iter().zip(actual) {
        if !lhs.is_finite() || !rhs.is_finite() {
            return Err(io::Error::other("projection output contains a non-finite value").into());
        }
        bit_mismatch += usize::from(lhs.to_bits() != rhs.to_bits());
        let error = (lhs - rhs).abs();
        max_abs = max_abs.max(error);
        sum_abs += f64::from(error);
    }
    let mean_abs = sum_abs / element_count as f64;
    if max_abs > MAX_ABS_TOLERANCE {
        return Err(io::Error::other(format!(
            "projection layout max_abs {max_abs:e} exceeds {MAX_ABS_TOLERANCE:e}"
        ))
        .into());
    }
    Ok((bit_mismatch, max_abs, mean_abs))
}

fn measure<F>(mut operation: F, device: &<B as Backend>::Device) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Tensor<B, 3>,
{
    let mut warmup_output = operation();
    for _ in 1..WARMUP {
        warmup_output = operation();
    }
    sync_device(device, "projection warmup")?;
    drop(warmup_output);

    sync_device(device, "projection pre-timer")?;
    let started = Instant::now();
    let mut output = operation();
    for _ in 1..ITERATIONS {
        output = operation();
    }
    sync_device(device, "projection device-complete")?;
    let elapsed = started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
    drop(output);
    Ok(elapsed)
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bench_batch(
    case: ShapeCase,
    batch: usize,
    sequence: usize,
    input: Tensor<B, 3>,
    weight_column: &Tensor<B, 2>,
    weight_row: &Tensor<B, 2>,
    device: &<B as Backend>::Device,
) -> Result<BatchResult, Box<dyn Error>> {
    let select_weight = |layout| match layout {
        WeightLayout::RowMajor => weight_row.clone(),
        WeightLayout::ColumnMajor => weight_column.clone(),
    };
    let production = case.production.for_batch(batch);
    let reference = project(
        input.clone(),
        select_weight(production.weight),
        production,
        batch,
        sequence,
        case,
    );
    for variant in VARIANTS {
        let candidate = project(
            input.clone(),
            select_weight(variant.weight),
            variant,
            batch,
            sequence,
            case,
        );
        let (bit_mismatch, max_abs, mean_abs) = full_comparison(reference.clone(), candidate)?;
        println!(
            "correctness case={} batch={batch} sequence={sequence} variant={} elements={} bit_mismatch={bit_mismatch} max_abs={max_abs:.9e} mean_abs={mean_abs:.9e}",
            case.name,
            variant.label(),
            batch * sequence * case.n,
        );
    }

    let mut samples: [Vec<f64>; VARIANTS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..VARIANTS.len() {
            let index = (trial + offset) % VARIANTS.len();
            let variant = VARIANTS[index];
            let weight = select_weight(variant.weight);
            samples[index].push(measure(
                || {
                    project(
                        input.clone(),
                        weight.clone(),
                        variant,
                        batch,
                        sequence,
                        case,
                    )
                },
                device,
            )?);
        }
    }

    let medians_us = std::array::from_fn(|index| median(&samples[index]));
    let minimums_us: [f64; VARIANTS.len()] =
        std::array::from_fn(|index| samples[index].iter().copied().fold(f64::INFINITY, f64::min));
    let maximums_us: [f64; VARIANTS.len()] = std::array::from_fn(|index| {
        samples[index]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let production_us = medians_us[variant_index(production)];
    for (index, variant) in VARIANTS.into_iter().enumerate() {
        println!(
            "timing case={} batch={batch} sequence={sequence} variant={} input={} weight={} median_us={:.6} min_us={:.6} max_us={:.6} speedup_vs_production={:.6}",
            case.name,
            variant.label(),
            variant.input.label(),
            variant.weight.label(),
            medians_us[index],
            minimums_us[index],
            maximums_us[index],
            production_us / medians_us[index],
        );
    }
    Ok(BatchResult { medians_us })
}

fn bench_shape(
    case: ShapeCase,
    sequence: usize,
    device: &<B as Backend>::Device,
) -> Result<ShapeResult, Box<dyn Error>> {
    // Reproduce checkpoint loading: contiguous physical PyTorch `[N,K]`, then
    // a metadata-only transpose to Burn's column-major logical `[K,N]` view.
    let physical =
        Tensor::<B, 2>::random([case.n, case.k], Distribution::Uniform(-0.05, 0.05), device);
    sync_device(device, "physical weight creation")?;
    let weight_column = physical.transpose();
    let column_primitive = weight_column.clone().into_primitive().tensor();
    if column_primitive.is_contiguous()
        || &column_primitive.meta.strides()[..] != [1, case.k].as_slice()
    {
        return Err(io::Error::other(format!(
            "{} source weight does not have checkpoint column-major layout",
            case.name
        ))
        .into());
    }
    sync_device(device, "row-major pack pre-sync")?;
    let started = Instant::now();
    let weight_row = pack_row_major(weight_column.clone());
    sync_device(device, "row-major pack device-complete")?;
    let pack_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    let row_primitive = weight_row.clone().into_primitive().tensor();
    if !row_primitive.is_contiguous() || &row_primitive.meta.strides()[..] != [case.n, 1].as_slice()
    {
        return Err(io::Error::other(format!(
            "{} packed weight is not canonical row-major",
            case.name
        ))
        .into());
    }

    let b1_input = Tensor::<B, 3>::random(
        [1, sequence, case.k],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let b2_input = Tensor::<B, 3>::random(
        [2, sequence, case.k],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let b1 = bench_batch(
        case,
        1,
        sequence,
        b1_input,
        &weight_column,
        &weight_row,
        device,
    )?;
    let b2 = bench_batch(
        case,
        2,
        sequence,
        b2_input,
        &weight_column,
        &weight_row,
        device,
    )?;
    Ok(ShapeResult {
        case,
        b1,
        b2,
        pack_us,
        weight_bytes: case.k * case.n * core::mem::size_of::<f32>(),
    })
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (adapter_index, sequence) = parse_args()?;
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);
    println!(
        "dit_projection_layout schema=2 device={device:?} sequence={sequence} warmup={WARMUP} iterations={ITERATIONS} trials={TRIALS} timer=pre_sync_to_device_complete readback=full_owned_contiguous_f32_outside_timer seed={SEED}"
    );

    let results = SHAPES
        .into_iter()
        .map(|case| bench_shape(case, sequence, &device))
        .collect::<Result<Vec<_>, _>>()?;

    let mut production_workload_us = 0.0;
    let mut best_workload_us = 0.0;
    let mut cache_bytes = 0_usize;
    let mut pack_us = 0.0;
    for result in &results {
        let production_b1 = result.case.production.for_batch(1);
        let production_b2 = result.case.production.for_batch(2);
        let (best_b1_variant, best_b1_us) = result.b1.best();
        let (best_b2_variant, best_b2_us) = result.b2.best();
        let production_case_us = B1_CALLS as f64 * result.b1.for_variant(production_b1)
            + B2_CALLS as f64 * result.b2.for_variant(production_b2);
        let best_case_us = B1_CALLS as f64 * best_b1_us + B2_CALLS as f64 * best_b2_us;
        production_workload_us += production_case_us;
        best_workload_us += best_case_us;
        pack_us += result.pack_us * LAYERS as f64;
        if best_b1_variant.weight == WeightLayout::RowMajor
            || best_b2_variant.weight == WeightLayout::RowMajor
        {
            cache_bytes += result.weight_bytes * LAYERS;
        }
        println!(
            "decision case={} sequence={sequence} production_b1={} production_b2={} best_b1={} best_b2={} production_workload_ms={:.6} best_workload_ms={:.6} saving_ms={:.6}",
            result.case.name,
            production_b1.label(),
            production_b2.label(),
            best_b1_variant.label(),
            best_b2_variant.label(),
            production_case_us / 1_000.0,
            best_case_us / 1_000.0,
            (production_case_us - best_case_us) / 1_000.0,
        );
    }
    println!(
        "aggregate sequence={sequence} calls_per_shape={} production_workload_ms={:.6} best_observed_workload_ms={:.6} saving_ms={:.6} speedup={:.6} additive_row_cache_mib={:.3} twelve_layer_pack_ms={:.6}",
        B1_CALLS + B2_CALLS,
        production_workload_us / 1_000.0,
        best_workload_us / 1_000.0,
        (production_workload_us - best_workload_us) / 1_000.0,
        production_workload_us / best_workload_us,
        mib(cache_bytes),
        pack_us / 1_000.0,
    );
    Ok(())
}
