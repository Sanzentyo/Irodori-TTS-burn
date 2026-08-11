//! Validate and measure a bounded k=7 Conv1d tile matrix against the accepted
//! T64/O16/WG128 production kernel.
//!
//! Four-case screening (short/high-C and long/low-C, dilation 1 and 9):
//! `cargo run --release --bin bench_conv1d_k7_tile_candidates -- <adapter-index>`
//!
//! One exact production case, indexed in shape-major/dilation-minor order:
//! `cargo run --release --bin bench_conv1d_k7_tile_candidates -- <adapter-index> --case 0`
//!
//! All four production shapes and all three dilations:
//! `cargo run --release --bin bench_conv1d_k7_tile_candidates -- <adapter-index> --all --winner-only`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    module::{Param, ParamId},
    nn::{Initializer, PaddingConfig1d, conv::Conv1d, conv::Conv1dConfig},
    tensor::{Distribution, Tensor, TensorData, TensorPrimitive, backend::Backend},
};
use irodori_tts_wgpu::WgpuRaw;

#[path = "../kernels/conv1d_k7_tile_candidates.rs"]
mod conv1d_k7_tile_candidates;
#[path = "../kernels/conv1d_k7_tiled.rs"]
mod conv1d_k7_tiled;

use conv1d_k7_tile_candidates::{Conv1dK7TileCandidate, conv1d_k7_same_tile_candidate_wgsl};
use conv1d_k7_tiled::{Conv1dK7Dilation, conv1d_k7_same_tiled_wgsl};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 5;
const DEFAULT_TRIALS: usize = 7;
const CPU_CHANNELS: usize = 32;
const CPU_LENGTH: usize = 73;
const KERNEL_SIZE: usize = 7;
const PRODUCTION_TOLERANCE: f32 = 2.0e-3;
const CPU_TOLERANCE: f32 = 2.0e-5;
const EXACT_CANDIDATE_TOLERANCE: f32 = 0.0;
const PORTABLE_WORKGROUP_STORAGE_BYTES: usize = 16 * 1024;
const CANDIDATES: [Conv1dK7TileCandidate; 3] = [
    Conv1dK7TileCandidate::Time32Output16,
    Conv1dK7TileCandidate::Time32Output32,
    Conv1dK7TileCandidate::Time64Output32,
];

#[derive(Clone, Copy, Debug)]
struct ConvCase {
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
}

#[derive(Debug)]
struct Args {
    adapter_index: Option<usize>,
    all_cases: bool,
    winner_only: bool,
    case_index: Option<usize>,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            adapter_index: None,
            all_cases: false,
            winner_only: false,
            case_index: None,
            warmup: DEFAULT_WARMUP,
            iterations: DEFAULT_ITERATIONS,
            trials: DEFAULT_TRIALS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct CaseTiming {
    case: ConvCase,
    accepted: Timing,
    winner: Option<Timing>,
}

fn usage() -> &'static str {
    "usage: bench_conv1d_k7_tile_candidates [adapter-index] [--all | --case N] [--winner-only] \
     [--warmup N] [--iterations N] [--trials N]"
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    value.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {value:?} for {option}: {error}")).into()
    })
}

fn next_positive_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = next_usize(args, option)?;
    if value == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(value)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--all" => parsed.all_cases = true,
            "--winner-only" => parsed.winner_only = true,
            "--case" => parsed.case_index = Some(next_usize(&mut args, "--case")?),
            "--warmup" => parsed.warmup = next_positive_usize(&mut args, "--warmup")?,
            "--iterations" => parsed.iterations = next_positive_usize(&mut args, "--iterations")?,
            "--trials" => parsed.trials = next_positive_usize(&mut args, "--trials")?,
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ if argument.starts_with('-') => {
                return Err(
                    io::Error::other(format!("unknown option {argument:?}; {}", usage())).into(),
                );
            }
            _ if parsed.adapter_index.is_none() => {
                parsed.adapter_index = Some(argument.parse::<usize>().map_err(|error| {
                    io::Error::other(format!(
                        "invalid adapter index {argument:?}: {error}; {}",
                        usage()
                    ))
                })?);
            }
            _ => {
                return Err(io::Error::other(format!(
                    "unexpected positional argument {argument:?}; {}",
                    usage()
                ))
                .into());
            }
        }
    }
    if parsed.all_cases && parsed.case_index.is_some() {
        return Err(io::Error::other("--all and --case are mutually exclusive").into());
    }
    Ok(parsed)
}

fn all_production_cases() -> Vec<ConvCase> {
    [(768, 600), (384, 6_000), (192, 48_000), (96, 96_000)]
        .into_iter()
        .flat_map(|(channels, length)| {
            [
                Conv1dK7Dilation::One,
                Conv1dK7Dilation::Three,
                Conv1dK7Dilation::Nine,
            ]
            .into_iter()
            .map(move |dilation| ConvCase {
                channels,
                length,
                dilation,
            })
        })
        .collect()
}

fn selected_cases(args: &Args) -> Result<Vec<ConvCase>, Box<dyn Error>> {
    let all = all_production_cases();
    if args.all_cases {
        return Ok(all);
    }
    if let Some(index) = args.case_index {
        return all
            .get(index)
            .copied()
            .map(|case| vec![case])
            .ok_or_else(|| {
                io::Error::other(format!(
                    "--case index {index} is out of range; valid range is 0..{}",
                    all.len()
                ))
                .into()
            });
    }
    Ok([0, 2, 9, 11].into_iter().map(|index| all[index]).collect())
}

fn selected_candidates(args: &Args) -> &'static [Conv1dK7TileCandidate] {
    if args.winner_only {
        &CANDIDATES[2..]
    } else {
        &CANDIDATES
    }
}

fn current_forward(
    conv: &Conv1d<B>,
    input: Tensor<B, 3>,
    dilation: Conv1dK7Dilation,
) -> Tensor<B, 3> {
    let bias = conv
        .bias
        .as_ref()
        .expect("benchmark convolution must have a bias");
    let output = conv1d_k7_same_tiled_wgsl(
        input.into_primitive().tensor(),
        conv.weight.val().into_primitive().tensor(),
        bias.val().into_primitive().tensor(),
        dilation,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn candidate_forward(
    conv: &Conv1d<B>,
    input: Tensor<B, 3>,
    dilation: Conv1dK7Dilation,
    candidate: Conv1dK7TileCandidate,
) -> Tensor<B, 3> {
    let bias = conv
        .bias
        .as_ref()
        .expect("benchmark convolution must have a bias");
    let input = input.into_primitive().tensor();
    let weight = conv.weight.val().into_primitive().tensor();
    let bias = bias.val().into_primitive().tensor();
    let output = if candidate == Conv1dK7TileCandidate::Time64Output32 {
        let dilation = irodori_tts_wgpu::kernels::conv1d_k7_tiled::Conv1dK7Dilation::try_from(
            dilation.value(),
        )
        .expect("benchmark dilation must be supported");
        irodori_tts_wgpu::kernels::conv1d_k7_tiled_o32::conv1d_k7_same_tiled_o32_wgsl(
            input, weight, bias, dilation,
        )
    } else {
        conv1d_k7_same_tile_candidate_wgsl(input, weight, bias, dilation.value(), candidate)
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn synchronize(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn measure<F>(warmup: usize, iterations: usize, trials: usize, mut operation: F) -> Timing
where
    F: FnMut() -> Tensor<B, 3>,
{
    let mut warmup_output = None;
    for _ in 0..warmup {
        warmup_output = Some(operation());
    }
    synchronize(warmup_output.expect("warmup count must be non-zero"));

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let started = Instant::now();
        let mut output = None;
        for _ in 0..iterations {
            output = Some(operation());
        }
        synchronize(output.expect("iteration count must be non-zero"));
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    Timing {
        median_us: samples[samples.len() / 2],
        min_us: samples[0],
        max_us: samples[samples.len() - 1],
    }
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn max_abs_diff_cpu(actual: Tensor<B, 3>, expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    let actual = actual.into_data().to_vec::<f32>()?;
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "CPU reference length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

fn check_error(name: &str, error: f32, tolerance: f32) -> Result<(), Box<dyn Error>> {
    if error.is_finite() && error <= tolerance {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{name} max_abs={error:.3e} exceeds tolerance {tolerance:.3e}"
        ))
        .into())
    }
}

fn deterministic_values(length: usize, modulus: usize, multiplier: usize, scale: f32) -> Vec<f32> {
    let centre = (modulus / 2) as f32;
    (0..length)
        .map(|index| (((index % modulus) * multiplier) % modulus) as f32 - centre)
        .map(|value| value * scale)
        .collect()
}

/// Scalar f32 reference using the shaders' input-channel then kernel order.
fn cpu_reference_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    length: usize,
    dilation: usize,
) -> Vec<f32> {
    let padding = 3 * dilation;
    let mut output = vec![0.0_f32; channels * length];
    for output_channel in 0..channels {
        for output_time in 0..length {
            let mut accumulator = bias[output_channel];
            for input_channel in 0..channels {
                for kernel_index in 0..KERNEL_SIZE {
                    let source_time = output_time as isize + (kernel_index * dilation) as isize
                        - padding as isize;
                    if (0..length as isize).contains(&source_time) {
                        let input_index = input_channel * length + source_time as usize;
                        let weight_index = (output_channel * channels + input_channel)
                            * KERNEL_SIZE
                            + kernel_index;
                        accumulator = input[input_index].mul_add(weight[weight_index], accumulator);
                    }
                }
            }
            output[output_channel * length + output_time] = accumulator;
        }
    }
    output
}

fn conv_with_data(
    device: &<B as Backend>::Device,
    channels: usize,
    dilation: Conv1dK7Dilation,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
) -> Conv1d<B> {
    let dilation_value = dilation.value();
    let mut conv = Conv1dConfig::new(channels, channels, KERNEL_SIZE)
        .with_dilation(dilation_value)
        .with_padding(PaddingConfig1d::Explicit(
            3 * dilation_value,
            3 * dilation_value,
        ))
        .with_bias(true)
        .with_initializer(Initializer::Zeros)
        .init::<B>(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = Some(Param::initialized(ParamId::new(), bias));
    conv
}

fn validate_cpu_reference(
    device: &<B as Backend>::Device,
    candidates: &[Conv1dK7TileCandidate],
) -> Result<(), Box<dyn Error>> {
    let input_values = deterministic_values(CPU_CHANNELS * CPU_LENGTH, 29, 11, 1.0 / 16.0);
    let weight_values = deterministic_values(
        CPU_CHANNELS * CPU_CHANNELS * KERNEL_SIZE,
        31,
        7,
        1.0 / 256.0,
    );
    let bias_values = deterministic_values(CPU_CHANNELS, 17, 5, 1.0 / 128.0);

    for dilation in [
        Conv1dK7Dilation::One,
        Conv1dK7Dilation::Three,
        Conv1dK7Dilation::Nine,
    ] {
        let input = Tensor::<B, 3>::from_data(
            TensorData::new(input_values.clone(), [1, CPU_CHANNELS, CPU_LENGTH]),
            device,
        );
        let weight = Tensor::<B, 3>::from_data(
            TensorData::new(
                weight_values.clone(),
                [CPU_CHANNELS, CPU_CHANNELS, KERNEL_SIZE],
            ),
            device,
        );
        let bias =
            Tensor::<B, 1>::from_data(TensorData::new(bias_values.clone(), [CPU_CHANNELS]), device);
        let conv = conv_with_data(device, CPU_CHANNELS, dilation, weight, bias);
        let expected = cpu_reference_conv1d(
            &input_values,
            &weight_values,
            &bias_values,
            CPU_CHANNELS,
            CPU_LENGTH,
            dilation.value(),
        );
        let current = current_forward(&conv, input.clone(), dilation);
        let current_error = max_abs_diff_cpu(current.clone(), &expected)?;
        check_error(
            "accepted CPU-reference smoke test",
            current_error,
            CPU_TOLERANCE,
        )?;

        for &candidate in candidates {
            let candidate_output = candidate_forward(&conv, input.clone(), dilation, candidate);
            let cpu_error = max_abs_diff_cpu(candidate_output.clone(), &expected)?;
            let exact_error = max_abs_diff(current.clone(), candidate_output)?;
            check_error(
                "candidate CPU-reference smoke test",
                cpu_error,
                CPU_TOLERANCE,
            )?;
            check_error(
                "candidate-versus-accepted smoke test",
                exact_error,
                EXACT_CANDIDATE_TOLERANCE,
            )?;
            println!(
                "CPU reference d={}: {:17} scalar max_abs={cpu_error:.3e}, accepted max_abs={exact_error:.3e}",
                dilation.value(),
                candidate.label(),
            );
        }
    }
    Ok(())
}

fn useful_macs(case: ConvCase) -> u128 {
    let length = case.length as u128;
    let dilation = case.dilation.value() as u128;
    let valid_positions = (0..KERNEL_SIZE)
        .map(|kernel_index| {
            let offset = (kernel_index as i128 - 3) * dilation as i128;
            length.saturating_sub(offset.unsigned_abs())
        })
        .sum::<u128>();
    (case.channels as u128).pow(2) * valid_positions
}

fn gflops(macs: u128, time_us: f64) -> f64 {
    2.0 * macs as f64 / (time_us * 1_000.0)
}

fn current_shared_bytes(dilation: usize) -> usize {
    let input_elements = 16 * (64 + 6 * dilation);
    let weight_elements = 16 * 16 * KERNEL_SIZE;
    (input_elements + weight_elements) * core::mem::size_of::<f32>()
}

fn print_timing(label: &str, timing: Timing, macs: u128, relative_to_current: f64) {
    println!(
        "  {label:17} median={:10.1} us [{:10.1}, {:10.1}], {:8.2} GFLOP/s, current/candidate={relative_to_current:6.3}x",
        timing.median_us,
        timing.min_us,
        timing.max_us,
        gflops(macs, timing.median_us),
    );
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: ConvCase,
    args: &Args,
    candidates: &[Conv1dK7TileCandidate],
) -> Result<CaseTiming, Box<dyn Error>> {
    let dilation = case.dilation.value();
    let conv = Conv1dConfig::new(case.channels, case.channels, KERNEL_SIZE)
        .with_dilation(dilation)
        .with_padding(PaddingConfig1d::Explicit(3 * dilation, 3 * dilation))
        .with_bias(true)
        .init::<B>(device);
    let input = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );

    let burn_output = conv.forward(input.clone());
    let current_output = current_forward(&conv, input.clone(), case.dilation);
    let burn_error = max_abs_diff(burn_output, current_output.clone())?;
    check_error(
        "accepted production Conv1d",
        burn_error,
        PRODUCTION_TOLERANCE,
    )?;

    let current = measure(args.warmup, args.iterations, args.trials, || {
        current_forward(&conv, input.clone(), case.dilation)
    });
    let macs = useful_macs(case);
    let current_shared = current_shared_bytes(dilation);
    println!(
        "C={:4} L={:6} d={dilation}: MAC={:.6}G, accepted-vs-Burn max_abs={burn_error:.3e}",
        case.channels,
        case.length,
        macs as f64 / 1.0e9,
    );
    println!(
        "  {:17} tile=64x16 wg=128 acc=8 shared={current_shared:5}B dispatch={}x{}",
        "accepted",
        case.length.div_ceil(64),
        case.channels / 16,
    );
    print_timing("accepted", current, macs, 1.0);

    let mut winner = None;
    for &candidate in candidates {
        let candidate_output = candidate_forward(&conv, input.clone(), case.dilation, candidate);
        let exact_error = max_abs_diff(current_output.clone(), candidate_output)?;
        check_error(
            "candidate-versus-accepted production output",
            exact_error,
            EXACT_CANDIDATE_TOLERANCE,
        )?;
        let timing = measure(args.warmup, args.iterations, args.trials, || {
            candidate_forward(&conv, input.clone(), case.dilation, candidate)
        });
        let shared_bytes = candidate.shared_memory_bytes(dilation);
        let portability = if shared_bytes <= PORTABLE_WORKGROUP_STORAGE_BYTES {
            "portable"
        } else {
            "native-limit"
        };
        println!(
            "  {:17} tile={}x{} wg={} acc={} shared={shared_bytes:5}B {portability:12} dispatch={}x{} exact max_abs={exact_error:.3e}",
            candidate.label(),
            candidate.time_tile(),
            candidate.output_channel_tile(),
            16 * candidate.local_channel_lanes(),
            candidate.accumulators_per_invocation(),
            case.length.div_ceil(candidate.time_tile()),
            case.channels / candidate.output_channel_tile(),
        );
        print_timing(
            candidate.label(),
            timing,
            macs,
            current.median_us / timing.median_us,
        );
        if candidate == Conv1dK7TileCandidate::Time64Output32 {
            winner = Some(timing);
        }
    }
    Ok(CaseTiming {
        case,
        accepted: current,
        winner,
    })
}

fn print_winner_summary(results: &[CaseTiming]) {
    if results.is_empty() || results.iter().any(|result| result.winner.is_none()) {
        return;
    }

    println!("winner summary (sums of independently measured medians):");
    for stage in results.chunks(3) {
        let accepted_us = stage
            .iter()
            .map(|result| result.accepted.median_us)
            .sum::<f64>();
        let winner_us = stage
            .iter()
            .map(|result| {
                result
                    .winner
                    .expect("winner presence was validated")
                    .median_us
            })
            .sum::<f64>();
        let case = stage[0].case;
        println!(
            "  C={:4} L={:6}: accepted={accepted_us:10.1} us, t64-o32={winner_us:10.1} us, speedup={:6.3}x",
            case.channels,
            case.length,
            accepted_us / winner_us,
        );
    }
    let accepted_us = results
        .iter()
        .map(|result| result.accepted.median_us)
        .sum::<f64>();
    let winner_us = results
        .iter()
        .map(|result| {
            result
                .winner
                .expect("winner presence was validated")
                .median_us
        })
        .sum::<f64>();
    println!(
        "  all {:2}: accepted={accepted_us:10.1} us, t64-o32={winner_us:10.1} us, speedup={:6.3}x",
        results.len(),
        accepted_us / winner_us,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let cases = selected_cases(&args)?;
    let candidates = selected_candidates(&args);
    let device = args
        .adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, 0);

    println!(
        "DACVAE Conv1d k=7 tile candidates device={device:?}, warmup={}, iterations={}, trials={}, cases={}",
        args.warmup,
        args.iterations,
        args.trials,
        cases.len(),
    );
    for dilation in [1, 3, 9] {
        println!(
            "shared d={dilation}: accepted={:5}B, t32-o16={:5}B, t32-o32={:5}B, t64-o32={:5}B",
            current_shared_bytes(dilation),
            Conv1dK7TileCandidate::Time32Output16.shared_memory_bytes(dilation),
            Conv1dK7TileCandidate::Time32Output32.shared_memory_bytes(dilation),
            Conv1dK7TileCandidate::Time64Output32.shared_memory_bytes(dilation),
        );
    }
    validate_cpu_reference(&device, candidates)?;
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        results.push(benchmark_case(&device, case, &args, candidates)?);
    }
    if args.all_cases && args.winner_only {
        print_winner_summary(&results);
    }
    Ok(())
}
