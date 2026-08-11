//! Validate and measure the isolated tiled DACVAE k=7 Conv1d candidate.
//!
//! Quick production extremes:
//! `cargo run --release --bin bench_conv1d_k7_tiled -- <adapter-index>`
//!
//! All four production shapes and all three dilations:
//! `cargo run --release --bin bench_conv1d_k7_tiled -- <adapter-index> --all`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    module::{Param, ParamId},
    nn::{Initializer, PaddingConfig1d, conv::Conv1d, conv::Conv1dConfig},
    tensor::{Distribution, Tensor, TensorData, TensorPrimitive, backend::Backend},
};
use irodori_tts_wgpu::WgpuRaw;

#[path = "../kernels/conv1d_k7_tiled.rs"]
mod conv1d_k7_tiled;

use conv1d_k7_tiled::{Conv1dK7Dilation, conv1d_k7_same_tiled_wgsl};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 5;
const DEFAULT_TRIALS: usize = 5;
const CPU_CHANNELS: usize = 16;
const CPU_LENGTH: usize = 73;
const KERNEL_SIZE: usize = 7;
const PRODUCTION_TOLERANCE: f32 = 2.0e-3;
const CPU_TOLERANCE: f32 = 2.0e-5;

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
    warmup: usize,
    iterations: usize,
    trials: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            adapter_index: None,
            all_cases: false,
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

fn usage() -> &'static str {
    "usage: bench_conv1d_k7_tiled [adapter-index] [--all] \
     [--warmup N] [--iterations N] [--trials N]"
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let parsed = value.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {value:?} for {option}: {error}"))
    })?;
    if parsed == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(parsed)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--all" => parsed.all_cases = true,
            "--warmup" => parsed.warmup = next_usize(&mut args, "--warmup")?,
            "--iterations" => parsed.iterations = next_usize(&mut args, "--iterations")?,
            "--trials" => parsed.trials = next_usize(&mut args, "--trials")?,
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
    Ok(parsed)
}

fn quick_cases() -> Vec<ConvCase> {
    vec![
        ConvCase {
            channels: 768,
            length: 600,
            dilation: Conv1dK7Dilation::Nine,
        },
        ConvCase {
            channels: 96,
            length: 96_000,
            dilation: Conv1dK7Dilation::One,
        },
    ]
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

fn custom_forward(
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

/// Scalar f32 reference using the shader's input-channel then kernel order.
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

fn validate_cpu_reference(device: &<B as Backend>::Device) -> Result<(), Box<dyn Error>> {
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
        let burn_error = max_abs_diff_cpu(conv.forward(input.clone()), &expected)?;
        let tiled_error = max_abs_diff_cpu(custom_forward(&conv, input, dilation), &expected)?;
        check_error("Burn CPU-reference smoke test", burn_error, CPU_TOLERANCE)?;
        check_error("tiled CPU-reference smoke test", tiled_error, CPU_TOLERANCE)?;
        println!(
            "CPU reference C={CPU_CHANNELS:3} L={CPU_LENGTH:3} d={}: \
             Burn max_abs={burn_error:.3e}, tiled max_abs={tiled_error:.3e}",
            dilation.value()
        );
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

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: ConvCase,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
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
    let tiled_output = custom_forward(&conv, input.clone(), case.dilation);
    let error = max_abs_diff(burn_output, tiled_output)?;
    check_error("production tiled Conv1d", error, PRODUCTION_TOLERANCE)?;

    let burn = measure(args.warmup, args.iterations, args.trials, || {
        conv.forward(input.clone())
    });
    let tiled = measure(args.warmup, args.iterations, args.trials, || {
        custom_forward(&conv, input.clone(), case.dilation)
    });
    let macs = useful_macs(case);
    println!(
        "C={:4} L={:6} d={dilation}: max_abs={error:.3e}, MAC={:.6}G\n\
         current Burn: median={:10.1} us [{:10.1}, {:10.1}], {:8.2} GFLOP/s\n\
         tiled WGSL:   median={:10.1} us [{:10.1}, {:10.1}], {:8.2} GFLOP/s, speedup={:5.2}x",
        case.channels,
        case.length,
        macs as f64 / 1.0e9,
        burn.median_us,
        burn.min_us,
        burn.max_us,
        gflops(macs, burn.median_us),
        tiled.median_us,
        tiled.min_us,
        tiled.max_us,
        gflops(macs, tiled.median_us),
        burn.median_us / tiled.median_us,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let device = args
        .adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, 0);

    println!(
        "Tiled DACVAE Conv1d k=7 device={device:?}, warmup={}, iterations={}, trials={}, cases={} ",
        args.warmup,
        args.iterations,
        args.trials,
        if args.all_cases { "all 12" } else { "quick 2" }
    );
    validate_cpu_reference(&device)?;
    let cases = if args.all_cases {
        all_production_cases()
    } else {
        quick_cases()
    };
    for case in cases {
        benchmark_case(&device, case, &args)?;
    }
    Ok(())
}
