//! Compare codec convolution operators with equivalent GEMM formulations.
//!
//! Run with:
//! `cargo run --release --bin bench_codec_convolutions -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    nn::{PaddingConfig1d, conv::Conv1d, conv::Conv1dConfig},
    tensor::{Distribution, Tensor, backend::Backend},
};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

#[derive(Clone)]
struct PackedPointwise {
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 3>>,
    output_channels: usize,
}

fn synchronize(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn measure<F>(mut operation: F) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    for _ in 0..WARMUP {
        let _ = operation();
    }
    synchronize(operation());

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = operation();
    }
    synchronize(operation());
    started.elapsed().as_secs_f64() * 1_000_000.0 / (ITERATIONS + 1) as f64
}

fn linearized_conv1d(conv: &Conv1d<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, _, length] = input.dims();
    let output_channels = conv.weight.dims()[0];
    let weight = conv
        .weight
        .val()
        .squeeze_dim::<2>(2)
        .transpose()
        .unsqueeze_dim::<3>(0);
    let mut output = input.swap_dims(1, 2).matmul(weight);
    if let Some(bias) = &conv.bias {
        output = output + bias.val().reshape([1, 1, output_channels]);
    }
    output
        .swap_dims(1, 2)
        .reshape([batch, output_channels, length])
}

fn pack_pointwise(conv: &Conv1d<B>) -> PackedPointwise {
    let output_channels = conv.weight.dims()[0];
    let weight = conv
        .weight
        .val()
        .squeeze_dim::<2>(2)
        .transpose()
        .add_scalar(0.0)
        .unsqueeze_dim::<3>(0);
    let bias = conv
        .bias
        .as_ref()
        .map(|bias| bias.val().reshape([1, 1, output_channels]));
    PackedPointwise {
        weight,
        bias,
        output_channels,
    }
}

fn packed_conv1d(packed: &PackedPointwise, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, _, length] = input.dims();
    let mut output = input.swap_dims(1, 2).matmul(packed.weight.clone());
    if let Some(bias) = &packed.bias {
        output = output + bias.clone();
    }
    output
        .swap_dims(1, 2)
        .reshape([batch, packed.output_channels, length])
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn bench_conv1x1(
    device: &<B as Backend>::Device,
    input_channels: usize,
    output_channels: usize,
    length: usize,
) -> Result<(), Box<dyn Error>> {
    let conv = Conv1dConfig::new(input_channels, output_channels, 1)
        .with_padding(PaddingConfig1d::Valid)
        .with_bias(true)
        .init::<B>(device);
    let input = Tensor::<B, 3>::random(
        [1, input_channels, length],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let expected = conv.forward(input.clone());
    let actual = linearized_conv1d(&conv, input.clone());
    let max_abs = max_abs_diff(expected, actual)?;
    let packed = pack_pointwise(&conv);
    let packed_max_abs = max_abs_diff(
        conv.forward(input.clone()),
        packed_conv1d(&packed, input.clone()),
    )?;

    let conv_us = measure(|| conv.forward(input.clone()));
    let matmul_us = measure(|| linearized_conv1d(&conv, input.clone()));
    let packed_us = measure(|| packed_conv1d(&packed, input.clone()));
    println!(
        "Conv1x1 Cin={input_channels:4} Cout={output_channels:4} L={length:6}: \
         conv={conv_us:9.1} us, \
         matmul={matmul_us:9.1} us ({:5.2}x), packed={packed_us:9.1} us ({:5.2}x), \
         max_abs={max_abs:.3e}/{packed_max_abs:.3e}",
        conv_us / matmul_us,
        conv_us / packed_us,
    );
    Ok(())
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
    B::seed(&device, 0);

    println!(
        "Codec convolution benchmark device={device:?} \
         ({WARMUP} warmup, {ITERATIONS} measured)"
    );
    for (input_channels, output_channels, length) in [
        (32, 1_024, 50),
        (768, 768, 600),
        (384, 384, 6_000),
        (192, 192, 48_000),
        (96, 96, 96_000),
    ] {
        bench_conv1x1(&device, input_channels, output_channels, length)?;
    }
    Ok(())
}
