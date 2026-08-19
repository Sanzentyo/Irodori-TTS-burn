//! Long-lived WGPU software-graph decode sessions.
//!
//! The captured dispatch plan is process-local and portable across WGPU
//! backends. It does not serialize native pipelines. A stable input buffer is
//! refreshed by one bit-preserving GPU copy, so RF latents remain GPU-resident.

use std::marker::PhantomData;

use burn::{backend::wgpu::WgpuRuntime, tensor::Tensor};
use cubecl::client::Graph;

use crate::{WgpuRaw, error::IrodoriError};

/// A fixed-shape codec dispatch plan with stable input and output addresses.
///
/// Only final CPU readback is exposed by the safe API. This prevents a caller
/// from retaining the reusable output tensor across the next replay. The model
/// borrow keeps every captured weight alive for the graph lifetime.
#[derive(Debug)]
pub struct CapturedCodecDecode<'model> {
    stable_input: Tensor<3>,
    output: Tensor<3>,
    graph: Graph<WgpuRuntime>,
    input_dims: [usize; 3],
    output_dims: [usize; 3],
    _model: PhantomData<&'model ()>,
}

impl CapturedCodecDecode<'_> {
    /// Fixed latent geometry accepted by this graph.
    pub fn input_dims(&self) -> [usize; 3] {
        self.input_dims
    }

    /// Fixed waveform geometry produced by this graph.
    pub fn output_dims(&self) -> [usize; 3] {
        self.output_dims
    }

    /// Refresh the stable GPU input, replay the codec, and return final F32
    /// audio on the CPU. No latent or intermediate tensor is read back.
    pub fn decode_to_cpu_f32(&mut self, latent: Tensor<3>) -> crate::error::Result<Vec<f32>> {
        if latent.dims() != self.input_dims {
            return Err(IrodoriError::Shape(format!(
                "captured codec expects latent {:?}, got {:?}",
                self.input_dims,
                latent.dims()
            )));
        }
        copy_into_stable_input(&latent, &self.stable_input)?;
        // SAFETY: this type owns the stable input, reusable output, and graph;
        // its model borrow keeps weights alive. `&mut self` serializes updates
        // and replay, and `into_data` completes the stream before returning.
        unsafe { self.graph.replay() };
        self.output
            .clone()
            .cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                IrodoriError::Dtype("captured codec output".to_owned(), error.to_string())
            })
    }
}

pub(super) fn capture_codec_decode<'model, M, F>(
    _model: &'model M,
    example: &Tensor<3>,
    decode: F,
) -> crate::error::Result<CapturedCodecDecode<'model>>
where
    F: Fn(Tensor<3>) -> Tensor<3>,
    M: ?Sized,
{
    let primitive = example
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| IrodoriError::UnsupportedMode("codec graph requires WGPU".to_owned()))?;
    let client = primitive.client;
    let stable_input = client
        .memory_persistent_allocation((), |()| {
            Tensor::<3>::zeros(example.dims(), &example.device())
        })
        .map_err(|error| {
            IrodoriError::Config(format!("stable codec input allocation failed: {error}"))
        })?;
    copy_into_stable_input(example, &stable_input)?;
    sync_client(&client, "codec graph input initialization")?;

    client.graph_prepare().map_err(|error| {
        IrodoriError::Config(format!("codec graph preparation failed: {error}"))
    })?;
    let priming_output = decode(stable_input.clone());
    sync_client(&client, "codec graph priming")?;
    drop(priming_output);
    client
        .start_capture()
        .map_err(|error| IrodoriError::Config(format!("codec graph capture failed: {error}")))?;
    let output = decode(stable_input.clone());
    let graph = client.stop_capture().map_err(|error| {
        IrodoriError::Config(format!("codec graph finalization failed: {error}"))
    })?;
    let output_dims = output.dims();

    Ok(CapturedCodecDecode {
        stable_input,
        output,
        graph,
        input_dims: example.dims(),
        output_dims,
        _model: PhantomData,
    })
}

fn copy_into_stable_input(source: &Tensor<3>, destination: &Tensor<3>) -> crate::error::Result<()> {
    let source = source
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| {
            IrodoriError::UnsupportedMode("codec graph source requires WGPU".to_owned())
        })?;
    let destination = destination
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|_| {
            IrodoriError::UnsupportedMode("codec graph destination requires WGPU".to_owned())
        })?;
    crate::kernels::contiguous_copy::copy_contiguous_into_wgsl(source, destination).map_err(
        |reason| IrodoriError::Config(format!("codec graph input copy rejected: {reason}")),
    )
}

fn sync_client(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    stage: &str,
) -> crate::error::Result<()> {
    cubecl::future::block_on(client.sync())
        .map_err(|error| IrodoriError::Config(format!("{stage} failed: {error}")))
}
