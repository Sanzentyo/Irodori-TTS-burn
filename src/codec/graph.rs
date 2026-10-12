//! Long-lived WGPU software-graph decode sessions.
//!
//! The captured dispatch plan is process-local and portable across WGPU
//! backends. It does not serialize native pipelines. A stable input buffer is
//! refreshed by one bit-preserving GPU copy, so RF latents remain GPU-resident.

use std::{collections::BTreeMap, marker::PhantomData};

use burn::{
    backend::wgpu::WgpuRuntime,
    tensor::{FloatDType, Tensor},
};
use cubecl::client::Graph;

use crate::{WgpuRaw, codec::DacVaeDecoder, error::IrodoriError};

#[derive(Debug)]
struct CapturedDecodeGraph {
    stable_input: Tensor<3>,
    output: Tensor<3>,
    graph: Graph<WgpuRuntime>,
    input_dims: [usize; 3],
    output_dims: [usize; 3],
}

/// A fixed-shape codec dispatch plan with stable input and output addresses.
///
/// Only final CPU readback is exposed by the safe API. This prevents a caller
/// from retaining the reusable output tensor across the next replay. The model
/// borrow keeps every captured weight alive for the graph lifetime.
#[derive(Debug)]
pub struct CapturedCodecDecode<'model> {
    inner: CapturedDecodeGraph,
    _model: PhantomData<&'model ()>,
}

/// One replayed codec output whose storage remains owned by its graph.
///
/// The mutable borrow of the captured session prevents another replay until
/// this value is consumed. Only an owned CPU result is exposed, so reusable
/// graph storage cannot escape into application state.
#[derive(Debug)]
pub struct CapturedCodecOutput<'graph> {
    output: &'graph Tensor<3>,
}

impl CapturedCodecOutput<'_> {
    /// Fixed waveform geometry produced by this replay.
    pub fn dims(&self) -> [usize; 3] {
        self.output.dims()
    }

    /// Complete final readback and return owned contiguous F32 audio.
    pub fn to_cpu_f32(self) -> crate::error::Result<Vec<f32>> {
        self.output
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                IrodoriError::Dtype("captured codec output".to_owned(), error.to_string())
            })
    }
}

impl CapturedCodecDecode<'_> {
    /// Fixed latent geometry accepted by this graph.
    pub fn input_dims(&self) -> [usize; 3] {
        self.inner.input_dims
    }

    /// Fixed waveform geometry produced by this graph.
    pub fn output_dims(&self) -> [usize; 3] {
        self.inner.output_dims
    }

    /// Refresh the stable GPU input, replay the codec through its captured F32
    /// consumer transform, and return final audio on the CPU. No latent or
    /// intermediate tensor is read back, and replay does not construct a
    /// per-request dtype-conversion dispatch.
    pub fn decode_to_cpu_f32(&mut self, latent: Tensor<3>) -> crate::error::Result<Vec<f32>> {
        self.inner.decode_to_cpu_f32(latent)
    }
}

/// An owned set of fixed-shape codec graphs.
///
/// The graphs are dropped before the decoder fields that own every captured
/// weight binding. This makes the graph lifetime invariant structural instead
/// of requiring a self-referential borrow. A mutable decode method serializes
/// updates to each graph's reusable input and output buffers.
#[derive(Debug)]
pub struct CapturedDacVaeDecoder {
    // Field order is safety-relevant: Rust drops fields in declaration order.
    graphs: BTreeMap<[usize; 3], CapturedDecodeGraph>,
    decoder: DacVaeDecoder,
}

impl CapturedDacVaeDecoder {
    pub(super) fn capture(
        decoder: DacVaeDecoder,
        input_geometries: impl IntoIterator<Item = [usize; 3]>,
        device: &burn::tensor::Device,
    ) -> crate::error::Result<Self> {
        let mut graphs = BTreeMap::new();
        for input_dims in input_geometries {
            validate_input_geometry(input_dims)?;
            if graphs.contains_key(&input_dims) {
                continue;
            }
            let example = Tensor::<3>::zeros(input_dims, device);
            let graph = capture_codec_decode_inner(&example, |latent| decoder.decode_wgsl(latent))?;
            graphs.insert(input_dims, graph);
        }
        if graphs.is_empty() {
            return Err(IrodoriError::Config(
                "captured codec requires at least one input geometry".to_owned(),
            ));
        }
        Ok(Self { graphs, decoder })
    }

    /// Fixed latent geometries admitted by this captured decoder.
    pub fn input_geometries(&self) -> impl ExactSizeIterator<Item = [usize; 3]> + '_ {
        self.graphs.keys().copied()
    }

    /// Replay the graph matching `latent` and return owned contiguous F32 audio.
    pub fn decode_to_cpu_f32(&mut self, latent: Tensor<3>) -> crate::error::Result<Vec<f32>> {
        self.enqueue(latent)?.to_cpu_f32()
    }

    /// Enqueue input refresh and graph replay without forcing CPU readback.
    ///
    /// The returned guard keeps this session mutably borrowed. A caller may
    /// therefore synchronize and record device-complete time before consuming
    /// final audio, but cannot enqueue a second request against the reusable
    /// graph storage in the meantime.
    pub fn enqueue(&mut self, latent: Tensor<3>) -> crate::error::Result<CapturedCodecOutput<'_>> {
        let input_dims = latent.dims();
        if !self.graphs.contains_key(&input_dims) {
            return Err(IrodoriError::Shape(format!(
                "captured codec has no graph for latent {input_dims:?}; admitted geometries: {:?}",
                self.graphs.keys().collect::<Vec<_>>()
            )));
        }
        let graph = self
            .graphs
            .get_mut(&input_dims)
            .expect("captured geometry was checked above");
        graph.enqueue(latent)
    }

    /// The resident decoder whose weights back every captured binding.
    pub fn decoder(&self) -> &DacVaeDecoder {
        &self.decoder
    }
}

impl CapturedDecodeGraph {
    fn decode_to_cpu_f32(&mut self, latent: Tensor<3>) -> crate::error::Result<Vec<f32>> {
        self.enqueue(latent)?.to_cpu_f32()
    }

    fn enqueue(&mut self, latent: Tensor<3>) -> crate::error::Result<CapturedCodecOutput<'_>> {
        if latent.dims() != self.input_dims {
            return Err(IrodoriError::Shape(format!(
                "captured codec expects latent {:?}, got {:?}",
                self.input_dims,
                latent.dims()
            )));
        }
        copy_into_stable_input(&latent, &self.stable_input)?;
        // SAFETY: both public owners keep every captured binding alive and
        // `&mut self` serializes stable-input refresh, replay, and readback.
        unsafe { self.graph.replay() };
        Ok(CapturedCodecOutput {
            output: &self.output,
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
    let inner = capture_codec_decode_inner(example, decode)?;
    Ok(CapturedCodecDecode {
        inner,
        _model: PhantomData,
    })
}

fn capture_codec_decode_inner<F>(
    example: &Tensor<3>,
    decode: F,
) -> crate::error::Result<CapturedDecodeGraph>
where
    F: Fn(Tensor<3>) -> Tensor<3>,
{
    validate_input_geometry(example.dims())?;
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
    let priming_output = decode(stable_input.clone()).cast(FloatDType::F32);
    sync_client(&client, "codec graph priming")?;
    drop(priming_output);
    client
        .start_capture()
        .map_err(|error| IrodoriError::Config(format!("codec graph capture failed: {error}")))?;
    // `CapturedCodecDecode` promises owned F32 CPU audio. Capture the final
    // conversion as part of that consumer boundary so a replay is the entire
    // fixed device graph rather than a graph followed by one host-enqueued
    // dispatch per request.
    let output = decode(stable_input.clone()).cast(FloatDType::F32);
    let graph = client.stop_capture().map_err(|error| {
        IrodoriError::Config(format!("codec graph finalization failed: {error}"))
    })?;
    let output_dims = output.dims();
    // Capture owns its live arena. Release only unused pages left in the
    // ordinary allocator by the priming run before another shape is captured.
    client.memory_cleanup();
    sync_client(&client, "codec graph allocator cleanup")?;

    Ok(CapturedDecodeGraph {
        stable_input,
        output,
        graph,
        input_dims: example.dims(),
        output_dims,
    })
}

fn validate_input_geometry(input_dims: [usize; 3]) -> crate::error::Result<()> {
    if input_dims.into_iter().any(|dim| dim == 0) {
        return Err(IrodoriError::Shape(format!(
            "captured codec dimensions must be non-zero, got {input_dims:?}"
        )));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_geometry_rejects_zero_dimensions() {
        assert!(validate_input_geometry([1, 112, 32]).is_ok());
        for dims in [[0, 112, 32], [1, 0, 32], [1, 112, 0]] {
            let error = validate_input_geometry(dims).expect_err("zero dimension must fail");
            assert!(error.to_string().contains("must be non-zero"));
        }
    }
}
