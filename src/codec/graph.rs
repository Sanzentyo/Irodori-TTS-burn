//! Long-lived WGPU software-graph decode sessions.
//!
//! The captured dispatch plan is process-local and portable across WGPU
//! backends. It does not serialize native pipelines. A stable input buffer is
//! refreshed by one bit-preserving GPU copy, so RF latents remain GPU-resident.

use std::{collections::BTreeMap, marker::PhantomData};

pub use crate::graph::CapturedTensorOutput as CapturedCodecOutput;
use crate::graph::{CapturedTensorGraph, capture_tensor_graph};
use burn::tensor::Tensor;

use crate::{codec::DacVaeDecoder, error::IrodoriError};

/// A fixed-shape codec dispatch plan with stable input and output addresses.
///
/// Only final CPU readback is exposed by the safe API. This prevents a caller
/// from retaining the reusable output tensor across the next replay. The model
/// borrow keeps every captured weight alive for the graph lifetime.
#[derive(Debug)]
pub struct CapturedCodecDecode<'model> {
    inner: CapturedTensorGraph,
    _model: PhantomData<&'model ()>,
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
    graphs: BTreeMap<[usize; 3], CapturedTensorGraph>,
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
            crate::graph::validate_input_geometry(input_dims)?;
            if graphs.contains_key(&input_dims) {
                continue;
            }
            let example = Tensor::<3>::zeros(input_dims, device);
            let graph = capture_tensor_graph(&example, |latent| Ok(decoder.decode_wgsl(latent)))?;
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

pub(super) fn capture_codec_decode<'model, M, F>(
    _model: &'model M,
    example: &Tensor<3>,
    decode: F,
) -> crate::error::Result<CapturedCodecDecode<'model>>
where
    F: Fn(Tensor<3>) -> Tensor<3>,
    M: ?Sized,
{
    let inner = capture_tensor_graph(example, |input| Ok(decode(input)))?;
    Ok(CapturedCodecDecode {
        inner,
        _model: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_graph_replays_generated_kernels_after_wgsl_input_copy() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let example = Tensor::<3>::zeros([1, 1, 8], &device);
        let owner = ();
        let mut captured =
            capture_codec_decode(&owner, &example, |input| input * 2.0 + 1.0).unwrap();
        for shift in [0.0_f32, 10.0] {
            let values = (0..8).map(|i| i as f32 + shift).collect::<Vec<_>>();
            let input = Tensor::<3>::from_data(
                burn::tensor::TensorData::new(values.clone(), [1, 1, 8]),
                &device,
            );
            assert_eq!(
                captured.decode_to_cpu_f32(input).unwrap(),
                values
                    .into_iter()
                    .map(|v| v * 2.0 + 1.0)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn captured_geometry_rejects_zero_dimensions() {
        assert!(crate::graph::validate_input_geometry([1, 112, 32]).is_ok());
        for dims in [[0, 112, 32], [1, 0, 32], [1, 112, 0]] {
            let error =
                crate::graph::validate_input_geometry(dims).expect_err("zero dimension must fail");
            assert!(error.to_string().contains("must be non-zero"));
        }
    }
}
