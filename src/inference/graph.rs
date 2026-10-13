//! Fixed conditioning and shape: only initial noise changes between replays.
use super::WgslInferenceEngine;
pub use crate::graph::CapturedTensorOutput as CapturedSamplingOutput;
use crate::{
    IrodoriError, PreparedSamplingRequest, SamplingRequest,
    graph::{CapturedTensorGraph, capture_tensor_graph},
};
use burn::tensor::Tensor;
use std::marker::PhantomData;

/// A captured denoising trajectory with fixed text, voice, shape and parameters.
/// The engine borrow prevents replacing weights or sampling parameters. Noise
/// must have the same shape, device and state precision as the capture example.
#[derive(Debug)]
pub struct CapturedSampling<'engine> {
    // Drop graph before the tensors whose addresses it captured.
    graph: CapturedTensorGraph,
    _request: PreparedSamplingRequest,
    _engine: PhantomData<&'engine WgslInferenceEngine>,
}

impl CapturedSampling<'_> {
    pub fn input_dims(&self) -> [usize; 3] {
        self.graph.input_dims
    }
    pub fn enqueue(&mut self, noise: Tensor<3>) -> crate::Result<CapturedSamplingOutput<'_>> {
        self.graph.enqueue(noise)
    }
    pub fn sample_to_cpu_f32(&mut self, noise: Tensor<3>) -> crate::Result<Vec<f32>> {
        self.enqueue(noise)?.to_cpu_f32()
    }
}

/// A captured denoising trajectory followed by the FP32 decoder. Replaying
/// returns audio directly, without reading the intermediate latent to the CPU.
#[cfg(feature = "codec")]
#[derive(Debug)]
pub struct CapturedSynthesis<'model> {
    sampling: CapturedSampling<'model>,
    _codec: PhantomData<&'model crate::codec::DacVaeDecoder>,
}

#[cfg(feature = "codec")]
impl CapturedSynthesis<'_> {
    pub fn input_dims(&self) -> [usize; 3] {
        self.sampling.input_dims()
    }
    pub fn synthesize_to_cpu_f32(&mut self, noise: Tensor<3>) -> crate::Result<Vec<f32>> {
        self.sampling.sample_to_cpu_f32(noise)
    }
}

impl WgslInferenceEngine {
    fn prepare_capture(
        &self,
        request: SamplingRequest,
    ) -> crate::Result<(PreparedSamplingRequest, Tensor<3>)> {
        if self.prepared_euler_cond_cache.is_none()
            || self.params.method != crate::SamplerMethod::Euler
        {
            return Err(IrodoriError::UnsupportedMode(
                "captured sampling requires Euler and a prepared timestep cache".into(),
            ));
        }
        if self.params.speaker_uncond_mode == crate::SpeakerUncondMode::Noise {
            return Err(IrodoriError::UnsupportedMode(
                "captured sampling requires deterministic speaker guidance".into(),
            ));
        }
        let mut request = self.prepare_sampling_request(request)?;
        if request.request.text_ids.dims()[0] != 1 {
            return Err(IrodoriError::UnsupportedMode(
                "captured sampling currently admits batch one".into(),
            ));
        }
        let example = request.request.initial_noise.clone().ok_or_else(|| {
            IrodoriError::MissingInput("capture requires an initial noise example".into())
        })?;
        let expected_dtype = self
            .params
            .state_precision
            .dtype(self.device.settings().float_dtype.into());
        if example.dtype() != expected_dtype {
            return Err(IrodoriError::Config(format!(
                "capture noise must use state dtype {expected_dtype:?}"
            )));
        }
        let aux = crate::AuxConditionInput::try_from_request(
            request.request.ref_latent.clone(),
            request.request.ref_mask.clone(),
            request.request.caption_ids.clone(),
            request.request.caption_mask.clone(),
        )?;
        request.encoded_condition = Some(self.model.encode_conditions(
            request.request.text_ids.clone(),
            request.request.text_mask.clone(),
            aux,
        )?);
        request.latent_rope = Some(
            self.model
                .precompute_latent_rope(request.sequence_length(), &self.device),
        );
        Ok((request, example))
    }

    /// Prepare and capture a fixed request. Initial preparation and compilation
    /// happen once; subsequent calls replace only the initial noise buffer.
    pub fn capture_sampling(
        &self,
        request: SamplingRequest,
    ) -> crate::Result<CapturedSampling<'_>> {
        let (request, example) = self.prepare_capture(request)?;
        let graph = capture_tensor_graph(&example, |noise| {
            let mut input = request.clone();
            input.request.initial_noise = Some(noise);
            self.sample_prepared(input)
        })?;
        Ok(CapturedSampling {
            graph,
            _request: request,
            _engine: PhantomData,
        })
    }

    /// Capture denoising and decoding together. The decoder must have completed
    /// `prepare_for_wgsl`; the returned borrow keeps both models immutable.
    #[cfg(feature = "codec")]
    pub fn capture_synthesis<'model>(
        &'model self,
        request: SamplingRequest,
        codec: &'model crate::codec::DacVaeDecoder,
    ) -> crate::Result<CapturedSynthesis<'model>> {
        if codec.out_proj.weight.val().dtype() != burn::tensor::DType::F32
            || codec.out_proj.weight.val().device() != self.device
        {
            return Err(IrodoriError::Config(
                "captured synthesis requires an FP32 decoder on the inference device".into(),
            ));
        }
        let (request, example) = self.prepare_capture(request)?;
        let graph = capture_tensor_graph(&example, |noise| {
            let mut input = request.clone();
            input.request.initial_noise = Some(noise);
            self.sample_prepared(input)
                .map(|latent| codec.decode_wgsl(latent.cast(burn::tensor::DType::F32)))
        })?;
        Ok(CapturedSynthesis {
            sampling: CapturedSampling {
                graph,
                _request: request,
                _engine: PhantomData,
            },
            _codec: PhantomData,
        })
    }
}
