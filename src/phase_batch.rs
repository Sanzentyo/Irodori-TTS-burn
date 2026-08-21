//! VRAM-bounded phase batching for production WGPU inference.
//!
//! An 8 GiB device cannot cheaply keep every RF and codec allocation resident
//! while serving unrelated requests.  This module therefore models offline or
//! throughput-oriented synthesis as explicit type-state transitions:
//!
//! ```text
//! RfResident -> LatentsResident -> CodecResident -> Complete
//! ```
//!
//! The transition out of `RfResident` drops the RF model and runs backend
//! memory cleanup while keeping the small sampled latents GPU-resident.  The
//! codec is attached only after that transition.  This is intentionally a
//! different API from an online low-latency session: callers must choose the
//! scheduling policy rather than accidentally receiving phase-batch latency.

use burn::tensor::{Bool, Device, Int};
use std::{
    collections::HashSet,
    fmt,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use burn::tensor::Tensor;

use crate::{
    IrodoriError, Result, SamplingRequest, WgslInferenceEngine,
    codec::{DacVaeCodec, DacVaeDecoder},
    unpatchify_latent,
};

/// Stable caller-visible identity for one synthesis item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchItemId(Arc<str>);

impl BatchItemId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(IrodoriError::Config(
                "batch item id must not be empty".to_string(),
            ));
        }
        Ok(Self(Arc::from(trimmed)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BatchItemId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Stable cache key for a prepared clone source or a designed voice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpeakerKey(Arc<str>);

impl SpeakerKey {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(IrodoriError::Config(
                "speaker key must not be empty".to_string(),
            ));
        }
        Ok(Self(Arc::from(trimmed)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SpeakerKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Semantic voice choice attached to a request and its timing record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VoiceIdentity {
    Unconditioned,
    Clone(SpeakerKey),
    Designed(SpeakerKey),
}

/// Validated geometry required to turn a patched RF result into codec input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGeometry {
    latent_frames: NonZeroUsize,
    patch_size: NonZeroUsize,
    latent_dim: NonZeroUsize,
}

impl OutputGeometry {
    pub fn new(latent_frames: usize, patch_size: usize, latent_dim: usize) -> Result<Self> {
        let latent_frames = NonZeroUsize::new(latent_frames).ok_or_else(|| {
            IrodoriError::Config("output latent_frames must be greater than zero".to_string())
        })?;
        let patch_size = NonZeroUsize::new(patch_size).ok_or_else(|| {
            IrodoriError::Config("output patch_size must be greater than zero".to_string())
        })?;
        let latent_dim = NonZeroUsize::new(latent_dim).ok_or_else(|| {
            IrodoriError::Config("output latent_dim must be greater than zero".to_string())
        })?;
        Ok(Self {
            latent_frames,
            patch_size,
            latent_dim,
        })
    }

    pub fn latent_frames(self) -> usize {
        self.latent_frames.get()
    }

    pub fn patch_size(self) -> usize {
        self.patch_size.get()
    }

    pub fn latent_dim(self) -> usize {
        self.latent_dim.get()
    }

    pub fn patched_frames(self) -> usize {
        self.latent_frames().div_ceil(self.patch_size())
    }

    fn validate_request(self, request: &SamplingRequest) -> Result<()> {
        if request.text_ids.dims()[0] != 1 {
            return Err(IrodoriError::Shape(
                "phase-batch items currently require batch size one; use one PlannedSynthesis per voice/request"
                    .to_string(),
            ));
        }
        if request.sequence_length != self.patched_frames() {
            return Err(IrodoriError::Shape(format!(
                "sampling request sequence_length={} does not match output geometry patched_frames={} (latent_frames={}, patch_size={})",
                request.sequence_length,
                self.patched_frames(),
                self.latent_frames(),
                self.patch_size(),
            )));
        }
        Ok(())
    }
}

/// One request prepared for the RF phase.
#[derive(Debug, Clone)]
pub struct PlannedSynthesis {
    pub id: BatchItemId,
    pub voice: VoiceIdentity,
    pub geometry: OutputGeometry,
    pub request: SamplingRequest,
}

impl PlannedSynthesis {
    pub fn new(
        id: BatchItemId,
        voice: VoiceIdentity,
        geometry: OutputGeometry,
        request: SamplingRequest,
    ) -> Result<Self> {
        geometry.validate_request(&request)?;
        Ok(Self {
            id,
            voice,
            geometry,
            request,
        })
    }
}

/// Device-complete RF timing for one item. No latent readback is performed.
#[derive(Debug, Clone)]
pub struct RfItemTiming {
    pub id: BatchItemId,
    pub voice: VoiceIdentity,
    /// All items in one tensor batch become device-complete together. This is
    /// one for [`PhaseBatch::sample_all`] and greater than one only for the
    /// explicit homogeneous tensor-batch API.
    pub tensor_batch_size: NonZeroUsize,
    pub device_complete: Duration,
}

/// Device-complete timing for one actual RF tensor batch.
#[derive(Debug, Clone)]
pub struct RfBatchTiming {
    pub ids: Vec<BatchItemId>,
    pub device_complete: Duration,
}

/// Device-complete and consumer-complete codec timing for one item.
#[derive(Debug, Clone)]
pub struct CodecItemTiming {
    pub id: BatchItemId,
    pub voice: VoiceIdentity,
    pub device_complete: Duration,
    /// Stops after the caller consumes the GPU audio. A WAV consumer normally
    /// performs its owned CPU readback inside this interval.
    pub consumer_complete: Duration,
}

/// Aggregate timings retained after all type-state transitions complete.
#[derive(Debug, Clone)]
pub struct PhaseBatchMetrics {
    pub rf_phase_wall: Duration,
    pub codec_phase_wall: Duration,
    pub rf_items: Vec<RfItemTiming>,
    pub rf_batches: Vec<RfBatchTiming>,
    pub codec_items: Vec<CodecItemTiming>,
}

struct ResidentLatent {
    id: BatchItemId,
    voice: VoiceIdentity,
    tensor: Tensor<3>,
}

/// RF model and all requests are resident. The codec is intentionally absent.
pub struct RfResident {
    engine: WgslInferenceEngine,
    requests: Vec<PlannedSynthesis>,
}

/// RF model has been released; sampled codec-ready latents remain on GPU.
pub struct LatentsResident {
    device: Device,
    latents: Vec<ResidentLatent>,
    rf_phase_wall: Duration,
    rf_items: Vec<RfItemTiming>,
    rf_batches: Vec<RfBatchTiming>,
}

/// Codec and sampled latents are resident; the RF model is absent.
pub struct CodecResident {
    device: Device,
    codec: DacVaeDecoder,
    latents: Vec<ResidentLatent>,
    rf_phase_wall: Duration,
    rf_items: Vec<RfItemTiming>,
    rf_batches: Vec<RfBatchTiming>,
}

/// Terminal state after every audio has been consumed.
pub struct Complete {
    metrics: PhaseBatchMetrics,
}

/// Type-state wrapper for the VRAM-bounded phase-batch scheduler.
pub struct PhaseBatch<State> {
    state: State,
}

impl PhaseBatch<RfResident> {
    pub fn new(engine: WgslInferenceEngine, requests: Vec<PlannedSynthesis>) -> Result<Self> {
        validate_unique_items(&requests)?;
        Ok(Self {
            state: RfResident { engine, requests },
        })
    }

    /// Sample every request while the RF model is loaded, then release the
    /// model without reading the resulting latents back to the CPU.
    pub fn sample_all(self) -> Result<PhaseBatch<LatentsResident>> {
        let RfResident { engine, requests } = self.state;
        let device = engine.device().clone();
        let phase_started = Instant::now();
        let mut latents = Vec::with_capacity(requests.len());
        let mut timings = Vec::with_capacity(requests.len());
        let mut batch_timings = Vec::with_capacity(requests.len());
        for planned in requests {
            sync(&device, "before RF item")?;
            let started = Instant::now();
            let patched = engine.sample(planned.request)?;
            sync(&device, "after RF item")?;
            let device_complete = started.elapsed();
            let unpatched = unpatchify_latent(
                patched,
                planned.geometry.patch_size(),
                planned.geometry.latent_dim(),
            );
            let available = unpatched.dims()[1];
            if available < planned.geometry.latent_frames() {
                return Err(IrodoriError::Shape(format!(
                    "RF output for {} has {available} unpatched frames, expected at least {}",
                    planned.id,
                    planned.geometry.latent_frames(),
                )));
            }
            let tensor = if available == planned.geometry.latent_frames() {
                unpatched
            } else {
                unpatched.slice([
                    0..1,
                    0..planned.geometry.latent_frames(),
                    0..planned.geometry.latent_dim(),
                ])
            };
            timings.push(RfItemTiming {
                id: planned.id.clone(),
                voice: planned.voice.clone(),
                tensor_batch_size: NonZeroUsize::MIN,
                device_complete,
            });
            batch_timings.push(RfBatchTiming {
                ids: vec![planned.id.clone()],
                device_complete,
            });
            latents.push(ResidentLatent {
                id: planned.id,
                voice: planned.voice,
                tensor,
            });
        }
        drop(engine);
        device.memory_cleanup();
        sync(&device, "after releasing RF model")?;
        Ok(PhaseBatch {
            state: LatentsResident {
                device,
                latents,
                rf_phase_wall: phase_started.elapsed(),
                rf_items: timings,
                rf_batches: batch_timings,
            },
        })
    }

    /// Sample all requests as one true RF tensor batch.
    ///
    /// This path is deliberately fail-closed: every request must have the
    /// same output geometry and the same optional-conditioning topology and
    /// tensor shapes. Values may differ, so multiple speakers are valid when
    /// their prepared reference geometry is identical. The model executes
    /// four Euler evaluations for the whole batch; it does not loop over
    /// requests internally.
    pub fn sample_homogeneous_tensor_batch(self) -> Result<PhaseBatch<LatentsResident>> {
        let RfResident { engine, requests } = self.state;
        validate_homogeneous_requests(&requests)?;
        let device = engine.device().clone();
        let phase_started = Instant::now();
        let batch_size = NonZeroUsize::new(requests.len()).expect("validated non-empty batch");
        let request = concatenate_requests(&requests);

        sync(&device, "before homogeneous RF tensor batch")?;
        let started = Instant::now();
        let patched = engine.sample(request)?;
        sync(&device, "after homogeneous RF tensor batch")?;
        let device_complete = started.elapsed();

        let geometry = requests[0].geometry;
        let unpatched = unpatchify_latent(patched, geometry.patch_size(), geometry.latent_dim());
        let [actual_batch, available, actual_dim] = unpatched.dims();
        if actual_batch != requests.len()
            || available < geometry.latent_frames()
            || actual_dim != geometry.latent_dim()
        {
            return Err(IrodoriError::Shape(format!(
                "homogeneous RF output [{actual_batch}, {available}, {actual_dim}] does not satisfy expected [{}, >= {}, {}]",
                requests.len(),
                geometry.latent_frames(),
                geometry.latent_dim(),
            )));
        }

        let ids = requests.iter().map(|planned| planned.id.clone()).collect();
        let mut timings = Vec::with_capacity(requests.len());
        let mut latents = Vec::with_capacity(requests.len());
        for (batch_index, planned) in requests.into_iter().enumerate() {
            let tensor = unpatched.clone().slice([
                batch_index..batch_index + 1,
                0..geometry.latent_frames(),
                0..geometry.latent_dim(),
            ]);
            timings.push(RfItemTiming {
                id: planned.id.clone(),
                voice: planned.voice.clone(),
                tensor_batch_size: batch_size,
                device_complete,
            });
            latents.push(ResidentLatent {
                id: planned.id,
                voice: planned.voice,
                tensor,
            });
        }

        drop(engine);
        device.memory_cleanup();
        sync(&device, "after releasing tensor-batched RF model")?;
        Ok(PhaseBatch {
            state: LatentsResident {
                device,
                latents,
                rf_phase_wall: phase_started.elapsed(),
                rf_items: timings,
                rf_batches: vec![RfBatchTiming {
                    ids,
                    device_complete,
                }],
            },
        })
    }
}

impl PhaseBatch<LatentsResident> {
    pub fn len(&self) -> usize {
        self.state.latents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.latents.is_empty()
    }

    /// Attach a codec only after the RF model has been released.
    pub fn with_codec(self, codec: DacVaeCodec) -> PhaseBatch<CodecResident> {
        self.with_decoder(codec.into_decoder())
    }

    /// Attach a decode-only codec without ever making encoder weights resident.
    pub fn with_decoder(mut self, mut codec: DacVaeDecoder) -> PhaseBatch<CodecResident> {
        codec.prepare_for_wgsl();
        let state = CodecResident {
            device: self.state.device,
            codec,
            latents: std::mem::take(&mut self.state.latents),
            rf_phase_wall: self.state.rf_phase_wall,
            rf_items: std::mem::take(&mut self.state.rf_items),
            rf_batches: std::mem::take(&mut self.state.rf_batches),
        };
        PhaseBatch { state }
    }
}

/// GPU audio handed to the caller exactly once.
pub struct BatchAudio {
    pub id: BatchItemId,
    pub voice: VoiceIdentity,
    pub tensor: Tensor<3>,
}

impl PhaseBatch<CodecResident> {
    /// Decode every latent while the codec is loaded. The consumer runs after
    /// device completion and owns the decision to perform a CPU readback.
    pub fn decode_all<F>(self, mut consume: F) -> Result<PhaseBatch<Complete>>
    where
        F: FnMut(BatchAudio) -> Result<()>,
    {
        let CodecResident {
            device,
            codec,
            latents,
            rf_phase_wall,
            rf_items,
            rf_batches,
        } = self.state;
        let phase_started = Instant::now();
        let mut codec_items = Vec::with_capacity(latents.len());
        for latent in latents {
            sync(&device, "before codec item")?;
            let started = Instant::now();
            let audio = codec.decode_wgsl(latent.tensor);
            sync(&device, "after codec item")?;
            let device_complete = started.elapsed();
            consume(BatchAudio {
                id: latent.id.clone(),
                voice: latent.voice.clone(),
                tensor: audio,
            })?;
            sync(&device, "after codec consumer")?;
            codec_items.push(CodecItemTiming {
                id: latent.id,
                voice: latent.voice,
                device_complete,
                consumer_complete: started.elapsed(),
            });
        }
        drop(codec);
        device.memory_cleanup();
        sync(&device, "after releasing codec")?;
        Ok(PhaseBatch {
            state: Complete {
                metrics: PhaseBatchMetrics {
                    rf_phase_wall,
                    codec_phase_wall: phase_started.elapsed(),
                    rf_items,
                    rf_batches,
                    codec_items,
                },
            },
        })
    }
}

impl PhaseBatch<Complete> {
    pub fn metrics(&self) -> &PhaseBatchMetrics {
        &self.state.metrics
    }

    pub fn into_metrics(self) -> PhaseBatchMetrics {
        self.state.metrics
    }
}

fn validate_unique_items(requests: &[PlannedSynthesis]) -> Result<()> {
    if requests.is_empty() {
        return Err(IrodoriError::Config(
            "phase batch must contain at least one request".to_string(),
        ));
    }
    let mut seen = HashSet::with_capacity(requests.len());
    for request in requests {
        if !seen.insert(request.id.clone()) {
            return Err(IrodoriError::Config(format!(
                "duplicate phase-batch item id: {}",
                request.id
            )));
        }
    }
    Ok(())
}

fn validate_homogeneous_requests(requests: &[PlannedSynthesis]) -> Result<()> {
    let first = requests.first().ok_or_else(|| {
        IrodoriError::Config("phase batch must contain at least one request".to_string())
    })?;
    let expected = request_shape_signature(&first.request);
    for request in &requests[1..] {
        if request.geometry != first.geometry {
            return Err(IrodoriError::Shape(format!(
                "homogeneous tensor batch requires one output geometry; {} differs from {}",
                request.id, first.id
            )));
        }
        let actual = request_shape_signature(&request.request);
        if actual != expected {
            return Err(IrodoriError::Shape(format!(
                "homogeneous tensor batch request {} has topology/shapes {actual:?}, expected {expected:?}",
                request.id
            )));
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct RequestShapeSignature {
    sequence_length: usize,
    text: [usize; 2],
    text_mask: [usize; 2],
    reference: Option<[usize; 3]>,
    reference_mask: Option<[usize; 2]>,
    caption: Option<[usize; 2]>,
    caption_mask: Option<[usize; 2]>,
    noise: Option<[usize; 3]>,
}

fn request_shape_signature(request: &SamplingRequest) -> RequestShapeSignature {
    RequestShapeSignature {
        sequence_length: request.sequence_length,
        text: request.text_ids.dims(),
        text_mask: request.text_mask.dims(),
        reference: request.ref_latent.as_ref().map(Tensor::dims),
        reference_mask: request.ref_mask.as_ref().map(Tensor::dims),
        caption: request.caption_ids.as_ref().map(Tensor::dims),
        caption_mask: request.caption_mask.as_ref().map(Tensor::dims),
        noise: request.initial_noise.as_ref().map(Tensor::dims),
    }
}

fn concatenate_requests(requests: &[PlannedSynthesis]) -> SamplingRequest {
    SamplingRequest {
        text_ids: Tensor::cat(
            requests
                .iter()
                .map(|planned| planned.request.text_ids.clone())
                .collect(),
            0,
        ),
        text_mask: Tensor::cat(
            requests
                .iter()
                .map(|planned| planned.request.text_mask.clone())
                .collect(),
            0,
        ),
        ref_latent: concatenate_optional_3(requests, |request| &request.ref_latent),
        ref_mask: concatenate_optional_bool_2(requests, |request| &request.ref_mask),
        sequence_length: requests[0].request.sequence_length,
        caption_ids: concatenate_optional_int_2(requests, |request| &request.caption_ids),
        caption_mask: concatenate_optional_bool_2(requests, |request| &request.caption_mask),
        initial_noise: concatenate_optional_3(requests, |request| &request.initial_noise),
    }
}

fn concatenate_optional_int_2<K>(requests: &[PlannedSynthesis], select: K) -> Option<Tensor<2, Int>>
where
    K: Fn(&SamplingRequest) -> &Option<Tensor<2, Int>>,
{
    let tensors = requests
        .iter()
        .map(|planned| select(&planned.request).as_ref().cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(Tensor::cat(tensors, 0))
}

fn concatenate_optional_bool_2<K>(
    requests: &[PlannedSynthesis],
    select: K,
) -> Option<Tensor<2, Bool>>
where
    K: Fn(&SamplingRequest) -> &Option<Tensor<2, Bool>>,
{
    let tensors = requests
        .iter()
        .map(|planned| select(&planned.request).as_ref().cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(Tensor::cat(tensors, 0))
}

fn concatenate_optional_3<K>(requests: &[PlannedSynthesis], select: K) -> Option<Tensor<3>>
where
    K: Fn(&SamplingRequest) -> &Option<Tensor<3>>,
{
    let tensors = requests
        .iter()
        .map(|planned| select(&planned.request).as_ref().cloned())
        .collect::<Option<Vec<_>>>()?;
    Some(Tensor::cat(tensors, 0))
}

fn sync(device: &Device, stage: &str) -> Result<()> {
    device.sync().map_err(|error| {
        IrodoriError::Config(format!("WGPU synchronization failed {stage}: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_trimmed_and_non_empty() {
        assert_eq!(BatchItemId::new("  item-1  ").unwrap().as_str(), "item-1");
        assert!(BatchItemId::new("  ").is_err());
        assert_eq!(
            SpeakerKey::new("  speaker-a  ").unwrap().as_str(),
            "speaker-a"
        );
        assert!(SpeakerKey::new("").is_err());
    }

    #[test]
    fn geometry_rejects_zero_and_rounds_patched_frames_up() {
        assert!(OutputGeometry::new(0, 1, 32).is_err());
        assert!(OutputGeometry::new(5, 0, 32).is_err());
        assert!(OutputGeometry::new(5, 2, 0).is_err());
        let geometry = OutputGeometry::new(5, 2, 32).unwrap();
        assert_eq!(geometry.patched_frames(), 3);
        assert_eq!(geometry.latent_frames(), 5);
        assert_eq!(geometry.latent_dim(), 32);
    }

    #[test]
    fn voice_identity_is_an_explicit_sum_type() {
        let clone = VoiceIdentity::Clone(SpeakerKey::new("alice").unwrap());
        let designed = VoiceIdentity::Designed(SpeakerKey::new("calm-low").unwrap());
        assert_ne!(clone, designed);
        assert_ne!(clone, VoiceIdentity::Unconditioned);
    }
}
