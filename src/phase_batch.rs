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

use std::{
    collections::HashSet,
    fmt,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use burn::tensor::{Tensor, backend::Backend};

use crate::{
    IrodoriError, Result, SamplingRequest, WgpuRaw, WgslInferenceEngine,
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

    fn validate_request(self, request: &SamplingRequest<WgpuRaw>) -> Result<()> {
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
    pub request: SamplingRequest<WgpuRaw>,
}

impl PlannedSynthesis {
    pub fn new(
        id: BatchItemId,
        voice: VoiceIdentity,
        geometry: OutputGeometry,
        request: SamplingRequest<WgpuRaw>,
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
    pub codec_items: Vec<CodecItemTiming>,
}

struct ResidentLatent {
    id: BatchItemId,
    voice: VoiceIdentity,
    tensor: Tensor<WgpuRaw, 3>,
}

/// RF model and all requests are resident. The codec is intentionally absent.
pub struct RfResident {
    engine: WgslInferenceEngine,
    requests: Vec<PlannedSynthesis>,
}

/// RF model has been released; sampled codec-ready latents remain on GPU.
pub struct LatentsResident {
    device: <WgpuRaw as Backend>::Device,
    latents: Vec<ResidentLatent>,
    rf_phase_wall: Duration,
    rf_items: Vec<RfItemTiming>,
}

/// Codec and sampled latents are resident; the RF model is absent.
pub struct CodecResident {
    device: <WgpuRaw as Backend>::Device,
    codec: DacVaeDecoder<WgpuRaw>,
    latents: Vec<ResidentLatent>,
    rf_phase_wall: Duration,
    rf_items: Vec<RfItemTiming>,
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
                device_complete,
            });
            latents.push(ResidentLatent {
                id: planned.id,
                voice: planned.voice,
                tensor,
            });
        }
        drop(engine);
        <WgpuRaw as Backend>::memory_cleanup(&device);
        sync(&device, "after releasing RF model")?;
        Ok(PhaseBatch {
            state: LatentsResident {
                device,
                latents,
                rf_phase_wall: phase_started.elapsed(),
                rf_items: timings,
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
    pub fn with_codec(self, codec: DacVaeCodec<WgpuRaw>) -> PhaseBatch<CodecResident> {
        self.with_decoder(codec.into_decoder())
    }

    /// Attach a decode-only codec without ever making encoder weights resident.
    pub fn with_decoder(mut self, mut codec: DacVaeDecoder<WgpuRaw>) -> PhaseBatch<CodecResident> {
        codec.prepare_for_wgsl();
        let state = CodecResident {
            device: self.state.device,
            codec,
            latents: std::mem::take(&mut self.state.latents),
            rf_phase_wall: self.state.rf_phase_wall,
            rf_items: std::mem::take(&mut self.state.rf_items),
        };
        PhaseBatch { state }
    }
}

/// GPU audio handed to the caller exactly once.
pub struct BatchAudio {
    pub id: BatchItemId,
    pub voice: VoiceIdentity,
    pub tensor: Tensor<WgpuRaw, 3>,
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
        <WgpuRaw as Backend>::memory_cleanup(&device);
        sync(&device, "after releasing codec")?;
        Ok(PhaseBatch {
            state: Complete {
                metrics: PhaseBatchMetrics {
                    rf_phase_wall,
                    codec_phase_wall: phase_started.elapsed(),
                    rf_items,
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

fn sync(device: &<WgpuRaw as Backend>::Device, stage: &str) -> Result<()> {
    <WgpuRaw as Backend>::sync(device).map_err(|error| {
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
