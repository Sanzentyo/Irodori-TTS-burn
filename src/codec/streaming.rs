//! Overlapped decoder windows keep the original latent convolution context.

use burn::tensor::{FloatDType, Tensor};
use std::{ops::ControlFlow, time::Instant};

use super::{DACVAE_LATENT_DIM, DacVaeDecoder};
use crate::{IrodoriError, Result};

/// Geometry of emitted chunks, in latent frames (one frame is 40 ms).
#[derive(Clone, Copy, Debug)]
pub struct DecodeWindow {
    pub first_frames: usize,
    pub subsequent_frames: usize,
}

impl Default for DecodeWindow {
    fn default() -> Self {
        Self {
            first_frames: 12,
            subsequent_frames: 96,
        }
    }
}

#[derive(Debug)]
pub struct DecodeStreamReport {
    pub samples: usize,
    pub chunks: usize,
    pub first_audio_seconds: Option<f64>,
    pub cancelled: bool,
}

impl DacVaeDecoder {
    /// Decode with ten latent frames of context on each side and emit only
    /// each window's interior. The original latent's tail remains context even
    /// when `sample_limit` trims the returned waveform.
    pub fn decode_stream(
        &self,
        latent: Tensor<3>,
        sample_limit: Option<usize>,
        window: DecodeWindow,
        mut sink: impl FnMut(&[f32]) -> Result<ControlFlow<()>>,
    ) -> Result<DecodeStreamReport> {
        let [batch, frames, dim] = latent.dims();
        if batch != 1
            || frames < 2
            || dim != DACVAE_LATENT_DIM
            || window.first_frames == 0
            || window.subsequent_frames == 0
        {
            return Err(IrodoriError::Shape(
                "stream decode needs [1, frames>=2, 32] and positive window lengths".into(),
            ));
        }
        let total = frames
            .checked_mul(self.hop_length())
            .ok_or_else(|| IrodoriError::Shape("decoded sample count overflow".into()))?;
        let limit = sample_limit.unwrap_or(total);
        if limit == 0 || limit > total {
            return Err(IrodoriError::Shape(
                "sample limit must be positive and fit the full latent".into(),
            ));
        }
        let started = Instant::now();
        let mut report = DecodeStreamReport {
            samples: 0,
            chunks: 0,
            first_audio_seconds: None,
            cancelled: false,
        };
        let mut at = 0;
        let needed = limit.div_ceil(self.hop_length());
        while at < needed {
            let size = if at == 0 {
                window.first_frames
            } else {
                window.subsequent_frames
            };
            let end = at.saturating_add(size).min(needed);
            let to = end.saturating_add(10).min(frames);
            let from = at.saturating_sub(10).min(to.saturating_sub(2));
            let decoded = self.decode_wgsl(
                latent
                    .clone()
                    .narrow(1, from, to - from)
                    .cast(FloatDType::F32),
            );
            let start_sample = (at - from) * self.hop_length();
            let count = ((end - at) * self.hop_length()).min(limit - report.samples);
            let samples = decoded
                .narrow(2, start_sample, count)
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| IrodoriError::Audio(format!("decoder readback failed: {e}")))?;
            if samples.iter().any(|v| !v.is_finite()) {
                return Err(IrodoriError::Audio(
                    "decoder returned non-finite audio".into(),
                ));
            }
            report
                .first_audio_seconds
                .get_or_insert_with(|| started.elapsed().as_secs_f64());
            report.samples += count;
            report.chunks += 1;
            if sink(&samples)?.is_break() {
                report.cancelled = true;
                break;
            }
            at = end;
        }
        Ok(report)
    }
}
