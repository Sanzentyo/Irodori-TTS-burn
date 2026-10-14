//! Overlapped decoder windows keep the original latent convolution context.

use burn::tensor::{FloatDType, Tensor};
use std::{
    ops::{ControlFlow, Range},
    time::Instant,
};

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
        sink: impl FnMut(&[f32]) -> Result<ControlFlow<()>>,
    ) -> Result<DecodeStreamReport> {
        decode_stream_with(
            latent,
            sample_limit,
            window,
            self.hop_length(),
            |input, crop| {
                self.decode_wgsl(input)
                    .narrow(2, crop.start, crop.len())
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| IrodoriError::Audio(format!("decoder readback failed: {e}")))
            },
            sink,
        )
    }
}

/// One exact window, shared by ordinary execution and captured execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeSlice {
    pub from: usize,
    pub to: usize,
    pub start_sample: usize,
    pub count: usize,
}

pub(super) struct DecodePlan {
    frames: usize,
    hop: usize,
    limit: usize,
    window: DecodeWindow,
    at: usize,
}

impl DecodePlan {
    pub(super) fn new(
        dims: [usize; 3],
        sample_limit: Option<usize>,
        window: DecodeWindow,
        hop: usize,
    ) -> Result<Self> {
        let [batch, frames, dim] = dims;
        if batch != 1
            || frames < 2
            || dim != DACVAE_LATENT_DIM
            || window.first_frames == 0
            || window.subsequent_frames == 0
            || hop == 0
        {
            return Err(IrodoriError::Shape(
                "stream decode needs [1, frames>=2, 32] and positive window lengths".into(),
            ));
        }
        let total = frames
            .checked_mul(hop)
            .ok_or_else(|| IrodoriError::Shape("decoded sample count overflow".into()))?;
        let limit = sample_limit.unwrap_or(total);
        if limit == 0 || limit > total {
            return Err(IrodoriError::Shape(
                "sample limit must be positive and fit the full latent".into(),
            ));
        }
        Ok(Self {
            frames,
            hop,
            limit,
            window,
            at: 0,
        })
    }
}

impl Iterator for DecodePlan {
    type Item = DecodeSlice;

    fn next(&mut self) -> Option<Self::Item> {
        let needed = self.limit.div_ceil(self.hop);
        if self.at >= needed {
            return None;
        }
        let size = if self.at == 0 {
            self.window.first_frames
        } else {
            self.window.subsequent_frames
        };
        let end = self.at.saturating_add(size).min(needed);
        let to = end.saturating_add(10).min(self.frames);
        let from = self.at.saturating_sub(10).min(to.saturating_sub(2));
        let slice = DecodeSlice {
            from,
            to,
            start_sample: (self.at - from) * self.hop,
            count: ((end - self.at) * self.hop).min(self.limit - self.at * self.hop),
        };
        self.at = end;
        Some(slice)
    }
}

pub(super) fn decode_stream_with(
    latent: Tensor<3>,
    sample_limit: Option<usize>,
    window: DecodeWindow,
    hop: usize,
    mut decode: impl FnMut(Tensor<3>, Range<usize>) -> Result<Vec<f32>>,
    mut sink: impl FnMut(&[f32]) -> Result<ControlFlow<()>>,
) -> Result<DecodeStreamReport> {
    let plan = DecodePlan::new(latent.dims(), sample_limit, window, hop)?;
    let started = Instant::now();
    let mut report = DecodeStreamReport {
        samples: 0,
        chunks: 0,
        first_audio_seconds: None,
        cancelled: false,
    };
    for slice in plan {
        let samples = decode(
            latent
                .clone()
                .narrow(1, slice.from, slice.to - slice.from)
                .cast(FloatDType::F32),
            slice.start_sample..slice.start_sample + slice.count,
        )?;
        if samples.len() != slice.count || samples.iter().any(|v| !v.is_finite()) {
            return Err(IrodoriError::Audio(
                "decoder returned invalid audio length or non-finite samples".into(),
            ));
        }
        report
            .first_audio_seconds
            .get_or_insert_with(|| started.elapsed().as_secs_f64());
        report.samples += slice.count;
        report.chunks += 1;
        if sink(&samples)?.is_break() {
            report.cancelled = true;
            break;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_plan_preserves_context_trim_and_coverage() {
        let hop = 1920;
        for frames in [2, 12, 13, 22, 50, 100, 112, 750] {
            for limit in [1, hop + 1, frames * hop - 317, frames * hop] {
                for window in [
                    DecodeWindow::default(),
                    DecodeWindow {
                        first_frames: 1,
                        subsequent_frames: 17,
                    },
                ] {
                    let plan = DecodePlan::new([1, frames, 32], Some(limit), window, hop).unwrap();
                    let mut emitted = 0;
                    for slice in plan {
                        assert!(slice.to <= frames && slice.to - slice.from >= 2);
                        assert_eq!(slice.from * hop + slice.start_sample, emitted);
                        assert!(slice.start_sample + slice.count <= (slice.to - slice.from) * hop);
                        emitted += slice.count;
                        assert_eq!(slice.to, (emitted.div_ceil(hop) + 10).min(frames));
                    }
                    assert_eq!(emitted, limit);
                }
            }
        }
    }

    #[test]
    fn streaming_plan_rejects_invalid_geometry_before_decoding() {
        for dims in [
            [0, 10, 32],
            [2, 10, 32],
            [1, 1, 32],
            [1, 10, 31],
            [1, usize::MAX, 32],
        ] {
            assert!(DecodePlan::new(dims, None, DecodeWindow::default(), 1920).is_err());
        }
        for limit in [0, 19201] {
            assert!(
                DecodePlan::new([1, 10, 32], Some(limit), DecodeWindow::default(), 1920).is_err()
            );
        }
    }
}
