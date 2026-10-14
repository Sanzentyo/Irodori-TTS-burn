//! Opt-in graph reuse for the exact windows emitted by the ordinary decoder.
//! Entry count and input geometry are bounded; these are not a measured VRAM
//! byte budget. Capture failures are returned, never silently rerouted.

use super::{
    DacVaeDecoder, DecodeStreamReport, DecodeWindow,
    streaming::{DecodePlan, decode_stream_with},
};
use crate::{
    IrodoriError, Result,
    graph::{CapturedTensorGraph, capture_tensor_graph_with_initial_output},
};
use burn::tensor::Tensor;
use std::{
    collections::VecDeque,
    num::NonZeroUsize,
    ops::{ControlFlow, Range},
};

#[derive(Clone, Copy, Debug)]
pub struct DecoderGraphLimits {
    entries: NonZeroUsize,
    input_frames: NonZeroUsize,
}

impl DecoderGraphLimits {
    pub fn new(entries: NonZeroUsize, input_frames: NonZeroUsize) -> Result<Self> {
        if input_frames.get() < 2 {
            return Err(IrodoriError::Config(
                "decoder graphs require at least two input frames".into(),
            ));
        }
        Ok(Self {
            entries,
            input_frames,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowKey {
    frames: usize,
    start: usize,
    count: usize,
}

/// A single owner contains both ordering and values; no parallel index can
/// retain an evicted GPU graph. Small bounded caches use linear lookup.
struct WindowCache<T> {
    entries: VecDeque<(WindowKey, T)>,
    limit: NonZeroUsize,
}

impl<T> WindowCache<T> {
    fn touch(&mut self, key: WindowKey) -> bool {
        let Some(index) = self.entries.iter().position(|(stored, _)| *stored == key) else {
            return false;
        };
        let entry = self
            .entries
            .remove(index)
            .expect("position is within the deque");
        self.entries.push_back(entry);
        true
    }

    fn make_room(&mut self) {
        if self.entries.len() == self.limit.get() {
            self.entries.pop_front();
        }
    }
}

/// Owns immutable decoder weights and a bounded set of process-local graphs.
/// Dropping/evicting a graph releases its captured bindings before the weights.
pub struct StreamingDacVaeDecoder {
    cache: WindowCache<CapturedTensorGraph>,
    generation: u32,
    limits: DecoderGraphLimits,
    decoder: DacVaeDecoder,
}

impl StreamingDacVaeDecoder {
    /// The decoder must have completed `prepare_for_wgsl` before this transition.
    pub fn new(decoder: DacVaeDecoder, limits: DecoderGraphLimits) -> Self {
        Self {
            cache: WindowCache {
                entries: VecDeque::new(),
                limit: limits.entries,
            },
            generation: cubecl_environment::environment::generation(),
            limits,
            decoder,
        }
    }

    pub fn cached_graphs(&self) -> usize {
        self.cache.entries.len()
    }

    pub fn decode_stream(
        &mut self,
        latent: Tensor<3>,
        sample_limit: Option<usize>,
        window: DecodeWindow,
        sink: impl FnMut(&[f32]) -> Result<ControlFlow<()>>,
    ) -> Result<DecodeStreamReport> {
        let hop = self.decoder.hop_length();
        let context = self.decoder.context_frames();
        // Reject an oversized request before any partial audio is emitted.
        for slice in DecodePlan::new(latent.dims(), sample_limit, window, hop, context)? {
            if slice.to - slice.from > self.limits.input_frames.get() {
                return Err(IrodoriError::Shape(
                    "decoder window exceeds graph input-frame limit".into(),
                ));
            }
        }
        if self.generation != cubecl_environment::environment::generation() {
            self.cache.entries.clear();
            self.generation = cubecl_environment::environment::generation();
        }
        decode_stream_with(
            latent,
            sample_limit,
            window,
            hop,
            context,
            |input, crop| self.decode_window(input, crop),
            sink,
        )
    }

    fn decode_window(&mut self, input: Tensor<3>, crop: Range<usize>) -> Result<Vec<f32>> {
        let key = WindowKey {
            frames: input.dims()[1],
            start: crop.start,
            count: crop.len(),
        };
        if !self.cache.touch(key) {
            // Evict before capture so the old graph does not inflate the new
            // capture's allocation peak. A failed capture leaves a valid cache.
            self.cache.make_room();
            let (graph, initial_output) =
                capture_tensor_graph_with_initial_output(&input, |input| {
                    Ok(self
                        .decoder
                        .decode_wgsl(input)
                        .narrow(2, key.start, key.count))
                })?;
            self.cache.entries.push_back((key, graph));
            return Ok(initial_output);
        }
        self.cache
            .entries
            .back_mut()
            .expect("the requested graph is most recent")
            .1
            .decode_to_cpu_f32(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_windows_include_the_output_crop_and_evict_least_recently_used() {
        let key = WindowKey {
            frames: 22,
            start: 0,
            count: 23040,
        };
        let trimmed = WindowKey {
            count: 23039,
            ..key
        };
        let mut cache = WindowCache {
            entries: VecDeque::new(),
            limit: NonZeroUsize::new(2).unwrap(),
        };
        cache.entries.push_back((key, 1));
        assert!(!cache.touch(trimmed));
        cache.entries.push_back((trimmed, 2));
        assert!(cache.touch(key));
        cache.make_room();
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries[0], (key, 1));
        assert!(!cache.touch(trimmed));
    }

    #[test]
    fn graph_limits_reject_one_frame() {
        assert!(DecoderGraphLimits::new(NonZeroUsize::MIN, NonZeroUsize::MIN).is_err());
        assert!(
            DecoderGraphLimits::new(NonZeroUsize::MIN, NonZeroUsize::new(116).unwrap()).is_ok()
        );
    }
}
