//! Profiling adapters for the production codec graph.
//!
//! The device profiler records CubeCL stream timestamps without synchronizing
//! after every operator.  The synchronized profiler is retained as a portable
//! diagnostic fallback and to make the measurement boundary explicit.

use std::time::{Duration, Instant};

use cubecl::{
    client::ComputeClient,
    future,
    profile::{ProfileDuration, TimingMethod},
};

use crate::{IrodoriError, WgpuRaw, error::Result};

type WgpuRuntime = burn::backend::wgpu::WgpuRuntime;

/// Origin of a codec-stage duration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecTimingSource {
    /// Hardware timestamps recorded on the active CubeCL stream.
    DeviceTimestamp,
    /// Host wall time around an operation and an explicit device sync.
    SynchronizedSystemClock,
}

/// One resolved codec-stage measurement.
#[derive(Clone, Copy, Debug)]
pub struct CodecStageTiming {
    pub label: &'static str,
    pub duration: Duration,
    pub source: CodecTimingSource,
}

pub(crate) trait CodecStageProfiler {
    type Error;

    fn profile<T, O>(
        &mut self,
        label: &'static str,
        operation: O,
    ) -> std::result::Result<T, Self::Error>
    where
        T: Send + 'static,
        O: FnOnce() -> T + Send;
}

pub(crate) struct NoopCodecStageProfiler;

impl CodecStageProfiler for NoopCodecStageProfiler {
    type Error = std::convert::Infallible;

    fn profile<T, O>(
        &mut self,
        _label: &'static str,
        operation: O,
    ) -> std::result::Result<T, Self::Error>
    where
        T: Send + 'static,
        O: FnOnce() -> T + Send,
    {
        Ok(operation())
    }
}

pub(crate) struct SynchronizedCodecStageProfiler<'a, E, S>
where
    S: FnMut(&'static str) -> std::result::Result<(), E>,
{
    synchronize: &'a mut S,
    timings: Vec<CodecStageTiming>,
}

impl<'a, E, S> SynchronizedCodecStageProfiler<'a, E, S>
where
    S: FnMut(&'static str) -> std::result::Result<(), E>,
{
    pub(crate) fn new(synchronize: &'a mut S) -> Self {
        Self {
            synchronize,
            timings: Vec::with_capacity(48),
        }
    }

    pub(crate) fn finish(self) -> Vec<CodecStageTiming> {
        self.timings
    }
}

impl<E, S> CodecStageProfiler for SynchronizedCodecStageProfiler<'_, E, S>
where
    S: FnMut(&'static str) -> std::result::Result<(), E>,
{
    type Error = E;

    fn profile<T, O>(&mut self, label: &'static str, operation: O) -> std::result::Result<T, E>
    where
        T: Send + 'static,
        O: FnOnce() -> T + Send,
    {
        let started = Instant::now();
        let output = operation();
        (self.synchronize)(label)?;
        self.timings.push(CodecStageTiming {
            label,
            duration: started.elapsed(),
            source: CodecTimingSource::SynchronizedSystemClock,
        });
        Ok(output)
    }
}

struct PendingDeviceTiming {
    label: &'static str,
    duration: ProfileDuration,
}

pub(crate) struct DeviceCodecStageProfiler {
    client: ComputeClient<WgpuRuntime>,
    pending: Vec<PendingDeviceTiming>,
}

impl DeviceCodecStageProfiler {
    pub(crate) fn from_tensor(tensor: &burn::prelude::Tensor<3>) -> Result<Self> {
        let primitive = tensor
            .clone()
            .try_into_primitive::<WgpuRaw>()
            .map_err(|_| IrodoriError::Profile("device profiling requires WGPU tensors".into()))?;
        Ok(Self {
            client: primitive.client,
            pending: Vec::with_capacity(48),
        })
    }

    pub(crate) fn finish(self) -> Result<Vec<CodecStageTiming>> {
        self.pending
            .into_iter()
            .map(|pending| {
                let source = match pending.duration.timing_method() {
                    TimingMethod::Device => CodecTimingSource::DeviceTimestamp,
                    TimingMethod::System => CodecTimingSource::SynchronizedSystemClock,
                };
                let ticks = future::block_on(pending.duration.resolve());
                Ok(CodecStageTiming {
                    label: pending.label,
                    duration: ticks.duration(),
                    source,
                })
            })
            .collect()
    }
}

impl CodecStageProfiler for DeviceCodecStageProfiler {
    type Error = IrodoriError;

    fn profile<T, O>(&mut self, label: &'static str, operation: O) -> Result<T>
    where
        T: Send + 'static,
        O: FnOnce() -> T + Send,
    {
        let (output, duration) = self
            .client
            .profile(operation, label)
            .map_err(|error| IrodoriError::Profile(format!("{label}: {error}")))?;
        self.pending.push(PendingDeviceTiming { label, duration });
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::{
        CodecStageProfiler, CodecTimingSource, NoopCodecStageProfiler,
        SynchronizedCodecStageProfiler,
    };

    #[test]
    fn noop_profiler_executes_once_without_a_receipt() {
        let mut profiler = NoopCodecStageProfiler;
        let value = profiler.profile("stage", || 42).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn synchronized_profiler_labels_its_clock_source() {
        let mut labels = Vec::new();
        let mut synchronize = |label| -> Result<(), Infallible> {
            labels.push(label);
            Ok(())
        };
        let mut profiler = SynchronizedCodecStageProfiler::new(&mut synchronize);
        let value = profiler.profile("codec_stage", || 7).unwrap();
        let timings = profiler.finish();

        assert_eq!(value, 7);
        assert_eq!(labels, ["codec_stage"]);
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].label, "codec_stage");
        assert_eq!(
            timings[0].source,
            CodecTimingSource::SynchronizedSystemClock
        );
    }
}
