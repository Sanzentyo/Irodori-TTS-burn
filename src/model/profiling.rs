//! Deferred GPU-timestamp profiling for the RF graph.
//!
//! Stage closures enqueue normally. Timestamp futures are resolved only after
//! the complete RF request, avoiding the per-stage synchronization that used
//! to distort dispatch batching and temporary live ranges.

use std::{cell::RefCell, collections::BTreeMap, time::Duration};

use burn::tensor::Tensor;
use cubecl::{
    future,
    profile::{ProfileDuration, TimingMethod},
};
use serde::{Deserialize, Serialize};

use crate::{IrodoriError, Result, WgpuRaw};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RfTimingSource {
    DeviceTimestamp,
    SynchronizedSystemClock,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RfDeviceStageTiming {
    pub ordinal: usize,
    pub component: String,
    pub stage: String,
    pub batch: usize,
    pub sequence: usize,
    pub duration_ns: u64,
    pub source: RfTimingSource,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RfDeviceStageAggregate {
    pub component: String,
    pub stage: String,
    pub calls: usize,
    pub total_ns: u64,
    pub mean_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RfDeviceProfileReceipt {
    pub schema_version: u32,
    pub timing_boundary: String,
    pub stages: Vec<RfDeviceStageTiming>,
    pub aggregates: Vec<RfDeviceStageAggregate>,
    pub summed_stage_ns: u64,
}

struct PendingStage {
    component: &'static str,
    stage: &'static str,
    batch: usize,
    sequence: usize,
    duration: ProfileDuration,
}

thread_local! {
    static PENDING: RefCell<Option<Vec<PendingStage>>> = const { RefCell::new(None) };
}

pub fn begin_rf_device_profile() -> Result<()> {
    PENDING.with(|pending| {
        let mut pending = pending.borrow_mut();
        if pending.is_some() {
            return Err(IrodoriError::Profile(
                "an RF device-timestamp profile is already active".to_owned(),
            ));
        }
        *pending = Some(Vec::with_capacity(40 * 12 * 16));
        Ok(())
    })
}

pub fn rf_device_profile_active() -> bool {
    PENDING.with(|pending| pending.borrow().is_some())
}

pub(crate) fn profile_rf_stage<const D: usize, T, O>(
    component: &'static str,
    stage: &'static str,
    batch: usize,
    sequence: usize,
    reference: &Tensor<D>,
    operation: O,
) -> T
where
    T: Send + 'static,
    O: FnOnce() -> T + Send,
{
    if !rf_device_profile_active() {
        return operation();
    }
    let primitive = reference
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .expect("RF device profiling requires WGPU raw tensors");
    let (output, duration) = primitive
        .client
        .profile(operation, stage)
        .unwrap_or_else(|error| {
            panic!("RF device profiling failed for {component}/{stage}: {error}")
        });
    PENDING.with(|pending| {
        pending
            .borrow_mut()
            .as_mut()
            .expect("RF device profile must remain active")
            .push(PendingStage {
                component,
                stage,
                batch,
                sequence,
                duration,
            });
    });
    output
}

pub fn finish_rf_device_profile() -> Result<RfDeviceProfileReceipt> {
    let pending = PENDING
        .with(|pending| pending.borrow_mut().take())
        .ok_or_else(|| {
            IrodoriError::Profile("no RF device-timestamp profile is active".to_owned())
        })?;
    let mut stages = Vec::with_capacity(pending.len());
    for (ordinal, pending) in pending.into_iter().enumerate() {
        let source = match pending.duration.timing_method() {
            TimingMethod::Device => RfTimingSource::DeviceTimestamp,
            TimingMethod::System => RfTimingSource::SynchronizedSystemClock,
        };
        let duration = future::block_on(pending.duration.resolve()).duration();
        stages.push(RfDeviceStageTiming {
            ordinal,
            component: pending.component.to_owned(),
            stage: pending.stage.to_owned(),
            batch: pending.batch,
            sequence: pending.sequence,
            duration_ns: duration_to_ns(duration)?,
            source,
        });
    }
    let mut grouped = BTreeMap::<(String, String), (usize, u64)>::new();
    for stage in &stages {
        let aggregate = grouped
            .entry((stage.component.clone(), stage.stage.clone()))
            .or_default();
        aggregate.0 += 1;
        aggregate.1 = aggregate
            .1
            .checked_add(stage.duration_ns)
            .ok_or_else(|| IrodoriError::Profile("RF stage timestamp total overflow".to_owned()))?;
    }
    let aggregates = grouped
        .into_iter()
        .map(
            |((component, stage), (calls, total_ns))| RfDeviceStageAggregate {
                component,
                stage,
                calls,
                total_ns,
                mean_ns: total_ns / calls as u64,
            },
        )
        .collect::<Vec<_>>();
    let summed_stage_ns = stages.iter().try_fold(0_u64, |total, stage| {
        total
            .checked_add(stage.duration_ns)
            .ok_or_else(|| IrodoriError::Profile("RF summed timestamp overflow".to_owned()))
    })?;
    Ok(RfDeviceProfileReceipt {
        schema_version: 1,
        timing_boundary: "deferred CubeCL stream timestamps; resolved after RF device completion"
            .to_owned(),
        stages,
        aggregates,
        summed_stage_ns,
    })
}

fn duration_to_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).map_err(|_| {
        IrodoriError::Profile("RF device timestamp exceeds u64 nanoseconds".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_is_exclusive_and_finish_resets_the_scope() {
        begin_rf_device_profile().unwrap();
        assert!(rf_device_profile_active());
        assert!(begin_rf_device_profile().is_err());
        let receipt = finish_rf_device_profile().unwrap();
        assert!(receipt.stages.is_empty());
        assert!(!rf_device_profile_active());
    }
}
