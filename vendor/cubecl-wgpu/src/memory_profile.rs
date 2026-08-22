//! Process-local high-water monitoring for the WGPU main-memory allocator.
//!
//! This intentionally observes CubeCL's portable WGPU allocation boundary,
//! rather than a vendor API.  It is a diagnostic aid: callers delimit one
//! stage with [`begin`] and [`finish`], while every main-pool reservation
//! contributes its post-reservation usage to the high-water mark.

use std::sync::{LazyLock, Mutex};

use cubecl_runtime::memory_management::MemoryUsage;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryHighWater {
    pub bytes_in_use: u64,
    pub bytes_reserved: u64,
    pub number_allocs: u64,
    pub reservation_events: u64,
}

#[derive(Debug, Default)]
struct Probe {
    active: bool,
    high_water: MemoryHighWater,
}

static PROBE: LazyLock<Mutex<Probe>> = LazyLock::new(|| Mutex::new(Probe::default()));

/// Start a new process-local measurement window.
///
/// Only one window may be active at a time. Returning `false` lets callers
/// fail closed instead of accidentally pooling overlapping stages.
pub fn begin() -> bool {
    let mut probe = PROBE.lock().expect("WGPU memory profiler lock poisoned");
    if probe.active {
        return false;
    }
    probe.active = true;
    probe.high_water = MemoryHighWater::default();
    true
}

/// Finish the active window and return its allocator high-water mark.
pub fn finish() -> Option<MemoryHighWater> {
    let mut probe = PROBE.lock().expect("WGPU memory profiler lock poisoned");
    if !probe.active {
        return None;
    }
    probe.active = false;
    Some(probe.high_water)
}

pub(crate) fn record_reservation(usage: MemoryUsage) {
    let mut probe = PROBE.lock().expect("WGPU memory profiler lock poisoned");
    if !probe.active {
        return;
    }
    probe.high_water.bytes_in_use = probe.high_water.bytes_in_use.max(usage.bytes_in_use);
    probe.high_water.bytes_reserved = probe.high_water.bytes_reserved.max(usage.bytes_reserved);
    probe.high_water.number_allocs = probe.high_water.number_allocs.max(usage.number_allocs);
    probe.high_water.reservation_events += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_high_water_without_pooling_windows() {
        assert!(begin());
        assert!(!begin());
        record_reservation(MemoryUsage {
            bytes_padding: 0,
            bytes_in_use: 64,
            bytes_reserved: 128,
            number_allocs: 1,
        });
        record_reservation(MemoryUsage {
            bytes_padding: 0,
            bytes_in_use: 32,
            bytes_reserved: 256,
            number_allocs: 2,
        });
        let peak = finish().expect("active window");
        assert_eq!(peak.bytes_in_use, 64);
        assert_eq!(peak.bytes_reserved, 256);
        assert_eq!(peak.number_allocs, 2);
        assert_eq!(peak.reservation_events, 2);
        assert!(finish().is_none());
    }
}
