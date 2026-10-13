//! Private, local-only driver cache in the existing `CubeCL` environment.
//!
//! Each pipeline has an immutable entry. This avoids serializing an ever-growing
//! device-wide cache on each synchronization during autotuning. New entries are
//! committed together, using the environment's existing transaction/locking.
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_environment::{
    bytes::Bytes,
    environment,
    persistence::{Database, Insertion, Origin},
};
use cubecl_server::{
    compiler::{KernelCacheKey, build_id_hash},
    config::{CubeClRuntimeConfig, RuntimeConfig},
    id::KernelId,
};
use std::{
    collections::{BTreeMap, HashSet},
    time::Instant,
};

const MAGIC: &[u8; 8] = b"CWGPC001";
const HEADER_BYTES: usize = 8 + 16 + 16 + 8;
// Bounds the private wire format; rejected entries never reach the driver.
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Default)]
struct Metrics {
    pipelines: usize,
    restored: usize,
    rejected: usize,
    stored: usize,
    bytes: usize,
    pipeline_ns: u128,
    restore_ns: u128,
    save_ns: u128,
}

#[derive(Debug)]
struct LocalCache {
    database: Database,
    namespace: String,
    available: BTreeMap<Vec<u8>, Vec<u8>>,
    pending: BTreeMap<Vec<u8>, Vec<u8>>,
    corrupt: HashSet<Vec<u8>>,
}

#[derive(Debug)]
pub(crate) struct NativePipelineCache {
    generation: u32,
    build_id: StableHash,
    enabled: bool,
    cache: Option<LocalCache>,
    metrics: Metrics,
    reported_pipelines: usize,
}

/// Bytes admitted by the local store, tied to a single pipeline.
pub(crate) struct PipelineCacheUse {
    pub(crate) handle: wgpu::PipelineCache,
    key: Vec<u8>,
    restored: bool,
}

fn namespace(info: &wgpu::AdapterInfo, build_id: StableHash) -> Option<String> {
    let key = wgpu::util::pipeline_cache_key(info)?;
    let driver = StableHasher::hash_one(&(&info.driver, &info.driver_info));
    Some(format!(
        "native-pipeline/wgpu-30/v1/{key}/{driver:032x}/{build_id:032x}"
    ))
}

fn encode(key: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&StableHasher::hash_one(&key).to_le_bytes());
    bytes.extend_from_slice(&StableHasher::hash_one(&payload).to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn decode<'a>(key: &[u8], bytes: &'a [u8], origin: Origin) -> Result<&'a [u8], &'static str> {
    if origin != Origin::Local {
        return Err("imported native cache");
    }
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err("invalid native cache header");
    }
    let payload = &bytes[HEADER_BYTES..];
    if payload.is_empty()
        || payload.len() > MAX_PAYLOAD_BYTES
        || bytes[40..48] != (payload.len() as u64).to_le_bytes()
        || bytes[8..24] != StableHasher::hash_one(&key).to_le_bytes()
        || bytes[24..40] != StableHasher::hash_one(&payload).to_le_bytes()
    {
        return Err("native cache integrity mismatch");
    }
    Ok(payload)
}

impl NativePipelineCache {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let enabled = match std::env::var("CUBECL_WGPU_NATIVE_PIPELINE_CACHE") {
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            Err(std::env::VarError::NotPresent) => true,
            other => {
                log::warn!(
                    "CUBECL_WGPU_NATIVE_PIPELINE_CACHE expects 0 or 1, disabling: {other:?}"
                );
                false
            }
        } && CubeClRuntimeConfig::get().compilation.cache;
        Self::open(device, enabled)
    }

    fn open(device: &wgpu::Device, enabled: bool) -> Self {
        // Capture the generation before opening: a concurrent environment switch
        // is detected on the next access, and pending old entries are discarded.
        let generation = environment::generation();
        let build_id = build_id_hash();
        let started = Instant::now();
        let mut metrics = Metrics::default();
        let cache = if enabled && device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            namespace(&device.adapter_info(), build_id).and_then(|namespace| {
                Database::open_active().map(|database| {
                    let mut available = BTreeMap::new();
                    let mut corrupt = HashSet::new();
                    database.scan_with_origin(&namespace, &mut |key, value, origin| match decode(
                        key, value, origin,
                    ) {
                        Ok(payload) => {
                            available.insert(key.to_vec(), payload.to_vec());
                        }
                        Err(reason) => {
                            metrics.rejected += 1;
                            if origin == Origin::Local {
                                corrupt.insert(key.to_vec());
                            }
                            log::debug!("Rejected {reason}");
                        }
                    });
                    if !corrupt.is_empty() {
                        log::warn!(
                            "Rejected {} corrupt local native pipeline cache entries",
                            corrupt.len()
                        );
                    }
                    LocalCache {
                        database,
                        namespace,
                        available,
                        corrupt,
                        pending: BTreeMap::new(),
                    }
                })
            })
        } else {
            None
        };
        metrics.restore_ns = started.elapsed().as_nanos();
        log::debug!(
            "native_pipeline_cache open enabled={} entries={} rejected={} restore_ns={}",
            cache.is_some(),
            cache.as_ref().map_or(0, |cache| cache.available.len()),
            metrics.rejected,
            metrics.restore_ns
        );
        Self {
            generation,
            build_id,
            enabled,
            cache,
            metrics,
            reported_pipelines: 0,
        }
    }

    /// Also tells the caller to invalidate its in-memory pipelines. In a
    /// WGSL-only build they would otherwise survive an environment switch.
    pub(crate) fn refresh(&mut self, device: &wgpu::Device) -> bool {
        if self.generation == environment::generation() {
            return false;
        }
        *self = Self::open(device, self.enabled);
        true
    }

    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        id: &KernelId,
    ) -> Option<PipelineCacheUse> {
        let cache = self.cache.as_mut()?;
        let hash: StableHash = StableHasher::hash_one(&KernelCacheKey::new(id, self.build_id));
        let key = hash.to_le_bytes().to_vec();
        let payload = cache.available.remove(&key);
        let restored = payload.is_some();
        let started = Instant::now();
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        // SAFETY: the only writer supplies unmodified PipelineCache::get_data
        // output. The private reader admits local-origin rows in this adapter /
        // driver / environment namespace, with exact build/kernel identity and
        // a verified length/checksum. Imported bundles never reach this call.
        // Owned bytes stay alive through the synchronous call, which copies them.
        // Persistent storage is trusted local application state: the checksum
        // detects accidental corruption, not a malicious local database writer.
        // wgpu explicitly permits persisted get_data output with that caveat.
        let handle = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("cubecl-native-local"),
                data: payload.as_deref(),
                fallback: true,
            })
        };
        let memory = memory.pop();
        let internal = internal.pop();
        let validation = validation.pop();
        let error = cubecl_environment::future::block_on(async {
            memory.await.or(internal.await).or(validation.await)
        });
        self.metrics.restore_ns += started.elapsed().as_nanos();
        if let Some(error) = error {
            log::warn!("Native pipeline cache unavailable: {error}");
            self.cache = None;
            return None;
        }
        self.metrics.restored += usize::from(restored);
        Some(PipelineCacheUse {
            handle,
            key,
            restored,
        })
    }

    pub(crate) fn compiled(&mut self, used: Option<PipelineCacheUse>, elapsed_ns: u128) {
        self.metrics.pipelines += 1;
        self.metrics.pipeline_ns += elapsed_ns;
        let started = Instant::now();
        if let (Some(cache), Some(used)) = (&mut self.cache, used)
            && !used.restored
            && let Some(payload) = used.handle.get_data()
        {
            if payload.is_empty() || payload.len() > MAX_PAYLOAD_BYTES {
                log::warn!(
                    "Native pipeline cache payload outside storage bounds: {} bytes",
                    payload.len()
                );
                return;
            }
            cache
                .pending
                .insert(used.key.clone(), encode(&used.key, &payload));
        }
        self.metrics.save_ns += started.elapsed().as_nanos();
    }

    /// No GPU wait and no I/O when no new pipelines were created. Cache failures
    /// are reported and do not change successfully compiled inference work.
    pub(crate) fn checkpoint(&mut self) {
        if self.generation != environment::generation() {
            // Never publish entries from a previous environment into the new one.
            return;
        }
        if self.reported_pipelines == self.metrics.pipelines {
            return;
        }
        let started = Instant::now();
        if let Some(cache) = &mut self.cache {
            let mut entries = Vec::new();
            for (key, value) in std::mem::take(&mut cache.pending) {
                self.metrics.bytes += value.len();
                if cache.corrupt.remove(&key) {
                    match cache
                        .database
                        .replace(&cache.namespace, &key, &value, Origin::Local)
                    {
                        Insertion::Stored => self.metrics.stored += 1,
                        other => log::warn!("Unable to repair native pipeline cache: {other:?}"),
                    }
                } else {
                    entries.push((Bytes::from_bytes_vec(key), Bytes::from_bytes_vec(value)));
                }
            }
            if !entries.is_empty() {
                let summary = cache.database.insert_many(
                    &cache.namespace,
                    &mut entries.into_iter(),
                    Origin::Local,
                );
                self.metrics.stored += summary.stored;
                if summary.failed != 0 {
                    log::warn!("Unable to save {} native pipeline caches", summary.failed);
                }
            }
        }
        self.metrics.save_ns += started.elapsed().as_nanos();
        self.reported_pipelines = self.metrics.pipelines;
        log::debug!(
            "native_pipeline_cache checkpoint enabled={} pipelines={} restored={} rejected={} stored={} bytes={} pipeline_ns={} restore_ns={} save_ns={}",
            self.cache.is_some(),
            self.metrics.pipelines,
            self.metrics.restored,
            self.metrics.rejected,
            self.metrics.stored,
            self.metrics.bytes,
            self.metrics.pipeline_ns,
            self.metrics.restore_ns,
            self.metrics.save_ns
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_envelope_rejects_imports_corruption_truncation_and_other_keys() {
        let key = b"pipeline-a";
        // This is parser-only data. It is never passed to wgpu.
        let original = encode(key, b"opaque test bytes");
        assert_eq!(
            decode(key, &original, Origin::Local).unwrap(),
            b"opaque test bytes"
        );
        assert!(decode(key, &original, Origin::Imported).is_err());
        assert!(decode(b"pipeline-b", &original, Origin::Local).is_err());
        for length in 0..original.len() {
            assert!(decode(key, &original[..length], Origin::Local).is_err());
        }
        for index in 0..original.len() {
            let mut changed = original.clone();
            changed[index] ^= 1;
            assert!(decode(key, &changed, Origin::Local).is_err());
        }
        assert!(decode(key, &encode(key, b""), Origin::Local).is_err());
    }

    #[test]
    fn native_namespace_separates_driver_and_adapter_and_rejects_other_apis() {
        let mut info = wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan);
        let original = namespace(&info, 7).unwrap();
        assert_ne!(namespace(&info, 8).unwrap(), original);
        info.driver_info = "updated driver".into();
        assert_ne!(namespace(&info, 7).unwrap(), original);
        info.driver_info.clear();
        info.device = 1;
        assert_ne!(namespace(&info, 7).unwrap(), original);
        info.backend = wgpu::Backend::Metal;
        assert!(namespace(&info, 7).is_none());
    }
}

#[cfg(test)]
mod device_tests;
