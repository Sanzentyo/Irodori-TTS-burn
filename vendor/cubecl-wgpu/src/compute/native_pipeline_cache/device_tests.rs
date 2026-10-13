//! Run explicitly with nextest, one test per process. No invented cache bytes
//! are ever supplied to the driver: every admitted payload comes from get_data.
use super::*;
use std::{path::PathBuf, sync::mpsc, time::SystemTime};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cubecl-native-cache-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        environment::set_root(&path);
        environment::activate("first");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        // The process owns its isolated directory. Database handles are dropped
        // before this guard, but Windows can retain the engine's shared handle.
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("Unable to remove native cache test directory: {error}");
        }
    }
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    cubecl_environment::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let kind = match std::env::var("CUBECL_NATIVE_CACHE_TEST_DEVICE").as_deref() {
            Ok("integrated") => wgpu::DeviceType::IntegratedGpu,
            Ok("discrete") | Err(_) => wgpu::DeviceType::DiscreteGpu,
            other => panic!("Expected integrated or discrete test device: {other:?}"),
        };
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .await
            .into_iter()
            .find(|adapter| adapter.get_info().device_type == kind)
            .expect("requested native Vulkan GPU");
        assert!(adapter.features().contains(wgpu::Features::PIPELINE_CACHE));
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::PIPELINE_CACHE,
                ..Default::default()
            })
            .await
            .unwrap()
    })
}

fn compile_and_execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut NativePipelineCache,
    id: &KernelId,
) {
    let used = cache.begin(device, id);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
             @compute @workgroup_size(1) fn main() { out[0] = 42u; }"
                .into(),
        ),
    });
    let started = Instant::now();
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: used.as_ref().map(|used| &used.handle),
    });
    cache.compiled(used, started.elapsed().as_nanos());
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
    queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    assert_eq!(
        &*readback.slice(..).get_mapped_range().unwrap(),
        &42_u32.to_le_bytes()
    );
    readback.unmap();
}

#[test]
#[ignore = "requires a Vulkan GPU with PIPELINE_CACHE; selects discrete or integrated via CUBECL_NATIVE_CACHE_TEST_DEVICE"]
fn native_roundtrip_repairs_corruption_rejects_imports_and_resets_environment() {
    let _directory = TestDirectory::new();
    let (device, queue) = device();
    let id = KernelId::new::<TestDirectory>();
    let mut cache = NativePipelineCache::open(&device, true);
    assert!(cache.cache.as_ref().unwrap().available.is_empty());
    compile_and_execute(&device, &queue, &mut cache, &id);
    cache.checkpoint();
    assert_eq!(cache.metrics.stored, 1);
    let namespace = cache.cache.as_ref().unwrap().namespace.clone();
    let db = Database::open_active().unwrap();
    let mut saved = Vec::new();
    db.scan_with_origin(&namespace, &mut |key, value, origin| {
        assert_eq!(origin, Origin::Local);
        saved.push((key.to_vec(), value.to_vec()));
    });
    let (key, original) = saved.pop().unwrap();
    assert!(saved.is_empty());
    assert!(decode(&key, &original, Origin::Local).is_ok());

    let mut restored = NativePipelineCache::open(&device, true);
    compile_and_execute(&device, &queue, &mut restored, &id);
    restored.checkpoint();
    assert_eq!(restored.metrics.restored, 1);
    assert_eq!(restored.metrics.stored, 0);

    // Flip a byte in real get_data output, keeping its original checksum.
    let mut corrupt = original.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(matches!(
        db.replace(&namespace, &key, &corrupt, Origin::Local),
        Insertion::Stored
    ));
    let mut repaired = NativePipelineCache::open(&device, true);
    assert_eq!(repaired.metrics.rejected, 1);
    compile_and_execute(&device, &queue, &mut repaired, &id);
    repaired.checkpoint();
    assert_eq!(repaired.metrics.restored, 0);
    assert_eq!(repaired.metrics.stored, 1);
    assert!(decode(&key, &db.get(&namespace, &key).unwrap(), Origin::Local).is_ok());

    assert!(matches!(
        db.replace(&namespace, &key, &original, Origin::Imported),
        Insertion::Stored
    ));
    let mut imported = NativePipelineCache::open(&device, true);
    assert_eq!(imported.metrics.rejected, 1);
    compile_and_execute(&device, &queue, &mut imported, &id);
    imported.checkpoint();
    assert_eq!(imported.metrics.restored, 0);
    assert_eq!(imported.metrics.stored, 1);
    db.scan_with_origin(&namespace, &mut |_, _, origin| {
        assert_eq!(origin, Origin::Local)
    });

    let mut disabled = NativePipelineCache::open(&device, false);
    assert!(disabled.cache.is_none());
    compile_and_execute(&device, &queue, &mut disabled, &id);
    disabled.checkpoint();
    assert_eq!(disabled.metrics.stored, 0);
    assert_eq!(disabled.metrics.restored, 0);

    // A switch before synchronization must discard the pending old entry.
    let pending_id = KernelId::new::<NativePipelineCache>();
    compile_and_execute(&device, &queue, &mut imported, &pending_id);
    assert_eq!(imported.cache.as_ref().unwrap().pending.len(), 1);
    environment::activate("second");
    imported.checkpoint();
    assert!(imported.refresh(&device));
    assert!(!imported.refresh(&device));
    assert!(imported.cache.as_ref().unwrap().pending.is_empty());
    assert!(imported.cache.as_ref().unwrap().available.is_empty());
    compile_and_execute(&device, &queue, &mut imported, &id);
    imported.checkpoint();
    assert_eq!(imported.metrics.restored, 0);
    assert_eq!(imported.metrics.stored, 1);
    let mut old_count = 0;
    db.scan(&namespace, &mut |_, _| old_count += 1);
    assert_eq!(old_count, 1, "pending old entry was discarded");
    let fresh = NativePipelineCache::open(&device, true);
    assert_eq!(fresh.cache.as_ref().unwrap().available.len(), 1);
}
