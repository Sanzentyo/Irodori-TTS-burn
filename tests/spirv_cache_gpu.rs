//! Explicit persistence acceptance. Run each cache policy in a fresh process.
#![cfg(feature = "spirv-cache")]

use burn::{
    backend::wgpu::WgpuDevice,
    tensor::{Device, Tensor, TensorData},
};
use cubecl_environment::{environment, persistence::Database};
use irodori_tts_burn::{WgpuFloatPrecision, backend_config};

fn execute(device: &Device, offset: f32) {
    let values = (0..128).map(|i| i as f32 + offset).collect::<Vec<_>>();
    let input = Tensor::<1>::from_data(TensorData::new(values.clone(), [128]), device);
    let actual = (input * 2.0 + 1.0).into_data().try_to_vec::<f32>().unwrap();
    let expected = values.iter().map(|x| x * 2.0 + 1.0).collect::<Vec<_>>();
    assert_eq!(actual, expected);
    device.sync().unwrap();
}

fn shader_entries(database: &Database) -> usize {
    let mut count = 0;
    for namespace in database.namespaces() {
        if namespace.starts_with("vulkan/") {
            database.scan(&namespace, &mut |_, _| count += 1);
        }
    }
    count
}

#[test]
#[ignore = "requires Vulkan GPU; select integrated with IRODORI_PACK_TEST_INTEGRATED=1 and set both cache switches explicitly"]
fn generated_shader_cache_resets_and_restores_across_environments() {
    let enabled = match std::env::var("CUBECL_WGPU_SPIRV_CACHE").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => panic!("set CUBECL_WGPU_SPIRV_CACHE to 0 or 1 for this acceptance test"),
    };
    let directory = tempfile::tempdir().unwrap();
    backend_config::configure_cubecl_persistent_cache_for_precision(
        directory.path(),
        WgpuFloatPrecision::Fp32,
    )
    .unwrap();
    let selected = if std::env::var_os("IRODORI_PACK_TEST_INTEGRATED").is_some() {
        cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0)
    } else {
        cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)
    };
    let device = backend_config::wgpu_device_with_precision(
        &WgpuDevice::new(selected),
        WgpuFloatPrecision::Fp32,
    )
    .unwrap();
    execute(&device, 0.0);
    let first_name = environment::active();
    let first = Database::open_active().unwrap();
    let original_count = shader_entries(&first);
    assert_eq!(original_count > 0, enabled);

    environment::activate("shader-cache-second");
    let second = Database::open_active().unwrap();
    assert_eq!(shader_entries(&second), 0);
    execute(&device, 16.0);
    assert_eq!(shader_entries(&second), original_count);
    assert_eq!(shader_entries(&first), original_count);

    environment::activate(first_name.as_ref());
    execute(&device, 32.0);
    assert_eq!(shader_entries(&first), original_count);
}
