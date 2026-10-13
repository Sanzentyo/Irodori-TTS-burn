#![cfg(feature = "packed-weights")]

use burn::backend::wgpu::{CubeDim, KernelSource, SourceKernel, SourceTemplate, WgpuDevice};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};
use wgslpack::{Archive, Dtype, Layout, TensorSpec};

#[derive(Debug)]
struct Decode;
impl KernelSource for Decode {
    fn lang(&self) -> &'static str {
        "wgsl"
    }

    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(format!(
            "{}\n{}",
            wgslpack::WGSL,
            r#"
// CubeCL suballocates these disjoint ranges from one storage buffer, so their
// binding access modes must agree. The shader only reads the two input ranges.
@group(0) @binding(0) var<storage, read_write> descriptors: array<WgslPackTensor>;
@group(0) @binding(1) var<storage, read_write> archive: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= 5u { return; }
    let t = descriptors[id.x];
    let packed = t.dtype == 2u || t.dtype == 3u;
    for (var i = 0u; i < 5u; i += 1u) {
        let word_index = t.data_offset.x / 4u + select(i, i / 2u, packed);
        output[id.x * 12u + i] = wgslpack_float(archive[word_index], t.dtype, i & 1u);
    }
    output[id.x * 12u + 5u] = f32(t.rank);
    output[id.x * 12u + 6u] = f32(t.shape[2]);
    output[id.x * 12u + 7u] = f32(t.strides[0]);
    output[id.x * 12u + 8u] = f32(t.strides[2]);
    output[id.x * 12u + 9u] = f32(t.data_length.x);
    output[id.x * 12u + 10u] = f32(t.name_length);
    output[id.x * 12u + 11u] = f32(t.data_offset.y);
}
"#
        ))
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

#[test]
#[ignore = "requires an explicit WGPU adapter; set IRODORI_PACK_TEST_INTEGRATED=1 for Intel"]
fn borrowed_archive_descriptors_and_payloads_execute_in_wgsl() {
    let kinds = [Dtype::F32, Dtype::F16, Dtype::Bf16, Dtype::I32, Dtype::U32];
    let names = ["a", "bb", "ccc", "dddd", "eeeee"];
    let specs = kinds
        .iter()
        .zip(names)
        .map(|(&dtype, name)| TensorSpec {
            name,
            dtype,
            shape: &[1, 1, 5],
        })
        .collect::<Vec<_>>();
    let layout = Layout::new(&specs, b"opaque").unwrap();
    let mut bytes = vec![0u8; layout.total_len() as usize];
    bytes[..layout.prefix().len()].copy_from_slice(layout.prefix());
    let values = [0.0f32, 1.5, -2.0, 0.125, 64.0];
    let signed = [-10i32, 0, 7, 123, i32::MAX];
    let unsigned = [0u32, 1, 17, 1000, u32::MAX];
    let mut expected = Vec::new();
    for (i, descriptor) in layout.index().tensors().enumerate() {
        let payload = match descriptor.dtype() {
            Dtype::F32 => values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            Dtype::F16 => values
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            Dtype::Bf16 => values
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            Dtype::I32 => signed.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::U32 => unsigned.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let range = descriptor.data_range();
        bytes[range.start as usize..range.end as usize].copy_from_slice(&payload);
        expected.extend(match descriptor.dtype() {
            Dtype::I32 => signed.map(|v| v as f32),
            Dtype::U32 => unsigned.map(|v| v as f32),
            _ => values,
        });
        expected.extend([
            3.0,
            5.0,
            5.0,
            1.0,
            payload.len() as f32,
            (i + 1) as f32,
            0.0,
        ]);
    }
    let archive = Archive::parse(&bytes).unwrap();
    let device = if std::env::var_os("IRODORI_PACK_TEST_INTEGRATED").is_some() {
        WgpuDevice::new(cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0))
    } else {
        WgpuDevice::new(cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0))
    };
    let client = cubecl::Device::from(device.clone()).client();
    let descriptors = client.create_from_slice(archive.index().descriptor_bytes());
    let payload = client.create_from_slice(&bytes);
    let output = client.empty(expected.len() * 4);
    client.launch(
        Box::new(SourceKernel::new(Decode, CubeDim::new_1d(32))),
        CubeCount::new_1d(1),
        KernelArguments::new()
            .with_buffer(descriptors.binding())
            .with_buffer(payload.binding())
            .with_buffer(output.clone().binding()),
    );
    let read = client.read_one(output).unwrap();
    let actual = read
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "requires IRODORI_PACK_TEST_CHECKPOINT pointing to an FP32 packed model with a duration predictor"]
fn packed_exact_load_omits_duration_parameters() {
    use irodori_tts_burn::{WgpuFloatPrecision, backend_config, weights};
    let path = std::env::var_os("IRODORI_PACK_TEST_CHECKPOINT").expect("set packed model fixture");
    let device = backend_config::wgpu_device_with_precision(
        &WgpuDevice::new(cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)),
        WgpuFloatPrecision::Fp32,
    )
    .unwrap();
    let (model, config) =
        weights::load_model_exact_only(std::path::Path::new(&path), &device).unwrap();
    assert!(!config.use_duration_predictor);
    assert!(!model.has_duration_predictor());
    device.sync().unwrap();
}
