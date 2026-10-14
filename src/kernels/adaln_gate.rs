//! Finalize packed cross-layer modulation, including the residual gate.

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate},
    tensor::{Shape, Tensor},
};
use cubecl::{
    CubeCount,
    prelude::KernelId,
    server::{KernelArguments, MetadataBindingInfo},
};

use super::precision::{KernelFloatPrecision, common_float_precision};

#[derive(Debug)]
struct ModulationKernel {
    precision: KernelFloatPrecision,
    dim: u32,
    slots: u32,
}

impl KernelSource for ModulationKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }
    fn source(&self) -> SourceTemplate {
        let (enable, scalar) = match self.precision {
            KernelFloatPrecision::F32 => ("", "f32"),
            KernelFloatPrecision::F16 => ("enable f16;", "f16"),
        };
        SourceTemplate::new(include_str!("adaln_gate.wgsl"))
            .register("enable", enable)
            .register("scalar", scalar)
            .register("dim", self.dim.to_string())
            .register("slots", self.slots.to_string())
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.dim, self.slots))
    }
}

pub(crate) fn finalize(up: Tensor<4>, bias: Tensor<4>, raw: Tensor<4>) -> Tensor<4> {
    let [batch, slots, singleton, dim] = up.dims();
    assert!(batch > 0 && slots > 0 && slots.is_multiple_of(3) && dim > 0 && singleton == 1);
    assert_eq!(bias.dims(), [1, slots, 1, dim]);
    assert_eq!(raw.dims(), [batch, 3, 1, dim]);
    let primitive = |t: Tensor<4>| {
        t.try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU modulation")
    };
    let up = primitive(up);
    let bias = primitive(bias);
    let raw = primitive(raw);
    let precision = common_float_precision([up.dtype, bias.dtype, raw.dtype])
        .expect("common F32/F16 modulation dtype");
    for tensor in [&up, &bias, &raw] {
        assert_eq!(tensor.device, up.device);
        assert!(tensor.is_contiguous());
        assert!(
            tensor.handle.size_in_used()
                >= (tensor.meta.num_elements() * precision.element_bytes()) as u64
        );
    }
    let elements = u32::try_from(up.meta.num_elements()).expect("modulation index exceeds u32");
    let properties = up.client.properties();
    assert!(properties.hardware.max_units_per_cube >= 256 && properties.hardware.max_bindings >= 5);
    assert!(properties.hardware.max_cube_dim.0 >= 256);
    assert!(properties.hardware.max_shared_memory_size >= 512 * precision.element_bytes());
    let groups = elements.div_ceil(256);
    let groups_x = groups.min(properties.hardware.max_cube_count.0).max(1);
    let groups_y = groups.div_ceil(groups_x);
    assert!(groups_y <= properties.hardware.max_cube_count.1);
    assert!(
        groups_x
            .checked_mul(groups_y)
            .and_then(|n| n.checked_mul(256))
            .is_some()
    );
    let bytes = elements as usize * precision.element_bytes();
    assert!(bytes as u64 <= properties.memory.max_page_size);
    let output = CubeTensor::new_contiguous(
        up.client.clone(),
        up.device.clone(),
        Shape::from([batch, slots, 1, dim]),
        up.client.empty(bytes),
        precision.dtype(),
    );
    let kernel = ModulationKernel {
        precision,
        dim: u32::try_from(dim).expect("modulation dim"),
        slots: u32::try_from(slots).expect("modulation slots"),
    };
    up.client.launch(
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(256))),
        CubeCount::new_2d(groups_x, groups_y),
        KernelArguments::new()
            .with_buffer(up.handle.binding())
            .with_buffer(bias.handle.binding())
            .with_buffer(raw.handle.binding())
            .with_buffer(output.handle.clone().binding())
            .with_info(MetadataBindingInfo::custom(vec![
                u64::from(elements) | (u64::from(groups_x) << 32),
                0,
            ])),
    );
    Tensor::from_primitive::<crate::WgpuRaw>(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, TensorData};

    #[test]
    fn packed_gate_matches_fp32_modulation() {
        compare(crate::WgpuFloatPrecision::Fp32);
    }
    #[test]
    fn packed_gate_matches_fp16_modulation() {
        compare(crate::WgpuFloatPrecision::Fp16);
    }

    fn compare(precision: crate::WgpuFloatPrecision) {
        let kind = if std::env::var_os("IRODORI_FUSION_TEST_INTEGRATED").is_some() {
            cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0)
        } else {
            cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)
        };
        let selected = burn::backend::wgpu::WgpuDevice::new(kind);
        let device =
            crate::backend_config::wgpu_device_with_precision(&selected, precision).unwrap();
        let dtype = match precision {
            crate::WgpuFloatPrecision::Fp32 => FloatDType::F32,
            crate::WgpuFloatPrecision::Fp16 => FloatDType::F16,
        };
        let tensor = |shape: [usize; 4], scale: f32| {
            Tensor::<4>::from_data(
                TensorData::new(
                    (0..shape.iter().product::<usize>())
                        .map(|i| ((i * 31 % 2047) as f32 - 1023.0) * scale)
                        .collect::<Vec<_>>(),
                    shape,
                ),
                &device,
            )
            .cast(dtype)
        };
        for (batch, modules, dim) in [
            (1, 2, 7),
            (2, 24, 1280),
            (3, 24, 1280),
            (1, 24, 1280),
            (2, 24, 1280),
        ] {
            let up = tensor([batch, modules * 3, 1, dim], 0.011);
            let bias = tensor([1, modules * 3, 1, dim], 0.00031);
            let raw = tensor([batch, 3, 1, dim], 0.0013);
            let reference = (up.clone() + bias.clone()).reshape([batch, modules, 3, dim])
                + raw.clone().reshape([batch, 1, 3, dim]);
            let gates = reference
                .clone()
                .narrow(2, 2, 1)
                .tanh()
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let mut expected = reference
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            for row in 0..batch * modules {
                expected[(row * 3 + 2) * dim..(row * 3 + 3) * dim]
                    .copy_from_slice(&gates[row * dim..(row + 1) * dim]);
            }
            let actual = finalize(up, bias, raw)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let mismatch = expected
                .iter()
                .zip(&actual)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "gate {precision:?} B{batch} modules{modules} dim{dim}: mismatches={mismatch} max_abs={max_abs:e}"
            );
            assert_eq!(mismatch, 0, "packed modulation changed");
        }
    }
}
