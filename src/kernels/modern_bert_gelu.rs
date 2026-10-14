//! GELU epilogue and gate multiplication, preserving Burn's storage rounding.
//!
//! Keep division/erf in the backend: the generated SPIR-V and WGSL polynomial
//! differ on Intel. Fuse the four remaining elementwise operations instead.

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate},
    tensor::{Shape, Tensor},
};
use cubecl::{
    CubeCount,
    prelude::KernelId,
    server::{KernelArguments, MetadataBindingInfo},
};

use super::precision::KernelFloatPrecision;

#[derive(Debug)]
struct GeluGateKernel {
    precision: KernelFloatPrecision,
    hidden: u32,
    capture_activation: bool,
}

impl KernelSource for GeluGateKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }

    fn source(&self) -> SourceTemplate {
        let (enable, scalar) = match self.precision {
            KernelFloatPrecision::F32 => ("", "f32"),
            KernelFloatPrecision::F16 => ("enable f16;", "f16"),
        };
        SourceTemplate::new(include_str!("modern_bert_gelu.wgsl"))
            .register("enable", enable)
            .register("scalar", scalar)
            .register("hidden", self.hidden.to_string())
            .register(
                "metadata_binding",
                if self.capture_activation { "4" } else { "3" },
            )
            .register(
                "activation_binding",
                if self.capture_activation {
                    "@group(0) @binding(3) var<storage, read_write> activation: array<F>;"
                } else {
                    ""
                },
            )
            .register(
                "activation_write",
                if self.capture_activation {
                    "activation[destination] = stage_a[lane ^ 1u];"
                } else {
                    ""
                },
            )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.hidden, self.capture_activation))
    }
}

pub(crate) fn apply(
    projected: Tensor<3>,
    capture_activation: bool,
) -> (Tensor<3>, Option<Tensor<3>>) {
    let [batch, sequence, width] = projected.dims();
    assert!(width > 0 && width.is_multiple_of(2));
    let erf = (projected.clone().narrow(2, 0, width / 2) / std::f64::consts::SQRT_2)
        .erf()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU GELU erf");
    let input = projected
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU GELU projection");
    assert!(input.is_contiguous());
    assert!(erf.is_contiguous());
    let precision = KernelFloatPrecision::from_dtype(input.dtype).expect("GELU F32/F16 input");
    let elements = u32::try_from(input.meta.num_elements() / 2).expect("GELU index exceeds u32");
    assert!(elements > 0);
    assert!(elements.checked_mul(2).is_some());
    let properties = input.client.properties();
    assert!(properties.hardware.max_units_per_cube >= 256);
    assert!(properties.hardware.max_cube_dim.0 >= 256);
    assert!(properties.hardware.max_bindings >= if capture_activation { 5 } else { 4 });
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
    assert!(input.handle.size_in_used() >= (bytes * 2) as u64);
    assert!(erf.handle.size_in_used() >= bytes as u64);
    assert!(bytes as u64 <= properties.memory.max_page_size);
    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::from([batch, sequence, width / 2]),
        input.client.empty(bytes),
        precision.dtype(),
    );
    let kernel = GeluGateKernel {
        precision,
        hidden: u32::try_from(width / 2).expect("GELU width exceeds u32"),
        capture_activation,
    };
    let activation = capture_activation.then(|| {
        CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            output.meta.shape().clone(),
            input.client.empty(bytes),
            precision.dtype(),
        )
    });
    let mut arguments = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(output.handle.clone().binding())
        .with_buffer(erf.handle.binding());
    if let Some(activation) = &activation {
        arguments = arguments.with_buffer(activation.handle.clone().binding());
    }
    // Reuse one pipeline across sequence lengths. CubeCL owns/caches the
    // immutable metadata uniform; no per-shape shader compilation is needed.
    arguments = arguments.with_info(MetadataBindingInfo::custom(vec![
        u64::from(elements) | (u64::from(groups_x) << 32),
        0,
    ]));
    input.client.launch(
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(256))),
        CubeCount::new_2d(groups_x, groups_y),
        arguments,
    );
    (
        Tensor::from_primitive::<crate::WgpuRaw>(output),
        activation.map(Tensor::from_primitive::<crate::WgpuRaw>),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{FloatDType, TensorData, activation::gelu};

    #[test]
    fn gelu_gate_matches_burn_fp32_storage_boundaries() {
        compare(crate::WgpuFloatPrecision::Fp32);
    }

    #[test]
    fn gelu_gate_matches_burn_fp16_storage_boundaries() {
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
        // Revisit a shape after changing its uniform metadata. Both execution
        // and observer pipelines must reuse their shader without stale bounds.
        for (hidden, rows) in [(7, 9), (7, 25), (7, 1), (7, 9), (3072, 3), (3072, 25)] {
            let values: Vec<f32> = (0..rows * hidden * 2)
                .map(|i| {
                    let j = i % (2 * hidden);
                    if j >= hidden {
                        ((i * 13 % 127) as f32 - 63.0) * 0.031
                    } else if i % 7 == 0 {
                        (i as f32 - 20.0) * 1.0e-8
                    } else {
                        ((i * 37 % 2003) as f32 - 1001.0) * 0.011
                    }
                })
                .collect();
            let dtype = match precision {
                crate::WgpuFloatPrecision::Fp32 => FloatDType::F32,
                crate::WgpuFloatPrecision::Fp16 => FloatDType::F16,
            };
            let input =
                Tensor::<3>::from_data(TensorData::new(values, [1, rows, 2 * hidden]), &device)
                    .cast(dtype);
            let expected_activation = gelu(input.clone().narrow(2, 0, hidden));
            let expected = expected_activation.clone() * input.clone().narrow(2, hidden, hidden);
            let actual = apply(input.clone(), false).0;
            let (observed, activation) = apply(input, true);
            assert_eq!(
                observed.clone().into_data().bytes(),
                actual.clone().into_data().bytes(),
                "recording must not change the production output"
            );
            assert_eq!(
                activation.unwrap().into_data().bytes(),
                expected_activation.into_data().bytes(),
                "recorded activation must be the actual pre-gate value"
            );
            let expected = expected
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let actual = actual
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let different: Vec<_> = expected
                .iter()
                .zip(&actual)
                .enumerate()
                .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                .collect();
            let maximum = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "{precision:?} hidden={hidden} rows={rows}: mismatches={} max_abs={maximum:e} first={:?}",
                different.len(),
                different.first()
            );
            assert!(different.is_empty(), "GELU storage rounding changed");
        }
    }
}
