//! Normalize using backend-owned reductions and an exact pointwise epilogue.

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate},
    tensor::{FloatDType, Shape, Tensor},
};
use cubecl::{
    CubeCount,
    prelude::KernelId,
    server::{KernelArguments, MetadataBindingInfo},
};

use super::precision::KernelFloatPrecision;

#[derive(Debug)]
struct NormEpilogue {
    precision: KernelFloatPrecision,
    width: u32,
}

impl KernelSource for NormEpilogue {
    fn lang(&self) -> &'static str {
        "wgsl"
    }
    fn source(&self) -> SourceTemplate {
        let (enable, scalar) = match self.precision {
            KernelFloatPrecision::F32 => ("", "f32"),
            KernelFloatPrecision::F16 => ("enable f16;", "f16"),
        };
        SourceTemplate::new(include_str!("modern_bert_norm_epilogue.wgsl"))
            .register("enable", enable)
            .register("scalar", scalar)
            .register("width", self.width.to_string())
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.width))
    }
}

pub(super) fn normalize(input: Tensor<3>, gamma: Tensor<1>) -> Tensor<3> {
    let shape = input.dims();
    assert_eq!(gamma.dims(), [shape[2]]);
    assert_eq!(input.dtype(), gamma.dtype());
    assert_eq!(input.device(), gamma.device());
    let input = input.cast(FloatDType::F32);
    // Keep exactly the same input shapes and strides for Burn's tuned reductions.
    let mean = input.clone().mean_dim(2);
    let centered = input - mean;
    let variance = (centered.clone() * centered.clone()).mean_dim(2);
    let centered = centered
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU norm");
    let variance = variance
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU variance");
    let gamma = gamma
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU gamma");
    let precision = KernelFloatPrecision::from_dtype(gamma.dtype).expect("F32/F16 gamma");
    assert!(centered.is_contiguous() && variance.is_contiguous() && gamma.is_contiguous());
    let elements = u32::try_from(centered.meta.num_elements()).expect("norm index exceeds u32");
    assert!(elements > 0 && shape[2] > 0);
    let properties = centered.client.properties();
    assert!(
        properties.hardware.max_units_per_cube >= 256 && properties.hardware.max_cube_dim.0 >= 256
    );
    assert!(
        properties.hardware.max_bindings >= 5 && properties.hardware.max_shared_memory_size >= 2048
    );
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
    for (tensor, element_bytes) in [
        (&centered, 4),
        (&variance, 4),
        (&gamma, precision.element_bytes()),
    ] {
        assert!(
            tensor.handle.size_in_used() >= (tensor.meta.num_elements() * element_bytes) as u64
        );
    }
    let output = CubeTensor::new_contiguous(
        centered.client.clone(),
        centered.device.clone(),
        Shape::from(shape),
        centered.client.empty(bytes),
        precision.dtype(),
    );
    let kernel = NormEpilogue {
        precision,
        width: u32::try_from(shape[2]).expect("norm width"),
    };
    centered.client.launch(
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(256))),
        CubeCount::new_2d(groups_x, groups_y),
        KernelArguments::new()
            .with_buffer(centered.handle.binding())
            .with_buffer(variance.handle.binding())
            .with_buffer(gamma.handle.binding())
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
    use burn::{nn::LayerNormConfig, tensor::TensorData};
    #[test]
    fn epilogue_matches_fp32_layer_norm() {
        compare(crate::WgpuFloatPrecision::Fp32);
    }
    #[test]
    fn epilogue_matches_fp16_layer_norm() {
        compare(crate::WgpuFloatPrecision::Fp16);
    }
    fn compare(precision: crate::WgpuFloatPrecision) {
        let kind = if std::env::var_os("IRODORI_FUSION_TEST_INTEGRATED").is_some() {
            cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0)
        } else {
            cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)
        };
        let device = crate::backend_config::wgpu_device_with_precision(
            &burn::backend::wgpu::WgpuDevice::new(kind),
            precision,
        )
        .unwrap();
        let dtype = match precision {
            crate::WgpuFloatPrecision::Fp32 => FloatDType::F32,
            crate::WgpuFloatPrecision::Fp16 => FloatDType::F16,
        };
        let mut norm = LayerNormConfig::new(768).with_bias(false).init(&device);
        norm.gamma = burn::module::Param::initialized(
            burn::module::ParamId::new(),
            Tensor::<1>::from_data(
                TensorData::new(
                    (0..768)
                        .map(|i| 0.5 + (i % 31) as f32 * 0.07)
                        .collect::<Vec<_>>(),
                    [768],
                ),
                &device,
            )
            .cast(dtype),
        );
        for [batch, sequence] in [[1, 3], [1, 25], [2, 17], [1, 129]] {
            let input = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * sequence * 768)
                        .map(|i| ((i * 31 % 997) as f32 - 498.0) * 0.013)
                        .collect::<Vec<_>>(),
                    [batch, sequence, 768],
                ),
                &device,
            )
            .cast(dtype);
            let reference = norm
                .forward(input.clone())
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let actual = normalize(input, norm.gamma.val())
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let different = reference
                .iter()
                .zip(&actual)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let maximum = reference
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!(
                "epilogue {precision:?} B{batch} S{sequence}: differences={different} max={maximum:e}"
            );
            assert_eq!(different, 0);
        }
    }
}
