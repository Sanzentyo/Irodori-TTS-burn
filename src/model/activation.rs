use burn::tensor::{DType, Tensor};

/// Preserve the existing FP32 graph; FP16 evaluates SiLU in FP32 and rounds
/// once, as the reference does, instead of rounding sigmoid before multiplying.
pub(super) fn silu<const D: usize>(input: Tensor<D>) -> Tensor<D> {
    if input.dtype() != DType::F16 {
        return burn::tensor::activation::silu(input);
    }
    let primitive = input
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("model uses the WGPU backend");
    Tensor::from_primitive::<crate::WgpuRaw>(crate::kernels::silu::silu_f16(primitive))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp16_silu_rounds_once_after_fp32_arithmetic() {
        use burn::tensor::{Device, TensorData};
        let device = Device::default();
        let values = (-8192..8192).map(|i| i as f32 / 1024.0).collect::<Vec<_>>();
        let input = Tensor::<1>::from_data(
            TensorData::new(values.clone(), [values.len()]),
            (&device, DType::F16),
        );
        let result = silu(input)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        let mut max = 0.0_f32;
        for (x, y) in values.into_iter().zip(result) {
            let x = half::f16::from_f32(x).to_f32();
            let expected = half::f16::from_f32(x / (1.0 + (-x).exp())).to_f32();
            max = max.max((y - expected).abs());
        }
        // At most one half-precision ULP near a rounding boundary on this range.
        assert!(max <= 0.00390625, "max error {max}");
    }
}
