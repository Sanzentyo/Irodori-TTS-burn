//! Fused FP16 ModernBERT rotary transform preserving separate half products.
use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate},
    tensor::{DType, Shape, Tensor},
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

#[derive(Debug)]
struct RotaryKernel {
    shape: [u32; 4],
    strides: [u32; 4],
    groups_x: u32,
}
impl KernelSource for RotaryKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }
    fn source(&self) -> SourceTemplate {
        let [batch, sequence, heads, width] = self.shape;
        SourceTemplate::new(include_str!("modern_bert_rope_f16.wgsl"))
            .register("elements", (batch * sequence * heads * width).to_string())
            .register("sequence", sequence.to_string())
            .register("heads", heads.to_string())
            .register("width", width.to_string())
            .register("groups_x", self.groups_x.to_string())
            .register("batch_stride", self.strides[0].to_string())
            .register("sequence_stride", self.strides[1].to_string())
            .register("head_stride", self.strides[2].to_string())
            .register("width_stride", self.strides[3].to_string())
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.shape, self.strides, self.groups_x))
    }
}

pub(crate) fn apply_f16(input: Tensor<4>, cosine: Tensor<2>, sine: Tensor<2>) -> Tensor<4> {
    let shape = input.dims();
    assert!(shape.iter().all(|&dimension| dimension > 0));
    assert!(shape[3].is_multiple_of(2));
    assert_eq!(cosine.dims(), [shape[1], shape[3]]);
    assert_eq!(sine.dims(), cosine.dims());
    let raw = |tensor| {
        Tensor::<2>::try_into_primitive::<crate::WgpuRaw>(tensor).expect("WGPU rotary coefficients")
    };
    let input = input
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU rotary input");
    let cosine = raw(cosine);
    let sine = raw(sine);
    for tensor in [&input, &cosine, &sine] {
        assert_eq!(tensor.dtype, DType::F16);
        assert_eq!(tensor.device, input.device);
    }
    assert!(cosine.is_contiguous() && sine.is_contiguous());
    let elements = u32::try_from(input.meta.num_elements()).expect("rotary tensor exceeds u32");
    let workgroups = elements.div_ceil(256);
    let limits = input.client.properties().hardware.max_cube_count;
    let groups_x = workgroups.min(limits.0).max(1);
    let groups_y = workgroups.div_ceil(groups_x).max(1);
    assert!(
        groups_y <= limits.1,
        "rotary dispatch exceeds adapter limits"
    );
    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::from(shape),
        input.client.empty(elements as usize * 2),
        DType::F16,
    );
    let kernel = RotaryKernel {
        shape: shape.map(|v| u32::try_from(v).expect("rotary dimension exceeds u32")),
        strides: std::array::from_fn(|i| {
            u32::try_from(input.meta.strides()[i]).expect("rotary stride exceeds u32")
        }),
        groups_x,
    };
    input.client.launch(
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(256))),
        CubeCount::new_2d(groups_x, groups_y),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(cosine.handle.binding())
            .with_buffer(sine.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Tensor::from_primitive::<crate::WgpuRaw>(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use half::f16;

    #[test]
    fn rotates_strided_qkv_view_preserving_half_rounding() {
        let selected = crate::backend_config::DeviceSelector::default().device();
        let device = crate::backend_config::wgpu_device_with_precision(
            &selected,
            crate::WgpuFloatPrecision::Fp16,
        )
        .unwrap();
        let input = Tensor::<3>::from_data(
            TensorData::new(
                (0..2 * 3 * 24)
                    .map(|i| {
                        let scale = if i % 3 == 0 { 0.000_003_7 } else { 0.037 };
                        f16::from_f32((i as f32 - 50.0) * scale)
                    })
                    .collect::<Vec<_>>(),
                [2, 3, 24],
            ),
            &device,
        )
        .narrow(2, 8, 8)
        .reshape([2, 3, 2, 4]);
        let coefficients = |sign: f32| {
            Tensor::<2>::from_data(
                TensorData::new(
                    (0..12)
                        .map(|i| f16::from_f32(sign * ((i % 7) as f32 - 2.0) * 0.17))
                        .collect::<Vec<_>>(),
                    [3, 4],
                ),
                &device,
            )
        };
        let cosine = coefficients(1.0);
        let sine = coefficients(-0.7);
        let x = input.clone().into_data().try_to_vec::<f16>().unwrap();
        let c = cosine.clone().into_data().try_to_vec::<f16>().unwrap();
        let s = sine.clone().into_data().try_to_vec::<f16>().unwrap();
        let expected = (0..x.len())
            .map(|index| {
                let column = index % 4;
                let position = (index / 8) % 3;
                let partner = if column < 2 { index + 2 } else { index - 2 };
                let rotated = x[partner].to_f32() * if column < 2 { -1.0 } else { 1.0 };
                let first = f16::from_f32(x[index].to_f32() * c[position * 4 + column].to_f32());
                let second = f16::from_f32(rotated * s[position * 4 + column].to_f32());
                f16::from_f32(first.to_f32() + second.to_f32()).to_bits()
            })
            .collect::<Vec<_>>();
        let actual = apply_f16(input, cosine, sine)
            .into_data()
            .try_to_vec::<f16>()
            .unwrap();
        assert_eq!(
            actual.into_iter().map(f16::to_bits).collect::<Vec<_>>(),
            expected
        );
    }
}
