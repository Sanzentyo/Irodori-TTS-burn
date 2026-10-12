//! One-dispatch Independent CFG combine and Euler update.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::common_float_precision;

const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
struct RfCfgEulerUpdateKernel {
    precision: super::precision::KernelFloatPrecision,
    elements: u32,
    cfg_scale_bits: u32,
    dt_bits: u32,
}

impl KernelSource for RfCfgEulerUpdateKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("rf_cfg_euler_update.wgsl"),
                include_str!("rf_cfg_euler_update_f16.wgsl"),
            )
            .register("elements", self.elements.to_string())
            .register(
                "cfg_scale",
                format!("{:.9}", f32::from_bits(self.cfg_scale_bits)),
            )
            .register("dt", format!("{:.9}", f32::from_bits(self.dt_bits)))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.elements,
            self.cfg_scale_bits,
            self.dt_bits,
        ))
    }
}

/// Fuse `v_cond + (v_cond - v_uncond) * scale` and `x_t + v * dt`.
///
/// The route is deliberately restricted to two-row Independent CFG topology.
/// Unsupported shapes, mixed dtypes, non-contiguous views, and non-finite
/// scalar policy values fail closed without allocating or dispatching.
pub fn try_rf_independent_cfg_euler_update_wgsl(
    x_t: CubeTensor<WgpuRuntime>,
    velocities: CubeTensor<WgpuRuntime>,
    cfg_scale: f32,
    dt: f32,
) -> Option<CubeTensor<WgpuRuntime>> {
    if x_t.meta.num_dims() != 3
        || velocities.meta.num_dims() != 3
        || !cfg_scale.is_finite()
        || !dt.is_finite()
    {
        return None;
    }
    let x_shape = x_t.meta.shape().dims::<3>();
    let velocity_shape = velocities.meta.shape().dims::<3>();
    let precision = common_float_precision([x_t.dtype, velocities.dtype])?;
    let compatible = velocity_shape == [x_shape[0].checked_mul(2)?, x_shape[1], x_shape[2]]
        && x_t.is_contiguous()
        && velocities.is_contiguous()
        && x_t.device == velocities.device;
    if !compatible {
        return None;
    }
    let elements = x_shape.into_iter().try_fold(1usize, usize::checked_mul)?;
    let elements_u32 = u32::try_from(elements).ok()?;
    let client = x_t.client.clone();
    let output_handle = client.empty(elements.checked_mul(precision.element_bytes())?);
    let output = CubeTensor::new_contiguous(
        client.clone(),
        x_t.device.clone(),
        Shape::from(x_shape),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            RfCfgEulerUpdateKernel {
                precision,
                elements: elements_u32,
                cfg_scale_bits: cfg_scale.to_bits(),
                dt_bits: dt.to_bits(),
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(elements_u32.div_ceil(WORKGROUP_SIZE)),
        KernelArguments::new()
            .with_buffer(x_t.handle.binding())
            .with_buffer(velocities.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_geometry_is_nonzero_and_bounded() {
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(50usize * 128, 6_400);
        assert_eq!(6_400_u32.div_ceil(WORKGROUP_SIZE), 25);
    }

    /// Directly checks the custom kernel against the five ordinary Burn
    /// elementwise operations, including F16 scalar and storage rounding.
    ///
    /// Ignored by default because WGPU device teardown is not reliable in the
    /// test harness. Run manually on an adapter with shader-f16 support.
    #[test]
    #[ignore = "requires a shader-f16 WGPU adapter"]
    fn f16_matches_reference_elementwise_graph_bitwise() {
        use burn::backend::wgpu::graphics::AutoGraphicsApi;
        use burn::backend::wgpu::{WgpuDevice, init_setup};
        use burn::tensor::Tensor;

        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::backend_config::WgpuFloatPrecision::Fp16,
        )
        .expect("test WGPU device must support F16");
        let x_values = [0.125, -0.25, 1.5, -2.0, 0.001, -0.003, 12.0, -9.0];
        let velocity_values = [
            0.5, -0.75, 0.125, 3.0, -0.02, 0.03, -4.0, 5.0, -0.25, 0.5, -0.5, 2.0, 0.015, -0.025,
            -3.0, 6.0,
        ];
        let cfg_scale = 2.333_333_3_f32;
        let dt = -0.333_333_34_f32;

        let x_t = Tensor::<1>::from_floats(x_values.as_slice(), &device).reshape([1, 2, 4]);
        let velocities =
            Tensor::<1>::from_floats(velocity_values.as_slice(), &device).reshape([2, 2, 4]);
        let chunks = velocities.clone().chunk(2, 0);
        let conditioned = chunks[0].clone();
        let reference = x_t.clone()
            + (conditioned.clone() + (conditioned - chunks[1].clone()) * cfg_scale) * dt;

        let actual = try_rf_independent_cfg_euler_update_wgsl(
            x_t.try_into_primitive::<crate::WgpuRaw>()
                .expect("x_t must use raw WGPU"),
            velocities
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("velocities must use raw WGPU"),
            cfg_scale,
            dt,
        )
        .expect("supported profile must select the custom kernel");
        let actual = Tensor::<3>::from_primitive::<crate::WgpuRaw>(actual);
        let expected = reference
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let actual = actual.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(actual, expected);
    }
}
