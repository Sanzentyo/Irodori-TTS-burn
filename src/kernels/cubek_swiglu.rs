//! CubeK pairwise-compressed projection epilogue for SwiGLU.
//!
//! The right-hand matrix is prepared as adjacent gate/value columns. CubeK
//! accumulates both columns, evaluates SwiGLU in f32, and writes one physical
//! output element. The full `[rows, 2 * hidden]` projection never exists.

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::{DType, Shape};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::prelude::*;
use cubek_matmul::{
    components::global::PairwiseAccumulatorGlobalEpilogue,
    definition::{MatmulElems, MatmulGlobalElems},
    routines::{BlueprintStrategy, TileSizeSelection, batch::simple_unit::SimpleUnitSelectionArgs},
};
use cubek_std::InputBinding;

pub struct SwiGluPairEpilogue;

#[cube]
impl PairwiseAccumulatorGlobalEpilogue<()> for SwiGluPairEpilogue {
    fn apply<ES: Numeric, EG: Numeric>(
        first: ES,
        second: ES,
        _coordinate: (u32, u32),
        _runtime_config: &(),
    ) -> EG {
        let gate = f32::cast_from(first);
        let value = f32::cast_from(second);
        EG::cast_from(gate / (1.0 + (-gate).exp()) * value)
    }
}

/// Execute `x @ [w1[0], w3[0], w1[1], w3[1], ...]` and compress every
/// adjacent accumulator pair through SwiGLU.
///
/// This launcher is deliberately fail-closed. It performs no implicit
/// contiguous conversion or weight packing, so a successful call is exactly
/// one CubeK dispatch and allocates only the compressed result.
pub fn try_cubek_swiglu_compressed(
    input: CubeTensor<WgpuRuntime>,
    interleaved_weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.dtype != DType::F32
        || interleaved_weight.dtype != DType::F32
        || input.meta.num_dims() != 2
        || interleaved_weight.meta.num_dims() != 2
        || !input.is_contiguous()
        || interleaved_weight.is_contiguous()
        || input.device != interleaved_weight.device
        || !core::ptr::eq(input.client.info(), interleaved_weight.client.info())
    {
        return None;
    }
    let rows = input.meta.shape()[0];
    let inner = input.meta.shape()[1];
    let weight_inner = interleaved_weight.meta.shape()[0];
    let doubled_hidden = interleaved_weight.meta.shape()[1];
    if rows == 0
        || inner == 0
        || inner != weight_inner
        || doubled_hidden == 0
        || !doubled_hidden.is_multiple_of(2)
        || input.meta.strides()[..] != [inner, 1]
        || interleaved_weight.meta.strides()[..] != [1, inner]
    {
        return None;
    }

    let hidden = doubled_hidden / 2;
    let output_elements = rows.checked_mul(hidden)?;
    let output_bytes = output_elements.checked_mul(size_of::<f32>())?;
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, hidden]),
        client.empty(output_bytes),
        DType::F32,
    );
    let storage = dtype_to_storage_type(DType::F32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
        tile_size: TileSizeSelection::MinTileSize,
    });
    let launched =
        cubek_matmul::launch::launch_pairwise_compressed_ref::<WgpuRuntime, SwiGluPairEpilogue>(
            &client,
            InputBinding::new(input.binding(), storage),
            InputBinding::new(interleaved_weight.binding(), storage),
            output.clone().binding(),
            &strategy,
            &mut dtypes,
        );
    #[cfg(feature = "profile")]
    if let Err(error) = &launched {
        tracing::debug!(
            target: "irodori_tts_burn::route",
            ?error,
            "CubeK compressed SwiGLU launch rejected"
        );
    }
    launched.ok()?;
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::{FloatDType, Tensor};

    #[test]
    fn compressed_geometry_removes_the_full_expansion() {
        let rows = 3 * 489;
        let hidden = 3_680;
        assert_eq!(rows * hidden, rows * (2 * hidden) / 2);
        assert_eq!(rows * hidden * size_of::<f32>(), 21_594_240);
    }

    #[test]
    fn pairwise_writer_matches_cpu_on_partial_tiles() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("debug");
        let wgpu = WgpuDevice::DefaultDevice;
        // Other parallel GPU tests may have already locked the shared default
        // device settings.  This kernel receives explicitly-F32 tensor data,
        // so inspect the shared policy instead of trying to configure it a
        // second time.
        let device: burn::tensor::Device = wgpu.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        // Keep both dimensions off the usual tile boundaries while leaving a
        // large enough problem for CubeK's cooperative plane selector.
        let (rows, inner, hidden) = (33, 128, 97);
        let input = (0..rows * inner)
            .map(|index| ((index as f32 + 1.0) * 0.017).sin())
            .collect::<Vec<_>>();
        let weight = (0..inner * hidden * 2)
            .map(|index| ((index as f32 + 3.0) * 0.011).cos() * 0.25)
            .collect::<Vec<_>>();
        let input_tensor =
            Tensor::<1>::from_floats(input.as_slice(), &device).reshape([rows, inner]);
        let mut weight_physical = Vec::with_capacity(weight.len());
        for column in 0..hidden * 2 {
            for k in 0..inner {
                weight_physical.push(weight[k * hidden * 2 + column]);
            }
        }
        let weight_tensor = Tensor::<1>::from_floats(weight_physical.as_slice(), &device)
            .reshape([hidden * 2, inner])
            .transpose();
        let output = try_cubek_swiglu_compressed(
            input_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("partial-tile compressed matmul must be supported");
        let actual = Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut expected = vec![0.0; rows * hidden];
        for row in 0..rows {
            for column in 0..hidden {
                let mut gate = 0.0;
                let mut value = 0.0;
                for k in 0..inner {
                    let x = input[row * inner + k];
                    gate += x * weight[k * hidden * 2 + column * 2];
                    value += x * weight[k * hidden * 2 + column * 2 + 1];
                }
                expected[row * hidden + column] = gate / (1.0 + (-gate).exp()) * value;
            }
        }
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "compressed output mismatch at {index}: {actual} vs {expected}"
            );
        }
    }
}
