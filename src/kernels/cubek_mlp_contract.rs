//! CubeK MLP contraction with an accumulator-domain gated-residual epilogue.
//!
//! The matmul core remains a reusable CubeK routine. The typed writer reads
//! the block residual and prepared gate only for valid output coordinates, so
//! the projected branch is never materialized or finalized by a second
//! dispatch. The right operand deliberately remains the logical source-column
//! view accepted by CubeK; no request-time packing or relayout is hidden here.

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::{DType, Shape};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::simple::{SimpleLayout, SimpleLayoutLaunch};
use cubecl::std::tensor::{View, launch::ViewArg, layout::Coords1d};
use cubek_matmul::{
    components::{global::AccumulatorGlobalStoreTransform, tile::TileMatmulKind},
    definition::{MatmulElems, MatmulGlobalElems},
    routines::{
        BlueprintStrategy, TileSizeSelection, batch::double_unit::DoubleUnitSelectionArgs,
        batch::simple::SimpleArgs, batch::simple_unit::SimpleUnitSelectionArgs,
    },
};
use cubek_std::InputBinding;

const INPUT_DIM: usize = 3_680;
const OUTPUT_DIM: usize = 1_280;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CubeKMlpContractAlgorithm {
    UnitMin,
    UnitMax,
    DoubleUnit,
    PlaneVec,
}

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
struct MlpResidualRuntimeArgs {
    residual: View<'static, f32, Coords1d>,
    gate: View<'static, f32, Coords1d>,
    sequence: u32,
}

struct MlpResidualTransform;

#[cube]
impl AccumulatorGlobalStoreTransform<MlpResidualRuntimeArgs> for MlpResidualTransform {
    fn apply<ES: Numeric, EG: Numeric>(
        value: ES,
        coordinate: (u32, u32),
        runtime: &mut MlpResidualRuntimeArgs,
    ) -> EG {
        let row = coordinate.0;
        let column = coordinate.1;
        let batch = row / runtime.sequence;
        let residual = runtime
            .residual
            .read((row * OUTPUT_DIM as u32 + column) as usize);
        let gate = runtime
            .gate
            .read((batch * OUTPUT_DIM as u32 + column) as usize);
        EG::cast_from(residual + gate * f32::cast_from(value))
    }
}

/// Execute strict-F32 `[rows,3680] @ [3680,1280]` and apply
/// `residual + gate * projection` before the primary output store.
///
/// The function performs no relayout, dtype conversion, or device transfer.
/// A successful call is one allocation and one CubeK dispatch.
pub fn try_cubek_mlp_contract_residual(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    algorithm: CubeKMlpContractAlgorithm,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.dtype != DType::F32
        || weight.dtype != DType::F32
        || residual.dtype != DType::F32
        || gate.dtype != DType::F32
        || input.meta.num_dims() != 2
        || weight.meta.num_dims() != 2
        || residual.meta.num_dims() != 2
        || gate.meta.num_dims() != 2
        || !input.is_contiguous()
        || !residual.is_contiguous()
        || !gate.is_contiguous()
        || input.device != weight.device
        || input.device != residual.device
        || input.device != gate.device
        || !core::ptr::eq(input.client.info(), weight.client.info())
        || !core::ptr::eq(input.client.info(), residual.client.info())
        || !core::ptr::eq(input.client.info(), gate.client.info())
    {
        #[cfg(feature = "profile")]
        tracing::debug!(
            target: "irodori_tts_burn::route",
            input_dtype = ?input.dtype,
            weight_dtype = ?weight.dtype,
            residual_dtype = ?residual.dtype,
            gate_dtype = ?gate.dtype,
            input_shape = ?input.meta.shape(),
            weight_shape = ?weight.meta.shape(),
            residual_shape = ?residual.meta.shape(),
            gate_shape = ?gate.meta.shape(),
            input_strides = ?input.meta.strides(),
            weight_strides = ?weight.meta.strides(),
            residual_strides = ?residual.meta.strides(),
            gate_strides = ?gate.meta.strides(),
            input_contiguous = input.is_contiguous(),
            weight_contiguous = weight.is_contiguous(),
            residual_contiguous = residual.is_contiguous(),
            gate_contiguous = gate.is_contiguous(),
            same_weight_device = input.device == weight.device,
            same_residual_device = input.device == residual.device,
            same_gate_device = input.device == gate.device,
            same_weight_client = core::ptr::eq(input.client.info(), weight.client.info()),
            same_residual_client = core::ptr::eq(input.client.info(), residual.client.info()),
            same_gate_client = core::ptr::eq(input.client.info(), gate.client.info()),
            "CubeK MLP residual transform rejected a physical input contract"
        );
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    if rows == 0
        || input.meta.shape().as_slice() != [rows, INPUT_DIM]
        || weight.meta.shape().as_slice() != [INPUT_DIM, OUTPUT_DIM]
        || residual.meta.shape().as_slice() != [rows, OUTPUT_DIM]
        || gate.meta.shape().as_slice() != [batch, OUTPUT_DIM]
        || input.meta.strides()[..] != [INPUT_DIM, 1]
        || weight.meta.strides()[..] != [1, INPUT_DIM]
        || residual.meta.strides()[..] != [OUTPUT_DIM, 1]
        || gate.meta.strides()[..] != [OUTPUT_DIM, 1]
    {
        #[cfg(feature = "profile")]
        tracing::debug!(
            target: "irodori_tts_burn::route",
            batch,
            sequence,
            rows,
            input_shape = ?input.meta.shape(),
            weight_shape = ?weight.meta.shape(),
            residual_shape = ?residual.meta.shape(),
            gate_shape = ?gate.meta.shape(),
            input_strides = ?input.meta.strides(),
            weight_strides = ?weight.meta.strides(),
            residual_strides = ?residual.meta.strides(),
            gate_strides = ?gate.meta.strides(),
            "CubeK MLP residual transform rejected its exact geometry"
        );
        return None;
    }

    let client = input.client.clone();
    let output_elements = rows.checked_mul(OUTPUT_DIM)?;
    let output_bytes = output_elements.checked_mul(size_of::<f32>())?;
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        client.empty(output_bytes),
        DType::F32,
    );
    let residual_binding = residual.binding();
    let gate_binding = gate.binding();
    let make_view = |binding: TensorBinding<WgpuRuntime>| {
        let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
        ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
    };
    let runtime_config = MlpResidualRuntimeArgsLaunch::new(
        make_view(residual_binding.clone()),
        make_view(gate_binding.clone()),
        u32::try_from(sequence).ok()?,
    );
    let runtime_address_type = residual_binding
        .required_address_type(size_of::<f32>())
        .max(gate_binding.required_address_type(size_of::<f32>()));
    let storage = dtype_to_storage_type(DType::F32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let lhs = InputBinding::new(input.binding(), storage);
    let rhs = InputBinding::new(weight.binding(), storage);
    let out = output.clone().binding();
    let launched = match algorithm {
        CubeKMlpContractAlgorithm::UnitMin | CubeKMlpContractAlgorithm::UnitMax => {
            let strategy = BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                tile_size: match algorithm {
                    CubeKMlpContractAlgorithm::UnitMin => TileSizeSelection::MinTileSize,
                    CubeKMlpContractAlgorithm::UnitMax => TileSizeSelection::MaxTileSize,
                    CubeKMlpContractAlgorithm::DoubleUnit | CubeKMlpContractAlgorithm::PlaneVec => {
                        unreachable!()
                    }
                },
            });
            cubek_matmul::launch::launch_accumulator_transform_unit_ref::<
                WgpuRuntime,
                MlpResidualRuntimeArgs,
                MlpResidualTransform,
            >(
                &client,
                lhs,
                rhs,
                out,
                runtime_config,
                runtime_address_type,
                &strategy,
                &mut dtypes,
            )
        }
        CubeKMlpContractAlgorithm::DoubleUnit => {
            let strategy = BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                tile_size: TileSizeSelection::MaxTileSize,
            });
            cubek_matmul::launch::launch_accumulator_transform_double_unit_ref::<
                WgpuRuntime,
                MlpResidualRuntimeArgs,
                MlpResidualTransform,
            >(
                &client,
                lhs,
                rhs,
                out,
                runtime_config,
                runtime_address_type,
                &strategy,
                &mut dtypes,
            )
        }
        CubeKMlpContractAlgorithm::PlaneVec => {
            let strategy = BlueprintStrategy::Inferred(SimpleArgs {
                tile_matmul: TileMatmulKind::PlaneVec,
                multi_rows: false,
            });
            cubek_matmul::launch::launch_accumulator_transform_plane_ref::<
                WgpuRuntime,
                MlpResidualRuntimeArgs,
                MlpResidualTransform,
            >(
                &client,
                lhs,
                rhs,
                out,
                runtime_config,
                runtime_address_type,
                &strategy,
                &mut dtypes,
            )
        }
    };
    #[cfg(feature = "profile")]
    if let Err(error) = &launched {
        tracing::debug!(
            target: "irodori_tts_burn::route",
            ?error,
            ?algorithm,
            "CubeK MLP residual transform launch rejected"
        );
    }
    launched.ok()?;
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_geometry_matches_the_released_mlp() {
        let rows = 3 * 489;
        assert_eq!(rows * INPUT_DIM, 5_398_560);
        assert_eq!(rows * OUTPUT_DIM, 1_877_760);
    }
}
