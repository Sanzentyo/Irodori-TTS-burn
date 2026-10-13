//! Exact-shape cooperative matrix candidates for explicit FP16 execution.
//! CubeK's ordinary tuner uses shape buckets; the outer tuner measures these
//! plans on the actual layout and applies its FP32-reference accuracy gate.

use burn::{backend::wgpu::CubeTensor, tensor::DType};
use burn_backend::cubecl::dtype_to_storage_type;
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_matmul::multi_level::Strategy;
use cubek_matmul::multi_level::components::tile::TileMatmulKind;
use cubek_matmul::multi_level::routines::batch::double_buffering::DoubleBufferingArgs;
use cubek_matmul::multi_level::routines::batch::ordered_double_buffering::OrderedSelectionArgs;
use cubek_matmul::multi_level::routines::batch::simple::SimpleArgs;
use cubek_matmul::routine::BlueprintStrategy;
use cubek_std::InputBinding;

#[derive(Clone, Copy, Debug)]
pub(super) enum CmmaPlan {
    SingleRow,
    MultiRow,
    Double,
    OrderedRows4,
    OrderedRows8,
}

pub(super) fn matmul(
    lhs: CubeTensor,
    rhs: CubeTensor,
    plan: CmmaPlan,
) -> Result<CubeTensor, String> {
    matmul_with_output(lhs, rhs, plan, DType::F16)
}

/// Preserve unrounded attention scores while using FP16 matrix inputs.
pub(super) fn matmul_f32_output(
    lhs: CubeTensor,
    rhs: CubeTensor,
    plan: CmmaPlan,
) -> Result<CubeTensor, String> {
    matmul_with_output(lhs, rhs, plan, DType::F32)
}

fn matmul_with_output(
    lhs: CubeTensor,
    rhs: CubeTensor,
    plan: CmmaPlan,
    output_dtype: DType,
) -> Result<CubeTensor, String> {
    if lhs.dtype != DType::F16
        || rhs.dtype != DType::F16
        || lhs.meta.num_dims() < 2
        || lhs.meta.num_dims() != rhs.meta.num_dims()
        || !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || lhs.device != rhs.device
        || lhs.client.service_id() != rhs.client.service_id()
    {
        return Err("cooperative candidate requires contiguous FP16 matrices on one device".into());
    }
    let rank = lhs.meta.num_dims();
    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let (m, k, rhs_k, n) = (
        lhs_shape[rank - 2],
        lhs_shape[rank - 1],
        rhs_shape[rank - 2],
        rhs_shape[rank - 1],
    );
    if m == 0 || n == 0 || k == 0 || k != rhs_k || lhs_shape[..rank - 2] != rhs_shape[..rank - 2] {
        return Err("cooperative matrix dimensions do not match".into());
    }
    let tile_matmul = TileMatmulKind::Cmma;
    let strategy = match plan {
        CmmaPlan::SingleRow | CmmaPlan::MultiRow => {
            Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                multi_rows: matches!(plan, CmmaPlan::MultiRow),
                tile_matmul,
            }))
        }
        CmmaPlan::Double => {
            Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                specialized: false,
                tile_matmul,
            }))
        }
        CmmaPlan::OrderedRows4 | CmmaPlan::OrderedRows8 => {
            Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                partition_k: Some(2),
                row_count: Some(if matches!(plan, CmmaPlan::OrderedRows4) {
                    4
                } else {
                    8
                }),
                rows_per_plane: Some(2),
                tile_matmul,
            }))
        }
    };
    let output = burn_cubecl::kernel::matmul::utils::init_matmul_output(&lhs, &rhs, output_dtype);
    let client = lhs.client.clone();
    let storage = dtype_to_storage_type(DType::F16);
    // Half storage retains FP32 stage/register accumulators.
    let mut elems = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: dtype_to_storage_type(output_dtype),
    });
    cubek_matmul::launch::launch_ref(
        &cubek_matmul::strategy::Strategy::MultiLevel(strategy),
        &client,
        InputBinding::new(lhs.binding(), storage),
        InputBinding::new(rhs.binding(), storage),
        output.clone().binding(),
        &mut elems,
    )
    .map_err(|error| format!("cooperative matrix plan unsupported: {error:?}"))?;
    Ok(output)
}
