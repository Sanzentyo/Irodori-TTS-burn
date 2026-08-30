//! Explicit strict-F32 CubeK matmul candidates.
//!
//! CubeCL's persistent autotuner buckets exact shapes into coarse power-of-two
//! keys. That is useful as an inner-kernel prior, but it cannot be the final
//! authority for an end-to-end RF route: distinct batch geometry can share a
//! key while preferring different algorithms. These launchers expose stable
//! algorithms to Irodori's exact-device route tuner without changing storage
//! precision or silently relayouting either operand.

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    routines::{BlueprintStrategy, TileSizeSelection, batch::simple_unit::SimpleUnitSelectionArgs},
    strategy::Strategy,
};
use cubek_std::InputBinding;

/// Execute a contiguous strict-F32 batched matmul with CubeK's
/// `SimpleUnit/MinTile` algorithm.
///
/// The function is deliberately fail-closed: it performs no hidden relayout,
/// dtype conversion, or device transfer. A successful call therefore means
/// one allocation and one matmul dispatch with a stable algorithm identity.
pub fn try_matmul_unit_min_f32(
    lhs: CubeTensor<WgpuRuntime>,
    rhs: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if lhs.dtype != DType::F32
        || rhs.dtype != DType::F32
        || lhs.meta.num_dims() < 2
        || lhs.meta.num_dims() != rhs.meta.num_dims()
        || !lhs.is_contiguous()
        || !rhs.is_contiguous()
        || lhs.device != rhs.device
        || !core::ptr::eq(lhs.client.info(), rhs.client.info())
    {
        return None;
    }
    let rank = lhs.meta.num_dims();
    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let inner = lhs_shape[rank - 1];
    if inner == 0
        || lhs_shape[rank - 2] == 0
        || rhs_shape[rank - 1] == 0
        || rhs_shape[rank - 2] != inner
        || lhs_shape[..rank - 2] != rhs_shape[..rank - 2]
    {
        return None;
    }

    let output = burn_cubecl::kernel::matmul::utils::init_matmul_output(&lhs, &rhs, DType::F32);
    let client = lhs.client.clone();
    let storage = dtype_to_storage_type(DType::F32);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
        tile_size: TileSizeSelection::MinTileSize,
    }));
    cubek_matmul::launch::launch_ref(
        &strategy,
        &client,
        InputBinding::new(lhs.binding(), storage),
        InputBinding::new(rhs.binding(), storage),
        output.clone().binding(),
        &mut dtypes,
    )
    .ok()?;
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_name_is_stable_for_receipts() {
        let strategy = Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        }));
        assert_eq!(strategy.to_string(), "matmul_simple_unit_min_tile_size");
    }

    #[test]
    #[ignore = "requires a WGPU adapter"]
    fn batched_partial_tiles_match_the_backend_reference() {
        use burn::backend::wgpu::WgpuDevice;
        use burn::tensor::{Tensor, TensorData};

        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let raw_device = WgpuDevice::DefaultDevice;
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::WgpuFloatPrecision::Fp32,
        )
        .unwrap();
        let (batch, heads, m, k, n) = (2, 3, 33, 35, 17);
        let lhs_values = (0..batch * heads * m * k)
            .map(|index| ((index as f32 + 1.0) * 0.013).sin() * 0.25)
            .collect::<Vec<_>>();
        let rhs_values = (0..batch * heads * k * n)
            .map(|index| ((index as f32 + 3.0) * 0.017).cos() * 0.25)
            .collect::<Vec<_>>();
        let lhs = Tensor::<1>::from_data(
            TensorData::new(lhs_values, [batch * heads * m * k]),
            &device,
        )
        .reshape([batch, heads, m, k]);
        let rhs = Tensor::<1>::from_data(
            TensorData::new(rhs_values, [batch * heads * k * n]),
            &device,
        )
        .reshape([batch, heads, k, n]);
        let expected = lhs.clone().matmul(rhs.clone()).into_data();
        let actual = try_matmul_unit_min_f32(
            lhs.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            rhs.try_into_primitive::<crate::WgpuRaw>().unwrap(),
        )
        .expect("partial-tile strict-F32 candidate must launch");
        let actual = Tensor::<4>::from_primitive::<crate::WgpuRaw>(actual).into_data();
        let expected = expected.to_vec::<f32>().unwrap();
        let actual = actual.to_vec::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs <= 5.0e-5, "max_abs={max_abs:e}");
    }
}
