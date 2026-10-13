use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes};
use crate::multi_level::args::*;
use crate::multi_level::components::CubeDimResource;
use crate::multi_level::components::batch::gemm::matmul::{matmul_entry, pairwise_matmul_entry};
use crate::multi_level::components::batch::gemm::{
    Gemm, GemmConfig, MatmulOperandLayouts, PairwiseGemm, PlanesSplit, Variant,
};
use crate::multi_level::components::batch::{BatchMatmulFamily, CheckBounds};
use crate::multi_level::components::global::PairwiseAccumulatorGlobalEpilogue;
use crate::multi_level::components::global::memory::GlobalLayoutConfig;
use crate::multi_level::components::stage::NumStages;
use crate::multi_level::definition::{
    Blueprint, CubeMappingLaunch, MatmulTypes, SwizzleModes, TilingScheme,
};
use cubecl::client::Client;
use cubecl::ir::{AddressType, DeviceProperties};
use cubecl::server::LaunchError;
use cubecl::{CubeCount, CubeDim};
use cubek_std::MatrixLayout;
use cubek_std::cube_count::HypercubeBlueprint;
use std::marker::PhantomData;

/// Unified GEMM family. Selects a kernel variant from operand layouts:
/// `Dot` (Row-Col) supports any `plane_dim` (plane-cooperative reduction
/// over K); `OuterM` / `OuterN` are CPU-only (require `plane_dim == 1`).
/// Col-Row has no variant: `launch_ref` normalizes it to Dot where planes
/// exist, and the CPU serves it through the `cpu_gemm` routine.
/// Also handles GEMV when one of `m`, `n` is 1: the vector side is
/// classified by `OperandLayout::Vector` and uses a layout-appropriate
/// variant.
pub struct GemmFamily {}

/// GEMM family whose Dot variant consumes adjacent RHS columns through a
/// typed compressed-output epilogue.
pub struct PairwiseGemmFamily<E: PairwiseAccumulatorGlobalEpilogue<()>> {
    _epilogue: PhantomData<E>,
}

/// Host-side work geometry used by [`crate::multi_level::routines::gemm::GemmRoutine`].
pub trait GemmOutputGeometry {
    fn work_columns(logical_columns: usize) -> usize;

    fn normalize_num_planes(target: usize, split_units: usize) -> usize;
}

impl GemmOutputGeometry for GemmFamily {
    fn work_columns(logical_columns: usize) -> usize {
        logical_columns
    }

    fn normalize_num_planes(target: usize, split_units: usize) -> usize {
        target.min(split_units).max(1)
    }
}

impl<E: PairwiseAccumulatorGlobalEpilogue<()>> GemmOutputGeometry for PairwiseGemmFamily<E> {
    fn work_columns(logical_columns: usize) -> usize {
        logical_columns
    }

    fn normalize_num_planes(target: usize, split_units: usize) -> usize {
        let candidate = target.min(split_units);
        candidate - candidate % 2
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GemmBlueprint {
    pub dtypes: MatmulElems,
    pub num_planes: usize,
    pub hypercube_blueprint: HypercubeBlueprint,
    pub variant: Variant,
    pub planes_split: PlanesSplit,
    pub check_bounds: CheckBounds,
}

impl Blueprint for GemmBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: self.variant.lhs_layout(),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: self.variant.rhs_layout(),
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        panic!("Gemm Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("Gemm Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for GemmFamily {
    type Matmul<MP: MatmulTypes> = Gemm<MP>;
    type Config = GemmConfig;
    type Blueprint = GemmBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(GemmConfig {
            plane_dim: device_props.hardware.plane_size_max,
            num_planes: blueprint.num_planes as u32,
            variant: blueprint.variant,
            planes_split: blueprint.planes_split,
            check_bounds: blueprint.check_bounds,
        })
    }

    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = ()>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        _config: ConfigRuntimeArg<MA>,
        cube_mapping: CubeMappingLaunch,
        blueprint: GemmBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                (),
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        Ok(CubeDimResource::Planes(blueprint.num_planes as u32))
    }

    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let plane_dim = client.properties().hardware.plane_size_max as usize;

        if vector_sizes.lhs != vector_sizes.rhs {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs vector sizes must be equal, got lhs:{:?}, rhs:{:?}",
                vector_sizes.lhs, vector_sizes.rhs
            ))));
        }

        if vector_sizes.out != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Out vector size must be 1, got {:?}",
                vector_sizes.out,
            ))));
        }

        let vs = vector_sizes.lhs;

        // Per-variant constraints. Dot supports plane-cooperative K reduction;
        // OuterM/OuterN are CPU-only because they don't reduce across units.
        match blueprint.variant {
            Variant::Dot => {
                let tile_dim = plane_dim * vs;
                if !problem.k.is_multiple_of(tile_dim) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by plane_dim * vector_size ({:?})",
                        problem.k, tile_dim,
                    ))));
                }
            }
            Variant::OuterN => {
                if plane_dim > 1 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterN variant requires plane_dim == 1 (CPU-only), got {}",
                        plane_dim,
                    ))));
                }
                if !problem.k.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                        problem.k, vs,
                    ))));
                }
                if !problem.n.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterN variant needs n ({}) divisible by vector_size ({})",
                        problem.n, vs,
                    ))));
                }
            }
            Variant::OuterM => {
                if plane_dim > 1 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterM variant requires plane_dim == 1 (CPU-only), got {}",
                        plane_dim,
                    ))));
                }
                if !problem.k.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Problem dimension k={:?} must be divisible by vector_size ({:?})",
                        problem.k, vs,
                    ))));
                }
                if !problem.m.is_multiple_of(vs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "OuterM variant needs m ({}) divisible by vector_size ({})",
                        problem.m, vs,
                    ))));
                }
            }
        }

        let derived = MatmulOperandLayouts::from_problem(problem)?.variant()?;
        if derived != blueprint.variant {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Blueprint variant {:?} disagrees with problem variant {:?}",
                blueprint.variant, derived
            ))));
        }

        Ok(())
    }
}

impl<E: PairwiseAccumulatorGlobalEpilogue<()>> BatchMatmulFamily<()> for PairwiseGemmFamily<E> {
    type Matmul<MP: MatmulTypes> = PairwiseGemm<MP, E>;
    type Config = GemmConfig;
    type Blueprint = GemmBlueprint;

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        <GemmFamily as BatchMatmulFamily<()>>::expand_config(
            device_props,
            blueprint,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        <GemmFamily as BatchMatmulFamily<()>>::num_stages()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = ()>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        _config: ConfigRuntimeArg<MA>,
        cube_mapping: CubeMappingLaunch,
        blueprint: GemmBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            pairwise_matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, E>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                (),
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };
        Ok(())
    }

    fn cubedim_resource(
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        <GemmFamily as BatchMatmulFamily<()>>::cubedim_resource(blueprint, dtypes, vector_sizes)
    }

    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        <GemmFamily as BatchMatmulFamily<()>>::validate_blueprint(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )?;
        if blueprint.variant != Variant::Dot || !problem.n.is_multiple_of(2) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "pairwise GEMM requires Dot layout and an even logical output width".to_owned(),
            )));
        }
        if blueprint.num_planes < 2 || !blueprint.num_planes.is_multiple_of(2) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "pairwise GEMM requires an even plane count of at least two, got {}",
                blueprint.num_planes
            ))));
        }
        Ok(())
    }
}
