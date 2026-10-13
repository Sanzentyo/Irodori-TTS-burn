pub mod launch;

use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes};
use crate::multi_level::args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg};
use crate::multi_level::components::batch::gemm::{
    GemmBlueprint, GemmConfig, GemmFamily, GemmOutputGeometry, MatmulOperandLayouts, PlanesSplit,
    Variant,
};
use crate::multi_level::components::batch::{BatchMatmulFamily, CheckBounds};
use crate::multi_level::components::stage::NumStages;
use crate::multi_level::definition::CubeMappingLaunch;
use crate::multi_level::{
    BatchMatmulRoutine, ExpandInfo, LaunchInfo, batch_validate_blueprint, num_concurrent_planes,
};
use crate::routine::{BlueprintStrategy, DeviceSettings, Routine};
use cubecl::client::Client;
use cubecl::ir::AddressType;
use cubecl::{CubeCount, CubeDim};
use cubek_std::cube_count::{CubeCountPlan, CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
use std::fmt::Display;
use std::marker::PhantomData;

pub struct GemmRoutine<F = GemmFamily> {
    pub _family: PhantomData<F>,
}

#[derive(Default, Clone)]
pub struct GemmStrategy {
    pub target_num_planes: Option<usize>,
}

impl Display for GemmStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{:?}", self.target_num_planes)
    }
}

/// Returns `(m_units, n_units)`: count of per-plane blocks along each
/// output axis for the chosen variant. Outer-product variants pack
/// `vector_size` cells per block along their natural-vector axis.
fn output_units<F: GemmOutputGeometry>(
    problem: &MatmulProblem,
    variant: Variant,
    vector_size: usize,
) -> (usize, usize) {
    let work_n = F::work_columns(problem.n);
    match variant {
        Variant::Dot => (problem.m, work_n),
        Variant::OuterN => (problem.m, work_n / vector_size),
        Variant::OuterM => (problem.m / vector_size, work_n),
    }
}

impl<F> Routine<()> for GemmRoutine<F>
where
    F: BatchMatmulFamily<(), Blueprint = GemmBlueprint, Config = GemmConfig> + GemmOutputGeometry,
{
    type Strategy = GemmStrategy;
    type Blueprint = GemmBlueprint;
}

impl<F> BatchMatmulRoutine<()> for GemmRoutine<F>
where
    F: BatchMatmulFamily<(), Blueprint = GemmBlueprint, Config = GemmConfig> + GemmOutputGeometry,
{
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = ()>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        config: ConfigRuntimeArg<MA>,
        cube_count_input: CubeMappingLaunch,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        {
            unsafe {
                <F>::launch_unchecked::<MA>(
                    client,
                    cube_dim,
                    cube_count,
                    address_type,
                    input,
                    output,
                    config,
                    cube_count_input,
                    blueprint,
                    dtypes,
                    vector_sizes,
                )?
            }
            Ok(())
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<F, ()>(client, blueprint, problem, dtypes, vector_sizes)
    }

    fn num_stages() -> NumStages {
        F::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let properties = device_settings.client.properties();

        match strategy {
            BlueprintStrategy::Forced(blueprint) => Ok(ExpandInfo {
                blueprint: blueprint.clone(),
                dtypes,
            }),
            BlueprintStrategy::Inferred(strategy) => {
                let target_num_planes = match strategy.target_num_planes {
                    Some(num_planes) => num_planes,
                    None => num_concurrent_planes(&properties.hardware),
                };

                let variant = MatmulOperandLayouts::from_problem(problem)?.variant()?;
                let planes_split = variant.planes_split();
                let vector_size = device_settings.vector_sizes.lhs;

                let (m_units, n_units) = output_units::<F>(problem, variant, vector_size);
                let split_units = match planes_split {
                    PlanesSplit::M => m_units,
                    PlanesSplit::N => n_units,
                };
                let num_planes = F::normalize_num_planes(target_num_planes, split_units);
                if num_planes == 0 {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(
                        "GEMM family could not form a valid plane group".to_owned(),
                    )));
                }

                let check_bounds = if split_units.is_multiple_of(num_planes) {
                    CheckBounds::None
                } else {
                    CheckBounds::Terminate
                };

                let blueprint = GemmBlueprint {
                    dtypes: dtypes.clone(),
                    num_planes,
                    hypercube_blueprint: HypercubeBlueprint::builder()
                        .cube_count_strategy(CubeCountStrategy::Flattened)
                        .global_order(GlobalOrder::RowMajor)
                        .build(),
                    variant,
                    planes_split,
                    check_bounds,
                };

                Ok(ExpandInfo { blueprint, dtypes })
            }
        }
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cube_dim = F::cubedim_resource(&blueprint, &dtypes, &device_settings.vector_sizes)?
            .to_cube_dim(device_settings.plane_dim)?;

        let variant = blueprint.variant;
        let vector_size = device_settings.vector_sizes.lhs;
        let (m_units, n_units) = output_units::<F>(problem, variant, vector_size);
        let (m_cubes, n_cubes) = match blueprint.planes_split {
            PlanesSplit::M => (
                m_units.div_ceil(blueprint.num_planes) as u32,
                n_units as u32,
            ),
            PlanesSplit::N => (
                m_units as u32,
                n_units.div_ceil(blueprint.num_planes) as u32,
            ),
        };

        let cube_count_plan = CubeCountPlan::from_blueprint(
            &blueprint.hypercube_blueprint,
            (m_cubes, n_cubes, problem.num_batches() as u32).into(),
            &device_settings.max_cube_count,
        );

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}
