use crate::{
    components::global::args::RuntimeArgs,
    forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory},
};
use cubecl::{
    prelude::TensorBinding,
    std::tensor::{
        launch::ViewArg,
        layout::simple::{SimpleLayout, SimpleLayoutLaunch},
    },
    {Runtime, client::ComputeClient},
};
use cubek_matmul::{
    args::{InputArg, OutputArg},
    routines::BlueprintStrategy,
};
use cubek_matmul::{
    definition::{MatmulElems, MatmulVectorSizes},
    routines::BatchMatmulRoutine,
};
use cubek_std::InputBinding;

use crate::components::{ConvSetupError, ConvolutionProblem};

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<
    R: Runtime,
    Args: ConcreteArgs<A>,
    A: BatchMatmulRoutine<RuntimeArgs>,
>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    out: TensorBinding<R>,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_kernel_concrete_impl::<R, Args, A>(
        client,
        input,
        weight,
        bias,
        None,
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        dtypes,
    )
}

/// Launch a concrete convolution while exposing one f32 parameter vector to
/// the selected global writer. The writer decides how to interpret the vector;
/// the convolution core and accumulator semantics remain unchanged.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(crate) fn launch_kernel_concrete_with_epilogue<
    R: Runtime,
    Args: ConcreteArgs<A>,
    A: BatchMatmulRoutine<RuntimeArgs>,
>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    epilogue_param: TensorBinding<R>,
    out: TensorBinding<R>,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_kernel_concrete_impl::<R, Args, A>(
        client,
        input,
        weight,
        bias,
        Some(epilogue_param),
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_kernel_concrete_impl<
    R: Runtime,
    Args: ConcreteArgs<A>,
    A: BatchMatmulRoutine<RuntimeArgs>,
>(
    client: &ComputeClient<R>,
    input: InputBinding<R>,
    weight: InputBinding<R>,
    bias: Option<InputBinding<R>>,
    epilogue_param: Option<TensorBinding<R>>,
    out: TensorBinding<R>,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = input {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = weight {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(
        &problem.as_matmul_problem(),
        &device_settings,
        blueprint_strategy,
    )?;

    let problem = Args::adjust_problem(client, problem, &expand_info.blueprint, dtypes);
    let launch_info = A::prepare(&problem.as_matmul_problem(), &device_settings, expand_info)?;

    let (input, mut runtime_args) = <InputArg<Args> as ConcreteInputsFactory<A>>::create(
        input,
        weight,
        bias,
        &launch_info.blueprint,
        &problem,
        dtypes,
    );
    if let Some(epilogue_param) = epilogue_param {
        let layout = SimpleLayoutLaunch::from_handle(epilogue_param.clone(), 1);
        runtime_args.epilogue_param = Some(ViewArg::new_tensor::<SimpleLayout>(
            epilogue_param.into_tensor_arg(),
            layout,
        ))
        .into();
    }
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        dtypes,
    );

    cubek_matmul::strategy::launch_kernel::<Args, R, A>(
        client,
        input,
        output,
        runtime_args,
        launch_info,
    )
    .map_err(ConvSetupError::Matmul)
}
