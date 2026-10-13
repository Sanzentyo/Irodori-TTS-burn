use crate::components::global::args::RuntimeArgs;
use crate::components::global::epilogue::PreparedPostCastEpilogue;
use crate::components::{ConvSetupError, ConvolutionProblem};
use crate::forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory};
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::launch::ViewArg;
use cubecl::std::tensor::layout::simple::{SimpleLayout, SimpleLayoutLaunch};
use cubek_matmul::definition::{AccumulatorOperand, MatmulElems, MatmulVectorSizes};
use cubek_matmul::multi_level::BatchMatmulRoutine;
use cubek_matmul::multi_level::args::{InputArg, OutputArg};
use cubek_matmul::routine::BlueprintStrategy;
use cubek_std::InputBinding;

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_kernel_concrete<Args: ConcreteArgs<A>, A: BatchMatmulRoutine<RuntimeArgs>>(
    client: &Client,
    input: InputBinding,
    weight: InputBinding,
    bias: Option<InputBinding>,
    out: TensorBinding,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_kernel_concrete_impl::<Args, A>(
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
    Args: ConcreteArgs<A>,
    A: BatchMatmulRoutine<RuntimeArgs>,
>(
    client: &Client,
    input: InputBinding,
    weight: InputBinding,
    bias: Option<InputBinding>,
    epilogue: PreparedPostCastEpilogue,
    out: TensorBinding,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    launch_kernel_concrete_impl::<Args, A>(
        client,
        input,
        weight,
        bias,
        Some(epilogue),
        out,
        problem,
        vector_sizes,
        blueprint_strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn launch_kernel_concrete_impl<Args: ConcreteArgs<A>, A: BatchMatmulRoutine<RuntimeArgs>>(
    client: &Client,
    input: InputBinding,
    weight: InputBinding,
    bias: Option<InputBinding>,
    epilogue: Option<PreparedPostCastEpilogue>,
    out: TensorBinding,
    problem: ConvolutionProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<Args::Config, A>,
    dtypes: &MatmulElems,
) -> Result<(), ConvSetupError> {
    // SyncBiasLoading stages one column-bias vector and broadcasts it over M;
    // charging a complete M-by-N accumulator rejects kernels that fit.
    let accumulator = match bias {
        Some(_) => AccumulatorOperand::ColumnBias,
        None => AccumulatorOperand::Absent,
    };

    let mut view_vector_sizes = vector_sizes;

    if let InputBinding::Quantized { scheme, .. } = input {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = weight {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let device_settings = A::device_settings(client, view_vector_sizes);
    let expand_info = A::expand_blueprint(
        &problem.as_matmul_problem(accumulator),
        &device_settings,
        blueprint_strategy,
    )?;

    let problem = Args::adjust_problem(client, problem, &expand_info.blueprint, dtypes);
    let launch_info = A::prepare(
        &problem.as_matmul_problem(accumulator),
        &device_settings,
        expand_info,
    )?;

    let (input, mut runtime_args) = <InputArg<Args> as ConcreteInputsFactory<A>>::create(
        input,
        weight,
        bias,
        &launch_info.blueprint,
        &problem,
        dtypes,
    );
    if let Some(epilogue) = epilogue {
        runtime_args.epilogue.f32_param = epilogue
            .f32_param
            .map(|binding| {
                let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
                ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
            })
            .into();
        runtime_args.epilogue.f16_input_0 = epilogue
            .f16_input_0
            .map(|binding| {
                let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
                ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
            })
            .into();
        runtime_args.epilogue.f16_input_1 = epilogue
            .f16_input_1
            .map(|binding| {
                let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
                ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
            })
            .into();
        runtime_args.epilogue.f16_output_0 = epilogue
            .f16_output_0
            .map(|binding| {
                let layout = SimpleLayoutLaunch::from_handle(binding.clone(), 1);
                ViewArg::new_tensor::<SimpleLayout>(binding.into_tensor_arg(), layout)
            })
            .into();
    }
    let output = <OutputArg<Args> as ConcreteOutputFactory<A>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        dtypes,
    );

    cubek_matmul::multi_level::launch_kernel::<Args, A>(
        client,
        input,
        output,
        runtime_args,
        launch_info,
    )
    .map_err(ConvSetupError::Matmul)
}
