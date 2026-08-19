//! Convolution epilogues implemented on CubeK's generic matmul writer hook.

use cubecl::{Runtime, client::ComputeClient, ir::AddressType, prelude::*};
use cubek_matmul::components::global::{
    AccumulatorGlobalStoreTransform, PostCastGlobalEpilogue,
};
use cubek_std::InputBinding;

use crate::components::{
    ConvSetupError, ConvolutionProblem, EpilogueSetupError, global::args::RuntimeArgs,
};

/// Host-side argument and validation contract owned by a post-cast epilogue.
pub trait PostCastEpilogueSpec: Send + Sync + 'static {
    type LaunchArgs<R: Runtime>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError>;
}

/// Marker used by routines that cannot accept epilogue parameters.
pub struct NoPostCastEpilogue;

impl PostCastEpilogueSpec for NoPostCastEpilogue {
    type LaunchArgs<R: Runtime> = ();

    fn prepare<R: Runtime>(
        _client: &ComputeClient<R>,
        _args: Self::LaunchArgs<R>,
        _problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        Err(ConvSetupError::Epilogue(EpilogueSetupError::Missing))
    }
}

pub struct PreparedPostCastEpilogue<R: Runtime> {
    pub(crate) f32_param: Option<TensorBinding<R>>,
    pub(crate) f16_input_0: Option<TensorBinding<R>>,
    pub(crate) f16_input_1: Option<TensorBinding<R>>,
    pub(crate) f16_output_0: Option<TensorBinding<R>>,
    pub(crate) address_type: AddressType,
}

/// A checked F32 parameter tensor associated with the client that owns it.
/// Private fields prevent callers from bypassing dtype and device validation.
pub struct F32EpilogueParameters<R: Runtime> {
    client: ComputeClient<R>,
    binding: TensorBinding<R>,
}

/// Typed bindings for an F16 pointwise residual + Snake store transform.
///
/// The primary convolution output is the contiguous NHWC activation. The
/// auxiliary output is the raw contiguous NCL residual state. Keeping these
/// roles in one required argument type prevents partially configured launches.
pub struct F16ResidualSnakeStoreParameters<R: Runtime> {
    client: ComputeClient<R>,
    residual_ncl: TensorBinding<R>,
    alpha: TensorBinding<R>,
    raw_ncl: TensorBinding<R>,
}

/// Typed F16 shortcut binding for an accumulator-domain residual store.
pub struct F16ResidualStoreParameters<R: Runtime> {
    client: ComputeClient<R>,
    residual_ncl: TensorBinding<R>,
}

/// Typed F16 shortcut and Snake parameter bindings for an activated-only
/// accumulator-domain pointwise store.
pub struct F16ResidualPostCastSnakeStoreParameters<R: Runtime> {
    client: ComputeClient<R>,
    residual_ncl: TensorBinding<R>,
    alpha: TensorBinding<R>,
}

impl<R: Runtime> F16ResidualPostCastSnakeStoreParameters<R> {
    pub fn try_new(
        client: &ComputeClient<R>,
        residual_ncl: InputBinding<R>,
        alpha: InputBinding<R>,
    ) -> Result<Self, ConvSetupError> {
        let expected = half::f16::as_type_native_unchecked().storage_type();
        let normal = |binding: InputBinding<R>| match binding {
            InputBinding::Normal(binding, dtype) if dtype == expected => Ok(binding),
            InputBinding::Normal(_, _) => {
                Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDtype))
            }
            InputBinding::Quantized { .. } => Err(ConvSetupError::Epilogue(
                EpilogueSetupError::QuantizedParameter,
            )),
        };
        let residual_ncl = normal(residual_ncl)?;
        let alpha = normal(alpha)?;
        for binding in [&residual_ncl, &alpha] {
            if !is_contiguous(&binding.shape, &binding.strides) {
                return Err(ConvSetupError::Epilogue(EpilogueSetupError::NonContiguous));
            }
            let required_bytes = binding
                .size()
                .checked_mul(core::mem::size_of::<half::f16>())
                .ok_or(ConvSetupError::Unknown)?;
            let actual_bytes = binding.handle.size_in_used() as usize;
            if actual_bytes < required_bytes {
                return Err(ConvSetupError::Epilogue(
                    EpilogueSetupError::BufferTooShort {
                        required_bytes,
                        actual_bytes,
                    },
                ));
            }
        }
        Ok(Self {
            client: client.clone(),
            residual_ncl,
            alpha,
        })
    }
}

impl<R: Runtime> F16ResidualStoreParameters<R> {
    pub fn try_new(
        client: &ComputeClient<R>,
        residual_ncl: InputBinding<R>,
    ) -> Result<Self, ConvSetupError> {
        let expected = half::f16::as_type_native_unchecked().storage_type();
        let residual_ncl = match residual_ncl {
            InputBinding::Normal(binding, dtype) if dtype == expected => binding,
            InputBinding::Normal(_, _) => {
                return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDtype));
            }
            InputBinding::Quantized { .. } => {
                return Err(ConvSetupError::Epilogue(
                    EpilogueSetupError::QuantizedParameter,
                ));
            }
        };
        if !is_contiguous(&residual_ncl.shape, &residual_ncl.strides) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::NonContiguous));
        }
        let required_bytes = residual_ncl
            .size()
            .checked_mul(core::mem::size_of::<half::f16>())
            .ok_or(ConvSetupError::Unknown)?;
        let actual_bytes = residual_ncl.handle.size_in_used() as usize;
        if actual_bytes < required_bytes {
            return Err(ConvSetupError::Epilogue(
                EpilogueSetupError::BufferTooShort {
                    required_bytes,
                    actual_bytes,
                },
            ));
        }
        Ok(Self {
            client: client.clone(),
            residual_ncl,
        })
    }
}

impl<R: Runtime> F16ResidualSnakeStoreParameters<R> {
    pub fn try_new(
        client: &ComputeClient<R>,
        residual_ncl: InputBinding<R>,
        alpha: InputBinding<R>,
        raw_ncl: InputBinding<R>,
    ) -> Result<Self, ConvSetupError> {
        let expected = half::f16::as_type_native_unchecked().storage_type();
        let normal = |binding: InputBinding<R>| match binding {
            InputBinding::Normal(binding, dtype) if dtype == expected => Ok(binding),
            InputBinding::Normal(_, _) => {
                Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDtype))
            }
            InputBinding::Quantized { .. } => Err(ConvSetupError::Epilogue(
                EpilogueSetupError::QuantizedParameter,
            )),
        };
        let residual_ncl = normal(residual_ncl)?;
        let alpha = normal(alpha)?;
        let raw_ncl = normal(raw_ncl)?;
        for binding in [&residual_ncl, &alpha, &raw_ncl] {
            if !is_contiguous(&binding.shape, &binding.strides) {
                return Err(ConvSetupError::Epilogue(EpilogueSetupError::NonContiguous));
            }
            let required_bytes = binding
                .size()
                .checked_mul(core::mem::size_of::<half::f16>())
                .ok_or(ConvSetupError::Unknown)?;
            let actual_bytes = binding.handle.size_in_used() as usize;
            if actual_bytes < required_bytes {
                return Err(ConvSetupError::Epilogue(
                    EpilogueSetupError::BufferTooShort {
                        required_bytes,
                        actual_bytes,
                    },
                ));
            }
        }
        Ok(Self {
            client: client.clone(),
            residual_ncl,
            alpha,
            raw_ncl,
        })
    }
}

impl<R: Runtime> F32EpilogueParameters<R> {
    pub fn try_new(
        client: &ComputeClient<R>,
        parameter: InputBinding<R>,
    ) -> Result<Self, ConvSetupError> {
        let (binding, dtype) = match parameter {
            InputBinding::Normal(binding, dtype) => (binding, dtype),
            InputBinding::Quantized { .. } => {
                return Err(ConvSetupError::Epilogue(
                    EpilogueSetupError::QuantizedParameter,
                ));
            }
        };
        if dtype != f32::as_type_native_unchecked().storage_type() {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDtype));
        }
        if !is_contiguous(&binding.shape, &binding.strides) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::NonContiguous));
        }
        let offset = binding.handle.offset_start.unwrap_or(0) as usize;
        if !offset.is_multiple_of(core::mem::align_of::<f32>()) {
            return Err(ConvSetupError::Epilogue(
                EpilogueSetupError::MisalignedOffset,
            ));
        }
        let required_bytes = binding.size() * core::mem::size_of::<f32>();
        let actual_bytes = binding.handle.size_in_used() as usize;
        if actual_bytes < required_bytes {
            return Err(ConvSetupError::Epilogue(
                EpilogueSetupError::BufferTooShort {
                    required_bytes,
                    actual_bytes,
                },
            ));
        }
        Ok(Self {
            client: client.clone(),
            binding,
        })
    }
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim > 1 && stride != expected {
            return false;
        }
        expected *= dim;
    }
    true
}

/// DAC-style Snake activation using a channel-indexed parameter vector.
///
/// Computation is deliberately promoted to f32 even when the convolution
/// output is f16. This matches a standalone mixed-precision Snake kernel while
/// avoiding its dispatch and intermediate global-memory round trip.
pub struct SnakeEpilogue;

impl PostCastEpilogueSpec for SnakeEpilogue {
    type LaunchArgs<R: Runtime> = F32EpilogueParameters<R>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        if !core::ptr::eq(client.info(), args.client.info()) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDevice));
        }
        let actual = args.binding.size();
        if actual < problem.out_channels {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::TooShort {
                required: problem.out_channels,
                actual,
            }));
        }
        let address_type = args
            .binding
            .required_address_type(core::mem::size_of::<f32>());
        Ok(PreparedPostCastEpilogue {
            f32_param: Some(args.binding),
            f16_input_0: None,
            f16_input_1: None,
            f16_output_0: None,
            address_type,
        })
    }
}

#[cube]
impl PostCastGlobalEpilogue<RuntimeArgs> for SnakeEpilogue {
    fn apply<E: Numeric>(value: E, coordinate: (u32, u32), runtime_config: &RuntimeArgs) -> E {
        let alpha = runtime_config.epilogue.f32_param.unwrap();
        let x = f32::cast_from(value);
        let a = alpha.read(coordinate.1 as usize);
        let sine = (a * x).sin();
        E::cast_from(x + sine * sine / (a + 1.0e-9))
    }
}

/// Snake activation using a prepared interleaved `[alpha, reciprocal]`
/// parameter pair for each output channel.
///
/// The reciprocal is computed once while preparing the model, replacing a
/// division in every output element with a multiplication. This is a distinct
/// numerical contract from [`SnakeEpilogue`] because the reciprocal is rounded
/// once to f32 before reuse.
pub struct PreparedSnakeEpilogue;

impl PostCastEpilogueSpec for PreparedSnakeEpilogue {
    type LaunchArgs<R: Runtime> = F32EpilogueParameters<R>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        if !core::ptr::eq(client.info(), args.client.info()) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDevice));
        }
        let required = problem
            .out_channels
            .checked_mul(2)
            .ok_or(ConvSetupError::Unknown)?;
        let actual = args.binding.size();
        if actual < required {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::TooShort {
                required,
                actual,
            }));
        }
        let address_type = args
            .binding
            .required_address_type(core::mem::size_of::<f32>());
        Ok(PreparedPostCastEpilogue {
            f32_param: Some(args.binding),
            f16_input_0: None,
            f16_input_1: None,
            f16_output_0: None,
            address_type,
        })
    }
}

#[cube]
impl PostCastGlobalEpilogue<RuntimeArgs> for PreparedSnakeEpilogue {
    fn apply<E: Numeric>(value: E, coordinate: (u32, u32), runtime_config: &RuntimeArgs) -> E {
        let parameters = runtime_config.epilogue.f32_param.unwrap();
        let parameter_offset = coordinate.1 as usize * 2;
        let a = parameters.read(parameter_offset);
        let reciprocal = parameters.read(parameter_offset + 1);
        let x = f32::cast_from(value);
        let sine = (a * x).sin();
        E::cast_from(x + sine * sine * reciprocal)
    }
}

/// Accumulator-domain projection store used by the pointwise differential
/// profile. Bias has already been accumulated by CubeK. This transform adds
/// the NCL shortcut in f32, stores raw NCL once, evaluates Snake from the same
/// unrounded raw value, and returns the activated NHWC primary output.
pub struct F16ResidualSnakeStore;

impl PostCastEpilogueSpec for F16ResidualSnakeStore {
    type LaunchArgs<R: Runtime> = F16ResidualSnakeStoreParameters<R>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        if !core::ptr::eq(client.info(), args.client.info()) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDevice));
        }
        let output_rows = problem.out_shape.iter().product::<usize>();
        let raw_elements = problem
            .batches
            .checked_mul(problem.out_channels)
            .and_then(|value| value.checked_mul(output_rows))
            .ok_or(ConvSetupError::Unknown)?;
        if problem.dimensionality.num_dims() != 1
            || problem.kernel_size != [1]
            || args.residual_ncl.size() != raw_elements
            || args.raw_ncl.size() != raw_elements
            || args.alpha.size() < problem.out_channels
        {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::TooShort {
                required: raw_elements,
                actual: args.residual_ncl.size().min(args.raw_ncl.size()),
            }));
        }
        let address_type = args
            .residual_ncl
            .required_address_type(core::mem::size_of::<half::f16>())
            .max(
                args.alpha
                    .required_address_type(core::mem::size_of::<half::f16>()),
            )
            .max(
                args.raw_ncl
                    .required_address_type(core::mem::size_of::<half::f16>()),
            );
        Ok(PreparedPostCastEpilogue {
            f32_param: None,
            f16_input_0: Some(args.residual_ncl),
            f16_input_1: Some(args.alpha),
            f16_output_0: Some(args.raw_ncl),
            address_type,
        })
    }
}

/// Accumulator-domain pointwise store that adds an NCL shortcut and returns
/// the raw result through the caller-provided logical output view.
pub struct F16ResidualStore;

impl PostCastEpilogueSpec for F16ResidualStore {
    type LaunchArgs<R: Runtime> = F16ResidualStoreParameters<R>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        if !core::ptr::eq(client.info(), args.client.info()) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDevice));
        }
        let output_rows = problem.out_shape.iter().product::<usize>();
        let output_elements = problem
            .batches
            .checked_mul(problem.out_channels)
            .and_then(|value| value.checked_mul(output_rows))
            .ok_or(ConvSetupError::Unknown)?;
        if problem.dimensionality.num_dims() != 1
            || problem.kernel_size != [1]
            || args.residual_ncl.size() != output_elements
        {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::TooShort {
                required: output_elements,
                actual: args.residual_ncl.size(),
            }));
        }
        let address_type = args
            .residual_ncl
            .required_address_type(core::mem::size_of::<half::f16>());
        Ok(PreparedPostCastEpilogue {
            f32_param: None,
            f16_input_0: Some(args.residual_ncl),
            f16_input_1: None,
            f16_output_0: None,
            address_type,
        })
    }
}

/// Accumulator-domain pointwise store that adds an NCL shortcut, preserves
/// the former F16 storage boundary, and returns only the Snake activation.
pub struct F16ResidualPostCastSnakeStore;

impl PostCastEpilogueSpec for F16ResidualPostCastSnakeStore {
    type LaunchArgs<R: Runtime> = F16ResidualPostCastSnakeStoreParameters<R>;

    fn prepare<R: Runtime>(
        client: &ComputeClient<R>,
        args: Self::LaunchArgs<R>,
        problem: &ConvolutionProblem,
    ) -> Result<PreparedPostCastEpilogue<R>, ConvSetupError> {
        if !core::ptr::eq(client.info(), args.client.info()) {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::WrongDevice));
        }
        let output_rows = problem.out_shape.iter().product::<usize>();
        let output_elements = problem
            .batches
            .checked_mul(problem.out_channels)
            .and_then(|value| value.checked_mul(output_rows))
            .ok_or(ConvSetupError::Unknown)?;
        if problem.dimensionality.num_dims() != 1
            || problem.kernel_size != [1]
            || args.residual_ncl.size() != output_elements
            || args.alpha.size() < problem.out_channels
        {
            return Err(ConvSetupError::Epilogue(EpilogueSetupError::TooShort {
                required: output_elements,
                actual: args.residual_ncl.size(),
            }));
        }
        let address_type = args
            .residual_ncl
            .required_address_type(core::mem::size_of::<half::f16>())
            .max(
                args.alpha
                    .required_address_type(core::mem::size_of::<half::f16>()),
            );
        Ok(PreparedPostCastEpilogue {
            f32_param: None,
            f16_input_0: Some(args.residual_ncl),
            f16_input_1: Some(args.alpha),
            f16_output_0: None,
            address_type,
        })
    }
}

#[cube]
impl AccumulatorGlobalStoreTransform<RuntimeArgs> for F16ResidualStore {
    fn apply<ES: Numeric, EG: Numeric>(
        value: ES,
        coordinate: (u32, u32),
        runtime_config: &mut RuntimeArgs,
    ) -> EG {
        let residual = runtime_config.epilogue.f16_input_0.unwrap();
        let rows = runtime_config.epilogue.output_rows;
        let batch = coordinate.0 / rows;
        let time = coordinate.0 - batch * rows;
        let channel = coordinate.1;
        let ncl_index = (batch * runtime_config.channels + channel) * rows + time;
        EG::cast_from(
            f32::cast_from(value) + f32::cast_from(residual.read(ncl_index as usize)),
        )
    }
}

#[cube]
impl AccumulatorGlobalStoreTransform<RuntimeArgs> for F16ResidualSnakeStore {
    fn apply<ES: Numeric, EG: Numeric>(
        value: ES,
        coordinate: (u32, u32),
        runtime_config: &mut RuntimeArgs,
    ) -> EG {
        let residual = runtime_config.epilogue.f16_input_0.unwrap();
        let alpha = runtime_config.epilogue.f16_input_1.unwrap();
        let mut raw_output = runtime_config.epilogue.f16_output_0.unwrap();
        let rows = runtime_config.epilogue.output_rows;
        let batch = coordinate.0 / rows;
        let time = coordinate.0 - batch * rows;
        let channel = coordinate.1;
        let ncl_index = (batch * runtime_config.channels + channel) * rows + time;
        let raw = f32::cast_from(value) + f32::cast_from(residual.read(ncl_index as usize));
        raw_output.write(ncl_index as usize, half::f16::cast_from(raw));
        let a = f32::cast_from(alpha.read(channel as usize));
        let sine = (a * raw).sin();
        EG::cast_from(raw + sine * sine / (a + 1.0e-9))
    }
}

#[cube]
impl AccumulatorGlobalStoreTransform<RuntimeArgs> for F16ResidualPostCastSnakeStore {
    fn apply<ES: Numeric, EG: Numeric>(
        value: ES,
        coordinate: (u32, u32),
        runtime_config: &mut RuntimeArgs,
    ) -> EG {
        let residual = runtime_config.epilogue.f16_input_0.unwrap();
        let alpha = runtime_config.epilogue.f16_input_1.unwrap();
        let rows = runtime_config.epilogue.output_rows;
        let batch = coordinate.0 / rows;
        let time = coordinate.0 - batch * rows;
        let channel = coordinate.1;
        let ncl_index = (batch * runtime_config.channels + channel) * rows + time;
        let raw = half::f16::cast_from(
            f32::cast_from(value) + f32::cast_from(residual.read(ncl_index as usize)),
        );
        let x = f32::cast_from(raw);
        let a = f32::cast_from(alpha.read(channel as usize));
        let sine = (a * x).sin();
        EG::cast_from(x + sine * sine / (a + 1.0e-9))
    }
}

#[cfg(test)]
mod tests {
    use super::is_contiguous;

    #[test]
    fn epilogue_parameters_require_contiguous_storage() {
        assert!(is_contiguous(&[1, 96, 1], &[96, 1, 1]));
        assert!(is_contiguous(&[96], &[1]));
        assert!(!is_contiguous(&[96, 2], &[1, 96]));
        assert!(!is_contiguous(&[96], &[2]));
        assert!(!is_contiguous(&[96], &[]));
    }
}
