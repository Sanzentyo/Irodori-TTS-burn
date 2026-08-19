//! Convolution epilogues implemented on CubeK's generic matmul writer hook.

use cubecl::{Runtime, client::ComputeClient, ir::AddressType, prelude::*};
use cubek_matmul::components::global::PostCastGlobalEpilogue;
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
    pub(crate) binding: TensorBinding<R>,
    pub(crate) address_type: AddressType,
}

/// A checked F32 parameter tensor associated with the client that owns it.
/// Private fields prevent callers from bypassing dtype and device validation.
pub struct F32EpilogueParameters<R: Runtime> {
    client: ComputeClient<R>,
    binding: TensorBinding<R>,
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
            binding: args.binding,
            address_type,
        })
    }
}

#[cube]
impl PostCastGlobalEpilogue<RuntimeArgs> for SnakeEpilogue {
    fn apply<E: Numeric>(value: E, coordinate: (u32, u32), runtime_config: &RuntimeArgs) -> E {
        let alpha = runtime_config.epilogue_param.unwrap();
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
            binding: args.binding,
            address_type,
        })
    }
}

#[cube]
impl PostCastGlobalEpilogue<RuntimeArgs> for PreparedSnakeEpilogue {
    fn apply<E: Numeric>(value: E, coordinate: (u32, u32), runtime_config: &RuntimeArgs) -> E {
        let parameters = runtime_config.epilogue_param.unwrap();
        let parameter_offset = coordinate.1 as usize * 2;
        let a = parameters.read(parameter_offset);
        let reciprocal = parameters.read(parameter_offset + 1);
        let x = f32::cast_from(value);
        let sine = (a * x).sin();
        E::cast_from(x + sine * sine * reciprocal)
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
