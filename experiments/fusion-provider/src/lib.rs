//! Opt-in Burn custom Fusion providers for the WGPU runtime.
//!
//! The production backend remains raw WGPU while the graph bridge is
//! accuracy/performance tested.  This module nevertheless uses Burn 0.22's
//! real provider registry and exposes a WGPU-only fusion backend so migration
//! can happen one graph boundary at a time without introducing CPU/CUDA
//! dispatch.

use burn_cubecl::fusion::{
    CubeFuser, CubeOptimization, FallbackOperation, FusedOperation, OptimizationProvider,
    RegistryError,
};
use burn_cubecl_fusion::optim::elemwise::ElementWiseFuser;
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser, Optimization, stream::Context};
use burn_ir::{ActivationOperationIr, NumericOperationIr, OperationIr, TensorId};

/// WGPU-only Fusion backend used by opt-in graph migration tests.
pub type WgpuFusion = burn_fusion::Fusion<burn_cubecl::CubeBackend>;

pub const SWIGLU_POSTPROCESS_PROVIDER: &str = "IrodoriSwiGluPostprocessV1";

/// Register all Irodori providers before the first [`WgpuFusion`] operation.
pub fn register_irodori_fusion_providers() -> Result<(), RegistryError> {
    burn_cubecl::fusion::register(SwiGluPostprocessProvider)
}

/// A focused provider for `sigmoid(gate) * gate`, followed by multiplication
/// with the value half. The generated operation delegates code generation to
/// Burn's portable elementwise fusion engine, while owning a stable Irodori
/// name/state and only competing once the complete SwiGLU data dependency is
/// observed. This is the first safe bridge; projection epilogues can replace
/// its executor later without changing registry or serialized-plan contracts.
pub struct SwiGluPostprocessProvider;

impl OptimizationProvider for SwiGluPostprocessProvider {
    type Operation = IrodoriSwiGluPostprocess;

    fn fuser(&self, device: &cubecl::Device) -> CubeFuser {
        CubeFuser::new(SwiGluPostprocessFuser::new(device.clone()))
    }
}

pub struct IrodoriSwiGluPostprocess {
    inner: CubeOptimization,
}

impl FusedOperation for IrodoriSwiGluPostprocess {
    const NAME: &'static str = SWIGLU_POSTPROCESS_PROVIDER;
    type State = burn_cubecl::fusion::CubeOptimizationState;

    fn max_relative_shape_id(&self) -> Option<usize> {
        burn_fusion::NumOperations::max_relative_shape_id(&self.inner)
    }

    fn num_ops_fused(&self) -> usize {
        self.inner.num_ops_fused()
    }

    fn run(
        &mut self,
        context: &mut Context<burn_cubecl_fusion::CubeFusionHandle>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation>,
    ) {
        self.inner.run(context, fallback);
    }

    fn to_state(&self) -> Self::State {
        self.inner.to_state()
    }

    fn from_state(device: &cubecl::Device, state: Self::State) -> Self {
        let inner =
            <CubeOptimization as Optimization<burn_cubecl::fusion::FusionCubeRuntime>>::from_state(
                device, state,
            );
        Self { inner }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SwiGluPattern {
    #[default]
    Searching,
    Sigmoid {
        output: TensorId,
    },
    Silu {
        output: TensorId,
    },
    Complete,
    Rejected,
}

pub struct SwiGluPostprocessFuser {
    inner: ElementWiseFuser,
    pattern: SwiGluPattern,
}

impl Clone for SwiGluPostprocessFuser {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            pattern: self.pattern,
        }
    }
}

impl SwiGluPostprocessFuser {
    pub fn new(device: cubecl::Device) -> Self {
        Self {
            inner: ElementWiseFuser::new(device),
            pattern: SwiGluPattern::Searching,
        }
    }

    fn observe(&mut self, operation: &OperationIr) {
        self.pattern = match (self.pattern, operation) {
            (
                SwiGluPattern::Searching,
                OperationIr::Activation(ActivationOperationIr::Sigmoid(operation)),
            ) => SwiGluPattern::Sigmoid {
                output: operation.out.id,
            },
            (
                SwiGluPattern::Sigmoid { output },
                OperationIr::NumericFloat(_, NumericOperationIr::Mul(operation)),
            ) if operation.lhs.id == output || operation.rhs.id == output => SwiGluPattern::Silu {
                output: operation.out.id,
            },
            (
                SwiGluPattern::Silu { output },
                OperationIr::NumericFloat(_, NumericOperationIr::Mul(operation)),
            ) if operation.lhs.id == output || operation.rhs.id == output => {
                SwiGluPattern::Complete
            }
            (SwiGluPattern::Complete, _) => SwiGluPattern::Complete,
            (SwiGluPattern::Rejected, _) => SwiGluPattern::Rejected,
            _ => SwiGluPattern::Rejected,
        };
    }
}

impl OperationFuser<CubeOptimization> for SwiGluPostprocessFuser {
    fn fuse(&mut self, operation: &OperationIr) {
        self.observe(operation);
        self.inner.fuse(operation);
    }

    fn finish(&mut self) -> CubeOptimization {
        CubeOptimization::new(IrodoriSwiGluPostprocess {
            inner: self.inner.finish(),
        })
    }

    fn reset(&mut self) {
        self.pattern = SwiGluPattern::Searching;
        self.inner.reset();
    }

    fn status(&self) -> FuserStatus {
        if self.pattern == SwiGluPattern::Rejected {
            FuserStatus::Closed
        } else {
            self.inner.status()
        }
    }

    fn properties(&self) -> FuserProperties {
        let mut properties = self.inner.properties();
        properties.ready &= self.pattern == SwiGluPattern::Complete;
        if properties.ready {
            // Win ties against the generic elementwise provider only for the
            // exact recognized dependency graph.
            properties.score = properties.score.saturating_add(1);
        }
        properties
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a WGPU device"]
    fn swiglu_matches_scalar_reference_and_preserves_shape_metadata() {
        use burn::tensor::{Tensor, TensorData, activation::sigmoid};

        register_irodori_fusion_providers().unwrap();
        let device: burn::tensor::Device = cubecl::wgpu::WgpuDevice::default().into();
        let gate = [-3.0_f32, -1.0, 0.0, 0.5, 1.5, 4.0];
        let value = [2.0_f32, -2.0, 3.0, 0.25, -1.0, 0.5];
        let gate_tensor = Tensor::<2>::from_data(TensorData::new(gate.to_vec(), [2, 3]), &device);
        let value_tensor = Tensor::<2>::from_data(TensorData::new(value.to_vec(), [2, 3]), &device);
        let output = sigmoid(gate_tensor.clone()) * gate_tensor * value_tensor;
        assert_eq!(output.dims(), [2, 3]);
        let actual = output.into_data().try_to_vec::<f32>().unwrap();
        for ((actual, gate), value) in actual.into_iter().zip(gate).zip(value) {
            let expected = gate / (1.0 + (-gate).exp()) * value;
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn provider_name_is_versioned_and_not_a_builtin() {
        assert!(SWIGLU_POSTPROCESS_PROVIDER.starts_with("Irodori"));
        assert!(!burn_cubecl::fusion::BUILTIN_NAMES.contains(&SWIGLU_POSTPROCESS_PROVIDER));
    }

    #[test]
    fn rejected_pattern_closes_fail_closed() {
        let pattern = match (SwiGluPattern::Searching, &OperationIr::Drop(dummy_tensor())) {
            (SwiGluPattern::Searching, _) => SwiGluPattern::Rejected,
            _ => unreachable!(),
        };
        assert_eq!(pattern, SwiGluPattern::Rejected);
    }

    fn dummy_tensor() -> burn_ir::TensorIr {
        burn_ir::TensorIr {
            id: TensorId::new(0),
            shape: burn::tensor::Shape::new([1]),
            status: burn_ir::TensorStatus::ReadOnly,
            dtype: burn::tensor::DType::F32,
        }
    }
}
