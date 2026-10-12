//! Opt-in Burn custom Fusion providers for the WGPU runtime.
//!
//! The production backend remains [`crate::WgpuRaw`] while the graph bridge is
//! accuracy/performance tested.  This module nevertheless uses Burn 0.22's
//! real provider registry and exposes a WGPU-only fusion backend so migration
//! can happen one graph boundary at a time without introducing CPU/CUDA
//! dispatch.

use burn_cubecl::{
    CubeRuntime,
    fusion::{
        CubeFuser, CubeOptimization, FallbackOperation, FusedOperation, OptimizationProvider,
        RegistryError,
    },
};
use burn_cubecl_fusion::optim::elemwise::ElementWiseFuser;
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser, Optimization, stream::Context};
use burn_ir::{ActivationOperationIr, NumericOperationIr, OperationIr, TensorId};

type WgpuRuntime = burn::backend::wgpu::WgpuRuntime;

/// WGPU-only Fusion backend used by opt-in graph migration tests.
pub type WgpuFusion = burn_fusion::Fusion<crate::WgpuRaw>;

pub const SWIGLU_POSTPROCESS_PROVIDER: &str = "IrodoriSwiGluPostprocessV1";

/// Register all Irodori providers before the first [`WgpuFusion`] operation.
pub fn register_irodori_fusion_providers() -> Result<(), RegistryError> {
    burn_cubecl::fusion::register::<WgpuRuntime>(SwiGluPostprocessProvider)
}

/// A focused provider for `sigmoid(gate) * gate`, followed by multiplication
/// with the value half. The generated operation delegates code generation to
/// Burn's portable elementwise fusion engine, while owning a stable Irodori
/// name/state and only competing once the complete SwiGLU data dependency is
/// observed. This is the first safe bridge; projection epilogues can replace
/// its executor later without changing registry or serialized-plan contracts.
pub struct SwiGluPostprocessProvider;

impl<R: CubeRuntime> OptimizationProvider<R> for SwiGluPostprocessProvider {
    type Operation = IrodoriSwiGluPostprocess<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        CubeFuser::new(SwiGluPostprocessFuser::new(device.clone()))
    }
}

pub struct IrodoriSwiGluPostprocess<R: CubeRuntime> {
    inner: CubeOptimization<R>,
}

impl<R: CubeRuntime> FusedOperation<R> for IrodoriSwiGluPostprocess<R> {
    const NAME: &'static str = SWIGLU_POSTPROCESS_PROVIDER;
    type State = burn_cubecl::fusion::CubeOptimizationState;

    fn num_ops_fused(&self) -> usize {
        self.inner.num_ops_fused()
    }

    fn run(
        &mut self,
        context: &mut Context<burn_cubecl_fusion::CubeFusionHandle<R>>,
        fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        self.inner.run(context, fallback);
    }

    fn to_state(&self) -> Self::State {
        self.inner.to_state()
    }

    fn from_state(device: &R::Device, state: Self::State) -> Self {
        let inner = <CubeOptimization<R> as Optimization<
            burn_cubecl::fusion::FusionCubeRuntime<R>,
        >>::from_state(device, state);
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

pub struct SwiGluPostprocessFuser<R: CubeRuntime> {
    inner: ElementWiseFuser<R>,
    pattern: SwiGluPattern,
}

impl<R: CubeRuntime> Clone for SwiGluPostprocessFuser<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            pattern: self.pattern,
        }
    }
}

impl<R: CubeRuntime> SwiGluPostprocessFuser<R> {
    pub fn new(device: R::Device) -> Self {
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

impl<R: CubeRuntime> OperationFuser<CubeOptimization<R>> for SwiGluPostprocessFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        self.observe(operation);
        self.inner.fuse(operation);
    }

    fn finish(&mut self) -> CubeOptimization<R> {
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

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
