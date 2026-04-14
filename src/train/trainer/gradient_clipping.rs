//! Global gradient norm clipping (PyTorch-compatible).

use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    prelude::ElementConversion,
    tensor::{Tensor, backend::AutodiffBackend},
};

use crate::train::LoraTextToLatentRfDiT;

type IB<B> = <B as AutodiffBackend>::InnerBackend;

/// Clip gradients by global L2 norm, matching PyTorch's `clip_grad_norm_`.
///
/// Computes the total L2 norm across **all** parameters, then scales each
/// gradient by `min(1, max_norm / (total_norm + ε))` if the total norm exceeds
/// `max_norm`.
///
/// This differs from burn's built-in per-parameter clipping — here we compute
/// one global norm and apply one uniform scale factor to all gradients.
pub(super) fn clip_grad_norm_global<B: AutodiffBackend>(
    mut grads: GradientsParams,
    model: &LoraTextToLatentRfDiT<B>,
    max_norm: f64,
) -> GradientsParams {
    // Pass 1: compute total squared norm.
    struct NormComputer<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sq_sum: f64,
        _b: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormComputer<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.get::<IB<B>, D>(param.id) {
                let sq: f64 = grad.clone().mul(grad).sum().into_scalar().elem();
                self.sq_sum += sq;
            }
        }
    }

    let mut computer = NormComputer::<B> {
        grads: &grads,
        sq_sum: 0.0,
        _b: std::marker::PhantomData,
    };
    model.visit(&mut computer);

    let total_norm = computer.sq_sum.sqrt();
    let clip_coef = (max_norm / (total_norm + 1e-6)).min(1.0);

    if clip_coef >= 1.0 {
        return grads;
    }

    // Pass 2: scale all gradients by clip_coef.
    struct NormScaler<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _b: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormScaler<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<IB<B>, D>(param.id) {
                self.grads.register::<IB<B>, D>(param.id, grad * self.scale);
            }
        }
    }

    let mut scaler = NormScaler::<B> {
        grads: &mut grads,
        scale: clip_coef as f32,
        _b: std::marker::PhantomData,
    };
    model.visit(&mut scaler);

    grads
}
