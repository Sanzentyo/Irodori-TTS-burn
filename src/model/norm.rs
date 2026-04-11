use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::silu, backend::Backend},
};

/// RMS Layer Normalisation over the last dimension.
///
/// `weight` shape: `[dim]`. Operates on 3-D tensors `[batch, seq, dim]`.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub(crate) weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            eps,
        }
    }

    /// `x`: `[batch, seq, dim]`
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rms = (x.clone() * x.clone())
            .mean_dim(2) // [B, S, 1] (keepdim)
            .add_scalar(self.eps as f32)
            .sqrt(); // [B, S, 1]
        // weight: [D] broadcasts to [1, 1, D] via burn's automatic broadcasting
        let w: Tensor<B, 3> = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0) // [1, D]
            .unsqueeze_dim::<3>(0); // [1, 1, D]
        x / rms * w
    }
}

/// Two-dimensional RMS norm for Q/K head normalisation.
///
/// `weight` shape: `[heads, head_dim]`. Operates on 4-D `[batch, seq, heads, head_dim]`.
#[derive(Module, Debug)]
pub struct HeadRmsNorm<B: Backend> {
    pub(crate) weight: Param<Tensor<B, 2>>,
    eps: f64,
}

impl<B: Backend> HeadRmsNorm<B> {
    pub fn new(heads: usize, head_dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([heads, head_dim], device)),
            eps,
        }
    }

    /// `x`: `[batch, seq, heads, head_dim]` — normalise over `head_dim` (dim 3).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let rms = (x.clone() * x.clone())
            .mean_dim(3) // [B, S, H, 1]
            .add_scalar(self.eps as f32)
            .sqrt(); // [B, S, H, 1]
        // weight: [H, D_h] → [1, 1, H, D_h]
        let w: Tensor<B, 4> = self
            .weight
            .val()
            .unsqueeze_dim::<3>(0) // [1, H, D_h]
            .unsqueeze_dim::<4>(0); // [1, 1, H, D_h]
        x / rms * w
    }
}

/// Echo-style low-rank Adaptive Layer Norm.
///
/// Given `cond_embed: [B, 1, model_dim * 3]`, produces:
/// - `h: [B, S, D]` — RMSNorm(x) modulated by scale/shift
/// - `gate: [B, 1, D]` — tanh gate for the residual path
///
/// Field names match the Python state_dict exactly:
/// `shift_down`, `scale_down`, `gate_down`, `shift_up`, `scale_up`, `gate_up`.
#[derive(Module, Debug)]
pub struct LowRankAdaLn<B: Backend> {
    pub(crate) shift_down: Linear<B>,
    pub(crate) scale_down: Linear<B>,
    pub(crate) gate_down: Linear<B>,
    pub(crate) shift_up: Linear<B>,
    pub(crate) scale_up: Linear<B>,
    pub(crate) gate_up: Linear<B>,
    eps: f64,
}

impl<B: Backend> LowRankAdaLn<B> {
    pub fn new(model_dim: usize, rank: usize, eps: f64, device: &B::Device) -> Self {
        let rank = rank.max(1).min(model_dim);

        // down projections: model_dim → rank, no bias
        let mk_down = || {
            LinearConfig::new(model_dim, rank)
                .with_bias(false)
                .init(device)
        };

        // up projections: rank → model_dim, with bias — zero-initialised
        let mk_up_zero = || {
            let mut l = LinearConfig::new(rank, model_dim)
                .with_bias(true)
                .init::<B>(device);
            l.weight = Param::initialized(ParamId::new(), Tensor::zeros([model_dim, rank], device));
            l.bias = Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([model_dim], device),
            ));
            l
        };

        Self {
            shift_down: mk_down(),
            scale_down: mk_down(),
            gate_down: mk_down(),
            shift_up: mk_up_zero(),
            scale_up: mk_up_zero(),
            gate_up: mk_up_zero(),
            eps,
        }
    }

    /// Returns `(modulated_x, gate)`.
    ///
    /// `x: [B, S, D]`, `cond_embed: [B, 1, D*3]`
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond_embed: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Split into shift / scale / gate: each [B, 1, D]
        let chunks = cond_embed.chunk(3, 2);
        let (raw_shift, raw_scale, raw_gate) =
            (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

        // Low-rank residual refinement
        let shift = self
            .shift_up
            .forward(self.shift_down.forward(silu(raw_shift.clone())))
            + raw_shift;
        let scale = self
            .scale_up
            .forward(self.scale_down.forward(silu(raw_scale.clone())))
            + raw_scale;
        let gate = self
            .gate_up
            .forward(self.gate_down.forward(silu(raw_gate.clone())))
            + raw_gate;

        // RMSNorm x: [B, S, D]
        let rms = (x.clone() * x.clone())
            .mean_dim(2)
            .add_scalar(self.eps as f32)
            .sqrt(); // [B, S, 1]
        let x_norm = x / rms;

        // Modulate: x_norm * (1 + scale) + shift   — broadcast [B,1,D] over [B,S,D]
        let modulated = x_norm * (scale + 1.0) + shift;
        let gate_out = gate.tanh();

        (modulated, gate_out)
    }
}
