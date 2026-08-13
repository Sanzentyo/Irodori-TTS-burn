use burn::tensor::Device;
use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{FloatDType, Tensor, activation::silu},
};

/// RMS Layer Normalisation over the last dimension.
///
/// `weight` shape: `[dim]`. Operates on 3-D tensors `[batch, seq, dim]`.
#[derive(Module, Debug)]
pub struct RmsNorm {
    pub(crate) weight: Param<Tensor<1>>,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            eps,
        }
    }

    /// `x`: `[batch, seq, dim]`
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let output_dtype: FloatDType = x.dtype().into();
        // Reference RMSNorm accumulates in float32 and casts only its result.
        let x = x.cast(FloatDType::F32);
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2) // [B, S, 1] (keepdim)
            .add_scalar(self.eps)
            .sqrt(); // [B, S, 1]
        // weight: [D] broadcasts to [1, 1, D] via burn's automatic broadcasting
        let w: Tensor<3> = self
            .weight
            .val()
            .cast(FloatDType::F32)
            .unsqueeze_dim::<2>(0) // [1, D]
            .unsqueeze_dim::<3>(0); // [1, 1, D]
        (x / rms * w).cast(output_dtype)
    }

    pub(crate) const fn epsilon(&self) -> f64 {
        self.eps
    }
}

impl RmsNorm {
    /// RMSNorm through the measured single-pass WGSL reduction kernel.
    pub(crate) fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, dim] = x.dims();
        let output = crate::kernels::rms_norm::rms_norm_wgsl(
            x.reshape([batch * seq_len, dim])
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            self.weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            self.eps,
        );
        Tensor::<2>::from_primitive::<crate::WgpuRaw>(output).reshape([batch, seq_len, dim])
    }
}

/// Two-dimensional RMS norm for Q/K head normalisation.
///
/// `weight` shape: `[heads, head_dim]`. Operates on 4-D `[batch, seq, heads, head_dim]`.
#[derive(Module, Debug)]
pub struct HeadRmsNorm {
    pub(crate) weight: Param<Tensor<2>>,
    eps: f64,
}

impl HeadRmsNorm {
    pub fn new(heads: usize, head_dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([heads, head_dim], device)),
            eps,
        }
    }

    /// `x`: `[batch, seq, heads, head_dim]` — normalise over `head_dim` (dim 3).
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let output_dtype: FloatDType = x.dtype().into();
        let x = x.cast(FloatDType::F32);
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3) // [B, S, H, 1]
            .add_scalar(self.eps)
            .sqrt(); // [B, S, H, 1]
        // weight: [H, D_h] → [1, 1, H, D_h]
        let w: Tensor<4> = self
            .weight
            .val()
            .cast(FloatDType::F32)
            .unsqueeze_dim::<3>(0) // [1, H, D_h]
            .unsqueeze_dim::<4>(0); // [1, 1, H, D_h]
        (x / rms * w).cast(output_dtype)
    }

    pub(crate) const fn epsilon(&self) -> f64 {
        self.eps
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
pub struct LowRankAdaLn {
    pub(crate) shift_down: Linear,
    pub(crate) scale_down: Linear,
    pub(crate) gate_down: Linear,
    pub(crate) shift_up: Linear,
    pub(crate) scale_up: Linear,
    pub(crate) gate_up: Linear,
    eps: f64,
    /// Legacy record-skipped slots remain empty in production. The WGSL
    /// wrapper owns the single cross-layer replacement cache instead.
    #[module(skip)]
    packed_down: Option<Tensor<4>>,
    #[module(skip)]
    packed_up: Option<Tensor<4>>,
    #[module(skip)]
    packed_bias: Option<Tensor<4>>,
}

/// Shift/scale/gate tensors after the low-rank residual projection.
#[derive(Debug)]
pub(crate) struct AdaLnModulation {
    pub(crate) shift: Tensor<3>,
    pub(crate) scale: Tensor<3>,
    pub(crate) gate: Tensor<3>,
}

impl AdaLnModulation {
    fn matches_condition(&self, cond_embed: &Tensor<3>) -> bool {
        let [batch, sequence, width] = cond_embed.dims();
        if sequence != 1 || width == 0 || !width.is_multiple_of(3) {
            return false;
        }
        let model_dim = width / 3;
        let device = cond_embed.device();
        [&self.shift, &self.scale, &self.gate]
            .into_iter()
            .all(|tensor| {
                tensor.dims() == [batch, 1, model_dim]
                    && tensor.dtype() == cond_embed.dtype()
                    && tensor.device() == device.clone()
            })
    }
}

impl LowRankAdaLn {
    pub fn new(model_dim: usize, rank: usize, eps: f64, device: &Device) -> Self {
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
                .init(device);
            // Row layout: weight shape is [d_input=rank, d_output=model_dim]
            l.weight = Param::initialized(ParamId::new(), Tensor::zeros([rank, model_dim], device));
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
            packed_down: None,
            packed_up: None,
            packed_bias: None,
        }
    }

    /// Returns `(modulated_x, gate)`.
    ///
    /// `x: [B, S, D]`, `cond_embed: [B, 1, D*3]`
    pub fn forward(&self, x: Tensor<3>, cond_embed: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let (shift, scale, gate) = self.modulation(cond_embed);

        // RMSNorm x: [B, S, D]
        let output_dtype: FloatDType = x.dtype().into();
        let x = x.cast(FloatDType::F32);
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(self.eps)
            .sqrt(); // [B, S, 1]
        let x_norm = x / rms;

        // Modulate: x_norm * (1 + scale) + shift   — broadcast [B,1,D] over [B,S,D]
        let modulated = (x_norm * (scale.cast(FloatDType::F32) + 1.0)
            + shift.cast(FloatDType::F32))
        .cast(output_dtype);
        let gate_out = gate.tanh();

        (modulated, gate_out)
    }

    pub(crate) fn modulation(&self, cond_embed: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
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
        (shift, scale, gate)
    }

    pub(crate) fn resolve_modulation(
        &self,
        cond_embed: Tensor<3>,
        precomputed: Option<AdaLnModulation>,
    ) -> AdaLnModulation {
        if let Some(modulation) = precomputed
            && modulation.matches_condition(&cond_embed)
        {
            return modulation;
        }
        let (shift, scale, gate) = self.modulation(cond_embed);
        AdaLnModulation { shift, scale, gate }
    }

    pub(crate) fn has_per_module_inference_cache(&self) -> bool {
        self.packed_down.is_some() || self.packed_up.is_some() || self.packed_bias.is_some()
    }
}

impl LowRankAdaLn {
    fn modulation_matches_input(modulation: &AdaLnModulation, input: &Tensor<3>) -> bool {
        let [batch, _, model_dim] = input.dims();
        let device = input.device();
        [&modulation.shift, &modulation.scale, &modulation.gate]
            .into_iter()
            .all(|tensor| {
                tensor.dims() == [batch, 1, model_dim]
                    && tensor.dtype() == input.dtype()
                    && tensor.device() == device.clone()
            })
    }

    /// Low-rank modulation with fused WGSL RMSNorm/scale/shift.
    ///
    /// A valid cross-layer slice is consumed directly. Missing or stale slices
    /// fall back to the source linear projections, so unsupported shapes never
    /// depend on an inference cache or panic.
    pub(crate) fn forward_wgsl(
        &self,
        x: Tensor<3>,
        cond_embed: Tensor<3>,
        precomputed: Option<AdaLnModulation>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [batch, seq_len, dim] = x.dims();
        let modulation = match precomputed {
            Some(modulation) if Self::modulation_matches_input(&modulation, &x) => Some(modulation),
            _ => None,
        };
        let modulation = self.resolve_modulation(cond_embed, modulation);
        let output = crate::kernels::fused_adaln::fused_adaln_wgsl(
            x.reshape([batch * seq_len, dim])
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            modulation
                .scale
                .reshape([batch, dim])
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            modulation
                .shift
                .reshape([batch, dim])
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            batch,
            seq_len,
            self.eps,
        );
        let output =
            Tensor::<2>::from_primitive::<crate::WgpuRaw>(output).reshape([batch, seq_len, dim]);
        (output, modulation.gate.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rms_norm_preserves_shape() {
        let device = Default::default();
        let norm = RmsNorm::new(16, 1e-6, &device);
        let x = Tensor::<3>::ones([2, 4, 16], &device);
        let y = norm.forward(x);
        assert_eq!(y.dims(), [2, 4, 16]);
    }

    #[test]
    fn rms_norm_unit_scale_preserves_magnitude() {
        let device = Default::default();
        let norm = RmsNorm::new(8, 1e-6, &device);
        // All-ones input: RMS = 1.0, weight = ones → output ≈ input
        let x = Tensor::<3>::ones([1, 1, 8], &device);
        let y = norm.forward(x);
        let data: Vec<f32> = y.into_data().to_vec().unwrap();
        for v in &data {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn head_rms_norm_preserves_shape() {
        let device = Default::default();
        let norm = HeadRmsNorm::new(4, 8, 1e-6, &device);
        let x = Tensor::<4>::ones([2, 3, 4, 8], &device);
        let y = norm.forward(x);
        assert_eq!(y.dims(), [2, 3, 4, 8]);
    }

    #[test]
    fn low_rank_adaln_forward_shapes() {
        let device = Default::default();
        let adaln = LowRankAdaLn::new(32, 8, 1e-6, &device);
        let x = Tensor::<3>::ones([2, 5, 32], &device);
        let cond = Tensor::<3>::ones([2, 1, 96], &device); // 32 * 3
        let (modulated, gate) = adaln.forward(x, cond);
        assert_eq!(modulated.dims(), [2, 5, 32]);
        assert_eq!(gate.dims(), [2, 1, 32]);
    }

    #[test]
    fn low_rank_adaln_zero_init_gate_is_tanh_of_input() {
        // With zero-initialized up projections, shift_up/scale_up/gate_up
        // output zero, so modulation = raw values (no low-rank refinement)
        let device = Default::default();
        let adaln = LowRankAdaLn::new(16, 4, 1e-6, &device);
        let cond = Tensor::<3>::ones([1, 1, 48], &device) * 0.5;
        let x = Tensor::<3>::ones([1, 3, 16], &device);
        let (_, gate) = adaln.forward(x, cond);
        // gate = tanh(raw_gate + gate_up(gate_down(silu(raw_gate))))
        //      = tanh(0.5 + 0) = tanh(0.5)
        let expected = 0.5_f32.tanh();
        let data: Vec<f32> = gate.into_data().to_vec().unwrap();
        for v in &data {
            assert!(
                (v - expected).abs() < 1e-5,
                "expected tanh(0.5)={expected}, got {v}"
            );
        }
    }
}
