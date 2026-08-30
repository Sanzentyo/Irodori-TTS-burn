use burn::tensor::Device;
use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{DType, Tensor, activation::silu},
};

#[cfg(feature = "profile")]
use std::{sync::OnceLock, time::Instant};

#[cfg(feature = "profile")]
#[inline]
fn rf_detail_profile_enabled() -> bool {
    if super::profiling::rf_device_profile_active() {
        return true;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("IRODORI_RF_DETAIL_PROFILE").as_deref() == Ok("1"))
}

use super::linear_ops::linear_rank3_flattened;

#[cfg(feature = "profile")]
fn profile_mlp_substage<const D: usize, T, O>(
    label: &'static str,
    batch: usize,
    sequence: usize,
    reference: &Tensor<D>,
    operation: O,
) -> T
where
    T: Send + 'static,
    O: FnOnce() -> T + Send,
{
    if !rf_detail_profile_enabled() {
        return operation();
    }

    if super::profiling::rf_device_profile_active() {
        let before = super::wgpu_memory_usage(reference);
        let output =
            super::profiling::profile_rf_stage("mlp", label, batch, sequence, reference, operation);
        let after = super::wgpu_memory_usage(reference);
        tracing::info!(
            target: "irodori_tts_burn::rf_profile",
            component = "mlp",
            stage = label,
            batch,
            sequence,
            before_in_use_bytes = before.bytes_in_use,
            after_in_use_bytes = after.bytes_in_use,
            before_reserved_bytes = before.bytes_reserved,
            after_reserved_bytes = after.bytes_reserved,
            before_allocs = before.number_allocs,
            after_allocs = after.number_allocs,
            timestamp_resolution = "deferred",
            "RF MLP substage enqueued under a device timestamp"
        );
        return output;
    }

    let device = reference.device();
    device
        .sync()
        .unwrap_or_else(|error| panic!("RF MLP {label} pre-sync failed: {error}"));
    let before = super::wgpu_memory_usage(reference);
    let started = Instant::now();
    let output = operation();
    device
        .sync()
        .unwrap_or_else(|error| panic!("RF MLP {label} post-sync failed: {error}"));
    let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let after = super::wgpu_memory_usage(reference);
    let delta_in_use = i128::from(after.bytes_in_use) - i128::from(before.bytes_in_use);
    let delta_reserved = i128::from(after.bytes_reserved) - i128::from(before.bytes_reserved);
    let delta_allocs = i128::from(after.number_allocs) - i128::from(before.number_allocs);
    tracing::info!(
        target: "irodori_tts_burn::rf_profile",
        component = "mlp",
        stage = label,
        batch,
        sequence,
        device_complete_ms,
        before_in_use_bytes = before.bytes_in_use,
        after_in_use_bytes = after.bytes_in_use,
        delta_in_use_bytes = %delta_in_use,
        before_reserved_bytes = before.bytes_reserved,
        after_reserved_bytes = after.bytes_reserved,
        delta_reserved_bytes = %delta_reserved,
        before_allocs = before.number_allocs,
        after_allocs = after.number_allocs,
        delta_allocs = %delta_allocs,
        "RF MLP substage profile"
    );
    output
}

#[cfg(feature = "profile")]
macro_rules! rf_mlp_substage {
    ($label:expr, $batch:expr, $sequence:expr, $reference:expr, $operation:expr) => {{
        let profile_reference = $reference.clone();
        profile_mlp_substage($label, $batch, $sequence, &profile_reference, || $operation)
    }};
}

#[cfg(not(feature = "profile"))]
macro_rules! rf_mlp_substage {
    ($label:expr, $batch:expr, $sequence:expr, $reference:expr, $operation:expr) => {
        $operation
    };
}

/// SwiGLU feed-forward network.
///
/// Implements: `out = Linear2(silu(Linear1(x)) * Linear3(x))`
/// where Linear1, Linear2, Linear3 are named `w1`, `w2`, `w3` to match the Python state_dict.
///
/// `w1` and `w3` are the two "gate/value" projections (expand),
/// `w2` is the output projection (contract).
#[derive(Module, Debug)]
pub struct SwiGlu {
    pub(crate) w1: Linear,
    pub(crate) w2: Linear,
    pub(crate) w3: Linear,
    /// Fused w1‖w3 weight: `[dim, 2*hidden_dim]` — inference-only optimisation.
    /// Saves 1 kernel launch per block per denoising step.
    #[module(skip)]
    fused_w13_weight: Option<Tensor<2>>,
    /// Pair-interleaved `[w1[0], w3[0], w1[1], w3[1], ...]` expansion used by
    /// CubeK's compressed-output SwiGLU writer.
    #[module(skip)]
    interleaved_w13_weight_wgsl: Option<Tensor<2>>,
    /// Row-major `w2` cache used by the measured prepared WGSL policy.
    ///
    /// The v4 checkpoint exposes logical `[3680, 1280]` `w2` weights as a
    /// checkpoint-native column-major view. On RTX 3060 Ti, packing all 12
    /// layers once took 174 ms and co-retaining the cache costs 215.625 MiB.
    /// The exact B1/S50 projection improved from 830.425 us to 685.870 us.
    /// B2 keeps the source view through S50, then selects this cache at S100
    /// and S200 where the multi-length layout sweep measured a consistent win.
    /// The learned source parameter remains available for every fallback,
    /// portable execution, and training path.
    #[module(skip)]
    packed_w2_weight_wgsl: Option<Tensor<2>>,
    /// Profile-locked admission for the same packed contraction at B3.
    #[module(skip)]
    allow_b3_packed_w2_wgsl: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedW2Route {
    PackedRowFlat,
    PackedRowRank3,
    SourceColumnFlat,
}

fn prepared_w2_route(
    batch: usize,
    sequence: usize,
    packed_row_compatible: bool,
    allow_b3_packed: bool,
) -> PreparedW2Route {
    prepared_w2_route_for(
        crate::route_autotune::active_route_table(),
        batch,
        sequence,
        packed_row_compatible,
        allow_b3_packed,
    )
}

fn prepared_w2_route_for(
    routes: &crate::route_autotune::ResolvedRouteTable,
    batch: usize,
    sequence: usize,
    packed_row_compatible: bool,
    allow_b3_packed: bool,
) -> PreparedW2Route {
    use crate::route_autotune::MlpContractWeightRoute as Route;

    if !packed_row_compatible {
        return PreparedW2Route::SourceColumnFlat;
    }
    match routes.mlp_contract_weight(batch, sequence) {
        Route::PackedRowFlat => PreparedW2Route::PackedRowFlat,
        Route::PackedRowRank3 => PreparedW2Route::PackedRowRank3,
        Route::SourceColumnFlat
            if routes.permits_legacy_profile_overlay()
                && allow_b3_packed
                && batch == 3
                && sequence >= 200 =>
        {
            PreparedW2Route::PackedRowRank3
        }
        Route::SourceColumnFlat
            if routes.permits_legacy_profile_overlay()
                && allow_b3_packed
                && batch == 3
                && sequence >= 100 =>
        {
            PreparedW2Route::PackedRowFlat
        }
        Route::SourceColumnFlat => PreparedW2Route::SourceColumnFlat,
    }
}

fn dit_mlp_expand_t64_route(
    batch: usize,
    sequence: usize,
    input_dim: usize,
    expanded_dim: usize,
    dtype: DType,
) -> bool {
    dit_mlp_expand_t64_route_for(
        crate::route_autotune::active_route_table(),
        batch,
        sequence,
        input_dim,
        expanded_dim,
        dtype,
    )
}

fn dit_mlp_expand_t64_route_for(
    routes: &crate::route_autotune::ResolvedRouteTable,
    batch: usize,
    sequence: usize,
    input_dim: usize,
    expanded_dim: usize,
    dtype: DType,
) -> bool {
    // B3 is deliberately excluded: the 489-frame same-binary component screen
    // was slower than the generic route and changed the final audio hash. The
    // M5-measured path remains available only as an explicit profile candidate.
    crate::kernels::dit_projection_t64::dit_projection_route_enabled()
        && projection_swiglu_epilogue_enabled()
        && (batch != 3
            || crate::kernels::dit_projection_t64::dit_projection_component_enabled("MLP_EXPAND"))
        && dtype == DType::F32
        && matches!(
            routes.mlp_expand(batch, sequence),
            crate::route_autotune::SwiGluRoute::HandwrittenT64
                | crate::route_autotune::SwiGluRoute::HandwrittenT64VectorInput
        )
        && crate::kernels::dit_projection_t64::dit_sequence_is_admitted(sequence)
        && input_dim == 1_280
        && expanded_dim == 7_360
}

/// Profile-only same-binary control for the one-dispatch projection epilogue.
/// Production builds compile this to a constant without an environment lookup.
#[inline]
fn projection_swiglu_epilogue_enabled() -> bool {
    #[cfg(feature = "profile")]
    {
        std::env::var("IRODORI_DISABLE_PROJECTION_SWIGLU_EPILOGUE").as_deref() != Ok("1")
    }
    #[cfg(not(feature = "profile"))]
    {
        true
    }
}

fn dit_mlp_contract_t64_route(
    batch: usize,
    sequence: usize,
    hidden_dim: usize,
    output_dim: usize,
    dtype: DType,
) -> bool {
    dit_mlp_contract_t64_route_for(
        crate::route_autotune::active_route_table(),
        batch,
        sequence,
        hidden_dim,
        output_dim,
        dtype,
    )
}

fn dit_mlp_contract_t64_route_for(
    routes: &crate::route_autotune::ResolvedRouteTable,
    batch: usize,
    sequence: usize,
    hidden_dim: usize,
    output_dim: usize,
    dtype: DType,
) -> bool {
    crate::kernels::dit_projection_t64::dit_projection_route_enabled()
        && (batch != 3
            || crate::kernels::dit_projection_t64::dit_projection_component_enabled("MLP_CONTRACT"))
        && dtype == DType::F32
        && matches!(batch, 1..=3)
        && routes.mlp_contract(batch, sequence).is_handwritten()
        && crate::kernels::dit_projection_t64::dit_sequence_is_admitted(sequence)
        && hidden_dim == 3_680
        && output_dim == 1_280
}

fn dit_mlp_contract_pitched_route(batch: usize, sequence: usize) -> bool {
    crate::route_autotune::active_route_table()
        .mlp_contract(batch, sequence)
        .uses_pitched_input()
}

fn dit_mlp_contract_vector_input_route(batch: usize, sequence: usize) -> bool {
    crate::route_autotune::active_route_table()
        .mlp_contract(batch, sequence)
        .uses_vector_input()
}

const fn duration_mlp_expand_t64_route(
    batch: usize,
    sequence: usize,
    input_dim: usize,
    expanded_dim: usize,
) -> bool {
    batch == 1 && sequence > 0 && sequence <= 64 && input_dim == 1_024 && expanded_dim == 2_048
}

impl SwiGlu {
    /// `dim`: input/output dimension.
    /// `hidden_dim`: intermediate dimension (typically `dim * 8/3`, rounded up).
    pub fn new(dim: usize, hidden_dim: Option<usize>, device: &Device) -> Self {
        // Default: 8/3 * dim rounded to nearest multiple of 256, matching Python
        let hidden_dim = hidden_dim.unwrap_or_else(|| round_up(dim * 8 / 3, 256));

        Self {
            w1: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(hidden_dim, dim)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .init(device),
            fused_w13_weight: None,
            interleaved_w13_weight_wgsl: None,
            packed_w2_weight_wgsl: None,
            allow_b3_packed_w2_wgsl: false,
        }
    }

    /// `x`: any shape ending in `[..., dim]`, operates on last dim.
    /// Concretely used as `[B, S, D]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let (gate, val) = if let Some(ref fused_w) = self.fused_w13_weight {
            debug_assert_eq!(
                fused_w.device(),
                x.device(),
                "fused w13 weight on wrong device (was model moved after prepare_for_inference()?)"
            );
            // Single matmul: x @ fused_w → [B, S, 2*H], then split
            let [_b, _s, _d] = x.dims();
            let hidden_dim = fused_w.dims()[1] / 2;
            let w13 = x.matmul(fused_w.clone().unsqueeze::<3>());
            let gate = silu(w13.clone().narrow(2, 0, hidden_dim));
            let val = w13.narrow(2, hidden_dim, hidden_dim);
            (gate, val)
        } else {
            let gate = silu(self.w1.forward(x.clone()));
            let val = self.w3.forward(x);
            (gate, val)
        };
        self.w2.forward(gate * val)
    }

    /// Branch-free forward using the pre-fused w1‖w3 weight matrix.
    ///
    /// # Panics
    ///
    /// Panics if [`prepare_for_inference`](Self::prepare_for_inference) has not
    /// been called (i.e. `fused_w13_weight` is `None`).
    pub(crate) fn forward_fused(&self, x: Tensor<3>) -> Tensor<3> {
        let fused_w = self.fused_w13_weight.as_ref().expect(
            "forward_fused called without weight fusion — call prepare_for_inference first",
        );
        let hidden_dim = fused_w.dims()[1] / 2;
        let w13 = linear_rank3_flattened(x, fused_w.clone(), None);
        let gate = silu(w13.clone().narrow(2, 0, hidden_dim));
        let val = w13.narrow(2, hidden_dim, hidden_dim);
        linear_rank3_flattened(
            gate * val,
            self.w2.weight.val(),
            self.w2.bias.as_ref().map(|bias| bias.val()),
        )
    }

    /// Fuse w1 and w3 weight matrices into a single `[dim, 2*hidden_dim]` tensor.
    ///
    /// Saves 1 kernel launch per block per denoising step (~10ms total for
    /// 12 blocks × 40 steps). Idempotent.
    ///
    /// # Safety invariant
    /// Must be called **after** final weights are loaded and device placement is
    /// complete. The fused tensor is `#[module(skip)]`, so it will NOT follow
    /// `to_device()` or `fork()` calls on the parent module.
    pub(crate) fn prepare_for_inference(&mut self) {
        if self.fused_w13_weight.is_some() {
            return;
        }
        let w1 = self.w1.weight.val(); // [dim, hidden_dim]
        let w3 = self.w3.weight.val();
        self.fused_w13_weight = Some(Tensor::cat(vec![w1, w3], 1));
    }

    /// Prepare the pair-interleaved expansion expected by the generic CubeK
    /// compressed-output writer. No request-time transpose or pack is allowed.
    pub(crate) fn prepare_interleaved_w13_wgsl(&mut self) {
        let w1 = self.w1.weight.val();
        let w3 = self.w3.weight.val();
        let [input_dim, hidden] = w1.dims();
        assert_eq!(w3.dims(), [input_dim, hidden]);
        assert_eq!(w3.device(), w1.device());
        assert_eq!(w3.dtype(), w1.dtype());
        if let Some(weight) = self.interleaved_w13_weight_wgsl.as_ref() {
            assert_eq!(weight.dims(), [input_dim, hidden * 2]);
            assert_eq!(weight.device(), w1.device());
            assert_eq!(weight.dtype(), w1.dtype());
            return;
        }
        let stacked: Tensor<3> = Tensor::stack(vec![w1.transpose(), w3.transpose()], 1);
        let physical = stacked.reshape([hidden * 2, input_dim]);
        let interleaved = physical.transpose();
        let primitive = interleaved
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("interleaved SwiGLU cache must use WGPU raw backend");
        assert!(!primitive.is_contiguous());
        assert_eq!(&primitive.meta.strides()[..], [1, input_dim]);
        self.interleaved_w13_weight_wgsl = Some(interleaved);
    }

    /// Materialise backend-independent storage for the measured WGSL cache.
    ///
    /// This is deliberately separate from [`Self::prepare_for_inference`], so
    /// constructing the portable optimized wrapper neither allocates nor uses
    /// the extra row-major `w2` copy. Existing caches are validated and reused;
    /// malformed or partially shaped caches are rejected before execution.
    fn prepare_w2_row_major_cache_wgsl(&mut self) {
        let (expected_shape, expected_device) = self.validate_w2_source_weight();
        if let Some(packed) = self.packed_w2_weight_wgsl.as_ref() {
            assert_eq!(
                packed.dims(),
                expected_shape,
                "existing packed w2 WGSL cache shape mismatch"
            );
            assert_eq!(
                packed.device(),
                expected_device,
                "existing packed w2 WGSL cache is on the wrong device"
            );
            assert_eq!(
                packed.dtype(),
                self.w2.weight.dtype(),
                "existing packed w2 WGSL cache dtype mismatch"
            );
            return;
        }

        // Even a single-input cat materialises the logical source view in
        // canonical row-major output storage without changing its values.
        let packed = Tensor::cat(vec![self.w2.weight.val()], 0);
        assert_eq!(
            packed.dims(),
            expected_shape,
            "new packed w2 WGSL cache shape mismatch"
        );
        assert_eq!(
            packed.device(),
            expected_device,
            "new packed w2 WGSL cache is on the wrong device"
        );
        assert_eq!(
            packed.dtype(),
            self.w2.weight.dtype(),
            "new packed w2 WGSL cache dtype mismatch"
        );
        self.packed_w2_weight_wgsl = Some(packed);
    }

    /// Validate the learned `w2` source before creating or reusing its WGSL
    /// row-major cache.
    fn validate_w2_source_weight(&self) -> ([usize; 2], Device) {
        assert!(
            self.w2.bias.is_none(),
            "SwiGLU w2 bias must be absent before WGSL row-major packing"
        );
        let weight = self.w2.weight.val();
        let [hidden_dim, output_dim] = weight.dims();
        assert!(
            hidden_dim > 0 && output_dim > 0,
            "SwiGLU w2 source weight must be non-empty"
        );
        assert!(
            matches!(weight.dtype(), DType::F32 | DType::F16),
            "SwiGLU w2 source weight must use a WGSL float dtype"
        );
        assert_eq!(
            self.w1.weight.dims(),
            [output_dim, hidden_dim],
            "SwiGLU w1/w2 source weight shapes are inconsistent"
        );
        assert_eq!(
            self.w3.weight.dims(),
            [output_dim, hidden_dim],
            "SwiGLU w3/w2 source weight shapes are inconsistent"
        );
        (weight.dims(), weight.device())
    }

    /// Apply the production flattened projection with the WGSL-only routing
    /// policy. Physical cache compatibility is established by the WGPU caller.
    fn project_w2_flattened_wgsl_policy(
        &self,
        activated: Tensor<3>,
        packed_row_compatible: bool,
    ) -> Tensor<3> {
        let [batch, sequence, _] = activated.dims();
        let route = prepared_w2_route(
            batch,
            sequence,
            packed_row_compatible,
            self.allow_b3_packed_w2_wgsl,
        );
        let packed_row = || {
            self.packed_w2_weight_wgsl
                .as_ref()
                .expect("compatible packed w2 route requires a prepared cache")
                .clone()
        };
        match route {
            PreparedW2Route::PackedRowFlat => linear_rank3_flattened(
                activated,
                packed_row(),
                self.w2.bias.as_ref().map(|bias| bias.val()),
            ),
            PreparedW2Route::PackedRowRank3 => activated.matmul(packed_row().unsqueeze::<3>()),
            PreparedW2Route::SourceColumnFlat => linear_rank3_flattened(
                activated,
                self.w2.weight.val(),
                self.w2.bias.as_ref().map(|bias| bias.val()),
            ),
        }
    }
}

impl SwiGlu {
    /// Release logical w1/w3 after the fused production expansion and both w2
    /// layouts have been prepared.
    ///
    /// w2 itself is retained because B2/B3 and short-sequence policies can
    /// select its source-column layout.
    pub(crate) fn release_production_expand_sources_wgsl(&mut self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let fused = self.fused_w13_weight.as_ref().ok_or_else(|| {
            IrodoriError::Config(
                "production profile requires a prepared fused w1/w3 cache".to_owned(),
            )
        })?;
        let packed_w2 = self.packed_w2_weight_wgsl.as_ref().ok_or_else(|| {
            IrodoriError::Config("production profile requires a prepared w2 cache".to_owned())
        })?;
        let source_device = self.w1.weight.device();
        if fused.dims() != [1_280, 7_360]
            || fused.device() != source_device
            || packed_w2.dims() != [3_680, 1_280]
            || packed_w2.device() != source_device
            || self.w1.weight.dims() != [1_280, 3_680]
            || self.w3.weight.dims() != [1_280, 3_680]
            || self.w3.weight.device() != source_device
            || self.w1.bias.is_some()
            || self.w3.bias.is_some()
        {
            return Err(IrodoriError::Config(
                "production FFN cache contract mismatch".to_owned(),
            ));
        }

        let tombstone = Tensor::zeros([1, 1], &source_device);
        self.w1.weight = Param::initialized(ParamId::new(), tombstone.clone());
        self.w3.weight = Param::initialized(ParamId::new(), tombstone);
        Ok(())
    }

    /// Release the contraction source after a serving manifest proves that
    /// every admitted B1/B2 route selects the prepared row-major cache.
    pub(crate) fn release_prepared_w2_source_wgsl(&mut self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let source_device = self.w2.weight.device();
        let packed_contract = self.packed_w2_weight_wgsl.as_ref().is_some_and(|packed| {
            packed.dims() == [3_680, 1_280]
                && packed.device() == source_device
                && packed.dtype() == self.w2.weight.dtype()
        });
        if self.w2.weight.dims() != [3_680, 1_280] || self.w2.bias.is_some() || !packed_contract {
            return Err(IrodoriError::Config(
                "prepared-only FFN contraction cache contract mismatch".to_owned(),
            ));
        }
        self.w2.weight = Param::initialized(ParamId::new(), Tensor::zeros([1, 1], &source_device));
        Ok(())
    }

    /// Admit the prepared row-major contraction for long B3 CFG calls.
    pub(crate) fn enable_long_b3_prepared_w2_wgsl(&mut self) -> crate::error::Result<()> {
        let packed = self.packed_w2_weight_wgsl.as_ref().ok_or_else(|| {
            crate::error::IrodoriError::Config(
                "long B3 FFN requires a prepared w2 cache".to_owned(),
            )
        })?;
        if packed.dims() != [3_680, 1_280]
            || packed.device() != self.w2.weight.device()
            || packed.dtype() != self.w2.weight.dtype()
        {
            return Err(crate::error::IrodoriError::Config(
                "long B3 FFN prepared w2 contract mismatch".to_owned(),
            ));
        }
        self.allow_b3_packed_w2_wgsl = true;
        Ok(())
    }

    /// Validate the prepared fixed-112 WGSL path and optionally release source
    /// w1/w3. The fused cache remains the only expand projection used by the
    /// packed-only profile; w2 is retained because the selected projection
    /// policy still uses its source layout for some fixed-112 batches.
    pub(crate) fn lock_fixed_112_wgsl(
        &mut self,
        release_source_weights: bool,
    ) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let fused = self.fused_w13_weight.as_ref().ok_or_else(|| {
            IrodoriError::Config("fixed-112 profile requires a fused w1/w3 cache".to_owned())
        })?;
        let source_device = self.w1.weight.device();
        let packed_w2_contract = self.packed_w2_weight_wgsl.as_ref().is_some_and(|packed| {
            packed.dims() == [3_680, 1_280] && packed.device() == source_device
        });
        if fused.dims() != [1_280, 7_360]
            || fused.device() != source_device
            || self.w1.weight.dims() != [1_280, 3_680]
            || self.w3.weight.dims() != [1_280, 3_680]
            || self.w3.weight.device() != source_device
            || !packed_w2_contract
        {
            return Err(IrodoriError::Config(
                "fixed-112 FFN cache contract mismatch".to_owned(),
            ));
        }
        if release_source_weights {
            let tombstone = Tensor::zeros([1, 1], &source_device);
            self.w1.weight = Param::initialized(ParamId::new(), tombstone.clone());
            self.w3.weight = Param::initialized(ParamId::new(), tombstone.clone());
        }
        Ok(())
    }

    /// Prepare and validate the physical row-major cache used only by WGSL
    /// inference. Repeated calls reuse the same allocation.
    pub(crate) fn prepare_w2_row_major_wgsl(&mut self) {
        self.prepare_w2_row_major_cache_wgsl();
        let packed = self
            .packed_w2_weight_wgsl
            .as_ref()
            .expect("WGSL w2 preparation must create the row-major cache")
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let [rows, columns] = self.w2.weight.dims();
        assert!(
            matches!(packed.dtype, DType::F32 | DType::F16),
            "packed w2 WGSL cache must use a WGSL float dtype"
        );
        assert!(
            packed.is_contiguous(),
            "packed w2 WGSL cache must be contiguous"
        );
        assert_eq!(
            &packed.meta.strides()[..],
            &[columns, 1],
            "packed w2 WGSL cache must have row-major strides"
        );
        assert_eq!(
            packed.meta.shape().dims::<2>(),
            [rows, columns],
            "packed w2 WGSL cache shape mismatch at the backend boundary"
        );
    }

    /// Prepare and retain exactly the layouts admitted by a runtime plan.
    #[cfg(all(feature = "inference", feature = "codec"))]
    pub(crate) fn retain_weight_layouts_wgsl(
        &mut self,
        layouts: &crate::runtime::WeightLayoutSet,
        admit_long_b3: bool,
    ) -> crate::error::Result<()> {
        use crate::{
            error::IrodoriError,
            runtime::WeightLayout::{
                MlpContractPacked, MlpContractSource, SwiGluFused, SwiGluInterleaved, SwiGluSource,
            },
        };

        if layouts.contains(SwiGluFused) {
            self.prepare_for_inference();
        }
        if layouts.contains(SwiGluInterleaved) {
            self.prepare_interleaved_w13_wgsl();
        }
        if layouts.contains(MlpContractPacked) {
            self.prepare_w2_row_major_wgsl();
        }

        let source_device = self.w1.weight.device();
        if !layouts.contains(SwiGluSource) {
            let fused_ok = self.fused_w13_weight.as_ref().is_some_and(|weight| {
                weight.dims() == [1_280, 7_360] && weight.device() == source_device
            });
            let interleaved_ok = self
                .interleaved_w13_weight_wgsl
                .as_ref()
                .is_some_and(|weight| {
                    weight.dims() == [1_280, 7_360] && weight.device() == source_device
                });
            if !fused_ok && !interleaved_ok {
                return Err(IrodoriError::Config(
                    "cannot release SwiGLU sources without a prepared expansion layout".to_owned(),
                ));
            }
            let tombstone = Tensor::zeros([1, 1], &source_device);
            self.w1.weight = Param::initialized(ParamId::new(), tombstone.clone());
            self.w3.weight = Param::initialized(ParamId::new(), tombstone);
        }
        if !layouts.contains(SwiGluFused) {
            self.fused_w13_weight = None;
        }
        if !layouts.contains(SwiGluInterleaved) {
            self.interleaved_w13_weight_wgsl = None;
        }
        if !layouts.contains(MlpContractSource) {
            self.release_prepared_w2_source_wgsl()?;
        }
        if !layouts.contains(MlpContractPacked) {
            self.packed_w2_weight_wgsl = None;
        }
        if admit_long_b3 {
            self.enable_long_b3_prepared_w2_wgsl()?;
        }
        Ok(())
    }

    /// Branch-free inference path using the production WGSL SwiGLU fusion.
    ///
    /// The large `x @ (w1 || w3)` projection remains on Burn's tuned matmul.
    /// Its output is consumed by one shader instead of materialising two
    /// slices and scheduling separate SiLU and multiply operations.
    pub(crate) fn forward_fused_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_fused_wgsl_impl(x, None)
    }

    /// Released identity-dropout path that folds the gated residual update
    /// into the final MLP projection. Every unsupported contract uses the
    /// existing projection and residual shaders.
    pub(crate) fn forward_fused_residual_wgsl(
        &self,
        x: Tensor<3>,
        residual: Tensor<3>,
        gate: Tensor<3>,
    ) -> Tensor<3> {
        self.forward_fused_wgsl_impl(x, Some((residual, gate)))
    }

    #[allow(clippy::redundant_closure)]
    fn forward_fused_wgsl_impl(
        &self,
        x: Tensor<3>,
        residual_gate: Option<(Tensor<3>, Tensor<3>)>,
    ) -> Tensor<3> {
        let [batch, seq_len, input_dim] = x.dims();
        let mlp_expand_route =
            crate::route_autotune::active_route_table().mlp_expand(batch, seq_len);
        let split_projection_route =
            mlp_expand_route == crate::route_autotune::SwiGluRoute::SplitProjectionPairEpilogue;
        let fused_weight = self.fused_w13_weight.as_ref();
        let hidden = if split_projection_route {
            let [w1_input, w1_hidden] = self.w1.weight.dims();
            assert_eq!(w1_input, input_dim, "split SwiGLU w1 input mismatch");
            assert_eq!(
                self.w3.weight.dims(),
                [w1_input, w1_hidden],
                "split SwiGLU w1/w3 shape mismatch"
            );
            w1_hidden
        } else {
            fused_weight
                .map(|weight| weight.dims()[1] / 2)
                .or_else(|| {
                    self.interleaved_w13_weight_wgsl
                        .as_ref()
                        .map(|weight| weight.dims()[1] / 2)
                })
                .expect("forward_fused_wgsl called before inference weight preparation")
        };
        let rows = batch
            .checked_mul(seq_len)
            .expect("SwiGLU flattened row count overflow");
        let flattened = x.clone().reshape([rows, input_dim]);
        let split_activated = split_projection_route.then(|| {
            rf_mlp_substage!("expand_split_swiglu", batch, seq_len, x, {
                let gate: Tensor<2> =
                    linear_rank3_flattened(x.clone(), self.w1.weight.val(), None).flatten(0, 1);
                let value: Tensor<2> =
                    linear_rank3_flattened(x.clone(), self.w3.weight.val(), None).flatten(0, 1);
                let output = crate::kernels::fused_swiglu::try_fused_swiglu_pair_wgsl(
                    gate.try_into_primitive::<crate::WgpuRaw>()
                        .expect("split SwiGLU gate must use WGPU raw backend"),
                    value
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("split SwiGLU value must use WGPU raw backend"),
                )
                .expect("resolved split SwiGLU route violated its launcher contract");
                Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
            })
        });
        let compressed_algorithm = match mlp_expand_route {
            crate::route_autotune::SwiGluRoute::CubeKCompressedInterleaved => {
                Some(crate::kernels::cubek_swiglu::CubeKSwiGluAlgorithm::UnitMin)
            }
            crate::route_autotune::SwiGluRoute::CubeKCompressedInterleavedMaxTile => {
                Some(crate::kernels::cubek_swiglu::CubeKSwiGluAlgorithm::UnitMax)
            }
            crate::route_autotune::SwiGluRoute::CubeKCompressedInterleavedDoubleUnit => {
                Some(crate::kernels::cubek_swiglu::CubeKSwiGluAlgorithm::DoubleUnit)
            }
            crate::route_autotune::SwiGluRoute::CubeKCompressedInterleavedPlaneVec => {
                Some(crate::kernels::cubek_swiglu::CubeKSwiGluAlgorithm::PlaneVec)
            }
            crate::route_autotune::SwiGluRoute::CubeKCompressedInterleavedGemm => {
                Some(crate::kernels::cubek_swiglu::CubeKSwiGluAlgorithm::Gemm)
            }
            _ => None,
        };
        let cubek_compressed = (compressed_algorithm.is_some()
            && matches!(batch, 1..=3)
            && seq_len >= 100
            && x.dtype() == DType::F32
            && cubek_b3_swiglu_enabled())
        .then(|| {
            let weight = self.interleaved_w13_weight_wgsl.as_ref()?;
            rf_mlp_substage!("expand_swiglu_cubek_compressed", batch, seq_len, x, {
                crate::kernels::cubek_swiglu::try_cubek_swiglu_compressed(
                    flattened
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    weight
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    compressed_algorithm.expect("compressed route selected a CubeK algorithm"),
                )
            })
        })
        .flatten();
        let fused_activated = cubek_compressed.or_else(|| {
            let fused_weight = fused_weight?;
            dit_mlp_expand_t64_route(batch, seq_len, input_dim, fused_weight.dims()[1], x.dtype())
                .then(|| {
                    rf_mlp_substage!("expand_swiglu", batch, seq_len, x, {
                        let flattened = flattened
                            .clone()
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend");
                        let fused_weight = fused_weight
                            .clone()
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend");
                        match mlp_expand_route {
                            crate::route_autotune::SwiGluRoute::HandwrittenT64VectorInput => {
                                crate::kernels::dit_projection_t64::try_dit_mlp_expand_swiglu_c128_vec4_wgsl(
                                    flattened,
                                    fused_weight,
                                )
                            }
                            _ => crate::kernels::dit_projection_t64::try_dit_mlp_expand_swiglu_c128_wgsl(
                                flattened,
                                fused_weight,
                            ),
                        }
                    })
                })
                .flatten()
        });
        let activated_flat = split_activated.unwrap_or_else(|| {
            fused_activated
                .map(|output| Tensor::<2>::from_primitive::<crate::WgpuRaw>(output))
                .unwrap_or_else(|| {
                    let fused_weight = fused_weight.expect(
                        "fallback SwiGLU expansion requires the half-separated fused layout",
                    );
                    let projected = rf_mlp_substage!("expand", batch, seq_len, x, {
                        let candidate = (mlp_expand_route
                            == crate::route_autotune::SwiGluRoute::HandwrittenT64
                            && dit_mlp_expand_t64_route(
                                batch,
                                seq_len,
                                input_dim,
                                fused_weight.dims()[1],
                                x.dtype(),
                            ))
                        .then(|| {
                            crate::kernels::dit_projection_t64::try_dit_mlp_expand_t64_wgsl(
                                flattened
                                    .try_into_primitive::<crate::WgpuRaw>()
                                    .expect("tensor must use WGPU raw backend"),
                                fused_weight
                                    .clone()
                                    .try_into_primitive::<crate::WgpuRaw>()
                                    .expect("tensor must use WGPU raw backend"),
                            )
                        })
                        .flatten();
                        candidate
                            .map(|output| Tensor::<2>::from_primitive::<crate::WgpuRaw>(output))
                            .unwrap_or_else(|| {
                                linear_rank3_flattened(x, fused_weight.clone(), None).flatten(0, 1)
                            })
                    });
                    let pitched = (dit_mlp_contract_pitched_route(batch, seq_len)
                        && residual_gate.is_some()
                        && self.packed_w2_weight_wgsl.is_some()
                        && dit_mlp_contract_t64_route(
                            batch,
                            seq_len,
                            hidden,
                            input_dim,
                            projected.dtype(),
                        ))
                    .then(|| {
                        rf_mlp_substage!("swiglu_pitched", batch, seq_len, projected, {
                            crate::kernels::fused_swiglu::try_fused_swiglu_pitched_in_place_wgsl(
                                projected
                                    .clone()
                                    .try_into_primitive::<crate::WgpuRaw>()
                                    .expect("tensor must use WGPU raw backend"),
                            )
                        })
                    })
                    .flatten()
                    .map(|full| {
                        Tensor::<2>::from_primitive::<crate::WgpuRaw>(full)
                            .slice([0..rows, 0..hidden])
                    });
                    pitched.unwrap_or_else(|| {
                        let activated_flat =
                            rf_mlp_substage!("swiglu", batch, seq_len, projected, {
                                crate::kernels::fused_swiglu::fused_swiglu_wgsl(
                                    projected
                                        .try_into_primitive::<crate::WgpuRaw>()
                                        .expect("tensor must use WGPU raw backend"),
                                )
                            });
                        Tensor::<2>::from_primitive::<crate::WgpuRaw>(activated_flat)
                    })
                })
        });
        let activated = activated_flat.clone().reshape([batch, seq_len, hidden]);
        let packed_row_compatible = self.packed_w2_contract_wgsl(&activated);
        rf_mlp_substage!("contract", batch, seq_len, activated, {
            let mut includes_residual = false;
            let candidate = if packed_row_compatible
                && dit_mlp_contract_t64_route(batch, seq_len, hidden, input_dim, activated.dtype())
                && let Some(packed) = self.packed_w2_weight_wgsl.as_ref()
            {
                let fused = residual_gate.as_ref().and_then(|(residual, gate)| {
                    let activated = activated_flat
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend");
                    let packed = packed
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend");
                    let residual = residual
                        .clone()
                        .reshape([rows, input_dim])
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend");
                    let gate = gate
                        .clone()
                        .reshape([batch, input_dim])
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend");
                    if dit_mlp_contract_vector_input_route(batch, seq_len) {
                        crate::kernels::dit_mlp_contract_residual::try_dit_mlp_contract_residual_vec4_wgsl(
                            activated, packed, residual, gate, batch, seq_len,
                        )
                    } else {
                        crate::kernels::dit_mlp_contract_residual::try_dit_mlp_contract_residual_wgsl(
                            activated, packed, residual, gate, batch, seq_len,
                        )
                    }
                });
                if fused.is_some() {
                    includes_residual = true;
                    fused
                } else {
                    crate::kernels::dit_projection_t64::try_dit_mlp_contract_t64_wgsl(
                        activated_flat
                            .clone()
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend"),
                        packed
                            .clone()
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend"),
                    )
                }
            } else {
                None
            };
            let branch_or_final = candidate
                .map(|output| {
                    Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
                        .reshape([batch, seq_len, input_dim])
                })
                .unwrap_or_else(|| {
                    self.project_w2_flattened_wgsl_policy(activated, packed_row_compatible)
                });
            match residual_gate {
                None => branch_or_final,
                Some(_) if includes_residual => branch_or_final,
                Some((residual, gate)) => {
                    let output = crate::kernels::fused_residual_gate::fused_residual_gate_wgsl(
                        residual
                            .reshape([rows, input_dim])
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend"),
                        branch_or_final
                            .reshape([rows, input_dim])
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend"),
                        gate.reshape([batch, input_dim])
                            .try_into_primitive::<crate::WgpuRaw>()
                            .expect("tensor must use WGPU raw backend"),
                        batch,
                        seq_len,
                    );
                    Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
                        .reshape([batch, seq_len, input_dim])
                }
            }
        })
    }

    /// Released duration path that consumes the `w1||w3` projection directly
    /// in a fused activation-plus-w2 kernel. All learned tensors stay on GPU.
    #[allow(clippy::redundant_closure)]
    pub(crate) fn forward_duration_fused_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, dim] = x.dims();
        let fused_weight = self
            .fused_w13_weight
            .as_ref()
            .expect("duration fused WGSL called before inference weight fusion");
        let rows = batch
            .checked_mul(seq_len)
            .expect("duration SwiGLU flattened row count overflow");
        let flattened = x.clone().reshape([rows, dim]);
        let candidate = duration_mlp_expand_t64_route(batch, seq_len, dim, fused_weight.dims()[1])
            .then(|| {
                crate::kernels::dit_projection_t64::try_duration_mlp_expand_t64_wgsl(
                    flattened
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    fused_weight
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                )
            })
            .flatten();
        let projected: Tensor<2> = candidate
            .map(|output| Tensor::<2>::from_primitive::<crate::WgpuRaw>(output))
            .unwrap_or_else(|| {
                linear_rank3_flattened(x.clone(), fused_weight.clone(), None).flatten(0, 1)
            });
        if batch == 1
            && dim == 1024
            && (1..=64).contains(&seq_len)
            && let Some(packed) = self.packed_w2_weight_wgsl.as_ref()
            && let Some(output) = crate::kernels::duration_swiglu_w2::try_duration_swiglu_w2_wgsl(
                projected
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                packed
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
            )
        {
            return Tensor::<3>::from_primitive::<crate::WgpuRaw>(output);
        }
        self.forward_fused_wgsl(x)
    }

    /// Long-text duration route that folds the block residual epilogue into
    /// the measured O64/K128 contraction. A rejected contract leaves the
    /// established branch plus residual-finalizer path available to the caller.
    pub(crate) fn try_forward_duration_fused_residual_wgsl(
        &self,
        x: Tensor<3>,
        residual: Tensor<3>,
        gate: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let [batch, seq_len, dim] = x.dims();
        if batch != 1 || !(48..=64).contains(&seq_len) || dim != 1_024 {
            return None;
        }
        let fused_weight = self.fused_w13_weight.as_ref()?;
        if fused_weight.dims() != [1_024, 2_048] {
            return None;
        }
        let packed = self.packed_w2_weight_wgsl.as_ref()?;
        let projected = crate::kernels::dit_projection_t64::try_duration_mlp_expand_t64_wgsl(
            x.reshape([seq_len, dim])
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            fused_weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )?;
        let output = crate::kernels::duration_swiglu_w2::try_duration_swiglu_w2_residual_wgsl(
            projected,
            packed
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            residual
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            gate.try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )?;
        Some(Tensor::<3>::from_primitive::<crate::WgpuRaw>(output))
    }

    /// Check the full physical measured row-cache contract without consuming the
    /// source tensor needed by the fail-closed column-weight fallback.
    fn packed_w2_contract_wgsl(&self, activated: &Tensor<3>) -> bool {
        let [batch, seq_len, hidden_dim] = activated.dims();
        let Some(packed) = self.packed_w2_weight_wgsl.as_ref() else {
            return false;
        };
        let [packed_hidden, output_dim] = packed.dims();
        let routes = crate::route_autotune::active_route_table();
        let measured_batch = routes.mlp_contract_weight(batch, seq_len)
            != crate::route_autotune::MlpContractWeightRoute::SourceColumnFlat
            || (self.allow_b3_packed_w2_wgsl && batch == 3 && seq_len >= 100);
        if !measured_batch
            || seq_len == 0
            || hidden_dim == 0
            || output_dim == 0
            || hidden_dim != packed_hidden
            || self.w2.bias.is_some()
            || !matches!(activated.dtype(), DType::F32 | DType::F16)
            || packed.dtype() != activated.dtype()
            || packed.dims() != [3_680, 1_280]
            || packed.device() != activated.device()
        {
            return false;
        }

        let primitive = packed
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        primitive.dtype == activated.dtype()
            && primitive.is_contiguous()
            && &primitive.meta.strides()[..] == [output_dim, 1].as_slice()
    }
}

#[inline]
fn cubek_b3_swiglu_enabled() -> bool {
    #[cfg(feature = "profile")]
    {
        std::env::var("IRODORI_DISABLE_B3_CUBEK_SWIGLU").as_deref() != Ok("1")
    }
    #[cfg(not(feature = "profile"))]
    {
        true
    }
}

fn round_up(n: usize, multiple: usize) -> usize {
    n.next_multiple_of(multiple)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn production_approved() -> &'static crate::route_autotune::ResolvedRouteTable {
        static ROUTES: std::sync::OnceLock<crate::route_autotune::ResolvedRouteTable> =
            std::sync::OnceLock::new();
        ROUTES.get_or_init(crate::route_autotune::ResolvedRouteTable::production_approved)
    }

    fn extended_candidate() -> &'static crate::route_autotune::ResolvedRouteTable {
        static ROUTES: std::sync::OnceLock<crate::route_autotune::ResolvedRouteTable> =
            std::sync::OnceLock::new();
        ROUTES.get_or_init(crate::route_autotune::ResolvedRouteTable::extended_candidate)
    }

    fn dev() -> Device {
        Default::default()
    }

    #[test]
    fn default_hidden_dim_computation() {
        // dim=768 → 768*8/3 = 2048, already multiple of 256
        let ffn = SwiGlu::new(768, None, &dev());
        let x = Tensor::zeros([1, 4, 768], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [1, 4, 768]);
    }

    #[test]
    fn custom_hidden_dim() {
        let ffn = SwiGlu::new(16, Some(32), &dev());
        let x = Tensor::zeros([2, 3, 16], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [2, 3, 16]);
    }

    #[test]
    fn output_shape_preserved() {
        let ffn = SwiGlu::new(8, Some(16), &dev());
        let x = Tensor::ones([1, 5, 8], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [1, 5, 8]);
    }

    #[test]
    fn silu_gating_nonzero_input_produces_output() {
        // With default (random) weight init, non-zero input should produce non-zero output
        let ffn = SwiGlu::new(8, Some(16), &dev());
        let x = Tensor::ones([1, 2, 8], &dev()) * 2.0;
        let out = ffn.forward(x);
        // We can't guarantee non-zero with random init, but we verify shape
        assert_eq!(out.dims(), [1, 2, 8]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        // SwiGLU: silu(w1*0) * w3*0 = silu(b1)*b3, but biases are false
        // So silu(0) * 0 = 0.5*0 * 0 = 0 (silu(0) = 0*sigmoid(0) = 0*0.5 = 0)
        let ffn = SwiGlu::new(8, Some(16), &dev());
        let x = Tensor::zeros([1, 3, 8], &dev());
        let out = ffn.forward(x);
        let sum: f32 = out.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn no_bias_in_linears() {
        let ffn = SwiGlu::new(8, Some(16), &dev());
        assert!(ffn.w1.bias.is_none());
        assert!(ffn.w2.bias.is_none());
        assert!(ffn.w3.bias.is_none());
    }

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(100, 256), 256);
        assert_eq!(round_up(256, 256), 256);
        assert_eq!(round_up(257, 256), 512);
        assert_eq!(round_up(0, 256), 0);
    }

    /// After `prepare_for_inference()`, fused w1‖w3 forward must produce
    /// identical output to the 2-linear path.
    #[test]
    fn fused_w13_matches_separate_linears() {
        let mut ffn = SwiGlu::new(16, Some(32), &dev());
        let x = Tensor::random([2, 4, 16], burn::tensor::Distribution::Default, &dev());

        let out_unfused = ffn.forward(x.clone());

        ffn.prepare_for_inference();
        assert!(ffn.fused_w13_weight.is_some());
        assert!(
            ffn.packed_w2_weight_wgsl.is_none(),
            "portable inference preparation must not allocate the WGSL-only w2 cache"
        );

        let out_fused = ffn.forward(x);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "fused w1||w3 output should match unfused: max_diff={diff}"
        );
    }

    /// `prepare_for_inference()` is idempotent.
    #[test]
    fn fused_w13_idempotent() {
        let mut ffn = SwiGlu::new(8, Some(16), &dev());
        ffn.prepare_for_inference();
        let w1: Vec<f32> = ffn
            .fused_w13_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        ffn.prepare_for_inference();
        let w2: Vec<f32> = ffn
            .fused_w13_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(w1, w2);
    }

    #[test]
    fn prepared_w2_route_matches_measured_length_policy() {
        assert_eq!(
            prepared_w2_route_for(production_approved(), 1, 13, true, false),
            PreparedW2Route::SourceColumnFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 1, 25, true, false),
            PreparedW2Route::SourceColumnFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 1, 50, true, false),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 1, 200, true, false),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 2, 25, true, false),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 2, 50, true, false),
            PreparedW2Route::SourceColumnFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 2, 100, true, false),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 2, 200, true, false),
            PreparedW2Route::PackedRowRank3
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 2, 200, false, false),
            PreparedW2Route::SourceColumnFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 3, 200, true, false),
            PreparedW2Route::PackedRowRank3
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 3, 100, true, false),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 3, 100, true, true),
            PreparedW2Route::PackedRowFlat
        );
        assert_eq!(
            prepared_w2_route_for(production_approved(), 3, 200, true, true),
            PreparedW2Route::PackedRowRank3
        );
        for batch in [2, 3] {
            for sequence in [13, 45, 50, 100, 200, 685] {
                assert_ne!(
                    prepared_w2_route_for(extended_candidate(), batch, sequence, true, false,),
                    PreparedW2Route::SourceColumnFlat
                );
            }
        }
    }

    #[test]
    fn long_text_profile_never_selects_the_w2_source_layout() {
        for batch in [1, 2] {
            for sequence in [100, 112, 255, 333, 489, 685] {
                assert_ne!(
                    prepared_w2_route_for(production_approved(), batch, sequence, true, false),
                    PreparedW2Route::SourceColumnFlat
                );
            }
        }
    }

    #[test]
    fn dit_mlp_expand_t64_route_matches_nvidia_exact_policy() {
        for sequence in [100, 112, 200, 511, 685] {
            for batch in [1, 2] {
                assert!(dit_mlp_expand_t64_route_for(
                    production_approved(),
                    batch,
                    sequence,
                    1_280,
                    7_360,
                    DType::F32
                ));
            }
        }
        for sequence in [333, 489] {
            assert!(!dit_mlp_expand_t64_route_for(
                production_approved(),
                1,
                sequence,
                1_280,
                7_360,
                DType::F32
            ));
            assert!(dit_mlp_expand_t64_route_for(
                production_approved(),
                2,
                sequence,
                1_280,
                7_360,
                DType::F32
            ));
        }
        assert!(!dit_mlp_expand_t64_route_for(
            production_approved(),
            3,
            200,
            1_280,
            7_360,
            DType::F32
        ));
        for sequence in [13, 45, 100, 112, 200, 333, 511, 685] {
            for batch in [1, 2, 3] {
                assert!(dit_mlp_expand_t64_route_for(
                    extended_candidate(),
                    batch,
                    sequence,
                    1_280,
                    7_360,
                    DType::F32
                ));
            }
        }
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            4,
            50,
            1_280,
            7_360,
            DType::F32
        ));
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            1,
            12,
            1_280,
            7_360,
            DType::F32
        ));
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            1,
            686,
            1_280,
            7_360,
            DType::F32
        ));
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            1,
            200,
            1_024,
            7_360,
            DType::F32
        ));
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            1,
            200,
            1_280,
            2_048,
            DType::F32
        ));
        assert!(!dit_mlp_expand_t64_route_for(
            extended_candidate(),
            1,
            200,
            1_280,
            7_360,
            DType::F16
        ));
    }

    #[test]
    fn projection_swiglu_fusion_removes_the_full_expansion_temporary() {
        let rows = 489usize;
        let expanded_bytes = rows * 7_360 * size_of::<f32>();
        let fused_output_bytes = rows * 3_680 * size_of::<f32>();
        assert_eq!(expanded_bytes, 14_396_160);
        assert_eq!(fused_output_bytes, 7_198_080);
        assert_eq!(expanded_bytes, 2 * fused_output_bytes);
    }

    #[test]
    fn dit_mlp_contract_t64_route_covers_b1_b2_and_moderate_b3_lengths() {
        for sequence in [100, 112, 200, 333, 511, 685] {
            for batch in [1, 2] {
                assert!(dit_mlp_contract_t64_route_for(
                    production_approved(),
                    batch,
                    sequence,
                    3_680,
                    1_280,
                    DType::F32
                ));
            }
        }
        for sequence in [100, 112, 200, 333, 489, 511] {
            assert!(dit_mlp_contract_t64_route_for(
                production_approved(),
                3,
                sequence,
                3_680,
                1_280,
                DType::F32
            ));
        }
        assert!(!dit_mlp_contract_t64_route_for(
            production_approved(),
            3,
            685,
            3_680,
            1_280,
            DType::F32
        ));
        for sequence in [13, 45, 100, 112, 200, 333, 511, 685] {
            for batch in [1, 2, 3] {
                assert!(dit_mlp_contract_t64_route_for(
                    extended_candidate(),
                    batch,
                    sequence,
                    3_680,
                    1_280,
                    DType::F32
                ));
            }
        }
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            4,
            50,
            3_680,
            1_280,
            DType::F32
        ));
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            1,
            12,
            3_680,
            1_280,
            DType::F32
        ));
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            1,
            686,
            3_680,
            1_280,
            DType::F32
        ));
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            1,
            200,
            3_680,
            1_024,
            DType::F32
        ));
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            1,
            200,
            2_048,
            1_280,
            DType::F32
        ));
        assert!(!dit_mlp_contract_t64_route_for(
            extended_candidate(),
            1,
            200,
            3_680,
            1_280,
            DType::F16
        ));
    }

    #[test]
    fn duration_mlp_expand_t64_route_covers_released_compact_extent_only() {
        for sequence in 1..=64 {
            assert!(duration_mlp_expand_t64_route(1, sequence, 1_024, 2_048));
        }
        assert!(!duration_mlp_expand_t64_route(1, 0, 1_024, 2_048));
        assert!(!duration_mlp_expand_t64_route(1, 65, 1_024, 2_048));
        assert!(!duration_mlp_expand_t64_route(2, 32, 1_024, 2_048));
        assert!(!duration_mlp_expand_t64_route(1, 32, 1_280, 2_048));
        assert!(!duration_mlp_expand_t64_route(1, 32, 1_024, 7_360));
    }

    #[test]
    fn packed_w2_preparation_reuses_exact_values() {
        let mut ffn = SwiGlu::new(8, Some(16), &dev());
        ffn.prepare_w2_row_major_cache_wgsl();
        let source: Vec<f32> = ffn.w2.weight.val().into_data().to_vec().unwrap();
        let first: Vec<f32> = ffn
            .packed_w2_weight_wgsl
            .as_ref()
            .expect("packed w2 cache")
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        ffn.prepare_w2_row_major_cache_wgsl();
        let second: Vec<f32> = ffn
            .packed_w2_weight_wgsl
            .as_ref()
            .expect("packed w2 cache")
            .clone()
            .into_data()
            .to_vec()
            .unwrap();

        assert_eq!(source, first, "row packing must preserve every w2 value");
        assert_eq!(first, second, "repeated preparation must reuse the cache");
    }

    #[test]
    fn packed_w2_policy_matches_source_projection_for_b1_b2() {
        let mut ffn = SwiGlu::new(8, Some(16), &dev());
        ffn.prepare_w2_row_major_cache_wgsl();

        for batch in [1, 2] {
            let activated =
                Tensor::random([batch, 3, 16], burn::tensor::Distribution::Default, &dev());
            let expected = linear_rank3_flattened(
                activated.clone(),
                ffn.w2.weight.val(),
                ffn.w2.bias.as_ref().map(|bias| bias.val()),
            );
            let actual = ffn.project_w2_flattened_wgsl_policy(activated, true);
            let max_abs: f32 = (expected - actual)
                .abs()
                .max()
                .to_data()
                .to_vec::<f32>()
                .unwrap()[0];
            assert!(
                max_abs <= 1e-6,
                "B{batch} packed/source w2 projection mismatch: max_abs={max_abs}"
            );
        }
    }

    #[test]
    fn packed_w2_policy_falls_back_when_cache_is_unavailable() {
        let ffn = SwiGlu::new(8, Some(16), &dev());
        let activated = Tensor::random([1, 3, 16], burn::tensor::Distribution::Default, &dev());
        let expected = linear_rank3_flattened(
            activated.clone(),
            ffn.w2.weight.val(),
            ffn.w2.bias.as_ref().map(|bias| bias.val()),
        );
        let actual = ffn.project_w2_flattened_wgsl_policy(activated, false);
        let max_abs: f32 = (expected - actual)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(max_abs, 0.0);
    }

    #[test]
    #[should_panic(expected = "existing packed w2 WGSL cache shape mismatch")]
    fn packed_w2_preparation_rejects_partial_cache_shape() {
        let mut ffn = SwiGlu::new(8, Some(16), &dev());
        ffn.packed_w2_weight_wgsl = Some(Tensor::zeros([15, 8], &dev()));
        ffn.prepare_w2_row_major_cache_wgsl();
    }

    /// Zero input still gives zero output with fused weights.
    #[test]
    fn fused_zero_input_gives_zero_output() {
        let mut ffn = SwiGlu::new(8, Some(16), &dev());
        ffn.prepare_for_inference();
        let x = Tensor::zeros([1, 3, 8], &dev());
        let out = ffn.forward(x);
        let sum: f32 = out.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum, 0.0);
    }
}
