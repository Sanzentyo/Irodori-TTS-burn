//! Exact-shape benchmark for the production JointAttention materialization kernels.
//!
//! This starts after the accepted combined QKV+gate projection and ends at the
//! contiguous input to `wo`. It keeps Burn/CubeCL's tuned SDPA unchanged and
//! measures the two production fusions separately and together for B=1/2, S=50,
//! H=20, Dh=64, and compacted context length 3.
//!
//! Intended command after explicit GPU authorization:
//! `cargo run --release --bin bench_joint_attention_materializations -- <adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{
        Bool, Distribution, Tensor, TensorPrimitive,
        backend::Backend,
        module::{attention as burn_attention, linear},
        ops::AttentionModuleOptions,
    },
};
use irodori_tts_wgpu::{WgpuRaw, kernels::qkv_postprocess::fused_qkv_gate_postprocess_wgsl};

use irodori_tts_wgpu::kernels::joint_attention_materialization::{
    COMBINED_DIM, CONTEXT_LEN, DirectPackedKvOutput, HEAD_DIM, MODEL_DIM, NUM_HEADS,
    REFERENCE_SEQ_LEN, current_kv_cat_logical_bytes, direct_kv_saved_logical_bytes,
    direct_packed_kv_wgsl, direct_shared_bytes, post_sdpa_layout_gate_wgsl,
    post_sdpa_saved_logical_bytes,
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_TRIALS: usize = 7;
const SEED: u64 = 0;
const EPS: f64 = 1.0e-5;
const PARITY_TOLERANCE: f32 = 1.0e-6;
const RF_LAYER_CALLS: usize = 4 * 12;
const SEQ_LEN: usize = REFERENCE_SEQ_LEN;
const TOTAL_KV_LEN: usize = SEQ_LEN + CONTEXT_LEN;

#[derive(Debug)]
struct Args {
    adapter_index: Option<usize>,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            adapter_index: None,
            warmup: DEFAULT_WARMUP,
            iterations: DEFAULT_ITERATIONS,
            trials: DEFAULT_TRIALS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone)]
struct Inputs {
    q_weight: Tensor<B, 2>,
    k_weight: Tensor<B, 2>,
    qk_weight: Tensor<B, 3>,
    rope_cos: Tensor<B, 2>,
    rope_sin: Tensor<B, 2>,
    ctx_k: Tensor<B, 4>,
    ctx_v: Tensor<B, 4>,
    ctx_kv: Tensor<B, 5>,
    mask: Tensor<B, 2, Bool>,
    wo_weight: Tensor<B, 2>,
}

struct Prepared {
    q: Tensor<B, 4>,
    k_all: Tensor<B, 4>,
    v_all: Tensor<B, 4>,
    combined: Tensor<B, 3>,
}

#[derive(Clone, Copy)]
enum PreparePath {
    Current,
    DirectPackedKv,
}

#[derive(Clone, Copy)]
enum PostPath {
    Current,
    LayoutGate,
}

fn usage() -> &'static str {
    "usage: bench_joint_attention_materializations [adapter-index] \
     [--warmup N] [--iterations N] [--trials N]"
}

fn next_positive_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let parsed = value.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {value:?} for {option}: {error}"))
    })?;
    if parsed == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(parsed)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--warmup" => parsed.warmup = next_positive_usize(&mut args, "--warmup")?,
            "--iterations" => parsed.iterations = next_positive_usize(&mut args, "--iterations")?,
            "--trials" => parsed.trials = next_positive_usize(&mut args, "--trials")?,
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ if argument.starts_with('-') => {
                return Err(
                    io::Error::other(format!("unknown option {argument:?}; {}", usage())).into(),
                );
            }
            _ if parsed.adapter_index.is_none() => {
                parsed.adapter_index = Some(argument.parse::<usize>().map_err(|error| {
                    io::Error::other(format!(
                        "invalid adapter index {argument:?}: {error}; {}",
                        usage()
                    ))
                })?);
            }
            _ => {
                return Err(io::Error::other(format!(
                    "unexpected positional argument {argument:?}; {}",
                    usage()
                ))
                .into());
            }
        }
    }
    Ok(parsed)
}

fn make_inputs(batch: usize, device: &<B as Backend>::Device) -> Inputs {
    let q_weight = Tensor::random(
        [NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(0.5, 1.5),
        device,
    );
    let k_weight = Tensor::random(
        [NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(0.5, 1.5),
        device,
    );
    let qk_weight = Tensor::<B, 2>::stack::<3>(vec![q_weight.clone(), k_weight.clone()], 0);
    let ctx_k = Tensor::random(
        [batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let ctx_v = Tensor::random(
        [batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let ctx_kv = Tensor::<B, 4>::stack::<5>(vec![ctx_k.clone(), ctx_v.clone()], 0);
    let mask = Tensor::<B, 2>::ones([batch, TOTAL_KV_LEN], device).greater_elem(0.0);
    Inputs {
        q_weight,
        k_weight,
        qk_weight,
        rope_cos: Tensor::random(
            [SEQ_LEN, HEAD_DIM / 2],
            Distribution::Uniform(-1.0, 1.0),
            device,
        ),
        rope_sin: Tensor::random(
            [SEQ_LEN, HEAD_DIM / 2],
            Distribution::Uniform(-1.0, 1.0),
            device,
        ),
        ctx_k,
        ctx_v,
        ctx_kv,
        mask,
        wo_weight: Tensor::random(
            [MODEL_DIM, MODEL_DIM],
            Distribution::Uniform(-0.025, 0.025),
            device,
        ),
    }
}

fn copy_combined(input: Tensor<B, 3>) -> Tensor<B, 3> {
    // One-input cat intentionally produces independent physical storage. The
    // QKV shaders update sigmoid(gate) in place, so benchmark paths must not
    // alias one another. This setup copy is outside every measured closure.
    Tensor::cat(vec![input], 0)
}

fn cube_to_tensor3(
    tensor: burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
) -> Tensor<B, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(tensor))
}

fn cube_to_tensor4(
    tensor: burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
) -> Tensor<B, 4> {
    Tensor::from_primitive(TensorPrimitive::Float(tensor))
}

fn current_prepare(combined: Tensor<B, 3>, inputs: &Inputs) -> Prepared {
    let output = fused_qkv_gate_postprocess_wgsl(
        combined.into_primitive().tensor(),
        inputs.q_weight.clone().into_primitive().tensor(),
        inputs.k_weight.clone().into_primitive().tensor(),
        inputs.rope_cos.clone().into_primitive().tensor(),
        inputs.rope_sin.clone().into_primitive().tensor(),
        EPS,
    );
    let q = cube_to_tensor4(output.qkv.q);
    let k_self = cube_to_tensor4(output.qkv.k);
    let v_self = cube_to_tensor4(output.qkv.v);
    Prepared {
        q,
        k_all: Tensor::cat(vec![k_self, inputs.ctx_k.clone()], 1),
        v_all: Tensor::cat(vec![v_self, inputs.ctx_v.clone()], 1),
        combined: cube_to_tensor3(output.combined),
    }
}

fn direct_prepare(combined: Tensor<B, 3>, inputs: &Inputs) -> Prepared {
    let DirectPackedKvOutput {
        q,
        k_all,
        v_all,
        combined,
    } = direct_packed_kv_wgsl(
        combined.into_primitive().tensor(),
        inputs.qk_weight.clone().into_primitive().tensor(),
        inputs.rope_cos.clone().into_primitive().tensor(),
        inputs.rope_sin.clone().into_primitive().tensor(),
        inputs.ctx_kv.clone().into_primitive().tensor(),
        EPS,
    );
    Prepared {
        q: cube_to_tensor4(q),
        k_all: cube_to_tensor4(k_all),
        v_all: cube_to_tensor4(v_all),
        combined: cube_to_tensor3(combined),
    }
}

fn prepare(path: PreparePath, combined: Tensor<B, 3>, inputs: &Inputs) -> Prepared {
    match path {
        PreparePath::Current => current_prepare(combined, inputs),
        PreparePath::DirectPackedKv => direct_prepare(combined, inputs),
    }
}

fn tuned_sdpa_raw(prepared: Prepared, mask: Tensor<B, 2, Bool>) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let mask = mask.bool_not().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };
    let output = burn_attention(
        prepared.q.swap_dims(1, 2),
        prepared.k_all.swap_dims(1, 2),
        prepared.v_all.swap_dims(1, 2),
        Some(mask),
        None,
        options,
    );
    (output, prepared.combined)
}

fn current_post_sdpa(attention: Tensor<B, 4>, combined: Tensor<B, 3>) -> Tensor<B, 3> {
    let batch = attention.dims()[0];
    let token_major = attention
        .swap_dims(1, 2)
        .reshape([batch, SEQ_LEN, MODEL_DIM]);
    let gate = combined.narrow(2, 3 * MODEL_DIM, MODEL_DIM);
    gate * token_major
}

fn fused_post_sdpa(attention: Tensor<B, 4>, combined: Tensor<B, 3>) -> Tensor<B, 3> {
    cube_to_tensor3(post_sdpa_layout_gate_wgsl(
        attention.into_primitive().tensor(),
        combined.into_primitive().tensor(),
    ))
}

fn finish_post_sdpa(
    path: PostPath,
    attention: Tensor<B, 4>,
    combined: Tensor<B, 3>,
) -> Tensor<B, 3> {
    match path {
        PostPath::Current => current_post_sdpa(attention, combined),
        PostPath::LayoutGate => fused_post_sdpa(attention, combined),
    }
}

fn full_tail(
    combined: Tensor<B, 3>,
    inputs: &Inputs,
    prepare_path: PreparePath,
    post_path: PostPath,
) -> Tensor<B, 3> {
    let prepared = prepare(prepare_path, combined, inputs);
    let (attention, combined) = tuned_sdpa_raw(prepared, inputs.mask.clone());
    finish_post_sdpa(post_path, attention, combined)
}

fn wo_projection(input: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let batch = input.dims()[0];
    linear(input.reshape([batch * SEQ_LEN, MODEL_DIM]), weight, None)
        .reshape([batch, SEQ_LEN, MODEL_DIM])
}

fn max_abs_diff<const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn check_error(name: &str, error: f32) -> Result<(), Box<dyn Error>> {
    if error.is_finite() && error <= PARITY_TOLERANCE {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{name} max_abs={error:.3e} exceeds tolerance {PARITY_TOLERANCE:.3e}"
        ))
        .into())
    }
}

fn sync_3d(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn sync_4d(tensor: Tensor<B, 4>) {
    let _ = tensor.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

fn sync_5d(tensor: Tensor<B, 5>) {
    let _ = tensor.slice([0..1, 0..1, 0..1, 0..1, 0..1]).into_data();
}

fn measure<T, F, S>(args: &Args, mut operation: F, synchronize: S) -> Timing
where
    F: FnMut() -> T,
    S: Fn(T),
{
    let mut warmup_output = None;
    for _ in 0..args.warmup {
        warmup_output = Some(operation());
    }
    synchronize(warmup_output.expect("warmup count must be non-zero"));

    let mut samples = Vec::with_capacity(args.trials);
    for _ in 0..args.trials {
        let started = Instant::now();
        let mut output = None;
        for _ in 0..args.iterations {
            output = Some(operation());
        }
        synchronize(output.expect("iteration count must be non-zero"));
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / args.iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    Timing {
        median_us: samples[samples.len() / 2],
        min_us: samples[0],
        max_us: samples[samples.len() - 1],
    }
}

fn print_timing(label: &str, timing: Timing, baseline: Option<Timing>) {
    if let Some(baseline) = baseline {
        println!(
            "  {label:28} median={:.1} us [{:.1}, {:.1}], speedup={:.3}x",
            timing.median_us,
            timing.min_us,
            timing.max_us,
            baseline.median_us / timing.median_us,
        );
    } else {
        println!(
            "  {label:28} median={:.1} us [{:.1}, {:.1}]",
            timing.median_us, timing.min_us, timing.max_us,
        );
    }
}

fn pack_logical_bytes(batch: usize) -> usize {
    let qk_plane = NUM_HEADS * HEAD_DIM * size_of::<f32>();
    let ctx_plane = batch * CONTEXT_LEN * MODEL_DIM * size_of::<f32>();
    4 * (qk_plane + ctx_plane)
}

fn packed_persistent_bytes(batch: usize) -> usize {
    2 * NUM_HEADS * HEAD_DIM * size_of::<f32>()
        + 2 * batch * CONTEXT_LEN * MODEL_DIM * size_of::<f32>()
}

fn measure_pack(args: &Args, inputs: &Inputs) -> Timing {
    measure(
        args,
        || {
            let qk = Tensor::<B, 2>::stack::<3>(
                vec![inputs.q_weight.clone(), inputs.k_weight.clone()],
                0,
            );
            let ctx =
                Tensor::<B, 4>::stack::<5>(vec![inputs.ctx_k.clone(), inputs.ctx_v.clone()], 0);
            (qk, ctx)
        },
        |(_qk, ctx)| sync_5d(ctx),
    )
}

fn print_trajectory_delta(label: &str, baseline: Timing, optimized: Timing) {
    let per_layer_us = baseline.median_us - optimized.median_us;
    println!(
        "  {label:28} delta={per_layer_us:+.1} us/layer, {:+.3} ms/{RF_LAYER_CALLS} calls",
        per_layer_us * RF_LAYER_CALLS as f64 / 1_000.0,
    );
}

fn validate_batch(batch: usize, base: Tensor<B, 3>, inputs: &Inputs) -> Result<(), Box<dyn Error>> {
    let current = current_prepare(copy_combined(base.clone()), inputs);
    let direct = direct_prepare(copy_combined(base), inputs);

    let q_error = max_abs_diff(current.q.clone(), direct.q.clone())?;
    let k_error = max_abs_diff(current.k_all.clone(), direct.k_all.clone())?;
    let v_error = max_abs_diff(current.v_all.clone(), direct.v_all.clone())?;
    let gate_error = max_abs_diff(current.combined.clone(), direct.combined.clone())?;
    for (name, error) in [
        ("Q", q_error),
        ("K_all", k_error),
        ("V_all", v_error),
        ("combined/gate", gate_error),
    ] {
        check_error(name, error)?;
    }

    let (current_attention, current_combined) = tuned_sdpa_raw(current, inputs.mask.clone());
    let (direct_attention, direct_combined) = tuned_sdpa_raw(direct, inputs.mask.clone());
    let sdpa_error = max_abs_diff(current_attention.clone(), direct_attention.clone())?;
    check_error("tuned SDPA", sdpa_error)?;

    let current_wo_input = current_post_sdpa(current_attention.clone(), current_combined);
    let direct_wo_input = fused_post_sdpa(direct_attention, direct_combined);
    let wo_input_error = max_abs_diff(current_wo_input.clone(), direct_wo_input.clone())?;
    check_error("wo input", wo_input_error)?;

    let output_primitive = direct_wo_input.clone().into_primitive().tensor();
    assert!(
        output_primitive.is_contiguous(),
        "fused wo input must be contiguous"
    );
    assert_eq!(
        &output_primitive.meta.strides()[..],
        &[SEQ_LEN * MODEL_DIM, MODEL_DIM, 1],
        "fused wo input stride mismatch"
    );
    let current_wo = wo_projection(current_wo_input, inputs.wo_weight.clone());
    let direct_wo = wo_projection(direct_wo_input, inputs.wo_weight.clone());
    let wo_error = max_abs_diff(current_wo, direct_wo)?;
    check_error("wo projection", wo_error)?;

    println!(
        "B={batch} parity: Q={q_error:.3e}, K_all={k_error:.3e}, V_all={v_error:.3e}, \
         combined/gate={gate_error:.3e}, SDPA={sdpa_error:.3e}, wo_input={wo_input_error:.3e}, \
         wo={wo_error:.3e}"
    );
    Ok(())
}

fn benchmark_batch(
    args: &Args,
    batch: usize,
    device: &<B as Backend>::Device,
) -> Result<(), Box<dyn Error>> {
    let inputs = make_inputs(batch, device);
    let base = Tensor::random(
        [batch, SEQ_LEN, COMBINED_DIM],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    validate_batch(batch, base.clone(), &inputs)?;
    let pack_timing = measure_pack(args, &inputs);

    let current_prepare_input = copy_combined(base.clone());
    let direct_prepare_input = copy_combined(base.clone());
    let current_prepare_timing = measure(
        args,
        || current_prepare(current_prepare_input.clone(), &inputs).v_all,
        sync_4d,
    );
    let direct_prepare_timing = measure(
        args,
        || direct_prepare(direct_prepare_input.clone(), &inputs).v_all,
        sync_4d,
    );

    // Create one stable, sigmoid-processed gate and tuned-SDPA output. Neither
    // post-SDPA path mutates these inputs, so this isolates only layout+gate.
    let post_prepared = current_prepare(copy_combined(base.clone()), &inputs);
    let (post_attention, post_combined) = tuned_sdpa_raw(post_prepared, inputs.mask.clone());
    sync_4d(post_attention.clone());
    let current_post_timing = measure(
        args,
        || current_post_sdpa(post_attention.clone(), post_combined.clone()),
        sync_3d,
    );
    let fused_post_timing = measure(
        args,
        || fused_post_sdpa(post_attention.clone(), post_combined.clone()),
        sync_3d,
    );

    let current_full_input = copy_combined(base.clone());
    let direct_full_input = copy_combined(base.clone());
    let post_full_input = copy_combined(base.clone());
    let combined_full_input = copy_combined(base);
    let current_full = measure(
        args,
        || {
            full_tail(
                current_full_input.clone(),
                &inputs,
                PreparePath::Current,
                PostPath::Current,
            )
        },
        sync_3d,
    );
    let direct_full = measure(
        args,
        || {
            full_tail(
                direct_full_input.clone(),
                &inputs,
                PreparePath::DirectPackedKv,
                PostPath::Current,
            )
        },
        sync_3d,
    );
    let post_full = measure(
        args,
        || {
            full_tail(
                post_full_input.clone(),
                &inputs,
                PreparePath::Current,
                PostPath::LayoutGate,
            )
        },
        sync_3d,
    );
    let combined_full = measure(
        args,
        || {
            full_tail(
                combined_full_input.clone(),
                &inputs,
                PreparePath::DirectPackedKv,
                PostPath::LayoutGate,
            )
        },
        sync_3d,
    );

    println!("B={batch} separated:");
    print_timing("current QKV-post + K/V cat", current_prepare_timing, None);
    print_timing(
        "direct packed K/V",
        direct_prepare_timing,
        Some(current_prepare_timing),
    );
    print_timing("current SDPA layout + gate", current_post_timing, None);
    print_timing(
        "fused SDPA layout + gate",
        fused_post_timing,
        Some(current_post_timing),
    );
    println!("B={batch} full tail through wo input:");
    print_timing("current", current_full, None);
    print_timing("direct K/V only", direct_full, Some(current_full));
    print_timing("layout+gate only", post_full, Some(current_full));
    print_timing("both production fusions", combined_full, Some(current_full));
    println!("B={batch} four-step x 12-layer projection:");
    print_trajectory_delta(
        "direct K/V separated",
        current_prepare_timing,
        direct_prepare_timing,
    );
    print_trajectory_delta(
        "layout+gate separated",
        current_post_timing,
        fused_post_timing,
    );
    print_trajectory_delta("direct K/V full-tail", current_full, direct_full);
    print_trajectory_delta("layout+gate full-tail", current_full, post_full);
    print_trajectory_delta("both fusions full-tail", current_full, combined_full);
    let gross_combined_ms =
        (current_full.median_us - combined_full.median_us) * RF_LAYER_CALLS as f64 / 1_000.0;
    let pack_all_layers_ms = pack_timing.median_us * 12.0 / 1_000.0;
    println!(
        "  one-time pack/layer median={:.1} us [{:.1}, {:.1}], logical={:.3}MiB, \
         persistent={:.3}MiB; x12={pack_all_layers_ms:.3} ms; combined net={:+.3} ms",
        pack_timing.median_us,
        pack_timing.min_us,
        pack_timing.max_us,
        pack_logical_bytes(batch) as f64 / (1024.0 * 1024.0),
        packed_persistent_bytes(batch) as f64 / (1024.0 * 1024.0),
        gross_combined_ms - pack_all_layers_ms,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let device = args
        .adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);
    println!(
        "JointAttention production materializations device={device:?}, S={SEQ_LEN}, H={NUM_HEADS}, \
         Dh={HEAD_DIM}, ctx={CONTEXT_LEN}, warmup={}, iterations={}, trials={}",
        args.warmup, args.iterations, args.trials,
    );
    println!(
        "production resources: direct WG32/shared={}B/bindings=8; post WG256/shared=0B/bindings=3",
        direct_shared_bytes(),
    );
    for batch in [1, 2] {
        println!(
            "B={batch} traffic: current K/V cats={:.3}MiB/layer; direct saves={:.3}MiB/layer \
             ({:.3}MiB/{RF_LAYER_CALLS} calls, 192 dispatches); post saves={:.3}MiB/layer \
             ({:.3}MiB/{RF_LAYER_CALLS} calls, 48 dispatches)",
            current_kv_cat_logical_bytes(batch, SEQ_LEN) as f64 / (1024.0 * 1024.0),
            direct_kv_saved_logical_bytes(batch, SEQ_LEN) as f64 / (1024.0 * 1024.0),
            (RF_LAYER_CALLS * direct_kv_saved_logical_bytes(batch, SEQ_LEN)) as f64
                / (1024.0 * 1024.0),
            post_sdpa_saved_logical_bytes(batch, SEQ_LEN) as f64 / (1024.0 * 1024.0),
            (RF_LAYER_CALLS * post_sdpa_saved_logical_bytes(batch, SEQ_LEN)) as f64
                / (1024.0 * 1024.0),
        );
        benchmark_batch(&args, batch, &device)?;
    }
    Ok(())
}
