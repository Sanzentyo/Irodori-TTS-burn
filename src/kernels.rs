//! Custom WGSL compute kernels for the WGPU backend.
//!
//! These kernels target the non-fusion WGPU backend (`CubeBackend<WgpuRuntime, ...>`)
//! and provide optimised implementations of hot-path operations in the DiT model.
//!
//! # Architecture
//!
//! Each sub-module contains:
//! - A `.wgsl` compute shader file
//! - A Rust launcher that implements [`KernelSource`] and handles buffer management
//!
//! # Platform compatibility
//!
//! All kernels use WGSL core features only (workgroup shared memory, barriers).
//! Vulkan/DX12/Metal-specific optimisations (subgroup ops) are commented as
//! future improvements for when the WGSL subgroups extension stabilises.
//!
pub mod conv1d_k7_residue_d1_snake;
pub mod conv1d_k7_snake_epilogue;
pub mod conv1d_k7_stem_direct;
pub mod conv1d_k7_t128;
pub mod conv1d_k7_t128_snake_epilogue;
pub mod conv1d_k7_t256_snake_epilogue;
pub mod conv1d_k7_t256_snake_vec4_store;
pub mod conv1d_k7_tiled;
pub mod conv1d_k7_tiled_o32;
pub mod conv1d_k7_tiled_o64;
pub mod conv_transpose1d_cached_col2im;
pub mod conv_transpose1d_cached_col2im_case0;
pub mod conv_transpose1d_polyphase;
pub mod dit_projection_t64;
pub mod duration_block_preprocess;
pub mod duration_output_finalize;
pub mod duration_residual_finalize;
pub mod duration_swiglu_w2;
#[allow(dead_code)]
pub mod fused_adaln;
pub mod fused_residual_gate;
pub mod fused_sdpa_native;
pub mod fused_swiglu;
pub mod joint_attention_materialization;
pub mod modern_bert_residual_layer_norm;
pub mod pointwise_residual_direct_tiled;
pub mod pointwise_residual_finalizer;
pub mod pointwise_residual_snake_pair;
pub mod qkv_postprocess;
#[allow(dead_code)]
pub mod rms_norm;
pub mod snake;
pub mod text_cfg_kv_derive;
pub mod wm_head_fused_final_t240_c16;
pub mod wm_head_snake_nlc;

#[cfg(test)]
mod tests {
    /// Raw [`burn_cubecl::template::SourceKernel`] shaders do not carry the
    /// generated CubeCL representation that normally upgrades every binding to
    /// read-write for a sliced memory pool. WGPU validates storage access for
    /// the shared physical buffer, not each logical allocation range, so a
    /// read-only/read-write mix can invalidate an otherwise non-overlapping
    /// dispatch.
    #[test]
    fn production_source_kernels_use_uniform_read_write_storage_bindings() {
        let shaders = [
            (
                "conv1d_k7_residue_pack",
                include_str!("kernels/conv1d_k7_residue_pack.wgsl"),
            ),
            (
                "conv1d_k7_residue_d1_snake",
                include_str!("kernels/conv1d_k7_residue_d1_snake.wgsl"),
            ),
            (
                "conv1d_k7_snake_epilogue",
                include_str!("kernels/conv1d_k7_snake_epilogue.wgsl"),
            ),
            (
                "conv1d_k7_stem_direct",
                include_str!("kernels/conv1d_k7_stem_direct.wgsl"),
            ),
            (
                "conv1d_k7_t128",
                include_str!("kernels/conv1d_k7_t128.wgsl"),
            ),
            (
                "conv1d_k7_t128_snake_epilogue",
                include_str!("kernels/conv1d_k7_t128_snake_epilogue.wgsl"),
            ),
            (
                "conv1d_k7_t256_snake_epilogue",
                include_str!("kernels/conv1d_k7_t256_snake_epilogue.wgsl"),
            ),
            (
                "conv1d_k7_t256_snake_vec4_store",
                include_str!("kernels/conv1d_k7_t256_snake_vec4_store.wgsl"),
            ),
            (
                "conv1d_k7_tiled",
                include_str!("kernels/conv1d_k7_tiled.wgsl"),
            ),
            (
                "conv1d_k7_tiled_o32",
                include_str!("kernels/conv1d_k7_tiled_o32.wgsl"),
            ),
            (
                "conv1d_k7_tiled_o64",
                include_str!("kernels/conv1d_k7_tiled_o64.wgsl"),
            ),
            (
                "conv_transpose1d_cached_col2im",
                include_str!("kernels/conv_transpose1d_cached_col2im.wgsl"),
            ),
            (
                "conv_transpose1d_polyphase",
                include_str!("kernels/conv_transpose1d_polyphase.wgsl"),
            ),
            (
                "conv_transpose1d_weight_pack",
                include_str!("kernels/conv_transpose1d_weight_pack.wgsl"),
            ),
            (
                "duration_output_finalize",
                include_str!("kernels/duration_output_finalize.wgsl"),
            ),
            (
                "duration_block_preprocess",
                include_str!("kernels/duration_block_preprocess.wgsl"),
            ),
            (
                "duration_input_projection_t64",
                include_str!("kernels/duration_input_projection_t64.wgsl"),
            ),
            (
                "duration_residual_finalize",
                include_str!("kernels/duration_residual_finalize.wgsl"),
            ),
            (
                "duration_swiglu_w2",
                include_str!("kernels/duration_swiglu_w2.wgsl"),
            ),
            (
                "duration_swiglu_w2_o64_vec4",
                include_str!("kernels/duration_swiglu_w2_o64_vec4.wgsl"),
            ),
            (
                "dit_projection_t64",
                include_str!("kernels/dit_projection_t64.wgsl"),
            ),
            ("fused_adaln", include_str!("kernels/fused_adaln.wgsl")),
            (
                "fused_residual_gate",
                include_str!("kernels/fused_residual_gate.wgsl"),
            ),
            ("fused_swiglu", include_str!("kernels/fused_swiglu.wgsl")),
            (
                "joint_attention_direct_kv",
                include_str!("kernels/joint_attention_direct_kv.wgsl"),
            ),
            (
                "joint_attention_post_sdpa",
                include_str!("kernels/joint_attention_post_sdpa.wgsl"),
            ),
            (
                "modern_bert_residual_layer_norm",
                include_str!("kernels/modern_bert_residual_layer_norm.wgsl"),
            ),
            (
                "pointwise_residual_direct_t64_o96_vec4_raw",
                include_str!("kernels/pointwise_residual_direct_t64_o96_vec4_raw.wgsl"),
            ),
            (
                "pointwise_residual_direct_t64_o96_vec4_pair",
                include_str!("kernels/pointwise_residual_direct_t64_o96_vec4_pair.wgsl"),
            ),
            (
                "pointwise_residual_finalizer",
                include_str!("kernels/pointwise_residual_finalizer.wgsl"),
            ),
            (
                "pointwise_residual_snake_pair",
                include_str!("kernels/pointwise_residual_snake_pair.wgsl"),
            ),
            ("rms_norm", include_str!("kernels/rms_norm.wgsl")),
            (
                "qkv_postprocess",
                include_str!("kernels/qkv_postprocess.wgsl"),
            ),
            ("snake", include_str!("kernels/snake.wgsl")),
            (
                "text_cfg_kv_derive",
                include_str!("kernels/text_cfg_kv_derive.wgsl"),
            ),
            (
                "wm_head_fused_final_t240_c16",
                include_str!("kernels/wm_head_fused_final_t240_c16.wgsl"),
            ),
            (
                "wm_head_snake_nlc",
                include_str!("kernels/wm_head_snake_nlc.wgsl"),
            ),
        ];

        for (name, shader) in shaders {
            let storage_bindings = shader
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
                .collect::<Vec<_>>();
            assert!(
                !storage_bindings.is_empty(),
                "{name} must declare at least one storage binding"
            );
            assert!(
                storage_bindings
                    .iter()
                    .all(|line| line.contains("var<storage, read_write>")),
                "{name} mixes storage access modes: {storage_bindings:?}"
            );
        }
    }
}
