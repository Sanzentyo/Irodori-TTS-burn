//! NVTX profiling helpers for nsys/nvtx range annotations.
//!
//! All public items compile to no-ops when the `profile` feature is disabled,
//! so there is zero runtime overhead in normal (non-profiling) builds.
//!
//! # Usage
//!
//! ```ignore
//! // Wrap an expression in an NVTX range (name shows up in nsys timeline).
//! nvtx_range!("attention_forward", {
//!     self.compute_attention(q, k, v, mask)
//! });
//!
//! // Or use the scoped guard directly for longer blocks.
//! let _guard = NvtxRange::new("dit_forward");
//! // ... work ...
//! drop(_guard); // or let it fall out of scope
//! ```
//!
//! Build with profiling enabled:
//! ```sh
//! just bench-cuda-profile
//! # or
//! cargo run --release --features "cli,profile" --bin bench_realmodel -- --backend cuda
//! ```

// ── NVTX-enabled path ────────────────────────────────────────────────────────

#[cfg(feature = "profile")]
pub use nvtx_impl::*;

#[cfg(feature = "profile")]
mod nvtx_impl {
    /// RAII guard that pushes an NVTX range on construction and pops on drop.
    pub struct NvtxRange;

    impl NvtxRange {
        /// Begin a named NVTX range. The range ends when the guard is dropped.
        #[inline]
        pub fn new(name: &str) -> Self {
            nvtx::range_push!("{}", name);
            Self
        }
    }

    impl Drop for NvtxRange {
        #[inline]
        fn drop(&mut self) {
            nvtx::range_pop!();
        }
    }

    /// Wrap `$expr` in an NVTX range named `$name`.
    ///
    /// The range spans exactly the evaluation of `$expr`.
    #[macro_export]
    macro_rules! nvtx_range {
        ($name:expr, $expr:expr) => {{
            let _guard = $crate::profiling::NvtxRange::new($name);
            $expr
        }};
    }
}

// ── No-op path (default) ─────────────────────────────────────────────────────

#[cfg(not(feature = "profile"))]
#[allow(unused_imports)]
pub use noop_impl::*;

#[cfg(not(feature = "profile"))]
mod noop_impl {
    /// Zero-cost stub: does nothing when `profile` feature is disabled.
    #[allow(dead_code)]
    pub struct NvtxRange;

    #[allow(dead_code)]
    impl NvtxRange {
        #[inline(always)]
        pub fn new(_name: &str) -> Self {
            Self
        }
    }

    /// No-op when `profile` feature is disabled; compiles away entirely.
    #[macro_export]
    macro_rules! nvtx_range {
        ($name:expr, $expr:expr) => {
            $expr
        };
    }
}
