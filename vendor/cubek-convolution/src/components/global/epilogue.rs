//! Convolution epilogues implemented on CubeK's generic matmul writer hook.

use cubecl::prelude::*;
use cubek_matmul::components::global::GlobalEpilogue;

use super::args::RuntimeArgs;

/// DAC-style Snake activation using a channel-indexed parameter vector.
///
/// Computation is deliberately promoted to f32 even when the convolution
/// output is f16. This matches a standalone mixed-precision Snake kernel while
/// avoiding its dispatch and intermediate global-memory round trip.
pub struct SnakeEpilogue;

#[cube]
impl GlobalEpilogue<RuntimeArgs> for SnakeEpilogue {
    fn apply<E: Numeric>(value: E, coordinate: (u32, u32), runtime_config: &RuntimeArgs) -> E {
        let alpha = runtime_config.epilogue_param.unwrap();
        let x = f32::cast_from(value);
        let a = alpha.read(coordinate.1 as usize);
        let sine = (a * x).sin();
        E::cast_from(x + sine * sine / (a + 1.0e-9))
    }
}
