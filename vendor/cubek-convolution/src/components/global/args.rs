use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::{View, ViewMut, layout::Coords1d},
    },
};
use half::f16;

use crate::components::{ConvolutionOperation, ConvolutionParams};

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct EpilogueRuntimeArgs {
    /// Optional read-only f32 parameters for scalar post-cast epilogues.
    pub f32_param: ComptimeOption<View<'static, f32, Coords1d>>,
    /// Generic f16 auxiliary inputs owned and validated by the transform type.
    pub f16_input_0: ComptimeOption<View<'static, f16, Coords1d>>,
    pub f16_input_1: ComptimeOption<View<'static, f16, Coords1d>>,
    /// Generic f16 auxiliary output owned and validated by the transform type.
    pub f16_output_0: ComptimeOption<ViewMut<'static, f16, Coords1d>>,
    /// Number of logical spatial rows per batch in the matrix output.
    pub output_rows: u32,
}

#[derive(CubeType, CubeLaunch, Clone)]
#[expand(derive(Clone))]
pub struct RuntimeArgs {
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod<u32>,
    #[cube(comptime)]
    pub operation: ConvolutionOperation,
    #[cube(comptime)]
    pub params: ConvolutionParams,
    /// Typed auxiliary bindings consumed by a custom output transform.
    /// Standard convolution routines leave every optional view absent.
    pub epilogue: EpilogueRuntimeArgs,
}
