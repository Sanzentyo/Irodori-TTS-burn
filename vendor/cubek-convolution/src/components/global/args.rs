use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::{View, layout::Coords1d},
    },
};

use crate::components::{ConvolutionOperation, ConvolutionParams};

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
    /// Optional read-only parameters consumed by a custom output epilogue.
    /// Standard convolution routines leave this absent, so they acquire no
    /// extra storage binding or runtime cost.
    pub epilogue_param: ComptimeOption<View<'static, f32, Coords1d>>,
}
