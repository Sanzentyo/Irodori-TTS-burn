use crate::multi_level::args::RuntimeConfig;
use crate::multi_level::components::global::memory::GlobalMemoryConfig;
use crate::multi_level::components::global::{
    PlaneFlowPartitionRule, WriteEventListener, WriteTiling,
};
use crate::multi_level::components::stage::{Stage, StageFamily};
use crate::multi_level::definition;
use crate::multi_level::definition::MatrixTypes;
use crate::multi_level::stage::StageMemoryConfig;
use cubecl::prelude::*;
use cubecl::std::tensor::ViewMut;
use cubecl::std::tensor::layout::Coords2d;
use cubek_std::InvalidConfigError;

pub type WriterStage<GW, RC, MT> = <<GW as GlobalWriterFamily<RC>>::Stage as StageFamily>::Stage<
    definition::Stage<MT>,
    definition::StageSize<MT>,
    WriteTiling,
>;

pub trait GlobalWriterFamily<RC: RuntimeConfig>: 'static + Send + Sync {
    type Stage: StageFamily;
    type Writer<'a, IP: MatrixTypes>: GlobalWriter<
            'a,
            IP,
            RC,
            Stage = <Self::Stage as StageFamily>::Stage<IP::Stage, IP::StageSize, WriteTiling>,
        >;

    /// Verify that configs are valid for a writer, otherwise return an error stating why
    fn validate_with_config(_config: &GlobalWriterConfig) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<'a, IP: MatrixTypes, RC: RuntimeConfig>:
    WriteEventListener + CubeType + 'a
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage>;

    /// Init this writer from a global tensor and config
    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        valid_shape: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    /// Stage used by this writer
    fn stage(this: &Self) -> Self::Stage;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalWriterConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub plane_flow_partition_rule: PlaneFlowPartitionRule,
    pub plane_dim: u32,
}
