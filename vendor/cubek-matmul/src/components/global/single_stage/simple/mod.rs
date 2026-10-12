mod matmul;
mod setup;

pub use matmul::{DirectStagePartition, ErasedStagePartition, StagePartitionMode};
pub use setup::SimpleMatmulFamily;
