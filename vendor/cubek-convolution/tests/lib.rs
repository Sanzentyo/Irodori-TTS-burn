use cubek_convolution::{
    components::global::epilogue::{NoPostCastEpilogue, PreparedSnakeEpilogue, SnakeEpilogue},
    routines::{
        Routine,
        simple::{
            SimpleSyncCyclicConv, SimpleSyncCyclicPostCastEpilogueConv,
            SimpleSyncCyclicStridedPostCastEpilogueConv,
        },
    },
};

fn assert_standard<R: Routine<PostCastEpilogue = NoPostCastEpilogue>>() {}
fn assert_snake<R: Routine<PostCastEpilogue = SnakeEpilogue>>() {}
fn assert_prepared_snake<R: Routine<PostCastEpilogue = PreparedSnakeEpilogue>>() {}

#[test]
fn standard_and_parameterized_routines_have_distinct_launch_contracts() {
    assert_standard::<SimpleSyncCyclicConv>();
    assert_snake::<SimpleSyncCyclicPostCastEpilogueConv<SnakeEpilogue>>();
    assert_snake::<SimpleSyncCyclicStridedPostCastEpilogueConv<SnakeEpilogue>>();
    assert_prepared_snake::<SimpleSyncCyclicPostCastEpilogueConv<PreparedSnakeEpilogue>>();
}
