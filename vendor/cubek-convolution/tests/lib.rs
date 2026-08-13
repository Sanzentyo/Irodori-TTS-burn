use cubek_convolution::{
    components::global::epilogue::{NoPostCastEpilogue, SnakeEpilogue},
    routines::{
        Routine,
        simple::{SimpleSyncCyclicConv, SimpleSyncCyclicPostCastEpilogueConv},
    },
};

fn assert_standard<R: Routine<PostCastEpilogue = NoPostCastEpilogue>>() {}
fn assert_snake<R: Routine<PostCastEpilogue = SnakeEpilogue>>() {}

#[test]
fn standard_and_parameterized_routines_have_distinct_launch_contracts() {
    assert_standard::<SimpleSyncCyclicConv>();
    assert_snake::<SimpleSyncCyclicPostCastEpilogueConv<SnakeEpilogue>>();
}
