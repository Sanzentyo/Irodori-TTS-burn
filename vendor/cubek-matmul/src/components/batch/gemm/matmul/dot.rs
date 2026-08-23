use cubecl::{cube, num_traits::Zero, std::tensor::View, std::tensor::layout::Coords2d};
use cubecl::{prelude::*, std::tensor::ViewMut};

use crate::components::batch::{
    CheckBounds,
    gemm::io::{read, write},
};
use crate::components::global::PairwiseAccumulatorGlobalEpilogue;

/// Plane-cooperative dot product over K — one output cell per plane.
///
/// Units within a plane share the K traversal in `plane_dim`-wide steps and
/// accumulate a `Vector<AccR, vs>` of partials; a final horizontal (and
/// cross-unit, when `plane_dim > 1`) sum produces the scalar to write. Tile
/// starts are swizzled by `plane_id` so concurrent planes hit K at staggered
/// offsets. When `plane_dim == 1` (CPU path) the cross-unit reduction
/// degenerates to a plain `Vector::vector_sum` and every plane writes its
/// own cell.
///
/// Layout precondition: lhs is row-major [M, K], rhs is col-major [K, N]
/// (i.e. K is the contiguous axis on both operands).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dot<
    L: CubePrimitive,
    R: CubePrimitive,
    O: CubePrimitive,
    AccR: Numeric,
    N: Size,
>(
    lhs: View<L, Coords2d>,
    rhs: View<R, Coords2d>,
    out: ViewMut<O, Coords2d>,
    m_pos: u32,
    n_pos: u32,
    k_dim: u32,
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let (out_m, out_n) = out.shape();
        if m_pos >= out_m || n_pos >= out_n {
            terminate!();
        }
    }

    let tile_size = plane_dim * vector_size;
    let num_tiles_k = k_dim / tile_size;

    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles_k {
        let swizzled_tile_index = (tile_index + plane_id) % num_tiles_k;
        let k_base = swizzled_tile_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        let lhs_val = read(&lhs, (m_pos, k_pos), check_bounds);
        let rhs_val = read(&rhs, (k_pos, n_pos), check_bounds);

        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    if comptime!(plane_dim > 1) {
        let sum = O::cast_from(plane_sum(Vector::vector_sum(acc)));
        if unit_id == 0 {
            write(out, (m_pos, n_pos), sum, check_bounds);
        }
    } else {
        let sum = O::cast_from(Vector::vector_sum(acc));
        write(out, (m_pos, n_pos), sum, check_bounds);
    };
}

/// Plane-cooperative dot product that evaluates one logical output column per
/// plane, then joins adjacent plane results through `E`.
///
/// This is the compressed-output counterpart of [`execute_dot`]. It preserves
/// the same K traversal, vectorization and plane-level parallelism while
/// avoiding a materialized `M x 2N` projection. Plane sums rendezvous in a
/// tiny cube-local array; even planes combine their sum with the adjacent odd
/// plane and write one physical `M x N` output cell.
///
/// All units must reach the cube barrier. In particular, the logical-N tail is
/// masked rather than terminated.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_pairwise_dot<
    L: CubePrimitive,
    R: CubePrimitive,
    EG: Numeric,
    NG: Size,
    AccR: Numeric,
    N: Size,
    E: PairwiseAccumulatorGlobalEpilogue<()>,
>(
    lhs: View<L, Coords2d>,
    rhs: View<R, Coords2d>,
    out: ViewMut<Vector<EG, NG>, Coords2d>,
    m_pos: u32,
    logical_n_pos: u32,
    k_dim: u32,
    logical_n: u32,
    #[comptime] plane_dim: u32,
    #[comptime] num_planes: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;
    let (out_m, _) = out.shape();
    let valid = m_pos < out_m && logical_n_pos < logical_n;
    let tile_size = plane_dim * vector_size;
    let num_tiles_k = k_dim / tile_size;
    let mut acc = Vector::<AccR, N>::zero();

    if valid {
        for tile_index in 0..num_tiles_k {
            let swizzled_tile_index = (tile_index + plane_id) % num_tiles_k;
            let k_base = swizzled_tile_index * plane_dim;
            let k_pos = (k_base + unit_id) * vector_size;
            let lhs_val = Vector::<AccR, N>::cast_from(read(
                &lhs,
                (m_pos, k_pos),
                check_bounds,
            ));
            let rhs_val = Vector::<AccR, N>::cast_from(read(
                &rhs,
                (k_pos, logical_n_pos),
                check_bounds,
            ));
            acc += lhs_val * rhs_val;
        }
    }

    let reduced = if comptime!(plane_dim > 1) {
        plane_sum(Vector::vector_sum(acc))
    } else {
        Vector::vector_sum(acc)
    };
    let mut plane_sums = Shared::<[AccR]>::new_slice(num_planes as usize);
    if unit_id == 0 {
        plane_sums[plane_id as usize] = reduced;
    }
    sync_cube();

    if unit_id == 0
        && plane_id.is_multiple_of(2)
        && valid
        && logical_n_pos + 1 < logical_n
    {
        let physical_n_pos = logical_n_pos / 2;
        let runtime_config = ();
        let value = E::apply::<AccR, EG>(
            plane_sums[plane_id as usize],
            plane_sums[plane_id as usize + 1],
            (m_pos, physical_n_pos),
            &runtime_config,
        );
        let mut output = Vector::<EG, NG>::empty();
        output.insert(0, value);
        write(out, (m_pos, physical_n_pos), output, check_bounds);
    }
}
