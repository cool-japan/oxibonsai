//! Dense FP32 GEMM: `out[m × n] = a[m × k] · w[n × k]ᵀ (+ bias[n])`.
//!
//! The batched sibling of [`gemv_f32`], in the
//! weight layout that function and the dense linear layer already share: one
//! weight **row** per output feature. `a` holds `m` input rows of `k` values,
//! `w` holds `n` weight rows of `k` values, and `out[i * n + j]` is the dot
//! product of input row `i` with weight row `j`, plus `bias[j]` when a bias is
//! given. Everything is row-major `f32`.
//!
//! # Bit-exactness
//!
//! Every output element is **bit-identical** to the corresponding element of
//! calling [`gemv_f32`] once per row of `a` (and then
//! adding the bias). It is not a similar-looking accumulation: the elements
//! come out of `dot_tile` (in `gemv_f32`), the routine
//! [`dot_f32`](crate::gemv_f32::dot_f32) is the `1 × 1` instance of, so the
//! lane accumulators, their fold order and the remainder handling are the
//! ones the GEMV uses, by construction. The bias is added to the finished dot
//! product (`dot + bias[j]`, one rounding) and never seeds an accumulator.
//! The result therefore does not depend on the tile a value was computed in,
//! on the number of Rayon threads, or on whether the call ran in parallel at
//! all.
//!
//! NaN and infinity follow the ordinary IEEE rules of that arithmetic: a NaN
//! in row `i` of `a` makes output row `i` NaN, a NaN in row `j` of `w` (or in
//! `bias[j]`) makes output column `j` NaN, and nothing else is affected. Which
//! NaN *payload* comes out is whatever the hardware's NaN propagation picks
//! and is not part of the contract.
//!
//! # Where the speed comes from
//!
//! Not from a new accumulation order — from the traversal:
//!
//! * **Register tiles.** The output is computed eight elements at a time
//!   (`4 × 2`, `2 × 4` or `1 × 8` dot products per step, whichever fits the
//!   rows left) in one loop nest. One dot product alone is two vector
//!   accumulator chains of dependent adds, which leaves the FP pipes idle;
//!   eight interleaved dot products (sixteen chains) keep them full, and every
//!   operand chunk loaded feeds several outputs.
//! * **Macro tiles.** The output is cut into `8 × 64` blocks (8 rows of `a`
//!   against 64 rows of `w`). A block keeps its 64 weight rows hot in L2 and
//!   its 8 input rows in L1 while it runs, so the weight matrix is streamed
//!   once per 8 input rows instead of once per row.
//! * **Rayon over blocks.** Blocks are independent, so they are the parallel
//!   work items, ordered so that the blocks that share a weight tile are
//!   adjacent (and so usually run on one thread). Calls below
//!   [`PAR_MIN_MACS`] multiply-accumulates, and calls made from a
//!   single-thread pool, run on the calling thread. (A single input row is
//!   not tiled at all: see the contract summary.)
//!
//! The per-row loop this replaces re-streams the whole weight matrix for every
//! input row and pays one Rayon fork/join per row. Both effects grow with `m`,
//! which is what a batched caller (a ViT block over hundreds of patches, a
//! prefill chunk) has plenty of.
//!
//! The tile shapes were chosen on an Apple M3 (four performance and four
//! efficiency cores) and are not tuned for x86-64. Register tiles of eight
//! dot products (`4 × 2`, `2 × 4`) measured within noise of each other; four
//! dot products (`2 × 2`) run about a third as fast, too few independent
//! chains to hide the add latency; sixteen (`4 × 4`) spill the accumulators
//! and lose a little. Macro tiles from `8 × 32` to `32 × 32` and `16 × 64`
//! were within noise of `8 × 64`. The `2 × 4` and `1 × 8` shapes exist for
//! the row remainder: with only `4 × 2` (and single dot products for the
//! tail) a call with `m < 4` ran no faster than the per-row loop.
//!
//! # Example
//!
//! ```
//! use oxibonsai_kernels::gemm_f32;
//!
//! // Two input rows of three values; two weight rows (one per output feature).
//! let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let w = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
//! let bias = [10.0f32, 20.0];
//! let mut out = [0.0f32; 4];
//! gemm_f32(&a, &w, Some(&bias), 2, 3, 2, &mut out)?;
//! assert_eq!(out, [11.0, 26.0, 14.0, 35.0]);
//! # Ok::<(), oxibonsai_kernels::KernelError>(())
//! ```
//!
//! # Contract summary
//!
//! * `m == 0` is a no-op that returns `Ok` without inspecting anything else.
//! * `m == 1` is a GEMV and runs as one (`gemv_f32`, then the bias): a single
//!   row has no weight reuse for tiling to exploit. It therefore follows the
//!   GEMV's own parallel threshold (`PAR_MIN_ROWS`), not [`PAR_MIN_MACS`].
//! * Slices longer than the shapes need are accepted (the `gemv_f32`
//!   convention): only the leading `m * k` of `a`, `n * k` of `w`, `n` of
//!   `bias` and `m * n` of `out` are read or written, and the rest of `out` is
//!   left untouched.
//! * Unlike `gemv_f32`, an **empty** `w` is not the all-zero projection: with
//!   `n * k > 0` it is a length mismatch and reported as one. A general
//!   matrix multiply that silently returned zeros for a weight matrix that
//!   failed to load would hide the bug.
//! * No caller-supplied length or shape can make this panic; every mismatch is
//!   a typed [`KernelError`].

use crate::error::{KernelError, KernelResult};
use crate::gemv_f32::{dot_tile, gemv_f32};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Rows of `a` (output rows) per macro tile.
const TILE_ROWS: usize = 8;

/// Rows of `w` (output columns) per macro tile.
const TILE_COLS: usize = 64;

/// Tiled calls (`m >= 2`) with fewer than this many multiply-accumulates
/// (`m * n * k`) run on the calling thread.
///
/// A Rayon fork/join costs tens of microseconds when the pool has to be woken,
/// and `2^19` multiply-accumulates is roughly 50 µs of the tiled kernel on an
/// Apple M3 core. The value is deliberately conservative: with a pool that is
/// already running, fanning out pays from about `2^17`, but a call that small
/// gains little either way and a cold pool would lose. It only decides
/// *where* the arithmetic runs: the result is bit-identical on either side of
/// it.
pub const PAR_MIN_MACS: usize = 1 << 19;

/// `out[m × n] = a[m × k] · w[n × k]ᵀ (+ bias[n])`, row-major, bit-identical to
/// [`gemv_f32`] run once per row of `a` (see the
/// [module docs](self) for what that guarantees and how the speed is won).
///
/// `bias`, when given, is added to each finished dot product:
/// `out[i * n + j] = dot(a[i], w[j]) + bias[j]`.
///
/// # Errors
///
/// - [`KernelError::UnsupportedOperation`] when `m * k`, `n * k` or `m * n`
///   overflows `usize` (checked first, so a nonsensical shape is reported as
///   such rather than as a short buffer).
/// - [`KernelError::NamedDimensionMismatch`] naming `"input"` when
///   `a.len() < m * k`, `"weights"` when `w.len() < n * k` (including an empty
///   `w`), or `"bias"` when a bias holds fewer than `n` values.
/// - [`KernelError::NamedBufferTooSmall`] naming `"output"` when
///   `out.len() < m * n`.
///
/// `m == 0` returns `Ok(())` before any of that is checked.
pub fn gemm_f32(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> KernelResult<()> {
    if m == 0 {
        return Ok(());
    }
    let a_len = checked_extent("input", m, k)?;
    let w_len = checked_extent("weights", n, k)?;
    let out_len = checked_extent("output", m, n)?;
    if a.len() < a_len {
        return Err(KernelError::NamedDimensionMismatch {
            name: "input",
            expected: a_len,
            got: a.len(),
        });
    }
    if w.len() < w_len {
        return Err(KernelError::NamedDimensionMismatch {
            name: "weights",
            expected: w_len,
            got: w.len(),
        });
    }
    if let Some(bias) = bias {
        if bias.len() < n {
            return Err(KernelError::NamedDimensionMismatch {
                name: "bias",
                expected: n,
                got: bias.len(),
            });
        }
    }
    if out.len() < out_len {
        return Err(KernelError::NamedBufferTooSmall {
            name: "output",
            needed: out_len,
            available: out.len(),
        });
    }
    if n == 0 {
        return Ok(());
    }
    if m == 1 {
        return single_row(
            &a[..k],
            &w[..w_len],
            bias.map(|bias| &bias[..n]),
            k,
            &mut out[..n],
        );
    }

    let operands = Operands {
        a: &a[..a_len],
        w: &w[..w_len],
        bias: bias.map(|bias| &bias[..n]),
        k,
    };
    let out = &mut out[..out_len];

    #[cfg(not(target_arch = "wasm32"))]
    if should_parallelize(m, k, n) {
        operands.run_parallel(m, n, out);
        return Ok(());
    }
    operands.run_sequential(m, n, out);
    Ok(())
}

/// A single input row is a GEMV: `gemv_f32`, then the bias added to each
/// finished dot product (the same one rounding as in the tiled path).
///
/// One row streams every weight row exactly once, so there is no reuse for
/// tiling to win, and the GEMV's row-parallel loop streams a large weight
/// matrix better than register tiles do (the tiled path measured about 0.7x of
/// it on a `1152 × 4304` matrix). It is also exactly what a per-row loop of
/// one would have run, so a batch of one costs nothing extra.
fn single_row(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    out: &mut [f32],
) -> KernelResult<()> {
    gemv_f32(w, a, out, out.len(), k)?;
    if let Some(bias) = bias {
        for (value, &b) in out.iter_mut().zip(bias) {
            *value += b;
        }
    }
    Ok(())
}

/// `rows * cols`, or the typed error naming the operand whose extent
/// overflows.
fn checked_extent(name: &str, rows: usize, cols: usize) -> KernelResult<usize> {
    rows.checked_mul(cols).ok_or_else(|| {
        KernelError::UnsupportedOperation(format!(
            "gemm_f32: the {name} extent {rows} x {cols} overflows usize"
        ))
    })
}

/// Whether a call of this shape runs on the Rayon pool: enough work to repay
/// the hand-off, at least two macro tiles to hand out, and a pool with more
/// than one thread.
#[cfg(not(target_arch = "wasm32"))]
fn should_parallelize(m: usize, k: usize, n: usize) -> bool {
    let macs = m.saturating_mul(n).saturating_mul(k);
    let tiles = m.div_ceil(TILE_ROWS).saturating_mul(n.div_ceil(TILE_COLS));
    macs >= PAR_MIN_MACS && tiles > 1 && rayon::current_num_threads() > 1
}

/// The validated operands of one call, shared by every tile: `a` is exactly
/// `m * k`, `w` exactly `n * k` and `bias` exactly `n` long.
struct Operands<'a> {
    a: &'a [f32],
    w: &'a [f32],
    bias: Option<&'a [f32]>,
    k: usize,
}

impl Operands<'_> {
    /// Every block on the calling thread, weight-tile-major so each `w` tile
    /// stays hot across all the row blocks that use it.
    fn run_sequential(&self, m: usize, n: usize, out: &mut [f32]) {
        for j0 in (0..n).step_by(TILE_COLS) {
            let nb = (n - j0).min(TILE_COLS);
            for i0 in (0..m).step_by(TILE_ROWS) {
                let mb = (m - i0).min(TILE_ROWS);
                self.block(i0, mb, j0, nb, &mut |r, c, value| {
                    out[(i0 + r) * n + j0 + c] = value;
                });
            }
        }
    }

    /// Every block as a Rayon work item.
    ///
    /// A block writes `mb` short runs of one output column range, which is not
    /// a contiguous piece of `out`, so the disjoint `&mut` pieces are cut up
    /// front, once per call: one segment per (output row, column tile), laid
    /// out column-tile-major and padded to whole row groups so that
    /// `TILE_ROWS` consecutive segments are one block's rows. That costs
    /// `O(m * n / TILE_COLS)` pointers — noise next to the `m * n * k`
    /// multiply-accumulates — and keeps the whole thing in safe Rust.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_parallel(&self, m: usize, n: usize, out: &mut [f32]) {
        let col_tiles = n.div_ceil(TILE_COLS);
        let row_groups = m.div_ceil(TILE_ROWS);
        let mut cursors: Vec<std::slice::ChunksMut<'_, f32>> = out
            .chunks_mut(n)
            .map(|row| row.chunks_mut(TILE_COLS))
            .collect();
        let mut segments: Vec<&mut [f32]> = Vec::with_capacity(col_tiles * row_groups * TILE_ROWS);
        for col_tile in 0..col_tiles {
            for cursor in &mut cursors {
                segments.extend(cursor.next());
            }
            // Pad this column tile's rows out to a whole number of blocks with
            // empty segments (never read: a block only touches its first `mb`
            // rows, and `mb` comes from `m`, not from the padding).
            segments.resize_with((col_tile + 1) * row_groups * TILE_ROWS, Default::default);
        }
        segments
            .par_chunks_mut(TILE_ROWS)
            .enumerate()
            .for_each(|(job, rows)| {
                let (col_tile, row_group) = (job / row_groups, job % row_groups);
                let i0 = row_group * TILE_ROWS;
                let j0 = col_tile * TILE_COLS;
                let mb = (m - i0).min(TILE_ROWS);
                let nb = rows.first().map_or(0, |segment| segment.len());
                self.block(i0, mb, j0, nb, &mut |r, c, value| {
                    rows[r][c] = value;
                });
            });
    }

    /// The `mb × nb` block whose top-left output is `(i0, j0)`, handing each
    /// finished value to `store(row, col, value)` with block-relative
    /// coordinates.
    ///
    /// Rows are consumed four at a time, then two, then one, and each group of
    /// rows is paired with as many weight rows as keep the same number of
    /// accumulator chains live (`4 × 2`, `2 × 4`, `1 × 8` — eight dot products
    /// each), so the tail of a block, and a whole small-`m` call, does not fall
    /// off the fast path.
    fn block<S: FnMut(usize, usize, f32)>(
        &self,
        i0: usize,
        mb: usize,
        j0: usize,
        nb: usize,
        store: &mut S,
    ) {
        let mut ri = 0;
        while ri < mb {
            match mb - ri {
                4.. => {
                    self.row_group::<4, 2, 1, S>(i0, ri, j0, nb, store);
                    ri += 4;
                }
                2..=3 => {
                    self.row_group::<2, 4, 2, S>(i0, ri, j0, nb, store);
                    ri += 2;
                }
                _ => {
                    self.row_group::<1, 8, 4, S>(i0, ri, j0, nb, store);
                    ri += 1;
                }
            }
        }
    }

    /// `MR` rows of the block against all `nb` weight rows, in panels of `N1`
    /// weight rows where they fit, then `N2`, then one at a time.
    #[inline(always)]
    fn row_group<const MR: usize, const N1: usize, const N2: usize, S: FnMut(usize, usize, f32)>(
        &self,
        i0: usize,
        ri: usize,
        j0: usize,
        nb: usize,
        store: &mut S,
    ) {
        let mut cj = 0;
        while cj < nb {
            let left = nb - cj;
            if left >= N1 {
                self.panel::<MR, N1, S>(i0, ri, j0, cj, store);
                cj += N1;
            } else if left >= N2 {
                self.panel::<MR, N2, S>(i0, ri, j0, cj, store);
                cj += N2;
            } else {
                self.panel::<MR, 1, S>(i0, ri, j0, cj, store);
                cj += 1;
            }
        }
    }

    /// One `MR × NR` register tile: the dot products of input rows
    /// `i0 + ri ..` with weight rows `j0 + cj ..`, bias added, stored.
    #[inline(always)]
    fn panel<const MR: usize, const NR: usize, S: FnMut(usize, usize, f32)>(
        &self,
        i0: usize,
        ri: usize,
        j0: usize,
        cj: usize,
        store: &mut S,
    ) {
        let a_rows: [&[f32]; MR] = std::array::from_fn(|r| row(self.a, i0 + ri + r, self.k));
        let w_rows: [&[f32]; NR] = std::array::from_fn(|s| row(self.w, j0 + cj + s, self.k));
        let tile = dot_tile::<MR, NR>(a_rows, w_rows);
        for (r, tile_row) in tile.iter().enumerate() {
            for (s, &dot) in tile_row.iter().enumerate() {
                let value = match self.bias {
                    Some(bias) => dot + bias[j0 + cj + s],
                    None => dot,
                };
                store(ri + r, cj + s, value);
            }
        }
    }
}

/// Row `index` of the row-major `[rows × k]` matrix `matrix`.
///
/// The callers only ask for rows below the validated `m` / `n`, so the range
/// is always in bounds; the slice index is that invariant made explicit.
#[inline(always)]
fn row(matrix: &[f32], index: usize, k: usize) -> &[f32] {
    &matrix[index * k..(index + 1) * k]
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::gemv_f32::dot_f32;

    const SENTINEL_BITS: u32 = 0x7FC0_DEAD;

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 40) as u32 as f32) / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// The element-by-element definition: `dot_f32(a[i], w[j]) (+ bias[j])`.
    fn definition(
        a: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let dot = dot_f32(&a[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
                out.push(bias.map_or(dot, |bias| dot + bias[j]));
            }
        }
        out
    }

    fn assert_bit_identical(context: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{context}: length");
        for (idx, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{context}: element {idx}");
        }
    }

    /// The two drivers, called directly so the parallel decomposition runs
    /// whatever the parallel threshold and the host's core count, must agree
    /// with the element-by-element definition (and so with each other) on
    /// shapes that leave a partial macro tile in both directions and a
    /// remainder for every register tile.
    #[test]
    fn both_drivers_reproduce_the_definition_on_partial_tiles() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build a rayon pool");
        for (m, k, n) in [
            (1usize, 5usize, 1usize),
            (1, 9, 70),
            (3, 1, 3),
            (5, 2, 66),
            (8, 16, 64),
            (9, 13, 65),
            (16, 16, 128),
            (17, 8, 129),
            (33, 31, 200),
        ] {
            let a = values(m * k, 1);
            let w = values(n * k, 2);
            let bias = values(n, 3);
            for bias in [None, Some(bias.as_slice())] {
                let want = definition(&a, &w, bias, m, k, n);
                let operands = Operands {
                    a: &a,
                    w: &w,
                    bias,
                    k,
                };
                let mut sequential = vec![f32::from_bits(SENTINEL_BITS); m * n];
                operands.run_sequential(m, n, &mut sequential);
                assert_bit_identical(
                    &format!("sequential {m}x{k}x{n} bias={}", bias.is_some()),
                    &sequential,
                    &want,
                );
                let mut parallel = vec![f32::from_bits(SENTINEL_BITS); m * n];
                pool.install(|| operands.run_parallel(m, n, &mut parallel));
                assert_bit_identical(
                    &format!("parallel {m}x{k}x{n} bias={}", bias.is_some()),
                    &parallel,
                    &want,
                );
            }
        }
    }

    /// The parallel decision needs enough work, more than one macro tile to
    /// hand out, and a pool with more than one thread — and nothing else.
    #[test]
    fn the_parallel_decision_needs_work_tiles_and_threads() {
        let pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build a rayon pool")
        };
        pool(4).install(|| {
            assert!(!should_parallelize(1, 1, 1), "trivial work stays put");
            // Two macro tiles' worth of columns, and exactly the threshold.
            assert!(should_parallelize(8, PAR_MIN_MACS / (8 * 65) + 1, 65));
            assert!(
                !should_parallelize(8, PAR_MIN_MACS / (8 * 65) - 1, 65),
                "just under the threshold stays on the calling thread"
            );
            // Any amount of work in one macro tile has nothing to split.
            assert!(!should_parallelize(8, 1 << 20, TILE_COLS));
            assert!(!should_parallelize(1, 1 << 20, 1));
            // A huge shape cannot overflow the work estimate.
            assert!(should_parallelize(usize::MAX, usize::MAX, usize::MAX));
        });
        pool(1).install(|| {
            assert!(
                !should_parallelize(64, 4096, 4096),
                "a single-thread pool never fans out"
            );
        });
    }
}
