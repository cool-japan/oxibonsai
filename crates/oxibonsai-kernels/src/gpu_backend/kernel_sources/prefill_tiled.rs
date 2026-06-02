//! Tiled fused-Metal TQ2_0_g128 GEMM kernel (`v8`) for the **large-M** path.
//!
//! `gemm_tq2_g128_v7` (in [`super::prefill`]) was designed for LLM decode
//! (`M = 1`) / small prefill: its grid is `[ceil(N/8), 1, 1]` and it walks the
//! batch *serially* in 8-column outer chunks, reloading + re-decoding each
//! weight row once **per 8 columns**. For the DiT (`M` up to 1536) that is
//! `M/8 = 192` serial chunks with 192× redundant weight decode and *zero*
//! M-parallelism in the grid, so the GPU runs far below peak.
//!
//! `gemm_tq2_g128_v8_tiled` fixes this with a register-blocked tiled GEMM:
//!
//! * **2-D grid** — `x` over N-tiles (`TN` weight rows), `y` over M-tiles
//!   (`TM` batch columns). This is the key change: the grid now parallelizes
//!   `M`.
//! * **K-tiling by 128** (`TK = 128` = exactly one ternary block per K-step).
//! * Per K-block each threadgroup stages, into threadgroup memory,
//!   `TN × 128` **dequantized** weights (decoded once per M-tile, vs `v7`'s
//!   `M/8` re-decodes) and `TM × 128` input values.
//! * **Register blocking** — each thread accumulates a *column* of `RN = 4`
//!   output rows in registers. Per staged K-element it loads ONE A value
//!   (reused across all `RN` rows) and `RN` weights → `RN` FMAs, giving high
//!   arithmetic intensity. A large `TN` is what slashes how many times the
//!   `A` matrix is re-streamed from global memory (`N / TN` passes) — the
//!   dominant cost at the big DiT `N` (e.g. `27648`).
//!
//! Tiles: `TN = 32`, `TM = 16`, `TK = 128`, `RN = 4` output rows per thread →
//! `(TN / RN) × TM = 8 × 16 = 128` threads / threadgroup. Thread `(rg, tm)`
//! owns output rows `[row_base + rg·RN .. +RN)` at column `col_base + tm`.
//!
//! Threadgroup-memory budget:
//! * `Wsh`: `32 × 128 × 4 B = 16 KiB`
//! * `Ash`: `16 × 128 × 4 B =  8 KiB`
//! * total `24 KiB` ≤ the 32 KiB Apple-Silicon threadgroup limit (M3 max
//!   `maxThreadgroupMemoryLength = 32768`).
//!
//! This register-blocked config measures **~2.7× faster than `v7`** at the DiT
//! shapes (`M = 1536`, `N ∈ {3072, 27648}`, `K = 3072`), versus ~1.37× for the
//! naïve one-output-per-thread tiling it replaced.
//!
//! Decode (`v8_decode_tq2`) is byte-for-byte identical to `v7`'s
//! `pf_decode_tq2` (`{0 → -1, 1 → 0, 2 → +1, 3 → 0}`) and to the CPU reference
//! `BlockTQ2_0_g128::dequant`, so `v8` is numerically equivalent to `v7`.
//!
//! Weight buffer is the **same** SoA layout `v7` reads (no upload / reformat
//! change): for `nb = N · (K/128)` blocks,
//! `[0 : nb·2)` = f16 scales (block `b` at byte `b·2`),
//! `[nb·2 : nb·2 + nb·32)` = qs (block `b` at byte `nb·2 + b·32`),
//! block `b = r·(K/128) + kb` covering `W[r, kb·128 .. kb·128+128]`.
//!
//! Boundaries: `N` / `M` tile edges are clamped with `min(...)` so the kernel
//! is correct for arbitrary (non-tile-multiple) shapes used by the unit tests
//! (e.g. `N = 40`, `M = 33`). `K` is required to be a multiple of 128 (the
//! caller validates this), so K needs no partial-block guard.

/// V8 tiled batched ternary (TQ2_0_g128) GEMM.
///
/// Computes `out[M,N] = A[M,K] · dequant(B[N,K])ᵀ` with the same column-major
/// buffer conventions as `gemm_tq2_g128_v7`:
/// `inputs[col·k + elem]`, `outputs[col·n_rows + row]` (`col = m`, `row = n`).
///
/// Buffers:
/// - buffer(0) = `soa_raw`    (u8, TQ2_0_g128 weights, SoA `[N·(K/128)·2 B scales][·32 B qs]`)
/// - buffer(1) = `inputs`     (f32, `M × K`, column-major == row-major `[M,K]`)
/// - buffer(2) = `outputs`    (f32, `M × N`, column-major == row-major `[M,N]`)
/// - buffer(3) = `n_rows`     (u32, `N`)
/// - buffer(4) = `batch_size` (u32, `M`)
/// - buffer(5) = `k`          (u32, multiple of 128)
///
/// Dispatch: `[ceil(N/32), ceil(M/16), 1]` threadgroups, `[128, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_TQ2_G128_V8_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tile dimensions (must match dispatch_gemm_tq2_v8 in metal_dispatch.rs).
//
// Register-blocked: each thread computes a column of V8_RN output rows
// (rows reuse the single staged A value per K-element -> high arithmetic
// intensity, and a large V8_TN slashes how many times the A matrix is
// re-streamed from global = the dominant cost for the big-N DiT shapes).
//
//   threads/threadgroup = (V8_TN / V8_RN) * V8_TM = 8 * 16 = 128
//   threadgroup mem     = Wsh(32*128*4) + Ash(16*128*4) = 16 + 8 = 24 KiB
constant constexpr uint V8_TN = 32u;   // weight rows per threadgroup
constant constexpr uint V8_TM = 16u;   // batch columns per threadgroup
constant constexpr uint V8_TK = 128u;  // K elements per step = one ternary block
constant constexpr uint V8_RN = 4u;    // output rows accumulated per thread
constant constexpr uint V8_ROWGROUPS = V8_TN / V8_RN;          // 8
constant constexpr uint V8_THREADS   = V8_ROWGROUPS * V8_TM;   // 128
constant constexpr uint V8_WBYTES    = V8_TN * 32u;            // qs bytes per K-block (1024)
constant constexpr uint V8_AELEMS    = V8_TM * V8_TK;          // A floats per K-block (2048)

// 2-bit ternary decode, identical to v7 pf_decode_tq2 / CPU dequant:
//   code 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0 (reserved).
inline float v8_decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

kernel void gemm_tq2_g128_v8_tiled(
    device const uchar*  soa_raw    [[buffer(0)]],
    device const float*  inputs     [[buffer(1)]],
    device       float*  outputs    [[buffer(2)]],
    constant uint&       n_rows     [[buffer(3)]],
    constant uint&       batch_size [[buffer(4)]],
    constant uint&       k          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    // This threadgroup's output tile origin.
    const uint row_base = tgid.x * V8_TN;   // first weight row (N)
    const uint col_base = tgid.y * V8_TM;   // first batch column (M)

    // Thread -> (row-group, micro-column) within the tile.
    //   lid in [0, V8_THREADS) ; rg in [0, V8_ROWGROUPS) ; tm in [0, V8_TM).
    // Thread (rg, tm) owns output rows [rg*V8_RN .. rg*V8_RN+V8_RN) at column tm.
    const uint rg = lid / V8_TM;            // 0..7  -> row-group
    const uint tm = lid % V8_TM;            // 0..15 -> output col offset
    const uint row0 = row_base + rg * V8_RN;  // first absolute weight row for this thread
    const uint col  = col_base + tm;          // absolute batch column for this thread

    const uint blocks_per_row = k / V8_TK;            // K/128 ternary blocks per row
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;    // qs section starts after f16 scales

    // Threadgroup staging:
    //   Wsh[r][kk] = dequantized weight for tile row `r`,    k = kb*128 + kk.
    //   Ash[c][kk] = input for tile column `c`,              k = kb*128 + kk.
    threadgroup float Wsh[V8_TN][V8_TK];   // 32 * 128 * 4 = 16 KiB
    threadgroup float Ash[V8_TM][V8_TK];   // 16 * 128 * 4 =  8 KiB

    // Per-thread register accumulators (one per output row in the row-group).
    float acc[V8_RN];
    for (uint r = 0u; r < V8_RN; r++) acc[r] = 0.0f;

    // How many tile rows / columns are actually valid (boundary clamp).
    const uint valid_rows = (row_base < n_rows)     ? min(V8_TN, n_rows - row_base)     : 0u;
    const uint valid_cols = (col_base < batch_size) ? min(V8_TM, batch_size - col_base) : 0u;

    for (uint kb = 0u; kb < blocks_per_row; kb++) {
        // ── Stage V8_TN dequantized weight rows (V8_TN * 32 qs bytes).
        // Each thread decodes V8_WBYTES/V8_THREADS qs bytes (4 weights each).
        for (uint i = lid; i < V8_WBYTES; i += V8_THREADS) {
            const uint w_row_local = i / 32u;     // 0..V8_TN-1
            const uint w_byte      = i % 32u;     // 0..31 (byte within the 128-wide block)
            const uint base = w_byte * 4u;        // 4 weights per byte
            if (w_row_local < valid_rows) {
                const uint w_row    = row_base + w_row_local;
                const uint block_idx = w_row * blocks_per_row + kb;
                const float scale =
                    float(*(device const half*)(soa_raw + block_idx * 2u));
                const uchar byte = soa_raw[qs_offset + block_idx * 32u + w_byte];
                Wsh[w_row_local][base + 0u] = scale * v8_decode_tq2((uint(byte)      ) & 3u);
                Wsh[w_row_local][base + 1u] = scale * v8_decode_tq2((uint(byte) >> 2u) & 3u);
                Wsh[w_row_local][base + 2u] = scale * v8_decode_tq2((uint(byte) >> 4u) & 3u);
                Wsh[w_row_local][base + 3u] = scale * v8_decode_tq2((uint(byte) >> 6u) & 3u);
            } else {
                // Padded weight row: zero so out-of-range rows contribute 0.
                Wsh[w_row_local][base + 0u] = 0.0f;
                Wsh[w_row_local][base + 1u] = 0.0f;
                Wsh[w_row_local][base + 2u] = 0.0f;
                Wsh[w_row_local][base + 3u] = 0.0f;
            }
        }

        // ── Stage V8_TM input columns (V8_TM * 128 floats).
        // Read inputs[col*k + kb*128 + e].
        {
            const uint k_off = kb * V8_TK;
            for (uint i = lid; i < V8_AELEMS; i += V8_THREADS) {
                const uint a_col_local = i / V8_TK;   // 0..V8_TM-1
                const uint a_elem      = i % V8_TK;   // 0..127
                float v = 0.0f;
                if (a_col_local < valid_cols) {
                    const uint a_col = col_base + a_col_local;
                    v = inputs[a_col * k + k_off + a_elem];
                }
                Ash[a_col_local][a_elem] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Accumulate this thread's V8_RN outputs over the 128-wide block.
        // Each K-element loads ONE A value (reused across V8_RN rows) and
        // V8_RN W values -> V8_RN FMAs (high reuse from threadgroup memory).
        // Compute regardless of boundary: padded rows/cols were staged to 0.
        {
            threadgroup const float* acol = &Ash[tm][0];
            threadgroup const float* w0 = &Wsh[rg * V8_RN + 0u][0];
            threadgroup const float* w1 = &Wsh[rg * V8_RN + 1u][0];
            threadgroup const float* w2 = &Wsh[rg * V8_RN + 2u][0];
            threadgroup const float* w3 = &Wsh[rg * V8_RN + 3u][0];
            for (uint e = 0u; e < V8_TK; e++) {
                const float a = acol[e];
                acc[0] += w0[e] * a;
                acc[1] += w1[e] * a;
                acc[2] += w2[e] * a;
                acc[3] += w3[e] * a;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Write back (column-major: outputs[col*n_rows + row]).
    if (tm < valid_cols) {
        for (uint r = 0u; r < V8_RN; r++) {
            const uint row_local = rg * V8_RN + r;
            if (row_local < valid_rows) {
                outputs[col * n_rows + (row0 + r)] = acc[r];
            }
        }
    }
}
"#;
