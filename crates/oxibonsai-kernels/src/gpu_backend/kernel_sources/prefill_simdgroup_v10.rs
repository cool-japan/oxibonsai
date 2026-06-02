//! `simdgroup_matrix`-based fused-Metal TQ2_0_g128 GEMM kernel (`v10`) — a
//! staging-optimized evolution of [`super::prefill_simdgroup`] (`v9`) for the
//! **large-M** DiT path.
//!
//! `v9` (`gemm_tq2_g128_v9_simdgroup`, `TM=TN=64`, `TK=32`, 128 threads / 4
//! simdgroups, a `4×4` grid of `simdgroup_float8x8` accumulators) is
//! *staging-bound*: per K-tile its dequant-scatter (≈1 thread per weight row,
//! 8 qs bytes decoded serially) sits behind a `threadgroup_barrier`, so the 8×8
//! matrix units idle while the staging runs (it reaches only ~4-6 % of the M3
//! GPU peak). `v10` keeps the *exact* op, SoA weight buffer, decode bits,
//! boundary semantics, and the `64×64` / `4`-simdgroup / `128`-thread shape of
//! `v9`, but optimizes the staging on two axes that measured **~3.86× faster
//! than `v9`** on the two big DiT shapes (`M = 1536`, `N ∈ {3072, 27648}`,
//! `K = 3072`) with parity identical to `v9` (max-abs err ≈ 1.2e-5):
//!
//! * **Lever 1 — stage `D` as `half` (EXACT here).** `D[k,n] = dequant(B[n,k])
//!   = code × scale`, `code ∈ {-1,0,+1}`, and `scale` is the per-block f16
//!   already stored in the GGUF. So `code × scale ∈ {-scale, 0, +scale}` — every
//!   value is *exactly* representable in f16 (zero added rounding). (`v9`'s f16
//!   attempt staged **`A`**, the full-range f32 activations, which *do* lose
//!   precision over `K = 3072`; `D` is the different, exact case.) `v10` keeps
//!   **`A` in `f32`**, stages **`D` in `half`** (`Dsh` is `half[]`, 4 KiB vs the
//!   f32 8 KiB), loads the `D` fragment as `simdgroup_half8x8`, and
//!   **accumulates in `f32`** via the mixed-precision
//!   `simdgroup_multiply_accumulate(acc_f32, a_f32, d_half, acc_f32)`. This
//!   halves `D`'s threadgroup memory *and* its store bandwidth with parity
//!   identical to `v9` (`D` exact, `A` unchanged).
//! * **Lever 2 — vectorize the dequant-scatter.** `v9` activates only `TN = 64`
//!   threads for the scatter (one per weight row), each decoding its `QSLICE = 8`
//!   qs bytes one at a time. `v10` spreads the scatter over **all 128 threads**
//!   as `(n_local, qbyte)` work-items (`TN × QSLICE = 512` items, 4 per thread),
//!   decoding 4 codes per byte. This doubles the active threads and removes the
//!   per-byte serial decode, raising the staging throughput that gates the MACs.
//!
//! Two further levers were implemented and benchmarked but **did not pay off on
//! M3** and are therefore *not* adopted (the staging is occupancy-/register-
//! sensitive, so anything that shrinks resident threadgroups or spills
//! accumulators costs more than it saves):
//!
//! * **Lever 3 — double-buffer the K-tile staging.** Ping/pong `Ash`/`Dsh`
//!   slabs (24 KiB) to overlap staging with compute *regressed* to ~2.15× over
//!   `v9` (vs the single-buffered ~3.86×): the doubled tgmem cut occupancy more
//!   than the software-pipeline overlap recovered. So `v10` stays
//!   **single-buffered** (12 KiB, same footprint as `v9`).
//! * **Lever 4 — wider tiles.** `128×64` / `8`-simdgroup (20 KiB) measured
//!   ~3.37×, and `128×128` / `TK=16` / `8`-simdgroup with 32 accumulators/thread
//!   *collapsed* to ~0.31× (register spilling). Neither beat the `64×64`
//!   single-buffered config, so `v10` keeps `v9`'s `64×64` tile.
//!
//! Decode (`v10_decode_tq2`) is byte-for-byte identical to `v9`/`v8`/`v7`
//! (`{0 → -1, 1 → 0, 2 → +1, 3 → 0}`) and to the CPU reference
//! `BlockTQ2_0_g128::dequant`. The weight buffer is the **same** SoA layout
//! (`[0 : nb·2)` f16 scales, `[nb·2 : nb·2+nb·32)` qs, block
//! `b = r·(K/128) + kb`) — no upload/reformat change. Boundary clamps
//! (`min(...)`, padded rows/cols staged as zero) match `v9`, so `v10` is correct
//! for arbitrary (non-tile-multiple) shapes. `K` must be a multiple of 128;
//! `TK = 32` divides 128, so a K-tile never straddles two ternary blocks.

/// V10 `simdgroup_matrix` batched ternary (TQ2_0_g128) GEMM — staging-optimized
/// `v9` (f16-`D` staging + vectorized dequant-scatter; single-buffered `64×64`).
///
/// Computes `out[M,N] = A[M,K] · dequant(B[N,K])ᵀ` with the same column-major
/// buffer conventions as `gemm_tq2_g128_v7/v8/v9`: `inputs[col·k + elem]`,
/// `outputs[col·n_rows + row]` (`col = m`, `row = n`).
///
/// Buffers:
/// - buffer(0) = `soa_raw`    (u8, TQ2_0_g128 weights, SoA `[N·(K/128)·2 B scales][·32 B qs]`)
/// - buffer(1) = `inputs`     (f32, `M × K`, column-major == row-major `[M,K]`)
/// - buffer(2) = `outputs`    (f32, `M × N`, column-major == row-major `[M,N]`)
/// - buffer(3) = `n_rows`     (u32, `N`)
/// - buffer(4) = `batch_size` (u32, `M`)
/// - buffer(5) = `k`          (u32, multiple of 128)
///
/// Dispatch: `[ceil(N/64), ceil(M/64), 1]` threadgroups,
/// `[V10_SIMDGROUPS·32, 1, 1] = [128, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_TQ2_G128_V10_SIMDGROUP: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tile / simdgroup geometry (must match dispatch_gemm_tq2_v10 in metal_dispatch.rs).
//
//   output tile          : V10_TM x V10_TN = 64 x 64
//   simdgroups/threadgroup: V10_SIMDGROUPS = 4  (-> 128 threads)
//   per-simdgroup quadrant: V10_SG_M x V10_SG_N = 32 x 32
//   accumulator grid      : (V10_SG_M/8) x (V10_SG_N/8) = 4 x 4 simdgroup_float8x8
//   K step                : V10_TK = 32 (= 4 fragments of 8; divides 128)
//   threadgroup memory (single-buffered, same footprint as v9):
//       Ash f32 (64*32*4 = 8 KiB) + Dsh half (32*64*2 = 4 KiB) = 12 KiB
//   (A stays f32 for parity; D is half because code*scale in {-scale,0,+scale}
//    is EXACT in f16 -> zero added rounding, parity identical to v9.
//    Double-buffering this was measured slower — it cut occupancy; see the
//    module header.)
constant constexpr uint V10_TM = 64u;
constant constexpr uint V10_TN = 64u;
constant constexpr uint V10_TK = 32u;
constant constexpr uint V10_SIMDGROUPS = 4u;
constant constexpr uint V10_THREADS = V10_SIMDGROUPS * 32u;   // 128
constant constexpr uint V10_SG_M = 32u;                       // rows per simdgroup quadrant (M)
constant constexpr uint V10_SG_N = 32u;                       // cols per simdgroup quadrant (N)
constant constexpr uint V10_FRAG = 8u;                        // hardware matrix edge
constant constexpr uint V10_MFRAGS = V10_SG_M / V10_FRAG;     // 4 accumulator rows
constant constexpr uint V10_NFRAGS = V10_SG_N / V10_FRAG;     // 4 accumulator cols
constant constexpr uint V10_KFRAGS = V10_TK / V10_FRAG;       // 4 K-fragments per K-tile
constant constexpr uint V10_AELEMS = V10_TM * V10_TK;         // A floats staged per K-tile (2048)
constant constexpr uint V10_QSLICE = V10_TK / 4u;             // qs bytes for a 32-wide K slice (8)
constant constexpr uint V10_DUNITS = V10_TN * V10_QSLICE;     // (n,qbyte) scatter work-items (512)

// 2-bit ternary decode, identical to v9 v9_decode_tq2 / v8 / v7 / CPU dequant:
//   code 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0 (reserved).
inline float v10_decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

// Stage one V10_TK-wide K-tile into the Ash/Dsh threadgroup buffers.  Mirrors
// v9's per-tile staging but (Lever 1) writes Dsh as half and (Lever 2) spreads
// the dequant-scatter across all 128 threads as (n,qbyte) work-items.
inline void v10_stage_tile(
    device const uchar* soa_raw,
    device const float* inputs,
    threadgroup float*  Ash,        // base: V10_TM * V10_TK floats
    threadgroup half*   Dsh,        // base: V10_TK * V10_TN halfs
    uint kt,
    uint k,
    uint col_base,
    uint row_base,
    uint blocks_per_row,
    uint qs_offset,
    uint valid_rows,
    uint valid_cols,
    uint lid)
{
    const uint k_off = kt * V10_TK;            // global K offset of this tile
    const uint kb    = k_off / 128u;           // ternary block index (V10_TK divides 128)
    const uint kin   = k_off % 128u;           // element offset within the 128-block
    const uint qbyte0 = kin / 4u;              // first qs byte of this 32-wide slice

    // -- Stage Dsh (half, transposed): all 128 threads cooperate over the
    //    V10_DUNITS = V10_TN * V10_QSLICE (n, qbyte) work-items, 4 each.
    //    Each item decodes one qs byte -> 4 codes -> 4 Dsh[kk*TN + n] entries.
    for (uint u = lid; u < V10_DUNITS; u += V10_THREADS) {
        const uint n_local = u / V10_QSLICE;   // 0..V10_TN-1 -> tile-local weight col
        const uint qb      = u % V10_QSLICE;   // 0..V10_QSLICE-1 qs byte within the slice
        const uint kk      = qb * 4u;          // first K element this byte covers
        if (n_local < valid_rows) {
            const uint w_row     = row_base + n_local;
            const uint block_idx = w_row * blocks_per_row + kb;
            const half scale_h   = *(device const half*)(soa_raw + block_idx * 2u);
            const device uchar* qs = soa_raw + qs_offset + block_idx * 32u + qbyte0;
            const uint byte = uint(qs[qb]);
            Dsh[(kk + 0u) * V10_TN + n_local] = scale_h * half(v10_decode_tq2((byte      ) & 3u));
            Dsh[(kk + 1u) * V10_TN + n_local] = scale_h * half(v10_decode_tq2((byte >> 2u) & 3u));
            Dsh[(kk + 2u) * V10_TN + n_local] = scale_h * half(v10_decode_tq2((byte >> 4u) & 3u));
            Dsh[(kk + 3u) * V10_TN + n_local] = scale_h * half(v10_decode_tq2((byte >> 6u) & 3u));
        } else {
            // Padded weight column: zero so out-of-range N contributes 0.
            Dsh[(kk + 0u) * V10_TN + n_local] = half(0.0);
            Dsh[(kk + 1u) * V10_TN + n_local] = half(0.0);
            Dsh[(kk + 2u) * V10_TN + n_local] = half(0.0);
            Dsh[(kk + 3u) * V10_TN + n_local] = half(0.0);
        }
    }

    // -- Stage Ash (f32): V10_TM input columns x V10_TK elems
    //    (read inputs[col*k + k_off + e]).  All 128 threads, 16 elems each.
    for (uint i = lid; i < V10_AELEMS; i += V10_THREADS) {
        const uint a_row = i / V10_TK;     // 0..V10_TM-1 -> tile-local M row
        const uint a_kk  = i % V10_TK;     // 0..V10_TK-1
        float v = 0.0f;
        if (a_row < valid_cols) {
            const uint a_col = col_base + a_row;
            v = inputs[a_col * k + k_off + a_kk];
        }
        Ash[a_row * V10_TK + a_kk] = v;
    }
}

// Accumulate one staged K-tile into the simdgroup accumulators:
// acc[mi][ni] += A_frag (f32) * D_frag (half) over the V10_TK K-tile.
inline void v10_accumulate_tile(
    threadgroup const float* Ash,
    threadgroup const half*  Dsh,
    thread simdgroup_float8x8 acc[V10_MFRAGS][V10_NFRAGS],
    uint sg_m0,
    uint sg_n0)
{
    for (uint kf = 0u; kf < V10_KFRAGS; kf++) {
        simdgroup_float8x8 afrag[V10_MFRAGS];
        simdgroup_half8x8  dfrag[V10_NFRAGS];
        for (uint mi = 0u; mi < V10_MFRAGS; mi++) {
            const uint a_off = (sg_m0 + mi * V10_FRAG) * V10_TK + kf * V10_FRAG;
            simdgroup_load(afrag[mi], Ash + a_off, V10_TK);
        }
        for (uint ni = 0u; ni < V10_NFRAGS; ni++) {
            const uint d_off = (kf * V10_FRAG) * V10_TN + (sg_n0 + ni * V10_FRAG);
            simdgroup_load(dfrag[ni], Dsh + d_off, V10_TN);
        }
        for (uint mi = 0u; mi < V10_MFRAGS; mi++) {
            for (uint ni = 0u; ni < V10_NFRAGS; ni++) {
                simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
            }
        }
    }
}

kernel void gemm_tq2_g128_v10_simdgroup(
    device const uchar*  soa_raw    [[buffer(0)]],
    device const float*  inputs     [[buffer(1)]],
    device       float*  outputs    [[buffer(2)]],
    constant uint&       n_rows     [[buffer(3)]],
    constant uint&       batch_size [[buffer(4)]],
    constant uint&       k          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    // This threadgroup's output tile origin.
    const uint row_base = tgid.x * V10_TN;   // first weight row (N)
    const uint col_base = tgid.y * V10_TM;   // first batch column (M)

    // simdgroup -> 32x32 quadrant of the 64x64 tile.
    const uint sg_mi = sgid / (V10_TN / V10_SG_N);   // 0..1 -> quadrant row
    const uint sg_ni = sgid % (V10_TN / V10_SG_N);   // 0..1 -> quadrant col
    const uint sg_m0 = sg_mi * V10_SG_M;             // tile-local first M row of this simdgroup
    const uint sg_n0 = sg_ni * V10_SG_N;             // tile-local first N col of this simdgroup

    const uint blocks_per_row = k / 128u;            // K/128 ternary blocks per weight row
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;   // qs section starts after f16 scales
    const uint k_tiles        = k / V10_TK;          // number of V10_TK-wide K-tiles

    // Single-buffered threadgroup staging (12 KiB, same footprint as v9; D is
    // half so Dsh is 4 KiB not 8 KiB). Double-buffering was measured slower.
    //   Ash[m][kk] = input  for tile row m, global k = kt*V10_TK + kk. (ld = V10_TK)
    //   Dsh[kk][n] = weight for tile col n, global k = kt*V10_TK + kk.  (ld = V10_TN, half)
    threadgroup float Ash[V10_TM * V10_TK];   // 64*32*4 = 8 KiB
    threadgroup half  Dsh[V10_TK * V10_TN];   // 32*64*2 = 4 KiB

    // Per-simdgroup accumulators: 4x4 grid of 8x8 f32 fragments (the 32x32 quadrant).
    simdgroup_float8x8 acc[V10_MFRAGS][V10_NFRAGS];
    for (uint mi = 0u; mi < V10_MFRAGS; mi++) {
        for (uint ni = 0u; ni < V10_NFRAGS; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Boundary extents for this tile (out-of-range rows/cols staged as zero).
    const uint valid_rows = (row_base < n_rows)     ? min(V10_TN, n_rows - row_base)     : 0u;
    const uint valid_cols = (col_base < batch_size) ? min(V10_TM, batch_size - col_base) : 0u;

    // Per K-tile: stage (f16 D + f32 A) -> barrier -> matrix-MAC -> barrier.
    for (uint kt = 0u; kt < k_tiles; kt++) {
        v10_stage_tile(soa_raw, inputs, Ash, Dsh, kt, k, col_base, row_base,
                       blocks_per_row, qs_offset, valid_rows, valid_cols, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v10_accumulate_tile(Ash, Dsh, acc, sg_m0, sg_n0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Write back the 32x32 quadrant (column-major: outputs[col*n_rows + row]).
    // Stage each 8x8 accumulator to threadgroup memory, then scatter with the
    // boundary clamp (output is column-major, so a per-element scatter is
    // required anyway).  Reuse Ash as the f32 scratch (it is dead here).
    threadgroup float* Csh = Ash;   // V10_TM*V10_TK = 2048 floats >= V10_SG_M*V10_SG_N (1024)

    for (uint sg = 0u; sg < V10_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < V10_MFRAGS; mi++) {
                for (uint ni = 0u; ni < V10_NFRAGS; ni++) {
                    const uint c_off = (mi * V10_FRAG) * V10_SG_N + ni * V10_FRAG;
                    simdgroup_store(acc[mi][ni], Csh + c_off, V10_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint idx = lid % 32u; idx < V10_SG_M * V10_SG_N; idx += 32u) {
                const uint mm = idx / V10_SG_N;          // 0..31 local M within quadrant
                const uint nn = idx % V10_SG_N;          // 0..31 local N within quadrant
                const uint m_local = sg_m0 + mm;         // tile-local M
                const uint n_local = sg_n0 + nn;         // tile-local N
                if (m_local < valid_cols && n_local < valid_rows) {
                    const uint col = col_base + m_local;
                    const uint row = row_base + n_local;
                    outputs[col * n_rows + row] = Csh[mm * V10_SG_N + nn];
                }
            }
        }
    }
}
"#;
