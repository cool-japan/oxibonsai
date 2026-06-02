//! `simdgroup_matrix`-based fused-Metal TQ2_0_g128 GEMM kernel (`v9`) for the
//! **large-M** DiT path.
//!
//! `gemm_tq2_g128_v8_tiled` ([`super::prefill_tiled`]) is a register-blocked
//! scalar-FMA GEMM: each thread accumulates a column of `RN = 4` outputs with
//! plain `acc += w * a` FMAs. At the big DiT shape
//! (`M = 1536`, `N = 27648`, `K = 3072`) it reaches only ~3-5 % of the M3 GPU
//! peak — the matmul moves ~210 MB but takes ~735 ms (~286 MB/s ≪ bandwidth),
//! so it is **compute/occupancy-bound, not memory-bound**. The lever is Apple's
//! `simdgroup_float8x8` + `simdgroup_multiply_accumulate` hardware 8×8×8 MAC
//! units, which do the FMAs the scalar kernel cannot keep up with.
//!
//! `gemm_tq2_g128_v9_simdgroup` rewrites the op as
//! `C[M,N] = A[M,K] · D[K,N]` where `D[k,n] = dequant(B[n,k])` (= the ternary
//! weight, **transposed** into the K×N orientation the matrix units want):
//!
//! * **Threadgroup output tile** — `V9_TM × V9_TN = 64 × 64`. The threadgroup
//!   has `V9_SIMDGROUPS = 4` simdgroups (128 threads); each simdgroup owns one
//!   `32 × 32` quadrant of the tile, held as a `4 × 4` grid of
//!   `simdgroup_float8x8` accumulators (f32 accumulate → parity with the CPU /
//!   `v8` reference over `K = 3072`).
//! * **K-tiling by `V9_TK = 32`** (= 4 fragments of 8). Per K-tile every
//!   threadgroup stages, into threadgroup memory, an `Ash[64][32]` slab of the
//!   inputs and a `Dsh[32][64]` slab of **dequantized, transposed** weights,
//!   both as `float` (8 KiB + 8 KiB = 16 KiB ≤ the 32 KiB M3 limit, still
//!   leaving room for occupancy). `float` (not `half`) staging is required for
//!   parity: f16-rounding the full-range `A` inputs and accumulating over
//!   `K = 3072` drifts past the unit-test `1e-3` bound (measured ~1.3e-3),
//!   whereas the `simdgroup_float8x8` path holds max-abs err ≲ 7e-4.
//! * **Dequant amortized by the matrix units** — each ternary weight element is
//!   decoded exactly once per K-tile into `Dsh[kk][n]` (a transposed scatter)
//!   and then reused across all 64 `M` columns of the tile by the 8×8 MACs.
//!   This is the whole point: the scalar `v8` re-derives every weight per
//!   output-row FMA; `v9` derives it once and the silicon multiplies it into 64
//!   accumulators.
//!
//! Decode (`v9_decode_tq2`) is byte-for-byte identical to `v8`'s
//! `v8_decode_tq2` / `v7`'s `pf_decode_tq2` (`{0 → -1, 1 → 0, 2 → +1, 3 → 0}`)
//! and to the CPU reference `BlockTQ2_0_g128::dequant`, so `v9` is numerically
//! equivalent (f32 accumulate over f32-staged operands, matching `v8`'s scalar
//! f32 FMAs; the `dit_parity` cosine gate validates parity stays ≥ 0.999).
//!
//! Weight buffer is the **same** SoA layout `v7` / `v8` read (no upload /
//! reformat change): for `nb = N · (K/128)` blocks,
//! `[0 : nb·2)` = f16 scales (block `b` at byte `b·2`),
//! `[nb·2 : nb·2 + nb·32)` = qs (block `b` at byte `nb·2 + b·32`),
//! block `b = r·(K/128) + kb` covering `W[r, kb·128 .. kb·128+128]`.
//!
//! Boundaries: `N` / `M` tile edges are clamped with `min(...)` (out-of-range
//! rows/cols are staged as zero so the matrix MACs still produce correct
//! in-range outputs), so the kernel is correct for arbitrary (non-tile-multiple)
//! shapes used by the unit tests (e.g. `N = 40`, `M = 33`). `K` is required to
//! be a multiple of 128 (the caller validates this); `V9_TK = 32` divides 128,
//! so a K-tile never straddles two ternary blocks.

/// V9 `simdgroup_matrix` batched ternary (TQ2_0_g128) GEMM.
///
/// Computes `out[M,N] = A[M,K] · dequant(B[N,K])ᵀ` with the same column-major
/// buffer conventions as `gemm_tq2_g128_v7` / `gemm_tq2_g128_v8_tiled`:
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
/// Dispatch: `[ceil(N/64), ceil(M/64), 1]` threadgroups,
/// `[V9_SIMDGROUPS·32, 1, 1] = [128, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_TQ2_G128_V9_SIMDGROUP: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tile / simdgroup geometry (must match dispatch_gemm_tq2_v9 in metal_dispatch.rs).
//
//   output tile          : V9_TM x V9_TN = 64 x 64
//   simdgroups/threadgroup: V9_SIMDGROUPS = 4  (-> 128 threads)
//   per-simdgroup quadrant: V9_SG_M x V9_SG_N = 32 x 32
//   accumulator grid      : (V9_SG_M/8) x (V9_SG_N/8) = 4 x 4 simdgroup_float8x8
//   K step                : V9_TK = 32 (= 4 fragments of 8; divides 128)
//   threadgroup memory     : Ash(64*32*4) + Dsh(32*64*4) = 8 + 8 = 16 KiB (float)
//   (float, not half: f16-rounded A inputs drift past the 1e-3 unit bound at K=3072)
constant constexpr uint V9_TM = 64u;
constant constexpr uint V9_TN = 64u;
constant constexpr uint V9_TK = 32u;
constant constexpr uint V9_SIMDGROUPS = 4u;
constant constexpr uint V9_THREADS = V9_SIMDGROUPS * 32u;   // 128
constant constexpr uint V9_SG_M = 32u;                      // rows per simdgroup quadrant (M)
constant constexpr uint V9_SG_N = 32u;                      // cols per simdgroup quadrant (N)
constant constexpr uint V9_FRAG = 8u;                       // hardware matrix edge
constant constexpr uint V9_MFRAGS = V9_SG_M / V9_FRAG;      // 4 accumulator rows
constant constexpr uint V9_NFRAGS = V9_SG_N / V9_FRAG;      // 4 accumulator cols
constant constexpr uint V9_KFRAGS = V9_TK / V9_FRAG;        // 4 K-fragments per K-tile
constant constexpr uint V9_AELEMS = V9_TM * V9_TK;          // A floats staged per K-tile (2048)
constant constexpr uint V9_QSLICE = V9_TK / 4u;             // qs bytes for a 32-wide K slice (8)

// 2-bit ternary decode, identical to v8 v8_decode_tq2 / v7 pf_decode_tq2 /
// CPU dequant:  code 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0 (reserved).
inline float v9_decode_tq2(uint code) {
    return select(select(0.0f, -1.0f, code == 0u), 1.0f, code == 2u);
}

kernel void gemm_tq2_g128_v9_simdgroup(
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
    const uint row_base = tgid.x * V9_TN;   // first weight row (N)
    const uint col_base = tgid.y * V9_TM;   // first batch column (M)

    // simdgroup -> 32x32 quadrant of the 64x64 tile.
    //   sgid in [0, V9_SIMDGROUPS) ; (sg_mi, sg_ni) in [0,2)x[0,2).
    const uint sg_mi = sgid / (V9_TN / V9_SG_N);   // 0..1  -> quadrant row
    const uint sg_ni = sgid % (V9_TN / V9_SG_N);   // 0..1  -> quadrant col
    const uint sg_m0 = sg_mi * V9_SG_M;            // tile-local first M row of this simdgroup
    const uint sg_n0 = sg_ni * V9_SG_N;            // tile-local first N col of this simdgroup

    const uint blocks_per_row = k / 128u;            // K/128 ternary blocks per weight row
    const uint total_blocks   = n_rows * blocks_per_row;
    const uint qs_offset      = total_blocks * 2u;   // qs section starts after f16 scales
    const uint k_tiles        = k / V9_TK;           // number of V9_TK-wide K-tiles

    // Threadgroup staging (float — see header: half loses parity at K=3072):
    //   Ash[m][kk] = input  for tile row m,  global k = kt*V9_TK + kk.   (row-major, ld = V9_TK)
    //   Dsh[kk][n] = weight  for tile col n,  global k = kt*V9_TK + kk.   (row-major, ld = V9_TN)
    threadgroup float Ash[V9_TM * V9_TK];   // 64 * 32 * 4 = 8 KiB
    threadgroup float Dsh[V9_TK * V9_TN];   // 32 * 64 * 4 = 8 KiB

    // Per-simdgroup accumulators: a 4x4 grid of 8x8 f32 fragments (the 32x32 quadrant).
    simdgroup_float8x8 acc[V9_MFRAGS][V9_NFRAGS];
    for (uint mi = 0u; mi < V9_MFRAGS; mi++) {
        for (uint ni = 0u; ni < V9_NFRAGS; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Boundary extents for this tile (out-of-range rows/cols staged as zero).
    const uint valid_rows = (row_base < n_rows)     ? min(V9_TN, n_rows - row_base)     : 0u;
    const uint valid_cols = (col_base < batch_size) ? min(V9_TM, batch_size - col_base) : 0u;

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k_off = kt * V9_TK;            // global K offset of this tile
        const uint kb    = k_off / 128u;          // ternary block index (V9_TK divides 128)
        const uint kin   = k_off % 128u;          // element offset within the 128-block
        const uint qbyte = kin / 4u;              // first qs byte of this 32-wide slice

        // -- Stage Dsh: dequantize V9_TN weight rows x V9_TK elems, transposed.
        // One thread per (weight row n) decodes its 8 qs bytes -> 32 codes,
        // scattering to Dsh[kk][n].  V9_THREADS=128 >= V9_TN=64, so threads
        // [V9_TN, V9_THREADS) idle here (they participate in Ash below).
        if (lid < V9_TN) {
            const uint n_local = lid;             // 0..V9_TN-1 -> tile-local weight col
            if (n_local < valid_rows) {
                const uint w_row     = row_base + n_local;
                const uint block_idx = w_row * blocks_per_row + kb;
                const half scale_h   = *(device const half*)(soa_raw + block_idx * 2u);
                const float scale    = float(scale_h);
                const device uchar* qs = soa_raw + qs_offset + block_idx * 32u + qbyte;
                for (uint qb = 0u; qb < V9_QSLICE; qb++) {
                    const uint byte = uint(qs[qb]);
                    const uint kk = qb * 4u;
                    Dsh[(kk + 0u) * V9_TN + n_local] = scale * v9_decode_tq2((byte      ) & 3u);
                    Dsh[(kk + 1u) * V9_TN + n_local] = scale * v9_decode_tq2((byte >> 2u) & 3u);
                    Dsh[(kk + 2u) * V9_TN + n_local] = scale * v9_decode_tq2((byte >> 4u) & 3u);
                    Dsh[(kk + 3u) * V9_TN + n_local] = scale * v9_decode_tq2((byte >> 6u) & 3u);
                }
            } else {
                // Padded weight column: zero so out-of-range N contributes 0.
                for (uint kk = 0u; kk < V9_TK; kk++) {
                    Dsh[kk * V9_TN + n_local] = 0.0f;
                }
            }
        }

        // -- Stage Ash: V9_TM input columns x V9_TK elems (read inputs[col*k + k_off + e]).
        // All 128 threads cooperate; V9_AELEMS = 2048 = 16 per thread.
        for (uint i = lid; i < V9_AELEMS; i += V9_THREADS) {
            const uint a_row = i / V9_TK;     // 0..V9_TM-1 -> tile-local M row
            const uint a_kk  = i % V9_TK;     // 0..V9_TK-1
            float v = 0.0f;
            if (a_row < valid_cols) {
                const uint a_col = col_base + a_row;
                v = inputs[a_col * k + k_off + a_kk];
            }
            Ash[a_row * V9_TK + a_kk] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Matrix MACs: acc[mi][ni] += A_frag * D_frag over the V9_TK K-tile.
        // A fragment : Ash[(sg_m0+mi*8)+ .. ][kf*8 + ..]  (row-major, ld = V9_TK)
        // D fragment : Dsh[kf*8 + ..][(sg_n0+ni*8)+ ..]   (row-major, ld = V9_TN)
        for (uint kf = 0u; kf < V9_KFRAGS; kf++) {
            simdgroup_float8x8 afrag[V9_MFRAGS];
            simdgroup_float8x8 dfrag[V9_NFRAGS];
            for (uint mi = 0u; mi < V9_MFRAGS; mi++) {
                const uint a_off = (sg_m0 + mi * V9_FRAG) * V9_TK + kf * V9_FRAG;
                simdgroup_load(afrag[mi], Ash + a_off, V9_TK);
            }
            for (uint ni = 0u; ni < V9_NFRAGS; ni++) {
                const uint d_off = (kf * V9_FRAG) * V9_TN + (sg_n0 + ni * V9_FRAG);
                simdgroup_load(dfrag[ni], Dsh + d_off, V9_TN);
            }
            for (uint mi = 0u; mi < V9_MFRAGS; mi++) {
                for (uint ni = 0u; ni < V9_NFRAGS; ni++) {
                    simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Write back the 32x32 quadrant (column-major: outputs[col*n_rows + row]).
    // Stage each 8x8 accumulator to threadgroup memory, then scatter with the
    // boundary clamp (the matrix store wants a contiguous destination; the
    // output is column-major, so a per-element scatter is required anyway).
    threadgroup float Csh[V9_SG_M * V9_SG_N];   // 32 * 32 * 4 = 4 KiB (one simdgroup at a time)

    for (uint sg = 0u; sg < V9_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < V9_MFRAGS; mi++) {
                for (uint ni = 0u; ni < V9_NFRAGS; ni++) {
                    const uint c_off = (mi * V9_FRAG) * V9_SG_N + ni * V9_FRAG;
                    simdgroup_store(acc[mi][ni], Csh + c_off, V9_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            // Scatter Csh[mm][nn] -> outputs[col*n_rows + row] with clamps.
            for (uint idx = lid % 32u; idx < V9_SG_M * V9_SG_N; idx += 32u) {
                const uint mm = idx / V9_SG_N;          // 0..31 local M within quadrant
                const uint nn = idx % V9_SG_N;          // 0..31 local N within quadrant
                const uint m_local = sg_m0 + mm;        // tile-local M
                const uint n_local = sg_n0 + nn;        // tile-local N
                if (m_local < valid_cols && n_local < valid_rows) {
                    const uint col = col_base + m_local;
                    const uint row = row_base + n_local;
                    outputs[col * n_rows + row] = Csh[mm * V9_SG_N + nn];
                }
            }
        }
    }
}
"#;
