//! `simdgroup_matrix`-based **f32-EXACT** GEMM kernel (`gemm_f32_simdgroup`) for
//! the large-M text-encoder (TE, Qwen3-4B) path.
//!
//! This is the f32 sibling of the ternary
//! `gemm_tq2_g128_v9_simdgroup` ([`super::prefill_simdgroup`]). The FLUX.2 text
//! encoder weights are **pure f32** (offline-dequantized from MLX 4-bit), so
//! there is no quantized format to decode — the op is a plain
//! `C[M,N] = A[M,K] · W[N,K]ᵀ` in f32, identical to the CPU
//! `oxibonsai-image::gemm::gemm_abt`. Reassociating the sum on the GPU stays
//! cos ≈ 1.0 vs the CPU reference (the `te_parity` gate), so parity is trivially
//! safe.
//!
//! The kernel is a near-verbatim copy of `v9` — the **only** difference is the
//! weight-staging step. Where `v9` decodes 2-bit ternary codes, applies the
//! per-128-block f16 scale, and scatters into the transposed `Dsh[kk][n]` tile,
//! this kernel **loads the f32 weight `W[n, k]` directly** into `Dsh[kk][n]`.
//! Everything else carries over unchanged:
//!
//! * **Threadgroup output tile** — `F32_TM × F32_TN = 64 × 64`. The threadgroup
//!   has `F32_SIMDGROUPS = 4` simdgroups (128 threads); each simdgroup owns one
//!   `32 × 32` quadrant of the tile, held as a `4 × 4` grid of
//!   `simdgroup_float8x8` accumulators (f32 accumulate → parity with the CPU).
//! * **K-tiling by `F32_TK = 32`** (= 4 fragments of 8). Per K-tile every
//!   threadgroup stages, into threadgroup memory, an `Ash[64][32]` slab of the
//!   inputs and a `Dsh[32][64]` slab of **transposed** weights, both as `float`
//!   (8 KiB + 8 KiB = 16 KiB ≤ the 32 KiB M3 limit). f32 (not half) staging is
//!   required for parity (and is the natural choice here: the weights are
//!   already f32).
//! * **Matrix MACs** — each weight element is loaded once per K-tile into
//!   `Dsh[kk][n]` and reused across all 64 `M` columns of the tile by the 8×8
//!   `simdgroup_multiply_accumulate` units (f32 accumulate).
//!
//! Boundaries: `N` / `M` tile edges are clamped with `min(...)` (out-of-range
//! rows/cols staged as zero), so the kernel is correct for arbitrary
//! (non-tile-multiple) shapes used by the unit tests (e.g. `N = 40`, `M = 33`).
//! Unlike the ternary kernel, `K` is **not** constrained to a multiple of 128;
//! the last K-tile is clamped to `min(F32_TK, k - k_off)` and out-of-range
//! elements are staged as zero, so any `K ≥ 1` is handled. (TE shapes use
//! `K ∈ {2560, 9728}`, both multiples of 32, so the common path takes full
//! 32-wide K-tiles.)

/// f32-exact `simdgroup_matrix` GEMM.
///
/// Computes `out[M,N] = A[M,K] · W[N,K]ᵀ` with the same column-major buffer
/// conventions as the ternary `gemm_tq2_g128_v9` kernel:
/// `inputs[col·k + elem]`, `outputs[col·n_rows + row]` (`col = m`, `row = n`),
/// which coincide with row-major `[M,K]` / `[M,N]`.
///
/// Buffers:
/// - buffer(0) = `weights`    (f32, `N × K` row-major == `weights[n*k + elem]`)
/// - buffer(1) = `inputs`     (f32, `M × K`, column-major == row-major `[M,K]`)
/// - buffer(2) = `outputs`    (f32, `M × N`, column-major == row-major `[M,N]`)
/// - buffer(3) = `n_rows`     (u32, `N`)
/// - buffer(4) = `batch_size` (u32, `M`)
/// - buffer(5) = `k`          (u32, inner dim; any `K ≥ 1`)
///
/// Dispatch: `[ceil(N/64), ceil(M/64), 1]` threadgroups,
/// `[F32_SIMDGROUPS·32, 1, 1] = [128, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_F32_SIMDGROUP: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tile / simdgroup geometry (must match dispatch_gemm_f32 in metal_dispatch.rs).
//
//   output tile          : F32_TM x F32_TN = 64 x 64
//   simdgroups/threadgroup: F32_SIMDGROUPS = 4  (-> 128 threads)
//   per-simdgroup quadrant: F32_SG_M x F32_SG_N = 32 x 32
//   accumulator grid      : (F32_SG_M/8) x (F32_SG_N/8) = 4 x 4 simdgroup_float8x8
//   K step                : F32_TK = 32 (= 4 fragments of 8)
//   threadgroup memory     : Ash(64*32*4) + Dsh(32*64*4) = 8 + 8 = 16 KiB (float)
constant constexpr uint F32_TM = 64u;
constant constexpr uint F32_TN = 64u;
constant constexpr uint F32_TK = 32u;
constant constexpr uint F32_SIMDGROUPS = 4u;
constant constexpr uint F32_THREADS = F32_SIMDGROUPS * 32u;   // 128
constant constexpr uint F32_SG_M = 32u;                       // rows per simdgroup quadrant (M)
constant constexpr uint F32_SG_N = 32u;                       // cols per simdgroup quadrant (N)
constant constexpr uint F32_FRAG = 8u;                        // hardware matrix edge
constant constexpr uint F32_MFRAGS = F32_SG_M / F32_FRAG;     // 4 accumulator rows
constant constexpr uint F32_NFRAGS = F32_SG_N / F32_FRAG;     // 4 accumulator cols
constant constexpr uint F32_KFRAGS = F32_TK / F32_FRAG;       // 4 K-fragments per K-tile
constant constexpr uint F32_AELEMS = F32_TM * F32_TK;         // A floats staged per K-tile (2048)

kernel void gemm_f32_simdgroup(
    device const float*  weights    [[buffer(0)]],
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
    const uint row_base = tgid.x * F32_TN;   // first weight row (N)
    const uint col_base = tgid.y * F32_TM;   // first batch column (M)

    // simdgroup -> 32x32 quadrant of the 64x64 tile.
    //   sgid in [0, F32_SIMDGROUPS) ; (sg_mi, sg_ni) in [0,2)x[0,2).
    const uint sg_mi = sgid / (F32_TN / F32_SG_N);   // 0..1  -> quadrant row
    const uint sg_ni = sgid % (F32_TN / F32_SG_N);   // 0..1  -> quadrant col
    const uint sg_m0 = sg_mi * F32_SG_M;             // tile-local first M row of this simdgroup
    const uint sg_n0 = sg_ni * F32_SG_N;             // tile-local first N col of this simdgroup

    const uint k_tiles = (k + F32_TK - 1u) / F32_TK; // number of F32_TK-wide K-tiles (ceil)

    // Threadgroup staging (float — the weights are already f32):
    //   Ash[m][kk] = input  for tile row m,  global k = kt*F32_TK + kk.   (row-major, ld = F32_TK)
    //   Dsh[kk][n] = weight  for tile col n,  global k = kt*F32_TK + kk.   (row-major, ld = F32_TN)
    threadgroup float Ash[F32_TM * F32_TK];   // 64 * 32 * 4 = 8 KiB
    threadgroup float Dsh[F32_TK * F32_TN];   // 32 * 64 * 4 = 8 KiB

    // Per-simdgroup accumulators: a 4x4 grid of 8x8 f32 fragments (the 32x32 quadrant).
    simdgroup_float8x8 acc[F32_MFRAGS][F32_NFRAGS];
    for (uint mi = 0u; mi < F32_MFRAGS; mi++) {
        for (uint ni = 0u; ni < F32_NFRAGS; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Boundary extents for this tile (out-of-range rows/cols staged as zero).
    const uint valid_rows = (row_base < n_rows)     ? min(F32_TN, n_rows - row_base)     : 0u;
    const uint valid_cols = (col_base < batch_size) ? min(F32_TM, batch_size - col_base) : 0u;

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k_off  = kt * F32_TK;                              // global K offset of this tile
        const uint k_span = (k_off < k) ? min(F32_TK, k - k_off) : 0u; // valid K elems in this tile

        // -- Stage Dsh: load F32_TN weight rows x F32_TK elems, transposed.
        // One thread per (weight row n) copies its F32_TK weight floats,
        // scattering to Dsh[kk][n].  F32_THREADS=128 >= F32_TN=64, so threads
        // [F32_TN, F32_THREADS) idle here (they participate in Ash below).
        if (lid < F32_TN) {
            const uint n_local = lid;             // 0..F32_TN-1 -> tile-local weight col
            if (n_local < valid_rows) {
                const uint w_row = row_base + n_local;
                const device float* wrow = weights + (ulong)w_row * (ulong)k + (ulong)k_off;
                for (uint kk = 0u; kk < F32_TK; kk++) {
                    Dsh[kk * F32_TN + n_local] = (kk < k_span) ? wrow[kk] : 0.0f;
                }
            } else {
                // Padded weight column: zero so out-of-range N contributes 0.
                for (uint kk = 0u; kk < F32_TK; kk++) {
                    Dsh[kk * F32_TN + n_local] = 0.0f;
                }
            }
        }

        // -- Stage Ash: F32_TM input columns x F32_TK elems (read inputs[col*k + k_off + e]).
        // All 128 threads cooperate; F32_AELEMS = 2048 = 16 per thread.
        for (uint i = lid; i < F32_AELEMS; i += F32_THREADS) {
            const uint a_row = i / F32_TK;     // 0..F32_TM-1 -> tile-local M row
            const uint a_kk  = i % F32_TK;     // 0..F32_TK-1
            float v = 0.0f;
            if (a_row < valid_cols && a_kk < k_span) {
                const uint a_col = col_base + a_row;
                v = inputs[(ulong)a_col * (ulong)k + (ulong)k_off + (ulong)a_kk];
            }
            Ash[a_row * F32_TK + a_kk] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Matrix MACs: acc[mi][ni] += A_frag * D_frag over the F32_TK K-tile.
        // A fragment : Ash[(sg_m0+mi*8)+ .. ][kf*8 + ..]  (row-major, ld = F32_TK)
        // D fragment : Dsh[kf*8 + ..][(sg_n0+ni*8)+ ..]   (row-major, ld = F32_TN)
        for (uint kf = 0u; kf < F32_KFRAGS; kf++) {
            simdgroup_float8x8 afrag[F32_MFRAGS];
            simdgroup_float8x8 dfrag[F32_NFRAGS];
            for (uint mi = 0u; mi < F32_MFRAGS; mi++) {
                const uint a_off = (sg_m0 + mi * F32_FRAG) * F32_TK + kf * F32_FRAG;
                simdgroup_load(afrag[mi], Ash + a_off, F32_TK);
            }
            for (uint ni = 0u; ni < F32_NFRAGS; ni++) {
                const uint d_off = (kf * F32_FRAG) * F32_TN + (sg_n0 + ni * F32_FRAG);
                simdgroup_load(dfrag[ni], Dsh + d_off, F32_TN);
            }
            for (uint mi = 0u; mi < F32_MFRAGS; mi++) {
                for (uint ni = 0u; ni < F32_NFRAGS; ni++) {
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
    threadgroup float Csh[F32_SG_M * F32_SG_N];   // 32 * 32 * 4 = 4 KiB (one simdgroup at a time)

    for (uint sg = 0u; sg < F32_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < F32_MFRAGS; mi++) {
                for (uint ni = 0u; ni < F32_NFRAGS; ni++) {
                    const uint c_off = (mi * F32_FRAG) * F32_SG_N + ni * F32_FRAG;
                    simdgroup_store(acc[mi][ni], Csh + c_off, F32_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            // Scatter Csh[mm][nn] -> outputs[col*n_rows + row] with clamps.
            for (uint idx = lid % 32u; idx < F32_SG_M * F32_SG_N; idx += 32u) {
                const uint mm = idx / F32_SG_N;          // 0..31 local M within quadrant
                const uint nn = idx % F32_SG_N;          // 0..31 local N within quadrant
                const uint m_local = sg_m0 + mm;         // tile-local M
                const uint n_local = sg_n0 + nn;         // tile-local N
                if (m_local < valid_cols && n_local < valid_rows) {
                    const uint col = col_base + m_local;
                    const uint row = row_base + n_local;
                    outputs[(ulong)col * (ulong)n_rows + (ulong)row] = Csh[mm * F32_SG_N + nn];
                }
            }
        }
    }
}
"#;

/// **bf16-input / f32-accumulate** variant of [`MSL_GEMM_F32_SIMDGROUP`].
///
/// Identical tiling, buffers, and dispatch geometry as `gemm_f32_simdgroup`
/// (so [`dispatch_gemm_bf16`](crate::gpu_backend::metal_dispatch) reuses the same
/// shape), but stages the input and weight tiles as `bfloat` and runs the matrix
/// MACs with `simdgroup_matrix<bfloat,8,8>` fragments into an `simdgroup_float8x8`
/// accumulator. Apple-GPU `bfloat` `simdgroup_matrix` (M3+/Metal 3.1) throughput
/// is ~2× the `float` path and the staged tiles are half the bytes, so this is a
/// large speedup. bf16 is deliberate over `half`: it has f32's full exponent
/// range (the TE activations/weights overflow `half`'s ±65504, which corrupts the
/// conditioning) and it is the model's **native** precision — the reference TE
/// runs in bf16 — so rounding the operands to bf16 is parity-clean (cos ≈ 1.0),
/// not lossy. Inputs/outputs/weights stay f32 in global memory; only the internal
/// compute precision is bf16. Use `gemm_f32_simdgroup` where bit-exactness is
/// required (e.g. the VAE conv path).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GEMM_BF16_SIMDGROUP: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint H_TM = 64u;
constant constexpr uint H_TN = 64u;
constant constexpr uint H_TK = 32u;
constant constexpr uint H_SIMDGROUPS = 4u;
constant constexpr uint H_THREADS = H_SIMDGROUPS * 32u;   // 128
constant constexpr uint H_SG_M = 32u;
constant constexpr uint H_SG_N = 32u;
constant constexpr uint H_FRAG = 8u;
constant constexpr uint H_MFRAGS = H_SG_M / H_FRAG;       // 4
constant constexpr uint H_NFRAGS = H_SG_N / H_FRAG;       // 4
constant constexpr uint H_KFRAGS = H_TK / H_FRAG;         // 4
constant constexpr uint H_AELEMS = H_TM * H_TK;           // 2048

kernel void gemm_bf16_simdgroup(
    device const float*  weights    [[buffer(0)]],
    device const float*  inputs     [[buffer(1)]],
    device       float*  outputs    [[buffer(2)]],
    constant uint&       n_rows     [[buffer(3)]],
    constant uint&       batch_size [[buffer(4)]],
    constant uint&       k          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint row_base = tgid.x * H_TN;
    const uint col_base = tgid.y * H_TM;

    const uint sg_mi = sgid / (H_TN / H_SG_N);
    const uint sg_ni = sgid % (H_TN / H_SG_N);
    const uint sg_m0 = sg_mi * H_SG_M;
    const uint sg_n0 = sg_ni * H_SG_N;

    const uint k_tiles = (k + H_TK - 1u) / H_TK;

    // bf16 staging: 4 KiB + 4 KiB (half the f32 kernel's threadgroup memory).
    threadgroup bfloat Ash[H_TM * H_TK];
    threadgroup bfloat Dsh[H_TK * H_TN];

    // f32 accumulators (the MACs read bf16 fragments, accumulate in float).
    simdgroup_float8x8 acc[H_MFRAGS][H_NFRAGS];
    for (uint mi = 0u; mi < H_MFRAGS; mi++) {
        for (uint ni = 0u; ni < H_NFRAGS; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint valid_rows = (row_base < n_rows)     ? min(H_TN, n_rows - row_base)     : 0u;
    const uint valid_cols = (col_base < batch_size) ? min(H_TM, batch_size - col_base) : 0u;

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k_off  = kt * H_TK;
        const uint k_span = (k_off < k) ? min(H_TK, k - k_off) : 0u;

        if (lid < H_TN) {
            const uint n_local = lid;
            if (n_local < valid_rows) {
                const uint w_row = row_base + n_local;
                const device float* wrow = weights + (ulong)w_row * (ulong)k + (ulong)k_off;
                for (uint kk = 0u; kk < H_TK; kk++) {
                    Dsh[kk * H_TN + n_local] = (kk < k_span) ? (bfloat)wrow[kk] : (bfloat)0.0f;
                }
            } else {
                for (uint kk = 0u; kk < H_TK; kk++) {
                    Dsh[kk * H_TN + n_local] = (bfloat)0.0f;
                }
            }
        }

        for (uint i = lid; i < H_AELEMS; i += H_THREADS) {
            const uint a_row = i / H_TK;
            const uint a_kk  = i % H_TK;
            bfloat v = (bfloat)0.0f;
            if (a_row < valid_cols && a_kk < k_span) {
                const uint a_col = col_base + a_row;
                v = (bfloat)inputs[(ulong)a_col * (ulong)k + (ulong)k_off + (ulong)a_kk];
            }
            Ash[a_row * H_TK + a_kk] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0u; kf < H_KFRAGS; kf++) {
            simdgroup_matrix<bfloat, 8, 8> afrag[H_MFRAGS];
            simdgroup_matrix<bfloat, 8, 8> dfrag[H_NFRAGS];
            for (uint mi = 0u; mi < H_MFRAGS; mi++) {
                const uint a_off = (sg_m0 + mi * H_FRAG) * H_TK + kf * H_FRAG;
                simdgroup_load(afrag[mi], Ash + a_off, H_TK);
            }
            for (uint ni = 0u; ni < H_NFRAGS; ni++) {
                const uint d_off = (kf * H_FRAG) * H_TN + (sg_n0 + ni * H_FRAG);
                simdgroup_load(dfrag[ni], Dsh + d_off, H_TN);
            }
            for (uint mi = 0u; mi < H_MFRAGS; mi++) {
                for (uint ni = 0u; ni < H_NFRAGS; ni++) {
                    simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Csh[H_SG_M * H_SG_N];

    for (uint sg = 0u; sg < H_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < H_MFRAGS; mi++) {
                for (uint ni = 0u; ni < H_NFRAGS; ni++) {
                    const uint c_off = (mi * H_FRAG) * H_SG_N + ni * H_FRAG;
                    simdgroup_store(acc[mi][ni], Csh + c_off, H_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint idx = lid % 32u; idx < H_SG_M * H_SG_N; idx += 32u) {
                const uint mm = idx / H_SG_N;
                const uint nn = idx % H_SG_N;
                const uint m_local = sg_m0 + mm;
                const uint n_local = sg_n0 + nn;
                if (m_local < valid_cols && n_local < valid_rows) {
                    const uint col = col_base + m_local;
                    const uint row = row_base + n_local;
                    outputs[(ulong)col * (ulong)n_rows + (ulong)row] = Csh[mm * H_SG_N + nn];
                }
            }
        }
    }
}
"#;
