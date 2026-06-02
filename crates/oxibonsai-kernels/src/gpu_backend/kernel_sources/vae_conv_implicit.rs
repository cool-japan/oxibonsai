//! **im2col-free implicit-GEMM** Conv2d kernel (`conv2d_f32_implicit`) for the
//! high-resolution FLUX.2 VAE decoder convs (`k=3, pad=1, stride=1`).
//!
//! The existing GPU conv path ([`super::vae`]'s `im2col_f32` + `gemm_f32_simdgroup`)
//! **materializes the full im2col patch matrix in global memory** (~3.6 GB for a
//! 512×512 `k=3` conv, ~7.2 GB of round-trip traffic) before the GEMM. That
//! materialization is ~half the conv time (measured ≈240 GFLOP/s for the im2col
//! path vs ≈490 GFLOP/s for the bare GEMM). This kernel removes it: the conv
//! patches are **gathered on-the-fly directly into threadgroup memory**, exactly
//! like a standard implicit-GEMM convolution, so im2col never touches DRAM.
//!
//! # The op (bit-for-bit the same as `super::vae::conv` / `encode_conv2d_f32`)
//!
//! `conv2d k=3 pad=1 stride=1`, NCHW, batch 1:
//! ```text
//! out[co, oh, ow] = bias[co]
//!   + Σ_{kh,kw,ci} W[co, kh, kw, ci] · in[ci, oh+kh-pad, ow+kw-pad]   (0 if OOB)
//! ```
//! with the weight in the exported **MLX layout `[C_out, kH, kW, C_in]`**
//! (flattened row-major == `[C_out, K²·C_in]`, the contraction index ordered
//! `(kh, kw, ci)` — matching `oxibonsai-image::vae::conv::build_im2col`). The
//! per-channel `bias` is **not** added here; the host adds it on download (as the
//! im2col path does), keeping this kernel a pure GEMM.
//!
//! # Implicit-GEMM mapping (adapted verbatim from `gemm_f32_simdgroup`)
//!
//! This is the GEMM `out[C_out, P] = W[C_out, K²·C_in] · Patches[K²·C_in, P]`
//! where `P = H_out·W_out` output pixels. The tile geometry, A/W staging,
//! `simdgroup_float8x8` matrix MACs (f32 accumulate), and boundary clamps are
//! the **identical** structure as `gemm_f32_simdgroup`; the **only** change is
//! that the patch operand's tile is *gathered* with conv indexing instead of read
//! from a contiguous row.
//!
//! Mapping chosen for a coalesced output store and a coalesced gather:
//! - **M = `C_out`** (tile rows, `tgid.y`). The M-operand `Ash[m][kk]` is the
//!   **conv weight read directly** — `W[row of C_out][k_off+kk]`, a contiguous
//!   row (no relayout), exactly the A-staging of `gemm_f32_simdgroup`.
//! - **N = `P = H_out·W_out`** (tile cols, `tgid.x`). The N-operand
//!   `Dsh[kk][n]` is a **gathered conv patch**: contraction index `j = k_off+kk`
//!   decodes to `(kh, kw, ci)`; output pixel `p = col_base+n` decodes to
//!   `(oh, ow)`; the value is `in[ci, oh+kh-pad, ow+kw-pad]` (0 if OOB).
//! - **Output store** is row-major `[C_out, P]` → `out[(row_base+m)*P + p]`,
//!   which **is NCHW `[C_out, H_out, W_out]`** (consecutive `p` → consecutive
//!   addresses, fully coalesced). No transpose, no per-element column-major
//!   scatter (the host then just adds bias[co] per output channel).
//!
//! `K = K²·C_in` is arbitrary (`k_span` clamps the last K-tile), and the M/N tile
//! edges are clamped with `min(...)` (out-of-range staged as zero), so the kernel
//! is correct for **any** VAE shape (spatial 64..512, C 96..384) and the
//! non-tile-multiple unit-test shapes (`H=W=40`, odd `C`). Everything is f32
//! end-to-end (parity with the CPU reference; reassociated sums only).
//!
//! The kernel asserts nothing about `k`/`pad` itself — the indexing is fully
//! general (`k`, `pad` are kernel arguments) — but the host
//! ([`crate::gpu_backend::metal_graph`]'s `encode_conv2d_f32`) only routes the
//! `k=3, pad=1, stride=1` convs here and keeps the tiled-im2col path as the
//! fallback for everything else.

/// f32-exact **implicit-GEMM** (im2col-free) `simdgroup_matrix` Conv2d.
///
/// Computes `out[C_out, P] = W[C_out, K²·C_in] · Patches[K²·C_in, P]` with the
/// conv patches gathered on-the-fly into threadgroup memory (never materialized
/// to global memory). `P = H_out·W_out`, `H_out = H + 2·pad − k + 1`.
///
/// Buffers:
/// - buffer(0) = `weight`  (f32, MLX `[C_out, K²·C_in]` row-major == `weight[co*kk_cin + j]`)
/// - buffer(1) = `input`   (f32, NCHW `[C_in, H·W]` == `input[ci*H*W + ih*W + iw]`)
/// - buffer(2) = `output`  (f32, NCHW `[C_out, P]` == `output[co*P + p]`, written, NO bias)
/// - buffer(3) = `c_out`   (u32, output channels = M = N-of-the-GEMM rows)
/// - buffer(4) = `p`       (u32, output pixels `H_out·W_out` = GEMM columns)
/// - buffer(5) = `kk_cin`  (u32, contraction dim `k·k·C_in`)
/// - buffer(6) = `c_in`    (u32)
/// - buffer(7) = `h`       (u32, input height)
/// - buffer(8) = `w`       (u32, input width)
/// - buffer(9) = `k`       (u32, kernel edge; square)
/// - buffer(10) = `pad`    (u32)
/// - buffer(11) = `w_out`  (u32, output width)
///
/// Dispatch: `[ceil(P/64), ceil(C_out/64), 1]` threadgroups,
/// `[CONV_SIMDGROUPS·32, 1, 1] = [128, 1, 1]` threads (must match
/// `dispatch_conv2d_f32_implicit`).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_CONV2D_F32_IMPLICIT: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tile / simdgroup geometry (must match dispatch_conv2d_f32_implicit).
// Identical to MSL_GEMM_F32_SIMDGROUP so the matrix-MAC structure carries over.
//
//   output tile          : CONV_TM x CONV_TN = 64 x 64   (M=C_out, N=P pixels)
//   simdgroups/threadgroup: CONV_SIMDGROUPS = 4  (-> 128 threads)
//   per-simdgroup quadrant: CONV_SG_M x CONV_SG_N = 32 x 32
//   accumulator grid      : (CONV_SG_M/8) x (CONV_SG_N/8) = 4 x 4 simdgroup_float8x8
//   K step                : CONV_TK = 32 (= 4 fragments of 8)
//   threadgroup memory     : Ash(64*32*4) + Dsh(32*64*4) = 8 + 8 = 16 KiB (float)
constant constexpr uint CONV_TM = 64u;
constant constexpr uint CONV_TN = 64u;
constant constexpr uint CONV_TK = 32u;
constant constexpr uint CONV_SIMDGROUPS = 4u;
constant constexpr uint CONV_THREADS = CONV_SIMDGROUPS * 32u;   // 128
constant constexpr uint CONV_SG_M = 32u;                        // rows per simdgroup quadrant (M=C_out)
constant constexpr uint CONV_SG_N = 32u;                        // cols per simdgroup quadrant (N=P)
constant constexpr uint CONV_FRAG = 8u;                         // hardware matrix edge
constant constexpr uint CONV_MFRAGS = CONV_SG_M / CONV_FRAG;    // 4 accumulator rows
constant constexpr uint CONV_NFRAGS = CONV_SG_N / CONV_FRAG;    // 4 accumulator cols
constant constexpr uint CONV_KFRAGS = CONV_TK / CONV_FRAG;      // 4 K-fragments per K-tile
constant constexpr uint CONV_AELEMS = CONV_TM * CONV_TK;        // weight floats staged per K-tile (2048)
constant constexpr uint CONV_DELEMS = CONV_TK * CONV_TN;        // patch floats staged per K-tile (2048)

kernel void conv2d_f32_implicit(
    device const float*  weight  [[buffer(0)]],
    device const float*  input   [[buffer(1)]],
    device       float*  output  [[buffer(2)]],
    constant uint&       c_out    [[buffer(3)]],
    constant uint&       p        [[buffer(4)]],
    constant uint&       kk_cin   [[buffer(5)]],
    constant uint&       c_in     [[buffer(6)]],
    constant uint&       h        [[buffer(7)]],
    constant uint&       w        [[buffer(8)]],
    constant uint&       k        [[buffer(9)]],
    constant uint&       pad      [[buffer(10)]],
    constant uint&       w_out    [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    // This threadgroup's output tile origin.
    const uint col_base = tgid.x * CONV_TN;   // first output pixel  (N = P)
    const uint row_base = tgid.y * CONV_TM;   // first output channel (M = C_out)

    // simdgroup -> 32x32 quadrant of the 64x64 tile.
    const uint sg_mi = sgid / (CONV_TN / CONV_SG_N);   // 0..1 -> quadrant row (M)
    const uint sg_ni = sgid % (CONV_TN / CONV_SG_N);   // 0..1 -> quadrant col (N)
    const uint sg_m0 = sg_mi * CONV_SG_M;              // tile-local first M row of this simdgroup
    const uint sg_n0 = sg_ni * CONV_SG_N;              // tile-local first N col of this simdgroup

    const uint k_tiles = (kk_cin + CONV_TK - 1u) / CONV_TK;   // number of CONV_TK-wide K-tiles (ceil)

    // Threadgroup staging (float — both operands are f32):
    //   Ash[m][kk] = weight[row_base+m][k_off+kk]   (contiguous weight row, ld = CONV_TK)
    //   Dsh[kk][n] = gathered patch for pixel col_base+n, contraction k_off+kk (ld = CONV_TN)
    threadgroup float Ash[CONV_TM * CONV_TK];   // 64 * 32 * 4 = 8 KiB
    threadgroup float Dsh[CONV_TK * CONV_TN];   // 32 * 64 * 4 = 8 KiB

    // Per-simdgroup accumulators: a 4x4 grid of 8x8 f32 fragments (the 32x32 quadrant).
    simdgroup_float8x8 acc[CONV_MFRAGS][CONV_NFRAGS];
    for (uint mi = 0u; mi < CONV_MFRAGS; mi++) {
        for (uint ni = 0u; ni < CONV_NFRAGS; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Boundary extents for this tile (out-of-range rows/cols staged as zero).
    const uint valid_m = (row_base < c_out) ? min(CONV_TM, c_out - row_base) : 0u;   // valid C_out
    const uint valid_n = (col_base < p)     ? min(CONV_TN, p - col_base)     : 0u;   // valid pixels

    const uint kc = k * c_in;            // for decoding j -> (kh, kw, ci)
    const ulong hw_plane = (ulong)h * (ulong)w;

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k_off  = kt * CONV_TK;                                  // global contraction offset
        const uint k_span = (k_off < kk_cin) ? min(CONV_TK, kk_cin - k_off) : 0u;

        // -- Stage Ash: CONV_TM weight rows x CONV_TK contraction elems.
        // All 128 threads cooperate; CONV_AELEMS = 2048 = 16 per thread.
        // Ash[a_row][a_kk] = weight[(row_base+a_row) * kk_cin + (k_off+a_kk)].
        for (uint i = lid; i < CONV_AELEMS; i += CONV_THREADS) {
            const uint a_row = i / CONV_TK;    // 0..CONV_TM-1 -> tile-local M (C_out) row
            const uint a_kk  = i % CONV_TK;    // 0..CONV_TK-1 -> contraction within tile
            float v = 0.0f;
            if (a_row < valid_m && a_kk < k_span) {
                const uint w_row = row_base + a_row;
                v = weight[(ulong)w_row * (ulong)kk_cin + (ulong)k_off + (ulong)a_kk];
            }
            Ash[a_row * CONV_TK + a_kk] = v;
        }

        // -- Stage Dsh: GATHER conv patches (im2col-free).
        // Dsh[d_kk][d_n], ld = CONV_TN. d_n -> pixel col_base+d_n -> (oh, ow);
        // contraction k_off+d_kk -> (kh, kw, ci); value = in[ci, oh+kh-pad, ow+kw-pad].
        // All 128 threads cooperate; CONV_DELEMS = 2048 = 16 per thread.
        for (uint i = lid; i < CONV_DELEMS; i += CONV_THREADS) {
            const uint d_kk = i / CONV_TN;     // 0..CONV_TK-1 -> contraction within tile
            const uint d_n  = i % CONV_TN;     // 0..CONV_TN-1 -> tile-local N (pixel) col
            float v = 0.0f;
            if (d_n < valid_n && d_kk < k_span) {
                const uint j = k_off + d_kk;          // global contraction index
                // Decode j = (kh*k + kw)*c_in + ci  ->  (kh, kw, ci).
                const uint kh  = j / kc;
                const uint rem = j % kc;
                const uint kw  = rem / c_in;
                const uint ci  = rem % c_in;
                // Decode output pixel p_idx = oh*w_out + ow.
                const uint p_idx = col_base + d_n;
                const uint oh = p_idx / w_out;
                const uint ow = p_idx % w_out;
                // Padded source coordinates (same geometry as build_im2col).
                const uint ih_p = oh + kh;
                const uint iw_p = ow + kw;
                if (ih_p >= pad && ih_p < h + pad && iw_p >= pad && iw_p < w + pad) {
                    const uint ih = ih_p - pad;
                    const uint iw = iw_p - pad;
                    v = input[(ulong)ci * hw_plane + (ulong)ih * (ulong)w + (ulong)iw];
                }
            }
            Dsh[d_kk * CONV_TN + d_n] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // -- Matrix MACs: acc[mi][ni] += A_frag * D_frag over the CONV_TK K-tile.
        // A fragment : Ash[(sg_m0+mi*8)+ .. ][kf*8 + ..]  (row-major, ld = CONV_TK)
        // D fragment : Dsh[kf*8 + ..][(sg_n0+ni*8)+ ..]   (row-major, ld = CONV_TN)
        for (uint kf = 0u; kf < CONV_KFRAGS; kf++) {
            simdgroup_float8x8 afrag[CONV_MFRAGS];
            simdgroup_float8x8 dfrag[CONV_NFRAGS];
            for (uint mi = 0u; mi < CONV_MFRAGS; mi++) {
                const uint a_off = (sg_m0 + mi * CONV_FRAG) * CONV_TK + kf * CONV_FRAG;
                simdgroup_load(afrag[mi], Ash + a_off, CONV_TK);
            }
            for (uint ni = 0u; ni < CONV_NFRAGS; ni++) {
                const uint d_off = (kf * CONV_FRAG) * CONV_TN + (sg_n0 + ni * CONV_FRAG);
                simdgroup_load(dfrag[ni], Dsh + d_off, CONV_TN);
            }
            for (uint mi = 0u; mi < CONV_MFRAGS; mi++) {
                for (uint ni = 0u; ni < CONV_NFRAGS; ni++) {
                    simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -- Write back the 32x32 quadrant. Output is row-major [C_out, P]:
    // output[(row_base+m_local) * p + (col_base+n_local)]  ==  NCHW [C_out, H_out, W_out].
    // Stage each 8x8 accumulator to threadgroup memory, then store row-major with
    // the boundary clamp (the matrix store wants a contiguous destination).
    threadgroup float Csh[CONV_SG_M * CONV_SG_N];   // 32 * 32 * 4 = 4 KiB (one simdgroup at a time)

    for (uint sg = 0u; sg < CONV_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < CONV_MFRAGS; mi++) {
                for (uint ni = 0u; ni < CONV_NFRAGS; ni++) {
                    const uint c_off = (mi * CONV_FRAG) * CONV_SG_N + ni * CONV_FRAG;
                    simdgroup_store(acc[mi][ni], Csh + c_off, CONV_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            // Store Csh[mm][nn] -> output[(row_base+mm) * p + (col_base+nn)] with clamps.
            // Iterate so consecutive lanes write consecutive nn (coalesced over pixels).
            for (uint idx = lid % 32u; idx < CONV_SG_M * CONV_SG_N; idx += 32u) {
                const uint mm = idx / CONV_SG_N;          // 0..31 local M (C_out) within quadrant
                const uint nn = idx % CONV_SG_N;          // 0..31 local N (pixel) within quadrant
                const uint m_local = sg_m0 + mm;          // tile-local M
                const uint n_local = sg_n0 + nn;          // tile-local N
                if (m_local < valid_m && n_local < valid_n) {
                    const uint co = row_base + m_local;   // output channel
                    const uint px = col_base + n_local;   // output pixel
                    output[(ulong)co * (ulong)p + (ulong)px] = Csh[mm * CONV_SG_N + nn];
                }
            }
        }
    }
}
"#;
