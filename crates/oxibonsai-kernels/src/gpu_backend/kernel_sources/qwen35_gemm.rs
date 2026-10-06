//! Tiled `simdgroup_matrix` GEMMs for the Qwen3.5 / PrismML Bonsai 2 hybrid
//! stack's batched prefill (design §4.2, `qwen35_prefill` in
//! `metal_full_layer`).
//!
//! # Why a GEMM, not the GEMV per token
//!
//! A prefill chunk of `t` tokens used to run every folded projection as the
//! decode GEMV with one grid row per token (`q35_gemv_*`, grid `[rows / 8,
//! t]`): every token re-streams the whole weight matrix, so a 512-token
//! chunk of the 27B moved 512 × 6.89 GB and ran at the decode rate
//! (≈ 125 ms per token for `PQ2_0`). These kernels stage a 64-token × 64-row
//! output tile per threadgroup and walk `K` in 32-wide slices, so each
//! weight is decoded once per 64 tokens and the multiply-adds run on the
//! 8 × 8 hardware matrix units.
//!
//! # Tile geometry
//!
//! The `v10` ternary DiT GEMM's shape (`prefill_simdgroup_v10.rs`): a
//! `64 (M, tokens) × 64 (N, weight rows)` output tile per threadgroup,
//! 4 simdgroups (128 threads) each owning a `32 × 32` quadrant held as a
//! `4 × 4` grid of `8 × 8` `f32` accumulators, and a `K` slice of 32
//! (`Q35G_TK`), which divides every format's 128-element super-block (and a
//! `Q2_0_g64` block's 64), so a slice never straddles two scales.
//!
//! # Operands
//!
//! * **Weights** are decoded from the on-disk (AoS) blocks in place — the
//!   same bytes the GEMVs read, so a memory-mapped model stays zero-copy —
//!   and staged as `half`. That is **exact** for every quantized format:
//!   a weight is `code · d` with `code ∈ {-1, 0, +1, +2}` (`PQ2_0` /
//!   `Q2_0_g64` / `PTQ1_0`), `{-1, 0, +1}` (`TQ2_0_g128`) or `±1`
//!   (`Q1_0_g128`) and `d` an `f16`, so `code · d` is an `f16` again.
//!   Unquantized `f32` weights (`q35_gemm_f32`) are staged as `f32`.
//! * **Activations** are staged as `f32` by the default entry points; the
//!   `_ha` entry points stage them as `half` (the PrismML fork's own Metal
//!   `mul_mm` stages both operands in half), trading activation precision
//!   for threadgroup bandwidth.
//! * Every product accumulates in `f32`.
//!
//! # Entry points
//!
//! `q35_gemm_<fmt>` for `<fmt>` ∈ `pq2`, `ptq1`, `tq2`, `q2g64`, `q1`,
//! `f32`, and the half-activation twins `q35_gemm_<fmt>_ha` of the five
//! quantized formats. The buffers and scalars are exactly the GEMV's —
//! weights (0, at the matrix's byte offset), `x` (1, `[m_cols][k]`), `y` (2,
//! `[m_cols][n_rows]`), `n_rows` (3), `k` (4, a multiple of 128), `m_cols`
//! (5), `accumulate` (6: `y += W x` when non-zero, the residual add of the
//! output projections) — so the encoder swaps the pipeline and the grid and
//! nothing else. Grid `[ceil(n_rows / 64), ceil(m_cols / 64), 1]`, 128
//! threads.
//!
//! # Numerics
//!
//! The decode is the GEMV's, bit for bit (`q35_decode16` / `q35_ptq1_rows`
//! in `kernel_sources::qwen35`); only the summation order differs — the
//! GEMV sums lane-sliced partials then `simd_sum`s them, the GEMM sums
//! `8`-element matrix products along `K` — so a GEMM row agrees with the
//! GEMV to rounding (the tests hold it to a relative `1e-5`), not bitwise.
//! Out-of-range rows and columns of a partial tile are staged as zero and
//! never written.

/// The batched-prefill GEMMs (see the module docs). Self-contained: its
/// helpers and constants carry the `q35g_` / `Q35G_` prefixes, so its place
/// in the combined library does not matter.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_QWEN35_GEMM: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint Q35G_TM = 64u;        // tokens (M) per output tile
constant constexpr uint Q35G_TN = 64u;        // weight rows (N) per output tile
constant constexpr uint Q35G_TK = 32u;        // K elements per staged slice
constant constexpr uint Q35G_THREADS = 128u;  // 4 simdgroups
constant constexpr uint Q35G_SG_M = 32u;      // quadrant rows (tokens)
constant constexpr uint Q35G_SG_N = 32u;      // quadrant columns (weight rows)
constant constexpr uint Q35G_FRAG = 8u;
constant constexpr uint Q35G_MFRAGS = Q35G_SG_M / Q35G_FRAG;  // 4
constant constexpr uint Q35G_NFRAGS = Q35G_SG_N / Q35G_FRAG;  // 4
constant constexpr uint Q35G_KFRAGS = Q35G_TK / Q35G_FRAG;    // 4
constant constexpr uint Q35G_A_VEC4 = Q35G_TM * Q35G_TK / 4u; // 512 float4 per slice

// Weight formats of the staging decoder.
constant constexpr uint Q35G_PQ2 = 0u;    // PQ2_0: 34 B, d first, value (code - 1) * d
constant constexpr uint Q35G_TQ2 = 1u;    // TQ2_0_g128: 34 B, qs first, d last, 0b11 = 0
constant constexpr uint Q35G_Q2G64 = 2u;  // Q2_0_g64: two 18 B blocks, d first, code - 1
constant constexpr uint Q35G_Q1 = 3u;     // Q1_0_g128: 18 B, d first, bit set = +d
constant constexpr uint Q35G_PTQ1 = 4u;   // PTQ1_0: 28 B, qs[24] qh[2] d

// A 32-bit little-endian word at a 2-byte-aligned address (every block
// format here has an even stride but not a multiple of four).
inline uint q35g_u32(device const uchar* p) {
    device const ushort* h = (device const ushort*)p;
    return uint(h[0]) | (uint(h[1]) << 16);
}

// The 2-bit code `c` of a PQ2_0 / Q2_0_g64 weight as its value in units of
// `d`: `c - 1` (0b11 = +2).
inline half q35g_code_m1(uint c) {
    return half(float(c) - 1.0f);
}

// The 2-bit code of a TQ2_0_g128 weight: 0 -> -1, 2 -> +1, 1 and 3 -> 0.
inline half q35g_code_tq2(uint c) {
    return (c == 0u) ? half(-1.0h) : ((c == 2u) ? half(1.0h) : half(0.0h));
}

// Stage the 32 x 64 weight slice starting at K offset `k_off` of rows
// `row0 .. row0 + valid_n` into `Dsh[kk * 64 + n]` as half (exact: see the
// module docs). Thread `lid` owns weight row `lid % 64` and the 16-element
// half `lid / 64` of the slice.
template <uint FMT>
inline void q35g_stage_w(device const uchar* w, threadgroup half* Dsh,
                         uint row0, uint valid_n, uint nsb, uint k_off, uint lid) {
    const uint n = lid % Q35G_TN;
    const uint part = lid / Q35G_TN;          // 0 or 1: elements [16 part, 16 part + 16)
    const uint kk0 = part * 16u;
    if (n >= valid_n) {
        for (uint j = 0u; j < 16u; ++j) {
            Dsh[(kk0 + j) * Q35G_TN + n] = half(0.0h);
        }
        return;
    }
    const uint kb = k_off / 128u;             // super-block of this slice
    const uint kin = k_off % 128u;            // 0, 32, 64 or 96 within it
    const uint row = row0 + n;
    if (FMT == Q35G_PQ2 || FMT == Q35G_TQ2 || FMT == Q35G_Q2G64) {
        device const uchar* blk;
        uint qs_off;
        half d;
        if (FMT == Q35G_PQ2) {
            blk = w + (row * nsb + kb) * 34u;
            d = *(device const half*)blk;
            qs_off = 2u + kin / 4u + part * 4u;
        } else if (FMT == Q35G_TQ2) {
            blk = w + (row * nsb + kb) * 34u;
            d = *(device const half*)(blk + 32u);
            qs_off = kin / 4u + part * 4u;
        } else {
            blk = w + (row * nsb + kb) * 36u + (kin / 64u) * 18u;
            d = *(device const half*)blk;
            qs_off = 2u + (kin % 64u) / 4u + part * 4u;
        }
        const uint q = q35g_u32(blk + qs_off);
        for (uint j = 0u; j < 16u; ++j) {
            const uint c = (q >> (2u * j)) & 3u;
            const half v = (FMT == Q35G_TQ2) ? q35g_code_tq2(c) : q35g_code_m1(c);
            Dsh[(kk0 + j) * Q35G_TN + n] = v * d;
        }
    } else if (FMT == Q35G_Q1) {
        device const uchar* blk = w + (row * nsb + kb) * 18u;
        const half d = *(device const half*)blk;
        const uint bits = uint(*(device const ushort*)(blk + 2u + kin / 8u + part * 2u));
        for (uint j = 0u; j < 16u; ++j) {
            Dsh[(kk0 + j) * Q35G_TN + n] = ((bits >> j) & 1u) ? d : -d;
        }
    } else {
        // PTQ1_0: element e of the super-block is trit `t` of byte `b`:
        //   e < 80:        b = e % 16,               t = e / 16
        //   80 <= e < 120: b = 16 + (e - 80) % 8,    t = (e - 80) / 8
        //   e >= 120:      b = 24 + (e - 120) % 2,   t = (e - 120) / 2   (the qh bytes)
        // and the code of trit t of byte q is F(t + 1) - 3 F(t) with
        // F(n) = floor(q * 3^n / 256), exact in f32 (the GEMV's decode).
        device const uchar* blk = w + (row * nsb + kb) * 28u;
        const half d = *(device const half*)(blk + 26u);
        for (uint j = 0u; j < 16u; ++j) {
            const uint e = kin + kk0 + j;
            uint b;
            uint t;
            if (e < 80u) {
                b = e & 15u;
                t = e >> 4;
            } else if (e < 120u) {
                b = 16u + ((e - 80u) & 7u);
                t = (e - 80u) >> 3;
            } else {
                b = 24u + ((e - 120u) & 1u);
                t = (e - 120u) >> 1;
            }
            const float q = float(blk[b]);
            float p = 1.0f;
            for (uint i = 0u; i < t; ++i) {
                p *= 3.0f;
            }
            const float lo = floor(q * p / 256.0f);
            const float hi = floor(q * (3.0f * p) / 256.0f);
            const float code = fma(lo, -3.0f, hi);
            Dsh[(kk0 + j) * Q35G_TN + n] = half(code - 1.0f) * d;
        }
    }
}

// Stage the 64 x 32 activation slice (tokens `m0 .. m0 + valid_m`, K
// offset `k_off`) into `Ash[m * 32 + kk]`, float4 at a time; rows past
// `valid_m` are zero.
template <typename AT>
inline void q35g_stage_x(device const float* x, threadgroup AT* Ash,
                         uint m0, uint valid_m, uint k, uint k_off, uint lid) {
    for (uint i = lid; i < Q35G_A_VEC4; i += Q35G_THREADS) {
        const uint m = i / (Q35G_TK / 4u);
        const uint k4 = i % (Q35G_TK / 4u);
        float4 v = float4(0.0f);
        if (m < valid_m) {
            v = *(device const float4*)(x + (m0 + m) * k + k_off + k4 * 4u);
        }
        threadgroup AT* dst = Ash + m * Q35G_TK + k4 * 4u;
        dst[0] = AT(v.x);
        dst[1] = AT(v.y);
        dst[2] = AT(v.z);
        dst[3] = AT(v.w);
    }
}

// Write one simdgroup's 32 x 32 quadrant back through threadgroup scratch
// (`Csh`, 1024 floats: the activation slice's storage, dead after the last
// K slice — 8 KiB as f32, exactly 4 KiB as half), one simdgroup at a time, with the tile clamps and
// the optional residual add (`y = y_old + W x`, the sum rounded first, as
// the GEMV does).
inline void q35g_write_back(thread simdgroup_float8x8 acc[Q35G_MFRAGS][Q35G_NFRAGS],
                            threadgroup float* Csh, device float* y,
                            uint m0, uint row0, uint valid_m, uint valid_n,
                            uint n_rows, uint accumulate,
                            uint sg_m0, uint sg_n0, uint sgid, uint lid) {
    for (uint sg = 0u; sg < 4u; ++sg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
                for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
                    simdgroup_store(acc[mi][ni],
                                    Csh + (mi * Q35G_FRAG) * Q35G_SG_N + ni * Q35G_FRAG,
                                    Q35G_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint idx = lid % 32u; idx < Q35G_SG_M * Q35G_SG_N; idx += 32u) {
                const uint m = sg_m0 + idx / Q35G_SG_N;
                const uint n = sg_n0 + idx % Q35G_SG_N;
                if (m < valid_m && n < valid_n) {
                    device float* dst = y + (m0 + m) * n_rows + row0 + n;
                    const float v = Csh[idx];
                    *dst = (accumulate != 0u) ? (*dst + v) : v;
                }
            }
        }
    }
}

// The quantized-weight GEMM body: `AT` is the staged activation type
// (`float` or `half`), the weights are staged as exact `half`.
template <uint FMT, typename AT>
inline void q35g_gemm_quant(device const uchar* w, device const float* x, device float* y,
                            uint n_rows, uint k, uint m_cols, uint accumulate,
                            uint2 tgid, uint lid, uint sgid,
                            threadgroup AT* Ash, threadgroup half* Dsh, threadgroup float* Csh) {
    const uint row0 = tgid.x * Q35G_TN;
    const uint m0 = tgid.y * Q35G_TM;
    if (row0 >= n_rows || m0 >= m_cols) {
        return;
    }
    const uint valid_n = min(Q35G_TN, n_rows - row0);
    const uint valid_m = min(Q35G_TM, m_cols - m0);
    const uint sg_m0 = (sgid / 2u) * Q35G_SG_M;
    const uint sg_n0 = (sgid % 2u) * Q35G_SG_N;
    const uint nsb = k / 128u;
    simdgroup_float8x8 acc[Q35G_MFRAGS][Q35G_NFRAGS];
    for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
        for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k_off = 0u; k_off < k; k_off += Q35G_TK) {
        q35g_stage_w<FMT>(w, Dsh, row0, valid_n, nsb, k_off, lid);
        q35g_stage_x<AT>(x, Ash, m0, valid_m, k, k_off, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < Q35G_KFRAGS; ++kf) {
            simdgroup_matrix<AT, 8, 8> afrag[Q35G_MFRAGS];
            simdgroup_half8x8 dfrag[Q35G_NFRAGS];
            for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
                simdgroup_load(afrag[mi], Ash + (sg_m0 + mi * Q35G_FRAG) * Q35G_TK + kf * Q35G_FRAG,
                               Q35G_TK);
            }
            for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
                simdgroup_load(dfrag[ni], Dsh + (kf * Q35G_FRAG) * Q35G_TN + sg_n0 + ni * Q35G_FRAG,
                               Q35G_TN);
            }
            for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
                for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
                    simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    q35g_write_back(acc, Csh, y, m0, row0, valid_m, valid_n, n_rows, accumulate,
                    sg_m0, sg_n0, sgid, lid);
}

// Unquantized f32 weights: staged as f32 (not exact in half), f32 MACs.
inline void q35g_gemm_f32(device const float* w, device const float* x, device float* y,
                          uint n_rows, uint k, uint m_cols, uint accumulate,
                          uint2 tgid, uint lid, uint sgid,
                          threadgroup float* Ash, threadgroup float* Dsh, threadgroup float* Csh) {
    const uint row0 = tgid.x * Q35G_TN;
    const uint m0 = tgid.y * Q35G_TM;
    if (row0 >= n_rows || m0 >= m_cols) {
        return;
    }
    const uint valid_n = min(Q35G_TN, n_rows - row0);
    const uint valid_m = min(Q35G_TM, m_cols - m0);
    const uint sg_m0 = (sgid / 2u) * Q35G_SG_M;
    const uint sg_n0 = (sgid % 2u) * Q35G_SG_N;
    simdgroup_float8x8 acc[Q35G_MFRAGS][Q35G_NFRAGS];
    for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
        for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    const uint n = lid % Q35G_TN;
    const uint kk0 = (lid / Q35G_TN) * 16u;
    for (uint k_off = 0u; k_off < k; k_off += Q35G_TK) {
        if (n < valid_n) {
            device const float4* src = (device const float4*)(w + (row0 + n) * k + k_off + kk0);
            for (uint j4 = 0u; j4 < 4u; ++j4) {
                const float4 v = src[j4];
                Dsh[(kk0 + 4u * j4 + 0u) * Q35G_TN + n] = v.x;
                Dsh[(kk0 + 4u * j4 + 1u) * Q35G_TN + n] = v.y;
                Dsh[(kk0 + 4u * j4 + 2u) * Q35G_TN + n] = v.z;
                Dsh[(kk0 + 4u * j4 + 3u) * Q35G_TN + n] = v.w;
            }
        } else {
            for (uint j = 0u; j < 16u; ++j) {
                Dsh[(kk0 + j) * Q35G_TN + n] = 0.0f;
            }
        }
        q35g_stage_x<float>(x, Ash, m0, valid_m, k, k_off, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kf = 0u; kf < Q35G_KFRAGS; ++kf) {
            simdgroup_float8x8 afrag[Q35G_MFRAGS];
            simdgroup_float8x8 dfrag[Q35G_NFRAGS];
            for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
                simdgroup_load(afrag[mi], Ash + (sg_m0 + mi * Q35G_FRAG) * Q35G_TK + kf * Q35G_FRAG,
                               Q35G_TK);
            }
            for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
                simdgroup_load(dfrag[ni], Dsh + (kf * Q35G_FRAG) * Q35G_TN + sg_n0 + ni * Q35G_FRAG,
                               Q35G_TN);
            }
            for (uint mi = 0u; mi < Q35G_MFRAGS; ++mi) {
                for (uint ni = 0u; ni < Q35G_NFRAGS; ++ni) {
                    simdgroup_multiply_accumulate(acc[mi][ni], afrag[mi], dfrag[ni], acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    q35g_write_back(acc, Csh, y, m0, row0, valid_m, valid_n, n_rows, accumulate,
                    sg_m0, sg_n0, sgid, lid);
}

kernel void q35_gemm_f32(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup float Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_f32((device const float*)w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid,
                  Ash, Dsh, Ash);
}

kernel void q35_gemm_pq2(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_PQ2, float>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_ptq1(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_PTQ1, float>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_tq2(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_TQ2, float>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_q2g64(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_Q2G64, float>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_q1(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_Q1, float>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_pq2_ha(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_PQ2, half>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_ptq1_ha(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_PTQ1, half>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_tq2_ha(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_TQ2, half>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_q2g64_ha(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_Q2G64, half>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}

kernel void q35_gemm_q1_ha(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Ash[Q35G_TM * Q35G_TK];
    threadgroup half Dsh[Q35G_TK * Q35G_TN];
    q35g_gemm_quant<Q35G_Q1, half>(w, x, y, n_rows, k, m_cols, accumulate, tgid, lid, sgid, Ash, Dsh,
        (threadgroup float*)Ash);
}
"#;
