//! Metal kernels of the Qwen3-VL vision tower (the Bonsai 2 `mmproj`,
//! bonsai2-design.md §6), driven by `metal_vision`: one command buffer per
//! image, every activation resident.
//!
//! | kernel | CPU reference (`oxibonsai-model::vision::tower`) |
//! |---|---|
//! | `vit_layer_norm` | `layer_norm_rows` (mean-centred, with bias) |
//! | `vit_qkv_rope_pack` | `apply_rope_qk` + the head-major pack of `attention` |
//! | `vit_gemm_f16w` / `vit_gemm_f32w` / `vit_gemm_q80` | `KernelDispatcher::gemm_f32` + bias / `gelu_tanh` / the residual add |
//! | `vit_add_rows` | the position-embedding add of `PatchEmbed::embed_planar` |
//!
//! The attention itself is the DiT's flash kernel
//! (`joint_attention_flash_f32`, non-causal, head-major `q/k/v`, token-major
//! output), which serves the tower's `head_dim` of 72.
//!
//! # GEMM
//!
//! `C[M, N] = A[M, K] · W[N, K]ᵀ` with the q35 GEMMs' tile (64 × 64 output,
//! 4 simdgroups, 32-wide `K` slices, `f32` accumulate), activations staged
//! as `f32` and kept at `f32` precision by the `float × half` simdgroup
//! products. Weights by storage:
//!
//! * `vit_gemm_q80` — the file's `Q8_0` blocks, **exact**: every 32-wide
//!   slice is one block per weight row, so the slice's product is taken
//!   with the block's int8 values (exact in `half`) into a per-slice `f32`
//!   partial, which is then scaled by the block's `f16` scale through an
//!   8 × 8 diagonal fragment — `Σ_b d_b · Σ_{k∈b} a_k q_k`, the value the
//!   dequantised weights `d · q` give, with no weight rounded to `half`.
//!   `K` must be a multiple of 32 (a `Q8_0` row's own constraint).
//! * `vit_gemm_f16w` — `half` weights (the file's `F16` matrices, exact).
//! * `vit_gemm_f32w` — `f32` weights (the patch kernel, and any matrix
//!   stored otherwise).
//!
//! The latter two add each slice's fresh `f32` partial into the accumulator
//! through the identity, as the `Q8_0` path adds its scaled partial: a
//! long `K` then keeps the CPU GEMM's accuracy, where one running chain of
//! products into the accumulator loses about an order of magnitude.
//! Partials are taken one 8-column fragment at a time, which keeps the
//! simdgroup's registers to the accumulators plus one fragment column (a
//! whole column block of partials beside them spills, at a third of the
//! rate). `K` need not be a multiple of 32 for the latter two (the tower's
//! FFN width is 4304): a partial last slice is staged with zeros. `K` must
//! be a multiple of 4 (a float4 is either wholly inside `K` or wholly past
//! it).
//!
//! The epilogue applies, in the CPU's order, `+ bias[n]` (flag bit 0), the
//! tanh GELU (bit 1) and the residual add `out = out_old + v` (bit 2).
//!
//! # GELU and fast math
//!
//! The library is compiled with fast math on, under which `tanh` may take a
//! fast approximation; the GELU therefore calls `precise::tanh` (as the
//! PrismML fork's own Metal kernel does) and evaluates the CPU's expression
//! `0.5 x (1 + tanh(√(2/π) x (1 + 0.044715 x²)))` in the CPU's operation
//! order with contraction off.
//!
//! # RoPE
//!
//! The tower's 2-D rotation pairs channel `j` with `j + head_dim / 2`
//! across the whole head (`GGML_ROPE_TYPE_VISION`); a pair is rotated as
//! the CPU's NEON path does it — each product rounded on its own
//! (contraction off), then subtracted / added — so it is bitwise the CPU's.

/// The vision-tower kernels (see the module docs). Self-contained: helpers
/// and constants carry the `vit_` / `VIT_` prefixes.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_VISION: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint VIT_TM = 64u;
constant constexpr uint VIT_TN = 64u;
constant constexpr uint VIT_TK = 32u;
constant constexpr uint VIT_THREADS = 128u;
constant constexpr uint VIT_SG_M = 32u;
constant constexpr uint VIT_SG_N = 32u;
constant constexpr uint VIT_FRAG = 8u;
constant constexpr uint VIT_MFRAGS = VIT_SG_M / VIT_FRAG;
constant constexpr uint VIT_NFRAGS = VIT_SG_N / VIT_FRAG;
constant constexpr uint VIT_KFRAGS = VIT_TK / VIT_FRAG;
constant constexpr uint VIT_A_VEC4 = VIT_TM * VIT_TK / 4u;
// Column fragments the exact Q8_0 GEMM takes partials of per pass (see its
// body): one keeps the registers to the accumulators and a fragment column.
constant constexpr uint VIT_Q80_NSTEP = 1u;

// Epilogue flags of the GEMMs.
constant constexpr uint VIT_BIAS = 1u;
constant constexpr uint VIT_GELU = 2u;
constant constexpr uint VIT_ACCUMULATE = 4u;

// Weight storage of the generic GEMM staging (`Q8_0` has its own).
constant constexpr uint VIT_W_F16 = 0u;
constant constexpr uint VIT_W_F32 = 1u;

#pragma METAL fp contract(off)
// The tanh GELU of the reference, in the CPU's operation order.
inline float vit_gelu(float x) {
    const float inner = (0.7978846f * x) * (1.0f + (0.044715f * x) * x);
    return (0.5f * x) * (1.0f + precise::tanh(inner));
}

// One rotation pair, each product rounded on its own.
inline float2 vit_rotate_pair(float x0, float x1, float c, float s) {
    return float2(x0 * c - x1 * s, x1 * c + x0 * s);
}
#pragma METAL fp contract(fast)

// LayerNorm of one `dim`-wide row per threadgroup: the mean, then the
// (biased) variance of the centred values, then `(x - mean) * inv * w + b`.
kernel void vit_layer_norm(
    device const float* x        [[buffer(0)]],
    device const float* w        [[buffer(1)]],
    device const float* b        [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& dim           [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tsz  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float partial[32];
    device const float* xr = x + row * dim;
    const uint nsg = (tsz + 31u) / 32u;

    float s = 0.0f;
    for (uint i = tid; i < dim; i += tsz) {
        s += xr[i];
    }
    s = simd_sum(s);
    if (lane == 0u) {
        partial[sgid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0u; g < nsg; ++g) {
        total += partial[g];
    }
    const float mean = total / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float v = 0.0f;
    for (uint i = tid; i < dim; i += tsz) {
        const float c = xr[i] - mean;
        v = fma(c, c, v);
    }
    v = simd_sum(v);
    if (lane == 0u) {
        partial[sgid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint g = 0u; g < nsg; ++g) {
        var += partial[g];
    }
    const float inv = 1.0f / sqrt(var / float(dim) + eps);
    device float* orow = out + row * dim;
    for (uint i = tid; i < dim; i += tsz) {
        orow[i] = ((xr[i] - mean) * inv) * w[i] + b[i];
    }
}

// Rotate the Q and K parts of every `[Q | K | V]` row by the token's angle
// row and write all three head-major (`[heads][n][head_dim]`). Thread
// `(j, h, t)` owns pair `j` (channels `j`, `j + head_dim / 2`) of head `h`
// of token `t`.
kernel void vit_qkv_rope_pack(
    device const float* qkv      [[buffer(0)]],
    device const float* cos_t    [[buffer(1)]],
    device const float* sin_t    [[buffer(2)]],
    device float* q              [[buffer(3)]],
    device float* k              [[buffer(4)]],
    device float* v              [[buffer(5)]],
    constant uint& n             [[buffer(6)]],
    constant uint& heads         [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint half_d = head_dim / 2u;
    const uint j = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (j >= half_d || h >= heads || t >= n) {
        return;
    }
    const uint hidden = heads * head_dim;
    device const float* row = qkv + t * 3u * hidden + h * head_dim;
    const float c = cos_t[t * half_d + j];
    const float s = sin_t[t * half_d + j];
    const uint dst = (h * n + t) * head_dim;
    const float2 rq = vit_rotate_pair(row[j], row[j + half_d], c, s);
    q[dst + j] = rq.x;
    q[dst + j + half_d] = rq.y;
    const float2 rk = vit_rotate_pair(row[hidden + j], row[hidden + j + half_d], c, s);
    k[dst + j] = rk.x;
    k[dst + j + half_d] = rk.y;
    v[dst + j] = row[2u * hidden + j];
    v[dst + j + half_d] = row[2u * hidden + j + half_d];
}

// `out[i] += add[i]` for `i < count`.
kernel void vit_add_rows(
    device float* out            [[buffer(0)]],
    device const float* add      [[buffer(1)]],
    constant uint& count         [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < count) {
        out[i] = out[i] + add[i];
    }
}

// Stage the activation slice (rows `m0 .. m0 + valid_m`, K `k_off ..
// k_off + 32`) into `Ash[m * 32 + kk]`, zero past `valid_m` or `k`.
inline void vit_stage_a(device const float* a, threadgroup float* Ash,
                        uint m0, uint valid_m, uint k, uint k_off, uint lid) {
    for (uint i = lid; i < VIT_A_VEC4; i += VIT_THREADS) {
        const uint m = i / (VIT_TK / 4u);
        const uint k4 = i % (VIT_TK / 4u);
        const uint kk = k_off + k4 * 4u;
        float4 val = float4(0.0f);
        if (m < valid_m && kk < k) {
            val = *(device const float4*)(a + (m0 + m) * k + kk);
        }
        *(threadgroup float4*)(Ash + m * VIT_TK + k4 * 4u) = val;
    }
}

// Stage the weight slice `W[row0 + n][k_off + kk]` transposed into
// `Dsh[kk * 64 + n]`, zero past `valid_n` or `k`. Thread `lid` owns row
// `lid % 64` and the 16-element half `lid / 64`, read four elements at a
// time (`k` is a multiple of 4, so a group is wholly inside `k` or past it).
template <uint WF, typename DT>
inline void vit_stage_w(device const uchar* w, threadgroup DT* Dsh,
                        uint row0, uint valid_n, uint k, uint k_off, uint lid) {
    const uint n = lid % VIT_TN;
    const uint kk0 = (lid / VIT_TN) * 16u;
    const uint row = row0 + n;
    for (uint j4 = 0u; j4 < 4u; ++j4) {
        const uint kk = k_off + kk0 + 4u * j4;
        float4 val = float4(0.0f);
        if (n < valid_n && kk < k) {
            if (WF == VIT_W_F16) {
                val = float4(*(device const half4*)((device const half*)w + row * k + kk));
            } else {
                val = *(device const float4*)((device const float*)w + row * k + kk);
            }
        }
        const uint base = (kk0 + 4u * j4) * VIT_TN + n;
        Dsh[base] = DT(val.x);
        Dsh[base + VIT_TN] = DT(val.y);
        Dsh[base + 2u * VIT_TN] = DT(val.z);
        Dsh[base + 3u * VIT_TN] = DT(val.w);
    }
}

// Stage one 32-wide `K` slice of `Q8_0` weight rows exactly. The slice is
// one 34-byte block per row (`d` as f16, then 32 int8, `k / 32` blocks a
// row): its int8 values, exact in `half`, go transposed into
// `Dsh[kk * 64 + n]` as the other formats' values do, and its scale onto
// the diagonal of row `n`'s 8 × 8 scale fragment,
// `Gsh[(n / 8) * 64 + (n % 8) * 9]` (the off-diagonal entries stay zero).
// Rows past `valid_n` stage zeros. Thread `lid` owns row `lid % 64` and the
// 16 values `lid / 64`.
inline void vit_stage_q80(device const uchar* w, threadgroup half* Dsh, threadgroup half* Gsh,
                          uint row0, uint valid_n, uint k, uint k_off, uint lid) {
    const uint n = lid % VIT_TN;
    const uint kk0 = (lid / VIT_TN) * 16u;
    const bool live = n < valid_n;
    device const uchar* blk = w + ((row0 + n) * (k / 32u) + k_off / 32u) * 34u;
    for (uint j4 = 0u; j4 < 4u; ++j4) {
        half4 val = half4(0.0h);
        if (live) {
            val = half4(char4(*(device const packed_char4*)(blk + 2u + kk0 + 4u * j4)));
        }
        const uint base = (kk0 + 4u * j4) * VIT_TN + n;
        Dsh[base] = val.x;
        Dsh[base + VIT_TN] = val.y;
        Dsh[base + 2u * VIT_TN] = val.z;
        Dsh[base + 3u * VIT_TN] = val.w;
    }
    if (kk0 == 0u) {
        Gsh[(n / 8u) * 64u + (n % 8u) * 9u] = live ? *(device const half*)blk : half(0.0h);
    }
}

// Write a tile's accumulators back through the (now dead) activation slice
// `Ash`, one simdgroup at a time, with the epilogue: `+ bias[n]`, the tanh
// GELU, the residual add — in the CPU's order.
inline void vit_gemm_store(thread simdgroup_float8x8 (&acc)[VIT_MFRAGS][VIT_NFRAGS],
                           device float* c, device const float* bias, uint n_rows, uint flags,
                           uint row0, uint m0, uint valid_n, uint valid_m, uint sg_m0,
                           uint sg_n0, uint lid, uint sgid, threadgroup float* Ash) {
    threadgroup float* Csh = Ash;
    for (uint sg = 0u; sg < 4u; ++sg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                for (uint ni = 0u; ni < VIT_NFRAGS; ++ni) {
                    simdgroup_store(acc[mi][ni], Csh + (mi * VIT_FRAG) * VIT_SG_N + ni * VIT_FRAG,
                                    VIT_SG_N);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint idx = lid % 32u; idx < VIT_SG_M * VIT_SG_N; idx += 32u) {
                const uint m = sg_m0 + idx / VIT_SG_N;
                const uint n = sg_n0 + idx % VIT_SG_N;
                if (m < valid_m && n < valid_n) {
                    float val = Csh[idx];
                    if ((flags & VIT_BIAS) != 0u) {
                        val = val + bias[row0 + n];
                    }
                    if ((flags & VIT_GELU) != 0u) {
                        val = vit_gelu(val);
                    }
                    device float* dst = c + (m0 + m) * n_rows + row0 + n;
                    *dst = ((flags & VIT_ACCUMULATE) != 0u) ? (*dst + val) : val;
                }
            }
        }
    }
}

// `f16` / `f32` weights: per 32-wide slice a fresh `f32` partial, added
// into the accumulator through the identity (`acc += partial · I`) — the
// accumulation the exact `Q8_0` path does, and far more accurate over a
// long `K` than one chain of products into the accumulator (the tower's
// 4304-wide FFN: about 1e-6 relative against 1.3e-7 this way, the CPU
// GEMM's own).
template <uint WF, typename DT>
inline void vit_gemm_body(device const uchar* w, device const float* a, device float* c,
                          device const float* bias, uint n_rows, uint k, uint m_rows,
                          uint flags, uint2 tgid, uint lid, uint sgid,
                          threadgroup float* Ash, threadgroup DT* Dsh) {
    const uint row0 = tgid.x * VIT_TN;
    const uint m0 = tgid.y * VIT_TM;
    if (row0 >= n_rows || m0 >= m_rows) {
        return;
    }
    const uint valid_n = min(VIT_TN, n_rows - row0);
    const uint valid_m = min(VIT_TM, m_rows - m0);
    const uint sg_m0 = (sgid / 2u) * VIT_SG_M;
    const uint sg_n0 = (sgid % 2u) * VIT_SG_N;
    const simdgroup_matrix<DT, 8, 8> ident = simdgroup_matrix<DT, 8, 8>(DT(1.0f));
    simdgroup_float8x8 acc[VIT_MFRAGS][VIT_NFRAGS];
    for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
        for (uint ni = 0u; ni < VIT_NFRAGS; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k_off = 0u; k_off < k; k_off += VIT_TK) {
        vit_stage_w<WF, DT>(w, Dsh, row0, valid_n, k, k_off, lid);
        vit_stage_a(a, Ash, m0, valid_m, k, k_off, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ni = 0u; ni < VIT_NFRAGS; ++ni) {
            simdgroup_float8x8 part[VIT_MFRAGS];
            for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                part[mi] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint kf = 0u; kf < VIT_KFRAGS; ++kf) {
                simdgroup_matrix<DT, 8, 8> dfrag;
                simdgroup_load(dfrag, Dsh + (kf * VIT_FRAG) * VIT_TN + sg_n0 + ni * VIT_FRAG,
                               VIT_TN);
                for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                    simdgroup_float8x8 afrag;
                    simdgroup_load(afrag,
                                   Ash + (sg_m0 + mi * VIT_FRAG) * VIT_TK + kf * VIT_FRAG, VIT_TK);
                    simdgroup_multiply_accumulate(part[mi], afrag, dfrag, part[mi]);
                }
            }
            for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                simdgroup_multiply_accumulate(acc[mi][ni], part[mi], ident, acc[mi][ni]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    vit_gemm_store(acc, c, bias, n_rows, flags, row0, m0, valid_n, valid_m, sg_m0, sg_n0, lid,
                   sgid, Ash);
}

// The exact `Q8_0` GEMM (see the module docs): per 32-wide slice, a fresh
// `f32` partial of the activations against the block's int8 values, then
// `acc += partial · diag(d)`.
inline void vit_gemm_q80_body(device const uchar* w, device const float* a, device float* c,
                              device const float* bias, uint n_rows, uint k, uint m_rows,
                              uint flags, uint2 tgid, uint lid, uint sgid,
                              threadgroup float* Ash, threadgroup half* Dsh,
                              threadgroup half* Gsh) {
    const uint row0 = tgid.x * VIT_TN;
    const uint m0 = tgid.y * VIT_TM;
    if (row0 >= n_rows || m0 >= m_rows) {
        return;
    }
    const uint valid_n = min(VIT_TN, n_rows - row0);
    const uint valid_m = min(VIT_TM, m_rows - m0);
    const uint sg_m0 = (sgid / 2u) * VIT_SG_M;
    const uint sg_n0 = (sgid % 2u) * VIT_SG_N;
    for (uint i = lid; i < VIT_TN * VIT_FRAG; i += VIT_THREADS) {
        Gsh[i] = half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc[VIT_MFRAGS][VIT_NFRAGS];
    for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
        for (uint ni = 0u; ni < VIT_NFRAGS; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k_off = 0u; k_off < k; k_off += VIT_TK) {
        vit_stage_q80(w, Dsh, Gsh, row0, valid_n, k, k_off, lid);
        vit_stage_a(a, Ash, m0, valid_m, k, k_off, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // `VIT_Q80_NSTEP` column fragments at a time keeps the per-slice
        // partials to a few fragments beside the sixteen accumulators.
        for (uint nh = 0u; nh < VIT_NFRAGS; nh += VIT_Q80_NSTEP) {
            simdgroup_float8x8 part[VIT_MFRAGS][VIT_Q80_NSTEP];
            for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                for (uint nj = 0u; nj < VIT_Q80_NSTEP; ++nj) {
                    part[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (uint kf = 0u; kf < VIT_KFRAGS; ++kf) {
                simdgroup_float8x8 afrag[VIT_MFRAGS];
                simdgroup_half8x8 qfrag[VIT_Q80_NSTEP];
                for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                    simdgroup_load(afrag[mi],
                                   Ash + (sg_m0 + mi * VIT_FRAG) * VIT_TK + kf * VIT_FRAG, VIT_TK);
                }
                for (uint nj = 0u; nj < VIT_Q80_NSTEP; ++nj) {
                    simdgroup_load(qfrag[nj],
                                   Dsh + (kf * VIT_FRAG) * VIT_TN + sg_n0 + (nh + nj) * VIT_FRAG,
                                   VIT_TN);
                }
                for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                    for (uint nj = 0u; nj < VIT_Q80_NSTEP; ++nj) {
                        simdgroup_multiply_accumulate(part[mi][nj], afrag[mi], qfrag[nj],
                                                      part[mi][nj]);
                    }
                }
            }
            for (uint nj = 0u; nj < VIT_Q80_NSTEP; ++nj) {
                simdgroup_half8x8 gfrag;
                simdgroup_load(gfrag,
                               Gsh + (sg_n0 / VIT_FRAG + nh + nj) * (VIT_FRAG * VIT_FRAG),
                               VIT_FRAG);
                for (uint mi = 0u; mi < VIT_MFRAGS; ++mi) {
                    simdgroup_multiply_accumulate(acc[mi][nh + nj], part[mi][nj], gfrag,
                                                  acc[mi][nh + nj]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    vit_gemm_store(acc, c, bias, n_rows, flags, row0, m0, valid_n, valid_m, sg_m0, sg_n0, lid,
                   sgid, Ash);
}

// Buffers: weights (0), activations `[m_rows][k]` (1), output `[m_rows]
// [n_rows]` (2), bias `[n_rows]` (3, read only with the bias flag); scalars
// `n_rows` (4), `k` (5), `m_rows` (6), `flags` (7). Grid `[ceil(n_rows /
// 64), ceil(m_rows / 64), 1]`, 128 threads.
kernel void vit_gemm_f16w(
    device const uchar* w        [[buffer(0)]],
    device const float* a        [[buffer(1)]],
    device float* c              [[buffer(2)]],
    device const float* bias     [[buffer(3)]],
    constant uint& n_rows        [[buffer(4)]],
    constant uint& k             [[buffer(5)]],
    constant uint& m_rows        [[buffer(6)]],
    constant uint& flags         [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[VIT_TM * VIT_TK];
    threadgroup half Dsh[VIT_TK * VIT_TN];
    vit_gemm_body<VIT_W_F16, half>(w, a, c, bias, n_rows, k, m_rows, flags, tgid, lid, sgid,
                                   Ash, Dsh);
}

kernel void vit_gemm_f32w(
    device const uchar* w        [[buffer(0)]],
    device const float* a        [[buffer(1)]],
    device float* c              [[buffer(2)]],
    device const float* bias     [[buffer(3)]],
    constant uint& n_rows        [[buffer(4)]],
    constant uint& k             [[buffer(5)]],
    constant uint& m_rows        [[buffer(6)]],
    constant uint& flags         [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[VIT_TM * VIT_TK];
    threadgroup float Dsh[VIT_TK * VIT_TN];
    vit_gemm_body<VIT_W_F32, float>(w, a, c, bias, n_rows, k, m_rows, flags, tgid, lid, sgid,
                                    Ash, Dsh);
}

kernel void vit_gemm_q80(
    device const uchar* w        [[buffer(0)]],
    device const float* a        [[buffer(1)]],
    device float* c              [[buffer(2)]],
    device const float* bias     [[buffer(3)]],
    constant uint& n_rows        [[buffer(4)]],
    constant uint& k             [[buffer(5)]],
    constant uint& m_rows        [[buffer(6)]],
    constant uint& flags         [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float Ash[VIT_TM * VIT_TK];
    threadgroup half Dsh[VIT_TK * VIT_TN];
    threadgroup half Gsh[VIT_TN * VIT_FRAG];
    vit_gemm_q80_body(w, a, c, bias, n_rows, k, m_rows, flags, tgid, lid, sgid, Ash, Dsh, Gsh);
}
"#;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::MSL_VISION;

    /// Every entry point `metal_vision` resolves by name is declared at the
    /// start of a line (the runtime verifier's line-anchored scan), and the
    /// source cannot end the build script's raw string early.
    #[test]
    fn every_vision_entry_point_is_declared() {
        for name in [
            "vit_layer_norm",
            "vit_qkv_rope_pack",
            "vit_add_rows",
            "vit_gemm_f16w",
            "vit_gemm_f32w",
            "vit_gemm_q80",
        ] {
            let entry = format!("kernel void {name}(");
            assert!(
                MSL_VISION.lines().any(|l| l.starts_with(&entry)),
                "{entry} missing"
            );
        }
        assert_eq!(MSL_VISION.matches("kernel void ").count(), 6);
        assert!(!MSL_VISION.contains("\"#"));
        assert!(MSL_VISION.contains("precise::tanh"));
    }
}
