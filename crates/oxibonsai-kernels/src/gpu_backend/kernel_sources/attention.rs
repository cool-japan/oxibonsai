//! Attention Metal kernels (fused and batched attention operations).
//!
//! Contains fused QK-norm, QK-RoPE, KV-store, batched attention
//! score/softmax/weighted-sum kernels.
//!
//! # Partial RoPE (MET-M2)
//!
//! `fused_qk_rope` and `fused_qk_norm_rope` rotate the whole head
//! (`n_rot == head_dim`). Qwen3.5 / Bonsai 2 rotates only the first
//! `rope.dimension_count = 64` of its 256 dims, so each constant also carries
//! a `*_partial` entry point with an explicit `n_rot` scalar binding: it
//! rotates the NeoX split-half pairs `(j, j + n_rot/2)` for `j < n_rot/2` and
//! copies `[n_rot, head_dim)` through verbatim. Both entry points of a
//! constant run the same templated per-head body; the full-rotation entry
//! instantiates it with `n_rot = head_dim` and the historical `fma` pair
//! arithmetic, so every existing caller computes exactly what it did before
//! (pinned by this module's tests).
//!
//! The partial entry points round each product of a rotated pair
//! separately — `x0*c - x1*s`, `x1*c + x0*s` — which is the arithmetic of the
//! CPU NEON path (`vmulq` then the non-fused `vmlsq`/`vmlaq`), so a partial
//! rotation is bitwise-reproducible against `rope_partial_splithalf_simd`.
//! Metal contracts `a*b - c*d` into an FMA by default (`fp contract(fast)` is
//! the MSL default), so the exact pair helper is defined inside a
//! `#pragma METAL fp contract(off)` region that is closed again with
//! `fp contract(fast)` — the default — before any other code, because every
//! constant here is concatenated into one combined translation unit.
//!
//! # Head dimension (MET-01)
//!
//! `batched_attention_scores_v2` stages the query in
//! `threadgroup float shared_q[256]` (`ATTN_V2_MAX_HEAD_DIM`, 1 KiB of the
//! 32 KiB threadgroup budget) and accumulates with a strided loop over
//! `head_dim`, so it is correct for every `head_dim <= 256` at its fixed
//! 128-thread (four-simdgroup) dispatch. For `head_dim <= 128` the loop runs
//! once per thread and the arithmetic is unchanged.
//!
//! # KV offsets (MET-07)
//!
//! Every KV-cache layer offset is bound as `constant ulong&` and every
//! derived cache offset is computed in 64 bits, so a cache whose element
//! count leaves the 32-bit range addresses correctly.

/// Fused QK-Norm: apply RMSNorm to both Q and K heads in a single dispatch.
///
/// The first `nq` threadgroups normalise Q heads, the remaining `nkv`
/// threadgroups normalise K heads.  Each threadgroup processes one head
/// using shared-memory parallel reduction for sum-of-squares.
///
/// Replaces two separate `batched_rmsnorm_v2` dispatches (Q-norm + K-norm).
///
/// Buffers:
///   - `q_in`     `[nq × head_dim]` (f32)
///   - `k_in`     `[nkv × head_dim]` (f32)
///   - `q_out`    `[nq × head_dim]` (f32)
///   - `k_out`    `[nkv × head_dim]` (f32)
///   - `q_weight` `[head_dim]` (f32)
///   - `k_weight` `[head_dim]` (f32)
///   - `nq`       (u32 scalar)
///   - `nkv`      (u32 scalar)
///   - `head_dim` (u32 scalar)
///   - `eps`      (f32 scalar)
///
/// Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_qk_norm(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* q_weight [[buffer(4)]],
    device const float* k_weight [[buffer(5)]],
    constant uint& nq            [[buffer(6)]],
    constant uint& nkv           [[buffer(7)]],
    constant uint& head_dim      [[buffer(8)]],
    constant float& eps          [[buffer(9)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    // First nq groups = Q heads, remaining nkv groups = K heads
    const bool is_q = (gid < nq);
    const uint head_idx = is_q ? gid : (gid - nq);

    device const float* in_ptr  = is_q ? (q_in + head_idx * head_dim)  : (k_in + head_idx * head_dim);
    device float* out_ptr       = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);
    device const float* w_ptr   = is_q ? q_weight : k_weight;

    // Sum of squares via shared-memory reduction
    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Apply normalization with weight
    for (uint i = tid; i < head_dim; i += tpg) {
        out_ptr[i] = in_ptr[i] * rms_inv * w_ptr[i];
    }
}
"#;

/// Fused QK-RoPE: apply rotary position embedding to both Q and K heads
/// in a single dispatch.
///
/// Two entry points share one per-head body (see the module docs):
///
/// * `fused_qk_rope` — full rotation (`n_rot == head_dim = 2 * half_dim`),
///   the historical `fma` pair arithmetic. Thread groups are 2-D:
///   `(ceil(half_dim/64), nq + nkv)`; groups with `gid.y < nq` rotate Q, the
///   rest rotate K. Buffers: `q_in`, `k_in`, `q_out`, `k_out` (`[heads ×
///   head_dim]` f32), `cos_buf`/`sin_buf` (`[half_dim]` f32), then the
///   scalars `nq`, `nkv`, `half_dim`.
/// * `fused_qk_rope_partial` — rotate the first `n_rot` dims of each head
///   (NeoX pairs `(j, j + n_rot/2)`), copy `[n_rot, head_dim)` verbatim, with
///   exact (non-contracted) pair arithmetic. Thread groups are
///   `(ceil(head_dim/64), nq + nkv)` of 64 threads; thread `d` rotates pair
///   `d` when `d < n_rot/2` and copies element `d` when `d >= n_rot`.
///   Buffers: `q_in`, `k_in`, `q_out`, `k_out`, `cos_buf`/`sin_buf`
///   (`[n_rot/2]`), then the scalars `nq`, `nkv`, `head_dim`, `n_rot` and
///   `q_in_stride` — the float distance between consecutive Q heads of
///   `q_in` (`head_dim` for a packed buffer, `2 * head_dim` for Qwen3.5's
///   `[q | gate]` interleave). Outputs are packed `[heads × head_dim]`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float2 qk_rope_pair_fma(float x0, float x1, float c, float s) {
    return float2(fma(x0, c, -(x1 * s)), fma(x0, s, x1 * c));
}

#pragma METAL fp contract(off)
// Each product rounded on its own, exactly as the CPU NEON path computes the
// pair (`vmulq_f32` then the non-fused `vmlsq_f32` / `vmlaq_f32`).
inline float2 qk_rope_pair_exact(float x0, float x1, float c, float s) {
    return float2(x0 * c - x1 * s, x1 * c + x0 * s);
}
#pragma METAL fp contract(fast)

// One element `d` of one head: rotate the pair (d, d + n_rot/2) when
// d < n_rot/2, copy the element when d >= n_rot, nothing otherwise.
template <bool EXACT>
inline void qk_rope_head_elem(device const float* in_ptr, device float* out_ptr,
                              device const float* cos_buf, device const float* sin_buf,
                              uint d, uint n_rot) {
    const uint half_rot = n_rot / 2u;
    if (d < half_rot) {
        const float c = cos_buf[d];
        const float s = sin_buf[d];
        const float x0 = in_ptr[d];
        const float x1 = in_ptr[d + half_rot];
        const float2 r = EXACT ? qk_rope_pair_exact(x0, x1, c, s) : qk_rope_pair_fma(x0, x1, c, s);
        out_ptr[d]            = r.x;
        out_ptr[d + half_rot] = r.y;
    } else if (d >= n_rot) {
        out_ptr[d] = in_ptr[d];
    }
}

kernel void fused_qk_rope(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* cos_buf  [[buffer(4)]],
    device const float* sin_buf  [[buffer(5)]],
    constant uint& nq            [[buffer(6)]],
    constant uint& nkv           [[buffer(7)]],
    constant uint& half_dim      [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= half_dim) return;

    const bool is_q = (gid.y < nq);
    const uint head_idx = is_q ? gid.y : (gid.y - nq);
    const uint head_dim = half_dim * 2u;

    device const float* in_ptr = is_q ? (q_in + head_idx * head_dim) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);

    qk_rope_head_elem<false>(in_ptr, out_ptr, cos_buf, sin_buf, d, head_dim);
}

kernel void fused_qk_rope_partial(
    device const float* q_in      [[buffer(0)]],
    device const float* k_in      [[buffer(1)]],
    device float* q_out           [[buffer(2)]],
    device float* k_out           [[buffer(3)]],
    device const float* cos_buf   [[buffer(4)]],
    device const float* sin_buf   [[buffer(5)]],
    constant uint& nq             [[buffer(6)]],
    constant uint& nkv            [[buffer(7)]],
    constant uint& head_dim       [[buffer(8)]],
    constant uint& n_rot          [[buffer(9)]],
    constant uint& q_in_stride    [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    if (d >= head_dim || gid.y >= nq + nkv) return;

    const bool is_q = (gid.y < nq);
    const uint head_idx = is_q ? gid.y : (gid.y - nq);

    device const float* in_ptr = is_q ? (q_in + head_idx * q_in_stride) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);

    qk_rope_head_elem<true>(in_ptr, out_ptr, cos_buf, sin_buf, d, n_rot);
}
"#;

/// Fused QK-Norm + QK-RoPE: apply RMSNorm then rotary position embedding
/// to both Q and K heads in a single dispatch, eliminating intermediate
/// normalised buffers.
///
/// Two entry points share one per-head body (see the module docs):
///
/// * `fused_qk_norm_rope` — full rotation, the historical arithmetic
///   (`rsqrt`, `x * inv * w`, `fma` pairs). The first `nq` threadgroups
///   process Q heads, the remaining `nkv` threadgroups process K heads; each
///   computes the RMSNorm via a shared-memory reduction and applies the
///   rotation in the same pass. Buffers: `q_in`, `k_in`, `q_out`, `k_out`,
///   `q_weight`/`k_weight` (`[head_dim]`), `cos_buf`/`sin_buf`
///   (`[half_dim]`), then the scalars `nq`, `nkv`, `head_dim`, `eps`.
///   Dispatch: `[nq + nkv, 1, 1]` threadgroups, `[256, 1, 1]` threads.
/// * `fused_qk_norm_rope_partial` — the same shape over `n_rot` of
///   `head_dim` rotated dims (tail copied), with the CPU's RMSNorm operation
///   order (`1/sqrt`, `w * (x * inv)`) and exact pair arithmetic, over
///   `t_len` tokens at once: threadgroup `(h, t)` normalises + rotates head
///   `h` of token `t`. Buffers: `q_in`, `k_in`, `q_out`, `k_out`,
///   `q_weight`, `k_weight`, `cos_buf`/`sin_buf` (`[t_len][n_rot/2]`, row
///   `t` holding token `t`'s angles), then the scalars `nq`, `nkv`,
///   `head_dim`, `eps`, `n_rot`, `q_in_stride` (floats between consecutive Q
///   heads), `q_in_token_stride` and `k_in_token_stride` (floats between
///   consecutive tokens of `q_in` / `k_in`). Outputs are packed
///   `[t_len][heads × head_dim]`. Dispatch: `[nq + nkv, t_len, 1]`
///   threadgroups, `[256, 1, 1]` threads (a power of two).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_QK_NORM_ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float2 qk_norm_rope_pair_fma(float x0, float x1, float c, float s) {
    return float2(fma(x0, c, -(x1 * s)), fma(x0, s, x1 * c));
}

#pragma METAL fp contract(off)
// Each product rounded on its own, exactly as the CPU NEON path computes the
// pair (`vmulq_f32` then the non-fused `vmlsq_f32` / `vmlaq_f32`).
inline float2 qk_norm_rope_pair_exact(float x0, float x1, float c, float s) {
    return float2(x0 * c - x1 * s, x1 * c + x0 * s);
}
#pragma METAL fp contract(fast)

// RMSNorm (shared-memory tree reduction; `tpg` must be a power of two
// <= 256) followed by the rotation of the first `n_rot` dims of one head and
// a verbatim copy of `[n_rot, head_dim)`.
template <bool EXACT>
inline void qk_norm_rope_head(device const float* in_ptr, device float* out_ptr,
                              device const float* w_ptr,
                              device const float* cos_buf, device const float* sin_buf,
                              uint head_dim, uint n_rot, float eps,
                              threadgroup float* shared_sum, uint tid, uint tpg) {
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint half_rot = n_rot / 2u;
    if (EXACT) {
        const float inv = 1.0f / sqrt(shared_sum[0] / float(head_dim) + eps);
        for (uint d = tid; d < half_rot; d += tpg) {
            const float normed_lo = w_ptr[d] * (in_ptr[d] * inv);
            const float normed_hi = w_ptr[d + half_rot] * (in_ptr[d + half_rot] * inv);
            const float2 r = qk_norm_rope_pair_exact(normed_lo, normed_hi, cos_buf[d], sin_buf[d]);
            out_ptr[d]            = r.x;
            out_ptr[d + half_rot] = r.y;
        }
        for (uint d = n_rot + tid; d < head_dim; d += tpg) {
            out_ptr[d] = w_ptr[d] * (in_ptr[d] * inv);
        }
    } else {
        float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);
        for (uint d = tid; d < half_rot; d += tpg) {
            float normed_lo = in_ptr[d] * rms_inv * w_ptr[d];
            float normed_hi = in_ptr[d + half_rot] * rms_inv * w_ptr[d + half_rot];

            float c = cos_buf[d];
            float s = sin_buf[d];
            const float2 r = qk_norm_rope_pair_fma(normed_lo, normed_hi, c, s);
            out_ptr[d]            = r.x;
            out_ptr[d + half_rot] = r.y;
        }
    }
}

kernel void fused_qk_norm_rope(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* q_weight [[buffer(4)]],
    device const float* k_weight [[buffer(5)]],
    device const float* cos_buf  [[buffer(6)]],
    device const float* sin_buf  [[buffer(7)]],
    constant uint& nq            [[buffer(8)]],
    constant uint& nkv           [[buffer(9)]],
    constant uint& head_dim      [[buffer(10)]],
    constant float& eps          [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const bool is_q = (gid < nq);
    const uint head_idx = is_q ? gid : (gid - nq);

    device const float* in_ptr = is_q ? (q_in + head_idx * head_dim) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);
    device const float* w_ptr  = is_q ? q_weight : k_weight;

    threadgroup float shared_sum[256];
    qk_norm_rope_head<false>(in_ptr, out_ptr, w_ptr, cos_buf, sin_buf,
                             head_dim, head_dim, eps, shared_sum, tid, tpg);
}

kernel void fused_qk_norm_rope_partial(
    device const float* q_in            [[buffer(0)]],
    device const float* k_in            [[buffer(1)]],
    device float* q_out                 [[buffer(2)]],
    device float* k_out                 [[buffer(3)]],
    device const float* q_weight        [[buffer(4)]],
    device const float* k_weight        [[buffer(5)]],
    device const float* cos_buf         [[buffer(6)]],
    device const float* sin_buf         [[buffer(7)]],
    constant uint& nq                   [[buffer(8)]],
    constant uint& nkv                  [[buffer(9)]],
    constant uint& head_dim             [[buffer(10)]],
    constant float& eps                 [[buffer(11)]],
    constant uint& n_rot                [[buffer(12)]],
    constant uint& q_in_stride          [[buffer(13)]],
    constant uint& q_in_token_stride    [[buffer(14)]],
    constant uint& k_in_token_stride    [[buffer(15)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint2 tpg2 [[threads_per_threadgroup]])
{
    const uint head = tg.x;
    const uint t = tg.y;
    const uint tpg = tpg2.x;
    if (head >= nq + nkv) return;
    const bool is_q = (head < nq);
    const uint head_idx = is_q ? head : (head - nq);
    const uint half_rot = n_rot / 2u;

    device const float* in_ptr = is_q
        ? (q_in + t * q_in_token_stride + head_idx * q_in_stride)
        : (k_in + t * k_in_token_stride + head_idx * head_dim);
    device float* out_ptr = is_q
        ? (q_out + (t * nq + head_idx) * head_dim)
        : (k_out + (t * nkv + head_idx) * head_dim);
    device const float* w_ptr = is_q ? q_weight : k_weight;

    threadgroup float shared_sum[256];
    qk_norm_rope_head<true>(in_ptr, out_ptr, w_ptr, cos_buf + t * half_rot, sin_buf + t * half_rot,
                            head_dim, n_rot, eps, shared_sum, tid, tpg);
}
"#;

/// Fused KV-Store: copy both K and V heads into the GPU KV cache in a
/// single dispatch.
///
/// Thread groups are 2-D: `(ceil(head_dim/64), nkv)`.  Each thread copies
/// one element of one head for both K and V simultaneously.
///
/// Replaces two separate `kv_cache_store` dispatches (K-store + V-store).
///
/// Buffers:
///   - `k_data`        `[nkv × head_dim]` (f32, after RoPE)
///   - `v_data`        `[nkv × head_dim]` (f32, raw from QKV)
///   - `k_cache`       `[n_layers × nkv × max_seq × head_dim]` (f16)
///   - `v_cache`       `[n_layers × nkv × max_seq × head_dim]` (f16)
///   - `head_dim`      (u32 scalar)
///   - `nkv`           (u32 scalar)
///   - `max_seq`       (u32 scalar)
///   - `pos`           (u32 scalar)
///   - `layer_offset`  (u64 scalar, `= layer_idx * nkv * max_seq * head_dim`)
///
/// Dispatch: `[ceil(head_dim/64), nkv, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_FUSED_KV_STORE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_kv_store(
    device const float* k_data   [[buffer(0)]],
    device const float* v_data   [[buffer(1)]],
    device half* k_cache          [[buffer(2)]],
    device half* v_cache          [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& nkv           [[buffer(5)]],
    constant uint& max_seq       [[buffer(6)]],
    constant uint& pos           [[buffer(7)]],
    constant ulong& layer_offset [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint head = gid.y;
    if (d >= head_dim || head >= nkv) return;

    const uint src_offset = head * head_dim + d;
    const ulong dst_offset = layer_offset
        + (ulong(head) * ulong(max_seq) + ulong(pos)) * ulong(head_dim) + ulong(d);

    k_cache[dst_offset] = half(k_data[src_offset]);
    v_cache[dst_offset] = half(v_data[src_offset]);
}
"#;

/// Batched attention scores: all Q heads compute dot-product scores against
/// cached K with GQA mapping (`kv_head = q_head / heads_per_group`).
///
/// One threadgroup per (q_head, position) pair. Each threadgroup of 256
/// threads performs a parallel dot-product reduction over `head_dim`.
///
/// Buffers:
///   - `queries`            `[n_q × head_dim]` (f32)
///   - `k_cache`            `[n_kv × max_seq × head_dim]` (f32)
///   - `all_scores`         `[n_q × max_seq]` (f32, output at `q_head*max_seq+pos`)
///   - `head_dim`           (u32 scalar)
///   - `n_q`                (u32 scalar)
///   - `n_kv`               (u32 scalar)
///   - `heads_per_group`    (u32 scalar)
///   - `max_seq`            (u32 scalar)
///   - `seq_len`            (u32 scalar)
///   - `inv_sqrt_hd`        (f32 scalar)
///   - `cache_layer_offset` (u64 scalar, `= layer_idx * n_kv * max_seq * head_dim`)
///
/// Dispatch: `[n_q, seq_len, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_SCORES: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_attention_scores(
    device const float* queries,
    device const half* k_cache,
    device float* all_scores,
    constant uint& head_dim,
    constant uint& n_q,
    constant uint& n_kv,
    constant uint& heads_per_group,
    constant uint& max_seq,
    constant uint& seq_len,
    constant float& inv_sqrt_hd,
    constant ulong& cache_layer_offset,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntpitg [[threads_per_threadgroup]])
{
    uint q_head = tgpig.x;
    uint pos_t = tgpig.y;
    uint tid = tpitg.x;
    uint tg_size = ntpitg.x;
    if (q_head >= n_q || pos_t >= seq_len) return;

    uint kv_head = q_head / heads_per_group;

    device const float* query = queries + q_head * head_dim;
    device const half* key = k_cache + cache_layer_offset
        + (ulong(kv_head) * ulong(max_seq) + ulong(pos_t)) * ulong(head_dim);

    // Parallel dot product with shared memory reduction
    threadgroup float shared[256];
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_size) {
        partial = fma(query[i], float(key[i]), partial);
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        all_scores[q_head * max_seq + pos_t] = shared[0] * inv_sqrt_hd;
    }
}
"#;

/// Largest `head_dim` [`MSL_BATCHED_ATTENTION_SCORES_V2`] stages and scores
/// completely: the size of its `threadgroup float shared_q[...]` array and
/// its `ATTN_V2_MAX_HEAD_DIM` guard (both asserted equal in this module's
/// tests). 256 floats are 1 KiB of the 32 KiB threadgroup budget measured on
/// the M3, and cover Bonsai 2's `head_dim = 256` (MET-01).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY: u32 = 256;

/// Threads per threadgroup `batched_attention_scores_v2` is written for:
/// four simdgroups, reduced through `threadgroup float sg_partial[4]`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const ATTENTION_SCORES_V2_THREADS: u64 = 128;

/// Batched attention scores V2: fixed 128-thread TGs + position batching.
///
/// Each TG handles one Q head and processes `batch_stride` positions.
/// The Q vector is loaded into shared memory once and reused across
/// positions. Every thread accumulates the dims `tid, tid + 128, …` of each
/// dot product (`head_dim <= 256`, see the module docs), then the four
/// simdgroup partials are summed.
///
/// Buffers:
///   - `queries`            `[n_q × head_dim]` (f32)
///   - `k_cache`            `[slots × n_kv × max_seq × head_dim]` (f16)
///   - `all_scores`         `[n_q × max_seq]` (f32)
///   - `head_dim` .. `inv_sqrt_hd` as in [`MSL_BATCHED_ATTENTION_SCORES`]
///   - `cache_layer_offset` (u64 scalar)
///   - `batch_stride`       (u32 scalar, positions per threadgroup)
///
/// Grid: `[n_q, ceil(seq_len / batch_stride), 1]` TGs, `[128, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_SCORES_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint ATTN_V2_MAX_HEAD_DIM = 256u;

kernel void batched_attention_scores_v2(
    device const float* queries          [[buffer(0)]],
    device const half* k_cache            [[buffer(1)]],
    device float* all_scores             [[buffer(2)]],
    constant uint& head_dim              [[buffer(3)]],
    constant uint& n_q                   [[buffer(4)]],
    constant uint& n_kv                  [[buffer(5)]],
    constant uint& heads_per_group       [[buffer(6)]],
    constant uint& max_seq               [[buffer(7)]],
    constant uint& seq_len               [[buffer(8)]],
    constant float& inv_sqrt_hd          [[buffer(9)]],
    constant ulong& cache_layer_offset   [[buffer(10)]],
    constant uint& batch_stride          [[buffer(11)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint3 ntpitg [[threads_per_threadgroup]])
{
    uint q_head = tgpig.x;
    uint batch_id = tgpig.y;
    uint tg_size = ntpitg.x;
    // The host refuses head_dim > ATTN_V2_MAX_HEAD_DIM before dispatch; the
    // guard keeps the staging writes in bounds regardless.
    if (q_head >= n_q || head_dim > ATTN_V2_MAX_HEAD_DIM) return;

    uint kv_head = q_head / heads_per_group;
    uint pos_start = batch_id * batch_stride;

    // Load Q vector into shared memory (reused across all positions);
    // ATTN_V2_MAX_HEAD_DIM floats.
    threadgroup float shared_q[256];
    for (uint i = tid; i < head_dim; i += tg_size) {
        shared_q[i] = queries[q_head * head_dim + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each position in this batch
    for (uint pos_t = pos_start; pos_t < min(pos_start + batch_stride, seq_len); pos_t++) {
        device const half* key = k_cache + cache_layer_offset
            + (ulong(kv_head) * ulong(max_seq) + ulong(pos_t)) * ulong(head_dim);

        // Strided dot product: dims tid, tid + tg_size, ... For
        // head_dim <= tg_size this is exactly one product per thread.
        float my_prod = 0.0f;
        uint i = tid;
        if (i < head_dim) {
            my_prod = shared_q[i] * float(key[i]);
            i += tg_size;
        }
        for (; i < head_dim; i += tg_size) {
            my_prod = fma(shared_q[i], float(key[i]), my_prod);
        }

        // SIMD-level reduction first (fast, within simdgroup)
        float sg_sum = simd_sum(my_prod);

        // Cross-simdgroup reduction via shared memory
        // 128 threads = 4 simdgroups (32 threads each)
        threadgroup float sg_partial[4];
        uint sgid = tid / 32u;
        uint lane = tid % 32u;
        if (lane == 0u) {
            sg_partial[sgid] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float total = sg_partial[0] + sg_partial[1] + sg_partial[2] + sg_partial[3];
            all_scores[q_head * max_seq + pos_t] = total * inv_sqrt_hd;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Batched softmax: per-head numerically-stable softmax.
///
/// One threadgroup per Q head. Three-pass approach:
/// 1. Find max (parallel reduction)
/// 2. Compute exp(x - max) and accumulate sum
/// 3. Normalize by sum
///
/// Buffers:
///   - `all_scores` `[n_q × max_seq]` (f32, in-place)
///   - `n_q`        (u32 scalar)
///   - `max_seq`    (u32 scalar)
///   - `seq_len`    (u32 scalar)
///
/// Dispatch: `[n_q, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_softmax(
    device float* all_scores,
    constant uint& n_q,
    constant uint& max_seq,
    constant uint& seq_len,
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgpig >= n_q) return;

    device float* scores = all_scores + tgpig * max_seq;
    threadgroup float shared[256];

    // Pass 1: max
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_len; i += tg_size) {
        local_max = max(local_max, scores[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exp + sum
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        float e = exp(scores[i] - gmax);
        scores[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gsum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: normalize
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    for (uint i = tid; i < seq_len; i += tg_size) {
        scores[i] *= inv_sum;
    }
}
"#;

/// Batched attention weighted sum: per-head `output[d] = Σ_t scores[t] × V[t][d]`.
///
/// One thread per (dimension, q_head) pair with GQA mapping.
///
/// Buffers:
///   - `all_scores`         `[n_q × max_seq]` (f32)
///   - `v_cache`            `[n_kv × max_seq × head_dim]` (f16)
///   - `attn_out`           `[n_q × head_dim]` (f32, output)
///   - `head_dim`           (u32 scalar)
///   - `n_q`                (u32 scalar)
///   - `n_kv`               (u32 scalar)
///   - `heads_per_group`    (u32 scalar)
///   - `max_seq`            (u32 scalar)
///   - `seq_len`            (u32 scalar)
///   - `cache_layer_offset` (u64 scalar, `= layer_idx * n_kv * max_seq * head_dim`)
///
/// Dispatch: `[ceil(head_dim/64), n_q, 1]` threadgroups, `[64, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_BATCHED_ATTENTION_WEIGHTED_SUM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batched_attention_weighted_sum(
    device const float* all_scores,
    device const half* v_cache,
    device float* attn_out,
    constant uint& head_dim,
    constant uint& n_q,
    constant uint& n_kv,
    constant uint& heads_per_group,
    constant uint& max_seq,
    constant uint& seq_len,
    constant ulong& cache_layer_offset,
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint q_head = gid.y;
    if (d >= head_dim || q_head >= n_q) return;

    uint kv_head = q_head / heads_per_group;
    device const float* scores = all_scores + q_head * max_seq;
    device const half* values = v_cache + cache_layer_offset
        + ulong(kv_head) * ulong(max_seq) * ulong(head_dim);

    float acc = 0.0f;
    for (uint t = 0u; t < seq_len; t++) {
        acc = fma(scores[t], float(values[ulong(t) * ulong(head_dim) + ulong(d)]), acc);
    }
    attn_out[q_head * head_dim + d] = acc;
}
"#;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    //! On-device checks of the MET-01 / MET-M2 / MET-07 changes: the
    //! `head_dim`-generic score kernel, the partial-RoPE entry points (bitwise
    //! against the CPU), and the full-rotation entry points' unchanged
    //! arithmetic.

    use super::*;
    use crate::gpu_backend::metal_graph::MetalGraph;
    use metal::{
        Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };

    /// Deterministic xorshift64* stream in `[-1, 1)`.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * scale).collect()
        }
    }

    /// `None` only without a Metal device: on a device, a combined library
    /// that fails to build fails the test instead of skipping it.
    fn graph() -> Option<MetalGraph> {
        Device::system_default()?;
        match MetalGraph::new() {
            Ok(graph) => Some(graph),
            Err(e) => panic!("the combined Metal library must build on this device: {e}"),
        }
    }

    fn shared_f32(device: &Device, data: &[f32]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len().max(1) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn shared_zeroed(device: &Device, bytes: usize) -> Buffer {
        let buf = device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared);
        // SAFETY: freshly allocated shared buffer of at least `bytes` bytes.
        unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes) };
        buf
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        // SAFETY: every caller sized `buf` for at least `n` floats.
        (0..n)
            .map(|i| unsafe { std::ptr::read(ptr.add(i)) })
            .collect()
    }

    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, v: u32) {
        enc.set_bytes(index, 4, &v as *const u32 as *const std::ffi::c_void);
    }

    fn set_f32(enc: &metal::ComputeCommandEncoderRef, index: u64, v: f32) {
        enc.set_bytes(index, 4, &v as *const f32 as *const std::ffi::c_void);
    }

    fn set_u64(enc: &metal::ComputeCommandEncoderRef, index: u64, v: u64) {
        enc.set_bytes(index, 8, &v as *const u64 as *const std::ffi::c_void);
    }

    fn run(graph: &MetalGraph, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(
            cmd.status(),
            metal::MTLCommandBufferStatus::Completed,
            "command buffer did not complete"
        );
    }

    fn standalone(device: &Device, src: &str, name: &str) -> ComputePipelineState {
        let lib = device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("standalone MSL compiles");
        let f = lib.get_function(name, None).expect("entry point");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline")
    }

    /// The MSL staging array and the Rust capacity constant must agree.
    #[test]
    fn scores_v2_capacity_constant_matches_the_msl_staging_array() {
        assert!(MSL_BATCHED_ATTENTION_SCORES_V2.contains(&format!(
            "constant constexpr uint ATTN_V2_MAX_HEAD_DIM = {ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY}u;"
        )));
        assert!(MSL_BATCHED_ATTENTION_SCORES_V2.contains(&format!(
            "threadgroup float shared_q[{ATTENTION_SCORES_V2_HEAD_DIM_CAPACITY}];"
        )));
        assert!(
            MSL_BATCHED_ATTENTION_SCORES_V2.contains("head_dim > ATTN_V2_MAX_HEAD_DIM) return;")
        );
        assert!(MSL_BATCHED_ATTENTION_SCORES_V2.contains("threadgroup float sg_partial[4];"));
        assert_eq!(ATTENTION_SCORES_V2_THREADS, 4 * 32);
    }

    /// MET-01: the score kernel is exact for every `head_dim` up to 256, at a
    /// non-zero KV slot, with the dispatcher's fixed 128-thread shape. The
    /// pre-fix kernel returned the first-128-dim partial sum (4.13628 instead
    /// of 7.43115 at `head_dim` 256 on this M3).
    #[test]
    fn scores_v2_is_head_dim_generic_up_to_256() {
        let Some(graph) = graph() else {
            return;
        };
        let device = &graph.device;
        for head_dim in [64usize, 128, 192, 256] {
            let (n_q, n_kv, max_seq, slots, slot) = (6usize, 2usize, 24usize, 3usize, 2usize);
            let seq_len = 19usize;
            let mut rng = Rng::new(head_dim as u64);
            let q = rng.vec(n_q * head_dim, 1.0);
            let cache_len = slots * n_kv * max_seq * head_dim;
            let k_f32 = rng.vec(cache_len, 1.0);
            let k_half: Vec<half::f16> = k_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
            let k_buf = device.new_buffer_with_data(
                k_half.as_ptr() as *const std::ffi::c_void,
                (k_half.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let q_buf = shared_f32(device, &q);
            let scores = shared_zeroed(device, n_q * max_seq * 4);
            let layer_offset = (slot * n_kv * max_seq * head_dim) as u64;
            let inv = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_group = (n_q / n_kv) as u32;
            let batch_stride = 16u32;
            run(&graph, |enc| {
                enc.set_compute_pipeline_state(&graph.pipelines.batched_attention_scores_v2);
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&k_buf), 0);
                enc.set_buffer(2, Some(&scores), 0);
                set_u32(enc, 3, head_dim as u32);
                set_u32(enc, 4, n_q as u32);
                set_u32(enc, 5, n_kv as u32);
                set_u32(enc, 6, heads_per_group);
                set_u32(enc, 7, max_seq as u32);
                set_u32(enc, 8, seq_len as u32);
                set_f32(enc, 9, inv);
                set_u64(enc, 10, layer_offset);
                set_u32(enc, 11, batch_stride);
                enc.dispatch_thread_groups(
                    MTLSize::new(n_q as u64, seq_len.div_ceil(16) as u64, 1),
                    MTLSize::new(ATTENTION_SCORES_V2_THREADS, 1, 1),
                );
            });
            let got = read_f32(&scores, n_q * max_seq);
            for h in 0..n_q {
                let kv = h / heads_per_group as usize;
                for pos in 0..seq_len {
                    let base = layer_offset as usize + (kv * max_seq + pos) * head_dim;
                    let want: f64 = (0..head_dim)
                        .map(|d| {
                            f64::from(q[h * head_dim + d]) * f64::from(k_half[base + d].to_f32())
                        })
                        .sum::<f64>()
                        * f64::from(inv);
                    let g = f64::from(got[h * max_seq + pos]);
                    assert!(
                        (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "head_dim {head_dim} head {h} pos {pos}: gpu {g} vs cpu {want}"
                    );
                }
            }
        }
    }

    /// The full-rotation `fused_qk_rope` keeps its historical `fma` pair
    /// arithmetic bit for bit.
    #[test]
    fn full_rotation_rope_keeps_its_fma_pair_arithmetic() {
        let Some(graph) = graph() else {
            return;
        };
        let device = &graph.device;
        let (nq, nkv, head_dim) = (3usize, 2usize, 128usize);
        let half = head_dim / 2;
        let mut rng = Rng::new(7);
        let q = rng.vec(nq * head_dim, 3.0);
        let k = rng.vec(nkv * head_dim, 3.0);
        let cos = rng.vec(half, 1.0);
        let sin = rng.vec(half, 1.0);
        let (qb, kb, cb, sb) = (
            shared_f32(device, &q),
            shared_f32(device, &k),
            shared_f32(device, &cos),
            shared_f32(device, &sin),
        );
        let qo = shared_zeroed(device, nq * head_dim * 4);
        let ko = shared_zeroed(device, nkv * head_dim * 4);
        run(&graph, |enc| {
            enc.set_compute_pipeline_state(&graph.pipelines.fused_qk_rope);
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&qo), 0);
            enc.set_buffer(3, Some(&ko), 0);
            enc.set_buffer(4, Some(&cb), 0);
            enc.set_buffer(5, Some(&sb), 0);
            set_u32(enc, 6, nq as u32);
            set_u32(enc, 7, nkv as u32);
            set_u32(enc, 8, half as u32);
            enc.dispatch_thread_groups(
                MTLSize::new(half.div_ceil(64) as u64, (nq + nkv) as u64, 1),
                MTLSize::new(64, 1, 1),
            );
        });
        for (input, out, heads) in [(&q, &qo, nq), (&k, &ko, nkv)] {
            let got = read_f32(out, heads * head_dim);
            for h in 0..heads {
                for d in 0..half {
                    let x0 = input[h * head_dim + d];
                    let x1 = input[h * head_dim + d + half];
                    let lo = x0.mul_add(cos[d], -(x1 * sin[d]));
                    let hi = x0.mul_add(sin[d], x1 * cos[d]);
                    assert_eq!(got[h * head_dim + d].to_bits(), lo.to_bits());
                    assert_eq!(got[h * head_dim + d + half].to_bits(), hi.to_bits());
                }
            }
        }
    }

    /// MET-M2: the partial entry point is bitwise the CPU's
    /// `rope_partial_splithalf_simd` on Bonsai 2's geometry (`n_rot` 64 of
    /// 256, Q heads read through the `[q | gate]` interleave), with the tail
    /// copied verbatim.
    #[test]
    fn partial_rope_is_bitwise_the_cpu_split_half_rotation() {
        let Some(graph) = graph() else {
            return;
        };
        let pso = graph
            .pipeline_for("fused_qk_rope_partial")
            .expect("fused_qk_rope_partial is in the combined library");
        let device = &graph.device;
        for (nq, nkv, head_dim, n_rot) in [(24usize, 4usize, 256usize, 64usize), (4, 2, 64, 16)] {
            let half = n_rot / 2;
            let mut rng = Rng::new((head_dim * 31 + n_rot) as u64);
            // Q is interleaved `[q | gate]` per head: stride 2 * head_dim.
            let q_all = rng.vec(nq * head_dim * 2, 4.0);
            let k = rng.vec(nkv * head_dim, 4.0);
            let cos = rng.vec(half, 1.0);
            let sin = rng.vec(half, 1.0);
            let (qb, kb, cb, sb) = (
                shared_f32(device, &q_all),
                shared_f32(device, &k),
                shared_f32(device, &cos),
                shared_f32(device, &sin),
            );
            let qo = shared_zeroed(device, nq * head_dim * 4);
            let ko = shared_zeroed(device, nkv * head_dim * 4);
            run(&graph, |enc| {
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(&kb), 0);
                enc.set_buffer(2, Some(&qo), 0);
                enc.set_buffer(3, Some(&ko), 0);
                enc.set_buffer(4, Some(&cb), 0);
                enc.set_buffer(5, Some(&sb), 0);
                set_u32(enc, 6, nq as u32);
                set_u32(enc, 7, nkv as u32);
                set_u32(enc, 8, head_dim as u32);
                set_u32(enc, 9, n_rot as u32);
                set_u32(enc, 10, (2 * head_dim) as u32);
                enc.dispatch_thread_groups(
                    MTLSize::new(head_dim.div_ceil(64) as u64, (nq + nkv) as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            });
            let got_q = read_f32(&qo, nq * head_dim);
            let got_k = read_f32(&ko, nkv * head_dim);
            let mut want = vec![0.0f32; head_dim];
            for h in 0..nq {
                let src = &q_all[h * 2 * head_dim..h * 2 * head_dim + head_dim];
                crate::rope_mrope::rope_partial_splithalf_simd(
                    src, &mut want, head_dim, n_rot, &cos, &sin,
                )
                .expect("cpu rope");
                check_rope_head(
                    &got_q[h * head_dim..(h + 1) * head_dim],
                    &want,
                    src,
                    &cos,
                    &sin,
                );
            }
            for h in 0..nkv {
                let src = &k[h * head_dim..(h + 1) * head_dim];
                crate::rope_mrope::rope_partial_splithalf_simd(
                    src, &mut want, head_dim, n_rot, &cos, &sin,
                )
                .expect("cpu rope");
                check_rope_head(
                    &got_k[h * head_dim..(h + 1) * head_dim],
                    &want,
                    src,
                    &cos,
                    &sin,
                );
            }
        }
    }

    /// `got` must equal the non-fused scalar rotation bit for bit (the
    /// arithmetic the NEON path performs), equal the CPU dispatcher's output
    /// bit for bit on aarch64, and carry the tail through unchanged.
    fn check_rope_head(got: &[f32], cpu: &[f32], src: &[f32], cos: &[f32], sin: &[f32]) {
        let half = cos.len();
        let n_rot = 2 * half;
        for d in 0..half {
            let (x0, x1) = (src[d], src[d + half]);
            let lo = (x0 * cos[d]) - (x1 * sin[d]);
            let hi = (x1 * cos[d]) + (x0 * sin[d]);
            assert_eq!(got[d].to_bits(), lo.to_bits(), "rotated lo {d}");
            assert_eq!(got[d + half].to_bits(), hi.to_bits(), "rotated hi {d}");
        }
        for d in n_rot..got.len() {
            assert_eq!(
                got[d].to_bits(),
                src[d].to_bits(),
                "tail {d} must be copied"
            );
        }
        if cfg!(target_arch = "aarch64") {
            for (i, (g, c)) in got.iter().zip(cpu).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    c.to_bits(),
                    "vs rope_partial_splithalf_simd at {i}"
                );
            }
        }
    }

    /// The partial norm + rotation kernel matches the CPU's per-head RMSNorm
    /// followed by `rope_partial_splithalf_simd`, over several tokens with
    /// per-token angles and the `[q | gate]` interleave.
    #[test]
    fn partial_norm_rope_matches_the_cpu_norm_then_rotation() {
        let Some(graph) = graph() else {
            return;
        };
        let pso = graph
            .pipeline_for("fused_qk_norm_rope_partial")
            .expect("fused_qk_norm_rope_partial is in the combined library");
        let device = &graph.device;
        let (nq, nkv, head_dim, n_rot, t_len) = (24usize, 4usize, 256usize, 64usize, 3usize);
        let half = n_rot / 2;
        let eps = 1e-6f32;
        let mut rng = Rng::new(99);
        let q_all = rng.vec(t_len * nq * head_dim * 2, 3.0);
        let k = rng.vec(t_len * nkv * head_dim, 3.0);
        let qw = rng.vec(head_dim, 1.5);
        let kw = rng.vec(head_dim, 1.5);
        let cos = rng.vec(t_len * half, 1.0);
        let sin = rng.vec(t_len * half, 1.0);
        let bufs: Vec<Buffer> = [&q_all, &k, &qw, &kw, &cos, &sin]
            .iter()
            .map(|v| shared_f32(device, v))
            .collect();
        let qo = shared_zeroed(device, t_len * nq * head_dim * 4);
        let ko = shared_zeroed(device, t_len * nkv * head_dim * 4);
        run(&graph, |enc| {
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&bufs[0]), 0);
            enc.set_buffer(1, Some(&bufs[1]), 0);
            enc.set_buffer(2, Some(&qo), 0);
            enc.set_buffer(3, Some(&ko), 0);
            enc.set_buffer(4, Some(&bufs[2]), 0);
            enc.set_buffer(5, Some(&bufs[3]), 0);
            enc.set_buffer(6, Some(&bufs[4]), 0);
            enc.set_buffer(7, Some(&bufs[5]), 0);
            set_u32(enc, 8, nq as u32);
            set_u32(enc, 9, nkv as u32);
            set_u32(enc, 10, head_dim as u32);
            set_f32(enc, 11, eps);
            set_u32(enc, 12, n_rot as u32);
            set_u32(enc, 13, (2 * head_dim) as u32);
            set_u32(enc, 14, (2 * nq * head_dim) as u32);
            set_u32(enc, 15, (nkv * head_dim) as u32);
            enc.dispatch_thread_groups(
                MTLSize::new((nq + nkv) as u64, t_len as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        let got_q = read_f32(&qo, t_len * nq * head_dim);
        let got_k = read_f32(&ko, t_len * nkv * head_dim);
        let mut normed = vec![0.0f32; head_dim];
        let mut want = vec![0.0f32; head_dim];
        let mut worst = 0.0f32;
        for t in 0..t_len {
            let (c, s) = (
                &cos[t * half..(t + 1) * half],
                &sin[t * half..(t + 1) * half],
            );
            for h in 0..nq {
                let lo = t * 2 * nq * head_dim + h * 2 * head_dim;
                crate::rms_norm_simd(&q_all[lo..lo + head_dim], &qw, &mut normed, eps)
                    .expect("cpu norm");
                crate::rope_mrope::rope_partial_splithalf_simd(
                    &normed, &mut want, head_dim, n_rot, c, s,
                )
                .expect("cpu rope");
                let g = &got_q[(t * nq + h) * head_dim..(t * nq + h + 1) * head_dim];
                for (a, b) in g.iter().zip(&want) {
                    worst = worst.max((a - b).abs() / b.abs().max(1.0));
                }
            }
            for h in 0..nkv {
                let lo = (t * nkv + h) * head_dim;
                crate::rms_norm_simd(&k[lo..lo + head_dim], &kw, &mut normed, eps)
                    .expect("cpu norm");
                crate::rope_mrope::rope_partial_splithalf_simd(
                    &normed, &mut want, head_dim, n_rot, c, s,
                )
                .expect("cpu rope");
                let g = &got_k[lo..lo + head_dim];
                for (a, b) in g.iter().zip(&want) {
                    worst = worst.max((a - b).abs() / b.abs().max(1.0));
                }
            }
        }
        assert!(
            worst < 2e-6,
            "partial norm+rope worst relative error {worst:e}"
        );
    }

    /// The full-rotation `fused_qk_norm_rope` entry point computes exactly
    /// what its pre-`n_rot` text computed: the historical kernel, compiled
    /// on its own, and this constant, compiled the same way, agree bit for
    /// bit.
    #[test]
    fn full_rotation_norm_rope_is_bit_identical_to_the_historical_kernel() {
        const HISTORICAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_qk_norm_rope_historical(
    device const float* q_in     [[buffer(0)]],
    device const float* k_in     [[buffer(1)]],
    device float* q_out          [[buffer(2)]],
    device float* k_out          [[buffer(3)]],
    device const float* q_weight [[buffer(4)]],
    device const float* k_weight [[buffer(5)]],
    device const float* cos_buf  [[buffer(6)]],
    device const float* sin_buf  [[buffer(7)]],
    constant uint& nq            [[buffer(8)]],
    constant uint& nkv           [[buffer(9)]],
    constant uint& head_dim      [[buffer(10)]],
    constant float& eps          [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const bool is_q = (gid < nq);
    const uint head_idx = is_q ? gid : (gid - nq);
    const uint half_dim = head_dim / 2u;

    device const float* in_ptr = is_q ? (q_in + head_idx * head_dim) : (k_in + head_idx * head_dim);
    device float* out_ptr      = is_q ? (q_out + head_idx * head_dim) : (k_out + head_idx * head_dim);
    device const float* w_ptr  = is_q ? q_weight : k_weight;

    threadgroup float shared_sum[256];
    float local_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(head_dim) + eps);

    for (uint d = tid; d < half_dim; d += tpg) {
        float normed_lo = in_ptr[d] * rms_inv * w_ptr[d];
        float normed_hi = in_ptr[d + half_dim] * rms_inv * w_ptr[d + half_dim];

        float c = cos_buf[d];
        float s = sin_buf[d];
        out_ptr[d]            = fma(normed_lo, c, -(normed_hi * s));
        out_ptr[d + half_dim] = fma(normed_lo, s,   normed_hi * c);
    }
}
"#;
        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let historical = standalone(&device, HISTORICAL, "fused_qk_norm_rope_historical");
        let current = standalone(&device, MSL_FUSED_QK_NORM_ROPE, "fused_qk_norm_rope");
        for (nq, nkv, head_dim) in [(16usize, 8usize, 128usize), (4, 2, 64)] {
            let mut rng = Rng::new(head_dim as u64 + 5);
            let inputs = [
                rng.vec(nq * head_dim, 3.0),
                rng.vec(nkv * head_dim, 3.0),
                rng.vec(head_dim, 1.5),
                rng.vec(head_dim, 1.5),
                rng.vec(head_dim / 2, 1.0),
                rng.vec(head_dim / 2, 1.0),
            ];
            let bufs: Vec<Buffer> = inputs.iter().map(|v| shared_f32(&device, v)).collect();
            let mut outs = Vec::new();
            for pso in [&historical, &current] {
                let qo = shared_zeroed(&device, nq * head_dim * 4);
                let ko = shared_zeroed(&device, nkv * head_dim * 4);
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&bufs[0]), 0);
                enc.set_buffer(1, Some(&bufs[1]), 0);
                enc.set_buffer(2, Some(&qo), 0);
                enc.set_buffer(3, Some(&ko), 0);
                enc.set_buffer(4, Some(&bufs[2]), 0);
                enc.set_buffer(5, Some(&bufs[3]), 0);
                enc.set_buffer(6, Some(&bufs[4]), 0);
                enc.set_buffer(7, Some(&bufs[5]), 0);
                set_u32(enc, 8, nq as u32);
                set_u32(enc, 9, nkv as u32);
                set_u32(enc, 10, head_dim as u32);
                set_f32(enc, 11, 1e-6);
                enc.dispatch_thread_groups(
                    MTLSize::new((nq + nkv) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let mut all = read_f32(&qo, nq * head_dim);
                all.extend(read_f32(&ko, nkv * head_dim));
                outs.push(all);
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(
                bits(&outs[0]),
                bits(&outs[1]),
                "head_dim {head_dim}: the full-rotation entry point changed its arithmetic"
            );
        }
    }

    /// MET-07: every KV offset binding is a 64-bit `ulong`.
    #[test]
    fn kv_layer_offsets_are_bound_as_ulong() {
        for (src, needle) in [
            (MSL_FUSED_KV_STORE, "constant ulong& layer_offset"),
            (
                MSL_BATCHED_ATTENTION_SCORES,
                "constant ulong& cache_layer_offset",
            ),
            (
                MSL_BATCHED_ATTENTION_SCORES_V2,
                "constant ulong& cache_layer_offset",
            ),
            (
                MSL_BATCHED_ATTENTION_WEIGHTED_SUM,
                "constant ulong& cache_layer_offset",
            ),
        ] {
            assert!(src.contains(needle), "missing `{needle}`");
            assert!(!src.contains("constant uint& layer_offset"));
            assert!(!src.contains("constant uint& cache_layer_offset"));
        }
    }
}
