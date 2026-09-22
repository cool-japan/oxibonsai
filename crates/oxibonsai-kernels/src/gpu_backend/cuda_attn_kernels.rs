//! CUDA C kernel source strings for OxiBonsai attention operations.
//!
//! # Attention kernel inventory
//!
//! | Kernel                      | Grid                                     | Block      | Notes                                           |
//! |-----------------------------|------------------------------------------|------------|-------------------------------------------------|
//! | `fused_qk_norm`             | `(nq+nkv, 1, 1)`                         | `(256,1,1)`| RMSNorm both Q and K heads in one dispatch      |
//! | `fused_qk_rope`             | `(ceil(half_dim/64), nq+nkv, 1)`         | `(64,1,1)` | Rotary position embedding for Q and K           |
//! | `fused_qk_norm_rope`        | `(nq+nkv, 1, 1)`                         | `(256,1,1)`| Combined RMSNorm + RoPE for Q and K             |
//! | `fused_kv_store`            | `(ceil(head_dim/64), nkv, 1)`            | `(64,1,1)` | Store K (after RoPE) and V into FP16 KV cache   |
//! | `batched_attn_scores_v2`    | `(n_q, ceil(seq_len/batch_stride), 1)`   | `(128,1,1)`| Batched dot-product attention scores (dynamic shared memory — see F3) |

//! | `batched_softmax`           | `(n_q, 1, 1)`                            | `(256,1,1)`| Per-head numerically-stable softmax             |
//! | `batched_attn_weighted_sum` | `(ceil(head_dim/64), n_q, 1)`            | `(64,1,1)` | Weighted sum of V vectors by softmax weights    |
//!
//! # FP16 KV cache
//!
//! K and V are stored in FP16 (u16 on the Rust side) to halve VRAM usage.
//! `fused_kv_store` writes FP16; the score and weighted-sum kernels read FP16.
//!
//! The layer base offset into that cache is an `unsigned long long` in all
//! three kernels that take one (finding **F4**). It used to be an
//! `unsigned int` that the Rust side had *already* truncated with `as u32`, so
//! a cache above `2^32` elements aliased two layers onto the same addresses
//! with no error. `CudaKvCache::layer_offset_elements` now returns `u64` and
//! the kernels consume it at full width.

//!
//! # Weight layout
//!
//! Q/K/V/O/gate+up/down weight blocks use the Q1_0_G128 SoA layout identical
//! to the FFN kernels in `cuda_kernels.rs`.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// Combined CUDA C source for all 7 attention kernels used in the decode path.
///
/// Compiled once at process startup (lazily on first `encode_attn_phase` call)
/// via `cudarc::nvrtc::compile_ptx`.  All kernels share a single PTX module.
pub const CUDA_ATTENTION_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA attention kernels — decode path
   All math intrinsics (sqrtf, expf, fmaxf, rsqrtf) are NVRTC built-ins;
   no CUDA SDK headers are required or included.
   ========================================================================= */

/* ── FP16 ↔ FP32 helpers ─────────────────────────────────────────────────── */
static __device__ __forceinline__ float fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

static __device__ __forceinline__ unsigned short float_to_fp16(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

/* =========================================================================
   Kernel 1 — fused_qk_norm
   Fused RMSNorm for Q and K heads.

   Grid:  (nq + nkv, 1, 1)
   Block: (256, 1, 1)
   First nq CTAs normalise Q heads; remaining nkv CTAs normalise K heads.
   Each CTA uses shared-memory parallel reduction for sum-of-squares.
   ========================================================================= */
extern "C" __global__ void fused_qk_norm(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    float*       __restrict__ q_out,
    float*       __restrict__ k_out,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    unsigned int nq,
    unsigned int nkv,
    unsigned int head_dim,
    float        eps
) {
    const unsigned int gid = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tpg = blockDim.x;

    const bool is_q = (gid < nq);
    const unsigned int head_idx = is_q ? gid : (gid - nq);

    const float* __restrict__ in_ptr  = is_q ? (q_in + (unsigned long long)head_idx * head_dim)
                                              : (k_in + (unsigned long long)head_idx * head_dim);
    float*       __restrict__ out_ptr = is_q ? (q_out + (unsigned long long)head_idx * head_dim)
                                              : (k_out + (unsigned long long)head_idx * head_dim);
    const float* __restrict__ w_ptr   = is_q ? q_weight : k_weight;

    __shared__ float shared_sum[256];
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    __syncthreads();

    for (unsigned int stride = tpg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared_sum[0] / (float)head_dim + eps);

    for (unsigned int i = tid; i < head_dim; i += tpg) {
        out_ptr[i] = in_ptr[i] * rms_inv * w_ptr[i];
    }
}

/* =========================================================================
   Kernel 2 — fused_qk_rope
   Apply rotary position embedding to Q and K.

   Grid:  (ceil(half_dim/64), nq + nkv, 1)
   Block: (64, 1, 1)
   blockIdx.y < nq  →  apply to Q head blockIdx.y
   blockIdx.y >= nq →  apply to K head (blockIdx.y - nq)
   ========================================================================= */
extern "C" __global__ void fused_qk_rope(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    float*       __restrict__ q_out,
    float*       __restrict__ k_out,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    unsigned int nq,
    unsigned int nkv,
    unsigned int half_dim
) {
    const unsigned int d       = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head_y  = blockIdx.y;
    if (d >= half_dim) return;

    const bool is_q = (head_y < nq);
    const unsigned int head_idx = is_q ? head_y : (head_y - nq);
    const unsigned int head_dim = half_dim * 2u;

    const float* __restrict__ in_ptr  = is_q ? (q_in + (unsigned long long)head_idx * head_dim)
                                              : (k_in + (unsigned long long)head_idx * head_dim);
    float*       __restrict__ out_ptr = is_q ? (q_out + (unsigned long long)head_idx * head_dim)
                                              : (k_out + (unsigned long long)head_idx * head_dim);

    float c  = cos_buf[d];
    float s  = sin_buf[d];
    float x0 = in_ptr[d];
    float x1 = in_ptr[d + half_dim];
    out_ptr[d]          = x0 * c - x1 * s;
    out_ptr[d + half_dim] = x0 * s + x1 * c;
}

/* =========================================================================
   Kernel 3 — fused_qk_norm_rope
   Combined RMSNorm + RoPE for Q and K in a single dispatch.

   Grid:  (nq + nkv, 1, 1)
   Block: (256, 1, 1)
   Each CTA processes one Q or K head:
     Phase 1: parallel sum-of-squares → rms_inv
     Phase 2: normalise each element pair and apply RoPE in-place
   ========================================================================= */
extern "C" __global__ void fused_qk_norm_rope(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    float*       __restrict__ q_out,
    float*       __restrict__ k_out,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_buf,
    const float* __restrict__ sin_buf,
    unsigned int nq,
    unsigned int nkv,
    unsigned int head_dim,
    float        eps
) {
    const unsigned int gid = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tpg = blockDim.x;

    const bool is_q = (gid < nq);
    const unsigned int head_idx = is_q ? gid : (gid - nq);
    const unsigned int half_dim = head_dim >> 1u;

    const float* __restrict__ in_ptr  = is_q ? (q_in + (unsigned long long)head_idx * head_dim)
                                              : (k_in + (unsigned long long)head_idx * head_dim);
    float*       __restrict__ out_ptr = is_q ? (q_out + (unsigned long long)head_idx * head_dim)
                                              : (k_out + (unsigned long long)head_idx * head_dim);
    const float* __restrict__ w_ptr   = is_q ? q_weight : k_weight;

    /* Phase 1: sum of squares */
    __shared__ float shared_sum[256];
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += tpg) {
        float v = in_ptr[i];
        local_sq += v * v;
    }
    shared_sum[tid] = local_sq;
    __syncthreads();

    for (unsigned int stride = tpg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared_sum[0] / (float)head_dim + eps);

    /* Phase 2: normalise + RoPE in one pass over [0, half_dim) */
    for (unsigned int d = tid; d < half_dim; d += tpg) {
        float normed_lo = in_ptr[d]          * rms_inv * w_ptr[d];
        float normed_hi = in_ptr[d + half_dim] * rms_inv * w_ptr[d + half_dim];

        float c = cos_buf[d];
        float s = sin_buf[d];
        out_ptr[d]          = normed_lo * c - normed_hi * s;
        out_ptr[d + half_dim] = normed_lo * s + normed_hi * c;
    }
}

/* =========================================================================
   Kernel 4 — fused_kv_store
   Store K (post-RoPE) and V into the FP16 KV cache in a single dispatch.

   Grid:  (ceil(head_dim/64), nkv, 1)
   Block: (64, 1, 1)
   Each thread stores one element of both K and V for one head.

   k_cache / v_cache layout: [n_layers × nkv × max_seq × head_dim]  (FP16)
   dst_offset = layer_offset + (head * max_seq + pos) * head_dim + d
   ========================================================================= */
extern "C" __global__ void fused_kv_store(
    const float*          __restrict__ k_data,
    const float*          __restrict__ v_data,
    unsigned short*       __restrict__ k_cache,
    unsigned short*       __restrict__ v_cache,
    unsigned int head_dim,
    unsigned int nkv,
    unsigned int max_seq,
    const unsigned int* __restrict__ d_pos_seqlen,
    unsigned long long layer_offset
) {
    const unsigned int pos  = d_pos_seqlen[0];
    const unsigned int d    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head = blockIdx.y;
    if (d >= head_dim || head >= nkv) return;

    const unsigned long long src_off = (unsigned long long)head * head_dim + d;
    const unsigned long long dst_off = layer_offset
                                     + ((unsigned long long)head * max_seq + pos) * head_dim
                                     + d;


    k_cache[dst_off] = float_to_fp16(k_data[src_off]);
    v_cache[dst_off] = float_to_fp16(v_data[src_off]);
}

/* =========================================================================
   Kernel 5 — batched_attn_scores_v2
   Batched dot-product attention scores with Q-vector shared-memory caching.

   Grid:  (n_q, ceil(seq_len / batch_stride), 1)
   Block: (128, 1, 1)

   Each CTA handles one Q-head and processes `batch_stride` positions.
   The Q vector is staged in shared memory once and reused for each position.

   F3 — head_dim > 128.  This kernel used to declare `__shared__ float
   shared_q[128]` and stage/multiply exactly one element per thread
   (`if (tid < head_dim) ...`), which silently truncated the QK dot product to
   `min(head_dim, 128)`.  Bonsai 2 27B has `head_dim = 256`
   (`qwen35.attention.key_length`), so every score was computed from half the
   vector — wrong numbers, no error.  Three coupled changes fix it:

     (a) the Q staging area and the cross-warp partials both live in ONE
         dynamically-sized shared block, laid out as
         `[head_dim floats | (blockDim.x / 32) floats]`.  The launcher
         (`cuda_full_layer/launchers.rs`) is the single place that sizes it,
         via `cuda_device_negotiation::attn_scores_shared_bytes`;
     (b) staging AND accumulation are strided loops over `head_dim`, so one
         thread covers several dimensions when `head_dim > blockDim.x`;
     (c) the cross-warp reduction sums `blockDim.x / 32` partials instead of
         the hardcoded four.  Simply raising the block to 256 threads — the
         "obvious" fix — would have written `warp_sums[4..7]` out of bounds
         AND still dropped half the partials.

   No cap is baked into the kernel: the reachable `head_dim` is whatever
   `attn_scores_shared_bytes` will size, and the launcher rejects the rest
   with a named error rather than corrupting shared memory.
   ========================================================================= */
extern "C" __global__ void batched_attn_scores_v2(
    const float*          __restrict__ queries,
    const unsigned short* __restrict__ k_cache,
    float*                __restrict__ all_scores,
    unsigned int head_dim,
    unsigned int n_q,
    unsigned int n_kv,
    unsigned int heads_per_group,
    unsigned int max_seq,
    const unsigned int* __restrict__ d_pos_seqlen,
    float        inv_sqrt_hd,
    unsigned long long cache_layer_offset,
    unsigned int batch_stride
) {
    const unsigned int seq_len  = d_pos_seqlen[1];
    const unsigned int q_head   = blockIdx.x;
    const unsigned int batch_id = blockIdx.y;
    const unsigned int tid      = threadIdx.x;
    if (q_head >= n_q) return;

    const unsigned int kv_head  = q_head / heads_per_group;
    const unsigned int pos_start = batch_id * batch_stride;
    const unsigned int n_warps   = blockDim.x >> 5u;

    /* One dynamic shared block, split by hand (F3a):
         shared_q  : head_dim floats
         warp_sums : n_warps floats                                        */
    /* No __restrict__ on these two: they are derived from one allocation, and
       promising the compiler they never alias would be a claim about the
       layout rather than about the accesses. */
    extern __shared__ float attn_smem[];
    float* shared_q  = attn_smem;
    float* warp_sums = attn_smem + head_dim;

    /* Stage Q — strided, so head_dim may exceed blockDim.x (F3b). */
    for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
        shared_q[i] = queries[(unsigned long long)q_head * head_dim + i];
    }
    __syncthreads();

    const unsigned int warp_id = tid >> 5u;
    const unsigned int lane    = tid & 31u;

    for (unsigned int pos_t = pos_start;
         pos_t < pos_start + batch_stride && pos_t < seq_len;
         pos_t++)
    {
        const unsigned short* __restrict__ key =
            k_cache + cache_layer_offset
            + ((unsigned long long)kv_head * max_seq + pos_t) * head_dim;

        /* Strided accumulation over the FULL head_dim (F3b). */
        float my_prod = 0.0f;
        for (unsigned int i = tid; i < head_dim; i += blockDim.x) {
            my_prod = fmaf(shared_q[i], fast_fp16_to_float(key[i]), my_prod);
        }

        /* Warp-level reduction using __shfl_down_sync.  Every thread of the
           block reaches this point (the only early return is uniform over the
           block), so the full 0xffffffff mask is correct. */
        unsigned int mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            my_prod += __shfl_down_sync(mask, my_prod, offset);
        }

        /* Cross-warp reduction over blockDim.x/32 warps (F3c). */
        if (lane == 0u) {
            warp_sums[warp_id] = my_prod;
        }
        __syncthreads();

        if (tid == 0u) {
            float total = 0.0f;
            for (unsigned int w = 0u; w < n_warps; ++w) {
                total += warp_sums[w];
            }
            all_scores[(unsigned long long)q_head * max_seq + pos_t] = total * inv_sqrt_hd;
        }
        __syncthreads();
    }
}


/* =========================================================================
   Kernel 6 — batched_softmax
   Per-head numerically-stable softmax (in-place).

   Grid:  (n_q, 1, 1)
   Block: (256, 1, 1)

   Three-pass approach:
     Pass 1: find max (parallel reduction)
     Pass 2: compute exp(x − max) and sum
     Pass 3: normalise by sum
   ========================================================================= */
extern "C" __global__ void batched_softmax(
    float*       __restrict__ all_scores,
    unsigned int n_q,
    unsigned int max_seq,
    const unsigned int* __restrict__ d_pos_seqlen
) {
    const unsigned int seq_len = d_pos_seqlen[1];
    const unsigned int tgpig  = blockIdx.x;
    const unsigned int tid    = threadIdx.x;
    const unsigned int tg_size = blockDim.x;
    if (tgpig >= n_q) return;

    float* __restrict__ scores = all_scores + (unsigned long long)tgpig * max_seq;

    __shared__ float shared[256];

    /* Pass 1: max */
    float local_max = -3.402823e+38f;
    for (unsigned int i = tid; i < seq_len; i += tg_size) {
        float v = scores[i];
        if (v > local_max) local_max = v;
    }
    shared[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }
    float gmax = shared[0];
    __syncthreads();

    /* Pass 2: exp + sum */
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < seq_len; i += tg_size) {
        float e = expf(scores[i] - gmax);
        scores[i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float gsum = shared[0];
    __syncthreads();

    /* Pass 3: normalise */
    float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    for (unsigned int i = tid; i < seq_len; i += tg_size) {
        scores[i] *= inv_sum;
    }
}

/* =========================================================================
   Kernel 7 — batched_attn_weighted_sum
   Weighted sum of V vectors: output[d] = Σ_t scores[t] × V[t][d].

   Grid:  (ceil(head_dim/64), n_q, 1)
   Block: (64, 1, 1)

   Each thread accumulates one output dimension d for one Q head.
   GQA mapping: kv_head = q_head / heads_per_group.
   V is stored in FP16 in the KV cache.
   ========================================================================= */
extern "C" __global__ void batched_attn_weighted_sum(
    const float*          __restrict__ all_scores,
    const unsigned short* __restrict__ v_cache,
    float*                __restrict__ attn_out,
    unsigned int head_dim,
    unsigned int n_q,
    unsigned int n_kv,
    unsigned int heads_per_group,
    unsigned int max_seq,
    const unsigned int* __restrict__ d_pos_seqlen,
    unsigned long long cache_layer_offset
) {
    const unsigned int seq_len = d_pos_seqlen[1];
    const unsigned int d      = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int q_head = blockIdx.y;
    if (d >= head_dim || q_head >= n_q) return;

    const unsigned int kv_head = q_head / heads_per_group;
    const float* __restrict__ scores =
        all_scores + (unsigned long long)q_head * max_seq;
    const unsigned short* __restrict__ values =
        v_cache + cache_layer_offset
        + (unsigned long long)kv_head * max_seq * head_dim;


    float acc = 0.0f;
    for (unsigned int t = 0u; t < seq_len; t++) {
        acc += scores[t] * fast_fp16_to_float(values[(unsigned long long)t * head_dim + d]);
    }
    attn_out[(unsigned long long)q_head * head_dim + d] = acc;
}
"#;
