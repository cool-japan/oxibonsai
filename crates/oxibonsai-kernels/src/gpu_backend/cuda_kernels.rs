//! CUDA C kernel source strings for OxiBonsai Q1_0_G128 inference.
//!
//! # Kernel versions
//!
//! | Kernel                      | Shared mem  | Notes |
//! |-----------------------------|-------------|-------|
//! | `gemv_q1_g128_v7`           | none        | V7 baseline; V7 + hardware fp16 |
//! | `gemv_q1_g128_v7_residual`  | none        | V7 + fused residual add |
//! | `gemv_q1_g128_v8`           | k×4 + pad   | V8: shared-mem input cache, coalesced access |
//! | `gemv_q1_g128_v8_residual`  | k×4 + pad   | V8 + fused residual add |
//! | `gemv_q1_g128_v9`           | none        | V9: 128-bit PTX vector load + `__ldg()` scale |
//! | `gemv_q1_g128_v9_residual`  | none        | V9 + fused residual add |
//! | `rmsnorm_weighted_v2`       | 8×4 bytes   | Two-phase warp-shuffle RMSNorm |
//! | `residual_add`              | none        | In-place a += b |
//! | `swiglu_fused`              | none        | SiLU(gate) × up |
//! | `fused_gate_up_swiglu_q1`   | none        | Fused gate+up GEMV + SwiGLU in epilogue |
//! | `argmax_f32`                | 2×256×4     | Single-block argmax |
//!
//! # V8 design — shared-memory padded input cache
//!
//! V7 non-coalesced read: lane `l` processes block `b=l`; at inner-loop
//! iteration `bit`, it reads `input[b*128 + bit]`.  Across a warp the 32
//! lanes read stride-128 addresses — one cache line miss per lane per bit.
//!
//! V8 fix: the 256-thread block cooperatively loads the input vector into
//! `__shared__` before compute.  The shared array uses a stride of 129
//! (= 128 + 1 padding float) so that lane `l` accesses bank `(l + bit) % 32` —
//! all 32 banks occupied, zero bank conflicts.
//!
//! # Weight layout (SoA)
//!
//! ```text
//! [scales: n_rows × n_blocks × 2 bytes FP16]
//! [data:   n_rows × n_blocks × 16 bytes (128 sign bits)]
//! ```

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// Combined CUDA C source for all V7/V8-quality OxiBonsai kernels.
///
/// Compiled once at process startup via `cudarc::nvrtc::compile_ptx`.
pub const CUDA_V7_KERNELS_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA kernels — V8 quality, SoA weight layout
   No CUDA SDK headers required; all intrinsics are NVRTC built-ins.
   ========================================================================= */

/* ── Hardware FP16 → FP32 via PTX (1 instruction on all SM6.0+ GPUs) ───── */
static __device__ __forceinline__ float fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── Software fallback (used by V7 tests / pre-SM6.0 paths) ──────────────── */
static __device__ __forceinline__ float fp16_to_float(unsigned short h) {
    return fast_fp16_to_float(h);
}

/* ── SiLU activation: x · σ(x) ─────────────────────────────────────────── */
static __device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ── 16-byte-aligned quant-section loads (finding F10) ───────────────────────
   `ld.global.nc.v4.u32` REQUIRES a 16-byte-aligned address; a misaligned one
   raises CUDA_ERROR_MISALIGNED_ADDRESS, which kills the whole context.

   In the SoA weight layout the quant section starts immediately after the FP16
   scales, at byte `scales_bytes = total_blocks * 2`, and each block occupies a
   whole number of 16-byte units (16 B for `Q1_0_g128`, 32 B for
   `TQ2_0_g128`).  The buffer base is a fresh `cuMemAlloc` (256-byte aligned,
   never a sub-slice), so alignment of every quant load reduces to

       scales_bytes % 16 == 0   <=>   total_blocks % 8 == 0.

   The previous in-source comment argued only `% 2` — i.e. 4-byte alignment —
   which proves nothing about the 16-byte requirement these instructions carry.
   It happens to hold for every current Qwen3 tensor, so the bug is latent, but
   any weight shape whose `n_rows * (k / 128)` is not a multiple of 8 aborted.

   Rather than change the on-device layout (the TQ2 twins in
   `cuda_prefill_kernels.rs` recompute the quant offset independently, so a
   padded layout would silently desynchronise them), the vector load is guarded
   and falls back to four scalar `__ldg` reads.  The predicate depends only on
   kernel arguments, so the branch is uniform across the whole grid and costs no
   divergence; `cuda_device_negotiation::soa_qs_vector_load_ok` is its host-side
   twin and is unit-tested on every host.

   NOTE for reviewers: `scripts/check_cuda.sh` removes every inline-PTX
   statement before its tier-2 (clang++ `-fsyntax-only`) pass, so
   in that pass the aligned branch of `soa_load4_u32` has an empty body and
   its four outputs are left uninitialised. A clean tier-2 result therefore
   proves less than usual for THIS helper; only a real `nvcc --cuda` (tier 1)
   pass, on a box with the CUDA toolkit, validates the guarded vector load. */

static __device__ __forceinline__ unsigned int soa_quant_aligned16(
    unsigned long long total_blocks)
{
    return (((total_blocks * 2ull) & 15ull) == 0ull) ? 1u : 0u;
}

static __device__ __forceinline__ void soa_load4_u32(
    const unsigned int* p,
    unsigned int aligned16,
    unsigned int &w0, unsigned int &w1, unsigned int &w2, unsigned int &w3)
{
    if (aligned16) {
        asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
            : "l"((unsigned long long)p));
    } else {
        w0 = __ldg(p + 0u);
        w1 = __ldg(p + 1u);
        w2 = __ldg(p + 2u);
        w3 = __ldg(p + 3u);
    }
}


/* =========================================================================
   Kernel 1 — gemv_q1_g128_v7
   Q1_0_G128 matrix-vector product (SoA, 8 rows per block, hardware fp16).

   Grid:  (⌈n_rows/8⌉, 1, 1)   — 8 output rows per CTA
   Block: (256, 1, 1)           — 8 warps × 32 lanes
   Each lane processes blocks:  lane, lane+32, lane+64, …
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v7(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int n_blocks = k >> 7u;

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int  g     = row * n_blocks + b;
        const float         scale = fast_fp16_to_float(scales[g]);
        const unsigned int* bp    = data + (unsigned long long)g * 4u;
        const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
        const unsigned int  base  = b * 128u;
        float block_sum = 0.0f;

        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial;
}

/* =========================================================================
   Kernel 2 — gemv_q1_g128_v7_residual
   V7 GEMV + fused residual add:  output[row] = dot(weight[row], input) + residual[row]
   Saves one kernel launch + one buffer round-trip vs. separate kernels.
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v7_residual(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    const float*         __restrict__ residual,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int n_blocks = k >> 7u;

    const unsigned short* scales =
        (const unsigned short*)blocks;
    const unsigned int* data =
        (const unsigned int*)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int  g     = row * n_blocks + b;
        const float         scale = fast_fp16_to_float(scales[g]);
        const unsigned int* bp    = data + (unsigned long long)g * 4u;
        const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
        const unsigned int  base  = b * 128u;
        float block_sum = 0.0f;

        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial + residual[row];
}

/* =========================================================================
   Kernel 3 — gemv_q1_g128_v8
   V8: shared-memory padded input cache — eliminates non-coalesced global reads.

   Strategy
   ─────────
   Problem: in V7, lane l reads global `input[b*128 + bit]` where b=l.
   Across the 32-lane warp, stride = 128 floats — one L1 miss per lane per bit.

   Fix: all 256 threads cooperatively load input into __shared__ with stride 129
   (128 + 1 padding float per block).  After __syncthreads(), each lane reads
   shared memory.  Bank for sh[b*129 + e] = (b*129 + e) % 32 = (b + e) % 32
   → all 32 lanes hit distinct banks at every inner-loop iteration → zero conflicts.

   Grid:  (⌈n_rows/8⌉, 1, 1)
   Block: (256, 1, 1)
   Shared: n_blocks × 129 × 4 bytes   (must be ≤ device shared-mem limit)
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v8(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    /* Dynamic shared memory: n_blocks × 129 floats (1 padding per block) */
    extern __shared__ float sh[];

    const unsigned int n_blocks = k >> 7u;
    const unsigned int stride   = 129u;  /* 128 data + 1 padding */

    /* Phase 1: cooperative coalesced load of input into shared memory */
    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        const unsigned int b = i >> 7u;   /* block index = i / 128 */
        const unsigned int e = i & 127u;  /* element within block  */
        sh[b * stride + e] = input[i];
    }
    __syncthreads();

    /* Phase 2: each warp computes one output row using shared-mem input */
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned short* scales =
        (const unsigned short*)blocks;
    const unsigned int* data =
        (const unsigned int*)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int  g     = row * n_blocks + b;
        const float         scale = fast_fp16_to_float(scales[g]);
        const unsigned int* bp    = data + (unsigned long long)g * 4u;
        const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];

        /* Pointer into padded shared mem for this block */
        const float* sp = sh + b * stride;

        float block_sum = 0.0f;
        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            /* Bank = (b * 129 + bit + K) % 32 = (b + bit + K) % 32
               For 32 lanes: banks = {(l + bit + K) % 32} — all distinct */
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 32u]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 64u]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 96u];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial;
}

/* =========================================================================
   Kernel 4 — gemv_q1_g128_v8_residual
   V8 GEMV + fused residual add.  Same shared-mem technique as V8.
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v8_residual(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    const float*         __restrict__ residual,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    extern __shared__ float sh[];

    const unsigned int n_blocks = k >> 7u;
    const unsigned int stride   = 129u;

    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        sh[(i >> 7u) * stride + (i & 127u)] = input[i];
    }
    __syncthreads();

    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned short* scales =
        (const unsigned short*)blocks;
    const unsigned int* data =
        (const unsigned int*)(blocks + (unsigned long long)n_rows * n_blocks * 2u);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int  g  = row * n_blocks + b;
        const float      scale = fast_fp16_to_float(scales[g]);
        const unsigned int* bp = data + (unsigned long long)g * 4u;
        const unsigned int w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
        const float*        sp = sh + b * stride;
        float block_sum = 0.0f;

        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 32u]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 64u]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * sp[bit + 96u];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial + residual[row];
}

/* =========================================================================
   Kernel 5 — rmsnorm_weighted_v2
   Root-mean-square layer normalisation with per-element weights.

   Algorithm (O(n) two-phase, fixed 256 threads):
     Phase 1: partial sum-of-squares per thread → warp shuffle → warp_ss[8]
     Phase 2: thread 0 sums 8 warp partials, computes rms_inv, broadcasts
     Phase 3: output[i] = input[i] * rms_inv * weight[i]
   ========================================================================= */
extern "C" __global__ void rmsnorm_weighted_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    unsigned int n,
    float eps
) {
    __shared__ float warp_ss[8];

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;

    float pss = 0.0f;
    for (unsigned int i = tid; i < n; i += 256u) {
        float x = input[i];
        pss += x * x;
    }

    pss += __shfl_down_sync(0xffffffffu, pss, 16u);
    pss += __shfl_down_sync(0xffffffffu, pss,  8u);
    pss += __shfl_down_sync(0xffffffffu, pss,  4u);
    pss += __shfl_down_sync(0xffffffffu, pss,  2u);
    pss += __shfl_down_sync(0xffffffffu, pss,  1u);

    if (lane == 0u) warp_ss[warp] = pss;
    __syncthreads();

    if (tid == 0u) {
        float total = 0.0f;
        for (unsigned int w = 0u; w < 8u; ++w) total += warp_ss[w];
        warp_ss[0] = 1.0f / sqrtf(total / (float)n + eps);
    }
    __syncthreads();

    const float rms_inv = warp_ss[0];
    for (unsigned int i = tid; i < n; i += 256u) {
        output[i] = input[i] * rms_inv * weight[i];
    }
}

/* =========================================================================
   Kernel 6 — residual_add
   In-place: a[i] += b[i]
   ========================================================================= */
extern "C" __global__ void residual_add(
    float*       __restrict__ a,
    const float* __restrict__ b,
    unsigned int n
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid < n) a[gid] += b[gid];
}

/* =========================================================================
   Kernel 7 — swiglu_fused
   output[i] = SiLU(gate_up[i]) × gate_up[i + n]
   Layout: gate_up = [gate: n elements | up: n elements]
   ========================================================================= */
extern "C" __global__ void swiglu_fused(
    const float* __restrict__ gate_up,
    float*       __restrict__ output,
    unsigned int n
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid < n) output[gid] = silu(gate_up[gid]) * gate_up[gid + n];
}

/* =========================================================================
   Kernel 8 — argmax_f32
   Single-block argmax with shared-memory tree reduction.

   TIE-BREAK CONTRACT (perf-11 / RT-22): on equal values this returns the
   SMALLEST ORIGINAL INDEX, byte-for-byte the same rule as the MSL `argmax`
   in kernel_sources/utility.rs, llama.cpp, and the CPU sampler's
   `argmax_first`.  A strict `>` alone compares reduction SLOTS, not indices,
   so the surviving maximum depended on the fold path (measured on the MSL
   twin: equal maxima at 1000/2000 returned 2000; 144 of 320 randomized
   multi-way ties returned a non-minimal index).  Two halves:
     - the per-thread strided scan keeps `>`, which already retains the
       smallest index one thread sees because `i` ascends;
     - every tree stage takes the partner slot when it is strictly greater
       OR equal with a smaller original index.
   A thread that scans nothing seeds `local_idx = n` (out of range) so it
   loses every tie instead of winning with a `0` it never observed.
   NOTE: `__syncthreads()` must stay OUTSIDE the `if (tid < s)` block — a
   barrier inside a divergent branch hangs the block — and the partner reads
   must stay INSIDE it, since the original relied on the `&&` short-circuit
   to keep `sm_val[tid + s]` in bounds.
   ========================================================================= */
extern "C" __global__ void argmax_f32(
    const float*  __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int n
) {
    __shared__ float       sm_val[256];
    __shared__ unsigned int sm_idx[256];

    const unsigned int tid = threadIdx.x;
    float        local_max = -3.402823e+38f;
    unsigned int local_idx = n;   /* out-of-range sentinel: loses every tie */

    for (unsigned int i = tid; i < n; i += 256u) {
        if (input[i] > local_max) { local_max = input[i]; local_idx = i; }
    }

    sm_val[tid] = local_max;
    sm_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float        ov = sm_val[tid + s];
            const unsigned int oi = sm_idx[tid + s];
            const float        mv = sm_val[tid];
            const unsigned int mi = sm_idx[tid];
            if (ov > mv || (ov == mv && oi < mi)) {
                sm_val[tid] = ov;
                sm_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    /* Only reachable with the sentinel when n == 0 (nothing to scan). */
    if (tid == 0u) output[0] = (sm_idx[0] < n) ? sm_idx[0] : 0u;
}

/* =========================================================================
   Kernel 9 — fused_gate_up_swiglu_q1
   Reads one gate row AND one up row from the SoA Q1 weight matrix in a
   single kernel, computes both dot products, then applies SiLU(gate)*up
   in the epilogue.  Halves dispatch count vs. separate GEMV + SwiGLU.

   Weight layout (SoA, concatenated gate+up):
     [scales: 2*n_rows * n_blocks * 2 bytes FP16]  <- gate rows first, then up
     [data:   2*n_rows * n_blocks * 16 bytes]

   Grid:  (ceil(n_rows/8), 1, 1)  -- 8 output elements per CTA
   Block: (256, 1, 1)             -- 8 warps, each handles 1 output element

   Each warp:
     - computes gate_sum for row `warp_id + blockIdx.x*8`
     - computes up_sum   for row `warp_id + blockIdx.x*8 + n_rows`
     - applies output[row] = silu(gate_sum) * up_sum
   ========================================================================= */
extern "C" __global__ void fused_gate_up_swiglu_q1(
    const unsigned char* __restrict__ blocks,   /* SoA: [gate scales | up scales | gate data | up data] */
    const float*         __restrict__ input,
    float*               __restrict__ output,   /* [n_rows] SiLU(gate)*up */
    unsigned int n_rows,                        /* = intermediate_size */
    unsigned int k                              /* = hidden_size */
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    /* total_rows = 2 * n_rows (gate rows 0..n_rows, up rows n_rows..2*n_rows) */
    const unsigned int total_rows = 2u * n_rows;
    const unsigned int n_blocks   = k >> 7u;  /* k / 128 */

    /* Scales pointer: first 2*n_rows * n_blocks FP16 values */
    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;

    /* Data pointer: skip ALL scales — both gate and up halves
       data offset = 2 * n_rows * n_blocks * 2 bytes (each FP16 is 2 bytes) */
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks +
            (unsigned long long)total_rows * n_blocks * 2u);

    float gate_partial = 0.0f;
    float up_partial   = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        /* ── Gate block ──────────────────────────────────────────────── */
        {
            const unsigned int  g      = row * n_blocks + b;
            const float         scale  = fast_fp16_to_float(scales[g]);
            const unsigned int* bp     = data + (unsigned long long)g * 4u;
            const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
            const unsigned int  base   = b * 128u;
            float block_sum = 0.0f;

            #pragma unroll
            for (unsigned int bit = 0u; bit < 32u; ++bit) {
                block_sum +=
                    (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
                  + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
                  + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
                  + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
            }
            gate_partial += scale * block_sum;
        }

        /* ── Up block ────────────────────────────────────────────────── */
        {
            const unsigned int  g_up   = (row + n_rows) * n_blocks + b;
            const float         scale  = fast_fp16_to_float(scales[g_up]);
            const unsigned int* bp     = data + (unsigned long long)g_up * 4u;
            const unsigned int  w0 = bp[0u], w1 = bp[1u], w2 = bp[2u], w3 = bp[3u];
            const unsigned int  base   = b * 128u;
            float block_sum = 0.0f;

            #pragma unroll
            for (unsigned int bit = 0u; bit < 32u; ++bit) {
                block_sum +=
                    (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
                  + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
                  + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
                  + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
            }
            up_partial += scale * block_sum;
        }
    }

    /* Warp-shuffle reduce gate_partial */
    gate_partial += __shfl_down_sync(0xffffffffu, gate_partial, 16u);
    gate_partial += __shfl_down_sync(0xffffffffu, gate_partial,  8u);
    gate_partial += __shfl_down_sync(0xffffffffu, gate_partial,  4u);
    gate_partial += __shfl_down_sync(0xffffffffu, gate_partial,  2u);
    gate_partial += __shfl_down_sync(0xffffffffu, gate_partial,  1u);

    /* Warp-shuffle reduce up_partial */
    up_partial += __shfl_down_sync(0xffffffffu, up_partial, 16u);
    up_partial += __shfl_down_sync(0xffffffffu, up_partial,  8u);
    up_partial += __shfl_down_sync(0xffffffffu, up_partial,  4u);
    up_partial += __shfl_down_sync(0xffffffffu, up_partial,  2u);
    up_partial += __shfl_down_sync(0xffffffffu, up_partial,  1u);

    /* Lane 0 writes fused output: SiLU(gate) * up */
    if (lane == 0u) output[row] = silu(gate_partial) * up_partial;
}

/* =========================================================================
   Kernel 10 — gemv_q1_g128_v9
   V9: 128-bit vectorized weight load + __ldg() scale read.
   Identical launch parameters to V7.  Uses PTX ld.global.nc.v4.u32 to
   load 4 × u32 (128 bits) in a single instruction, reducing instruction
   count and improving L2 bandwidth utilisation on sm_75 (Turing).

   Grid:  (ceil(n_rows/8), 1, 1)
   Block: (256, 1, 1)
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v9(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int n_blocks = k >> 7u;

    const unsigned long long total_blocks = (unsigned long long)n_rows * n_blocks;
    /* F10: the 128-bit vector load below needs a 16-byte-aligned address. */
    const unsigned int aligned16 = soa_quant_aligned16(total_blocks);

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + total_blocks * 2ull);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int g = row * n_blocks + b;
        /* __ldg: read-only data hint → texture cache path for scale */
        const float scale = fast_fp16_to_float(__ldg(&scales[g]));

        /* PTX 128-bit non-caching vector load for weight bits (F10-guarded) */
        const unsigned int* bp = data + (unsigned long long)g * 4u;
        unsigned int w0, w1, w2, w3;
        soa_load4_u32(bp, aligned16, w0, w1, w2, w3);


        const unsigned int base = b * 128u;
        float block_sum = 0.0f;

        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial;
}

/* =========================================================================
   Kernel TQ2 — gemv_tq2_g128_v1
   Ternary (TQ2_0_g128) GEMV.  SoA layout:
     [scales: n_rows * n_blocks * 2 bytes FP16 LE]
     [qs:     n_rows * n_blocks * 32 bytes  (4 codes/byte, 0b00→-1, 0b01→0, 0b10→+1, 0b11→0)]

   Grid:  (ceil(n_rows/8), 1, 1)   — 8 output rows per CTA
   Block: (256, 1, 1)              — 8 warps × 32 lanes
   Each lane processes blocks: lane, lane+32, … via 128-bit vector loads of qs.
   ========================================================================= */
static __device__ __forceinline__ float decode_tq2(unsigned int code) {
    /* 0b00 → -1, 0b10 → +1, others → 0  (codes 0b01 and 0b11 are zero values) */
    return (code == 0u) ? -1.0f : ((code == 2u) ? 1.0f : 0.0f);
}

static __device__ __forceinline__ float byte_dot_tq2(unsigned int b, const float* x) {
    return decode_tq2((b      ) & 3u) * x[0]
         + decode_tq2((b >> 2u) & 3u) * x[1]
         + decode_tq2((b >> 4u) & 3u) * x[2]
         + decode_tq2((b >> 6u) & 3u) * x[3];
}

extern "C" __global__ void gemv_tq2_g128_v1(
    const unsigned char* __restrict__ soa_raw,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 7u;
    const unsigned long long total_blocks = (unsigned long long)n_rows * blocks_per_row;
    const unsigned long long qs_offset    = total_blocks * 2ull;
    /* F10: both 128-bit vector loads below need a 16-byte-aligned address,
       which holds iff qs_offset % 16 == 0 (block stride is 32 B, base is a
       256-byte-aligned cuMemAlloc). */
    const unsigned int aligned16 = soa_quant_aligned16(total_blocks);

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)soa_raw;

    float partial = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned long long block_idx = (unsigned long long)row * blocks_per_row + b;
        const float scale = fast_fp16_to_float(__ldg(&scales[block_idx]));

        /* qs is 32 bytes per block: 2x 128-bit vector loads for 8 u32s, or
           8 scalar __ldg reads when the section is not 16-byte aligned. */
        const unsigned int* qs_words =
            (const unsigned int*)(soa_raw + qs_offset + block_idx * 32u);
        unsigned int w0, w1, w2, w3, w4, w5, w6, w7;
        soa_load4_u32(qs_words,      aligned16, w0, w1, w2, w3);
        soa_load4_u32(qs_words + 4u, aligned16, w4, w5, w6, w7);


        const float* x = input + b * 128u;
        float block_sum =
            byte_dot_tq2((w0      ) & 0xFFu, x +   0u)
          + byte_dot_tq2((w0 >>  8u) & 0xFFu, x +   4u)
          + byte_dot_tq2((w0 >> 16u) & 0xFFu, x +   8u)
          + byte_dot_tq2((w0 >> 24u) & 0xFFu, x +  12u)
          + byte_dot_tq2((w1      ) & 0xFFu, x +  16u)
          + byte_dot_tq2((w1 >>  8u) & 0xFFu, x +  20u)
          + byte_dot_tq2((w1 >> 16u) & 0xFFu, x +  24u)
          + byte_dot_tq2((w1 >> 24u) & 0xFFu, x +  28u)
          + byte_dot_tq2((w2      ) & 0xFFu, x +  32u)
          + byte_dot_tq2((w2 >>  8u) & 0xFFu, x +  36u)
          + byte_dot_tq2((w2 >> 16u) & 0xFFu, x +  40u)
          + byte_dot_tq2((w2 >> 24u) & 0xFFu, x +  44u)
          + byte_dot_tq2((w3      ) & 0xFFu, x +  48u)
          + byte_dot_tq2((w3 >>  8u) & 0xFFu, x +  52u)
          + byte_dot_tq2((w3 >> 16u) & 0xFFu, x +  56u)
          + byte_dot_tq2((w3 >> 24u) & 0xFFu, x +  60u)
          + byte_dot_tq2((w4      ) & 0xFFu, x +  64u)
          + byte_dot_tq2((w4 >>  8u) & 0xFFu, x +  68u)
          + byte_dot_tq2((w4 >> 16u) & 0xFFu, x +  72u)
          + byte_dot_tq2((w4 >> 24u) & 0xFFu, x +  76u)
          + byte_dot_tq2((w5      ) & 0xFFu, x +  80u)
          + byte_dot_tq2((w5 >>  8u) & 0xFFu, x +  84u)
          + byte_dot_tq2((w5 >> 16u) & 0xFFu, x +  88u)
          + byte_dot_tq2((w5 >> 24u) & 0xFFu, x +  92u)
          + byte_dot_tq2((w6      ) & 0xFFu, x +  96u)
          + byte_dot_tq2((w6 >>  8u) & 0xFFu, x + 100u)
          + byte_dot_tq2((w6 >> 16u) & 0xFFu, x + 104u)
          + byte_dot_tq2((w6 >> 24u) & 0xFFu, x + 108u)
          + byte_dot_tq2((w7      ) & 0xFFu, x + 112u)
          + byte_dot_tq2((w7 >>  8u) & 0xFFu, x + 116u)
          + byte_dot_tq2((w7 >> 16u) & 0xFFu, x + 120u)
          + byte_dot_tq2((w7 >> 24u) & 0xFFu, x + 124u);

        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial;
}

/* =========================================================================
   Kernel 11 — gemv_q1_g128_v9_residual
   V9 GEMV + fused residual add.  output[row] = dot(weight[row], input) + residual[row]
   ========================================================================= */
extern "C" __global__ void gemv_q1_g128_v9_residual(
    const unsigned char* __restrict__ blocks,
    const float*         __restrict__ input,
    const float*         __restrict__ residual,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int n_blocks = k >> 7u;

    const unsigned long long total_blocks = (unsigned long long)n_rows * n_blocks;
    /* F10: the 128-bit vector load below needs a 16-byte-aligned address. */
    const unsigned int aligned16 = soa_quant_aligned16(total_blocks);

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)blocks;
    const unsigned int* __restrict__ data =
        (const unsigned int* __restrict__)(blocks + total_blocks * 2ull);

    float partial = 0.0f;

    for (unsigned int b = lane; b < n_blocks; b += 32u) {
        const unsigned int g = row * n_blocks + b;
        const float scale = fast_fp16_to_float(__ldg(&scales[g]));

        const unsigned int* bp = data + (unsigned long long)g * 4u;
        unsigned int w0, w1, w2, w3;
        soa_load4_u32(bp, aligned16, w0, w1, w2, w3);


        const unsigned int base = b * 128u;
        float block_sum = 0.0f;

        #pragma unroll
        for (unsigned int bit = 0u; bit < 32u; ++bit) {
            block_sum +=
                (((w0 >> bit) & 1u) ? 1.0f : -1.0f) * input[base       + bit]
              + (((w1 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 32u + bit]
              + (((w2 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 64u + bit]
              + (((w3 >> bit) & 1u) ? 1.0f : -1.0f) * input[base + 96u + bit];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial + residual[row];
}
"#;

/// Shared-memory bytes required by the V8 kernel for a given `k`.
///
/// V8 pads each 128-element block by 1 float to eliminate bank conflicts:
/// `shared_bytes = (k / 128) * 129 * 4`
///
/// Returns `None` if the result exceeds `max_shared_bytes`.
#[inline]
pub fn v8_shared_mem_bytes(k: usize, max_shared_bytes: usize) -> Option<u32> {
    let n_blocks = k / 128;
    let bytes = n_blocks * 129 * 4;
    if bytes <= max_shared_bytes {
        Some(bytes as u32)
    } else {
        None
    }
}

/// Like `v8_shared_mem_bytes` but uses the sm_75 extended 64 KB limit.
///
/// On Turing (sm_75), `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536)`
/// allows up to 64 KB dynamic shared memory per block, enabling V8 for k=14336.
#[inline]
pub fn v8_extended_shared_mem_bytes(k: usize) -> Option<u32> {
    v8_shared_mem_bytes(k, 65_536)
}
