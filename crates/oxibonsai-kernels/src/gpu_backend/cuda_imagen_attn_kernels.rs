//! CUDA-C kernel source for the FLUX.2 DiT joint flash-attention
//! (`joint_attention_flash_f32`).
//!
//! **CUDA prototype mirror** of the Metal `joint_attention_flash_f32`
//! (`gpu_backend/kernel_sources/dit_attention_flash.rs`) and the CPU reference
//! `oxibonsai-image/src/math.rs::joint_attention`. This is the *correctness-first*
//! CUDA port: it implements the identical **flash-attention v2** (online softmax)
//! reordering of the CPU full-row, max-subtracted softmax, but **scalarizes** the
//! Metal `simdgroup_float8x8` matrix MACs into plain FP32 scalar/warp-tiled MACs.
//!
//! # Parity-first design (everything f32)
//!
//! Q, K, V, the scores `S`, the softmax probabilities `P`, the running max `m`,
//! the running sum `l`, and the output accumulator `O` are **all f32**, with
//! **f32 accumulation** throughout. There are deliberately **no tensor cores**
//! (`wmma` / `mma.sync`) and **no f16 staging**: the Metal v9/v10 lesson is that
//! f16 staging of full-range attention values drifts past the `1e-3` parity bound
//! (only ternary `code×scale` was f16-exact, which does not apply here). The
//! online-softmax reorder is mathematically *exact* vs the CPU full-row softmax
//! (modulo f32 reassociation), so parity holds (expect max-abs ≈ 1e-5).
//!
//! Tensor-Core (mma.sync) perf tuning deferred to the hardware phase.
//!
//! # The op (matches the CPU reference bit-for-bit in behaviour)
//!
//! ```text
//!   per head h (num_heads = 24 in the DiT):
//!     S[qi,ki] = scale · Σ_d q[h,qi,d]·k[h,ki,d],  scale = 1/sqrt(head_dim)
//!     softmax over ki (FULL ROW, NON-causal/bidirectional, max-subtracted)
//!     O[qi,d]  = Σ_ki softmax(S)[qi,ki] · v[h,ki,d]
//!   then transpose to token-major:
//!     out[qi*(num_heads*head_dim) + h*head_dim + d] = O[h,qi,d]
//! ```
//!
//! Layout (identical to the Metal kernel):
//!   - `q`/`k`/`v`: head-major `[num_heads, seq, head_dim]` f32 (RoPE already
//!     applied upstream to q,k):  `q[h*seq*D + n*D + d]`.
//!   - `out`: token-major TRANSPOSED `[seq, num_heads*head_dim]` f32.
//!
//! `seq = N ≤ 1536`, `head_dim = D = 128`.
//!
//! # Kernel structure (flash-attention v2, scalar warp-tiled)
//!
//! grid = `(seq.div_ceil(BQ), num_heads, 1)`, block = `(128, 1, 1)`,
//! `BQ = 64`, `BK = 32`. `__shared__ float Ksh[BK*128], Vsh[BK*128]`
//! (32 KiB static — under the 48 KiB default per-block limit, so no opt-in
//! `cudaFuncSetAttribute` is required).
//!
//! Each thread owns a strided subset of the `BQ` query rows of the tile
//! (`128` threads cover `64` rows, so each thread owns its `row, row+128, …` —
//! here just one row). For each owned query row `qi < seq` it keeps the query
//! vector `Q[128]` in registers, plus the running softmax state
//! `m = -inf, l = 0, O[128] = {0}`. The key loop streams `BK`-key tiles:
//! cooperatively stage `Ksh`/`Vsh` from device `k`/`v` (clamping the partial
//! tile at the `seq` edge to zero), `__syncthreads()`, then for each in-range
//! key `kk` accumulate the online softmax:
//!
//! ```text
//!   s      = scale · Σ_d Q[d]·Ksh[kk*128+d]
//!   m_new  = fmaxf(m, s)
//!   corr   = (m == -inf) ? 0 : expf(m - m_new)
//!   p      = expf(s - m_new)
//!   l      = l*corr + p
//!   O[d]   = O[d]*corr + p*Vsh[kk*128+d]   (all d)
//!   m      = m_new
//! ```
//!
//! then `__syncthreads()` before re-staging `Ksh`/`Vsh`. After all tiles:
//! `out[qi*(num_heads*head_dim) + h*head_dim + d] = (l > 0) ? O[d]/l : 0`.
//!
//! Register pressure (`Q[128] + O[128]` = 256 f32 per thread) is high → low
//! occupancy is **acceptable** for this parity prototype (spilling `O`/`Q` to
//! local memory is fine; correctness is the goal, perf tuning is the hardware
//! phase).

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// CUDA-C source for the DiT joint flash-attention kernel
/// `joint_attention_flash_f32`.
///
/// Compiled at process startup via `cudarc::nvrtc::compile_ptx`. No CUDA SDK
/// headers are required; `expf`/`fmaxf` are NVRTC built-ins.
///
/// Entry point signature (matches the Metal kernel's buffer order):
///   `q`         `[num_heads × seq × head_dim]` (f32, head-major)
///   `k`         `[num_heads × seq × head_dim]` (f32, head-major)
///   `v`         `[num_heads × seq × head_dim]` (f32, head-major)
///   `out`       `[seq × (num_heads*head_dim)]` (f32, token-major transposed)
///   `num_heads` (u32 scalar)
///   `seq`       (u32 scalar)
///   `head_dim`  (u32 scalar)
///   `scale`     (f32 scalar)
///
/// Launch: grid `(ceil(seq/BQ), num_heads, 1)`, block `(128, 1, 1)`,
/// `shared_mem_bytes = 0` (the kernel uses only the 32 KiB *static* `Ksh`/`Vsh`).
pub const CUDA_IMAGEN_ATTN_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA imagen kernels — DiT joint flash-attention (parity prototype)

   joint_attention_flash_f32: flash-attention v2 (online softmax), PLAIN FP32
   scalar/warp-tiled MACs with f32 accumulate throughout. NO tensor cores, NO
   f16 staging (parity-first; tensor-core perf tuning deferred to the hardware
   phase). Mirrors the Metal joint_attention_flash_f32 ALGORITHM and the CPU
   reference oxibonsai-image::math::joint_attention.

   No CUDA SDK headers required; expf/fmaxf are NVRTC built-ins.
   ========================================================================= */

/* Tile geometry (must match the host launch: grid=(ceil(seq/BQ),num_heads,1),
   block=(128,1,1)).
     query tile (rows of `out` per block) : FA_BQ = 64
     key tile   (keys per online step)    : FA_BK = 32
     threads per block                    : FA_THREADS = 128
     head_dim compile-time bound          : FA_DMAX (per-thread Q[]/O[] array
       bound). OVERRIDABLE via a prepended `#define FA_DMAX <N>u`: the host
       compiles this one source twice — a lean DiT variant (FA_DMAX=128, 32 KiB
       shared, no >48 KiB opt-in → full L1) and a wider VAE variant (FA_DMAX=384,
       96 KiB shared) — see cudagraph_global_group::new. Only [0,head_dim) is ever
       touched, so each variant is exact for its head_dim. Defaults to 384. */
#define FA_BQ 8u                    /* query rows per block (= FA_WARPS*FA_RPW) */
#define FA_BK 32u
#define FA_THREADS 128u
#define FA_WARPS 4u                  /* FA_THREADS / 32 */
#define FA_RPW (FA_BQ / FA_WARPS)    /* query rows per WARP = 2 */
#ifndef FA_DMAX
#define FA_DMAX 384u
#endif
/* Head-dim slots held per lane: lane `t` owns dims {t, t+32, t+64, ...}, so each
   lane keeps ceil(head_dim/32) f32 of Q and O in REGISTERS (no local-memory
   spill — the core win over the thread-per-row prototype). */
#define FA_MAXDPL ((FA_DMAX + 31u) / 32u)

/* Running-max init sentinel. Uses the literal -FLT_MAX (the proven in-repo
   idiom in cuda_attn_kernels.rs) rather than the INFINITY macro, so the kernel
   depends on no <math.h>/<cfloat> include. It plays the exact role of the
   Metal kernel's -INFINITY: the first key sets m_new = max(sentinel, s) = s
   (any real score s exceeds -FLT_MAX), and the `m == FA_NEG_INF` test below
   forces corr = 0 on that first key (so l = p and O = p*v), bit-for-bit the
   Metal kernel's `(m_old == -INFINITY) ? 0 : exp(m_old - m_new)`. */
#define FA_NEG_INF (-3.402823e+38f)

/* Flash-attention v2 (online softmax) joint multi-head scaled-dot-product
   attention (NON-causal / bidirectional), scalar FP32 MACs, writing the
   token-major TRANSPOSED output directly.

   One block computes the FA_BQ output rows out[q0 .. q0+FA_BQ, :] for the
   head `h = blockIdx.y` and the query-tile `blockIdx.x`. Each of the 128
   threads owns the tile-local query rows {tid, tid+FA_THREADS, ...} (with
   FA_BQ=64 <= FA_THREADS=128, each valid row is owned by exactly one thread).
*/
extern "C" __global__ void joint_attention_flash_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    unsigned int num_heads,
    unsigned int seq,
    unsigned int head_dim,
    float scale
) {
    const unsigned int q_tile = blockIdx.x;          /* query-tile index */
    const unsigned int h      = blockIdx.y;          /* head index       */
    if (h >= num_heads) return;
    const unsigned int q0 = q_tile * FA_BQ;          /* first query row of tile */
    if (q0 >= seq) return;

    const unsigned int tid  = threadIdx.x;           /* 0 .. FA_THREADS-1 */
    const unsigned int lane = tid & 31u;             /* 0 .. 31 (warp lane)  */
    const unsigned int wid  = tid >> 5u;             /* 0 .. FA_WARPS-1      */

    /* Per-head base offset into the head-major [num_heads, seq, head_dim]
       tensors. */
    const unsigned long long head_off =
        (unsigned long long)h * (unsigned long long)seq * (unsigned long long)head_dim;

    /* Dynamic shared K/V tile (Ksh ‖ Vsh), FA_BK*head_dim*2 floats — sized at
       launch (DiT head_dim=128 → 32 KiB; VAE head_dim=384 → 96 KiB via the
       host's cuFuncSetAttribute opt-in). All f32, no f16 staging. */
    extern __shared__ float fa_smem[];
    float* Ksh = fa_smem;
    float* Vsh = fa_smem + (FA_BK * head_dim);

    /* ── WARP-COOPERATIVE rows. Each WARP owns FA_RPW tile-local query rows
       {wid, wid+FA_WARPS, ...}; within a warp the 32 lanes split head_dim
       (lane t -> dims {t, t+32, ...}), so Q/O live in REGISTERS (no spill) and
       the score is a lane-partial dot finished by a butterfly warp reduction.
       `owns[]` is warp-uniform (depends on wid, not lane), so every lane of a
       warp takes the same path through the block-collective staging + the two
       __syncthreads() and the __shfl_xor_sync below — no barrier/shuffle
       divergence. Only the final writeback is guarded per row. */
    unsigned int qrow[FA_RPW];
    bool owns[FA_RPW];
    float Q[FA_RPW][FA_MAXDPL];
    float O[FA_RPW][FA_MAXDPL];
    float m[FA_RPW];
    float l[FA_RPW];
    #pragma unroll
    for (unsigned int rr = 0u; rr < FA_RPW; rr++) {
        const unsigned int row = wid + rr * FA_WARPS;     /* tile-local row */
        qrow[rr] = q0 + row;                              /* global query row */
        owns[rr] = (row < FA_BQ) && (qrow[rr] < seq);
        m[rr] = FA_NEG_INF;
        l[rr] = 0.0f;
        #pragma unroll
        for (unsigned int s = 0u; s < FA_MAXDPL; s++) { Q[rr][s] = 0.0f; O[rr][s] = 0.0f; }
        if (owns[rr]) {
            const float* qsrc = q + head_off + (unsigned long long)qrow[rr] * head_dim;
            #pragma unroll
            for (unsigned int s = 0u; s < FA_MAXDPL; s++) {
                const unsigned int d = lane + s * 32u;
                if (d < head_dim) Q[rr][s] = qsrc[d];
            }
        }
    }

    /* ── Online-softmax key loop. The cooperative Ksh/Vsh staging and the two
       __syncthreads() are BLOCK-COLLECTIVE — run by ALL FA_THREADS every tile. */
    const unsigned int k_tiles = (seq + FA_BK - 1u) / FA_BK;
    for (unsigned int kt = 0u; kt < k_tiles; kt++) {
        const unsigned int k0 = kt * FA_BK;
        const unsigned int k_valid =
            (k0 < seq) ? min(FA_BK, seq - k0) : 0u;

        for (unsigned int i = tid; i < FA_BK * head_dim; i += FA_THREADS) {
            const unsigned int kk = i / head_dim;
            const unsigned int d  = i % head_dim;
            float kv = 0.0f;
            float vv = 0.0f;
            if (kk < k_valid) {
                const unsigned long long base =
                    head_off + (unsigned long long)(k0 + kk) * head_dim + d;
                kv = k[base];
                vv = v[base];
            }
            Ksh[kk * head_dim + d] = kv;
            Vsh[kk * head_dim + d] = vv;
        }
        __syncthreads();

        for (unsigned int kk = 0u; kk < k_valid; kk++) {
            const float* ksp = Ksh + kk * head_dim;
            const float* vsp = Vsh + kk * head_dim;
            #pragma unroll
            for (unsigned int rr = 0u; rr < FA_RPW; rr++) {
                /* lane-partial dot over this lane's head dims */
                float partial = 0.0f;
                #pragma unroll
                for (unsigned int s = 0u; s < FA_MAXDPL; s++) {
                    const unsigned int d = lane + s * 32u;
                    if (d < head_dim) partial += Q[rr][s] * ksp[d];
                }
                /* butterfly warp reduction — every lane ends with the full score */
                #pragma unroll
                for (unsigned int off = 16u; off > 0u; off >>= 1u)
                    partial += __shfl_xor_sync(0xffffffffu, partial, off);
                const float sc = partial * scale;

                const float m_new = fmaxf(m[rr], sc);
                /* corr = exp(m - m_new); m == sentinel on the first key => 0
                   (mirrors the Metal `(m_old == -INFINITY) ? 0 : ...`). */
                const float corr = (m[rr] == FA_NEG_INF) ? 0.0f : expf(m[rr] - m_new);
                const float p = expf(sc - m_new);
                l[rr] = l[rr] * corr + p;
                #pragma unroll
                for (unsigned int s = 0u; s < FA_MAXDPL; s++) {
                    const unsigned int d = lane + s * 32u;
                    if (d < head_dim) O[rr][s] = O[rr][s] * corr + p * vsp[d];
                }
                m[rr] = m_new;
            }
        }
        __syncthreads();                             /* before re-staging Ksh/Vsh */
    }

    /* ── Writeback: out[qi, h*head_dim + d] = O[d] / l (token-major). ── */
    const unsigned int inner = num_heads * head_dim;
    #pragma unroll
    for (unsigned int rr = 0u; rr < FA_RPW; rr++) {
        if (!owns[rr]) continue;
        const float inv = (l[rr] > 0.0f) ? (1.0f / l[rr]) : 0.0f;
        const unsigned long long obase =
            (unsigned long long)qrow[rr] * inner + (unsigned long long)h * head_dim;
        #pragma unroll
        for (unsigned int s = 0u; s < FA_MAXDPL; s++) {
            const unsigned int d = lane + s * 32u;
            if (d < head_dim) out[obase + d] = O[rr][s] * inv;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The CUDA source string must contain the kernel entry point so the
    /// integration / dispatch layer (and the Linux NVRTC compile) can locate it.
    #[test]
    fn src_has_entry_point() {
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("joint_attention_flash_f32"));
    }

    /// Sanity: parity-first invariants — no tensor cores / f16 staging, and the
    /// online-softmax math (max-subtraction) is present.
    #[test]
    fn src_is_parity_first_fp32() {
        // No tensor-core / wmma usage anywhere in the prototype.
        assert!(!CUDA_IMAGEN_ATTN_SRC.contains("wmma"));
        assert!(!CUDA_IMAGEN_ATTN_SRC.contains("mma.sync"));
        // No f16 staging: the staging buffers and all accumulators are `float`.
        assert!(!CUDA_IMAGEN_ATTN_SRC.contains("__half"));
        assert!(!CUDA_IMAGEN_ATTN_SRC.contains("half2"));
        // Online softmax stabilisation + normaliser present.
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("fmaxf"));
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("expf"));
        // f32 (dynamic) shared staging for K and V (no f16): a single
        // `extern __shared__ float` arena split into Ksh ‖ Vsh.
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("extern __shared__ float fa_smem"));
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("float* Ksh = fa_smem"));
        assert!(CUDA_IMAGEN_ATTN_SRC.contains("float* Vsh = fa_smem"));
    }
}
