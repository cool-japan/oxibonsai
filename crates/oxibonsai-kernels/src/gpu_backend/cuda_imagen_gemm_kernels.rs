//! CUDA C kernel source for the OxiBonsai image-generation (FLUX.2) GEMM path.
//!
//! These two CUDA-core GEMMs mirror, line-for-line, the proven Metal kernels of
//! the image pipeline:
//!
//! | CUDA kernel | Metal sibling                           | Op                                   |
//! |-------------|-----------------------------------------|--------------------------------------|
//! | `gemm_f32`  | `gemm_f32_simdgroup` (`prefill_f32_simdgroup.rs`) | f32-exact `out[M,N]=A[M,K]·W[N,K]ᵀ`  |
//! | `gemm_tq2`  | `gemm_tq2_g128_v10_simdgroup` (`prefill_simdgroup_v10.rs`) | ternary `out=A·dequant(B)ᵀ`          |
//!
//! # Parity-first design (NOT a perf kernel)
//!
//! These are **plain FP32 CUDA-core** tiled GEMMs with **f32 accumulate** — they
//! deliberately do **not** use tensor cores / `wmma` / `mma.sync` / TF32, which
//! truncate the mantissa and would break the `cos ≥ 0.999` parity gate the
//! Metal path holds. The op is the column-major
//! `out[M,N] = A[M,K] · W[N,K]ᵀ` (== row-major `[M,K]`/`[M,N]`) of the CPU
//! reference `oxibonsai-image::gemm::gemm_abt`; reassociating the sum on the GPU
//! stays cos ≈ 1.0. **Tensor-Core perf tuning is deferred to the hardware
//! phase.**
//!
//! # FULL-M (the cap-of-8 fix)
//!
//! The existing CUDA Q1 GEMV path has a known cap-of-8 trap (capping the batch
//! dimension at 8 columns). **Both kernels here process the FULL `M`** via
//! `grid.y` tiles plus an in-kernel `m_local < M` clamp — every output column is
//! computed. The parity tests exercise `M > 8` shapes (e.g. `M ∈ {33,40,96}`) to
//! prove this.
//!
//! # TQ2 weight buffer (SoA)
//!
//! `gemm_tq2` consumes the **same** SoA layout `CudaGraph::get_or_upload_weight_tq2_soa`
//! produces (mirrors the Metal `reformat_tq2_aos_to_soa`):
//!
//! ```text
//! [scales: N·(K/128) × 2 bytes FP16 LE]
//! [qs:     N·(K/128) × 32 bytes]   (4 ternary codes/byte LSB-first)
//! block index b = row·(K/128) + kb
//! ```
//!
//! The decode (`decode_tq2`) and the `code × scale ∈ {-scale, 0, +scale}`
//! f16-exact staging are byte-for-byte the Metal `v10` (and CPU
//! `BlockTQ2_0_g128::dequant`) semantics.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

/// Combined CUDA C source for the two image-generation GEMM kernels
/// (`gemm_f32` + `gemm_tq2`).
///
/// Compiled once at process startup via `cudarc::nvrtc::compile_ptx`. NVRTC
/// compiles each source string independently, so the `fast_fp16_to_float` and
/// `decode_tq2` device helpers are re-pasted at the top of *this* string
/// (verbatim copies of the ones in `cuda_kernels.rs`).
pub const CUDA_IMAGEN_GEMM_SRC: &str = r#"
/* =========================================================================
   OxiBonsai CUDA image-generation (FLUX.2) GEMM kernels.
   PARITY-FIRST: plain FP32 CUDA-core MACs, f32 accumulate. No tensor cores
   / wmma / mma.sync / TF32 (they truncate the mantissa and break the
   cos >= 0.999 parity gate). Tensor-Core perf tuning deferred to the
   hardware phase.
   No CUDA SDK headers required; all intrinsics are NVRTC built-ins.
   ========================================================================= */

/* -- Hardware FP16 -> FP32 via PTX (1 instruction on all SM6.0+ GPUs) ----- */
/*    Verbatim copy of cuda_kernels.rs::fast_fp16_to_float (NVRTC compiles   */
/*    each source string independently, so the helper must live here too).   */
static __device__ __forceinline__ float fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* -- 2-bit ternary decode -------------------------------------------------- */
/*    Verbatim copy of cuda_kernels.rs::decode_tq2: 0b00 -> -1, 0b10 -> +1,   */
/*    others (0b01, 0b11) -> 0. Matches the Metal v10 v10_decode_tq2 and the   */
/*    CPU BlockTQ2_0_g128::dequant.                                            */
static __device__ __forceinline__ float decode_tq2(unsigned int code) {
    return (code == 0u) ? -1.0f : ((code == 2u) ? 1.0f : 0.0f);
}

/* =========================================================================
   Kernel 1 — gemm_f32
   f32-EXACT batched GEMM: out[M,N] = A[M,K] . W[N,K]^T.
   CUDA-core sibling of the Metal gemm_f32_simdgroup (prefill_f32_simdgroup.rs).

   Buffers / conventions (column-major == row-major [M,K] / [M,N]):
     weights : f32, [N, K] row-major  -> weights[n*K + e]
     inputs  : f32, [M, K]            -> inputs[col*K + e]   (col = m)
     outputs : f32, [M, N]            -> outputs[col*N + row] (row = n)
     n_rows  : N
     batch_size : M  (FULL — no cap-of-8)
     k       : inner dim (arbitrary K >= 1; last BK tile is zero-clamped)

   Tiling (plain CUDA cores, f32 accumulate):
     output tile   BM x BN = 64 x 64
     K step        BK = 16
     block         (16, 16) = 256 threads
     micro-tile    4 x 4 per thread  (TM x TN; BM/16 = 4, BN/16 = 4)
     shared mem    As[64*16] + Ws[64*16] = 2 * 4 KiB = 8 KiB
     grid          (ceil(N/64), ceil(M/64), 1)
   Cooperative tile load with OOB -> 0 clamp; column-major writeback with an
   m_local < M && n_local < N clamp so arbitrary (non-tile-multiple) shapes
   are correct.  FULL-M (grid.y + m_local clamp) — fixes the cap-of-8 trap.
   ========================================================================= */
extern "C" __global__ void gemm_f32(
    const float* __restrict__ weights,     /* [N, K] row-major */
    const float* __restrict__ inputs,      /* [M, K] col-major (col*K + e) */
    float*       __restrict__ outputs,     /* [M, N] col-major (col*N + row) */
    unsigned int n_rows,                   /* N */
    unsigned int batch_size,               /* M, FULL */
    unsigned int k
) {
    const unsigned int BM = 64u;
    const unsigned int BN = 64u;
    const unsigned int BK = 16u;
    const unsigned int TM = 4u;            /* BM / blockDim.y (64/16) */
    const unsigned int TN = 4u;            /* BN / blockDim.x (64/16) */

    /* Tile origin: blockIdx.x -> N (weight rows), blockIdx.y -> M (batch cols). */
    const unsigned int row_base = blockIdx.x * BN;   /* first weight row (N) */
    const unsigned int col_base = blockIdx.y * BM;   /* first batch column (M) */

    /* This thread owns a TM x TN micro-tile within the 64x64 output tile.
       (ty, tx) in [0,16)x[0,16); micro-tile rows = ty*TM.., cols = tx*TN.. */
    const unsigned int tx = threadIdx.x;             /* 0..15 -> N sub-block */
    const unsigned int ty = threadIdx.y;             /* 0..15 -> M sub-block */
    const unsigned int tid = ty * blockDim.x + tx;   /* 0..255 flat */

    /* Shared staging:
         As[m_local][kk] = inputs[(col_base+m_local)*K + k_off+kk]  (ld = BK)
         Ws[n_local][kk] = weights[(row_base+n_local)*K + k_off+kk] (ld = BK) */
    __shared__ float As[64u * 16u];   /* 4 KiB */
    __shared__ float Ws[64u * 16u];   /* 4 KiB */

    /* Per-thread f32 accumulators (4x4 micro-tile). */
    float acc[4u][4u];
    #pragma unroll
    for (unsigned int i = 0u; i < TM; ++i)
        #pragma unroll
        for (unsigned int j = 0u; j < TN; ++j)
            acc[i][j] = 0.0f;

    const unsigned int k_tiles = (k + BK - 1u) / BK;  /* ceil(K / BK) */

    for (unsigned int kt = 0u; kt < k_tiles; ++kt) {
        const unsigned int k_off = kt * BK;

        /* -- Cooperative load As (64*16 = 1024 floats, 4 per thread). -------- */
        #pragma unroll
        for (unsigned int r = 0u; r < (64u * 16u) / 256u; ++r) {
            const unsigned int idx = tid + r * 256u;     /* 0..1023 */
            const unsigned int a_row = idx / BK;         /* 0..63  -> M local */
            const unsigned int a_kk  = idx % BK;         /* 0..15  -> K local */
            const unsigned int g_col = col_base + a_row;
            const unsigned int g_k   = k_off + a_kk;
            float v = 0.0f;
            if (g_col < batch_size && g_k < k) {
                v = inputs[(unsigned long long)g_col * k + g_k];
            }
            As[a_row * BK + a_kk] = v;
        }

        /* -- Cooperative load Ws (64*16 = 1024 floats, 4 per thread). -------- */
        #pragma unroll
        for (unsigned int r = 0u; r < (64u * 16u) / 256u; ++r) {
            const unsigned int idx = tid + r * 256u;     /* 0..1023 */
            const unsigned int w_row = idx / BK;         /* 0..63  -> N local */
            const unsigned int w_kk  = idx % BK;         /* 0..15  -> K local */
            const unsigned int g_row = row_base + w_row;
            const unsigned int g_k   = k_off + w_kk;
            float v = 0.0f;
            if (g_row < n_rows && g_k < k) {
                v = weights[(unsigned long long)g_row * k + g_k];
            }
            Ws[w_row * BK + w_kk] = v;
        }

        __syncthreads();

        /* -- Micro-tile MACs over this BK slice (f32 accumulate). ----------- */
        #pragma unroll
        for (unsigned int kk = 0u; kk < BK; ++kk) {
            float a_reg[4u];
            float w_reg[4u];
            #pragma unroll
            for (unsigned int i = 0u; i < TM; ++i)
                a_reg[i] = As[(ty * TM + i) * BK + kk];
            #pragma unroll
            for (unsigned int j = 0u; j < TN; ++j)
                w_reg[j] = Ws[(tx * TN + j) * BK + kk];
            #pragma unroll
            for (unsigned int i = 0u; i < TM; ++i)
                #pragma unroll
                for (unsigned int j = 0u; j < TN; ++j)
                    acc[i][j] += a_reg[i] * w_reg[j];
        }

        __syncthreads();
    }

    /* -- Write back the 4x4 micro-tile (column-major outputs[col*N + row]).
       Clamp with m_local < M && n_local < N so out-of-range rows/cols of an
       edge tile are not written.  FULL-M: every column is covered by grid.y. */
    #pragma unroll
    for (unsigned int i = 0u; i < TM; ++i) {
        const unsigned int m_local = ty * TM + i;        /* 0..63 tile-local M */
        const unsigned int col = col_base + m_local;
        if (col >= batch_size) continue;
        #pragma unroll
        for (unsigned int j = 0u; j < TN; ++j) {
            const unsigned int n_local = tx * TN + j;    /* 0..63 tile-local N */
            const unsigned int row = row_base + n_local;
            if (row < n_rows) {
                outputs[(unsigned long long)col * n_rows + row] = acc[i][j];
            }
        }
    }
}

/* =========================================================================
   Kernel 2 — gemm_tq2
   Ternary (TQ2_0_g128) batched GEMM: out[M,N] = A[M,K] . dequant(B[N,K])^T.
   CUDA-core sibling of the Metal gemm_tq2_g128_v10_simdgroup
   (prefill_simdgroup_v10.rs): same SoA weight buffer, same decode bits, same
   f16-exact code*scale staging, but plain CUDA-core 4x4 micro-tile MACs with
   f32 accumulate (parity-first; no tensor cores).

   Buffers / conventions (column-major == row-major [M,K] / [M,N]):
     soa_raw : u8, TQ2_0_g128 SoA
                 [scales: N*(K/128) * 2 B FP16 LE]
                 [qs:     N*(K/128) * 32 B]   (4 codes/byte LSB-first)
                 block index b = row*(K/128) + kb
     inputs  : f32, [M, K]  -> inputs[col*K + e]   (col = m)
     outputs : f32, [M, N]  -> outputs[col*N + row] (row = n)
     n_rows  : N
     batch_size : M  (FULL — no cap-of-8)
     k       : inner dim, MUST be a multiple of 128

   Tiling (plain CUDA cores, f32 accumulate):
     output tile   BM x BN = 32 x 32
     K step        BK = 128  (exactly one ternary block per K-tile)
     block         (16, 16) = 256 threads
     micro-tile    2 x 2 per thread  (TM x TN; BM/16 = 2, BN/16 = 2)
     shared mem    As[32*128] + Ws[32*128] = 2 * 16 KiB = 32 KiB
     grid          (ceil(N/32), ceil(M/32), 1)
   Per K-tile: decode each in-tile weight row's 32 qs bytes -> 128 dequantized
   f32 (code*scale, EXACT in f16 -> staged as f32, parity-identical), stage the
   128 input elems, run 128 f32 MACs.  Boundary clamps (OOB rows/cols staged as
   zero) match v10.  FULL-M (grid.y + m_local clamp) — fixes the cap-of-8 trap.
   ========================================================================= */
extern "C" __global__ void gemm_tq2(
    const unsigned char* __restrict__ soa_raw,
    const float*         __restrict__ inputs,   /* [M, K] col-major (col*K + e) */
    float*               __restrict__ outputs,  /* [M, N] col-major (col*N + row) */
    unsigned int n_rows,                        /* N */
    unsigned int batch_size,                    /* M, FULL */
    unsigned int k                              /* %128 == 0 */
) {
    /* Wide register-blocked FP32 tile (parity-safe; NO tensor cores):
         output tile  BM x BN = 128 x 128
         K step       BK = 8        (the 128-wide ternary block = 16 K-tiles)
         block        (16,16) = 256 threads ; micro-tile TM x TN = 8 x 8 / thread
         shared       As[8*128] + Ws[8*128] = 2 * 4 KiB = 8 KiB, **TRANSPOSED**
                      ([kk][m] / [kk][n]) so the inner-MAC reads are contiguous
                      in m/n and bank-conflict-free across the warp.
         grid         (ceil(N/128), ceil(M/128), 1)
       This lifts the arithmetic intensity from the old 2x2 (4 MAC / 4 shared
       loads) to 8x8 (64 MAC / 16 shared loads = 4:1) — the dominant DiT-GEMM
       lever. f32 accumulate keeps cos >= 0.999. FULL-M (grid.y + col clamp). */
    const unsigned int BM = 128u;
    const unsigned int BN = 128u;
    const unsigned int BK = 8u;
    const unsigned int TM = 8u;            /* BM / blockDim.y (128/16) */
    const unsigned int TN = 8u;            /* BN / blockDim.x (128/16) */

    const unsigned int row_base = blockIdx.x * BN;   /* first weight row (N) */
    const unsigned int col_base = blockIdx.y * BM;   /* first batch column (M) */

    const unsigned int tx = threadIdx.x;             /* 0..15 -> N sub-block */
    const unsigned int ty = threadIdx.y;             /* 0..15 -> M sub-block */
    const unsigned int tid = ty * blockDim.x + tx;   /* 0..255 flat */

    /* SoA pointers (mirror Metal v10 / gemv_tq2_g128_v1):
         scales = (const unsigned short*)soa_raw
         qs     = soa_raw + (N * nblk) * 2
         block index for weight row n, K-block kb = n*nblk + kb. */
    const unsigned int nblk = k >> 7u;                       /* K / 128 */
    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)soa_raw;
    const unsigned char* __restrict__ qs_base =
        soa_raw + (unsigned long long)n_rows * nblk * 2u;

    /* TRANSPOSED shared tiles (leading dim = BM / BN):
         As[kk*BM + m_local] = inputs[(col_base+m_local)*K + k_off + kk]
         Ws[kk*BN + n_local] = dequant(B[row_base+n_local, k_off + kk]) */
    __shared__ float As[8u * 128u];   /* 4 KiB */
    __shared__ float Ws[8u * 128u];   /* 4 KiB */

    float acc[8u][8u];
    #pragma unroll
    for (unsigned int i = 0u; i < TM; ++i)
        #pragma unroll
        for (unsigned int j = 0u; j < TN; ++j)
            acc[i][j] = 0.0f;

    const unsigned int k_tiles = k / BK;   /* K / 8 */

    for (unsigned int kt = 0u; kt < k_tiles; ++kt) {
        const unsigned int k_off = kt * BK;

        /* -- Stage As transposed (128*8 = 1024 floats, 4 per thread). ------- */
        #pragma unroll
        for (unsigned int r = 0u; r < (128u * 8u) / 256u; ++r) {
            const unsigned int idx = tid + r * 256u;     /* 0..1023 */
            const unsigned int m_local = idx & 127u;     /* 0..127 -> M local */
            const unsigned int kk      = idx >> 7u;      /* 0..7   -> K local */
            const unsigned int g_col   = col_base + m_local;
            float v = 0.0f;
            if (g_col < batch_size) {
                v = inputs[(unsigned long long)g_col * k + k_off + kk];
            }
            As[kk * BM + m_local] = v;
        }

        /* -- Stage Ws transposed by decoding the ternary block. ------------
           This 8-wide K-tile = 2 qs bytes per weight row. 128 rows * 2 bytes =
           256 work-items, 1 per thread. The 128-block is constant within the
           tile (blk = k_off/128); the byte offset into it = (k_off%128)/4. */
        {
            const unsigned int blk         = k_off >> 7u;           /* kt / 16 */
            const unsigned int byte_in_blk = (k_off & 127u) >> 2u;  /* (kt%16)*2 */
            const unsigned int n_local = tid >> 1u;                 /* 0..127 -> N local */
            const unsigned int wb      = tid & 1u;                  /* which of the 2 bytes */
            const unsigned int e_local = wb * 4u;                   /* 0 or 4 (K within tile) */
            float scale = 0.0f;
            unsigned int byte = 0u;
            if (row_base + n_local < n_rows) {
                const unsigned long long b =
                    (unsigned long long)(row_base + n_local) * nblk + blk;
                scale = fast_fp16_to_float(scales[b]);
                byte  = (unsigned int)qs_base[b * 32u + byte_in_blk + wb];
            }
            /* (scale==0 for padded N rows -> Ws contributes 0.) */
            #pragma unroll
            for (unsigned int c = 0u; c < 4u; ++c) {
                Ws[(e_local + c) * BN + n_local] =
                    scale * decode_tq2((byte >> (2u * c)) & 3u);
            }
        }

        __syncthreads();

        /* -- 8x8 micro-tile MACs over this 8-wide K-tile (f32 accumulate).
           Transposed shared => As[e*BM + m] / Ws[e*BN + n] are contiguous in
           m/n, so the per-warp shared loads are bank-conflict-free. */
        #pragma unroll
        for (unsigned int e = 0u; e < BK; ++e) {
            float a_reg[8u];
            float w_reg[8u];
            #pragma unroll
            for (unsigned int i = 0u; i < TM; ++i)
                a_reg[i] = As[e * BM + ty * TM + i];
            #pragma unroll
            for (unsigned int j = 0u; j < TN; ++j)
                w_reg[j] = Ws[e * BN + tx * TN + j];
            #pragma unroll
            for (unsigned int i = 0u; i < TM; ++i)
                #pragma unroll
                for (unsigned int j = 0u; j < TN; ++j)
                    acc[i][j] += a_reg[i] * w_reg[j];
        }

        __syncthreads();
    }

    /* -- Write back the 8x8 micro-tile (column-major outputs[col*N + row]),
       clamped m_local < M && n_local < N. FULL-M: grid.y covers all columns. */
    #pragma unroll
    for (unsigned int i = 0u; i < TM; ++i) {
        const unsigned int m_local = ty * TM + i;        /* 0..127 tile-local M */
        const unsigned int col = col_base + m_local;
        if (col >= batch_size) continue;
        #pragma unroll
        for (unsigned int j = 0u; j < TN; ++j) {
            const unsigned int n_local = tx * TN + j;    /* 0..127 tile-local N */
            const unsigned int row = row_base + n_local;
            if (row < n_rows) {
                outputs[(unsigned long long)col * n_rows + row] = acc[i][j];
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::CUDA_IMAGEN_GEMM_SRC;

    /// The combined source must declare both GEMM entry points and re-paste the
    /// ternary decode helper (NVRTC compiles this string independently, so the
    /// helper has to be present in-string).
    #[test]
    fn src_has_entry_points() {
        assert!(
            CUDA_IMAGEN_GEMM_SRC.contains("gemm_f32"),
            "missing gemm_f32 entry point"
        );
        assert!(
            CUDA_IMAGEN_GEMM_SRC.contains("gemm_tq2"),
            "missing gemm_tq2 entry point"
        );
        assert!(
            CUDA_IMAGEN_GEMM_SRC.contains("decode_tq2"),
            "missing decode_tq2 device helper"
        );
    }
}
