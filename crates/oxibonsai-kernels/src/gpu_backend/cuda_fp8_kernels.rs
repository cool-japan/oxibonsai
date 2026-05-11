//! CUDA C kernel source strings for OxiBonsai FP8 GEMV operations.
//!
//! # FP8 kernel catalogue
//!
//! | Kernel              | Description                                    |
//! |---------------------|------------------------------------------------|
//! | `gemv_fp8_e4m3`     | FP8 E4M3FN GEMV, AoS blocks, warp-per-row     |
//! | `gemv_fp8_e5m2`     | FP8 E5M2 GEMV, AoS blocks, warp-per-row       |
//! | `dequant_fp8_e4m3`  | FP8 E4M3FN block dequantization               |
//! | `dequant_fp8_e5m2`  | FP8 E5M2 block dequantization                 |
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! Block[i] = [q0, q1, ..., q31, scale_lo, scale_hi]
//!             ^^^^^^^^^^^^^^^^^ 32 FP8 bytes ^^^^   ^^ FP16 LE scale ^^
//! ```
//!
//! This matches the `#[repr(C)]` layout of `BlockFP8E4M3 { qs: [u8; 32], d: f16 }`:
//! `qs` occupies bytes 0-31, `d` (FP16) occupies bytes 32-33.
//!
//! # Grid / block dimensions
//!
//! Both GEMV kernels use:
//! - Grid:  `(ceil(n_rows / 8), 1, 1)` — 8 warps per CTA, one warp per output row
//! - Block: `(256, 1, 1)` — 8 warps × 32 lanes

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};

// =============================================================================
// CUDA C kernel source
// =============================================================================

/// CUDA C source for FP8 GEMV and dequantization kernels.
///
/// Block layout: AoS, 34 bytes/block.
/// Byte order: `[q0..q31, scale_lo, scale_hi]`
/// (weights first, FP16 scale at bytes 32-33 — matches `BlockFP8E4M3/E5M2 repr(C)`)
pub const CUDA_FP8_KERNELS_SRC: &str = r#"
/* ==========================================================================
   OxiBonsai CUDA FP8 GEMV + dequant kernels.

   Block layout (AoS, 34 bytes/block):
     bytes  0-31: 32 FP8 quantized weights
     bytes 32-33: FP16 LE block scale

   This matches BlockFP8E4M3 / BlockFP8E5M2 #[repr(C)] layout:
     struct { qs: [u8; 32], d: f16 }

   GEMV grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA, 1 warp/row
   GEMV block: (256, 1, 1)
   ========================================================================== */

/* ── Hardware FP16 → FP32 via PTX (1 instruction, SM 6.0+) ─────────────── */
static __device__ __forceinline__ float fp8_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ── FP8 E4M3FN decode (OFP8, bias=7, 4-bit exp, 3-bit mantissa) ─────────
   Format: s[7] exp[6:3] man[2:0], bias=7
   Normal:  (-1)^s * 2^(exp-7) * (1 + man/8)
   Denorm:  (-1)^s * 2^(-6) * (man/8)
   NaN:     exp=0b1111 AND man=0b111 (patterns 0x7f, 0xff) → 0 for inference
   ─────────────────────────────────────────────────────────────────────────── */
static __device__ __forceinline__ float fp8_e4m3_to_float(unsigned char b) {
    /* NaN patterns: 0x7f and 0xff → treat as 0 for inference */
    if (b == 0x7Fu || b == 0xFFu) return 0.0f;
    const unsigned int sign = (b >> 7u) & 1u;
    const unsigned int exp  = (b >> 3u) & 15u;  /* 4-bit exponent */
    const unsigned int mant = b & 7u;            /* 3-bit mantissa */
    float val;
    if (exp == 0u) {
        /* Denormal: (-1)^s * 2^(-6) * (mant/8) */
        val = (float)mant * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        /* Normal: 2^(exp-7) * (1 + mant/8)
           Assemble as IEEE-754 f32: ((exp - 7 + 127) << 23) | (mant << 20) */
        val = __int_as_float(((exp - 7u + 127u) << 23u) | (mant << 20u));
    }
    return sign ? -val : val;
}

/* ── FP8 E5M2 decode (standard, bias=15, 5-bit exp, 2-bit mantissa) ──────
   Format: s[7] exp[6:2] man[1:0], bias=15
   Normal:  (-1)^s * 2^(exp-15) * (1 + man/4)
   Denorm:  (-1)^s * 2^(-14) * (man/4)
   Inf/NaN: exp=31 → 0 for inference
   ─────────────────────────────────────────────────────────────────────────── */
static __device__ __forceinline__ float fp8_e5m2_to_float(unsigned char b) {
    const unsigned int exp  = (b >> 2u) & 31u;  /* 5-bit exponent */
    const unsigned int mant = b & 3u;            /* 2-bit mantissa */
    if (exp == 31u) return 0.0f;                 /* Inf / NaN → 0 */
    const unsigned int sign = (b >> 7u) & 1u;
    float val;
    if (exp == 0u) {
        /* Denormal: (-1)^s * 2^(-14) * (mant/4) */
        val = (float)mant * (1.0f / 4.0f) * (1.0f / 16384.0f);
    } else {
        /* Normal: 2^(exp-15) * (1 + mant/4)
           Assemble as IEEE-754 f32: ((exp - 15 + 127) << 23) | (mant << 21) */
        val = __int_as_float(((exp - 15u + 127u) << 23u) | (mant << 21u));
    }
    return sign ? -val : val;
}

/* ==========================================================================
   Kernel 1 — gemv_fp8_e4m3
   FP8 E4M3FN GEMV.

   Block layout: [q0..q31, scale_lo, scale_hi]
   blocks_per_row = k / 32  (32 weights per block)
   block_idx      = row * blocks_per_row + b

   Grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_fp8_e4m3(
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

    const unsigned int blocks_per_row = k >> 5u;  /* k / 32 */
    float acc = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned int block_idx = row * blocks_per_row + b;
        const unsigned int base_byte = block_idx * 34u;

        /* Scale is at bytes 32-33 (after the 32 weight bytes) */
        const unsigned short scale_bits =
            (unsigned short)(blocks[base_byte + 32u])
          | ((unsigned short)(blocks[base_byte + 33u]) << 8u);
        const float scale = fp8_fp16_to_float(scale_bits);

        /* Dot product: 32 FP8 weights at bytes 0-31 */
        const unsigned int inp_base = b * 32u;
        float block_sum = 0.0f;
        #pragma unroll 8
        for (unsigned int w = 0u; w < 32u; ++w) {
            block_sum += fp8_e4m3_to_float(blocks[base_byte + w]) * input[inp_base + w];
        }
        acc += scale * block_sum;
    }

    /* Warp-shuffle reduction across 32 lanes */
    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 2 — gemv_fp8_e5m2
   FP8 E5M2 GEMV.  Identical structure to gemv_fp8_e4m3.
   ========================================================================== */
extern "C" __global__ void gemv_fp8_e5m2(
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

    const unsigned int blocks_per_row = k >> 5u;
    float acc = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned int block_idx = row * blocks_per_row + b;
        const unsigned int base_byte = block_idx * 34u;

        /* Scale at bytes 32-33 */
        const unsigned short scale_bits =
            (unsigned short)(blocks[base_byte + 32u])
          | ((unsigned short)(blocks[base_byte + 33u]) << 8u);
        const float scale = fp8_fp16_to_float(scale_bits);

        const unsigned int inp_base = b * 32u;
        float block_sum = 0.0f;
        #pragma unroll 8
        for (unsigned int w = 0u; w < 32u; ++w) {
            block_sum += fp8_e5m2_to_float(blocks[base_byte + w]) * input[inp_base + w];
        }
        acc += scale * block_sum;
    }

    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}

/* ==========================================================================
   Kernel 3 — dequant_fp8_e4m3
   Dequantize all FP8 E4M3FN weights to f32.

   Each thread handles one weight from the flattened weight matrix (n_rows × k).
   gid = row * k + col  (global weight index)

   Grid:  (ceil(n_rows * k / 256), 1, 1)
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void dequant_fp8_e4m3(
    const unsigned char* __restrict__ blocks,
    float*               __restrict__ output,
    unsigned int n_elements,   /* = n_rows * k */
    unsigned int k
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid >= n_elements) return;

    const unsigned int col   = gid % k;
    const unsigned int row   = gid / k;
    const unsigned int b     = col >> 5u;      /* col / 32 */
    const unsigned int w_idx = col & 31u;      /* col % 32 */
    const unsigned int blocks_per_row = k >> 5u;
    const unsigned int block_idx  = row * blocks_per_row + b;
    const unsigned int base_byte  = block_idx * 34u;

    /* Scale at bytes 32-33 */
    const unsigned short scale_bits =
        (unsigned short)(blocks[base_byte + 32u])
      | ((unsigned short)(blocks[base_byte + 33u]) << 8u);
    const float scale = fp8_fp16_to_float(scale_bits);

    output[gid] = scale * fp8_e4m3_to_float(blocks[base_byte + w_idx]);
}

/* ==========================================================================
   Kernel 4 — dequant_fp8_e5m2
   Dequantize all FP8 E5M2 weights to f32.
   ========================================================================== */
extern "C" __global__ void dequant_fp8_e5m2(
    const unsigned char* __restrict__ blocks,
    float*               __restrict__ output,
    unsigned int n_elements,
    unsigned int k
) {
    const unsigned int gid = blockIdx.x * 256u + threadIdx.x;
    if (gid >= n_elements) return;

    const unsigned int col   = gid % k;
    const unsigned int row   = gid / k;
    const unsigned int b     = col >> 5u;
    const unsigned int w_idx = col & 31u;
    const unsigned int blocks_per_row = k >> 5u;
    const unsigned int block_idx  = row * blocks_per_row + b;
    const unsigned int base_byte  = block_idx * 34u;

    const unsigned short scale_bits =
        (unsigned short)(blocks[base_byte + 32u])
      | ((unsigned short)(blocks[base_byte + 33u]) << 8u);
    const float scale = fp8_fp16_to_float(scale_bits);

    output[gid] = scale * fp8_e5m2_to_float(blocks[base_byte + w_idx]);
}
"#;

// =============================================================================
// CudaFp8Modules — process-wide singleton for compiled FP8 kernels
// =============================================================================

/// Compiled CUDA function handles for the 4 FP8 kernels.
pub struct CudaFp8Modules {
    pub gemv_e4m3: CudaFunction,
    pub gemv_e5m2: CudaFunction,
    pub dequant_e4m3: CudaFunction,
    pub dequant_e5m2: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaFp8Modules {}
unsafe impl Sync for CudaFp8Modules {}

struct CudaFp8State {
    modules: Mutex<Option<Arc<CudaFp8Modules>>>,
}

unsafe impl Send for CudaFp8State {}
unsafe impl Sync for CudaFp8State {}

static FP8_STATE: OnceLock<CudaFp8State> = OnceLock::new();

fn fp8_state() -> &'static CudaFp8State {
    FP8_STATE.get_or_init(|| CudaFp8State {
        modules: Mutex::new(None),
    })
}

/// Compile (or return cached) FP8 CUDA modules.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_fp8_modules(graph: &CudaGraph) -> Result<Arc<CudaFp8Modules>, CudaGraphError> {
    let state = fp8_state();
    let mut guard = state
        .modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_FP8_KERNELS_SRC, "fp8_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module fp8: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaFp8Modules {
        gemv_e4m3: load("gemv_fp8_e4m3")?,
        gemv_e5m2: load("gemv_fp8_e5m2")?,
        dequant_e4m3: load("dequant_fp8_e4m3")?,
        dequant_e5m2: load("dequant_fp8_e5m2")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Public host functions
// =============================================================================

/// Run FP8 E4M3FN GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 34 bytes per block: `[q0..q31, scale_lo, scale_hi]`
/// - Total length: `n_rows * (k / 32) * 34`
///
/// `input` must have length `>= k`.
/// `output` must have length `>= n_rows` (results are written, not accumulated).
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemv_fp8_e4m3(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    if k == 0 || k % 32 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E4M3 GEMV: k={k} must be a positive multiple of 32"
        )));
    }
    let blocks_per_row = k / 32;
    let expected_bytes = n_rows * blocks_per_row * 34;
    if blocks_bytes.len() < expected_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E4M3 blocks_bytes too short: {} < {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    if input.len() < k {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E4M3 GEMV: input.len()={} < k={k}",
            input.len()
        )));
    }
    if output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E4M3 GEMV: output.len()={} < n_rows={n_rows}",
            output.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let mods = init_fp8_modules(&graph)?;

    let d_blocks: CudaSlice<u8> = graph
        .stream_arc()
        .clone_htod(&blocks_bytes[..expected_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod fp8 blocks: {e}")))?;
    let d_input: CudaSlice<f32> = graph
        .stream_arc()
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod fp8 input: {e}")))?;
    let mut d_output: CudaSlice<f32> = graph
        .stream_arc()
        .alloc_zeros::<f32>(n_rows)
        .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros fp8 output: {e}")))?;

    let grid_x = (n_rows as u32).div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: kernel arguments match the CUDA kernel signature; all device
    // buffers are valid on the graph stream and have the correct element counts.
    unsafe {
        graph
            .stream_arc()
            .launch_builder(&mods.gemv_e4m3)
            .arg(&d_blocks)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(n_rows as u32))
            .arg(&(k as u32))
            .launch(cfg)
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_fp8_e4m3 launch: {e}")))?;
    }

    let host_out: Vec<f32> = graph
        .stream_arc()
        .clone_dtoh(&d_output)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh fp8 output: {e}")))?;

    output[..n_rows].copy_from_slice(&host_out);
    Ok(())
}

/// Run FP8 E5M2 GEMV on GPU.
///
/// Same semantics as [`cuda_gemv_fp8_e4m3`] but for E5M2 format.
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemv_fp8_e5m2(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    if k == 0 || k % 32 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E5M2 GEMV: k={k} must be a positive multiple of 32"
        )));
    }
    let blocks_per_row = k / 32;
    let expected_bytes = n_rows * blocks_per_row * 34;
    if blocks_bytes.len() < expected_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E5M2 blocks_bytes too short: {} < {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    if input.len() < k {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E5M2 GEMV: input.len()={} < k={k}",
            input.len()
        )));
    }
    if output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "FP8 E5M2 GEMV: output.len()={} < n_rows={n_rows}",
            output.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let mods = init_fp8_modules(&graph)?;

    let d_blocks: CudaSlice<u8> = graph
        .stream_arc()
        .clone_htod(&blocks_bytes[..expected_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod fp8 blocks: {e}")))?;
    let d_input: CudaSlice<f32> = graph
        .stream_arc()
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod fp8 input: {e}")))?;
    let mut d_output: CudaSlice<f32> = graph
        .stream_arc()
        .alloc_zeros::<f32>(n_rows)
        .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros fp8 output: {e}")))?;

    let grid_x = (n_rows as u32).div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: kernel arguments match the CUDA kernel signature; all device
    // buffers are valid on the graph stream and have the correct element counts.
    unsafe {
        graph
            .stream_arc()
            .launch_builder(&mods.gemv_e5m2)
            .arg(&d_blocks)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(n_rows as u32))
            .arg(&(k as u32))
            .launch(cfg)
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_fp8_e5m2 launch: {e}")))?;
    }

    let host_out: Vec<f32> = graph
        .stream_arc()
        .clone_dtoh(&d_output)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh fp8 output: {e}")))?;

    output[..n_rows].copy_from_slice(&host_out);
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the kernel source contains the E4M3 GEMV entry point.
    #[test]
    fn test_cuda_fp8_kernel_source_has_gemv_e4m3() {
        assert!(
            CUDA_FP8_KERNELS_SRC.contains("gemv_fp8_e4m3"),
            "CUDA_FP8_KERNELS_SRC must contain gemv_fp8_e4m3"
        );
    }

    /// Verify the kernel source contains the E5M2 GEMV entry point.
    #[test]
    fn test_cuda_fp8_kernel_source_has_gemv_e5m2() {
        assert!(
            CUDA_FP8_KERNELS_SRC.contains("gemv_fp8_e5m2"),
            "CUDA_FP8_KERNELS_SRC must contain gemv_fp8_e5m2"
        );
    }

    /// Verify the kernel source contains both dequantization entry points.
    #[test]
    fn test_cuda_fp8_kernel_source_has_dequant() {
        assert!(
            CUDA_FP8_KERNELS_SRC.contains("dequant_fp8_e4m3"),
            "CUDA_FP8_KERNELS_SRC must contain dequant_fp8_e4m3"
        );
        assert!(
            CUDA_FP8_KERNELS_SRC.contains("dequant_fp8_e5m2"),
            "CUDA_FP8_KERNELS_SRC must contain dequant_fp8_e5m2"
        );
    }

    /// Verify the block layout comment documents the correct byte order.
    #[test]
    fn test_cuda_fp8_kernel_source_documents_layout() {
        /* The kernel source must document that scale is at bytes 32-33 (weights-first layout). */
        assert!(
            CUDA_FP8_KERNELS_SRC.contains("base_byte + 32u"),
            "CUDA_FP8_KERNELS_SRC must access scale at byte offset 32"
        );
    }

    /// CI-GPU-gated: CPU vs GPU FP8 E4M3 GEMV parity.
    ///
    /// Uses a 16-row × 128-column weight matrix (4 blocks/row, 64 blocks total).
    /// Each block: scale=1.0 (FP16), alternating weights 0x38 (E4M3=1.0) and 0x00 (E4M3=0.0).
    /// Input: all-ones.
    /// Expected: each output row = 16 * (1.0 * 1.0 + 1.0 * 0.0) * 4 blocks = 64.0
    ///           (16 non-zero weights per block × 4 blocks/row × scale=1.0)
    #[test]
    fn test_cuda_gemv_fp8_e4m3_matches_cpu() {
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_fp8_e4m3_matches_cpu — no CUDA device");
            return;
        }

        let n_rows = 16usize;
        let k = 128usize;
        let blocks_per_row = k / 32;
        let total_blocks = n_rows * blocks_per_row;

        /* Build blocks in AoS layout: [q0..q31, scale_lo, scale_hi] */
        let mut blocks_bytes = vec![0u8; total_blocks * 34];
        let scale_bits = half::f16::from_f32(1.0f32).to_bits().to_le_bytes();

        for i in 0..total_blocks {
            let base = i * 34;
            /* Weights at bytes 0-31: alternating 0x38 (E4M3 = 1.0) and 0x00 (E4M3 = 0.0) */
            for w in 0..32usize {
                blocks_bytes[base + w] = if w % 2 == 0 { 0x38u8 } else { 0x00u8 };
            }
            /* Scale at bytes 32-33 */
            blocks_bytes[base + 32] = scale_bits[0];
            blocks_bytes[base + 33] = scale_bits[1];
        }

        let input = vec![1.0f32; k];

        /* CPU reference */
        let mut cpu_out = vec![0.0f32; n_rows];
        crate::gemv_fp8::gemv_fp8_e4m3(
            oxibonsai_core::BlockFP8E4M3::slice_from_bytes(&blocks_bytes)
                .expect("cpu slice_from_bytes"),
            &input,
            &mut cpu_out,
            n_rows,
            k,
        )
        .expect("CPU fp8 e4m3 gemv");

        /* GPU */
        let mut gpu_out = vec![0.0f32; n_rows];
        cuda_gemv_fp8_e4m3(&blocks_bytes, &input, &mut gpu_out, n_rows, k)
            .expect("GPU fp8 e4m3 gemv");

        for (i, (c, g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-3,
                "E4M3 row {i}: cpu={c}, gpu={g}, diff={}",
                (c - g).abs()
            );
        }
    }

    /// CI-GPU-gated: CPU vs GPU FP8 E5M2 GEMV parity.
    ///
    /// Same structure as the E4M3 test but using E5M2 encoding.
    /// 0x3C = E5M2 1.0 (exp=15, man=0 → 2^(15-15) × 1.0 = 1.0).
    #[test]
    fn test_cuda_gemv_fp8_e5m2_matches_cpu() {
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_fp8_e5m2_matches_cpu — no CUDA device");
            return;
        }

        let n_rows = 16usize;
        let k = 128usize;
        let blocks_per_row = k / 32;
        let total_blocks = n_rows * blocks_per_row;

        let mut blocks_bytes = vec![0u8; total_blocks * 34];
        let scale_bits = half::f16::from_f32(1.0f32).to_bits().to_le_bytes();

        for i in 0..total_blocks {
            let base = i * 34;
            for w in 0..32usize {
                /* 0x3C = E5M2 1.0 */
                blocks_bytes[base + w] = if w % 2 == 0 { 0x3Cu8 } else { 0x00u8 };
            }
            blocks_bytes[base + 32] = scale_bits[0];
            blocks_bytes[base + 33] = scale_bits[1];
        }

        let input = vec![1.0f32; k];

        let mut cpu_out = vec![0.0f32; n_rows];
        crate::gemv_fp8::gemv_fp8_e5m2(
            oxibonsai_core::BlockFP8E5M2::slice_from_bytes(&blocks_bytes)
                .expect("cpu slice_from_bytes"),
            &input,
            &mut cpu_out,
            n_rows,
            k,
        )
        .expect("CPU fp8 e5m2 gemv");

        let mut gpu_out = vec![0.0f32; n_rows];
        cuda_gemv_fp8_e5m2(&blocks_bytes, &input, &mut gpu_out, n_rows, k)
            .expect("GPU fp8 e5m2 gemv");

        for (i, (c, g)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-3,
                "E5M2 row {i}: cpu={c}, gpu={g}, diff={}",
                (c - g).abs()
            );
        }
    }

    /// Validate input dimension guard (k not multiple of 32).
    #[test]
    fn test_cuda_gemv_fp8_e4m3_bad_k() {
        let blocks = vec![0u8; 34];
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 31);
        assert!(
            result.is_err(),
            "k=31 (not multiple of 32) should return error"
        );
    }

    /// Validate output buffer size guard.
    #[test]
    fn test_cuda_gemv_fp8_e5m2_output_too_small() {
        let blocks = vec![0u8; 34];
        let input = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 0];
        let result = cuda_gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 32);
        assert!(result.is_err(), "empty output buffer should return error");
    }

    /// Verify FP8 block types have the expected AoS layout size (34 bytes each).
    ///
    /// This matches the CUDA kernel's hard-coded block stride of 34 bytes
    /// (`BLOCK_BYTES 34` in the CUDA C source). If this size ever changes,
    /// the CUDA kernel source must be updated to match.
    #[test]
    fn test_fp8_block_accessors_exist() {
        use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2};
        assert_eq!(
            std::mem::size_of::<BlockFP8E4M3>(),
            34,
            "BlockFP8E4M3 must be 34 bytes (32 FP8 weights + 2-byte FP16 scale)"
        );
        assert_eq!(
            std::mem::size_of::<BlockFP8E5M2>(),
            34,
            "BlockFP8E5M2 must be 34 bytes (32 FP8 weights + 2-byte FP16 scale)"
        );
    }
}
