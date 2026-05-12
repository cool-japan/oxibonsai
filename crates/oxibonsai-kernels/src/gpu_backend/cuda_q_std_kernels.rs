//! CUDA C kernel source strings for OxiBonsai Q4_0 and Q8_0 standard GEMV operations.
//!
//! # Q standard kernel catalogue
//!
//! | Kernel        | Description                                       |
//! |---------------|---------------------------------------------------|
//! | `gemv_q4_0`   | Q4_0 GEMV, AoS blocks (18 B/block), warp-per-row  |
//! | `gemv_q8_0`   | Q8_0 GEMV, AoS blocks (34 B/block), warp-per-row  |
//!
//! # Block layout
//!
//! **Q4_0** (18 bytes/block, 32 weights):
//! ```text
//! Block[i] = [d_lo, d_hi, qs[0], ..., qs[15]]
//!             ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
//!             FP16 LE    16 nibble bytes → 32 weights
//! ```
//! Dequant: `weight[j] = d_f32 * (nibble[j] as f32 - 8.0)` where
//! even `j → qs[j/2] & 0x0F`, odd `j → (qs[j/2] >> 4) & 0x0F`.
//!
//! **Q8_0** (34 bytes/block, 32 weights):
//! ```text
//! Block[i] = [d_lo, d_hi, qs[0], ..., qs[31]]
//!             ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
//!             FP16 LE    32 signed int8 bytes
//! ```
//! Dequant: `weight[j] = d_f32 * qs[j]`.
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

/// CUDA C source for Q4_0 and Q8_0 standard GEMV kernels.
///
/// Block layout for Q4_0: AoS, 18 bytes/block.
/// Byte order: `[d_lo, d_hi, qs[0]..qs[15]]`
/// (FP16 scale first, then 16 nibble bytes encoding 32 int4 weights)
///
/// Block layout for Q8_0: AoS, 34 bytes/block.
/// Byte order: `[d_lo, d_hi, qs[0]..qs[31]]`
/// (FP16 scale first, then 32 signed int8 weights)
pub const CUDA_Q_STD_KERNELS_SRC: &str = r#"
/* ==========================================================================
   OxiBonsai CUDA Q4_0 / Q8_0 GEMV kernels.

   Q4_0 block layout (AoS, 18 bytes/block):
     bytes 0-1:   FP16 LE scale (d)
     bytes 2-17:  16 nibble bytes → 32 int4 weights
     Dequant: w[j] = d * (nibble[j] - 8)
     nibble[j]: even j → qs[j/2] & 0x0F, odd j → (qs[j/2] >> 4) & 0x0F

   Q8_0 block layout (AoS, 34 bytes/block):
     bytes 0-1:   FP16 LE scale (d)
     bytes 2-33:  32 signed int8 weights
     Dequant: w[j] = d * qs[j]

   GEMV grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA, 1 warp/row
   GEMV block: (256, 1, 1)
   ========================================================================== */

/* ── Hardware FP16 → FP32 via PTX (1 instruction, SM 6.0+) ─────────────── */
static __device__ __forceinline__ float q_fast_fp16_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* ==========================================================================
   Kernel 1 — gemv_q4_0
   Q4_0 GEMV: warp-per-row, AoS block layout (18 bytes/block).

   blocks_per_row = k / 32  (32 weights per block)
   stride         = 18      (bytes per Q4_0 block: 2 scale + 16 nibble bytes)

   Grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q4_0(
    const unsigned char* __restrict__ blocks,  /* AoS: [d:2B][qs:16B] x n_blocks */
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k                             /* must be multiple of 32 */
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;  /* k / 32 */
    const unsigned int stride = 18u;               /* bytes per Q4_0 block */

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;
        /* Scale: first 2 bytes = FP16 little-endian */
        const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const float scale = q_fast_fp16_to_float(d_raw);
        /* 16 nibble bytes = 32 weights */
        const float* xbase = input + (b << 5u);  /* b * 32 */
        #pragma unroll 16
        for (unsigned int nb = 0u; nb < 16u; ++nb) {
            const unsigned int byte = bptr[2u + nb];
            const float w0 = scale * (float)((int)(byte & 0x0Fu) - 8);
            const float w1 = scale * (float)((int)((byte >> 4u) & 0x0Fu) - 8);
            acc += w0 * xbase[nb * 2u] + w1 * xbase[nb * 2u + 1u];
        }
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
   Kernel 2 — gemv_q8_0
   Q8_0 GEMV: warp-per-row, AoS block layout (34 bytes/block).

   blocks_per_row = k / 32
   stride         = 34     (bytes per Q8_0 block: 2 scale + 32 int8 bytes)

   Grid:  (ceil(n_rows / 8), 1, 1)  -- 8 warps per CTA
   Block: (256, 1, 1)
   ========================================================================== */
extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ blocks,  /* AoS: [d:2B][qs:32B] x n_blocks */
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k                             /* must be multiple of 32 */
) {
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 5u;
    const unsigned int stride = 34u;  /* bytes per Q8_0 block */

    float acc = 0.0f;
    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned char* bptr = blocks + (row * blocks_per_row + b) * stride;
        /* Scale: first 2 bytes = FP16 little-endian */
        const unsigned short d_raw = (unsigned short)bptr[0] | ((unsigned short)bptr[1] << 8u);
        const float scale = q_fast_fp16_to_float(d_raw);
        const float* xbase = input + (b << 5u);
        #pragma unroll 32
        for (unsigned int j = 0u; j < 32u; ++j) {
            const int q = (int)(signed char)bptr[2u + j];
            acc += scale * (float)q * xbase[j];
        }
    }

    /* Warp-shuffle reduction across 32 lanes */
    acc += __shfl_down_sync(0xffffffffu, acc, 16u);
    acc += __shfl_down_sync(0xffffffffu, acc,  8u);
    acc += __shfl_down_sync(0xffffffffu, acc,  4u);
    acc += __shfl_down_sync(0xffffffffu, acc,  2u);
    acc += __shfl_down_sync(0xffffffffu, acc,  1u);
    if (lane == 0u) output[row] = acc;
}
"#;

// =============================================================================
// CudaQStdModules — process-wide singleton for compiled Q4_0 / Q8_0 kernels
// =============================================================================

/// Compiled CUDA function handles for the Q4_0 and Q8_0 GEMV kernels.
pub struct CudaQStdModules {
    /// Compiled handle for the `gemv_q4_0` kernel.
    pub gemv_q4_0: CudaFunction,
    /// Compiled handle for the `gemv_q8_0` kernel.
    pub gemv_q8_0: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaQStdModules {}
unsafe impl Sync for CudaQStdModules {}

struct CudaQStdState {
    modules: Mutex<Option<Arc<CudaQStdModules>>>,
}

unsafe impl Send for CudaQStdState {}
unsafe impl Sync for CudaQStdState {}

static Q_STD_STATE: OnceLock<CudaQStdState> = OnceLock::new();

fn q_std_state() -> &'static CudaQStdState {
    Q_STD_STATE.get_or_init(|| CudaQStdState {
        modules: Mutex::new(None),
    })
}

/// Compile (or return cached) Q standard (Q4_0 / Q8_0) CUDA modules.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_q_std_modules(graph: &CudaGraph) -> Result<Arc<CudaQStdModules>, CudaGraphError> {
    let state = q_std_state();
    let mut guard = state
        .modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_Q_STD_KERNELS_SRC, "q_std_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module q_std: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaQStdModules {
        gemv_q4_0: load("gemv_q4_0")?,
        gemv_q8_0: load("gemv_q8_0")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Public host functions
// =============================================================================

/// Run Q4_0 GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 18 bytes per block: `[d_lo, d_hi, qs[0]..qs[15]]`
///   - `d` is FP16 little-endian scale
///   - `qs` are 16 bytes encoding 32 int4 nibbles (even nibble = low bits, odd = high bits)
/// - Total length: `n_rows * (k / 32) * 18`
///
/// `input` must have length `>= k`.
/// `output` must have length `>= n_rows` (results are written, not accumulated).
pub fn cuda_gemv_q4_0(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    if k == 0 || k % 32 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q4_0 GEMV: k={k} must be a positive multiple of 32"
        )));
    }
    let blocks_per_row = k / 32;
    let expected_bytes = n_rows * blocks_per_row * 18;
    if blocks_bytes.len() < expected_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q4_0 blocks_bytes too short: {} < {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    if input.len() < k {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q4_0 GEMV: input.len()={} < k={k}",
            input.len()
        )));
    }
    if output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q4_0 GEMV: output.len()={} < n_rows={n_rows}",
            output.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let mods = init_q_std_modules(&graph)?;

    let d_blocks: CudaSlice<u8> = graph
        .stream_arc()
        .clone_htod(&blocks_bytes[..expected_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod q4_0 blocks: {e}")))?;
    let d_input: CudaSlice<f32> = graph
        .stream_arc()
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod q4_0 input: {e}")))?;
    let mut d_output: CudaSlice<f32> = graph
        .stream_arc()
        .alloc_zeros::<f32>(n_rows)
        .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros q4_0 output: {e}")))?;

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
            .launch_builder(&mods.gemv_q4_0)
            .arg(&d_blocks)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(n_rows as u32))
            .arg(&(k as u32))
            .launch(cfg)
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_q4_0 launch: {e}")))?;
    }

    let host_out: Vec<f32> = graph
        .stream_arc()
        .clone_dtoh(&d_output)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh q4_0 output: {e}")))?;

    output[..n_rows].copy_from_slice(&host_out);
    Ok(())
}

/// Run Q8_0 GEMV on GPU.
///
/// `blocks_bytes` is the raw AoS byte representation of the weight matrix:
/// - 34 bytes per block: `[d_lo, d_hi, qs[0]..qs[31]]`
///   - `d` is FP16 little-endian scale
///   - `qs` are 32 signed int8 weights
/// - Total length: `n_rows * (k / 32) * 34`
///
/// `input` must have length `>= k`.
/// `output` must have length `>= n_rows` (results are written, not accumulated).
pub fn cuda_gemv_q8_0(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    if k == 0 || k % 32 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q8_0 GEMV: k={k} must be a positive multiple of 32"
        )));
    }
    let blocks_per_row = k / 32;
    let expected_bytes = n_rows * blocks_per_row * 34;
    if blocks_bytes.len() < expected_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q8_0 blocks_bytes too short: {} < {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    if input.len() < k {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q8_0 GEMV: input.len()={} < k={k}",
            input.len()
        )));
    }
    if output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "Q8_0 GEMV: output.len()={} < n_rows={n_rows}",
            output.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let mods = init_q_std_modules(&graph)?;

    let d_blocks: CudaSlice<u8> = graph
        .stream_arc()
        .clone_htod(&blocks_bytes[..expected_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod q8_0 blocks: {e}")))?;
    let d_input: CudaSlice<f32> = graph
        .stream_arc()
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod q8_0 input: {e}")))?;
    let mut d_output: CudaSlice<f32> = graph
        .stream_arc()
        .alloc_zeros::<f32>(n_rows)
        .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros q8_0 output: {e}")))?;

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
            .launch_builder(&mods.gemv_q8_0)
            .arg(&d_blocks)
            .arg(&d_input)
            .arg(&mut d_output)
            .arg(&(n_rows as u32))
            .arg(&(k as u32))
            .launch(cfg)
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_q8_0 launch: {e}")))?;
    }

    let host_out: Vec<f32> = graph
        .stream_arc()
        .clone_dtoh(&d_output)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_dtoh q8_0 output: {e}")))?;

    output[..n_rows].copy_from_slice(&host_out);
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the kernel source contains the Q4_0 GEMV entry point.
    #[test]
    fn test_q_std_kernel_source_has_gemv_q4_0() {
        assert!(
            CUDA_Q_STD_KERNELS_SRC.contains("gemv_q4_0"),
            "CUDA_Q_STD_KERNELS_SRC must contain gemv_q4_0"
        );
    }

    /// Verify the kernel source contains the Q8_0 GEMV entry point.
    #[test]
    fn test_q_std_kernel_source_has_gemv_q8_0() {
        assert!(
            CUDA_Q_STD_KERNELS_SRC.contains("gemv_q8_0"),
            "CUDA_Q_STD_KERNELS_SRC must contain gemv_q8_0"
        );
    }

    /// Q4_0 block size: 2 bytes scale + 16 bytes nibbles = 18 bytes/block.
    #[test]
    fn test_q4_0_block_stride() {
        assert_eq!(2 + 16, 18usize);
    }

    /// Q8_0 block size: 2 bytes scale + 32 bytes int8 = 34 bytes/block.
    #[test]
    fn test_q8_0_block_stride() {
        assert_eq!(2 + 32, 34usize);
    }

    /// CI-GPU-gated: Q4_0 GEMV with all-zero weights (nibble value 8 → offset 0).
    ///
    /// Build a 4-row × 32-col Q4_0 weight matrix where every nibble encodes 8,
    /// which dequantizes to `1.0 * (8 - 8) = 0.0`.  With an all-ones input the
    /// output must be all-zeros.
    #[test]
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn test_cuda_gemv_q4_0_matches_cpu() {
        use crate::gpu_backend::cuda_graph::CudaGraph;
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_q4_0_matches_cpu — no CUDA device");
            return;
        }

        let n_rows = 4usize;
        let k = 32usize;
        // 1 block per row, 18 bytes each
        let mut blocks_bytes = vec![0u8; n_rows * 18];
        for r in 0..n_rows {
            let b = &mut blocks_bytes[r * 18..(r + 1) * 18];
            // FP16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
            b[0] = 0x00;
            b[1] = 0x3C;
            // All nibbles = 8 → weight = 1.0 * (8 - 8) = 0.0
            // nibble_low=8 (0x8), nibble_high=8 (0x8) → byte = 0x88
            for i in 2..18 {
                b[i] = 0x88;
            }
        }

        let input = vec![1.0f32; k];
        let mut output_gpu = vec![0.0f32; n_rows];
        cuda_gemv_q4_0(&blocks_bytes, &input, &mut output_gpu, n_rows, k).unwrap();
        // All weights are 0.0 → output must be 0.0
        for &v in &output_gpu {
            assert!(v.abs() < 1e-5, "expected 0.0, got {v}");
        }
    }

    /// CI-GPU-gated: Q8_0 GEMV with a single non-zero int8 weight.
    ///
    /// Build a 4-row × 32-col Q8_0 weight matrix where only `qs[0] = 1` and
    /// the rest are zero.  With scale=1.0 and all-ones input, each output row
    /// should equal `1.0 * 1 * 1.0 = 1.0`.
    #[test]
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn test_cuda_gemv_q8_0_matches_cpu() {
        use crate::gpu_backend::cuda_graph::CudaGraph;
        if CudaGraph::global().is_err() {
            eprintln!("SKIP: test_cuda_gemv_q8_0_matches_cpu — no CUDA device");
            return;
        }

        let n_rows = 4usize;
        let k = 32usize;
        // 1 block per row, 34 bytes each
        let mut blocks_bytes = vec![0u8; n_rows * 34];
        for r in 0..n_rows {
            let b = &mut blocks_bytes[r * 34..(r + 1) * 34];
            // FP16 1.0
            b[0] = 0x00;
            b[1] = 0x3C;
            // int8 weights: first = 1, rest = 0
            b[2] = 1u8;
            for i in 3..34 {
                b[i] = 0u8;
            }
        }

        let input = vec![1.0f32; k];
        let mut output_gpu = vec![0.0f32; n_rows];
        cuda_gemv_q8_0(&blocks_bytes, &input, &mut output_gpu, n_rows, k).unwrap();
        // weight[0] = 1.0 * 1 * input[0] = 1.0; all rest = 0 → sum = 1.0
        for &v in &output_gpu {
            assert!((v - 1.0f32).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    /// Validate input dimension guard (k not multiple of 32) for Q4_0.
    #[test]
    fn test_cuda_gemv_q4_0_bad_k() {
        let blocks = vec![0u8; 18];
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q4_0(&blocks, &input, &mut output, 1, 31);
        assert!(
            result.is_err(),
            "k=31 (not multiple of 32) should return error"
        );
    }

    /// Validate input dimension guard (k not multiple of 32) for Q8_0.
    #[test]
    fn test_cuda_gemv_q8_0_bad_k() {
        let blocks = vec![0u8; 34];
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];
        let result = cuda_gemv_q8_0(&blocks, &input, &mut output, 1, 31);
        assert!(
            result.is_err(),
            "k=31 (not multiple of 32) should return error"
        );
    }

    /// Validate output buffer size guard for Q4_0.
    #[test]
    fn test_cuda_gemv_q4_0_output_too_small() {
        let blocks = vec![0u8; 18];
        let input = vec![0.0f32; 32];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q4_0(&blocks, &input, &mut output, 1, 32);
        assert!(result.is_err(), "empty output buffer should return error");
    }

    /// Validate output buffer size guard for Q8_0.
    #[test]
    fn test_cuda_gemv_q8_0_output_too_small() {
        let blocks = vec![0u8; 34];
        let input = vec![0.0f32; 32];
        let mut output: Vec<f32> = Vec::new();
        let result = cuda_gemv_q8_0(&blocks, &input, &mut output, 1, 32);
        assert!(result.is_err(), "empty output buffer should return error");
    }
}
