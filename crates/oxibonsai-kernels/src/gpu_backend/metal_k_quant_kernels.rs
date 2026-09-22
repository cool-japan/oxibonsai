//! Direct Metal dispatch engine for OxiBonsai K-quant GEMV
//! (`Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_K`).
//!
//! Metal counterpart of `cuda_k_quant_kernels.rs`, mirroring the Phase 27
//! `metal_fp8_kernels.rs` architecture:
//!
//! - Independent singleton (own [`metal::Device`] + [`metal::CommandQueue`]).
//! - Six compute pipelines, compiled lazily from MSL source at first use (no
//!   offline Metal Toolchain required).
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`).
//!
//! All K-quant formats use `QK_K = 256` weights per super-block, so `k` must be
//! a positive multiple of 256.
//!
//! # Public API
//!
//! - [`metal_gemv_q2k`] / [`metal_gemv_q3k`] / [`metal_gemv_q4k`]
//! - [`metal_gemv_q5k`] / [`metal_gemv_q6k`] / [`metal_gemv_q8k`]

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::OnceLock;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use super::kernel_sources::{
    MSL_GEMV_Q2K_V1, MSL_GEMV_Q3K_V1, MSL_GEMV_Q4K_V1, MSL_GEMV_Q5K_V1, MSL_GEMV_Q6K_V1,
    MSL_GEMV_Q8K_V1,
};
use super::metal_graph::{commit_and_wait, MetalGraphError};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Weights per K-quant super-block (`QK_K`).
const QK_K: usize = 256;
/// Bytes per `Q2_K` super-block.
const Q2K_BLOCK_BYTES: usize = 84;
/// Bytes per `Q3_K` super-block.
const Q3K_BLOCK_BYTES: usize = 110;
/// Bytes per `Q4_K` super-block.
const Q4K_BLOCK_BYTES: usize = 144;
/// Bytes per `Q5_K` super-block.
const Q5K_BLOCK_BYTES: usize = 176;
/// Bytes per `Q6_K` super-block.
const Q6K_BLOCK_BYTES: usize = 210;
/// Bytes per `Q8_K` super-block.
const Q8K_BLOCK_BYTES: usize = 292;
/// Simdgroups per threadgroup (one output row per simdgroup).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

// ═══════════════════════════════════════════════════════════════════════════
// Singleton state
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide Metal K-quant dispatch state (six compiled pipelines).
struct MetalKQuantState {
    device: Device,
    queue: CommandQueue,
    pipeline_q2k: ComputePipelineState,
    pipeline_q3k: ComputePipelineState,
    pipeline_q4k: ComputePipelineState,
    pipeline_q5k: ComputePipelineState,
    pipeline_q6k: ComputePipelineState,
    pipeline_q8k: ComputePipelineState,
}

// SAFETY: `metal::Device` / `metal::CommandQueue` are reference-counted
// Objective-C objects documented by Apple as safe to share across threads once
// initialised. This mirrors `metal_fp8_kernels::MetalFp8State`.
unsafe impl Send for MetalKQuantState {}
unsafe impl Sync for MetalKQuantState {}

impl MetalKQuantState {
    fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();

        Ok(Self {
            pipeline_q2k: compile_pipeline(&device, &options, MSL_GEMV_Q2K_V1, "gemv_q2k")?,
            pipeline_q3k: compile_pipeline(&device, &options, MSL_GEMV_Q3K_V1, "gemv_q3k")?,
            pipeline_q4k: compile_pipeline(&device, &options, MSL_GEMV_Q4K_V1, "gemv_q4k")?,
            pipeline_q5k: compile_pipeline(&device, &options, MSL_GEMV_Q5K_V1, "gemv_q5k")?,
            pipeline_q6k: compile_pipeline(&device, &options, MSL_GEMV_Q6K_V1, "gemv_q6k")?,
            pipeline_q8k: compile_pipeline(&device, &options, MSL_GEMV_Q8K_V1, "gemv_q8k")?,
            device,
            queue,
        })
    }
}

/// Compile one MSL source string into a named compute pipeline.
fn compile_pipeline(
    device: &Device,
    options: &CompileOptions,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalGraphError> {
    let library = device
        .new_library_with_source(source, options)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} library: {e}")))?;
    let function = library
        .get_function(entry, None)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} function: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} pipeline: {e}")))
}

/// Lazy process-wide singleton.
fn state() -> Result<&'static MetalKQuantState, MetalGraphError> {
    static STATE: OnceLock<Result<MetalKQuantState, MetalGraphError>> = OnceLock::new();
    match STATE.get_or_init(MetalKQuantState::new) {
        Ok(s) => Ok(s),
        Err(e) => Err(clone_err(e)),
    }
}

fn clone_err(e: &MetalGraphError) -> MetalGraphError {
    match e {
        MetalGraphError::DeviceNotFound => MetalGraphError::DeviceNotFound,
        MetalGraphError::CompilationFailed(s) => MetalGraphError::CompilationFailed(s.clone()),
        MetalGraphError::BufferCreationFailed => MetalGraphError::BufferCreationFailed,
        MetalGraphError::EncodingFailed(s) => MetalGraphError::EncodingFailed(s.clone()),
        MetalGraphError::ExecutionFailed(s) => MetalGraphError::ExecutionFailed(s.clone()),
        MetalGraphError::InvalidDimensions(s) => MetalGraphError::InvalidDimensions(s.clone()),
        MetalGraphError::CommandBufferFailed {
            what,
            status,
            error,
        } => MetalGraphError::CommandBufferFailed {
            what,
            status: *status,
            error: error.clone(),
        },
        MetalGraphError::BufferTooLarge {
            what,
            requested,
            max,
        } => MetalGraphError::BufferTooLarge {
            what,
            requested: *requested,
            max: *max,
        },
        // FIX-06 / MET-02: `MetalGraphError` gained a typed
        // `WeightKindMismatch` variant, and this hand-written clone is an
        // exhaustive match, so it must name it. See the package deviations:
        // all four copies of `clone_err` should be replaced by a `Clone`
        // derive on `MetalGraphError` itself.
        MetalGraphError::WeightKindMismatch { expected, found } => {
            MetalGraphError::WeightKindMismatch {
                expected: *expected,
                found: *found,
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public dispatch functions
// ═══════════════════════════════════════════════════════════════════════════

/// `Q2_K` GEMV on the Metal GPU.
///
/// # Arguments
/// - `blocks`: raw AoS super-block bytes, length `n_rows * (k / 256) * 84`.
/// - `input`: dense FP32 input vector, length `k`.
/// - `output`: dense FP32 output vector, length `n_rows`.
/// - `n_rows`: number of output rows.
/// - `k`: input dimension (must be a positive multiple of 256).
///
/// # Errors
/// Returns [`MetalGraphError::DeviceNotFound`] on systems without a Metal device,
/// [`MetalGraphError::CompilationFailed`] on pipeline-build failure, or
/// [`MetalGraphError::EncodingFailed`] for shape/buffer mismatches.
pub fn metal_gemv_q2k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q2k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q2K_BLOCK_BYTES,
        "Q2_K",
    )
}

/// `Q3_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q3k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q3k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q3K_BLOCK_BYTES,
        "Q3_K",
    )
}

/// `Q4_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q4k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q4k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q4K_BLOCK_BYTES,
        "Q4_K",
    )
}

/// `Q5_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q5k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q5k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q5K_BLOCK_BYTES,
        "Q5_K",
    )
}

/// `Q6_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q6k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q6k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q6K_BLOCK_BYTES,
        "Q6_K",
    )
}

/// `Q8_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q8k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q8k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q8K_BLOCK_BYTES,
        "Q8_K",
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_k_quant_gemv(
    s: &MetalKQuantState,
    pipeline: &ComputePipelineState,
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    block_bytes: usize,
    format: &str,
) -> Result<(), MetalGraphError> {
    // ── Validate dimensions ─────────────────────────────────────────────────
    if k == 0 || !k.is_multiple_of(QK_K) {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: k = {k} must be a non-zero multiple of {QK_K}"
        )));
    }
    let blocks_per_row = k / QK_K;
    let expected_block_bytes = n_rows.saturating_mul(blocks_per_row) * block_bytes;
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: blocks.len() = {} expected {expected_block_bytes} (n_rows = {n_rows}, k = {k})",
            blocks.len()
        )));
    }
    if input.len() != k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: input.len() = {} expected {k}",
            input.len()
        )));
    }
    if output.len() != n_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: output.len() = {} expected {n_rows}",
            output.len()
        )));
    }
    if n_rows == 0 {
        return Ok(());
    }

    // ── Allocate buffers (shared storage) ───────────────────────────────────
    let block_buf = s.device.new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = s.device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(input) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = s.device.new_buffer(
        (n_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Zero-initialise output (some drivers leave new buffers uninitialised).
    unsafe {
        std::ptr::write_bytes(output_buf.contents() as *mut f32, 0u8, n_rows);
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_rows = {n_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;

    // ── Encode + commit ─────────────────────────────────────────────────────
    let cmd = s.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&block_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &n_rows_u32 as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const std::ffi::c_void,
    );

    let n_tgs = n_rows.div_ceil(SIMDS_PER_TG) as u64;
    let grid = metal::MTLSize::new(n_tgs, 1, 1);
    let tg_size = metal::MTLSize::new(THREADS_PER_TG, 1, 1);
    encoder.dispatch_thread_groups(grid, tg_size);
    encoder.end_encoding();

    commit_and_wait(cmd, "metal_gemv_k_quant")?;

    // ── Read output back ────────────────────────────────────────────────────
    unsafe {
        let src = output_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), n_rows);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — CI-GPU-gated parity, host-only constant checks elsewhere
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use oxibonsai_core::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K};

    #[test]
    fn block_size_constants_match_core() {
        assert_eq!(QK_K, oxibonsai_core::quant_k::QK_K);
        assert_eq!(Q2K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q2_K_BYTES);
        assert_eq!(Q3K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q3K_BYTES);
        assert_eq!(Q4K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q4_K_BYTES);
        assert_eq!(Q5K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q5K_BYTES);
        assert_eq!(Q6K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q6K_BYTES);
        assert_eq!(Q8K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q8K_BYTES);
    }

    /// `k` not a multiple of 256 is rejected before any GPU work.
    #[test]
    fn q4k_bad_k_rejected() {
        if state().is_err() {
            return;
        }
        let blocks = vec![0u8; Q4K_BLOCK_BYTES];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        assert!(metal_gemv_q4k(&blocks, &input, &mut output, 1, 255).is_err());
    }

    // ───────────────────────────────────────────────────────────────────────
    // MSL-side golden blocks (`core-gguf-K0` / `KQUANT-CPU-fallout`)
    //
    // The parity tests in `tests/metal_k_quant_gemv_parity.rs` compare a GPU
    // row sum against the CPU row sum under a relative tolerance. That hid the
    // Q6_K half of this bug at rel = 0.0119 for months. The tests below pin the
    // *individual dequantized weights* the MSL kernels produce, so a kernel
    // edit that re-breaks the ggml block walk fails loudly instead of drifting
    // inside a tolerance.
    // ───────────────────────────────────────────────────────────────────────

    /// 8-byte-aligned byte buffer so `slice_from_bytes` can reinterpret it.
    #[repr(C, align(8))]
    struct Aligned<const N: usize>([u8; N]);

    /// Signature shared by the six public `metal_gemv_q*` entry points.
    type GemvFn = fn(&[u8], &[f32], &mut [f32], usize, usize) -> Result<(), MetalGraphError>;

    /// Recover all 256 dequantized weights of one super-block **from the GPU**,
    /// in a single dispatch.
    ///
    /// Row `r` of a `256 x 65536` weight matrix holds `golden` at block slot `r`
    /// and all-zero blocks everywhere else — an all-zero block has `d = 0`, so
    /// it dequantizes to 256 exact zeros in every K-quant format. The input is
    /// the concatenation of 256 one-hot vectors (`x[b * 256 + i] = (i == b)`),
    /// hence
    ///
    /// ```text
    /// out[r] = Σ_b Σ_i y_b[i] · x[b·256 + i] = y_golden[r]
    /// ```
    ///
    /// i.e. the kernel's own value for element `r`, surrounded only by exact
    /// zero terms. Returns `None` when the host has no Metal device.
    fn gpu_dequant_block(gemv: GemvFn, golden: &[u8]) -> Option<Vec<f32>> {
        let block_bytes = golden.len();
        let k = QK_K * QK_K;
        let mut blocks = vec![0u8; QK_K * QK_K * block_bytes];
        for r in 0..QK_K {
            let off = (r * QK_K + r) * block_bytes;
            blocks[off..off + block_bytes].copy_from_slice(golden);
        }
        let mut input = vec![0.0f32; k];
        for b in 0..QK_K {
            input[b * QK_K + b] = 1.0;
        }
        let mut out = vec![0.0f32; QK_K];
        match gemv(&blocks, &input, &mut out, QK_K, k) {
            Ok(()) => Some(out),
            Err(e) if e.to_string().contains("no Metal-capable GPU device") => None,
            Err(e) => panic!("Metal K-quant golden probe failed: {e}"),
        }
    }

    /// Element-wise comparison against a golden row. The tolerance is only a
    /// guard against f32 noise in the (exact) zero accumulation — every layout
    /// error moves a value by O(1).
    fn assert_elementwise(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-5f32 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "{label}: element {i}: gpu {g} != golden {w}"
            );
        }
    }

    /// Deterministic byte stream for the dense blocks (no `rand` dependency).
    fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 33) as u8
            })
            .collect()
    }

    /// GPU decode of `golden` vs. the byte-exact CPU reference decode of the
    /// same bytes — `BlockQ*K::dequant` is this package's normative oracle.
    macro_rules! assert_gpu_matches_cpu_reference {
        ($label:literal, $blk:ty, $metal:path, $bytes:expr) => {{
            let buf = $bytes;
            let Some(gpu) = gpu_dequant_block($metal, &buf.0) else {
                return;
            };
            let blocks = <$blk>::slice_from_bytes(&buf.0).expect("aligned block parses");
            assert_eq!(blocks.len(), 1);
            let mut cpu = vec![0.0f32; QK_K];
            <$blk>::dequant(blocks, &mut cpu).expect("cpu dequant");
            assert_elementwise($label, &gpu, &cpu);
            gpu
        }};
    }

    // ── Q2_K ───────────────────────────────────────────────────────────────

    /// `d = 0.5`, `dmin = 0.25`, `scales[0] = 0x12` (sc 2, mn 1),
    /// `scales[1] = 0x31` (sc 1, mn 3), `qs[0] = 0xE4`, `qs[16] = 0x1B`.
    /// ggml's `is++` / shift-0,2,4,6 walk emits `y[0..16]` from `qs[0..16] >> 0`
    /// under `scales[0]` and `y[16..32]` from `qs[16..32] >> 0` under
    /// `scales[1]`; the pre-fix element-sequential kernel produced `y[1] = 0.75`.
    fn q2k_sparse_golden() -> Aligned<84> {
        let mut b = [0u8; 84];
        b[0] = 0x12;
        b[1] = 0x31;
        b[16] = 0xE4;
        b[32] = 0x1B;
        b[80..82].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        b[82..84].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        Aligned(b)
    }

    #[test]
    fn metal_q2k_hand_derived_golden_block() {
        let gpu = assert_gpu_matches_cpu_reference!(
            "Q2_K golden vs CPU",
            BlockQ2K,
            metal_gemv_q2k,
            q2k_sparse_golden()
        );
        let mut want = vec![0.0f32; QK_K];
        want[0..16].fill(-0.25);
        want[16] = 0.75;
        want[17..32].fill(-0.75);
        assert_elementwise("Q2_K hand golden", &gpu, &want);
    }

    #[test]
    fn metal_q2k_dense_block_matches_cpu_reference() {
        let mut b = [0u8; 84];
        b[0..80].copy_from_slice(&lcg_bytes(80, 0x2ACE_0001));
        b[80..82].copy_from_slice(&f16::from_f32(0.0137).to_bits().to_le_bytes());
        b[82..84].copy_from_slice(&f16::from_f32(0.0091).to_bits().to_le_bytes());
        assert_gpu_matches_cpu_reference!("Q2_K dense", BlockQ2K, metal_gemv_q2k, Aligned(b));
    }

    // ── Q3_K ───────────────────────────────────────────────────────────────

    /// `d = 0.25`; sub-block 0's 6-bit code is `2 | (2 << 4) = 34` → signed
    /// scale `+2`, every other code is `0` → `-32`. `hmask[0] = 0x01`,
    /// `qs[0] = 0x03`. ggml subtracts 4 when the hmask bit is **clear**, and
    /// element 32 lands in the `m = 2` step, pinning the `m <<= 1` stepping.
    fn q3k_sparse_golden() -> Aligned<110> {
        let mut b = [0u8; 110];
        b[0] = 0x01;
        b[32] = 0x03;
        b[96] = 0x02;
        b[104] = 0x02;
        b[108..110].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        Aligned(b)
    }

    #[test]
    fn metal_q3k_hand_derived_golden_block() {
        let gpu = assert_gpu_matches_cpu_reference!(
            "Q3_K golden vs CPU",
            BlockQ3K,
            metal_gemv_q3k,
            q3k_sparse_golden()
        );
        let mut want = vec![32.0f32; QK_K];
        want[0] = 1.5;
        want[1..16].fill(-2.0);
        assert_elementwise("Q3_K hand golden", &gpu, &want);
    }

    #[test]
    fn metal_q3k_dense_block_matches_cpu_reference() {
        let mut b = [0u8; 110];
        b[0..108].copy_from_slice(&lcg_bytes(108, 0x3BEE_0002));
        b[108..110].copy_from_slice(&f16::from_f32(0.0037).to_bits().to_le_bytes());
        assert_gpu_matches_cpu_reference!("Q3_K dense", BlockQ3K, metal_gemv_q3k, Aligned(b));
    }

    // ── Q4_K ───────────────────────────────────────────────────────────────

    /// `d = 0.5`, `dmin = 0.25`, `qs[0] = 0x57`; the `scales` array exercises
    /// both halves of `get_scale_min_k4` (`j < 4` reads a full 6-bit byte,
    /// `j >= 4` splices in the top two bits of bytes `j - 4` / `j`):
    ///
    /// ```text
    /// j = 0: sc = 132 & 63 = 4                     m = 66 & 63 = 2
    /// j = 1: sc = 3                                m = 1
    /// j = 4: sc = (0x31 & 0xF) | ((132 >> 6) << 4) m = (0x31 >> 4) | ((66 >> 6) << 4)
    ///           = 33                                  = 19
    /// ```
    ///
    /// and the 32-low-then-32-high nibble emission order gives `y[0] = 13.5`,
    /// `y[32] = 7.25`, `y[128..160] = -4.75`.
    fn q4k_sparse_golden() -> Aligned<144> {
        let mut b = [0u8; 144];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        b[4] = 132;
        b[5] = 3;
        b[8] = 66;
        b[9] = 1;
        b[12] = 0x31;
        b[16] = 0x57;
        Aligned(b)
    }

    #[test]
    fn metal_q4k_hand_derived_golden_block() {
        let gpu = assert_gpu_matches_cpu_reference!(
            "Q4_K golden vs CPU",
            BlockQ4K,
            metal_gemv_q4k,
            q4k_sparse_golden()
        );
        let mut want = vec![0.0f32; QK_K];
        want[0] = 13.5;
        want[1..32].fill(-0.5);
        want[32] = 7.25;
        want[33..64].fill(-0.25);
        want[128..160].fill(-4.75);
        assert_elementwise("Q4_K hand golden", &gpu, &want);
    }

    #[test]
    fn metal_q4k_dense_block_matches_cpu_reference() {
        let mut b = [0u8; 144];
        b[0..2].copy_from_slice(&f16::from_f32(0.0211).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(0.0074).to_bits().to_le_bytes());
        b[4..144].copy_from_slice(&lcg_bytes(140, 0x4C0D_0003));
        assert_gpu_matches_cpu_reference!("Q4_K dense", BlockQ4K, metal_gemv_q4k, Aligned(b));
    }

    // ── Q5_K ───────────────────────────────────────────────────────────────

    /// Q4_K's scale layout plus the 5th bit: `d = 0.5`, `dmin = 0.25`,
    /// `scales[0] = 4`, `scales[1] = 3`, `scales[4] = 2`, `scales[5] = 1`,
    /// `qh[0] = 0x03`, `qs[0] = 0x57`. The `+16` on **both** lanes appears only
    /// if the kernel applies the `u1`/`u2` masks (bits 0 and 1 of `qh[0]`)
    /// rather than a per-element `qh[i / 8] >> (i % 8)` bit:
    /// `y[0] = 45.5`, `y[32] = 31.25`.
    fn q5k_sparse_golden() -> Aligned<176> {
        let mut b = [0u8; 176];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        b[4] = 4;
        b[5] = 3;
        b[8] = 2;
        b[9] = 1;
        b[16] = 0x03;
        b[48] = 0x57;
        Aligned(b)
    }

    #[test]
    fn metal_q5k_hand_derived_golden_block() {
        let gpu = assert_gpu_matches_cpu_reference!(
            "Q5_K golden vs CPU",
            BlockQ5K,
            metal_gemv_q5k,
            q5k_sparse_golden()
        );
        let mut want = vec![0.0f32; QK_K];
        want[0] = 45.5;
        want[1..32].fill(-0.5);
        want[32] = 31.25;
        want[33..64].fill(-0.25);
        assert_elementwise("Q5_K hand golden", &gpu, &want);
    }

    #[test]
    fn metal_q5k_dense_block_matches_cpu_reference() {
        let mut b = [0u8; 176];
        b[0..2].copy_from_slice(&f16::from_f32(0.0163).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(0.0052).to_bits().to_le_bytes());
        b[4..176].copy_from_slice(&lcg_bytes(172, 0x5D1E_0004));
        assert_gpu_matches_cpu_reference!("Q5_K dense", BlockQ5K, metal_gemv_q5k, Aligned(b));
    }

    // ── Q6_K ───────────────────────────────────────────────────────────────

    /// `d = 0.5`; `scales = [2, 0, 3, 0, -1, 0, 4, 0, …]`; `ql[0] = 0x9A`,
    /// `ql[32] = 0x27`, `qh[0] = 0xE4`. ggml's four-lane interleave with
    /// `sc[is + 0/2/4/6]` gives `y[0] = -22`, `y[32] = -13.5`, `y[64] = -4.5`,
    /// `y[96] = 36`; for `l in 1..16` the zero nibbles give `q = -32`.
    fn q6k_sparse_golden() -> Aligned<210> {
        let mut b = [0u8; 210];
        b[0] = 0x9A;
        b[32] = 0x27;
        b[128] = 0xE4;
        b[192] = 2i8 as u8;
        b[194] = 3i8 as u8;
        b[196] = (-1i8) as u8;
        b[198] = 4i8 as u8;
        b[208..210].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        Aligned(b)
    }

    #[test]
    fn metal_q6k_hand_derived_golden_block() {
        let gpu = assert_gpu_matches_cpu_reference!(
            "Q6_K golden vs CPU",
            BlockQ6K,
            metal_gemv_q6k,
            q6k_sparse_golden()
        );
        let mut want = vec![0.0f32; QK_K];
        want[0] = -22.0;
        want[32] = -13.5;
        want[64] = -4.5;
        want[96] = 36.0;
        for l in 1..16usize {
            want[l] = -32.0;
            want[l + 32] = -48.0;
            want[l + 64] = 16.0;
            want[l + 96] = -64.0;
        }
        assert_elementwise("Q6_K hand golden", &gpu, &want);
    }

    #[test]
    fn metal_q6k_dense_block_matches_cpu_reference() {
        let mut b = [0u8; 210];
        b[0..208].copy_from_slice(&lcg_bytes(208, 0x6E2F_0005));
        b[208..210].copy_from_slice(&f16::from_f32(0.00042).to_bits().to_le_bytes());
        assert_gpu_matches_cpu_reference!("Q6_K dense", BlockQ6K, metal_gemv_q6k, Aligned(b));
    }
}
