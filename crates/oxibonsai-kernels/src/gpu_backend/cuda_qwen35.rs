//! Host-side CUDA wiring for the `qwen35` hybrid path (PrismML Bonsai 2,
//! finding **F13**): compiles and launches the kernels in
//! [`super::kernel_sources::cuda_qwen35_kernels`], and transcodes `PTQ1_0`
//! (ggml 143) weights into the existing ternary GEMV's SoA layout.
//!
//! **CUDA is UNVALIDATED.** No CUDA hardware ran any of the launches below:
//! every one is transcribed from the CPU/Metal reference
//! (`oxibonsai_model::hybrid::metal`, `kernel_sources::qwen35`) and is
//! compile-checked only (`cargo check --target x86_64-unknown-linux-gnu
//! --features native-cuda`) plus `scripts/check_cuda.sh`'s kernel-source
//! syntax pass. A real-device cosine-similarity comparison against the CPU
//! reference, on real layer-0 activations, is required per kernel —
//! `fwht_signed` (forward and inverse), `conv1d_silu`, `l2_normalize`,
//! `sigmoid_gate`, `gated_rmsnorm`, `partial_rope`, `gdn_step` and
//! `gemv_pq2_g128_v1` — before this path is advertised as working, at the
//! same `cos` value of `0.999` or higher the Metal kernels in
//! `metal_full_layer::qwen35` were accepted against. The checklist of those
//! runs is `tests/cuda_hybrid_parity_plan.rs`.
//!
//! # Scope
//!
//! **There is no CUDA hybrid (`qwen35`) forward pass.** Nothing in this
//! workspace launches the seven kernels below: a Bonsai 2 model on a
//! `native-cuda` build runs its hybrid layers on the CPU path. This module
//! is the device-side groundwork for such a forward pass — the seven
//! primitives Bonsai 2's hybrid layer needs beyond the plain-transformer
//! CUDA kernels, each with a launch wrapper, plus the `PTQ1_0` weight
//! transcode — and it is **not** a complete surface: the per-token host
//! preparation those launches assume has no CUDA implementation either:
//!
//! - the v-head gather that expands `q` / `k` from `n_k_heads` to one
//!   vector per v-head (`gdn_step` takes them already expanded), and the
//!   re-ordering of the output gate `z` into `o`'s v-head order
//!   (`gated_rmsnorm` does no gather);
//! - the de-interleave of `attn_q`'s per-head `[q | gate]` projection into
//!   separate `q` and contiguous `gate` buffers (`sigmoid_gate` reads `gate`
//!   contiguously);
//! - the per-token decay / beta preparation (`exp(-exp(A_log) *
//!   softplus(a + dt_bias))` and `sigmoid(b)`), which `gdn_step` takes as
//!   inputs;
//! - the recurrent-state and KV-cache residency, and the layer loop itself.
//!
//! The Metal path's fully GPU-resident model
//! (`oxibonsai_model::hybrid::metal::HybridMetalRunner`, mapped-region
//! binding, a process-wide weight-residency cache) is the shape a CUDA
//! forward pass would mirror.

use oxibonsai_core::quant_prism::BlockPTQ1_0;

use super::kernel_sources::cuda_qwen35_kernels::decode_pq2_code as _decode_pq2_code_reexport_check;

/// Block byte size of the 2-bit SoA format `gemv_tq2_g128_v1`
/// ([`crate::gpu_backend::cuda_kernels`]) consumes: one `f16` scale plus 32
/// bytes of packed 2-bit codes per 128 weights.
const TQ2_SOA_BLOCK_BYTES: usize = 34;
const TQ2_SOA_SCALE_BYTES: usize = 2;
const TQ2_SOA_QS_BYTES: usize = 32;

/// Transcode `PTQ1_0` (ggml **143**) blocks into the same 2-bit SoA layout
/// `gemv_tq2_g128_v1` reads — the lossless `PTQ1_0` -> 2-bit transcode that
/// lets the existing ternary GEMV/GEMM stack serve `PTQ1_0` weights
/// (finding **F13**).
///
/// `PTQ1_0`'s 128 trit codes per block are already `{0, 1, 2}` (`y = (code -
/// 1) * d`, [`BlockPTQ1_0::dequant`]) — exactly the values
/// `TQ2_0_g128`/`PQ2_0`'s 2-bit codes `0b00/0b01/0b10` decode to under
/// `decode_tq2` (`0b11` is simply never produced by ternary data, on either
/// side), so no new decode table and no new GEMV kernel are needed: this
/// function unpacks each block's trits with
/// [`BlockPTQ1_0::decode_codes`] (the exact algorithm transcribed from
/// `dequantize_row_ptq1_0`, `ggml-quants.c:2255-2285` — interleaved stage
/// order, wrapping `u8` trit arithmetic, all reproduced there, not
/// re-derived here) and re-packs them 4-per-byte LSB-first into the SoA
/// layout `gemv_tq2_g128_v1` already serves, scales first.
///
/// Returns `None` when `blocks` is empty (no stable "one SoA buffer" to
/// return).
///
/// The caller must upload this output with
/// [`crate::gpu_backend::cuda_graph::CudaGraph::upload_weight_soa_direct`]
/// (or its `_for_epoch` twin) — **not**
/// `get_or_upload_weight_tq2_soa`/`_for_epoch`, which reformats its input
/// as AoS and would corrupt already-SoA bytes.
#[cfg(feature = "native-cuda")]
#[must_use]
pub fn ptq1_blocks_to_tq2_soa(blocks: &[BlockPTQ1_0]) -> Option<Vec<u8>> {
    if blocks.is_empty() {
        return None;
    }
    let n = blocks.len();
    let mut soa = vec![0u8; n * TQ2_SOA_BLOCK_BYTES];
    let (scales_section, qs_section) = soa.split_at_mut(n * TQ2_SOA_SCALE_BYTES);

    let mut codes = [0u8; 128];
    for (i, block) in blocks.iter().enumerate() {
        block.decode_codes(&mut codes);

        let bits = block.d.to_bits().to_le_bytes();
        scales_section[i * TQ2_SOA_SCALE_BYTES] = bits[0];
        scales_section[i * TQ2_SOA_SCALE_BYTES + 1] = bits[1];

        let qs_block = &mut qs_section[i * TQ2_SOA_QS_BYTES..(i + 1) * TQ2_SOA_QS_BYTES];
        for (byte_idx, qs_byte) in qs_block.iter_mut().enumerate() {
            let base = byte_idx * 4;
            // LSB-first, matching `decode_tq2`'s `(b >> 2*lane) & 3`.
            *qs_byte = (codes[base] & 0b11)
                | ((codes[base + 1] & 0b11) << 2)
                | ((codes[base + 2] & 0b11) << 4)
                | ((codes[base + 3] & 0b11) << 6);
        }
    }
    Some(soa)
}

/// Re-export check: [`super::kernel_sources::cuda_qwen35_kernels`]'s
/// `decode_pq2_code` stays reachable from this module's callers without a
/// second implementation to drift from the kernel source's table. Not
/// itself part of this module's public API (the kernel-sources module is).
#[cfg(feature = "native-cuda")]
const _: fn(u8) -> f32 = _decode_pq2_code_reexport_check;

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod device {
    // Nothing in this workspace launches these kernels (see the module doc:
    // there is no CUDA hybrid forward pass), so every item here is unused
    // until one exists; the re-export at the bottom of the parent module
    // carries the matching `#[allow(unused_imports)]`.
    #![allow(dead_code)]

    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

    use super::super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
    use super::super::kernel_sources::cuda_qwen35_kernels::{
        qwen35_fwht_dims_ok, qwen35_gdn_head_k_dim_ok, qwen35_gdn_out_scale,
        qwen35_partial_rope_dims_ok, CUDA_QWEN35_CONV1D_KERNELS_SRC, CUDA_QWEN35_FWHT_KERNELS_SRC,
        CUDA_QWEN35_GATED_RMSNORM_KERNELS_SRC, CUDA_QWEN35_GDN_KERNELS_SRC,
        CUDA_QWEN35_L2NORM_KERNELS_SRC, CUDA_QWEN35_PARTIAL_ROPE_KERNELS_SRC,
        CUDA_QWEN35_SIGMOID_GATE_KERNELS_SRC,
    };

    /// Largest power of two `<= n` (minimum `1`), for sizing a
    /// shared-memory tree-reduction kernel's block dimension: `l2_normalize`
    /// and `gated_rmsnorm` both halve their active-thread count each pass
    /// (`stride >>= 1` down to `0`), which only visits every element when
    /// the starting thread count is a power of two.
    const fn largest_pow2_at_most(n: u32) -> u32 {
        if n <= 1 {
            return 1;
        }
        // `31 - leading_zeros` is the index of the highest set bit, so
        // `1 << that` is the largest power of two `<= n` for any `n >= 1`.
        1u32 << (31 - n.leading_zeros())
    }

    /// Compiled CUDA function handles for the seven `qwen35` hybrid-layer
    /// kernels (finding **F13**). Held in [`qwen35_modules`], not on
    /// [`CudaGraph`] itself: `CudaGraph`'s struct definition and constructor
    /// (`cudagraph_type.rs` / `cudagraph_global_group.rs`) are defined
    /// elsewhere, so this state follows the same process-wide-`OnceLock`
    /// shape `cuda_prefill::state`'s `CudaPrefillModules` already uses.
    pub(crate) struct Qwen35Modules {
        pub(crate) fwht_signed: CudaFunction,
        pub(crate) conv1d_silu: CudaFunction,
        pub(crate) l2_normalize: CudaFunction,
        pub(crate) sigmoid_gate: CudaFunction,
        pub(crate) gated_rmsnorm: CudaFunction,
        pub(crate) partial_rope: CudaFunction,
        pub(crate) gdn_step: CudaFunction,
    }

    // SAFETY: CudaFunction is Send in cudarc; this mirrors every other
    // `Cuda*Modules` struct in this crate (e.g. `CudaPrefillModules`).
    unsafe impl Send for Qwen35Modules {}
    unsafe impl Sync for Qwen35Modules {}

    fn qwen35_state() -> &'static Mutex<Option<Arc<Qwen35Modules>>> {
        static STATE: OnceLock<Mutex<Option<Arc<Qwen35Modules>>>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(None))
    }

    /// Compile and cache the seven `qwen35` CUDA kernels. Idempotent.
    ///
    /// Each `CUDA_QWEN35_*_KERNELS_SRC` string is fully self-contained
    /// (including its own copy of any `q35_sigmoid`/`q35_silu`/
    /// `q35_fp16_to_f32` helper it calls) and compiles as its own PTX
    /// module: NVRTC compiles a `pub const CUDA_*` string as an independent
    /// translation unit, and `scripts/check_cuda.sh` extracts and
    /// syntax-checks each one in isolation the same way, so a kernel that
    /// depended on a *different* constant's helpers would pass here but
    /// fail that gate.
    pub(crate) fn qwen35_modules(graph: &CudaGraph) -> Result<Arc<Qwen35Modules>, CudaGraphError> {
        let state = qwen35_state();
        let mut guard = state.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(ref m) = *guard {
            return Ok(Arc::clone(m));
        }

        let load_one =
            |src: &str, tag: &str, fn_name: &str| -> Result<CudaFunction, CudaGraphError> {
                let ptx = compile_or_load_ptx(src, tag)?;
                let module = graph
                    .context_arc()
                    .load_module(ptx)
                    .map_err(|e| CudaGraphError::DriverError(format!("load_module {tag}: {e}")))?;
                module.load_function(fn_name).map_err(|e| {
                    CudaGraphError::DriverError(format!("load_function({fn_name}): {e}"))
                })
            };

        let fwht_signed = load_one(CUDA_QWEN35_FWHT_KERNELS_SRC, "qwen35_fwht", "fwht_signed")?;
        let l2_normalize = load_one(
            CUDA_QWEN35_L2NORM_KERNELS_SRC,
            "qwen35_l2norm",
            "l2_normalize",
        )?;
        let partial_rope = load_one(
            CUDA_QWEN35_PARTIAL_ROPE_KERNELS_SRC,
            "qwen35_partial_rope",
            "partial_rope",
        )?;
        let gdn_step = load_one(CUDA_QWEN35_GDN_KERNELS_SRC, "qwen35_gdn", "gdn_step")?;
        let conv1d_silu = load_one(
            CUDA_QWEN35_CONV1D_KERNELS_SRC,
            "qwen35_conv1d",
            "conv1d_silu",
        )?;
        let sigmoid_gate = load_one(
            CUDA_QWEN35_SIGMOID_GATE_KERNELS_SRC,
            "qwen35_sigmoid_gate",
            "sigmoid_gate",
        )?;
        let gated_rmsnorm = load_one(
            CUDA_QWEN35_GATED_RMSNORM_KERNELS_SRC,
            "qwen35_gated_rmsnorm",
            "gated_rmsnorm",
        )?;

        let modules = Arc::new(Qwen35Modules {
            fwht_signed,
            conv1d_silu,
            l2_normalize,
            sigmoid_gate,
            gated_rmsnorm,
            partial_rope,
            gdn_step,
        });
        *guard = Some(Arc::clone(&modules));
        Ok(modules)
    }

    /// Launch `fwht_signed`: blockwise Hadamard transform with fused
    /// sign-flip (finding **F13**). `x` is rotated in place.
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] when `(width, block)` fail
    /// [`qwen35_fwht_dims_ok`] (see that function for the exact rule).
    ///
    /// # Safety
    /// `x` and `signs` must be valid device pointers on `graph.stream_arc()`,
    /// `x` sized `rows * width` and `signs` at least `width`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fwht_signed(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        x: &mut CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        width: u32,
        block: u32,
        rows: u32,
        inverse: bool,
        scale: f32,
    ) -> Result<(), CudaGraphError> {
        if !qwen35_fwht_dims_ok(width as usize, block as usize) {
            return Err(CudaGraphError::InvalidDimensions(format!(
                "fwht_signed: invalid dims width={width} block={block}"
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (width / block, rows.max(1), 1),
            block_dim: (block.min(1024), 1, 1),
            shared_mem_bytes: 0,
        };
        let inverse_u32: u32 = u32::from(inverse);
        graph
            .stream_arc()
            .launch_builder(&mods.fwht_signed)
            .arg(x)
            .arg(signs)
            .arg(&width)
            .arg(&block)
            .arg(&inverse_u32)
            .arg(&scale)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("fwht_signed launch: {e}")))
    }

    /// Launch `conv1d_silu`: depthwise causal conv1d (kernel width 4) fused
    /// with SiLU over `t_len` tokens, updating the causal window in place
    /// (finding **F13**).
    ///
    /// `state` is channel-major, 3 taps per channel, oldest first — the same
    /// layout `oxibonsai_model::hybrid::recurrent_cache`'s `RecurrentCache` conv
    /// window keeps and `oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode`
    /// updates in place, **not** a frame-major buffer with the new sample
    /// folded in. `x`/`out` are token-major, matching the Metal reference
    /// `q35_conv1d_silu`'s `x[t*channels+c]`/`out[t*channels+c]`
    /// (`kernel_sources::qwen35`).
    ///
    /// # Safety
    /// All slices must be valid device pointers on `graph.stream_arc()`:
    /// `x` and `out` sized `t_len * channels`, `weight` sized `4 * channels`,
    /// `state` sized `3 * channels`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_conv1d_silu(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        channels: u32,
        t_len: u32,
    ) -> Result<(), CudaGraphError> {
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.conv1d_silu)
            .arg(x)
            .arg(weight)
            .arg(state)
            .arg(out)
            .arg(&channels)
            .arg(&t_len)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("conv1d_silu launch: {e}")))
    }

    /// Launch `l2_normalize`: per-row L2 normalisation in place (finding
    /// **F13**; the gated-delta-net's per-head `q`/`k` normalisation).
    ///
    /// # Safety
    /// `x` must be a valid device pointer on `graph.stream_arc()`, sized
    /// `n_rows * row_width`.
    pub unsafe fn launch_l2_normalize(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        x: &mut CudaSlice<f32>,
        row_width: u32,
        n_rows: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        // `l2_normalize`'s shared-memory tree reduction (`stride >>= 1` down
        // to 0) is only correct when `blockDim.x` is a power of two — an odd
        // or non-power-of-two block would drop the top bit's partner and
        // silently under-sum. The kernel's per-thread strided load loop
        // (`for i = tid; i < row_width; i += tsz`) already covers any
        // `row_width`, so the block size only needs to be a power of two,
        // not equal to `row_width`.
        let block = largest_pow2_at_most(row_width.clamp(1, 256));
        let cfg = LaunchConfig {
            grid_dim: (n_rows.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.l2_normalize)
            .arg(x)
            .arg(&row_width)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("l2_normalize launch: {e}")))
    }

    /// Launch `sigmoid_gate`: `out = attn * sigmoid(gate)`, elementwise
    /// (finding **F13**; full-attention layers' output gate).
    ///
    /// # Safety
    /// All slices must be valid device pointers on `graph.stream_arc()`,
    /// each sized `n`.
    pub unsafe fn launch_sigmoid_gate(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        attn: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: u32,
    ) -> Result<(), CudaGraphError> {
        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.sigmoid_gate)
            .arg(attn)
            .arg(gate)
            .arg(out)
            .arg(&n)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("sigmoid_gate launch: {e}")))
    }

    /// Launch `gated_rmsnorm`: `out = weight * RMSNorm(o) * silu(z)` per
    /// v-head (finding **F13**; the Gated-DeltaNet output norm).
    ///
    /// The caller must have already re-ordered `z` into `o`'s v-head order
    /// (`tiled(m) = (m % rep) * n_k_heads + m / rep`, `prism.hadamard.
    /// gdn_v_grouped`) before this launch — the kernel does no gather.
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] when `head_dim` is outside
    /// `1..=1024` (one CTA of `head_dim`-rounded-down-to-a-power-of-two
    /// threads per v-head).
    ///
    /// # Safety
    /// All slices must be valid device pointers on `graph.stream_arc()`: `o`
    /// / `z` / `out` sized `n_v_heads * head_dim`, `weight` sized `head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_gated_rmsnorm(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        o: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        head_dim: u32,
        n_v_heads: u32,
        eps: f32,
    ) -> Result<(), CudaGraphError> {
        if head_dim == 0 || head_dim > 1024 {
            return Err(CudaGraphError::InvalidDimensions(format!(
                "gated_rmsnorm: head_dim={head_dim} must be in 1..=1024"
            )));
        }
        // Same power-of-two requirement as `launch_l2_normalize`'s block
        // size, for the same tree-reduction reason; the kernel's strided
        // per-thread loops (both the sum-of-squares reduction and the
        // final elementwise write) already cover `head_dim` not dividing
        // evenly into the block.
        let block = largest_pow2_at_most(head_dim);
        let cfg = LaunchConfig {
            grid_dim: (n_v_heads.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.gated_rmsnorm)
            .arg(o)
            .arg(z)
            .arg(weight)
            .arg(out)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gated_rmsnorm launch: {e}")))
    }

    /// Launch `partial_rope`: NEOX half-split RoPE (pairs `(i, i + n_rot/2)`)
    /// on the first `n_rot` of `head_dim` dimensions, in place, dimensions
    /// `[n_rot..head_dim)` untouched (finding **F13**: the existing
    /// `fused_qk_rope`/`fused_qk_norm_rope`
    /// ([`crate::gpu_backend::cuda_attn_kernels`]) hardcode `half_dim =
    /// head_dim >> 1`, wrong for Bonsai 2's `n_rot = 64` on a 256-wide head).
    /// Text-only: no 3-axis M-RoPE position schedule (the vision tower and
    /// its `<|image_pad|>`/`<|video_pad|>` position schedule are a separate
    /// concern, design doc §8.2).
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] when `(head_dim, n_rot)` fail
    /// [`qwen35_partial_rope_dims_ok`].
    ///
    /// # Safety
    /// `x` must be a valid device pointer on `graph.stream_arc()`, sized
    /// `n_heads * head_dim`; `cos`/`sin` sized at least `n_rot / 2`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_partial_rope(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        x: &mut CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        head_dim: u32,
        n_rot: u32,
        n_heads: u32,
    ) -> Result<(), CudaGraphError> {
        if !qwen35_partial_rope_dims_ok(head_dim as usize, n_rot as usize) {
            return Err(CudaGraphError::InvalidDimensions(format!(
                "partial_rope: invalid dims head_dim={head_dim} n_rot={n_rot}"
            )));
        }
        const BLOCK: u32 = 64;
        let half_rot = n_rot / 2;
        let cfg = LaunchConfig {
            grid_dim: (half_rot.div_ceil(BLOCK), n_heads.max(1), 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.partial_rope)
            .arg(x)
            .arg(cos)
            .arg(sin)
            .arg(&head_dim)
            .arg(&n_rot)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("partial_rope launch: {e}")))
    }

    /// Launch `gdn_step`: one Gated-DeltaNet recurrence step for every
    /// v-head, in place on `state` (finding **F13**). See
    /// [`CUDA_QWEN35_GDN_KERNELS_SRC`](super::super::kernel_sources::cuda_qwen35_kernels::CUDA_QWEN35_GDN_KERNELS_SRC)
    /// for the exact per-thread step order (decay, then `kv`, then delta,
    /// then `o`, then the `1/sqrt(head_v_dim)` output scale — order-sensitive,
    /// must match the CPU reference). The scale is derived here from
    /// `head_v_dim` via
    /// [`qwen35_gdn_out_scale`](super::super::kernel_sources::cuda_qwen35_kernels::qwen35_gdn_out_scale),
    /// the same value the CPU reference's `GdnDims::out_scale` computes, so
    /// the caller never has to pass it separately or risk it drifting from
    /// the geometry.
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] when `head_k_dim` fails
    /// [`qwen35_gdn_head_k_dim_ok`] (the kernel's fixed-size per-thread
    /// local array caps it at 256) or `head_v_dim` is outside `1..=1024`
    /// (one thread per state column).
    ///
    /// # Safety
    /// All slices must be valid device pointers on `graph.stream_arc()`:
    /// `state` sized `n_v_heads * head_k_dim * head_v_dim`, `q`/`k` sized
    /// `n_v_heads * head_k_dim`, `v`/`out` sized `n_v_heads * head_v_dim`,
    /// `decay`/`beta` sized `n_v_heads`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_gdn_step(
        graph: &CudaGraph,
        mods: &Qwen35Modules,
        state: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        head_k_dim: u32,
        head_v_dim: u32,
        n_v_heads: u32,
    ) -> Result<(), CudaGraphError> {
        if !qwen35_gdn_head_k_dim_ok(head_k_dim as usize) {
            return Err(CudaGraphError::InvalidDimensions(format!(
                "gdn_step: head_k_dim={head_k_dim} must be in 1..=256 (the kernel's local array)"
            )));
        }
        if head_v_dim == 0 || head_v_dim > 1024 {
            return Err(CudaGraphError::InvalidDimensions(format!(
                "gdn_step: head_v_dim={head_v_dim} must be in 1..=1024"
            )));
        }
        let out_scale = qwen35_gdn_out_scale(head_v_dim as usize);
        let cfg = LaunchConfig {
            grid_dim: (n_v_heads.max(1), 1, 1),
            block_dim: (head_v_dim, 1, 1),
            shared_mem_bytes: 2 * head_k_dim * 4,
        };
        graph
            .stream_arc()
            .launch_builder(&mods.gdn_step)
            .arg(state)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(decay)
            .arg(beta)
            .arg(out)
            .arg(&head_k_dim)
            .arg(&out_scale)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gdn_step launch: {e}")))
    }
}

// Re-exported for a CUDA hybrid forward pass to call once one exists (see
// this file's module doc): nothing launches these seven kernels today — they
// are compiled and syntax-checked only — so the re-export is unused until
// that caller is written.
#[allow(unused_imports)]
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub(crate) use device::{
    launch_conv1d_silu, launch_fwht_signed, launch_gated_rmsnorm, launch_gdn_step,
    launch_l2_normalize, launch_partial_rope, launch_sigmoid_gate, qwen35_modules, Qwen35Modules,
};

#[cfg(feature = "native-cuda")]
#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Build a `PTQ1_0` block whose 128 trits are all `1` (decodes to `0.0`
    /// regardless of scale) — [`BlockPTQ1_0::zeroed`] already derives the
    /// right raw bytes for this from the trit codec, so the test does not
    /// need to re-derive the base-3 packing.
    fn zero_block(d: f32) -> BlockPTQ1_0 {
        let mut b = BlockPTQ1_0::zeroed();
        b.d = f16::from_f32(d);
        b
    }

    #[test]
    fn ptq1_transcode_of_an_all_zero_block_is_all_zero_codes() {
        let blocks = [zero_block(0.5)];
        let soa = ptq1_blocks_to_tq2_soa(&blocks).expect("one block transcodes");
        assert_eq!(soa.len(), TQ2_SOA_BLOCK_BYTES);
        let scale_bits = u16::from_le_bytes([soa[0], soa[1]]);
        assert_eq!(scale_bits, f16::from_f32(0.5).to_bits());
        // Trit code 1 decodes to 0 under `(code - 1) * d`; packed 2-bit code
        // 0b01 per lane, LSB-first: every qs byte is 0b01_01_01_01 = 0x55.
        for &byte in &soa[TQ2_SOA_SCALE_BYTES..] {
            assert_eq!(byte, 0x55, "trit code 1 (zero) packs to 2-bit code 0b01");
        }
    }

    #[test]
    fn ptq1_transcode_round_trips_through_the_tq2_decode_table() {
        // A block built directly from known trit codes via `quantize`
        // (rather than `zeroed()`) exercises the real encode path; decode it
        // both ways (PTQ1_0's own `dequant` and this module's SoA + the
        // scalar TQ2 decode table) and require the two to agree exactly —
        // the whole point of the lossless-transcode design (no new decode
        // table to desynchronise from PTQ1_0's own).
        let mut input = [0.0f32; 128];
        for (i, v) in input.iter_mut().enumerate() {
            *v = match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
        }
        let blocks = BlockPTQ1_0::quantize(&input).expect("quantize 128 floats");
        assert_eq!(blocks.len(), 1);

        let mut reference = [0.0f32; 128];
        BlockPTQ1_0::dequant(&blocks, &mut reference).expect("dequant");

        let soa = ptq1_blocks_to_tq2_soa(&blocks).expect("transcode");
        let d = f16::from_le_bytes([soa[0], soa[1]]).to_f32();
        let decode_tq2 = |code: u8| -> f32 {
            match code & 0b11 {
                0 => -1.0,
                2 => 1.0,
                _ => 0.0,
            }
        };
        let mut transcoded = [0.0f32; 128];
        for (byte_idx, &byte) in soa[TQ2_SOA_SCALE_BYTES..].iter().enumerate() {
            for lane in 0..4 {
                let code = (byte >> (lane * 2)) & 0b11;
                transcoded[byte_idx * 4 + lane] = decode_tq2(code) * d;
            }
        }
        assert_eq!(
            reference, transcoded,
            "PTQ1_0's own dequant and the SoA-transcoded TQ2 decode must agree bit-for-bit"
        );
    }

    #[test]
    fn ptq1_transcode_rejects_empty_input() {
        assert_eq!(ptq1_blocks_to_tq2_soa(&[]), None);
    }

    #[test]
    fn ptq1_transcode_concatenates_multiple_blocks_scales_first() {
        let blocks = [zero_block(1.0), zero_block(2.0), zero_block(4.0)];
        let soa = ptq1_blocks_to_tq2_soa(&blocks).expect("three blocks");
        assert_eq!(soa.len(), 3 * TQ2_SOA_BLOCK_BYTES);
        // All three scales come first (SoA), then all three qs sections.
        let scales_end = 3 * TQ2_SOA_SCALE_BYTES;
        for (i, expect) in [1.0f32, 2.0, 4.0].into_iter().enumerate() {
            let got = f16::from_le_bytes([soa[i * 2], soa[i * 2 + 1]]).to_f32();
            assert_eq!(got, expect);
        }
        assert_eq!(soa.len() - scales_end, 3 * TQ2_SOA_QS_BYTES);
    }
}
