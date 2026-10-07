//! CudaGraph GEMV encoding entry points for the 2-bit block formats:
//! `TQ2_0_g128` (ggml 42, qs-first) and PrismML's `PQ2_0` (ggml 142, d-first,
//! finding **F16**).
//!
//! Both formats share the SoA buffer layout the `gemv_tq2_g128_v1` kernel
//! reads (`[all d: N×2 B FP16 LE][all qs: N×32 B]`) and differ only in their
//! decode table for the reserved 2-bit code `0b11` (`0.0` for `TQ2_0_g128`,
//! `+2.0` for `PQ2_0`). The CUDA twin of MET-11 keeps that difference out of
//! the shared GEMV kernel entirely: `PQ2_0` gets its own decode-table kernel
//! ([`crate::gpu_backend::kernel_sources::cuda_qwen35_kernels`]) and its own
//! reformat/upload/GEMV path below, mirroring `gemv_tq2_g128_v1`'s dispatch
//! shape so both stay easy to compare.
//!
//! Both cached-GEMV entry points refuse a malformed launch geometry — a `k`
//! that is not a positive multiple of 128, a short `input`, or a cached
//! weight buffer that is not exactly `n_rows * (k / 128) * 34` bytes — as
//! [`CudaGraphError::InvalidDimensions`] before any launch.
//!
//! **CUDA is unvalidated**: no CUDA hardware has run the launches below, so
//! every one is compile-checked only.

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

use super::super::kernel_sources::cuda_qwen35_kernels::{
    gemv_2bit_check_dims, gemv_2bit_check_weight_len, CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC,
};
use super::types::{CudaGraphError, Pq2GemvBuffers, TernaryGemvBuffers};
use super::{compile_or_load_ptx, cudagraph_type::CudaGraph};

/// Reject a GEMV whose `k` is not a positive multiple of 128 (the 2-bit
/// block size — the SoA layout has no partial-block encoding), or whose
/// `input` is shorter than `k`: `encode_gemv_tq2_cached` used to slice
/// `&input[..k]` with neither check, panicking on a short buffer instead of
/// returning an error. Checked before any lock is taken or buffer allocated,
/// so a caller mistake never leaves the shared GEMV pool in a
/// resized-but-unused state. The predicate itself
/// ([`gemv_2bit_check_dims`]) is host-tested in `cuda_qwen35_kernels`.
fn check_gemv_2bit_dims(k: usize, input_len: usize, kernel: &str) -> Result<(), CudaGraphError> {
    gemv_2bit_check_dims(k, input_len)
        .map_err(|e| CudaGraphError::InvalidDimensions(format!("{kernel}: {e}")))
}

/// Reject a launch over a cached weight buffer whose byte length is not
/// exactly the `n_rows x k` 2-bit SoA size ([`gemv_2bit_check_weight_len`]):
/// the kernels derive the codes offset from `n_rows`, so a buffer uploaded
/// for a different row count would be read at the wrong offset, and past
/// its end when it is shorter. This is the kernels-crate entry point's own
/// copy of the guard `oxibonsai-model`'s fused ternary call site applies,
/// so no caller can bypass it.
fn check_gemv_2bit_weight(
    weight_len: usize,
    n_rows: usize,
    k: usize,
    kernel: &str,
) -> Result<(), CudaGraphError> {
    gemv_2bit_check_weight_len(weight_len, n_rows, k)
        .map_err(|e| CudaGraphError::InvalidDimensions(format!("{kernel}: {e}")))
}

/// Process-wide pool for [`CudaGraph::encode_gemv_pq2_cached`]'s input/output
/// buffers.
///
/// `PQ2_0` support (finding **F16**) adds no field to [`CudaGraph`] itself
/// (`cudagraph_type.rs` / `cudagraph_global_group.rs`), so this pool cannot
/// live as a `CudaGraph` field the way [`TernaryGemvBuffers`] does on
/// `self.tq2_gemv_buffers`. It follows the same process-wide-`OnceLock`
/// shape `cuda_prefill::state` and `cuda_full_layer`'s `full_layer_state`
/// already use for exactly this situation.
fn pq2_gemv_state() -> &'static Mutex<Option<Pq2GemvBuffers>> {
    static STATE: OnceLock<Mutex<Option<Pq2GemvBuffers>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

impl CudaGraph {
    /// Launch `gemv_tq2_g128_v1` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_gemv_tq2_v1(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&self.modules.gemv_tq2_g128_v1)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_tq2_v1 launch: {e}")))
    }
    /// Public wrapper — launch `gemv_tq2_g128_v1` directly from cached device slices.
    ///
    /// Used by the full-forward ternary path (`encode_layer_into_ternary`) where the
    /// weight is already on device and the input/output slices live in the shared
    /// `CudaFullLayerBuffers` — no H2D/D2H or pool allocation is needed.
    ///
    /// # Safety
    /// All slices must be valid device pointers allocated on `self.stream`.
    pub unsafe fn launch_gemv_tq2_v1_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        self.launch_gemv_tq2_v1(d_weight, d_input, d_output, n_rows, k)
    }

    /// Execute a TQ2 (ternary) GEMV using a pre-cached SoA weight handle.
    ///
    /// Uses a process-wide reusable input/output buffer pool that grows to fit
    /// the largest GEMV seen so far — eliminates the per-call cuMemAlloc/Free
    /// round-trip that otherwise dominates short-kernel dispatch overhead.
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] when `k` is not a positive
    /// multiple of 128, `input` is shorter than `k`, or the cached buffer is
    /// not exactly `n_rows * (k / 128) * 34` bytes (all refused before any
    /// launch); [`CudaGraphError::WeightNotFound`] for an unknown handle;
    /// [`CudaGraphError::DriverError`] for a device failure.
    pub fn encode_gemv_tq2_cached(
        &self,
        weight_id: u64,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        // Refuse a malformed `k` / short `input` before touching the weight
        // cache lock or the GEMV buffer pool, so a caller mistake never
        // panics on the `&input[..k]` slice below (it used to) and never
        // leaves the pool resized for a request that was rejected.
        check_gemv_2bit_dims(k, input.len(), "encode_gemv_tq2_cached")?;
        let d_weight = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache
                .get(&weight_id)
                .map(Arc::clone)
                .ok_or(CudaGraphError::WeightNotFound(weight_id))?
        };
        check_gemv_2bit_weight(d_weight.len(), n_rows, k, "encode_gemv_tq2_cached")?;
        let mut buf_guard = self
            .tq2_gemv_buffers
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match buf_guard.as_ref() {
            Some(b) => !b.fits(k, n_rows),
            None => true,
        };
        if needs_alloc {
            let in_cap = match buf_guard.as_ref() {
                Some(b) => b.input_capacity.max(k),
                None => k,
            };
            let out_cap = match buf_guard.as_ref() {
                Some(b) => b.output_capacity.max(n_rows),
                None => n_rows,
            };
            let d_input = self.stream.alloc_zeros::<f32>(in_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros tq2 input pool: {e}"))
            })?;
            let d_output = self.stream.alloc_zeros::<f32>(out_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros tq2 output pool: {e}"))
            })?;
            *buf_guard = Some(TernaryGemvBuffers {
                d_input,
                d_output,
                input_capacity: in_cap,
                output_capacity: out_cap,
            });
        }
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("tq2 gemv buffers missing".into()))?;
        {
            let mut d_in_view = bufs.d_input.slice_mut(0..k);
            self.stream
                .memcpy_htod(&input[..k], &mut d_in_view)
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_htod tq2 input: {e}")))?;
        }
        unsafe {
            self.launch_gemv_tq2_v1(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                n_rows as u32,
                k as u32,
            )?;
        }
        let mut host = vec![0.0f32; n_rows];
        {
            let d_out_view = bufs.d_output.slice(0..n_rows);
            self.stream
                .memcpy_dtoh(&d_out_view, &mut host[..n_rows])
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh tq2 output: {e}")))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync tq2: {e}")))?;
        Ok(host)
    }

    // ═══════════════════════════════════════════════════════════════════
    // PQ2_0 (finding F16, CUDA twin of MET-11)
    // ═══════════════════════════════════════════════════════════════════

    /// Launch `gemv_pq2_g128_v1` on the default stream.
    ///
    /// # Safety
    /// Caller must ensure all slices are valid device pointers on `self.stream`.
    unsafe fn launch_gemv_pq2_v1(
        &self,
        d_weight: &CudaSlice<u8>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let func = pq2_gemv_function(self)?;
        let grid_x = n_rows.div_ceil(8);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream
            .launch_builder(&func)
            .arg(d_weight)
            .arg(d_input)
            .arg(d_output)
            .arg(&n_rows)
            .arg(&k)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("gemv_pq2_v1 launch: {e}")))
    }

    /// Return a cached `PQ2_0` weight slice, or reformat AoS bytes to SoA
    /// and upload (finding **F16**).
    ///
    /// Mirrors [`Self::get_or_upload_weight_tq2_soa`], but for `PQ2_0`'s
    /// **d-first** 34-byte AoS block (`{ d: f16, qs: [u8; 32] }`, the mirror
    /// image of `TQ2_0_g128`'s qs-first layout — see
    /// `reformat_pq2_aos_bytes_to_soa`) and the `gemv_pq2_g128_v1` kernel,
    /// which differs from `gemv_tq2_g128_v1` only in its decode table
    /// (`0b11 -> +2.0` instead of `0.0`).
    ///
    /// **Unattributed**: never freed by
    /// [`Self::release_model_epoch`](super::cudagraph_type::CudaGraph::release_model_epoch)
    /// (finding F-M3); use
    /// [`Self::get_or_upload_weight_pq2_soa_for_epoch`] when the owning
    /// model is known.
    pub fn get_or_upload_weight_pq2_soa(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError> {
        self.get_or_upload_weight_pq2_soa_for_epoch(
            handle_id,
            aos_bytes,
            crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH,
        )
    }

    /// [`Self::get_or_upload_weight_pq2_soa`], attributing the upload to
    /// `model_epoch` so the model's `Drop` can free it (finding **F-M3**).
    pub fn get_or_upload_weight_pq2_soa_for_epoch(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
        model_epoch: u64,
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError> {
        let cached = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.get(&handle_id).map(Arc::clone)
        };
        if let Some(existing) = cached {
            self.register_model_weight(model_epoch, handle_id)?;
            return Ok(existing);
        }
        let soa = reformat_pq2_aos_bytes_to_soa(aos_bytes).ok_or_else(|| {
            CudaGraphError::WeightLayoutError(format!(
                "PQ2 AoS bytes length {} not divisible by 34",
                aos_bytes.len()
            ))
        })?;
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod pq2_soa: {e}")))?;
        let arc = Arc::new(d_weight);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.insert(handle_id, Arc::clone(&arc));
        }
        // Outside the cache lock: the lock order is always "weight cache,
        // then epoch registry" (matching the TQ2 upload path).
        self.register_model_weight(model_epoch, handle_id)?;
        Ok(arc)
    }

    /// Upload typed `PQ2_0` blocks in SoA layout under `handle_id`,
    /// replacing any buffer already cached there — the `PQ2_0` twin of
    /// [`Self::upload_weight_tq2_soa`] (finding **F16**), reformatted by
    /// `reformat_pq2_blocks_to_soa` for the `gemv_pq2_g128_v1` kernel.
    ///
    /// **Unattributed**: never freed by
    /// [`Self::release_model_epoch`](super::cudagraph_type::CudaGraph::release_model_epoch)
    /// (finding F-M3); use [`Self::upload_weight_pq2_soa_for_epoch`] when
    /// the owning model is known.
    ///
    /// # Errors
    /// [`CudaGraphError::InvalidDimensions`] for an empty block slice;
    /// [`CudaGraphError::DriverError`] when the upload fails.
    pub fn upload_weight_pq2_soa(
        &self,
        handle_id: u64,
        blocks: &[oxibonsai_core::BlockPQ2_0],
    ) -> Result<(), CudaGraphError> {
        self.upload_weight_pq2_soa_for_epoch(
            handle_id,
            blocks,
            crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH,
        )
    }

    /// [`Self::upload_weight_pq2_soa`], attributing the upload to
    /// `model_epoch` so the model's `Drop` can free it (finding **F-M3**).
    ///
    /// # Errors
    /// As [`Self::upload_weight_pq2_soa`].
    pub fn upload_weight_pq2_soa_for_epoch(
        &self,
        handle_id: u64,
        blocks: &[oxibonsai_core::BlockPQ2_0],
        model_epoch: u64,
    ) -> Result<(), CudaGraphError> {
        if blocks.is_empty() {
            return Err(CudaGraphError::InvalidDimensions(
                "upload_weight_pq2_soa: no PQ2_0 blocks to upload".to_string(),
            ));
        }
        let soa = reformat_pq2_blocks_to_soa(blocks);
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod pq2 blocks: {e}")))?;
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.insert(handle_id, Arc::new(d_weight));
        }
        self.register_model_weight(model_epoch, handle_id)?;
        Ok(())
    }

    /// Upload bytes that are **already** in the TQ2 SoA layout
    /// (`[N×2 B FP16 scales][N×32 B qs]`), with no AoS→SoA reformat —
    /// unlike [`Self::get_or_upload_weight_tq2_soa`] and
    /// [`Self::get_or_upload_weight_pq2_soa`], which both reformat their
    /// input.
    ///
    /// For `PTQ1_0` (ggml 143, finding **F13**): its 128 trit codes are
    /// already unpacked into this exact SoA shape by
    /// [`crate::gpu_backend::cuda_qwen35::ptq1_blocks_to_tq2_soa`] (the
    /// "lossless transcode PTQ1_0->2-bit" design target), so calling
    /// `get_or_upload_weight_tq2_soa` on that output would silently
    /// re-reformat already-SoA bytes as if they were AoS — reading `d`
    /// where `qs` starts and vice versa, corrupting every weight. This
    /// entry point is the one `PTQ1_0` callers must use instead; the
    /// resulting buffer serves `gemv_tq2_g128_v1`
    /// ([`crate::gpu_backend::cuda_kernels`]) exactly like a native
    /// `TQ2_0_g128` upload — `PTQ1_0`'s codes decode under the same
    /// `0b00/0b01/0b10` table, `0b11` never emitted by ternary data on
    /// either side.
    ///
    /// **Unattributed**: never freed by
    /// [`Self::release_model_epoch`](super::cudagraph_type::CudaGraph::release_model_epoch);
    /// use [`Self::upload_weight_soa_direct_for_epoch`] when the owning
    /// model is known.
    ///
    /// # Errors
    /// [`CudaGraphError::WeightLayoutError`] when `soa_bytes.len()` is not a
    /// positive multiple of 34.
    pub fn upload_weight_soa_direct(
        &self,
        handle_id: u64,
        soa_bytes: &[u8],
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError> {
        self.upload_weight_soa_direct_for_epoch(
            handle_id,
            soa_bytes,
            crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH,
        )
    }

    /// [`Self::upload_weight_soa_direct`], attributing the upload to
    /// `model_epoch` so the model's `Drop` can free it.
    pub fn upload_weight_soa_direct_for_epoch(
        &self,
        handle_id: u64,
        soa_bytes: &[u8],
        model_epoch: u64,
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError> {
        const BLOCK_BYTES: usize = 34;
        let cached = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.get(&handle_id).map(Arc::clone)
        };
        if let Some(existing) = cached {
            self.register_model_weight(model_epoch, handle_id)?;
            return Ok(existing);
        }
        if soa_bytes.is_empty() || !soa_bytes.len().is_multiple_of(BLOCK_BYTES) {
            return Err(CudaGraphError::WeightLayoutError(format!(
                "upload_weight_soa_direct: {} bytes not a positive multiple of {BLOCK_BYTES}",
                soa_bytes.len()
            )));
        }
        let d_weight = self
            .stream
            .clone_htod(soa_bytes)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod soa_direct: {e}")))?;
        let arc = Arc::new(d_weight);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.insert(handle_id, Arc::clone(&arc));
        }
        self.register_model_weight(model_epoch, handle_id)?;
        Ok(arc)
    }

    /// Execute a `PQ2_0` GEMV using a pre-cached SoA weight handle (finding
    /// **F16**).
    ///
    /// Mirrors [`Self::encode_gemv_tq2_cached`] exactly (including its own
    /// process-wide reusable input/output buffer pool,
    /// `pq2_gemv_state`), swapping the kernel and the pool so a `PQ2_0`
    /// GEMV never contends with a concurrent `TQ2_0_g128` one over the same
    /// buffers.
    ///
    /// # Errors
    /// As [`Self::encode_gemv_tq2_cached`] — including the refusal of a
    /// cached buffer that is not exactly `n_rows * (k / 128) * 34` bytes.
    ///
    /// **CUDA is unvalidated**: no CUDA hardware has run this GEMV.
    pub fn encode_gemv_pq2_cached(
        &self,
        weight_id: u64,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, CudaGraphError> {
        // The same `k` / short-`input` guard as `encode_gemv_tq2_cached`,
        // applied to the PQ2 twin from the start rather than retrofitted:
        // refuse before touching any lock or buffer.
        check_gemv_2bit_dims(k, input.len(), "encode_gemv_pq2_cached")?;
        let d_weight = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache
                .get(&weight_id)
                .map(Arc::clone)
                .ok_or(CudaGraphError::WeightNotFound(weight_id))?
        };
        check_gemv_2bit_weight(d_weight.len(), n_rows, k, "encode_gemv_pq2_cached")?;
        let mut buf_guard = pq2_gemv_state()
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        let needs_alloc = match buf_guard.as_ref() {
            Some(b) => !b.fits(k, n_rows),
            None => true,
        };
        if needs_alloc {
            let in_cap = match buf_guard.as_ref() {
                Some(b) => b.input_capacity.max(k),
                None => k,
            };
            let out_cap = match buf_guard.as_ref() {
                Some(b) => b.output_capacity.max(n_rows),
                None => n_rows,
            };
            let d_input = self.stream.alloc_zeros::<f32>(in_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros pq2 input pool: {e}"))
            })?;
            let d_output = self.stream.alloc_zeros::<f32>(out_cap).map_err(|e| {
                CudaGraphError::DriverError(format!("alloc_zeros pq2 output pool: {e}"))
            })?;
            *buf_guard = Some(Pq2GemvBuffers {
                d_input,
                d_output,
                input_capacity: in_cap,
                output_capacity: out_cap,
            });
        }
        let bufs = buf_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("pq2 gemv buffers missing".into()))?;
        {
            let mut d_in_view = bufs.d_input.slice_mut(0..k);
            self.stream
                .memcpy_htod(&input[..k], &mut d_in_view)
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_htod pq2 input: {e}")))?;
        }
        unsafe {
            self.launch_gemv_pq2_v1(
                &d_weight,
                &bufs.d_input,
                &mut bufs.d_output,
                n_rows as u32,
                k as u32,
            )?;
        }
        let mut host = vec![0.0f32; n_rows];
        {
            let d_out_view = bufs.d_output.slice(0..n_rows);
            self.stream
                .memcpy_dtoh(&d_out_view, &mut host[..n_rows])
                .map_err(|e| CudaGraphError::DriverError(format!("memcpy_dtoh pq2 output: {e}")))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("stream sync pq2: {e}")))?;
        Ok(host)
    }
}

/// Compile and cache `gemv_pq2_g128_v1` (finding **F16**). Idempotent.
///
/// The kernel source is fully self-contained (its own copy of
/// `q35_fp16_to_f32`, not a shared prelude) — see
/// [`CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC`]'s doc for why.
fn pq2_gemv_function(graph: &CudaGraph) -> Result<Arc<CudaFunction>, CudaGraphError> {
    static STATE: OnceLock<Mutex<Option<Arc<CudaFunction>>>> = OnceLock::new();
    let state = STATE.get_or_init(|| Mutex::new(None));
    let mut guard = state.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
    if let Some(ref f) = *guard {
        return Ok(Arc::clone(f));
    }
    let ptx = compile_or_load_ptx(CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC, "gemv_pq2_g128_v1")?;
    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module gemv_pq2: {e}")))?;
    let func = module
        .load_function("gemv_pq2_g128_v1")
        .map_err(|e| CudaGraphError::DriverError(format!("load_function gemv_pq2_g128_v1: {e}")))?;
    let arc = Arc::new(func);
    *guard = Some(Arc::clone(&arc));
    Ok(arc)
}

/// Reformat raw `PQ2_0` AoS bytes to SoA layout (finding **F16**, CUDA twin
/// of MET-11 / Metal's `reformat_pq2_aos_to_soa`).
///
/// Each AoS block is 34 bytes in PrismML's `block_pq2_0` field order
/// (`ggml-common.h`): `[d: f16 LE (2 bytes)][qs: [u8; 32]]` — the FP16 scale
/// **first**, the mirror image of `TQ2_0_g128`'s qs-first layout
/// (`reformat_tq2_aos_bytes_to_soa`,
/// [`cudagraph_reformat_tq2_blocks_to_soa_group`](super::cudagraph_reformat_tq2_blocks_to_soa_group)).
///
/// SoA output is **identical in shape** to the TQ2 one — `[N×2 bytes FP16
/// scales][N×32 bytes qs]` — which is exactly why `gemv_pq2_g128_v1` can
/// reuse `gemv_tq2_g128_v1`'s dispatch shape unchanged: the two formats
/// differ only in their decode table, never in the buffer a GEMV kernel
/// reads.
///
/// Returns `None` when `aos_bytes.len()` is not a positive multiple of 34.
///
/// [`reformat_pq2_blocks_to_soa`] is the typed twin over
/// `&[oxibonsai_core::BlockPQ2_0]`; the two produce byte-identical output
/// for the same blocks.
pub(super) fn reformat_pq2_aos_bytes_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
    const BLOCK_BYTES: usize = 34;
    const SCALE_BYTES: usize = 2;
    const QS_BYTES: usize = 32;
    if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(BLOCK_BYTES) {
        return None;
    }
    let n = aos_bytes.len() / BLOCK_BYTES;
    let mut soa = Vec::with_capacity(aos_bytes.len());
    // Scales pass: the FP16 scale is the FIRST 2 bytes of each block
    // (`{ d, qs }` field order — the mirror image of TQ2's `{ qs, d }`).
    for i in 0..n {
        let src = i * BLOCK_BYTES;
        soa.extend_from_slice(&aos_bytes[src..src + SCALE_BYTES]);
    }
    // Quant codes pass: the 32 qs bytes are the LAST 32 bytes of each block.
    for i in 0..n {
        let src = i * BLOCK_BYTES + SCALE_BYTES;
        soa.extend_from_slice(&aos_bytes[src..src + QS_BYTES]);
    }
    Some(soa)
}

/// Reformat typed `PQ2_0` blocks to SoA bytes (finding **F16**) — the typed
/// twin of [`reformat_pq2_aos_bytes_to_soa`], as the `TQ2_0_g128` path's
/// `reformat_tq2_blocks_to_soa` is of `reformat_tq2_aos_bytes_to_soa`.
///
/// Output: `[N×2 bytes FP16 scales LE][N×32 bytes qs]`, byte-identical to
/// [`reformat_pq2_aos_bytes_to_soa`] over the same blocks' on-disk bytes.
/// It reads the typed `d` / `qs` fields, so the d-first vs qs-first offset
/// question the byte reformatter has to get right cannot arise here.
pub(super) fn reformat_pq2_blocks_to_soa(blocks: &[oxibonsai_core::BlockPQ2_0]) -> Vec<u8> {
    let mut soa = Vec::with_capacity(blocks.len() * 34);
    for block in blocks {
        soa.extend_from_slice(&block.d.to_bits().to_le_bytes());
    }
    for block in blocks {
        soa.extend_from_slice(&block.qs);
    }
    soa
}

#[cfg(test)]
mod pq2_tests {
    use super::*;

    /// Hand-built 3-block d-first AoS input, matching
    /// `metal_graph::reformat`'s `pq2_reformat_round_trips_hand_built_blocks_bit_exactly`
    /// fixture (scales `1.0`, `2.0`, `0.5`; `qs` bytes `1..=32` per block) so
    /// the CUDA and Metal reformatters can be compared by inspection even
    /// though this host cannot run both.
    fn hand_built_pq2_aos() -> Vec<u8> {
        let scales: [u16; 3] = [0x3C00, 0x4000, 0x3800]; // f16 1.0, 2.0, 0.5
        let mut aos = Vec::with_capacity(3 * 34);
        for &s in &scales {
            aos.extend_from_slice(&s.to_le_bytes());
            for j in 1u8..=32u8 {
                aos.push(j);
            }
        }
        aos
    }

    #[test]
    fn pq2_reformat_moves_the_leading_scale_and_trailing_qs_to_soa_sections() {
        let aos = hand_built_pq2_aos();
        let soa = reformat_pq2_aos_bytes_to_soa(&aos).expect("34 * 3 bytes reformats");
        assert_eq!(soa.len(), aos.len());
        // Scales section: 3 * 2 bytes, in block order.
        assert_eq!(u16::from_le_bytes([soa[0], soa[1]]), 0x3C00);
        assert_eq!(u16::from_le_bytes([soa[2], soa[3]]), 0x4000);
        assert_eq!(u16::from_le_bytes([soa[4], soa[5]]), 0x3800);
        // Qs section starts right after: block 0's 32 bytes are 1..=32.
        let qs0 = &soa[6..6 + 32];
        assert_eq!(qs0, (1u8..=32u8).collect::<Vec<u8>>().as_slice());
    }

    #[test]
    fn pq2_reformat_disagrees_with_a_qs_first_reading_of_the_same_bytes() {
        // One PQ2 block: scale 1.0 (d-first), qs = 1..=32. This is the
        // canonical MET-11 / F16 trap: `TQ2_0_g128`'s reformatter
        // (`cudagraph_reformat_tq2_blocks_to_soa_group::
        // reformat_tq2_aos_bytes_to_soa`, private to that module and so not
        // called directly here) would read the SAME 34 bytes as
        // qs-first — its first 32 bytes as `qs` and its last 2 as the
        // scale — silently swapping the real scale (`1.0`) for a "scale"
        // built from `qs[30..32]` (`0x1F1E` as raw bits) and losing the
        // real trailing two `qs` bytes entirely. This test reproduces that
        // qs-first reading inline (that reformatter is private to its own
        // module) and asserts PQ2's own, correct reformat differs from it.
        let mut aos = Vec::with_capacity(34);
        aos.extend_from_slice(&0x3C00u16.to_le_bytes());
        for j in 1u8..=32u8 {
            aos.push(j);
        }
        let pq2 = reformat_pq2_aos_bytes_to_soa(&aos).expect("pq2 reformat");

        let mut qs_first_reading = Vec::with_capacity(34);
        qs_first_reading.extend_from_slice(&aos[32..34]); // "scale" = last 2 bytes
        qs_first_reading.extend_from_slice(&aos[0..32]); // "qs" = first 32 bytes

        assert_ne!(
            pq2, qs_first_reading,
            "PQ2_0's own d-first reformat must not coincide with a qs-first \
             misreading of the same bytes — the exact MET-11-class corruption \
             the two dedicated reformatters (one per format) exist to prevent"
        );
    }

    /// Three typed blocks with distinct scales and codes (every 2-bit code
    /// 0..=3 appears, including the reserved `+2` code `0b11`).
    fn typed_pq2_blocks() -> Vec<oxibonsai_core::BlockPQ2_0> {
        [1.0f32, -0.5, 0.25]
            .iter()
            .enumerate()
            .map(|(b, &d)| {
                let mut qs = [0u8; 32];
                for (j, q) in qs.iter_mut().enumerate() {
                    *q = ((b * 37 + j * 11) % 256) as u8;
                }
                oxibonsai_core::BlockPQ2_0 {
                    d: half::f16::from_f32(d),
                    qs,
                }
            })
            .collect()
    }

    /// The blocks' on-disk (`d`-first) AoS bytes, built field by field.
    fn aos_bytes_of(blocks: &[oxibonsai_core::BlockPQ2_0]) -> Vec<u8> {
        let mut aos = Vec::with_capacity(blocks.len() * 34);
        for block in blocks {
            aos.extend_from_slice(&block.d.to_bits().to_le_bytes());
            aos.extend_from_slice(&block.qs);
        }
        aos
    }

    #[test]
    fn pq2_typed_reformat_matches_the_byte_reformat() {
        let blocks = typed_pq2_blocks();
        let typed = reformat_pq2_blocks_to_soa(&blocks);
        let bytes = reformat_pq2_aos_bytes_to_soa(&aos_bytes_of(&blocks))
            .expect("three 34-byte blocks reformat");
        assert_eq!(
            typed, bytes,
            "typed and byte PQ2 reformats must agree bit for bit"
        );
        assert!(reformat_pq2_blocks_to_soa(&[]).is_empty());
    }

    #[test]
    fn pq2_typed_soa_decoded_with_the_kernel_table_matches_core_dequant() {
        // The whole F16 chain on the host: typed blocks -> SoA -> the
        // `gemv_pq2_g128_v1` decode table (`decode_pq2_code`, host twin) must
        // reproduce `BlockPQ2_0::dequant` exactly, `0b11 -> +2` included.
        use crate::gpu_backend::kernel_sources::cuda_qwen35_kernels::decode_pq2_code;
        let blocks = typed_pq2_blocks();
        let n = blocks.len();
        let soa = reformat_pq2_blocks_to_soa(&blocks);
        let mut reference = vec![0.0f32; n * 128];
        oxibonsai_core::BlockPQ2_0::dequant(&blocks, &mut reference).expect("dequant");
        for b in 0..n {
            let d = half::f16::from_le_bytes([soa[2 * b], soa[2 * b + 1]]).to_f32();
            let qs = &soa[2 * n + 32 * b..2 * n + 32 * (b + 1)];
            for j in 0..128 {
                let code = (qs[j / 4] >> (2 * (j % 4))) & 0b11;
                assert_eq!(
                    decode_pq2_code(code) * d,
                    reference[b * 128 + j],
                    "block {b} lane {j}"
                );
            }
        }
    }

    #[test]
    fn pq2_reformat_rejects_misaligned_and_empty_input() {
        assert!(reformat_pq2_aos_bytes_to_soa(&[]).is_none());
        assert!(reformat_pq2_aos_bytes_to_soa(&[0u8; 33]).is_none());
        assert!(reformat_pq2_aos_bytes_to_soa(&[0u8; 35]).is_none());
        assert!(reformat_pq2_aos_bytes_to_soa(&[0u8; 68]).is_some());
    }

    #[test]
    fn check_gemv_2bit_dims_rejects_non_block_multiples_and_short_input() {
        // The exact defect class this guard exists for: a `k` that is not a
        // multiple of 128, and an `input` shorter than `k`, must be rejected
        // before any panic-prone slice — as `InvalidDimensions`.
        assert!(check_gemv_2bit_dims(128, 128, "t").is_ok());
        assert!(check_gemv_2bit_dims(0, 128, "t").is_err());
        assert!(check_gemv_2bit_dims(127, 128, "t").is_err());
        assert!(check_gemv_2bit_dims(129, 129, "t").is_err());
        assert!(matches!(
            check_gemv_2bit_dims(128, 127, "t"),
            Err(CudaGraphError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn check_gemv_2bit_weight_refuses_a_buffer_of_the_wrong_size() {
        assert!(check_gemv_2bit_weight(34 * 6, 6, 128, "t").is_ok());
        assert!(matches!(
            check_gemv_2bit_weight(34 * 4, 6, 128, "t"),
            Err(CudaGraphError::InvalidDimensions(_))
        ));
    }
}
