//! `MetalGraph` type: device + queue + pipelines + lazily-allocated buffers,
//! plus weight upload/caching, single-GEMV dispatch, and the fused FFN phase.

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::gpu_backend::metal_full_layer;
use crate::gpu_backend::metal_prefill;

use super::buffers::{alloc_buf, download_f32, upload_bytes, upload_f32, MetalBuffers};
use super::error::{MetalGraphError, MetalWeightHandle};
use super::pipelines::MetalPipelines;
use super::reformat::{reformat_q1_aos_to_soa, reformat_tq2_aos_to_soa};

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide singleton for `MetalGraph`.
static GLOBAL_METAL_GRAPH: OnceLock<Mutex<Option<Arc<MetalGraph>>>> = OnceLock::new();

/// Direct Metal dispatch engine for the FFN pipeline.
///
/// Holds a Metal device, command queue, pre-compiled pipeline states, and
/// lazily allocated intermediate buffers.  All FFN operations are encoded
/// into a single command buffer with a single compute encoder, then
/// committed and synchronously waited upon.
pub struct MetalGraph {
    pub(crate) device: Device,
    pub(crate) command_queue: CommandQueue,
    pub(crate) pipelines: MetalPipelines,
    /// Lazily allocated intermediate buffers, protected by a mutex for
    /// interior mutability (buffer contents are mutated on each dispatch).
    buffers: Mutex<Option<MetalBuffers>>,
    /// Lazy cache of GPU-resident weight buffers, keyed by `GpuWeightHandle` id.
    weight_cache: Mutex<HashMap<u64, Arc<MetalWeightHandle>>>,
    /// Lazily allocated KV cache for all layers.
    pub(crate) kv_cache: Mutex<Option<metal_full_layer::GpuKvCache>>,
    /// Lazily allocated full-layer intermediate buffers.
    pub(crate) full_layer_buffers: Mutex<Option<metal_full_layer::FullLayerBuffers>>,
    /// Lazily allocated logits output buffer for fused LM head dispatch.
    pub(crate) logits_buf: Mutex<Option<Buffer>>,
    /// Persistent 4-byte buffer for GPU argmax token ID output (greedy decoding).
    pub(crate) token_id_buf: Mutex<Option<Buffer>>,
    /// Lazily allocated prefill buffers for batch processing.
    pub(crate) prefill_buffers: Mutex<Option<metal_prefill::PrefillBuffers>>,
}

// Metal objects (Device, CommandQueue, etc.) are Send+Sync in the metal crate.
unsafe impl Send for MetalGraph {}
unsafe impl Sync for MetalGraph {}

impl MetalGraph {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Create a new `MetalGraph` bound to the system default Metal device.
    ///
    /// Compiles all MSL kernels into pipeline states.  This is an expensive
    /// operation — prefer `global()` for repeated use.
    pub fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let command_queue = device.new_command_queue();
        let pipelines = MetalPipelines::compile(&device)?;

        Ok(Self {
            device,
            command_queue,
            pipelines,
            buffers: Mutex::new(None),
            weight_cache: Mutex::new(HashMap::new()),
            kv_cache: Mutex::new(None),
            full_layer_buffers: Mutex::new(None),
            logits_buf: Mutex::new(None),
            token_id_buf: Mutex::new(None),
            prefill_buffers: Mutex::new(None),
        })
    }

    /// Get or create the process-wide `MetalGraph` singleton.
    pub fn global() -> Result<Arc<Self>, MetalGraphError> {
        let mutex = GLOBAL_METAL_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("MetalGraph lock poisoned".into()))?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let graph = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&graph));
        Ok(graph)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Weight management
    // ─────────────────────────────────────────────────────────────────────

    /// Upload raw packed weight bytes to a GPU-resident Metal buffer.
    ///
    /// The returned handle can be passed to `encode_ffn_phase` or
    /// `encode_gemv` without further copies.
    pub fn upload_weight(&self, data: &[u8]) -> Result<MetalWeightHandle, MetalGraphError> {
        let buffer = upload_bytes(&self.device, data)?;
        Ok(MetalWeightHandle {
            byte_len: data.len(),
            buffer,
        })
    }

    /// Get a cached `MetalWeightHandle` or upload raw bytes and cache it.
    ///
    /// `key` is typically the `GpuWeightHandle`'s `u64` ID.
    pub fn get_or_upload_weight(
        &self,
        key: u64,
        raw_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_weight(raw_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_weight`, but accepts a closure that produces the bytes.
    ///
    /// This avoids unnecessary allocation when the weight is already cached.
    pub fn get_or_upload_weight_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let bytes = data_fn();
        let handle = Arc::new(self.upload_weight(&bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Upload Q1_0_g128 weight bytes in SoA layout for optimal GPU coalescing.
    ///
    /// Automatically reformats AoS → SoA during upload. The returned handle
    /// contains weights in SoA format ready for V7 kernels.
    pub fn upload_q1_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        let soa_data = reformat_q1_aos_to_soa(aos_data).ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "Q1 SoA reformat failed: input length {} is not a multiple of 18",
                aos_data.len()
            ))
        })?;
        let buffer = upload_bytes(&self.device, &soa_data)?;
        Ok(MetalWeightHandle {
            byte_len: soa_data.len(),
            buffer,
        })
    }

    /// Get a cached SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_q1_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_q1_weight_soa(aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_q1_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_q1_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let aos_bytes = data_fn();
        let handle = Arc::new(self.upload_q1_weight_soa(&aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Upload TQ2_0_g128 (ternary) weight bytes in SoA layout.
    ///
    /// Reformats 34-byte AoS blocks `{ qs:[u8;32], d:f16 }` into SoA
    /// `[N × 2B scales][N × 32B qs]` ready for `gemv_tq2_g128_v1`.
    pub fn upload_tq2_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        let soa_data = reformat_tq2_aos_to_soa(aos_data).ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "TQ2 SoA reformat failed: input length {} is not a multiple of 34",
                aos_data.len()
            ))
        })?;
        let buffer = upload_bytes(&self.device, &soa_data)?;
        Ok(MetalWeightHandle {
            byte_len: soa_data.len(),
            buffer,
        })
    }

    /// Get a cached TQ2 SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_tq2_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let handle = Arc::new(self.upload_tq2_weight_soa(aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    /// Like `get_or_upload_tq2_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_tq2_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))?;
        if let Some(w) = cache.get(&key) {
            return Ok(Arc::clone(w));
        }
        let aos_bytes = data_fn();
        let handle = Arc::new(self.upload_tq2_weight_soa(&aos_bytes)?);
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Single GEMV dispatch
    // ─────────────────────────────────────────────────────────────────────

    /// Execute a single Q1_0_g128 GEMV: `output = weight × input`.
    ///
    /// `weight` must have been uploaded via `upload_weight`.
    /// `input` and `output` are CPU-side f32 slices.
    ///
    /// - `n_rows`: number of output rows (weight matrix rows)
    /// - `k`: number of input elements (weight matrix columns, must be multiple of 128)
    pub fn encode_gemv(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        if input.len() < k {
            return Err(MetalGraphError::EncodingFailed(format!(
                "input too short: need {k}, got {}",
                input.len()
            )));
        }
        if output.len() < n_rows {
            return Err(MetalGraphError::EncodingFailed(format!(
                "output too short: need {n_rows}, got {}",
                output.len()
            )));
        }

        let opts = MTLResourceOptions::StorageModeShared;
        let input_bytes = std::mem::size_of_val(input) as u64;
        let output_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;

        let input_buf = alloc_buf(&self.device, input_bytes, opts)?;
        let output_buf = alloc_buf(&self.device, output_bytes, opts)?;

        unsafe { upload_f32(&input_buf, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        self.dispatch_gemv_q1(
            encoder,
            &weight.buffer,
            &input_buf,
            &output_buf,
            n_rows as u32,
            k as u32,
        );

        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&output_buf, &mut output[..n_rows]) };

        Ok(())
    }

    /// Execute a single TQ2_0_g128 (ternary) GEMV: `output = weight × input`.
    ///
    /// Mirror of `encode_gemv` for ternary weights. `weight` must have been
    /// uploaded via [`upload_tq2_weight_soa`](Self::upload_tq2_weight_soa).
    pub fn encode_gemv_tq2(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        if input.len() < k {
            return Err(MetalGraphError::EncodingFailed(format!(
                "input too short: need {k}, got {}",
                input.len()
            )));
        }
        if output.len() < n_rows {
            return Err(MetalGraphError::EncodingFailed(format!(
                "output too short: need {n_rows}, got {}",
                output.len()
            )));
        }

        let opts = MTLResourceOptions::StorageModeShared;
        let input_bytes = std::mem::size_of_val(input) as u64;
        let output_bytes = (n_rows * std::mem::size_of::<f32>()) as u64;

        let input_buf = alloc_buf(&self.device, input_bytes, opts)?;
        let output_buf = alloc_buf(&self.device, output_bytes, opts)?;

        unsafe { upload_f32(&input_buf, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        self.dispatch_gemv_tq2(
            encoder,
            &weight.buffer,
            &input_buf,
            &output_buf,
            n_rows as u32,
            k as u32,
        );

        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe { download_f32(&output_buf, &mut output[..n_rows]) };

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // FFN phase dispatch (7 operations, 1 encoder)
    // ─────────────────────────────────────────────────────────────────────

    /// Encode the full FFN phase as 7 sequential operations in one encoder.
    ///
    /// # Data flow
    ///
    /// 1. Upload `hidden`, `attn_out`, `norm_weight` to GPU
    /// 2. GEMV(attn_proj_weight, attn_out) → proj_buf
    /// 3. residual_add(hidden_buf, proj_buf)
    /// 4. rmsnorm_weighted(hidden_buf, norm_weight_buf) → normed_buf
    /// 5. GEMV(gate_up_weight, normed_buf) → gate_up_buf
    /// 6. swiglu_fused(gate_up_buf) → swiglu_buf
    /// 7. GEMV(down_weight, swiglu_buf) → down_buf
    /// 8. residual_add(hidden_buf, down_buf)
    /// 9. Read hidden_buf back to `hidden`
    ///
    /// All operations share one command buffer and one encoder.  Metal's
    /// automatic hazard tracking on shared-mode buffers ensures correct
    /// ordering of read-after-write dependencies.
    ///
    /// # Parameters
    ///
    /// - `hidden`: hidden state (read + written in-place), length = `hidden_size`
    /// - `attn_out`: attention output, length = `hidden_size`
    /// - `norm_weight`: RMSNorm weight vector, length = `hidden_size`
    /// - `attn_proj_weight`: pre-uploaded Q1 weight handle (hidden×hidden)
    /// - `gate_up_weight`: pre-uploaded Q1 weight handle ((intermediate*2)×hidden)
    /// - `down_weight`: pre-uploaded Q1 weight handle (hidden×intermediate)
    /// - `hidden_size`: dimension of the hidden state
    /// - `intermediate_size`: dimension of the MLP intermediate layer
    /// - `eps`: RMSNorm epsilon (typically 1e-6)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        attn_proj_weight: &MetalWeightHandle,
        gate_up_weight: &MetalWeightHandle,
        down_weight: &MetalWeightHandle,
        hidden_size: usize,
        intermediate_size: usize,
        eps: f32,
    ) -> Result<(), MetalGraphError> {
        static FFN_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
        let call_num = FFN_CALL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let t_total = Instant::now();

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden too short: need {hidden_size}, got {}",
                hidden.len()
            )));
        }
        if attn_out.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "attn_out too short: need {hidden_size}, got {}",
                attn_out.len()
            )));
        }
        if norm_weight.len() < hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "norm_weight too short: need {hidden_size}, got {}",
                norm_weight.len()
            )));
        }

        // ── Ensure intermediate buffers ──────────────────────────────────
        let t0 = Instant::now();
        let guard = self.acquire_buffers(hidden_size, intermediate_size)?;
        let bufs = guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("buffers not allocated".into()))?;
        let dt_acquire = t0.elapsed();

        // ── Step 1: Upload CPU → GPU ─────────────────────────────────────
        let t1 = Instant::now();
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden[..hidden_size]);
            upload_f32(&bufs.attn_out_buf, &attn_out[..hidden_size]);
            upload_f32(&bufs.norm_weight_buf, &norm_weight[..hidden_size]);
        }
        let dt_upload = t1.elapsed();

        // ── Create command buffer + single encoder ───────────────────────
        let t2 = Instant::now();
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        let dt_encode_setup = t2.elapsed();

        let h = hidden_size as u32;
        let inter = intermediate_size as u32;

        // ── Step 2: GEMV(attn_proj, attn_out) → proj_buf ────────────────
        // n_rows = hidden_size, k = hidden_size
        self.dispatch_gemv_q1(
            encoder,
            &attn_proj_weight.buffer,
            &bufs.attn_out_buf,
            &bufs.proj_buf,
            h,
            h,
        );

        // ── Step 3: residual_add(hidden_buf, proj_buf) ───────────────────
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.proj_buf, h);

        // ── Step 4: rmsnorm_weighted(hidden_buf, norm_weight_buf) → normed_buf
        self.dispatch_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            &bufs.norm_weight_buf,
            &bufs.normed_buf,
            eps,
            h,
        );

        // ── Step 5: Fused gate+up+SwiGLU → swiglu_buf ──────────────────
        self.dispatch_fused_gate_up_swiglu(
            encoder,
            &gate_up_weight.buffer,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter,
            h,
        );

        // ── Step 7: GEMV(down, swiglu) → down_buf ───────────────────────
        // n_rows = hidden_size, k = intermediate_size
        self.dispatch_gemv_q1(
            encoder,
            &down_weight.buffer,
            &bufs.swiglu_buf,
            &bufs.down_buf,
            h,
            inter,
        );

        // ── Step 8: residual_add(hidden_buf, down_buf) ──────────────────
        self.dispatch_residual_add(encoder, &bufs.hidden_buf, &bufs.down_buf, h);

        // ── Commit and wait ──────────────────────────────────────────────
        encoder.end_encoding();
        cmd_buf.commit();
        let t3 = Instant::now();
        cmd_buf.wait_until_completed();
        let dt_gpu_wait = t3.elapsed();

        // ── Step 9: Read back ────────────────────────────────────────────
        let t4 = Instant::now();
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }
        let dt_download = t4.elapsed();

        let dt_total = t_total.elapsed();
        if call_num % 36 == 0 {
            tracing::debug!(
                "MetalGraph FFN #{}: acquire={}µs upload={}µs encode={}µs gpu_wait={}µs download={}µs total={}µs",
                call_num,
                dt_acquire.as_micros(),
                dt_upload.as_micros(),
                dt_encode_setup.as_micros(),
                dt_gpu_wait.as_micros(),
                dt_download.as_micros(),
                dt_total.as_micros(),
            );
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // QKV phase dispatch (single GEMV, 1 encoder)
    // ─────────────────────────────────────────────────────────────────────

    /// Encode a fused QKV projection as a single GEMV dispatch.
    ///
    /// This is a thin wrapper around [`encode_gemv`](Self::encode_gemv) that
    /// provides a named entry point specifically for the Q/K/V projection
    /// hot-path in `block.rs`.
    ///
    /// - `input`: normed hidden state (length ≥ `k`)
    /// - `output`: fused QKV output (length ≥ `n_rows`)
    /// - `weight`: pre-uploaded fused Q+K+V weight handle
    /// - `n_rows`: total output rows (q_rows + k_rows + v_rows)
    /// - `k`: input dimension (hidden_size)
    pub fn encode_qkv_phase(
        &self,
        input: &[f32],
        output: &mut [f32],
        weight: &MetalWeightHandle,
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        self.encode_gemv(weight, input, output, n_rows, k)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal: buffer management
    // ─────────────────────────────────────────────────────────────────────

    /// Acquire the intermediate buffer set, allocating or re-allocating as
    /// needed.  Returns a mutex guard whose inner `Option` is guaranteed to
    /// be `Some`.
    fn acquire_buffers(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<MetalBuffers>>, MetalGraphError> {
        let mut guard = self
            .buffers
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("buffer lock poisoned".into()))?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(hidden_size, intermediate_size),
            None => true,
        };

        if needs_alloc {
            *guard = Some(MetalBuffers::allocate(
                &self.device,
                hidden_size,
                intermediate_size,
            )?);
        }

        Ok(guard)
    }

    /// Expose the device reference for external buffer allocation.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
