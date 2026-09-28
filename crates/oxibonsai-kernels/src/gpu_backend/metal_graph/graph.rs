//! `MetalGraph` type: one dispatch **session** — its own command queue and
//! lazily-allocated workspace buffers — over the process-shared
//! [`MetalDevice`] (device + pipelines + weight cache). Carries the per-format
//! weight uploads, single-GEMV dispatch, the DiT GEMM/attention entry points
//! and the fused FFN phase; construction, session binding and the shared
//! weight cache live in the `session` sibling module.

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::gpu_backend::metal_full_layer;
use crate::gpu_backend::metal_full_layer::types::{
    next_model_epoch, WeightKey, WeightKind, LEGACY_MODEL_EPOCH,
};
use crate::gpu_backend::metal_prefill;

use super::buffers::{
    alloc_buf, commit_and_wait, download_f32, upload_bytes, upload_f32, MetalBuffers,
};
use super::error::{MetalGraphError, MetalWeightHandle};
use super::pipelines::MetalPipelines;
use super::reformat::{
    reformat_pq2_aos_to_soa, reformat_q1_aos_to_soa, reformat_tq2_aos_to_soa,
    validate_tq2_ternary_codes, Q2_BLOCK_BYTES, Q2_QS_BYTES, Q2_SCALE_BYTES,
};

mod weight_cache;

/// Shared-device / per-session split (`MET-08`); see [`session`].
///
/// Declared here with an explicit `#[path]` (it resolves relative to this
/// file's directory, i.e. `metal_graph/session.rs`) so the module sits beside
/// its siblings rather than under `graph/`.
#[path = "session.rs"]
pub mod session;

pub use session::{MetalDevice, SessionScope};

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide count of pooled DiT-GEMM buffer (re)allocations.
///
/// Incremented once per *grow* of either the pooled input or output buffer in
/// [`MetalGraph::encode_gemm_tq2`] (i.e. each time a fresh `device.new_buffer`
/// backs the pool because the requested byte length exceeded the current cap).
/// Steady-state, after the pool has ramped to the largest shape a forward ever
/// requests, this stays flat — every subsequent matmul reuses the resident
/// buffers and triggers **zero** new allocations. Exposed via
/// [`MetalGraph::gemm_pool_alloc_count`] as a deterministic, load-independent
/// proof that the pool engages (vs. the pre-pool path, which allocated two
/// fresh buffers on *every* one of the ~100 matmuls/forward).
static GEMM_POOL_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of pooled joint-attention buffer (re)allocations.
///
/// Incremented once per *grow* of any pooled q/k/v/out buffer in the DiT
/// joint-attention pool ([`JointAttnIoPool`]). Steady-state, after the pool has
/// ramped to the largest attention shape a forward requests, this stays flat —
/// every subsequent call reuses the resident buffers and triggers zero new
/// allocations. Exposed via [`MetalGraph::joint_attn_pool_alloc_count`].
static JOINT_ATTN_POOL_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Resizable shared-storage I/O scratch for the DiT GEMM path
/// ([`MetalGraph::encode_gemm_tq2`]).
///
/// Holds one input and one output `StorageModeShared` buffer that **grow to the
/// max byte length ever seen and are reused** across all ~100 ternary matmuls of
/// a DiT forward (~400/image), eliminating the per-call alloc/free of a fresh
/// input buffer *and* a fresh output buffer (up to ~170 MB for the
/// `M=1536, N=27648` single-block projection). Caps are grow-only: a buffer is
/// reallocated solely when the requested size exceeds its current capacity,
/// never shrunk. An oversized buffer is harmless — the v9 kernel and
/// `download_f32` touch only the first `m*k` / `m*n_rows` floats, so reusing a
/// larger buffer for a smaller matmul is bit-identical to a tightly-sized one.
struct GemmIoPool {
    /// Shared-storage input scratch (`m*k` f32 written per call).
    input: Buffer,
    /// Allocated capacity of `input`, in bytes.
    input_cap_bytes: usize,
    /// Shared-storage output scratch (`m*n_rows` f32 read back per call).
    output: Buffer,
    /// Allocated capacity of `output`, in bytes.
    output_cap_bytes: usize,
}

/// Pre-allocated, grow-only shared-storage q/k/v/out buffers for the DiT
/// joint-attention path.
///
/// The *resident-scenario* buffer set for the flash joint-attention path: the
/// q/k/v inputs and the out output are each allocated once and reused across
/// every `encode_joint_attention_flash_pooled` call (and across the resident
/// kernel-only benchmark loop), eliminating the four `device.new_buffer` per
/// call. Grow-only: a buffer is reallocated solely when
/// the requested size exceeds its current capacity, never shrunk (an oversized
/// buffer is harmless — the kernel and `download_f32` touch only the first
/// `qkv_len` / `out_len` floats).
pub(super) struct JointAttnIoPool {
    /// Shared-storage q scratch (`num_heads*seq*head_dim` f32).
    pub(super) q: Buffer,
    /// Shared-storage k scratch (`num_heads*seq*head_dim` f32).
    pub(super) k: Buffer,
    /// Shared-storage v scratch (`num_heads*seq*head_dim` f32).
    pub(super) v: Buffer,
    /// Allocated capacity of each of `q`/`k`/`v`, in bytes (all grown together).
    qkv_cap_bytes: usize,
    /// Shared-storage out scratch (`seq*num_heads*head_dim` f32).
    pub(super) out: Buffer,
    /// Allocated capacity of `out`, in bytes.
    out_cap_bytes: usize,
}

/// One **session** of the direct Metal dispatch engine (`MET-08`).
///
/// A session owns the mutable GPU workspace of a single logical decoder: its
/// own `CommandQueue`, its own device KV cache, its own full-layer and prefill
/// intermediates, its own logits and argmax output buffers. Everything that is
/// immutable or deliberately shared — the `Device`, the compiled pipelines,
/// the weight cache and the DiT I/O pools — lives on the process-global
/// [`MetalDevice`] behind `shared`. N sessions therefore cost N KV caches, and
/// 1x the weights of a model whose replicas key each weight to one slot — the
/// ternary route today; the Q1 route still uploads one copy per replica (see
/// the `session` module's *Sizing reality* for the byte numbers).
///
/// Sessions are what make concurrent GPU inference possible: the `MutexGuard`s
/// a fused forward holds from `acquire_full_layer_buffers` through submission
/// and the synchronous completion wait are now *per session*, so two replicas
/// genuinely overlap instead of serialising on one another. See
/// [`MetalGraph::with_session`] for how a caller selects one, and the
/// `session` module docs for the sizing reality (604 MB of KV per session for
/// the 8B at `ctx = 4096`).
pub struct MetalGraph {
    /// Process-global device state shared with every other session: device,
    /// pipelines, weight cache, DiT I/O pools, upload counters.
    shared: Arc<MetalDevice>,
    /// The Metal device (a refcounted handle to the same `MTLDevice` as
    /// `shared.device`; kept as a field so the encode paths can reach it
    /// without a second indirection).
    pub(crate) device: Device,
    /// This session's **own** command queue, so submissions from two sessions
    /// overlap on the GPU instead of queueing behind each other.
    pub(crate) command_queue: CommandQueue,
    /// Compiled pipeline states, shared with every session (an `Arc` clone of
    /// `shared.pipelines`; field access auto-derefs).
    pub(crate) pipelines: Arc<MetalPipelines>,
    /// Process-unique session id, used by the thread-local binding.
    session_id: u64,
    /// Binding token of the thread that bound this session last (`0` =
    /// never bound). A thread whose binding no longer matches has had the
    /// session claimed away and treats its binding as stale — see the
    /// `session` module docs.
    pub(super) owner_thread: AtomicU64,
    /// Lazily allocated intermediate buffers, protected by a mutex for
    /// interior mutability (buffer contents are mutated on each dispatch).
    buffers: Mutex<Option<MetalBuffers>>,
    /// Lazily allocated KV cache for all layers — **per session**, which is
    /// what stops two replicas of the same model shape from trampling each
    /// other's attention state (`GpuKvCache::matches` compares only the
    /// shape).
    pub(crate) kv_cache: Mutex<Option<metal_full_layer::GpuKvCache>>,
    /// Lazily allocated full-layer intermediate buffers.
    pub(crate) full_layer_buffers: Mutex<Option<metal_full_layer::FullLayerBuffers>>,
    /// Lazily allocated logits output buffer for fused LM head dispatch.
    pub(crate) logits_buf: Mutex<Option<Buffer>>,
    /// Persistent 4-byte buffer for GPU argmax token ID output (greedy decoding).
    pub(crate) token_id_buf: Mutex<Option<Buffer>>,
    /// Lazily allocated prefill buffers for batch processing.
    ///
    /// Held as a [`metal_prefill::PrefillBufferCache`] — the buffer set plus
    /// the batch capacity it was allocated for — so the cache is **bucketed
    /// and grow-only** rather than exact-match on `batch_size`, which used to
    /// reallocate ~172 MB on every distinct prompt length (perf-M3).
    pub(crate) prefill_buffers: Mutex<Option<metal_prefill::PrefillBufferCache>>,
    /// Lazily compiled batched-prefill attention pipelines (perf-01).
    ///
    /// `None` inside the `OnceLock` means the two kernels are unavailable on
    /// this device/toolchain and prefill falls back to the per-token attention
    /// loop; they live in their own Metal library so the combined metallib
    /// shared with the image crate is untouched. Shared across sessions (an
    /// `Arc` clone of `shared.prefill_attn`) because it is a compiled library,
    /// not workspace state — a new session must not re-compile it. See
    /// `metal_prefill::attention`.
    pub(crate) prefill_attn: Arc<OnceLock<Option<metal_prefill::attention::PrefillAttnPipelines>>>,
}

// Metal objects (Device, CommandQueue, etc.) are Send+Sync in the metal crate.
unsafe impl Send for MetalGraph {}
unsafe impl Sync for MetalGraph {}

/// Debug-build cross-check: the directly-written SoA buffer must be byte-for-byte
/// what the allocating reformatter produces.
///
/// `reference` is a closure and the `cfg!` test is a plain early return, so a
/// release build never calls it (LLVM removes the branch) — the whole point of
/// MET-03 is not to allocate that second copy — while rustc still sees the
/// reference reformatters as used, so they cannot rot.
fn debug_check_q2_soa(buffer: &Buffer, reference: impl FnOnce() -> Option<Vec<u8>>, what: &str) {
    if !cfg!(debug_assertions) {
        return;
    }
    let Some(reference) = reference() else {
        return;
    };
    // SAFETY: `write_q2_soa_into_new_buffer` produced this shared-storage
    // buffer with a non-null `contents()` and a length of exactly the AoS byte
    // length, which is what the reformatter also returns.
    let written =
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, reference.len()) };
    assert_eq!(
        written, reference,
        "{what} direct SoA write diverged from reformat_*_aos_to_soa"
    );
}

/// Allocate a 2-bit SoA weight buffer and write both sections straight into it
/// (MET-03).
///
/// `split_block` receives one 34-byte AoS block and returns its `(qs, scale)`
/// slices, which is the only thing that differs between `TQ2_0_g128` (qs-first)
/// and `PQ2_0` (`d`-first). The buffer layout produced is the shared SoA one:
/// `[N x 2 B scales][N x 32 B qs]`.
///
/// Returns `None` when the input is empty or not a whole number of blocks —
/// the same rejection the allocating reformatter makes — and
/// `Some(Err(..))` when the Metal allocation itself fails.
#[allow(clippy::type_complexity)]
fn write_q2_soa_into_new_buffer<'a>(
    device: &Device,
    aos_bytes: &'a [u8],
    split_block: impl Fn(&'a [u8]) -> (&'a [u8], &'a [u8]),
) -> Option<Result<Buffer, MetalGraphError>> {
    if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(Q2_BLOCK_BYTES) {
        return None;
    }
    let n_blocks = aos_bytes.len() / Q2_BLOCK_BYTES;
    let buf = match alloc_buf(
        device,
        aos_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    ) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    // SAFETY: `alloc_buf` has already rejected a null `contents()` for a
    // shared-storage buffer, the allocation is `aos_bytes.len()` bytes, and
    // the two destination ranges below are disjoint and together exactly that
    // length. Nothing else can hold this buffer yet — it was created here.
    unsafe {
        let base = buf.contents() as *mut u8;
        let scales = base;
        let qs = base.add(n_blocks * Q2_SCALE_BYTES);
        for (i, block) in aos_bytes.chunks_exact(Q2_BLOCK_BYTES).enumerate() {
            let (block_qs, block_scale) = split_block(block);
            std::ptr::copy_nonoverlapping(
                block_scale.as_ptr(),
                scales.add(i * Q2_SCALE_BYTES),
                Q2_SCALE_BYTES,
            );
            std::ptr::copy_nonoverlapping(block_qs.as_ptr(), qs.add(i * Q2_QS_BYTES), Q2_QS_BYTES);
        }
    }
    Some(Ok(buf))
}

impl MetalGraph {
    // ─────────────────────────────────────────────────────────────────────
    // Weight management
    //
    // Construction, session selection and the (shared) weight cache live in
    // the `session` module; what stays here is the per-format upload/encode
    // machinery.
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
            kind: WeightKind::RawF32,
        })
    }

    /// Allocate a fresh model epoch for [`WeightKey`]s.
    ///
    /// Re-exported on `MetalGraph` so model code can reach it without naming
    /// the `metal_full_layer` module. One epoch per loaded model: carry it on
    /// the model, use it in every weight key, and pass it to
    /// [`Self::release_model`] on unload.
    #[must_use]
    pub fn next_model_epoch() -> u64 {
        next_model_epoch()
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
            kind: WeightKind::Q1Soa,
        })
    }

    /// Get a cached SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_q1_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::Q1Soa, key), || {
            self.upload_q1_weight_soa(aos_bytes)
        })
    }

    /// Like `get_or_upload_q1_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_q1_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::Q1Soa, key), || {
            self.upload_q1_weight_soa(&data_fn())
        })
    }

    /// Upload TQ2_0_g128 (ternary) weight bytes in SoA layout.
    ///
    /// Reformats 34-byte AoS blocks `{ qs:[u8;32], d:f16 }` (**qs first**) into
    /// SoA `[N × 2B scales][N × 32B qs]` ready for `gemv_tq2_g128_v1`.
    ///
    /// The block bytes are first screened for the reserved 2-bit code `0b11`,
    /// which ternary data never contains but `PQ2_0` uses for `+2`: a hit means
    /// a `PQ2_0` tensor reached the ternary path, whose `d`-first blocks would
    /// otherwise be reformatted with `qs[0..2]` as the scale and decoded with
    /// `0b11 → 0`, silently (MET-11). Use [`Self::upload_pq2_weight_soa`] for
    /// those. The scan is one linear pass over bytes the reformatter is about
    /// to read anyway.
    pub fn upload_tq2_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        if let Err(violation) = validate_tq2_ternary_codes(aos_data) {
            return Err(MetalGraphError::ExecutionFailed(format!(
                "TQ2 SoA upload rejected: {violation}"
            )));
        }
        // MET-03: allocate the destination buffer FIRST and write the two SoA
        // sections straight into it. The previous shape —
        // `reformat_tq2_aos_to_soa(aos) -> Vec<u8>` then `upload_bytes` —
        // held a full second host copy of every matrix while it was being
        // uploaded, which for the 27B is a 7.2 GB load-time peak on top of the
        // mmap. The allocating reformatter is kept (it is the PQ2 path's
        // sibling and this function's unit-test oracle); only the upload path
        // stopped using it.
        let buffer = write_q2_soa_into_new_buffer(&self.device, aos_data, |block| {
            // TQ2_0_g128 AoS is qs-first: qs[0..32], then the f16 scale.
            (&block[..Q2_QS_BYTES], &block[Q2_QS_BYTES..])
        })
        .ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "TQ2 SoA reformat failed: input length {} is not a multiple of 34",
                aos_data.len()
            ))
        })??;
        debug_check_q2_soa(&buffer, || reformat_tq2_aos_to_soa(aos_data), "TQ2_0_g128");
        Ok(MetalWeightHandle {
            byte_len: aos_data.len(),
            buffer,
            kind: WeightKind::Tq2Soa,
        })
    }

    /// Get a cached TQ2 SoA weight handle or reformat AoS→SoA and upload.
    ///
    /// Keyed under [`LEGACY_MODEL_EPOCH`]; callers that own a model epoch use
    /// [`Self::get_or_upload_tq2_weight_soa_for_epoch`].
    pub fn get_or_upload_tq2_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_tq2_weight_soa_for_epoch(LEGACY_MODEL_EPOCH, key, aos_bytes)
    }

    /// Like `get_or_upload_tq2_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_tq2_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_tq2_weight_soa_lazy_for_epoch(LEGACY_MODEL_EPOCH, key, data_fn)
    }

    /// Epoch-keyed TQ2 SoA lookup-or-upload (`MET-02`).
    ///
    /// The buffer is cached under
    /// `WeightKey::new(model_epoch, WeightKind::Tq2Soa, slot)`, so
    /// [`Self::release_model`]`(model_epoch)` frees it together with the rest
    /// of that model, and a caller keyed under a *different* epoch can never
    /// be served it.
    ///
    /// # Errors
    ///
    /// Propagates the upload's validation/allocation failure, or a kind
    /// mismatch on the slot.
    pub fn get_or_upload_tq2_weight_soa_for_epoch(
        &self,
        model_epoch: u64,
        slot: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(
            WeightKey::new(model_epoch, WeightKind::Tq2Soa, slot),
            || self.upload_tq2_weight_soa(aos_bytes),
        )
    }

    /// Lazy twin of [`Self::get_or_upload_tq2_weight_soa_for_epoch`]: `data_fn`
    /// runs only on a cache miss, so a resident buffer never materialises its
    /// AoS bytes (the Q‖K‖V / gate‖up concatenations).
    ///
    /// # Errors
    ///
    /// As [`Self::get_or_upload_tq2_weight_soa_for_epoch`].
    pub fn get_or_upload_tq2_weight_soa_lazy_for_epoch(
        &self,
        model_epoch: u64,
        slot: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(
            WeightKey::new(model_epoch, WeightKind::Tq2Soa, slot),
            || self.upload_tq2_weight_soa(&data_fn()),
        )
    }

    /// Upload `PQ2_0` (PrismML ggml type 142) weight bytes in SoA layout.
    ///
    /// `PQ2_0` blocks are the mirror image of `TQ2_0_g128`: same 34 bytes and
    /// same 128 weights, but `d` comes **first** and the code `0b11` means
    /// `+2`. The SoA buffer this produces has the identical shape to the TQ2
    /// one — only the kernel's decode table differs (`decode_pq2`), which is
    /// why the two must never share a cache slot.
    pub fn upload_pq2_weight_soa(
        &self,
        aos_data: &[u8],
    ) -> Result<MetalWeightHandle, MetalGraphError> {
        // MET-03, same direct write as the TQ2 path above — and the one that
        // matters most, since `PQ2_0` is Bonsai 2's 7.2 GB 27B format.
        let buffer = write_q2_soa_into_new_buffer(&self.device, aos_data, |block| {
            // PQ2_0 AoS is d-first: the f16 scale, then qs[2..34].
            (&block[Q2_SCALE_BYTES..], &block[..Q2_SCALE_BYTES])
        })
        .ok_or_else(|| {
            MetalGraphError::ExecutionFailed(format!(
                "PQ2 SoA reformat failed: input length {} is not a multiple of 34",
                aos_data.len()
            ))
        })??;
        debug_check_q2_soa(&buffer, || reformat_pq2_aos_to_soa(aos_data), "PQ2_0");
        Ok(MetalWeightHandle {
            byte_len: aos_data.len(),
            buffer,
            kind: WeightKind::Pq2Soa,
        })
    }

    /// Get a cached PQ2 SoA weight handle or reformat AoS→SoA and upload.
    pub fn get_or_upload_pq2_weight_soa(
        &self,
        key: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::Pq2Soa, key), || {
            self.upload_pq2_weight_soa(aos_bytes)
        })
    }

    /// Like `get_or_upload_pq2_weight_soa`, but accepts a closure that produces AoS bytes.
    pub fn get_or_upload_pq2_weight_soa_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::Pq2Soa, key), || {
            self.upload_pq2_weight_soa(&data_fn())
        })
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
        commit_and_wait(cmd_buf, "encode_gemv_q1")?;

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
        commit_and_wait(cmd_buf, "encode_gemv_tq2")?;

        unsafe { download_f32(&output_buf, &mut output[..n_rows]) };

        Ok(())
    }

    /// Execute a batched TQ2_0_g128 (ternary) GEMM: `output = input × weightᵀ`.
    ///
    /// Batched counterpart to [`encode_gemv_tq2`](Self::encode_gemv_tq2),
    /// dispatching the tiled `gemm_tq2_g128_v8_tiled` kernel (2-D grid,
    /// register-blocked) optimized for the large-M DiT path; it handles
    /// arbitrary batch sizes (no cap-of-8) and is numerically equivalent to the
    /// serial-M `gemm_tq2_g128_v7`. `weight` must have been uploaded via
    /// [`get_or_upload_tq2_weight_soa`](Self::get_or_upload_tq2_weight_soa) (or
    /// the `_lazy` variant).
    ///
    /// # Layout
    ///
    /// `input` and `output` are **column-major** with the batch as the outer
    /// dimension, which coincides exactly with a row-major `[M, K]` / `[M, N]`:
    /// - `input[m * k + elem]` — row-major `[M, K]`, i.e. column `m` of the kernel.
    /// - `output[m * n_rows + row]` — row-major `[M, N]`, i.e. column `m` of the kernel.
    ///
    /// So a caller holding row-major `input[M,K]`, ternary `weight[N,K]`, and
    /// `out[M,N]` (computing `out = input · weightᵀ`) can pass its buffers
    /// directly — no transpose, no reshuffle.
    ///
    /// # Parameters
    ///
    /// - `weight`: pre-uploaded SoA TQ2 weight handle (`N` rows × `k` cols).
    /// - `input`: length `m * k` f32.
    /// - `output`: length `m * n_rows` f32 (overwritten).
    /// - `m`: batch size (rows of `A`); arbitrary, including values `> 8`.
    /// - `n_rows`: `N`, the number of weight rows.
    /// - `k`: inner dimension; **must** be a multiple of 128.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::InvalidDimensions`] if `k % 128 != 0`,
    /// `input.len() != m * k`, or `output.len() != m * n_rows`; or a buffer /
    /// execution error if the GPU work cannot be encoded.
    pub fn encode_gemm_tq2(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        if !k.is_multiple_of(128) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_tq2: k must be a multiple of 128, got {k}"
            )));
        }
        let expected_in = m.checked_mul(k).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_tq2: m*k overflow (m={m}, k={k})"
            ))
        })?;
        if input.len() != expected_in {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_tq2: input len {} != m*k {expected_in} (m={m}, k={k})",
                input.len()
            )));
        }
        let expected_out = m.checked_mul(n_rows).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_tq2: m*n_rows overflow (m={m}, n_rows={n_rows})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_tq2: output len {} != m*n_rows {expected_out} (m={m}, n_rows={n_rows})",
                output.len()
            )));
        }

        // Degenerate empty GEMM: nothing to do (avoids a zero-byte buffer alloc).
        if expected_in == 0 || expected_out == 0 {
            return Ok(());
        }

        // ── Acquire pooled shared-storage I/O buffers (resize-to-max) ─────
        // Unified memory: shared storage means the CPU write and the GPU read
        // alias the same pages, so wrapping `input` is a plain memcpy with no
        // device transfer. Instead of allocating a fresh `m*k` input buffer and
        // a fresh `m*n_rows` output buffer on every call (the big single-block
        // projection alone is ~170 MB, churned ~80×/image), reuse a
        // process-wide pool that grows to the largest shape ever requested. The
        // pool's buffers may be larger than this call needs after a prior grow;
        // that is bit-identical because the v9 kernel and `download_f32` only
        // ever touch the first `m*k` / `m*n_rows` floats.
        let input_bytes = std::mem::size_of_val(input);
        let output_bytes = expected_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_gemm_tq2: output byte size overflow (m={m}, n_rows={n_rows})"
                ))
            })?;

        // The pool lock is held across the ENTIRE encode (upload → dispatch →
        // commit → wait → download). The single shared input/output pair must
        // not be aliased by two concurrent `encode_gemm_tq2` calls writing
        // different matmuls into the same pages. The DiT forward is strictly
        // sequential (`forward.rs` runs one block, hence one matmul, at a time),
        // so this serialization is free in practice; were two callers to race,
        // they would simply queue on this mutex rather than corrupt each other.
        let mut pool_guard =
            self.shared.gemm_io_pool.lock().map_err(|_| {
                MetalGraphError::ExecutionFailed("gemm_io_pool lock poisoned".into())
            })?;
        let pool =
            Self::ensure_gemm_pool(&self.device, &mut pool_guard, input_bytes, output_bytes)?;

        unsafe { upload_f32(&pool.input, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        // DiT large-M path: use the staging-optimized simdgroup_matrix v10 GEMM,
        // which drives Apple's 8×8×8 hardware MAC units (f32 accumulate) but
        // optimizes v9's threadgroup staging — it stages the dequantized weight
        // as half (EXACT for ternary code×scale ∈ {-scale,0,+scale}, zero added
        // rounding) and spreads the dequant-scatter across all 128 threads. It
        // is numerically equivalent to v7/v8/v9 (unit parity max-abs err ≲
        // 1.2e-5 ≪ 1e-3; dit_parity cos ≥ 0.999) and measures ~3.86× faster than
        // v9 (≈2.6× over v8) on the big DiT shapes (M=1536, N∈{3072,27648},
        // K=3072). v8/v9 are retained as fallbacks; every LLM forward/prefill
        // path still calls dispatch_gemm_tq2_v7 unchanged.
        self.dispatch_gemm_tq2_v10(
            encoder,
            &weight.buffer,
            &pool.input,
            &pool.output,
            n_rows as u32,
            k as u32,
            m as u32,
        );

        encoder.end_encoding();
        commit_and_wait(cmd_buf, "encode_gemm_tq2")?;

        // `output.len()` is authoritative (== expected_out); the pooled output
        // buffer may be larger after a grow, so copy exactly the requested span.
        unsafe { download_f32(&pool.output, output) };

        Ok(())
    }

    /// Execute a batched **f32-exact** GEMM: `output = input × weightᵀ`.
    ///
    /// f32 sibling of [`encode_gemm_tq2`](Self::encode_gemm_tq2) for the FLUX.2
    /// text encoder (Qwen3-4B), whose weights are pure f32 (no quantized
    /// format). Dispatches the `gemm_f32_simdgroup` kernel (Apple
    /// `simdgroup_float8x8` 8×8×8 HW MACs, f32 accumulate), which is numerically
    /// equivalent to the CPU `gemm_abt` (cos ≈ 1.0 — reassociated sums only).
    /// Reuses the same process-wide `GemmIoPool` as the ternary path, so there
    /// is no per-call big I/O allocation.
    ///
    /// # Layout
    ///
    /// `input` / `output` are **column-major** with the batch as the outer
    /// dimension, which coincides exactly with a row-major `[M, K]` / `[M, N]`:
    /// - `input[m * k + elem]` — row-major `[M, K]`, i.e. column `m` of the kernel.
    /// - `output[m * n_rows + row]` — row-major `[M, N]`, i.e. column `m`.
    ///
    /// `weight` is the pre-uploaded **row-major f32** `[N, K]` handle (from
    /// [`get_or_upload_f32_weight`](Self::get_or_upload_f32_weight), keyed by the
    /// weight slice pointer and cached across forwards), so a caller holding
    /// row-major `input[M,K]`, f32 `weight[N,K]`, and `out[M,N]` (computing
    /// `out = input · weightᵀ`) can pass its buffers directly — no transpose.
    ///
    /// # Parameters
    ///
    /// - `weight`: pre-uploaded row-major f32 weight handle (`N` rows × `k` cols).
    /// - `input`: length `m * k` f32.
    /// - `output`: length `m * n_rows` f32 (overwritten).
    /// - `m`: batch size (rows of `A`); arbitrary.
    /// - `n_rows`: `N`, the number of weight rows.
    /// - `k`: inner dimension; arbitrary (`≥ 1`, no `% 128` constraint — the
    ///   f32 kernel clamps its last K-tile).
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::InvalidDimensions`] if `input.len() != m * k`,
    /// `output.len() != m * n_rows`, the weight handle is too small for `n_rows *
    /// k` f32, or a size overflow; or a buffer / execution error if the GPU work
    /// cannot be encoded.
    pub fn encode_gemm_f32(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        self.encode_gemm_simdgroup(weight, input, output, m, n_rows, k, false)
    }

    /// bf16-input / f32-accumulate sibling of [`Self::encode_gemm_f32`]. Same
    /// buffers and result shape; the operands are rounded to `bfloat` inside the
    /// `gemm_bf16_simdgroup` kernel for ~2× throughput. bf16 is the model's
    /// native precision, so parity is render-level (cos ≈ 1.0). Used for the
    /// text encoder; see `OXI_TE_GEMM_F32` to force the exact f32 path.
    ///
    /// If the bf16 kernel is unavailable on this device (M1/M2 or macOS < 14,
    /// see [`Self::bf16_gemm_available`]) the dispatch transparently uses the
    /// exact f32 kernel instead.
    ///
    /// # Errors
    /// As [`Self::encode_gemm_f32`].
    pub fn encode_gemm_bf16(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<(), MetalGraphError> {
        self.encode_gemm_simdgroup(weight, input, output, m, n_rows, k, true)
    }

    /// Whether the optional bf16 TE GEMM kernel compiled on this device.
    ///
    /// `false` on a device/toolchain without `bfloat` simdgroup_matrix support
    /// (M1/M2 or macOS < 14); callers (the TE GPU path) then route to
    /// [`Self::encode_gemm_f32`]. Even when this returns `true`,
    /// [`Self::encode_gemm_bf16`] independently re-checks and falls back, so a
    /// stale `true` can never produce a wrong result.
    pub fn bf16_gemm_available(&self) -> bool {
        self.pipelines.gemm_bf16_simdgroup.is_some()
    }

    /// Shared encode for the f32 and bf16 text-encoder GEMM kernels; `bf16`
    /// selects the staging precision (`gemm_bf16_simdgroup` vs
    /// `gemm_f32_simdgroup`). Everything else — validation, the resident I/O
    /// pool, and the f32 host buffers — is identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_gemm_simdgroup(
        &self,
        weight: &MetalWeightHandle,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
        bf16: bool,
    ) -> Result<(), MetalGraphError> {
        // ── Validate ─────────────────────────────────────────────────────
        let expected_in = m.checked_mul(k).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: m*k overflow (m={m}, k={k})"
            ))
        })?;
        if input.len() != expected_in {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: input len {} != m*k {expected_in} (m={m}, k={k})",
                input.len()
            )));
        }
        let expected_out = m.checked_mul(n_rows).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: m*n_rows overflow (m={m}, n_rows={n_rows})"
            ))
        })?;
        if output.len() != expected_out {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: output len {} != m*n_rows {expected_out} (m={m}, n_rows={n_rows})",
                output.len()
            )));
        }
        // The weight buffer must hold at least N*K f32 (row-major `[N,K]`).
        let weight_floats = n_rows.checked_mul(k).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: n_rows*k overflow (n_rows={n_rows}, k={k})"
            ))
        })?;
        let weight_bytes = weight_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_gemm_f32: weight byte size overflow (n_rows={n_rows}, k={k})"
                ))
            })?;
        if weight.byte_len < weight_bytes {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "encode_gemm_f32: weight handle holds {} bytes < n_rows*k*4 {weight_bytes} (n_rows={n_rows}, k={k})",
                weight.byte_len
            )));
        }

        // Degenerate empty GEMM: nothing to do (avoids a zero-byte buffer alloc).
        if expected_in == 0 || expected_out == 0 {
            return Ok(());
        }

        // ── Acquire pooled shared-storage I/O buffers (resize-to-max) ─────
        // Same resident pool as the ternary path: shared storage means the CPU
        // write and the GPU read alias the same pages, so wrapping `input` is a
        // plain memcpy with no device transfer. The pool grows to the largest
        // shape ever requested and is reused; a larger-than-needed buffer is
        // bit-identical because the kernel and `download_f32` only touch the
        // first `m*k` / `m*n_rows` floats.
        let input_bytes = std::mem::size_of_val(input);
        let output_bytes = expected_out
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                MetalGraphError::InvalidDimensions(format!(
                    "encode_gemm_f32: output byte size overflow (m={m}, n_rows={n_rows})"
                ))
            })?;

        // The pool lock is held across the ENTIRE encode (upload → dispatch →
        // commit → wait → download); the single shared input/output pair must
        // not be aliased by two concurrent encodes. The TE forward is strictly
        // sequential, so this serialization is free in practice.
        let mut pool_guard =
            self.shared.gemm_io_pool.lock().map_err(|_| {
                MetalGraphError::ExecutionFailed("gemm_io_pool lock poisoned".into())
            })?;
        let pool =
            Self::ensure_gemm_pool(&self.device, &mut pool_guard, input_bytes, output_bytes)?;

        unsafe { upload_f32(&pool.input, input) };

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        // Text-encoder large-M path: the simdgroup_matrix GEMM. Same 64×64-tile /
        // 4-simdgroup shape as the ternary v9/v10, staging the f32 weight tile
        // directly (no dequant). The f32 kernel accumulates in f32 (parity with
        // the CPU gemm_abt, cos ≥ 0.999); the bf16 kernel rounds the operands to
        // bf16 — the model's native precision — for ~2× throughput at
        // render-level parity (cos ≈ 1.0).
        // bf16 is dispatched only when it was requested *and* the optional bf16
        // kernel actually compiled on this device (M3+/Metal 3.1). On M1/M2 or an
        // older toolchain `gemm_bf16_simdgroup` is `None`, so this transparently
        // falls back to the exact f32 kernel — same result shape, f32 numerics.
        let use_bf16 = bf16_dispatch_selected(bf16, self.pipelines.gemm_bf16_simdgroup.is_some());
        let bf16_pso = if use_bf16 {
            self.pipelines.gemm_bf16_simdgroup.as_ref()
        } else {
            None
        };
        match bf16_pso {
            Some(pso) => self.dispatch_gemm_bf16(
                pso,
                encoder,
                &weight.buffer,
                &pool.input,
                &pool.output,
                n_rows as u32,
                k as u32,
                m as u32,
            ),
            None => self.dispatch_gemm_f32(
                encoder,
                &weight.buffer,
                &pool.input,
                &pool.output,
                n_rows as u32,
                k as u32,
                m as u32,
            ),
        }

        encoder.end_encoding();
        commit_and_wait(cmd_buf, "encode_gemm_simdgroup")?;

        // `output.len()` is authoritative (== expected_out); the pooled output
        // buffer may be larger after a grow, so copy exactly the requested span.
        unsafe { download_f32(&pool.output, output) };

        Ok(())
    }

    /// Ensure the pooled DiT-GEMM I/O buffers are each at least the requested
    /// byte length, allocating (growing) only the buffer(s) that are too small.
    ///
    /// Mirrors the resize-on-mismatch idiom of `acquire_buffers` /
    /// `acquire_prefill_buffers`, but **grows only** (never shrinks): once a
    /// buffer has reached the largest shape a forward requests it is reused for
    /// every subsequent, smaller matmul. Each fresh `alloc_buf` bumps
    /// [`GEMM_POOL_ALLOC_COUNT`] so the engagement of the pool is observable
    /// without timing. Returns a `&mut GemmIoPool` borrowed from `guard`,
    /// guaranteed to satisfy both requested capacities.
    fn ensure_gemm_pool<'g>(
        device: &Device,
        guard: &'g mut std::sync::MutexGuard<'_, Option<GemmIoPool>>,
        input_bytes: usize,
        output_bytes: usize,
    ) -> Result<&'g mut GemmIoPool, MetalGraphError> {
        let opts = MTLResourceOptions::StorageModeShared;

        match guard.as_mut() {
            // Fresh pool: allocate both buffers exactly once.
            None => {
                let input = alloc_buf(device, input_bytes as u64, opts)?;
                let output = alloc_buf(device, output_bytes as u64, opts)?;
                // Two buffers materialized for a previously-empty pool.
                GEMM_POOL_ALLOC_COUNT.fetch_add(2, Ordering::Relaxed);
                **guard = Some(GemmIoPool {
                    input,
                    input_cap_bytes: input_bytes,
                    output,
                    output_cap_bytes: output_bytes,
                });
            }
            // Existing pool: grow only the buffer(s) that no longer fit.
            Some(pool) => {
                if pool.input_cap_bytes < input_bytes {
                    pool.input = alloc_buf(device, input_bytes as u64, opts)?;
                    pool.input_cap_bytes = input_bytes;
                    GEMM_POOL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                if pool.output_cap_bytes < output_bytes {
                    pool.output = alloc_buf(device, output_bytes as u64, opts)?;
                    pool.output_cap_bytes = output_bytes;
                    GEMM_POOL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        guard
            .as_mut()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("gemm_io_pool not allocated".into()))
    }

    /// Number of pooled DiT-GEMM buffer (re)allocations since process start.
    ///
    /// Counts each *grow* of the pooled input or output buffer in
    /// [`encode_gemm_tq2`](Self::encode_gemm_tq2) (a fresh `device.new_buffer`
    /// because the requested byte length exceeded the resident capacity). After
    /// the pool ramps to the largest shape a forward requests, this value stays
    /// constant across all further matmuls — a deterministic, load-independent
    /// proof that the pool eliminates the pre-pool path's two-allocs-per-matmul
    /// churn. Read it before and after a warmed forward: the delta should be
    /// ~0, not ~200.
    pub fn gemm_pool_alloc_count() -> u64 {
        GEMM_POOL_ALLOC_COUNT.load(Ordering::Relaxed)
    }

    // ─────────────────────────────────────────────────────────────────────
    // DiT joint attention (standalone R&D)
    // ─────────────────────────────────────────────────────────────────────

    /// Validate the joint-attention dims/lengths against the kernel caps.
    ///
    /// Returns `(qkv_len, out_len, scale)` on success. Uses
    /// [`MetalGraphError::InvalidDimensions`] for every check (matching
    /// `encode_gemm_f32` / the VAE ops), since these are shape/length errors.
    fn joint_attn_validate(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(usize, usize, f32), MetalGraphError> {
        use crate::gpu_backend::kernel_sources::{DIT_ATTN_MAX_HEAD_DIM, DIT_ATTN_MAX_SEQ};

        if num_heads == 0 || seq == 0 || head_dim == 0 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention: dims must be non-zero (num_heads={num_heads}, seq={seq}, head_dim={head_dim})"
            )));
        }
        if seq > DIT_ATTN_MAX_SEQ {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention: seq {seq} exceeds kernel cap {DIT_ATTN_MAX_SEQ}"
            )));
        }
        if head_dim > DIT_ATTN_MAX_HEAD_DIM {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention: head_dim {head_dim} exceeds kernel cap {DIT_ATTN_MAX_HEAD_DIM}"
            )));
        }

        let qkv_len = num_heads * seq * head_dim;
        let out_len = seq * num_heads * head_dim;
        if q.len() < qkv_len || k.len() < qkv_len || v.len() < qkv_len {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention: q/k/v too short (need {qkv_len}, got {}/{}/{})",
                q.len(),
                k.len(),
                v.len()
            )));
        }
        if out.len() < out_len {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention: out too short (need {out_len}, got {})",
                out.len()
            )));
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        Ok((qkv_len, out_len, scale))
    }

    /// Ensure the pooled joint-attention buffers are each at least the requested
    /// byte length, growing only the buffer(s) that are too small.
    ///
    /// Mirrors [`Self::ensure_gemm_pool`]: grow-only (never shrinks). Each fresh
    /// `alloc_buf` bumps [`JOINT_ATTN_POOL_ALLOC_COUNT`]. The three q/k/v buffers
    /// share one capacity (they are always the same length) and grow together.
    pub(super) fn ensure_joint_attn_pool<'g>(
        device: &Device,
        guard: &'g mut std::sync::MutexGuard<'_, Option<JointAttnIoPool>>,
        qkv_bytes: usize,
        out_bytes: usize,
    ) -> Result<&'g mut JointAttnIoPool, MetalGraphError> {
        let opts = MTLResourceOptions::StorageModeShared;

        match guard.as_mut() {
            None => {
                let q = alloc_buf(device, qkv_bytes as u64, opts)?;
                let k = alloc_buf(device, qkv_bytes as u64, opts)?;
                let v = alloc_buf(device, qkv_bytes as u64, opts)?;
                let out = alloc_buf(device, out_bytes as u64, opts)?;
                // q + k + v + out materialized for a previously-empty pool.
                JOINT_ATTN_POOL_ALLOC_COUNT.fetch_add(4, Ordering::Relaxed);
                **guard = Some(JointAttnIoPool {
                    q,
                    k,
                    v,
                    qkv_cap_bytes: qkv_bytes,
                    out,
                    out_cap_bytes: out_bytes,
                });
            }
            Some(pool) => {
                if pool.qkv_cap_bytes < qkv_bytes {
                    pool.q = alloc_buf(device, qkv_bytes as u64, opts)?;
                    pool.k = alloc_buf(device, qkv_bytes as u64, opts)?;
                    pool.v = alloc_buf(device, qkv_bytes as u64, opts)?;
                    pool.qkv_cap_bytes = qkv_bytes;
                    JOINT_ATTN_POOL_ALLOC_COUNT.fetch_add(3, Ordering::Relaxed);
                }
                if pool.out_cap_bytes < out_bytes {
                    pool.out = alloc_buf(device, out_bytes as u64, opts)?;
                    pool.out_cap_bytes = out_bytes;
                    JOINT_ATTN_POOL_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        guard
            .as_mut()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("joint_attn_pool not allocated".into()))
    }

    /// Number of pooled joint-attention buffer (re)allocations since process
    /// start (sum of all q/k/v/out grows). After the pool ramps to the largest
    /// shape requested this stays flat — a deterministic, load-independent proof
    /// that the pool eliminates the standalone path's four-allocs-per-call churn.
    pub fn joint_attn_pool_alloc_count() -> u64 {
        JOINT_ATTN_POOL_ALLOC_COUNT.load(Ordering::Relaxed)
    }

    /// **Resident-scenario** entry point: upload q/k/v into the pooled GPU
    /// buffers **once**, before a kernel-only timing loop.
    ///
    /// Allocates/grows the `JointAttnIoPool` to the requested shape and copies
    /// q/k/v into the resident shared buffers. Pair with
    /// [`Self::joint_attn_flash_resident_dispatch`] (the per-iteration encode+wait,
    /// with **no** upload/download) and [`Self::joint_attn_resident_download`]
    /// (a single readback for verification). This is the honest model of the
    /// fused-resident DiT, where q/k/v are produced on-GPU by the preceding
    /// matmuls and the attention output is consumed on-GPU — so neither the
    /// upload nor the download is on the per-step critical path.
    ///
    /// # Errors
    /// As [`Self::encode_joint_attention_flash`] (dim/length validation against the
    /// kernel caps; the `out` slice is only length-checked here, not written).
    #[allow(clippy::too_many_arguments)]
    pub fn joint_attn_resident_prepare(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), MetalGraphError> {
        let (qkv_len, out_len, _scale) =
            Self::joint_attn_validate(q, k, v, out, num_heads, seq, head_dim)?;

        let qkv_bytes = qkv_len * std::mem::size_of::<f32>();
        let out_bytes = out_len * std::mem::size_of::<f32>();

        let mut guard = self.shared.joint_attn_pool.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("joint_attn_pool lock poisoned".into())
        })?;
        let pool = Self::ensure_joint_attn_pool(&self.device, &mut guard, qkv_bytes, out_bytes)?;

        unsafe {
            upload_f32(&pool.q, &q[..qkv_len]);
            upload_f32(&pool.k, &k[..qkv_len]);
            upload_f32(&pool.v, &v[..qkv_len]);
        }
        Ok(())
    }

    /// Download the resident joint-attention output (left in the pool by
    /// [`Self::joint_attn_flash_resident_dispatch`]) into a host slice, for
    /// one-shot parity verification after a kernel-only loop. Not on the timed
    /// path.
    ///
    /// # Errors
    /// Returns [`MetalGraphError::ExecutionFailed`] if the pool is unprepared or
    /// `out` is longer than the resident output capacity.
    pub fn joint_attn_resident_download(&self, out: &mut [f32]) -> Result<(), MetalGraphError> {
        let guard = self.shared.joint_attn_pool.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("joint_attn_pool lock poisoned".into())
        })?;
        let pool = guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed(
                "joint_attn_resident_download: pool not prepared".into(),
            )
        })?;
        let out_floats = pool.out_cap_bytes / std::mem::size_of::<f32>();
        if out.len() > out_floats {
            return Err(MetalGraphError::ExecutionFailed(format!(
                "joint_attn_resident_download: out len {} exceeds resident capacity {out_floats}",
                out.len()
            )));
        }
        unsafe { download_f32(&pool.out, out) };
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // DiT joint attention — FLASH (simdgroup_matrix) variant (standalone R&D)
    // ─────────────────────────────────────────────────────────────────────

    /// Validate the flash-attention shape: the shared
    /// [`Self::joint_attn_validate`] checks plus the extra constraints of the
    /// `joint_attention_flash_f32` kernel — `head_dim` must be a multiple of the
    /// `8`-wide matrix-unit edge and must not exceed `128` (the compile-time
    /// width of the kernel's `KVsh` threadgroup staging). The DiT `head_dim = 128`
    /// and all parity shapes satisfy both.
    pub(super) fn joint_attn_flash_validate(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(usize, usize, f32), MetalGraphError> {
        let (qkv_len, out_len, scale) =
            Self::joint_attn_validate(q, k, v, out, num_heads, seq, head_dim)?;
        if !head_dim.is_multiple_of(8) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention_flash: head_dim {head_dim} must be a multiple of 8"
            )));
        }
        if head_dim > 128 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention_flash: head_dim {head_dim} exceeds the flash kernel cap (128)"
            )));
        }
        Ok((qkv_len, out_len, scale))
    }

    /// Execute FLUX.2 DiT joint (txt+img) multi-head scaled-dot-product attention
    /// on the GPU via the flash-attention (online-softmax, `simdgroup_float8x8`
    /// HW-matrix) kernel `joint_attention_flash_f32` (the kernel built to beat the
    /// rayon+NEON CPU at the DiT shape), matching the CPU reference
    /// `oxibonsai-image::math::joint_attention` in behaviour.
    ///
    /// `q`, `k`, `v` are head-major `[num_heads × seq × head_dim]` f32 (RoPE
    /// already applied upstream to q,k). `out` receives the token-major
    /// transposed result `[seq × (num_heads*head_dim)]`. Non-causal (full
    /// bidirectional softmax over keys); `scale = 1/sqrt(head_dim)`. This is the
    /// **standalone** path: fresh input/output buffers are allocated per call.
    ///
    /// # Errors
    /// Returns [`MetalGraphError::InvalidDimensions`] if the slice lengths are
    /// inconsistent with `num_heads*seq*head_dim`, if any dimension is zero, if
    /// `seq`/`head_dim` exceed the kernel's compile-time caps, or if `head_dim` is
    /// not a multiple of `8` or exceeds the flash kernel's `128` cap.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_joint_attention_flash(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), MetalGraphError> {
        let (qkv_len, out_len, scale) =
            Self::joint_attn_flash_validate(q, k, v, out, num_heads, seq, head_dim)?;

        let opts = MTLResourceOptions::StorageModeShared;
        let qkv_bytes = (qkv_len * std::mem::size_of::<f32>()) as u64;
        let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;

        let q_buf = alloc_buf(&self.device, qkv_bytes, opts)?;
        let k_buf = alloc_buf(&self.device, qkv_bytes, opts)?;
        let v_buf = alloc_buf(&self.device, qkv_bytes, opts)?;
        let out_buf = alloc_buf(&self.device, out_bytes, opts)?;

        unsafe {
            upload_f32(&q_buf, &q[..qkv_len]);
            upload_f32(&k_buf, &k[..qkv_len]);
            upload_f32(&v_buf, &v[..qkv_len]);
        }

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_joint_attention_flash(
            encoder,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            num_heads as u32,
            seq as u32,
            head_dim as u32,
            scale,
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf, "encode_joint_attention_flash")?;

        unsafe { download_f32(&out_buf, &mut out[..out_len]) };

        Ok(())
    }

    /// Pooled / resident-buffer variant of [`Self::encode_joint_attention_flash`],
    /// reusing the process-wide `JointAttnIoPool` instead of allocating four
    /// fresh buffers per call. Still uploads q/k/v and downloads out each call, but
    /// pays zero per-call allocation after the pool warms up.
    ///
    /// # Errors
    /// As [`Self::encode_joint_attention_flash`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_joint_attention_flash_pooled(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), MetalGraphError> {
        let (qkv_len, out_len, scale) =
            Self::joint_attn_flash_validate(q, k, v, out, num_heads, seq, head_dim)?;

        let qkv_bytes = qkv_len * std::mem::size_of::<f32>();
        let out_bytes = out_len * std::mem::size_of::<f32>();

        let mut guard = self.shared.joint_attn_pool.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("joint_attn_pool lock poisoned".into())
        })?;
        let pool = Self::ensure_joint_attn_pool(&self.device, &mut guard, qkv_bytes, out_bytes)?;

        unsafe {
            upload_f32(&pool.q, &q[..qkv_len]);
            upload_f32(&pool.k, &k[..qkv_len]);
            upload_f32(&pool.v, &v[..qkv_len]);
        }

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_joint_attention_flash(
            encoder,
            &pool.q,
            &pool.k,
            &pool.v,
            &pool.out,
            num_heads as u32,
            seq as u32,
            head_dim as u32,
            scale,
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf, "encode_joint_attention_flash_pooled")?;

        unsafe { download_f32(&pool.out, &mut out[..out_len]) };

        Ok(())
    }

    /// **Resident-scenario** per-iteration dispatch for the **flash** kernel:
    /// encode + commit + wait against the already-resident pooled q/k/v (uploaded
    /// by [`Self::joint_attn_resident_prepare`]), with **no** host upload/download.
    /// This is the kernel-only number for the resident scenario — the honest
    /// model of the fused-resident DiT (q/k/v produced on-GPU, output consumed
    /// on-GPU), dispatching `joint_attention_flash_f32`.
    ///
    /// # Errors
    /// Returns [`MetalGraphError::InvalidDimensions`] for zero/over-cap dims (incl.
    /// the flash `head_dim` constraints), or [`MetalGraphError::ExecutionFailed`]
    /// if the pool is unprepared or too small for the requested shape.
    pub fn joint_attn_flash_resident_dispatch(
        &self,
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), MetalGraphError> {
        use crate::gpu_backend::kernel_sources::{DIT_ATTN_MAX_HEAD_DIM, DIT_ATTN_MAX_SEQ};

        if num_heads == 0 || seq == 0 || head_dim == 0 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention_flash: dims must be non-zero (num_heads={num_heads}, seq={seq}, head_dim={head_dim})"
            )));
        }
        if seq > DIT_ATTN_MAX_SEQ || head_dim > DIT_ATTN_MAX_HEAD_DIM {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention_flash: seq {seq} / head_dim {head_dim} exceed caps \
                 ({DIT_ATTN_MAX_SEQ}/{DIT_ATTN_MAX_HEAD_DIM})"
            )));
        }
        if !head_dim.is_multiple_of(8) || head_dim > 128 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "joint_attention_flash: head_dim {head_dim} must be a multiple of 8 and <= 128"
            )));
        }

        let qkv_len = num_heads * seq * head_dim;
        let out_len = seq * num_heads * head_dim;
        let qkv_bytes = qkv_len * std::mem::size_of::<f32>();
        let out_bytes = out_len * std::mem::size_of::<f32>();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let guard = self.shared.joint_attn_pool.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("joint_attn_pool lock poisoned".into())
        })?;
        let pool = guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed(
                "joint_attn_flash_resident_dispatch: pool not prepared (call joint_attn_resident_prepare first)"
                    .into(),
            )
        })?;
        if pool.qkv_cap_bytes < qkv_bytes || pool.out_cap_bytes < out_bytes {
            return Err(MetalGraphError::ExecutionFailed(
                "joint_attn_flash_resident_dispatch: resident pool smaller than requested shape"
                    .into(),
            ));
        }

        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        self.dispatch_joint_attention_flash(
            encoder,
            &pool.q,
            &pool.k,
            &pool.v,
            &pool.out,
            num_heads as u32,
            seq as u32,
            head_dim as u32,
            scale,
        );
        encoder.end_encoding();
        commit_and_wait(cmd_buf, "joint_attn_flash_resident_dispatch")?;
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
        // MET-04: `commit_and_wait` replaces the raw commit/wait pair and turns a
        // non-`Completed` command-buffer status into a named error instead of
        // letting the readback below hand back the previous token's bytes. The
        // `Instant` now spans commit+wait rather than wait alone — at batch 1
        // `commit()` is a submission, not a sync, so the measured GPU wait is
        // unchanged in practice, and `dt_gpu_wait` remains the GPU-bound term
        // of the per-call breakdown below.
        let t3 = Instant::now();
        commit_and_wait(cmd_buf, "encode_ffn_phase")?;
        let dt_gpu_wait = t3.elapsed();

        // ── Step 9: Read back ────────────────────────────────────────────
        let t4 = Instant::now();
        unsafe {
            download_f32(&bufs.hidden_buf, &mut hidden[..hidden_size]);
        }
        let dt_download = t4.elapsed();

        let dt_total = t_total.elapsed();
        if call_num.is_multiple_of(36) {
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

/// Pure selection for the TE GEMM staging precision: bf16 is dispatched only
/// when it was both requested (`OXI_TE_GEMM_F32` unset) *and* the optional
/// `gemm_bf16_simdgroup` pipeline compiled on this device (M3+/Metal 3.1). On a
/// device/toolchain without `bfloat` simdgroup support the pipeline is `None`
/// and the TE GEMM falls back to the exact f32 kernel. Factored out so the
/// fallback (the unsupported-device invariant) is unit-testable without a GPU.
#[inline]
fn bf16_dispatch_selected(bf16_requested: bool, bf16_available: bool) -> bool {
    bf16_requested && bf16_available
}

#[cfg(test)]
mod tests {
    use super::bf16_dispatch_selected;

    #[test]
    fn bf16_dispatch_selects_f32_when_pipeline_unavailable() {
        // The unsupported-device invariant: bf16 requested but the optional
        // kernel did not compile on this device → select the f32 kernel.
        assert!(!bf16_dispatch_selected(true, false));
        // bf16 requested and available → bf16.
        assert!(bf16_dispatch_selected(true, true));
        // f32 forced (OXI_TE_GEMM_F32=1) → f32 regardless of availability.
        assert!(!bf16_dispatch_selected(false, true));
        assert!(!bf16_dispatch_selected(false, false));
    }
}
