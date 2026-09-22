//! Batched prefill attention: pipelines, dispatch and buffer-capacity cache.
//!
//! This module carries the three pieces that turn the prefill attention phase
//! from `6 × batch × layer` tiny dispatches into **two batch-wide dispatches
//! per layer** (perf-01), plus the bucketed buffer cache that stops ~172 MB of
//! Metal buffers being reallocated on every distinct prompt length (perf-M3):
//!
//! 1. [`PrefillAttnPipelines`] — the `prefill_qkv_prepare` and
//!    `prefill_flash_attention` compute pipelines, compiled **lazily into
//!    their own Metal library**. `metal_graph/pipelines.rs`'s combined library
//!    and `build.rs`'s `ACTIVE_KERNELS` whitelist are outside this package's
//!    ownership; a separate best-effort library is the same shape the optional
//!    bf16 TE GEMM already uses (`try_compile_bf16_pipeline`), and it keeps the
//!    combined metallib — shared with the image crate's DiT/VAE paths — byte
//!    identical. A compile failure is **not** an error: it yields `None` and
//!    the caller falls back to the historical per-token attention loop.
//! 2. The two dispatch helpers, which own the grid geometry and must stay in
//!    sync with the `PFA_*` MSL constants (asserted in
//!    `kernel_sources::prefill`'s own tests).
//! 3. [`PrefillBufferCache`] — `PrefillBuffers` plus the batch capacity they
//!    were allocated for, so the cache can be **grow-only and bucketed**
//!    instead of exact-match on `batch_size`.

use metal::{Buffer, ComputePipelineState, Device, Library, MTLSize};
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::kernel_sources;
use super::super::metal_graph::{set_scalar, MetalGraph, MetalGraphError};
use super::types::PrefillBuffers;

// ═══════════════════════════════════════════════════════════════════════════
// Buffer-capacity cache (perf-M3)
// ═══════════════════════════════════════════════════════════════════════════

/// Granularity of the prefill buffer-capacity buckets, in prompt tokens.
///
/// `PrefillBuffers` used to be cached exact-match on `batch_size`, so every
/// distinct prompt length reallocated the whole set (~172 MB for the 8B at a
/// 2 k prompt). Rounding the requested batch up to a multiple of this constant
/// and reusing any resident set whose capacity is large enough collapses a
/// sweep of arbitrary prompt lengths onto a handful of allocations, and a
/// repeat of that sweep onto **zero** — see
/// [`MetalGraph::prefill_buffer_alloc_count`].
pub(crate) const PREFILL_BATCH_BUCKET: usize = 128;

/// Process-wide count of prefill buffer-set allocations.
///
/// Incremented once per `PrefillBuffers::allocate` — i.e. each time the
/// resident set is missing, too small, or has different model dimensions.
/// Exposed by [`MetalGraph::prefill_buffer_alloc_count`] so the bucketing can
/// be proven without timing.
static PREFILL_BUFFER_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of **batched** prefill attention layer-phases encoded.
///
/// Incremented once per layer that takes the two-dispatch path. Because the
/// per-token fallback is numerically equivalent, a parity test alone cannot
/// tell the two apart — this counter is how a caller proves the batched
/// kernels actually ran. Exposed by
/// [`MetalGraph::prefill_batched_attention_count`].
static PREFILL_BATCHED_ATTN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Round a requested prefill batch up to its buffer-capacity bucket.
#[inline]
#[must_use]
pub(crate) fn prefill_batch_capacity(batch_size: usize) -> usize {
    batch_size.max(1).div_ceil(PREFILL_BATCH_BUCKET) * PREFILL_BATCH_BUCKET
}

/// A resident `PrefillBuffers` set together with the batch capacity it was
/// allocated for.
///
/// `PrefillBuffers::matches` is exact on its `batch_size` argument and the
/// field is private (the type is owned by another package), so the capacity is
/// tracked here: the buffers are allocated *as if* the batch were
/// `capacity`, and every smaller batch reuses them. The column-major layout
/// makes an oversized set harmless — a batch of `n` touches only the first `n`
/// columns of each buffer.
pub(crate) struct PrefillBufferCache {
    /// The buffer set, sized for `capacity` prompt tokens.
    pub(crate) bufs: PrefillBuffers,
    /// Prompt tokens the set can hold (a multiple of [`PREFILL_BATCH_BUCKET`]).
    capacity: usize,
}

impl PrefillBufferCache {
    /// Allocate a set sized for `capacity` tokens and count the allocation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate(
        device: &Device,
        capacity: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, MetalGraphError> {
        let bufs = PrefillBuffers::allocate(
            device,
            capacity,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        )?;
        PREFILL_BUFFER_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        Ok(Self { bufs, capacity })
    }

    /// Whether this resident set can serve `batch_size` tokens at these model
    /// dimensions: same geometry, and capacity at least the requested batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn serves(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        self.capacity >= batch_size
            && self.bufs.matches(
                self.capacity,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            )
    }

    /// Lower bound for a reallocation at these dimensions.
    ///
    /// The resident capacity when the geometry is unchanged (so growing for a
    /// longer prompt never *shrinks* the set and a later shorter prompt still
    /// hits), or `0` when the dimensions differ — a different model's buffers
    /// carry no useful capacity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn growth_floor(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> usize {
        if self.bufs.matches(
            self.capacity,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        ) {
            self.capacity
        } else {
            0
        }
    }

    /// Capacity, in prompt tokens, of the resident set.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dimensions passed to both batched attention kernels
// ═══════════════════════════════════════════════════════════════════════════

/// Per-layer scalar arguments shared by `prefill_qkv_prepare` and
/// `prefill_flash_attention`.
///
/// Grouped into one struct because the two kernels take 8 and 11 scalars
/// respectively and every one of them is a dimension of the same layer.
pub(crate) struct PrefillAttnDims {
    /// Query heads.
    pub(crate) nq: u32,
    /// Key/value heads (GQA groups).
    pub(crate) nkv: u32,
    /// `nq / nkv`.
    pub(crate) heads_per_group: u32,
    /// Per-head width.
    pub(crate) head_dim: u32,
    /// RMSNorm epsilon for the Q/K head norms.
    pub(crate) eps: f32,
    /// KV-cache positions per head (`max_seq_len`).
    pub(crate) max_seq: u32,
    /// Absolute position of the first prompt token in this batch.
    pub(crate) pos_start: u32,
    /// Prompt tokens in this batch.
    pub(crate) batch_size: u32,
    /// `1 / sqrt(head_dim)`.
    pub(crate) scale: f32,
    /// Element offset of this layer's slab in the flat KV cache.
    ///
    /// Held as `u64` on the Rust side (`GpuKvCache::layer_offset_elements`)
    /// and narrowed to `u32` at the dispatch boundary, because the MSL
    /// bindings — like `fused_kv_store`'s and
    /// `batched_attention_scores_v2`'s — declare it `constant uint&`.
    /// `check_kv_cache_geometry` refuses any geometry whose total element
    /// count leaves the 32-bit range, so the narrowing cannot lose bits;
    /// widening the bindings to `ulong` belongs with that cap, not here.
    pub(crate) layer_offset: u64,
}

impl PrefillAttnDims {
    /// `(nq + 2·nkv) · head_dim` — the column stride of the batched QKV
    /// buffer, which is also the row stride of the Q operand.
    #[inline]
    fn qkv_dim(&self) -> u32 {
        (self.nq + 2 * self.nkv) * self.head_dim
    }

    /// `nq · head_dim` — the column stride of the batched attention output.
    #[inline]
    fn attn_dim(&self) -> u32 {
        self.nq * self.head_dim
    }

    /// The `u32` layer offset the MSL bindings expect (see
    /// [`Self::layer_offset`]).
    #[inline]
    fn layer_offset_u32(&self) -> u32 {
        self.layer_offset as u32
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipelines
// ═══════════════════════════════════════════════════════════════════════════

/// The two batched-prefill attention pipelines.
pub(crate) struct PrefillAttnPipelines {
    /// `prefill_qkv_prepare`: batch-wide QK RMSNorm + RoPE + KV-cache store.
    qkv_prepare: ComputePipelineState,
    /// `prefill_flash_attention`: batch-wide causal GQA flash attention.
    flash_attention: ComputePipelineState,
}

impl PrefillAttnPipelines {
    /// Compile both kernels into their own library, best effort.
    ///
    /// Returns `None` — never an error — if the library or either pipeline
    /// cannot be built (no Metal toolchain and a driver that rejects the
    /// source, a GPU without `simdgroup_matrix`, …). The caller then runs the
    /// historical per-token attention loop, so the worst case is the old
    /// performance, not a failed prefill.
    fn compile(device: &Device) -> Option<Self> {
        let library = load_library(device)?;
        let qkv_prepare = pipeline_for(&library, device, "prefill_qkv_prepare")?;
        let flash_attention = pipeline_for(&library, device, "prefill_flash_attention")?;
        Some(Self {
            qkv_prepare,
            flash_attention,
        })
    }
}

/// Whether the batched attention path can serve this layer geometry.
///
/// `head_dim` must be a multiple of the `8×8` matrix-unit edge and fit the
/// flash kernel's threadgroup-memory budget; `nkv` must divide `nq` (GQA). Any
/// other shape — notably Bonsai 2's `head_dim = 256` — takes the per-token
/// fallback.
#[must_use]
pub(crate) fn batched_attention_supported(nq: usize, nkv: usize, head_dim: usize) -> bool {
    nkv != 0
        && nq != 0
        && nq.is_multiple_of(nkv)
        && head_dim != 0
        && head_dim.is_multiple_of(8)
        && head_dim <= kernel_sources::PREFILL_FLASH_MAX_HEAD_DIM
}

/// MSL source of the standalone batched-prefill attention library.
fn library_source() -> String {
    let mut src = String::with_capacity(
        kernel_sources::MSL_PREFILL_QKV_PREPARE.len()
            + kernel_sources::MSL_PREFILL_FLASH_ATTENTION.len()
            + 2,
    );
    src.push_str(kernel_sources::MSL_PREFILL_QKV_PREPARE);
    src.push('\n');
    src.push_str(kernel_sources::MSL_PREFILL_FLASH_ATTENTION);
    src.push('\n');
    src
}

/// Extract one named pipeline, returning `None` on any failure.
fn pipeline_for(library: &Library, device: &Device, name: &str) -> Option<ComputePipelineState> {
    let func = match library.get_function(name, None) {
        Ok(f) => f,
        Err(e) => {
            tracing::info!("batched prefill attention: function '{name}' unavailable ({e})");
            return None;
        }
    };
    match device.new_compute_pipeline_state_with_function(&func) {
        Ok(pso) => Some(pso),
        Err(e) => {
            tracing::info!("batched prefill attention: pipeline '{name}' unavailable ({e})");
            None
        }
    }
}

/// Load the standalone library: disk cache → `xcrun` → runtime MSL compile.
///
/// Mirrors `metal_graph::pipelines::load_or_compile_library` (whose helpers are
/// private to that module) minus the embedded metallib, which only ever holds
/// the combined kernel set. The disk cache matters here: without it every
/// process start would pay a fresh MSL compile on the TTFT path.
fn load_library(device: &Device) -> Option<Library> {
    let src = library_source();
    let hash = source_hash(&src);
    let cache_name = format!("prefill_attn_{hash:016x}.metallib");

    if let Some(cache_dir) = cache_dir() {
        let cache_path = cache_dir.join(&cache_name);
        if let Ok(data) = std::fs::read(&cache_path) {
            if let Ok(lib) = device.new_library_with_data(&data) {
                return Some(lib);
            }
        }
        if let Some(lib) = compile_via_xcrun(device, &src, &cache_path) {
            return Some(lib);
        }
    }

    match device.new_library_with_source(&src, &metal::CompileOptions::new()) {
        Ok(lib) => Some(lib),
        Err(e) => {
            tracing::info!("batched prefill attention: MSL compilation failed ({e})");
            None
        }
    }
}

/// 64-bit hash of the MSL source, used as the disk-cache key.
fn source_hash(src: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut hasher);
    hasher.finish()
}

/// `~/.cache/oxibonsai/`, shared with the combined metallib cache.
fn cache_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".cache").join("oxibonsai"))
}

/// Compile MSL → `.metallib` via `xcrun`, cache it, and load it.
///
/// Intermediates go under `std::env::temp_dir()` in a per-process directory so
/// concurrent builds (the test suite runs several binaries at once) cannot
/// overwrite each other's `.air`.
fn compile_via_xcrun(device: &Device, src: &str, cache_path: &std::path::Path) -> Option<Library> {
    let tmp_dir =
        std::env::temp_dir().join(format!("oxibonsai_prefill_attn_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).ok()?;
    let metal_path = tmp_dir.join("prefill_attn.metal");
    let air_path = tmp_dir.join("prefill_attn.air");
    let lib_path = tmp_dir.join("prefill_attn.metallib");
    std::fs::write(&metal_path, src).ok()?;

    let metal_str = metal_path.to_str()?;
    let air_str = air_path.to_str()?;
    let lib_str = lib_path.to_str()?;

    let out = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c", metal_str, "-o", air_str])
        .output()
        .ok()?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        tracing::debug!(
            "batched prefill attention: xcrun metal failed: {}",
            &stderr[..stderr.len().min(500)]
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return None;
    }
    let out = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", air_str, "-o", lib_str])
        .output()
        .ok()?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return None;
    }

    let data = std::fs::read(&lib_path).ok();
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let data = data?;
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(cache_path, &data);
    device.new_library_with_data(&data).ok()
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph hooks
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// The batched-prefill attention pipelines, compiled on first use.
    ///
    /// `None` means "unavailable on this device/toolchain"; callers fall back
    /// to the per-token attention loop. The compile happens **outside** the
    /// prefill-buffer and KV-cache locks (see `encode_full_forward_prefill*`),
    /// so a first-run MSL compile never serialises other GPU work.
    pub(crate) fn prefill_attn_pipelines(&self) -> Option<&PrefillAttnPipelines> {
        self.prefill_attn
            .get_or_init(|| PrefillAttnPipelines::compile(&self.device))
            .as_ref()
    }

    /// Number of prefill buffer-set allocations since process start (perf-M3).
    ///
    /// Flat across a repeated sweep of distinct prompt lengths once the
    /// bucketed capacity has ramped: the acceptance evidence that the
    /// exact-match-on-`batch_size` reallocation is gone.
    #[must_use]
    pub fn prefill_buffer_alloc_count() -> u64 {
        PREFILL_BUFFER_ALLOC_COUNT.load(Ordering::Relaxed)
    }

    /// Number of layer attention phases encoded through the **batched**
    /// two-dispatch path since process start (perf-01).
    ///
    /// Zero after a prefill means every layer silently took the per-token
    /// fallback — numerically identical, and ~30x slower.
    #[must_use]
    pub fn prefill_batched_attention_count() -> u64 {
        PREFILL_BATCHED_ATTN_COUNT.load(Ordering::Relaxed)
    }

    /// Record that one layer's attention phase took the batched path.
    #[inline]
    pub(crate) fn note_batched_attention_layer() {
        PREFILL_BATCHED_ATTN_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    /// Encode the batch-wide QK-norm + RoPE + KV-store dispatch.
    ///
    /// Replaces `batch ×` (`fused_qk_norm` + `fused_qk_rope` +
    /// `fused_kv_store`). The Q section of `qkv` is rewritten **in place**
    /// with the normalised, rotated queries the flash kernel then consumes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_prefill_qkv_prepare(
        &self,
        pipelines: &PrefillAttnPipelines,
        encoder: &metal::ComputeCommandEncoderRef,
        qkv: &Buffer,
        q_norm: &Buffer,
        k_norm: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        dims: &PrefillAttnDims,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.qkv_prepare);
        encoder.set_buffer(0, Some(qkv), 0);
        encoder.set_buffer(1, Some(q_norm), 0);
        encoder.set_buffer(2, Some(k_norm), 0);
        encoder.set_buffer(3, Some(cos), 0);
        encoder.set_buffer(4, Some(sin), 0);
        encoder.set_buffer(5, Some(k_cache), 0);
        encoder.set_buffer(6, Some(v_cache), 0);
        let layer_offset = dims.layer_offset_u32();
        unsafe {
            set_scalar(encoder, 7, &dims.nq);
            set_scalar(encoder, 8, &dims.nkv);
            set_scalar(encoder, 9, &dims.head_dim);
            set_scalar(encoder, 10, &dims.eps);
            set_scalar(encoder, 11, &dims.max_seq);
            set_scalar(encoder, 12, &dims.pos_start);
            set_scalar(encoder, 13, &dims.batch_size);
            set_scalar(encoder, 14, &layer_offset);
        }
        let slots = u64::from(dims.nq + 2 * dims.nkv);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims.batch_size), slots, 1),
            MTLSize::new(kernel_sources::PREFILL_QKV_PREPARE_THREADS as u64, 1, 1),
        );
    }

    /// Encode the batch-wide causal GQA flash-attention dispatch.
    ///
    /// Replaces `batch ×` (`batched_attention_scores_v2` + `batched_softmax` +
    /// `batched_attention_weighted_sum`) and never materialises the
    /// `batch × n_ctx` score matrix.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_prefill_flash_attention(
        &self,
        pipelines: &PrefillAttnPipelines,
        encoder: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        out: &Buffer,
        dims: &PrefillAttnDims,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.flash_attention);
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(k_cache), 0);
        encoder.set_buffer(2, Some(v_cache), 0);
        encoder.set_buffer(3, Some(out), 0);
        let q_row_stride = dims.qkv_dim();
        let out_row_stride = dims.attn_dim();
        let layer_offset = dims.layer_offset_u32();
        unsafe {
            set_scalar(encoder, 4, &dims.nq);
            set_scalar(encoder, 5, &dims.nkv);
            set_scalar(encoder, 6, &dims.heads_per_group);
            set_scalar(encoder, 7, &dims.head_dim);
            set_scalar(encoder, 8, &q_row_stride);
            set_scalar(encoder, 9, &out_row_stride);
            set_scalar(encoder, 10, &dims.max_seq);
            set_scalar(encoder, 11, &dims.pos_start);
            set_scalar(encoder, 12, &dims.batch_size);
            set_scalar(encoder, 13, &dims.scale);
            set_scalar(encoder, 14, &layer_offset);
        }
        let q_tiles = (dims.batch_size as usize).div_ceil(kernel_sources::PREFILL_FLASH_BQ) as u64;
        encoder.dispatch_thread_groups(
            MTLSize::new(q_tiles, u64::from(dims.nq), 1),
            MTLSize::new(kernel_sources::PREFILL_FLASH_THREADS as u64, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_buckets_round_up_and_are_monotone() {
        assert_eq!(prefill_batch_capacity(0), PREFILL_BATCH_BUCKET);
        assert_eq!(prefill_batch_capacity(1), PREFILL_BATCH_BUCKET);
        assert_eq!(prefill_batch_capacity(128), 128);
        assert_eq!(prefill_batch_capacity(129), 256);
        assert_eq!(prefill_batch_capacity(1501), 1536);
        let mut prev = 0;
        for n in [1usize, 7, 63, 64, 127, 128, 200, 512, 1000, 1501, 4096] {
            let cap = prefill_batch_capacity(n);
            assert!(cap >= n, "capacity {cap} must cover batch {n}");
            assert!(cap >= prev, "capacity must be monotone in batch size");
            assert!(cap.is_multiple_of(PREFILL_BATCH_BUCKET));
            prev = cap;
        }
    }

    #[test]
    fn supported_geometries_are_the_ones_the_kernel_can_serve() {
        // Ternary-Bonsai-1.7B / 8B decode geometry.
        assert!(batched_attention_supported(16, 8, 128));
        // The synthetic parity fixture (head_dim 32).
        assert!(batched_attention_supported(4, 2, 32));
        // Bonsai 2 full-attention layers: head_dim 256 is above the cap.
        assert!(!batched_attention_supported(24, 4, 256));
        // Non-multiple-of-8 head_dim cannot feed the 8x8 matrix units.
        assert!(!batched_attention_supported(4, 2, 36));
        // nkv must divide nq for the GQA index to be exact.
        assert!(!batched_attention_supported(6, 4, 64));
        assert!(!batched_attention_supported(4, 0, 64));
    }

    /// perf-M3: a sweep of distinct prompt lengths must not reallocate the
    /// ~172 MB buffer set per length. Device-backed but dispatch-free.
    #[test]
    fn bucketed_cache_stops_per_prompt_length_reallocation() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = match MetalGraph::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        // Small model dimensions: this test is about the cache, not the GPU.
        let (h, inter, nq, nkv, hd, max_seq) = (128usize, 256, 4, 2, 32, 4096);
        // 20 distinct prompt lengths, deliberately not in bucket order.
        let lengths: Vec<usize> = (0..20).map(|i| 37 + i * 91).collect();

        let before_first = MetalGraph::prefill_buffer_alloc_count();
        for &n in &lengths {
            let guard = graph
                .acquire_prefill_buffers(n, h, inter, nq, nkv, hd, max_seq)
                .expect("acquire_prefill_buffers");
            let cache = guard.as_ref().expect("buffers allocated");
            assert!(cache.capacity() >= n);
        }
        let first_pass = MetalGraph::prefill_buffer_alloc_count() - before_first;
        assert!(
            first_pass < lengths.len() as u64,
            "first pass allocated {first_pass} times for {} distinct lengths — \
             the cache is still (near) exact-match",
            lengths.len()
        );

        // Second pass over the very same lengths: the capacity has ramped to
        // the high-water bucket, so nothing may reallocate.
        let before_second = MetalGraph::prefill_buffer_alloc_count();
        for &n in &lengths {
            let _guard = graph
                .acquire_prefill_buffers(n, h, inter, nq, nkv, hd, max_seq)
                .expect("acquire_prefill_buffers");
        }
        assert_eq!(
            MetalGraph::prefill_buffer_alloc_count() - before_second,
            0,
            "a repeated sweep of 20 distinct prompt lengths must allocate nothing"
        );
    }

    /// perf-01 structural guard: the batched attention phase must stay wired
    /// into **both** prefill layer encoders, and the `v10` GEMM selector into
    /// the ternary one.
    ///
    /// The batched path and the per-token fallback are numerically equivalent,
    /// so an edit that dropped back to the loop would leave every parity test
    /// green while quietly restoring the ~13x slower prefill this package
    /// removed. [`MetalGraph::prefill_batched_attention_count`] catches that at
    /// runtime; this pins it structurally, the way
    /// `graph_rs_has_no_unchecked_command_buffer_commit` pins MET-04.
    #[test]
    fn prefill_layer_encoders_still_dispatch_the_batched_attention_phase() {
        let src = include_str!("functions.rs");
        assert_eq!(
            src.matches("match attn {").count(),
            2,
            "both encode_layer_prefill and encode_layer_prefill_ternary must \
             choose between the batched phase and the per-token fallback"
        );
        assert!(
            src.contains("self.encode_attention_batched("),
            "the batched two-dispatch attention phase is no longer dispatched (perf-01)"
        );
        assert!(
            src.contains("fn encode_attention_per_token("),
            "the per-token fallback must stay: it serves head_dim > 128 (Bonsai 2) \
             and any device where the batched kernels fail to compile"
        );
        assert_eq!(
            src.matches("self.dispatch_gemm_tq2_prefill(").count(),
            4,
            "all four ternary prefill GEMMs must go through the v10/v7 selector (perf-02)"
        );
    }

    /// The pipelines must actually build on this machine — otherwise the whole
    /// package silently degrades to the per-token path it exists to replace.
    #[test]
    fn prefill_attention_pipelines_compile_on_this_device() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = match MetalGraph::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        assert!(
            graph.prefill_attn_pipelines().is_some(),
            "prefill_qkv_prepare / prefill_flash_attention failed to compile; \
             batched prefill would fall back to the per-token loop"
        );
    }
}
