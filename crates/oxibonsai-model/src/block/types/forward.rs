//! Single-token forward pass (`TransformerBlock::forward`).

use crate::error::ModelResult;
use crate::kv_cache::KvCache;
use crate::layers::rope::RopeTable;
use crate::layers::swiglu::try_swiglu;
use oxibonsai_kernels::traits::OneBitKernel;
use std::time::Instant;

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::block::functions::blocks_as_bytes;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )
))]
use crate::block::functions::blocks_as_bytes_ternary;

use crate::block::functions::compute_gqa_attention;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::block::functions::try_metal_gemv_ternary_fused;
use crate::block::functions::{advance_kv_cache_to, validate_shapes};

use super::block_def::TransformerBlock;
use super::scratch::ScratchBuffers;

// Weight-cache epoch of this block's CUDA uploads (finding F-M3):
// `model_epoch: u64` is the last parameter of
// `oxibonsai_kernels::try_cuda_qkv` / `try_cuda_ffn` so a model's `Drop` can
// free exactly its own GPU weights, but the block carried no epoch and
// registered its uploads **unattributed** (never released). The block now
// carries its model's namespace (`block_def::TransformerBlock::slot_namespace`,
// set by `BonsaiModel`'s constructors to the model's `cuda_model_epoch` on a
// CUDA build), so the CUDA arms below attribute their uploads to
// `self.slot_namespace.epoch()` and the model's drop frees them. A standalone
// block's namespace is a fresh epoch drawn from the same CUDA counter, so it
// can never name another model's buffers.

/// Private CUDA weight-cache namespace for the **fused** Q‖K‖V ternary
/// upload (finding **M-21**, blocking fix).
///
/// `CudaGraph::weight_cache` is a single `HashMap<u64, Arc<CudaSlice<u8>>>`
/// shared by every TQ2 producer. The Q projection’s own per-matrix upload
/// already owns `attn_q.gpu_handle().id()` in that map —
/// `LinearTernary::upload_to_gpu` (`layers/linear.rs`) →
/// `NativeCudaBackend::upload_weights_ternary`
/// (`cuda_graph/nativecudabackend_traits.rs`) →
/// `upload_weight_tq2_soa_for_epoch`
/// (`cuda_graph/cudagraph_reformat_tq2_blocks_to_soa_group.rs`) inserts the
/// **Q-only** SoA buffer under exactly that id. Setting bit 63 moves the fused
/// concatenation into a namespace no other producer writes.
///
/// The tag cannot alias a real handle id: `cuda_graph::functions`’
/// `alloc_handle_id` hands out `NEXT_HANDLE_ID: AtomicU64::new(1)` values via
/// `fetch_add(1, Relaxed)`, so reaching `1 << 63` would need 2^63 uploads —
/// unreachable in any process lifetime. Untagged ids therefore always have
/// bit 63 clear, and tagged ones always have it set.
#[cfg(any(
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ),
    test
))]
const CUDA_FUSED_QKV_SLOT_TAG: u64 = 1u64 << 63;

/// Map a `GpuWeightHandle` id to its fused-QKV slot (see
/// [`CUDA_FUSED_QKV_SLOT_TAG`]).
#[cfg(any(
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ),
    test
))]
fn cuda_fused_qkv_slot(handle_id: u64) -> u64 {
    handle_id | CUDA_FUSED_QKV_SLOT_TAG
}

/// CUDA weight-cache slot for a ternary fused GEMV (`M-21`).
///
/// A **dedicated** fused handle (`fused_*_handle_ternary`, minted by
/// `upload_to_gpu` for exactly this concatenation) is used as-is: no other
/// producer writes that id, and with `NativeCudaBackend` the upload already
/// put the concatenation's SoA there, so tagging it would only upload a
/// second copy. Without one, the slot is derived from a single projection's
/// handle (`fallback`), whose raw id that projection's own per-matrix upload
/// already owns — so it is moved into the private tagged namespace
/// ([`cuda_fused_qkv_slot`]). `None` when the block was never uploaded.
#[cfg(any(
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ),
    test
))]
fn cuda_ternary_fused_slot(dedicated_id: Option<u64>, fallback_id: Option<u64>) -> Option<u64> {
    match (dedicated_id, fallback_id) {
        (Some(fused), _) => Some(fused),
        (None, Some(single)) => Some(cuda_fused_qkv_slot(single)),
        (None, None) => None,
    }
}

/// Byte length a `TQ2_0_g128` SoA weight buffer must have for `total_rows`
/// output rows and `k` input columns, or `None` when `k` is not a whole number
/// of 128-wide quant groups.
///
/// The SoA layout the CUDA TQ2 kernels consume is
/// `[N × 2 B FP16 scales][N × 32 B qs]` with `N = total_rows * (k / 128)`,
/// i.e. 34 bytes per block — the same 34-byte stride
/// `reformat_tq2_aos_bytes_to_soa` requires of its input.
#[cfg(any(
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ),
    test
))]
fn cuda_tq2_soa_len_bytes(total_rows: usize, k: usize) -> Option<usize> {
    const BLOCK_BYTES: usize = 34;
    const GROUP: usize = 128;
    if k == 0 || !k.is_multiple_of(GROUP) {
        return None;
    }
    total_rows
        .checked_mul(k / GROUP)
        .and_then(|blocks| blocks.checked_mul(BLOCK_BYTES))
}

/// Fused ternary (TQ2_0_g128) QKV GEMV on CUDA — the twin of
/// `block::functions::try_metal_gemv_ternary_fused` (**M-21**).
///
/// Ternary models get no `fused_qkv_handle` (see `upload.rs`), so the 1-bit
/// fused path above cannot serve them: `OneBitKernel::gemv_cached` always
/// decodes its buffer as `Q1_0_g128` and would read a 34-byte-block tensor as
/// 18-byte blocks. This uploads Q‖K‖V once and runs one GEMV over the
/// concatenation, exactly as the Metal path does.
///
/// # Why the slot is tagged, and why the earlier argument here was wrong
///
/// A previous revision of this comment justified the slot as
/// "`attn_q.gpu_handle().id()`, not the weight’s mmap address: ids come from
/// a process-global monotonic counter, so they are never reused across model
/// loads". That is true and it is **not** the property this call site needs.
/// The id is unique, but it is not *private*: `weight_cache` is one flat map
/// and the Q projection’s own per-matrix upload already occupies that id with
/// a **Q-only** buffer. `get_or_upload_weight_tq2_soa_lazy_for_epoch` returns
/// early on a hit, so the fused bytes were never built, and
/// `encode_gemv_tq2_cached` then launched `gemv_tq2_g128_v1` with
/// `n_rows = q_rows + 2 * k_rows` over a `q_rows`-sized allocation: the
/// in-kernel `qs_offset = total_blocks * 2` is recomputed from the larger row
/// count, so even the Q rows decode against the wrong `qs` bytes and K/V read
/// past the end of the allocation. The Metal twin is safe only because
/// `MetalGraph` keeps a *different* map (stated in `upload.rs`), so the port
/// was not faithful.
///
/// Both halves of the fix are required and both are here:
/// 1. The caller derives `fused_slot` with [`cuda_ternary_fused_slot`]: a
///    dedicated fused handle's own id (`M-21`, a namespace no other producer
///    writes), or a single projection's id moved into the private tagged
///    namespace by [`cuda_fused_qkv_slot`]. The **same** value keys the
///    upload and the launch below.
/// 2. The cached buffer’s byte length is checked against
///    [`cuda_tq2_soa_len_bytes`] before the launch, so any future second
///    consumer of the map is caught as an `Err` and the caller’s existing CPU
///    fallback engages instead of a `CUDA_ERROR_ILLEGAL_ADDRESS`.
///
/// Generic over the number of parts (`M-21`): Q‖K‖V passes three, gate‖up
/// two.
///
/// The upload registers under `model_epoch` — the block's namespace epoch,
/// i.e. its model's `cuda_model_epoch` — so the model's drop releases it.
///
/// Not yet run on CUDA hardware by a dedicated test: no hardware test
/// targeted this per-block wrapper. The TQ2 GEMV it launches passed CUDA-P18 on the real fused
/// Q‖K‖V and gate‖up shapes (RTX A4000, CUDA 12.0, 2026-10-07).
#[cfg(all(
    feature = "native-cuda",
    not(all(feature = "metal", target_os = "macos")),
    any(target_os = "linux", target_os = "windows")
))]
fn try_cuda_gemv_ternary_fused(
    input: &[f32],
    output: &mut [f32],
    fused_slot: u64,
    aos_parts: &[&[u8]],
    n_rows: usize,
    k: usize,
    model_epoch: u64,
) -> Result<(), oxibonsai_kernels::CudaGraphError> {
    let graph = oxibonsai_kernels::CudaGraph::global()?;
    let expected_bytes = cuda_tq2_soa_len_bytes(n_rows, k).ok_or_else(|| {
        oxibonsai_kernels::CudaGraphError::DriverError(format!(
            "fused ternary QKV GEMV needs k to be a multiple of 128, got k={k}"
        ))
    })?;
    let d_weight = graph.get_or_upload_weight_tq2_soa_lazy_for_epoch(
        fused_slot,
        || {
            let total: usize = aos_parts.iter().map(|part| part.len()).sum();
            let mut fused = Vec::with_capacity(total);
            for part in aos_parts {
                fused.extend_from_slice(part);
            }
            fused
        },
        model_epoch,
    )?;
    if d_weight.len() != expected_bytes {
        return Err(oxibonsai_kernels::CudaGraphError::DriverError(format!(
            "fused ternary QKV slot {fused_slot:#018x} holds {} bytes, expected \
             {expected_bytes} for {n_rows} rows x k={k} (TQ2_0_g128 SoA, 34 B per \
             block); refusing to launch over a buffer this call site does not own",
            d_weight.len()
        )));
    }
    let rows = graph.encode_gemv_tq2_cached(fused_slot, input, n_rows, k)?;
    if rows.len() < n_rows || output.len() < n_rows {
        return Err(oxibonsai_kernels::CudaGraphError::DriverError(format!(
            "fused ternary QKV GEMV produced {} rows into a {}-element output, expected {n_rows}",
            rows.len(),
            output.len()
        )));
    }
    output[..n_rows].copy_from_slice(&rows[..n_rows]);
    Ok(())
}

impl<'a> TransformerBlock<'a> {
    /// Forward pass for a single token at position `pos`.
    ///
    /// - `hidden`: Input/output hidden state `[hidden_size]`. Modified in-place.
    /// - `pos`: Current token position in the sequence.
    /// - `kv_cache`: KV cache to store/retrieve K and V vectors.
    /// - `rope`: Precomputed RoPE table.
    /// - `kernel`: 1-bit kernel dispatcher.
    #[allow(clippy::needless_late_init)]
    #[tracing::instrument(skip_all, fields(layer = self.layer_idx))]
    pub fn forward(
        &self,
        hidden: &mut [f32],
        pos: usize,
        kv_cache: &mut KvCache,
        rope: &RopeTable,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<()> {
        // M-12: validate before any KV-cache access (including the GPU
        // full-layer fast paths below, which compute the same
        // `heads_per_group = nq / nkv` and index the cache with the same
        // unchecked stride internally).
        validate_shapes(
            self.layer_idx,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.attn_q.out_features(),
            kv_cache,
            pos,
        )?;
        // M-19: maintain the cache's own sequence-length cursor so
        // `seq_len()`/`utilization_ratio()` reflect reality; see
        // `advance_kv_cache_to`'s doc comment for why this is safe to call
        // once per layer per token rather than once per token.
        advance_kv_cache_to(kv_cache, pos);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if let Some(Ok(())) = self.try_full_layer_gpu(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            if let Some(Ok(())) = self.try_full_layer_cuda(hidden, pos, rope, kv_cache) {
                return Ok(());
            }
        }
        let h = self.hidden_size;
        let hd = self.head_dim;
        let nq = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nq / nkv;
        let total_start = Instant::now();
        let mut scratch = self.scratch.lock().map_err(|e| {
            crate::error::ModelError::Internal(format!("scratch lock poisoned: {e}"))
        })?;
        scratch.clear();
        let ScratchBuffers {
            normed,
            q_all,
            k_all,
            v_all,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attn_out,
            attn_proj,
            gate_out,
            up_out,
            swiglu_out,
            down_out,
            fused_qkv,
            fused_gate_up,
        } = &mut *scratch;
        let norm_us: u128;
        let qkv_us: u128;
        let qknorm_us: u128;
        let rope_us: u128;
        let cache_us: u128;
        let attn_us: u128;
        let ffn_us: u128;
        {
            let norm_start = Instant::now();
            self.attn_norm.forward(hidden, normed)?;
            norm_us = norm_start.elapsed().as_micros();
            let qkv_start = Instant::now();
            if let Some(fused_handle) = self.fused_qkv_handle {
                let q_rows = nq * hd;
                let k_rows = nkv * hd;
                let total_rows = q_rows + k_rows + k_rows;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let metal_ok = {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_metal_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                        )
                        .is_ok()
                    } else {
                        false
                    }
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let metal_ok = false;
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                // F-M2: a CPU-tier run must never
                // reach `try_cuda_qkv` at all. `CudaGraph::global()` opens the
                // device and compiles six NVRTC modules on first call; without
                // this gate a `KernelTier::Reference` run on a CUDA box paid
                // that attempt once per layer per token. Mirrors the
                // `_gpu_kernel` gate at `model/types/mod.rs:891`.
                let cuda_ok = if !metal_ok && kernel.is_gpu_accelerated() {
                    if let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
                        self.attn_q.blocks_1bit(),
                        self.attn_k.blocks_1bit(),
                        self.attn_v.blocks_1bit(),
                    ) {
                        let q_bytes = blocks_as_bytes(q_blk);
                        let k_bytes = blocks_as_bytes(k_blk);
                        let v_bytes = blocks_as_bytes(v_blk);
                        oxibonsai_kernels::try_cuda_qkv(
                            normed,
                            fused_qkv,
                            fused_handle.id(),
                            q_bytes,
                            k_bytes,
                            v_bytes,
                            total_rows,
                            h,
                            self.slot_namespace.epoch(),
                        )
                        .is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                };

                #[cfg(not(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                )))]
                let cuda_ok = false;
                if !metal_ok && !cuda_ok {
                    // `fused_qkv_handle` (see `upload.rs`) is built ONLY for
                    // 1-bit weights today, but guard explicitly rather than
                    // relying on that invariant: `OneBitKernel::gemv_cached`
                    // always decodes its buffer as `Q1_0_g128`, so it must
                    // never be called with a handle built over a different
                    // format (M-21 hardening — ternary/PQ2_0/PTQ1_0 would
                    // silently read garbage otherwise).
                    if self.attn_q.blocks_1bit().is_some() {
                        kernel.gemv_cached(fused_handle, normed, fused_qkv, total_rows, h)?;
                    } else {
                        self.attn_q.forward_vec(normed, q_all)?;
                        self.attn_k.forward_vec(normed, k_all)?;
                        self.attn_v.forward_vec(normed, v_all)?;
                    }
                }
                if metal_ok || cuda_ok || self.attn_q.blocks_1bit().is_some() {
                    q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                    k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                    v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
                }
            } else {
                // M-21: ternary models get no 1-bit `fused_qkv_handle` (see
                // the long comment in `upload.rs` for why), so the Metal
                // fused-QKV fast path is keyed on `blocks_ternary()` directly,
                // gated on GPU residency (`upload_to_gpu()` ran on a GPU
                // tier) so a CPU-tier block stays on the CPU path exactly like
                // the 1-bit branch above does. Its slot is the mapped
                // `attn_q` address under the block's namespace epoch
                // (`ternary_fused_qkv_slot`, MET-02): the model's own
                // full-forward cache keys its fused Q‖K‖V buffer on the very
                // same (epoch, slot), every replica of the GGUF mapping shares
                // it, and a model re-loaded at a reused address mints a new
                // epoch, so a freed mapping's buffer can never be bound.
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let ternary_metal_ok =
                    self.try_fused_qkv_ternary_metal(normed, fused_qkv, q_all, k_all, v_all);
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let ternary_metal_ok = false;
                // M-21: the CUDA twin of the Metal
                // branch above. Same gating — real ternary blocks on all three
                // projections, a GPU handle that seeds the upload slot, and a
                // GPU kernel tier (F-M2) so a CPU-tier run never opens the
                // device — and the same fall-through to the CPU projections
                // when any of that is missing or the GEMV fails.
                //
                // The slot comes from `cuda_ternary_fused_slot`: the dedicated
                // `fused_qkv_handle_ternary` id when `upload_to_gpu` built one
                // (its buffer already IS the concatenation on
                // `NativeCudaBackend`), else `attn_q`'s id moved into the
                // private tagged namespace, because that raw id is owned by the
                // Q projection's own per-matrix upload (M-21 blocking fix).
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                let ternary_cuda_ok = {
                    let blocks = if kernel.is_gpu_accelerated() {
                        (
                            self.attn_q.blocks_ternary(),
                            self.attn_k.blocks_ternary(),
                            self.attn_v.blocks_ternary(),
                        )
                    } else {
                        (None, None, None)
                    };
                    if let ((Some(q_blk), Some(k_blk), Some(v_blk)), Some(fused_slot)) = (
                        blocks,
                        cuda_ternary_fused_slot(
                            self.fused_qkv_handle_ternary.map(|hnd| hnd.id()),
                            self.attn_q.gpu_handle().map(|hnd| hnd.id()),
                        ),
                    ) {
                        let q_rows = nq * hd;
                        let k_rows = nkv * hd;
                        let total_rows = q_rows + k_rows + k_rows;
                        let q_bytes = blocks_as_bytes_ternary(q_blk);
                        let k_bytes = blocks_as_bytes_ternary(k_blk);
                        let v_bytes = blocks_as_bytes_ternary(v_blk);
                        match try_cuda_gemv_ternary_fused(
                            normed,
                            fused_qkv,
                            fused_slot,
                            &[q_bytes, k_bytes, v_bytes],
                            total_rows,
                            h,
                            self.slot_namespace.epoch(),
                        ) {
                            Ok(()) => {
                                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                                k_all[..k_rows]
                                    .copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                                v_all[..k_rows]
                                    .copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
                                true
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "fused ternary QKV GEMV on CUDA failed, falling back to CPU"
                                );
                                false
                            }
                        }
                    } else {
                        false
                    }
                };
                #[cfg(not(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                )))]
                let ternary_cuda_ok = false;
                if !ternary_metal_ok && !ternary_cuda_ok {
                    self.attn_q.forward_vec(normed, q_all)?;
                    self.attn_k.forward_vec(normed, k_all)?;
                    self.attn_v.forward_vec(normed, v_all)?;
                }
            }
            qkv_us = qkv_start.elapsed().as_micros();
        }
        let qknorm_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            self.attn_q_norm
                .forward(&q_all[start..start + hd], &mut q_normed[start..start + hd])?;
        }
        for head in 0..nkv {
            let start = head * hd;
            self.attn_k_norm
                .forward(&k_all[start..start + hd], &mut k_normed[start..start + hd])?;
        }
        qknorm_us = qknorm_start.elapsed().as_micros();
        let rope_start = Instant::now();
        for head in 0..nq {
            let start = head * hd;
            rope.apply(
                &q_normed[start..start + hd],
                &mut q_rope[start..start + hd],
                pos,
            )?;
        }
        for head in 0..nkv {
            let start = head * hd;
            rope.apply(
                &k_normed[start..start + hd],
                &mut k_rope[start..start + hd],
                pos,
            )?;
        }
        rope_us = rope_start.elapsed().as_micros();
        let cache_start = Instant::now();
        for head in 0..nkv {
            let start = head * hd;
            // Sparse-KV: the fallible stores, so an
            // out-of-range layer/head/position or a wrong-width key surfaces
            // as an error here instead of being silently dropped and read
            // back as zeros by the attention below.
            kv_cache.try_store_key(self.layer_idx, head, pos, &k_rope[start..start + hd])?;
            kv_cache.try_store_value(self.layer_idx, head, pos, &v_all[start..start + hd])?;
        }
        cache_us = cache_start.elapsed().as_micros();
        let seq_len = pos + 1;
        let attn_start = Instant::now();
        compute_gqa_attention(
            q_rope,
            attn_out,
            kv_cache,
            self.layer_idx,
            nq,
            heads_per_group,
            hd,
            seq_len,
        )?;
        attn_us = attn_start.elapsed().as_micros();
        let ffn_start = Instant::now();
        let did_batch_ffn = if let (
            Some(attn_proj_handle),
            Some(gate_up_handle),
            Some(down_handle),
        ) = (
            self.attn_output.gpu_handle(),
            self.fused_gate_up_handle,
            self.ffn_down.gpu_handle(),
        ) {
            // M-21 hardening: `batch_ffn_phase`/`try_metal_ffn` always
            // decode every buffer as `Q1_0_g128` regardless of the
            // handle's true origin (`GpuWeightHandle` carries no format
            // tag). `fused_gate_up_handle` is built only for 1-bit
            // weights today (see `upload.rs`), but this whole-layer
            // batched path must not rely on that invariant alone —
            // require every matrix to genuinely be 1-bit before
            // entering it, so a ternary (or future PQ2_0/PTQ1_0) model
            // can never reach the wrong decoder here.
            if self.attn_output.blocks_1bit().is_some()
                && self.ffn_gate.blocks_1bit().is_some()
                && self.ffn_up.blocks_1bit().is_some()
                && self.ffn_down.blocks_1bit().is_some()
            {
                let inter = self.ffn_gate.out_features();
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) = (
                        self.attn_output.blocks_1bit(),
                        self.ffn_gate.blocks_1bit(),
                        self.ffn_up.blocks_1bit(),
                        self.ffn_down.blocks_1bit(),
                    ) {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let metal_result = oxibonsai_kernels::try_metal_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                        );
                        if metal_result.is_ok() {
                            true
                        } else {
                            tracing::warn!(
                                error = ? metal_result.err(),
                                "MetalGraph FFN failed, falling back"
                            );
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(all(
                    feature = "native-cuda",
                    not(all(feature = "metal", target_os = "macos")),
                    any(target_os = "linux", target_os = "windows")
                ))]
                {
                    // F-M2: see the `try_cuda_qkv`
                    // gate above — a CPU-tier run must not open the device.
                    if let (Some(attn_proj_blk), Some(gate_blk), Some(up_blk), Some(down_blk)) =
                        if kernel.is_gpu_accelerated() {
                            (
                                self.attn_output.blocks_1bit(),
                                self.ffn_gate.blocks_1bit(),
                                self.ffn_up.blocks_1bit(),
                                self.ffn_down.blocks_1bit(),
                            )
                        } else {
                            (None, None, None, None)
                        }
                    {
                        let attn_proj_bytes = blocks_as_bytes(attn_proj_blk);
                        let gate_bytes = blocks_as_bytes(gate_blk);
                        let up_bytes = blocks_as_bytes(up_blk);
                        let down_bytes = blocks_as_bytes(down_blk);
                        let cuda_result = oxibonsai_kernels::try_cuda_ffn(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle.id(),
                            attn_proj_bytes,
                            gate_up_handle.id(),
                            gate_bytes,
                            up_bytes,
                            down_handle.id(),
                            down_bytes,
                            h,
                            inter,
                            self.slot_namespace.epoch(),
                        );
                        if cuda_result.is_ok() {
                            true
                        } else {
                            tracing::warn!(
                                error = ? cuda_result.err(),
                                "CudaGraph FFN failed, falling back"
                            );
                            kernel.batch_ffn_phase(
                                hidden,
                                attn_out,
                                self.ffn_norm.weight(),
                                self.ffn_norm.eps(),
                                attn_proj_handle,
                                gate_up_handle,
                                down_handle,
                                h,
                                inter,
                                nq * hd,
                            )?
                        }
                    } else {
                        kernel.batch_ffn_phase(
                            hidden,
                            attn_out,
                            self.ffn_norm.weight(),
                            self.ffn_norm.eps(),
                            attn_proj_handle,
                            gate_up_handle,
                            down_handle,
                            h,
                            inter,
                            nq * hd,
                        )?
                    }
                }
                #[cfg(not(any(
                    all(feature = "metal", target_os = "macos"),
                    all(
                        feature = "native-cuda",
                        any(target_os = "linux", target_os = "windows")
                    )
                )))]
                {
                    kernel.batch_ffn_phase(
                        hidden,
                        attn_out,
                        self.ffn_norm.weight(),
                        self.ffn_norm.eps(),
                        attn_proj_handle,
                        gate_up_handle,
                        down_handle,
                        h,
                        inter,
                        nq * hd,
                    )?
                }
            } else {
                false
            }
        } else {
            false
        };
        if !did_batch_ffn {
            self.attn_output.forward_vec(attn_out, attn_proj)?;
            for i in 0..h {
                hidden[i] += attn_proj[i];
            }
            self.ffn_norm.forward(hidden, normed)?;
            if let Some(fused_handle) = self.fused_gate_up_handle {
                // M-21 hardening: mirrors the QKV-side guard above —
                // `gemv_cached` always decodes `Q1_0_g128`, so it must never
                // see a handle built over a different format.
                if self.ffn_gate.blocks_1bit().is_some() {
                    let inter = gate_out.len();
                    let total_rows = inter * 2;
                    kernel.gemv_cached(fused_handle, normed, fused_gate_up, total_rows, h)?;
                    gate_out[..inter].copy_from_slice(&fused_gate_up[..inter]);
                    up_out[..inter].copy_from_slice(&fused_gate_up[inter..total_rows]);
                } else {
                    self.ffn_gate.forward_vec(normed, gate_out)?;
                    self.ffn_up.forward_vec(normed, up_out)?;
                }
            } else if !self.try_fused_gate_up_ternary(
                normed,
                fused_gate_up,
                gate_out,
                up_out,
                kernel,
            ) {
                // Not a GPU-uploaded ternary block (or the fused GEMV
                // failed): the two projections, per matrix.
                self.ffn_gate.forward_vec(normed, gate_out)?;
                self.ffn_up.forward_vec(normed, up_out)?;
            }
            try_swiglu(gate_out, up_out, swiglu_out)?;
            self.ffn_down.forward_vec(swiglu_out, down_out)?;
            for i in 0..h {
                hidden[i] += down_out[i];
            }
        }
        ffn_us = ffn_start.elapsed().as_micros();
        let total_us = total_start.elapsed().as_micros();
        tracing::debug!(
            target : "block_profile",
            "L{layer}: norm={norm_us}µs qkv={qkv_us}µs qknorm={qknorm_us}µs rope={rope_us}µs cache={cache_us}µs attn={attn_us}µs ffn={ffn_us}µs total={total_us}µs",
            layer = self.layer_idx,
        );
        Ok(())
    }

    /// `M-21` / `MET-02`: one fused GEMV for a GPU-uploaded **ternary**
    /// block's Q‖K‖V projection on Metal, writing `q_all` / `k_all` /
    /// `v_all` — the one ternary fused-QKV arm every block forward kind
    /// (`forward`, `forward_with_sliding_window`, `forward_with_stats`) runs,
    /// so they all bind the same `MetalGraph` buffer.
    ///
    /// Keyed on [`Self::ternary_fused_qkv_slot`] (the mapped `attn_q`
    /// address) under the block's namespace epoch — the model's mapping
    /// epoch, i.e. the exact key the model's full-forward ternary cache holds
    /// its Q‖K‖V concatenation under, which is byte-identical to the one built
    /// here — through the size-checked N-part kernels entry, which refuses a
    /// resident buffer of the wrong length. Returns `false` (and the caller
    /// runs the three per-matrix projections) for a non-ternary block, a
    /// block never uploaded to a GPU tier, or any Metal error.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn try_fused_qkv_ternary_metal(
        &self,
        normed: &[f32],
        fused_qkv: &mut [f32],
        q_all: &mut [f32],
        k_all: &mut [f32],
        v_all: &mut [f32],
    ) -> bool {
        let (Some(q_blk), Some(k_blk), Some(v_blk)) = (
            self.attn_q.blocks_ternary(),
            self.attn_k.blocks_ternary(),
            self.attn_v.blocks_ternary(),
        ) else {
            return false;
        };
        let Some(slot) = self.ternary_fused_qkv_slot() else {
            return false;
        };
        let q_rows = self.num_heads * self.head_dim;
        let k_rows = self.num_kv_heads * self.head_dim;
        let total_rows = q_rows + k_rows + k_rows;
        if fused_qkv.len() < total_rows
            || q_all.len() < q_rows
            || k_all.len() < k_rows
            || v_all.len() < k_rows
        {
            return false;
        }
        let parts = [
            blocks_as_bytes_ternary(q_blk),
            blocks_as_bytes_ternary(k_blk),
            blocks_as_bytes_ternary(v_blk),
        ];
        // Before and after the upload-or-bind: see `MappingState::mark_gpu_used`.
        self.slot_namespace.mark_gpu_used();
        let result = try_metal_gemv_ternary_fused(
            normed,
            fused_qkv,
            self.slot_namespace.epoch(),
            slot,
            &parts,
            total_rows,
            self.hidden_size,
        );
        self.slot_namespace.mark_gpu_used();
        match result {
            Ok(()) => {
                q_all[..q_rows].copy_from_slice(&fused_qkv[..q_rows]);
                k_all[..k_rows].copy_from_slice(&fused_qkv[q_rows..q_rows + k_rows]);
                v_all[..k_rows].copy_from_slice(&fused_qkv[q_rows + k_rows..total_rows]);
                true
            }
            Err(e) => {
                tracing::debug!(
                    layer = self.layer_idx, error = %e,
                    "fused ternary QKV GEMV on Metal failed, using per-matrix projections"
                );
                false
            }
        }
    }

    /// `M-21`: one fused GEMV for a **ternary** block's gate‖up projection
    /// instead of two, writing `gate_out` / `up_out` — shared by every block
    /// forward kind.
    ///
    /// Engages only for a block whose gate and up projections are both
    /// `TQ2_0_g128` and that `upload_to_gpu` gave a GPU slot
    /// ([`Self::ternary_fused_gate_up_slot`] under the block's namespace
    /// epoch on Metal — the model cache's own `gate_up` key —
    /// `cuda_ternary_fused_slot` on CUDA, behind a GPU kernel tier there).
    /// Returns `false` — and the caller runs the two per-matrix projections —
    /// on a CPU tier, a non-ternary block, a never-uploaded block, or any GPU
    /// error; a CPU-tier run therefore behaves exactly as before.
    ///
    /// `kernel` is read only by the CUDA arm (its F-M2 tier gate): on Metal a
    /// CPU-tier kernel never produced the GPU handles the slot needs, and a
    /// build with neither backend never fuses.
    #[cfg_attr(
        not(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        )),
        allow(unused_variables)
    )]
    pub(super) fn try_fused_gate_up_ternary(
        &self,
        normed: &[f32],
        fused_gate_up: &mut [f32],
        gate_out: &mut [f32],
        up_out: &mut [f32],
        kernel: &dyn OneBitKernel,
    ) -> bool {
        let (Some(gate_blk), Some(up_blk)) =
            (self.ffn_gate.blocks_ternary(), self.ffn_up.blocks_ternary())
        else {
            return false;
        };
        let inter = gate_out.len();
        let total_rows = inter * 2;
        if up_out.len() != inter || fused_gate_up.len() < total_rows {
            return false;
        }
        let h = self.hidden_size;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let fused_ok = match self.ternary_fused_gate_up_slot() {
            Some(slot) => {
                let parts = [
                    blocks_as_bytes_ternary(gate_blk),
                    blocks_as_bytes_ternary(up_blk),
                ];
                self.slot_namespace.mark_gpu_used();
                let result = try_metal_gemv_ternary_fused(
                    normed,
                    &mut fused_gate_up[..total_rows],
                    self.slot_namespace.epoch(),
                    slot,
                    &parts,
                    total_rows,
                    h,
                );
                self.slot_namespace.mark_gpu_used();
                match result {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::debug!(
                            layer = self.layer_idx, error = %e,
                            "fused ternary gate+up GEMV on Metal failed, using per-matrix projections"
                        );
                        false
                    }
                }
            }
            None => false,
        };
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        let fused_ok = match cuda_ternary_fused_slot(
            self.fused_gate_up_handle_ternary.map(|hnd| hnd.id()),
            self.ffn_gate.gpu_handle().map(|hnd| hnd.id()),
        ) {
            // F-M2: a CPU-tier run never opens the CUDA device.
            Some(slot) if kernel.is_gpu_accelerated() => {
                let parts = [
                    blocks_as_bytes_ternary(gate_blk),
                    blocks_as_bytes_ternary(up_blk),
                ];
                match try_cuda_gemv_ternary_fused(
                    normed,
                    &mut fused_gate_up[..total_rows],
                    slot,
                    &parts,
                    total_rows,
                    h,
                    self.slot_namespace.epoch(),
                ) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "fused ternary gate+up GEMV on CUDA failed, falling back to CPU"
                        );
                        false
                    }
                }
            }
            _ => false,
        };
        #[cfg(not(any(
            all(feature = "metal", target_os = "macos"),
            all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            )
        )))]
        let fused_ok = false;
        if fused_ok {
            gate_out.copy_from_slice(&fused_gate_up[..inter]);
            up_out.copy_from_slice(&fused_gate_up[inter..total_rows]);
        }
        fused_ok
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — M-21 fused-QKV slot namespace and SoA length predicate
//
// Both are pure integer arithmetic over the CUDA weight-cache contract, so
// they run on every host, including this CUDA-less one. Nothing here opens a
// device or links cudarc.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::{
        cuda_fused_qkv_slot, cuda_ternary_fused_slot, cuda_tq2_soa_len_bytes,
        CUDA_FUSED_QKV_SLOT_TAG,
    };

    /// M-21: a dedicated fused handle keys the CUDA fused GEMV untagged (its
    /// id is private to the concatenation), a single projection's handle only
    /// through the tagged namespace, and a never-uploaded block not at all.
    #[test]
    fn cuda_ternary_fused_slot_prefers_the_dedicated_handle() {
        assert_eq!(cuda_ternary_fused_slot(Some(41), Some(7)), Some(41));
        assert_eq!(cuda_ternary_fused_slot(Some(41), None), Some(41));
        assert_eq!(
            cuda_ternary_fused_slot(None, Some(7)),
            Some(cuda_fused_qkv_slot(7))
        );
        assert_ne!(cuda_ternary_fused_slot(None, Some(7)), Some(7));
        assert_eq!(cuda_ternary_fused_slot(None, None), None);
    }

    /// Every id the allocator can mint is moved into a disjoint namespace, and
    /// the original id survives in the low 63 bits.
    #[test]
    fn fused_qkv_slot_is_disjoint_from_raw_handle_ids() {
        // 1 is the first value `alloc_handle_id` returns
        // (`NEXT_HANDLE_ID: AtomicU64::new(1)`); the rest are arbitrary later
        // values of the same `fetch_add(1)` counter.
        for id in [
            1u64,
            2,
            3,
            12_345,
            u32::MAX as u64,
            1u64 << 62,
            (1u64 << 63) - 1,
        ] {
            let slot = cuda_fused_qkv_slot(id);
            assert_ne!(slot, id, "slot for id {id} must not equal the raw id");
            assert_eq!(
                slot,
                id | (1u64 << 63),
                "slot for id {id} must be the id with bit 63 set"
            );
            assert_eq!(
                slot & !CUDA_FUSED_QKV_SLOT_TAG,
                id,
                "low 63 bits must round-trip"
            );
            assert_ne!(slot & CUDA_FUSED_QKV_SLOT_TAG, 0, "tag bit must be set");
        }
    }

    /// The aliasing invariant the tag relies on: `alloc_handle_id` starts at 1
    /// and only ever `fetch_add(1)`s, so no id it can return has bit 63 set
    /// (that would take 2^63 uploads). Untagged and tagged slots are therefore
    /// permanently disjoint.
    #[test]
    fn allocator_ids_never_carry_the_fused_tag() {
        assert_eq!(CUDA_FUSED_QKV_SLOT_TAG, 1u64 << 63);
        for id in [1u64, 1_000_000, 1u64 << 40, (1u64 << 63) - 1] {
            assert_eq!(
                id & CUDA_FUSED_QKV_SLOT_TAG,
                0,
                "a reachable allocator id ({id}) must have bit 63 clear"
            );
        }
        // Two distinct raw ids stay distinct after tagging.
        assert_ne!(cuda_fused_qkv_slot(7), cuda_fused_qkv_slot(8));
    }

    /// The length predicate against the real Ternary-Bonsai-1.7B QKV shape:
    /// `nq = 16`, `nkv = 8`, `head_dim = 128`, `hidden = k = 2048`.
    /// `total_rows = 16*128 + 2*(8*128) = 4096`, `blocks_per_row = 2048/128 =
    /// 16`, so the fused SoA buffer is `4096 * 16 * 34` bytes.
    #[test]
    fn tq2_soa_len_matches_the_17b_fused_qkv_shape() {
        let (nq, nkv, hd, k) = (16usize, 8usize, 128usize, 2048usize);
        let total_rows = nq * hd + 2 * (nkv * hd);
        assert_eq!(total_rows, 4096);
        let expected = cuda_tq2_soa_len_bytes(total_rows, k);
        assert_eq!(expected, Some(2_228_224));

        // A Q-only buffer — exactly what the colliding slot used to hand back —
        // is a third of that, so the check rejects it.
        let q_only = cuda_tq2_soa_len_bytes(nq * hd, k).expect("q-only length");
        assert_eq!(q_only, 1_114_112);
        assert_ne!(Some(q_only), expected);
    }

    /// A near-miss byte length is not merely unequal — it is unreachable: for a
    /// fixed `k`, valid SoA lengths are spaced `34 * (k / 128)` bytes apart
    /// (one whole output row), so nothing within a row of the correct length is
    /// a legal buffer. This is what makes the equality check in
    /// `try_cuda_gemv_ternary_fused` a real guard rather than a coincidence
    /// filter: a foreign buffer in the slot can only pass by being byte-exactly
    /// the right shape.
    #[test]
    fn tq2_soa_len_rejects_near_misses() {
        let k = 2048usize;
        let row_stride = 34 * (k / 128);
        assert_eq!(row_stride, 544);
        let len = cuda_tq2_soa_len_bytes(4096, k).expect("length");

        let reachable: std::collections::HashSet<usize> = (0usize..8192)
            .filter_map(|rows| cuda_tq2_soa_len_bytes(rows, k))
            .collect();
        assert!(reachable.contains(&len));
        for delta in [1usize, 2, 34, 33, 543] {
            for wrong in [len - delta, len + delta] {
                assert_ne!(wrong, len);
                assert!(
                    !reachable.contains(&wrong),
                    "{wrong} (len {len} ± {delta}) must not be a valid SoA length for k={k}"
                );
            }
        }
        // One whole row away IS reachable — the guard rejects it on equality,
        // which is exactly the Q-only-vs-fused case.
        assert!(reachable.contains(&(len - row_stride)));
        assert!(reachable.contains(&(len + row_stride)));
    }

    /// `k` must be a whole number of 128-wide quant groups; anything else has
    /// no TQ2_0_g128 SoA length at all.
    #[test]
    fn tq2_soa_len_requires_whole_quant_groups() {
        assert_eq!(cuda_tq2_soa_len_bytes(4096, 0), None);
        assert_eq!(cuda_tq2_soa_len_bytes(4096, 127), None);
        assert_eq!(cuda_tq2_soa_len_bytes(4096, 2049), None);
        assert_eq!(cuda_tq2_soa_len_bytes(1, 128), Some(34));
        assert_eq!(cuda_tq2_soa_len_bytes(0, 128), Some(0));
    }

    /// The 34-byte block stride is the contract, not an accident: one block
    /// is 32 bytes of 2-bit codes plus a 2-byte FP16 scale.
    #[test]
    fn tq2_soa_len_is_34_bytes_per_block() {
        for (rows, k, blocks) in [
            (1usize, 128usize, 1usize),
            (8, 256, 16),
            (4096, 2048, 65_536),
        ] {
            assert_eq!(cuda_tq2_soa_len_bytes(rows, k), Some(blocks * 34));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — M-21 on Metal: the ternary fused QKV and gate+up arms engage, stay
// on the ternary path, and match the CPU reference
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_fused_ternary_tests {
    use crate::block::TransformerBlock;
    use crate::kv_cache::KvCache;
    use crate::layers::linear::{LinearLayer, LinearTernary};
    use crate::layers::rms_norm::RmsNorm;
    use crate::layers::rope::RopeTable;
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use oxibonsai_kernels::gpu_backend::metal_full_layer::types::{WeightKey, WeightKind};
    use oxibonsai_kernels::{KernelDispatcher, KernelTier, MetalGraph, MetalGraphError};
    use std::sync::Arc;

    const H: usize = 256;
    const HD: usize = 64;
    const NQ: usize = 4;
    const NKV: usize = 2;
    const INTER: usize = 512;
    const SEQ: usize = 16;

    /// Varied ternary blocks (codes in {0, 1, 2} only — never the reserved
    /// `0b11`), so every projection is a genuinely different matrix.
    fn blocks(n: usize, seed: u64) -> Vec<BlockTQ2_0_g128> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for byte in qs.iter_mut() {
                    for lane in 0..4 {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        *byte |= (((state >> 33) % 3) as u8) << (2 * lane);
                    }
                }
                BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(0.02 + (seed % 7) as f32 * 0.003),
                }
            })
            .collect()
    }

    struct Weights {
        q: Vec<BlockTQ2_0_g128>,
        k: Vec<BlockTQ2_0_g128>,
        v: Vec<BlockTQ2_0_g128>,
        o: Vec<BlockTQ2_0_g128>,
        gate: Vec<BlockTQ2_0_g128>,
        up: Vec<BlockTQ2_0_g128>,
        down: Vec<BlockTQ2_0_g128>,
    }

    impl Weights {
        fn new() -> Self {
            let bpr = H / 128;
            Self {
                q: blocks(NQ * HD * bpr, 1),
                k: blocks(NKV * HD * bpr, 2),
                v: blocks(NKV * HD * bpr, 3),
                o: blocks(H * (NQ * HD / 128), 4),
                gate: blocks(INTER * bpr, 5),
                up: blocks(INTER * bpr, 6),
                down: blocks(H * (INTER / 128), 7),
            }
        }

        fn block(&self, kernel: &Arc<KernelDispatcher>) -> TransformerBlock<'_> {
            TransformerBlock::new(
                0,
                RmsNorm::new(vec![1.0; H], 1e-6),
                lin(&self.q, NQ * HD, H, kernel),
                lin(&self.k, NKV * HD, H, kernel),
                lin(&self.v, NKV * HD, H, kernel),
                lin(&self.o, H, NQ * HD, kernel),
                RmsNorm::new(vec![1.0; HD], 1e-6),
                RmsNorm::new(vec![1.0; HD], 1e-6),
                RmsNorm::new(vec![1.0; H], 1e-6),
                lin(&self.gate, INTER, H, kernel),
                lin(&self.up, INTER, H, kernel),
                lin(&self.down, H, INTER, kernel),
                NQ,
                NKV,
                HD,
                H,
            )
        }
    }

    /// A ternary linear layer over borrowed blocks.
    fn lin<'w>(
        blocks: &'w [BlockTQ2_0_g128],
        out: usize,
        inp: usize,
        kernel: &Arc<KernelDispatcher>,
    ) -> LinearLayer<'w> {
        LinearTernary::new(blocks, out, inp, Arc::clone(kernel))
            .expect("ternary linear")
            .into()
    }

    /// Residency probe against the bound session's weight cache, under the
    /// key the block's fused arms upload with: `(epoch, Tq2Soa, slot)`.
    fn resident_bytes(graph: &MetalGraph, epoch: u64, slot: u64) -> Option<usize> {
        graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Tq2Soa, slot), || {
                Err(MetalGraphError::ExecutionFailed("probe".into()))
            })
            .ok()
            .map(|handle| handle.byte_len())
    }

    /// The address of a weight slice — the mapped-tensor slot a block's fused
    /// arm keys the concatenation starting with it on.
    fn address_of(blocks: &[BlockTQ2_0_g128]) -> u64 {
        blocks.as_ptr() as u64
    }

    /// `M-21` end to end on Metal:
    ///
    /// 1. `upload_to_gpu` leaves every 1-bit fused field (and its gated
    ///    accessor) empty — so the block is not diverted into the 1-bit fused
    ///    branch — and, on Metal, uploads no ternary concatenation to the
    ///    kernel's own cache either (C2: nothing on Metal reads it), while the
    ///    fused slots come alive: the addresses of the Q and gate weights,
    ///    under the block's namespace epoch;
    /// 2. `forward` then runs the ternary fused-QKV arm **and** the fused
    ///    gate‖up GEMV — proven by both concatenations being resident in
    ///    `MetalGraph`'s cache, at the exact fused byte length, under exactly
    ///    that `(epoch, slot)` key and under no legacy one;
    /// 3. and the result matches the CPU reference block (no upload, reference
    ///    tier) over several positions.
    ///
    /// Runs in its own isolated device so the residency probes see only this
    /// test's uploads.
    #[test]
    fn ternary_fused_qkv_and_gate_up_engage_and_match_the_cpu_reference() {
        let Ok(isolated) = MetalGraph::new() else {
            return; // no Metal device on this host
        };
        let isolated = Arc::new(isolated);
        MetalGraph::with_session(&isolated, || {
            let weights = Weights::new();
            let cpu_kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));
            let gpu_kernel = Arc::new(KernelDispatcher::auto_detect());
            let reference = weights.block(&cpu_kernel);
            let mut gpu = weights.block(&gpu_kernel);
            assert_eq!(
                gpu.ternary_fused_qkv_slot(),
                None,
                "a block never uploaded to a GPU tier must not engage the fused arm"
            );
            gpu.upload_to_gpu(gpu_kernel.as_ref());

            assert!(
                gpu.fused_qkv_gpu_handle().is_none() && gpu.fused_gate_up_gpu_handle().is_none(),
                "a ternary block must expose no 1-bit fused handle"
            );
            assert!(
                gpu.fused_qkv_gpu_handle_ternary().is_none()
                    && gpu.fused_gate_up_gpu_handle_ternary().is_none(),
                "C2: on Metal no ternary concatenation goes to the kernel's own cache"
            );
            let qkv_slot = gpu
                .ternary_fused_qkv_slot()
                .expect("a GPU-uploaded ternary block has a fused-QKV slot");
            let gate_up_slot = gpu
                .ternary_fused_gate_up_slot()
                .expect("a GPU-uploaded ternary block has a fused gate‖up slot");
            assert_ne!(qkv_slot, gate_up_slot);
            assert_eq!(
                qkv_slot,
                address_of(&weights.q),
                "the fused-QKV slot is the mapped Q weight's address, not an upload-handle id"
            );
            assert_eq!(gate_up_slot, address_of(&weights.gate));
            let epoch = gpu.gpu_slot_epoch();
            assert_ne!(
                epoch,
                reference.gpu_slot_epoch(),
                "two standalone blocks never share a namespace"
            );

            let rope = RopeTable::new(HD, SEQ, 10_000.0);
            let mut ref_kv = KvCache::new(1, NKV, HD, SEQ);
            let mut gpu_kv = KvCache::new(1, NKV, HD, SEQ);
            let mut worst = 0f32;
            for pos in 0..4usize {
                let input: Vec<f32> = (0..H)
                    .map(|i| ((i * 13 + pos * 5) % 29) as f32 * 0.01 - 0.14)
                    .collect();
                let mut ref_hidden = input.clone();
                reference
                    .forward(
                        &mut ref_hidden,
                        pos,
                        &mut ref_kv,
                        &rope,
                        cpu_kernel.as_ref(),
                    )
                    .expect("CPU reference forward");
                let mut gpu_hidden = input.clone();
                gpu.forward(
                    &mut gpu_hidden,
                    pos,
                    &mut gpu_kv,
                    &rope,
                    gpu_kernel.as_ref(),
                )
                .expect("GPU-uploaded forward");
                assert_ne!(
                    ref_hidden, input,
                    "the forward must change the hidden state"
                );
                for (a, b) in ref_hidden.iter().zip(&gpu_hidden) {
                    worst = worst.max((a - b).abs());
                }
            }
            assert!(
                worst < 1e-3,
                "fused ternary forward diverged from the CPU reference: max |diff| = {worst}"
            );

            let bpr = H / 128;
            let qkv_rows = NQ * HD + 2 * NKV * HD;
            assert_eq!(
                resident_bytes(&isolated, epoch, qkv_slot),
                Some(qkv_rows * bpr * 34),
                "the ternary fused-QKV arm did not run (its Q‖K‖V buffer is not resident)"
            );
            assert_eq!(
                resident_bytes(&isolated, epoch, gate_up_slot),
                Some(2 * INTER * bpr * 34),
                "the fused gate‖up GEMV did not run (its gate‖up buffer is not resident)"
            );
            assert_eq!(
                resident_bytes(&isolated, oxibonsai_kernels::LEGACY_MODEL_EPOCH, qkv_slot),
                None,
                "the fused buffer is keyed under the block's epoch, not the legacy one"
            );
        });
    }

    /// The three block forward kinds, as far as the ternary fused arms go.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ForwardKind {
        Forward,
        SlidingWindow,
        Stats,
    }

    /// Run one forward kind of `block` at position 0 over a fresh KV cache.
    fn run_forward(
        kind: ForwardKind,
        block: &TransformerBlock<'_>,
        input: &[f32],
        rope: &RopeTable,
        kernel: &KernelDispatcher,
    ) -> Vec<f32> {
        let mut hidden = input.to_vec();
        let mut kv = KvCache::new(1, NKV, HD, SEQ);
        match kind {
            ForwardKind::Forward => block
                .forward(&mut hidden, 0, &mut kv, rope, kernel)
                .expect("forward"),
            ForwardKind::SlidingWindow => block
                .forward_with_sliding_window(&mut hidden, 0, &mut kv, rope, kernel, None)
                .expect("forward_with_sliding_window"),
            ForwardKind::Stats => {
                block
                    .forward_with_stats(&mut hidden, 0, &mut kv, rope, kernel)
                    .expect("forward_with_stats");
            }
        }
        hidden
    }

    /// **Every** block forward kind — `forward`,
    /// `forward_with_sliding_window` and `forward_with_stats` — runs the
    /// ternary fused arms (Q‖K‖V and gate‖up) on its own, is not diverted into
    /// the 1-bit branch, matches the CPU reference, and binds the **same**
    /// resident concatenations: whichever kind runs first uploads exactly the
    /// two ternary buffers (and nothing of 1-bit kind), and running the other
    /// two kinds afterwards does not grow `MetalGraph`'s resident bytes.
    ///
    /// Before B2 the sliding-window and stats forwards keyed the fused QKV on
    /// the Q projection's handle id (the legacy slot) and had no fused
    /// gate‖up arm, so mixing them with `forward` held the Q‖K‖V
    /// concatenation twice.
    #[test]
    fn every_block_forward_kind_runs_the_ternary_fused_arms_on_one_buffer() {
        if MetalGraph::new().is_err() {
            return; // no Metal device on this host
        }
        let weights = Weights::new();
        let cpu_kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));
        let gpu_kernel = Arc::new(KernelDispatcher::auto_detect());
        let reference = weights.block(&cpu_kernel);
        let rope = RopeTable::new(HD, SEQ, 10_000.0);
        let input: Vec<f32> = (0..H)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.01 - 0.15)
            .collect();
        let expected = run_forward(ForwardKind::Forward, &reference, &input, &rope, &cpu_kernel);
        let bpr = H / 128;
        let qkv_bytes = (NQ * HD + 2 * NKV * HD) * bpr * 34;
        let gate_up_bytes = 2 * INTER * bpr * 34;
        let kinds = [
            ForwardKind::Forward,
            ForwardKind::SlidingWindow,
            ForwardKind::Stats,
        ];
        for first in kinds {
            let Ok(isolated) = MetalGraph::new() else {
                return;
            };
            let isolated = Arc::new(isolated);
            MetalGraph::with_session(&isolated, || {
                let mut gpu = weights.block(&gpu_kernel);
                gpu.upload_to_gpu(gpu_kernel.as_ref());
                assert!(
                    gpu.fused_qkv_gpu_handle().is_none()
                        && gpu.fused_gate_up_gpu_handle().is_none(),
                    "{first:?}: the ternary block must not reach the 1-bit branch"
                );
                let epoch = gpu.gpu_slot_epoch();
                let qkv_slot = gpu.ternary_fused_qkv_slot().expect("fused-QKV slot");
                let gate_up_slot = gpu.ternary_fused_gate_up_slot().expect("gate‖up slot");

                let out = run_forward(first, &gpu, &input, &rope, &gpu_kernel);
                let worst = expected
                    .iter()
                    .zip(&out)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst < 1e-3,
                    "{first:?}: diverged from the CPU reference by {worst}"
                );
                assert_eq!(
                    resident_bytes(&isolated, epoch, qkv_slot),
                    Some(qkv_bytes),
                    "{first:?} did not run the ternary fused-QKV arm"
                );
                assert_eq!(
                    resident_bytes(&isolated, epoch, gate_up_slot),
                    Some(gate_up_bytes),
                    "{first:?} did not run the ternary fused gate‖up arm"
                );
                assert_eq!(
                    isolated.cached_weight_count().expect("count"),
                    2,
                    "{first:?}: exactly the two ternary concatenations are resident — no 1-bit \
                     (Q1Soa) buffer, no second copy"
                );
                let resident = isolated.bytes_uploaded();
                assert_eq!(resident, (qkv_bytes + gate_up_bytes) as u64);

                for other in kinds.into_iter().filter(|k| *k != first) {
                    let out = run_forward(other, &gpu, &input, &rope, &gpu_kernel);
                    let worst = expected
                        .iter()
                        .zip(&out)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 1e-3,
                        "{other:?} after {first:?}: diverged by {worst}"
                    );
                    assert_eq!(
                        isolated.bytes_uploaded(),
                        resident,
                        "{other:?} after {first:?} grew the resident bytes: a second copy of a \
                         ternary concatenation"
                    );
                }
            });
        }
    }
}
