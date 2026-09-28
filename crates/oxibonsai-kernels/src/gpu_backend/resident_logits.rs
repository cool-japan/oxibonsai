//! Sampled decode on the fused Metal route: top-k of the **resident** logits
//! (`perf-11`, sampled half).
//!
//! Split out of `gpu_backend/mod.rs` (headroom under the 2000-line ceiling)
//! and re-exported from [`crate::gpu_backend`]. Built only with the Metal
//! backend.

#![cfg(all(feature = "metal", target_os = "macos"))]

use super::{metal_graph, MetalGraph, MetalGraphError, MAX_RESIDENT_TOPK};

/// The `k` highest `(token id, logit)` pairs of the last fused forward's
/// logit row, as the GPU `topk_f32` kernel produced them.
///
/// Sorted by descending logit; an exact tie resolves to the **smaller** token
/// id (the kernel's documented rule, the same first-index convention as the
/// GPU argmax and the CPU samplers). A slot the kernel could not fill — only
/// possible when fewer than `k` logits are finite and greater than
/// `-INFINITY` — carries `(0, -INFINITY)`; callers must treat a non-finite
/// value inside the range they sample from as "not representable" and fall
/// back to the full row.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentLogitsTopK {
    /// Candidate token ids, best first.
    pub ids: Vec<u32>,
    /// Their logits, in the same order.
    pub values: Vec<f32>,
}

/// Run `topk_f32` over the logit row the **last fused forward on this
/// thread's Metal session** left in its `logits_buf`, and download only the
/// `k` winning `(id, logit)` pairs — `8·k` bytes instead of the `4·vocab`
/// bytes (993 KB at Bonsai 2's 248 320 vocabulary) a full-row download
/// costs (`perf-11`).
///
/// The caller must have just run a fused forward that writes `logits_buf`
/// on this thread's session (`BonsaiModel::forward_greedy_gpu` does: it
/// encodes the LM head into `logits_buf`, then the argmax), with no other
/// forward on the same session in between — i.e. the usual contract of one
/// engine per session (`EngineLease` binds a replica's own session to the
/// thread using it). The `logits_buf` lock is held for the whole dispatch,
/// so a concurrent forward on the same session cannot overwrite the row
/// mid-read.
///
/// `k` is clamped to `vocab` and to [`MAX_RESIDENT_TOPK`].
///
/// # Errors
///
/// [`MetalGraphError::BufferCreationFailed`] when no fused forward has run on
/// this session yet (no `logits_buf`), or when it is shorter than `vocab`
/// floats; [`MetalGraphError::InvalidDimensions`] for `vocab == 0` or
/// `k == 0`; a command-buffer failure from the dispatch itself.
pub fn metal_resident_logits_topk(
    vocab: usize,
    k: usize,
) -> Result<ResidentLogitsTopK, MetalGraphError> {
    use metal::MTLResourceOptions;

    if vocab == 0 || k == 0 {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "metal_resident_logits_topk: vocab={vocab} and k={k} must both be > 0"
        )));
    }
    let k = k.min(vocab).min(MAX_RESIDENT_TOPK);
    let vocab_u32 = u32::try_from(vocab).map_err(|_| {
        MetalGraphError::InvalidDimensions(format!(
            "metal_resident_logits_topk: vocab={vocab} does not fit u32"
        ))
    })?;
    let k_u32 = u32::try_from(k).map_err(|_| {
        MetalGraphError::InvalidDimensions(format!(
            "metal_resident_logits_topk: k={k} does not fit u32"
        ))
    })?;

    let graph = MetalGraph::global()?;
    let guard = graph
        .logits_buf
        .lock()
        .map_err(|_| MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into()))?;
    let logits = guard
        .as_ref()
        .ok_or(MetalGraphError::BufferCreationFailed)?;
    let needed = (vocab as u64).saturating_mul(4);
    if logits.length() < needed {
        return Err(MetalGraphError::BufferCreationFailed);
    }

    let shared = MTLResourceOptions::StorageModeShared;
    let ids_buf = metal_graph::alloc_buf(&graph.device, (k as u64) * 4, shared)?;
    let vals_buf = metal_graph::alloc_buf(&graph.device, (k as u64) * 4, shared)?;

    let cmd = graph.command_queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    let dispatched =
        graph.dispatch_topk_f32(encoder, logits, &ids_buf, &vals_buf, vocab_u32, k_u32);
    encoder.end_encoding();
    dispatched?;
    metal_graph::commit_and_wait(cmd, "resident logits top-k")?;

    let mut ids = vec![0u32; k];
    let mut values = vec![0.0f32; k];
    // SAFETY: both buffers are `StorageModeShared`, non-null (checked by
    // `alloc_buf`), hold exactly `k` 4-byte elements, and the command buffer
    // that wrote them has completed (`commit_and_wait` returned `Ok`).
    unsafe {
        std::ptr::copy_nonoverlapping(ids_buf.contents() as *const u32, ids.as_mut_ptr(), k);
        metal_graph::download_f32(&vals_buf, &mut values);
    }
    Ok(ResidentLogitsTopK { ids, values })
}

/// Download the full logit row the last fused forward on this thread's Metal
/// session left in `logits_buf` — the counted full-row fallback of the
/// sampled top-k path (`perf-11`), which needs no second forward.
///
/// Same session contract as [`metal_resident_logits_topk`].
///
/// # Errors
///
/// [`MetalGraphError::BufferCreationFailed`] when no fused forward has left a
/// `logits_buf` of at least `vocab` floats on this session;
/// [`MetalGraphError::InvalidDimensions`] for `vocab == 0`.
pub fn metal_resident_logits_download(vocab: usize) -> Result<Vec<f32>, MetalGraphError> {
    if vocab == 0 {
        return Err(MetalGraphError::InvalidDimensions(
            "metal_resident_logits_download: vocab must be > 0".into(),
        ));
    }
    let graph = MetalGraph::global()?;
    let guard = graph
        .logits_buf
        .lock()
        .map_err(|_| MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into()))?;
    let logits = guard
        .as_ref()
        .ok_or(MetalGraphError::BufferCreationFailed)?;
    if logits.length() < (vocab as u64).saturating_mul(4) {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let mut out = vec![0.0f32; vocab];
    // SAFETY: `logits_buf` is a `StorageModeShared` buffer (allocated by the
    // fused LM-head path) holding at least `vocab` floats (checked above), and
    // every command buffer that writes it is committed and waited on before
    // the fused forward returns.
    unsafe {
        metal_graph::download_f32(logits, &mut out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_topk_cap_matches_the_kernel_dispatcher_cap() {
        assert_eq!(
            MAX_RESIDENT_TOPK as u32,
            super::super::metal_dispatch::MAX_TOPK_F32,
            "the public cap must equal the kernel's fixed picked[] bound"
        );
    }
}
