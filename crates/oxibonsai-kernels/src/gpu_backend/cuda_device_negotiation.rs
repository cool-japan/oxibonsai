//! Host-side CUDA device-capability negotiation, kept pure and **not**
//! `cfg`-gated so it is exercised by the ordinary CPU test suite on every
//! host.
//!
//! `cuda_graph::cudagraph_global_group` — where these two findings' device
//! queries and kernel-source rewrite actually live — is
//! `cfg(all(feature = "native-cuda", any(target_os = "linux", target_os =
//! "windows")))`: on this project's macOS development host that condition is
//! never true for *any* feature flag, so nothing in that module (or its
//! sibling `cuda_graph` files) is ever compiled here, and a `#[cfg(test)]`
//! added there is dead on arrival (verified: `cargo check -p
//! oxibonsai-kernels --features native-cuda -v 2>&1 | grep -c cudagraph`
//! prints `0`). The two decisions below are plain integer arithmetic with no
//! CUDA dependency, so splitting them out — leaving only the actual device
//! query and the `CUDA_IMAGEN_ATTN_SRC` kernel-source rewrite behind the
//! `cfg` gate — is what lets their test coverage run on every host,
//! including this one.
//!
//! - [`parse_device_ordinal`]: `OXIBONSAI_CUDA_DEVICE` value → device
//!   ordinal (finding **F-M4**), consumed by
//!   `cudagraph_global_group::CudaGraph::selected_device`.
//! - [`select_flash_tile_keys`] / [`flash_shared_bytes`]: choose the widest
//!   flash-attention key-tile depth (`FA_BK`) that fits the device's dynamic
//!   shared-memory opt-in maximum (finding **F5**), consumed by
//!   `cudagraph_global_group::CudaGraph::new`.

/// `head_dim` cap of the wide flash-attention build (`FA_DMAX`), mirroring
/// `cudagraph_imagen_attn_group::DIT_FLASH_HEAD_DIM_CAP`. Also this module's
/// per-key-tile byte multiplier in [`flash_shared_bytes`].
pub const FLASH_HEAD_DIM_CAP: usize = 384;

/// Key-tile depths (`FA_BK`) the wide flash-attention build can be compiled
/// with, widest first (finding **F5**). `32` keys × `384` head dims × (K‖V)
/// × `4` B = `98_304` B, which only Volta (96 KiB opt-in) and Ampere and
/// later (99 328 B) can opt into; the `16`-key build needs `49_152` B, which
/// fits Turing's 64 KiB opt-in *and* Pascal's 48 KiB limit.
pub const FLASH_TILE_KEYS: [usize; 3] = [32, 16, 8];

/// Dynamic shared memory the wide flash-attention build needs at `tile_keys`.
pub const fn flash_shared_bytes(tile_keys: usize) -> usize {
    tile_keys * FLASH_HEAD_DIM_CAP * 2 * std::mem::size_of::<f32>()
}

/// Widest key tile whose dynamic shared footprint fits `optin_max` (finding
/// **F5**). Never fails: the narrowest documented tile is the floor.
pub fn select_flash_tile_keys(optin_max: usize) -> usize {
    FLASH_TILE_KEYS
        .iter()
        .copied()
        .find(|&t| flash_shared_bytes(t) <= optin_max)
        .unwrap_or(FLASH_TILE_KEYS[FLASH_TILE_KEYS.len() - 1])
}

/// `head_dim` cap of the **lean** flash-attention build (`FA_DMAX = 128`), the
/// one `cudagraph_imagen_attn_group` dispatches to for the DiT. It requests at
/// most `32 × 128 × 2 × 4 = 32 768` B of dynamic shared memory, which is inside
/// the 48 KiB every CUDA device gives a block without any opt-in, so it is
/// unaffected by F5 and deliberately gets no opt-in (keeping its large default
/// L1 for the register-resident `Q[]`/`O[]` streaming).
pub const FLASH_LEAN_HEAD_DIM_CAP: usize = 128;

/// Key-tile depth the lean build is compiled with — must equal
/// `cudagraph_imagen_attn_group::DIT_FLASH_BK` (asserted there).
pub const FLASH_LEAN_TILE_KEYS: usize = 32;

/// Dynamic shared memory one flash-attention launch needs: the `Ksh ‖ Vsh`
/// staging arena, `tile_keys × head_dim` floats each.
///
/// This is the arithmetic the **launcher** must use (finding **F5**). Before the
/// fix `launch_joint_attention_flash_resident` hardcoded `tile_keys = 32` while
/// `CudaGraph::new` had opted the wide build into only `49 152` B on a 64 KiB
/// device, so every wide launch asked for `32 × 384 × 2 × 4 = 98 304` B and
/// failed with `CUDA_ERROR_INVALID_VALUE` — exactly the Turing/Pascal fleet F5
/// set out to rescue.
pub const fn flash_launch_shared_bytes(tile_keys: usize, head_dim: usize) -> usize {
    tile_keys * head_dim * 2 * std::mem::size_of::<f32>()
}

/// Key tile the launcher must size its shared memory from, for `head_dim`.
///
/// `head_dim <= 128` dispatches to the lean build (fixed tile); anything wider
/// dispatches to the device-negotiated wide build, whose tile is whatever
/// [`select_flash_tile_keys`] chose at init and which `CudaGraph::
/// flash_large_tile_keys` publishes.
pub fn flash_launch_tile_keys(head_dim: usize, large_tile_keys: usize) -> usize {
    if head_dim <= FLASH_LEAN_HEAD_DIM_CAP {
        FLASH_LEAN_TILE_KEYS
    } else {
        large_tile_keys
    }
}

/// Widest `head_dim` the wide build can serve once the device has granted
/// `granted_shared` bytes at `tile_keys` (finding **F5**).
///
/// `granted_shared == 0` means the device refused the opt-in outright, so the
/// wide build is unusable and this is `0`: every `head_dim > 128` must then be
/// refused **before** launch rather than failing inside it.
pub fn flash_large_max_head_dim(granted_shared: usize, tile_keys: usize) -> usize {
    if tile_keys == 0 {
        return 0;
    }
    (granted_shared / (tile_keys * 2 * std::mem::size_of::<f32>())).min(FLASH_HEAD_DIM_CAP)
}

/// Whether this device can serve `head_dim` at all (finding **F5**).
///
/// `head_dim <= 128` always can — it uses the lean build, which needs no opt-in.
/// Wider head dims need the wide build's granted arena, so they are bounded by
/// `large_max_head_dim` (`CudaGraph::flash_large_max_head_dim`).
pub fn flash_head_dim_supported(head_dim: usize, large_max_head_dim: usize) -> bool {
    head_dim <= FLASH_LEAN_HEAD_DIM_CAP || head_dim <= large_max_head_dim
}

/// The wide build's key tile and the shared-memory request to opt into, for a
/// device whose `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` is
/// `optin_max` (finding **F5**).
///
/// Never asks for more than the device allows, so the opt-in is refused only by
/// a device that reported a limit it will not honour.
pub fn negotiate_flash_large(optin_max: usize) -> (usize, usize) {
    let tile_keys = select_flash_tile_keys(optin_max);
    (tile_keys, flash_shared_bytes(tile_keys).min(optin_max))
}

/// Parse an `OXIBONSAI_CUDA_DEVICE` value into a device ordinal (finding
/// **F-M4**). An unparsable value logs a warning and falls back to device
/// `0` rather than panicking or silently miscounting devices; callers only
/// ever invoke this once the env var is confirmed present, so "not set" is
/// handled by the caller, not here.
pub fn parse_device_ordinal(raw: &str) -> usize {
    match raw.trim().parse::<usize>() {
        Ok(ordinal) => ordinal,
        Err(_) => {
            tracing::warn!("OXIBONSAI_CUDA_DEVICE={raw:?} is not a device ordinal; using device 0");
            0
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// F3 — attention-score shared-memory sizing (head_dim > 128)
// CUDA is unvalidated: this session has no CUDA hardware, so everything below
// is host-side arithmetic checked by unit tests, never run through nvcc/NVRTC
// or a real launch.
// ═════════════════════════════════════════════════════════════════════════════

/// Threads per `batched_attn_scores_v2` CTA. Fixed by
/// `cuda_full_layer::launchers::launch_batched_attn_scores_v2`; the kernel no
/// longer assumes it equals `head_dim`.
pub const ATTN_SCORES_BLOCK_DIM: u32 = 128;

/// Largest `head_dim` `batched_attn_scores_v2` will be launched with
/// (finding **F3**).
///
/// The kernel itself has no cap — it stages `head_dim` floats in dynamically
/// sized shared memory and strides over them — so this is purely a sanity
/// bound on how much shared memory one CTA may ask for. `1024` floats plus the
/// cross-warp partials is 4 KiB, comfortably inside the 48 KiB every CUDA
/// device grants a block without an opt-in, and four times Bonsai 2 27B's
/// `head_dim = 256` (`qwen35.attention.key_length`), the value that motivated
/// the finding. The pre-F3 kernel silently truncated the QK dot product to
/// `min(head_dim, 128)`; anything above this bound is now a named error
/// instead.
pub const CUDA_ATTN_MAX_HEAD_DIM: u32 = 1024;

/// Dynamic shared memory one `batched_attn_scores_v2` CTA needs (finding
/// **F3**): `head_dim` floats for the staged Q vector followed by
/// `block_dim / 32` floats for the cross-warp partials.
///
/// The kernel derives both sub-arrays from the same `extern __shared__` block
/// with exactly this split, so this function and the kernel must agree. It is
/// the single source of truth for `LaunchConfig::shared_mem_bytes`.
///
/// # Errors
/// - `head_dim == 0`, or above [`CUDA_ATTN_MAX_HEAD_DIM`];
/// - `block_dim` not a non-zero multiple of the 32-lane warp — the kernel's
///   cross-warp reduction sums exactly `block_dim / 32` partials, so a partial
///   warp would drop lanes.
pub fn attn_scores_shared_bytes(head_dim: u32, block_dim: u32) -> Result<u32, String> {
    if head_dim == 0 {
        return Err("batched_attn_scores_v2: head_dim must be non-zero".to_string());
    }
    if head_dim > CUDA_ATTN_MAX_HEAD_DIM {
        return Err(format!(
            "batched_attn_scores_v2: head_dim {head_dim} exceeds the supported maximum \
             {CUDA_ATTN_MAX_HEAD_DIM}; the Q vector is staged in shared memory, so a wider \
             head would overrun one CTA's shared-memory budget"
        ));
    }
    if block_dim == 0 || !block_dim.is_multiple_of(32) {
        return Err(format!(
            "batched_attn_scores_v2: block_dim {block_dim} must be a non-zero multiple of the \
             32-lane warp size"
        ));
    }
    let floats = head_dim + block_dim / 32;
    floats
        .checked_mul(std::mem::size_of::<f32>() as u32)
        .ok_or_else(|| format!("batched_attn_scores_v2: shared-memory size overflows: {floats}"))
}

// ═════════════════════════════════════════════════════════════════════════════
// F4 — 64-bit KV-cache addressing
// CUDA is unvalidated: this session has no CUDA hardware, so everything below
// is host-side arithmetic checked by unit tests, never run through nvcc/NVRTC
// or a real launch.
// ═════════════════════════════════════════════════════════════════════════════

/// Bytes per KV-cache element (`half`), mirroring
/// `metal_full_layer::types::KV_ELEMENT_BYTES`.
pub const CUDA_KV_ELEMENT_BYTES: u64 = 2;

/// Element offset of `layer_idx`'s slab inside a flat CUDA KV cache
/// (finding **F4**).
///
/// Computed entirely in `u64`. `CudaKvCache::layer_offset_elements` returned
/// `u32` via a silent `as u32`, and the three attention kernels took the value
/// as `unsigned int`, so the whole linear index wrapped once
/// `n_layers * n_kv * max_seq * head_dim` passed `2^32` — two layers aliased
/// onto the same addresses with no error anywhere. The kernels now take
/// `unsigned long long`, so unlike the Metal twin (whose MSL bindings are still
/// `constant uint&`, capping it at `u32::MAX` elements) there is no 32-bit
/// ceiling left on this path.
#[inline]
pub fn cuda_kv_layer_offset_elements(
    layer_idx: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> u64 {
    (layer_idx as u64) * (n_kv as u64) * (max_seq as u64) * (head_dim as u64)
}

/// Validate a requested KV-cache geometry and return the element count of one
/// cache buffer (K or V) — finding **F4**.
///
/// `acquire_kv_cache` computed `n_layers * n_kv * max_seq * head_dim` in
/// `usize` with no overflow check at all before handing it to `alloc_zeros`.
/// Every multiplication here is checked, so an absurd `--max-seq-len` produces
/// a named error rather than a wrapped allocation.
///
/// # Errors
/// Any zero dimension, or an overflow in the element or byte count.
pub fn check_cuda_kv_cache_geometry(
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<u64, String> {
    if n_layers == 0 || n_kv == 0 || max_seq == 0 || head_dim == 0 {
        return Err(format!(
            "CUDA KV cache geometry must be non-zero, got n_layers={n_layers}, n_kv={n_kv}, \
             max_seq={max_seq}, head_dim={head_dim}"
        ));
    }
    let per_position = (n_layers as u64)
        .checked_mul(n_kv as u64)
        .and_then(|v| v.checked_mul(head_dim as u64))
        .ok_or_else(|| {
            format!(
                "CUDA KV cache geometry overflows: n_layers={n_layers} x n_kv={n_kv} x \
                 head_dim={head_dim}"
            )
        })?;
    let total_elements = per_position.checked_mul(max_seq as u64).ok_or_else(|| {
        format!("CUDA KV cache geometry overflows: {per_position} x max_seq={max_seq}")
    })?;
    let byte_len = total_elements
        .checked_mul(CUDA_KV_ELEMENT_BYTES)
        .ok_or_else(|| {
            format!(
                "CUDA KV cache byte length overflows: {total_elements} x {CUDA_KV_ELEMENT_BYTES}"
            )
        })?;
    // `alloc_zeros` takes a `usize` element count; on a 32-bit host the u64
    // arithmetic above would otherwise be truncated at the call.
    if total_elements > usize::MAX as u64 || byte_len > usize::MAX as u64 {
        return Err(format!(
            "CUDA KV cache of {total_elements} elements ({byte_len} bytes) does not fit in a \
             host usize"
        ));
    }
    Ok(total_elements)
}

// ═════════════════════════════════════════════════════════════════════════════
// F10 — 16-byte alignment of the SoA quant section
// ═════════════════════════════════════════════════════════════════════════════

/// Byte offset of the quant section inside an SoA weight buffer: the FP16
/// scales come first, one per block.
#[inline]
pub const fn soa_qs_offset_bytes(total_blocks: u64) -> u64 {
    total_blocks * 2
}

/// Whether the SoA quant section is 16-byte aligned, i.e. whether the CUDA
/// GEMV kernels may use `ld.global.nc.v4.u32` (finding **F10**).
///
/// The PTX vector load requires a 16-byte-aligned address. The buffer base is
/// a fresh `cuMemAlloc` (256-byte aligned, never a sub-slice) and each block's
/// quant stride is a whole number of 16-byte units, so the requirement reduces
/// to `soa_qs_offset_bytes(total_blocks) % 16 == 0`, i.e.
/// `total_blocks % 8 == 0`. The kernels' in-source comment only ever argued
/// `% 2` (4-byte). This is the host-side twin of the device-side
/// `soa_quant_aligned16` predicate in `cuda_kernels.rs`; the kernels fall back
/// to scalar `__ldg` loads when it is false.
///
/// That fallback is only a "slower, not a `CUDA_ERROR_MISALIGNED_ADDRESS`
/// abort" story for **even** `total_blocks`: the scalar reads are four
/// naturally-aligned `unsigned int` loads starting at byte
/// `2 * total_blocks`, which itself needs 4-byte (2-mod-4) alignment. Even
/// `total_blocks` gives a multiple-of-4 byte offset, so the fallback is
/// correct, just not vectorised. Odd `total_blocks` gives a byte offset that
/// is `2 mod 4`, so the scalar `__ldg` reads are themselves misaligned and
/// still abort — matching the pre-fix behaviour for that one sub-case, not a
/// regression. `total_blocks = n_rows * (k / 128)` is odd only when both
/// `n_rows` and `k / 128` are odd; `n_rows` is a weight tensor's output-row
/// count, and no shape in this module's `soa_alignment_holds_for_shipped_shapes_and_fails_for_others`
/// test (nor any GGUF tensor dimension named in this crate's model support so
/// far) is odd, so the odd case has not been observed in practice. That is
/// evidence from the shapes checked, not a proof over every possible weight
/// shape, so the guarantee above must be stated as "even `total_blocks`", not
/// "any non-multiple-of-8 shape".
///
/// No non-test, non-doc caller wires this predicate (or [`soa_total_blocks`])
/// against the SoA reformat path
/// (`cuda_graph::cudagraph_reformat_tq2_blocks_to_soa_group` /
/// `cudagraph_reformat_q1_aos_to_soa_group`, both in this package). That is
/// intentional, not an oversight: the F10 fix (spec item 5) took the in-kernel
/// grid-uniform branch — `soa_quant_aligned16` / `soa_load4_u32` in
/// `cuda_kernels.rs` — as the single source of truth, so every launch is
/// correct regardless of shape without the host needing to pre-check or
/// reject anything; adding a host-side assert at the reformat sites would be
/// redundant defensive coding for a branch that can never be wrong. This
/// function and [`soa_total_blocks`] exist to give that in-kernel arithmetic
/// unit-test coverage on hosts (like this one) that cannot compile the CUDA
/// modules at all; they are a host-side test twin of the device predicate,
/// not a host-side gate.
#[inline]
pub const fn soa_qs_vector_load_ok(total_blocks: u64) -> bool {
    soa_qs_offset_bytes(total_blocks).is_multiple_of(16)
}

/// Number of SoA blocks a `[n_rows x k]` weight has at group size 128.
#[inline]
pub const fn soa_total_blocks(n_rows: u64, k: u64) -> u64 {
    n_rows * (k / 128)
}

// ═════════════════════════════════════════════════════════════════════════════
// F9 — prefill chunk staging layout
// ═════════════════════════════════════════════════════════════════════════════

/// Element ranges of token `t` inside a prefill chunk's staging buffers
/// (finding **F9**).
///
/// Returns `(pos_seqlen_range, rope_range)`:
/// - the `[pos, pos + 1]` pair lives at `2t..2t+2` of the positions buffer;
/// - this token's RoPE cosines and sines live at
///   `t*half_dim..(t+1)*half_dim` of each of the two rope buffers.
///
/// Split out of `cuda_full_layer::CudaPrefillRopeChunk::token` so the indexing
/// — the one part of the F9 hoist that can be wrong without any compiler
/// noticing — is unit-tested on hosts that cannot compile the CUDA modules.
///
/// # Errors
/// `t` outside the uploaded chunk, `half_dim == 0`, or an overflowing range.
pub fn prefill_chunk_token_ranges(
    t: usize,
    n_tokens: usize,
    half_dim: usize,
) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>), String> {
    if t >= n_tokens {
        return Err(format!(
            "prefill chunk: token {t} outside the uploaded chunk of {n_tokens} tokens"
        ));
    }
    if half_dim == 0 {
        return Err("prefill chunk: half_dim must be non-zero".to_string());
    }
    let pos_lo = t
        .checked_mul(2)
        .ok_or_else(|| format!("prefill chunk: position offset overflows at token {t}"))?;
    let pos_hi = pos_lo
        .checked_add(2)
        .ok_or_else(|| format!("prefill chunk: position offset overflows at token {t}"))?;
    let rope_lo = t
        .checked_mul(half_dim)
        .ok_or_else(|| format!("prefill chunk: rope offset overflows at token {t}"))?;
    let rope_hi = rope_lo
        .checked_add(half_dim)
        .ok_or_else(|| format!("prefill chunk: rope offset overflows at token {t}"))?;
    Ok((pos_lo..pos_hi, rope_lo..rope_hi))
}

#[cfg(test)]
mod tests {

    use super::*;

    // ── F9: prefill chunk staging layout ────────────────────────────────────

    /// F9: every token must map to its own two-element position pair and its
    /// own `half_dim` RoPE window, with no overlap and no gap. A wrong window
    /// here feeds one token's RoPE to another — plausible output, silently
    /// wrong, and invisible to the compiler.
    #[test]
    fn prefill_chunk_token_ranges_tile_the_buffers_exactly() {
        let (n_tokens, half_dim) = (7usize, 64usize);
        let mut prev_pos_end = 0usize;
        let mut prev_rope_end = 0usize;
        for t in 0..n_tokens {
            let (pos, rope) = prefill_chunk_token_ranges(t, n_tokens, half_dim)
                .expect("token inside the chunk must resolve");
            assert_eq!(pos.start, prev_pos_end, "position gap/overlap at t={t}");
            assert_eq!(pos.end - pos.start, 2, "position pair must be 2 elements");
            assert_eq!(rope.start, prev_rope_end, "rope gap/overlap at t={t}");
            assert_eq!(rope.end - rope.start, half_dim, "rope window at t={t}");
            prev_pos_end = pos.end;
            prev_rope_end = rope.end;
        }
        // The two buffers are exactly filled.
        assert_eq!(prev_pos_end, n_tokens * 2);
        assert_eq!(prev_rope_end, n_tokens * half_dim);
    }

    /// F9: the hoist must not let a token index escape the uploaded chunk —
    /// that would read a stale position and write K/V at the wrong slot of the
    /// SHARED decode KV cache.
    #[test]
    fn prefill_chunk_token_ranges_reject_out_of_chunk_indices() {
        assert!(prefill_chunk_token_ranges(0, 0, 64).is_err());
        assert!(prefill_chunk_token_ranges(4, 4, 64).is_err());
        assert!(prefill_chunk_token_ranges(3, 4, 64).is_ok());
        assert!(prefill_chunk_token_ranges(0, 4, 0).is_err());
        assert!(prefill_chunk_token_ranges(usize::MAX - 1, usize::MAX, 64).is_err());
    }

    // ── F3: attention-score shared-memory sizing ────────────────────────────

    /// F3: the kernel's shared block is `[head_dim floats | n_warps floats]`.
    /// The launcher must size exactly that, for every head dim the finding
    /// names — including Bonsai 2 27B's 256, which the pre-fix kernel
    /// truncated to 128 with no error.
    #[test]
    fn attn_scores_shared_bytes_covers_q_vector_and_warp_partials() {
        let warps = ATTN_SCORES_BLOCK_DIM / 32;
        assert_eq!(warps, 4);
        for head_dim in [64u32, 128, 192, 256, 512, CUDA_ATTN_MAX_HEAD_DIM] {
            let bytes = attn_scores_shared_bytes(head_dim, ATTN_SCORES_BLOCK_DIM)
                .expect("supported head_dim must size");
            // Assert the DECOMPOSITION, not just the total. The kernel splits
            // this one `extern __shared__` block by hand as `[0, head_dim)` for
            // the staged Q vector and `[head_dim, head_dim + n_warps)` for the
            // cross-warp partials (`attn_smem + head_dim` in
            // `cuda_attn_kernels.rs`). If either side drifts, `warp_sums` runs
            // off the end of the allocation -- silent shared-memory corruption
            // on the device, not a fault. This is the only place the two sides
            // of that contract can be checked without a GPU.
            let staged_q_bytes = head_dim * 4;
            let warp_partial_bytes = (ATTN_SCORES_BLOCK_DIM / 32) * 4;
            assert_eq!(
                bytes,
                staged_q_bytes + warp_partial_bytes,
                "head_dim={head_dim}"
            );
            assert_eq!(bytes, (head_dim + warps) * 4, "head_dim={head_dim}");
            // Must stay inside the 48 KiB every device grants without opt-in.
            assert!(bytes <= 48 * 1024, "head_dim={head_dim} bytes={bytes}");
        }
    }

    /// F3: Bonsai 2 27B is the model the finding is about — assert its exact
    /// geometry rather than only the general rule.
    #[test]
    fn attn_scores_shared_bytes_admits_bonsai2_head_dim_256() {
        // qwen35.attention.key_length = qwen35.attention.value_length = 256.
        let bytes = attn_scores_shared_bytes(256, ATTN_SCORES_BLOCK_DIM)
            .expect("Bonsai 2's head_dim = 256 must be supported");
        assert_eq!(bytes, (256 + 4) * 4);
        // Twice the old fixed `__shared__ float shared_q[128]` staging area:
        // exactly the half of every Q vector the pre-F3 kernel dropped.
        assert!(bytes > 128 * 4 * 2);
    }

    /// F3: every rejected input must be rejected *before* launch, not by
    /// corrupting shared memory inside the kernel.
    #[test]
    fn attn_scores_shared_bytes_rejects_unsupported_geometry() {
        assert!(attn_scores_shared_bytes(0, ATTN_SCORES_BLOCK_DIM).is_err());
        assert!(
            attn_scores_shared_bytes(CUDA_ATTN_MAX_HEAD_DIM + 1, ATTN_SCORES_BLOCK_DIM).is_err()
        );
        // A partial warp would make the kernel's `blockDim.x / 32` reduction
        // silently drop lanes.
        assert!(attn_scores_shared_bytes(128, 0).is_err());
        assert!(attn_scores_shared_bytes(128, 48).is_err());
        assert!(attn_scores_shared_bytes(128, 96).is_ok());
    }

    // ── F4: 64-bit KV-cache addressing ──────────────────────────────────────

    /// F4: the offset must be computed in `u64`. The pre-fix
    /// `layer_offset_elements` returned `u32`, so an 8B-class model at a long
    /// context aliased two layers onto one address range.
    #[test]
    fn kv_layer_offset_is_computed_in_64_bits() {
        let (n_kv, max_seq, head_dim) = (8usize, 512usize, 128usize);
        assert_eq!(cuda_kv_layer_offset_elements(0, n_kv, max_seq, head_dim), 0);
        assert_eq!(
            cuda_kv_layer_offset_elements(1, n_kv, max_seq, head_dim),
            (8 * 512 * 128) as u64
        );

        // The reachability case from the finding: 80 GB-class hardware, 8B
        // geometry at 128 K context. The old `as u32` wrapped here.
        let (n_kv, max_seq, head_dim) = (8usize, 131_072usize, 128usize);
        let per_layer = (n_kv * max_seq * head_dim) as u64;
        for layer_idx in [1usize, 33, 35] {
            let offset = cuda_kv_layer_offset_elements(layer_idx, n_kv, max_seq, head_dim);
            assert_eq!(offset, layer_idx as u64 * per_layer);
            // Truncation to u32 is what the fix removes — prove the value
            // really does leave the 32-bit range for the layers the verdict
            // names, so this test would fail against the old code.
            if layer_idx >= 33 {
                assert!(
                    offset > u32::MAX as u64,
                    "layer {layer_idx} must exceed u32"
                );
                assert_ne!(offset, u64::from(offset as u32));
            }
        }
    }

    /// F4: `acquire_kv_cache` had no overflow check at all on
    /// `n_layers * n_kv * max_seq * head_dim`.
    #[test]
    fn kv_cache_geometry_is_checked_end_to_end() {
        assert_eq!(
            check_cuda_kv_cache_geometry(36, 8, 4096, 128),
            Ok(36 * 8 * 4096 * 128)
        );
        // Bonsai 2 27B: 64 layers, 4 KV heads, head_dim 256.
        assert_eq!(
            check_cuda_kv_cache_geometry(64, 4, 262_144, 256),
            Ok(64u64 * 4 * 262_144 * 256)
        );
        for bad in [(0, 8, 8, 8), (8, 0, 8, 8), (8, 8, 0, 8), (8, 8, 8, 0)] {
            assert!(check_cuda_kv_cache_geometry(bad.0, bad.1, bad.2, bad.3).is_err());
        }
        // Overflow rather than wraparound.
        assert!(check_cuda_kv_cache_geometry(usize::MAX, usize::MAX, 2, 2).is_err());
        assert!(check_cuda_kv_cache_geometry(1 << 20, 1 << 20, 1 << 20, 1 << 20).is_err());
    }

    /// F4: a geometry above the 32-bit range must now be *accepted*, because
    /// the CUDA kernels take `unsigned long long`. This is the difference from
    /// the Metal twin, which still has to refuse it.
    #[test]
    fn kv_cache_geometry_accepts_beyond_the_32_bit_range() {
        let total = check_cuda_kv_cache_geometry(36, 8, 131_072, 128)
            .expect("8B at 128K context must be allowed on CUDA");
        assert!(
            total > u32::MAX as u64,
            "total={total} should exceed u32::MAX"
        );
    }

    // ── F10: SoA quant-section alignment ────────────────────────────────────

    /// F10: the predicate the kernels branch on is `total_blocks % 8 == 0`,
    /// not the `% 2` the old in-source comment argued.
    #[test]
    fn soa_vector_load_requires_sixteen_byte_alignment() {
        for blocks in 0u64..64 {
            assert_eq!(
                soa_qs_vector_load_ok(blocks),
                blocks.is_multiple_of(8),
                "total_blocks={blocks}"
            );
            assert_eq!(soa_qs_offset_bytes(blocks), blocks * 2);
        }
        // The old comment's claim (`% 2`, i.e. 4-byte) is strictly weaker:
        // these are even but NOT 16-byte aligned, and were the silent aborts.
        for blocks in [2u64, 4, 6, 10, 12, 14] {
            assert!(blocks.is_multiple_of(2));
            assert!(!soa_qs_vector_load_ok(blocks));
        }
    }

    /// F10: every Qwen3 tensor shape currently shipped is aligned — which is
    /// why the bug is latent — while a plausible shape is not.
    #[test]
    fn soa_alignment_holds_for_shipped_shapes_and_fails_for_others() {
        // 1.7B / 8B hidden sizes, k = hidden_size, blocks_per_row = k / 128.
        for &(n_rows, k) in &[
            (2048u64, 2048u64),
            (4096, 4096),
            (11008, 4096),
            (4096, 11008),
            (5120, 5120),
            (17408, 5120),
        ] {
            let blocks = soa_total_blocks(n_rows, k);
            assert!(
                soa_qs_vector_load_ok(blocks),
                "n_rows={n_rows} k={k} blocks={blocks}"
            );
        }
        // A shape that is NOT a multiple of 8 blocks: one row of k = 512
        // gives 4 blocks, so the section starts 8 bytes into a 16-byte unit.
        let blocks = soa_total_blocks(1, 512);
        assert_eq!(blocks, 4);
        assert!(!soa_qs_vector_load_ok(blocks));
        assert_eq!(soa_qs_offset_bytes(blocks) % 16, 8);
    }

    /// F5: the key-tile choice must keep the wide attention kernel inside the
    /// opt-in maximum of every GPU generation the finding names. This is the
    /// hardware-free half of the F5 acceptance test; it now runs on every
    /// host (previously dead on macOS — see the module doc above).
    #[test]
    fn flash_tile_selection_fits_every_documented_optin_limit() {
        // Ampere/Ada/Hopper (99 328 B) and Volta (98 304 B) keep the 32 tile.
        assert_eq!(select_flash_tile_keys(99_328), 32);
        assert_eq!(select_flash_tile_keys(98_304), 32);
        // Turing (SM 7.5, 64 KiB) and Pascal (48 KiB) must step down.
        assert_eq!(select_flash_tile_keys(65_536), 16);
        assert_eq!(select_flash_tile_keys(49_152), 16);
        // Anything smaller still yields a usable, documented tile.
        assert_eq!(select_flash_tile_keys(32_768), 8);
        assert_eq!(select_flash_tile_keys(0), 8);
        for optin in [99_328usize, 98_304, 65_536, 49_152, 32_768] {
            let tile = select_flash_tile_keys(optin);
            assert!(
                flash_shared_bytes(tile) <= optin,
                "tile {tile} needs {} B but the device allows {optin} B",
                flash_shared_bytes(tile)
            );
        }
    }

    /// **F5 acceptance (hardware-free).** The launcher's shared-memory request
    /// must never exceed what `CudaGraph::new` opted into, for every device
    /// class the finding names crossed with every `head_dim` the two builds
    /// serve.
    ///
    /// This is the defect: the launcher used to compute `32 * head_dim * 2 * 4`
    /// unconditionally, so on a 64 KiB-optin Turing (where the negotiation had
    /// stepped down to a 16-key tile and opted into 49 152 B) a `head_dim = 384`
    /// launch asked for 98 304 B and died with `CUDA_ERROR_INVALID_VALUE`.
    /// Row 2 below is exactly that case.
    #[test]
    fn flash_launch_request_never_exceeds_the_granted_budget() {
        // (device optin max, label)
        let devices: [(usize, &str); 4] = [
            (49_152, "Pascal SM 6.x — 48 KiB"),
            (65_536, "Turing SM 7.5 (T4, RTX 20xx) — 64 KiB"),
            (98_304, "Volta SM 7.0 — 96 KiB"),
            (167_936, "Ampere/Ada/Hopper — 164 KiB"),
        ];
        for (optin_max, label) in devices {
            let (tile_keys, granted) = negotiate_flash_large(optin_max);
            assert!(
                granted <= optin_max,
                "{label}: opted into {granted} B of a {optin_max} B budget"
            );
            let max_head_dim = flash_large_max_head_dim(granted, tile_keys);
            for head_dim in [64usize, 128, 256, 384] {
                assert!(
                    flash_head_dim_supported(head_dim, max_head_dim),
                    "{label}: head_dim {head_dim} must be servable (max {max_head_dim})"
                );
                let launch_tile = flash_launch_tile_keys(head_dim, tile_keys);
                let requested = flash_launch_shared_bytes(launch_tile, head_dim);
                let budget = if head_dim <= FLASH_LEAN_HEAD_DIM_CAP {
                    // The lean build takes no opt-in: its ceiling is the 48 KiB
                    // every device grants a block by default.
                    49_152
                } else {
                    granted
                };
                assert!(
                    requested <= budget,
                    "{label}: head_dim {head_dim} requests {requested} B \
                     at tile {launch_tile} but only {budget} B are available"
                );
            }
        }
    }

    /// F5: across the four device classes above the negotiation always ends up
    /// able to serve the full 384 head dim, so the pre-launch guard rejects
    /// *nothing* there — asserted explicitly so the row above is not misread as
    /// covering rejection. The rejecting cases are the next test.
    #[test]
    fn documented_devices_all_reach_the_full_head_dim_cap() {
        for optin_max in [49_152usize, 65_536, 98_304, 167_936] {
            let (tile_keys, granted) = negotiate_flash_large(optin_max);
            assert_eq!(
                flash_large_max_head_dim(granted, tile_keys),
                FLASH_HEAD_DIM_CAP,
                "optin {optin_max} should still reach the full {FLASH_HEAD_DIM_CAP} head dim"
            );
        }
    }

    /// F5: the pre-launch `head_dim` guard must reject **exactly** the
    /// combinations the device cannot serve — the case
    /// `joint_attn_flash_validate` now checks before touching the GPU.
    ///
    /// `granted == 0` is the device that refused the opt-in outright: the wide
    /// build is unusable, every `head_dim > 128` must be refused, and the lean
    /// build (`head_dim <= 128`) must keep working — the whole point of F5 is
    /// that one unusable kernel no longer takes the backend down.
    #[test]
    fn head_dim_guard_rejects_exactly_the_unsupported_combinations() {
        // (granted_shared, tile_keys, expected max_head_dim, servable head dims)
        let cases: [(usize, usize, usize, &[usize]); 4] = [
            // Opt-in refused: only the lean build survives.
            (0, 16, 0, &[64, 128]),
            // Partial grant at the 32 tile: 16 384 / (32*8) = 64.
            (16_384, 32, 64, &[64, 128]),
            // Partial grant at the 8 tile: 16 384 / (8*8) = 256.
            (16_384, 8, 256, &[64, 128, 256]),
            // Full grant: everything the kernel caps at.
            (49_152, 16, 384, &[64, 128, 256, 384]),
        ];
        for (granted, tile_keys, expected_max, servable) in cases {
            let max_head_dim = flash_large_max_head_dim(granted, tile_keys);
            assert_eq!(
                max_head_dim, expected_max,
                "granted {granted} B at tile {tile_keys}"
            );
            for head_dim in [64usize, 128, 256, 384] {
                let expected = servable.contains(&head_dim);
                assert_eq!(
                    flash_head_dim_supported(head_dim, max_head_dim),
                    expected,
                    "granted {granted} B at tile {tile_keys}: head_dim {head_dim} \
                     should be {}",
                    if expected { "servable" } else { "refused" }
                );
                if expected && head_dim > FLASH_LEAN_HEAD_DIM_CAP {
                    assert!(
                        flash_launch_shared_bytes(tile_keys, head_dim) <= granted,
                        "granted {granted} B at tile {tile_keys}: head_dim \
                         {head_dim} is accepted but does not fit"
                    );
                }
            }
        }
        // A zero tile cannot stage anything; guard against the division.
        assert_eq!(flash_large_max_head_dim(98_304, 0), 0);
    }

    /// F5: the launcher picks the lean build's fixed tile below the lean cap and
    /// the device-negotiated tile above it — the dispatch the shared-memory size
    /// must agree with.
    #[test]
    fn launch_tile_follows_the_dispatched_build() {
        for large_tile_keys in FLASH_TILE_KEYS {
            for head_dim in [8usize, 64, 128] {
                assert_eq!(
                    flash_launch_tile_keys(head_dim, large_tile_keys),
                    FLASH_LEAN_TILE_KEYS,
                    "head_dim {head_dim} uses the lean build"
                );
            }
            for head_dim in [136usize, 256, 384] {
                assert_eq!(
                    flash_launch_tile_keys(head_dim, large_tile_keys),
                    large_tile_keys,
                    "head_dim {head_dim} uses the wide build"
                );
            }
        }
        // The lean build's own worst case stays inside the no-opt-in 48 KiB.
        assert_eq!(
            flash_launch_shared_bytes(FLASH_LEAN_TILE_KEYS, FLASH_LEAN_HEAD_DIM_CAP),
            32_768
        );
        // `flash_shared_bytes` is the wide build's worst case — the same
        // arithmetic at the full head-dim cap, kept as one source of truth.
        for tile_keys in FLASH_TILE_KEYS {
            assert_eq!(
                flash_shared_bytes(tile_keys),
                flash_launch_shared_bytes(tile_keys, FLASH_HEAD_DIM_CAP)
            );
        }
    }

    /// F-M4: table-tests the parsing `selected_device_defaults_to_zero`
    /// (`cudagraph_global_group.rs`) could not, since that test only checks
    /// the memoised `OnceLock`'s output shape ("only the shape is checked
    /// here" — its own comment) and is itself dead on this host. Every case
    /// the finding named: empty, a plain ordinal, one with surrounding
    /// whitespace, non-numeric garbage, and a negative number (not a valid
    /// `usize`).
    #[test]
    fn parse_device_ordinal_table() {
        let cases: [(&str, usize); 5] = [("", 0), ("3", 3), (" 3 ", 3), ("abc", 0), ("-1", 0)];
        for (raw, expected) in cases {
            assert_eq!(
                parse_device_ordinal(raw),
                expected,
                "parse_device_ordinal({raw:?}) should be {expected}"
            );
        }
    }
}
