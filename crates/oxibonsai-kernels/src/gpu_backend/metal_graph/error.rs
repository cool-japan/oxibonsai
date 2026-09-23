//! Error type and GPU-resident weight handle for the Metal graph dispatch engine.

use metal::{Buffer, MTLCommandBufferStatus};
use std::fmt;

use crate::gpu_backend::metal_full_layer::types::{WeightKind, WEIGHT_KIND_MISMATCH_TAG};

// ═══════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════

/// Errors raised by the Metal graph dispatch engine.
///
/// `Clone` is derived (every payload is `String`, `Option<String>`,
/// `&'static str`, `MTLCommandBufferStatus` or [`WeightKind`], all of which
/// are `Clone`) so a caller that must both log an error and return it — the
/// FP8/K-quant/Q-std kernel families all do — can simply `e.clone()`. Four
/// hand-written, exhaustive `clone_err` matches used to do that, and every
/// new variant broke all four files.
#[derive(Debug, Clone)]
pub enum MetalGraphError {
    /// No Metal-capable GPU device was found on the system.
    DeviceNotFound,
    /// MSL shader compilation failed.
    CompilationFailed(String),
    /// A GPU buffer could not be allocated.
    BufferCreationFailed,
    /// An encoding operation failed (pipeline not found, etc.).
    EncodingFailed(String),
    /// A command buffer execution failed or timed out.
    ExecutionFailed(String),
    /// Supplied dimensions or buffer lengths are inconsistent (e.g. `k` not a
    /// multiple of 128, or a slice length mismatching `m*k` / `m*n_rows`).
    InvalidDimensions(String),
    /// A committed command buffer finished with a non-`Completed` status
    /// (`Error`, or still `Committed`/`Scheduled` if waited on incorrectly)
    /// instead of running to completion.
    ///
    /// Surfacing this — rather than blindly downloading whatever bytes
    /// happen to be in the output buffer and returning `Ok(())` — is the
    /// MET-04 fix: on a GPU fault the compute never ran, so the shared
    /// output buffer still holds the *previous* call's contents (or,
    /// for a freshly allocated buffer, undefined/zeroed bytes).
    CommandBufferFailed {
        /// Short static label identifying which command buffer failed
        /// (e.g. `"encode_full_layer"`, `"encode_tail_and_commit"`).
        what: &'static str,
        /// The raw `MTLCommandBufferStatus` (anything other than `Completed`).
        status: MTLCommandBufferStatus,
        /// `NSError.localizedDescription` read from `-[MTLCommandBuffer
        /// error]`, when the driver supplied one.
        error: Option<String>,
    },
    /// A requested Metal buffer allocation exceeds `MTLDevice::maxBufferLength`.
    ///
    /// `-[MTLDevice newBufferWithLength:options:]` returns `nil` above this
    /// limit (or under GPU memory pressure); the `metal` crate's
    /// `foreign_types` wrapper then treats that `nil` pointer as undefined
    /// behavior. Checking the requested size up front, before calling into
    /// `new_buffer`, turns that into an ordinary, named error (MET-06).
    BufferTooLarge {
        /// Short static label identifying the allocation path.
        what: &'static str,
        /// The requested allocation size, in bytes.
        requested: u64,
        /// `MTLDevice::maxBufferLength`, in bytes.
        max: u64,
    },
    /// A weight-cache slot already holds a buffer in a different on-GPU
    /// format than the one requested (MET-02).
    ///
    /// The SoA layouts are not interchangeable — handing a `Q1_0_g128` SoA
    /// buffer to the ternary GEMV decodes garbage — so the cache refuses the
    /// lookup instead of serving a stale hit. This is the typed form of what
    /// was previously an [`MetalGraphError::ExecutionFailed`] whose message
    /// merely *started* with [`WEIGHT_KIND_MISMATCH_TAG`], so callers can now
    /// discriminate the condition without string matching.
    WeightKindMismatch {
        /// The [`WeightKind`] the caller asked for.
        expected: WeightKind,
        /// The [`WeightKind`] already resident in that slot.
        found: WeightKind,
        /// Per-model weight identity of the colliding slot (`WeightKey::slot`)
        /// — *which* tensor collided, context the replaced string form
        /// carried and the typed variant originally dropped.
        slot: u64,
        /// The loaded-model epoch of the colliding key (`WeightKey::model_epoch`).
        model_epoch: u64,
    },
}

impl fmt::Display for MetalGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "no Metal-capable GPU device found"),
            Self::CompilationFailed(msg) => write!(f, "MSL compilation failed: {msg}"),
            Self::BufferCreationFailed => write!(f, "Metal buffer allocation failed"),
            Self::EncodingFailed(msg) => write!(f, "Metal encoding failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "Metal execution failed: {msg}"),
            Self::InvalidDimensions(msg) => write!(f, "Metal invalid dimensions: {msg}"),
            Self::CommandBufferFailed {
                what,
                status,
                error,
            } => match error {
                Some(msg) => write!(
                    f,
                    "Metal command buffer '{what}' did not complete: status={status:?}: {msg}"
                ),
                None => write!(
                    f,
                    "Metal command buffer '{what}' did not complete: status={status:?}"
                ),
            },
            Self::BufferTooLarge {
                what,
                requested,
                max,
            } => write!(
                f,
                "Metal buffer allocation '{what}' requested {requested} bytes, \
                 exceeding device max_buffer_length {max} bytes"
            ),
            // The message intentionally opens with `WEIGHT_KIND_MISMATCH_TAG`:
            // that substring is the stable contract callers and tests match on
            // (`metal_graph::graph::weight_cache`'s acceptance tests), and
            // interpolating the constant rather than re-typing the literal keeps
            // the two from drifting apart.
            Self::WeightKindMismatch {
                expected,
                found,
                slot,
                model_epoch,
            } => write!(
                f,
                "{WEIGHT_KIND_MISMATCH_TAG}: epoch {model_epoch} slot {slot} holds a {found} \
                 buffer but {expected} was requested"
            ),
        }
    }
}

impl std::error::Error for MetalGraphError {}

// ═══════════════════════════════════════════════════════════════════════════
// Weight handle
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque handle to a weight buffer already resident on the GPU.
///
/// Stores the raw `metal::Buffer` directly so the graph can bind it
/// without going through any abstraction layer.
pub struct MetalWeightHandle {
    /// Raw Metal buffer containing packed weight data.
    pub(crate) buffer: Buffer,
    /// Size in bytes.
    pub(crate) byte_len: usize,
    /// On-GPU layout of `buffer` (MET-02).
    ///
    /// Carried on the handle — not only on the cache key — so a handle that
    /// reaches the wrong kernel can self-identify instead of silently
    /// decoding one quantization's bytes with another's decoder. Every
    /// construction site in `graph.rs` already knows its kind.
    pub(crate) kind: WeightKind,
}

impl MetalWeightHandle {
    /// Size of the weight data in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// On-GPU layout of this buffer.
    ///
    /// A kernel that only accepts one layout can check this before binding,
    /// turning what used to be a silent garbage decode into a named error.
    pub fn kind(&self) -> WeightKind {
        self.kind
    }
}

impl fmt::Debug for MetalWeightHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalWeightHandle")
            .field("byte_len", &self.byte_len)
            .field("kind", &self.kind)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — MET-04 (checked command buffers) and MET-02 (typed kind mismatch)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::metal_full_layer::types::{next_model_epoch, WeightKey};
    use crate::gpu_backend::metal_graph::MetalGraph;
    use metal::Device;

    /// `n_blocks` valid ternary (qs-first) 34-byte `TQ2_0_g128` blocks: scale
    /// `1.0`, 2-bit codes cycling through `00/01/10` only (never the reserved
    /// `0b11`, which the upload validator rejects for a ternary tensor).
    fn tq2_blocks(n_blocks: usize) -> Vec<u8> {
        tq2_block_structs(n_blocks)
            .iter()
            .flat_map(tq2_block_bytes)
            .collect()
    }

    /// The same blocks as [`tq2_blocks`], but **typed**.
    ///
    /// `O4`: the scalar-reference call sites used to cast the `Vec<u8>` above
    /// to `*const BlockTQ2_0_g128`. `BlockTQ2_0_g128` has alignment 2 (its
    /// `d: f16`) while a `Vec<u8>` is only guaranteed 1-aligned, so that cast
    /// was UB by the letter — it worked only because the system allocator
    /// over-aligns, and Miri flags it. Building the typed vector first and
    /// deriving the bytes from it inverts the direction and removes the cast.
    fn tq2_block_structs(n_blocks: usize) -> Vec<oxibonsai_core::BlockTQ2_0_g128> {
        (0..n_blocks)
            .map(|i| {
                let mut qs = [0u8; 32];
                for (j, byte) in qs.iter_mut().enumerate() {
                    let c = |s: usize| ((i + j + s) % 3) as u8;
                    *byte = c(0) | (c(1) << 2) | (c(2) << 4) | (c(3) << 6);
                }
                oxibonsai_core::BlockTQ2_0_g128 {
                    qs,
                    // 0x3C00 == 1.0 in IEEE binary16.
                    d: half::f16::from_bits(0x3C00),
                }
            })
            .collect()
    }

    /// The on-disk 34 bytes of one block: `[qs 32 B][d f16 LE]`.
    fn tq2_block_bytes(block: &oxibonsai_core::BlockTQ2_0_g128) -> Vec<u8> {
        let mut out = Vec::with_capacity(34);
        out.extend_from_slice(&block.qs);
        out.extend_from_slice(&block.d.to_bits().to_le_bytes());
        out
    }

    /// `n_blocks` 18-byte `Q1_0_g128` blocks (any byte pattern is a valid Q1
    /// block — the format has no reserved codes).
    fn q1_blocks(n_blocks: usize) -> Vec<u8> {
        (0..n_blocks * 18).map(|i| (i % 251) as u8).collect()
    }

    // ── MET-04 ──────────────────────────────────────────────────────────

    /// MET-04 regression guard for `graph.rs` specifically.
    ///
    /// The first MET-04 sweep left eight raw `commit()` /
    /// `wait_until_completed()` pairs behind in `graph.rs` — every one of them
    /// a dispatch whose GPU fault was ignored before the caller downloaded and
    /// returned the output buffer as `Ok`. A real hardware fault is not
    /// reproducible on Apple Silicon (see `buffers.rs`'s own tests, where an
    /// oversized threadgroup and a ~4 GB out-of-bounds write both completed
    /// cleanly), so the *behavioural* half is covered there by
    /// `map_command_buffer_status_error_is_err_not_ok`. What that cannot catch
    /// is a dispatch site that never routes through the checked helper at all
    /// — which is exactly how these eight survived the first sweep. This pins
    /// the structural half: no unchecked wait, and no bare commit.
    #[test]
    fn graph_rs_has_no_unchecked_command_buffer_commit() {
        let src = include_str!("graph.rs");
        assert!(
            !src.contains("wait_until_completed"),
            "graph.rs must not call wait_until_completed() directly — every \
             commit/wait pair goes through buffers::commit_and_wait so a \
             non-Completed status becomes MetalGraphError::CommandBufferFailed \
             instead of a silently stale readback (MET-04)"
        );
        for (lineno, line) in src.lines().enumerate() {
            let code = line.trim_start();
            assert!(
                !(code.starts_with("cmd_buf.commit()") || code.starts_with("cmd.commit()")),
                "graph.rs:{}: bare commit() — use commit_and_wait(cmd, \"<site>\")? (MET-04)",
                lineno + 1
            );
        }
        // And the helper really is in use. `>=`, not `==`: the eight sites
        // MET-04 missed must all be routed, but a later package adding a
        // ninth *checked* dispatch here is correct work, not a regression.
        assert!(
            src.matches("commit_and_wait(cmd_buf, ").count() >= 8,
            "expected at least the 8 known graph.rs dispatch sites to call \
             commit_and_wait"
        );
    }

    /// MET-04 happy path through one of the eight rewritten sites: a real
    /// ternary GEMV still completes and still returns the GPU's data, i.e. the
    /// added status check does not reject a normal submission and the `?` on
    /// `commit_and_wait` did not truncate the readback.
    #[test]
    fn checked_commit_still_returns_gpu_results_from_encode_gemv_tq2() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");

        let n_rows = 8usize;
        let k = 256usize;
        let blocks_per_row = k / 128;
        // `O4`: build the typed blocks first and derive the byte blob from
        // them, rather than casting a 1-aligned `Vec<u8>` to a 2-aligned
        // `*const BlockTQ2_0_g128`.
        let blocks = tq2_block_structs(n_rows * blocks_per_row);
        let aos: Vec<u8> = blocks.iter().flat_map(tq2_block_bytes).collect();
        let handle = graph
            .upload_tq2_weight_soa(&aos)
            .expect("upload_tq2_weight_soa");
        assert_eq!(handle.kind(), WeightKind::Tq2Soa);

        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // Scalar reference over the same blocks.
        let mut expected = vec![0f32; n_rows];
        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut expected, n_rows, k)
            .expect("scalar reference GEMV");

        let mut got = vec![0f32; n_rows];
        graph
            .encode_gemv_tq2(&handle, &input, &mut got, n_rows, k)
            .expect("encode_gemv_tq2 must succeed on a healthy device");

        for (i, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "row {i}: expected {a}, got {b} — a checked commit must still \
                 hand back the GPU's own output"
            );
        }
        // A non-zero result rules out "returned the zeroed output buffer".
        assert!(
            got.iter().any(|v| v.abs() > 1e-6),
            "degenerate fixture: every output row is zero, so this could not \
             distinguish real data from a stale/zeroed buffer"
        );
    }

    /// `CommandBufferFailed` names the failing site in its `Display`, which is
    /// the whole point of threading a distinct `what` through each of the eight
    /// `graph.rs` dispatches.
    #[test]
    fn command_buffer_failed_display_names_site_and_status() {
        let err = MetalGraphError::CommandBufferFailed {
            what: "encode_ffn_phase",
            status: MTLCommandBufferStatus::Error,
            error: Some("Insufficient Memory".to_string()),
        };
        let msg = err.to_string();
        assert!(msg.contains("encode_ffn_phase"), "{msg}");
        assert!(msg.contains("Error"), "{msg}");
        assert!(msg.contains("Insufficient Memory"), "{msg}");

        let no_desc = MetalGraphError::CommandBufferFailed {
            what: "encode_gemv_tq2",
            status: MTLCommandBufferStatus::Error,
            error: None,
        };
        assert!(no_desc.to_string().contains("encode_gemv_tq2"));
    }

    // ── MET-02: typed weight-kind mismatch ──────────────────────────────

    /// The `Display` of the typed variant keeps the stable tag every existing
    /// caller and test greps for, and names both kinds.
    #[test]
    fn weight_kind_mismatch_display_keeps_the_stable_tag() {
        let err = MetalGraphError::WeightKindMismatch {
            expected: WeightKind::Tq2Soa,
            found: WeightKind::Q1Soa,
            slot: 0x2a,
            model_epoch: 7,
        };
        let msg = err.to_string();
        assert!(msg.starts_with(WEIGHT_KIND_MISMATCH_TAG), "{msg}");
        assert!(msg.contains("q1_soa"), "{msg}");
        assert!(msg.contains("tq2_soa"), "{msg}");
        assert_eq!(
            msg,
            "weight cache kind mismatch: epoch 7 slot 42 holds a q1_soa buffer but tq2_soa was \
             requested"
        );
    }

    /// MET-02 acceptance, now **typed**: inserting slot K as `Q1Soa` and
    /// requesting it as `Tq2Soa` must yield
    /// [`MetalGraphError::WeightKindMismatch`] — not a stale hit, and not an
    /// untyped `ExecutionFailed` a caller can only discriminate by string.
    #[test]
    fn requesting_a_q1_slot_as_tq2_yields_the_typed_mismatch_error() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let epoch = next_model_epoch();
        let slot = 2_000_000u64;

        let q1 = graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Q1Soa, slot), || {
                graph.upload_q1_weight_soa(&q1_blocks(4))
            })
            .expect("Q1 upload");
        assert_eq!(q1.kind(), WeightKind::Q1Soa);

        match graph.get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Tq2Soa, slot), || {
            graph.upload_tq2_weight_soa(&tq2_blocks(4))
        }) {
            Err(MetalGraphError::WeightKindMismatch {
                expected,
                found,
                slot: reported_slot,
                model_epoch,
            }) => {
                assert_eq!(expected, WeightKind::Tq2Soa);
                assert_eq!(found, WeightKind::Q1Soa);
                // `O5`: the typed variant now names *which* tensor collided,
                // context that previously survived only in the log line.
                assert_eq!(reported_slot, slot);
                assert_eq!(model_epoch, epoch);
            }
            Ok(_) => panic!("a Q1 buffer must never be served to the ternary path"),
            Err(other) => panic!("expected WeightKindMismatch, got {other:?}"),
        }

        graph.release_model(epoch).expect("release epoch");
    }

    /// Every upload path stamps the handle with its own on-GPU layout, so a
    /// handle that reaches the wrong kernel can be identified.
    #[test]
    fn upload_paths_stamp_their_weight_kind_on_the_handle() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");

        assert_eq!(
            graph
                .upload_weight(&[0u8; 64])
                .expect("raw f32 upload")
                .kind(),
            WeightKind::RawF32
        );
        assert_eq!(
            graph
                .upload_q1_weight_soa(&q1_blocks(2))
                .expect("q1 upload")
                .kind(),
            WeightKind::Q1Soa
        );
        assert_eq!(
            graph
                .upload_tq2_weight_soa(&tq2_blocks(2))
                .expect("tq2 upload")
                .kind(),
            WeightKind::Tq2Soa
        );
        // PQ2 shares the 34-byte shape but decodes `0b11` as `+2`, so the same
        // bytes are a different kind.
        assert_eq!(
            graph
                .upload_pq2_weight_soa(&tq2_blocks(2))
                .expect("pq2 upload")
                .kind(),
            WeightKind::Pq2Soa
        );
    }
}
