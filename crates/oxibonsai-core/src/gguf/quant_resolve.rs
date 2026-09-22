//! Settles the three on-disk readings of ggml tensor type id **42**.
//!
//! Three mutually incompatible layouts ship under one id:
//!
//! | file family | group | bytes | byte order | `quantization_version` |
//! |---|---|---|---|---|
//! | `Ternary-Bonsai-{1.7B,8B}` (OxiBonsai's own writer) | 128 | 34 | `qs` first | **string** `"TQ2_0_G128"` |
//! | `Ternary-Bonsai-27B-Q2_0` (PrismML gen-1) | 128 | 34 | `d` first | `u32 2` |
//! | `Ternary-Bonsai-2-27B-Q2_0` (mainline `block_q2_0`) | 64 | 18 | `d` first | `u32 2` |
//!
//! `general.file_type` is **not** a discriminator — the gen-1 g128 file and
//! the gen-2 g64 file both report 41.
//!
//! # How the group size is settled
//!
//! Not by `offset[i+1] - offset[i]`: GGUF tensor offsets are **padded**
//! (`offset[i] = Σ_{j<i} GGML_PAD(ggml_nbytes(j), alignment)`,
//! `ggml/src/gguf.cpp:780-793`), so that delta is an upper bound, not the
//! size, and an equality test on it rejects every valid file.
//!
//! Instead this replays llama.cpp's own invariant for each candidate
//! geometry: walk the tensors in offset order accumulating
//! `GGML_PAD(nbytes_i, alignment)` — with `nbytes` computed **per row**
//! (`nrows * ceil(ne0/blck) * type_size`) — and keep the candidates that
//! reproduce every declared offset exactly. Measured on the real files, the
//! two candidates differ by hundreds of megabytes, so exactly one survives.
//!
//! An unpadded running sum is accepted as a second size model because
//! OxiBonsai's own writer historically laid tensor data back-to-back
//! (core-gguf-10); which model matched is reported in
//! [`Resolved42::size_model`].
//!
//! # How the byte order is settled
//!
//! It **cannot** be settled by offsets — `d`-first and `qs`-first give
//! byte-identical sizes. The structural data assertion of
//! [`crate::quant_ternary::sniff_two_bit_layout`] (no reserved `0b11` code in
//! a ternary checkpoint, finite non-negative FP16 scale) is therefore the
//! only discriminator, and is mandatory: [`resolve_type_42`] resolves the
//! order only for the legacy string tag (which nothing but OxiBonsai emits)
//! and otherwise returns [`BonsaiError::AmbiguousQuantType`] naming
//! [`resolve_type_42_with_sample`]. It never guesses.

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::metadata::{MetadataStore, MetadataValue};
use crate::gguf::tensor_info::{padded_size, row_size_bytes, TensorInfo};
use crate::gguf::types::GgufTensorType;
use crate::quant_ternary::{sniff_two_bit_layout, TwoBitLayout, SNIFF_DEFAULT_BLOCKS};

/// The ggml type id this module disambiguates.
pub const AMBIGUOUS_TYPE_ID: u32 = 42;

/// The legacy `general.quantization_version` string OxiBonsai's own writer
/// emits. Every PrismML file stores `u32 2` there instead.
pub const LEGACY_QVER_STRING: &str = "TQ2_0_G128";

/// Which spelling of `general.quantization_version` the file carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LegacyVersionTag {
    /// The **string** `"TQ2_0_G128"` — written only by OxiBonsai's own
    /// converter, so it is definitive evidence for the legacy qs-first g128
    /// layout.
    Tq2_0G128,
    /// A numeric version (`2`, …) or the key is absent — carries no ordering
    /// information at all.
    #[default]
    Numeric,
}

impl LegacyVersionTag {
    /// Classify a `general.quantization_version` value.
    pub fn from_value(value: Option<&MetadataValue>) -> Self {
        match value.and_then(|v| v.as_str()) {
            Some(s) if s.eq_ignore_ascii_case(LEGACY_QVER_STRING) => Self::Tq2_0G128,
            _ => Self::Numeric,
        }
    }

    /// Classify straight from a metadata store.
    pub fn from_metadata(md: &MetadataStore) -> Self {
        Self::from_value(md.get("general.quantization_version"))
    }
}

/// Which running-sum model reproduced the file's offset table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeModel {
    /// `offset[i] == Σ_{j<i} GGML_PAD(nbytes_j, alignment)` — what ggml
    /// writes and what llama.cpp validates.
    Padded,
    /// `offset[i] == Σ_{j<i} nbytes_j` — no inter-tensor padding. Produced by
    /// OxiBonsai's writer before core-gguf-10; llama.cpp rejects such files.
    Unpadded,
    /// Both models agree (every tensor's size is already a multiple of the
    /// alignment), which is the common case for block-aligned models.
    Both,
}

/// What settled the `d`-first vs `qs`-first question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderEvidence {
    /// The group size is 64; mainline `block_q2_0` has only one byte order.
    NotApplicable,
    /// `general.quantization_version` is the legacy string `"TQ2_0_G128"`.
    LegacyVersionString,
    /// The structural sniff over real block bytes.
    DataSniff,
}

/// The settled reading of ggml type id 42 for one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved42 {
    /// The resolved variant. Its [`GgufTensorType::wire_id`] is still 42.
    pub tensor_type: GgufTensorType,
    /// Elements per block under the resolved reading.
    pub block_size: usize,
    /// Bytes per block under the resolved reading.
    pub block_bytes: usize,
    /// Which running-sum model reproduced the offsets.
    pub size_model: SizeModel,
    /// What settled the byte order.
    pub order_evidence: OrderEvidence,
    /// The `general.quantization_version` spelling seen in the file.
    pub legacy_version_tag: LegacyVersionTag,
}

/// The two geometries id 42 can have. The byte order is *not* part of this —
/// it produces identical sizes and must come from the data.
const GROUP_CANDIDATES: [(usize, usize); 2] = [(128, 34), (64, 18)];

/// Per-candidate outcome of the offset replay.
#[derive(Debug, Clone, Copy)]
struct ReplayOutcome {
    block_size: usize,
    block_bytes: usize,
    padded_ok: bool,
    unpadded_ok: bool,
}

impl ReplayOutcome {
    fn survives(&self) -> bool {
        self.padded_ok || self.unpadded_ok
    }

    fn size_model(&self) -> SizeModel {
        match (self.padded_ok, self.unpadded_ok) {
            (true, true) => SizeModel::Both,
            (true, false) => SizeModel::Padded,
            _ => SizeModel::Unpadded,
        }
    }
}

/// Build the per-tensor byte extents [`resolve_type_42`] expects.
///
/// `extents[i]` is an upper bound on the bytes available to `tensors[i]`:
/// the gap to the next tensor in offset order, or `data_len - offset` for the
/// last one. The result is indexed like `tensors`, not like the sorted order.
pub fn compute_extents(tensors: &[TensorInfo], data_len: u64) -> BonsaiResult<Vec<u64>> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| {
        tensors[a]
            .offset
            .cmp(&tensors[b].offset)
            .then_with(|| tensors[a].name.cmp(&tensors[b].name))
    });
    let mut extents = vec![0u64; tensors.len()];
    for (pos, &idx) in order.iter().enumerate() {
        let start = tensors[idx].offset;
        let end = match order.get(pos + 1) {
            Some(&next) => tensors[next].offset,
            None => data_len,
        };
        extents[idx] = end.checked_sub(start).ok_or_else(|| {
            BonsaiError::tensor_layout(
                tensors[idx].name.clone(),
                format!("offset {start} exceeds the end of the tensor data section ({end})"),
            )
        })?;
    }
    Ok(extents)
}

/// Replay llama.cpp's offset invariant for one candidate geometry.
fn replay_candidate(
    tensors: &[TensorInfo],
    extents: &[u64],
    alignment: u64,
    block_size: usize,
    block_bytes: usize,
) -> Option<ReplayOutcome> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&a, &b| {
        tensors[a]
            .offset
            .cmp(&tensors[b].offset)
            .then_with(|| tensors[a].name.cmp(&tensors[b].name))
    });

    let mut running_padded = 0u64;
    let mut running_unpadded = 0u64;
    let mut padded_ok = true;
    let mut unpadded_ok = true;

    for &idx in &order {
        let info = &tensors[idx];
        let nbytes = if info.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID {
            let ne0 = info.ne0();
            if !ne0.is_multiple_of(block_size as u64) {
                // This candidate cannot even describe the tensor's rows.
                return None;
            }
            let nrows = info
                .shape
                .iter()
                .skip(1)
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
            nrows
                .checked_mul(ne0 / block_size as u64)?
                .checked_mul(block_bytes as u64)?
        } else {
            row_size_bytes(info.tensor_type, &info.shape)
        };

        // The declared gap to the next tensor is an upper bound on the size.
        if nbytes > extents[idx] {
            return None;
        }
        if info.offset != running_padded {
            padded_ok = false;
        }
        if info.offset != running_unpadded {
            unpadded_ok = false;
        }
        if !padded_ok && !unpadded_ok {
            return None;
        }
        running_padded = running_padded.checked_add(padded_size(nbytes, alignment)?)?;
        running_unpadded = running_unpadded.checked_add(nbytes)?;
    }

    Some(ReplayOutcome {
        block_size,
        block_bytes,
        padded_ok,
        unpadded_ok,
    })
}

/// Common front half of both entry points: validate the inputs and settle the
/// group size.
fn settle_group(
    tensors: &[TensorInfo],
    alignment: u64,
    extents: Option<&[u64]>,
) -> BonsaiResult<ReplayOutcome> {
    let extents = extents.ok_or_else(|| BonsaiError::AmbiguousQuantType {
        type_id: AMBIGUOUS_TYPE_ID,
        hint: "no per-tensor byte extents were supplied, so the offset invariant cannot be \
               replayed; build them with quant_resolve::compute_extents() — resolving id 42 \
               without evidence is never allowed"
            .to_string(),
    })?;

    if extents.len() != tensors.len() {
        return Err(BonsaiError::AmbiguousQuantType {
            type_id: AMBIGUOUS_TYPE_ID,
            hint: format!(
                "extents has {} entries for {} tensors; they must be index-matched",
                extents.len(),
                tensors.len()
            ),
        });
    }

    let first_42 = tensors
        .iter()
        .find(|t| t.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
        .ok_or_else(|| BonsaiError::AmbiguousQuantType {
            type_id: AMBIGUOUS_TYPE_ID,
            hint: "the tensor list contains no ggml type-42 tensor to resolve".to_string(),
        })?;

    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(BonsaiError::AlignmentError {
            expected: alignment as usize,
            offset: 0,
        });
    }

    let survivors: Vec<ReplayOutcome> = GROUP_CANDIDATES
        .iter()
        .filter_map(|&(bs, bb)| replay_candidate(tensors, extents, alignment, bs, bb))
        .filter(ReplayOutcome::survives)
        .collect();

    match survivors.len() {
        1 => Ok(survivors[0]),
        0 => Err(BonsaiError::tensor_layout(
            first_42.name.clone(),
            format!(
                "no ggml type-42 group size reproduces the file's offset table at alignment \
                 {alignment}: tried group 128 / 34 B and group 64 / 18 B, both padded and \
                 unpadded running sums"
            ),
        )),
        n => Err(BonsaiError::AmbiguousQuantType {
            type_id: AMBIGUOUS_TYPE_ID,
            hint: format!(
                "{n} group-size candidates reproduce the offset table; the file does not \
                 determine its own layout"
            ),
        }),
    }
}

/// Settle ggml type id 42 from the file's offset table and metadata alone.
///
/// Resolves the group size by replaying llama.cpp's offset invariant (see the
/// module docs). The 34-byte reading additionally needs a byte order, which
/// offsets cannot supply:
///
/// * `general.quantization_version == "TQ2_0_G128"` (the string form, which
///   only OxiBonsai's own writer emits) settles it as the legacy qs-first
///   [`GgufTensorType::TQ2_0_g128`];
/// * otherwise this returns [`BonsaiError::AmbiguousQuantType`] pointing at
///   [`resolve_type_42_with_sample`]. It never falls back to a default.
///
/// `extents` is mandatory — see [`compute_extents`]. A call with `None`
/// errors rather than guessing.
pub fn resolve_type_42(
    tensors: &[TensorInfo],
    alignment: u64,
    qver: Option<&MetadataValue>,
    extents: Option<&[u64]>,
) -> BonsaiResult<Resolved42> {
    let outcome = settle_group(tensors, alignment, extents)?;
    let tag = LegacyVersionTag::from_value(qver);

    if outcome.block_size == 64 {
        return Ok(Resolved42 {
            tensor_type: GgufTensorType::Q2_0G64,
            block_size: outcome.block_size,
            block_bytes: outcome.block_bytes,
            size_model: outcome.size_model(),
            order_evidence: OrderEvidence::NotApplicable,
            legacy_version_tag: tag,
        });
    }

    match tag {
        LegacyVersionTag::Tq2_0G128 => Ok(Resolved42 {
            tensor_type: GgufTensorType::TQ2_0_g128,
            block_size: outcome.block_size,
            block_bytes: outcome.block_bytes,
            size_model: outcome.size_model(),
            order_evidence: OrderEvidence::LegacyVersionString,
            legacy_version_tag: tag,
        }),
        LegacyVersionTag::Numeric => Err(BonsaiError::AmbiguousQuantType {
            type_id: AMBIGUOUS_TYPE_ID,
            hint: "the offset table settles group 128 / 34 B, but d-first and qs-first have \
                   identical sizes and general.quantization_version is numeric; call \
                   resolve_type_42_with_sample() with the raw bytes of a type-42 tensor so the \
                   0b11-code / scale-sanity assertion can decide"
                .to_string(),
        }),
    }
}

/// Settle ggml type id 42 using the offset table **and** the real bytes of a
/// type-42 tensor.
///
/// `sample` is the raw data of one declared-ternary type-42 tensor (the first
/// quantized tensor is the conventional choice). The structural sniff is the
/// only thing that can separate `d`-first from `qs`-first, so this is the
/// entry point a loader should use.
///
/// A sniff verdict that contradicts the offset replay is a hard
/// [`BonsaiError::QuantLayoutMismatch`], not a warning.
pub fn resolve_type_42_with_sample(
    tensors: &[TensorInfo],
    alignment: u64,
    qver: Option<&MetadataValue>,
    extents: Option<&[u64]>,
    sample: &[u8],
) -> BonsaiResult<Resolved42> {
    let outcome = settle_group(tensors, alignment, extents)?;
    let tag = LegacyVersionTag::from_value(qver);
    let sample_name = tensors
        .iter()
        .find(|t| t.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
        .map(|t| t.name.clone())
        .unwrap_or_else(|| "<type-42 tensor>".to_string());

    let verdict = sniff_two_bit_layout(sample, SNIFF_DEFAULT_BLOCKS);

    if outcome.block_size == 64 {
        // The replay says group 64; the data must not say group 128.
        if matches!(verdict, TwoBitLayout::DFirst34 | TwoBitLayout::QsFirst34) {
            return Err(BonsaiError::QuantLayoutMismatch {
                tensor: sample_name,
                assumed: TwoBitLayout::DFirst18.name().to_string(),
                suggestion: verdict.name().to_string(),
            });
        }
        return Ok(Resolved42 {
            tensor_type: GgufTensorType::Q2_0G64,
            block_size: outcome.block_size,
            block_bytes: outcome.block_bytes,
            size_model: outcome.size_model(),
            order_evidence: OrderEvidence::NotApplicable,
            legacy_version_tag: tag,
        });
    }

    let (tensor_type, order_evidence) = match verdict {
        TwoBitLayout::QsFirst34 => (GgufTensorType::TQ2_0_g128, OrderEvidence::DataSniff),
        TwoBitLayout::DFirst34 => {
            if tag == LegacyVersionTag::Tq2_0G128 {
                // The file claims to be ours but its bytes are d-first.
                return Err(BonsaiError::QuantLayoutMismatch {
                    tensor: sample_name,
                    assumed: TwoBitLayout::QsFirst34.name().to_string(),
                    suggestion: TwoBitLayout::DFirst34.name().to_string(),
                });
            }
            (GgufTensorType::Q2_0G128DFirst, OrderEvidence::DataSniff)
        }
        TwoBitLayout::DFirst18 => {
            return Err(BonsaiError::QuantLayoutMismatch {
                tensor: sample_name,
                assumed: "group-128/34B (from the offset table)".to_string(),
                suggestion: TwoBitLayout::DFirst18.name().to_string(),
            });
        }
        TwoBitLayout::Ambiguous => match tag {
            // An all-zero or too-small sample is genuinely undecidable; the
            // legacy string tag is then the only remaining evidence.
            LegacyVersionTag::Tq2_0G128 => (
                GgufTensorType::TQ2_0_g128,
                OrderEvidence::LegacyVersionString,
            ),
            LegacyVersionTag::Numeric => {
                return Err(BonsaiError::AmbiguousQuantType {
                    type_id: AMBIGUOUS_TYPE_ID,
                    hint: format!(
                        "the {sample_name} sample is structurally clean under more than one \
                         byte order (typically an all-zero or very small tensor) and \
                         general.quantization_version is numeric; supply a larger sample from a \
                         real weight tensor"
                    ),
                })
            }
        },
    };

    Ok(Resolved42 {
        tensor_type,
        block_size: outcome.block_size,
        block_bytes: outcome.block_bytes,
        size_model: outcome.size_model(),
        order_evidence,
        legacy_version_tag: tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tensor_info::align_up;

    fn info(name: &str, shape: &[u64], ty: GgufTensorType, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape: shape.to_vec(),
            tensor_type: ty,
            offset,
        }
    }

    /// Lay out `specs` as a real GGUF would: offsets are the running padded
    /// sum of the per-row sizes under `(block_size, block_bytes)` for id-42
    /// tensors.
    fn lay_out(
        specs: &[(&str, Vec<u64>, GgufTensorType)],
        alignment: u64,
        g42: (usize, usize),
        padded: bool,
    ) -> (Vec<TensorInfo>, u64) {
        let mut tensors = Vec::new();
        let mut running = 0u64;
        for (name, shape, ty) in specs {
            let nbytes = if ty.wire_id() == AMBIGUOUS_TYPE_ID {
                let ne0 = shape[0];
                let nrows: u64 = shape.iter().skip(1).product();
                nrows * (ne0 / g42.0 as u64) * g42.1 as u64
            } else {
                row_size_bytes(*ty, shape)
            };
            tensors.push(info(name, shape, *ty, running));
            running += if padded {
                align_up(nbytes, alignment).expect("align")
            } else {
                nbytes
            };
        }
        (tensors, running)
    }

    fn specs() -> Vec<(&'static str, Vec<u64>, GgufTensorType)> {
        vec![
            (
                "token_embd.weight",
                vec![128, 40],
                GgufTensorType::TQ2_0_g128,
            ),
            ("blk.0.attn_norm.weight", vec![128], GgufTensorType::F32),
            (
                "blk.0.ffn_up.weight",
                vec![128, 96],
                GgufTensorType::TQ2_0_g128,
            ),
            ("output.weight", vec![128, 40], GgufTensorType::TQ2_0_g128),
        ]
    }

    // ── The mandatory-evidence contract ───────────────────────────────────

    /// A call with no extents must ERROR, never guess.
    #[test]
    fn missing_extents_is_an_error() {
        let (tensors, _) = lay_out(&specs(), 32, (128, 34), true);
        let err = resolve_type_42(&tensors, 32, None, None).expect_err("must refuse");
        match err {
            BonsaiError::AmbiguousQuantType { type_id, hint } => {
                assert_eq!(type_id, 42);
                assert!(hint.contains("extents"), "hint: {hint}");
            }
            other => panic!("expected AmbiguousQuantType, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_extent_length_is_an_error() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let mut ext = compute_extents(&tensors, data_len).expect("extents");
        ext.pop();
        assert!(resolve_type_42(&tensors, 32, None, Some(&ext)).is_err());
    }

    /// d-first vs qs-first cannot be decided without the data, so the 4-arg
    /// entry point must refuse rather than default.
    #[test]
    fn numeric_qver_without_a_sample_is_ambiguous_not_defaulted() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::Uint32(2);
        match resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)) {
            Err(BonsaiError::AmbiguousQuantType { type_id, hint }) => {
                assert_eq!(type_id, 42);
                assert!(hint.contains("resolve_type_42_with_sample"), "hint: {hint}");
            }
            other => panic!("expected AmbiguousQuantType, got {other:?}"),
        }
    }

    // ── Group size from the offset replay ─────────────────────────────────

    #[test]
    fn offset_replay_picks_group_128_over_group_64() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let r = resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)).expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::TQ2_0_g128);
        assert_eq!(r.block_size, 128);
        assert_eq!(r.block_bytes, 34);
        assert_eq!(r.order_evidence, OrderEvidence::LegacyVersionString);
    }

    #[test]
    fn offset_replay_picks_group_64_for_a_mainline_q2_0_file() {
        let (tensors, data_len) = lay_out(&specs(), 32, (64, 18), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::Uint32(2);
        let r = resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)).expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::Q2_0G64);
        assert_eq!(r.block_size, 64);
        assert_eq!(r.block_bytes, 18);
        assert_eq!(r.order_evidence, OrderEvidence::NotApplicable);
    }

    /// The finder's proposed rule — `offset[i+1] - offset[i] == size` — would
    /// reject this file, because the deltas are padded up to 32 bytes.
    /// Rows whose padded size differs between the two candidate geometries:
    /// `[128, 10]` is `10 * 1 * 34 = 340 → 352` at group 128, but
    /// `10 * 2 * 18 = 360 → 384` at group 64.
    fn ragged_specs() -> Vec<(&'static str, Vec<u64>, GgufTensorType)> {
        vec![
            ("a", vec![128, 10], GgufTensorType::TQ2_0_g128),
            ("b", vec![128, 10], GgufTensorType::TQ2_0_g128),
            ("c", vec![128, 10], GgufTensorType::TQ2_0_g128),
        ]
    }

    /// The finder's proposed rule — `offset[i+1] - offset[i] == size` — would
    /// reject this file, because the deltas are padded up to 32 bytes.
    #[test]
    fn padded_offsets_do_not_equal_the_unpadded_sizes() {
        let (tensors, data_len) = lay_out(&ragged_specs(), 32, (128, 34), true);
        assert_eq!(tensors[1].offset, 352, "offset must be the PADDED sum");
        assert_ne!(
            tensors[1].offset - tensors[0].offset,
            340,
            "the delta is an upper bound, not the size"
        );
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let r = resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)).expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::TQ2_0_g128);
        assert_eq!(r.size_model, SizeModel::Padded);
    }

    /// OxiBonsai's own writer historically emitted unpadded offsets; those
    /// files must still resolve, and the model used must be reported.
    #[test]
    fn unpadded_offsets_still_resolve_and_are_reported() {
        let (tensors, data_len) = lay_out(&ragged_specs(), 32, (128, 34), false);
        assert_eq!(tensors[1].offset, 340);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let r = resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)).expect("resolves");
        assert_eq!(r.size_model, SizeModel::Unpadded);
    }

    /// A file small enough that alignment padding hides the size difference
    /// between the two geometries is genuinely undecidable — `[128, 3]` is
    /// 102 B at group 128 and 108 B at group 64, and both pad to 128. The
    /// resolver must say so rather than pick one.
    #[test]
    fn a_file_where_padding_hides_the_difference_is_ambiguous_not_guessed() {
        let degenerate: Vec<(&str, Vec<u64>, GgufTensorType)> = vec![
            ("a", vec![128, 3], GgufTensorType::TQ2_0_g128),
            ("b", vec![128, 3], GgufTensorType::TQ2_0_g128),
        ];
        let (tensors, data_len) = lay_out(&degenerate, 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        match resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)) {
            Err(BonsaiError::AmbiguousQuantType { type_id, hint }) => {
                assert_eq!(type_id, 42);
                assert!(hint.contains("candidates"), "hint: {hint}");
            }
            other => panic!("expected AmbiguousQuantType, got {other:?}"),
        }
    }

    #[test]
    fn a_file_whose_offsets_match_nothing_is_a_hard_error() {
        let (mut tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        tensors[2].offset += 7; // corrupt one offset
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::Uint32(2);
        match resolve_type_42(&tensors, 32, Some(&qver), Some(&ext)) {
            Err(BonsaiError::TensorLayout { reason, .. }) => {
                assert!(reason.contains("offset table"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout, got {other:?}"),
        }
    }

    #[test]
    fn invalid_alignment_is_rejected() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        assert!(resolve_type_42(&tensors, 0, None, Some(&ext)).is_err());
        assert!(resolve_type_42(&tensors, 3, None, Some(&ext)).is_err());
    }

    #[test]
    fn no_type_42_tensor_is_an_error() {
        let tensors = vec![info("a", &[128], GgufTensorType::F32, 0)];
        let ext = vec![512u64];
        assert!(resolve_type_42(&tensors, 32, None, Some(&ext)).is_err());
    }

    // ── Byte order from the data sniff ────────────────────────────────────

    /// Build `n_blocks` of structurally clean 34-byte blocks in the given
    /// order, so the sniff has real evidence.
    fn sample_34(d_first: bool, n_blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; n_blocks * 34];
        let d = half::f16::from_f32(0.0415).to_le_bytes();
        for i in 0..n_blocks {
            let base = i * 34;
            let (s, c) = if d_first {
                (base, base + 2)
            } else {
                (base + 32, base)
            };
            out[s] = d[0];
            out[s + 1] = d[1];
            for k in 0..32 {
                out[c + k] = 0b10_01_00_01;
            }
        }
        out
    }

    #[test]
    fn sniff_settles_qs_first_legacy_layout() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let sample = sample_34(false, 64);
        let r = resolve_type_42_with_sample(&tensors, 32, Some(&qver), Some(&ext), &sample)
            .expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::TQ2_0_g128);
        assert_eq!(r.order_evidence, OrderEvidence::DataSniff);
    }

    #[test]
    fn sniff_settles_d_first_gen1_layout() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::Uint32(2);
        let sample = sample_34(true, 64);
        let r = resolve_type_42_with_sample(&tensors, 32, Some(&qver), Some(&ext), &sample)
            .expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::Q2_0G128DFirst);
        assert_eq!(r.tensor_type.wire_id(), 42);
        assert_eq!(r.order_evidence, OrderEvidence::DataSniff);
    }

    /// A file that claims the legacy string but whose bytes are d-first is a
    /// hard error — silently trusting either side is exactly the silent
    /// mis-decode this package exists to remove.
    #[test]
    fn sniff_contradicting_the_legacy_tag_is_a_hard_error() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let sample = sample_34(true, 64);
        match resolve_type_42_with_sample(&tensors, 32, Some(&qver), Some(&ext), &sample) {
            Err(BonsaiError::QuantLayoutMismatch {
                assumed,
                suggestion,
                ..
            }) => {
                assert!(assumed.contains("qs-first"), "assumed: {assumed}");
                assert!(suggestion.contains("d-first"), "suggestion: {suggestion}");
            }
            other => panic!("expected QuantLayoutMismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_all_zero_sample_with_a_numeric_qver_stays_ambiguous() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::Uint32(2);
        let sample = vec![0u8; 34 * 18];
        assert!(matches!(
            resolve_type_42_with_sample(&tensors, 32, Some(&qver), Some(&ext), &sample),
            Err(BonsaiError::AmbiguousQuantType { .. })
        ));
    }

    #[test]
    fn an_all_zero_sample_falls_back_to_the_legacy_tag() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        let qver = MetadataValue::String(LEGACY_QVER_STRING.to_string());
        let sample = vec![0u8; 34 * 18];
        let r = resolve_type_42_with_sample(&tensors, 32, Some(&qver), Some(&ext), &sample)
            .expect("resolves");
        assert_eq!(r.tensor_type, GgufTensorType::TQ2_0_g128);
        assert_eq!(r.order_evidence, OrderEvidence::LegacyVersionString);
    }

    #[test]
    fn version_tag_classification() {
        assert_eq!(
            LegacyVersionTag::from_value(Some(&MetadataValue::String("TQ2_0_G128".into()))),
            LegacyVersionTag::Tq2_0G128
        );
        assert_eq!(
            LegacyVersionTag::from_value(Some(&MetadataValue::Uint32(2))),
            LegacyVersionTag::Numeric
        );
        assert_eq!(
            LegacyVersionTag::from_value(None),
            LegacyVersionTag::Numeric
        );
    }

    #[test]
    fn compute_extents_covers_every_tensor_exactly_once() {
        let (tensors, data_len) = lay_out(&specs(), 32, (128, 34), true);
        let ext = compute_extents(&tensors, data_len).expect("extents");
        assert_eq!(ext.len(), tensors.len());
        assert_eq!(ext.iter().sum::<u64>(), data_len);
    }
}
