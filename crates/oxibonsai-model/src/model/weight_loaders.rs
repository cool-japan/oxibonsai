// Weight-loading helper functions extracted from model.rs.
// These are `pub(super)` so only `model.rs` can call them.

use oxibonsai_core::config::Qwen3Config;
#[cfg(test)]
use oxibonsai_core::config::RopeScaling;
use oxibonsai_core::gguf::quant_resolve::{
    compute_extents, resolve_type_42_with_sample, AMBIGUOUS_TYPE_ID,
};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::{row_size_bytes, tensor_names, TensorInfo};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_ternary::{sniff_sample_byte_cap, SNIFF_DEFAULT_BLOCKS};
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_core::{
    BlockFP8E4M3, BlockFP8E5M2, BlockPQ2_0, BlockPTQ1_0, BlockQ2K, BlockQ2_0G64, BlockQ3K,
    BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, BlockTQ2_0, BlockTQ2_0_g128,
    BonsaiError,
};

use crate::block::TransformerBlock;
use crate::convert::mlx_image::pack::bf16_to_f32;
use crate::error::{ModelError, ModelResult};
use crate::layers::linear::{
    Linear1Bit, LinearFP8E4M3, LinearFP8E5M2, LinearPQ2_0, LinearPTQ1_0, LinearQ2_0G64,
    LinearTernary,
};
use crate::layers::linear_kquant_ext::{LinearQ5K, LinearQ6K};
use crate::layers::linear_kquant_full::{LinearQ2K, LinearQ3K, LinearQ4K, LinearQ8K};
use crate::layers::linear_standard::{LinearQ4_0, LinearQ8_0};
use crate::layers::rms_norm::RmsNorm;

use super::types::OutputWeight;

// ─────────────────────────────────────────────────────────────────────────────
// Unified dequantization dispatch (M-09 / M-10 / K-12)
//
// `dequant_any` is the ONLY quant-type-to-`Vec<f32>` dispatch table in this
// crate. Before this, `load_f32_tensor` and `load_output_weight` each carried
// their own hand-written table and had drifted apart (`load_f32_tensor` had
// no BF16/Q2_K/Q3_K/Q8_K arm even though `load_output_weight` did, and both
// still used `GgufTensorType`-by-hand rather than `is_executable()`). Every
// `GgufTensorType` variant is named explicitly below — there is no `_ =>`
// wildcard — so the compiler forces a deliberate decision the moment a new
// variant is added to the enum, instead of letting it silently fall through
// to whatever the last arm happened to be (the exact class of bug M-10 found
// in `load_transformer_block`'s boolean ladder).
// ─────────────────────────────────────────────────────────────────────────────

/// Decode a flat (block-size-1) tensor — `F32`/`F16`/`BF16` — into exactly `n`
/// `f32` values, `elem_bytes` bytes per element.
///
/// Requires `data.len() == n * elem_bytes` exactly; a mismatch is reported
/// rather than silently truncated or zero-padded.
fn dequant_flat(
    tensor_type: GgufTensorType,
    data: &[u8],
    n: usize,
    elem_bytes: usize,
    decode_elem: impl Fn(&[u8]) -> f32,
) -> ModelResult<Vec<f32>> {
    let expected_bytes = n.saturating_mul(elem_bytes);
    if data.len() != expected_bytes {
        return Err(ModelError::ShapeMismatch {
            name: format!("<{tensor_type} dequant_any>"),
            expected: vec![expected_bytes],
            actual: vec![data.len()],
        });
    }
    let mut out = vec![0.0f32; n];
    for (i, chunk) in data.chunks_exact(elem_bytes).enumerate() {
        out[i] = decode_elem(chunk);
    }
    Ok(out)
}

/// Decode a slice of `group`-sized quantization blocks (already zero-copy
/// cast from the tensor bytes via `slice`) into exactly `n` `f32` values.
///
/// `n` must equal `blocks.len() * group` — the caller's declared element
/// count (from the GGUF tensor shape) must agree with what the byte length
/// actually decodes to. A mismatch (e.g. a truncated or misdeclared tensor)
/// is a hard error rather than a buffer whose length silently disagrees with
/// what was asked for.
fn dequant_blocks<T>(
    tensor_type: GgufTensorType,
    slice: oxibonsai_core::BonsaiResult<&[T]>,
    n: usize,
    group: usize,
    decode: impl FnOnce(&[T], &mut [f32]) -> oxibonsai_core::BonsaiResult<()>,
) -> ModelResult<Vec<f32>> {
    let blocks = slice.map_err(ModelError::Core)?;
    let decoded_len = blocks.len().saturating_mul(group);
    if decoded_len != n {
        return Err(ModelError::ShapeMismatch {
            name: format!("<{tensor_type} dequant_any>"),
            expected: vec![n],
            actual: vec![decoded_len],
        });
    }
    let mut out = vec![0.0f32; n];
    decode(blocks, &mut out).map_err(ModelError::Core)?;
    Ok(out)
}

/// Single, canonical dequantization dispatch used by every weight loader in
/// this module (M-09). `n` is the tensor's expected element count (the
/// product of its GGUF shape, e.g.
/// [`oxibonsai_core::gguf::tensor_info::TensorInfo::element_count`]).
///
/// Every executable [`GgufTensorType`] (per
/// [`GgufTensorType::is_executable`]) has a real decoder here — including
/// `TQ2_0` (mainline) and `Q2_0G128DFirst`, which no caller in this file
/// asked for directly but which the type system still reports as
/// executable, so misclassifying them as non-executable would itself be a
/// drift bug. Every other variant — one this build can parse but has no
/// kernel for — reports [`BonsaiError::NonExecutableQuantType`] by name
/// (M-10): there is no arm that falls through to a decoder that does not
/// match the on-disk format.
pub(super) fn dequant_any(
    tensor_type: GgufTensorType,
    data: &[u8],
    n: usize,
) -> ModelResult<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => dequant_flat(tensor_type, data, n, 4, |c| {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
        }),
        GgufTensorType::F16 => dequant_flat(tensor_type, data, n, 2, |c| {
            half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()
        }),
        // BF16: `ssm_alpha.weight`/`ssm_beta.weight` on every linear layer of
        // the 27B are BF16 (K-12). A dedicated NEON/AVX2 `gemv_bf16` kernel is
        // NOT built here — the finding's own sizing (94 MB total across both
        // 27B files) makes a one-time widen-to-f32 at load simpler and just
        // as fast as a bespoke kernel would be for tensors this small.
        GgufTensorType::BF16 => dequant_flat(tensor_type, data, n, 2, |c| {
            bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
        }),
        GgufTensorType::Q1_0_g128 => dequant_blocks(
            tensor_type,
            BlockQ1_0G128::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            |blocks, out| {
                let group = tensor_type.block_size();
                for (i, block) in blocks.iter().enumerate() {
                    let base = i * group;
                    for j in 0..group {
                        out[base + j] = block.weight(j);
                    }
                }
                Ok(())
            },
        ),
        GgufTensorType::TQ2_0_g128 => dequant_blocks(
            tensor_type,
            BlockTQ2_0_g128::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockTQ2_0_g128::dequant,
        ),
        GgufTensorType::TQ2_0 => dequant_blocks(
            tensor_type,
            BlockTQ2_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockTQ2_0::dequant,
        ),
        GgufTensorType::Q4_0 => dequant_blocks(
            tensor_type,
            BlockQ4_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ4_0::dequant,
        ),
        GgufTensorType::Q8_0 => dequant_blocks(
            tensor_type,
            BlockQ8_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ8_0::dequant,
        ),
        GgufTensorType::Q2_K => dequant_blocks(
            tensor_type,
            BlockQ2K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ2K::dequant,
        ),
        GgufTensorType::Q3_K => dequant_blocks(
            tensor_type,
            BlockQ3K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ3K::dequant,
        ),
        GgufTensorType::Q4_K => dequant_blocks(
            tensor_type,
            BlockQ4K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ4K::dequant,
        ),
        GgufTensorType::Q5_K => dequant_blocks(
            tensor_type,
            BlockQ5K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ5K::dequant,
        ),
        GgufTensorType::Q6_K => dequant_blocks(
            tensor_type,
            BlockQ6K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ6K::dequant,
        ),
        GgufTensorType::Q8_K => dequant_blocks(
            tensor_type,
            BlockQ8K::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ8K::dequant,
        ),
        GgufTensorType::F8_E4M3 => dequant_blocks(
            tensor_type,
            BlockFP8E4M3::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockFP8E4M3::dequant,
        ),
        GgufTensorType::F8_E5M2 => dequant_blocks(
            tensor_type,
            BlockFP8E5M2::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockFP8E5M2::dequant,
        ),
        GgufTensorType::Q2_0G64 => dequant_blocks(
            tensor_type,
            BlockQ2_0G64::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockQ2_0G64::dequant,
        ),
        // `Q2_0G128DFirst` (PrismML gen-1, ggml id 42 read as group 128) is
        // wire-identical to `PQ2_0` (`d` first, then `qs[32]`, 34 bytes) —
        // same struct, same decode, only the *logical* type id differs so the
        // id-42 resolver can tell the two apart.
        GgufTensorType::Q2_0G128DFirst => dequant_blocks(
            tensor_type,
            BlockPQ2_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockPQ2_0::dequant,
        ),
        GgufTensorType::PQ2_0 => dequant_blocks(
            tensor_type,
            BlockPQ2_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockPQ2_0::dequant,
        ),
        GgufTensorType::PTQ1_0 => dequant_blocks(
            tensor_type,
            BlockPTQ1_0::slice_from_bytes(data),
            n,
            tensor_type.block_size(),
            BlockPTQ1_0::dequant,
        ),
        // Every remaining variant is a type this build's *parser* recognises
        // (`GgufTensorType::from_id` succeeds) but no kernel in this build
        // can execute (`is_executable()` is `false`) — plain integer types,
        // llama.cpp k-quant variants OxiBonsai has never implemented, and
        // legacy asymmetric formats no current model file uses. Named
        // individually (not `_`) so adding a new variant to the enum is a
        // compile error here, not a silent fall-through.
        GgufTensorType::Q4_1
        | GgufTensorType::Q5_0
        | GgufTensorType::Q5_1
        | GgufTensorType::Q8_1
        | GgufTensorType::IQ2_XXS
        | GgufTensorType::IQ2_XS
        | GgufTensorType::IQ3_XXS
        | GgufTensorType::IQ1_S
        | GgufTensorType::IQ4_NL
        | GgufTensorType::IQ3_S
        | GgufTensorType::IQ2_S
        | GgufTensorType::IQ4_XS
        | GgufTensorType::I8
        | GgufTensorType::I16
        | GgufTensorType::I32
        | GgufTensorType::I64
        | GgufTensorType::F64
        | GgufTensorType::IQ1_M
        | GgufTensorType::TQ1_0
        | GgufTensorType::MXFP4
        | GgufTensorType::NVFP4 => Err(ModelError::Core(BonsaiError::non_executable_quant_type(
            tensor_type.name(),
            tensor_type,
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ggml wire id 42 resolution (B2-09 fix)
//
// `TensorInfo::tensor_type` is only ever the PARSE-TIME guess for wire id 42
// -- always `GgufTensorType::TQ2_0_g128` (see
// `oxibonsai_core::gguf::quant_resolve`'s module doc) -- regardless of
// whether the file is actually that legacy qs-first layout, PrismML gen-1
// d-first (`GgufTensorType::Q2_0G128DFirst`), or mainline group-64
// (`GgufTensorType::Q2_0G64`). Every load-time decision in this module that
// switches on a tensor's type must route through [`apply_resolved_type`]
// instead of reading `TensorInfo::tensor_type` directly, or a real PrismML
// file silently decodes under the wrong layout: same byte count either way,
// so it "succeeds" with the `d`-first FP16 scale bytes read back as 2-bit
// codes.
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve ggml wire id 42's on-disk layout ONCE for a whole GGUF file.
///
/// The whole file shares one on-disk reading of id 42 --
/// [`resolve_type_42_with_sample`] settles the group size by replaying
/// every tensor's offset and the byte order from one sample -- so this
/// resolves once per file (mirroring
/// `oxibonsai_model::gguf_loader::load_tensor_metadata_resolved`'s
/// pattern) rather than once per tensor or once per layer: `Ok(None)` when
/// the file has no wire-id-42 tensor at all (every type then passes through
/// [`apply_resolved_type`] unchanged), `Ok(Some(ty))` otherwise.
///
/// Callers that already hold this value (e.g. the per-layer loop in
/// `model/types/mod.rs`) must compute it once and thread it through rather
/// than calling this again per layer -- the replay walks every tensor's
/// offset and the sniff samples real block bytes, so repeating it 64 times
/// for the 27B would be pure waste for an answer that cannot change within
/// one file.
pub(super) fn resolve_id42_once(gguf: &GgufFile<'_>) -> ModelResult<Option<GgufTensorType>> {
    let infos: Vec<TensorInfo> = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let Some(sample_info) = infos
        .iter()
        .find(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
    else {
        return Ok(None);
    };
    let sample_name = sample_info.name.clone();
    let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
    let extents = compute_extents(&infos, data_len).map_err(ModelError::Core)?;
    let alignment = gguf
        .metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;
    let sample = gguf.tensor_data(&sample_name).map_err(ModelError::Core)?;
    let sample = &sample[..sample
        .len()
        .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];
    match resolve_type_42_with_sample(
        &infos,
        alignment,
        gguf.metadata.get("general.quantization_version"),
        Some(&extents),
        sample,
    ) {
        Ok(resolved) => Ok(Some(resolved.tensor_type)),
        // `AmbiguousQuantType` means the offset table AND the data sniff
        // both genuinely fail to determine a layout -- `quant_resolve.rs`'s
        // own module doc says this is only reachable in practice for an
        // all-zero, too-small, or otherwise degenerate sample (every real
        // PrismML file resolves conclusively; see
        // `quant_prism_golden.rs::id_42_resolves_correctly_on_every_real_model_present`).
        // That is exactly the shape of the many pre-existing synthetic
        // `TQ2_0_g128` test fixtures across the workspace (built in-process
        // via `GgufWriter`, never carrying `general.quantization_version`,
        // sized for a fast unit test rather than a real checkpoint) --
        // files this package does not own and cannot add the tag to.
        // Falling back to the legacy qs-first reading here reproduces
        // exactly what every caller did before this package's fix existed,
        // for the one case (genuinely inconclusive evidence) where nothing
        // -- old code or new -- ever had grounds to prefer one reading over
        // another. `TensorLayout` (the offsets are internally
        // inconsistent) and `QuantLayoutMismatch` (the data actively
        // contradicts a declared legacy tag) are NOT included here: both
        // are active evidence of a real problem, not merely insufficient
        // evidence, and stay hard errors.
        Err(BonsaiError::AmbiguousQuantType { hint, .. }) => {
            tracing::warn!(
                tensor = %sample_name,
                reason = %hint,
                "ggml wire id 42 could not be conclusively resolved (degenerate/synthetic \
                 sample); falling back to the legacy qs-first TQ2_0_g128 reading"
            );
            Ok(None)
        }
        Err(other) => Err(ModelError::Core(other)),
    }
}

/// Apply [`resolve_id42_once`]'s per-file resolution to one tensor's raw
/// (parse-time) type.
///
/// Every type but wire id 42 passes through unchanged. `resolved_42: None`
/// covers two cases the caller cannot tell apart and does not need to:
/// the file has no wire-id-42 tensor at all, or [`resolve_id42_once`]
/// could not conclusively resolve one (see its own doc for why that is
/// the historical legacy reading, not a guess). Either way `raw` -- already
/// `TQ2_0_g128` for any wire-id-42 tensor per `GgufTensorType::from_id` --
/// is the answer.
pub(super) fn apply_resolved_type(
    raw: GgufTensorType,
    resolved_42: Option<GgufTensorType>,
) -> GgufTensorType {
    if raw.wire_id() != AMBIGUOUS_TYPE_ID {
        return raw;
    }
    resolved_42.unwrap_or(raw)
}

/// [`GgufFile::tensor_data`], but sized from a RESOLVED type instead of
/// [`oxibonsai_core::gguf::tensor_info::TensorInfo::data_size`]'s raw
/// parse-time guess.
///
/// `TensorInfo::data_size` (and therefore `GgufFile::tensor_data`) is
/// documented as "provisional for wire id 42": it always sizes a
/// wire-id-42 tensor as the legacy qs-first group-128 reading (34 bytes per
/// 128-element row), which byte-UNDER-counts a genuine mainline group-64
/// tensor (36 bytes per 128 elements, i.e. two 18-byte blocks) and would
/// hand every group-64 block loader a slice truncated by 2 bytes per 128
/// elements -- silently dropping the tail of every row instead of failing
/// loudly, were it not for the loader's own exact-byte-length check turning
/// that into a clear `ModelError` instead. That doc's own fix is to
/// "recompute [the size] from the settled resolved type ... via
/// `row_size_bytes`", which is exactly what this does; every other
/// (non-wire-id-42) tensor is byte-identical to plain `tensor_data`.
fn tensor_data_resolved<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    resolved_type: GgufTensorType,
) -> ModelResult<&'a [u8]> {
    let info = gguf.tensors.require(name).map_err(ModelError::Core)?;
    if resolved_type == info.tensor_type {
        return gguf.tensor_data(name).map_err(ModelError::Core);
    }
    let size = row_size_bytes(resolved_type, &info.shape);
    let data_len = gguf.data.len() as u64;
    let data_offset = gguf.data_offset as u64;
    let start = data_offset
        .checked_add(info.offset)
        .ok_or(ModelError::Core(BonsaiError::UnexpectedEof {
            offset: u64::MAX,
        }))?;
    let end = start
        .checked_add(size)
        .ok_or(ModelError::Core(BonsaiError::UnexpectedEof {
            offset: u64::MAX,
        }))?;
    if end > data_len {
        return Err(ModelError::Core(BonsaiError::UnexpectedEof { offset: end }));
    }
    Ok(&gguf.data[start as usize..end as usize])
}

/// [`resolve_id42_once`] + [`apply_resolved_type`] for a single ad hoc
/// tensor lookup, for callers (namely [`load_f32_tensor`]) that do not
/// already hold a precomputed `resolved_42` for this file.
///
/// Cheap in the common case: every tensor `load_f32_tensor` is actually
/// called with today (RMSNorm weights, `ssm_a`, `ssm_alpha`/`ssm_beta`, an
/// F32/F16/BF16-typed output/embedding tensor) has a non-42 wire id, so this
/// returns `info.tensor_type` immediately with no scan at all. The
/// expensive per-file replay only runs on the rare tensor whose wire id
/// really is 42.
fn resolve_sample_tensor_type(
    gguf: &GgufFile<'_>,
    info: &TensorInfo,
) -> ModelResult<GgufTensorType> {
    if info.tensor_type.wire_id() != AMBIGUOUS_TYPE_ID {
        return Ok(info.tensor_type);
    }
    let resolved_42 = resolve_id42_once(gguf)?;
    Ok(apply_resolved_type(info.tensor_type, resolved_42))
}

/// Dominant quantization type across the **weight** tensors of `gguf` (M-25).
///
/// Deterministic in two ways the previous `HashMap` + `max_by_key` was not:
///
/// * F32/F16 tensors (norms, `ssm_a`, `ssm_dt.bias`, …) are excluded. They are
///   never what "the model's quantization" means, and on a small or mixed file
///   they can out-number the quantized matrices and win outright — which is
///   what makes the reported [`ModelVariant`] meaningful at all.
/// * Selection runs over a [`BTreeMap`](std::collections::BTreeMap) keyed by
///   [`GgufTensorType`](oxibonsai_core::GgufTensorType), so one file always
///   yields one answer regardless of hash iteration order. Ties break towards
///   whichever tied variant is declared **first** in the `GgufTensorType`
///   enum body, which is what the derived `Ord` orders by — that is the same
///   as ascending [`wire_id`](oxibonsai_core::GgufTensorType::wire_id) for
///   every variant with its own ggml id, but NOT for the two ggml-id-42
///   aliases with sentinel Rust discriminants (`Q2_0G64`, `Q2_0G128DFirst`):
///   both are declared, and so win ties, ahead of `PQ2_0`/`PTQ1_0` despite
///   their far larger Rust discriminant values (`wire_id` alone actually
///   agrees with this outcome here, since both alias ggml id 42, well below
///   142/143 — it is the discriminant, not the wire id, that this ordering
///   diverges from). Still fully deterministic either way, which is all this
///   finding asked for.
///
/// RAW: this buckets every wire-id-42 tensor under `TQ2_0_g128` (the
/// parse-time guess) regardless of its actual on-disk layout. Use
/// [`resolved_dominant_weight_quant_type`] for anything that reports a
/// variant/size to a human or feeds `ModelVariant` detection (B2-09) — this
/// raw form only remains `pub(super)` for [`dominant_from_counts`]'s own
/// unit tests, which exercise the pure selection logic over hand-written
/// `(type, count)` pairs.
pub(super) fn dominant_weight_quant_type(gguf: &GgufFile<'_>) -> GgufTensorType {
    dominant_from_counts(gguf.tensors.count_by_type().into_iter())
}

/// [`resolve_id42_once`] + [`apply_resolved_type`] applied to
/// [`dominant_weight_quant_type`]'s answer (B2-09).
///
/// The raw tally can only ever land on `TQ2_0_g128` by counting wire-id-42
/// tensors (that is the ONLY way `GgufTensorType::from_id` produces that
/// variant), so `resolved_42` is guaranteed `Some` whenever relabelling is
/// actually needed; every other winning type passes through untouched with
/// no per-file replay at all.
pub(super) fn resolved_dominant_weight_quant_type(
    gguf: &GgufFile<'_>,
) -> ModelResult<GgufTensorType> {
    let raw = dominant_weight_quant_type(gguf);
    if raw != GgufTensorType::TQ2_0_g128 {
        return Ok(raw);
    }
    let resolved_42 = resolve_id42_once(gguf)?;
    Ok(apply_resolved_type(raw, resolved_42))
}

/// Selection half of [`dominant_weight_quant_type`], over `(type, count)` pairs.
pub(super) fn dominant_from_counts<I>(counts: I) -> GgufTensorType
where
    I: Iterator<Item = (GgufTensorType, usize)>,
{
    let mut tally: std::collections::BTreeMap<GgufTensorType, usize> =
        std::collections::BTreeMap::new();
    for (ty, count) in counts {
        if matches!(ty, GgufTensorType::F32 | GgufTensorType::F16) {
            continue;
        }
        *tally.entry(ty).or_insert(0) += count;
    }
    let mut best: Option<(GgufTensorType, usize)> = None;
    for (ty, count) in tally {
        // `tally` iterates in ascending type-id order and the comparison is
        // strict, so the first (lowest-id) type keeps a tie.
        let wins = match best {
            Some((_, best_count)) => count > best_count,
            None => true,
        };
        if wins {
            best = Some((ty, count));
        }
    }
    best.map(|(ty, _)| ty).unwrap_or(GgufTensorType::Q1_0_g128)
}

/// Load an FP32 tensor from GGUF by name.
///
/// Thin wrapper around [`dequant_any`] (M-09): every quant type this crate
/// can execute is handled there, in one place, so this function and
/// [`load_output_weight`] can never drift apart again the way they had
/// (`load_f32_tensor` used to lack BF16/Q2_K/Q3_K/Q8_K arms that
/// `load_output_weight` already had).
pub(super) fn load_f32_tensor(gguf: &GgufFile<'_>, name: &str) -> ModelResult<Vec<f32>> {
    let info = gguf.tensors.require(name).map_err(ModelError::Core)?;
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    let element_count = info.element_count();
    let n = usize::try_from(element_count).map_err(|_| {
        ModelError::Core(BonsaiError::tensor_layout(
            name,
            format!("element count {element_count} overflows usize on this platform"),
        ))
    })?;
    // B2-09 fix: resolve the real on-disk layout before screening/decoding
    // (see the "ggml wire id 42 resolution" section above) -- `info.tensor_type`
    // alone is only ever the parse-time guess for wire id 42.
    let resolved_type = resolve_sample_tensor_type(gguf, info)?;
    // CQ-14: screen a declared-ternary tensor before decoding it, so a
    // `PQ2_0` tensor mis-declared as ggml id 42 fails loudly here instead of
    // decoding every `+2` weight as `0`.
    screen_ternary_codes(name, data, resolved_type)?;
    let values = dequant_any(resolved_type, data, n)?;
    // B2-05's acceptance criterion: a Gated DeltaNet `ssm_a` (`A = -exp(A_log)`,
    // GGUF `blk.N.ssm_a`, always F32) must be strictly non-positive, or the
    // decay gate `exp(g)` in `gdn_step`/`gdn_chunk` exceeds 1 and the
    // recurrence diverges. Reject at load with a named error, not a log
    // line — this is the single point every `ssm_a` load (present or
    // future) passes through, so a future hybrid-model loader inherits the
    // check automatically by calling `load_f32_tensor` like every other
    // named tensor.
    if name.ends_with(".ssm_a") || name == "ssm_a" {
        oxibonsai_kernels::gated_delta_net::validate_a_neg(&values)
            .map_err(|e| ModelError::InvalidTensor(format!("{name}: {e}")))?;
    }
    Ok(values)
}

/// Load Q1_0_g128 weight blocks from GGUF (zero-copy).
pub(super) fn load_1bit_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ1_0G128]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Reject a ternary tensor that carries the reserved `0b11` two-bit code
/// (CQ-14, load side).
///
/// `TQ2_0_g128` (ggml id 42, `qs` first) and `TQ2_0` (id 35) are *declared*
/// ternary: their decoders map `0b11` to `0`. `PQ2_0` (142) and the mainline
/// group-64 `Q2_0` decode the very same bit pattern as `+2`, and `PQ2_0` is
/// stored under **id 42 as well** in PrismML gen-1 files. A `PQ2_0` tensor
/// mis-declared as `TQ2_0_g128` therefore loads and runs, silently reading
/// every `+2` weight as `0`.
///
/// `PQ2_0` and `Q2_0_g64` are deliberately **exempt**: `0b11` is a legal `+2`
/// there, so screening them would reject valid files. This is the load-side
/// twin of the converter-side screen `crate::quantize` already applies when
/// writing (CONVERT-EXPORT wave 1).
///
/// # Errors
///
/// [`ModelError::InvalidTensor`] naming the tensor and the offending block.
fn screen_ternary_codes(name: &str, data: &[u8], tensor_type: GgufTensorType) -> ModelResult<()> {
    use oxibonsai_core::gguf::writer::TensorType;

    // Only the two layouts whose declared meaning is ternary. Named
    // explicitly rather than converted wholesale, because the `GgufTensorType`
    // -> `TensorType` relation is not a bijection (two variants share ggml id
    // 42) and a blanket conversion would be exactly the kind of drift CQ-14
    // is about.
    let declared = match tensor_type {
        GgufTensorType::TQ2_0_g128 => TensorType::TQ2_0_g128,
        GgufTensorType::TQ2_0 => TensorType::TQ2_0,
        _ => return Ok(()),
    };
    crate::quantize::validate_ternary_codes(name, data, declared)
        .map_err(|e| ModelError::InvalidTensor(e.to_string()))
}

/// Load TQ2\_0\_g128 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockTQ2_0_g128` pointing directly into the
/// memory-mapped GGUF data.  The lifetime is tied to the `GgufFile`.
///
/// **Deliberately not screened for the reserved `0b11` code** (CQ-14). This is
/// the zero-copy *weight* path, and that contract already has an enforcement
/// point downstream: the Metal SoA upload rejects `0b11` in a tensor declared
/// ternary (MET-11,
/// `oxibonsai_kernels::gpu_backend::metal_graph::reformat::validate_tq2_ternary_codes`).
/// The gap CQ-14 identified is the *dequantize-to-f32* path, where nothing
/// checked and the CPU decoder maps `0b11 → 0` silently — that is where
/// [`screen_ternary_codes`] is applied, from [`load_f32_tensor`].
pub(super) fn load_ternary_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockTQ2_0_g128]> {
    let data = tensor_data_resolved(gguf, name, GgufTensorType::TQ2_0_g128)?;
    oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load FP8 E4M3FN weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockFP8E4M3` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_fp8_e4m3_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockFP8E4M3]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    oxibonsai_core::BlockFP8E4M3::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load FP8 E5M2 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockFP8E5M2` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_fp8_e5m2_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockFP8E5M2]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    oxibonsai_core::BlockFP8E5M2::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load Q4_0 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockQ4_0` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_q4_0_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ4_0]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ4_0::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load Q8_0 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockQ8_0` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_q8_0_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ8_0]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ8_0::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load `PQ2_0` weight blocks from GGUF (zero-copy; B2-09).
///
/// Also the loader for `Q2_0G128DFirst` (the PrismML gen-1 reading of ggml
/// id 42), which is wire-identical to `PQ2_0` — see `dequant_any`'s comment.
pub(super) fn load_pq2_0_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockPQ2_0]> {
    let data = tensor_data_resolved(gguf, name, GgufTensorType::PQ2_0)?;
    BlockPQ2_0::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load `PTQ1_0` weight blocks from GGUF (zero-copy; B2-09).
pub(super) fn load_ptq1_0_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockPTQ1_0]> {
    let data = tensor_data_resolved(gguf, name, GgufTensorType::PTQ1_0)?;
    BlockPTQ1_0::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load mainline group-64 `Q2_0` weight blocks from GGUF (zero-copy; B2-09).
///
/// The tensor sample that routes a real file here always arrives with wire
/// id 42, whose `TensorInfo::tensor_type` is only ever the group-128
/// parse-time guess (see [`tensor_data_resolved`]'s doc) -- byte-UNDER-sized
/// for this variant's real group-64 rows, which is exactly the mismatch
/// [`tensor_data_resolved`] corrects before this ever reaches
/// `BlockQ2_0G64::slice_from_bytes`.
pub(super) fn load_q2_0_g64_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ2_0G64]> {
    let data = tensor_data_resolved(gguf, name, GgufTensorType::Q2_0G64)?;
    BlockQ2_0G64::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load Q5_K weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockQ5K` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_q5k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ5K]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ5K::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load Q6_K weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockQ6K` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_q6k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ6K]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ6K::slice_from_bytes(data).map_err(ModelError::Core)
}

pub(super) fn load_q2k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ2K]> {
    BlockQ2K::slice_from_bytes(gguf.tensor_data(name).map_err(ModelError::Core)?)
        .map_err(ModelError::Core)
}
pub(super) fn load_q3k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ3K]> {
    BlockQ3K::slice_from_bytes(gguf.tensor_data(name).map_err(ModelError::Core)?)
        .map_err(ModelError::Core)
}
pub(super) fn load_q4k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ4K]> {
    BlockQ4K::slice_from_bytes(gguf.tensor_data(name).map_err(ModelError::Core)?)
        .map_err(ModelError::Core)
}
pub(super) fn load_q8k_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ8K]> {
    BlockQ8K::slice_from_bytes(gguf.tensor_data(name).map_err(ModelError::Core)?)
        .map_err(ModelError::Core)
}

/// Load a single Transformer block's weights from GGUF.
///
/// Automatically detects whether the model uses Q1\_0\_g128 (1-bit) or
/// TQ2\_0\_g128 (ternary) quantization by inspecting the attention Q tensor.
///
/// `resolved_42` is [`resolve_id42_once`]'s per-file resolution of ggml wire
/// id 42, computed ONCE by the caller (`model/types/mod.rs`, before the
/// per-layer loop) and threaded through here rather than re-resolved per
/// layer (B2-09): the replay walks every tensor's offset and the sniff
/// samples real block bytes, so redoing it once per layer would repeat that
/// work 64 times for the 27B for an answer that cannot change within one
/// file. Pass `None` when the caller already knows the file has no
/// wire-id-42 tensor at all.
pub(super) fn load_transformer_block<'a>(
    gguf: &'a GgufFile<'a>,
    config: &Qwen3Config,
    layer_idx: usize,
    kernel: &std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    resolved_42: Option<GgufTensorType>,
) -> ModelResult<TransformerBlock<'a>> {
    // M-12, load-time half: the four shape invariants, checked once per layer
    // here — before any `Linear*` is built and before the KV cache is
    // allocated from the same `config` — instead of once per layer per token
    // inside `block::functions::validate_shapes`. `validate_config_shapes`
    // runs first because the divisor it guards (`num_kv_heads`) is the one the
    // per-layer widths below are derived from.
    validate_config_shapes(config)?;
    validate_layer_tensor_shapes(gguf, config, layer_idx)?;

    let h = config.hidden_size;
    let nq = config.num_attention_heads;
    let nkv = config.num_kv_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_size;

    let blk = |suffix: &str| -> String { tensor_names::block_tensor(layer_idx, suffix) };

    // Detect quantization type from the Q projection tensor. B2-09: the RAW
    // `sample_info.tensor_type` is only ever the parse-time guess for wire
    // id 42 -- `resolved_type` is what every arm below actually matches on.
    let sample_name = blk(tensor_names::ATTN_Q);
    let sample_info = gguf
        .tensors
        .require(&sample_name)
        .map_err(ModelError::Core)?;
    let resolved_type = apply_resolved_type(sample_info.tensor_type, resolved_42);
    // RMSNorm weights (always FP32).
    let attn_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_NORM))?;
    let ffn_norm_w = load_f32_tensor(gguf, &blk(tensor_names::FFN_NORM))?;
    let q_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_Q_NORM))?;
    let k_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_K_NORM))?;

    // M-10: dispatch on the sample tensor's RESOLVED quantization type via
    // an exhaustive `match` rather than a boolean ladder ending in a bare
    // `else`. The old `else` branch silently decoded ANY unmatched type
    // (BF16, TQ2_0, PQ2_0, PTQ1_0, Q2_0_g64, or a genuinely unrecognised
    // type) as `Q1_0_g128` garbage — no error, no warning, wrong weights.
    // Every variant is named individually (the final arm is an
    // exhaustive OR-pattern, not a `_`/`other` wildcard), so adding a new
    // `GgufTensorType` variant is a compile error here until a human
    // decides whether it gets a real `Linear*` wrapper or joins the
    // "not wired here yet" list.
    match resolved_type {
        GgufTensorType::TQ2_0_g128 => {
            let q_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearTernary::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearTernary::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearTernary::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearTernary::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearTernary::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearTernary::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearTernary::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded ternary transformer block");
            Ok(block)
        }
        GgufTensorType::F8_E4M3 => {
            // FP8 E4M3FN path.
            let q_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearFP8E4M3::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearFP8E4M3::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearFP8E4M3::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearFP8E4M3::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearFP8E4M3::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearFP8E4M3::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearFP8E4M3::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded FP8 E4M3FN transformer block");
            Ok(block)
        }
        GgufTensorType::F8_E5M2 => {
            // FP8 E5M2 path.
            let q_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearFP8E5M2::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearFP8E5M2::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearFP8E5M2::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearFP8E5M2::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearFP8E5M2::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearFP8E5M2::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearFP8E5M2::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded FP8 E5M2 transformer block");
            Ok(block)
        }
        GgufTensorType::Q4_0 => {
            // Q4_0 (4-bit symmetric) path.
            let q_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_q4_0_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ4_0::new(q_blocks, nq * hd, h)?.into(),
                LinearQ4_0::new(k_blocks, nkv * hd, h)?.into(),
                LinearQ4_0::new(v_blocks, nkv * hd, h)?.into(),
                LinearQ4_0::new(o_blocks, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ4_0::new(gate_blocks, inter, h)?.into(),
                LinearQ4_0::new(up_blocks, inter, h)?.into(),
                LinearQ4_0::new(down_blocks, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q4_0 transformer block");
            Ok(block)
        }
        GgufTensorType::Q8_0 => {
            // Q8_0 (8-bit symmetric) path.
            let q_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_q8_0_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ8_0::new(q_blocks, nq * hd, h)?.into(),
                LinearQ8_0::new(k_blocks, nkv * hd, h)?.into(),
                LinearQ8_0::new(v_blocks, nkv * hd, h)?.into(),
                LinearQ8_0::new(o_blocks, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ8_0::new(gate_blocks, inter, h)?.into(),
                LinearQ8_0::new(up_blocks, inter, h)?.into(),
                LinearQ8_0::new(down_blocks, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q8_0 transformer block");
            Ok(block)
        }
        GgufTensorType::Q5_K => {
            // Q5_K (5-bit K-quant) path.
            let q_blocks = load_q5k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_q5k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_q5k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_q5k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_q5k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_q5k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_q5k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ5K::new(q_blocks, nq * hd, h)?.into(),
                LinearQ5K::new(k_blocks, nkv * hd, h)?.into(),
                LinearQ5K::new(v_blocks, nkv * hd, h)?.into(),
                LinearQ5K::new(o_blocks, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ5K::new(gate_blocks, inter, h)?.into(),
                LinearQ5K::new(up_blocks, inter, h)?.into(),
                LinearQ5K::new(down_blocks, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q5_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q6_K => {
            // Q6_K (6-bit K-quant) path.
            let q_blocks = load_q6k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_q6k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_q6k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_q6k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_q6k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_q6k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_q6k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ6K::new(q_blocks, nq * hd, h)?.into(),
                LinearQ6K::new(k_blocks, nkv * hd, h)?.into(),
                LinearQ6K::new(v_blocks, nkv * hd, h)?.into(),
                LinearQ6K::new(o_blocks, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ6K::new(gate_blocks, inter, h)?.into(),
                LinearQ6K::new(up_blocks, inter, h)?.into(),
                LinearQ6K::new(down_blocks, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q6_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q2_K => {
            let q_b = load_q2k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_b = load_q2k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_b = load_q2k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_b = load_q2k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_b = load_q2k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_b = load_q2k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_b = load_q2k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;
            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ2K::new(q_b, nq * hd, h)?.into(),
                LinearQ2K::new(k_b, nkv * hd, h)?.into(),
                LinearQ2K::new(v_b, nkv * hd, h)?.into(),
                LinearQ2K::new(o_b, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ2K::new(gate_b, inter, h)?.into(),
                LinearQ2K::new(up_b, inter, h)?.into(),
                LinearQ2K::new(down_b, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q2_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q3_K => {
            let q_b = load_q3k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_b = load_q3k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_b = load_q3k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_b = load_q3k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_b = load_q3k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_b = load_q3k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_b = load_q3k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;
            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ3K::new(q_b, nq * hd, h)?.into(),
                LinearQ3K::new(k_b, nkv * hd, h)?.into(),
                LinearQ3K::new(v_b, nkv * hd, h)?.into(),
                LinearQ3K::new(o_b, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ3K::new(gate_b, inter, h)?.into(),
                LinearQ3K::new(up_b, inter, h)?.into(),
                LinearQ3K::new(down_b, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q3_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q4_K => {
            let q_b = load_q4k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_b = load_q4k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_b = load_q4k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_b = load_q4k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_b = load_q4k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_b = load_q4k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_b = load_q4k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;
            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ4K::new(q_b, nq * hd, h)?.into(),
                LinearQ4K::new(k_b, nkv * hd, h)?.into(),
                LinearQ4K::new(v_b, nkv * hd, h)?.into(),
                LinearQ4K::new(o_b, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ4K::new(gate_b, inter, h)?.into(),
                LinearQ4K::new(up_b, inter, h)?.into(),
                LinearQ4K::new(down_b, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q4_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q8_K => {
            let q_b = load_q8k_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_b = load_q8k_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_b = load_q8k_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_b = load_q8k_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_b = load_q8k_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_b = load_q8k_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_b = load_q8k_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;
            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ8K::new(q_b, nq * hd, h)?.into(),
                LinearQ8K::new(k_b, nkv * hd, h)?.into(),
                LinearQ8K::new(v_b, nkv * hd, h)?.into(),
                LinearQ8K::new(o_b, h, nq * hd)?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ8K::new(gate_b, inter, h)?.into(),
                LinearQ8K::new(up_b, inter, h)?.into(),
                LinearQ8K::new(down_b, h, inter)?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q8_K transformer block");
            Ok(block)
        }
        GgufTensorType::Q1_0_g128 => {
            // Q1_0_g128 (1-bit) path.
            let q_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                Linear1Bit::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                Linear1Bit::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                Linear1Bit::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                Linear1Bit::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                Linear1Bit::new(gate_blocks, inter, h, kernel.clone())?.into(),
                Linear1Bit::new(up_blocks, inter, h, kernel.clone())?.into(),
                Linear1Bit::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded transformer block");
            Ok(block)
        }
        // `PQ2_0` (142) and `Q2_0G128DFirst` (the PrismML gen-1 RESOLVED
        // reading of ambiguous ggml id 42, never the raw wire id -- see
        // `resolved_type` above) share one wire layout (B2-09; see
        // `dequant_any`'s own comment above), so both build a `LinearPQ2_0`.
        // This is a generic uniform-Qwen3 wrapper: the real Bonsai 2 27B
        // hybrid layout does not fit `TransformerBlock` (different tensor
        // names/widths for the linear-attention layers, `q|gate` interleave
        // for full-attention ones) — that model is loaded through
        // `hybrid/weights.rs` (B2-10), not this function. This arm exists so
        // ANY GGUF that legitimately uses these quant types in a uniform
        // Qwen3 shape loads correctly instead of hitting the "no wrapper"
        // error below; a hybrid file fails this arm's own shape checks
        // loudly (its `attn_q` tensor won't even exist, or its width
        // disagrees with `nq * hd`), never silently. Reachable because the
        // match dispatches on `resolved_type`, not the raw wire id: a
        // `Ternary-Bonsai-27B-Q2_0.gguf`-shaped file (gen-1, d-first, group
        // 128) resolves here even though its on-disk id is 42, same as
        // `TQ2_0_g128` above.
        GgufTensorType::PQ2_0 | GgufTensorType::Q2_0G128DFirst => {
            let q_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_pq2_0_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearPQ2_0::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearPQ2_0::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearPQ2_0::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearPQ2_0::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearPQ2_0::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearPQ2_0::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearPQ2_0::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded PQ2_0 transformer block");
            Ok(block)
        }
        GgufTensorType::PTQ1_0 => {
            let q_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_ptq1_0_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearPTQ1_0::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearPTQ1_0::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearPTQ1_0::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearPTQ1_0::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearPTQ1_0::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearPTQ1_0::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearPTQ1_0::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded PTQ1_0 transformer block");
            Ok(block)
        }
        GgufTensorType::Q2_0G64 => {
            let q_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
            let k_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::ATTN_K))?;
            let v_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::ATTN_V))?;
            let o_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
            let gate_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
            let up_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::FFN_UP))?;
            let down_blocks = load_q2_0_g64_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(attn_norm_w, config.rms_norm_eps),
                LinearQ2_0G64::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
                LinearQ2_0G64::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearQ2_0G64::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
                LinearQ2_0G64::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
                RmsNorm::new(q_norm_w, config.rms_norm_eps),
                RmsNorm::new(k_norm_w, config.rms_norm_eps),
                RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
                LinearQ2_0G64::new(gate_blocks, inter, h, kernel.clone())?.into(),
                LinearQ2_0G64::new(up_blocks, inter, h, kernel.clone())?.into(),
                LinearQ2_0G64::new(down_blocks, h, inter, kernel.clone())?.into(),
                nq,
                nkv,
                hd,
                h,
            );
            tracing::trace!(layer = layer_idx, "loaded Q2_0G64 transformer block");
            Ok(block)
        }
        // `dequant_any` CAN decode these (`is_executable() == true`), but no
        // `Linear*` attention/FFN kernel wrapper exists for a transformer
        // block in this build yet (a dense FP32 block path is future work,
        // and mainline `TQ2_0` id 35 has no `Linear*` wrapper — distinct
        // from `TQ2_0_g128` id 42 above). Deliberately NOT
        // `BonsaiError::NonExecutableQuantType`: that message is generated
        // from `is_executable()` and would list this exact type under
        // "executable types are …", contradicting itself.
        GgufTensorType::F32
        | GgufTensorType::F16
        | GgufTensorType::BF16
        | GgufTensorType::TQ2_0 => Err(ModelError::Internal(format!(
            "tensor '{sample_name}' uses quantization type {} (id {}): `dequant_any` can \
             decode it, but no transformer-block Linear*/attention-FFN kernel wrapper for it \
             exists in this build yet",
            resolved_type,
            resolved_type.wire_id(),
        ))),
        // Genuinely non-executable (`is_executable() == false`): the
        // generated "executable types are …" message is accurate here.
        // Named individually, not `_`, so adding a new `GgufTensorType`
        // variant is a compile error at this match until a human decides
        // which of the two arms above it belongs in.
        GgufTensorType::Q4_1
        | GgufTensorType::Q5_0
        | GgufTensorType::Q5_1
        | GgufTensorType::Q8_1
        | GgufTensorType::IQ2_XXS
        | GgufTensorType::IQ2_XS
        | GgufTensorType::IQ3_XXS
        | GgufTensorType::IQ1_S
        | GgufTensorType::IQ4_NL
        | GgufTensorType::IQ3_S
        | GgufTensorType::IQ2_S
        | GgufTensorType::IQ4_XS
        | GgufTensorType::I8
        | GgufTensorType::I16
        | GgufTensorType::I32
        | GgufTensorType::I64
        | GgufTensorType::F64
        | GgufTensorType::IQ1_M
        | GgufTensorType::TQ1_0
        | GgufTensorType::MXFP4
        | GgufTensorType::NVFP4 => Err(ModelError::Core(BonsaiError::non_executable_quant_type(
            sample_name.clone(),
            resolved_type,
        ))),
    }
}

/// Load the output (LM head) weight — may be Q1_0_g128 or FP32.
///
/// Falls back to the token embedding table (M-Missed-2) when `output.weight`
/// is absent, exactly as llama.cpp does: many Qwen3-family checkpoints tie
/// the LM head to the embedding table and never write a separate
/// `output.weight` tensor. GGUF stores both `output.weight` and
/// `token_embd.weight` in the same row-major `[in_features, out_features]`
/// (`[hidden_size, vocab_size]`) layout — row `v` is `hidden_size`
/// contiguous elements for both the embedding of token `v` and the LM-head
/// weight row that produces logit `v` — so the embedding tensor's raw bytes
/// are directly usable as the output projection's weight blocks with no
/// data transpose, only reuse of the one tensor for both roles.
///
/// `resolved_42` is [`resolve_id42_once`]'s per-file resolution of ggml wire
/// id 42, exactly as [`load_transformer_block`] takes it (B2-09) — the raw
/// `output.weight`/`token_embd.weight` tensor type is only ever the
/// parse-time guess for wire id 42, so matching on it directly would send a
/// PrismML gen-1 or mainline group-64 output tensor down the wrong decode
/// path instead of either its correct one or a loud "no wrapper" error.
pub(super) fn load_output_weight<'a>(
    gguf: &'a GgufFile<'a>,
    config: &Qwen3Config,
    kernel: &std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    resolved_42: Option<GgufTensorType>,
) -> ModelResult<OutputWeight<'a>> {
    let (name, info) = match gguf.tensors.get(tensor_names::OUTPUT) {
        Some(info) => (tensor_names::OUTPUT, info),
        None => {
            let info = gguf
                .tensors
                .require(tensor_names::TOKEN_EMBD)
                .map_err(ModelError::Core)?;
            tracing::info!("output.weight absent; tying LM head to token_embd.weight");
            (tensor_names::TOKEN_EMBD, info)
        }
    };
    let resolved_type = apply_resolved_type(info.tensor_type, resolved_42);

    // Derive actual output dimensions from the tensor shape rather than
    // config.vocab_size, which may reflect the tokenizer vocabulary rather
    // than the model's actual output projection size.
    let out_features = if info.shape.len() >= 2 {
        info.shape[1] as usize
    } else {
        config.vocab_size
    };
    let in_features = if !info.shape.is_empty() {
        info.shape[0] as usize
    } else {
        config.hidden_size
    };

    match resolved_type {
        GgufTensorType::Q1_0_g128 => {
            let blocks = load_1bit_blocks(gguf, name)?;
            let linear = Linear1Bit::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::OneBit(linear))
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = load_ternary_blocks(gguf, name)?;
            let linear = LinearTernary::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::Ternary(linear))
        }
        GgufTensorType::F8_E4M3 => {
            let blocks = load_fp8_e4m3_blocks(gguf, name)?;
            let linear = LinearFP8E4M3::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::FP8E4M3(linear))
        }
        GgufTensorType::F8_E5M2 => {
            let blocks = load_fp8_e5m2_blocks(gguf, name)?;
            let linear = LinearFP8E5M2::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::FP8E5M2(linear))
        }
        GgufTensorType::Q4_0 => {
            let blocks = load_q4_0_blocks(gguf, name)?;
            let linear = LinearQ4_0::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q4_0(linear))
        }
        GgufTensorType::Q8_0 => {
            let blocks = load_q8_0_blocks(gguf, name)?;
            let linear = LinearQ8_0::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q8_0(linear))
        }
        GgufTensorType::Q5_K => {
            let blocks = load_q5k_blocks(gguf, name)?;
            let linear = LinearQ5K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q5K(linear))
        }
        GgufTensorType::Q6_K => {
            let blocks = load_q6k_blocks(gguf, name)?;
            let linear = LinearQ6K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q6K(linear))
        }
        GgufTensorType::Q2_K => {
            let blocks = load_q2k_blocks(gguf, name)?;
            let linear = LinearQ2K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q2K(linear))
        }
        GgufTensorType::Q3_K => {
            let blocks = load_q3k_blocks(gguf, name)?;
            let linear = LinearQ3K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q3K(linear))
        }
        GgufTensorType::Q4_K => {
            let blocks = load_q4k_blocks(gguf, name)?;
            let linear = LinearQ4K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q4K(linear))
        }
        GgufTensorType::Q8_K => {
            let blocks = load_q8k_blocks(gguf, name)?;
            let linear = LinearQ8K::new(blocks, out_features, in_features)?;
            Ok(OutputWeight::Q8K(linear))
        }
        // BF16 has no dedicated `Linear*` kernel wrapper (K-12: a one-time
        // widen at load is enough — no shipped model uses BF16 for its
        // output/embedding tensor, only for the much smaller
        // `ssm_alpha`/`ssm_beta` per-layer tensors) — dequantize to a dense
        // FP32 projection through the same `dequant_any` table F32/F16 use.
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 => {
            let weights = load_f32_tensor(gguf, name)?;
            Ok(OutputWeight::Fp32 {
                weights,
                out_features,
                in_features,
            })
        }
        // `dequant_any` CAN decode these, but no `Linear*` output-projection
        // wrapper exists in this build yet. Deliberately NOT
        // `NonExecutableQuantType`: its generated "executable types are …"
        // list would include this exact type, contradicting itself.
        GgufTensorType::TQ2_0
        | GgufTensorType::Q2_0G64
        | GgufTensorType::Q2_0G128DFirst
        | GgufTensorType::PQ2_0
        | GgufTensorType::PTQ1_0 => Err(ModelError::Internal(format!(
            "output tensor '{name}' uses quantization type {} (id {}): `dequant_any` can \
             decode it, but no output-projection Linear* kernel wrapper for it exists in \
             this build yet",
            resolved_type,
            resolved_type.wire_id(),
        ))),
        // Genuinely non-executable (`is_executable() == false`): named
        // individually (not `_`), matching `dequant_any` and
        // `load_transformer_block` above, so a new `GgufTensorType` variant
        // is a compile error here too, not a silent fall-through.
        GgufTensorType::Q4_1
        | GgufTensorType::Q5_0
        | GgufTensorType::Q5_1
        | GgufTensorType::Q8_1
        | GgufTensorType::IQ2_XXS
        | GgufTensorType::IQ2_XS
        | GgufTensorType::IQ3_XXS
        | GgufTensorType::IQ1_S
        | GgufTensorType::IQ4_NL
        | GgufTensorType::IQ3_S
        | GgufTensorType::IQ2_S
        | GgufTensorType::IQ4_XS
        | GgufTensorType::I8
        | GgufTensorType::I16
        | GgufTensorType::I32
        | GgufTensorType::I64
        | GgufTensorType::F64
        | GgufTensorType::IQ1_M
        | GgufTensorType::TQ1_0
        | GgufTensorType::MXFP4
        | GgufTensorType::NVFP4 => Err(ModelError::Core(BonsaiError::non_executable_quant_type(
            name,
            resolved_type,
        ))),
    }
}

mod wiring;

// Load-time wiring extracted into a child module so this file stays under
// the 2000-line policy limit: `wiring` turns `Qwen3Config::rope_scaling`
// into a real `RopeTable` (M-08) and enforces the shape invariants a bad
// configuration would otherwise break silently (M-12). Re-exported here so
// every caller keeps addressing them as `weight_loaders::<name>`.
pub(super) use wiring::{
    build_rope_table, build_rope_table_or_unscaled, validate_config_shapes,
    validate_layer_tensor_shapes,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
