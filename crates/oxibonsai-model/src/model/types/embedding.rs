//! Token-embedding table with **row-wise** dequantization (M-02 / M-32 /
//! perf-04) and an O(1)-memory synthetic table for the config-only
//! constructors (M-33).
//!
//! ## Why this type exists
//!
//! `BonsaiModel` used to hold the whole embedding table as one eagerly
//! dequantized `Arc<[f32]>`. Measured resident cost of that table:
//!
//! | model                      | `token_embd.weight` | FP32 resident |
//! |----------------------------|---------------------|---------------|
//! | Ternary-Bonsai-1.7B        | `[2048, 151669]`    | 1.16 GiB      |
//! | Bonsai-8B / Ternary-8B     | `[4096, 151669]`    | 2.31 GiB      |
//! | Bonsai 2 27B               | `[5120, 248320]`    | 4.74 GiB      |
//!
//! A decode step reads exactly **one** row (5120 f32 = 20 KB for the 27B), so
//! the table is kept in its on-disk quantized form — borrowed zero-copy from
//! the memory map — and only the looked-up row is dequantized, straight into
//! the caller's scratch buffer ([`EmbeddingTable::copy_row`]). Nothing is
//! allocated per token.
//!
//! Every layout the **shipped and target** models actually use for
//! `token_embd.weight` gets the row-wise win: `Q1_0_g128` (41), `TQ2_0` (35),
//! `TQ2_0_g128` (42, `qs`-first), the `d`-first group-128 and group-64 layouts
//! that also travel under id 42, and PrismML's `PQ2_0` (142) / `PTQ1_0` (143).
//! Decoding always runs through `oxibonsai-core`'s own block decoder for the
//! format, so a row lookup cannot drift from the authoritative reader.
//!
//! This is **not** every layout the tree can read: a `token_embd.weight` of
//! `Q4_0`, `Q8_0`, any K-quant, `F8_E4M3`/`F8_E5M2` or `TQ1_0` — all formats
//! [`OutputWeight`](super::OutputWeight) loads for the *output* projection —
//! has no row-wise decoder here yet and still falls back to eager
//! [`load_f32_tensor`](crate::model::weight_loaders::load_f32_tensor) below.
//! None of the shipped or 27B-target GGUFs embed their token table that way
//! today; add a decoder here if one ever does.
//!
//! ## No dense escape hatch (M-02, wave 2.5)
//!
//! This type used to implement `Index<Range<usize>>`, which materialized the
//! whole FP32 table into a `OnceLock` on first use. Every batched GPU gather
//! in the sibling modules (`forward_metal.rs`, `forward_metal_fp8.rs`,
//! `forward_cuda/*.rs`, `forward_cuda_fp8.rs`) went through it — including
//! `try_metal_prefill_with_lm_head_ternary`, which **every** multi-token
//! prompt on a default Ternary-Bonsai Metal build reaches — so the measured
//! resident-set win of M-02 was zero by default and the materialization
//! simply moved from load time to first prefill, where it costs TTFT.
//!
//! Those sites now call [`EmbeddingTable::copy_row`] per token, exactly as the
//! decode path does, and the escape hatch is **gone**: there is no `Index`
//! impl, no `dense` `OnceLock` and no `dense_view`, so it cannot come back by
//! accident. [`EmbeddingTable::resident_bytes`] is therefore identically `0`
//! for every quantized table, for the whole life of the model. `len()` still
//! reports the *virtual* element count and never materializes anything.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_prism::{
    BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::{BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

use crate::error::{ModelError, ModelResult};

/// Backing storage for the embedding rows.
enum EmbeddingSource<'a> {
    /// Fully dequantized FP32 table, shared across engine-pool replicas.
    ///
    /// Used when the GGUF stores `token_embd.weight` as F32/F16 (or any type
    /// with no row-wise decoder here), and when the engine pool hands a
    /// pre-loaded table to [`crate::model::BonsaiModel::from_gguf_with_embd`].
    Dense(Arc<[f32]>),
    /// `Q1_0_g128` (ggml id 41) — `Bonsai-8B`.
    OneBit(&'a [BlockQ1_0G128]),
    /// `TQ2_0_g128` (id 42, `qs` first) — `Ternary-Bonsai-{1.7B,8B}`.
    Ternary(&'a [BlockTQ2_0_g128]),
    /// `TQ2_0` (id 35), group 256.
    TernaryG256(&'a [BlockTQ2_0]),
    /// PrismML `PQ2_0` (id 142) and the wire-identical `d`-first group-128
    /// layout that PrismML gen-1 files store under id 42.
    Prism2Bit(&'a [BlockPQ2_0]),
    /// Mainline `Q2_0` (id 42, group 64).
    Prism2BitG64(&'a [BlockQ2_0G64]),
    /// PrismML `PTQ1_0` (id 143), five base-3 trits per byte.
    PrismTrit(&'a [BlockPTQ1_0]),
    /// Every element of every row has this value — no storage at all.
    ///
    /// This is what the weightless constructors use (M-33): `new` used to
    /// allocate `vocab × hidden` f32 (2.5 GiB for the 8B config, 4.9 GiB
    /// together with the LM head for the 27B) just to read zeros back out.
    Constant(f32),
}

impl EmbeddingSource<'_> {
    /// Elements this source can supply, or `None` for a synthetic table.
    fn capacity(&self) -> Option<usize> {
        match self {
            Self::Dense(table) => Some(table.len()),
            Self::OneBit(blocks) => Some(blocks.len() * QK1_0_G128),
            Self::Ternary(blocks) => Some(blocks.len() * QK_TQ2_0_G128),
            Self::TernaryG256(blocks) => Some(blocks.len() * QK_TQ2_0),
            Self::Prism2Bit(blocks) => Some(blocks.len() * QK_PQ2_0),
            Self::Prism2BitG64(blocks) => Some(blocks.len() * QK_Q2_0_G64),
            Self::PrismTrit(blocks) => Some(blocks.len() * QK_PTQ1_0),
            Self::Constant(_) => None,
        }
    }

    /// Write `out.len()` elements starting at flat element index `start`.
    ///
    /// Total by construction: elements past the end of the source are written
    /// as `0.0`, which [`EmbeddingTable::check_capacity`] makes unreachable for
    /// any index below `vocab × hidden`.
    fn write_range(&self, start: usize, out: &mut [f32]) {
        match self {
            Self::Dense(table) => {
                let available = table.len().saturating_sub(start).min(out.len());
                out[..available].copy_from_slice(&table[start..start + available]);
                out[available..].fill(0.0);
            }
            Self::OneBit(blocks) => {
                // `BlockQ1_0G128` decodes per element, so no staging is needed.
                for (i, slot) in out.iter_mut().enumerate() {
                    let elem = start + i;
                    *slot = match blocks.get(elem / QK1_0_G128) {
                        Some(block) => block.weight(elem % QK1_0_G128),
                        None => 0.0,
                    };
                }
            }
            Self::Ternary(blocks) => {
                dequant_blockwise(QK_TQ2_0_G128, blocks.len(), start, out, |b, dst| {
                    BlockTQ2_0_g128::dequant(&blocks[b..b + 1], dst).is_ok()
                });
            }
            Self::TernaryG256(blocks) => {
                dequant_blockwise(QK_TQ2_0, blocks.len(), start, out, |b, dst| {
                    BlockTQ2_0::dequant(&blocks[b..b + 1], dst).is_ok()
                });
            }
            Self::Prism2Bit(blocks) => {
                dequant_blockwise(QK_PQ2_0, blocks.len(), start, out, |b, dst| {
                    BlockPQ2_0::dequant(&blocks[b..b + 1], dst).is_ok()
                });
            }
            Self::Prism2BitG64(blocks) => {
                dequant_blockwise(QK_Q2_0_G64, blocks.len(), start, out, |b, dst| {
                    BlockQ2_0G64::dequant(&blocks[b..b + 1], dst).is_ok()
                });
            }
            Self::PrismTrit(blocks) => {
                dequant_blockwise(QK_PTQ1_0, blocks.len(), start, out, |b, dst| {
                    BlockPTQ1_0::dequant(&blocks[b..b + 1], dst).is_ok()
                });
            }
            Self::Constant(value) => out.fill(*value),
        }
    }

    /// Short, stable name of this layout, for load-time diagnostics.
    fn kind(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense-f32",
            Self::OneBit(_) => "Q1_0_g128",
            Self::Ternary(_) => "TQ2_0_g128",
            Self::TernaryG256(_) => "TQ2_0",
            Self::Prism2Bit(_) => "PQ2_0",
            Self::Prism2BitG64(_) => "Q2_0_g64",
            Self::PrismTrit(_) => "PTQ1_0",
            Self::Constant(_) => "synthetic",
        }
    }
}

/// Largest group size of any block format decoded here (`TQ2_0` = 256), so the
/// per-block staging buffer in [`dequant_blockwise`] can live on the stack.
const MAX_GROUP: usize = QK_TQ2_0;

/// Decode `out.len()` elements starting at flat element index `start`, one
/// group at a time, through the format's own block decoder.
///
/// `fill_block(block_idx, dst)` fills `dst` (exactly `qk` elements) and reports
/// success. Rows need not be group-aligned: the leading and trailing groups are
/// decoded into a stack buffer and sliced. Nothing is allocated, which is what
/// keeps `forward_into` allocation-free.
fn dequant_blockwise<F>(
    qk: usize,
    n_blocks: usize,
    start: usize,
    out: &mut [f32],
    mut fill_block: F,
) where
    F: FnMut(usize, &mut [f32]) -> bool,
{
    debug_assert!(qk > 0 && qk <= MAX_GROUP);
    let qk = qk.clamp(1, MAX_GROUP);
    let mut group = [0.0f32; MAX_GROUP];
    let mut written = 0usize;
    while written < out.len() {
        let elem = start + written;
        let block_idx = elem / qk;
        let lane = elem % qk;
        let take = (qk - lane).min(out.len() - written);
        let dst = &mut group[..qk];
        if block_idx >= n_blocks || !fill_block(block_idx, dst) {
            dst.fill(0.0);
        }
        out[written..written + take].copy_from_slice(&dst[lane..lane + take]);
        written += take;
    }
}

/// Token-embedding table (`[vocab_size × hidden_size]`, row-major).
pub(super) struct EmbeddingTable<'a> {
    src: EmbeddingSource<'a>,
    vocab: usize,
    hidden: usize,
}

impl<'a> EmbeddingTable<'a> {
    /// Wrap an already-dequantized table.
    ///
    /// Holds **exactly one** clone of `table` (in `src`). The engine pool's
    /// identity check counts these references, so a second internal clone
    /// would be visible to it.
    pub(super) fn dense(table: Arc<[f32]>, vocab: usize, hidden: usize) -> Self {
        Self {
            src: EmbeddingSource::Dense(table),
            vocab,
            hidden,
        }
    }

    /// Synthesize a table whose every element is `value`, with no storage.
    pub(super) fn constant(value: f32, vocab: usize, hidden: usize) -> Self {
        Self {
            src: EmbeddingSource::Constant(value),
            vocab,
            hidden,
        }
    }

    /// Build the table for `token_embd.weight` of `gguf`, keeping it quantized
    /// whenever a row-wise decoder exists for its layout.
    ///
    /// Falls back to the eager dequantization of
    /// [`load_f32_tensor`](crate::model::weight_loaders::load_f32_tensor) for
    /// any other tensor type, which preserves the previous behaviour bit for
    /// bit.
    pub(super) fn from_gguf(
        gguf: &'a GgufFile<'a>,
        vocab: usize,
        hidden: usize,
    ) -> ModelResult<Self> {
        let name = tensor_names::TOKEN_EMBD;
        let info = gguf.tensors.require(name).map_err(ModelError::Core)?;
        let needed = vocab
            .checked_mul(hidden)
            .ok_or_else(|| ModelError::Internal(format!("{name}: vocab*hidden overflows usize")))?;
        let quantized = Self::quantized_source(gguf, name, info.tensor_type)?;
        match quantized {
            Some(src) => {
                Self::check_capacity(name, src.capacity().unwrap_or(needed), needed)?;
                tracing::debug!(
                    tensor = name,
                    layout = src.kind(),
                    elements = needed,
                    "token embedding kept quantized; rows are dequantized on lookup"
                );
                Ok(Self { src, vocab, hidden })
            }
            None => {
                let table = crate::model::weight_loaders::load_f32_tensor(gguf, name)?;
                Self::check_capacity(name, table.len(), needed)?;
                tracing::debug!(
                    tensor = name,
                    tensor_type = ?info.tensor_type,
                    "no row-wise decoder for this layout; token embedding dequantized eagerly"
                );
                Ok(Self::dense(table.into(), vocab, hidden))
            }
        }
    }

    /// Borrow `name`'s blocks for a layout that can be decoded row-wise, or
    /// `None` when the caller must fall back to eager dequantization.
    fn quantized_source(
        gguf: &'a GgufFile<'a>,
        name: &str,
        tensor_type: GgufTensorType,
    ) -> ModelResult<Option<EmbeddingSource<'a>>> {
        // Only touch the tensor bytes for a layout we can actually decode.
        if !matches!(
            tensor_type,
            GgufTensorType::Q1_0_g128
                | GgufTensorType::TQ2_0_g128
                | GgufTensorType::TQ2_0
                | GgufTensorType::PQ2_0
                | GgufTensorType::Q2_0G128DFirst
                | GgufTensorType::Q2_0G64
                | GgufTensorType::PTQ1_0
        ) {
            return Ok(None);
        }
        let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
        let src = match tensor_type {
            GgufTensorType::Q1_0_g128 => EmbeddingSource::OneBit(
                BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            GgufTensorType::TQ2_0_g128 => EmbeddingSource::Ternary(
                BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            GgufTensorType::TQ2_0 => EmbeddingSource::TernaryG256(
                BlockTQ2_0::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            // `Q2_0G128DFirst` is wire-identical to `PQ2_0`; only the ggml id
            // differs (see `gguf::types`), so one decoder serves both.
            GgufTensorType::PQ2_0 | GgufTensorType::Q2_0G128DFirst => EmbeddingSource::Prism2Bit(
                BlockPQ2_0::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            GgufTensorType::Q2_0G64 => EmbeddingSource::Prism2BitG64(
                BlockQ2_0G64::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            GgufTensorType::PTQ1_0 => EmbeddingSource::PrismTrit(
                BlockPTQ1_0::slice_from_bytes(data).map_err(ModelError::Core)?,
            ),
            other => {
                return Err(ModelError::Internal(format!(
                    "{name}: {other} passed the row-wise layout filter but has no decoder"
                )))
            }
        };
        Ok(Some(src))
    }

    /// Reject a table that cannot cover `vocab × hidden` elements.
    ///
    /// Checked once at load so that [`copy_row`](Self::copy_row) and the dense
    /// materialization in `Index` are total functions afterwards.
    fn check_capacity(name: &str, have: usize, needed: usize) -> ModelResult<()> {
        if have < needed {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![needed],
                actual: vec![have],
            });
        }
        Ok(())
    }

    /// Virtual element count (`vocab × hidden`) — never materializes.
    pub(super) fn len(&self) -> usize {
        self.vocab.saturating_mul(self.hidden)
    }

    /// `true` when the table holds no elements (`vocab == 0 || hidden == 0`).
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shared FP32 handle, if (and only if) one already exists.
    ///
    /// Returns `None` for a quantized or synthetic table: those have no dense
    /// allocation to share, and this accessor deliberately does **not**
    /// materialize one (that is what makes the engine-pool seam cheap for
    /// quantized models — every replica borrows the same memory map).
    pub(super) fn dense_handle(&self) -> Option<Arc<[f32]>> {
        match &self.src {
            EmbeddingSource::Dense(table) => Some(Arc::clone(table)),
            _ => None,
        }
    }

    /// Resident bytes this table owns.
    ///
    /// Blocks borrowed from the memory map and the synthetic
    /// [`EmbeddingSource::Constant`] table cost nothing, so this is
    /// identically `0` for every quantized layout — for the whole life of the
    /// model, because no consumer can materialize one any more (M-02, wave
    /// 2.5: the `Index<Range<usize>>` escape hatch is gone). Only a table the
    /// GGUF stores dense, or one handed in by the engine pool, costs its FP32
    /// bytes; that allocation is shared across replicas, so the same bytes are
    /// reported by each replica holding it.
    pub(super) fn resident_bytes(&self) -> usize {
        let elements = match &self.src {
            EmbeddingSource::Dense(table) => table.len(),
            _ => 0,
        };
        elements * std::mem::size_of::<f32>()
    }

    /// Copy the embedding row of `token_id` into `out[..hidden]`.
    ///
    /// This is the per-token path: it dequantizes exactly one row and allocates
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] if `out` is shorter than `hidden_size`;
    /// [`ModelError::MissingTensor`] if `token_id` is outside the vocabulary.
    pub(super) fn copy_row(&self, token_id: u32, out: &mut [f32]) -> ModelResult<()> {
        let hidden = self.hidden;
        if out.len() < hidden {
            return Err(ModelError::ShapeMismatch {
                name: "token_embd row destination".to_string(),
                expected: vec![hidden],
                actual: vec![out.len()],
            });
        }
        let token = token_id as usize;
        if token >= self.vocab {
            return Err(ModelError::MissingTensor {
                name: format!("token_id {token_id} out of range (vocab={})", self.vocab),
            });
        }
        self.src.write_range(token * hidden, &mut out[..hidden]);
        Ok(())
    }

    /// Copy the embedding rows of `token_ids` into `out`, one contiguous
    /// `hidden`-element row each (M-02, wave 2.5).
    ///
    /// This is what the batched GPU gather paths call instead of the deleted
    /// dense `Index` hatch: it allocates nothing and touches only the
    /// `token_ids.len()` rows actually referenced, rather than materializing
    /// `vocab × hidden` FP32 elements (1.16 / 2.31 / 4.74 GiB for the
    /// 1.7B / 8B / 27B) to read a few hundred of them.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] if `out` is shorter than
    /// `token_ids.len() * hidden_size`; [`ModelError::MissingTensor`] if any
    /// token id is outside the vocabulary.
    ///
    /// Compiled only where it is reachable. Every batched embedding gather in
    /// the tree lives in the Metal or CUDA forward modules, which are
    /// themselves feature- and target-gated; a plain CPU build has no batched
    /// gather at all (`forward_sequential` walks the prompt one token at a
    /// time through [`copy_row`](Self::copy_row)), so an unconditional
    /// definition would be dead code there. The `test` arm keeps the parity
    /// assertions that pin this against the format's own whole-table
    /// `dequant` compiling on every host.
    #[cfg(any(
        test,
        all(feature = "metal", target_os = "macos"),
        all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        )
    ))]
    pub(super) fn copy_rows(&self, token_ids: &[u32], out: &mut [f32]) -> ModelResult<()> {
        let hidden = self.hidden;
        let needed = token_ids.len().saturating_mul(hidden);
        if out.len() < needed {
            return Err(ModelError::ShapeMismatch {
                name: "token_embd batch destination".to_string(),
                expected: vec![needed],
                actual: vec![out.len()],
            });
        }
        for (t, &token_id) in token_ids.iter().enumerate() {
            self.copy_row(token_id, &mut out[t * hidden..(t + 1) * hidden])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Deterministic, non-degenerate table values (mixed signs and magnitudes
    /// so every quantizer uses more than one code).
    fn table_values(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.031)
            .collect()
    }

    /// Row-wise lookup must agree, element for element, with the format's own
    /// whole-table `dequant` — at a group-aligned `hidden` and at one that
    /// straddles group boundaries.
    macro_rules! row_parity {
        ($name:ident, $block:ty, $qk:expr, $variant:ident) => {
            #[test]
            fn $name() {
                for &hidden in &[$qk, 100usize] {
                    let vocab = 5usize;
                    let total = vocab * hidden;
                    let padded = total.div_ceil($qk) * $qk;
                    let blocks = <$block>::quantize(&table_values(padded)).expect("quantize");
                    let mut reference = vec![0.0f32; blocks.len() * $qk];
                    <$block>::dequant(&blocks, &mut reference).expect("dequant");

                    let table = EmbeddingTable {
                        src: EmbeddingSource::$variant(&blocks),
                        vocab,
                        hidden,
                    };
                    let mut row = vec![0.0f32; hidden];
                    for token in 0..vocab {
                        table.copy_row(token as u32, &mut row).expect("row");
                        assert_eq!(
                            row.as_slice(),
                            &reference[token * hidden..(token + 1) * hidden],
                            "{}: hidden={hidden} token={token}",
                            stringify!($name),
                        );
                    }
                    // The batched gather must produce exactly the same numbers
                    // as the whole-table `dequant` it replaced (M-02, wave 2.5
                    // — this is the assertion the deleted dense `Index` hatch
                    // used to carry), and cost no resident bytes doing it.
                    let ids: Vec<u32> = (0..vocab as u32).collect();
                    let mut batch = vec![0.0f32; total];
                    table.copy_rows(&ids, &mut batch).expect("batch");
                    assert_eq!(batch.as_slice(), &reference[..total]);
                    assert_eq!(table.resident_bytes(), 0);
                }
            }
        };
    }

    row_parity!(
        ternary_g128_rows_match_whole_table,
        BlockTQ2_0_g128,
        QK_TQ2_0_G128,
        Ternary
    );
    row_parity!(
        ternary_g256_rows_match_whole_table,
        BlockTQ2_0,
        QK_TQ2_0,
        TernaryG256
    );
    row_parity!(
        prism_pq2_0_rows_match_whole_table,
        BlockPQ2_0,
        QK_PQ2_0,
        Prism2Bit
    );
    row_parity!(
        prism_q2_0_g64_rows_match_whole_table,
        BlockQ2_0G64,
        QK_Q2_0_G64,
        Prism2BitG64
    );
    row_parity!(
        prism_ptq1_0_rows_match_whole_table,
        BlockPTQ1_0,
        QK_PTQ1_0,
        PrismTrit
    );

    /// `Q1_0_g128` has no `quantize`/`dequant` pair, so the reference here is
    /// the documented formula itself (bit set -> `+d`, clear -> `-d`), matching
    /// `weight_loaders::load_f32_tensor`.
    #[test]
    fn one_bit_rows_match_the_documented_formula() {
        for &hidden in &[QK1_0_G128, 100usize] {
            let vocab = 5usize;
            let total = vocab * hidden;
            let n_blocks = total.div_ceil(QK1_0_G128);
            let blocks: Vec<BlockQ1_0G128> = (0..n_blocks)
                .map(|i| BlockQ1_0G128 {
                    d: f16::from_f32(0.25 + i as f32 * 0.125),
                    qs: std::array::from_fn(|j| ((i * 31 + j * 17) % 256) as u8),
                })
                .collect();
            let mut reference = vec![0.0f32; n_blocks * QK1_0_G128];
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                for j in 0..QK1_0_G128 {
                    let bit = (block.qs[j / 8] >> (j % 8)) & 1;
                    reference[i * QK1_0_G128 + j] = if bit != 0 { d } else { -d };
                }
            }

            let table = EmbeddingTable {
                src: EmbeddingSource::OneBit(&blocks),
                vocab,
                hidden,
            };
            let mut row = vec![0.0f32; hidden];
            for token in 0..vocab {
                table.copy_row(token as u32, &mut row).expect("row");
                assert_eq!(
                    row.as_slice(),
                    &reference[token * hidden..(token + 1) * hidden],
                    "hidden={hidden} token={token}"
                );
            }
            let ids: Vec<u32> = (0..vocab as u32).collect();
            let mut batch = vec![0.0f32; total];
            table.copy_rows(&ids, &mut batch).expect("batch");
            assert_eq!(batch.as_slice(), &reference[..total]);
            assert_eq!(table.resident_bytes(), 0);
        }
    }

    /// The batched gather bounds-checks both the destination and every token
    /// id, instead of panicking the way the deleted `Index` hatch's slice
    /// would have (M-02 / M-29).
    #[test]
    fn batched_gather_is_bounds_checked() {
        let blocks = BlockPQ2_0::quantize(&table_values(4 * QK_PQ2_0)).expect("quantize");
        let table = EmbeddingTable {
            src: EmbeddingSource::Prism2Bit(&blocks),
            vocab: 4,
            hidden: QK_PQ2_0,
        };
        let mut short = vec![0.0f32; QK_PQ2_0];
        assert!(
            table.copy_rows(&[0, 1], &mut short).is_err(),
            "a destination shorter than tokens*hidden must be rejected"
        );
        let mut out = vec![0.0f32; 2 * QK_PQ2_0];
        assert!(
            table.copy_rows(&[0, 4], &mut out).is_err(),
            "a token id at or beyond the vocabulary must be rejected"
        );
        assert!(table.copy_rows(&[0, 3], &mut out).is_ok());
    }

    /// A source shorter than `vocab * hidden` never reads out of bounds; the
    /// load-time capacity check is what keeps this unreachable in practice.
    #[test]
    fn a_short_source_zero_fills_instead_of_panicking() {
        let blocks = BlockPQ2_0::quantize(&table_values(QK_PQ2_0)).expect("quantize");
        let table = EmbeddingTable {
            src: EmbeddingSource::Prism2Bit(&blocks),
            vocab: 4,
            hidden: QK_PQ2_0,
        };
        let mut row = vec![7.0f32; QK_PQ2_0];
        table.copy_row(3, &mut row).expect("row past the blocks");
        assert!(row.iter().all(|&v| v == 0.0), "{:?}", &row[..8]);
    }

    #[test]
    fn capacity_check_rejects_a_truncated_table() {
        assert!(EmbeddingTable::check_capacity("t", 10, 20).is_err());
        assert!(EmbeddingTable::check_capacity("t", 20, 20).is_ok());
    }
}
