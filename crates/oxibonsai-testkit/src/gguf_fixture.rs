//! One shared GGUF fixture builder covering every quant format (T-07).
//!
//! Before this crate existed, 13 test files across 4 crates each hand-rolled
//! their own synthetic-GGUF byte builder (three of them literally named
//! `build_synthetic_ternary_gguf` with subtly divergent bodies). This module
//! is the single replacement: a deterministic weight generator, a
//! format-dispatching quantizer covering every [`FixtureQuant`] this
//! workspace's kernels execute, and a small [`GgufFixtureBuilder`] on top of
//! [`oxibonsai_core::GgufWriter`] for tests that need a complete, parseable
//! GGUF file rather than raw block bytes.
//!
//! # Cross-crate wiring
//!
//! `oxibonsai-testkit` is a workspace member and a dev-dependency of the
//! crates whose tests use it; any remaining hand-rolled builder elsewhere is
//! a candidate to re-point at this module, which is the complete, tested,
//! canonical implementation.

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::{
    BlockFP8E4M3, BlockFP8E5M2, BlockPQ2_0, BlockPTQ1_0, BlockQ2K, BlockQ2_0G64, BlockQ3K,
    BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, BlockTQ2_0, BlockTQ2_0_g128,
    GgufWriter, MetadataWriteValue, TensorEntry, TensorType, WriteError, QK_FP8, QK_PQ2_0,
    QK_PTQ1_0, QK_Q2_0_G64, QK_Q4_0, QK_Q8_0, QK_TQ2_0, QK_TQ2_0_G128,
};

/// K-quant super-block size. Not re-exported from `oxibonsai_core` (each
/// `gemv_q*k.rs` kernel hardcodes it as a local `const QK_K: usize = 256`);
/// mirrored here for the same reason.
const QK_K: usize = 256;

// ─── Deterministic PRNG ─────────────────────────────────────────────────────

/// A small, deterministic, seedable PRNG shared by every fixture builder in
/// this workspace, replacing the ad-hoc `state = state.wrapping_mul(...)`
/// LCGs that used to be hand-copied into each of the 13 duplicate builders
/// (T-07) — including, in three of them, a bug where `(state >> 33) as u8`
/// was used directly as a packed ternary byte, landing on the reserved
/// `0b11` two-bit code in about a quarter of lanes (CQ-14). Use
/// [`Lcg::next_valid_tq2_byte`] for ternary-coded fixtures to avoid that
/// class of bug entirely.
#[derive(Debug, Clone)]
pub struct Lcg(u64);

impl Lcg {
    /// A generator seeded with `seed`. Zero is remapped to a fixed non-zero
    /// constant: zero is a fixed point of this LCG (`0 * mult + 0 == 0`
    /// forever), which would silently make every "random" value identical.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Advance the generator and return the next raw 64-bit state.
    ///
    /// Same multiplier/increment as every ad-hoc LCG this fixture replaces,
    /// kept identical on purpose: porting a call site to this shared
    /// generator with the same seed reproduces byte-identical fixtures.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }

    /// A pseudo-random `f32` in `[-1.0, 1.0]`.
    pub fn next_signed_unit_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32 & 0x00FF_FFFF; // 24 bits of entropy
        let unit = f64::from(bits) / f64::from(0x00FF_FFFF_u32); // [0.0, 1.0], f64 for precision
        unit.mul_add(2.0, -1.0).clamp(-1.0, 1.0) as f32
    }

    /// A pseudo-random byte.
    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }

    /// One ternary 2-bit lane in `{0, 1, 2}` — never the reserved `3`
    /// (`0b11`), which `TQ2_0`/`PQ2_0`-family kernels treat as invalid.
    pub fn next_ternary_lane(&mut self) -> u8 {
        ((self.next_u64() >> 33) % 3) as u8
    }

    /// One byte packing four valid ternary lanes (CQ-14-safe: every lane in
    /// `{0, 1, 2}`), matching
    /// `crates/oxibonsai-model/src/model/types/gpu_cache.rs::tq2_pattern`.
    pub fn next_valid_tq2_byte(&mut self) -> u8 {
        let mut byte = 0u8;
        for lane in 0..4u8 {
            byte |= self.next_ternary_lane() << (2 * lane);
        }
        byte
    }
}

/// `n` deterministic weights in `[-1.0, 1.0]`, seeded by `seed`.
///
/// Every call with the same `(n, seed)` produces byte-identical output,
/// across processes and across runs — the property every duplicate fixture
/// builder this replaces relied on individually.
#[must_use]
pub fn deterministic_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_signed_unit_f32()).collect()
}

// ─── Quant format dispatch ──────────────────────────────────────────────────

/// Every quant format this workspace's kernels can execute (mirrors
/// `oxibonsai_core::GgufTensorType::is_executable`), plus the two
/// unquantized element types. One [`quantize_bytes`] call covers all of
/// them, replacing the 13 format-specific hand-rolled builders (T-07).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum FixtureQuant {
    F32,
    F16,
    /// PrismML 1-bit, 128-element groups (ggml id 41).
    Q1_0G128,
    /// Legacy OxiBonsai ternary, `qs` first / `d` last (ggml id 42, as used
    /// by `models/Ternary-Bonsai-{1.7B,8B}.gguf`).
    TQ2_0_g128,
    /// llama.cpp upstream ternary, 256-element groups (ggml id 35).
    TQ2_0,
    /// 4-bit, 32-element groups (ggml id 2).
    Q4_0,
    /// 8-bit, 32-element groups (ggml id 8).
    Q8_0,
    /// K-quant, 256-element super-blocks (ggml ids 10/11/12/13/14/15).
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    /// PrismML FP8 E4M3FN (ggml id 43).
    F8E4M3,
    /// PrismML FP8 E5M2 (ggml id 44).
    F8E5M2,
    /// PrismML Bonsai-2 `PQ2_0`, `d` first (ggml id 142).
    PQ2_0,
    /// PrismML Bonsai-2 `PTQ1_0`, `d` last (ggml id 143).
    PTQ1_0,
    /// Mainline `block_q2_0`, group 64, `d` first (stored under ggml id 42).
    Q2_0G64,
}

impl FixtureQuant {
    /// Elements per quantized block (1 for the unquantized types).
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q1_0G128 => QK1_0_G128,
            Self::TQ2_0_g128 => QK_TQ2_0_G128,
            Self::TQ2_0 => QK_TQ2_0,
            Self::Q4_0 => QK_Q4_0,
            Self::Q8_0 => QK_Q8_0,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => QK_K,
            Self::F8E4M3 | Self::F8E5M2 => QK_FP8,
            Self::PQ2_0 => QK_PQ2_0,
            Self::PTQ1_0 => QK_PTQ1_0,
            Self::Q2_0G64 => QK_Q2_0_G64,
        }
    }

    /// The `oxibonsai_core::TensorType` this format serialises as when
    /// writing a full GGUF file with [`GgufFixtureBuilder`].
    ///
    /// Every format has one, including the six K-quants (ggml ids 10-15): the
    /// writer's `TensorType` carries `Q2_K`, `Q3_K` and `Q8_K` next to
    /// `Q4_K`/`Q5_K`/`Q6_K`, so a fixture file can embed any of them.
    #[must_use]
    pub const fn writer_type(self) -> TensorType {
        match self {
            Self::F32 => TensorType::F32,
            Self::F16 => TensorType::F16,
            Self::Q1_0G128 => TensorType::Q1_0G128,
            Self::TQ2_0_g128 => TensorType::TQ2_0_g128,
            Self::TQ2_0 => TensorType::TQ2_0,
            Self::Q4_0 => TensorType::Q4_0,
            Self::Q8_0 => TensorType::Q8_0,
            Self::Q2K => TensorType::Q2_K,
            Self::Q3K => TensorType::Q3_K,
            Self::Q4K => TensorType::Q4_K,
            Self::Q5K => TensorType::Q5_K,
            Self::Q6K => TensorType::Q6_K,
            Self::Q8K => TensorType::Q8_K,
            Self::F8E4M3 => TensorType::F8_E4M3,
            Self::F8E5M2 => TensorType::F8_E5M2,
            Self::PQ2_0 => TensorType::PQ2_0,
            Self::PTQ1_0 => TensorType::PTQ1_0,
            Self::Q2_0G64 => TensorType::Q2_0G64,
        }
    }
}

/// Errors a fixture builder can produce.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// A `Block*::quantize`/pack call failed (e.g. `weights.len()` was not a
    /// multiple of the format's block size).
    #[error("quantizing to {quant:?} failed: {source}")]
    Quantize {
        quant: FixtureQuant,
        #[source]
        source: oxibonsai_core::BonsaiError,
    },
    /// `weights.len()` was not a multiple of `quant.block_size()`, checked
    /// by this crate before calling into a format with no such internal
    /// check of its own (currently only [`FixtureQuant::Q1_0G128`]).
    #[error("{len} weights is not a multiple of {quant:?}'s block size ({block_size})")]
    NotBlockAligned {
        quant: FixtureQuant,
        len: usize,
        block_size: usize,
    },
    /// [`oxibonsai_core::GgufWriter::write`] itself failed.
    #[error(transparent)]
    Write(#[from] WriteError),
}

/// Reinterpret a slice of plain-old-data quantized blocks as raw bytes.
///
/// # Safety contract
///
/// Every `Block*` type in `oxibonsai-core` is `#[repr(C)]`, `Copy`, and
/// carries a `const _: () = assert!(size_of::<Self>() == BLOCK_*_BYTES);`
/// guarding that it has no trailing padding that would matter — the same
/// invariant `crates/oxibonsai-kernels/tests/metal_k_quant_gemv_parity.rs`
/// and `crates/oxibonsai-kernels/tests/cuda_tq2_gemv_parity.rs` already rely
/// on at their kernel-call boundary. This is the one place that pattern is
/// implemented instead of re-copy-pasted.
fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
    let ptr = blocks.as_ptr().cast::<u8>();
    let len = std::mem::size_of_val(blocks);
    // SAFETY: `ptr` is derived from a live `&[T]` of length `blocks.len()`,
    // so `ptr..ptr+len` (`len == blocks.len() * size_of::<T>()`) is entirely
    // within that allocation; `T: Copy` and the block-level invariant above
    // mean every byte in range is initialized. `u8` has no alignment
    // requirement, so an arbitrary alignment of `ptr` is not a soundness
    // concern for this read (unlike casting the other direction back to
    // `&[T]`, which callers of this module never do with these bytes).
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

/// Pack `weights` into [`BlockQ1_0G128`] blocks (128 weights per block).
///
/// `oxibonsai_core::tensor` exposes no encoder for this format (production
/// code only ever *reads* Q1\_0\_g128 files written by PrismML's own
/// tooling), so this fixture packer defines its own convention: the
/// per-block scale is the mean absolute weight in that block (falling back
/// to `1.0` for an all-zero block, so `d` is never zero — a zero scale would
/// make every reconstructed weight exactly zero regardless of its sign
/// bit), and each bit is `1` for a non-negative weight, `0` for negative.
///
/// # Errors
///
/// Returns [`FixtureError::NotBlockAligned`] if `weights.len()` is not a
/// multiple of 128.
pub fn pack_q1_0_g128(weights: &[f32]) -> Result<Vec<BlockQ1_0G128>, FixtureError> {
    if weights.is_empty() || !weights.len().is_multiple_of(QK1_0_G128) {
        return Err(FixtureError::NotBlockAligned {
            quant: FixtureQuant::Q1_0G128,
            len: weights.len(),
            block_size: QK1_0_G128,
        });
    }
    let mut blocks = Vec::with_capacity(weights.len() / QK1_0_G128);
    for group in weights.chunks_exact(QK1_0_G128) {
        let mean_abs = group.iter().map(|w| w.abs()).sum::<f32>() / group.len() as f32;
        let scale = if mean_abs > 0.0 { mean_abs } else { 1.0 };
        let mut qs = [0u8; QK1_0_G128 / 8];
        for (i, &w) in group.iter().enumerate() {
            if w >= 0.0 {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        blocks.push(BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs,
        });
    }
    Ok(blocks)
}

/// Quantize `weights` into raw on-disk block bytes for `quant`, dispatching
/// to each format's own `Block*::quantize`. This is the "one fixture
/// builder covering every quant format" T-07 asks for at the block-bytes
/// level; [`GgufFixtureBuilder`] wraps it to also produce a complete file.
///
/// # Errors
///
/// Propagates the underlying `Block*::quantize` error (typically: `weights`
/// not a multiple of `quant.block_size()`).
pub fn quantize_bytes(quant: FixtureQuant, weights: &[f32]) -> Result<Vec<u8>, FixtureError> {
    let wrap = |r: oxibonsai_core::BonsaiResult<Vec<u8>>| {
        r.map_err(|source| FixtureError::Quantize { quant, source })
    };
    match quant {
        FixtureQuant::F32 => Ok(weights.iter().flat_map(|w| w.to_le_bytes()).collect()),
        FixtureQuant::F16 => Ok(weights
            .iter()
            .flat_map(|w| f16::from_f32(*w).to_le_bytes())
            .collect()),
        FixtureQuant::Q1_0G128 => Ok(blocks_to_bytes(&pack_q1_0_g128(weights)?)),
        FixtureQuant::TQ2_0_g128 => {
            wrap(BlockTQ2_0_g128::quantize(weights).map(|b| blocks_to_bytes(&b)))
        }
        FixtureQuant::TQ2_0 => wrap(BlockTQ2_0::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q4_0 => wrap(BlockQ4_0::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q8_0 => wrap(BlockQ8_0::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q2K => wrap(BlockQ2K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q3K => wrap(BlockQ3K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q4K => wrap(BlockQ4K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q5K => wrap(BlockQ5K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q6K => wrap(BlockQ6K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q8K => wrap(BlockQ8K::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::F8E4M3 => wrap(BlockFP8E4M3::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::F8E5M2 => wrap(BlockFP8E5M2::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::PQ2_0 => wrap(BlockPQ2_0::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::PTQ1_0 => wrap(BlockPTQ1_0::quantize(weights).map(|b| blocks_to_bytes(&b))),
        FixtureQuant::Q2_0G64 => wrap(BlockQ2_0G64::quantize(weights).map(|b| blocks_to_bytes(&b))),
    }
}

// ─── Full-GGUF-file builder ─────────────────────────────────────────────────

/// Builds a complete, parseable GGUF file from deterministic tensors of any
/// [`FixtureQuant`], on top of `oxibonsai_core::GgufWriter`.
///
/// ```
/// use oxibonsai_testkit::gguf_fixture::{FixtureQuant, GgufFixtureBuilder};
///
/// // Shape is GGUF order (`ne0` first / innermost); `ne0` must be a
/// // multiple of the format's block size (128 for TQ2_0_g128) because
/// // ggml quantizes each row independently.
/// let bytes = GgufFixtureBuilder::new()
///     .metadata_str("general.architecture", "qwen3")
///     .tensor("token_embd.weight", &[128, 8], FixtureQuant::TQ2_0_g128, 1)
///     .expect("add tensor")
///     .tensor("output_norm.weight", &[8], FixtureQuant::F32, 2)
///     .expect("add tensor")
///     .build()
///     .expect("build gguf");
/// assert!(bytes.starts_with(b"GGUF"));
/// ```
pub struct GgufFixtureBuilder<'a> {
    writer: GgufWriter<'a>,
}

// `GgufWriter` does not implement `Debug`, so this cannot be `#[derive]`d;
// hand-written instead of omitted so `Result<&mut Self, _>::expect_err` (a
// natural thing for a test to call) has a `Debug` bound to satisfy.
impl std::fmt::Debug for GgufFixtureBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFixtureBuilder")
            .field("tensor_count", &self.tensor_count())
            .finish()
    }
}

impl<'a> GgufFixtureBuilder<'a> {
    /// A new, empty builder (default alignment: 32 bytes, matching
    /// `GgufWriter::new`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: GgufWriter::new(),
        }
    }

    /// Append a string metadata key-value pair.
    pub fn metadata_str(&mut self, key: &str, value: &str) -> &mut Self {
        self.writer
            .add_metadata(key, MetadataWriteValue::Str(value.to_string()));
        self
    }

    /// Append a `u32` metadata key-value pair (most `qwen3`/`qwen35`
    /// integer hyperparameters — `block_count`, `embedding_length`, … —
    /// are stored as this type).
    pub fn metadata_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.writer
            .add_metadata(key, MetadataWriteValue::U32(value));
        self
    }

    /// Append an `f32` metadata key-value pair.
    pub fn metadata_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.writer
            .add_metadata(key, MetadataWriteValue::F32(value));
        self
    }

    /// Append an arbitrary metadata key-value pair — the escape hatch for
    /// any [`MetadataWriteValue`] variant the typed helpers above don't
    /// cover (arrays, bools, …), such as the `qwen35`/`prism.hadamard.*`
    /// array and bool keys [`crate::qwen35_fixture::synthetic_qwen35_gguf`]
    /// needs.
    pub fn metadata(&mut self, key: &str, value: MetadataWriteValue) -> &mut Self {
        self.writer.add_metadata(key, value);
        self
    }

    /// Append a tensor named `name` with shape `shape` (GGUF order: the
    /// fastest-varying / innermost dimension first), filled with
    /// [`deterministic_weights`] seeded by `seed` and quantized to `quant`.
    ///
    /// # Errors
    ///
    /// - Whatever [`quantize_bytes`] returns for a shape whose element count
    ///   is not a multiple of `quant.block_size()`.
    pub fn tensor(
        &mut self,
        name: &str,
        shape: &[u64],
        quant: FixtureQuant,
        seed: u64,
    ) -> Result<&mut Self, FixtureError> {
        let tensor_type = quant.writer_type();
        let element_count: u64 = shape.iter().product();
        let weights = deterministic_weights(element_count as usize, seed);
        let data = quantize_bytes(quant, &weights)?;
        self.writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: shape.to_vec(),
            tensor_type,
            data,
        });
        Ok(self)
    }

    /// Append a tensor from caller-supplied raw bytes (e.g. a hand-crafted
    /// malformed block).
    pub fn tensor_raw(
        &mut self,
        name: &str,
        shape: &[u64],
        tensor_type: TensorType,
        data: Vec<u8>,
    ) -> &mut Self {
        self.writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: shape.to_vec(),
            tensor_type,
            data,
        });
        self
    }

    /// Number of tensors queued so far.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.writer.tensor_count()
    }

    /// Serialise the queued metadata and tensors into a complete GGUF file.
    ///
    /// # Errors
    ///
    /// Propagates [`WriteError`] (e.g. a duplicate tensor name).
    pub fn build(&self) -> Result<Vec<u8>, FixtureError> {
        Ok(self.writer.to_bytes()?)
    }
}

impl Default for GgufFixtureBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── A non-degenerate tiny dense `qwen3` model ──────────────────────────────

/// [`tiny_dense_qwen3_gguf`]'s dimensions, matching
/// `oxibonsai_core::config::Qwen3Config::tiny_test()` exactly except for
/// the vocabulary (32, not `tiny_test`'s real 151936 — small enough to
/// keep the embedding/output tensors cheap; every other dimension is the
/// same size).
pub const TINY_DENSE_HIDDEN: usize = 64;
pub const TINY_DENSE_INTERMEDIATE: usize = 128;
pub const TINY_DENSE_LAYERS: usize = 2;
pub const TINY_DENSE_HEADS: usize = 4;
pub const TINY_DENSE_KV_HEADS: usize = 2;
pub const TINY_DENSE_HEAD_DIM: usize = 16;
pub const TINY_DENSE_VOCAB: usize = 32;
pub const TINY_DENSE_CONTEXT: usize = 512;

/// A complete, parseable, non-degenerate dense `qwen3` GGUF: every weight is
/// [`deterministic_weights`] seeded from `seed` (never all-zero, never
/// constant across a tensor), at [`TINY_DENSE_HIDDEN`]/[`TINY_DENSE_INTERMEDIATE`]/
/// [`TINY_DENSE_LAYERS`]/[`TINY_DENSE_HEADS`]/[`TINY_DENSE_KV_HEADS`]/
/// [`TINY_DENSE_HEAD_DIM`] — `Qwen3Config::tiny_test()`'s own sizes.
///
/// `Qwen3Config::tiny_test()` combined with an all-zero-weight model (e.g.
/// `BonsaiModel::new`, which leaves `blocks` empty) produces an all-zero
/// logit vector, under which a logit-shape setting such as a repetition
/// penalty is the identity on every logit and so cannot be observed to do
/// anything. This fixture, loaded through the real `BonsaiModel::from_gguf`,
/// gives such a test real, moving logits to check a setting against instead.
///
/// Every 2-D weight matrix is [`FixtureQuant::Q8_0`] (32-element row
/// groups): [`FixtureQuant::TQ2_0_g128`]/[`FixtureQuant::Q1_0G128`] need a
/// row length that is a whole multiple of 128, which [`TINY_DENSE_HIDDEN`]
/// (64) does not satisfy. The 1-D norm vectors stay [`FixtureQuant::F32`].
///
/// # Errors
/// Propagates [`FixtureError`] from an underlying [`GgufFixtureBuilder::tensor`]
/// call (e.g. an unsupported quant/writer mismatch) or from
/// [`GgufFixtureBuilder::build`] (e.g. a duplicate tensor name).
pub fn tiny_dense_qwen3_gguf(seed: u64) -> Result<Vec<u8>, FixtureError> {
    let h = TINY_DENSE_HIDDEN;
    let inter = TINY_DENSE_INTERMEDIATE;
    let nq = TINY_DENSE_HEADS;
    let nkv = TINY_DENSE_KV_HEADS;
    let hd = TINY_DENSE_HEAD_DIM;
    let vocab = TINY_DENSE_VOCAB;

    let mut b = GgufFixtureBuilder::new();
    b.metadata_str("general.architecture", "qwen3");
    b.metadata_str("general.name", "testkit-tiny-dense-qwen3");
    b.metadata_u32("qwen3.embedding_length", h as u32);
    b.metadata_u32("qwen3.block_count", TINY_DENSE_LAYERS as u32);
    b.metadata_u32("qwen3.attention.head_count", nq as u32);
    b.metadata_u32("qwen3.attention.head_count_kv", nkv as u32);
    b.metadata_u32("qwen3.attention.key_length", hd as u32);
    b.metadata_u32("qwen3.attention.value_length", hd as u32);
    b.metadata_u32("qwen3.feed_forward_length", inter as u32);
    b.metadata_u32("qwen3.vocab_size", vocab as u32);
    b.metadata_u32("qwen3.context_length", TINY_DENSE_CONTEXT as u32);
    b.metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    b.metadata_f32("qwen3.rope.freq_base", 10_000.0);

    // A distinct seed per tensor, deterministically derived from `seed`, so
    // no two tensors are byte-identical copies of each other.
    let mut state = seed ^ 0x7151_7151_7151_7151;
    let mut next_seed = move || {
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        state
    };

    b.tensor(
        "token_embd.weight",
        &[h as u64, vocab as u64],
        FixtureQuant::Q8_0,
        next_seed(),
    )?;
    b.tensor(
        "output_norm.weight",
        &[h as u64],
        FixtureQuant::F32,
        next_seed(),
    )?;
    b.tensor(
        "output.weight",
        &[h as u64, vocab as u64],
        FixtureQuant::Q8_0,
        next_seed(),
    )?;

    for layer in 0..TINY_DENSE_LAYERS {
        let pfx = format!("blk.{layer}");
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            b.tensor(
                &format!("{pfx}.{name}"),
                &[h as u64],
                FixtureQuant::F32,
                next_seed(),
            )?;
        }
        for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            b.tensor(
                &format!("{pfx}.{name}"),
                &[hd as u64],
                FixtureQuant::F32,
                next_seed(),
            )?;
        }
        b.tensor(
            &format!("{pfx}.attn_q.weight"),
            &[h as u64, (nq * hd) as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.attn_k.weight"),
            &[h as u64, (nkv * hd) as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.attn_v.weight"),
            &[h as u64, (nkv * hd) as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.attn_output.weight"),
            &[(nq * hd) as u64, h as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.ffn_gate.weight"),
            &[h as u64, inter as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.ffn_up.weight"),
            &[h as u64, inter as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
        b.tensor(
            &format!("{pfx}.ffn_down.weight"),
            &[inter as u64, h as u64],
            FixtureQuant::Q8_0,
            next_seed(),
        )?;
    }
    b.build()
}

// ─── In-place patch of one metadata value in a real GGUF image ──────────────

/// GGUF's type tag for a `FLOAT32` metadata value.
const GGUF_METADATA_TYPE_FLOAT32: u32 = 6;

/// Overwrite one `FLOAT32` metadata value of a GGUF file image in place,
/// changing nothing else — how a real-model test builds a counterpart of a
/// shipped file that differs in exactly one hyperparameter (e.g.
/// `qwen3.rope.scaling.factor` 4.0 -> 1.0 on `Bonsai-8B.gguf`), since a
/// parsed `GgufFile`'s metadata store has no mutator.
///
/// A GGUF metadata key is a `u64` little-endian byte length followed by the
/// key bytes with no terminator, then a `u32` type tag, then the value. The
/// key is located by its literal bytes, and nothing is written unless every
/// check passes: the key occurs exactly once in the whole image, the 8 bytes
/// before it are a `u64` equal to its length, the tag after it is `FLOAT32`,
/// and the current value equals `expected` exactly.
///
/// # Errors
///
/// A description of the first check that failed.
pub fn patch_gguf_f32_metadata(
    bytes: &mut [u8],
    key: &str,
    expected: f32,
    replacement: f32,
) -> Result<(), String> {
    let needle = key.as_bytes();
    if needle.is_empty() {
        return Err("empty metadata key".to_string());
    }
    let occurrences: Vec<usize> = bytes
        .windows(needle.len())
        .enumerate()
        .filter_map(|(i, window)| (window == needle).then_some(i))
        .collect();
    let [key_offset] = occurrences[..] else {
        return Err(format!(
            "expected exactly one occurrence of the key {key:?} in the GGUF image, found {} \
             -- refusing to patch blind",
            occurrences.len()
        ));
    };

    let length_prefix: [u8; 8] = key_offset
        .checked_sub(8)
        .and_then(|start| bytes.get(start..key_offset))
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| format!("{key:?} sits too close to the start of the image"))?;
    let declared_len = u64::from_le_bytes(length_prefix);
    if declared_len != needle.len() as u64 {
        return Err(format!(
            "the 8 bytes before {key:?} read {declared_len}, not its length {} -- not a \
             metadata key at this offset",
            needle.len()
        ));
    }

    let tag_offset = key_offset + needle.len();
    let tag: [u8; 4] = bytes
        .get(tag_offset..tag_offset + 4)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| format!("{key:?} is truncated before its type tag"))?;
    let tag = u32::from_le_bytes(tag);
    if tag != GGUF_METADATA_TYPE_FLOAT32 {
        return Err(format!(
            "{key:?} has type tag {tag}, not FLOAT32 ({GGUF_METADATA_TYPE_FLOAT32})"
        ));
    }

    let value_offset = tag_offset + 4;
    let value_slot = bytes
        .get_mut(value_offset..value_offset + 4)
        .ok_or_else(|| format!("{key:?} is truncated before its value"))?;
    let mut current = [0u8; 4];
    current.copy_from_slice(value_slot);
    let current = f32::from_le_bytes(current);
    if current.to_bits() != expected.to_bits() {
        return Err(format!(
            "{key:?} currently holds {current}, not the expected {expected}"
        ));
    }
    value_slot.copy_from_slice(&replacement.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;

    #[test]
    fn patch_gguf_f32_metadata_changes_exactly_the_one_value() {
        const KEY: &str = "qwen3.rope.scaling.factor";
        let original = GgufFixtureBuilder::new()
            .metadata_str("general.architecture", "qwen3")
            .metadata_f32(KEY, 4.0)
            .metadata_f32("qwen3.rope.freq_base", 1_000_000.0)
            .tensor("output_norm.weight", &[8], FixtureQuant::F32, 3)
            .expect("add tensor")
            .build()
            .expect("build gguf");

        let mut patched = original.clone();
        patch_gguf_f32_metadata(&mut patched, KEY, 4.0, 1.0).expect("patch the factor");

        let gguf = GgufFile::parse(&patched).expect("the patched image still parses");
        assert_eq!(gguf.metadata.get_f32(KEY).expect("factor"), 1.0);
        assert_eq!(
            gguf.metadata
                .get_f32("qwen3.rope.freq_base")
                .expect("freq_base"),
            1_000_000.0,
            "a neighbouring value must be untouched"
        );
        let changed = original
            .iter()
            .zip(&patched)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            (1..=4).contains(&changed),
            "only the value's own 4 bytes may change, {changed} bytes did"
        );
    }

    #[test]
    fn patch_gguf_f32_metadata_refuses_a_wrong_expectation_or_type() {
        let mut bytes = GgufFixtureBuilder::new()
            .metadata_f32("a.float", 2.0)
            .metadata_u32("a.int", 7)
            .build()
            .expect("build gguf");
        let pristine = bytes.clone();
        assert!(patch_gguf_f32_metadata(&mut bytes, "a.float", 3.0, 1.0).is_err());
        assert!(patch_gguf_f32_metadata(&mut bytes, "a.int", 7.0, 1.0).is_err());
        assert!(patch_gguf_f32_metadata(&mut bytes, "a.missing", 1.0, 2.0).is_err());
        assert_eq!(bytes, pristine, "a refused patch must write nothing");
    }

    #[test]
    fn deterministic_weights_are_reproducible_and_bounded() {
        let a = deterministic_weights(256, 42);
        let b = deterministic_weights(256, 42);
        assert_eq!(a, b, "same (n, seed) must reproduce byte-identical weights");
        assert_eq!(a.len(), 256);
        assert!(a.iter().all(|w| (-1.0..=1.0).contains(w)));
        // A different seed must (overwhelmingly likely) differ somewhere.
        let c = deterministic_weights(256, 43);
        assert_ne!(a, c, "different seeds should not collide");
    }

    #[test]
    fn deterministic_weights_are_not_all_identical() {
        // Regression guard: a broken generator that always returns the same
        // value would still pass the "bounded" assertion above.
        let w = deterministic_weights(64, 7);
        let first = w[0];
        assert!(
            w.iter().any(|&x| x != first),
            "a real PRNG must not produce a constant sequence"
        );
    }

    #[test]
    fn tiny_dense_qwen3_gguf_builds_and_parses() {
        let bytes = tiny_dense_qwen3_gguf(0xD1_5E).expect("build tiny dense qwen3 gguf");
        assert!(bytes.starts_with(b"GGUF"));
        let gguf = GgufFile::parse(&bytes).expect("parse tiny dense qwen3 gguf");
        assert_eq!(
            gguf.metadata
                .get_string("general.architecture")
                .expect("general.architecture"),
            "qwen3"
        );
        // Every layer's tensors are present.
        for layer in 0..TINY_DENSE_LAYERS {
            for suffix in [
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{layer}.{suffix}");
                assert!(gguf.tensors.require(&name).is_ok(), "missing tensor {name}");
            }
        }
    }

    #[test]
    fn tiny_dense_qwen3_gguf_is_deterministic_and_not_all_zero() {
        let a = tiny_dense_qwen3_gguf(0xD1_5E).expect("build a");
        let b = tiny_dense_qwen3_gguf(0xD1_5E).expect("build b");
        assert_eq!(a, b, "same seed must reproduce byte-identical output");
        let c = tiny_dense_qwen3_gguf(0xD1_5F).expect("build c");
        assert_ne!(a, c, "different seeds should not collide");

        // The embedding tensor's raw bytes must not be all-zero — the
        // degenerate case this fixture exists to avoid.
        let gguf = GgufFile::parse(&a).expect("parse");
        let info = gguf.tensors.require("token_embd.weight").expect("tensor");
        let start = gguf.data_offset + info.offset as usize;
        let row_bytes = TensorType::F32.row_bytes(&info.shape) as usize;
        assert!(
            a[start..start + row_bytes].iter().any(|&byte| byte != 0),
            "token_embd.weight must not be all-zero bytes"
        );
    }

    #[test]
    fn next_valid_tq2_byte_never_emits_the_reserved_pattern() {
        let mut rng = Lcg::new(123);
        for _ in 0..10_000 {
            let byte = rng.next_valid_tq2_byte();
            for lane in 0..4 {
                let code = (byte >> (2 * lane)) & 0b11;
                assert_ne!(
                    code, 0b11,
                    "lane {lane} of byte {byte:#010b} is the reserved ternary code"
                );
            }
        }
    }

    #[test]
    fn lcg_is_deterministic_across_instances() {
        let mut a = Lcg::new(99);
        let mut b = Lcg::new(99);
        for _ in 0..50 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn lcg_zero_seed_is_remapped_to_a_nonzero_fixed_point() {
        let mut rng = Lcg::new(0);
        // If 0 were used verbatim, next_u64() would stay 0 forever
        // (0 * mult + 0 == 0).
        let mut saw_nonzero = false;
        for _ in 0..8 {
            if rng.next_u64() != 0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero, "seed 0 must not be a degenerate fixed point");
    }

    #[test]
    fn quantize_bytes_covers_every_fixture_quant_variant() {
        // One block's worth of weights per format; every variant must
        // succeed and return exactly `block_bytes` bytes.
        let all = [
            (FixtureQuant::F32, 4),
            (FixtureQuant::F16, 2),
            (FixtureQuant::Q1_0G128, 18),
            (FixtureQuant::TQ2_0_g128, 34),
            (FixtureQuant::TQ2_0, 66),
            (FixtureQuant::Q4_0, 18),
            (FixtureQuant::Q8_0, 34),
            (FixtureQuant::Q2K, 84),
            (FixtureQuant::Q3K, 110),
            (FixtureQuant::Q4K, 144),
            (FixtureQuant::Q5K, 176),
            (FixtureQuant::Q6K, 210),
            (FixtureQuant::Q8K, 292),
            (FixtureQuant::F8E4M3, 34),
            (FixtureQuant::F8E5M2, 34),
            (FixtureQuant::PQ2_0, 34),
            (FixtureQuant::PTQ1_0, 28),
            (FixtureQuant::Q2_0G64, 18),
        ];
        for (quant, expected_block_bytes) in all {
            let weights = deterministic_weights(quant.block_size(), 7);
            let bytes = quantize_bytes(quant, &weights)
                .unwrap_or_else(|e| panic!("quantize_bytes({quant:?}) failed: {e}"));
            assert_eq!(
                bytes.len(),
                expected_block_bytes,
                "{quant:?}: one block must be {expected_block_bytes} bytes"
            );
        }
    }

    #[test]
    fn quantize_bytes_rejects_misaligned_length() {
        let weights = deterministic_weights(FixtureQuant::TQ2_0_g128.block_size() - 1, 1);
        assert!(
            quantize_bytes(FixtureQuant::TQ2_0_g128, &weights).is_err(),
            "one weight short of a full block must fail, not silently truncate"
        );
    }

    #[test]
    fn pack_q1_0_g128_roundtrips_signs() {
        let mut weights = vec![0.0f32; QK1_0_G128];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = if i % 2 == 0 { 0.7 } else { -0.7 };
        }
        let blocks = pack_q1_0_g128(&weights).expect("pack");
        assert_eq!(blocks.len(), 1);
        for i in 0..QK1_0_G128 {
            let expected_positive = i % 2 == 0;
            assert_eq!(
                blocks[0].sign_bit(i),
                expected_positive,
                "sign bit {i} mismatch"
            );
        }
    }

    #[test]
    fn pack_q1_0_g128_rejects_non_multiple_of_128() {
        let weights = vec![0.1f32; 127];
        assert!(pack_q1_0_g128(&weights).is_err());
        let empty: Vec<f32> = vec![];
        assert!(pack_q1_0_g128(&empty).is_err());
    }

    #[test]
    fn builder_produces_a_file_the_real_reader_parses() {
        // GGUF shape convention: shape[0] (`ne0`) is the innermost/fastest-
        // varying dimension and ggml quantizes each row (every *other*
        // dimension held fixed) independently, so `ne0` itself must be a
        // multiple of the format's block size — `[128, 2]` means
        // "in_features=128, out_features=2" (2 rows of 128), not `[8, 32]`.
        let bytes = GgufFixtureBuilder::new()
            .metadata_str("general.architecture", "qwen3")
            .metadata_u32("qwen3.block_count", 2)
            .tensor("token_embd.weight", &[128, 2], FixtureQuant::TQ2_0_g128, 1)
            .expect("add token_embd")
            .tensor("output_norm.weight", &[8], FixtureQuant::F32, 2)
            .expect("add output_norm")
            .tensor("blk.0.ffn_down.weight", &[128, 16], FixtureQuant::F8E4M3, 3)
            .expect("add ffn_down")
            .build()
            .expect("build");

        assert!(bytes.starts_with(b"GGUF"));
        let file = GgufFile::parse(&bytes).expect("the real reader must parse this fixture");
        assert_eq!(file.tensors.len(), 3);
        assert_eq!(
            file.metadata
                .get_string("general.architecture")
                .expect("architecture key"),
            "qwen3"
        );
        let embd = file.tensor_data("token_embd.weight").expect("tensor data");
        // shape [128, 2] -> 256 elements / 128 per TQ2_0_g128 block = 2 blocks * 34 bytes.
        assert_eq!(embd.len(), 68);
        let norm = file.tensor_data("output_norm.weight").expect("tensor data");
        assert_eq!(norm.len(), 8 * 4, "F32: 8 elements * 4 bytes each");
    }

    /// Every K-quant — `Q2K`, `Q3K` and `Q8K` included — goes through the
    /// builder into a file the real reader parses: the tensor carries the
    /// format's ggml type id and exactly `rows * block_bytes` of data, and
    /// the data is the same bytes [`quantize_bytes`] produces for the seed.
    #[test]
    fn builder_embeds_every_k_quant_format_in_a_file_the_real_reader_parses() {
        use oxibonsai_core::GgufTensorType;

        // (format, ggml id, bytes per 256-weight super-block)
        let cases = [
            (FixtureQuant::Q2K, 10_u32, std::mem::size_of::<BlockQ2K>()),
            (FixtureQuant::Q3K, 11, std::mem::size_of::<BlockQ3K>()),
            (FixtureQuant::Q4K, 12, std::mem::size_of::<BlockQ4K>()),
            (FixtureQuant::Q5K, 13, std::mem::size_of::<BlockQ5K>()),
            (FixtureQuant::Q6K, 14, std::mem::size_of::<BlockQ6K>()),
            (FixtureQuant::Q8K, 15, std::mem::size_of::<BlockQ8K>()),
        ];
        for (quant, ggml_id, block_bytes) in cases {
            // ne0 = one super-block, two rows.
            let shape = [quant.block_size() as u64, 2];
            let bytes = GgufFixtureBuilder::new()
                .metadata_str("general.architecture", "qwen3")
                .tensor("w", &shape, quant, 7)
                .unwrap_or_else(|e| panic!("{quant:?}: add tensor: {e}"))
                .build()
                .unwrap_or_else(|e| panic!("{quant:?}: build: {e}"));
            let file = GgufFile::parse(&bytes)
                .unwrap_or_else(|e| panic!("{quant:?}: the real reader must parse: {e}"));
            let info = file
                .tensors
                .get("w")
                .unwrap_or_else(|| panic!("{quant:?}: tensor `w` is listed"));
            assert_eq!(
                info.tensor_type,
                GgufTensorType::from_id(ggml_id).expect("a known ggml id"),
                "{quant:?} is written under ggml id {ggml_id}"
            );
            let data = file.tensor_data("w").expect("tensor data");
            assert_eq!(data.len(), 2 * block_bytes, "{quant:?}: two super-blocks");
            let expected = quantize_bytes(quant, &deterministic_weights(2 * quant.block_size(), 7))
                .expect("quantize");
            assert_eq!(data, expected.as_slice(), "{quant:?}: the bytes written");
        }
    }

    #[test]
    fn builder_default_matches_new() {
        let a = GgufFixtureBuilder::default();
        assert_eq!(a.tensor_count(), 0);
    }

    /// The generic `metadata()` escape hatch must reach the real reader for
    /// variants none of the typed helpers (`metadata_str`/`_u32`/`_f32`)
    /// cover — array and bool values, which `qwen35_fixture` needs for the
    /// `prism.hadamard.*` contract.
    #[test]
    fn metadata_writes_array_and_bool_variants_the_typed_helpers_do_not_cover() {
        let bytes = GgufFixtureBuilder::new()
            .metadata_str("general.architecture", "qwen3")
            .metadata(
                "qwen3.rope.dimension_sections",
                MetadataWriteValue::ArrayU32(vec![3, 3, 2, 0]),
            )
            .metadata(
                "prism.hadamard.sign_values",
                MetadataWriteValue::ArrayI32(vec![-1, 1, 1, -1]),
            )
            .metadata(
                "prism.hadamard.weight_names",
                MetadataWriteValue::ArrayStr(vec!["output.weight".to_string()]),
            )
            .metadata(
                "prism.hadamard.gdn_v_grouped",
                MetadataWriteValue::Bool(true),
            )
            .tensor("output_norm.weight", &[8], FixtureQuant::F32, 1)
            .expect("add tensor")
            .build()
            .expect("build gguf");

        let file = GgufFile::parse(&bytes).expect("the real reader must parse this fixture");
        assert_eq!(
            file.metadata
                .get("qwen3.rope.dimension_sections")
                .and_then(|v| v.as_array())
                .expect("array u32 key")
                .len(),
            4
        );
        assert_eq!(
            file.metadata
                .get("prism.hadamard.sign_values")
                .and_then(|v| v.as_array())
                .expect("array i32 key")
                .len(),
            4
        );
        assert_eq!(
            file.metadata
                .get("prism.hadamard.weight_names")
                .and_then(|v| v.as_array())
                .expect("array str key")
                .len(),
            1
        );
        assert_eq!(
            file.metadata
                .get("prism.hadamard.gdn_v_grouped")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }
}
