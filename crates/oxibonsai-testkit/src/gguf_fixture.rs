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
//! # Cross-crate wiring (deviation)
//!
//! See this crate's `Cargo.toml` doc comment: `oxibonsai-testkit` is not yet
//! a registered workspace member, so the 13 pre-existing duplicate builders
//! could not be re-pointed at this module without editing Cargo manifests
//! outside this package's `owned_files`. That re-point is recorded as a
//! deviation; this module is nonetheless the complete, tested, canonical
//! implementation those call sites should adopt once the wiring lands.

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
    /// `None` for `Q2K`/`Q3K`/`Q8K`: `oxibonsai_core::gguf::writer::TensorType`
    /// (a file this package does not own) has no variant for ggml ids
    /// 10/11/15 yet — only `Q4_K`/`Q5_K`/`Q6_K` are wired into the writer
    /// today. [`quantize_bytes`] still produces correct raw block bytes for
    /// these three (every K-quant kernel test in this workspace — e.g.
    /// `metal_k_quant_gemv_parity.rs` — consumes raw blocks directly and
    /// never goes through a full GGUF file), only [`GgufFixtureBuilder`]
    /// cannot embed them in a whole file. Recorded as a deviation (writer.rs
    /// needs three more `TensorType` variants; not in this package's
    /// `owned_files`).
    #[must_use]
    pub const fn writer_type(self) -> Option<TensorType> {
        match self {
            Self::F32 => Some(TensorType::F32),
            Self::F16 => Some(TensorType::F16),
            Self::Q1_0G128 => Some(TensorType::Q1_0G128),
            Self::TQ2_0_g128 => Some(TensorType::TQ2_0_g128),
            Self::TQ2_0 => Some(TensorType::TQ2_0),
            Self::Q4_0 => Some(TensorType::Q4_0),
            Self::Q8_0 => Some(TensorType::Q8_0),
            Self::Q4K => Some(TensorType::Q4_K),
            Self::Q5K => Some(TensorType::Q5_K),
            Self::Q6K => Some(TensorType::Q6_K),
            Self::F8E4M3 => Some(TensorType::F8_E4M3),
            Self::F8E5M2 => Some(TensorType::F8_E5M2),
            Self::PQ2_0 => Some(TensorType::PQ2_0),
            Self::PTQ1_0 => Some(TensorType::PTQ1_0),
            Self::Q2_0G64 => Some(TensorType::Q2_0G64),
            Self::Q2K | Self::Q3K | Self::Q8K => None,
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
    /// [`FixtureQuant::writer_type`] returned `None`: see its doc comment.
    #[error(
        "{quant:?} has no oxibonsai_core::gguf::writer::TensorType mapping yet; \
         use quantize_bytes() directly for this format instead of GgufFixtureBuilder"
    )]
    UnsupportedByWriter { quant: FixtureQuant },
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

    /// Append a tensor named `name` with shape `shape` (GGUF order: the
    /// fastest-varying / innermost dimension first), filled with
    /// [`deterministic_weights`] seeded by `seed` and quantized to `quant`.
    ///
    /// # Errors
    ///
    /// - [`FixtureError::UnsupportedByWriter`] if `quant` has no
    ///   `oxibonsai_core::TensorType` mapping (`Q2K`/`Q3K`/`Q8K` today; see
    ///   [`FixtureQuant::writer_type`]).
    /// - Whatever [`quantize_bytes`] returns for a shape whose element count
    ///   is not a multiple of `quant.block_size()`.
    pub fn tensor(
        &mut self,
        name: &str,
        shape: &[u64],
        quant: FixtureQuant,
        seed: u64,
    ) -> Result<&mut Self, FixtureError> {
        let tensor_type = quant
            .writer_type()
            .ok_or(FixtureError::UnsupportedByWriter { quant })?;
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
    /// malformed block, or a format `quant.writer_type()` cannot reach).
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;

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

    #[test]
    fn builder_rejects_k_quant_formats_the_writer_cannot_serialise_yet() {
        // Q2K/Q3K/Q8K: writer.rs (not owned by this package) has no
        // TensorType variant for ggml ids 10/11/15 yet.
        for quant in [FixtureQuant::Q2K, FixtureQuant::Q3K, FixtureQuant::Q8K] {
            let mut b = GgufFixtureBuilder::new();
            let err = b
                .tensor("w", &[quant.block_size() as u64], quant, 1)
                .expect_err("must report the writer gap, not silently drop the tensor");
            assert!(matches!(err, FixtureError::UnsupportedByWriter { .. }));
        }
    }

    #[test]
    fn builder_default_matches_new() {
        let a = GgufFixtureBuilder::default();
        assert_eq!(a.tensor_count(), 0);
    }
}
