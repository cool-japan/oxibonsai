//! `oxibonsai-testkit` — shared, dev-only test-support crate (T-07 / T-05).
//!
//! Never published (`publish = false` in this crate's `Cargo.toml`) and
//! never a runtime dependency of any shipped crate or binary — only ever a
//! `[dev-dependencies]` entry. It exists to stop each test crate from
//! re-inventing the same four things:
//!
//! - [`gguf_fixture`] — one deterministic GGUF fixture builder covering
//!   every quant format this workspace executes, replacing 13 independently
//!   written, subtly-divergent builders (T-07).
//! - [`qwen35_fixture`] — a complete, public synthetic Bonsai 2 hybrid
//!   (`qwen35`) GGUF built on [`gguf_fixture`], for any crate that needs a
//!   loadable hybrid model without a multi-GB real one (EMBED-WIRE handover
//!   (3)).
//! - [`mmproj_fixture`] — a small synthetic Qwen3-VL vision projector
//!   (`clip` architecture, `qwen3vl_merger` projector) GGUF with the same
//!   tensor inventory and type mix as the real Bonsai 2 mmproj, for
//!   vision-tower tests that must not need the 0.63 GB real file.
//! - [`capability`] — the JSONL hardware/fixture-capability self-skip
//!   report contract (T-05), so a skipped hardware-dependent test is
//!   visibly distinct from one that ran and passed.
//! - [`temp_path`] — collision-free temp file helpers built on
//!   `std::env::temp_dir()` (never a hardcoded absolute path).
//! - [`golden`] — golden-vector comparison helpers (`max_abs_diff`,
//!   `cosine_similarity`, `assert_allclose`) for parity tests.
//! - [`workspace`] — resolving this workspace's root and its (gitignored,
//!   often-absent-in-a-worktree) `models/` directory robustly.
//! - [`parity`] — the cross-tier per-step logit/token comparison a
//!   real-model greedy parity gate needs (token chain first, then a
//!   bit-exact or relative-bound numeric check), generic over the caller's
//!   own kernel-tier type so this crate never depends on
//!   `oxibonsai-kernels`.
//!
//! This crate is a real member of the main Cargo workspace (root
//! `Cargo.toml`'s `[workspace] members`), taken as a `[dev-dependencies]`
//! entry by every producer crate above.

pub mod capability;
pub mod dense_fixture;
pub mod gguf_fixture;
pub mod parity;
pub mod qwen35_fixture;

/// A synthetic Qwen3-VL vision projector (`mmproj`) GGUF.
///
/// The real Bonsai 2 projector (`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`) is
/// a `general.architecture = "clip"` file with the `qwen3vl_merger`
/// projector: a patch-embedding convolution stored as two temporal slices
/// (`v.patch_embd.weight`, `v.patch_embd.weight.1`) plus `v.patch_embd.bias`,
/// a learned `v.position_embd.weight` over a square grid, `block_count`
/// pre-LayerNorm ViT blocks (`v.blk.N.{ln1,attn_qkv,attn_out,ln2,ffn_up,
/// ffn_down}`), `v.post_ln` and the two-layer merger MLP `mm.0` / `mm.2`.
/// This module builds the same inventory at toy sizes with the real file's
/// type mix — `Q8_0` for `attn_qkv` / `attn_out` / `ffn_up` / `mm.0` /
/// `mm.2`, `F16` for `ffn_down`, `F32` for everything else — so a loader and
/// a tower can be exercised end to end (and against an independent
/// reference) without the real weights.
///
/// Every value is [`gguf_fixture::deterministic_weights`] output scaled to a
/// range that keeps activations of order one through the whole graph:
/// matrices by roughly `1 / sqrt(fan_in)` (the fused QKV projection twice
/// that, so the attention is visibly non-uniform), LayerNorm scales around
/// one, biases and position embeddings small.
///
/// The builder is split into [`mmproj_fixture::metadata`],
/// [`mmproj_fixture::tensors`] and [`mmproj_fixture::assemble`] so a negative
/// test can drop, reshape, retype or add a tensor, or change a metadata key,
/// before the bytes are written.
pub mod mmproj_fixture {
    use oxibonsai_core::MetadataWriteValue;

    use crate::gguf_fixture::{
        deterministic_weights, quantize_bytes, FixtureError, FixtureQuant, GgufFixtureBuilder,
    };

    /// Dimensions of a synthetic projector. [`MmprojFixtureSpec::tiny`] is
    /// the canonical small configuration (2 blocks, hidden 64, 4 heads,
    /// patch 16, spatial merge 2).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct MmprojFixtureSpec {
        /// `clip.vision.embedding_length`.
        pub hidden: usize,
        /// `clip.vision.attention.head_count`.
        pub heads: usize,
        /// `clip.vision.feed_forward_length`.
        pub ffn: usize,
        /// `clip.vision.block_count`.
        pub blocks: usize,
        /// `clip.vision.patch_size`.
        pub patch: usize,
        /// Side of the stored square position-embedding grid
        /// (`v.position_embd.weight` holds `pos_side * pos_side` rows).
        pub pos_side: usize,
        /// Output width of `mm.0` (the real file uses `4 * hidden`).
        pub merger_hidden: usize,
        /// `clip.vision.projection_dim` — the output width of `mm.2`.
        pub projection_dim: usize,
        /// Base seed; every tensor derives its own seed from it.
        pub seed: u64,
    }

    impl MmprojFixtureSpec {
        /// 2 blocks, hidden 64, 4 heads (head dim 16), FFN 96, patch 16, an
        /// 8 x 8 stored position grid (image size 128), merger
        /// 256 -> 256 -> 80.
        #[must_use]
        pub const fn tiny() -> Self {
            Self {
                hidden: 64,
                heads: 4,
                ffn: 96,
                blocks: 2,
                patch: 16,
                pos_side: 8,
                merger_hidden: 256,
                projection_dim: 80,
                seed: 0x5EED_C11F,
            }
        }
    }

    /// One tensor of the fixture: its GGUF-order shape (`ne0` first), the
    /// format it is written in and its values before quantisation.
    #[derive(Debug, Clone)]
    pub struct FixtureTensor {
        /// Tensor name, e.g. `v.blk.0.attn_qkv.weight`.
        pub name: String,
        /// GGUF-order shape (fastest-varying dimension first).
        pub shape: Vec<u64>,
        /// Storage format.
        pub quant: FixtureQuant,
        /// `shape.iter().product()` values, quantised to `quant` on write.
        pub values: Vec<f32>,
    }

    fn tensor(
        name: String,
        shape: &[usize],
        quant: FixtureQuant,
        values: Vec<f32>,
    ) -> FixtureTensor {
        FixtureTensor {
            name,
            shape: shape.iter().map(|&d| d as u64).collect(),
            quant,
            values,
        }
    }

    /// `n` deterministic values in `[offset - scale, offset + scale]`.
    fn values(n: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
        deterministic_weights(n, seed)
            .into_iter()
            .map(|v| v * scale + offset)
            .collect()
    }

    /// The metadata of the fixture, in the real file's key spelling.
    #[must_use]
    pub fn metadata(spec: &MmprojFixtureSpec) -> Vec<(String, MetadataWriteValue)> {
        let u32_of = |v: usize| MetadataWriteValue::U32(u32::try_from(v).unwrap_or(u32::MAX));
        let text = |v: &str| MetadataWriteValue::Str(v.to_string());
        vec![
            ("general.architecture".to_string(), text("clip")),
            ("general.type".to_string(), text("mmproj")),
            (
                "general.quantization_version".to_string(),
                MetadataWriteValue::U32(2),
            ),
            (
                "clip.has_vision_encoder".to_string(),
                MetadataWriteValue::Bool(true),
            ),
            ("clip.projector_type".to_string(), text("qwen3vl_merger")),
            ("clip.use_gelu".to_string(), MetadataWriteValue::Bool(true)),
            (
                "clip.vision.image_size".to_string(),
                u32_of(spec.pos_side * spec.patch),
            ),
            ("clip.vision.patch_size".to_string(), u32_of(spec.patch)),
            (
                "clip.vision.embedding_length".to_string(),
                u32_of(spec.hidden),
            ),
            (
                "clip.vision.feed_forward_length".to_string(),
                u32_of(spec.ffn),
            ),
            ("clip.vision.block_count".to_string(), u32_of(spec.blocks)),
            (
                "clip.vision.attention.head_count".to_string(),
                u32_of(spec.heads),
            ),
            (
                "clip.vision.attention.layer_norm_epsilon".to_string(),
                MetadataWriteValue::F32(1e-6),
            ),
            (
                "clip.vision.projection_dim".to_string(),
                u32_of(spec.projection_dim),
            ),
            (
                "clip.vision.spatial_merge_size".to_string(),
                MetadataWriteValue::U32(2),
            ),
            (
                "clip.vision.image_mean".to_string(),
                MetadataWriteValue::ArrayF32(vec![0.5, 0.5, 0.5]),
            ),
            (
                "clip.vision.image_std".to_string(),
                MetadataWriteValue::ArrayF32(vec![0.5, 0.5, 0.5]),
            ),
            (
                "clip.vision.is_deepstack_layers".to_string(),
                MetadataWriteValue::ArrayBool(vec![false; spec.blocks]),
            ),
        ]
    }

    /// Every tensor of the fixture, named and shaped exactly as in the real
    /// projector (`12 * blocks + 10` tensors).
    #[must_use]
    pub fn tensors(spec: &MmprojFixtureSpec) -> Vec<FixtureTensor> {
        let h = spec.hidden;
        let ffn = spec.ffn;
        let inv_sqrt = |n: usize| 1.0 / (n.max(1) as f32).sqrt();
        let mut seed = spec.seed;
        let mut next_seed = move || {
            seed = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x2545_F491);
            seed
        };
        let mut out = Vec::with_capacity(12 * spec.blocks + 10);
        for il in 0..spec.blocks {
            let name = |suffix: &str| format!("v.blk.{il}.{suffix}");
            let norm_scale = |seed: u64| values(h, seed, 0.2, 1.0);
            let small = |n: usize, seed: u64| values(n, seed, 0.1, 0.0);
            out.push(tensor(
                name("ln1.weight"),
                &[h],
                FixtureQuant::F32,
                norm_scale(next_seed()),
            ));
            out.push(tensor(
                name("ln1.bias"),
                &[h],
                FixtureQuant::F32,
                small(h, next_seed()),
            ));
            out.push(tensor(
                name("attn_qkv.weight"),
                &[h, 3 * h],
                FixtureQuant::Q8_0,
                values(3 * h * h, next_seed(), 2.0 * inv_sqrt(h), 0.0),
            ));
            out.push(tensor(
                name("attn_qkv.bias"),
                &[3 * h],
                FixtureQuant::F32,
                small(3 * h, next_seed()),
            ));
            out.push(tensor(
                name("attn_out.weight"),
                &[h, h],
                FixtureQuant::Q8_0,
                values(h * h, next_seed(), inv_sqrt(h), 0.0),
            ));
            out.push(tensor(
                name("attn_out.bias"),
                &[h],
                FixtureQuant::F32,
                small(h, next_seed()),
            ));
            out.push(tensor(
                name("ln2.weight"),
                &[h],
                FixtureQuant::F32,
                norm_scale(next_seed()),
            ));
            out.push(tensor(
                name("ln2.bias"),
                &[h],
                FixtureQuant::F32,
                small(h, next_seed()),
            ));
            out.push(tensor(
                name("ffn_up.weight"),
                &[h, ffn],
                FixtureQuant::Q8_0,
                values(h * ffn, next_seed(), inv_sqrt(h), 0.0),
            ));
            out.push(tensor(
                name("ffn_up.bias"),
                &[ffn],
                FixtureQuant::F32,
                small(ffn, next_seed()),
            ));
            out.push(tensor(
                name("ffn_down.weight"),
                &[ffn, h],
                FixtureQuant::F16,
                values(ffn * h, next_seed(), inv_sqrt(ffn), 0.0),
            ));
            out.push(tensor(
                name("ffn_down.bias"),
                &[h],
                FixtureQuant::F32,
                small(h, next_seed()),
            ));
        }
        let merged = 4 * h;
        let mid = spec.merger_hidden;
        let proj = spec.projection_dim;
        out.push(tensor(
            "mm.0.weight".to_string(),
            &[merged, mid],
            FixtureQuant::Q8_0,
            values(merged * mid, next_seed(), inv_sqrt(merged), 0.0),
        ));
        out.push(tensor(
            "mm.0.bias".to_string(),
            &[mid],
            FixtureQuant::F32,
            values(mid, next_seed(), 0.1, 0.0),
        ));
        out.push(tensor(
            "mm.2.weight".to_string(),
            &[mid, proj],
            FixtureQuant::Q8_0,
            values(mid * proj, next_seed(), inv_sqrt(mid), 0.0),
        ));
        out.push(tensor(
            "mm.2.bias".to_string(),
            &[proj],
            FixtureQuant::F32,
            values(proj, next_seed(), 0.1, 0.0),
        ));
        out.push(tensor(
            "v.post_ln.weight".to_string(),
            &[h],
            FixtureQuant::F32,
            values(h, next_seed(), 0.2, 1.0),
        ));
        out.push(tensor(
            "v.post_ln.bias".to_string(),
            &[h],
            FixtureQuant::F32,
            values(h, next_seed(), 0.1, 0.0),
        ));
        let p = spec.patch;
        let patch_len = p * p * 3;
        out.push(tensor(
            "v.patch_embd.bias".to_string(),
            &[h],
            FixtureQuant::F32,
            values(h, next_seed(), 0.1, 0.0),
        ));
        out.push(tensor(
            "v.patch_embd.weight".to_string(),
            &[p, p, 3, h],
            FixtureQuant::F32,
            values(patch_len * h, next_seed(), inv_sqrt(patch_len), 0.0),
        ));
        out.push(tensor(
            "v.patch_embd.weight.1".to_string(),
            &[p, p, 3, h],
            FixtureQuant::F32,
            values(patch_len * h, next_seed(), inv_sqrt(patch_len), 0.0),
        ));
        let n_pos = spec.pos_side * spec.pos_side;
        out.push(tensor(
            "v.position_embd.weight".to_string(),
            &[h, n_pos],
            FixtureQuant::F32,
            values(h * n_pos, next_seed(), 0.5, 0.0),
        ));
        out
    }

    /// Write `metadata` and `tensors` as one GGUF file.
    ///
    /// # Errors
    ///
    /// Propagates [`quantize_bytes`] (a value count that is not a multiple
    /// of the format's block size) and the writer's own errors, and reports
    /// [`FixtureError::UnsupportedByWriter`] for a format with no GGUF
    /// writer mapping.
    pub fn assemble(
        metadata: &[(String, MetadataWriteValue)],
        tensors: &[FixtureTensor],
    ) -> Result<Vec<u8>, FixtureError> {
        let mut builder = GgufFixtureBuilder::new();
        for (key, value) in metadata {
            builder.metadata(key, value.clone());
        }
        for t in tensors {
            let tensor_type = t
                .quant
                .writer_type()
                .ok_or(FixtureError::UnsupportedByWriter { quant: t.quant })?;
            let data = quantize_bytes(t.quant, &t.values)?;
            builder.tensor_raw(&t.name, &t.shape, tensor_type, data);
        }
        builder.build()
    }

    /// The complete synthetic projector for `spec`.
    ///
    /// # Errors
    ///
    /// See [`assemble`].
    pub fn synthetic_mmproj_gguf(spec: &MmprojFixtureSpec) -> Result<Vec<u8>, FixtureError> {
        assemble(&metadata(spec), &tensors(spec))
    }

    /// A deterministic `width x height` RGB8 test image (row-major, three
    /// bytes per pixel), built from integer arithmetic only so it is
    /// identical on every platform.
    ///
    /// It is deliberately asymmetric: red is a horizontal ramp, green a
    /// vertical one, blue an irregular block pattern, and a diagonal band
    /// inverts the red channel — so a transposed axis, a swapped colour
    /// plane or a mis-ordered patch changes the picture a vision tower
    /// sees, and different regions of the image look different.
    #[must_use]
    pub fn pattern_rgb8(width: usize, height: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        let wd = width.saturating_sub(1).max(1);
        let hd = height.saturating_sub(1).max(1);
        for y in 0..height {
            for x in 0..width {
                let mut r = x * 255 / wd;
                let g = y * 255 / hd;
                let b = if (x / 24 + y / 20).is_multiple_of(3) {
                    230
                } else {
                    (x * y / 7) % 97 + 40
                };
                if (x + 2 * y) % 64 < 8 {
                    r = 255 - r;
                }
                data.extend([r as u8, g as u8, b as u8]);
            }
        }
        data
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use oxibonsai_core::gguf::reader::GgufFile;
        use oxibonsai_core::GgufTensorType;

        #[test]
        fn tiny_fixture_parses_with_the_real_inventory_and_type_mix() {
            let spec = MmprojFixtureSpec::tiny();
            let bytes = synthetic_mmproj_gguf(&spec).expect("build fixture");
            let gguf = GgufFile::parse(&bytes).expect("parse fixture");
            assert_eq!(gguf.tensors.len(), 12 * spec.blocks + 10);
            assert_eq!(
                gguf.metadata
                    .get_string("general.architecture")
                    .expect("arch"),
                "clip"
            );
            assert_eq!(
                gguf.metadata
                    .get_string("clip.projector_type")
                    .expect("projector"),
                "qwen3vl_merger"
            );
            let qkv = gguf
                .tensors
                .get("v.blk.1.attn_qkv.weight")
                .expect("qkv present");
            assert_eq!(qkv.shape, vec![64, 192]);
            assert_eq!(qkv.tensor_type, GgufTensorType::Q8_0);
            let down = gguf
                .tensors
                .get("v.blk.0.ffn_down.weight")
                .expect("ffn_down present");
            assert_eq!(down.tensor_type, GgufTensorType::F16);
            let patch = gguf
                .tensors
                .get("v.patch_embd.weight.1")
                .expect("second temporal slice present");
            assert_eq!(patch.shape, vec![16, 16, 3, 64]);
            let pos = gguf
                .tensors
                .get("v.position_embd.weight")
                .expect("position embedding present");
            assert_eq!(pos.shape, vec![64, 64]);
        }

        #[test]
        fn fixture_is_deterministic_and_seeded() {
            let spec = MmprojFixtureSpec::tiny();
            let a = synthetic_mmproj_gguf(&spec).expect("first build");
            let b = synthetic_mmproj_gguf(&spec).expect("second build");
            assert_eq!(a, b);
            let mut other = spec.clone();
            other.seed ^= 1;
            let c = synthetic_mmproj_gguf(&other).expect("reseeded build");
            assert_ne!(a, c, "a different seed must change the weights");
        }

        #[test]
        fn pattern_is_sized_deterministic_and_asymmetric() {
            let img = pattern_rgb8(64, 32);
            assert_eq!(img.len(), 64 * 32 * 3);
            assert_eq!(img, pattern_rgb8(64, 32));
            let px = |x: usize, y: usize| img[(y * 64 + x) * 3..(y * 64 + x) * 3 + 3].to_vec();
            // Transposing a coordinate pair changes the pixel...
            assert_ne!(px(5, 1), px(1, 5));
            // ...and so does swapping two channels of one pixel.
            let p = px(40, 20);
            assert!(
                p[0] != p[1] || p[1] != p[2],
                "pixel channels all equal: {p:?}"
            );
        }

        #[test]
        fn assemble_lets_a_test_drop_a_tensor() {
            let spec = MmprojFixtureSpec::tiny();
            let mut all = tensors(&spec);
            all.retain(|t| t.name != "v.patch_embd.bias");
            let bytes = assemble(&metadata(&spec), &all).expect("build");
            let gguf = GgufFile::parse(&bytes).expect("parse");
            assert!(gguf.tensors.get("v.patch_embd.bias").is_none());
            assert_eq!(gguf.tensors.len(), 12 * spec.blocks + 9);
        }
    }
}

/// Collision-free temp-path helpers built on `std::env::temp_dir()`.
///
/// Every helper here resolves through `std::env::temp_dir()` — never a
/// hardcoded absolute path — per this workspace's testing policy.
pub mod temp_path {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A process- and call-unique path under the OS temp directory, with the
    /// given filename `prefix` and `suffix` (e.g. `".gguf"`). Does not
    /// create the file; the caller writes to it (or see
    /// [`write_temp_file`]/[`TempFile`]).
    ///
    /// Uniqueness: process id + a monotonically increasing atomic counter +
    /// wall-clock nanoseconds, so concurrent tests in the same process (or
    /// concurrent `nextest` processes on the same machine) never collide.
    #[must_use]
    pub fn unique_path(prefix: &str, suffix: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxibonsai-testkit-{prefix}-{}-{n}-{nanos}{suffix}",
            std::process::id()
        ));
        path
    }

    /// Write `bytes` to a fresh unique temp file and return its path.
    ///
    /// The caller owns cleanup (the OS temp directory reaps stale files
    /// eventually; use [`TempFile`] for RAII cleanup within one test).
    ///
    /// # Errors
    /// Propagates the underlying `std::fs::write` error.
    pub fn write_temp_file(prefix: &str, suffix: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
        let path = unique_path(prefix, suffix);
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    /// An owned temp file that deletes itself on drop.
    #[derive(Debug)]
    pub struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        /// Write `bytes` to a fresh unique temp file and wrap it for RAII
        /// cleanup.
        ///
        /// # Errors
        /// Propagates the underlying `std::fs::write` error.
        pub fn write(prefix: &str, suffix: &str, bytes: &[u8]) -> std::io::Result<Self> {
            Ok(Self {
                path: write_temp_file(prefix, suffix, bytes)?,
            })
        }

        /// The file's path.
        #[must_use]
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn unique_path_never_collides_across_many_calls() {
            let mut seen = std::collections::HashSet::new();
            for _ in 0..256 {
                let p = unique_path("collision-check", ".bin");
                assert!(seen.insert(p), "unique_path produced a duplicate");
            }
        }

        #[test]
        fn write_temp_file_roundtrips_bytes() {
            let path = write_temp_file("roundtrip", ".gguf", b"GGUF-test-bytes").expect("write");
            let back = std::fs::read(&path).expect("read back");
            assert_eq!(back, b"GGUF-test-bytes");
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn temp_file_deletes_itself_on_drop() {
            let path = {
                let f = TempFile::write("raii", ".gguf", b"data").expect("write");
                let p = f.path().to_path_buf();
                assert!(p.exists());
                p
            };
            assert!(!path.exists(), "TempFile must delete its file on drop");
        }

        #[test]
        fn paths_are_never_hardcoded_absolute_literals() {
            // Regression guard for the project rule "never hardcode
            // absolute paths": every path this module returns must live
            // under std::env::temp_dir(), not some fixed literal.
            let p = unique_path("policy-check", ".tmp");
            assert!(p.starts_with(std::env::temp_dir()));
        }
    }
}

/// Golden-vector comparison helpers shared by every parity/regression test.
pub mod golden {
    /// The largest absolute difference between `actual` and `expected`.
    ///
    /// # Panics
    /// Panics if the two slices have different lengths — this is a test
    /// helper, and a length mismatch is always a test bug, never expected
    /// input worth a `Result`.
    #[must_use]
    pub fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "max_abs_diff: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max)
    }

    /// Cosine similarity between two equal-length vectors, in `[-1.0, 1.0]`
    /// (`1.0` for identical direction). Returns `0.0` if either vector is
    /// all-zero, rather than the `NaN` a raw division would give.
    ///
    /// # Panics
    /// Panics on a length mismatch (see [`max_abs_diff`]).
    #[must_use]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let norm_a: f64 = a
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = b
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)) as f32
    }

    /// Asserts every element of `actual` is within `atol + rtol *
    /// |expected|` of `expected` (numpy's `allclose` convention), panicking
    /// with the first offending index and both values on failure.
    ///
    /// # Panics
    /// On a length mismatch, or the first out-of-tolerance element.
    pub fn assert_allclose(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "assert_allclose: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let bound = atol + rtol * e.abs();
            let diff = (a - e).abs();
            assert!(
                diff <= bound,
                "assert_allclose: index {i}: actual={a} expected={e} diff={diff} > bound={bound} (atol={atol}, rtol={rtol})"
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn max_abs_diff_finds_the_largest_gap() {
            let a = [1.0, 2.0, 3.0];
            let b = [1.0, 2.5, 2.0];
            assert!((max_abs_diff(&a, &b) - 1.0).abs() < 1e-6);
        }

        #[test]
        #[should_panic(expected = "length mismatch")]
        fn max_abs_diff_panics_on_length_mismatch() {
            let _ = max_abs_diff(&[1.0], &[1.0, 2.0]);
        }

        #[test]
        fn cosine_similarity_is_one_for_identical_vectors() {
            let a = [1.0, 2.0, 3.0, -4.0];
            assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_is_minus_one_for_opposite_vectors() {
            let a = [1.0, 2.0, 3.0];
            let b = [-1.0, -2.0, -3.0];
            assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_is_zero_for_orthogonal_vectors() {
            let a = [1.0, 0.0];
            let b = [0.0, 1.0];
            assert!(cosine_similarity(&a, &b).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_of_zero_vector_is_zero_not_nan() {
            let a = [0.0, 0.0, 0.0];
            let b = [1.0, 2.0, 3.0];
            assert_eq!(cosine_similarity(&a, &b), 0.0);
        }

        #[test]
        fn assert_allclose_accepts_within_tolerance() {
            assert_allclose(&[1.0001], &[1.0], 1e-3, 0.0);
        }

        #[test]
        #[should_panic(expected = "diff=")]
        fn assert_allclose_rejects_outside_tolerance() {
            assert_allclose(&[1.1], &[1.0], 1e-3, 0.0);
        }
    }
}

/// Locating this workspace's root and its `models/` directory.
///
/// (Wave-1 gatekeeper OPTIONAL #O7): several real-model acceptance tests
/// (`crates/oxibonsai-core/tests/quant_prism_golden.rs`,
/// `crates/oxibonsai-model/src/gguf_loader.rs`'s own test module — neither
/// owned by this package) silently no-op when `models/` is empty, which is
/// the normal state inside an isolated wave worktree (confirmed empirically
/// this wave: this worktree's own `models/` holds only a `.gitkeep`). This
/// module is the fix's building block: it resolves the real `models/`
/// directory the same, robust way regardless of which crate's test binary
/// calls in, with an env-var override for CI or a custom layout, and it is
/// what those two files should call once they can take this crate as a
/// dev-dependency (see the `Cargo.toml` doc comment — that wiring is a
/// deviation, this helper is not). It deliberately does not hardcode, guess
/// at, or symlink to any *other* checkout's path (e.g. the main tree this
/// worktree was created from): only `$OXIBONSAI_MODELS_DIR`, set by
/// whatever created the environment, can point here at anything outside
/// this workspace's own `models/` directory.
pub mod workspace {
    use std::path::PathBuf;

    /// The workspace root, resolved from this crate's own (fixed at compile
    /// time) `CARGO_MANIFEST_DIR`: every workspace member lives exactly two
    /// directories below the root as `crates/<name>`, so this is stable
    /// regardless of which crate's test binary actually calls in — the same
    /// technique [`crate::capability::report_path`] uses for the workspace
    /// `target/` directory.
    #[must_use]
    pub fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    /// The real-model directory: `$OXIBONSAI_MODELS_DIR` if set to a
    /// non-empty value, else `<workspace root>/models`.
    #[must_use]
    pub fn models_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("OXIBONSAI_MODELS_DIR") {
            if !dir.is_empty() {
                return PathBuf::from(dir);
            }
        }
        root().join("models")
    }

    /// `models_dir().join(filename)`, if that path exists and is a
    /// non-empty file — `None` otherwise (including the common "gitignored,
    /// absent in a fresh worktree" case), so callers can self-skip on
    /// `None` rather than opening a zero-byte placeholder.
    #[must_use]
    pub fn find_model(filename: &str) -> Option<PathBuf> {
        let path = models_dir().join(filename);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() > 0 => Some(path),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        /// Guards this module's two env-mutating tests against each other
        /// (independent of `capability`'s own lock: different env vars).
        static ENV_LOCK: Mutex<()> = Mutex::new(());

        #[test]
        fn root_contains_this_crates_own_cargo_toml_two_levels_up() {
            let candidate = root().join("crates/oxibonsai-testkit/Cargo.toml");
            assert!(
                candidate.exists(),
                "root() = {:?} does not contain this crate at {:?}",
                root(),
                candidate
            );
        }

        #[test]
        fn models_dir_env_override_takes_priority() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // MINOR (verifier wave 3): this path is never opened (only
            // compared against `models_dir()`'s return value), but a fixed
            // absolute literal would still trip a naive "never hardcode
            // absolute paths" policy grep. `std::env::temp_dir()` satisfies
            // the policy and the override-priority assertion equally well.
            let override_path = std::env::temp_dir().join("oxibonsai-testkit-override-check");
            std::env::set_var("OXIBONSAI_MODELS_DIR", &override_path);
            assert_eq!(models_dir(), override_path);
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
        }

        #[test]
        fn find_model_returns_none_for_a_missing_file() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
            assert!(find_model("this-file-does-not-exist-in-any-checkout.gguf").is_none());
        }

        #[test]
        fn find_model_finds_this_crates_own_cargo_toml_as_a_smoke_test() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::env::set_var(
                "OXIBONSAI_MODELS_DIR",
                root().join("crates/oxibonsai-testkit"),
            );
            assert!(
                find_model("Cargo.toml").is_some(),
                "find_model must locate an existing non-empty file"
            );
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
        }
    }
}
