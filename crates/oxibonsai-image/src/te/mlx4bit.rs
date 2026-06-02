//! Pure-Rust MLX 4-bit (packed-affine) weight loader for the Qwen3-4B text
//! encoder.
//!
//! This loads the **native 4-bit safetensors** export
//! (`text_encoder-mlx-4bit/model.safetensors`, ~2.1 GB) directly, instead of
//! the ~15 GB dequantised f32 `.npy` dump. It restores Bonsai's "small model"
//! property end-to-end: the on-disk / distribution footprint drops from 15 GB
//! to 2.1 GB while the forward pass is byte-for-byte the f32 numerics it always
//! was (each linear is dequantised on first `get` and then cached).
//!
//! # Format (MLX `bits=4`, `group_size=64`, LSB-first along the input dim)
//!
//! A quantized linear `[out, in]` is stored as a triple in the safetensors:
//!
//! ```text
//! <name>.weight   U32  [out, in/8]    8 nibbles per u32, LSB-first along `in`
//! <name>.scales   BF16 [out, in/64]   per-group affine scale
//! <name>.biases   BF16 [out, in/64]   per-group affine bias
//! ```
//!
//! Dequant (computed in f32), per output row `o` and input index `i`:
//!
//! ```text
//! code = (weight[o*(in/8) + i/8] >> (4 * (i % 8))) & 0xF   // unsigned 0..15
//! g    = i / 64
//! w[o*in + i] = code * bf16(scales[o*(in/64) + g]) + bf16(biases[o*(in/64) + g])
//! ```
//!
//! This is the standard MLX affine form (`code * scale + bias`), validated to
//! byte-match `mx.dequantize(...)` against the f32 `.npy` oracle (see
//! `examples/te_4bit_check.rs`).
//!
//! Plain (non-quantized) tensors — the RMSNorm vectors — are stored as a single
//! BF16 `<name>.weight` (no `.scales` / `.biases`) and decoded bf16→f32 to a
//! `[len]` vector.
//!
//! # Ownership model
//!
//! [`Mlx4bitModel`] owns the [`memmap2::Mmap`] for the file's lifetime and
//! re-runs the (cheap, header-only) [`SafeTensors::deserialize`] on each
//! `load_tensor` access. This sidesteps the self-referential lifetime that a
//! stored `SafeTensors<'a>` borrowing the owned `Mmap` would create, at the
//! negligible cost of re-parsing the JSON header per loaded tensor (there are
//! ~253 quantized + ~37 plain tensors total, each loaded once and then cached
//! by [`super::weights::TeWeights`]). The mmap itself is never copied.

use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::te::error::{TeError, TeResult};
use crate::te::weights::Tensor;

/// MLX quantization group size (input features per scale/bias).
const GROUP_SIZE: usize = 64;

/// Number of 4-bit nibbles packed into one `u32` weight word.
const NIBBLES_PER_U32: usize = 8;

/// Safetensors suffix for the packed-weight sub-tensor of a quantized module.
const SUFFIX_WEIGHT: &str = ".weight";
/// Safetensors suffix for the per-group scale sub-tensor.
const SUFFIX_SCALES: &str = ".scales";
/// Safetensors suffix for the per-group bias sub-tensor.
const SUFFIX_BIASES: &str = ".biases";

/// Reinterpret a bfloat16 bit pattern as `f32`.
///
/// bfloat16 is the top 16 bits of an IEEE-754 `f32`, so the conversion is an
/// exact left-shift by 16 with no rounding.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Dequantize one MLX 4-bit packed-affine linear into a row-major `[out * in]`
/// f32 buffer.
///
/// `weight` is the U32 packed buffer (`out * in/8` words, 8 LSB-first nibbles
/// each); `scales` / `biases` are the BF16 bit patterns (`out * in/64` each).
/// Each input feature `i` of output row `o` is dequantized as
/// `code * scale[group] + bias[group]` with `code ∈ 0..15` unsigned and
/// `group = i / 64`.
///
/// # Errors
///
/// [`TeError::Shape`] if `in_features` is not a positive multiple of 64, or if
/// any of the three buffers does not have exactly the length implied by
/// `(out_features, in_features)`.
pub fn dequantize_mlx_4bit_affine(
    weight: &[u32],
    scales: &[u16],
    biases: &[u16],
    out_features: usize,
    in_features: usize,
) -> TeResult<Vec<f32>> {
    if in_features == 0 || in_features % GROUP_SIZE != 0 {
        return Err(TeError::Shape(format!(
            "mlx4bit dequant: in_features ({in_features}) must be a positive multiple of {GROUP_SIZE}"
        )));
    }

    let weight_cols = in_features / NIBBLES_PER_U32; // in/8
    let group_cols = in_features / GROUP_SIZE; // in/64

    let expect_weight = out_features * weight_cols;
    if weight.len() != expect_weight {
        return Err(TeError::Shape(format!(
            "mlx4bit dequant: weight len {} != expected {expect_weight} (out={out_features}, in/8={weight_cols})",
            weight.len()
        )));
    }
    let expect_groups = out_features * group_cols;
    if scales.len() != expect_groups {
        return Err(TeError::Shape(format!(
            "mlx4bit dequant: scales len {} != expected {expect_groups} (out={out_features}, in/64={group_cols})",
            scales.len()
        )));
    }
    if biases.len() != expect_groups {
        return Err(TeError::Shape(format!(
            "mlx4bit dequant: biases len {} != expected {expect_groups} (out={out_features}, in/64={group_cols})",
            biases.len()
        )));
    }

    let mut out = vec![0.0f32; out_features * in_features];

    // Dequantize one output row `o` into `row` (`in_features` long). Each row is
    // independent, so this is the unit of parallelism below. Walk the row in
    // 8-nibble words; the group (and thus scale/bias) is constant for every span
    // of 64 input features = 8 words.
    let dequant_row = |o: usize, row: &mut [f32]| {
        let weight_row = o * weight_cols;
        let group_row = o * group_cols;
        for col in 0..weight_cols {
            let word = weight[weight_row + col];
            let i_base = col * NIBBLES_PER_U32;
            let group = i_base / GROUP_SIZE;
            let scale = bf16_to_f32(scales[group_row + group]);
            let bias = bf16_to_f32(biases[group_row + group]);
            for nibble in 0..NIBBLES_PER_U32 {
                let code = ((word >> (4 * nibble)) & 0xF) as f32;
                row[i_base + nibble] = code * scale + bias;
            }
        }
    };

    // This dominates the GPU-TE forward (the no-cache 4-bit source re-dequantises
    // every Linear weight per pass). Rows are independent → split across CPUs via
    // scoped threads (bit-identical to the serial loop; `wasm`-safe — no rayon).
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(out_features.max(1));
    if threads <= 1 || out_features < 4 {
        for (o, row) in out.chunks_mut(in_features).enumerate() {
            dequant_row(o, row);
        }
    } else {
        let rows_per = out_features.div_ceil(threads);
        let dequant_ref = &dequant_row;
        std::thread::scope(|scope| {
            let mut base = 0usize;
            for block in out.chunks_mut(rows_per * in_features) {
                let start = base;
                base += block.len() / in_features;
                scope.spawn(move || {
                    for (r, row) in block.chunks_mut(in_features).enumerate() {
                        dequant_ref(start + r, row);
                    }
                });
            }
        });
    }

    Ok(out)
}

/// A memory-mapped MLX 4-bit text-encoder safetensors file.
///
/// Owns the [`Mmap`] and re-parses the safetensors header per access (see the
/// module docs for the ownership rationale). Tensor names are mapped from the
/// loader's dotted `base` (e.g. `layers.0.self_attn.q_proj`) to the
/// `model.`-prefixed safetensors name (e.g. `model.layers.0.self_attn.q_proj`).
pub struct Mlx4bitModel {
    path: PathBuf,
    mmap: Mmap,
}

impl Mlx4bitModel {
    /// Open and memory-map a 4-bit `model.safetensors` file, validating that it
    /// parses as a safetensors container.
    ///
    /// # Errors
    ///
    /// [`TeError::Io`] if the file cannot be opened or mapped; [`TeError::Npy`]
    /// (reused as the container-parse error variant) if the safetensors header
    /// is malformed.
    pub fn open(path: &Path) -> TeResult<Self> {
        let file = std::fs::File::open(path).map_err(|source| TeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: read-only mapping of a file we do not mutate while mapped —
        // the standard safetensors model-loading pattern (mirrors the DiT
        // converter's `convert::mlx_image` path).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| TeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        // Validate the header parses up front so a corrupt file fails at open.
        SafeTensors::deserialize(&mmap).map_err(|e| TeError::Npy {
            path: path.display().to_string(),
            reason: format!("safetensors header: {e}"),
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
        })
    }

    /// Re-parse the safetensors header over the owned mmap.
    fn view(&self) -> TeResult<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.mmap).map_err(|e| TeError::Npy {
            path: self.path.display().to_string(),
            reason: format!("safetensors header: {e}"),
        })
    }

    /// Load a tensor by the loader's dotted `base` name.
    ///
    /// If `model.{base}.scales` exists → dequantize the U32/BF16 quantized
    /// triple to a `[out, in]` f32 [`Tensor`]; otherwise decode the plain BF16
    /// `model.{base}.weight` to a `[len]` f32 [`Tensor`].
    ///
    /// # Errors
    ///
    /// [`TeError::MissingWeight`] if neither form is present; [`TeError::Npy`]
    /// on a header/dtype problem; [`TeError::Shape`] on a buffer-length or
    /// rank mismatch.
    pub fn load_tensor(&self, base: &str) -> TeResult<Tensor> {
        let st = self.view()?;
        let prefixed = format!("model.{base}");
        let weight_name = format!("{prefixed}{SUFFIX_WEIGHT}");

        // Quantized iff `.scales` (and `.biases`) accompany a U32 `.weight`.
        let scales_name = format!("{prefixed}{SUFFIX_SCALES}");
        let has_scales = st.tensor(&scales_name).is_ok();

        if has_scales {
            return self.load_quantized(&st, base, &prefixed);
        }

        // Plain BF16 vector (RMSNorm). Missing → the tensor truly is absent.
        if st.tensor(&weight_name).is_err() {
            return Err(TeError::MissingWeight {
                name: base.to_string(),
            });
        }
        self.load_plain_bf16(&st, base, &weight_name)
    }

    /// Dequantize the `model.{base}.{weight,scales,biases}` quantized triple.
    fn load_quantized(&self, st: &SafeTensors<'_>, base: &str, prefixed: &str) -> TeResult<Tensor> {
        let weight_name = format!("{prefixed}{SUFFIX_WEIGHT}");
        let scales_name = format!("{prefixed}{SUFFIX_SCALES}");
        let biases_name = format!("{prefixed}{SUFFIX_BIASES}");

        let w_view = st.tensor(&weight_name).map_err(|e| self.parse_err(&e))?;
        let s_view = st.tensor(&scales_name).map_err(|e| self.parse_err(&e))?;
        let b_view = st.tensor(&biases_name).map_err(|e| self.parse_err(&e))?;

        if w_view.dtype() != Dtype::U32 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.weight dtype {:?}, expected U32", w_view.dtype()),
            });
        }
        if s_view.dtype() != Dtype::BF16 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.scales dtype {:?}, expected BF16", s_view.dtype()),
            });
        }
        if b_view.dtype() != Dtype::BF16 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.biases dtype {:?}, expected BF16", b_view.dtype()),
            });
        }

        // weight [out, in/8]; scales/biases [out, in/64].
        let w_shape = w_view.shape();
        if w_shape.len() != 2 {
            return Err(TeError::Shape(format!(
                "{base}.weight rank {}, expected 2",
                w_shape.len()
            )));
        }
        let out_features = w_shape[0];
        let in_features = w_shape[1] * NIBBLES_PER_U32;

        let weight = u32_from_le_bytes(w_view.data());
        let scales = u16_from_le_bytes(s_view.data());
        let biases = u16_from_le_bytes(b_view.data());

        let data =
            dequantize_mlx_4bit_affine(&weight, &scales, &biases, out_features, in_features)?;

        Ok(Tensor {
            data,
            shape: vec![out_features, in_features],
        })
    }

    /// Dequantize **only** the listed `rows` of the quantized tensor
    /// `model.{base}` (shape `[out, cols]`) into a flat `[rows.len() * cols]`
    /// f32 buffer, in the rows' given order.
    ///
    /// This is the RAM-frugal embedding gather: instead of dequantizing the
    /// whole `[vocab, hidden]` table (1.5 GB f32 for Qwen3-4B) just to keep a
    /// handful of token rows, it touches only each requested row's
    /// `cols / 8` packed `u32` words and `cols / 64` scale/bias groups.
    ///
    /// Per row it applies the **identical** `code * scale + bias` affine form as
    /// [`dequantize_mlx_4bit_affine`] (same LSB-first nibble order, same
    /// `group = i / 64`, same bf16→f32 of scale/bias), so for any row `r` the
    /// produced `cols` values are byte-for-byte equal to that row's slice of a
    /// full dequant.
    ///
    /// # Errors
    ///
    /// [`TeError::MissingWeight`] if `model.{base}` is not a quantized triple;
    /// [`TeError::Npy`] on a header/dtype problem; [`TeError::Shape`] if `cols`
    /// is not a positive multiple of 64, if the on-disk `in_features` disagrees
    /// with `cols`, or if any requested row index is `>= out_features`.
    pub fn gather_quant_rows(&self, base: &str, rows: &[usize], cols: usize) -> TeResult<Vec<f32>> {
        if cols == 0 || cols % GROUP_SIZE != 0 {
            return Err(TeError::Shape(format!(
                "mlx4bit gather: cols ({cols}) must be a positive multiple of {GROUP_SIZE}"
            )));
        }

        let st = self.view()?;
        let prefixed = format!("model.{base}");
        let weight_name = format!("{prefixed}{SUFFIX_WEIGHT}");
        let scales_name = format!("{prefixed}{SUFFIX_SCALES}");
        let biases_name = format!("{prefixed}{SUFFIX_BIASES}");

        // The gather only makes sense for the quantized form; a plain BF16
        // vector has no `.scales`, so treat its absence as "no such quantized
        // tensor" (the f32-path fallback in `TeWeights` covers any other case).
        if st.tensor(&scales_name).is_err() {
            return Err(TeError::MissingWeight {
                name: base.to_string(),
            });
        }

        let w_view = st.tensor(&weight_name).map_err(|e| self.parse_err(&e))?;
        let s_view = st.tensor(&scales_name).map_err(|e| self.parse_err(&e))?;
        let b_view = st.tensor(&biases_name).map_err(|e| self.parse_err(&e))?;

        if w_view.dtype() != Dtype::U32 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.weight dtype {:?}, expected U32", w_view.dtype()),
            });
        }
        if s_view.dtype() != Dtype::BF16 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.scales dtype {:?}, expected BF16", s_view.dtype()),
            });
        }
        if b_view.dtype() != Dtype::BF16 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!("{base}.biases dtype {:?}, expected BF16", b_view.dtype()),
            });
        }

        let w_shape = w_view.shape();
        if w_shape.len() != 2 {
            return Err(TeError::Shape(format!(
                "{base}.weight rank {}, expected 2",
                w_shape.len()
            )));
        }
        let out_features = w_shape[0];
        let in_features = w_shape[1] * NIBBLES_PER_U32;
        if in_features != cols {
            return Err(TeError::Shape(format!(
                "{base}: on-disk in_features {in_features} != requested cols {cols}"
            )));
        }

        let weight = u32_from_le_bytes(w_view.data());
        let scales = u16_from_le_bytes(s_view.data());
        let biases = u16_from_le_bytes(b_view.data());

        let weight_cols = cols / NIBBLES_PER_U32; // cols/8
        let group_cols = cols / GROUP_SIZE; // cols/64
        if weight.len() != out_features * weight_cols {
            return Err(TeError::Shape(format!(
                "{base}.weight len {} != expected {} (out={out_features}, cols/8={weight_cols})",
                weight.len(),
                out_features * weight_cols
            )));
        }
        if scales.len() != out_features * group_cols || biases.len() != out_features * group_cols {
            return Err(TeError::Shape(format!(
                "{base}.scales/biases len {}/{} != expected {} (out={out_features}, cols/64={group_cols})",
                scales.len(),
                biases.len(),
                out_features * group_cols
            )));
        }

        let mut out = vec![0.0f32; rows.len() * cols];
        for (dst_row, &r) in rows.iter().enumerate() {
            if r >= out_features {
                return Err(TeError::Shape(format!(
                    "{base}: gather row {r} >= out_features {out_features}"
                )));
            }
            let weight_row = r * weight_cols;
            let group_row = r * group_cols;
            let out_base = dst_row * cols;
            // Identical traversal to `dequantize_mlx_4bit_affine`'s inner loop:
            // walk 8-nibble words, scale/bias constant per 64-input group.
            for col in 0..weight_cols {
                let word = weight[weight_row + col];
                let i_base = col * NIBBLES_PER_U32;
                let group = i_base / GROUP_SIZE;
                let scale = bf16_to_f32(scales[group_row + group]);
                let bias = bf16_to_f32(biases[group_row + group]);
                let dst = out_base + i_base;
                for nibble in 0..NIBBLES_PER_U32 {
                    let code = ((word >> (4 * nibble)) & 0xF) as f32;
                    out[dst + nibble] = code * scale + bias;
                }
            }
        }
        Ok(out)
    }

    /// Decode a plain BF16 `[len]` vector (an RMSNorm weight).
    fn load_plain_bf16(
        &self,
        st: &SafeTensors<'_>,
        base: &str,
        weight_name: &str,
    ) -> TeResult<Tensor> {
        let view = st.tensor(weight_name).map_err(|e| self.parse_err(&e))?;
        if view.dtype() != Dtype::BF16 {
            return Err(TeError::Npy {
                path: self.path.display().to_string(),
                reason: format!(
                    "{base}.weight (plain) dtype {:?}, expected BF16",
                    view.dtype()
                ),
            });
        }
        let bits = u16_from_le_bytes(view.data());
        let numel: usize = view.shape().iter().product();
        if bits.len() < numel {
            return Err(TeError::Shape(format!(
                "{base}.weight: bf16 payload short ({} < {numel})",
                bits.len()
            )));
        }
        let data: Vec<f32> = bits[..numel].iter().map(|&b| bf16_to_f32(b)).collect();
        Ok(Tensor {
            data,
            shape: view.shape().to_vec(),
        })
    }

    /// Map a safetensors error into the TE parse-error variant.
    fn parse_err(&self, e: &safetensors::SafeTensorError) -> TeError {
        TeError::Npy {
            path: self.path.display().to_string(),
            reason: e.to_string(),
        }
    }
}

/// Decode a little-endian `u32` buffer from raw bytes (a trailing partial word
/// is ignored; callers validate the element count against the expected shape).
fn u32_from_le_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode a little-endian `u16` (bf16 bit-pattern) buffer from raw bytes.
fn u16_from_le_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip `f32` → bf16 bits via `half`, for synthetic scale/bias buffers.
    fn f32_to_bf16(value: f32) -> u16 {
        half::bf16::from_f32(value).to_bits()
    }

    /// Pack 8 nibble codes (LSB-first) into one `u32` word.
    fn pack_word(codes: &[u8; 8]) -> u32 {
        let mut w = 0u32;
        for (n, &c) in codes.iter().enumerate() {
            w |= ((c & 0xF) as u32) << (4 * n);
        }
        w
    }

    #[test]
    fn bf16_to_f32_exact_for_representable() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.0625, 2000.0, 0.125] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v, "value {v}");
        }
    }

    #[test]
    fn single_group_affine_matches_formula() {
        // One row, in=64 → 8 words, 1 group. code in 0..15 cycling.
        let in_features = 64usize;
        let scale = 0.125f32; // bf16-exact
        let bias = -1.0f32; // bf16-exact
        let mut weight = Vec::new();
        let mut expected = Vec::new();
        for w in 0..(in_features / 8) {
            let mut codes = [0u8; 8];
            for (n, c) in codes.iter_mut().enumerate() {
                let code = ((w * 8 + n) % 16) as u8;
                *c = code;
                expected.push(code as f32 * scale + bias);
            }
            weight.push(pack_word(&codes));
        }
        let scales = vec![f32_to_bf16(scale)];
        let biases = vec![f32_to_bf16(bias)];

        let out =
            dequantize_mlx_4bit_affine(&weight, &scales, &biases, 1, in_features).expect("dequant");
        assert_eq!(out, expected);
    }

    #[test]
    fn multi_row_multi_group_distinct_scales() {
        // out=3, in=128 → 16 words/row, 2 groups/row.
        let out = 3usize;
        let in_features = 128usize;
        let words_per_row = in_features / 8; // 16
        let groups_per_row = in_features / 64; // 2

        let mut weight = vec![0u32; out * words_per_row];
        let mut scales = vec![0u16; out * groups_per_row];
        let mut biases = vec![0u16; out * groups_per_row];
        let mut expected = vec![0.0f32; out * in_features];

        for o in 0..out {
            for g in 0..groups_per_row {
                let scale = 1.0f32 / ((1 << (o + g + 1)) as f32); // bf16-exact
                let bias = -(1.0f32 / ((1 << (o + 1)) as f32)); // bf16-exact
                scales[o * groups_per_row + g] = f32_to_bf16(scale);
                biases[o * groups_per_row + g] = f32_to_bf16(bias);
                for wlocal in 0..(64 / 8) {
                    let w_idx = g * (64 / 8) + wlocal;
                    let mut codes = [0u8; 8];
                    for (n, c) in codes.iter_mut().enumerate() {
                        let i = g * 64 + wlocal * 8 + n;
                        let code = ((o + i) % 16) as u8;
                        *c = code;
                        expected[o * in_features + i] = code as f32 * scale + bias;
                    }
                    weight[o * words_per_row + w_idx] = pack_word(&codes);
                }
            }
        }

        let out_buf = dequantize_mlx_4bit_affine(&weight, &scales, &biases, out, in_features)
            .expect("dequant");
        assert_eq!(out_buf, expected);
    }

    #[test]
    fn nibble_order_is_lsb_first() {
        // A word with distinct nibbles 0..7 must map left-to-right to inputs
        // 0..7 (LSB-first along the input dim).
        let codes = [0u8, 1, 2, 3, 4, 5, 6, 7];
        let weight = vec![pack_word(&codes)];
        let scales = vec![f32_to_bf16(1.0)];
        let biases = vec![f32_to_bf16(0.0)];
        // in=64 needs 8 words; pad the rest with zero words (codes all 0).
        let mut full = weight;
        full.extend(std::iter::repeat_n(0u32, 7));
        let out = dequantize_mlx_4bit_affine(&full, &scales, &biases, 1, 64).expect("dequant");
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(out[i], c as f32, "input {i}");
        }
    }

    #[test]
    fn rejects_unaligned_in_features() {
        let err = dequantize_mlx_4bit_affine(&[0u32; 1], &[0u16; 1], &[0u16; 1], 1, 60)
            .expect_err("in not multiple of 64 must error");
        assert!(matches!(err, TeError::Shape(_)));
    }

    #[test]
    fn rejects_wrong_weight_len() {
        // in=64 → 8 words/row expected; give 4.
        let err = dequantize_mlx_4bit_affine(&[0u32; 4], &[0u16; 1], &[0u16; 1], 1, 64)
            .expect_err("short weight must error");
        assert!(matches!(err, TeError::Shape(_)));
    }

    /// A raw-bytes [`safetensors::View`] so a test can serialize synthetic
    /// quantized tensors into an in-memory safetensors container.
    struct RawTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl safetensors::View for RawTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.bytes)
        }
        fn data_len(&self) -> usize {
            self.bytes.len()
        }
    }

    fn u32_le_bytes(words: &[u32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(words.len() * 4);
        for &w in words {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    fn u16_le_bytes(words: &[u16]) -> Vec<u8> {
        let mut b = Vec::with_capacity(words.len() * 2);
        for &w in words {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    /// `gather_quant_rows` must reproduce, per row, byte-identical values to a
    /// full `dequantize_mlx_4bit_affine` of the same quantized tensor.
    #[test]
    fn gather_rows_bit_equal_to_full_dequant() {
        // out=5, cols=128 → 16 words/row, 2 groups/row.
        let out = 5usize;
        let cols = 128usize;
        let words_per_row = cols / NIBBLES_PER_U32; // 16
        let groups_per_row = cols / GROUP_SIZE; // 2

        let mut weight = vec![0u32; out * words_per_row];
        let mut scales = vec![0u16; out * groups_per_row];
        let mut biases = vec![0u16; out * groups_per_row];

        // Distinct (bf16-exact) scale/bias per (row, group); pseudo-random codes.
        for o in 0..out {
            for g in 0..groups_per_row {
                let scale = 1.0f32 / ((1 << (o + g + 1)) as f32);
                let bias = -(1.0f32 / ((1 << (o + 1)) as f32));
                scales[o * groups_per_row + g] = f32_to_bf16(scale);
                biases[o * groups_per_row + g] = f32_to_bf16(bias);
            }
            for col in 0..words_per_row {
                let mut codes = [0u8; 8];
                for (n, c) in codes.iter_mut().enumerate() {
                    *c = (((o * 31 + col * 7 + n * 3) ^ (o + col)) % 16) as u8;
                }
                weight[o * words_per_row + col] = pack_word(&codes);
            }
        }

        let full =
            dequantize_mlx_4bit_affine(&weight, &scales, &biases, out, cols).expect("full dequant");

        // Serialize an in-memory safetensors with the `model.{base}.*` names the
        // loader expects, write to a temp file, and open it.
        let base = "layers.0.self_attn.q_proj";
        let prefixed = format!("model.{base}");
        let tensors = vec![
            (
                format!("{prefixed}{SUFFIX_WEIGHT}"),
                RawTensor {
                    dtype: Dtype::U32,
                    shape: vec![out, words_per_row],
                    bytes: u32_le_bytes(&weight),
                },
            ),
            (
                format!("{prefixed}{SUFFIX_SCALES}"),
                RawTensor {
                    dtype: Dtype::BF16,
                    shape: vec![out, groups_per_row],
                    bytes: u16_le_bytes(&scales),
                },
            ),
            (
                format!("{prefixed}{SUFFIX_BIASES}"),
                RawTensor {
                    dtype: Dtype::BF16,
                    shape: vec![out, groups_per_row],
                    bytes: u16_le_bytes(&biases),
                },
            ),
        ];
        let blob = safetensors::serialize(tensors, None).expect("serialize");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxibonsai_te_gather_test_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &blob).expect("write temp safetensors");

        let model = Mlx4bitModel::open(&path).expect("open synthetic model");

        // Gather a non-contiguous, repeated, out-of-order set of rows.
        let rows = [3usize, 0, 4, 0, 2];
        let gathered = model
            .gather_quant_rows(base, &rows, cols)
            .expect("gather rows");
        assert_eq!(gathered.len(), rows.len() * cols);

        for (dst_row, &r) in rows.iter().enumerate() {
            let got = &gathered[dst_row * cols..(dst_row + 1) * cols];
            let want = &full[r * cols..(r + 1) * cols];
            // Bit-for-bit equality (not just approximate).
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {r} (dst {dst_row}) differs from full dequant"
            );
        }

        // Out-of-range row index must error, not panic.
        let err = model
            .gather_quant_rows(base, &[out], cols)
            .expect_err("row >= out_features must error");
        assert!(matches!(err, TeError::Shape(_)));

        let _ = std::fs::remove_file(&path);
    }
}
