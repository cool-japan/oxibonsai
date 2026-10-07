//! An independent `f64` evaluation of the Qwen3-VL vision projector graph,
//! straight from a GGUF's bytes — the ground truth the `f32` tower is tested
//! against (bonsai2-design.md §6.2 and §8.2).
//!
//! # Independence
//!
//! Nothing here calls the tower's code. The only thing shared with it is
//! the GGUF container parser (`oxibonsai_core::gguf`). Everything else is
//! written out a second time, from the reference sources rather than from
//! the tower:
//!
//! - weights are decoded from the raw tensor bytes by this file's own
//!   `F32` / `F16` / `BF16` / `Q8_0` readers (`Q8_0`: 34-byte blocks, `d` as
//!   `f16` first, then 32 signed bytes);
//! - the patch convolution is ggml's `conv_2d` (im2col + matmul) written as
//!   a direct sum, run once per temporal slice and added, exactly like
//!   `build_inp_with_temporal_merge`;
//! - the merge-window reordering and the position-embedding resize are
//!   **emulations of the ggml tensor operations** the reference graph uses
//!   (`permute`, `cont`, `reshape`, `upscale` in bilinear align-corners
//!   mode) on a small strided-view type, not a closed-form index formula —
//!   so the tower's formula is checked against the op sequence itself;
//! - the rotary embedding transliterates `ggml_mrope_cache_init` (vision
//!   mode) and `rotate_pairs(ne0, n_dims)` into `f64`, fed with the position
//!   table built by the same loop the reference's input setup uses;
//! - LayerNorm, the tanh GELU, the softmax attention and every matrix
//!   product accumulate in `f64`.
//!
//! # Compiled in two places
//!
//! The crate declares this file as a `#[cfg(test)]` module, and
//! `tests/vision_mmproj_tests.rs` includes it by path for the real-weight
//! check. It therefore uses only `std`, `oxibonsai_core`, `half` and
//! `rayon` — no path into the crate itself — and reports errors as plain
//! `String`s.
//!
//! # Memory
//!
//! Matrices are never materialised: each linear layer decodes one weight
//! row at a time (per Rayon worker) and applies it to every token, so the
//! real projector is evaluated with a few MiB of `f64` scratch.

#![allow(dead_code)]

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use rayon::prelude::*;

/// Result type of the reference: errors are descriptive strings.
pub type RefResult<T> = Result<T, String>;

/// Frequency base of the ViT's rotary embedding (hard-coded in the
/// reference graph).
pub const ROPE_FREQ_BASE: f64 = 10_000.0;

// ─────────────────────────────────────────────────────────────────────────
//  Metadata
// ─────────────────────────────────────────────────────────────────────────

/// The hyper-parameters the graph needs, read from the metadata and two
/// tensor shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct RefConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub blocks: usize,
    pub patch: usize,
    pub eps: f64,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub projection_dim: usize,
    pub merger_hidden: usize,
    pub pos_side: usize,
}

fn meta_usize(gguf: &GgufFile<'_>, key: &str) -> RefResult<usize> {
    gguf.metadata
        .get(key)
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .ok_or_else(|| format!("metadata key {key} missing or not an integer"))
}

fn meta_f64(gguf: &GgufFile<'_>, key: &str) -> RefResult<f64> {
    gguf.metadata
        .get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("metadata key {key} missing or not a float"))
}

fn meta_triple(gguf: &GgufFile<'_>, key: &str) -> RefResult<[f64; 3]> {
    let values = gguf
        .metadata
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("metadata key {key} missing or not an array"))?;
    if values.len() < 3 {
        return Err(format!("{key}: expected 3 values, found {}", values.len()));
    }
    let mut out = [0.0; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .ok_or_else(|| format!("{key}: element is not a float"))?;
    }
    Ok(out)
}

fn shape_of(gguf: &GgufFile<'_>, name: &str) -> RefResult<Vec<usize>> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor {name} missing"))?;
    Ok(info.shape.iter().map(|&d| d as usize).collect())
}

/// Read the graph's hyper-parameters from `gguf`.
///
/// # Errors
///
/// A description of the first missing or malformed key or tensor shape.
pub fn read_config(gguf: &GgufFile<'_>) -> RefResult<RefConfig> {
    let hidden = meta_usize(gguf, "clip.vision.embedding_length")?;
    let heads = meta_usize(gguf, "clip.vision.attention.head_count")?;
    if heads == 0 || !hidden.is_multiple_of(heads) {
        return Err(format!("hidden {hidden} not divisible by {heads} heads"));
    }
    let pos_shape = shape_of(gguf, "v.position_embd.weight")?;
    let n_pos = *pos_shape
        .get(1)
        .ok_or("v.position_embd.weight is not 2-D")?;
    let pos_side = (n_pos as f64).sqrt().round() as usize;
    if pos_side * pos_side != n_pos {
        return Err(format!("{n_pos} positions is not a square grid"));
    }
    let mm0 = shape_of(gguf, "mm.0.weight")?;
    let merger_hidden = *mm0.get(1).ok_or("mm.0.weight is not 2-D")?;
    Ok(RefConfig {
        hidden,
        heads,
        head_dim: hidden / heads,
        ffn: meta_usize(gguf, "clip.vision.feed_forward_length")?,
        blocks: meta_usize(gguf, "clip.vision.block_count")?,
        patch: meta_usize(gguf, "clip.vision.patch_size")?,
        eps: meta_f64(gguf, "clip.vision.attention.layer_norm_epsilon")?,
        mean: meta_triple(gguf, "clip.vision.image_mean")?,
        std: meta_triple(gguf, "clip.vision.image_std")?,
        projection_dim: meta_usize(gguf, "clip.vision.projection_dim")?,
        merger_hidden,
        pos_side,
    })
}

// ─────────────────────────────────────────────────────────────────────────
//  Raw tensor decoding
// ─────────────────────────────────────────────────────────────────────────

/// A tensor's raw bytes and its ggml shape (`shape[0]` fastest).
pub struct RawTensor<'a> {
    name: String,
    ty: GgufTensorType,
    data: &'a [u8],
    shape: Vec<usize>,
}

/// Look up tensor `name`.
///
/// # Errors
///
/// When the tensor is missing or its bytes lie outside the file.
pub fn raw_tensor<'a>(gguf: &GgufFile<'a>, name: &str) -> RefResult<RawTensor<'a>> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor {name} missing"))?;
    let data = gguf
        .tensor_data(name)
        .map_err(|e| format!("tensor {name}: {e}"))?;
    Ok(RawTensor {
        name: name.to_string(),
        ty: info.tensor_type,
        data,
        shape: info.shape.iter().map(|&d| d as usize).collect(),
    })
}

impl RawTensor<'_> {
    /// Elements per row (`ne0`).
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Number of rows (the product of the remaining dimensions).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    /// Decode row `r` into `out` (`row_len` values).
    ///
    /// # Errors
    ///
    /// For a storage type this reference does not decode, a short buffer
    /// or a row outside the tensor.
    pub fn decode_row(&self, r: usize, out: &mut [f64]) -> RefResult<()> {
        let n = self.row_len();
        if out.len() != n || r >= self.rows() {
            return Err(format!(
                "{}: row {r} of {} ({} values) into a buffer of {}",
                self.name,
                self.rows(),
                n,
                out.len()
            ));
        }
        let fetch = |start: usize, len: usize| {
            self.data.get(start..start + len).ok_or_else(|| {
                format!("{}: bytes {start}..{} out of range", self.name, start + len)
            })
        };
        match self.ty {
            GgufTensorType::F32 => {
                let bytes = fetch(r * n * 4, n * 4)?;
                for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
                    *o = f64::from(f32::from_le_bytes(*c));
                }
            }
            GgufTensorType::F16 => {
                let bytes = fetch(r * n * 2, n * 2)?;
                for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                    *o = half::f16::from_bits(u16::from_le_bytes(*c)).to_f64();
                }
            }
            GgufTensorType::BF16 => {
                let bytes = fetch(r * n * 2, n * 2)?;
                for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
                    let bits = u32::from(u16::from_le_bytes(*c)) << 16;
                    *o = f64::from(f32::from_bits(bits));
                }
            }
            GgufTensorType::Q8_0 => {
                // 34-byte blocks of 32: `d` (f16) first, then 32 x i8.
                const QK: usize = 32;
                const BYTES: usize = 34;
                if !n.is_multiple_of(QK) {
                    return Err(format!("{}: Q8_0 row of {n} values", self.name));
                }
                let blocks = n / QK;
                let bytes = fetch(r * blocks * BYTES, blocks * BYTES)?;
                for (b, block) in bytes.as_chunks::<BYTES>().0.iter().enumerate() {
                    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f64();
                    for (j, &q) in block[2..].iter().enumerate() {
                        out[b * QK + j] = d * f64::from(q as i8);
                    }
                }
            }
            other => {
                return Err(format!(
                    "{}: storage type {} not decoded by the f64 reference",
                    self.name,
                    other.name()
                ));
            }
        }
        Ok(())
    }

    /// Decode the whole tensor, row after row.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_row`].
    pub fn decode_all(&self) -> RefResult<Vec<f64>> {
        let n = self.row_len();
        let mut out = vec![0.0; n * self.rows()];
        for (r, row) in out.chunks_mut(n.max(1)).enumerate() {
            self.decode_row(r, row)?;
        }
        Ok(out)
    }
}

fn vector(gguf: &GgufFile<'_>, name: &str, len: usize) -> RefResult<Vec<f64>> {
    let t = raw_tensor(gguf, name)?;
    let v = t.decode_all()?;
    if v.len() != len {
        return Err(format!("{name}: {} values, expected {len}", v.len()));
    }
    Ok(v)
}

/// `y[t][o] = sum_i x[t][i] * W[o][i] (+ b[o])` for `n` rows of `x`, with
/// `W` decoded one row at a time.
///
/// # Errors
///
/// On a shape mismatch or a decoding error.
pub fn linear(x: &[f64], n: usize, w: &RawTensor<'_>, bias: Option<&[f64]>) -> RefResult<Vec<f64>> {
    let k = w.row_len();
    let m = w.rows();
    if x.len() != n * k {
        return Err(format!(
            "{}: input holds {} values, expected {n} x {k}",
            w.name,
            x.len()
        ));
    }
    if let Some(b) = bias {
        if b.len() != m {
            return Err(format!("{}: bias of {} for {m} outputs", w.name, b.len()));
        }
    }
    let columns = (0..m)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; k],
            |row, o| -> RefResult<Vec<f64>> {
                w.decode_row(o, row)?;
                let b = bias.map_or(0.0, |b| b[o]);
                Ok((0..n)
                    .map(|t| dot(&x[t * k..(t + 1) * k], row) + b)
                    .collect())
            },
        )
        .collect::<RefResult<Vec<_>>>()?;
    let mut y = vec![0.0; n * m];
    for (o, column) in columns.iter().enumerate() {
        for (t, value) in column.iter().enumerate() {
            y[t * m + o] = *value;
        }
    }
    Ok(y)
}

/// A four-accumulator `f64` dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (ca, a_remainder) = a.as_chunks::<4>();
    let (cb, b_remainder) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb.iter()) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in a_remainder.iter().zip(b_remainder) {
        sum += x * y;
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────
//  ggml tensor-op emulation
// ─────────────────────────────────────────────────────────────────────────

/// A strided view with ggml's conventions: `ne[0]` is the fastest-varying
/// axis and `nb[i]` the element stride of axis `i`.
#[derive(Clone, Debug)]
pub struct GTensor {
    data: Arc<[f64]>,
    ne: [usize; 4],
    nb: [usize; 4],
}

impl GTensor {
    /// A contiguous tensor over `data` with shape `ne`.
    ///
    /// # Errors
    ///
    /// When `data.len()` is not the product of `ne`.
    pub fn new(data: Vec<f64>, ne: [usize; 4]) -> RefResult<Self> {
        if data.len() != ne.iter().product::<usize>() {
            return Err(format!("{} values for shape {ne:?}", data.len()));
        }
        Ok(Self {
            data: data.into(),
            ne,
            nb: [1, ne[0], ne[0] * ne[1], ne[0] * ne[1] * ne[2]],
        })
    }

    /// Shape.
    #[must_use]
    pub fn ne(&self) -> [usize; 4] {
        self.ne
    }

    /// Element `(i0, i1, i2, i3)`.
    #[must_use]
    pub fn at(&self, i: [usize; 4]) -> f64 {
        self.data[i[0] * self.nb[0] + i[1] * self.nb[1] + i[2] * self.nb[2] + i[3] * self.nb[3]]
    }

    /// `ggml_permute(a, axes[0], axes[1], axes[2], axes[3])`: source axis
    /// `i` becomes destination axis `axes[i]`.
    #[must_use]
    pub fn permute(&self, axes: [usize; 4]) -> Self {
        let mut ne = [0; 4];
        let mut nb = [0; 4];
        for (i, &axis) in axes.iter().enumerate() {
            ne[axis] = self.ne[i];
            nb[axis] = self.nb[i];
        }
        Self {
            data: Arc::clone(&self.data),
            ne,
            nb,
        }
    }

    /// `ggml_cont`: materialise in logical order (`i0` fastest).
    #[must_use]
    pub fn cont(&self) -> Self {
        let mut out = Vec::with_capacity(self.ne.iter().product());
        for i3 in 0..self.ne[3] {
            for i2 in 0..self.ne[2] {
                for i1 in 0..self.ne[1] {
                    for i0 in 0..self.ne[0] {
                        out.push(self.at([i0, i1, i2, i3]));
                    }
                }
            }
        }
        let ne = self.ne;
        Self {
            data: out.into(),
            ne,
            nb: [1, ne[0], ne[0] * ne[1], ne[0] * ne[1] * ne[2]],
        }
    }

    fn is_contiguous(&self) -> bool {
        let ne = self.ne;
        self.nb == [1, ne[0], ne[0] * ne[1], ne[0] * ne[1] * ne[2]]
    }

    /// `ggml_reshape_*`: reinterpret a contiguous tensor.
    ///
    /// # Errors
    ///
    /// When the tensor is not contiguous or the element counts differ.
    pub fn reshape(&self, ne: [usize; 4]) -> RefResult<Self> {
        if !self.is_contiguous() {
            return Err("reshape of a non-contiguous view".to_string());
        }
        if ne.iter().product::<usize>() != self.ne.iter().product::<usize>() {
            return Err(format!("reshape {:?} -> {ne:?}", self.ne));
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            ne,
            nb: [1, ne[0], ne[0] * ne[1], ne[0] * ne[1] * ne[2]],
        })
    }

    /// `ggml_cont_4d(a, ne...)`: `cont` then `reshape`.
    ///
    /// # Errors
    ///
    /// As [`Self::reshape`].
    pub fn cont_as(&self, ne: [usize; 4]) -> RefResult<Self> {
        self.cont().reshape(ne)
    }

    /// The elements in logical order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        self.cont().data.to_vec()
    }
}

/// `ggml_conv_2d(kernel, input, s, s, 0, 0, 1, 1)` for a kernel
/// `[KW, KH, C, OC]` and an input `[W, H, C, 1]`, returning
/// `[OW, OH, OC, 1]`: `dst[ow, oh, oc] = sum_{c, kh, kw} K[kw, kh, c, oc] *
/// I[ow * s + kw, oh * s + kh, c]`.
///
/// # Errors
///
/// On a channel-count mismatch.
pub fn conv2d(kernel: &GTensor, input: &GTensor, stride: usize) -> RefResult<GTensor> {
    let [kw, kh, kc, oc] = kernel.ne();
    let [w, h, c, n] = input.ne();
    if kc != c || n != 1 {
        return Err(format!(
            "conv2d kernel {:?} on input {:?}",
            kernel.ne(),
            input.ne()
        ));
    }
    let ow = (w - kw) / stride + 1;
    let oh = (h - kh) / stride + 1;
    let mut out = vec![0.0; ow * oh * oc];
    out.par_chunks_mut(ow * oh)
        .enumerate()
        .for_each(|(o, plane)| {
            for y in 0..oh {
                for x in 0..ow {
                    let mut acc = 0.0;
                    for ci in 0..c {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                acc += kernel.at([kx, ky, ci, o])
                                    * input.at([x * stride + kx, y * stride + ky, ci, 0]);
                            }
                        }
                    }
                    plane[y * ow + x] = acc;
                }
            }
        });
    GTensor::new(out, [ow, oh, oc, 1])
}

/// ggml `upscale` in `BILINEAR | ALIGN_CORNERS` mode, resizing axes 0 and 1
/// of `src` (`[W0, H0, C, N]`) to `w x h`, evaluated in exact `f64`: scale
/// `(out - 1) / (in - 1)` (or `out / in` when a side is 1), source
/// coordinate `i / scale`, neighbours clamped to the grid, fraction from
/// the clamped lower neighbour clamped to `[0, 1]`.
///
/// # Errors
///
/// For an empty target.
pub fn upscale_bilinear_align_corners(src: &GTensor, w: usize, h: usize) -> RefResult<GTensor> {
    let [w0, h0, c, n] = src.ne();
    if w == 0 || h == 0 || w0 == 0 || h0 == 0 {
        return Err("empty upscale".to_string());
    }
    let taps = |out: usize, len: usize| -> Vec<(usize, usize, f64)> {
        let scale = if out > 1 && len > 1 {
            (out - 1) as f64 / (len - 1) as f64
        } else {
            out as f64 / len as f64
        };
        (0..out)
            .map(|i| {
                let s = i as f64 / scale;
                let lo = s.floor() as i64;
                let last = len as i64 - 1;
                let lo_c = lo.clamp(0, last);
                let hi_c = (lo + 1).clamp(0, last);
                let frac = (s - lo_c as f64).clamp(0.0, 1.0);
                (lo_c as usize, hi_c as usize, frac)
            })
            .collect()
    };
    let xs = taps(w, w0);
    let ys = taps(h, h0);
    let mut out = Vec::with_capacity(w * h * c * n);
    for i3 in 0..n {
        for i2 in 0..c {
            for &(y0, y1, dy) in &ys {
                for &(x0, x1, dx) in &xs {
                    let a = src.at([x0, y0, i2, i3]);
                    let b = src.at([x1, y0, i2, i3]);
                    let cc = src.at([x0, y1, i2, i3]);
                    let d = src.at([x1, y1, i2, i3]);
                    out.push(
                        a * (1.0 - dx) * (1.0 - dy)
                            + b * dx * (1.0 - dy)
                            + cc * (1.0 - dx) * dy
                            + d * dx * dy,
                    );
                }
            }
        }
    }
    GTensor::new(out, [w, h, c, n])
}

/// The spatial-merge re-layout of `clip_graph_qwen3vl::build`, applied to a
/// `[pw, ph, n_embd, 1]` tensor (a convolution output):
///
/// ```text
/// permute(1, 2, 0, 3) -> cont_4d(2 n_embd, pw/2, ph, 1)
///   -> reshape_4d(2 n_embd, pw/2, 2, ph/2) -> permute(0, 2, 1, 3)
///   -> cont_3d(n_embd, pw * ph, 1)
/// ```
///
/// # Errors
///
/// For odd grid sides or a reshape failure.
pub fn spatial_merge_layout(t: &GTensor) -> RefResult<GTensor> {
    let [pw, ph, n_embd, _] = t.ne();
    if !pw.is_multiple_of(2) || !ph.is_multiple_of(2) {
        return Err(format!("odd patch grid {pw} x {ph}"));
    }
    let t = t.permute([1, 2, 0, 3]);
    let t = t.cont_as([n_embd * 2, pw / 2, ph, 1])?;
    let t = t.reshape([n_embd * 2, pw / 2, 2, ph / 2])?;
    let t = t.permute([0, 2, 1, 3]);
    t.cont_as([n_embd, pw * ph, 1, 1])
}

/// The same re-layout applied to a `[n_embd, pw * ph]` raster-order tensor
/// (the resized position embedding), which the reference feeds straight
/// into `cont_4d(2 n_embd, pw/2, ph, 1)`.
///
/// # Errors
///
/// As [`spatial_merge_layout`].
pub fn spatial_merge_layout_rows(t: &GTensor, pw: usize, ph: usize) -> RefResult<GTensor> {
    let [n_embd, n, _, _] = t.ne();
    if n != pw * ph || !pw.is_multiple_of(2) || !ph.is_multiple_of(2) {
        return Err(format!("{n} rows for a {pw} x {ph} grid"));
    }
    let t = t.cont_as([n_embd * 2, pw / 2, ph, 1])?;
    let t = t.reshape([n_embd * 2, pw / 2, 2, ph / 2])?;
    let t = t.permute([0, 2, 1, 3]);
    t.cont_as([n_embd, pw * ph, 1, 1])
}

/// `resize_position_embeddings(BILINEAR | ALIGN_CORNERS)`: the stored
/// `[n_embd, side * side]` table resized to `[n_embd, pw * ph]` (raster
/// order) through the reference's `reshape_3d -> permute(2, 0, 1, 3) ->
/// interpolate -> permute(1, 2, 0, 3) -> cont_2d` sequence; returned
/// unchanged when the grid already matches.
///
/// # Errors
///
/// On a malformed table.
pub fn resize_position_embeddings(
    table: &[f64],
    n_embd: usize,
    side: usize,
    pw: usize,
    ph: usize,
) -> RefResult<GTensor> {
    let pos = GTensor::new(table.to_vec(), [n_embd, side * side, 1, 1])?;
    if pw == side && ph == side {
        return Ok(pos);
    }
    let pos = pos.reshape([n_embd, side, side, 1])?;
    let pos = pos.permute([2, 0, 1, 3]);
    let pos = upscale_bilinear_align_corners(&pos, pw, ph)?;
    let pos = pos.permute([1, 2, 0, 3]);
    pos.cont_as([n_embd, pw * ph, 1, 1])
}

/// The `positions` input of the reference for a `pw x ph` patch grid:
/// four channels (`y, x, y, x`) per token, laid out channel-major exactly
/// as the reference's input-setup loop fills them.
#[must_use]
pub fn vision_positions(pw: usize, ph: usize) -> Vec<i64> {
    let n = pw * ph;
    let mut positions = vec![0i64; 4 * n];
    let mut ptr = 0;
    for y in (0..ph).step_by(2) {
        for x in (0..pw).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    positions[ptr] = (y + dy) as i64;
                    positions[n + ptr] = (x + dx) as i64;
                    positions[2 * n + ptr] = (y + dy) as i64;
                    positions[3 * n + ptr] = (x + dx) as i64;
                    ptr += 1;
                }
            }
        }
    }
    positions
}

// ─────────────────────────────────────────────────────────────────────────
//  RoPE (GGML_ROPE_TYPE_VISION), transliterated to f64
// ─────────────────────────────────────────────────────────────────────────

/// `ggml_mrope_cache_init` with `indep_sects = true`, `is_imrope = false`,
/// `freq_scale = 1`, `ext_factor = 0`, `mscale = 1` and no frequency
/// factors, in `f64`: one `(cos, sin)` per rotation pair, `ne0 / 2` pairs.
#[must_use]
pub fn mrope_vision_cache(
    p: [i64; 4],
    sections: [usize; 4],
    ne0: usize,
    n_dims: usize,
    freq_base: f64,
) -> Vec<(f64, f64)> {
    let theta_scale = freq_base.powf(-2.0 / n_dims as f64);
    let base = p.map(|v| v as f64);
    let (mut theta_t, mut theta_h, mut theta_w, mut theta_e) = (base[0], base[1], base[2], base[3]);
    let sect_dims: usize = sections.iter().sum();
    let sec_w = sections[1] + sections[0];
    let sec_e = sections[2] + sec_w;
    let mut cache = Vec::with_capacity(ne0 / 2);
    for i0 in (0..ne0).step_by(2) {
        let sector = (i0 / 2) % sect_dims.max(1);
        if sector == 0 {
            theta_t = base[0];
        } else if sector == sections[0] {
            theta_h = base[1];
        } else if sector == sec_w {
            theta_w = base[2];
        } else if sector == sec_e {
            theta_e = base[3];
        }
        let mut theta = theta_t;
        if sector >= sections[0] && sector < sec_w {
            theta = theta_h;
        } else if sector >= sec_w && sector < sec_w + sections[2] {
            theta = theta_w;
        } else if sector >= sec_w + sections[2] {
            theta = theta_e;
        }
        cache.push((theta.cos(), theta.sin()));
        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
    cache
}

/// `rotate_pairs(ne0, n_offset = n_dims, cache, x, x)` (the vision branch):
/// pair `i0 / 2` couples `x[i0 / 2]` with `x[i0 / 2 + n_dims]`.
pub fn rotate_pairs_vision(x: &mut [f64], n_dims: usize, cache: &[(f64, f64)]) {
    let ne0 = x.len();
    for i0 in (0..ne0).step_by(2) {
        let ic = i0 / 2;
        let (cos_theta, sin_theta) = cache[i0 / 2];
        let x0 = x[ic];
        let x1 = x[ic + n_dims];
        x[ic] = x0 * cos_theta - x1 * sin_theta;
        x[ic + n_dims] = x0 * sin_theta + x1 * cos_theta;
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Elementwise pieces
// ─────────────────────────────────────────────────────────────────────────

/// `ggml_norm` (mean-centred, biased variance) then `* w + b`, per row.
fn layer_norm(x: &[f64], dim: usize, w: &[f64], b: &[f64], eps: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (o, row) in out.chunks_mut(dim).zip(x.chunks(dim)) {
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for (((dst, &v), &wi), &bi) in o.iter_mut().zip(row).zip(w).zip(b) {
            *dst = (v - mean) * inv * wi + bi;
        }
    }
    out
}

/// `ggml_gelu`: `0.5 x (1 + tanh(sqrt(2 / pi) x (1 + 0.044715 x^2)))`.
#[must_use]
pub fn gelu_tanh(x: f64) -> f64 {
    let k = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * x * (1.0 + 0.044_715 * x * x)).tanh())
}

// ─────────────────────────────────────────────────────────────────────────
//  The graph
// ─────────────────────────────────────────────────────────────────────────

/// The reference's output: `n_rows x dim` merged embeddings, row-major.
#[derive(Debug, Clone)]
pub struct RefOutput {
    pub rows: Vec<f64>,
    pub n_rows: usize,
    pub dim: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Normalise RGB8 pixels (`(y * width + x) * 3 + c`) to the planar input
/// (`[3][height][width]`): `(v / 255 - mean) / std`.
///
/// # Errors
///
/// When `rgb` is not `width * height * 3` bytes.
pub fn normalize_rgb8(
    cfg: &RefConfig,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> RefResult<Vec<f64>> {
    if rgb.len() != width * height * 3 {
        return Err(format!(
            "{} bytes for a {width} x {height} image",
            rgb.len()
        ));
    }
    let plane = width * height;
    let mut planar = vec![0.0; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            planar[c * plane + i] = (f64::from(rgb[i * 3 + c]) / 255.0 - cfg.mean[c]) / cfg.std[c];
        }
    }
    Ok(planar)
}

/// Evaluate the projector on an RGB8 image.
///
/// # Errors
///
/// A description of the first problem met.
pub fn encode_rgb8(
    gguf: &GgufFile<'_>,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> RefResult<RefOutput> {
    let cfg = read_config(gguf)?;
    let planar = normalize_rgb8(&cfg, width, height, rgb)?;
    encode_planar(gguf, width, height, &planar)
}

/// Evaluate the projector on an already-normalised planar image
/// (`[3][height][width]`, the reference graph's `inp_raw`).
///
/// # Errors
///
/// A description of the first problem met.
pub fn encode_planar(
    gguf: &GgufFile<'_>,
    width: usize,
    height: usize,
    planar: &[f64],
) -> RefResult<RefOutput> {
    let cfg = read_config(gguf)?;
    let (h, p, heads, hd) = (cfg.hidden, cfg.patch, cfg.heads, cfg.head_dim);
    if !width.is_multiple_of(2 * p) || !height.is_multiple_of(2 * p) || width == 0 || height == 0 {
        return Err(format!(
            "{width} x {height} is not a whole number of merge windows"
        ));
    }
    let (pw, ph) = (width / p, height / p);
    let n = pw * ph;

    // Input and both temporal convolutions (build_inp_with_temporal_merge).
    let inp = GTensor::new(planar.to_vec(), [width, height, 3, 1])?;
    let k0 = GTensor::new(
        raw_tensor(gguf, "v.patch_embd.weight")?.decode_all()?,
        [p, p, 3, h],
    )?;
    let k1 = GTensor::new(
        raw_tensor(gguf, "v.patch_embd.weight.1")?.decode_all()?,
        [p, p, 3, h],
    )?;
    let c0 = conv2d(&k0, &inp, p)?;
    let c1 = conv2d(&k1, &inp, p)?;
    let summed: Vec<f64> = c0
        .to_vec()
        .iter()
        .zip(c1.to_vec())
        .map(|(a, b)| a + b)
        .collect();
    let conv = GTensor::new(summed, [pw, ph, h, 1])?;

    // Merge-window layout, patch bias, resized + re-laid-out position table.
    let mut x = spatial_merge_layout(&conv)?.to_vec();
    let bias = vector(gguf, "v.patch_embd.bias", h)?;
    let table = vector(
        gguf,
        "v.position_embd.weight",
        h * cfg.pos_side * cfg.pos_side,
    )?;
    let pos = resize_position_embeddings(&table, h, cfg.pos_side, pw, ph)?;
    let pos = spatial_merge_layout_rows(&pos, pw, ph)?.to_vec();
    for t in 0..n {
        for c in 0..h {
            x[t * h + c] += bias[c];
            x[t * h + c] += pos[t * h + c];
        }
    }

    // Rotary caches, one per token.
    let positions = vision_positions(pw, ph);
    let sections = [hd / 4; 4];
    let caches: Vec<Vec<(f64, f64)>> = (0..n)
        .map(|t| {
            mrope_vision_cache(
                [
                    positions[t],
                    positions[n + t],
                    positions[2 * n + t],
                    positions[3 * n + t],
                ],
                sections,
                hd,
                hd / 2,
                ROPE_FREQ_BASE,
            )
        })
        .collect();
    let scale = 1.0 / (hd as f64).sqrt();

    for il in 0..cfg.blocks {
        let name = |s: &str| format!("v.blk.{il}.{s}");
        let ln1_w = vector(gguf, &name("ln1.weight"), h)?;
        let ln1_b = vector(gguf, &name("ln1.bias"), h)?;
        let cur = layer_norm(&x, h, &ln1_w, &ln1_b, cfg.eps);
        let qkv_b = vector(gguf, &name("attn_qkv.bias"), 3 * h)?;
        let qkv = linear(
            &cur,
            n,
            &raw_tensor(gguf, &name("attn_qkv.weight"))?,
            Some(&qkv_b),
        )?;

        // Q / K / V views: [d_head, n_head, n_pos] at offsets 0, h, 2h.
        let mut q = vec![0.0; n * h];
        let mut k = vec![0.0; n * h];
        let mut v = vec![0.0; n * h];
        for t in 0..n {
            q[t * h..(t + 1) * h].copy_from_slice(&qkv[t * 3 * h..t * 3 * h + h]);
            k[t * h..(t + 1) * h].copy_from_slice(&qkv[t * 3 * h + h..t * 3 * h + 2 * h]);
            v[t * h..(t + 1) * h].copy_from_slice(&qkv[t * 3 * h + 2 * h..(t + 1) * 3 * h]);
        }
        for (t, cache) in caches.iter().enumerate() {
            for head in 0..heads {
                let range = t * h + head * hd..t * h + (head + 1) * hd;
                rotate_pairs_vision(&mut q[range.clone()], hd / 2, cache);
                rotate_pairs_vision(&mut k[range], hd / 2, cache);
            }
        }

        // softmax(scale * K^T Q) V per head, no mask.
        let per_head: Vec<Vec<f64>> = (0..heads)
            .into_par_iter()
            .map(|head| {
                let mut out = vec![0.0; n * hd];
                let mut scores = vec![0.0; n];
                for tq in 0..n {
                    let qv = &q[tq * h + head * hd..tq * h + (head + 1) * hd];
                    for (tk, s) in scores.iter_mut().enumerate() {
                        let kv = &k[tk * h + head * hd..tk * h + (head + 1) * hd];
                        *s = dot(qv, kv) * scale;
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (tk, s) in scores.iter().enumerate() {
                        let w = s / sum;
                        let vv = &v[tk * h + head * hd..tk * h + (head + 1) * hd];
                        for (o, val) in out[tq * hd..(tq + 1) * hd].iter_mut().zip(vv) {
                            *o += w * val;
                        }
                    }
                }
                out
            })
            .collect();
        let mut attn = vec![0.0; n * h];
        for (head, out) in per_head.iter().enumerate() {
            for t in 0..n {
                attn[t * h + head * hd..t * h + (head + 1) * hd]
                    .copy_from_slice(&out[t * hd..(t + 1) * hd]);
            }
        }
        let out_b = vector(gguf, &name("attn_out.bias"), h)?;
        let proj = linear(
            &attn,
            n,
            &raw_tensor(gguf, &name("attn_out.weight"))?,
            Some(&out_b),
        )?;
        for (xi, pi) in x.iter_mut().zip(&proj) {
            *xi += pi;
        }

        let ln2_w = vector(gguf, &name("ln2.weight"), h)?;
        let ln2_b = vector(gguf, &name("ln2.bias"), h)?;
        let cur = layer_norm(&x, h, &ln2_w, &ln2_b, cfg.eps);
        let up_b = vector(gguf, &name("ffn_up.bias"), cfg.ffn)?;
        let mut up = linear(
            &cur,
            n,
            &raw_tensor(gguf, &name("ffn_up.weight"))?,
            Some(&up_b),
        )?;
        for u in up.iter_mut() {
            *u = gelu_tanh(*u);
        }
        let down_b = vector(gguf, &name("ffn_down.bias"), h)?;
        let down = linear(
            &up,
            n,
            &raw_tensor(gguf, &name("ffn_down.weight"))?,
            Some(&down_b),
        )?;
        for (xi, di) in x.iter_mut().zip(&down) {
            *xi += di;
        }
    }

    // post_ln, then the merger on groups of four consecutive tokens.
    let post_w = vector(gguf, "v.post_ln.weight", h)?;
    let post_b = vector(gguf, "v.post_ln.bias", h)?;
    let normed = layer_norm(&x, h, &post_w, &post_b, cfg.eps);
    let n_merged = n / 4;
    let mm0_b = vector(gguf, "mm.0.bias", cfg.merger_hidden)?;
    let mut mid = linear(
        &normed,
        n_merged,
        &raw_tensor(gguf, "mm.0.weight")?,
        Some(&mm0_b),
    )?;
    for m in mid.iter_mut() {
        *m = gelu_tanh(*m);
    }
    let mm2_b = vector(gguf, "mm.2.bias", cfg.projection_dim)?;
    let rows = linear(
        &mid,
        n_merged,
        &raw_tensor(gguf, "mm.2.weight")?,
        Some(&mm2_b),
    )?;
    Ok(RefOutput {
        rows,
        n_rows: n_merged,
        dim: cfg.projection_dim,
        grid_h: ph / 2,
        grid_w: pw / 2,
    })
}

/// Largest per-row relative L2 error `|a - b| / |b|` between an `f32`
/// result and the `f64` reference (rows of `dim` values).
#[must_use]
pub fn max_row_relative_error(actual: &[f32], reference: &[f64], dim: usize) -> f64 {
    actual
        .chunks(dim.max(1))
        .zip(reference.chunks(dim.max(1)))
        .map(|(a, r)| {
            let diff = a
                .iter()
                .zip(r)
                .map(|(x, y)| (f64::from(*x) - y).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = r.iter().map(|y| y * y).sum::<f64>().sqrt();
            if norm == 0.0 {
                diff
            } else {
                diff / norm
            }
        })
        .fold(0.0, f64::max)
}
