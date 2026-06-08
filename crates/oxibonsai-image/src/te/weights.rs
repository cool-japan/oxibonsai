//! Weight registry for the Pure-Rust Qwen3-4B text encoder.
//!
//! Weights are loaded from the per-tensor `.npy` files exported by
//! `/tmp/bonsai_te_export_weights.py` (the dequantised f32 of every 4-bit
//! mlx-packed-affine linear, plus the bf16 RMSNorm vectors), in row-major
//! C-order. Each tensor is read on demand by its dotted name and cached.
//!
//! Naming (mirrors the export script):
//! - `embed_tokens`                        `[vocab, hidden]`
//! - `layers.{i}.self_attn.{q,k,v,o}_proj` quantised → f32 `[out, in]`
//! - `layers.{i}.self_attn.{q,k}_norm`     `[head_dim]`
//! - `layers.{i}.input_layernorm`          `[hidden]`
//! - `layers.{i}.post_attention_layernorm` `[hidden]`
//! - `layers.{i}.mlp.{gate,up,down}_proj`  quantised → f32 `[out, in]`
//! - `norm`                                `[hidden]` (final RMSNorm, unused by
//!   the stacked cond but loaded for completeness / the `te_hidden` parity of
//!   the final-norm output)
//!
//! A `Linear[out, in]` weight is row-major `w[n * in + k]` so it feeds the
//! shared [`crate::gemm::gemm_abt`] (`out[m,n] = Σ_k in[m,k] · w[n,k]`) directly,
//! computing `x · Wᵀ` — the same contraction the DiT linears use.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::te::config::TeConfig;
use crate::te::error::{TeError, TeResult};
use crate::te::mlx4bit::Mlx4bitModel;

/// A dense f32 tensor with an explicit shape (C-order / row-major).
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flat row-major data.
    pub data: Vec<f32>,
    /// Logical shape.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Where a [`TeWeights`] registry draws its tensors from.
///
/// Both sources resolve a dotted name (e.g. `layers.0.self_attn.q_proj`) to the
/// same f32 row-major [`Tensor`], so the cache and the entire downstream TE
/// forward are identical regardless of source.
enum Source {
    /// The per-tensor f32 `.npy` dump (the original ~15 GB export). Each
    /// `<name>.npy` is read with [`read_npy_f32`]. Byte-unchanged behaviour.
    NpyDir(PathBuf),
    /// The native ~2.1 GB MLX 4-bit safetensors, dequantised on demand.
    Mlx4bit(Box<Mlx4bitModel>),
}

/// A lazily-loaded, cached registry over the Qwen3 text-encoder weights, plus
/// the parsed [`TeConfig`].
///
/// The weights come from one of two interchangeable [`Source`]s — the f32
/// `.npy` dump ([`Self::open`]) or the 4-bit MLX safetensors
/// ([`Self::open_mlx_4bit`]) — both yielding identical f32 tensors by dotted
/// name. Every loaded tensor is cached, so dequant happens at most once per
/// tensor and the downstream forward is source-agnostic.
pub struct TeWeights {
    source: Source,
    config: TeConfig,
    cache: RefCell<HashMap<String, Rc<Tensor>>>,
    /// When set, the [`Source::Mlx4bit`] path also caches its dequantised f32
    /// tensors (like the `.npy` path always does), trading RAM for speed so a
    /// long-lived registry pays the dequant cost once instead of once per
    /// forward. Default off: the one-shot CLI keeps its ~2.5 GB low-RAM
    /// profile; [`crate::session::ImageSession`] flips it on to keep the
    /// ~16 GB of f32 encoder weights resident across REPL prompts.
    resident: Cell<bool>,
}

impl TeWeights {
    /// Open a text-encoder weights directory (the one containing
    /// `embed_tokens.npy`, `layers.*.npy`, `norm.npy`).
    ///
    /// The [`TeConfig`] defaults to the Qwen3-4B architecture; if a
    /// `weights_manifest.json` is present its scalar fields are honoured.
    ///
    /// # Errors
    /// [`TeError::Io`] if the directory cannot be inspected.
    pub fn open(dir: &Path) -> TeResult<Self> {
        if !dir.is_dir() {
            return Err(TeError::Io {
                path: dir.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "weights directory not found",
                ),
            });
        }
        let config = TeConfig::from_manifest_dir(dir).unwrap_or_default();
        Ok(Self {
            source: Source::NpyDir(dir.to_path_buf()),
            config,
            cache: RefCell::new(HashMap::new()),
            resident: Cell::new(false),
        })
    }

    /// Open the native MLX **4-bit safetensors** text encoder (the ~2.1 GB
    /// `text_encoder-mlx-4bit/model.safetensors`).
    ///
    /// Tensors are dequantised on first [`Self::get`] and cached, so the
    /// downstream forward is byte-for-byte the f32 numerics of the `.npy` path
    /// — only the on-disk footprint shrinks (15 GB → 2.1 GB). The [`TeConfig`]
    /// defaults to Qwen3-4B (matching the export constants).
    ///
    /// # Errors
    /// [`TeError::Io`] if the file cannot be opened/mapped; [`TeError::Npy`] if
    /// it is not a valid safetensors container.
    pub fn open_mlx_4bit(safetensors_path: &Path) -> TeResult<Self> {
        let model = Mlx4bitModel::open(safetensors_path)?;
        Ok(Self {
            source: Source::Mlx4bit(Box::new(model)),
            config: TeConfig::default(),
            cache: RefCell::new(HashMap::new()),
            resident: Cell::new(false),
        })
    }

    /// The parsed (or default) text-encoder configuration.
    pub fn config(&self) -> &TeConfig {
        &self.config
    }

    /// Keep dequantised f32 weights resident across forwards.
    ///
    /// Off by default. The `.npy` source already caches every tensor; this only
    /// changes the [`Source::Mlx4bit`] path, which otherwise re-dequantises each
    /// weight on every [`Self::get`] (the deliberate low-RAM policy). Turning it
    /// on holds the full f32 encoder (~16 GB) resident so repeated forwards skip
    /// the dequant. Intended for a long-lived [`crate::session::ImageSession`] on
    /// a high-memory machine, not the one-shot CLI.
    pub fn set_resident(&self, on: bool) {
        self.resident.set(on);
        if !on {
            self.cache.borrow_mut().clear();
        }
    }

    /// Fetch a tensor by dotted name.
    ///
    /// Dispatches on the registry's [`Source`]:
    /// - [`Source::NpyDir`] reads `<name>.npy` and **caches** the parsed f32
    ///   tensor (byte-unchanged behaviour: the f32 path keeps every loaded
    ///   tensor resident, so a tensor is read at most once).
    /// - [`Source::Mlx4bit`] dequantises `model.{name}` on the fly and
    ///   deliberately **does not cache** the result. Each returned `Rc<Tensor>`
    ///   owns a transient f32 buffer that is freed as soon as the caller drops
    ///   it — so over a forward, layer weights are released as the layer loop
    ///   advances instead of accumulating to ~16 GB of resident f32. (The
    ///   per-layer re-dequant of the tiny RMSNorm vectors is negligible.)
    ///
    /// Both sources still yield the same f32 row-major [`Tensor`] for a given
    /// name, so the downstream forward is source-agnostic.
    ///
    /// Caveat (GPU TE path): [`crate::te::cuda_gpu::te_matmul_gpu`] keys the
    /// resident device-weight cache by `weight.as_ptr()`. Under this Mlx4bit
    /// no-cache policy the dequantised buffer is freed as soon as the caller drops
    /// its `Rc`, and the allocator **recycles that address** for the next Linear —
    /// so the pointer is NOT a stable per-weight identity, and a naive
    /// get-or-upload returns a *stale* buffer on the inevitable collision (wrong
    /// weights → corrupted conditioning). The GPU path therefore evicts the key
    /// immediately after each GEMM (so every matmul re-uploads fresh); this keeps
    /// the ~16 GB → ~2.5 GB RAM win of the no-cache policy intact while staying
    /// correct. The TE GPU path is opt-in (`OXI_TE_GPU=1`).
    ///
    /// # Errors
    /// [`TeError::MissingWeight`] if the tensor is absent; [`TeError::Npy`] if
    /// the source file/tensor is malformed.
    pub fn get(&self, name: &str) -> TeResult<Rc<Tensor>> {
        match &self.source {
            Source::NpyDir(dir) => {
                if let Some(t) = self.cache.borrow().get(name) {
                    return Ok(t.clone());
                }
                let path = dir.join(format!("{name}.npy"));
                if !path.exists() {
                    return Err(TeError::MissingWeight {
                        name: name.to_string(),
                    });
                }
                let t = Rc::new(read_npy_f32(&path)?);
                self.cache.borrow_mut().insert(name.to_string(), t.clone());
                Ok(t)
            }
            // Default: no-cache, the dequantised f32 is transient (the core RAM
            // win). When `resident` is set, cache like the `.npy` path so a
            // long-lived registry dequantises each tensor at most once.
            Source::Mlx4bit(model) => {
                if self.resident.get() {
                    if let Some(t) = self.cache.borrow().get(name) {
                        return Ok(t.clone());
                    }
                    let t = Rc::new(model.load_tensor(name)?);
                    self.cache.borrow_mut().insert(name.to_string(), t.clone());
                    Ok(t)
                } else {
                    Ok(Rc::new(model.load_tensor(name)?))
                }
            }
        }
    }

    /// Fetch a 2-D linear weight, validating its `(out, in)` dimensions.
    ///
    /// # Errors
    /// As [`Self::get`], plus [`TeError::Shape`] if the rank or dims disagree.
    pub fn linear(&self, name: &str, out: usize, in_: usize) -> TeResult<Rc<Tensor>> {
        let t = self.get(name)?;
        if t.shape != [out, in_] {
            return Err(TeError::Shape(format!(
                "{name}: expected [{out}, {in_}], got {:?}",
                t.shape
            )));
        }
        Ok(t)
    }

    /// Fetch a 1-D weight (an RMSNorm vector), validating its length.
    ///
    /// # Errors
    /// As [`Self::get`], plus [`TeError::Shape`] if the rank or length disagree.
    pub fn vec1(&self, name: &str, len: usize) -> TeResult<Rc<Tensor>> {
        let t = self.get(name)?;
        if t.shape != [len] {
            return Err(TeError::Shape(format!(
                "{name}: expected [{len}], got {:?}",
                t.shape
            )));
        }
        Ok(t)
    }

    /// Gather the embedding rows for `ids` from the `[vocab, cols]` table `name`
    /// into a flat row-major `[ids.len() * cols]` f32 buffer (one row per id, in
    /// `ids` order).
    ///
    /// This is the RAM-frugal embedding lookup. For [`Source::Mlx4bit`] it
    /// dequantises **only** the requested rows via
    /// [`Mlx4bitModel::gather_quant_rows`] — avoiding the 1.5 GB f32 spike of
    /// dequantising the full `[151936, 2560]` table just to keep `seq` rows —
    /// and the per-row values are byte-identical to a full-table dequant. For
    /// [`Source::NpyDir`] it falls back to the current behaviour
    /// (`self.get(name)?` full load, then gather the rows), so the f32 path is
    /// byte-unchanged.
    ///
    /// Each id is validated `< cols`-rows of the table (`id < vocab`), returning
    /// the same [`TeError::Shape`] the forward used to raise inline.
    ///
    /// # Errors
    /// [`TeError::Shape`] if any `id >= vocab`, plus the errors of [`Self::get`]
    /// / [`Mlx4bitModel::gather_quant_rows`].
    pub fn embed_gather(&self, name: &str, ids: &[u32], cols: usize) -> TeResult<Vec<f32>> {
        match &self.source {
            Source::Mlx4bit(model) => {
                // Validate ids against the configured vocab before touching the
                // file (mirrors the forward's old inline bounds check).
                let vocab = self.config.vocab_size;
                let mut rows = Vec::with_capacity(ids.len());
                for &id in ids {
                    let row = id as usize;
                    if row >= vocab {
                        return Err(TeError::Shape(format!(
                            "token id {row} >= vocab_size {vocab}"
                        )));
                    }
                    rows.push(row);
                }
                model.gather_quant_rows(name, &rows, cols)
            }
            // f32 path: full-load + gather (byte-unchanged from the old forward).
            Source::NpyDir(_) => {
                let table = self.get(name)?;
                let vocab = self.config.vocab_size;
                if table.shape != [vocab, cols] {
                    return Err(TeError::Shape(format!(
                        "{name}: expected [{vocab}, {cols}], got {:?}",
                        table.shape
                    )));
                }
                let mut out = vec![0.0f32; ids.len() * cols];
                for (t, &id) in ids.iter().enumerate() {
                    let row = id as usize;
                    if row >= vocab {
                        return Err(TeError::Shape(format!(
                            "token id {row} >= vocab_size {vocab}"
                        )));
                    }
                    out[t * cols..(t + 1) * cols]
                        .copy_from_slice(&table.data[row * cols..(row + 1) * cols]);
                }
                Ok(out)
            }
        }
    }
}

/// Minimal NumPy `.npy` reader for `descr=='<f4'`, C-order (or Fortran, which is
/// reordered to C). Returns the f32 data and parsed shape. (Mirrors the reader
/// in [`crate::vae::weights`].)
///
/// # Errors
/// [`TeError::Io`] on read failure; [`TeError::Npy`] on a malformed header or
/// unsupported dtype.
pub fn read_npy_f32(path: &Path) -> TeResult<Tensor> {
    let bytes = std::fs::read(path).map_err(|e| TeError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let npy = |reason: String| TeError::Npy {
        path: path.display().to_string(),
        reason,
    };
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(npy("bad npy magic".to_string()));
    }
    let major = bytes[6];
    let (header_start, header_len) = if major >= 2 {
        if bytes.len() < 12 {
            return Err(npy("truncated v2 header length".to_string()));
        }
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    } else {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    };
    if header_start + header_len > bytes.len() {
        return Err(npy("header extends past file".to_string()));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| npy(format!("header utf8: {e}")))?;
    if !header.contains("'<f4'") {
        return Err(npy(format!("descr is not '<f4': {header}")));
    }
    let fortran = header.contains("'fortran_order': True");
    let s_idx = header
        .find("'shape':")
        .ok_or_else(|| npy("no shape key".to_string()))?;
    let open = header[s_idx..]
        .find('(')
        .map(|o| s_idx + o + 1)
        .ok_or_else(|| npy("no shape open paren".to_string()))?;
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .ok_or_else(|| npy("no shape close paren".to_string()))?;
    let shape: Vec<usize> = header[open..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| npy(format!("shape parse: {e}")))
        })
        .collect::<Result<_, _>>()?;
    let data_start = header_start + header_len;
    let payload = &bytes[data_start..];
    if payload.len() % 4 != 0 {
        return Err(npy("payload not f32-aligned".to_string()));
    }
    let numel: usize = shape.iter().product();
    if payload.len() / 4 < numel {
        return Err(npy(format!(
            "payload short ({} < {})",
            payload.len() / 4,
            numel
        )));
    }
    let raw: Vec<f32> = payload[..numel * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&raw, &shape)
    } else {
        raw
    };
    Ok(Tensor { data, shape })
}

/// Reorder a Fortran-stored (column-major) buffer into C (row-major) order.
fn fortran_to_c(src: &[f32], shape: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let mut f_stride = vec![1usize; ndim];
    for d in 1..ndim {
        f_stride[d] = f_stride[d - 1] * shape[d - 1];
    }
    let mut out = vec![0.0f32; numel];
    for (c_pos, slot) in out.iter_mut().enumerate() {
        let mut rem = c_pos;
        let mut f_off = 0usize;
        for d in 0..ndim {
            let stride_c: usize = shape[d + 1..].iter().product();
            let idx = rem / stride_c;
            rem %= stride_c;
            f_off += idx * f_stride[d];
        }
        *slot = src[f_off];
    }
    out
}
