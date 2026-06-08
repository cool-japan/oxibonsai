//! Weight registry for the FLUX.2 `AutoencoderKLFlux2` VAE decoder.
//!
//! Weights come from one of two interchangeable sources, both yielding the
//! same f32 row-major [`Tensor`] for a given dotted name (so the entire
//! downstream decode is source-agnostic):
//!
//! - `Safetensors` — the canonical diffusers
//!   `vae/diffusion_pytorch_model.safetensors` (FLUX.2 `AutoencoderKLFlux2`),
//!   read directly in Pure Rust via [`crate::vae::safetensors::VaeSafetensors`]
//!   (bf16→f32, conv-weight transpose, `to_out.0` un-nesting). This is the
//!   self-serve path — no Python dump step.
//! - `NpyDir` — the per-tensor f32 `.npy` files exported by
//!   `/tmp/bonsai_vae_export_weights.py` (the original dev-time golden dump).
//!
//! [`VaeWeights::open`] **auto-detects** the source from the path: a file ending
//! in `.safetensors` → the safetensors loader; a directory → the `.npy` loader.
//!
//! ## Conv weight layout (important)
//!
//! MLX `nn.Conv2d` stores its weight as **`[out, kH, kW, in]`** (NOT PyTorch
//! `[out, in, kH, kW]`). The exported `.npy` files preserve that layout, the
//! safetensors loader transposes the PyTorch checkpoint into it, and the
//! [`crate::vae::conv`] kernel consumes it directly. Linear weights are
//! `[out, in]` (PyTorch-compatible).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::vae::error::{VaeError, VaeResult};
use crate::vae::safetensors::VaeSafetensors;

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

/// Where a [`VaeWeights`] registry draws its tensors from.
///
/// Both sources resolve a dotted name (e.g. `decoder.conv_in.weight`) to the
/// same f32 row-major [`Tensor`] in the layout the decoder consumes, so the
/// cache and the entire downstream VAE decode are identical regardless of
/// source.
enum Source {
    /// The per-tensor f32 `.npy` dump. Each `<name>.npy` is read with
    /// [`read_npy_f32`].
    NpyDir(PathBuf),
    /// The native FLUX.2 `AutoencoderKLFlux2` safetensors, decoded (bf16→f32) and
    /// re-laid-out on demand to byte-match the `.npy` values.
    Safetensors(Box<VaeSafetensors>),
}

/// A lazily-loaded, cached registry over the VAE decode-path weights.
///
/// The weights come from one of two interchangeable sources, auto-detected
/// by [`Self::open`]; every loaded tensor is cached, so each is read/decoded at
/// most once and the downstream decode is source-agnostic.
pub struct VaeWeights {
    source: Source,
    cache: RefCell<HashMap<String, std::rc::Rc<Tensor>>>,
}

impl VaeWeights {
    /// Open a VAE weights source, **auto-detecting** the format from `path`:
    ///
    /// - a **file ending in `.safetensors`** → the Pure-Rust FLUX.2
    ///   `AutoencoderKLFlux2` safetensors loader
    ///   ([`crate::vae::safetensors::VaeSafetensors`]);
    /// - a **directory** → the per-tensor `.npy` dump (the one containing
    ///   `decoder.*.npy`, `bn.*.npy`, `post_quant_conv.*.npy`).
    ///
    /// # Errors
    /// [`VaeError::Io`] if `path` is neither a `.safetensors` file nor a readable
    /// directory; the safetensors loader's open errors otherwise.
    pub fn open(path: &Path) -> VaeResult<Self> {
        let is_safetensors = path.is_file()
            && path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        let source = if is_safetensors {
            Source::Safetensors(Box::new(VaeSafetensors::open(path)?))
        } else if path.is_dir() {
            Source::NpyDir(path.to_path_buf())
        } else {
            return Err(VaeError::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "VAE weights path is neither a .safetensors file nor a directory",
                ),
            });
        };
        Ok(Self {
            source,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// Fetch a tensor by dotted name (e.g. `decoder.conv_in.weight`), loading and
    /// caching it on first use. Both sources yield the same f32 [`Tensor`] in the
    /// decoder's expected layout.
    ///
    /// # Errors
    /// [`VaeError::MissingWeight`] if the tensor is absent; [`VaeError::Npy`] if
    /// the source file/tensor is malformed.
    pub fn get(&self, name: &str) -> VaeResult<std::rc::Rc<Tensor>> {
        if let Some(t) = self.cache.borrow().get(name) {
            return Ok(t.clone());
        }
        let tensor = match &self.source {
            Source::NpyDir(dir) => {
                let path = dir.join(format!("{name}.npy"));
                if !path.exists() {
                    return Err(VaeError::MissingWeight {
                        name: name.to_string(),
                    });
                }
                read_npy_f32(&path)?
            }
            Source::Safetensors(model) => model.load_tensor(name)?,
        };
        let t = std::rc::Rc::new(tensor);
        self.cache.borrow_mut().insert(name.to_string(), t.clone());
        Ok(t)
    }

    /// Fetch a 1-D tensor's data (e.g. a bias or norm weight), validating rank.
    ///
    /// # Errors
    /// As [`Self::get`], plus [`VaeError::Shape`] if the tensor is not 1-D.
    pub fn vec1(&self, name: &str) -> VaeResult<std::rc::Rc<Tensor>> {
        let t = self.get(name)?;
        if t.shape.len() != 1 {
            return Err(VaeError::Shape(format!(
                "{name}: expected 1-D, got {:?}",
                t.shape
            )));
        }
        Ok(t)
    }
}

/// Minimal NumPy `.npy` reader for `descr=='<f4'`, C-order (or Fortran, which is
/// reordered to C). Returns the f32 data and parsed shape.
///
/// # Errors
/// [`VaeError::Io`] on read failure; [`VaeError::Npy`] on a malformed header or
/// unsupported dtype.
pub fn read_npy_f32(path: &Path) -> VaeResult<Tensor> {
    let bytes = std::fs::read(path).map_err(|e| VaeError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let npy = |reason: String| VaeError::Npy {
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
