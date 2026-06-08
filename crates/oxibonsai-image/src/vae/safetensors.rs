//! Pure-Rust loader for the **FLUX.2 `AutoencoderKLFlux2` VAE** weights, read
//! directly from the canonical diffusers `*.safetensors` file — the
//! `vae/diffusion_pytorch_model.safetensors` of
//! [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev)
//! (subfolder `vae/`), `_class_name = "AutoencoderKLFlux2"`.
//!
//! This is the self-serve replacement for the dev-time Python `.npy` golden
//! dump (`/tmp/bonsai_vae_export_weights.py`): it resolves every dotted weight
//! name the decoder requests (the [`super::weights`] key contract) to the
//! matching safetensors tensor, applies the **identical** layout/name transforms
//! the Python export applied, and decodes the on-disk dtype to the f32 the
//! decoder consumes — yielding values byte-identical to the `.npy` path.
//!
//! # Name + layout mapping (authoritative; mirrors mflux `flux2_weight_mapping`)
//!
//! The diffusers checkpoint is plain PyTorch (`__metadata__.format == "pt"`),
//! so the only differences from the decoder's `.npy` key contract are:
//!
//! 1. **Conv weights are transposed.** PyTorch `nn.Conv2d` stores
//!    `[out, in, kH, kW]`; the decoder's `Conv2d` (and the MLX
//!    `.npy` dump) want `[out, kH, kW, in]`. So every 4-D `*.weight` tensor is
//!    permuted with axes `(0, 2, 3, 1)` — exactly mflux's
//!    `WeightTransforms.transpose_conv2d_weight` (`tensor.transpose(0, 2, 3, 1)`).
//!    This covers `conv_in`, `conv_out`, `post_quant_conv` (1×1), every resnet
//!    `conv1`/`conv2`, the 1×1 `conv_shortcut`s, and the `upsamplers.*.conv`.
//!
//! 2. **`to_out` is un-nested.** PyTorch wraps the attention output projection in
//!    a `ModuleList`, so the on-disk name is `…attentions.0.to_out.0.weight`
//!    (and `.0.bias`); the decoder asks for `…attentions.0.to_out.weight`. The
//!    mapping inserts the `.0.` segment when resolving such a key. The attention
//!    linears (`to_q`/`to_k`/`to_v`/`to_out`) are 2-D `[out, in]` and pass
//!    through **without** transpose (matching the `.npy` Linear layout).
//!
//! 3. **All other tensors pass through by name.** `bn.running_mean` /
//!    `bn.running_var` (1-D), the GroupNorm `weight`/`bias` (1-D), and every conv
//!    `bias` (1-D) keep their name and shape.
//!
//! Encoder-only tensors (`encoder.*`, `quant_conv.*`) and the unused
//! `bn.num_batches_tracked` (`I64`) are simply never requested by the decoder,
//! so they are ignored.
//!
//! # Dtype
//!
//! The FLUX.2 VAE checkpoint stores every float tensor as **BF16**. Decoding
//! bf16→f32 is the exact left-shift `(bits as u32) << 16` (bf16 is the high 16
//! bits of an IEEE-754 f32), so it is lossless — the produced f32 values match a
//! Python `tensor.float().numpy()` export bit-for-bit. `F32` tensors (should the
//! source ever store them) are read directly.
//!
//! # Ownership model
//!
//! [`crate::vae::safetensors::VaeSafetensors`] owns the [`memmap2::Mmap`] for the file's lifetime and
//! re-runs the (cheap, header-only) `SafeTensors::deserialize` on each
//! `load_tensor` access — the same pattern (and rationale) as
//! [`crate::te::mlx4bit::Mlx4bitModel`]. The VAE decoder loads each of its ~142
//! tensors once and [`crate::vae::weights::VaeWeights`] caches the result, so the
//! per-access header re-parse is negligible and the mmap is never copied.

use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::vae::error::{VaeError, VaeResult};
use crate::vae::weights::Tensor;

/// The infix PyTorch inserts for the `ModuleList`-wrapped attention output
/// projection: the decoder's `…to_out.weight` is `…to_out.0.weight` on disk.
const TO_OUT: &str = ".to_out.";
/// Replacement infix that re-nests `to_out` into its `ModuleList` slot 0.
const TO_OUT_NESTED: &str = ".to_out.0.";

/// Reinterpret a bfloat16 bit pattern as `f32`.
///
/// bfloat16 is the top 16 bits of an IEEE-754 `f32`, so the widening is an exact
/// left-shift by 16 with no rounding (lossless decode).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// A memory-mapped FLUX.2 `AutoencoderKLFlux2` VAE safetensors file.
///
/// Owns the [`Mmap`] and re-parses the safetensors header per access (see the
/// module docs). `load_tensor` takes the decoder's dotted `.npy`-style
/// key and returns the f32 [`Tensor`] in exactly the layout the decoder expects.
pub struct VaeSafetensors {
    path: PathBuf,
    mmap: Mmap,
}

impl VaeSafetensors {
    /// Open and memory-map a VAE `*.safetensors` file, validating that it parses
    /// as a safetensors container.
    ///
    /// # Errors
    /// [`VaeError::Io`] if the file cannot be opened or mapped; [`VaeError::Npy`]
    /// (reused as the container-parse error variant) if the safetensors header
    /// is malformed.
    pub fn open(path: &Path) -> VaeResult<Self> {
        let file = std::fs::File::open(path).map_err(|source| VaeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: read-only mapping of a file we do not mutate while mapped —
        // the standard safetensors model-loading pattern (mirrors the TE 4-bit
        // loader's `Mlx4bitModel::open`).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| VaeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        // Validate the header parses up front so a corrupt file fails at open.
        SafeTensors::deserialize(&mmap).map_err(|e| VaeError::Npy {
            path: path.display().to_string(),
            reason: format!("safetensors header: {e}"),
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
        })
    }

    /// Re-parse the safetensors header over the owned mmap.
    fn view(&self) -> VaeResult<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.mmap).map_err(|e| VaeError::Npy {
            path: self.path.display().to_string(),
            reason: format!("safetensors header: {e}"),
        })
    }

    /// Resolve a decoder `.npy`-style dotted key to its on-disk safetensors name.
    ///
    /// The only rewrite is the attention output projection: a `…to_out.weight` /
    /// `…to_out.bias` key maps to the `ModuleList`-nested `…to_out.0.weight` /
    /// `…to_out.0.bias`. Everything else is identity. (A name that already
    /// contains `.to_out.0.` is left untouched.)
    fn on_disk_name(key: &str) -> String {
        if key.contains(TO_OUT) && !key.contains(TO_OUT_NESTED) {
            key.replacen(TO_OUT, TO_OUT_NESTED, 1)
        } else {
            key.to_string()
        }
    }

    /// Load a tensor by the decoder's dotted `.npy`-style key (e.g.
    /// `decoder.conv_in.weight`), returning it as an f32 [`Tensor`] in the exact
    /// layout the decoder consumes:
    ///
    /// - a 4-D conv `*.weight` is decoded then permuted `[O,I,kH,kW] → [O,kH,kW,I]`;
    /// - every other tensor (1-D bn/norm/bias, 2-D attention linear) is decoded
    ///   in place with its on-disk shape.
    ///
    /// bf16 is decoded losslessly to f32; f32 is read directly.
    ///
    /// # Errors
    /// [`VaeError::MissingWeight`] if the (mapped) name is absent;
    /// [`VaeError::Npy`] on a header or unsupported-dtype problem;
    /// [`VaeError::Shape`] on a buffer-length / rank mismatch.
    pub fn load_tensor(&self, key: &str) -> VaeResult<Tensor> {
        let st = self.view()?;
        let name = Self::on_disk_name(key);
        let view = match st.tensor(&name) {
            Ok(v) => v,
            Err(_) => {
                return Err(VaeError::MissingWeight {
                    name: key.to_string(),
                })
            }
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let numel: usize = shape.iter().product();
        let raw = view.data();

        // Decode the on-disk dtype to a flat row-major f32 buffer.
        let flat = match view.dtype() {
            Dtype::BF16 => {
                if raw.len() != numel * 2 {
                    return Err(VaeError::Shape(format!(
                        "{key}: bf16 byte len {} != 2*{numel}",
                        raw.len()
                    )));
                }
                raw.chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect::<Vec<f32>>()
            }
            Dtype::F32 => {
                if raw.len() != numel * 4 {
                    return Err(VaeError::Shape(format!(
                        "{key}: f32 byte len {} != 4*{numel}",
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>()
            }
            other => {
                return Err(VaeError::Npy {
                    path: self.path.display().to_string(),
                    reason: format!("{key}: unsupported dtype {other:?} (expected BF16 or F32)"),
                })
            }
        };

        // 4-D conv weight: PyTorch `[O,I,kH,kW]` → decoder/MLX `[O,kH,kW,I]`.
        if shape.len() == 4 {
            let (out_c, in_c, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
            let data = transpose_oikk_to_okki(&flat, out_c, in_c, kh, kw);
            return Ok(Tensor {
                data,
                shape: vec![out_c, kh, kw, in_c],
            });
        }

        // 1-D (bn/norm/bias) and 2-D (attention linear) pass through unchanged.
        Ok(Tensor { data: flat, shape })
    }
}

/// Permute a flat row-major conv weight from PyTorch `[O, I, kH, kW]` to the
/// MLX/decoder layout `[O, kH, kW, I]` (axes `(0, 2, 3, 1)`).
///
/// `src` is `O*I*kH*kW` long, indexed `((o*I + i)*kH + a)*kW + b`; the output is
/// indexed `((o*kH + a)*kW + b)*I + i`.
fn transpose_oikk_to_okki(
    src: &[f32],
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0f32; out_c * kh * kw * in_c];
    for o in 0..out_c {
        for i in 0..in_c {
            for a in 0..kh {
                for b in 0..kw {
                    let s = ((o * in_c + i) * kh + a) * kw + b;
                    let d = ((o * kh + a) * kw + b) * in_c + i;
                    dst[d] = src[s];
                }
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_map_rewrites_to_out_only() {
        assert_eq!(
            VaeSafetensors::on_disk_name("decoder.mid_block.attentions.0.to_out.weight"),
            "decoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(
            VaeSafetensors::on_disk_name("decoder.mid_block.attentions.0.to_out.bias"),
            "decoder.mid_block.attentions.0.to_out.0.bias"
        );
        // Already-nested name is untouched (idempotent).
        assert_eq!(
            VaeSafetensors::on_disk_name("decoder.mid_block.attentions.0.to_out.0.weight"),
            "decoder.mid_block.attentions.0.to_out.0.weight"
        );
        // Non-`to_out` keys are identity.
        for k in [
            "decoder.conv_in.weight",
            "decoder.mid_block.attentions.0.to_q.weight",
            "bn.running_mean",
            "post_quant_conv.bias",
        ] {
            assert_eq!(VaeSafetensors::on_disk_name(k), k);
        }
    }

    #[test]
    fn bf16_decode_is_exact_left_shift() {
        // Values whose f32 bit-patterns have a zero low-16 mantissa survive a
        // bf16 round-trip exactly; decode must reproduce them.
        for &v in &[1.5f32, -2.25, 0.0, -0.0, 65536.0, 0.5] {
            let bits = (v.to_bits() >> 16) as u16;
            assert_eq!(bf16_to_f32(bits), f32::from_bits((v.to_bits() >> 16) << 16));
        }
        // Explicit bit check: bf16 0x3FC0 == 1.5.
        assert_eq!(bf16_to_f32(0x3FC0), 1.5);
    }

    #[test]
    fn conv_transpose_matches_manual_index() {
        // O=2, I=3, kH=2, kW=2; fill with a unique value per (o,i,a,b).
        let (o_c, i_c, kh, kw) = (2usize, 3usize, 2usize, 2usize);
        let mut src = vec![0.0f32; o_c * i_c * kh * kw];
        for o in 0..o_c {
            for i in 0..i_c {
                for a in 0..kh {
                    for b in 0..kw {
                        let s = ((o * i_c + i) * kh + a) * kw + b;
                        src[s] = (s + 1) as f32;
                    }
                }
            }
        }
        let dst = transpose_oikk_to_okki(&src, o_c, i_c, kh, kw);
        assert_eq!(dst.len(), o_c * kh * kw * i_c);
        // Spot-check the permutation: dst[(o,a,b,i)] == src[(o,i,a,b)].
        for o in 0..o_c {
            for i in 0..i_c {
                for a in 0..kh {
                    for b in 0..kw {
                        let s = ((o * i_c + i) * kh + a) * kw + b;
                        let d = ((o * kh + a) * kw + b) * i_c + i;
                        assert_eq!(dst[d], src[s], "mismatch at o{o} i{i} a{a} b{b}");
                    }
                }
            }
        }
    }
}
