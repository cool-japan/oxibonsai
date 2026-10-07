//! Read-side helpers for the ONNX → GGUF pipeline.
//!
//! Responsible for:
//!
//! * Parsing an `.onnx` protobuf file into [`ModelProto`] via `oxionnx-proto`.
//! * Memory-mapping the optional external-data sidecar (`.onnx_data`) so
//!   large weight tensors can be read zero-copy.
//! * Reading the raw bytes of a single initializer (inline or external).
//! * Converting raw bytes of supported element dtypes (`f32`, `f16`, `bf16`)
//!   into `Vec<f32>`.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use oxionnx_proto::parser::parse_model;
use oxionnx_proto::types::{ModelProto, TensorProto};

use super::error::OnnxImportError;

/// ONNX element-type codes we understand when converting an initializer to f32.
mod dtype_code {
    pub const FLOAT32: i32 = 1;
    pub const FLOAT16: i32 = 10;
    pub const BFLOAT16: i32 = 16;
}

/// An open ONNX model together with its optional memory-mapped sidecar.
pub struct OnnxReader {
    /// Path to the `.onnx` file (retained for error messages).
    pub onnx_path: PathBuf,
    /// Directory that holds the `.onnx` file and any sidecar files.
    pub base_dir: PathBuf,
    /// Parsed protobuf structure.
    pub model: ModelProto,
    /// Lazily-populated memory-maps, keyed by sidecar relative path.
    sidecars: Vec<(PathBuf, Mmap)>,
}

impl OnnxReader {
    /// Open `onnx_path` and parse the protobuf. Does not eagerly open any
    /// sidecar — sidecars are memory-mapped on first use.
    pub fn open(onnx_path: &Path) -> Result<Self, OnnxImportError> {
        let bytes = std::fs::read(onnx_path).map_err(|e| OnnxImportError::Io {
            path: onnx_path.to_path_buf(),
            source: e,
        })?;
        let model = parse_model(&bytes).map_err(|msg| OnnxImportError::Parse {
            path: onnx_path.to_path_buf(),
            msg,
        })?;
        let base_dir = onnx_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            onnx_path: onnx_path.to_path_buf(),
            base_dir,
            model,
            sidecars: Vec::new(),
        })
    }

    /// Locate an initializer by exact name.
    pub fn find_initializer(&self, name: &str) -> Option<&TensorProto> {
        self.model
            .graph
            .initializers
            .iter()
            .find(|t| t.name == name)
    }

    /// Return the raw bytes for an initializer, resolving external data as
    /// needed. The returned slice has a lifetime tied to `self` (either the
    /// initializer's `raw_data` or the memory-map of a sidecar).
    pub fn initializer_bytes<'a>(
        &'a mut self,
        tensor: &'a TensorProto,
    ) -> Result<&'a [u8], OnnxImportError> {
        if !is_external(tensor) {
            return Ok(tensor.raw_data.as_slice());
        }

        // Fetch "location", "offset", "length" from external_data entries.
        let location = external_entry(tensor, "location").ok_or_else(|| {
            OnnxImportError::MissingExternalEntry {
                tensor: tensor.name.clone(),
                key: "location",
            }
        })?;
        let offset: usize = external_entry(tensor, "offset")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|u| u as usize)
            .unwrap_or(0);
        let length: usize = external_entry(tensor, "length")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|u| u as usize)
            .unwrap_or(0);
        if length == 0 {
            // Fall back to a dtype-aware product of dims (only meaningful for
            // byte-sized dtypes like uint8/int8 used by MatMulNBits weights).
            return Err(OnnxImportError::MissingExternalEntry {
                tensor: tensor.name.clone(),
                key: "length",
            });
        }

        // Ensure the sidecar is memory-mapped.
        let sidecar_path = self.base_dir.join(location);
        self.ensure_sidecar_mapped(&sidecar_path)?;

        let mmap = self
            .sidecars
            .iter()
            .find(|(p, _)| p == &sidecar_path)
            .map(|(_, m)| m)
            .ok_or_else(|| {
                OnnxImportError::Other(format!(
                    "internal: sidecar {} was not mapped after ensure_sidecar_mapped",
                    sidecar_path.display()
                ))
            })?;

        let end = offset.checked_add(length).ok_or_else(|| {
            OnnxImportError::Other(format!(
                "offset {offset} + length {length} overflows usize for tensor '{}'",
                tensor.name
            ))
        })?;
        if end > mmap.len() {
            return Err(OnnxImportError::Other(format!(
                "external-data range {offset}..{end} exceeds sidecar size {} for tensor '{}'",
                mmap.len(),
                tensor.name
            )));
        }

        Ok(&mmap[offset..end])
    }

    /// Memory-map `path` if it is not already cached.
    fn ensure_sidecar_mapped(&mut self, path: &Path) -> Result<(), OnnxImportError> {
        if self.sidecars.iter().any(|(p, _)| p == path) {
            return Ok(());
        }
        let file = File::open(path).map_err(|e| OnnxImportError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        // SAFETY: the mapped file is read-only; `Mmap` is Send+Sync and the
        // underlying storage lives for the duration of the reader.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| OnnxImportError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        self.sidecars.push((path.to_path_buf(), mmap));
        Ok(())
    }
}

/// Return true if the tensor is stored externally (in a sidecar file).
pub fn is_external(tensor: &TensorProto) -> bool {
    tensor.data_location == 1 || !tensor.external_data.is_empty()
}

/// Look up a single key in the `external_data` key-value list.
pub fn external_entry<'a>(tensor: &'a TensorProto, key: &str) -> Option<&'a str> {
    tensor
        .external_data
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Convert raw little-endian bytes of a supported dtype into a `Vec<f32>`.
///
/// Returns [`OnnxImportError::UnsupportedDtype`] if `data_type` is anything
/// other than float32, float16, or bfloat16.
pub fn bytes_to_f32(
    bytes: &[u8],
    data_type: i32,
    tensor_name: &str,
) -> Result<Vec<f32>, OnnxImportError> {
    match data_type {
        dtype_code::FLOAT32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        dtype_code::FLOAT16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f32())
            .collect()),
        dtype_code::BFLOAT16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect()),
        other => Err(OnnxImportError::UnsupportedDtype {
            tensor: tensor_name.to_string(),
            dtype: other,
        }),
    }
}

/// Read an integer attribute from the `ints`/`i` fields of an ONNX attribute.
pub fn attr_int(attrs: &[oxionnx_proto::types::AttributeProto], name: &'static str) -> Option<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| match a.value.attr_type {
            2 => Some(a.value.i),
            _ => None,
        })
}

/// Locate `config.json` beside an ONNX file by walking up two ancestor
/// directories. The HF ONNX export layout is
/// `<model>/onnx/model_q2.onnx` with `<model>/config.json`, so we probe the
/// ONNX parent and grandparent before giving up.
pub fn locate_config_json(onnx_path: &Path) -> Result<PathBuf, OnnxImportError> {
    let mut dir = onnx_path.parent().map(Path::to_path_buf);
    for _ in 0..3 {
        if let Some(d) = dir.as_ref() {
            let candidate = d.join("config.json");
            if candidate.exists() {
                return Ok(candidate);
            }
            dir = d.parent().map(Path::to_path_buf);
        } else {
            break;
        }
    }
    Err(OnnxImportError::ConfigJsonMissing {
        onnx_path: onnx_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_f32_f32_roundtrip() {
        let values: [f32; 3] = [1.0, -2.5, 0.125];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, dtype_code::FLOAT32, "t").expect("ok");
        assert_eq!(out, values);
    }

    #[test]
    fn bytes_to_f32_f16_roundtrip() {
        let values = [half::f16::from_f32(1.0), half::f16::from_f32(-0.5)];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, dtype_code::FLOAT16, "t").expect("ok");
        assert_eq!(out.len(), values.len());
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - -0.5).abs() < 1e-4);
    }

    #[test]
    fn bytes_to_f32_unsupported_dtype_errors() {
        let err = bytes_to_f32(&[0, 0, 0, 0], 7 /* int64 */, "t").unwrap_err();
        match err {
            OnnxImportError::UnsupportedDtype { tensor, dtype } => {
                assert_eq!(tensor, "t");
                assert_eq!(dtype, 7);
            }
            _ => panic!("expected UnsupportedDtype, got {:?}", err),
        }
    }
}
