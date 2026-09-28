//! Where a model-loading command's GGUF bytes come from.
//!
//! The default is a read-only memory map of the file. With
//! `--ptq1-transcode` (design §2.1: "native PTQ1_0 GEMV is the default; the
//! transcode path is opt-in via `--ptq1-transcode`") every `PTQ1_0` tensor
//! (ggml id 143, 1.75 bits/weight) is instead re-encoded losslessly to
//! `PQ2_0` (id 142, 2.125 bits/weight — the layout the existing 2-bit
//! ternary kernels consume) in an in-memory GGUF image:
//!
//! * the header and the whole metadata section are copied **verbatim**
//!   (vocabulary, chat template, `prism.hadamard.*` — nothing is
//!   re-encoded, so nothing can drift);
//! * the tensor-info table is rewritten with the new type ids and offsets;
//! * each `PTQ1_0` tensor's blocks go through
//!   `BlockPTQ1_0::transcode_to_pq2` (trit codes transfer unchanged, `d` is
//!   copied bit for bit), every other tensor's bytes are copied as-is.
//!
//! The image is parsed back with the ordinary `GgufFile::parse`, so every
//! downstream consumer (the context guard, the variant classifier, the
//! hybrid loader) sees a genuine PQ2_0 file. The source memory map is
//! dropped once the image is built: the design's own trade-off — ~7.2 GB
//! of anonymous RAM for the 27B instead of a 5.9 GB page-cache mapping.

use oxibonsai_core::gguf::header::GgufHeader;
use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::GgufTensorType;

/// The alignment `GgufFile::parse` assumes when `general.alignment` is absent.
const DEFAULT_GGUF_ALIGNMENT: u64 = 32;

enum ModelBytes {
    /// A read-only memory map of the file.
    Mapped(Box<dyn std::ops::Deref<Target = [u8]> + Send + Sync>),
    /// An in-memory (transcoded) GGUF image.
    Owned(Vec<u8>),
}

/// A command's model bytes plus what they cost in RAM.
pub(crate) struct ModelSource {
    bytes: ModelBytes,
    transcoded_tensors: usize,
}

impl ModelSource {
    /// Open `path`: memory-map it, and — with `ptq1_transcode` — replace it
    /// with a transcoded in-memory image when it holds any `PTQ1_0` tensor
    /// (a file with none is used as-is, with a log line saying so).
    ///
    /// # Errors
    ///
    /// The file cannot be opened/parsed, or the transcode fails.
    pub(crate) fn open(path: &str, ptq1_transcode: bool) -> anyhow::Result<Self> {
        let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(path))
            .map_err(|e| anyhow::anyhow!("failed to open model '{path}': {e}"))?;
        if !ptq1_transcode {
            return Ok(Self {
                bytes: ModelBytes::Mapped(Box::new(mmap)),
                transcoded_tensors: 0,
            });
        }
        match transcode_ptq1_image(&mmap)? {
            Some((image, transcoded)) => {
                drop(mmap);
                tracing::info!(
                    transcoded_tensors = transcoded,
                    image_bytes = image.len(),
                    "--ptq1-transcode: every PTQ1_0 tensor was re-encoded to PQ2_0 in memory"
                );
                Ok(Self {
                    bytes: ModelBytes::Owned(image),
                    transcoded_tensors: transcoded,
                })
            }
            None => {
                tracing::info!(
                    "--ptq1-transcode: this file has no PTQ1_0 tensors; using it unchanged"
                );
                Ok(Self {
                    bytes: ModelBytes::Mapped(Box::new(mmap)),
                    transcoded_tensors: 0,
                })
            }
        }
    }

    /// The GGUF bytes to parse.
    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.bytes {
            ModelBytes::Mapped(map) => map,
            ModelBytes::Owned(image) => image,
        }
    }

    /// What the weights occupy (the file, or the transcoded image) — the
    /// `weight_bytes` term of the context guard.
    pub(crate) fn weight_bytes(&self) -> u64 {
        self.bytes().len() as u64
    }

    /// How many tensors were transcoded (`0` = the original file is used).
    pub(crate) fn transcoded_tensors(&self) -> usize {
        self.transcoded_tensors
    }

    /// Leak the bytes for a process-lifetime server (the engine pool
    /// borrows its GGUF for `'static`).
    #[cfg(feature = "server")]
    pub(crate) fn into_static(self) -> &'static [u8] {
        match self.bytes {
            ModelBytes::Mapped(map) => {
                let map: &'static (dyn std::ops::Deref<Target = [u8]> + Send + Sync) =
                    Box::leak(map);
                map
            }
            ModelBytes::Owned(image) => Box::leak(image.into_boxed_slice()),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// GGUF alignment of `gguf` (`general.alignment` as `UINT32`, else 32 —
/// exactly what `GgufFile::parse` itself accepted).
fn gguf_alignment(gguf: &GgufFile<'_>) -> u64 {
    match gguf.metadata.get("general.alignment") {
        Some(MetadataValue::Uint32(v)) if *v > 0 => u64::from(*v),
        _ => DEFAULT_GGUF_ALIGNMENT,
    }
}

/// Build a GGUF image of `src` with every `PTQ1_0` tensor transcoded to
/// `PQ2_0`. Returns `None` when `src` holds no `PTQ1_0` tensor.
///
/// # Errors
///
/// `src` does not parse as GGUF, a `PTQ1_0` tensor's data is malformed, or
/// the rebuilt image fails to parse back.
pub(crate) fn transcode_ptq1_image(src: &[u8]) -> anyhow::Result<Option<(Vec<u8>, usize)>> {
    let gguf = GgufFile::parse(src).map_err(|e| anyhow::anyhow!("not a valid GGUF: {e}"))?;
    let tensors = gguf.tensors.sorted_by_offset();
    let ptq1_count = tensors
        .iter()
        .filter(|info| info.tensor_type == GgufTensorType::PTQ1_0)
        .count();
    if ptq1_count == 0 {
        return Ok(None);
    }

    // Byte range of header + metadata, reproduced verbatim.
    let (header, header_end) =
        GgufHeader::parse(src, 0).map_err(|e| anyhow::anyhow!("GGUF header: {e}"))?;
    let (_, metadata_end) = MetadataStore::parse(src, header_end, header.metadata_kv_count)
        .map_err(|e| anyhow::anyhow!("GGUF metadata: {e}"))?;
    let alignment = gguf_alignment(&gguf);

    // New per-tensor (type, byte size), in data order.
    struct Planned<'s> {
        name: &'s str,
        shape: &'s [u64],
        tensor_type: GgufTensorType,
        new_size: u64,
        new_offset: u64,
    }
    let mut planned: Vec<Planned<'_>> = Vec::with_capacity(tensors.len());
    let mut cursor = 0u64;
    for info in &tensors {
        let (tensor_type, new_size) = if info.tensor_type == GgufTensorType::PTQ1_0 {
            let n_blocks = info.element_count() / oxibonsai_core::quant_prism::QK_PTQ1_0 as u64;
            (
                GgufTensorType::PQ2_0,
                n_blocks * oxibonsai_core::quant_prism::BLOCK_PQ2_0_BYTES as u64,
            )
        } else {
            (info.tensor_type, info.data_size())
        };
        let new_offset = align_up(cursor, alignment);
        cursor = new_offset + new_size;
        planned.push(Planned {
            name: &info.name,
            shape: &info.shape,
            tensor_type,
            new_size,
            new_offset,
        });
    }

    // Tensor-info table (GGUF: name, n_dims, dims, type, offset).
    let mut info_table = Vec::new();
    for plan in &planned {
        info_table.extend_from_slice(&(plan.name.len() as u64).to_le_bytes());
        info_table.extend_from_slice(plan.name.as_bytes());
        info_table.extend_from_slice(&(plan.shape.len() as u32).to_le_bytes());
        for dim in plan.shape {
            info_table.extend_from_slice(&dim.to_le_bytes());
        }
        info_table.extend_from_slice(&plan.tensor_type.wire_id().to_le_bytes());
        info_table.extend_from_slice(&plan.new_offset.to_le_bytes());
    }

    let data_start = align_up((metadata_end + info_table.len()) as u64, alignment);
    // The reader requires every tensor's extent padded to the alignment —
    // the last one's included — to lie inside the file.
    let total = usize::try_from(data_start + align_up(cursor, alignment))
        .map_err(|_| anyhow::anyhow!("transcoded image does not fit this address space"))?;
    let mut image: Vec<u8> = Vec::with_capacity(total);
    image.extend_from_slice(&src[..metadata_end]);
    image.extend_from_slice(&info_table);
    image.resize(data_start as usize, 0);

    let mut scratch: Vec<oxibonsai_core::quant_prism::BlockPQ2_0> = Vec::new();
    for plan in &planned {
        let target = (data_start + plan.new_offset) as usize;
        image.resize(target, 0);
        let data = gguf
            .tensor_data(plan.name)
            .map_err(|e| anyhow::anyhow!("tensor '{}': {e}", plan.name))?;
        if plan.tensor_type == GgufTensorType::PQ2_0
            && gguf
                .tensors
                .get(plan.name)
                .is_some_and(|info| info.tensor_type == GgufTensorType::PTQ1_0)
        {
            let blocks = oxibonsai_core::quant_prism::BlockPTQ1_0::slice_from_bytes(data)
                .map_err(|e| anyhow::anyhow!("PTQ1_0 tensor '{}': {e}", plan.name))?;
            scratch.clear();
            scratch.resize(
                blocks.len(),
                oxibonsai_core::quant_prism::BlockPQ2_0::zeroed(),
            );
            oxibonsai_core::quant_prism::BlockPTQ1_0::transcode_to_pq2(blocks, &mut scratch)
                .map_err(|e| anyhow::anyhow!("transcoding '{}': {e}", plan.name))?;
            for block in &scratch {
                image.extend_from_slice(&block.d.to_le_bytes());
                image.extend_from_slice(&block.qs);
            }
        } else {
            image.extend_from_slice(data);
        }
        anyhow::ensure!(
            image.len() == target + plan.new_size as usize,
            "tensor '{}': wrote {} bytes, planned {}",
            plan.name,
            image.len() - target,
            plan.new_size
        );
    }

    image.resize(total, 0);

    // The image must be a well-formed GGUF on its own terms.
    GgufFile::parse(&image)
        .map_err(|e| anyhow::anyhow!("the transcoded GGUF image failed to parse back: {e}"))?;
    Ok(Some((image, ptq1_count)))
}

#[cfg(test)]
#[path = "model_source_tests.rs"]
mod tests;
