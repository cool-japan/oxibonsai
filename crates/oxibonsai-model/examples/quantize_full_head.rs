//! Produces GGUFs whose LM head is quantized too, for the CUDA
//! Q-std/K-quant/FP8 path tests.
//!
//! `oxibonsai quantize` keeps `token_embd.weight`, `output_norm.weight` **and
//! `output.weight`** in FP32 (`ExportConfig::default_fp32_exceptions`), while
//! every CUDA Q4_0/Q8_0/K-quant/FP8 prefill and decode branch is selected by
//! the type of the LM head (`self.output_weight`). A file made by that
//! command therefore never reaches the branch it is named after. This
//! example runs the same streaming export pipeline
//! ([`export_to_gguf_streaming`]) with `output.weight` taken off the
//! exception list, mirroring `build_q_std_fixture` in
//! `tests/cuda_p11_q_std_kv_readback.rs`:
//!
//! * every 2-D weight, `output.weight` included, is written as `<format>`;
//! * `token_embd.weight` and `output_norm.weight` stay FP32 (the P11 rule:
//!   the embedding loader would only dequantize a Q-std/K-quant/FP8 table
//!   back to FP32 at load, and no CUDA branch of an untied model dispatches
//!   on it). A tied model (no `output.weight`) has its LM head in
//!   `token_embd.weight`, so that tensor is quantized instead;
//! * 1-D tensors and `*norm.weight` stay FP32 (`keep_fp32_by_kind`), and so
//!   does any tensor whose first dimension is not a block multiple (the
//!   writer's own fallback) — the run fails if that hits the LM head.
//!
//! The source must hold only F32/F16 tensors (re-quantizing an
//! already-quantized tensor would bake its error into the fixture).
//!
//! Publishing never overwrites anything, even with several fixture jobs
//! running concurrently against the same `models/` directory:
//!
//! * `<output>.partial` is created exclusively (`create_new`). If it already
//!   exists — another run is writing it, or a crashed run left it behind —
//!   the run stops before any work and leaves that file alone (it is never
//!   truncated or deleted by a run that did not create it);
//! * an existing `<output>` (any directory entry, a dangling symlink
//!   included) makes the run stop before any work;
//! * the partial file is re-opened and checked (LM-head type, tensor count)
//!   and then published with a hard link `<output>.partial` -> `<output>`.
//!   The OS refuses the link when `<output>` exists, so a file that appeared
//!   during the export (minutes) is never replaced — the run fails instead.
//!   Only after the link succeeded is the `.partial` name removed;
//! * on a filesystem without hard links the run fails and keeps the verified
//!   `<output>.partial`, for the user to move into place by hand.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p oxibonsai-model --example quantize_full_head -- \
//!     models/tb17-f32.gguf models/tb17h-q4_0.gguf q4_0
//! ```
//!
//! `<format>`: q4_0 q8_0 q2_k q3_k q4_k q5_k q6_k q8_k fp8_e4m3 fp8_e5m2, or
//! any other `ExportFormat` label (the export API refuses the ones it cannot
//! write).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_model::export::{
    export_to_gguf_streaming, ExportConfig, ExportError, ExportFormat, TensorPlan,
};

const LM_HEAD: &str = "output.weight";
const TOKEN_EMBD: &str = "token_embd.weight";
const OUTPUT_NORM: &str = "output_norm.weight";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, format] = args.as_slice() else {
        bail!(
            "usage: quantize_full_head <input.gguf> <output.gguf> <format>\n  formats: {}",
            format_names()
        );
    };
    let format = parse_format(format)?;
    run(Path::new(input), Path::new(output), format)
}

/// Every accepted spelling: the `oxibonsai quantize` names plus the labels.
fn format_names() -> String {
    let mut names = vec!["fp8_e4m3".to_owned(), "fp8_e5m2".to_owned()];
    names.extend(
        ExportFormat::ALL
            .iter()
            .map(|f| f.label().to_ascii_lowercase()),
    );
    names.join(" ")
}

fn parse_format(raw: &str) -> Result<ExportFormat> {
    // The `oxibonsai quantize` spellings that differ from the labels.
    let alias = match raw.to_ascii_lowercase().as_str() {
        "fp8_e4m3" => Some(ExportFormat::FP8E4M3),
        "fp8_e5m2" => Some(ExportFormat::FP8E5M2),
        "q1_0" => Some(ExportFormat::Q1_0G128),
        "ternary" => Some(ExportFormat::TernaryG128),
        _ => None,
    };
    alias
        .or_else(|| {
            ExportFormat::ALL
                .into_iter()
                .find(|f| f.label().eq_ignore_ascii_case(raw))
        })
        .ok_or_else(|| anyhow!("unknown format '{raw}'; one of: {}", format_names()))
}

/// Decode one F32/F16 source tensor to `f32`, checking its element count.
fn load_f32(gguf: &GgufFile<'_>, entry: &TensorPlan) -> Result<Vec<f32>, String> {
    let name = entry.name.as_str();
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor {name} vanished from the source"))?;
    let bytes = gguf.tensor_data(name).map_err(|e| e.to_string())?;
    let data: Vec<f32> = match info.tensor_type {
        GgufTensorType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect(),
        GgufTensorType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| half::f16::from_le_bytes(*h).to_f32())
            .collect(),
        other => return Err(format!("{name} is {}, not F32/F16", other.name())),
    };
    if data.len() != entry.num_elements() {
        return Err(format!(
            "{name}: decoded {} elements, the shape {:?} needs {}",
            data.len(),
            entry.shape,
            entry.num_elements()
        ));
    }
    Ok(data)
}

/// The `<output>.partial` this run created (exclusively). Dropping it
/// removes the file — after a successful publish that is only the second
/// name of the published inode — unless [`PartialFile::keep`] was called.
/// It is constructed only after `create_new` succeeded, so a run never
/// deletes a partial file that another run is writing.
struct PartialFile {
    path: PathBuf,
    remove_on_drop: bool,
}

impl PartialFile {
    /// Create `path` exclusively; refuses an existing file instead of
    /// truncating it.
    fn create_new(path: PathBuf) -> Result<(Self, std::fs::File)> {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => Ok((
                Self {
                    path,
                    remove_on_drop: true,
                },
                file,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => bail!(
                "{path:?} already exists: another quantize_full_head run is writing it, \
                 or a crashed run left it behind; remove it once no run is active"
            ),
            Err(e) => Err(e).with_context(|| format!("create {path:?}")),
        }
    }

    /// Leave the file on disk when this value is dropped.
    fn keep(&mut self) {
        self.remove_on_drop = false;
    }
}

impl Drop for PartialFile {
    fn drop(&mut self) {
        if !self.remove_on_drop {
            return;
        }
        if let Err(e) = std::fs::remove_file(&self.path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("note: could not remove {:?}: {e}", self.path);
            }
        }
    }
}

/// Publish the verified partial file as `output` without ever replacing an
/// existing entry: `link(2)` fails with `EEXIST` when `output` exists, unlike
/// `rename(2)`, which would silently replace it.
fn publish_no_clobber(partial: &mut PartialFile, output: &Path) -> Result<()> {
    match std::fs::hard_link(&partial.path, output) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => bail!(
            "{output:?} appeared while this run was exporting; refusing to overwrite it \
             (this run's output is discarded)"
        ),
        Err(e) => {
            partial.keep();
            bail!(
                "could not hard-link {:?} -> {output:?} ({e}); a filesystem without hard \
                 links? The verified file is kept at {:?}: move it into place by hand \
                 if {output:?} still does not exist",
                partial.path,
                partial.path
            )
        }
    }
}

fn run(input: &Path, output: &Path, format: ExportFormat) -> Result<()> {
    // Early exit only (the authoritative check is the hard link at the end).
    // `symlink_metadata` also sees a dangling symlink, which `link(2)` would
    // refuse as well.
    if std::fs::symlink_metadata(output).is_ok() {
        bail!("{output:?} already exists; refusing to overwrite it");
    }
    let target = format
        .tensor_type()
        .ok_or_else(|| anyhow!("{format:?} has no GGUF tensor type; nothing to write"))?;
    let start = Instant::now();
    let mmap = mmap_gguf_file(input).map_err(|e| anyhow!("mmap {input:?}: {e}"))?;
    let gguf = GgufFile::parse(&mmap).map_err(|e| anyhow!("parse {input:?}: {e}"))?;

    let mut names: Vec<&str> = gguf.tensors.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let mut plan = Vec::with_capacity(names.len());
    for &name in &names {
        let info = gguf
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("{input:?}: tensor {name} vanished"))?;
        if !matches!(info.tensor_type, GgufTensorType::F32 | GgufTensorType::F16) {
            bail!(
                "{input:?}: tensor {name} is {}; the fixture needs an all-F32/F16 source",
                info.tensor_type.name()
            );
        }
        plan.push(TensorPlan::new(
            name,
            info.shape.iter().map(|&d| d as usize).collect(),
        ));
    }

    // The LM head is `output.weight`, or `token_embd.weight` when tied.
    let tied = gguf.tensors.get(LM_HEAD).is_none();
    let lm_head = if tied { TOKEN_EMBD } else { LM_HEAD };
    let mut keep_fp32 = vec![OUTPUT_NORM.to_owned()];
    if !tied {
        keep_fp32.push(TOKEN_EMBD.to_owned());
    }
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("quantize-full-head");
    let source_name = input.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    let config = ExportConfig::new(format, stem)
        .with_fp32_layers(keep_fp32.clone())
        .with_source_metadata(&gguf.metadata)
        .context("carry the source metadata")?
        .with_description(&format!(
            "quantize_full_head: every 2-D weight incl. the LM head ({lm_head}) as {}; \
             FP32: {} + 1-D/norm tensors; source {source_name}",
            format.label(),
            keep_fp32.join(", ")
        ));
    println!(
        "quantize_full_head: {input:?} -> {output:?} as {} ({} tensors, LM head {lm_head}{})",
        format.label(),
        plan.len(),
        if tied { ", tied" } else { "" }
    );

    let mut partial_name = output.as_os_str().to_owned();
    partial_name.push(".partial");
    let (mut partial, file) = PartialFile::create_new(PathBuf::from(partial_name))?;
    let mut writer = std::io::BufWriter::with_capacity(8 << 20, file);
    let stats = export_to_gguf_streaming(
        &plan,
        |entry| {
            load_f32(&gguf, entry).map_err(|reason| ExportError::QuantizeError {
                name: entry.name.clone(),
                reason,
            })
        },
        &config,
        &[],
        &mut writer,
    )
    .with_context(|| format!("export {:?}", partial.path))?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|e| anyhow!("flush {:?}: {}", partial.path, e.error()))?;
    file.sync_all()?;
    drop(file);
    let encode_s = start.elapsed().as_secs_f64();

    verify(&partial.path, &plan, lm_head, target.wire_id())?;
    publish_no_clobber(&mut partial, output)?;
    // `<output>` now names the verified inode; drop the `.partial` name.
    drop(partial);
    let bytes = std::fs::metadata(output)?.len();
    println!(
        "wrote {output:?}: {bytes} bytes, {} tensors ({} quantized, {} FP32), \
         {:.2}:1 vs FP32, encode {encode_s:.1} s, total {:.1} s",
        stats.num_tensors,
        stats.quantized_tensors,
        stats.fp32_tensors,
        stats.compression_ratio,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Re-open the written file: same tensor count, LM head of the target type,
/// and a type histogram plus every 2-D tensor that stayed FP32.
fn verify(path: &Path, plan: &[TensorPlan], lm_head: &str, want_wire: u32) -> Result<()> {
    let mmap = mmap_gguf_file(path).map_err(|e| anyhow!("re-open {path:?}: {e}"))?;
    let gguf = GgufFile::parse(&mmap).map_err(|e| anyhow!("re-parse {path:?}: {e}"))?;
    if gguf.tensors.len() != plan.len() {
        bail!(
            "{path:?} holds {} tensors, the plan has {}",
            gguf.tensors.len(),
            plan.len()
        );
    }
    let head = gguf
        .tensors
        .get(lm_head)
        .ok_or_else(|| anyhow!("{path:?}: LM head {lm_head} missing"))?;
    println!(
        "lm_head {lm_head} type={} (ggml id {}) shape={:?}",
        head.tensor_type.name(),
        head.tensor_type.wire_id(),
        head.shape
    );
    if head.tensor_type.wire_id() != want_wire {
        bail!(
            "{path:?}: LM head {lm_head} came out {} (id {}), not ggml id {want_wire} — \
             its first dimension is probably not a block multiple",
            head.tensor_type.name(),
            head.tensor_type.wire_id()
        );
    }
    let mut histogram: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut fp32_2d: Vec<&str> = Vec::new();
    for (name, info) in gguf.tensors.iter() {
        *histogram.entry(info.tensor_type.name()).or_default() += 1;
        if info.tensor_type == GgufTensorType::F32 && info.shape.len() >= 2 {
            fp32_2d.push(name.as_str());
        }
    }
    fp32_2d.sort_unstable();
    let summary: Vec<String> = histogram.iter().map(|(t, n)| format!("{t}x{n}")).collect();
    println!("tensor types: {}", summary.join(" "));
    println!("2-D tensors kept FP32: {fp32_2d:?}");
    Ok(())
}
