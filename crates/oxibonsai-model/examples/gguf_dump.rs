//! GGUF tensor dump utility for OxiBonsai.
//!
//! Loads a GGUF file via `oxibonsai-core::gguf::reader::GgufFile`, prints per-tensor
//! `{name, type_code, type_name, shape, size_bytes}` lines sorted by name, and a
//! histogram aggregated by tensor type.
//!
//! Usage: cargo run -p oxibonsai-model --example gguf_dump --features mmap -- <path.gguf>

use std::collections::HashMap;
use std::path::PathBuf;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::types::GgufTensorType;

fn main() {
    if let Err(e) = run() {
        eprintln!("gguf_dump error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .ok_or_else(|| "usage: gguf_dump <path.gguf>".to_string())?;
    let path = PathBuf::from(path);

    let mmap = mmap_gguf_file(&path).map_err(|e| format!("mmap failed: {e}"))?;
    let file = GgufFile::parse(&mmap).map_err(|e| format!("parse failed: {e}"))?;

    println!("== {} ==", path.display());
    println!(
        "header: version={} tensors={} metadata_kv={}",
        file.header.version, file.header.tensor_count, file.header.metadata_kv_count
    );
    println!("data_offset={}", file.data_offset);
    println!();

    // Per-tensor lines, sorted by name.
    let mut entries: Vec<(&str, &oxibonsai_core::gguf::tensor_info::TensorInfo)> =
        file.tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    println!("== Per-tensor ==");
    for (name, info) in &entries {
        // `wire_id()` rather than a raw discriminant cast: `GgufTensorType`
        // carries sentinel variants (`Q2_0G64`, `Q2_0G128DFirst`) whose raw
        // discriminants are NOT the on-disk ggml id, so casting would print
        // `1073741866` instead of `42` for a legacy PrismML file
        // (core-gguf-12 / B2-01).
        let tid = info.tensor_type.wire_id();
        println!(
            "{:<40} type={} ({:<12}) shape={:?} bytes={}",
            name,
            tid,
            info.tensor_type.name(),
            info.shape,
            info.data_size()
        );
    }

    // Histogram by tensor type.
    let mut count: HashMap<GgufTensorType, (usize, u64)> = HashMap::new();
    for (_, info) in &entries {
        let e = count.entry(info.tensor_type).or_insert((0, 0));
        e.0 += 1;
        e.1 += info.data_size();
    }
    let mut hist: Vec<(GgufTensorType, usize, u64)> =
        count.into_iter().map(|(k, (c, b))| (k, c, b)).collect();
    hist.sort_by_key(|b| std::cmp::Reverse(b.2));

    println!();
    println!("== Type histogram (sorted by aggregate bytes desc) ==");
    let mut total: u64 = 0;
    for (ty, c, bytes) in &hist {
        total += *bytes;
        println!(
            "type={:<3} {:<12} count={:<5} bytes={:>13} ({:.2} MiB)",
            ty.wire_id(),
            ty.name(),
            c,
            bytes,
            (*bytes as f64) / (1024.0 * 1024.0)
        );
    }
    println!(
        "TOTAL tensor bytes: {} ({:.2} MiB)",
        total,
        (total as f64) / (1024.0 * 1024.0)
    );

    Ok(())
}
