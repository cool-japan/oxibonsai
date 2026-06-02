//! Oracle check for the MLX 4-bit text-encoder weight loader.
//!
//! For a representative set of tensors, this dequantises directly from the
//! native 4-bit `model.safetensors` (via [`oxibonsai_image::te::TeWeights`]'s
//! 4-bit source) AND reads the f32 `.npy` oracle (literally `mx.dequantize(...)`
//! output), then reports shape match, **max|Δ|**, and **cosine**.
//!
//! Pass criterion: `max|Δ| < 1e-2` and `cos ≥ 0.99999` for every listed tensor
//! — i.e. the Rust dequant must reproduce the MLX ground truth near-exactly. A
//! failure means the affine form / nibble packing is wrong.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p oxibonsai-image --example te_4bit_check -- \
//!     /path/to/text_encoder-mlx-4bit/model.safetensors \
//!     /tmp/bonsai_golden/te/weights
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use oxibonsai_image::te::{read_npy_f32, TeWeights};

/// Max absolute difference threshold for a pass.
const MAX_ABS_THRESH: f64 = 1e-2;
/// Minimum cosine similarity for a pass.
const COS_MIN: f64 = 0.99999;

/// Cosine similarity and max|Δ| (f64 accumulation).
fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    (cos, max_abs)
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let st_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: te_4bit_check <safetensors> <oracle_npy_dir>")?;
    let oracle_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: te_4bit_check <safetensors> <oracle_npy_dir>")?;

    if !st_path.is_file() {
        return Err(format!("safetensors not found: {}", st_path.display()));
    }
    if !oracle_dir.is_dir() {
        return Err(format!("oracle dir not found: {}", oracle_dir.display()));
    }

    println!("4-bit safetensors: {}", st_path.display());
    println!("f32 npy oracle:    {}", oracle_dir.display());

    let weights =
        TeWeights::open_mlx_4bit(&st_path).map_err(|e| format!("open 4-bit safetensors: {e}"))?;

    // Representative set: an embedding, attention in/out projs, an MLP down
    // proj (in=intermediate), a plain RMSNorm vector, a deep-layer MLP, and the
    // final norm. Mixes both `in` dims and both tensor kinds.
    let names = [
        "embed_tokens",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.down_proj",
        "layers.0.input_layernorm",
        "layers.35.mlp.gate_proj",
        "norm",
    ];

    let mut failures = 0usize;
    for name in names {
        let deq = match weights.get(name) {
            Ok(t) => t,
            Err(e) => {
                println!("  [FAIL] {name}: dequant error: {e}");
                failures += 1;
                continue;
            }
        };
        let oracle = match read_npy_f32(&oracle_dir.join(format!("{name}.npy"))) {
            Ok(t) => t,
            Err(e) => {
                println!("  [FAIL] {name}: oracle read error: {e}");
                failures += 1;
                continue;
            }
        };

        let shape_ok = deq.shape == oracle.shape;
        if !shape_ok || deq.data.len() != oracle.data.len() {
            println!(
                "  [FAIL] {name}: shape mismatch deq {:?} oracle {:?}",
                deq.shape, oracle.shape
            );
            failures += 1;
            continue;
        }

        let (cos, max_abs) = metrics(&deq.data, &oracle.data);
        let pass = max_abs < MAX_ABS_THRESH && cos >= COS_MIN;
        if !pass {
            failures += 1;
        }
        println!(
            "  [{}] {name}: shape {:?} max|Δ|={max_abs:.3e} cos={cos:.7} (need max|Δ|<{MAX_ABS_THRESH:.0e}, cos>={COS_MIN})",
            if pass { "PASS" } else { "FAIL" },
            deq.shape
        );
    }

    println!("\n=== {} tensors, {failures} failures ===", names.len());
    Ok(failures == 0)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("4-BIT ORACLE CHECK PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("4-BIT ORACLE CHECK FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
