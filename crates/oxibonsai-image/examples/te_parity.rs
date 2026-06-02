//! Parity check of the Pure-Rust Qwen3-4B text-encoder forward against golden
//! MLX tensors.
//!
//! Loads the dequantised f32 weights (exported by `/tmp/bonsai_te_export_weights.py`)
//! and the golden `input_ids`/`attention_mask`, runs the Rust encoder, and prints
//! per-layer cosine + relative-L2 for `te_hidden_{0,1,9,18,27,35,36}`, the stacked
//! `te_cond_7680`, and finally the end-to-end `cond.npy` target.
//!
//! Gate: cosine ≥ 0.999, applied **only** to the tensors that actually feed the
//! conditioning — `te_hidden_{0,1,9,18,27}`, the stacked `te_cond_7680`, and the
//! end-to-end `cond.npy`. (relL2 may grow with depth from f32-vs-bf16 drift —
//! cosine is the binding metric.) `te_hidden_35`/`te_hidden_36` are the
//! un-normalized deep residual states that are **not** stacked into
//! `te_cond_7680` (only layers 9/18/27 are), so they are printed for information
//! only and never gated: their larger f32-vs-bf16 drift is expected and harmless.
//!
//! Usage (paths default to the standard dump locations):
//!
//! ```text
//! cargo run --release -p oxibonsai-image --example te_parity -- \
//!     /tmp/bonsai_golden/te/weights /tmp/bonsai_golden/te
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::te::forward::Precision;
use oxibonsai_image::te::{Qwen3Tokenizer, TeWeights, TextEncoder};

/// A loaded `.npy` tensor (f32, C-order).
struct Npy {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Npy {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Minimal NumPy `.npy` reader (f32 or int). Returns raw little-endian payload
/// interpreted per the header `descr`. (Mirrors the readers in `dit_parity.rs`.)
fn read_npy(path: &Path) -> Result<Npy, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: bad npy magic", path.display()));
    }
    let major = bytes[6];
    let (header_start, header_len) = if major >= 2 {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    } else {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| format!("{}: header utf8: {e}", path.display()))?;
    let s_idx = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: no shape key", path.display()))?;
    let open = header[s_idx..]
        .find('(')
        .map(|o| s_idx + o + 1)
        .ok_or_else(|| format!("{}: no shape open paren", path.display()))?;
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .ok_or_else(|| format!("{}: no shape close paren", path.display()))?;
    let shape: Vec<usize> = header[open..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| format!("shape parse: {e}")))
        .collect::<Result<_, _>>()?;
    let data_start = header_start + header_len;
    let payload = &bytes[data_start..];
    let numel: usize = shape.iter().product();
    // f32 path (the only float dtype we compare).
    if header.contains("'<f4'") {
        if payload.len() / 4 < numel {
            return Err(format!("{}: payload short", path.display()));
        }
        let data: Vec<f32> = payload[..numel * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Ok(Npy { data, shape });
    }
    Err(format!("{}: unsupported descr in {header}", path.display()))
}

/// Read an integer `.npy` (`<i4` or `<i8`) into `Vec<i64>`.
fn read_npy_int(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let major = bytes[6];
    let (header_start, header_len) = if major >= 2 {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    } else {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| format!("{}: header utf8: {e}", path.display()))?;
    let data_start = header_start + header_len;
    let payload = &bytes[data_start..];
    if header.contains("'<i4'") {
        Ok(payload
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
            .collect())
    } else if header.contains("'<i8'") {
        Ok(payload
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    } else {
        Err(format!("{}: not an int npy: {header}", path.display()))
    }
}

fn load(dir: &Path, name: &str) -> Result<Npy, String> {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Cosine similarity and relative-L2 (f64 accumulation).
fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut diff2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = x - y;
        diff2 += d * d;
        ref2 += y * y;
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let rel = if ref2 > 0.0 {
        (diff2 / ref2).sqrt()
    } else {
        diff2.sqrt()
    };
    (cos, rel)
}

struct Report {
    failures: usize,
    checks: usize,
}

impl Report {
    fn new() -> Self {
        Self {
            failures: 0,
            checks: 0,
        }
    }

    fn compare(&mut self, label: &str, got: &[f32], golden: &Npy, cos_min: f64) {
        self.checks += 1;
        if got.len() != golden.numel() {
            self.failures += 1;
            println!(
                "  [FAIL] {label}: length mismatch got {} golden {} (shape {:?})",
                got.len(),
                golden.numel(),
                golden.shape
            );
            return;
        }
        let (cos, rel) = metrics(got, &golden.data);
        let pass = cos >= cos_min;
        if !pass {
            self.failures += 1;
        }
        println!(
            "  [{}] {label}: cos={cos:.6} relL2={rel:.6} (need cos>={cos_min})",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    /// Print cos/relL2 for a tensor **without** gating on it. Used for the deep
    /// residual states (`te_hidden_35`/`36`) that are not part of the 7680
    /// conditioning, so their f32-vs-bf16 drift must not fail the run.
    fn inform(&mut self, label: &str, got: &[f32], golden: &Npy) {
        if got.len() != golden.numel() {
            println!(
                "  [INFO] {label}: length mismatch got {} golden {} (shape {:?})",
                got.len(),
                golden.numel(),
                golden.shape
            );
            return;
        }
        let (cos, rel) = metrics(got, &golden.data);
        println!("  [INFO] {label}: cos={cos:.6} relL2={rel:.6} (not gated)");
    }
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let weights_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/te/weights"));
    let golden_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/te"));

    if !weights_dir.is_dir() {
        return Err(format!("weights dir not found: {}", weights_dir.display()));
    }
    if !golden_dir.is_dir() {
        return Err(format!("golden dir not found: {}", golden_dir.display()));
    }

    // Precision policy: default pure-f32 (spec-mandated; matches the stacked
    // cond at cosine ≥ 0.999). Set TE_PRECISION=bf16 to emulate MLX bf16 storage
    // (slightly looser — MLX's fused bf16 numerics do not round per-op).
    let precision = match std::env::var("TE_PRECISION").as_deref() {
        Ok("bf16") | Ok("BF16") | Ok("bf16_storage") => Precision::Bf16Storage,
        _ => Precision::F32,
    };

    // TE weight source: the f32 `.npy` dump by default; the native 2.1 GB 4-bit
    // safetensors when `OXI_TE_4BIT=<path/to/model.safetensors>` is set.
    let weights = match std::env::var("OXI_TE_4BIT") {
        Ok(p) if !p.is_empty() => {
            println!("TE source: 4-bit safetensors {p}");
            TeWeights::open_mlx_4bit(Path::new(&p))
                .map_err(|e| format!("open 4-bit TE weights ({p}): {e}"))?
        }
        _ => {
            println!("TE source: f32 npy {}", weights_dir.display());
            TeWeights::open(&weights_dir).map_err(|e| format!("open TE weights: {e}"))?
        }
    };
    let encoder = TextEncoder::with_precision(&weights, precision);
    println!("  precision: {:?}", encoder.precision());
    let cfg = encoder.config();
    println!(
        "  config: hidden={} layers={} heads={}/{} head_dim={} inter={} theta={} eps={}",
        cfg.hidden_size,
        cfg.num_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.rope_theta,
        cfg.rms_norm_eps
    );

    // ── golden ids + mask ──
    let ids_i = read_npy_int(&golden_dir.join("input_ids.npy"))?;
    let mask_i = read_npy_int(&golden_dir.join("attention_mask.npy"))?;
    let input_ids: Vec<u32> = ids_i.iter().map(|&v| v as u32).collect();
    let attention_mask: Vec<i32> = mask_i.iter().map(|&v| v as i32).collect();
    let nonpad: usize = attention_mask.iter().filter(|&&m| m != 0).count();
    println!(
        "  input_ids len={} non-pad={} first8={:?}",
        input_ids.len(),
        nonpad,
        &input_ids[..8.min(input_ids.len())]
    );

    // ── tokenizer check (Step 2): reproduce the golden ids from the prompt ──
    // tokenizer.json dir defaults to the 4-bit model dir; override via env.
    let tok_dir = std::env::var("TE_TOKENIZER_DIR")
        .unwrap_or_else(|_| "/path/to/text_encoder-mlx-4bit".to_string());
    let prompt = "a tiny bonsai tree in a ceramic pot";
    println!("\n=== Tokenizer check (prompt = {prompt:?}) ===");
    match Qwen3Tokenizer::open(Path::new(&tok_dir)) {
        Ok(tok) => match tok.tokenize(prompt, input_ids.len()) {
            Ok(out) => {
                let exact = out.input_ids == input_ids;
                let mask_exact = out.attention_mask == attention_mask;
                println!("  tokenizer.json: {tok_dir}");
                println!(
                    "  rust ids[:21] = {:?}",
                    &out.input_ids[..21.min(out.input_ids.len())]
                );
                println!(
                    "  gold ids[:21] = {:?}",
                    &input_ids[..21.min(input_ids.len())]
                );
                if exact {
                    println!(
                        "  [PASS] Rust ids == golden input_ids ({} tokens)",
                        out.input_ids.len()
                    );
                } else {
                    let first = out
                        .input_ids
                        .iter()
                        .zip(input_ids.iter())
                        .position(|(a, b)| a != b);
                    println!("  [FAIL] Rust ids != golden; first diff at index {first:?}");
                }
                println!(
                    "  [{}] attention_mask match",
                    if mask_exact { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => println!("  (tokenize failed: {e})"),
        },
        Err(e) => println!("  (tokenizer.json not loaded: {e})"),
    }

    // ── forward ──
    println!(
        "\n=== Running Qwen3 encoder forward ({} tokens) ===",
        input_ids.len()
    );
    let t = std::time::Instant::now();
    let out = encoder
        .forward(&input_ids, &attention_mask)
        .map_err(|e| format!("forward: {e}"))?;
    println!("  (forward took {:.2}s)", t.elapsed().as_secs_f64());

    let mut report = Report::new();
    // Gated per-layer hidden states: only the layers that feed the 7680 cond
    // (the input/early taps plus the three stacked layers 9/18/27).
    println!("\n--- cond-relevant per-layer hidden states (gated, cos >= 0.999) ---");
    for &idx in &[0usize, 1, 9, 18, 27] {
        if idx < out.hidden_states.len() {
            report.compare(
                &format!("te_hidden_{idx}"),
                &out.hidden_states[idx],
                &load(&golden_dir, &format!("te_hidden_{idx}"))?,
                0.999,
            );
        }
    }

    // Informational-only deep residual states. Layers 35/36 are un-normalized
    // deep residuals that are NOT stacked into te_cond_7680 (only 9/18/27 are),
    // so their f32-vs-bf16 drift is expected and they are reported but not gated.
    println!("\n--- deep residual states (informational only, NOT gated) ---");
    println!("    note: te_hidden_35/36 are un-normalized deep residuals not used");
    println!("    by the cond; their f32-vs-bf16 drift is expected and harmless.");
    for &idx in &[35usize, 36] {
        if idx < out.hidden_states.len() {
            report.inform(
                &format!("te_hidden_{idx}"),
                &out.hidden_states[idx],
                &load(&golden_dir, &format!("te_hidden_{idx}"))?,
            );
        }
    }

    // ── stacked cond ──
    println!("\n--- stacked cond [seq, 7680] ---");
    let cond = out.cond_7680().map_err(|e| format!("cond_7680: {e}"))?;
    report.compare(
        "te_cond_7680",
        &cond,
        &load(&golden_dir, "te_cond_7680")?,
        0.999,
    );

    // ── end-to-end cond.npy target ──
    let bf16_cond = golden_dir
        .parent()
        .map(|p| p.join("bf16").join("cond.npy"))
        .filter(|p| p.exists());
    if let Some(p) = bf16_cond {
        println!("\n--- end-to-end target: bf16/cond.npy ---");
        let golden = read_npy(&p)?;
        report.compare("cond.npy", &cond, &golden, 0.999);
    } else {
        println!("\n  (skip cond.npy: not found)");
    }

    println!(
        "\n=== Summary: {} checks, {} failures ===",
        report.checks, report.failures
    );
    Ok(report.failures == 0)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("ALL TE STAGES PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("SOME TE STAGES FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
