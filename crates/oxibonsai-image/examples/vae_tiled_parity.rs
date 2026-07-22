//! Parity check: tiled VAE decode vs untiled VAE decode.
//!
//! Validates that `decode_packed_latents_tiled` produces the same result as
//! `decode_packed_latents` (cos ≥ 0.999, relL2 ≤ 2e-2) using the standard
//! 32×32 latent golden.
//!
//! Usage (paths default to the standard dump locations):
//!
//! ```text
//! cargo run -p oxibonsai-image --example vae_tiled_parity -- \
//!     /tmp/bonsai_golden/vae/weights /tmp/bonsai_golden/vae
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::vae::{TileBoundary, TileConfig, VaeDecoder, VaeWeights};

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

/// Minimal NumPy `.npy` reader: v1.0/2.0, `descr=='<f4'`.
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
    if !header.contains("'<f4'") {
        return Err(format!("{}: descr is not '<f4': {header}", path.display()));
    }
    let fortran = header.contains("'fortran_order': True");
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
    if payload.len() % 4 != 0 {
        return Err(format!("{}: payload not f32-aligned", path.display()));
    }
    let raw: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let numel: usize = shape.iter().product();
    if raw.len() < numel {
        return Err(format!(
            "{}: payload short ({} < {})",
            path.display(),
            raw.len(),
            numel
        ));
    }
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&raw[..numel], &shape)
    } else {
        raw
    };
    Ok(Npy { data, shape })
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

/// Load a named golden tensor from the dump directory.
fn load(dir: &Path, name: &str) -> Result<Npy, String> {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Cosine similarity and relative-L2 between two equal-length slices.
fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    debug_assert_eq!(a.len(), b.len());
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
    let rel_l2 = if ref2 > 0.0 {
        (diff2 / ref2).sqrt()
    } else {
        diff2.sqrt()
    };
    (cos, rel_l2)
}

/// Accumulates pass/fail across all comparisons.
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

    fn check(&mut self, label: &str, got: &[f32], reference: &[f32], cos_min: f64, rel_max: f64) {
        self.checks += 1;
        if got.len() != reference.len() {
            self.failures += 1;
            println!(
                "  [FAIL] {label}: length mismatch got {} ref {}",
                got.len(),
                reference.len()
            );
            return;
        }
        let (cos, rel) = metrics(got, reference);
        let pass = cos >= cos_min && rel <= rel_max;
        if !pass {
            self.failures += 1;
        }
        println!(
            "  [{}] {label}: cos={cos:.8} relL2={rel:.8} (need cos>={cos_min}, relL2<={rel_max})",
            if pass { "PASS" } else { "FAIL" }
        );
    }
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let weights_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/vae/weights"));
    let golden_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/vae"));

    if !weights_dir.exists() {
        return Err(format!("weights dir not found: {}", weights_dir.display()));
    }
    if !golden_dir.is_dir() {
        return Err(format!("golden dir not found: {}", golden_dir.display()));
    }

    println!("Loading VAE weights: {}", weights_dir.display());
    let weights = VaeWeights::open(&weights_dir).map_err(|e| format!("open weights: {e}"))?;
    let decoder = VaeDecoder::from_weights(&weights).map_err(|e| format!("build decoder: {e}"))?;

    // Input: vae_in_packed [1,128,32,32] (flat NCHW).
    let packed = load(&golden_dir, "vae_in_packed")?;
    if packed.shape != vec![1, 128, 32, 32] {
        return Err(format!("vae_in_packed unexpected shape {:?}", packed.shape));
    }
    let (ph, pw) = (32usize, 32usize);

    println!("\n=== 1. Untiled baseline (reference) ===");
    let t = std::time::Instant::now();
    let untiled = decoder
        .decode_packed_latents(&packed.data, ph, pw, None)
        .map_err(|e| format!("untiled decode: {e}"))?;
    println!("  (untiled decode took {:.1}s)", t.elapsed().as_secs_f64());

    let mut report = Report::new();

    // Cross-check untiled against the MLX golden (sanity gate).
    if let Ok(golden) = load(&golden_dir, "vae_decoded") {
        if golden.numel() == untiled.data.len() {
            report.check(
                "untiled_vs_mlx_golden",
                &untiled.data,
                &golden.data,
                0.999,
                2e-2,
            );
        }
    }

    println!("\n=== 2. Tiled AfterConvNormOut (tile_px=128, forces 4×4=16 tiles at 512px) ===");
    let cfg_after_norm = TileConfig {
        tile_px: 128,
        boundary: TileBoundary::AfterConvNormOut,
    };
    let t = std::time::Instant::now();
    let tiled_norm = decoder
        .decode_packed_latents_tiled(&packed.data, ph, pw, cfg_after_norm)
        .map_err(|e| format!("tiled AfterConvNormOut decode: {e}"))?;
    println!(
        "  (tiled AfterConvNormOut took {:.1}s)",
        t.elapsed().as_secs_f64()
    );

    // Gate: cos >= 0.999 vs untiled (the binding production gate).
    report.check(
        "AfterConvNormOut_vs_untiled (cos >= 0.999)",
        &tiled_norm.data,
        &untiled.data,
        0.999,
        2e-2,
    );
    // Stricter in-process gate: same decode path, should be bit-exact or very
    // close (both run the identical CPU/GPU path; only the silu+conv_out is
    // tiled which is exact per-pixel). Gate: cos >= 0.99999.
    report.check(
        "AfterConvNormOut_vs_untiled (strict cos >= 0.99999)",
        &tiled_norm.data,
        &untiled.data,
        0.999_99,
        1e-3,
    );

    println!("\n=== 3. Tiled AfterMid (same fallback path as AfterConvNormOut) ===");
    let cfg_after_mid = TileConfig {
        tile_px: 128,
        boundary: TileBoundary::AfterMid,
    };
    let t = std::time::Instant::now();
    let tiled_mid = decoder
        .decode_packed_latents_tiled(&packed.data, ph, pw, cfg_after_mid)
        .map_err(|e| format!("tiled AfterMid decode: {e}"))?;
    println!("  (tiled AfterMid took {:.1}s)", t.elapsed().as_secs_f64());

    report.check(
        "AfterMid_vs_untiled (cos >= 0.999)",
        &tiled_mid.data,
        &untiled.data,
        0.999,
        2e-2,
    );
    report.check(
        "AfterMid_vs_AfterConvNormOut (should be identical)",
        &tiled_mid.data,
        &tiled_norm.data,
        0.999_99,
        1e-3,
    );

    println!(
        "\n=== Summary: {} checks, {} failures ===",
        report.checks, report.failures
    );
    Ok(report.failures == 0)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("ALL TILED-PARITY CHECKS PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("SOME TILED-PARITY CHECKS FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
