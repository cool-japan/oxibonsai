//! Stage-by-stage parity check of the Pure-Rust FLUX.2 SMALL VAE decoder against
//! the golden (untiled) MLX tensors.
//!
//! Usage (paths default to the standard dump locations):
//!
//! ```text
//! cargo run -p oxibonsai-image --example vae_parity -- \
//!     /tmp/bonsai_golden/vae/weights /tmp/bonsai_golden/vae
//! ```
//!
//! It loads the exported decode-path weights + the per-stage goldens, feeds
//! `vae_in_packed` into the Rust decoder (capturing every intermediate), and
//! prints per-stage cosine similarity + relative-L2 with PASS/FAIL. Gate:
//! cosine of at least 0.999 AND relL2 within 2e-2 per tensor (relaxed to 5e-2 on
//! the bf16-conv-dominated intermediate stages, where f32-vs-bf16 drift grows).
//! It validates against the UNTILED `vae_decoded` (not `vae_decoded_tiled`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::vae::{DecodeTaps, VaeDecoder, VaeWeights};

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

/// Minimal NumPy `.npy` reader: v1.0/2.0 header, `descr=='<f4'`,
/// `fortran_order==False`. Returns the f32 data and parsed shape.
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

    /// Compare `got` to a named golden with the given tolerances.
    fn compare(&mut self, label: &str, got: &[f32], golden: &Npy, cos_min: f64, rel_max: f64) {
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
        let pass = cos >= cos_min && rel <= rel_max;
        if !pass {
            self.failures += 1;
        }
        println!(
            "  [{}] {label}: cos={cos:.6} relL2={rel:.6} (need cos>={cos_min}, relL2<={rel_max})",
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

    if !weights_dir.is_dir() {
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

    println!("\n=== Running untiled VAE decode (this may take a few minutes) ===");
    let t = std::time::Instant::now();
    let mut taps = DecodeTaps::default();
    let decoded = decoder
        .decode_packed_latents(&packed.data, ph, pw, Some(&mut taps))
        .map_err(|e| format!("decode: {e}"))?;
    println!("  (decode took {:.1}s)", t.elapsed().as_secs_f64());
    // Prove whether the GPU VAE path actually executed (vs a silent CPU
    // fallback). With OXI_VAE_GPU=1 (default-on) on macOS/Metal this must be
    // `true`; the CUDA sibling reports the same on Linux/Windows. cfg-gated so the
    // default (non-GPU) build — incl. Linux CPU — still compiles (the GPU hook
    // modules only exist under their respective feature+target gates).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    println!(
        "  (vae_gpu_was_used = {})",
        oxibonsai_image::vae::gpu::vae_gpu_was_used()
    );
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    println!(
        "  (vae_gpu_was_used = {})",
        oxibonsai_image::vae::cuda_gpu::vae_gpu_was_used()
    );

    let mut report = Report::new();
    println!("\n--- Per-stage parity (untiled) ---");

    // Tolerance policy (mirrors the DiT parity harness's treatment of the same
    // f32-vs-bf16 drift): cosine >= 0.999 is the binding directional gate at
    // EVERY stage. The goldens were computed with convs/attn in bf16 (GroupNorm
    // in f32); this crate is pure f32, so any stage *after the first bf16 conv*
    // carries an irreducible ~2-4% magnitude delta vs the bf16 golden — verified
    // by replaying the identical MLX conv in f32 (relL2 0.0213 at vae_conv_in,
    // byte-matching this Rust output) vs bf16 (relL2 0.0). So relL2 is gated at
    // 5e-2 for the bf16-conv-dominated intermediates (the spec's own escape
    // hatch), while the pre-conv stages and the FINAL decoded image — the actual
    // deliverable, which re-converges because conv_out lands back in a small
    // range — keep the tight 2e-2 bound.
    const REL_TIGHT: f64 = 2e-2;
    const REL_BF16: f64 = 5e-2;

    // Pre-conv stages: pure-f32 reshapes/denorm, tight tolerance.
    if let Some(v) = taps.bn_denorm.as_ref() {
        report.compare(
            "vae_bn_denorm",
            v,
            &load(&golden_dir, "vae_bn_denorm")?,
            0.999,
            REL_TIGHT,
        );
    }
    if let Some(v) = taps.unpatchified.as_ref() {
        report.compare(
            "vae_unpatchified",
            v,
            &load(&golden_dir, "vae_unpatchified")?,
            0.999,
            REL_TIGHT,
        );
    }
    if let Some(v) = taps.post_quant_conv.as_ref() {
        // k=1 conv (32 products): bf16 drift still under 2e-2.
        report.compare(
            "vae_post_quant_conv",
            v,
            &load(&golden_dir, "vae_post_quant_conv")?,
            0.999,
            REL_TIGHT,
        );
    }
    // From conv_in onward the golden is bf16-rounded per conv → relL2 ~2-4%.
    if let Some(v) = taps.conv_in.as_ref() {
        report.compare(
            "vae_conv_in",
            v,
            &load(&golden_dir, "vae_conv_in")?,
            0.999,
            REL_BF16,
        );
    }
    if let Some(v) = taps.mid.as_ref() {
        report.compare(
            "vae_mid",
            v,
            &load(&golden_dir, "vae_mid")?,
            0.999,
            REL_BF16,
        );
    }
    for (i, v) in taps.up.iter().enumerate() {
        report.compare(
            &format!("vae_up{i}"),
            v,
            &load(&golden_dir, &format!("vae_up{i}"))?,
            0.999,
            REL_BF16,
        );
    }
    if let Some(v) = taps.conv_norm_out.as_ref() {
        report.compare(
            "vae_conv_norm_out",
            v,
            &load(&golden_dir, "vae_conv_norm_out")?,
            0.999,
            REL_BF16,
        );
    }
    // Final decoded image (UNTILED golden) — tight bound.
    report.compare(
        "vae_decoded",
        &decoded.data,
        &load(&golden_dir, "vae_decoded")?,
        0.999,
        REL_TIGHT,
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
            println!("ALL VAE STAGES PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("SOME VAE STAGES FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
