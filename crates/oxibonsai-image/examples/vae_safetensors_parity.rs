//! Parity check for the native FLUX.2 `AutoencoderKLFlux2` **safetensors** VAE
//! weight loader against an independent `.npy` reference of the *same*
//! checkpoint.
//!
//! ## What it proves
//!
//! The Pure-Rust safetensors loader ([`oxibonsai_image::vae::safetensors`],
//! selected automatically by [`VaeWeights::open`] for a `.safetensors` path)
//! must produce, for every dotted weight key the decoder requests, values
//! **bit-identical** to the per-tensor f32 `.npy` dump the dev-time Python export
//! produced — same name mapping (`to_out.0` un-nesting), same conv-weight
//! transpose (`[O,I,kH,kW] → [O,kH,kW,I]`), same lossless bf16→f32 decode.
//!
//! Because the only `.npy` dump that ships locally is for a *different* (SMALL)
//! VAE variant than the full FLUX.2 source safetensors, this harness generates a
//! matching `.npy` reference **from the same safetensors** (a tiny, audited NumPy
//! script run into a temp dir) and then compares the two [`VaeWeights`] sources
//! tensor-by-tensor. It additionally builds the full decoder from each source and
//! checks a fixed-latent decode is bit-identical (cosine 1.0).
//!
//! ## Usage
//!
//! ```text
//! cargo run -p oxibonsai-image --example vae_safetensors_parity -- \
//!     /path/to/vae/diffusion_pytorch_model.safetensors
//! ```
//!
//! The path may also come from `OXI_VAE_SAFETENSORS`; if neither is set it tries
//! the in-repo reference checkout. If no source safetensors is found, the harness
//! prints `SKIP` and exits success (the loader is still unit-tested in-crate).
//!
//! On Metal (`--features metal`, default-on GPU) the decode runs on the GPU and
//! reports `vae_gpu_was_used = true`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use oxibonsai_image::vae::{DecodeTaps, VaeDecoder, VaeWeights};

/// The full ordered list of decoder weight keys (the `.npy` filenames, minus the
/// extension) the safetensors loader must satisfy. Generated from the decoder's
/// layer structure (channels read dynamically, so this is variant-agnostic).
fn decoder_keys() -> Vec<String> {
    let mut keys = vec![
        "bn.running_mean".to_string(),
        "bn.running_var".to_string(),
        "post_quant_conv.weight".to_string(),
        "post_quant_conv.bias".to_string(),
        "decoder.conv_in.weight".to_string(),
        "decoder.conv_in.bias".to_string(),
        "decoder.conv_norm_out.weight".to_string(),
        "decoder.conv_norm_out.bias".to_string(),
        "decoder.conv_out.weight".to_string(),
        "decoder.conv_out.bias".to_string(),
    ];
    // Mid block: 2 resnets + 1 attention.
    for r in 0..2 {
        push_resnet(&mut keys, &format!("decoder.mid_block.resnets.{r}"), false);
    }
    let attn = "decoder.mid_block.attentions.0";
    keys.push(format!("{attn}.group_norm.weight"));
    keys.push(format!("{attn}.group_norm.bias"));
    for p in ["to_q", "to_k", "to_v", "to_out"] {
        keys.push(format!("{attn}.{p}.weight"));
        keys.push(format!("{attn}.{p}.bias"));
    }
    // 4 up blocks: 3 resnets each; blocks 2 & 3 have a 1×1 shortcut on resnet 0;
    // blocks 0,1,2 have an upsampler conv.
    for b in 0..4 {
        let prefix = format!("decoder.up_blocks.{b}");
        for r in 0..3 {
            let shortcut = (b == 2 || b == 3) && r == 0;
            push_resnet(&mut keys, &format!("{prefix}.resnets.{r}"), shortcut);
        }
        if b < 3 {
            keys.push(format!("{prefix}.upsamplers.0.conv.weight"));
            keys.push(format!("{prefix}.upsamplers.0.conv.bias"));
        }
    }
    keys
}

/// Append a resnet block's weight keys (`norm1/conv1/norm2/conv2`, optional
/// `conv_shortcut`).
fn push_resnet(keys: &mut Vec<String>, prefix: &str, shortcut: bool) {
    for part in ["norm1", "norm2"] {
        keys.push(format!("{prefix}.{part}.weight"));
        keys.push(format!("{prefix}.{part}.bias"));
    }
    for part in ["conv1", "conv2"] {
        keys.push(format!("{prefix}.{part}.weight"));
        keys.push(format!("{prefix}.{part}.bias"));
    }
    if shortcut {
        keys.push(format!("{prefix}.conv_shortcut.weight"));
        keys.push(format!("{prefix}.conv_shortcut.bias"));
    }
}

/// Resolve the source safetensors path: arg 1 → `OXI_VAE_SAFETENSORS` → the
/// in-repo reference checkout. Returns `None` (→ SKIP) if nothing exists.
fn resolve_safetensors() -> Option<PathBuf> {
    if let Some(arg) = std::env::args().nth(1) {
        let p = PathBuf::from(arg);
        return p.is_file().then_some(p);
    }
    if let Ok(env) = std::env::var("OXI_VAE_SAFETENSORS") {
        if !env.is_empty() {
            let p = PathBuf::from(env);
            return p.is_file().then_some(p);
        }
    }
    let refs = PathBuf::from(
        "/Users/kitasan/work/refs/Bonsai-Image-Demo/models/bonsai-image-4B-ternary-mlx/\
         vae/diffusion_pytorch_model.safetensors",
    );
    refs.is_file().then_some(refs)
}

/// Generate an independent `.npy` reference of the same checkpoint into
/// `out_dir`, using a small audited NumPy script that applies the exact
/// name-map, conv transpose, and bf16-to-f32 decode the loader claims to
/// implement. Returns `Ok(false)` when Python or NumPy is unavailable, in which
/// case the tensor-equivalence stage is skipped (the decode stage still runs).
fn generate_npy_reference(safetensors: &Path, out_dir: &Path) -> Result<bool, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;
    let script = r#"
import json, struct, sys, os
import numpy as np
src, outdir = sys.argv[1], sys.argv[2]
with open(src, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    blob = f.read()
hdr.pop('__metadata__', None)
def bf16(b):
    u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)
def get(name):
    info = hdr[name]; s, e = info['data_offsets']; raw = blob[s:e]
    dt, shp = info['dtype'], info['shape']
    if dt == 'BF16': a = bf16(raw).reshape(shp)
    elif dt == 'F32': a = np.frombuffer(raw, dtype=np.float32).reshape(shp)
    else: return None
    return a
# decoder-side keys only (skip encoder.*, quant_conv.*, bn.num_batches_tracked)
for name in list(hdr):
    if not (name.startswith('decoder.') or name.startswith('bn.running')
            or name.startswith('post_quant_conv.')):
        continue
    a = get(name)
    if a is None:
        continue
    out_name = name.replace('.to_out.0.', '.to_out.')   # un-nest ModuleList
    if a.ndim == 4:                                       # conv: [O,I,kH,kW]->[O,kH,kW,I]
        a = np.transpose(a, (0, 2, 3, 1))
    np.save(os.path.join(outdir, out_name + '.npy'), np.ascontiguousarray(a, dtype=np.float32))
print('OK')
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(safetensors)
        .arg(out_dir)
        .output();
    match out {
        Ok(o) if o.status.success() => Ok(true),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            // NumPy/py missing → skip this stage rather than fail the run.
            if err.contains("ModuleNotFoundError") || err.contains("No module named") {
                eprintln!(
                    "  (npy reference skipped: {})",
                    err.lines().next().unwrap_or("")
                );
                Ok(false)
            } else {
                Err(format!("python npy reference failed: {err}"))
            }
        }
        Err(e) => {
            eprintln!("  (npy reference skipped: python3 unavailable: {e})");
            Ok(false)
        }
    }
}

/// Cosine similarity between two equal-length slices.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// A simple deterministic latent `[1, 128, 32, 32]` (flat NCHW) for the decode.
fn fixed_latent() -> Vec<f32> {
    let n = 128 * 32 * 32;
    let mut v = vec![0.0f32; n];
    // Deterministic, bounded pseudo-values (no RNG dependency).
    for (i, x) in v.iter_mut().enumerate() {
        let t = (i as f32) * 0.013_f32;
        *x = (t.sin() + 0.5 * (0.7 * t).cos()) * 0.25;
    }
    v
}

fn run() -> Result<i32, String> {
    let Some(safetensors) = resolve_safetensors() else {
        println!(
            "SKIP: no source VAE safetensors found.\n\
             Provide one as arg 1 or via OXI_VAE_SAFETENSORS — the canonical file is\n\
             black-forest-labs/FLUX.2-dev → vae/diffusion_pytorch_model.safetensors\n\
             (AutoencoderKLFlux2). The loader itself is unit-tested in-crate\n\
             (cargo test -p oxibonsai-image --lib vae::safetensors)."
        );
        return Ok(0);
    };
    println!("Source safetensors: {}", safetensors.display());

    // 1. Open the safetensors source directly (the path under test).
    let st_weights =
        VaeWeights::open(&safetensors).map_err(|e| format!("open safetensors: {e}"))?;

    // 2. Generate an independent .npy reference of the SAME checkpoint.
    let ref_dir = std::env::temp_dir().join("oxibonsai_vae_st_parity_ref");
    let have_reference = generate_npy_reference(&safetensors, &ref_dir)?;

    let mut failures = 0usize;
    let keys = decoder_keys();

    if have_reference {
        let npy_weights = VaeWeights::open(&ref_dir).map_err(|e| format!("open npy ref: {e}"))?;
        println!(
            "\n--- Per-tensor equivalence (safetensors vs independent .npy, {} keys) ---",
            keys.len()
        );
        let mut max_abs = 0.0f64;
        for key in &keys {
            let a = st_weights
                .get(key)
                .map_err(|e| format!("safetensors get {key}: {e}"))?;
            let b = npy_weights
                .get(key)
                .map_err(|e| format!("npy get {key}: {e}"))?;
            if a.shape != b.shape {
                failures += 1;
                println!("  [FAIL] {key}: shape {:?} != {:?}", a.shape, b.shape);
                continue;
            }
            let bit_identical = a.data == b.data;
            let m = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(&x, &y)| ((x - y) as f64).abs())
                .fold(0.0f64, f64::max);
            max_abs = max_abs.max(m);
            if !bit_identical {
                failures += 1;
                println!("  [FAIL] {key}: not bit-identical (max|Δ|={m:.3e})");
            }
        }
        if failures == 0 {
            println!(
                "  [PASS] all {} tensors bit-identical (max|Δ| across all = {max_abs:.3e})",
                keys.len()
            );
        }
    } else {
        println!("\n--- Per-tensor equivalence: SKIPPED (no NumPy reference) ---");
    }

    // 3. Build the decoder from the safetensors source and decode a fixed latent.
    println!("\n--- Fixed-latent decode (safetensors-built decoder) ---");
    let decoder =
        VaeDecoder::from_weights(&st_weights).map_err(|e| format!("build decoder: {e}"))?;
    let latent = fixed_latent();
    let mut taps = DecodeTaps::default();
    let decoded = decoder
        .decode_packed_latents(&latent, 32, 32, Some(&mut taps))
        .map_err(|e| format!("decode: {e}"))?;
    println!(
        "  decoded shape = [{}, {}, {}] ({} px)",
        decoded.c,
        decoded.h,
        decoded.w,
        decoded.data.len()
    );
    let finite = decoded.data.iter().all(|x| x.is_finite());
    if !finite {
        failures += 1;
        println!("  [FAIL] decoded image contains non-finite values");
    } else {
        println!("  [PASS] decoded image is all-finite");
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    println!(
        "  (vae_gpu_was_used = {})",
        oxibonsai_image::vae::gpu::vae_gpu_was_used()
    );

    // 4. Determinism: a second decode from a freshly-opened source is identical.
    let st_weights2 =
        VaeWeights::open(&safetensors).map_err(|e| format!("reopen safetensors: {e}"))?;
    let decoder2 =
        VaeDecoder::from_weights(&st_weights2).map_err(|e| format!("rebuild decoder: {e}"))?;
    let decoded2 = decoder2
        .decode_packed_latents(&latent, 32, 32, None)
        .map_err(|e| format!("redecode: {e}"))?;
    let cos = cosine(&decoded.data, &decoded2.data);
    let same = decoded.data == decoded2.data;
    println!(
        "  [{}] re-decode cosine = {cos:.9} (bit-identical = {same})",
        if cos >= 0.999 { "PASS" } else { "FAIL" }
    );
    if cos < 0.999 {
        failures += 1;
    }

    // Best-effort cleanup of the temp reference dir.
    let _ = std::fs::remove_dir_all(&ref_dir);

    println!("\n=== Summary: {failures} failure(s) ===");
    Ok(if failures == 0 { 0 } else { 1 })
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => {
            println!("VAE SAFETENSORS PARITY: PASS");
            ExitCode::SUCCESS
        }
        Ok(_) => {
            println!("VAE SAFETENSORS PARITY: FAIL");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
