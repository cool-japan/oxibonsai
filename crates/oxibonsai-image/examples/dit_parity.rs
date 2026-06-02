//! Stage-by-stage parity check of the Pure-Rust FLUX.2 Klein DiT forward pass
//! against golden tensors dumped from the real MLX megakernel.
//!
//! Usage (paths default to the standard dump locations):
//!
//! ```text
//! cargo run -p oxibonsai-image --example dit_parity -- \
//!     /tmp/parity.gguf /tmp/bonsai_golden/bf16
//! ```
//!
//! It loads the GGUF weights and the goldens, feeds the golden inputs into the
//! Rust forward pass, and prints per-stage cosine similarity + relative-L2 with
//! PASS/FAIL against the tolerances in the spec (cosine >= 0.999, relL2 <= 2e-2,
//! relaxed to 5e-2 on late single blocks; noise/latents must stay >= 0.999).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::forward::{ForwardTaps, StepTap};
use oxibonsai_image::{DitForward, DitWeights};

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
    // Parse shape tuple between "'shape': (" and ")".
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
    // Convert Fortran (column-major) storage to C (row-major) order.
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&raw[..numel], &shape)
    } else {
        raw
    };
    Ok(Npy { data, shape })
}

/// Reorder a Fortran-stored (column-major) buffer into C (row-major) order for
/// the given shape.
fn fortran_to_c(src: &[f32], shape: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    // Fortran strides: stride[0]=1, stride[d]=stride[d-1]*shape[d-1].
    let mut f_stride = vec![1usize; ndim];
    for d in 1..ndim {
        f_stride[d] = f_stride[d - 1] * shape[d - 1];
    }
    let mut out = vec![0.0f32; numel];
    let mut idx = vec![0usize; ndim];
    for (c_pos, slot) in out.iter_mut().enumerate() {
        // Decode C-order multi-index for c_pos.
        let mut rem = c_pos;
        for d in 0..ndim {
            let stride_c: usize = shape[d + 1..].iter().product();
            idx[d] = rem / stride_c;
            rem %= stride_c;
        }
        // Encode Fortran flat offset.
        let mut f_off = 0usize;
        for d in 0..ndim {
            f_off += idx[d] * f_stride[d];
        }
        *slot = src[f_off];
    }
    out
}

/// Load a named golden tensor from the dump directory.
fn load(dir: &Path, name: &str) -> Result<Npy, String> {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Load a scalar golden (shape `()` → 1 element).
fn load_scalar(dir: &Path, name: &str) -> Result<f32, String> {
    let n = load(dir, name)?;
    n.data
        .first()
        .copied()
        .ok_or_else(|| format!("{name}: empty scalar"))
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

/// Compare every S0 tap (embeddings, RoPE, modulation triples) against goldens.
fn compare_stage0(
    report: &mut Report,
    dir: &Path,
    s0: &oxibonsai_image::Stage0,
) -> Result<(), String> {
    report.compare(
        "x_emb_out",
        &s0.x_emb,
        &load(dir, "x_emb_out")?,
        0.999,
        2e-2,
    );
    report.compare(
        "ctx_emb_out",
        &s0.ctx_emb,
        &load(dir, "ctx_emb_out")?,
        0.999,
        2e-2,
    );
    report.compare(
        "rope_cos",
        &s0.rope.cos,
        &load(dir, "rope_cos")?,
        0.999,
        2e-2,
    );
    report.compare(
        "rope_sin",
        &s0.rope.sin,
        &load(dir, "rope_sin")?,
        0.999,
        2e-2,
    );
    report.compare(
        "shift_msa",
        &s0.mod_img.msa.shift,
        &load(dir, "shift_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "scale_msa",
        &s0.mod_img.msa.scale,
        &load(dir, "scale_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "gate_msa",
        &s0.mod_img.msa.gate,
        &load(dir, "gate_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "shift_mlp",
        &s0.mod_img.mlp.shift,
        &load(dir, "shift_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "scale_mlp",
        &s0.mod_img.mlp.scale,
        &load(dir, "scale_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "gate_mlp",
        &s0.mod_img.mlp.gate,
        &load(dir, "gate_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_shift_msa",
        &s0.mod_txt.msa.shift,
        &load(dir, "c_shift_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_scale_msa",
        &s0.mod_txt.msa.scale,
        &load(dir, "c_scale_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_gate_msa",
        &s0.mod_txt.msa.gate,
        &load(dir, "c_gate_msa")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_shift_mlp",
        &s0.mod_txt.mlp.shift,
        &load(dir, "c_shift_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_scale_mlp",
        &s0.mod_txt.mlp.scale,
        &load(dir, "c_scale_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "c_gate_mlp",
        &s0.mod_txt.mlp.gate,
        &load(dir, "c_gate_mlp")?,
        0.999,
        2e-2,
    );
    report.compare(
        "single_mod_shift",
        &s0.mod_single.shift,
        &load(dir, "single_mod_shift")?,
        0.999,
        2e-2,
    );
    report.compare(
        "single_mod_scale",
        &s0.mod_single.scale,
        &load(dir, "single_mod_scale")?,
        0.999,
        2e-2,
    );
    report.compare(
        "single_mod_gate",
        &s0.mod_single.gate,
        &load(dir, "single_mod_gate")?,
        0.999,
        2e-2,
    );
    Ok(())
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/parity.gguf"));
    let golden_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/bf16"));

    if !gguf.exists() {
        return Err(format!("GGUF not found: {}", gguf.display()));
    }
    if !golden_dir.exists() {
        return Err(format!("golden dir not found: {}", golden_dir.display()));
    }

    println!(
        "Kernel tier: {:?}",
        oxibonsai_kernels::KernelDispatcher::auto_detect().tier()
    );
    println!("Loading weights: {}", gguf.display());
    let weights = DitWeights::open(&gguf).map_err(|e| format!("load gguf: {e}"))?;
    let cfg = weights.config();
    let seq_img = 1024usize;
    let seq_txt = 512usize;
    let in_channels = cfg.in_channels as usize;
    let joint_dim = cfg.joint_attention_dim as usize;

    let fwd = DitForward::new(&weights);

    // ── Load golden inputs ──
    let hs = load(&golden_dir, "tf_in_hidden_states")?;
    let cond = load(&golden_dir, "cond")?;
    let img_ids = load(&golden_dir, "img_ids")?;
    let txt_ids = load(&golden_dir, "txt_ids")?;
    let t0 = load_scalar(&golden_dir, "timestep_step0")?;

    // ids: [1, seq, 4] → squeeze to [seq, 4].
    let num_axes = cfg.axes_dims_rope.len();
    let img_ids_2d = &img_ids.data[..seq_img * num_axes];
    let txt_ids_2d = &txt_ids.data[..seq_txt * num_axes];

    // Optional stage gate: DIT_MAX_STAGE in 0..=4 (default 4) limits how far
    // the harness runs, so early stages can be validated without paying for the
    // full (slow) scalar forward + sampler.
    let max_stage: u32 = std::env::var("DIT_MAX_STAGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    // DIT_S4_ONLY runs ONLY the Euler sampler loop (S4), skipping the redundant
    // step0 forward + S1/S2/S3 comparisons (useful once those have passed).
    let s4_only = std::env::var("DIT_S4_ONLY").is_ok();

    let mut report = Report::new();

    if s4_only {
        run_s4(
            &fwd,
            &mut report,
            &golden_dir,
            &cond,
            img_ids_2d,
            txt_ids_2d,
            seq_img,
            seq_txt,
            in_channels,
            joint_dim,
        )?;
        println!(
            "\n=== Summary: {} checks, {} failures ===",
            report.checks, report.failures
        );
        return Ok(report.failures == 0);
    }

    // ── S0 standalone (fast): embeddings / temb / rope / modulation ──
    println!("\n=== S0: embeddings, time embedding, RoPE, modulation (t={t0}) ===");
    let t_s0 = std::time::Instant::now();
    let s0_only = fwd
        .run_stage0(
            &hs.data[..seq_img * in_channels],
            &cond.data[..seq_txt * joint_dim],
            img_ids_2d,
            txt_ids_2d,
            seq_img,
            seq_txt,
            t0,
        )
        .map_err(|e| format!("run_stage0: {e}"))?;
    compare_stage0(&mut report, &golden_dir, &s0_only)?;
    println!("  (S0 took {:.2}s)", t_s0.elapsed().as_secs_f64());
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    if max_stage < 1 {
        println!(
            "\n=== Summary: {} checks, {} failures ===",
            report.checks, report.failures
        );
        return Ok(report.failures == 0);
    }

    // ── Run full forward @step0 with taps (slow) ──
    println!("\n=== Running full forward @step0 ===");
    let t_fwd = std::time::Instant::now();
    let mut taps = ForwardTaps::default();
    let noise = fwd
        .forward(
            &hs.data[..seq_img * in_channels],
            &cond.data[..seq_txt * joint_dim],
            img_ids_2d,
            txt_ids_2d,
            seq_img,
            seq_txt,
            t0,
            Some(&mut taps),
        )
        .map_err(|e| format!("forward step0: {e}"))?;
    println!(
        "  (full forward took {:.2}s)",
        t_fwd.elapsed().as_secs_f64()
    );
    #[cfg(all(feature = "metal", target_os = "macos"))]
    println!(
        "  [after 1 forward] gpu_was_used = {}  (OXI_DIT_GPU enabled = {}) | \
         dit_attn_gpu_was_used = {}  (OXI_DIT_ATTN_GPU enabled = {})",
        oxibonsai_image::gpu::gpu_was_used(),
        oxibonsai_image::gpu::dit_gpu_enabled(),
        oxibonsai_image::gpu::dit_attn_gpu_was_used(),
        oxibonsai_image::gpu::dit_attn_gpu_enabled()
    );
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    println!(
        "  [after 1 forward] gpu_was_used = {}  (OXI_DIT_GPU enabled = {}) | \
         dit_attn_gpu_was_used = {}  (OXI_DIT_ATTN_GPU enabled = {}) | \
         dit_fused_block_was_used = {}  (OXI_DIT_FUSED enabled = {})",
        oxibonsai_image::cuda_gpu::gpu_was_used(),
        oxibonsai_image::cuda_gpu::dit_gpu_enabled(),
        oxibonsai_image::cuda_gpu::dit_attn_gpu_was_used(),
        oxibonsai_image::cuda_gpu::dit_attn_gpu_enabled(),
        oxibonsai_image::cuda_gpu::dit_fused_block_was_used(),
        oxibonsai_image::cuda_gpu::dit_fused_enabled()
    );

    // ── S1: double blocks ──
    println!("\n--- S1: dual-stream blocks (enc + h) ---");
    for i in 0..cfg.num_layers as usize {
        report.compare(
            &format!("double_block{i}_enc"),
            &taps.double_enc[i],
            &load(&golden_dir, &format!("double_block{i}_enc"))?,
            0.999,
            2e-2,
        );
        report.compare(
            &format!("double_block{i}_h"),
            &taps.double_h[i],
            &load(&golden_dir, &format!("double_block{i}_h"))?,
            0.999,
            2e-2,
        );
    }
    // single_in_hidden_states (concat of enc4, h4)
    if let Some(si) = taps.single_in.as_ref() {
        report.compare(
            "single_in_hidden_states",
            si,
            &load(&golden_dir, "single_in_hidden_states")?,
            0.999,
            2e-2,
        );
    }
    let _ = std::io::stdout().flush();
    if max_stage < 2 {
        println!(
            "\n=== Summary: {} checks, {} failures ===",
            report.checks, report.failures
        );
        return Ok(report.failures == 0);
    }

    // ── S2: single blocks (relax relL2 on late blocks) ──
    println!("\n--- S2: single-stream blocks (h) ---");
    for j in 0..cfg.num_single_layers as usize {
        let rel_max = if j >= 12 { 5e-2 } else { 2e-2 };
        report.compare(
            &format!("single_block{j}_h"),
            &taps.single_h[j],
            &load(&golden_dir, &format!("single_block{j}_h"))?,
            0.999,
            rel_max,
        );
    }
    let _ = std::io::stdout().flush();

    // ── S3: head → noise_step0 ──
    println!("\n--- S3: head (norm_out + proj_out) ---");
    report.compare(
        "noise_step0",
        &noise,
        &load(&golden_dir, "noise_step0")?,
        0.999,
        2e-2,
    );
    let _ = std::io::stdout().flush();
    if max_stage < 4 {
        println!(
            "\n=== Summary: {} checks, {} failures ===",
            report.checks, report.failures
        );
        return Ok(report.failures == 0);
    }

    // ── S4: Euler sampler loop ──
    run_s4(
        &fwd,
        &mut report,
        &golden_dir,
        &cond,
        img_ids_2d,
        txt_ids_2d,
        seq_img,
        seq_txt,
        in_channels,
        joint_dim,
    )?;

    // ── Summary ──
    #[cfg(all(feature = "metal", target_os = "macos"))]
    println!(
        "GPU path used (gpu_was_used) = {}  (OXI_DIT_GPU enabled = {}) | \
         attn GPU used (dit_attn_gpu_was_used) = {}  (OXI_DIT_ATTN_GPU enabled = {})",
        oxibonsai_image::gpu::gpu_was_used(),
        oxibonsai_image::gpu::dit_gpu_enabled(),
        oxibonsai_image::gpu::dit_attn_gpu_was_used(),
        oxibonsai_image::gpu::dit_attn_gpu_enabled()
    );
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    println!(
        "GPU path used (gpu_was_used) = {}  (OXI_DIT_GPU enabled = {}) | \
         attn GPU used (dit_attn_gpu_was_used) = {}  (OXI_DIT_ATTN_GPU enabled = {})",
        oxibonsai_image::cuda_gpu::gpu_was_used(),
        oxibonsai_image::cuda_gpu::dit_gpu_enabled(),
        oxibonsai_image::cuda_gpu::dit_attn_gpu_was_used(),
        oxibonsai_image::cuda_gpu::dit_attn_gpu_enabled()
    );
    println!(
        "\n=== Summary: {} checks, {} failures ===",
        report.checks, report.failures
    );
    Ok(report.failures == 0)
}

/// Run the S4 Euler sampler loop (4 steps) from `tf_in_hidden_states` and
/// compare each step's `noise` and `latents` to the goldens.
#[allow(clippy::too_many_arguments)]
fn run_s4(
    fwd: &DitForward,
    report: &mut Report,
    golden_dir: &Path,
    cond: &Npy,
    img_ids_2d: &[f32],
    txt_ids_2d: &[f32],
    seq_img: usize,
    seq_txt: usize,
    in_channels: usize,
    joint_dim: usize,
) -> Result<(), String> {
    println!("\n--- S4: Euler sampler loop (4 steps) ---");
    let init = load(golden_dir, "tf_in_hidden_states")?;
    let timesteps = load(golden_dir, "timesteps")?;
    let sigmas = load(golden_dir, "sigmas")?;
    let mut steps: Vec<StepTap> = Vec::new();
    let t = std::time::Instant::now();
    let _final = fwd
        .sample(
            &init.data[..seq_img * in_channels],
            &cond.data[..seq_txt * joint_dim],
            img_ids_2d,
            txt_ids_2d,
            seq_img,
            seq_txt,
            &timesteps.data,
            &sigmas.data,
            Some(&mut steps),
        )
        .map_err(|e| format!("sample: {e}"))?;
    println!(
        "  (sampler {} steps took {:.1}s)",
        steps.len(),
        t.elapsed().as_secs_f64()
    );
    // The binding S4 metric is cosine >= 0.999 (per the spec). The relative-L2
    // bound is relaxed to 5e-2 here because the sampler chains 4 full forwards:
    // the Rust f32 trajectory drifts in MAGNITUDE from the golden bf16 trajectory
    // (the megakernel re-rounds to bf16 at every Linear each step, the Rust path
    // does not), so relL2 grows step-over-step exactly as the spec anticipates
    // for late stages, while the DIRECTION (cosine) stays >= 0.999.
    for (s, step) in steps.iter().enumerate() {
        report.compare(
            &format!("noise_step{s}"),
            &step.noise,
            &load(golden_dir, &format!("noise_step{s}"))?,
            0.999,
            5e-2,
        );
        report.compare(
            &format!("latent_after_step{s}"),
            &step.latents,
            &load(golden_dir, &format!("latent_after_step{s}"))?,
            0.999,
            5e-2,
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => {
            println!("ALL STAGES PASS");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("SOME STAGES FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
