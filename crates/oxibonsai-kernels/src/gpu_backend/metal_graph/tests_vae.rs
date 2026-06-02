//! FLUX.2 VAE-decoder per-op f32 primitive parity tests.
//!
//! Self-contained CPU ports of `oxibonsai-image::vae` validate the GPU
//! conv2d (k1 / k3-im2col / k3-implicit), GroupNorm, SiLU, and nearest
//! upsample primitives. Split out of `tests.rs`.

use super::graph::MetalGraph;

// ═══════════════════════════════════════════════════════════════════════════
// FLUX.2 VAE decoder per-op f32 primitives — parity vs CPU reference
// ═══════════════════════════════════════════════════════════════════════════
//
// Self-contained CPU ports of `oxibonsai-image::vae` (the kernels crate must
// not depend on the image crate). Each GPU primitive is validated against its
// port; the parity bound is f32-scale (max-abs / relL2 ≈ 1e-4) per the spec.

/// Deterministic bounded f32 fill (`tanh`-ish), seeded by `seed`.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn vae_fill(n: usize, seed: u32) -> Vec<f32> {
    let mut v = vec![0f32; n];
    let mut lcg: u32 = 0x9E37_79B9 ^ seed.wrapping_mul(2_654_435_761);
    for slot in v.iter_mut() {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *slot = ((lcg >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    v
}

/// CPU im2col, mirroring `oxibonsai-image::vae::conv::build_im2col` exactly:
/// `patches[(oh*w_out+ow)*patch_dim + (kh*k+kw)*c_in + ci]` in `(kh,kw,ci)`
/// order, zero-padded.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn cpu_build_im2col(
    input: &[f32],
    patches: &mut [f32],
    in_ch: usize,
    h: usize,
    w: usize,
    k: usize,
    pad: usize,
    h_out: usize,
    w_out: usize,
) {
    let patch_dim = k * k * in_ch;
    let hw_plane = h * w;
    for oh in 0..h_out {
        for ow in 0..w_out {
            let row =
                &mut patches[(oh * w_out + ow) * patch_dim..(oh * w_out + ow + 1) * patch_dim];
            for kh in 0..k {
                let ih = oh + kh;
                if ih < pad || ih >= h + pad {
                    continue;
                }
                let ih = ih - pad;
                for kw in 0..k {
                    let iw = ow + kw;
                    if iw < pad || iw >= w + pad {
                        continue;
                    }
                    let iw = iw - pad;
                    let dst_base = (kh * k + kw) * in_ch;
                    let src_base = ih * w + iw;
                    for ci in 0..in_ch {
                        row[dst_base + ci] = input[ci * hw_plane + src_base];
                    }
                }
            }
        }
    }
}

/// Full CPU Conv2d reference (im2col + naive GEMM + transpose + bias), mirroring
/// `oxibonsai-image::vae::conv::Conv2d::forward`. Returns NCHW `[c_out, h_out,
/// w_out]`.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn cpu_conv2d(
    input: &[f32],
    weight: &[f32], // [c_out, k*k*c_in] row-major (MLX [c_out,kH,kW,c_in] flattened)
    bias: &[f32],   // [c_out]
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = h + 2 * pad + 1 - k;
    let w_out = w + 2 * pad + 1 - k;
    let patch_dim = k * k * c_in;
    let spatial = h_out * w_out;
    let mut patches = vec![0f32; spatial * patch_dim];
    cpu_build_im2col(input, &mut patches, c_in, h, w, k, pad, h_out, w_out);
    let mut out = vec![0f32; c_out * spatial];
    for oc in 0..c_out {
        let w_row = &weight[oc * patch_dim..(oc + 1) * patch_dim];
        let b = bias[oc];
        for hw in 0..spatial {
            let p_row = &patches[hw * patch_dim..(hw + 1) * patch_dim];
            let mut acc = 0f32;
            for kk in 0..patch_dim {
                acc += p_row[kk] * w_row[kk];
            }
            out[oc * spatial + hw] = acc + b;
        }
    }
    (out, h_out, w_out)
}

/// CPU GroupNorm reference, mirroring
/// `oxibonsai-image::vae::norm::GroupNorm::forward_inplace` (f64 accumulation).
#[cfg(all(feature = "metal", target_os = "macos"))]
fn cpu_groupnorm(
    x: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    hw: usize,
    num_groups: usize,
    eps: f32,
) {
    let gs = channels / num_groups;
    let group_elems = gs * hw;
    let inv_n = 1.0f64 / group_elems as f64;
    for g in 0..num_groups {
        let c0 = g * gs;
        let base = c0 * hw;
        let group = &mut x[base..base + group_elems];
        let mut mean = 0.0f64;
        for &v in group.iter() {
            mean += v as f64;
        }
        mean *= inv_n;
        let mut var = 0.0f64;
        for &v in group.iter() {
            let d = v as f64 - mean;
            var += d * d;
        }
        var *= inv_n;
        let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
        let mean_f = mean as f32;
        for ci in 0..gs {
            let c = c0 + ci;
            let wgt = weight[c];
            let bia = bias[c];
            let chan = &mut group[ci * hw..(ci + 1) * hw];
            for v in chan.iter_mut() {
                *v = (*v - mean_f) * inv_std * wgt + bia;
            }
        }
    }
}

/// Max-abs and relative-L2 error between two equal-length slices.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn err_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut num = 0f64;
    let mut den = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let e = (x - y).abs();
        if e > max_abs {
            max_abs = e;
        }
        num += (e as f64) * (e as f64);
        den += (x as f64) * (x as f64);
    }
    let rel_l2 = if den > 0.0 {
        (num.sqrt() / den.sqrt()) as f32
    } else {
        num.sqrt() as f32
    };
    (max_abs, rel_l2)
}

/// Parity: `encode_conv2d_f32` k=1 (pure channel-mix) vs CPU Conv2d.
///
/// Sweeps several `(C_in, C_out, H, W)` including odd channels and a 512×512
/// plane. Bound: max-abs < 1e-3, relL2 < 1e-4.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_conv2d_f32_k1_matches_cpu() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping conv2d k=1 parity");
            return;
        }
    };
    // (c_in, c_out, h, w)
    let cases = [
        (3usize, 8usize, 5usize, 7usize),
        (128, 32, 32, 32),
        (35, 17, 40, 40), // odd channels, non-tile-multiple plane
        (192, 96, 64, 64),
        (16, 3, 512, 512), // large plane (conv_out-like)
    ];
    let key_base: u64 = 8_100_000;
    let mut worst_max = 0f32;
    let mut worst_rel = 0f32;
    for (idx, &(c_in, c_out, h, w)) in cases.iter().enumerate() {
        let input = vae_fill(c_in * h * w, c_in as u32 * 31 + h as u32);
        let weight = vae_fill(c_out * c_in, c_out as u32 * 17 + c_in as u32); // [c_out, c_in]
        let bias = vae_fill(c_out, c_out as u32 * 7 + 3);
        let handle = graph
            .get_or_upload_f32_weight(key_base + idx as u64, &weight)
            .expect("upload conv k1 weight");

        let mut got = vec![0f32; c_out * h * w];
        graph
            .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 1, 0)
            .expect("encode_conv2d_f32 k1");

        let (expected, ho, wo) = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 1, 0);
        assert_eq!((ho, wo), (h, w));
        let (max_abs, rel_l2) = err_stats(&expected, &got);
        assert!(
            max_abs < 1e-3 && rel_l2 < 1e-4,
            "conv k1 C_in={c_in} C_out={c_out} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
        worst_max = worst_max.max(max_abs);
        worst_rel = worst_rel.max(rel_l2);
        eprintln!(
            "conv2d k1 C_in={c_in:>4} C_out={c_out:>4} {h}x{w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
    }
    eprintln!("conv2d k1 WORST max_abs={worst_max:e} relL2={worst_rel:e}");
}

/// Parity: `encode_conv2d_f32` k=3 (GPU im2col + GEMM, tiled) vs CPU Conv2d.
///
/// Sweeps several `(C_in, C_out, H, W)` including odd channels, a non-tile
/// plane (H=W=40), and a 512×512 plane (up2/up3-style). Bound: max-abs < 1e-3,
/// relL2 < 1e-4.
#[test]
#[ignore = "heavy: Metal conv2d k=3 parity sweep (512×512 planes, 192ch) — ~15 min on debug build; run explicitly with --run-ignored"]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_conv2d_f32_k3_matches_cpu() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping conv2d k=3 parity");
            return;
        }
    };
    // (c_in, c_out, h, w)
    let cases = [
        (4usize, 6usize, 6usize, 8usize),
        (35, 17, 40, 40),     // odd channels, non-tile-multiple plane
        (192, 96, 64, 64),    // mid-ish
        (96, 96, 128, 128),   // up0/up1-style
        (192, 192, 512, 512), // up2/up3-style large plane (multi-tile)
    ];
    let key_base: u64 = 8_200_000;
    let mut worst_max = 0f32;
    let mut worst_rel = 0f32;
    for (idx, &(c_in, c_out, h, w)) in cases.iter().enumerate() {
        let patch_dim = 9 * c_in;
        let input = vae_fill(c_in * h * w, c_in as u32 * 53 + w as u32);
        let weight = vae_fill(c_out * patch_dim, c_out as u32 * 41 + c_in as u32); // [c_out, 3*3*c_in]
        let bias = vae_fill(c_out, c_out as u32 * 11 + 5);
        let handle = graph
            .get_or_upload_f32_weight(key_base + idx as u64, &weight)
            .expect("upload conv k3 weight");

        let mut got = vec![0f32; c_out * h * w];
        graph
            .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 3, 1)
            .expect("encode_conv2d_f32 k3");

        let (expected, ho, wo) = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 3, 1);
        assert_eq!((ho, wo), (h, w)); // same padding
        let (max_abs, rel_l2) = err_stats(&expected, &got);
        assert!(
            max_abs < 1e-3 && rel_l2 < 1e-4,
            "conv k3 C_in={c_in} C_out={c_out} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
        worst_max = worst_max.max(max_abs);
        worst_rel = worst_rel.max(rel_l2);
        eprintln!(
            "conv2d k3 C_in={c_in:>4} C_out={c_out:>4} {h}x{w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
    }
    eprintln!("conv2d k3 WORST max_abs={worst_max:e} relL2={worst_rel:e}");
}

/// Parity: `encode_groupnorm_f32` (Kahan-f32 reduction) vs CPU GroupNorm (f64).
///
/// 32 groups, eps 1e-6, per-channel affine; sweeps several `(C, H, W)` incl. a
/// 512×512 plane (the most reduction elements). Bound: max-abs < 1e-3,
/// relL2 < 1e-4.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_groupnorm_f32_matches_cpu() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping groupnorm parity");
            return;
        }
    };
    let eps = 1e-6f32;
    let ng = 32usize;
    // (channels, h, w) — channels divisible by 32.
    let cases = [
        (32usize, 8usize, 8usize),
        (384, 64, 64),
        (192, 128, 128),
        (96, 512, 512), // gs=3, 786432 elems/group — worst-case reduction
        (384, 32, 32),
    ];
    let mut worst_max = 0f32;
    let mut worst_rel = 0f32;
    for &(channels, h, w) in &cases {
        let hw = h * w;
        // Use a slightly wider distribution so var is non-trivial.
        let base = vae_fill(channels * hw, channels as u32 * 23 + h as u32);
        let scaled: Vec<f32> = base.iter().map(|v| v * 4.0 + 0.3).collect();
        let weight = vae_fill(channels, channels as u32 * 13 + 1)
            .iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>();
        let bias = vae_fill(channels, channels as u32 * 29 + 9);

        let mut got = scaled.clone();
        graph
            .encode_groupnorm_f32(&mut got, &weight, &bias, channels, hw, ng, eps)
            .expect("encode_groupnorm_f32");

        let mut expected = scaled.clone();
        cpu_groupnorm(&mut expected, &weight, &bias, channels, hw, ng, eps);

        let (max_abs, rel_l2) = err_stats(&expected, &got);
        assert!(
            max_abs < 1e-3 && rel_l2 < 1e-4,
            "groupnorm C={channels} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
        worst_max = worst_max.max(max_abs);
        worst_rel = worst_rel.max(rel_l2);
        eprintln!(
            "groupnorm C={channels:>4} {h}x{w} (gs={}): max_abs={max_abs:e} relL2={rel_l2:e}",
            channels / ng
        );
    }
    eprintln!("groupnorm WORST max_abs={worst_max:e} relL2={worst_rel:e}");
}

/// Parity: `encode_silu_f32` vs CPU `x / (1 + exp(-x))`.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_silu_f32_matches_cpu() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping silu parity");
            return;
        }
    };
    for &n in &[1usize, 255, 256, 257, 4096, 384 * 512 * 512] {
        let base = vae_fill(n, n as u32 * 3 + 1);
        // Spread across a meaningful SiLU range.
        let x: Vec<f32> = base.iter().map(|v| v * 16.0).collect();
        let mut got = x.clone();
        graph.encode_silu_f32(&mut got).expect("encode_silu_f32");
        let expected: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let (max_abs, rel_l2) = err_stats(&expected, &got);
        assert!(
            max_abs < 1e-3 && rel_l2 < 1e-4,
            "silu n={n}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
        eprintln!("silu n={n:>9}: max_abs={max_abs:e} relL2={rel_l2:e}");
    }
}

/// Parity: `encode_upsample_nearest_f32` (`[C,H,W] → [C,2H,2W]`) vs CPU.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_upsample_nearest_f32_matches_cpu() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping upsample parity");
            return;
        }
    };
    // (c, h, w)
    let cases = [
        (1usize, 1usize, 2usize),
        (35, 40, 40), // odd channels, non-tile plane
        (384, 64, 64),
        (192, 128, 128),
        (96, 256, 256),
    ];
    for &(c, h, w) in &cases {
        let input = vae_fill(c * h * w, c as u32 * 19 + h as u32);
        let mut got = vec![0f32; c * 4 * h * w];
        graph
            .encode_upsample_nearest_f32(&input, &mut got, c, h, w)
            .expect("encode_upsample_nearest_f32");

        // CPU reference (ops::upsample_nearest2x).
        let h_out = h * 2;
        let w_out = w * 2;
        let hw = h * w;
        let mut expected = vec![0f32; c * h_out * w_out];
        for ci in 0..c {
            let src = &input[ci * hw..(ci + 1) * hw];
            let dst = &mut expected[ci * h_out * w_out..(ci + 1) * h_out * w_out];
            for ho in 0..h_out {
                let hh = ho / 2;
                for wo in 0..w_out {
                    let ww = wo / 2;
                    dst[ho * w_out + wo] = src[hh * w + ww];
                }
            }
        }
        let (max_abs, _rel) = err_stats(&expected, &got);
        // Exact gather — must be bit-identical.
        assert!(
            max_abs == 0.0,
            "upsample C={c} {h}x{w}: max_abs={max_abs:e} (expected exact)"
        );
        eprintln!("upsample C={c:>4} {h}x{w}: max_abs={max_abs:e} (exact)");
    }
}

/// Speed: GPU `encode_conv2d_f32` k=3 vs CPU im2col+GEMM on an up2/up3-style
/// 512×512 conv (`C_in ≈ 192`). Reports the warm median ratio (gate ≥ 9×).
///
/// The CPU reference here is the SAME math the image crate runs (im2col +
/// per-(oc,hw) dot), so the ratio reflects the real VAE-conv win. Load varies;
/// the ratio is load-robust (both paths see the same machine state).
#[test]
#[ignore = "heavy: Metal conv2d implicit-GEMM speed-ratio sweep — ~15 min on debug build; run explicitly with --run-ignored"]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_conv2d_f32_k3_speed_ratio() {
    use std::time::Instant;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping conv2d k=3 speed ratio");
            return;
        }
    };

    let (c_in, c_out, h, w) = (192usize, 192usize, 512usize, 512usize);
    let patch_dim = 9 * c_in;
    let input = vae_fill(c_in * h * w, 12345);
    let weight = vae_fill(c_out * patch_dim, 6789);
    let bias = vae_fill(c_out, 111);
    let handle = graph
        .get_or_upload_f32_weight(8_300_000, &weight)
        .expect("upload speed weight");

    let warmup = 1usize;
    let iters = 3usize;

    // GPU timing.
    let mut gpu_ms = Vec::with_capacity(iters);
    let mut got = vec![0f32; c_out * h * w];
    for _ in 0..warmup {
        graph
            .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 3, 1)
            .expect("gpu conv warmup");
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        graph
            .encode_conv2d_f32(&handle, &input, &bias, &mut got, c_in, c_out, h, w, 3, 1)
            .expect("gpu conv");
        gpu_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    // CPU timing (same math as oxibonsai-image::vae::conv).
    let mut cpu_ms = Vec::with_capacity(iters);
    for _ in 0..warmup {
        let _ = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 3, 1);
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, 3, 1);
        cpu_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    let gpu_med = median(gpu_ms);
    let cpu_med = median(cpu_ms);
    let ratio = cpu_med / gpu_med;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!(
        "CONV-SPEED k3 C_in={c_in} C_out={c_out} {h}x{w}: GPU={gpu_med:.2}ms CPU={cpu_med:.2}ms \
         CPU/GPU={ratio:.2}x (warmup {warmup}, median of {iters}){}",
        if load.is_empty() {
            String::new()
        } else {
            format!(" loadavg={}", load.trim())
        }
    );
    // Informational gate: the GPU conv should clearly beat the CPU im2col+GEMM.
    assert!(
        ratio >= 9.0,
        "conv2d k3 512x512 GPU/CPU ratio {ratio:.2}x < 9x target"
    );
}

/// Parity: the **im2col-free implicit-GEMM** conv kernel
/// (`encode_conv2d_f32_implicit`, k=3 pad=1) vs the CPU Conv2d reference.
///
/// Drives the implicit kernel DIRECTLY (not via `encode_conv2d_f32`'s routing, so
/// this is a true unit test of the new kernel — no chance of a silent im2col
/// fallback). Sweeps `C_in/C_out ∈ {96,192,384}` × `H=W ∈ {64,128,256,512}` plus
/// a non-tile-multiple plane with odd channels (`H=W=40`, `C_in=35`, `C_out=17`).
/// Bound: max-abs < 1e-3, relL2 ≲ 1e-4 (f32 reassociation only).
#[test]
#[ignore = "heavy: Metal conv2d implicit-GEMM full parity reference — ~15 min on debug build; run explicitly with --run-ignored"]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_conv2d_f32_implicit_matches_reference() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping conv2d implicit parity");
            return;
        }
    };
    // (c_in, c_out, h, w). VAE-representative channels {96,192,384} crossed with
    // spatial {64,128,256,512}, plus the odd / non-tile-multiple corner case.
    let cases = [
        (96usize, 96usize, 64usize, 64usize),
        (96, 192, 128, 128),
        (192, 192, 128, 128),
        (192, 384, 64, 64),
        (384, 384, 64, 64),
        (192, 192, 256, 256),
        (96, 96, 512, 512),
        (192, 192, 512, 512),
        (35, 17, 40, 40), // odd channels, non-tile-multiple plane
    ];
    let key_base: u64 = 8_400_000;
    let mut worst_max = 0f32;
    let mut worst_rel = 0f32;
    for (idx, &(c_in, c_out, h, w)) in cases.iter().enumerate() {
        let (k, pad) = (3usize, 1usize);
        let patch_dim = k * k * c_in;
        let spatial = h * w; // h_out=h, w_out=w for k=3 pad=1 stride=1
        let w_out = w;
        let input = vae_fill(c_in * h * w, c_in as u32 * 53 + w as u32 + idx as u32);
        let weight = vae_fill(c_out * patch_dim, c_out as u32 * 41 + c_in as u32); // [c_out, 3*3*c_in]
        let bias = vae_fill(c_out, c_out as u32 * 11 + 5);
        let handle = graph
            .get_or_upload_f32_weight(key_base + idx as u64, &weight)
            .expect("upload conv implicit weight");

        let mut got = vec![0f32; c_out * spatial];
        graph
            .encode_conv2d_f32_implicit(
                &handle, &input, &bias, &mut got, c_in, c_out, h, w, k, pad, spatial, patch_dim,
                w_out,
            )
            .expect("encode_conv2d_f32_implicit");

        let (expected, ho, wo) = cpu_conv2d(&input, &weight, &bias, c_in, c_out, h, w, k, pad);
        assert_eq!((ho, wo), (h, w));
        let (max_abs, rel_l2) = err_stats(&expected, &got);
        assert!(
            max_abs < 1e-3,
            "conv implicit C_in={c_in} C_out={c_out} H={h} W={w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
        worst_max = worst_max.max(max_abs);
        worst_rel = worst_rel.max(rel_l2);
        eprintln!(
            "conv2d implicit C_in={c_in:>4} C_out={c_out:>4} {h}x{w}: max_abs={max_abs:e} relL2={rel_l2:e}"
        );
    }
    eprintln!("conv2d implicit WORST max_abs={worst_max:e} relL2={worst_rel:e}");
}

/// Speed: **implicit-GEMM** conv vs the **tiled-im2col** conv on a 512×512 k=3
/// `C_in=C_out=192` conv (the up2/up3 shape). Reports warm-median ms + GFLOP/s
/// for each path (`FLOP ≈ 2·C_out·C_in·K²·H·W`). The implicit path should beat
/// the im2col path (no ~GB global im2col materialization).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_conv2d_f32_implicit_vs_im2col_speed() {
    use std::time::Instant;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping conv2d implicit-vs-im2col speed");
            return;
        }
    };

    let (c_in, c_out, h, w) = (192usize, 192usize, 512usize, 512usize);
    let (k, pad) = (3usize, 1usize);
    let patch_dim = k * k * c_in;
    let spatial = h * w;
    let w_out = w;
    let input = vae_fill(c_in * h * w, 24680);
    let weight = vae_fill(c_out * patch_dim, 13579);
    let bias = vae_fill(c_out, 222);
    let handle = graph
        .get_or_upload_f32_weight(8_500_000, &weight)
        .expect("upload speed weight");

    let warmup = 2usize;
    let iters = 11usize;
    let flop = 2.0 * c_out as f64 * c_in as f64 * (k * k) as f64 * h as f64 * w as f64;

    let mut imp = vec![0f32; c_out * spatial];
    let mut i2c = vec![0f32; c_out * spatial];

    // Parity sanity (the two paths must agree before timing means anything).
    graph
        .encode_conv2d_f32_implicit(
            &handle, &input, &bias, &mut imp, c_in, c_out, h, w, k, pad, spatial, patch_dim, w_out,
        )
        .expect("implicit");
    graph
        .encode_conv2d_f32_im2col(
            &handle, &input, &bias, &mut i2c, c_in, c_out, h, w, k, pad, spatial, patch_dim, w_out,
        )
        .expect("im2col");
    let (max_abs, _rel) = err_stats(&imp, &i2c);
    assert!(
        max_abs < 1e-2,
        "implicit vs im2col disagree: max_abs={max_abs:e}"
    );

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };

    // Implicit timing.
    let mut imp_ms = Vec::with_capacity(iters);
    for _ in 0..warmup {
        graph
            .encode_conv2d_f32_implicit(
                &handle, &input, &bias, &mut imp, c_in, c_out, h, w, k, pad, spatial, patch_dim,
                w_out,
            )
            .expect("implicit warmup");
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        graph
            .encode_conv2d_f32_implicit(
                &handle, &input, &bias, &mut imp, c_in, c_out, h, w, k, pad, spatial, patch_dim,
                w_out,
            )
            .expect("implicit");
        imp_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    // im2col timing.
    let mut i2c_ms = Vec::with_capacity(iters);
    for _ in 0..warmup {
        graph
            .encode_conv2d_f32_im2col(
                &handle, &input, &bias, &mut i2c, c_in, c_out, h, w, k, pad, spatial, patch_dim,
                w_out,
            )
            .expect("im2col warmup");
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        graph
            .encode_conv2d_f32_im2col(
                &handle, &input, &bias, &mut i2c, c_in, c_out, h, w, k, pad, spatial, patch_dim,
                w_out,
            )
            .expect("im2col");
        i2c_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let imp_med = median(imp_ms);
    let i2c_med = median(i2c_ms);
    let imp_gflops = flop / (imp_med * 1e-3) / 1e9;
    let i2c_gflops = flop / (i2c_med * 1e-3) / 1e9;
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!(
        "CONV-IMPL-SPEED k3 C_in={c_in} C_out={c_out} {h}x{w}: \
         implicit={imp_med:.2}ms ({imp_gflops:.0} GFLOP/s)  \
         im2col={i2c_med:.2}ms ({i2c_gflops:.0} GFLOP/s)  \
         speedup={:.2}x (warmup {warmup}, median of {iters}){}",
        i2c_med / imp_med,
        if load.is_empty() {
            String::new()
        } else {
            format!(" loadavg={}", load.trim())
        }
    );
}
