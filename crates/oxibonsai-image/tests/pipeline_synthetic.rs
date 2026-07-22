//! End-to-end **orchestration** coverage for the `oxibonsai-image` pipeline on a
//! clean checkout — no multi-GB golden GGUF, no VAE/TE goldens, no Metal.
//!
//! The headline entry points (`text_to_image`, `DitForward::sample`, the VAE
//! decode + latent packing + PNG stages) were previously exercised only by the
//! 1.3 GB `OXIBONSAI_DIT_GGUF` parity test or macOS/Metal-gated benches, so on a
//! clean CUDA/Linux (or CPU) checkout the whole generation *plumbing* had zero
//! functional coverage. This file closes that gap by building a **complete but
//! tiny** synthetic DiT (1 dual + 1 single block, hidden = 128) as an in-memory
//! GGUF, plus a **complete tiny** SMALL-VAE as a directory of `.npy` weights, and
//! driving the real orchestration end-to-end:
//!
//! * [`text_to_image`] via a golden-cond override (skips only the ~16 GB text
//!   encoder) → a valid PNG of the expected size, and byte-for-byte determinism.
//! * [`DitForward::sample`] with the **native** noise / flow-match schedule /
//!   position-id scaffolding → seed reproducibility and step-count sensitivity.
//! * The VAE decode + `latent_seq_to_packed_nchw` packing + `encode_rgb8` PNG
//!   stages, driven on the synthetic VAE.
//!
//! Everything runs on the CPU (the DiT GPU path is forced off per-test) in well
//! under a second, so it is safe for CI on any machine.

use std::path::{Path, PathBuf};

use half::bf16;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

use oxibonsai_image::pipeline::latent_seq_to_packed_nchw;
use oxibonsai_image::vae::{VaeDecoder, VaeWeights};
use oxibonsai_image::{
    encode_rgb8, sample, text_to_image, DitForward, DitWeights, GoldenOverride, TeSource,
    TextToImageCfg,
};

// ── Tiny synthetic architecture dimensions (all quantized dims 128-divisible) ──
const HIDDEN: usize = 128; // heads * head_dim
const HEAD_DIM: usize = 128;
const IN_CH: usize = 128; // = 4 * VAE latent channels (VAE packed_ch)
const JOINT: usize = 128; // joint_attention_dim
const FFN: usize = 128; // round(hidden * mlp_ratio=1.0)
const SEQ_TXT_PIPELINE: usize = 512; // the fixed pad length text_to_image uses
const NUM_AXES: usize = 4;

/// Unique scratch path under the system temp dir (policy: tests use
/// `std::env::temp_dir()`), tagged by `label` + pid + nanos so parallel runs
/// never collide.
fn scratch(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxibonsai_synth_{label}_{}_{nanos}",
        std::process::id()
    ))
}

/// Force the pure-CPU DiT/VAE path so the orchestration test is deterministic and
/// hermetic regardless of which GPU features the build enables (each nextest test
/// runs in its own process, so these env writes are isolated).
fn force_cpu() {
    std::env::set_var("OXI_DIT_GPU", "0");
    std::env::set_var("OXI_DIT_FUSED", "0");
    std::env::set_var("OXI_VAE_GPU", "0");
    std::env::set_var("OXI_TE_GPU", "0");
}

/// Deterministic small weight values (kept small so ternary quantisation and the
/// dense embedders never blow the activations up over the tiny stack).
fn wvals(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.013 + seed).sin() * scale)
        .collect()
}

/// bf16-encode a logical row-major buffer to little-endian bytes.
fn bf16_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Ternary-quantise a logical row-major `[out, in]` buffer (in must be 128-div)
/// into the on-disk `TQ2_0_g128` byte layout (`qs` then the f16 scale per block).
fn quant_bytes(flat: &[f32]) -> Vec<u8> {
    let blocks = BlockTQ2_0_g128::quantize(flat).expect("quantize synthetic weight");
    let mut bytes = Vec::with_capacity(blocks.len() * 34);
    for b in &blocks {
        bytes.extend_from_slice(&b.qs);
        bytes.extend_from_slice(&b.d.to_le_bytes());
    }
    bytes
}

/// Add a plain bf16 2-D tensor stored under its full `name` (logical `[out, in]`;
/// GGUF stores the shape reversed).
fn add_bf16(w: &mut GgufWriter, name: &str, out: usize, in_: usize, seed: f32, scale: f32) {
    let data = wvals(out * in_, seed, scale);
    w.add_tensor(TensorEntry {
        name: name.to_string(),
        shape: vec![in_ as u64, out as u64],
        tensor_type: TensorType::BF16,
        data: bf16_bytes(&data),
    });
}

/// Add a 1-D bf16 norm vector centred on 1.0 (an RMSNorm weight).
fn add_bf16_norm(w: &mut GgufWriter, name: &str, len: usize, seed: f32) {
    let data: Vec<f32> = (0..len)
        .map(|i| 1.0 + 0.02 * ((i as f32) * 0.05 + seed).sin())
        .collect();
    w.add_tensor(TensorEntry {
        name: name.to_string(),
        shape: vec![len as u64],
        tensor_type: TensorType::BF16,
        data: bf16_bytes(&data),
    });
}

/// Add a ternary-quantized linear stored under its base `name` (no `.weight`;
/// logical `[out, in]`, shape reversed).
fn add_quant(w: &mut GgufWriter, name: &str, out: usize, in_: usize, seed: f32, scale: f32) {
    let data = wvals(out * in_, seed, scale);
    w.add_tensor(TensorEntry {
        name: name.to_string(),
        shape: vec![in_ as u64, out as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: quant_bytes(&data),
    });
}

/// Build a complete, tiny `bonsai-image` DiT GGUF (1 dual + 1 single block) that
/// `DitForward::sample` can run end-to-end.
fn build_dit_gguf() -> Vec<u8> {
    let mut w = GgufWriter::new();

    // ── metadata: hidden = 1*128 = 128, ffn_inner = round(128*1.0) = 128 ──
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("bonsai-image".to_string()),
    );
    w.add_metadata("bonsai-image.num_layers", MetadataWriteValue::U32(1));
    w.add_metadata("bonsai-image.num_single_layers", MetadataWriteValue::U32(1));
    w.add_metadata(
        "bonsai-image.attention.head_count",
        MetadataWriteValue::U32(1),
    );
    w.add_metadata(
        "bonsai-image.attention.head_dim",
        MetadataWriteValue::U32(HEAD_DIM as u32),
    );
    w.add_metadata(
        "bonsai-image.joint_attention_dim",
        MetadataWriteValue::U32(JOINT as u32),
    );
    w.add_metadata(
        "bonsai-image.in_channels",
        MetadataWriteValue::U32(IN_CH as u32),
    );
    w.add_metadata("bonsai-image.mlp_ratio", MetadataWriteValue::F32(1.0));
    w.add_metadata(
        "bonsai-image.rope.axes_dims",
        MetadataWriteValue::ArrayU32(vec![32, 32, 32, 32]),
    );
    w.add_metadata("bonsai-image.rope.theta", MetadataWriteValue::F32(2000.0));
    w.add_metadata(
        "bonsai-image.guidance_embeds",
        MetadataWriteValue::Bool(false),
    );

    // ── top-level bf16 dense ──
    add_bf16(&mut w, "x_embedder.weight", HIDDEN, IN_CH, 0.1, 0.05);
    add_bf16(&mut w, "context_embedder.weight", HIDDEN, JOINT, 0.2, 0.05);
    add_bf16(
        &mut w,
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        HIDDEN,
        256,
        0.3,
        0.02,
    );
    add_bf16(
        &mut w,
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        HIDDEN,
        HIDDEN,
        0.4,
        0.02,
    );
    add_bf16(
        &mut w,
        "double_stream_modulation_img.linear.weight",
        3 * 2 * HIDDEN,
        HIDDEN,
        0.5,
        0.02,
    );
    add_bf16(
        &mut w,
        "double_stream_modulation_txt.linear.weight",
        3 * 2 * HIDDEN,
        HIDDEN,
        0.6,
        0.02,
    );
    add_bf16(
        &mut w,
        "single_stream_modulation.linear.weight",
        3 * HIDDEN,
        HIDDEN,
        0.7,
        0.02,
    );
    add_bf16(
        &mut w,
        "norm_out.linear.weight",
        2 * HIDDEN,
        HIDDEN,
        0.8,
        0.02,
    );
    add_bf16(&mut w, "proj_out.weight", IN_CH, HIDDEN, 0.9, 0.05);

    // ── dual-stream block 0 ──
    let db = "transformer_blocks.0";
    for (name, seed) in [
        (format!("{db}.attn.to_q"), 1.1),
        (format!("{db}.attn.to_k"), 1.2),
        (format!("{db}.attn.to_v"), 1.3),
        (format!("{db}.attn.add_q_proj"), 1.4),
        (format!("{db}.attn.add_k_proj"), 1.5),
        (format!("{db}.attn.add_v_proj"), 1.6),
        (format!("{db}.attn.to_out.0"), 1.7),
        (format!("{db}.attn.to_add_out"), 1.8),
    ] {
        add_quant(&mut w, &name, HIDDEN, HIDDEN, seed, 0.03);
    }
    add_quant(
        &mut w,
        &format!("{db}.ff.linear_in"),
        2 * FFN,
        HIDDEN,
        2.1,
        0.03,
    );
    add_quant(
        &mut w,
        &format!("{db}.ff.linear_out"),
        HIDDEN,
        FFN,
        2.2,
        0.03,
    );
    add_quant(
        &mut w,
        &format!("{db}.ff_context.linear_in"),
        2 * FFN,
        HIDDEN,
        2.3,
        0.03,
    );
    add_quant(
        &mut w,
        &format!("{db}.ff_context.linear_out"),
        HIDDEN,
        FFN,
        2.4,
        0.03,
    );
    for (name, seed) in [
        (format!("{db}.attn.norm_q.weight"), 3.1),
        (format!("{db}.attn.norm_k.weight"), 3.2),
        (format!("{db}.attn.norm_added_q.weight"), 3.3),
        (format!("{db}.attn.norm_added_k.weight"), 3.4),
    ] {
        add_bf16_norm(&mut w, &name, HEAD_DIM, seed);
    }

    // ── single-stream block 0 ──
    let sb = "single_transformer_blocks.0";
    // to_qkv_mlp_proj: [3*hidden + 2*ffn_inner, hidden]; to_out: [hidden, hidden+ffn_inner].
    add_quant(
        &mut w,
        &format!("{sb}.attn.to_qkv_mlp_proj"),
        3 * HIDDEN + 2 * FFN,
        HIDDEN,
        4.1,
        0.03,
    );
    add_quant(
        &mut w,
        &format!("{sb}.attn.to_out"),
        HIDDEN,
        HIDDEN + FFN,
        4.2,
        0.03,
    );
    add_bf16_norm(&mut w, &format!("{sb}.attn.norm_q.weight"), HEAD_DIM, 5.1);
    add_bf16_norm(&mut w, &format!("{sb}.attn.norm_k.weight"), HEAD_DIM, 5.2);

    w.to_bytes().expect("serialise synthetic DiT gguf")
}

// ── synthetic SMALL VAE (all channels = 32, so GroupNorm(32) is 1 ch/group) ──
const VAE_CH: usize = 32;

/// Minimal NumPy `.npy` writer (f32, C-order, v1.0 header), matching what the VAE
/// `.npy` loader ([`VaeWeights::open`] on a directory) parses.
fn write_npy(path: &Path, shape: &[usize], data: &[f32]) {
    let shape_body = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_tuple = if shape.len() == 1 {
        format!("({shape_body},)")
    } else {
        format!("({shape_body})")
    };
    let base = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
    // Pad so that 10 (preamble) + header_len is a multiple of 64, header ends '\n'.
    let pad = (64 - ((10 + base.len() + 1) % 64)) % 64;
    let mut header = base;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + data.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write npy");
}

/// Write `<name>.weight` [out, k, k, in] (MLX conv layout) + `<name>.bias` [out].
fn vconv(dir: &Path, name: &str, out: usize, in_: usize, k: usize, seed: f32) {
    write_npy(
        &dir.join(format!("{name}.weight.npy")),
        &[out, k, k, in_],
        &wvals(out * k * k * in_, seed, 0.08),
    );
    write_npy(
        &dir.join(format!("{name}.bias.npy")),
        &[out],
        &wvals(out, seed + 0.5, 0.02),
    );
}

/// Write a GroupNorm `<name>.weight`/`.bias` [c] (weight ~1.0).
fn vgn(dir: &Path, name: &str, c: usize, seed: f32) {
    let weight: Vec<f32> = (0..c)
        .map(|i| 1.0 + 0.02 * ((i as f32) + seed).sin())
        .collect();
    write_npy(&dir.join(format!("{name}.weight.npy")), &[c], &weight);
    write_npy(
        &dir.join(format!("{name}.bias.npy")),
        &[c],
        &wvals(c, seed + 0.3, 0.02),
    );
}

/// Write a Linear `<name>.weight` [out, in] + `<name>.bias` [out].
fn vlinear(dir: &Path, name: &str, out: usize, in_: usize, seed: f32) {
    write_npy(
        &dir.join(format!("{name}.weight.npy")),
        &[out, in_],
        &wvals(out * in_, seed, 0.08),
    );
    write_npy(
        &dir.join(format!("{name}.bias.npy")),
        &[out],
        &wvals(out, seed + 0.5, 0.02),
    );
}

/// Write a same-channel resnet block (no shortcut) under `prefix`.
fn vresnet(dir: &Path, prefix: &str, ch: usize, seed: f32) {
    vgn(dir, &format!("{prefix}.norm1"), ch, seed);
    vconv(dir, &format!("{prefix}.conv1"), ch, ch, 3, seed + 1.0);
    vgn(dir, &format!("{prefix}.norm2"), ch, seed + 2.0);
    vconv(dir, &format!("{prefix}.conv2"), ch, ch, 3, seed + 3.0);
}

/// Build a complete tiny SMALL-VAE `.npy` directory (fixed 4-up-block
/// architecture; every channel = 32) that `VaeDecoder::from_weights` can load and
/// `decode_packed_latents` can run.
fn build_vae_dir(dir: &Path) {
    std::fs::create_dir_all(dir).expect("mkdir vae");
    let c = VAE_CH;

    // BatchNorm-stats denorm over the 128 packed channels (var > 0).
    write_npy(
        &dir.join("bn.running_mean.npy"),
        &[4 * c],
        &vec![0.0f32; 4 * c],
    );
    write_npy(
        &dir.join("bn.running_var.npy"),
        &[4 * c],
        &vec![1.0f32; 4 * c],
    );

    // post_quant_conv (k=1), conv_in (k=3).
    vconv(dir, "post_quant_conv", c, c, 1, 0.1);
    vconv(dir, "decoder.conv_in", c, c, 3, 0.2);

    // mid block: resnet, attention, resnet.
    vresnet(dir, "decoder.mid_block.resnets.0", c, 0.3);
    vresnet(dir, "decoder.mid_block.resnets.1", c, 0.4);
    let attn = "decoder.mid_block.attentions.0";
    vgn(dir, &format!("{attn}.group_norm"), c, 0.5);
    vlinear(dir, &format!("{attn}.to_q"), c, c, 0.6);
    vlinear(dir, &format!("{attn}.to_k"), c, c, 0.7);
    vlinear(dir, &format!("{attn}.to_v"), c, c, 0.8);
    vlinear(dir, &format!("{attn}.to_out"), c, c, 0.9);

    // 4 up blocks (blocks 0,1,2 have an upsampler; block 3 does not).
    for b in 0..4 {
        let up = format!("decoder.up_blocks.{b}");
        for r in 0..3 {
            vresnet(
                dir,
                &format!("{up}.resnets.{r}"),
                c,
                1.0 + b as f32 + 0.1 * r as f32,
            );
        }
        if b < 3 {
            vconv(
                dir,
                &format!("{up}.upsamplers.0.conv"),
                c,
                c,
                3,
                2.0 + b as f32,
            );
        }
    }

    // Output stage.
    vgn(dir, "decoder.conv_norm_out", c, 6.0);
    vconv(dir, "decoder.conv_out", 3, c, 3, 6.5);
}

/// Write a golden `.npy` tensor.
fn write_golden(dir: &Path, name: &str, shape: &[usize], data: &[f32]) {
    write_npy(&dir.join(format!("{name}.npy")), shape, data);
}

/// **Headline API** end-to-end: drive `text_to_image` with a golden conditioning
/// override (so the ~16 GB text encoder is skipped) but the real synthetic DiT +
/// VAE, and assert it produces a valid PNG of the requested size — and that two
/// identical calls are byte-for-byte deterministic.
#[test]
fn text_to_image_end_to_end_via_golden_cond() {
    force_cpu();

    // Geometry for a 32×32 image: latent grid 2×2, seq_img = 4.
    let (width, height) = (32usize, 32usize);
    let (lat_h, lat_w) = sample::latent_grid(height, width);
    let seq_img = lat_h * lat_w;
    assert_eq!((lat_h, lat_w, seq_img), (2, 2, 4));

    // Synthetic DiT GGUF file.
    let gguf = build_dit_gguf();
    let dit_path = scratch("t2i").with_extension("gguf");
    std::fs::write(&dit_path, &gguf).expect("write dit gguf");

    // Synthetic VAE weights dir.
    let vae_dir = scratch("t2i_vae");
    build_vae_dir(&vae_dir);

    // Golden dir: cond + init latent + ids + schedule (skips TE + native scaffolding).
    let golden_dir = scratch("t2i_golden");
    std::fs::create_dir_all(&golden_dir).expect("mkdir golden");
    write_golden(
        &golden_dir,
        "cond",
        &[SEQ_TXT_PIPELINE, JOINT],
        &wvals(SEQ_TXT_PIPELINE * JOINT, 0.05, 0.4),
    );
    write_golden(
        &golden_dir,
        "tf_in_hidden_states",
        &[seq_img, IN_CH],
        &sample::create_noise(7, height, width),
    );
    write_golden(
        &golden_dir,
        "img_ids",
        &[seq_img, NUM_AXES],
        &sample::img_ids(lat_h, lat_w),
    );
    write_golden(
        &golden_dir,
        "txt_ids",
        &[SEQ_TXT_PIPELINE, NUM_AXES],
        &sample::txt_ids(SEQ_TXT_PIPELINE),
    );
    let (timesteps, sigmas) = sample::flow_match_schedule(seq_img, 2);
    write_golden(&golden_dir, "timesteps", &[timesteps.len()], &timesteps);
    write_golden(&golden_dir, "sigmas", &[sigmas.len()], &sigmas);

    let cfg = TextToImageCfg {
        prompt: "unused (golden cond)".to_string(),
        seed: 7,
        steps: 2,
        width,
        height,
        guidance: 1.0,
        dit_gguf: dit_path.clone(),
        vae_weights_dir: vae_dir.clone(),
        // te_source / tokenizer are never touched with use_golden_cond = true.
        te_source: TeSource::NpyDir(scratch("unused_te")),
        tokenizer_dir: scratch("unused_tok"),
        golden_override: Some(GoldenOverride {
            golden_dir: golden_dir.clone(),
            use_golden_cond: true,
            use_golden_latent: false,
            vae_golden_dir: None,
        }),
    };

    let out = text_to_image(&cfg).expect("text_to_image end-to-end");
    assert_eq!((out.width, out.height), (32, 32), "output image dimensions");
    assert!(
        out.png
            .starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']),
        "output must be a valid PNG stream"
    );
    assert!(out.png.len() > 8, "PNG must have content beyond the header");

    // Determinism: an identical config reproduces byte-for-byte.
    let out2 = text_to_image(&cfg).expect("text_to_image (repeat)");
    assert_eq!(
        out.png, out2.png,
        "identical config must produce byte-identical PNGs"
    );

    // Cleanup.
    let _ = std::fs::remove_file(&dit_path);
    let _ = std::fs::remove_dir_all(&vae_dir);
    let _ = std::fs::remove_dir_all(&golden_dir);
}

/// Drive `DitForward::sample` over the **native** sampling scaffolding
/// (`create_noise` / `flow_match_schedule` / `img_ids` / `txt_ids`) and assert:
/// the sampler is seed-reproducible, seed-sensitive, and step-count-sensitive —
/// i.e. the scheduler actually drives the Euler updates.
#[test]
fn dit_sample_seed_reproducibility_and_scheduler() {
    force_cpu();

    let (width, height) = (32usize, 32usize);
    let (lat_h, lat_w) = sample::latent_grid(height, width);
    let seq_img = lat_h * lat_w;
    let seq_txt = 4usize; // tiny text stream (we feed a synthetic cond directly)

    let weights = DitWeights::from_bytes(build_dit_gguf()).expect("load synthetic DiT");
    let fwd = DitForward::new(&weights);

    let cond = wvals(seq_txt * JOINT, 0.05, 0.4);
    let img_ids = sample::img_ids(lat_h, lat_w);
    let txt_ids = sample::txt_ids(seq_txt);

    let run = |seed: u64, steps: usize| -> Vec<f32> {
        let init = sample::create_noise(seed, height, width);
        assert_eq!(init.len(), seq_img * IN_CH, "native noise shape");
        let (timesteps, sigmas) = sample::flow_match_schedule(seq_img, steps);
        fwd.sample(
            &init, &cond, &img_ids, &txt_ids, seq_img, seq_txt, &timesteps, &sigmas, None,
        )
        .expect("dit sample")
    };

    let a = run(42, 2);
    assert_eq!(a.len(), seq_img * IN_CH, "sampled latent shape");
    assert!(
        a.iter().all(|v| v.is_finite()),
        "sampled latent must be finite"
    );

    // Reproducibility: same seed + steps → bit-identical.
    let a_again = run(42, 2);
    assert_eq!(a, a_again, "same seed must reproduce the latent exactly");

    // Seed sensitivity: a different seed changes the result.
    let b = run(43, 2);
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| x != y),
        "a different seed must change the sampled latent"
    );

    // Scheduler drives the update: a different step count changes the result.
    let a_1step = run(42, 1);
    assert!(
        a.iter().zip(a_1step.iter()).any(|(x, y)| x != y),
        "a different step count must change the sampled latent"
    );
}

/// Drive the VAE decode + latent packing + `encode_rgb8` PNG stages (the tail of
/// the pipeline) directly on the synthetic VAE, independent of the DiT — proving
/// the decode/pack/PNG plumbing on a clean checkout.
#[test]
fn vae_decode_pack_and_png_stage() {
    force_cpu();

    let (width, height) = (32usize, 32usize);
    let (lat_h, lat_w) = sample::latent_grid(height, width);
    let seq_img = lat_h * lat_w;

    let vae_dir = scratch("vae_stage");
    build_vae_dir(&vae_dir);
    let vae_weights = VaeWeights::open(&vae_dir).expect("open synthetic VAE");
    let vae = VaeDecoder::from_weights(&vae_weights).expect("build VAE decoder");

    // A DiT-shaped latent [seq_img, in_channels] → packed NCHW [128, ph, pw].
    let latent = wvals(seq_img * IN_CH, 0.02, 0.5);
    let packed = latent_seq_to_packed_nchw(&latent, seq_img, IN_CH, lat_h, lat_w)
        .expect("pack latent to NCHW");
    assert_eq!(packed.len(), IN_CH * lat_h * lat_w);

    let decoded = vae
        .decode_packed_latents(&packed, lat_h, lat_w, None)
        .expect("VAE decode");
    assert_eq!(decoded.c, 3, "RGB output channels");
    assert_eq!((decoded.h, decoded.w), (32, 32), "decoded image dims");
    assert!(
        decoded.data.iter().all(|v| v.is_finite()),
        "decoded pixels must be finite"
    );

    // clip(x/2 + 0.5) → u8, CHW → HWC (mirrors the pipeline's pixel stage), PNG.
    let plane = decoded.h * decoded.w;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..decoded.h {
        for x in 0..decoded.w {
            let hw = y * decoded.w + x;
            for ch in 0..3 {
                let v = (decoded.data[ch * plane + hw] / 2.0 + 0.5).clamp(0.0, 1.0);
                rgb[hw * 3 + ch] = (v * 255.0).round() as u8;
            }
        }
    }
    let png = encode_rgb8(decoded.w, decoded.h, &rgb).expect("encode png");
    assert!(
        png.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']),
        "encode_rgb8 must produce a valid PNG"
    );

    let _ = std::fs::remove_dir_all(&vae_dir);
}

/// `latent_seq_to_packed_nchw` must map sequence-major `[s, c]` (with
/// `s = hh*pw + ww`) to NCHW `[c, hh, ww]` — the exact transpose the VAE decode
/// consumes. Verified against a hand-computed reference plus the rejected
/// shape-mismatch error path.
#[test]
fn latent_packing_roundtrip() {
    let (ph, pw, ch) = (2usize, 3usize, 4usize);
    let seq = ph * pw;
    // latent[s, c] = s*10 + c so every element is uniquely identifiable.
    let latent: Vec<f32> = (0..seq * ch).map(|i| i as f32).collect();
    let packed = latent_seq_to_packed_nchw(&latent, seq, ch, ph, pw).expect("pack");
    assert_eq!(packed.len(), ch * seq);
    for hh in 0..ph {
        for ww in 0..pw {
            let s = hh * pw + ww;
            for c in 0..ch {
                let got = packed[c * seq + hh * pw + ww];
                let want = latent[s * ch + c];
                assert_eq!(got, want, "packed[c={c}, hh={hh}, ww={ww}] mismatch");
            }
        }
    }

    // Shape guards: seq_img != ph*pw and a wrong latent length both error.
    assert!(latent_seq_to_packed_nchw(&latent, seq + 1, ch, ph, pw).is_err());
    assert!(latent_seq_to_packed_nchw(&latent[..seq * ch - 1], seq, ch, ph, pw).is_err());
}
