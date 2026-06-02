//! Cross-implementation parity dump for the MLX → GGUF FLUX.2 DiT converter.
//!
//! This example validates [`oxibonsai_model::convert_mlx_image_to_gguf`]
//! end-to-end on a **real** checkpoint by converting the MLX-packed
//! `transformer.safetensors` to a temporary GGUF, reading a fixed set of
//! tensors back through the OxiBonsai GGUF reader, dequantizing them to `f32`,
//! and dumping the raw little-endian `f32` buffers to disk. An external numpy
//! reference (run against the original MLX safetensors) can then be compared
//! byte-for-byte against these dumps to prove the Rust converter and reader
//! agree with an independent implementation.
//!
//! # Usage
//!
//! ```text
//! cargo run -p oxibonsai-model --example mlx_image_parity --release -- \
//!     <path/to/transformer.safetensors> [out_dir]
//! ```
//!
//! `out_dir` defaults to `/tmp`. For each dumped tensor the example writes
//! `parity_rust_<sanitized_name>.f32` (where `.` and `/` in the GGUF name are
//! replaced with `_`) and prints the GGUF name, type, shape (as read back),
//! element count, and the first/last 8 `f32` values.
//!
//! # Ordering convention (read this before writing the numpy reference)
//!
//! The converter writes every tensor with a **reversed** shape: a logical
//! `[out, in]` weight is stored with GGUF dims `ne[0]=in`, so the shape printed
//! here is `[in, out]`. The dumped `f32` buffer, however, is in the **original
//! logical C-order**:
//!
//! * Quantized linears: block order is out-major (`row*(in/128)+group`), which
//!   [`BlockTQ2_0_g128::dequant`] flattens to row-major `[out, in]` —
//!   `f32[row*in + col] == W[row, col]`.
//! * BF16 passthrough: bytes are copied verbatim from the safetensors tensor,
//!   so the `f32` buffer is the original tensor in its native C-order.
//!
//! The numpy side must therefore reshape each `.f32` dump to the **original**
//! `[out, in]` (not the reversed GGUF shape).

use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxibonsai_model::convert_mlx_image_to_gguf;

/// Quantisation format the converter targets (matches `mlx_image_convert`).
const QUANT: &str = "tq2_0_g128";

/// The fixed set of GGUF tensor names this example reads back and dumps.
///
/// The first three are quantized linears (stored under their *base* name with
/// GGUF type `TQ2_0_g128`); the last two are BF16 auxiliary tensors (stored
/// under their *full* `.weight` name). Dispatch is on the read-back GGUF type,
/// so the same list works regardless of which bucket a name lands in.
const PARITY_TENSORS: &[&str] = &[
    // Quantized linears (base name, TQ2_0_g128).
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
    "transformer_blocks.0.attn.to_q",
    "single_transformer_blocks.0.attn.to_out",
    // BF16 auxiliary tensors (full name).
    "x_embedder.weight",
    "proj_out.weight",
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let from_path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            return Err("usage: mlx_image_parity <transformer.safetensors> [out_dir]".into());
        }
    };
    let out_dir = args.get(2).cloned().unwrap_or_else(|| "/tmp".to_string());

    let from_path = Path::new(&from_path);
    let out_dir = Path::new(&out_dir);
    fs::create_dir_all(out_dir)?;
    let gguf_path = out_dir.join("parity.gguf");

    println!("── OxiBonsai MLX (FLUX.2 DiT) → GGUF parity dump ──");
    println!("  source  : {}", from_path.display());
    println!("  gguf    : {}", gguf_path.display());
    println!("  out_dir : {}", out_dir.display());
    println!("  quant   : {QUANT}");

    // ── 1. Convert the real safetensors to a temporary GGUF ──────────────────
    let t0 = Instant::now();
    let stats = convert_mlx_image_to_gguf(from_path, &gguf_path, QUANT)?;
    let elapsed = t0.elapsed();

    println!();
    println!("── Conversion complete ──");
    println!("  total tensors      : {}", stats.n_tensors);
    println!("  TQ2_0_g128 weights : {}", stats.n_ternary);
    println!("  BF16 passthrough   : {}", stats.n_bf16);
    println!("  F16 passthrough    : {}", stats.n_f16);
    println!(
        "  output file size   : {} bytes ({:.2} MiB)",
        stats.output_bytes,
        (stats.output_bytes as f64) / (1024.0 * 1024.0)
    );
    println!("  elapsed            : {elapsed:.2?}");

    // ── 2. Read the fixed tensor set back and dump dequantized f32 ───────────
    println!();
    println!("── Dumping {} parity tensors ──", PARITY_TENSORS.len());
    let dumped = dump_tensors(&gguf_path, out_dir, PARITY_TENSORS)?;

    println!();
    println!(
        "── Done: dumped {dumped}/{} tensors to {} ──",
        PARITY_TENSORS.len(),
        out_dir.display()
    );

    Ok(())
}

/// Read each named tensor from `gguf_path`, dequantize to `f32`, and dump the
/// raw little-endian `f32` buffer to `<out_dir>/parity_rust_<sanitized>.f32`.
///
/// Dispatch is on the tensor's read-back GGUF type:
/// * [`GgufTensorType::TQ2_0_g128`] → ternary dequant via [`BlockTQ2_0_g128`].
/// * [`GgufTensorType::BF16`] → per-element widen `u16 → f32`.
/// * anything else → a warning (the parity set should never hit this).
///
/// A missing tensor prints a warning and is skipped (not an error). Returns the
/// number of tensors successfully dumped.
fn dump_tensors(gguf_path: &Path, out_dir: &Path, names: &[&str]) -> Result<usize, Box<dyn Error>> {
    // Read the whole GGUF into an allocator-aligned Vec. The 32-byte tensor-data
    // alignment in GGUF plus the Vec's base alignment makes the zero-copy
    // `slice_from_bytes` 2-byte-alignment check pass (same path the converter's
    // end-to-end test relies on); the warn-on-error branch below is the net.
    let gguf_bytes = fs::read(gguf_path)?;
    let parsed = GgufFile::parse(&gguf_bytes).map_err(|e| e.to_string())?;

    let mut dumped = 0usize;
    for &name in names {
        let info = match parsed.tensors.get(name) {
            Some(info) => info,
            None => {
                eprintln!("  [warn] tensor not found, skipping: {name}");
                continue;
            }
        };

        let raw = parsed.tensor_data(name).map_err(|e| e.to_string())?;

        let values: Vec<f32> = match info.tensor_type {
            GgufTensorType::TQ2_0_g128 => match dequant_ternary(raw) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  [warn] dequant failed for {name}: {e}; skipping");
                    continue;
                }
            },
            GgufTensorType::BF16 => decode_bf16(raw),
            other => {
                eprintln!("  [warn] unexpected GGUF type {other:?} for {name}; skipping");
                continue;
            }
        };

        let sanitized = sanitize(name);
        let out_path = out_dir.join(format!("parity_rust_{sanitized}.f32"));
        fs::write(&out_path, f32_slice_to_le_bytes(&values))?;

        println!(
            "  {name}\n    type   : {:?}\n    shape  : {:?}  (GGUF reversed; dump is original [out, in])\n    count  : {}\n    first8 : {:?}\n    last8  : {:?}\n    file   : {}",
            info.tensor_type,
            info.shape,
            values.len(),
            &values[..values.len().min(8)],
            &values[values.len().saturating_sub(8)..],
            out_path.display()
        );

        dumped += 1;
    }

    Ok(dumped)
}

/// Dequantize a raw `TQ2_0_g128` byte stream into a flat row-major `f32` buffer.
fn dequant_ternary(raw: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let blocks = BlockTQ2_0_g128::slice_from_bytes(raw).map_err(|e| e.to_string())?;
    let mut out = vec![0.0f32; blocks.len() * 128];
    BlockTQ2_0_g128::dequant(blocks, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// Decode a raw BF16 byte stream (little-endian `u16` bit patterns) to `f32`.
///
/// BF16 → f32 is an exact upper-half placement: `f32::from_bits(bf16 << 16)`.
fn decode_bf16(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Flatten an `f32` slice to its raw little-endian byte representation.
fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Sanitize a GGUF tensor name for use in a filename: `.`/`/` → `_`.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c == '.' || c == '/' { '_' } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use half::bf16;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    /// Owned raw byte buffers for one synthetic MLX quantized module, plus the
    /// expected row-major `[out, in]` dequant. Kept owned (no `'static`/leak) so
    /// the borrowing [`TensorView`]s can be built inline in the test's scope.
    struct QuantBuffers {
        weight_bytes: Vec<u8>,
        scales_bytes: Vec<u8>,
        biases_bytes: Vec<u8>,
        out: usize,
        weight_cols: usize,
        group_cols: usize,
        expected: Vec<f32>,
    }

    /// Build the 8 little-endian `u32` words packing one 128-code MLX group.
    fn group_words(codes: &[u8; 128]) -> [u32; 8] {
        let mut words = [0u32; 8];
        for (j, &q) in codes.iter().enumerate() {
            words[j / 16] |= (q as u32) << ((j % 16) * 2);
        }
        words
    }

    /// Round an `f32` to its bf16 bit pattern (no `#[cfg(test)]` lib helper is
    /// reachable from an example, so we go through `half` directly).
    fn bf16_bits(value: f32) -> u16 {
        bf16::from_f32(value).to_bits()
    }

    /// Build a tiny synthetic MLX-style quantized module's raw byte buffers
    /// (`weight` U32 `[out, in/16]`, `scales`/`biases` BF16 `[out, in/128]`) with
    /// `bias == -scale` (ternary), plus the expected row-major `[out, in]`
    /// dequant. Power-of-two scales are exact in both bf16 and the block's f16,
    /// so the parity assertion can be exact.
    fn build_quant_buffers(out: usize, in_features: usize) -> QuantBuffers {
        let group_cols = in_features / 128;
        let weight_cols = in_features / 16;

        let mut weight_words = vec![0u32; out * weight_cols];
        let mut scales_bits = vec![0u16; out * group_cols];
        let mut biases_bits = vec![0u16; out * group_cols];
        let mut expected = Vec::with_capacity(out * in_features);

        for row in 0..out {
            for g in 0..group_cols {
                let scale = 1.0_f32 / ((1u32 << (row + g + 1)) as f32);
                scales_bits[row * group_cols + g] = bf16_bits(scale);
                biases_bits[row * group_cols + g] = bf16_bits(-scale);

                let mut codes = [0u8; 128];
                for (j, c) in codes.iter_mut().enumerate() {
                    let q = ((row + g + j) % 3) as u8;
                    *c = q;
                    expected.push(scale * (q as f32 - 1.0));
                }
                let words = group_words(&codes);
                let wbase = row * weight_cols + g * 8;
                weight_words[wbase..wbase + 8].copy_from_slice(&words);
            }
        }

        QuantBuffers {
            weight_bytes: weight_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            scales_bytes: scales_bits.iter().flat_map(|s| s.to_le_bytes()).collect(),
            biases_bytes: biases_bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
            out,
            weight_cols,
            group_cols,
            expected,
        }
    }

    /// Append the three safetensors views for one quantized module, borrowing
    /// from the persistent [`QuantBuffers`].
    fn push_quant_views<'a>(
        views: &mut Vec<(String, TensorView<'a>)>,
        base: &str,
        q: &'a QuantBuffers,
    ) {
        views.push((
            format!("{base}.weight"),
            TensorView::new(Dtype::U32, vec![q.out, q.weight_cols], &q.weight_bytes)
                .expect("weight view"),
        ));
        views.push((
            format!("{base}.scales"),
            TensorView::new(Dtype::BF16, vec![q.out, q.group_cols], &q.scales_bytes)
                .expect("scales view"),
        ));
        views.push((
            format!("{base}.biases"),
            TensorView::new(Dtype::BF16, vec![q.out, q.group_cols], &q.biases_bytes)
                .expect("biases view"),
        ));
    }

    /// Read a `.f32` dump back into a `Vec<f32>`.
    fn read_f32_dump(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).expect("read dump");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn parity_dump_matches_reference_on_synthetic() {
        // Two quantized modules (the single_* and transformer_* style names from
        // the real parity set) plus one BF16 aux tensor.
        let q_base_a = "single_transformer_blocks.0.attn.to_qkv_mlp_proj";
        let q_base_b = "transformer_blocks.0.attn.to_q";
        let qa = build_quant_buffers(4, 256);
        let qb = build_quant_buffers(2, 128);

        // BF16 aux tensor (full name survives classification: no scales/biases).
        let bf16_name = "x_embedder.weight";
        let bf16_src: Vec<f32> = (0..32).map(|i| (i as f32) * 0.015625).collect();
        let bf16_bytes: Vec<u8> = bf16_src
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();

        let mut views: Vec<(String, TensorView<'_>)> = Vec::new();
        push_quant_views(&mut views, q_base_a, &qa);
        push_quant_views(&mut views, q_base_b, &qb);
        views.push((
            bf16_name.to_string(),
            TensorView::new(Dtype::BF16, vec![4, 8], &bf16_bytes).expect("bf16 view"),
        ));

        let st_bytes = safetensors::serialize(views, None).expect("serialize safetensors");

        // Per-process temp dir.
        let dir =
            std::env::temp_dir().join(format!("oxibonsai_parity_test_{}", std::process::id()));
        fs::create_dir_all(&dir).expect("mkdir temp");
        let st_path = dir.join("transformer.safetensors");
        let gguf_path = dir.join("parity.gguf");
        fs::write(&st_path, &st_bytes).expect("write safetensors");

        // Run the real converter, then the example's dump function.
        let stats = convert_mlx_image_to_gguf(&st_path, &gguf_path, QUANT)
            .expect("conversion should succeed");
        assert_eq!(stats.n_ternary, 2, "two quantized modules");
        assert_eq!(stats.n_bf16, 1, "one bf16 passthrough");

        let names = [q_base_a, q_base_b, bf16_name];
        let dumped = dump_tensors(&gguf_path, &dir, &names).expect("dump");
        assert_eq!(dumped, 3, "all three tensors dumped");

        // Quantized dumps: exact `scale*(q-1)` row-major [out, in].
        let got_a = read_f32_dump(&dir.join(format!("parity_rust_{}.f32", sanitize(q_base_a))));
        assert_eq!(got_a, qa.expected, "quant module A must match reference");
        let got_b = read_f32_dump(&dir.join(format!("parity_rust_{}.f32", sanitize(q_base_b))));
        assert_eq!(got_b, qb.expected, "quant module B must match reference");

        // BF16 dump: exact bf16-rounded values via the prescribed widening.
        let got_bf16 = read_f32_dump(&dir.join(format!("parity_rust_{}.f32", sanitize(bf16_name))));
        let expected_bf16: Vec<f32> = bf16_src
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect();
        assert_eq!(got_bf16, expected_bf16, "bf16 dump must match reference");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_tensor_is_skipped_not_fatal() {
        // Build a minimal GGUF with a single BF16 tensor, then ask for a name
        // that does not exist; dump_tensors must skip it and report 0 dumped.
        let bf16_bytes: Vec<u8> = [0.0f32, 1.0, -1.0, 0.5]
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        let views = vec![(
            "proj_out.weight".to_string(),
            TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).expect("bf16 view"),
        )];
        let st_bytes = safetensors::serialize(views, None).expect("serialize safetensors");

        let dir =
            std::env::temp_dir().join(format!("oxibonsai_parity_missing_{}", std::process::id()));
        fs::create_dir_all(&dir).expect("mkdir temp");
        let st_path = dir.join("transformer.safetensors");
        let gguf_path = dir.join("parity.gguf");
        fs::write(&st_path, &st_bytes).expect("write safetensors");
        convert_mlx_image_to_gguf(&st_path, &gguf_path, QUANT).expect("convert");

        let names = ["this.tensor.does.not.exist"];
        let dumped = dump_tensors(&gguf_path, &dir, &names).expect("dump");
        assert_eq!(dumped, 0, "missing tensor must be skipped");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_replaces_dots_and_slashes() {
        assert_eq!(sanitize("a.b/c.d"), "a_b_c_d");
        assert_eq!(
            sanitize("single_transformer_blocks.0.attn.to_q"),
            "single_transformer_blocks_0_attn_to_q"
        );
    }

    #[test]
    fn decode_bf16_widens_exactly() {
        // 1.0 in bf16 is 0x3f80; widened to f32 upper half it is exactly 1.0.
        let raw = 0x3f80u16.to_le_bytes();
        let got = decode_bf16(&raw);
        assert_eq!(got, vec![1.0f32]);
    }
}
