// Weight-loading helper functions extracted from model.rs.
// These are `pub(super)` so only `model.rs` can call them.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::layers::linear::{Linear1Bit, LinearFP8E4M3, LinearFP8E5M2, LinearTernary};
use crate::layers::rms_norm::RmsNorm;

use super::types::OutputWeight;

/// Load an FP32 tensor from GGUF by name.
pub(super) fn load_f32_tensor(gguf: &GgufFile<'_>, name: &str) -> ModelResult<Vec<f32>> {
    let info = gguf.tensors.require(name).map_err(ModelError::Core)?;

    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;

    match info.tensor_type {
        GgufTensorType::F32 => {
            let count = data.len() / 4;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        GgufTensorType::F16 => {
            let count = data.len() / 2;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(2).enumerate() {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = half::f16::from_bits(raw).to_f32();
            }
            Ok(out)
        }
        GgufTensorType::Q1_0_g128 => {
            let blocks = BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)?;
            let n = blocks.len() * QK1_0_G128;
            let mut out = vec![0.0f32; n];
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let base = i * QK1_0_G128;
                for j in 0..QK1_0_G128 {
                    let byte_index = j / 8;
                    let bit_offset = j % 8;
                    let bit = (block.qs[byte_index] >> bit_offset) & 1;
                    out[base + j] = if bit != 0 { d } else { -d };
                }
            }
            Ok(out)
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data)
                .map_err(ModelError::Core)?;
            let n = blocks.len() * oxibonsai_core::QK_TQ2_0_G128;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockTQ2_0_g128::dequant(blocks, &mut out).map_err(ModelError::Core)?;
            Ok(out)
        }
        other => Err(ModelError::MissingTensor {
            name: format!("{name}: expected F32, F16, Q1_0_g128, or TQ2_0_g128, got {other}"),
        }),
    }
}

/// Load Q1_0_g128 weight blocks from GGUF (zero-copy).
pub(super) fn load_1bit_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [BlockQ1_0G128]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load TQ2\_0\_g128 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockTQ2_0_g128` pointing directly into the
/// memory-mapped GGUF data.  The lifetime is tied to the `GgufFile`.
pub(super) fn load_ternary_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockTQ2_0_g128]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load a TQ2\_0\_g128 tensor and dequantize it to FP32.
///
/// Used for embedding tables stored in ternary format.  The returned
/// `Vec<f32>` has length `blocks.len() × 128`.
///
/// Currently unused — ternary token embeddings are not part of Qwen3/Bonsai
/// models yet, but the helper is provided for future model variants.
#[allow(dead_code)]
pub(super) fn load_ternary_embedding(gguf: &GgufFile<'_>, name: &str) -> ModelResult<Vec<f32>> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    let blocks =
        oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)?;
    let n = blocks.len() * oxibonsai_core::QK_TQ2_0_G128;
    let mut out = vec![0.0f32; n];
    oxibonsai_core::BlockTQ2_0_g128::dequant(blocks, &mut out).map_err(ModelError::Core)?;
    Ok(out)
}

/// Load FP8 E4M3FN weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockFP8E4M3` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_fp8_e4m3_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockFP8E4M3]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    oxibonsai_core::BlockFP8E4M3::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load FP8 E5M2 weight blocks from GGUF (zero-copy).
///
/// Returns a borrowed slice of `BlockFP8E5M2` pointing directly into the
/// memory-mapped GGUF data. The lifetime is tied to the `GgufFile`.
pub(super) fn load_fp8_e5m2_blocks<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
) -> ModelResult<&'a [oxibonsai_core::BlockFP8E5M2]> {
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    oxibonsai_core::BlockFP8E5M2::slice_from_bytes(data).map_err(ModelError::Core)
}

/// Load a single Transformer block's weights from GGUF.
///
/// Automatically detects whether the model uses Q1\_0\_g128 (1-bit) or
/// TQ2\_0\_g128 (ternary) quantization by inspecting the attention Q tensor.
pub(super) fn load_transformer_block<'a>(
    gguf: &'a GgufFile<'a>,
    config: &Qwen3Config,
    layer_idx: usize,
    kernel: &std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
) -> ModelResult<TransformerBlock<'a>> {
    let h = config.hidden_size;
    let nq = config.num_attention_heads;
    let nkv = config.num_kv_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_size;

    let blk = |suffix: &str| -> String { tensor_names::block_tensor(layer_idx, suffix) };

    // Detect quantization type from the Q projection tensor.
    let sample_name = blk(tensor_names::ATTN_Q);
    let sample_info = gguf
        .tensors
        .require(&sample_name)
        .map_err(ModelError::Core)?;
    let is_ternary = sample_info.tensor_type == GgufTensorType::TQ2_0_g128;
    let is_fp8_e4m3 = sample_info.tensor_type == GgufTensorType::F8_E4M3;
    let is_fp8_e5m2 = sample_info.tensor_type == GgufTensorType::F8_E5M2;

    // RMSNorm weights (always FP32).
    let attn_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_NORM))?;
    let ffn_norm_w = load_f32_tensor(gguf, &blk(tensor_names::FFN_NORM))?;
    let q_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_Q_NORM))?;
    let k_norm_w = load_f32_tensor(gguf, &blk(tensor_names::ATTN_K_NORM))?;

    if is_ternary {
        let q_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
        let k_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_K))?;
        let v_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_V))?;
        let o_blocks = load_ternary_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
        let gate_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
        let up_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_UP))?;
        let down_blocks = load_ternary_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

        let block = TransformerBlock::new(
            layer_idx,
            RmsNorm::new(attn_norm_w, config.rms_norm_eps),
            LinearTernary::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
            LinearTernary::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearTernary::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearTernary::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
            RmsNorm::new(q_norm_w, config.rms_norm_eps),
            RmsNorm::new(k_norm_w, config.rms_norm_eps),
            RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
            LinearTernary::new(gate_blocks, inter, h, kernel.clone())?.into(),
            LinearTernary::new(up_blocks, inter, h, kernel.clone())?.into(),
            LinearTernary::new(down_blocks, h, inter, kernel.clone())?.into(),
            nq,
            nkv,
            hd,
            h,
        );
        tracing::trace!(layer = layer_idx, "loaded ternary transformer block");
        Ok(block)
    } else if is_fp8_e4m3 {
        // FP8 E4M3FN path.
        let q_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
        let k_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_K))?;
        let v_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_V))?;
        let o_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
        let gate_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
        let up_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_UP))?;
        let down_blocks = load_fp8_e4m3_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

        let block = TransformerBlock::new(
            layer_idx,
            RmsNorm::new(attn_norm_w, config.rms_norm_eps),
            LinearFP8E4M3::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
            LinearFP8E4M3::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearFP8E4M3::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearFP8E4M3::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
            RmsNorm::new(q_norm_w, config.rms_norm_eps),
            RmsNorm::new(k_norm_w, config.rms_norm_eps),
            RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
            LinearFP8E4M3::new(gate_blocks, inter, h, kernel.clone())?.into(),
            LinearFP8E4M3::new(up_blocks, inter, h, kernel.clone())?.into(),
            LinearFP8E4M3::new(down_blocks, h, inter, kernel.clone())?.into(),
            nq,
            nkv,
            hd,
            h,
        );
        tracing::trace!(layer = layer_idx, "loaded FP8 E4M3FN transformer block");
        Ok(block)
    } else if is_fp8_e5m2 {
        // FP8 E5M2 path.
        let q_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
        let k_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_K))?;
        let v_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_V))?;
        let o_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
        let gate_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
        let up_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_UP))?;
        let down_blocks = load_fp8_e5m2_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

        let block = TransformerBlock::new(
            layer_idx,
            RmsNorm::new(attn_norm_w, config.rms_norm_eps),
            LinearFP8E5M2::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
            LinearFP8E5M2::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearFP8E5M2::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
            LinearFP8E5M2::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
            RmsNorm::new(q_norm_w, config.rms_norm_eps),
            RmsNorm::new(k_norm_w, config.rms_norm_eps),
            RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
            LinearFP8E5M2::new(gate_blocks, inter, h, kernel.clone())?.into(),
            LinearFP8E5M2::new(up_blocks, inter, h, kernel.clone())?.into(),
            LinearFP8E5M2::new(down_blocks, h, inter, kernel.clone())?.into(),
            nq,
            nkv,
            hd,
            h,
        );
        tracing::trace!(layer = layer_idx, "loaded FP8 E5M2 transformer block");
        Ok(block)
    } else {
        // Q1_0_g128 (1-bit) path.
        let q_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_Q))?;
        let k_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_K))?;
        let v_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_V))?;
        let o_blocks = load_1bit_blocks(gguf, &blk(tensor_names::ATTN_OUTPUT))?;
        let gate_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_GATE))?;
        let up_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_UP))?;
        let down_blocks = load_1bit_blocks(gguf, &blk(tensor_names::FFN_DOWN))?;

        let block = TransformerBlock::new(
            layer_idx,
            RmsNorm::new(attn_norm_w, config.rms_norm_eps),
            Linear1Bit::new(q_blocks, nq * hd, h, kernel.clone())?.into(),
            Linear1Bit::new(k_blocks, nkv * hd, h, kernel.clone())?.into(),
            Linear1Bit::new(v_blocks, nkv * hd, h, kernel.clone())?.into(),
            Linear1Bit::new(o_blocks, h, nq * hd, kernel.clone())?.into(),
            RmsNorm::new(q_norm_w, config.rms_norm_eps),
            RmsNorm::new(k_norm_w, config.rms_norm_eps),
            RmsNorm::new(ffn_norm_w, config.rms_norm_eps),
            Linear1Bit::new(gate_blocks, inter, h, kernel.clone())?.into(),
            Linear1Bit::new(up_blocks, inter, h, kernel.clone())?.into(),
            Linear1Bit::new(down_blocks, h, inter, kernel.clone())?.into(),
            nq,
            nkv,
            hd,
            h,
        );
        tracing::trace!(layer = layer_idx, "loaded transformer block");
        Ok(block)
    }
}

/// Load the output (LM head) weight — may be Q1_0_g128 or FP32.
pub(super) fn load_output_weight<'a>(
    gguf: &'a GgufFile<'a>,
    config: &Qwen3Config,
    kernel: &std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
) -> ModelResult<OutputWeight<'a>> {
    let info = gguf
        .tensors
        .require(tensor_names::OUTPUT)
        .map_err(ModelError::Core)?;

    // Derive actual output dimensions from the tensor shape rather than
    // config.vocab_size, which may reflect the tokenizer vocabulary rather
    // than the model's actual output projection size.
    let out_features = if info.shape.len() >= 2 {
        info.shape[1] as usize
    } else {
        config.vocab_size
    };
    let in_features = if !info.shape.is_empty() {
        info.shape[0] as usize
    } else {
        config.hidden_size
    };

    match info.tensor_type {
        GgufTensorType::Q1_0_g128 => {
            let blocks = load_1bit_blocks(gguf, tensor_names::OUTPUT)?;
            let linear = Linear1Bit::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::OneBit(linear))
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = load_ternary_blocks(gguf, tensor_names::OUTPUT)?;
            let linear = LinearTernary::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::Ternary(linear))
        }
        GgufTensorType::F8_E4M3 => {
            let blocks = load_fp8_e4m3_blocks(gguf, tensor_names::OUTPUT)?;
            let linear = LinearFP8E4M3::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::FP8E4M3(linear))
        }
        GgufTensorType::F8_E5M2 => {
            let blocks = load_fp8_e5m2_blocks(gguf, tensor_names::OUTPUT)?;
            let linear = LinearFP8E5M2::new(blocks, out_features, in_features, kernel.clone())?;
            Ok(OutputWeight::FP8E5M2(linear))
        }
        GgufTensorType::F32 | GgufTensorType::F16 => {
            let weights = load_f32_tensor(gguf, tensor_names::OUTPUT)?;
            Ok(OutputWeight::Fp32 {
                weights,
                out_features,
                in_features,
            })
        }
        other => Err(ModelError::MissingTensor {
            name: format!("output.weight: unsupported type {other}"),
        }),
    }
}
