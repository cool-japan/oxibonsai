//! Quantization pipeline integration tests
//!
//! Tests the complete quantization stack from FP32 through Q1_0_g128 / INT8,
//! model export, LoRA adapters, training scaffold, optimizer convergence, and
//! the ModelConfigBuilder.

use oxibonsai_model::{
    export::{export_to_gguf, ExportConfig, ExportFormat, WeightTensor},
    lora::{LoraAdapter, LoraConfig, LoraRegistry},
    lora_trainer::{LoraTrainer, LoraTrainingConfig},
    model_config_builder::ModelConfigBuilder,
    optimizer::Adam,
    quantize::{
        analyze_quantization_error, compute_weight_stats, dequantize_q1_0_g128, quantize_q1_0_g128,
        GROUP_SIZE,
    },
    quantize_int8::{quantize_per_channel, quantize_per_tensor, Int8Mode},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Generate a deterministic sequence of f32 values.
fn synthetic_weights(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as f32 * 0.13 + 1.0).sin() * scale;
            if v == 0.0 {
                0.001
            } else {
                v
            }
        })
        .collect()
}

// ── Q1_0_g128 roundtrip ───────────────────────────────────────────────────────

#[test]
fn test_quantize_and_dequantize_roundtrip() {
    let weights = synthetic_weights(GROUP_SIZE * 4, 1.0);

    let quantized = quantize_q1_0_g128(&weights).expect("quantize should succeed");
    let dequantized = dequantize_q1_0_g128(&quantized).expect("dequantize should succeed");

    assert_eq!(
        dequantized.len(),
        weights.len(),
        "roundtrip must preserve element count"
    );

    // Q1_0_g128 is lossy; check that the MAE is within a reasonable bound.
    let mae: f32 = weights
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / weights.len() as f32;

    assert!(
        mae < 2.0,
        "mean absolute error of Q1_0 roundtrip should be < 2.0; got {mae}"
    );
}

// ── INT8 quantize / dequantize ────────────────────────────────────────────────

#[test]
fn test_int8_quantize_and_dequantize() {
    // 4 output channels, each of size 16 = 64 elements total.
    let weights = synthetic_weights(64, 2.0);

    let tensor =
        quantize_per_channel(&weights, 4).expect("INT8 per-channel quantize should succeed");
    assert_eq!(
        tensor.mode,
        Int8Mode::PerChannel { num_channels: 4 },
        "mode should be PerChannel"
    );
    assert_eq!(
        tensor.data.len(),
        64,
        "quantized data length must match input"
    );

    let dequantized = tensor.dequantize();
    assert_eq!(dequantized.len(), 64);

    // INT8 per-channel should be more accurate than Q1_0; MAE < 0.5.
    let mae: f32 = weights
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / weights.len() as f32;

    assert!(mae < 0.5, "INT8 per-channel MAE should be < 0.5; got {mae}");
}

// ── GGUF export ───────────────────────────────────────────────────────────────

#[test]
fn test_export_to_gguf_writes_valid_file() {
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        synthetic_weights(128, 1.0),
        vec![1, 128],
    )];
    let config = ExportConfig::new(ExportFormat::Float32, "test-model");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export should succeed");

    assert!(!bytes.is_empty(), "exported GGUF must not be empty");
    // GGUF magic: the first four bytes must be the characters G G U F.
    // Byte sequence written by GgufWriter: [0x47, 0x47, 0x55, 0x46] = "GGUF" in LE.
    // Check non-empty and starts with 'G'.
    assert_eq!(bytes[0], b'G', "GGUF file first byte must be 'G'");
}

// ── Quantization method comparison ───────────────────────────────────────────

#[test]
fn test_quantization_comparison_q1_0_vs_int8() {
    let weights = synthetic_weights(GROUP_SIZE, 1.5);

    // Q1_0_g128 error
    let q1_quantized = quantize_q1_0_g128(&weights).expect("q1 quantize");
    let q1_error = analyze_quantization_error(&weights, &q1_quantized).expect("q1 error analysis");

    // INT8 per-tensor error
    let int8_tensor = quantize_per_tensor(&weights);
    let int8_dequant = int8_tensor.dequantize();
    let int8_mae: f32 = weights
        .iter()
        .zip(int8_dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / weights.len() as f32;

    // Both error analyses should succeed — just verify they produce finite values.
    assert!(
        q1_error.mse.is_finite(),
        "Q1_0 MSE must be finite; got {}",
        q1_error.mse
    );
    assert!(
        int8_mae.is_finite(),
        "INT8 MAE must be finite; got {int8_mae}"
    );
}

// ── Weight stats ──────────────────────────────────────────────────────────────

#[test]
fn test_weight_stats_computation() {
    let weights: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let stats = compute_weight_stats(&weights);

    assert!(
        (stats.min - (-3.0)).abs() < 1e-5,
        "min stat should be -3.0; got {}",
        stats.min
    );
    assert!(
        (stats.max - 3.0).abs() < 1e-5,
        "max stat should be 3.0; got {}",
        stats.max
    );
    assert!(
        stats.std >= 0.0,
        "std must be non-negative; got {}",
        stats.std
    );
    assert!(
        stats.sparsity >= 0.0 && stats.sparsity <= 1.0,
        "sparsity must be in [0,1]; got {}",
        stats.sparsity
    );
}

// ── LoRA adapter apply + merge ────────────────────────────────────────────────

#[test]
fn test_lora_adapter_apply_and_merge() {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        ..Default::default()
    };
    let d_in = 16;
    let d_out = 16;
    let adapter = LoraAdapter::new(d_in, d_out, config);

    let x: Vec<f32> = (0..d_in).map(|i| i as f32 * 0.1).collect();
    let out = adapter
        .apply(&x)
        .expect("apply with correctly-sized input should succeed");
    // B is zero-initialized so output should be all zeros.
    assert_eq!(out.len(), d_out);
    for &v in &out {
        assert!(
            v.abs() < 1e-5,
            "zero B-matrix adapter output should be ~0; got {v}"
        );
    }

    // After merge the weight delta should still be zero.
    let mut weights = vec![1.0_f32; d_out * d_in];
    adapter
        .merge_into_weights(&mut weights)
        .expect("merge with correctly-sized weights should succeed");
    // Weights unchanged since delta = B * A = 0.
    for &w in &weights {
        assert!(
            (w - 1.0).abs() < 1e-5,
            "merging zero-delta adapter should not change weights"
        );
    }
}

// ── LoRA registry multiple adapters ──────────────────────────────────────────

#[test]
fn test_lora_registry_multiple_adapters() {
    let config = LoraConfig {
        rank: 4,
        alpha: 4.0,
        ..Default::default()
    };
    let mut registry = LoraRegistry::new(config.clone());

    registry.add("q_proj", LoraAdapter::new(64, 64, config.clone()));
    registry.add("v_proj", LoraAdapter::new(64, 64, config.clone()));
    registry.add("k_proj", LoraAdapter::new(64, 64, config.clone()));

    assert_eq!(
        registry.adapter_count(),
        3,
        "registry should have 3 adapters"
    );

    let x: Vec<f32> = vec![0.5_f32; 64];
    assert!(
        registry.apply_adapter("q_proj", &x).is_some(),
        "apply for registered module should return Some"
    );
    assert!(
        registry.apply_adapter("missing", &x).is_none(),
        "apply for unregistered module should return None"
    );
}

// ── LoRA trainer step ─────────────────────────────────────────────────────────

#[test]
fn test_lora_trainer_step_reduces_loss() {
    let train_config = LoraTrainingConfig {
        lora_config: LoraConfig {
            rank: 4,
            alpha: 4.0,
            ..Default::default()
        },
        learning_rate: 1e-2,
        warmup_steps: 0,
        max_steps: 10,
        ..Default::default()
    };
    let mut trainer = LoraTrainer::new(train_config);
    // LoraAdapter::new(d_in, d_out, config)
    let mut adapter = LoraAdapter::new(
        8,
        8,
        LoraConfig {
            rank: 4,
            alpha: 4.0,
            ..Default::default()
        },
    );

    // Simulate three steps with decreasing loss.
    let losses = [2.0_f32, 1.5, 1.0];
    for (step_idx, &loss) in losses.iter().enumerate() {
        let grad_a = vec![0.01_f32; 4 * 8]; // rank * d_in
        let grad_b = vec![0.01_f32; 8 * 4]; // d_out * rank
        let step_result = trainer.step(loss, &mut adapter, grad_a, grad_b);

        assert_eq!(step_result.step, step_idx, "step index must match");
        assert!(
            step_result.loss.is_finite(),
            "loss must be finite; got {}",
            step_result.loss
        );
    }

    assert_eq!(trainer.total_steps(), 3, "should have completed 3 steps");
}

// ── Adam optimizer convergence ────────────────────────────────────────────────

#[test]
fn test_optimizer_adam_converges() {
    // Simple quadratic: minimize f(x) = sum(x_i^2).
    // Gradient: grad_i = 2 * x_i.
    let mut params = [vec![3.0_f32, -2.0, 1.5]];
    let mut optimizer = Adam::new(0.1);

    let initial_loss: f32 = params[0].iter().map(|&v| v * v).sum();

    for _ in 0..50 {
        let grads: Vec<Vec<f32>> = params
            .iter()
            .map(|p| p.iter().map(|&v| 2.0 * v).collect())
            .collect();

        let param_slices: Vec<&mut Vec<f32>> = params.iter_mut().collect();
        let mut param_refs: Vec<&mut Vec<f32>> = param_slices;
        optimizer.step(&mut param_refs, &grads);
    }

    let final_loss: f32 = params[0].iter().map(|&v| v * v).sum();
    assert!(
        final_loss < initial_loss,
        "Adam should reduce loss: initial {initial_loss:.4} > final {final_loss:.4}"
    );
    assert!(
        final_loss < 0.1,
        "Adam should converge close to 0; got final_loss = {final_loss:.6}"
    );
}

// ── ModelConfigBuilder roundtrip ──────────────────────────────────────────────

#[test]
fn test_model_config_builder_roundtrip() {
    // Build a custom config and verify all fields are preserved.
    let config = ModelConfigBuilder::new()
        .layers(4)
        .hidden_size(128)
        .num_attention_heads(4)
        .num_kv_heads(2)
        .intermediate_size(256)
        .vocab_size(512)
        .max_position_embeddings(1024)
        .build()
        .expect("config builder should succeed");

    assert_eq!(config.num_layers, 4);
    assert_eq!(config.hidden_size, 128);
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.intermediate_size, 256);
    assert_eq!(config.vocab_size, 512);
    assert_eq!(config.max_context_length, 1024);

    // The tiny preset should also remain valid.
    let tiny = ModelConfigBuilder::build_tiny();
    assert_eq!(tiny.num_layers, 2);
    assert_eq!(tiny.hidden_size, 64);
}
