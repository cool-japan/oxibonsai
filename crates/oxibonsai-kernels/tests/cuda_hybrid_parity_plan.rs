//! The CUDA-hardware parity checklist (test plan) for the CUDA paths that
//! need a real-device parity run: the `qwen35` hybrid-layer kernels (finding
//! **F13**), the `PQ2_0` / `PTQ1_0` 2-bit GEMV paths (finding **F16**), the
//! batch-prefill KV read-back (finding **F6**), the view-based batch-prefill
//! attention (finding **F9**) and the fused ternary gate‖up GEMV the
//! sliding-window and stats forwards reach.
//!
//! **Status (0.2.4, RTX A4000, CUDA 12.0, 2026-10-07):** CUDA-P11, P14, P15
//! and P18 pass on real models through the dedicated harnesses
//! (`crates/oxibonsai-model/tests/cuda_p11_q_std_kv_readback.rs`,
//! `crates/oxibonsai-runtime/tests/cuda_p14_p15_batch_prefill_vs_sequential.rs`,
//! `crates/oxibonsai-kernels/tests/cuda_p18_tq2_gemv_real_shapes.rs`); P09
//! partial (no `+2` codes in the available files); P12/P13 at CLI token level
//! only; P01-P08 and P10 not run (no 27B files on the CUDA host); P16/P17 not
//! run (no sliding-window / stats forward harness). The boxes below stay
//! manual: this file is the plan, not a record of results.
//!
//! [`CUDA_HARDWARE_PARITY_PLAN`] is the list of runs that must pass
//! on a real CUDA device — each at cosine similarity `>= 0.999` against the
//! named CPU reference, per row / per head, on real activations — before any
//! of these paths is advertised as working.
//!
//! The runs are manual, so this file never performs them and never claims
//! they ran:
//! - [`cuda_hardware_parity_plan_names_every_required_run`] checks the plan
//!   itself (host-only, runs everywhere);
//! - [`cuda_hardware_parity_plan_self_skips`] self-skips on every host and
//!   files a [`Capability::CudaHardware`] record with `"executed": false`
//!   in the release-gate capability manifest, so the manifest shows the plan
//!   exists and that this test did not execute it. It never writes
//!   `"executed": true`.

use oxibonsai_testkit::capability::{record_skipped, Capability};

/// Cosine-similarity threshold every run in the plan must meet.
const COS_THRESHOLD: &str = "cos >= 0.999";

/// The checklist. One run per line, `CUDA-Pnn`, each naming the CUDA entry
/// point, the CPU reference it is compared against, and the inputs.
const CUDA_HARDWARE_PARITY_PLAN: &str = "\
CUDA hardware parity checklist (manual; every run: cos >= 0.999 vs the CPU reference, \
per row / per head, on real Bonsai 2 27B layer-0 activations unless stated otherwise)

[ ] CUDA-P01 fwht_signed forward: cuda_qwen35::launch_fwht_signed(inverse = false, \
block = 1024, scale = 1/sqrt(1024)) over widths 5120, 6144 and 17408 with the \
checkpoint's prism.hadamard sign vectors, vs the CPU blockwise FWHT with fused signs \
(oxibonsai_kernels::hadamard::fwht_forward_signed). cos >= 0.999.
[ ] CUDA-P02 fwht_signed inverse: launch_fwht_signed(inverse = true) on token_embd rows \
(width 5120 as five 1024-wide blocks, h = FWHT_1024(z) * signs[5120]), vs the CPU \
inverse-embedding transform (hadamard::fwht_inverse_signed). cos >= 0.999.
[ ] CUDA-P03 conv1d_silu: launch_conv1d_silu over >= 4 consecutive tokens with the \
channel-major [3 x 10240] rolling state, vs ssm_ops::causal_conv1d_k4_decode + SiLU \
token by token; outputs cos >= 0.999 and the final state window equal.
[ ] CUDA-P04 l2_normalize: launch_l2_normalize on the q and k heads (head_k_dim 128), vs \
the CPU per-head L2 normalisation. cos >= 0.999.
[ ] CUDA-P05 sigmoid_gate: launch_sigmoid_gate (attn * sigmoid(gate)) on a full-attention \
layer's de-interleaved gate, vs the CPU gate. cos >= 0.999.
[ ] CUDA-P06 gated_rmsnorm: launch_gated_rmsnorm (weight * RMSNorm(o) * silu(z)) over 48 \
v-heads x 128, vs the CPU gated RMSNorm. cos >= 0.999.
[ ] CUDA-P07 partial_rope: launch_partial_rope with n_rot = 64 of head_dim = 256 (NEOX \
half-split pairs (i, i + 32)) at several positions, vs the CPU partial RoPE \
(rope_mrope::rope_partial_splithalf_simd with a partial_rope_build_table table); \
dimensions 64..256 must pass through unchanged. cos >= 0.999.
[ ] CUDA-P08 gdn_step: launch_gdn_step over >= 8 consecutive tokens with the state \
threaded between calls, vs gated_delta_net::gdn_step_with (out_scale 1/sqrt(128)); \
per-token outputs and the final state cos >= 0.999.
[ ] CUDA-P09 gemv_pq2_g128_v1: CudaGraph::encode_gemv_pq2_cached after \
get_or_upload_weight_pq2_soa / upload_weight_pq2_soa on real PQ2_0 (ggml 142) weights, \
including blocks that carry the +2 code (0b11), vs the CPU PQ2_0 GEMV \
(dequant_prism::gemv_pq2_0). cos >= 0.999.
[ ] CUDA-P10 PTQ1_0 -> TQ2 SoA transcode: cuda_qwen35::ptq1_blocks_to_tq2_soa + \
CudaGraph::upload_weight_soa_direct, served by gemv_tq2_g128_v1, on real PTQ1_0 \
(ggml 143) weights, vs the CPU PTQ1_0 GEMV (gemv_ptq1::gemv_ptq1_0). cos >= 0.999.
[ ] CUDA-P11 Q4_0/Q8_0 KV read-back: try_cuda_prefill_q_std(kv_readback_out = Some) at \
pos_start 0 on a real Q4_0 and a real Q8_0 GGUF, read-back K/V vs the K/V the \
sequential host path stores for the same prompt (cos >= 0.999 per layer and head), \
then the greedy tokens decoded after the prefill equal the sequential path's.
[ ] CUDA-P12 K-quant KV read-back: try_cuda_prefill_k_quant(kv_readback_out = Some), \
same comparison as CUDA-P11, for each K-quant format available. cos >= 0.999.
[ ] CUDA-P13 FP8 KV read-back: cuda_fp8_prefill::try_cuda_prefill_fp8_with_kv_readback, \
same comparison as CUDA-P11, E4M3 and E5M2. cos >= 0.999.
[ ] CUDA-P14 Q1 batch prefill, F9 view path: cuda_prefill::encode_prefill_layer \
(encode_attn_phase with hidden_in / attn_out views) on Bonsai-8B (Q1_0_g128), last-token \
logits vs the sequential per-token CUDA path (cos >= 0.999) and identical greedy decode.
[ ] CUDA-P15 TQ2 batch prefill, F9 view path: cuda_prefill::encode_prefill_layer_ternary \
(encode_attn_phase_tq2 with hidden_in / attn_out views) on Ternary-Bonsai-8B, same \
comparison as CUDA-P14. cos >= 0.999.
[ ] CUDA-P16 fused gate||up GEMV, sliding-window forward: the try_fused_gate_up_ternary \
call in block/types/forward_sw.rs (forward_with_sliding_window) on a native-cuda build \
with a GPU-uploaded ternary block, vs the per-matrix CPU ffn_gate / ffn_up projections. \
cos >= 0.999.
[ ] CUDA-P17 fused gate||up GEMV, stats forward: the try_fused_gate_up_ternary call in \
block/types/forward_stats.rs (forward_with_stats), same comparison as CUDA-P16. \
cos >= 0.999.
[ ] CUDA-P18 2-bit GEMV geometry guard: encode_gemv_tq2_cached over every projection and \
the LM head of Ternary-Bonsai-1.7B and -8B (real shapes) returns no InvalidDimensions \
and matches the CPU TQ2 GEMV. cos >= 0.999.
";

/// Every run the plan must name, as `(id, required phrases)`: each phrase
/// must appear on that run's line.
const REQUIRED_RUNS: &[(&str, &[&str])] = &[
    ("CUDA-P01", &["fwht_signed forward", "inverse = false"]),
    ("CUDA-P02", &["fwht_signed inverse", "inverse = true"]),
    ("CUDA-P03", &["conv1d_silu", "causal_conv1d_k4_decode"]),
    ("CUDA-P04", &["l2_normalize"]),
    ("CUDA-P05", &["sigmoid_gate"]),
    ("CUDA-P06", &["gated_rmsnorm"]),
    ("CUDA-P07", &["partial_rope", "n_rot = 64"]),
    ("CUDA-P08", &["gdn_step", "gdn_step_with"]),
    ("CUDA-P09", &["gemv_pq2_g128_v1", "+2 code"]),
    (
        "CUDA-P10",
        &["PTQ1_0 -> TQ2 SoA transcode", "gemv_tq2_g128_v1"],
    ),
    (
        "CUDA-P11",
        &["Q4_0/Q8_0 KV read-back", "try_cuda_prefill_q_std"],
    ),
    (
        "CUDA-P12",
        &["K-quant KV read-back", "try_cuda_prefill_k_quant"],
    ),
    (
        "CUDA-P13",
        &["FP8 KV read-back", "try_cuda_prefill_fp8_with_kv_readback"],
    ),
    ("CUDA-P14", &["Q1 batch prefill", "hidden_in / attn_out"]),
    ("CUDA-P15", &["TQ2 batch prefill", "encode_attn_phase_tq2"]),
    ("CUDA-P16", &["forward_sw.rs", "try_fused_gate_up_ternary"]),
    (
        "CUDA-P17",
        &["forward_stats.rs", "try_fused_gate_up_ternary"],
    ),
    ("CUDA-P18", &["encode_gemv_tq2_cached", "InvalidDimensions"]),
];

/// The plan's lines, each run joined back into one logical line (the
/// constant wraps its runs over several source lines with `\` escapes, so
/// every run already is one line; this only drops the header and blanks).
fn plan_runs() -> Vec<&'static str> {
    CUDA_HARDWARE_PARITY_PLAN
        .lines()
        .filter(|line| line.starts_with("[ ] CUDA-P"))
        .collect()
}

#[test]
fn cuda_hardware_parity_plan_names_every_required_run() {
    let runs = plan_runs();
    assert_eq!(
        runs.len(),
        REQUIRED_RUNS.len(),
        "every run in the plan must be one of the required runs, and vice versa"
    );
    for (id, phrases) in REQUIRED_RUNS {
        let line = runs
            .iter()
            .find(|line| line.contains(&format!("{id} ")))
            .unwrap_or_else(|| panic!("the plan must list {id}"));
        for phrase in *phrases {
            assert!(
                line.contains(phrase),
                "{id} must mention `{phrase}`: {line}"
            );
        }
        assert!(
            line.contains(COS_THRESHOLD),
            "{id} must state its acceptance threshold (`{COS_THRESHOLD}`): {line}"
        );
        assert!(
            line.starts_with("[ ] "),
            "{id} must stay an unchecked plan item — this file is the plan, not a record of results"
        );
    }
}

#[test]
fn cuda_hardware_parity_plan_self_skips() {
    const TEST_NAME: &str =
        "oxibonsai-kernels::cuda_hybrid_parity_plan::cuda_hardware_parity_plan_self_skips";
    // The plan is executed by hand on a CUDA device and has no automated
    // form, so this test skips on every host — including one with a CUDA
    // device — and records exactly that. It must never record
    // `"executed": true`.
    eprintln!(
        "SKIP: {} CUDA-hardware parity runs are manual and this test never executes them \
         (see the file header for the 0.2.4 status); checklist:\n{CUDA_HARDWARE_PARITY_PLAN}",
        plan_runs().len()
    );
    record_skipped(Capability::CudaHardware, TEST_NAME);
}
