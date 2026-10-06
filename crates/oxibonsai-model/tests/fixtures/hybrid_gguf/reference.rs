//! The independent f64 reference forward of the whole `qwen35` stack:
//! embedding with the inverse Hadamard fold, per-layer full attention and
//! Gated-DeltaNet linear attention, the shared FFN, the final norm and the
//! LM head, computed over the exact pre-quantization weight values of a plan.

use std::collections::BTreeMap;

use super::math::{
    apply_partial_rope_f64, causal_conv1d_step_f64, fwht_forward_signed_f64,
    fwht_inverse_signed_f64, gated_rms_norm_head_f64, gdn_step_fused_f64, l2_norm_f64,
    rms_norm_f64, sigmoid_f64, silu_f64, VHeadMap,
};
use super::plan::{tname, FixturePlan};
use super::spec::{
    HybridFixtureSpec, BLOCK_COUNT, N_HEAD, N_KV, RMS_EPS, ROPE_DIM, ROPE_FREQ_BASE,
    SSM_CONV_KERNEL, T_TOKENS, VOCAB,
};
use super::Xorshift64Star;

/// Precomputed f64 forward-pass outputs (the reference model's answer for
/// this fixture's fixed token sequence), stored once at build time so a
/// consuming test never has to re-derive it.
pub struct ReferenceForward {
    pub token_ids: Vec<usize>,
    /// `[T_TOKENS][vocab]` logits.
    pub logits: Vec<Vec<f64>>,
    /// `[T_TOKENS][hidden]` pre-final-norm residual stream, for diagnostics.
    /// Equal to `per_layer_hidden[BLOCK_COUNT - 1]`; kept as its own field
    /// since it predates `per_layer_hidden` and existing callers read it.
    pub final_hidden: Vec<Vec<f64>>,
    /// `per_layer_hidden[layer]` is the residual stream after block
    /// `layer`, `[t][hidden]` row-major — the same shape and meaning as
    /// `oxibonsai_model::hybrid::forward::LayerDump.layers[layer]`, so a
    /// caller with a real `LayerDump` can compare layer by layer instead of
    /// only at the end.
    pub per_layer_hidden: Vec<Vec<f64>>,
}

// ═════════════════════════════════════════════════════════════════════════
// 11. The f64 reference forward driver
// ═════════════════════════════════════════════════════════════════════════

/// Fetch tensor `name`'s planned values as `f64` (exact widening).
///
/// Panics (rather than silently returning `vec![]`) if `name` was never
/// planned: this is the f64 reference model's own ground truth, so a
/// name typo in the forward driver below must fail loudly, not degrade
/// into an all-zero output that `zip`-truncated arithmetic would silently
/// propagate. Every call site is reached only from a branch that matches
/// `build_plan`'s own construction of that name (full-attention names
/// only under `is_full_attention`, `ssm_*` names only under its `else`),
/// so this is a real invariant, not merely convenient.
fn f64_values(plan: &FixturePlan, name: &str) -> Vec<f64> {
    let planned = plan.tensors.get(name).unwrap_or_else(|| {
        panic!("hybrid fixture reference forward: tensor '{name}' was never planned by build_plan")
    });
    planned.values.iter().map(|&v| v as f64).collect()
}

/// `y = W^T x` for a row-major `[in, out]`-shaped weight (`W` stored as
/// `out` rows of `in` values each, matching GGUF's `[ne0=in, ne1=out]`
/// convention): `y[o] = sum_i W[o][i] * x[i]`.
fn matmul_row_major(w: &[f64], in_dim: usize, out_dim: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        y[o] = row.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum();
    }
    y
}

struct HadamardRuntime {
    block_size: usize,
    signs: BTreeMap<usize, Vec<f64>>,
}

impl HadamardRuntime {
    /// Panics on a width with no configured sign vector, rather than
    /// silently skipping the rotation: `rotate` is only ever called on the
    /// three widths `build_plan` populates `signs` for
    /// (`HIDDEN`/`attn_output_input_width` (== `inner_size` by this
    /// fixture's chosen dims) /`FFN`), so reaching `None` here means a call
    /// site and `build_plan`'s sign-width set have drifted apart — exactly
    /// the class of silent-wrong-math bug design §3.4 warns a skipped
    /// rotation is, so it must fail loudly rather than let the caller run
    /// un-rotated data through a folded matmul unnoticed.
    fn rotate(&self, x: &[f64]) -> Vec<f64> {
        let width = x.len();
        match self.signs.get(&width) {
            Some(s) => fwht_forward_signed_f64(x, s, self.block_size),
            None => panic!(
                "hybrid fixture reference forward: no Hadamard sign vector configured for \
                 width {width}; rotate() must only be called on a folded activation's width"
            ),
        }
    }

    fn inverse_embedding(&self, x: &[f64]) -> Vec<f64> {
        let width = x.len();
        match self.signs.get(&width) {
            Some(s) => fwht_inverse_signed_f64(x, s, self.block_size),
            None => panic!(
                "hybrid fixture reference forward: no Hadamard sign vector configured for \
                 width {width}; inverse_embedding() must only be called on the embedding width"
            ),
        }
    }
}

pub(super) fn run_reference_forward(
    spec: &HybridFixtureSpec,
    plan: &FixturePlan,
) -> ReferenceForward {
    let dims = plan.dims;
    let hadamard = if spec.hadamard {
        Some(HadamardRuntime {
            block_size: dims.hadamard_block,
            signs: plan.signs.clone(),
        })
    } else {
        None
    };
    let rotate = |x: &[f64]| -> Vec<f64> {
        match &hadamard {
            Some(h) => h.rotate(x),
            None => x.to_vec(),
        }
    };

    let mut rng = Xorshift64Star::new(spec.seed ^ 0xC0FF_EE00_D15E_A5E5);
    let token_ids: Vec<usize> = (0..T_TOKENS).map(|_| rng.next_usize_below(VOCAB)).collect();

    let full_output_in = dims.attn_output_input_width();
    let vhead_map = VHeadMap::new(dims.n_k_heads, dims.n_v_heads);

    // Per-linear-layer recurrent state.
    let linear_layers: Vec<usize> = (0..BLOCK_COUNT)
        .filter(|&l| !dims.is_full_attention(l))
        .collect();
    let mut conv_state: BTreeMap<usize, Vec<Vec<f64>>> = linear_layers
        .iter()
        .map(|&l| (l, vec![vec![0.0f64; SSM_CONV_KERNEL - 1]; dims.conv_dim]))
        .collect();
    let mut gdn_state: BTreeMap<usize, Vec<f64>> = linear_layers
        .iter()
        .map(|&l| {
            (
                l,
                vec![0.0f64; dims.n_v_heads * dims.head_v_dim * dims.head_k_dim],
            )
        })
        .collect();

    // Per-full-layer KV history.
    let full_layers: Vec<usize> = (0..BLOCK_COUNT)
        .filter(|&l| dims.is_full_attention(l))
        .collect();
    let mut k_hist: BTreeMap<usize, Vec<Vec<Vec<f64>>>> = full_layers
        .iter()
        .map(|&l| (l, vec![Vec::new(); N_KV]))
        .collect();
    let mut v_hist: BTreeMap<usize, Vec<Vec<Vec<f64>>>> = full_layers
        .iter()
        .map(|&l| (l, vec![Vec::new(); N_KV]))
        .collect();

    let token_embd = f64_values(plan, "token_embd.weight");
    let output_norm_w = f64_values(plan, "output_norm.weight");
    let output_w = f64_values(plan, "output.weight");

    let mut logits = Vec::with_capacity(T_TOKENS);
    let mut final_hidden = Vec::with_capacity(T_TOKENS);
    // `per_layer_hidden[layer]` is `[t][hidden]` row-major, matching
    // `oxibonsai_model::hybrid::forward::LayerDump.layers[layer]` exactly:
    // capturing the residual stream after every block, not only the final
    // one, so a mismatch against the real loader localises to one block
    // instead of only showing up in the final logits.
    let mut per_layer_hidden: Vec<Vec<f64>> = (0..BLOCK_COUNT)
        .map(|_| Vec::with_capacity(T_TOKENS * dims.hidden))
        .collect();

    for (pos, &tok) in token_ids.iter().enumerate() {
        let embed_row: Vec<f64> = token_embd[tok * dims.hidden..(tok + 1) * dims.hidden].to_vec();
        let mut h = match &hadamard {
            Some(hr) => hr.inverse_embedding(&embed_row),
            None => embed_row,
        };

        for layer in 0..BLOCK_COUNT {
            let residual = h.clone();
            let attn_norm_w = f64_values(plan, &tname(layer, "attn_norm.weight"));
            let normed = rms_norm_f64(&h, &attn_norm_w, RMS_EPS);

            if dims.is_full_attention(layer) {
                let a_rot = rotate(&normed);
                let wq = f64_values(plan, &tname(layer, "attn_q.weight"));
                let wk = f64_values(plan, &tname(layer, "attn_k.weight"));
                let wv = f64_values(plan, &tname(layer, "attn_v.weight"));
                let qfull = matmul_row_major(&wq, dims.hidden, N_HEAD * dims.head_dim * 2, &a_rot);
                let kfull = matmul_row_major(&wk, dims.hidden, N_KV * dims.head_dim, &a_rot);
                let vfull = matmul_row_major(&wv, dims.hidden, N_KV * dims.head_dim, &a_rot);

                let q_norm_w = f64_values(plan, &tname(layer, "attn_q_norm.weight"));
                let k_norm_w = f64_values(plan, &tname(layer, "attn_k_norm.weight"));

                // K/V for this position, per KV head, RMS-normed + RoPE'd.
                for kv in 0..N_KV {
                    let mut k_h = kfull[kv * dims.head_dim..(kv + 1) * dims.head_dim].to_vec();
                    k_h = rms_norm_f64(&k_h, &k_norm_w, RMS_EPS);
                    apply_partial_rope_f64(&mut k_h, pos, ROPE_DIM, ROPE_FREQ_BASE);
                    let v_h = vfull[kv * dims.head_dim..(kv + 1) * dims.head_dim].to_vec();
                    // `k_hist`/`v_hist` were pre-populated with one N_KV-slot
                    // vector for every layer satisfying `is_full_attention`,
                    // using that same predicate; we are inside the branch
                    // that predicate selected, so `layer` is always present,
                    // and `kv < N_KV` by the loop bound above.
                    k_hist
                        .get_mut(&layer)
                        .expect("full-attention layer must be present in k_hist")
                        .get_mut(kv)
                        .expect("kv head index must be < N_KV")
                        .push(k_h);
                    v_hist
                        .get_mut(&layer)
                        .expect("full-attention layer must be present in v_hist")
                        .get_mut(kv)
                        .expect("kv head index must be < N_KV")
                        .push(v_h);
                }

                let group_size = N_HEAD / N_KV;
                let scale = 1.0 / (dims.head_dim as f64).sqrt();
                let mut attn_out = vec![0.0f64; N_HEAD * dims.head_dim];
                let mut gate_all = vec![0.0f64; N_HEAD * dims.head_dim];
                for head in 0..N_HEAD {
                    let base = head * 2 * dims.head_dim;
                    let mut q_h = qfull[base..base + dims.head_dim].to_vec();
                    let gate_h = &qfull[base + dims.head_dim..base + 2 * dims.head_dim];
                    q_h = rms_norm_f64(&q_h, &q_norm_w, RMS_EPS);
                    apply_partial_rope_f64(&mut q_h, pos, ROPE_DIM, ROPE_FREQ_BASE);

                    let kv_head = head / group_size;
                    let ks = &k_hist[&layer][kv_head];
                    let vs = &v_hist[&layer][kv_head];
                    let scores: Vec<f64> = ks
                        .iter()
                        .map(|k| scale * k.iter().zip(&q_h).map(|(&ki, &qi)| ki * qi).sum::<f64>())
                        .collect();
                    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                    let sum_s: f64 = exp_s.iter().sum();
                    let weights: Vec<f64> = exp_s.iter().map(|&e| e / sum_s).collect();
                    let mut o_h = vec![0.0f64; dims.head_dim];
                    for (t, v) in vs.iter().enumerate() {
                        for d in 0..dims.head_dim {
                            o_h[d] += weights[t] * v[d];
                        }
                    }
                    attn_out[head * dims.head_dim..(head + 1) * dims.head_dim]
                        .copy_from_slice(&o_h);
                    gate_all[head * dims.head_dim..(head + 1) * dims.head_dim]
                        .copy_from_slice(gate_h);
                }
                debug_assert_eq!(attn_out.len(), full_output_in);
                let gated: Vec<f64> = attn_out
                    .iter()
                    .zip(&gate_all)
                    .map(|(&a, &g)| a * sigmoid_f64(g))
                    .collect();
                let o_rot = rotate(&gated);
                let wo = f64_values(plan, &tname(layer, "attn_output.weight"));
                let attn_result = matmul_row_major(&wo, full_output_in, dims.hidden, &o_rot);
                h = residual
                    .iter()
                    .zip(&attn_result)
                    .map(|(&r, &a)| r + a)
                    .collect();
            } else {
                let a_rot = rotate(&normed);
                let wqkv = f64_values(plan, &tname(layer, "attn_qkv.weight"));
                let wgate = f64_values(plan, &tname(layer, "attn_gate.weight"));
                let mut qkv = matmul_row_major(&wqkv, dims.hidden, dims.conv_dim, &a_rot);
                let z_tiled = matmul_row_major(&wgate, dims.hidden, dims.inner_size, &a_rot);

                // ssm_alpha/ssm_beta are NOT folded: consume the unrotated
                // `normed` (design §3.4's "easy-to-miss" trap).
                let w_alpha = f64_values(plan, &tname(layer, "ssm_alpha.weight"));
                let w_beta = f64_values(plan, &tname(layer, "ssm_beta.weight"));
                let alpha_raw_tiled =
                    matmul_row_major(&w_alpha, dims.hidden, dims.n_v_heads, &normed);
                let beta_raw_tiled =
                    matmul_row_major(&w_beta, dims.hidden, dims.n_v_heads, &normed);

                let conv_w_flat = f64_values(plan, &tname(layer, "ssm_conv1d.weight"));
                // ne=[kc, conv_dim] row-major-per-channel: channel c's kc
                // weights are conv_w_flat[c*kc .. c*kc+kc].
                let conv_w: Vec<Vec<f64>> = (0..dims.conv_dim)
                    .map(|c| conv_w_flat[c * SSM_CONV_KERNEL..(c + 1) * SSM_CONV_KERNEL].to_vec())
                    .collect();
                // `conv_state` was pre-populated for every layer NOT
                // satisfying `is_full_attention`, using that same
                // predicate; we are inside the `else` of that branch.
                let state = conv_state
                    .get_mut(&layer)
                    .expect("linear-attention layer must be present in conv_state");
                qkv = causal_conv1d_step_f64(state, &qkv, &conv_w, dims.conv_dim, SSM_CONV_KERNEL);
                qkv = qkv.into_iter().map(silu_f64).collect();

                let q_width = dims.n_k_heads * dims.head_k_dim;
                let k_width = dims.n_k_heads * dims.head_k_dim;
                let (q_all, rest) = qkv.split_at(q_width);
                let (k_all, v_tiled) = rest.split_at(k_width);

                // L2-norm the joint Q/K region, per K-head.
                let mut q_normed = vec![0.0f64; q_width];
                let mut k_normed = vec![0.0f64; k_width];
                for kh in 0..dims.n_k_heads {
                    let qs = &q_all[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let ks = &k_all[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let qn = l2_norm_f64(qs, 1e-6);
                    let kn = l2_norm_f64(ks, 1e-6);
                    q_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim].copy_from_slice(&qn);
                    k_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim].copy_from_slice(&kn);
                }

                let ssm_a = f64_values(plan, &tname(layer, "ssm_a"));
                let ssm_dt_bias = f64_values(plan, &tname(layer, "ssm_dt.bias"));

                // Same invariant as `conv_state` above: `gdn_state` is
                // pre-populated for every linear-attention layer.
                let state = gdn_state
                    .get_mut(&layer)
                    .expect("linear-attention layer must be present in gdn_state");
                let mut out_grouped = vec![0.0f64; dims.n_v_heads * dims.head_v_dim];
                for m in 0..dims.n_v_heads {
                    let tiled_j = vhead_map.tiled(m);
                    let kh = vhead_map.k_head(m);
                    let q_h = &q_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let k_h = &k_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let v_h = &v_tiled[tiled_j * dims.head_v_dim..(tiled_j + 1) * dims.head_v_dim];
                    let state_slab = &mut state[m * dims.head_v_dim * dims.head_k_dim
                        ..(m + 1) * dims.head_v_dim * dims.head_k_dim];
                    let (out_h, _decay, _delta) = gdn_step_fused_f64(
                        state_slab,
                        q_h,
                        k_h,
                        v_h,
                        alpha_raw_tiled[tiled_j],
                        beta_raw_tiled[tiled_j],
                        ssm_dt_bias[tiled_j],
                        ssm_a[tiled_j],
                        dims.head_k_dim,
                        dims.head_v_dim,
                    );
                    out_grouped[m * dims.head_v_dim..(m + 1) * dims.head_v_dim]
                        .copy_from_slice(&out_h);
                }

                // z is TILED (same as v); re-index to GROUPED to align with
                // `out_grouped` before the gated norm (design §3.4).
                let ssm_norm_w = f64_values(plan, &tname(layer, "ssm_norm.weight"));
                let mut gated_out = vec![0.0f64; dims.n_v_heads * dims.head_v_dim];
                for m in 0..dims.n_v_heads {
                    let tiled_j = vhead_map.tiled(m);
                    let o_h = &out_grouped[m * dims.head_v_dim..(m + 1) * dims.head_v_dim];
                    let z_h = &z_tiled[tiled_j * dims.head_v_dim..(tiled_j + 1) * dims.head_v_dim];
                    let g_h = gated_rms_norm_head_f64(o_h, z_h, &ssm_norm_w, RMS_EPS);
                    gated_out[m * dims.head_v_dim..(m + 1) * dims.head_v_dim].copy_from_slice(&g_h);
                }

                // The recurrent state is kept GROUPED (design §3.6), so
                // `ssm_out`'s fold degenerates to a plain `rotate`.
                let o_rot = rotate(&gated_out);
                let w_out = f64_values(plan, &tname(layer, "ssm_out.weight"));
                let ssm_result = matmul_row_major(&w_out, dims.inner_size, dims.hidden, &o_rot);
                h = residual
                    .iter()
                    .zip(&ssm_result)
                    .map(|(&r, &s)| r + s)
                    .collect();
            }

            // ── Shared FFN ───────────────────────────────────────────────
            let residual2 = h.clone();
            let post_norm_w = f64_values(plan, &tname(layer, "post_attention_norm.weight"));
            let f = rms_norm_f64(&h, &post_norm_w, RMS_EPS);
            let f_rot = rotate(&f);
            let w_gate = f64_values(plan, &tname(layer, "ffn_gate.weight"));
            let w_up = f64_values(plan, &tname(layer, "ffn_up.weight"));
            let gate_out = matmul_row_major(&w_gate, dims.hidden, dims.ffn, &f_rot);
            let up_out = matmul_row_major(&w_up, dims.hidden, dims.ffn, &f_rot);
            let m_vec: Vec<f64> = gate_out
                .iter()
                .zip(&up_out)
                .map(|(&g, &u)| silu_f64(g) * u)
                .collect();
            let m_rot = rotate(&m_vec);
            let w_down = f64_values(plan, &tname(layer, "ffn_down.weight"));
            let down_out = matmul_row_major(&w_down, dims.ffn, dims.hidden, &m_rot);
            h = residual2
                .iter()
                .zip(&down_out)
                .map(|(&r, &d)| r + d)
                .collect();

            per_layer_hidden[layer].extend_from_slice(&h);
        }

        final_hidden.push(h.clone());
        let normed_final = rms_norm_f64(&h, &output_norm_w, RMS_EPS);
        let final_rot = rotate(&normed_final);
        let token_logits = matmul_row_major(&output_w, dims.hidden, VOCAB, &final_rot);
        logits.push(token_logits);
    }

    ReferenceForward {
        token_ids,
        logits,
        final_hidden,
        per_layer_hidden,
    }
}
