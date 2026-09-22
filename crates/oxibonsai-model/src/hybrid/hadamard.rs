//! The Hadamard rotation hook for Bonsai 2's folded weights (M-15, design
//! §3.4 / §3.5).
//!
//! # What "folded" means
//!
//! PrismML's Bonsai 2 GGUFs store most weight matrices in a *rotated* basis:
//! `W_stored = W · H⁻¹` for the blockwise sign-flipped Walsh–Hadamard
//! transform `H`. Recovering `y = W x` therefore means rotating the
//! **activation** before the matmul:
//!
//! ```text
//! x' = FWHT_1024( x ⊙ signs[width] ) / sqrt(1024)      (per 1024-wide block)
//! y  = W_stored x'
//! ```
//!
//! and, for the one *inverse* tensor (`token_embd.weight`, whose rows are
//! stored rotated), un-rotating the looked-up row the other way round —
//! FWHT **then** signs.
//!
//! # Where the hook lives, and why not in `LinearLayer::forward`
//!
//! The transform is a property of the **activation**, not of the weight.
//! A full-attention layer feeds one rotated 5120-wide activation to `attn_q`,
//! `attn_k` *and* `attn_v`; a linear layer feeds one to `attn_qkv` *and*
//! `attn_gate`. Hiding the rotation behind each matmul would run it 7 times
//! per full layer and 6 times per linear layer instead of 4 — and the fork
//! already calls the rotation out as a measurable non-matmul cost at batch 1.
//!
//! So the hook is explicit and the block forward owns a [`HadamardScratch`]:
//! call [`HadamardHook::rotate`] **once** per (activation, width), then pass
//! the returned slice to every folded matmul that consumes it. This is the
//! "caller-discipline memoisation" that design §2.2/§3.4 specify in place of
//! the fork's pointer-keyed `hadamard_memo`; `oxibonsai-kernels`'
//! `hadamard` module deliberately exposes no memo primitive for the same
//! reason. [`HadamardScratch::transforms`] counts the passes so a test can
//! *prove* the discipline instead of documenting it
//! (`rotation_runs_once_per_activation_not_once_per_weight`).
//!
//! # What is NOT folded (the silent-accuracy trap)
//!
//! `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias` and every
//! norm are absent from `prism.hadamard.weight_names` (the real files list
//! exactly 401 folded names). They consume the **un-rotated** activation.
//! Feeding them `x'` is wrong math that still produces plausible logits, so
//! [`HadamardHook::is_folded`] is the check a forward must make rather than
//! assuming "this is a Bonsai 2 layer, rotate everything".

use std::sync::Arc;

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::hadamard_config::HadamardConfig;
use oxibonsai_kernels::hadamard::{
    fwht_forward_signed, fwht_forward_signed_batch, fwht_inverse_signed,
};

use crate::error::{ModelError, ModelResult};
use crate::hybrid::vhead_map::VHeadMap;

/// The blockwise Hadamard rotation of one model, bound to its
/// `prism.hadamard.*` contract.
///
/// Cheap to clone (one `Arc`), so every block can hold one.
#[derive(Debug, Clone)]
pub struct HadamardHook {
    config: Arc<HadamardConfig>,
}

/// Per-token rotation scratch owned by the block forward.
///
/// Holds one buffer per activation width the model actually rotates — three
/// for `qwen35` (27B: 5120 for the pre-attention and pre-FFN norms' output,
/// 6144 for `attn_output`/`ssm_out`'s input, 17408 for `ffn_down`'s input).
/// Allocated once per sequence, reused every token.
#[derive(Debug, Clone)]
pub struct HadamardScratch {
    /// `(width, buffer)`, deduplicated and ordered by width.
    buffers: Vec<(usize, Vec<f32>)>,
    /// Blockwise-FWHT passes performed since the last
    /// [`HadamardScratch::reset_counter`].
    transforms: usize,
}

impl HadamardScratch {
    /// Allocate the scratch for `config`'s three rotated activation widths.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] when a width is zero (a degenerate
    /// config that would otherwise allocate an empty buffer and silently
    /// rotate nothing).
    pub fn new(config: &HybridConfig) -> ModelResult<Self> {
        Self::for_widths(&rotated_widths(config))
    }

    /// Allocate a scratch for an explicit width list (deduplicated).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] when any width is zero.
    pub fn for_widths(widths: &[usize]) -> ModelResult<Self> {
        let mut buffers: Vec<(usize, Vec<f32>)> = Vec::with_capacity(widths.len());
        for &width in widths {
            if width == 0 {
                return Err(ModelError::ShapeInvariant {
                    tensor: "HadamardScratch width".to_string(),
                    expected: "nonzero".to_string(),
                    actual: "0".to_string(),
                });
            }
            if !buffers.iter().any(|(w, _)| *w == width) {
                buffers.push((width, vec![0.0; width]));
            }
        }
        buffers.sort_by_key(|(w, _)| *w);
        Ok(Self {
            buffers,
            transforms: 0,
        })
    }

    /// Widths this scratch can rotate, ascending.
    #[must_use]
    pub fn widths(&self) -> Vec<usize> {
        self.buffers.iter().map(|(w, _)| *w).collect()
    }

    /// Whether `width` has a buffer.
    #[must_use]
    pub fn has_width(&self, width: usize) -> bool {
        self.buffers.iter().any(|(w, _)| *w == width)
    }

    /// Blockwise-FWHT passes performed so far.
    ///
    /// The measure of the caller-discipline memo: a correct block forward
    /// performs **four** per layer regardless of how many folded matmuls it
    /// drives.
    #[inline]
    #[must_use]
    pub fn transforms(&self) -> usize {
        self.transforms
    }

    /// Zero the transform counter (per-token or per-test accounting).
    #[inline]
    pub fn reset_counter(&mut self) {
        self.transforms = 0;
    }

    /// The last rotation written for `width`, if any.
    ///
    /// Returns the buffer contents regardless of which activation produced
    /// them — it is the caller that knows which activation is live, which is
    /// exactly why this is not a cache lookup keyed by pointer identity.
    #[must_use]
    pub fn peek(&self, width: usize) -> Option<&[f32]> {
        self.buffers
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, buf)| buf.as_slice())
    }

    /// Claim the buffer for `width` for one new transform, counting it.
    ///
    /// Bumps [`HadamardScratch::transforms`] *before* handing out the
    /// buffer, because the returned `&mut` borrows `self` for as long as the
    /// caller holds it. Every fallible precondition is checked by the caller
    /// before claiming, so the count never runs ahead of the work.
    fn claim(&mut self, width: usize) -> ModelResult<&mut [f32]> {
        let available: Vec<usize> = self.buffers.iter().map(|(w, _)| *w).collect();
        let slot = self
            .buffers
            .iter_mut()
            .find(|(w, _)| *w == width)
            .map(|(_, buf)| buf);
        match slot {
            Some(buf) => {
                self.transforms += 1;
                Ok(buf.as_mut_slice())
            }
            None => Err(ModelError::ShapeInvariant {
                tensor: "HadamardScratch".to_string(),
                expected: format!("a scratch buffer for width {width}"),
                actual: format!("configured widths {available:?}"),
            }),
        }
    }
}

/// The activation widths a `qwen35` forward rotates (design §3.4).
///
/// * `hidden_size` — the pre-attention and pre-FFN RMSNorm outputs, shared by
///   `attn_q`/`attn_k`/`attn_v` (or `attn_qkv`/`attn_gate`) and by
///   `ffn_gate`/`ffn_up`, and by the LM head on the final `output_norm`;
/// * `num_attention_heads * head_dim` — `attn_output`'s input, which for the
///   27B (24 × 256 = 6144) coincides with `ssm_inner_size`, `ssm_out`'s
///   input;
/// * `intermediate_size` — `ffn_down`'s input.
#[must_use]
pub fn rotated_widths(config: &HybridConfig) -> Vec<usize> {
    vec![
        config.base.hidden_size,
        config.base.num_attention_heads * config.base.head_dim,
        config.ssm_inner_size,
        config.base.intermediate_size,
    ]
}

impl HadamardHook {
    /// Bind a hook to an already-parsed and validated contract.
    #[must_use]
    pub fn new(config: Arc<HadamardConfig>) -> Self {
        Self { config }
    }

    /// The contract this hook applies.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &HadamardConfig {
        &self.config
    }

    /// FWHT block width (27B: 1024).
    #[inline]
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Whether `name`'s input activation must be rotated before the matmul
    /// that consumes it.
    #[inline]
    #[must_use]
    pub fn is_folded(&self, name: &str) -> bool {
        self.config.is_folded(name)
    }

    /// Whether `name`'s lookup result needs the inverse transform
    /// (`token_embd.weight`).
    #[inline]
    #[must_use]
    pub fn is_inverse(&self, name: &str) -> bool {
        self.config.is_inverse(name)
    }

    /// Folded tensor names in this contract (27B: 401).
    #[inline]
    #[must_use]
    pub fn folded_count(&self) -> usize {
        self.config.folded.len()
    }

    /// Check at **load** that every width this model will rotate has a sign
    /// vector and divides the FWHT block size (M-15's correction: the
    /// metadata validation is fatal, not advisory).
    ///
    /// Without this, a truncated or mis-declared `sign_widths` table would
    /// first be noticed by the *token* that happens to reach `ffn_down`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Core`] carrying `BonsaiError::HadamardContract` for a
    /// missing sign vector; [`ModelError::ShapeInvariant`] when a width is
    /// not a whole number of FWHT blocks.
    pub fn validate_widths(&self, config: &HybridConfig) -> ModelResult<()> {
        let block = self.block_size();
        for width in rotated_widths(config) {
            let signs = self.config.signs_for(width).map_err(ModelError::Core)?;
            if signs.len() != width {
                return Err(ModelError::ShapeMismatch {
                    name: format!("prism.hadamard.sign_values[{width}]"),
                    expected: vec![width],
                    actual: vec![signs.len()],
                });
            }
            if block == 0 || !width.is_multiple_of(block) {
                return Err(ModelError::ShapeInvariant {
                    tensor: format!("rotated activation width {width}"),
                    expected: format!("a whole multiple of the FWHT block size {block}"),
                    actual: format!("{width} % {block} != 0"),
                });
            }
        }
        Ok(())
    }

    /// Rotate `src` (exactly `width` wide) into `scratch`'s buffer for
    /// `width` and return it: `signs`, then blockwise FWHT.
    ///
    /// Call **once** per (activation, width) per layer and reuse the
    /// returned slice for every folded matmul that consumes that activation
    /// — see the module docs.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `src.len() != width`;
    /// [`ModelError::ShapeInvariant`] when the scratch has no buffer for
    /// `width`; [`ModelError::Core`] when the contract has no sign vector for
    /// it; [`ModelError::Kernel`] from the transform itself.
    pub fn rotate<'s>(
        &self,
        src: &[f32],
        width: usize,
        scratch: &'s mut HadamardScratch,
    ) -> ModelResult<&'s [f32]> {
        if src.len() != width {
            return Err(ModelError::ShapeMismatch {
                name: format!("hadamard rotate input (width {width})"),
                expected: vec![width],
                actual: vec![src.len()],
            });
        }
        let signs = self.config.signs_for(width).map_err(ModelError::Core)?;
        let block = self.config.block_size;
        let buffer = scratch.claim(width)?;
        buffer.copy_from_slice(src);
        fwht_forward_signed(buffer, signs, block).map_err(ModelError::Kernel)?;
        Ok(buffer)
    }

    /// `ssm_out`'s variant: re-index the v-heads tiled → grouped, then
    /// signs, then FWHT.
    ///
    /// With the recurrent state kept in grouped order (design §3.3 — what
    /// this crate does), the Gated-DeltaNet output is *already* grouped and
    /// this degenerates to [`HadamardHook::rotate`]; the regrouping path
    /// exists for a caller that holds a tiled-order output, and
    /// `rotate_gdn_out_matches_rotate_after_regrouping` pins the two forms
    /// together.
    ///
    /// # Errors
    ///
    /// As [`HadamardHook::rotate`], plus [`ModelError::ShapeMismatch`] from
    /// [`VHeadMap::gather_grouped`].
    pub fn rotate_gdn_out<'s>(
        &self,
        src_tiled: &[f32],
        map: &VHeadMap,
        head_v_dim: usize,
        scratch: &'s mut HadamardScratch,
    ) -> ModelResult<&'s [f32]> {
        let width = map.n_v_heads() * head_v_dim;
        if map.is_identity() {
            return self.rotate(src_tiled, width, scratch);
        }
        if src_tiled.len() != width {
            return Err(ModelError::ShapeMismatch {
                name: format!("hadamard rotate_gdn_out input (width {width})"),
                expected: vec![width],
                actual: vec![src_tiled.len()],
            });
        }
        let signs = self.config.signs_for(width).map_err(ModelError::Core)?;
        let block = self.config.block_size;
        let buffer = scratch.claim(width)?;
        map.gather_grouped(src_tiled, head_v_dim, buffer)?;
        fwht_forward_signed(buffer, signs, block).map_err(ModelError::Kernel)?;
        Ok(buffer)
    }

    /// Rotate `rows` activations of `width` each, in place — the prefill
    /// shape, where one width is shared by a whole chunk of tokens.
    ///
    /// Rotating a `[rows][width]` batch in one call is the batched analogue
    /// of the once-per-activation rule: the same sign vector and the same
    /// fused scale pass serve every row.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `x.len() != rows * width`;
    /// [`ModelError::Core`] for a missing sign vector;
    /// [`ModelError::Kernel`] from the transform.
    pub fn rotate_rows_in_place(
        &self,
        x: &mut [f32],
        width: usize,
        rows: usize,
    ) -> ModelResult<()> {
        let expected = width
            .checked_mul(rows)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "hadamard rotate_rows_in_place".to_string(),
                expected: "rows * width representable as usize".to_string(),
                actual: format!("rows = {rows}, width = {width}"),
            })?;
        if x.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: format!("hadamard rotate_rows_in_place ({rows} x {width})"),
                expected: vec![expected],
                actual: vec![x.len()],
            });
        }
        let signs = self.config.signs_for(width).map_err(ModelError::Core)?;
        fwht_forward_signed_batch(x, signs, self.config.block_size, rows)
            .map_err(ModelError::Kernel)
    }

    /// Un-rotate one `token_embd.weight` row in place: FWHT, **then** signs
    /// (design §3.5 — the opposite order to [`HadamardHook::rotate`]).
    ///
    /// Cache nothing: the row changes every token, and one row is 5120
    /// values (40 blocks), which is negligible beside the layer stack.
    ///
    /// # Errors
    ///
    /// [`ModelError::Core`] when the contract has no sign vector for
    /// `row.len()`; [`ModelError::Kernel`] from the transform.
    pub fn inverse_embedding(&self, row: &mut [f32]) -> ModelResult<()> {
        let signs = self.config.signs_for(row.len()).map_err(ModelError::Core)?;
        fwht_inverse_signed(row, signs, self.config.block_size).map_err(ModelError::Kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::{bonsai2_config, bonsai2_hadamard};

    const HIDDEN: usize = 5120;
    const HEADS_WIDTH: usize = 6144;
    const FFN: usize = 17408;

    fn ramp(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.013 + seed).sin())
            .collect()
    }

    #[test]
    fn scratch_covers_exactly_the_three_qwen35_widths() {
        let config = bonsai2_config();
        let scratch = HadamardScratch::new(&config).expect("valid widths");
        assert_eq!(scratch.widths(), vec![HIDDEN, HEADS_WIDTH, FFN]);
        // 6144 is both `n_heads * head_dim` and `ssm_inner_size`; the
        // scratch must hold one buffer for it, not two.
        assert_eq!(rotated_widths(&config).len(), 4);
        assert!(scratch.has_width(HEADS_WIDTH));
        assert!(!scratch.has_width(1024));
    }

    #[test]
    fn rotate_matches_the_kernel_directly() {
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let mut scratch = HadamardScratch::new(&bonsai2_config()).expect("valid widths");
        let x = ramp(HIDDEN, 0.25);

        let rotated = hook.rotate(&x, HIDDEN, &mut scratch).expect("rotate");

        let mut reference = x.clone();
        let signs = hook.config().signs_for(HIDDEN).expect("signs");
        fwht_forward_signed(&mut reference, signs, hook.block_size()).expect("kernel");
        assert_eq!(rotated, reference.as_slice());
        assert_eq!(scratch.transforms(), 1);
    }

    #[test]
    fn inverse_embedding_undoes_rotate() {
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let mut scratch = HadamardScratch::new(&bonsai2_config()).expect("valid widths");
        let x = ramp(HIDDEN, 1.5);

        let mut round_trip = hook
            .rotate(&x, HIDDEN, &mut scratch)
            .expect("rotate")
            .to_vec();
        hook.inverse_embedding(&mut round_trip).expect("inverse");

        for (i, (got, want)) in round_trip.iter().zip(x.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "element {i}: got {got}, want {want}"
            );
        }
    }

    /// Design §3.4: `rotate_gdn_out` is `rotate` once the activation is in
    /// grouped order, and performs the regroup itself when it is not.
    #[test]
    fn rotate_gdn_out_matches_rotate_after_regrouping() {
        let config = bonsai2_config();
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let map = VHeadMap::new(config.n_k_heads(), config.n_v_heads()).expect("valid geometry");
        let head_v_dim = config.head_v_dim();
        let tiled = ramp(HEADS_WIDTH, 0.75);

        let mut grouped = vec![0.0f32; HEADS_WIDTH];
        map.gather_grouped(&tiled, head_v_dim, &mut grouped)
            .expect("gather");

        let mut scratch_a = HadamardScratch::new(&config).expect("valid widths");
        let via_rotate = hook
            .rotate(&grouped, HEADS_WIDTH, &mut scratch_a)
            .expect("rotate")
            .to_vec();

        let mut scratch_b = HadamardScratch::new(&config).expect("valid widths");
        let via_gdn = hook
            .rotate_gdn_out(&tiled, &map, head_v_dim, &mut scratch_b)
            .expect("rotate_gdn_out")
            .to_vec();

        assert_eq!(via_rotate, via_gdn);
        assert_eq!(scratch_b.transforms(), 1);

        // With one v-head per k-head the map is the identity and the two
        // entry points are the same call.
        let identity = VHeadMap::new(config.n_v_heads(), config.n_v_heads()).expect("identity");
        let mut scratch_c = HadamardScratch::new(&config).expect("valid widths");
        let via_identity = hook
            .rotate_gdn_out(&tiled, &identity, head_v_dim, &mut scratch_c)
            .expect("rotate_gdn_out")
            .to_vec();
        let mut scratch_d = HadamardScratch::new(&config).expect("valid widths");
        let plain = hook
            .rotate(&tiled, HEADS_WIDTH, &mut scratch_d)
            .expect("rotate")
            .to_vec();
        assert_eq!(via_identity, plain);
    }

    /// The whole point of the hook: replaying design §3.4's schedule for one
    /// layer runs the transform **four** times, not once per folded matmul.
    #[test]
    fn rotation_runs_once_per_activation_not_once_per_weight() {
        let config = bonsai2_config();
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let mut scratch = HadamardScratch::new(&config).expect("valid widths");

        let attn_norm_out = ramp(HIDDEN, 0.1);
        let attn_concat = ramp(HEADS_WIDTH, 0.2);
        let ffn_norm_out = ramp(HIDDEN, 0.3);
        let ffn_hidden = ramp(FFN, 0.4);

        // ── FULL layer: 7 folded matmuls, 4 activations ──────────────────
        let mut full_matmuls = 0usize;
        {
            let a = hook
                .rotate(&attn_norm_out, HIDDEN, &mut scratch)
                .expect("a");
            // attn_q, attn_k, attn_v all read the SAME rotated activation.
            for _ in 0..3 {
                assert_eq!(a.len(), HIDDEN);
                full_matmuls += 1;
            }
        }
        {
            let o = hook
                .rotate(&attn_concat, HEADS_WIDTH, &mut scratch)
                .expect("o");
            assert_eq!(o.len(), HEADS_WIDTH);
            full_matmuls += 1; // attn_output
        }
        {
            let f = hook.rotate(&ffn_norm_out, HIDDEN, &mut scratch).expect("f");
            for _ in 0..2 {
                assert_eq!(f.len(), HIDDEN);
                full_matmuls += 1; // ffn_gate, ffn_up
            }
        }
        {
            let m = hook.rotate(&ffn_hidden, FFN, &mut scratch).expect("m");
            assert_eq!(m.len(), FFN);
            full_matmuls += 1; // ffn_down
        }
        assert_eq!(full_matmuls, 7);
        assert_eq!(
            scratch.transforms(),
            4,
            "a full layer rotates 4 activations, not 7 weights"
        );

        // ── LINEAR layer: 6 folded matmuls, the same 4 activations ───────
        scratch.reset_counter();
        let mut linear_matmuls = 0usize;
        {
            let a = hook
                .rotate(&attn_norm_out, HIDDEN, &mut scratch)
                .expect("a");
            for _ in 0..2 {
                assert_eq!(a.len(), HIDDEN);
                linear_matmuls += 1; // attn_qkv, attn_gate
            }
        }
        {
            let o = hook
                .rotate(&attn_concat, HEADS_WIDTH, &mut scratch)
                .expect("o");
            assert_eq!(o.len(), HEADS_WIDTH);
            linear_matmuls += 1; // ssm_out
        }
        {
            let f = hook.rotate(&ffn_norm_out, HIDDEN, &mut scratch).expect("f");
            for _ in 0..2 {
                assert_eq!(f.len(), HIDDEN);
                linear_matmuls += 1; // ffn_gate, ffn_up
            }
        }
        {
            let m = hook.rotate(&ffn_hidden, FFN, &mut scratch).expect("m");
            assert_eq!(m.len(), FFN);
            linear_matmuls += 1; // ffn_down
        }
        assert_eq!(linear_matmuls, 6);
        assert_eq!(scratch.transforms(), 4);
    }

    #[test]
    fn batched_rotation_matches_the_single_row_form() {
        let config = bonsai2_config();
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let mut scratch = HadamardScratch::new(&config).expect("valid widths");
        let rows = 3usize;
        let mut batch: Vec<f32> = Vec::with_capacity(rows * HIDDEN);
        for r in 0..rows {
            batch.extend(ramp(HIDDEN, r as f32));
        }
        let original = batch.clone();

        hook.rotate_rows_in_place(&mut batch, HIDDEN, rows)
            .expect("batch rotate");

        for r in 0..rows {
            let single = hook
                .rotate(
                    &original[r * HIDDEN..(r + 1) * HIDDEN],
                    HIDDEN,
                    &mut scratch,
                )
                .expect("rotate");
            assert_eq!(&batch[r * HIDDEN..(r + 1) * HIDDEN], single, "row {r}");
        }
    }

    #[test]
    fn rejects_a_width_the_scratch_or_contract_does_not_know() {
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        let mut scratch = HadamardScratch::for_widths(&[HIDDEN]).expect("valid widths");
        let err = hook
            .rotate(&ramp(HEADS_WIDTH, 0.0), HEADS_WIDTH, &mut scratch)
            .expect_err("no scratch buffer for 6144");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
        assert!(err.to_string().contains("6144"), "{err}");
        // A failed claim must not be counted as work done.
        assert_eq!(scratch.transforms(), 0);

        let err = hook
            .rotate(&ramp(HIDDEN - 1, 0.0), HIDDEN, &mut scratch)
            .expect_err("short input");
        assert_eq!(err.error_code(), "SHAPE_MISMATCH");
        assert_eq!(scratch.transforms(), 0);
    }

    #[test]
    fn validate_widths_is_fatal_on_a_truncated_sign_table() {
        let config = bonsai2_config();
        let good = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        good.validate_widths(&config).expect("complete sign table");

        let mut truncated = bonsai2_hadamard();
        truncated.signs.remove(&FFN);
        let hook = HadamardHook::new(Arc::new(truncated));
        let err = hook
            .validate_widths(&config)
            .expect_err("a missing sign vector must be fatal at load");
        assert_eq!(err.error_code(), "CORE_ERROR");
        assert!(err.to_string().contains("17408"), "{err}");
    }

    #[test]
    fn fold_membership_excludes_the_ssm_gates() {
        let hook = HadamardHook::new(Arc::new(bonsai2_hadamard()));
        assert!(hook.is_folded("blk.0.attn_qkv.weight"));
        assert!(hook.is_folded("output.weight"));
        for unfolded in [
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_norm.weight",
            "blk.0.attn_norm.weight",
        ] {
            assert!(!hook.is_folded(unfolded), "{unfolded} must NOT be folded");
        }
        assert!(hook.is_inverse("token_embd.weight"));
        assert!(!hook.is_inverse("output.weight"));
    }
}
