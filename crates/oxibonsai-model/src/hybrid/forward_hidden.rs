//! Hidden-state forward pass of the `qwen35` hybrid stack — the pre-LM-head
//! seam embeddings need (`RT-08`), the hybrid twin of the dense
//! `BonsaiModel::forward_hidden`.
//!
//! # What it returns, and in which basis
//!
//! Row `i` of [`HybridModel::forward_hidden`] is `output_norm` applied to the
//! residual stream after the last block, for `tokens[i]` at position `i`:
//! exactly the vector [`run_chunk`] normalises for its tail token before the
//! LM head.
//!
//! For a Hadamard-folded checkpoint those rows are in the model's **own**
//! basis, not the rotated one. The embedding row is un-rotated right after
//! its lookup (design §3.5), so the residual stream is plain throughout, and
//! every folded projection consumes a rotated *copy* of its input; the final
//! norm is applied to the plain stream, and [`run_chunk`] rotates a copy of
//! that normed row only to feed the folded LM head. Pooled embeddings of a
//! folded and an unfolded checkpoint of the same weights are therefore
//! directly comparable — no sign vector or transform is baked into them.
//!
//! # How
//!
//! The same driver the prefill runs: the prompt is split into
//! [`HybridModel::prefill_chunk`]-token chunks and each goes through
//! [`run_chunk`] with no logit buffer, so the LM head is skipped entirely
//! and the chunk's post-stack residual rows are left in the scratch, where
//! `output_norm` is applied to each. Because the chunking is the prefill's
//! own, the rows are the very state [`HybridModel::forward_prefill`]
//! computes before its head — the tests below reproduce its logits bit for
//! bit from the last row.
//!
//! # Per-sequence state
//!
//! The pass writes KV positions `0..tokens.len()` and advances the recurrent
//! state, so it clears both — [`HybridModel::reset`], which touches only this
//! model's own KV cursor and recurrent state (RT-28) — before the first chunk
//! and again after the last, on the error path too: an embedding never sees
//! a previous sequence and never leaves its own behind. It must not be
//! interleaved with a generation on the same model; the runtime serialises
//! embedding requests on a dedicated engine.

use crate::error::{ModelError, ModelResult};
use crate::hybrid::forward::{run_chunk, scratch_short};
use crate::model::BonsaiModel;

use super::HybridModel;

impl HybridModel<'_> {
    /// Final-normed hidden state of every token, **before** the LM head.
    ///
    /// Returns a row-major `[n_tokens × hidden_size]` buffer in the model's
    /// own (un-rotated) basis — see the module docs. Positions are assigned
    /// `0..tokens.len()`: an embedding is a property of the text alone.
    /// Clears this model's KV cursor and recurrent state before and after.
    ///
    /// # Errors
    ///
    /// * [`ModelError::ShapeInvariant`] — `tokens` is empty, or the model has
    ///   a zero `hidden_size` (the dense path's typed errors).
    /// * [`ModelError::SequenceTooLong`] — more tokens than the KV window
    ///   ([`HybridModel::max_seq_len`]).
    /// * Anything the embedding lookup, the blocks or `output_norm` return
    ///   (e.g. a token id past the vocabulary).
    pub fn forward_hidden(&mut self, tokens: &[u32]) -> ModelResult<Vec<f32>> {
        let hidden = self.config.base.hidden_size;
        BonsaiModel::check_embedding_input(
            tokens,
            hidden,
            self.max_seq_len,
            "HybridModel::forward_hidden",
        )?;
        self.reset();
        let result = self.forward_hidden_inner(tokens, hidden);
        self.reset();
        result
    }

    /// Body of [`forward_hidden`](Self::forward_hidden), between the resets.
    fn forward_hidden_inner(&mut self, tokens: &[u32], hidden: usize) -> ModelResult<Vec<f32>> {
        let total = tokens.len();
        let len = total
            .checked_mul(hidden)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "HybridModel::forward_hidden".to_string(),
                expected: "n_tokens * hidden_size representable as usize".to_string(),
                actual: format!("{total} x {hidden}"),
            })?;
        let mut out = vec![0.0f32; len];
        let chunk = self.prefill_chunk.max(1);
        let mut offset = 0usize;
        while offset < total {
            let end = offset.saturating_add(chunk).min(total);
            let slice = tokens
                .get(offset..end)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "hidden-state chunk".to_string(),
                    expected: format!("{offset}..{end} within {total} tokens"),
                    actual: "out of range".to_string(),
                })?;
            {
                let mut ctx = self.ctx();
                // No logit buffer: the LM head is skipped, and the chunk's
                // post-stack residual rows stay in `scratch.resid`.
                run_chunk(&mut ctx, slice, offset, None, None)?;
            }
            let rows_len = (end - offset) * hidden;
            let rows = self
                .scratch
                .resid
                .get(..rows_len)
                .ok_or_else(|| scratch_short("resid", rows_len))?;
            let dst = out.get_mut(offset * hidden..end * hidden).ok_or_else(|| {
                ModelError::ShapeInvariant {
                    tensor: "HybridModel::forward_hidden rows".to_string(),
                    expected: format!("rows {offset}..{end} of {total}"),
                    actual: "out of range".to_string(),
                }
            })?;
            for (src, dst) in rows.chunks_exact(hidden).zip(dst.chunks_exact_mut(hidden)) {
                self.output_norm.forward(src, dst)?;
            }
            offset = end;
        }
        Ok(out)
    }

    /// Mean-pooled, L2-normalised sentence embedding for `tokens`: the
    /// average of [`forward_hidden`](Self::forward_hidden)'s rows, scaled to
    /// unit length, `[hidden_size]` floats.
    ///
    /// The dense model's recipe exactly (the pooling step is shared, so the
    /// two cannot pool differently): a (near-)zero mean is returned as the
    /// zero vector and a non-finite one collapses to it, rather than being
    /// divided into `NaN`s.
    ///
    /// # Errors
    ///
    /// Everything [`forward_hidden`](Self::forward_hidden) can return; in
    /// particular [`ModelError::ShapeInvariant`] for empty input.
    pub fn embed_mean_pooled(&mut self, tokens: &[u32]) -> ModelResult<Vec<f32>> {
        let hidden = self.config.base.hidden_size;
        BonsaiModel::check_embedding_input(
            tokens,
            hidden,
            self.max_seq_len,
            "HybridModel::embed_mean_pooled",
        )?;
        let states = self.forward_hidden(tokens)?;
        BonsaiModel::mean_pool_normalized(
            &states,
            hidden,
            tokens.len(),
            "HybridModel::embed_mean_pooled",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};
    use oxibonsai_core::gguf::reader::GgufFile;

    const PROMPT: [u32; 7] = [3, 9, 1, 4, 27, 8, 15];

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The row the LM head reads, turned into logits the way `run_chunk`
    /// does it: rotate a copy for a folded checkpoint, then project.
    fn logits_from_row(model: &HybridModel<'_>, row: &[f32]) -> Vec<f32> {
        let hidden = model.config().base.hidden_size;
        let vocab = model.config().base.vocab_size;
        let mut head_input = row.to_vec();
        if let Some(hook) = model.hadamard() {
            hook.rotate_rows_in_place(&mut head_input, hidden, 1)
                .expect("rotate the final row");
        }
        let mut logits = vec![0.0f32; vocab];
        model
            .lm_head()
            .forward_vec(&head_input, &mut logits)
            .expect("LM head");
        logits
    }

    #[test]
    fn forward_hidden_returns_one_row_per_token_and_leaves_no_state_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let hidden = model.config().base.hidden_size;
        let rows = model.forward_hidden(&PROMPT).expect("forward_hidden");
        assert_eq!(rows.len(), PROMPT.len() * hidden);
        assert!(rows.iter().all(|v| v.is_finite()));
        assert!(
            rows.iter().any(|v| *v != 0.0),
            "a real stack is not all zero"
        );
        assert_eq!(model.kv_cache().seq_len(), 0, "no KV history survives");
        assert_eq!(model.recurrent().token_count(), 0);
        for slot in 0..model.recurrent().n_layers() {
            assert!(model
                .recurrent()
                .ssm(slot)
                .expect("ssm")
                .iter()
                .all(|v| *v == 0.0));
        }
    }

    /// The rows are the pre-head state: rotating `forward_hidden`'s last row
    /// and applying the LM head reproduces `forward_prefill`'s last-token
    /// logits **bit for bit** — for a folded and an unfolded checkpoint.
    #[test]
    fn the_last_row_through_the_lm_head_is_forward_prefill_bit_for_bit_bonsai2() {
        for hadamard in [true, false] {
            let bytes = synthetic_gguf(
                FixtureShape::default(),
                FixtureOptions {
                    hadamard,
                    ..FixtureOptions::default()
                },
            );
            let gguf = GgufFile::parse(&bytes).expect("fixture parses");
            let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
            let hidden = model.config().base.hidden_size;
            let vocab = model.config().base.vocab_size;
            assert_eq!(model.hadamard().is_some(), hadamard);

            let rows = model.forward_hidden(&PROMPT).expect("forward_hidden");
            let last = &rows[(PROMPT.len() - 1) * hidden..];
            let from_rows = logits_from_row(&model, last);

            let mut prefill = vec![0.0f32; vocab];
            model
                .forward_prefill(&PROMPT, 0, &mut prefill)
                .expect("forward_prefill");
            assert_eq!(
                bits(&from_rows),
                bits(&prefill),
                "hadamard={hadamard}: the last hidden row must be exactly the LM head's input"
            );
        }
    }

    /// Chunking is invisible: a 3-token chunk (so 3 + 3 + a 1-token tail)
    /// gives exactly the rows of one unchunked pass.
    #[test]
    fn chunked_forward_hidden_equals_the_unchunked_rows_exactly_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let unchunked = model.forward_hidden(&PROMPT).expect("unchunked");
        model.set_prefill_chunk(3).expect("chunk 3");
        let chunked = model.forward_hidden(&PROMPT).expect("chunked");
        assert_eq!(bits(&chunked), bits(&unchunked));
    }

    /// Each row equals the `output_norm` of the residual the same prefix
    /// leaves behind, so a prefix's rows do not depend on what follows it.
    #[test]
    fn a_prefix_has_the_same_rows_as_the_whole_prompt_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let hidden = model.config().base.hidden_size;
        let whole = model.forward_hidden(&PROMPT).expect("whole");
        let prefix = model.forward_hidden(&PROMPT[..4]).expect("prefix");
        assert_eq!(bits(&prefix), bits(&whole[..4 * hidden]));
    }

    #[test]
    fn forward_hidden_is_deterministic_and_ignores_a_previous_sequence_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let first = model.forward_hidden(&PROMPT).expect("first");
        // Leave a live sequence behind: KV positions and recurrent state.
        let vocab = model.config().base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        model
            .forward_prefill(&[5, 6, 7, 8], 0, &mut logits)
            .expect("unrelated prefill");
        assert_eq!(model.recurrent().token_count(), 4);
        let second = model.forward_hidden(&PROMPT).expect("second");
        assert_eq!(
            bits(&first),
            bits(&second),
            "an embedding must not see a previous sequence's state"
        );
    }

    #[test]
    fn embed_mean_pooled_is_a_unit_vector_of_the_mean_row_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let hidden = model.config().base.hidden_size;
        let pooled = model.embed_mean_pooled(&PROMPT).expect("pooled");
        assert_eq!(pooled.len(), hidden);
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit norm, got {norm}");

        // The same vector, pooled by hand from the rows.
        let rows = model.forward_hidden(&PROMPT).expect("rows");
        let mut mean = vec![0.0f32; hidden];
        for row in rows.chunks_exact(hidden) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v;
            }
        }
        let inv = 1.0 / PROMPT.len() as f32;
        mean.iter_mut().for_each(|m| *m *= inv);
        let mean_norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
        mean.iter_mut().for_each(|m| *m /= mean_norm);
        assert_eq!(bits(&pooled), bits(&mean));

        // Different texts give different embeddings.
        let other = model.embed_mean_pooled(&[40, 41, 42]).expect("other");
        assert_ne!(bits(&pooled), bits(&other));
    }

    #[test]
    fn forward_hidden_refuses_empty_and_over_long_input_with_the_dense_errors_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 8).expect("model loads");
        assert!(matches!(
            model.forward_hidden(&[]),
            Err(ModelError::ShapeInvariant { .. })
        ));
        assert!(matches!(
            model.embed_mean_pooled(&[]),
            Err(ModelError::ShapeInvariant { .. })
        ));
        let too_long: Vec<u32> = (0..9).collect();
        assert!(matches!(
            model.forward_hidden(&too_long),
            Err(ModelError::SequenceTooLong {
                seq_len: 9,
                max_ctx: 8
            })
        ));
        // Exactly the window is fine.
        let window: Vec<u32> = (0..8).collect();
        let rows = model.forward_hidden(&window).expect("a full window embeds");
        assert_eq!(rows.len(), 8 * model.config().base.hidden_size);
    }

    #[test]
    fn a_token_past_the_vocabulary_is_refused_and_leaves_no_state_bonsai2() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let vocab = u32::try_from(model.config().base.vocab_size).expect("fits");
        model
            .forward_hidden(&[1, 2, vocab])
            .expect_err("a token past the vocabulary must be refused");
        assert_eq!(model.kv_cache().seq_len(), 0);
        assert_eq!(model.recurrent().token_count(), 0);
    }
}
