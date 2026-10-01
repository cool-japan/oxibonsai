//! Prefill of caller-supplied embedding rows — the model side of the vision
//! splice (bonsai2-design.md §6.2).
//!
//! A multimodal prompt is text tokens interleaved with images. The two
//! kinds of row reach block 0 differently:
//!
//! * a **text** row is `token_embd[id]` followed by the inverse Hadamard
//!   transform (design §3.5) — [`HybridModel::embed_token_rows`], the very
//!   code a text-only prefill runs;
//! * an **image** row is one merged token of the vision tower
//!   ([`crate::vision::VisionTower::encode`]), already in the unrotated
//!   embedding basis — copied verbatim, **never** inverse-transformed (see
//!   [`crate::vision::merger`] for why that trap is invisible to every norm
//!   check).
//!
//! [`HybridModel::assemble_prompt`] builds the rows and their 3-axis
//! rotary positions ([`crate::vision::mrope`]); [`HybridModel::forward_prefill_rows`]
//! runs them through the same chunk loop a token prefill uses
//! (`HybridModel::prefill_chunks`), with each full-attention layer rotating
//! a row at its own [`MropePos`]; afterwards the model keeps the M-RoPE
//! offset ([`HybridModel::rope_delta`]) so that later decode steps rotate
//! at the position the reference would give them. The offset is keyed by
//! the sequence position it takes effect at (`RopeOffsets`), so a sequence
//! rolled back to an earlier position continues at the offset in force
//! there.

use crate::error::{ModelError, ModelResult};
use crate::hybrid::forward::{ChunkInput, LayerDump};
use crate::hybrid::model::{HybridModel, PrefillSource};
use crate::layers::rope_mrope::MropePos;
use crate::vision::mrope::{next_rope_position, MropeCursor};
use crate::vision::GridSize;

/// One piece of a multimodal prompt, in prompt order.
#[derive(Debug, Clone, Copy)]
pub enum PromptPiece<'p> {
    /// Text token ids: embedded like any prompt (lookup + inverse Hadamard
    /// transform), rotating at consecutive text positions.
    Tokens(&'p [u32]),
    /// One image's merged rows (`grid.h * grid.w` rows of `hidden`, row-major
    /// over the merged grid) in the unrotated embedding basis, rotating at
    /// the image's 3-axis positions.
    Image {
        /// `grid.n_tokens() * hidden` floats.
        rows: &'p [f32],
        /// The merged grid the rows cover.
        grid: GridSize,
    },
}

/// A whole multimodal prompt as block 0 sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledPrompt {
    /// `positions.len() * hidden` floats: text rows embedded, image rows
    /// verbatim.
    pub rows: Vec<f32>,
    /// One rotary position per row.
    pub positions: Vec<MropePos>,
}

impl AssembledPrompt {
    /// Rows (sequence positions) the prompt occupies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the prompt has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// The M-RoPE offsets of one sequence (design §6.2), keyed by the sequence
/// position each takes effect at.
///
/// An image of merged grid `h x w` occupies `h * w` sequence positions but
/// only `max(h, w)` rotary ones, so every token after it rotates behind its
/// sequence position. A rows prefill that moves the offset records one
/// entry: the tokens from sequence position `from` on rotate at `pos -
/// delta`, until a later entry. Keeping the offsets by position, rather than
/// only the latest one, is what lets a sequence rolled back to an earlier
/// point — its KV cursor and recurrent state restored from a snapshot —
/// continue at the offset in force there: a forward at `pos` first forgets
/// every entry that takes effect after `pos`, which belongs to the
/// abandoned continuation. A text-only sequence records nothing (offset
/// `0` everywhere).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RopeOffsets {
    /// `(from, delta)` pairs, strictly increasing in `from`.
    entries: Vec<(usize, usize)>,
}

impl RopeOffsets {
    /// The offset of the token at sequence position `pos`.
    pub(crate) fn at(&self, pos: usize) -> usize {
        self.entries
            .iter()
            .rev()
            .find(|&&(from, _)| from <= pos)
            .map_or(0, |&(_, delta)| delta)
    }

    /// The sequence continues at position `pos`: forget the offsets that
    /// take effect after it — a rolled-back continuation's, or at `0` (a new
    /// sequence) every one.
    pub(crate) fn continue_at(&mut self, pos: usize) {
        if pos == 0 {
            self.entries.clear();
        } else {
            self.entries.retain(|&(from, _)| from <= pos);
        }
    }

    /// A prefill that began at sequence position `start` leaves the tokens
    /// from sequence position `end` on rotating at `pos - delta`: forget
    /// every offset past `start`, then record this one unless it is already
    /// the offset in force. A prefill of no rows (`end <= start`) records
    /// nothing — which is also what keeps `from` strictly increasing, every
    /// entry left after `continue_at(start)` taking effect at or before
    /// `start`.
    pub(crate) fn record(&mut self, start: usize, end: usize, delta: usize) {
        self.continue_at(start);
        if end > start && self.at(end) != delta {
            self.entries.push((end, delta));
        }
    }

    /// Forget every offset (the sequence is reset).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<'a> HybridModel<'a> {
    /// The rotary position a prompt starting at sequence position
    /// `start_pos` begins at, without changing any state: `0` for a new
    /// sequence, `start_pos` minus the offset in force there for a
    /// continuation.
    fn rope_start_for(&self, start_pos: usize) -> ModelResult<usize> {
        if start_pos == 0 {
            return Ok(0);
        }
        let delta = self.rope_delta_at(start_pos);
        start_pos
            .checked_sub(delta)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "rotary position".to_string(),
                expected: format!("a sequence position at or past the M-RoPE offset {delta}"),
                actual: start_pos.to_string(),
            })
    }

    /// Build the rows and rotary positions of `pieces`, a prompt whose first
    /// row sits at sequence position `start_pos`.
    ///
    /// Text pieces are embedded through [`HybridModel::embed_token_rows`];
    /// image rows are copied verbatim (no inverse Hadamard transform, see the
    /// module docs). Positions follow design §6.2: consecutive text
    /// positions, then each image at `(p0, p0 + row, p0 + col)` with text
    /// resuming at `p0 + max(h, w)`.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] for an image whose `rows` are not
    /// `grid.n_tokens() * hidden` long, [`ModelError::ShapeInvariant`] for an
    /// empty grid or a position past the rotary range, and the embedding
    /// lookup's errors (e.g. a token id past the vocabulary).
    pub fn assemble_prompt(
        &self,
        pieces: &[PromptPiece<'_>],
        start_pos: usize,
    ) -> ModelResult<AssembledPrompt> {
        let hidden = self.config().base.hidden_size;
        let total: usize = pieces
            .iter()
            .map(|piece| match piece {
                PromptPiece::Tokens(tokens) => tokens.len(),
                PromptPiece::Image { grid, .. } => grid.n_tokens(),
            })
            .sum();
        let mut rows = vec![0.0f32; total.saturating_mul(hidden)];
        let mut cursor = MropeCursor::new(self.rope_start_for(start_pos)?);
        let mut row = 0usize;
        for (index, piece) in pieces.iter().enumerate() {
            match *piece {
                PromptPiece::Tokens(tokens) => {
                    let dst = rows
                        .get_mut(row * hidden..(row + tokens.len()) * hidden)
                        .ok_or_else(|| short_rows(index))?;
                    self.embed_token_rows(tokens, dst)?;
                    cursor.push_text(tokens.len())?;
                    row += tokens.len();
                }
                PromptPiece::Image { rows: image, grid } => {
                    let n = grid.n_tokens();
                    let expected = n.saturating_mul(hidden);
                    if image.len() != expected {
                        return Err(ModelError::ShapeMismatch {
                            name: format!(
                                "image {index} rows ({} x {} merged grid)",
                                grid.h, grid.w
                            ),
                            expected: vec![n, hidden],
                            actual: vec![image.len()],
                        });
                    }
                    // Verbatim: the tower's rows are already in the basis
                    // block 0 reads (design §3.5 / §6.2).
                    rows.get_mut(row * hidden..(row + n) * hidden)
                        .ok_or_else(|| short_rows(index))?
                        .copy_from_slice(image);
                    cursor.push_image(grid)?;
                    row += n;
                }
            }
        }
        Ok(AssembledPrompt {
            rows,
            positions: cursor.into_positions(),
        })
    }

    /// Chunked prefill of caller-supplied rows starting at sequence position
    /// `start_pos`, writing the **last** row's `[vocab_size]` logits into
    /// `logits_out` when given.
    ///
    /// `rows` is `positions.len()` rows of `hidden` floats **already in the
    /// basis block 0 expects**: they are written into the residual stream
    /// verbatim, with no `token_embd` lookup and no inverse Hadamard
    /// transform — text rows must come from
    /// [`HybridModel::embed_token_rows`] (or
    /// [`HybridModel::assemble_prompt`]), image rows straight from the vision
    /// tower. Row `t` is stored at KV slot `start_pos + t`, advances the
    /// recurrent state once, and rotates at `positions[t]` in the
    /// full-attention layers (3-axis M-RoPE, design §6.2); the Gated-DeltaNet
    /// layers have no positional input. The rows run through the same chunk
    /// loop as [`HybridModel::forward_prefill`], so the result does not
    /// depend on [`HybridModel::prefill_chunk`] beyond floating-point
    /// batching.
    ///
    /// Afterwards the next token sits at sequence position `start_pos +
    /// positions.len()` and rotates at one past the largest axis `positions`
    /// used; the difference is kept as [`HybridModel::rope_delta`] for every
    /// later [`HybridModel::forward`] / [`HybridModel::forward_prefill`].
    ///
    /// An empty `positions` is a no-op.
    ///
    /// # Errors
    ///
    /// All checked before any state changes:
    /// [`ModelError::ShapeMismatch`] for `rows` of the wrong length,
    /// [`ModelError::PositionOutOfRange`] past the KV window, and
    /// [`ModelError::ShapeInvariant`] for rotary positions that run ahead of
    /// the sequence. Then anything the blocks, the caches or the kernels
    /// return.
    pub fn forward_prefill_rows(
        &mut self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
        logits_out: Option<&mut [f32]>,
    ) -> ModelResult<()> {
        let Some(rope_next) = self.check_rows(rows, positions, start_pos)? else {
            return Ok(());
        };
        self.prefill_chunks(
            PrefillSource::Rows { rows, positions },
            start_pos,
            logits_out,
        )?;
        self.set_rope_next(start_pos, start_pos + positions.len(), rope_next)
    }

    /// [`HybridModel::forward_prefill_rows`] as one unchunked chunk,
    /// recording every block's output — what entered block 0 included
    /// ([`LayerDump::embedding`]) — for diagnostics and the Hadamard-bypass
    /// gate.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::forward_prefill_rows`]; an empty `positions` is a
    /// [`ModelError::ShapeInvariant`] here (there is nothing to record).
    pub fn forward_prefill_rows_with_dump(
        &mut self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
        logits_out: Option<&mut [f32]>,
    ) -> ModelResult<LayerDump> {
        let rope_next = self
            .check_rows(rows, positions, start_pos)?
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "prefill rows".to_string(),
                expected: "at least one row to record".to_string(),
                actual: "0".to_string(),
            })?;
        let dump = self.run_single_chunk_with_dump(
            ChunkInput::Rows { rows, positions },
            start_pos,
            logits_out,
        )?;
        self.set_rope_next(start_pos, start_pos + positions.len(), rope_next)?;
        Ok(dump)
    }

    /// Assemble `pieces` ([`HybridModel::assemble_prompt`]) and prefill
    /// them ([`HybridModel::forward_prefill_rows`]) from sequence position
    /// `start_pos`. Returns the number of rows (sequence positions) the
    /// prompt occupied.
    ///
    /// # Errors
    ///
    /// As the two calls it makes.
    pub fn forward_prefill_pieces(
        &mut self,
        pieces: &[PromptPiece<'_>],
        start_pos: usize,
        logits_out: Option<&mut [f32]>,
    ) -> ModelResult<usize> {
        let prompt = self.assemble_prompt(pieces, start_pos)?;
        self.forward_prefill_rows(&prompt.rows, &prompt.positions, start_pos, logits_out)?;
        Ok(prompt.len())
    }

    /// Validate a rows prefill before touching any state; `Ok(None)` for an
    /// empty one, else the rotary position the token after it takes.
    fn check_rows(
        &self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
    ) -> ModelResult<Option<usize>> {
        let Some(rope_next) = next_rope_position(positions) else {
            return if rows.is_empty() {
                Ok(None)
            } else {
                Err(ModelError::ShapeMismatch {
                    name: "prefill rows".to_string(),
                    expected: vec![0],
                    actual: vec![rows.len()],
                })
            };
        };
        let hidden = self.config().base.hidden_size;
        let n = positions.len();
        if rows.len() != n.saturating_mul(hidden) {
            return Err(ModelError::ShapeMismatch {
                name: "prefill rows".to_string(),
                expected: vec![n, hidden],
                actual: vec![rows.len()],
            });
        }
        let end = start_pos
            .checked_add(n)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "prefill rows".to_string(),
                expected: "start_pos + rows representable as usize".to_string(),
                actual: format!("start_pos = {start_pos}, rows = {n}"),
            })?;
        if end > self.max_seq_len() {
            return Err(ModelError::PositionOutOfRange {
                pos: end - 1,
                max: self.max_seq_len(),
            });
        }
        if rope_next > end {
            return Err(ModelError::ShapeInvariant {
                tensor: "rotary positions".to_string(),
                expected: format!(
                    "no axis past its row's sequence position (next sequence position {end})"
                ),
                actual: format!("next rotary position {rope_next}"),
            });
        }
        Ok(Some(rope_next))
    }
}

/// The assembled row buffer is shorter than the pieces need — an internal
/// invariant (the buffer is sized from the same pieces), reported rather
/// than `unwrap`ped.
fn short_rows(piece: usize) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: format!("assembled prompt rows (piece {piece})"),
        expected: "rows sized from the pieces".to_string(),
        actual: "shorter".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};
    use crate::vision::mrope::text_positions;
    use oxibonsai_core::gguf::reader::GgufFile;

    fn fixture() -> Vec<u8> {
        synthetic_gguf(FixtureShape::default(), FixtureOptions::default())
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Deterministic, clearly-not-an-embedding rows standing in for a vision
    /// tower's output.
    fn image_rows(n: usize, hidden: usize, seed: u32) -> Vec<f32> {
        (0..n * hidden)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) as f32 / 16_777_216.0) - 0.5
            })
            .collect()
    }

    /// A text-only prompt through the rows path (text rows from
    /// `embed_token_rows`, text positions) is bit-identical to the token-id
    /// prefill: same logits, same next decode step.
    #[test]
    fn text_rows_prefill_is_bit_identical_to_the_token_prefill_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut by_ids = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        let mut by_rows = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        let vocab = by_ids.config().base.vocab_size;
        let hidden = by_ids.config().base.hidden_size;
        let tokens: Vec<u32> = vec![3, 17, 42, 5, 99, 250, 7, 1];

        let mut want = vec![0.0f32; vocab];
        by_ids
            .forward_prefill(&tokens, 0, &mut want)
            .expect("token prefill");

        let mut rows = vec![0.0f32; tokens.len() * hidden];
        by_rows.embed_token_rows(&tokens, &mut rows).expect("embed");
        let positions = text_positions(0, tokens.len()).expect("positions");
        let mut got = vec![0.0f32; vocab];
        by_rows
            .forward_prefill_rows(&rows, &positions, 0, Some(&mut got))
            .expect("rows prefill");
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(by_rows.rope_delta(), 0);

        let pos = tokens.len();
        let a = by_ids.forward_alloc(11, pos).expect("decode");
        let b = by_rows.forward_alloc(11, pos).expect("decode");
        assert_eq!(bits(&a), bits(&b));
    }

    /// The Hadamard bypass (design §3.5 / §6.2): an image row reaches block 0
    /// exactly as the tower produced it, while a text row is the
    /// inverse-transformed `token_embd` lookup — different from the raw
    /// lookup, and the transform would have changed the image row too.
    #[test]
    fn image_rows_bypass_the_inverse_hadamard_and_text_rows_take_it_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        let hidden = model.config().base.hidden_size;
        let hook = model.hadamard().cloned().expect("the fixture is folded");
        let grid = GridSize { h: 2, w: 2 };
        let image = image_rows(grid.n_tokens(), hidden, 7);
        let lead = [5u32, 6];
        let tail = [7u32];
        let pieces = [
            PromptPiece::Tokens(&lead),
            PromptPiece::Image { rows: &image, grid },
            PromptPiece::Tokens(&tail),
        ];
        let prompt = model.assemble_prompt(&pieces, 0).expect("assemble");
        assert_eq!(prompt.len(), 2 + 4 + 1);
        assert_eq!(
            prompt.positions,
            vec![
                MropePos::text(0),
                MropePos::text(1),
                MropePos { t: 2, h: 2, w: 2 },
                MropePos { t: 2, h: 2, w: 3 },
                MropePos { t: 2, h: 3, w: 2 },
                MropePos { t: 2, h: 3, w: 3 },
                MropePos::text(4),
            ]
        );
        let dump = model
            .forward_prefill_rows_with_dump(&prompt.rows, &prompt.positions, 0, None)
            .expect("prefill with dump");
        assert_eq!(dump.t_len(), 7);
        assert!(dump.tokens.is_empty());
        assert_eq!(dump.rotary_positions, prompt.positions);

        // Image rows: verbatim, bit for bit.
        for r in 0..grid.n_tokens() {
            let entered = dump.embedding_row(2 + r).expect("row");
            assert_eq!(
                bits(entered),
                bits(&image[r * hidden..(r + 1) * hidden]),
                "image row {r} must reach block 0 unchanged"
            );
            // ...and the bypass is load-bearing: the transform moves it.
            let mut transformed = image[r * hidden..(r + 1) * hidden].to_vec();
            hook.inverse_embedding(&mut transformed).expect("transform");
            assert_ne!(bits(&transformed), bits(entered));
        }

        // Text rows: the lookup followed by the inverse transform.
        for (t, token) in [(0usize, 5u32), (1, 6), (6, 7)] {
            let entered = dump.embedding_row(t).expect("row");
            let mut raw = vec![0.0f32; hidden];
            model
                .embedding()
                .row(token, hidden, &mut raw)
                .expect("lookup");
            assert_ne!(
                bits(entered),
                bits(&raw),
                "text row {t} must be inverse-transformed"
            );
            hook.inverse_embedding(&mut raw).expect("transform");
            assert_eq!(bits(entered), bits(&raw), "text row {t}");
        }
        // Seven sequence rows over five rotary positions.
        assert_eq!(model.rope_delta(), 2);
        assert_eq!(model.recurrent().token_count(), 7);
    }

    /// Decode after a multimodal prefill rotates at `pos - rope_delta`: a
    /// token fed by `forward` after the prefill gives exactly the logits of
    /// the same token placed as the prompt's last row at its explicit text
    /// position — and differs from rotating it at its sequence position.
    #[test]
    fn decode_after_an_image_rotates_at_the_offset_position_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let load = || {
            let mut m = HybridModel::from_gguf(&gguf, 64).expect("model loads");
            // One row per chunk: every path below then runs identical
            // single-row forwards, so the comparison is bitwise.
            m.set_prefill_chunk(1).expect("chunk");
            m
        };
        let mut decoded = load();
        let mut in_prompt = load();
        let mut wrong = load();
        let vocab = decoded.config().base.vocab_size;
        let hidden = decoded.config().base.hidden_size;
        let grid = GridSize { h: 3, w: 2 };
        let image = image_rows(grid.n_tokens(), hidden, 11);
        let lead = [9u32, 4, 2];
        let tail = [8u32, 31];
        let next = 123u32;

        let pieces = [
            PromptPiece::Tokens(&lead),
            PromptPiece::Image { rows: &image, grid },
            PromptPiece::Tokens(&tail),
        ];
        let rows = decoded
            .forward_prefill_pieces(&pieces, 0, None)
            .expect("prefill");
        assert_eq!(rows, 3 + 6 + 2);
        assert_eq!(decoded.rope_delta(), 6 - 3);
        let after = decoded.forward_alloc(next, rows).expect("decode");

        let tail_with_next = [8u32, 31, next];
        let pieces_with_next = [
            PromptPiece::Tokens(&lead),
            PromptPiece::Image { rows: &image, grid },
            PromptPiece::Tokens(&tail_with_next),
        ];
        let mut want = vec![0.0f32; vocab];
        in_prompt
            .forward_prefill_pieces(&pieces_with_next, 0, Some(&mut want))
            .expect("prefill");
        assert_eq!(bits(&after), bits(&want));

        // The same rows with the last token rotated at its *sequence*
        // position (no offset) — what a model that forgot the offset does.
        let mut prompt = wrong
            .assemble_prompt(&pieces_with_next, 0)
            .expect("assemble");
        let last = prompt.positions.len() - 1;
        prompt.positions[last] = MropePos::text(u32::try_from(last).expect("fits"));
        let mut forgot = vec![0.0f32; vocab];
        wrong
            .forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut forgot))
            .expect("prefill");
        assert_ne!(bits(&forgot), bits(&want));
    }

    /// The rows prefill runs through the shared chunk loop: a small chunk
    /// that splits the image agrees with one chunk (to floating-point
    /// batching, like the token prefill's own chunk invariance).
    #[test]
    fn rows_prefill_is_chunk_size_invariant_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut small = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        let mut whole = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        small.set_prefill_chunk(3).expect("chunk");
        let vocab = small.config().base.vocab_size;
        let hidden = small.config().base.hidden_size;
        let grid = GridSize { h: 2, w: 3 };
        let image = image_rows(grid.n_tokens(), hidden, 3);
        let lead = [1u32, 2, 3, 4];
        let tail = [5u32, 6];
        let pieces = [
            PromptPiece::Tokens(&lead),
            PromptPiece::Image { rows: &image, grid },
            PromptPiece::Tokens(&tail),
        ];
        let mut a = vec![0.0f32; vocab];
        let mut b = vec![0.0f32; vocab];
        small
            .forward_prefill_pieces(&pieces, 0, Some(&mut a))
            .expect("chunked");
        whole
            .forward_prefill_pieces(&pieces, 0, Some(&mut b))
            .expect("whole");
        for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
            let tol = 1e-5 + 1e-5 * f64::from(y.abs());
            assert!(
                (f64::from(x) - f64::from(y)).abs() <= tol,
                "logit[{i}] chunked {x} vs whole {y}"
            );
        }
        assert_eq!(small.rope_delta(), whole.rope_delta());
        assert_eq!(
            small.recurrent().token_count(),
            whole.recurrent().token_count()
        );
    }

    #[test]
    fn reset_and_a_new_sequence_clear_the_offset_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 64).expect("model loads");
        let hidden = model.config().base.hidden_size;
        let grid = GridSize { h: 2, w: 2 };
        let image = image_rows(grid.n_tokens(), hidden, 5);
        let pieces = [
            PromptPiece::Tokens(&[1, 2]),
            PromptPiece::Image { rows: &image, grid },
        ];
        model
            .forward_prefill_pieces(&pieces, 0, None)
            .expect("prefill");
        assert_eq!(model.rope_delta(), 2);
        model.reset();
        assert_eq!(model.rope_delta(), 0);

        model
            .forward_prefill_pieces(&pieces, 0, None)
            .expect("prefill");
        assert_eq!(model.rope_delta(), 2);
        // A decode at position 0 starts a new sequence.
        model.reset();
        model
            .forward_prefill_pieces(&pieces, 0, None)
            .expect("prefill");
        let vocab = model.config().base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        model.forward(3, 0, &mut logits).expect("restart");
        assert_eq!(model.rope_delta(), 0);
    }

    #[test]
    fn rope_offsets_are_keyed_by_the_position_they_take_effect_at() {
        let mut offsets = RopeOffsets::default();
        assert_eq!(offsets.at(0), 0);
        // An image in a prompt from 0: the tokens from 7 on rotate 2 behind.
        offsets.record(0, 7, 2);
        assert_eq!((offsets.at(6), offsets.at(7), offsets.at(100)), (0, 2, 2));
        // A second image, continuing at 10.
        offsets.record(10, 16, 8);
        assert_eq!((offsets.at(12), offsets.at(16)), (2, 8));
        // A rows prefill from 12 that moves nothing: the second image's
        // offset belongs to a continuation abandoned at 12, and the offset
        // in force there needs no new entry.
        offsets.record(12, 14, 2);
        assert_eq!(offsets, {
            let mut only_first = RopeOffsets::default();
            only_first.record(0, 7, 2);
            only_first
        });
        assert_eq!(offsets.at(100), 2);
        // Continuing at or after an entry keeps it; before it forgets it.
        offsets.continue_at(9);
        assert_eq!(offsets.at(9), 2);
        offsets.continue_at(5);
        assert_eq!(offsets, RopeOffsets::default());
        // A new sequence (position 0) forgets everything, as does a reset.
        offsets.record(0, 4, 1);
        offsets.continue_at(0);
        assert_eq!(offsets, RopeOffsets::default());
        offsets.record(0, 4, 1);
        offsets.clear();
        assert_eq!(offsets.at(4), 0);
        // A prefill of no rows records nothing.
        offsets.record(3, 3, 5);
        assert_eq!(offsets, RopeOffsets::default());
    }

    /// Roll `model` back to sequence position `k` the way an engine restores
    /// a sequence snapshot: the recurrent state from its deep copy, the KV
    /// cursor to `k` (the slots below it were never overwritten).
    fn roll_back(
        model: &mut HybridModel<'_>,
        snapshot: &crate::hybrid::RecurrentSnapshot,
        k: usize,
    ) {
        model.recurrent_mut().restore(snapshot).expect("restore");
        model.kv_cache_mut().set_seq_len(k);
    }

    /// Continue `model` at `k` with a text tail and one decode step: the
    /// tail's last logits and the step's. The tail is long enough for the
    /// decode step to sit past where either test's abandoned continuation
    /// ended, so an offset that continuation recorded would reach it.
    fn continue_text(model: &mut HybridModel<'_>, k: usize) -> (Vec<f32>, Vec<f32>) {
        let tail = [8u32, 31, 12, 40, 3, 77, 21, 6, 90, 14];
        let mut logits = vec![0.0f32; model.config().base.vocab_size];
        model
            .forward_prefill(&tail, k, &mut logits)
            .expect("text continuation");
        let next = model.forward_alloc(123, k + tail.len()).expect("decode");
        (logits, next)
    }

    /// A text sequence snapshotted at `k`, continued by an image prefill
    /// that is then rolled back, continues at the offset in force at `k`
    /// (none) — bit-identical to a model that never saw the image.
    #[test]
    fn a_rolled_back_text_sequence_forgets_the_abandoned_image_offset_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let load = || {
            let mut m = HybridModel::from_gguf(&gguf, 64).expect("model loads");
            m.set_prefill_chunk(1).expect("chunk");
            m
        };
        let mut model = load();
        let mut reference = load();
        let vocab = model.config().base.vocab_size;
        let hidden = model.config().base.hidden_size;
        let lead = [9u32, 4, 2, 7];
        let k = lead.len();
        let mut scratch = vec![0.0f32; vocab];

        model
            .forward_prefill(&lead, 0, &mut scratch)
            .expect("prefill");
        let snapshot = model.recurrent().snapshot();
        let grid = GridSize { h: 3, w: 2 };
        let image = image_rows(grid.n_tokens(), hidden, 13);
        let continuation = [
            PromptPiece::Image { rows: &image, grid },
            PromptPiece::Tokens(&[5]),
        ];
        model
            .forward_prefill_pieces(&continuation, k, None)
            .expect("image continuation");
        assert_eq!(model.rope_delta(), 6 - 3);
        roll_back(&mut model, &snapshot, k);
        assert_eq!(model.rope_delta(), 0, "the offset in force at k");
        let (got, got_next) = continue_text(&mut model, k);

        reference
            .forward_prefill(&lead, 0, &mut scratch)
            .expect("prefill");
        let (want, want_next) = continue_text(&mut reference, k);
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(bits(&got_next), bits(&want_next));
    }

    /// A sequence whose own image precedes the snapshot keeps that image's
    /// offset through the roll-back — neither zero nor the abandoned second
    /// image's larger one.
    #[test]
    fn a_rolled_back_sequence_keeps_the_offset_of_its_own_image_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let load = || {
            let mut m = HybridModel::from_gguf(&gguf, 64).expect("model loads");
            m.set_prefill_chunk(1).expect("chunk");
            m
        };
        let mut model = load();
        let mut reference = load();
        let hidden = model.config().base.hidden_size;
        let first = GridSize { h: 2, w: 2 };
        let image = image_rows(first.n_tokens(), hidden, 17);
        let lead = [
            PromptPiece::Tokens(&[1, 2]),
            PromptPiece::Image {
                rows: &image,
                grid: first,
            },
            PromptPiece::Tokens(&[3]),
        ];
        let k = model
            .forward_prefill_pieces(&lead, 0, None)
            .expect("prefill");
        assert_eq!(model.rope_delta(), 4 - 2);
        let snapshot = model.recurrent().snapshot();
        let second = GridSize { h: 3, w: 3 };
        let other = image_rows(second.n_tokens(), hidden, 19);
        model
            .forward_prefill_pieces(
                &[PromptPiece::Image {
                    rows: &other,
                    grid: second,
                }],
                k,
                None,
            )
            .expect("second image");
        assert_eq!(model.rope_delta(), 2 + (9 - 3));
        roll_back(&mut model, &snapshot, k);
        assert_eq!(model.rope_delta(), 2, "the first image's offset");
        let (got, got_next) = continue_text(&mut model, k);

        reference
            .forward_prefill_pieces(&lead, 0, None)
            .expect("prefill");
        let (want, want_next) = continue_text(&mut reference, k);
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(bits(&got_next), bits(&want_next));
    }

    #[test]
    fn malformed_rows_are_refused_before_any_state_changes_bonsai2() {
        let bytes = fixture();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 16).expect("model loads");
        let hidden = model.config().base.hidden_size;

        // Nothing to do.
        model
            .forward_prefill_rows(&[], &[], 0, None)
            .expect("empty is a no-op");

        // Rows / positions disagree.
        let positions = text_positions(0, 2).expect("positions");
        let err = model
            .forward_prefill_rows(&vec![0.0; hidden], &positions, 0, None)
            .expect_err("one row for two positions");
        assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");

        // Rotary positions ahead of the sequence.
        let ahead = vec![MropePos::text(5)];
        let err = model
            .forward_prefill_rows(&vec![0.0; hidden], &ahead, 0, None)
            .expect_err("rotary position past the sequence");
        assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err}");

        // Past the KV window (16).
        let long = text_positions(0, 17).expect("positions");
        let err = model
            .forward_prefill_rows(&vec![0.0; 17 * hidden], &long, 0, None)
            .expect_err("past the window");
        assert!(
            matches!(err, ModelError::PositionOutOfRange { .. }),
            "{err}"
        );

        // An image whose rows do not cover its grid.
        let grid = GridSize { h: 2, w: 2 };
        let short = vec![0.0; 3 * hidden];
        let err = model
            .assemble_prompt(&[PromptPiece::Image { rows: &short, grid }], 0)
            .expect_err("short image");
        assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");

        // None of the above touched the sequence.
        assert_eq!(model.recurrent().token_count(), 0);
        assert_eq!(model.kv_cache().seq_len(), 0);
        assert_eq!(model.rope_delta(), 0);
    }
}
