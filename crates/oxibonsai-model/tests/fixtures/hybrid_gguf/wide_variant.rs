//! The 1024-wide Hadamard variant: the real 27B's own block geometry
//! (`prism.hadamard.block_size == 1024`) at fixture scale. Its widths are
//! chosen so every foldable tensor's input is a whole multiple of the
//! 1024-wide block, which the narrow variants (two 128-wide blocks per hidden
//! vector) never exercise.

use oxibonsai_core::gguf::writer::TensorType;

use super::spec::HybridFixtureSpec;

/// The real 27B's own Hadamard block size (`prism.hadamard.block_size`,
/// design §7.0): every fixture variant with `hidden == HIDDEN_WIDE` uses
/// this instead of [`HADAMARD_BLOCK_SIZE`], so a single tensor's folded
/// width spans exactly one block (`HIDDEN_WIDE`) or a small whole number of
/// them (`FFN_WIDE`), rather than the many small `HADAMARD_BLOCK_SIZE`-wide
/// blocks the narrow fixtures use.
pub const HADAMARD_BLOCK_SIZE_WIDE: usize = 1024;
/// A hidden width that is itself exactly one Hadamard block: the narrow
/// fixtures' `HIDDEN` (256) is only two 128-wide blocks, so they never
/// exercise the kernel's `128->1024` NEON butterfly stages the real 27B's
/// own 1024-wide blocks need.
/// [`HybridFixtureSpec::hidden`]/[`HybridFixtureSpec::hadamard_block`]
/// select between this and [`HIDDEN`]/[`HADAMARD_BLOCK_SIZE`] per variant.
pub const HIDDEN_WIDE: usize = 1024;
/// Attention head width at [`HIDDEN_WIDE`]: `N_HEAD * HEAD_DIM_WIDE ==
/// 2048`, so `attn_output`'s folded (input) width is two whole Hadamard
/// blocks — the real 27B's `attn_output`/`ssm_out` share this same
/// property (both fold on a width that is a whole multiple of 1024).
/// Reused for the Gated-DeltaNet `head_k_dim`/`head_v_dim` too, so
/// `inner_size = SSM_TIME_STEP_RANK * HEAD_DIM_WIDE == 2048` gives
/// `ssm_out` the same width, mirroring the real model's own
/// `attn_output`/`ssm_out` share.
///
/// Distinct from [`HIDDEN_WIDE`] to mirror the real 27B, whose rotated
/// widths (5120 / 6144 / 17408) are pairwise distinct except that one
/// designed share. It is not an aliasing guard: `HadamardHook::rotate`
/// returns a slice that mutably borrows its `HadamardScratch`, so two claims
/// on one width are sequential by construction. At a 256-wide head (where
/// `attn_output`'s width equals `HIDDEN_WIDE`) the worst per-token logit
/// miss was 1.6e-4 over the absolute 1e-4 band — consistent with f32
/// accumulation over a wider rotated reduction, judged by an absolute rather
/// than a scale-relative band; that geometry would need a scale-relative
/// bound, not a different buffer layout. At this width the variant passes
/// at the same 1e-4 band the narrow variants use.
pub const HEAD_DIM_WIDE: usize = 512;
/// FFN width at [`HIDDEN_WIDE`]: three whole [`HADAMARD_BLOCK_SIZE_WIDE`]
/// blocks, distinct from both [`HIDDEN_WIDE`] and the
/// `attn_output`/`ssm_out` width above (same reasoning as
/// [`HEAD_DIM_WIDE`]'s own doc comment) — exercising multi-block folding
/// within a single tensor (the narrow fixtures' `FFN` is not a whole
/// multiple of `HADAMARD_BLOCK_SIZE` at all, so this is also new coverage).
pub const FFN_WIDE: usize = 3072;

/// The one 1024-wide variant: `hidden == HIDDEN_WIDE == hadamard_block`, so
/// `token_embd`/`output`/`ffn_gate`/`ffn_up`/`attn_q`/`attn_k`/`attn_v`/
/// `attn_qkv`/`attn_gate` each fold on exactly one 1024-wide Hadamard
/// block, `ffn_down` on three, and `attn_output`/`ssm_out` on two
/// (`HEAD_DIM_WIDE`-derived, see its own doc comment) — every foldable
/// tensor's input width is a whole multiple of 1024, the same property the
/// real 27B's own widths (5120/17408/6144) have, and every one of the
/// three widths is pairwise distinct except for the `attn_output`/`ssm_out`
/// share, mirroring the real 27B's own three-distinct-width shape exactly
/// (see `HEAD_DIM_WIDE`'s doc comment for why that distinctness matters).
/// `PQ2_0`, Hadamard on, V-head grouped: the real release band, and — like
/// every ternary quant format this generator draws from `{-1, 0, +1}` — an
/// exact (lossless) dequantization oracle, so any mismatch this variant
/// finds is arithmetic, never quantization noise.
#[must_use]
pub fn hadamard_1024_variant_spec(base_seed: u64) -> HybridFixtureSpec {
    HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: base_seed ^ 0x1024_1024_1024_1024,
        hidden: HIDDEN_WIDE,
        hadamard_block: HADAMARD_BLOCK_SIZE_WIDE,
    }
}
