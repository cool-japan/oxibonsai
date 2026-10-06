//! The fixture's fixed dimensions, the variant specification
//! ([`HybridFixtureSpec`] and the 24-variant matrix) and the dimensions
//! derived from a spec ([`Dims`]).

use oxibonsai_core::gguf::writer::TensorType;

use super::wide_variant::{FFN_WIDE, HEAD_DIM_WIDE};

// ═════════════════════════════════════════════════════════════════════════
// 7. Fixture specification and derived dimensions
// ═════════════════════════════════════════════════════════════════════════

/// Number of decoder layers in every generated fixture. `full_attention_interval
/// == 4` selects layers 3 and 7 as full-attention (16 % / 84 % split, the
/// same ratio class as the real 27B) and the rest as Gated-DeltaNet.
pub const BLOCK_COUNT: usize = 8;
pub const FULL_ATTENTION_INTERVAL: usize = 4;
pub const HIDDEN: usize = 256;
pub const FFN: usize = 384;
pub const N_HEAD: usize = 4;
pub const N_KV: usize = 2;
pub const HEAD_DIM: usize = 32;
pub const ROPE_DIM: usize = 16;
pub const ROPE_SECTIONS: [i32; 4] = [3, 3, 2, 0];
pub const ROPE_FREQ_BASE: f64 = 10_000.0;
pub const SSM_CONV_KERNEL: usize = 4;
pub const SSM_STATE_SIZE: usize = 32; // head_k_dim == head_v_dim
pub const SSM_TIME_STEP_RANK: usize = 4; // n_v_heads
pub const VOCAB: usize = 64;
pub const CONTEXT_LENGTH: usize = 512;
pub const HADAMARD_BLOCK_SIZE: usize = 128;
pub const RMS_EPS: f64 = 1e-6;
/// Number of tokens the embedded reference forward runs over — long enough
/// to exceed the conv1d kernel width and exercise multi-position causal
/// attention and multi-step GDN recurrence.
pub const T_TOKENS: usize = 6;

/// The six quantization formats a hybrid loader must bind (design §1.1).
pub const ALL_QUANT_TYPES: [TensorType; 6] = [
    TensorType::F32,
    TensorType::PQ2_0,
    TensorType::PTQ1_0,
    TensorType::Q2_0G64,
    TensorType::TQ2_0_g128,
    TensorType::Q1_0G128,
];

/// One fixture variant: a quantization format crossed with Hadamard
/// on/off and V-head grouping on/off (design §7.1's `HybridFixtureSpec`),
/// plus the Hadamard-block geometry: `hidden`/`hadamard_block` are
/// independent spec fields — not bare constants — precisely so a variant
/// can ask for a wider hidden size than the default
/// [`HIDDEN`]/[`HADAMARD_BLOCK_SIZE`] pair without touching every other
/// variant. [`all_variant_specs`]'s 24 canonical variants all use the
/// narrow (`HIDDEN`, `HADAMARD_BLOCK_SIZE`) pair, reproducing their
/// previous byte-for-byte output exactly; [`hadamard_1024_variant_spec`]
/// is the one wide variant.
#[derive(Debug, Clone, Copy)]
pub struct HybridFixtureSpec {
    pub quant: TensorType,
    pub hadamard: bool,
    pub gdn_v_grouped: bool,
    pub seed: u64,
    pub hidden: usize,
    pub hadamard_block: usize,
}

/// Every canonical (quant, hadamard, grouped) combination — the full cross
/// product (`6 x 2 x 2 = 24`), a strict superset of design §7.1's original
/// 16-variant matrix (4 quant formats it named explicitly) and this
/// generator's own 6-format list. Every variant uses the narrow
/// `HIDDEN`/`HADAMARD_BLOCK_SIZE` pair; see [`hadamard_1024_variant_spec`]
/// for the wide one.
pub fn all_variant_specs(base_seed: u64) -> Vec<HybridFixtureSpec> {
    let mut specs = Vec::with_capacity(ALL_QUANT_TYPES.len() * 4);
    for (qi, &quant) in ALL_QUANT_TYPES.iter().enumerate() {
        for (hi, &hadamard) in [false, true].iter().enumerate() {
            for (gi, &gdn_v_grouped) in [false, true].iter().enumerate() {
                let mix = (qi as u64) * 100 + (hi as u64) * 10 + (gi as u64);
                let seed = base_seed ^ mix.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                specs.push(HybridFixtureSpec {
                    quant,
                    hadamard,
                    gdn_v_grouped,
                    seed,
                    hidden: HIDDEN,
                    hadamard_block: HADAMARD_BLOCK_SIZE,
                });
            }
        }
    }
    specs
}

/// Every dimension derivable from a [`HybridFixtureSpec`]. `gdn_v_grouped`
/// selects `n_k_heads`: `2` (so `v_per_k == 2`, the real interesting case)
/// when grouped, or `4` (`v_per_k == 1`, the trivial/identity case) when
/// not — a V-head map with `rep > 1` and `gdn_v_grouped == false` is
/// exactly the configuration design §3.3 says a hybrid loader must
/// *refuse*, so the ordinary parity matrix never constructs it (see
/// [`build_invalid_ungrouped_fixture`] for the dedicated negative fixture).
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub inner_size: usize,
    pub conv_dim: usize,
    /// [`HybridFixtureSpec::hidden`], repeated here so every place that
    /// already threads `Dims`/`plan.dims` through can read the variant's
    /// hidden width without a second lookup into `spec`.
    pub hidden: usize,
    /// [`HybridFixtureSpec::hadamard_block`], repeated here for the same
    /// reason as `hidden`.
    pub hadamard_block: usize,
    /// Attention head width: [`HEAD_DIM`] at the narrow `hidden`,
    /// [`HEAD_DIM_WIDE`] at [`HIDDEN_WIDE`] (`N_HEAD * head_dim ==
    /// hidden` either way, by construction of the two constant pairs).
    pub head_dim: usize,
    /// FFN width: [`FFN`] at the narrow `hidden`, [`FFN_WIDE`] at
    /// [`HIDDEN_WIDE`].
    pub ffn: usize,
}

impl Dims {
    pub fn from_spec(spec: &HybridFixtureSpec) -> Self {
        let n_v_heads = SSM_TIME_STEP_RANK;
        let n_k_heads = if spec.gdn_v_grouped { 2 } else { n_v_heads };
        // `spec.hidden != HIDDEN` selects the wide preset as a whole (there
        // are exactly two: narrow and [`HIDDEN_WIDE`] — see
        // [`hadamard_1024_variant_spec`]). Every dimension that a folded
        // tensor's input width depends on scales together, so a mismatched
        // pair (e.g. a wide `hidden` with the narrow `head_dim`) can never
        // arise from a spec built by this file's own two constructors.
        let wide = spec.hidden != HIDDEN;
        let head_k_dim = if wide { HEAD_DIM_WIDE } else { SSM_STATE_SIZE };
        let head_v_dim = head_k_dim;
        let inner_size = n_v_heads * head_v_dim;
        let conv_dim = 2 * head_k_dim * n_k_heads + inner_size;
        let head_dim = if wide { HEAD_DIM_WIDE } else { HEAD_DIM };
        let ffn = if wide { FFN_WIDE } else { FFN };
        Self {
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            inner_size,
            conv_dim,
            hidden: spec.hidden,
            hadamard_block: spec.hadamard_block,
            head_dim,
            ffn,
        }
    }

    pub fn v_per_k(&self) -> usize {
        self.n_v_heads / self.n_k_heads.max(1)
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL)
    }

    pub fn attn_output_input_width(&self) -> usize {
        N_HEAD * self.head_dim
    }
}
