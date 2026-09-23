//! GGUF tensor → hybrid-layer binding (design §3.8).
//!
//! Every tensor of a `qwen35` file is resolved **by name** and checked
//! against [`HybridConfig`] before it is stored. A missing or mis-shaped
//! tensor is a hard [`ModelError`] naming the expected and the actual shape
//! — never a warning, never a silently zeroed buffer: a 27B that loads with
//! the wrong `ssm_conv1d` transposition produces fluent, wrong text.
//!
//! # Name table (all present in every real 27B file)
//!
//! ```text
//! token_embd.weight                  [5120, 248320]  quant, INVERSE-rotated
//! output_norm.weight                 [5120]          F32
//! output.weight                      [5120, 248320]  quant, folded
//! blk.N.attn_norm.weight             [5120]          F32
//! blk.N.post_attention_norm.weight   [5120]          F32
//! blk.N.ffn_{gate,up}.weight         [5120, 17408]   quant, folded
//! blk.N.ffn_down.weight              [17408, 5120]   quant, folded
//! -- full layers ((N+1) % 4 == 0) --
//! blk.N.attn_q.weight                [5120, 12288]   quant, folded (q|gate)
//! blk.N.attn_{k,v}.weight            [5120, 1024]    quant, folded
//! blk.N.attn_output.weight           [6144, 5120]    quant, folded
//! blk.N.attn_{q,k}_norm.weight       [256]           F32
//! -- linear layers --
//! blk.N.attn_qkv.weight              [5120, 10240]   quant, folded
//! blk.N.attn_gate.weight             [5120, 6144]    quant, folded
//! blk.N.ssm_alpha.weight             [5120, 48]      BF16, NOT folded
//! blk.N.ssm_beta.weight              [5120, 48]      BF16, NOT folded
//! blk.N.ssm_conv1d.weight            [4, 10240]      F32,  NOT folded
//! blk.N.ssm_a                        [48]            F32,  NOT folded (< 0)
//! blk.N.ssm_dt.bias                  [48]            F32,  NOT folded
//! blk.N.ssm_norm.weight              [128]           F32,  NOT folded
//! blk.N.ssm_out.weight               [6144, 5120]    quant, folded, GROUPED
//! ```
//!
//! GGUF shapes are `[in, out]` (`ne[0]` is the row length), so a
//! `[5120, 17408]` tensor is 17408 rows of 5120 — that is the order
//! [`expect_shape`] compares against and the order `Linear*::new` takes
//! (`out_features` first).
//!
//! # Two load-time refusals that exist because the failure is silent
//!
//! * **`ssm_a` must be `A = -exp(A_log)`, i.e. `<= 0`.** A positive element
//!   makes the decay `exp(g) > 1` and the recurrence diverge — hundreds of
//!   tokens later, far from the cause. [`oxibonsai_kernels::gated_delta_net::validate_a_neg`]
//!   is called here, at **load**, so a bad checkpoint names the offending
//!   index and value immediately (B2-05's loader-side requirement).
//! * **The `a_neg` / `dt_bias` argument order.** `gdn_step_f32` takes
//!   `(…, dt_bias, a_neg, …)` while `gdn_step` takes `(…, a_neg, dt_bias,
//!   …)`; both are `[n_v_heads]` `f32`, so a swap compiles and only shows up
//!   when `dt_bias` happens to hold a positive element. [`GdnGateWeights`]
//!   closes the class: the two vectors are bound **by field name** once,
//!   here, and every call site takes a ready-made [`GdnGates`] instead of
//!   two interchangeable slices.
//! * **The v-head index space.** `ssm_a` and `ssm_dt.bias` are GGUF *rows* in
//!   **tiled** v-head order (design §3.3's table: `ssm_a`/`ssm_dt.bias` read
//!   with scalar index `map.tiled(m)`), but every kernel entry point this
//!   crate calls indexes its gates by the **grouped** v-head index
//!   ([`crate::hybrid::vhead_map::GDN_HEAD_ORDER`] is
//!   [`oxibonsai_kernels::gated_delta_net::GdnHeadOrder::Grouped`]). Handing
//!   the raw tiled rows straight to [`GdnGateWeights`] would silently pair
//!   45 of the 27B's 48 v-heads with the wrong decay constant — the same
//!   class of bug the `v`/`z` activations avoid via
//!   [`crate::hybrid::vhead_map::VHeadMap::gather_grouped`] and
//!   `LinearScratch::{alpha_grouped,beta_grouped}`. [`bind_gdn_gates`]
//!   closes it the same way: both vectors are re-indexed through
//!   [`crate::hybrid::vhead_map::VHeadMap::gather_grouped_scalar`] before
//!   [`GdnGateWeights::new`] ever sees them, so `a_neg()[m]`/`dt_bias()[m]`
//!   are grouped like every other v-indexed quantity this package produces.

use std::sync::Arc;

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::gguf::quant_resolve::{
    compute_extents, resolve_type_42_with_sample, AMBIGUOUS_TYPE_ID,
};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::{row_size_bytes, TensorInfo};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_ternary::{sniff_sample_byte_cap, SNIFF_DEFAULT_BLOCKS};
use oxibonsai_core::tensor::QK1_0_G128;
use oxibonsai_core::{
    BlockPQ2_0, BlockPTQ1_0, BlockQ1_0G128, BlockQ2_0G64, BlockTQ2_0_g128, BonsaiError, QK_PQ2_0,
    QK_PTQ1_0, QK_Q2_0_G64, QK_TQ2_0_G128,
};
use oxibonsai_kernels::gated_delta_net::{validate_a_neg, GdnBeta, GdnDecay, GdnGates};
use oxibonsai_kernels::KernelDispatcher;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::vhead_map::VHeadMap;
use crate::layers::linear::{
    Linear1Bit, LinearLayer, LinearPQ2_0, LinearPTQ1_0, LinearQ2_0G64, LinearTernary,
};
use crate::layers::rms_norm::RmsNorm;
use oxibonsai_core::bf16::bf16_to_f32;

/// Tensor names a hybrid stack binds, as GGUF spells them.
pub mod names {
    /// `token_embd.weight` — quantized **and** inverse-rotated.
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    /// `output_norm.weight`.
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    /// `output.weight` — the LM head; never tied on a `qwen35` file.
    pub const OUTPUT: &str = "output.weight";

    /// Pre-attention RMSNorm.
    pub const ATTN_NORM: &str = "attn_norm.weight";
    /// Pre-FFN RMSNorm — GGUF spells it `post_attention_norm`, **not**
    /// `ffn_norm` as the dense Qwen3 files do.
    pub const POST_ATTENTION_NORM: &str = "post_attention_norm.weight";
    /// SwiGLU gate projection.
    pub const FFN_GATE: &str = "ffn_gate.weight";
    /// SwiGLU up projection.
    pub const FFN_UP: &str = "ffn_up.weight";
    /// SwiGLU down projection.
    pub const FFN_DOWN: &str = "ffn_down.weight";

    /// Fused `[q | gate]` projection of a full-attention layer.
    pub const ATTN_Q: &str = "attn_q.weight";
    /// Key projection.
    pub const ATTN_K: &str = "attn_k.weight";
    /// Value projection.
    pub const ATTN_V: &str = "attn_v.weight";
    /// Attention output projection.
    pub const ATTN_OUTPUT: &str = "attn_output.weight";
    /// Per-head query RMSNorm.
    pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
    /// Per-head key RMSNorm.
    pub const ATTN_K_NORM: &str = "attn_k_norm.weight";

    /// Fused `[q | k | v]` projection of a linear-attention layer.
    pub const ATTN_QKV: &str = "attn_qkv.weight";
    /// Gated-DeltaNet output gate `z`.
    pub const ATTN_GATE: &str = "attn_gate.weight";
    /// Raw decay projection (BF16, NOT folded).
    pub const SSM_ALPHA: &str = "ssm_alpha.weight";
    /// Raw β projection (BF16, NOT folded).
    pub const SSM_BETA: &str = "ssm_beta.weight";
    /// Depthwise causal conv1d weights, `[conv_kernel, conv_dim]`.
    pub const SSM_CONV1D: &str = "ssm_conv1d.weight";
    /// `A = -exp(A_log)`, `[n_v_heads]`, strictly non-positive. No
    /// `.weight` suffix in GGUF.
    pub const SSM_A: &str = "ssm_a";
    /// `dt` bias, `[n_v_heads]`. No `.weight` suffix in GGUF.
    pub const SSM_DT_BIAS: &str = "ssm_dt.bias";
    /// Gated RMSNorm weight, shared across v-heads, `[head_v_dim]`.
    pub const SSM_NORM: &str = "ssm_norm.weight";
    /// Gated-DeltaNet output projection (folded, GROUPED columns).
    pub const SSM_OUT: &str = "ssm_out.weight";
}

/// Tensor suffixes every layer carries, whichever kind it is.
pub const SHARED_LAYER_TENSORS: &[&str] = &[
    names::ATTN_NORM,
    names::POST_ATTENTION_NORM,
    names::FFN_GATE,
    names::FFN_UP,
    names::FFN_DOWN,
];

/// Tensor suffixes only a full-attention layer carries.
pub const FULL_LAYER_TENSORS: &[&str] = &[
    names::ATTN_Q,
    names::ATTN_K,
    names::ATTN_V,
    names::ATTN_OUTPUT,
    names::ATTN_Q_NORM,
    names::ATTN_K_NORM,
];

/// Tensor suffixes only a Gated-DeltaNet layer carries.
pub const LINEAR_LAYER_TENSORS: &[&str] = &[
    names::ATTN_QKV,
    names::ATTN_GATE,
    names::SSM_ALPHA,
    names::SSM_BETA,
    names::SSM_CONV1D,
    names::SSM_A,
    names::SSM_DT_BIAS,
    names::SSM_NORM,
    names::SSM_OUT,
];

/// `blk.<layer>.<suffix>`.
#[must_use]
pub fn block_tensor(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

// ─────────────────────────────────────────────────────────────────────────
//  BF16 matrix (ssm_alpha / ssm_beta)
// ─────────────────────────────────────────────────────────────────────────

/// A zero-copy BF16 weight matrix, `[out_features][in_features]` row-major.
///
/// `ssm_alpha`/`ssm_beta` are `[5120, 48]` BF16 on every linear layer: 480 KB
/// each, 94 MB across both 27B files. Widening them to f32 at load would
/// double that for no gain, and a dedicated `gemv_bf16` SIMD tier would be
/// wasted on a 48-row matrix — so the bytes stay mmap'd and are widened one
/// element at a time inside the dot product.
///
/// The bytes are kept as `&'a [u8]` rather than `&'a [u16]` deliberately: a
/// GGUF tensor offset is only guaranteed aligned to `general.alignment`,
/// and a `&[u8] → &[u16]` cast at an odd address is instant UB. Decoding
/// with [`u16::from_le_bytes`] costs nothing here and needs no `unsafe`.
#[derive(Debug, Clone)]
pub struct Bf16Matrix<'a> {
    storage: GateStorage<'a>,
    out_features: usize,
    in_features: usize,
}

/// How one gate projection's weights are held (gatekeeper REQUIRED #16).
///
/// Every Bonsai **2** file stores `ssm_alpha`/`ssm_beta` as BF16, which is
/// read zero-copy straight out of the mmap. The gen-1 27B builds
/// (`Bonsai-27B-Q1_0`, `Ternary-Bonsai-27B-{PQ2_0,Q2_0}`) store the same
/// two tensors **quantized** (ggml ids 41/142/42 per
/// `gguf_headers_summary.txt`), and `ModelVariant::Bonsai27B` advertises
/// those files as supported -- so refusing anything but BF16 made a
/// supported variant unloadable. A quantized gate is dequantized once at
/// load: `n_v_heads x hidden` is 48 x 5120 = 983 KB for the 27B, i.e. a
/// rounding error beside the 7.2 GB of weights, and the alternative (a
/// per-token quantized GEMV for a 48-row matrix) would be slower.
#[derive(Debug, Clone)]
enum GateStorage<'a> {
    /// Zero-copy BF16 bytes, `[out_features][in_features]` row-major.
    Bf16(&'a [u8]),
    /// Dequantized `f32`, same shape.
    Dense(Arc<[f32]>),
}

impl<'a> Bf16Matrix<'a> {
    /// Wrap a tensor's raw bytes.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `bytes.len() != 2 * out_features *
    /// in_features`.
    pub fn new(
        name: &str,
        bytes: &'a [u8],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        let expected = out_features
            .checked_mul(in_features)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: name.to_string(),
                expected: "out_features * in_features * 2 representable as usize".to_string(),
                actual: format!("{out_features} x {in_features}"),
            })?;
        if bytes.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![expected],
                actual: vec![bytes.len()],
            });
        }
        Ok(Self {
            storage: GateStorage::Bf16(bytes),
            out_features,
            in_features,
        })
    }

    /// Rows (`48` for `ssm_alpha`).
    #[inline]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Columns (`5120` for `ssm_alpha`).
    #[inline]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Wrap already-dequantized `f32` weights (a gen-1 27B checkpoint whose
    /// `ssm_alpha`/`ssm_beta` are quantized rather than BF16).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `values.len() != out_features *
    /// in_features`.
    pub fn from_dense(
        name: &str,
        values: Vec<f32>,
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        let expected =
            out_features
                .checked_mul(in_features)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: name.to_string(),
                    expected: "out_features * in_features representable as usize".to_string(),
                    actual: format!("{out_features} x {in_features}"),
                })?;
        if values.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![expected],
                actual: vec![values.len()],
            });
        }
        Ok(Self {
            storage: GateStorage::Dense(values.into()),
            out_features,
            in_features,
        })
    }

    /// One element, widened to `f32`.
    #[inline]
    #[must_use]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        let flat = row * self.in_features + col;
        match &self.storage {
            GateStorage::Bf16(bytes) => {
                let index = flat * 2;
                match bytes.get(index..index + 2) {
                    Some(pair) => bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])),
                    None => 0.0,
                }
            }
            GateStorage::Dense(values) => values.get(flat).copied().unwrap_or(0.0),
        }
    }

    /// `true` when the weights are held as dequantized `f32` rather than as
    /// a zero-copy BF16 view (a gen-1 27B checkpoint).
    #[inline]
    #[must_use]
    pub fn is_dequantized(&self) -> bool {
        matches!(self.storage, GateStorage::Dense(_))
    }

    /// `y = W x`, widening each weight on the fly.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `x` is not `in_features` long or
    /// `y` is shorter than `out_features`.
    pub fn forward_vec(&self, x: &[f32], y: &mut [f32]) -> ModelResult<()> {
        if x.len() != self.in_features {
            return Err(ModelError::ShapeMismatch {
                name: "Bf16Matrix input".to_string(),
                expected: vec![self.in_features],
                actual: vec![x.len()],
            });
        }
        if y.len() < self.out_features {
            return Err(ModelError::ShapeMismatch {
                name: "Bf16Matrix output".to_string(),
                expected: vec![self.out_features],
                actual: vec![y.len()],
            });
        }
        if let GateStorage::Dense(values) = &self.storage {
            for (row, out) in y.iter_mut().take(self.out_features).enumerate() {
                let base = row * self.in_features;
                let mut acc = 0.0f32;
                for (col, xv) in x.iter().enumerate() {
                    acc += values.get(base + col).copied().unwrap_or(0.0) * xv;
                }
                *out = acc;
            }
            return Ok(());
        }
        let GateStorage::Bf16(bytes) = &self.storage else {
            // Unreachable: the `Dense` arm returned above.
            return Err(ModelError::Internal(
                "Bf16Matrix storage discriminant changed under us".to_string(),
            ));
        };
        for (row, out) in y.iter_mut().take(self.out_features).enumerate() {
            let base = row * self.in_features * 2;
            let mut acc = 0.0f32;
            for (col, xv) in x.iter().enumerate() {
                let index = base + col * 2;
                if let Some(pair) = bytes.get(index..index + 2) {
                    acc += bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])) * xv;
                }
            }
            *out = acc;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Gated-DeltaNet gate binding (the argument-order footgun, closed)
// ─────────────────────────────────────────────────────────────────────────

/// The two `[n_v_heads]` gate vectors of one Gated-DeltaNet layer, bound
/// **by name**.
///
/// `ssm_a` and `ssm_dt.bias` are both `[48]` `f32` and the kernel entry
/// points disagree on their order (`gdn_step_f32(…, dt_bias, a_neg, …)` vs
/// `gdn_step(…, a_neg, dt_bias, …)`). Holding them in named fields and
/// handing out a ready-made [`GdnGates`] means no call site ever writes the
/// two slices positionally, so no future refactor can transpose them.
///
/// Both vectors are stored in **grouped** v-head order — [`bind_gdn_gates`]
/// re-indexes them from the GGUF's tiled row order before they reach
/// [`GdnGateWeights::new`], exactly like `LinearScratch::{alpha_grouped,
/// beta_grouped}` and the `v`/`z` activations. A caller must feed
/// [`GdnGateWeights::gates`] an `alpha_raw`/`beta_raw` pair in the same
/// grouped order (the module docs' third silent-failure class).
#[derive(Debug, Clone)]
pub struct GdnGateWeights {
    /// `ssm_a`: `A = -exp(A_log)`, one non-positive value per v-head, in
    /// **grouped** v-head order.
    a_neg: Arc<[f32]>,
    /// `ssm_dt.bias`: one value per v-head, sign unconstrained, in
    /// **grouped** v-head order.
    dt_bias: Arc<[f32]>,
}

impl GdnGateWeights {
    /// Bind the pair, validating both lengths and the sign of `a_neg`.
    ///
    /// The `a_neg` / `dt_bias` parameters are named, not interchangeable:
    /// pass the tensor called `ssm_a` as `a_neg` and `ssm_dt.bias` as
    /// `dt_bias`. A transposition is caught here whenever `dt_bias` holds a
    /// positive element, and by
    /// `gates_are_bound_by_name_not_position` unconditionally.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when either vector is not `n_v_heads`
    /// long; [`ModelError::InvalidTensor`] (from
    /// [`validate_a_neg`]) naming the index and value when `a_neg` holds a
    /// positive or NaN element.
    pub fn new(a_neg: Vec<f32>, dt_bias: Vec<f32>, n_v_heads: usize) -> ModelResult<Self> {
        for (name, vector) in [(names::SSM_A, &a_neg), (names::SSM_DT_BIAS, &dt_bias)] {
            if vector.len() != n_v_heads {
                return Err(ModelError::ShapeMismatch {
                    name: name.to_string(),
                    expected: vec![n_v_heads],
                    actual: vec![vector.len()],
                });
            }
        }
        validate_a_neg(&a_neg)
            .map_err(|e| ModelError::InvalidTensor(format!("{}: {e}", names::SSM_A)))?;
        Ok(Self {
            a_neg: a_neg.into(),
            dt_bias: dt_bias.into(),
        })
    }

    /// `ssm_a`, `[n_v_heads]`, every element `<= 0`.
    #[inline]
    #[must_use]
    pub fn a_neg(&self) -> &[f32] {
        &self.a_neg
    }

    /// `ssm_dt.bias`, `[n_v_heads]`.
    #[inline]
    #[must_use]
    pub fn dt_bias(&self) -> &[f32] {
        &self.dt_bias
    }

    /// Build the kernel's gate pair for one call from this layer's weights
    /// and the token's raw `alpha`/`beta` projections.
    ///
    /// Constructed with **struct-field syntax**, so `dt_bias` and `a_neg`
    /// cannot be transposed by a positional edit.
    #[must_use]
    pub fn gates<'g>(&'g self, alpha_raw: &'g [f32], beta_raw: &'g [f32]) -> GdnGates<'g> {
        GdnGates {
            beta: GdnBeta::Raw(beta_raw),
            decay: GdnDecay::ScalarRaw {
                alpha_raw,
                dt_bias: &self.dt_bias,
                a_neg: &self.a_neg,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Token embedding (row-wise dequantization)
// ─────────────────────────────────────────────────────────────────────────

/// `token_embd.weight`, kept quantized and decoded one row per token.
///
/// The 27B's table is 248 320 × 5120 — 5 GB dequantized, 1.3 GB as stored.
/// Only one row is needed per token, and that row still has to go through
/// [`crate::hybrid::HadamardHook::inverse_embedding`] afterwards, so nothing
/// is cached.
#[derive(Debug)]
pub enum HybridEmbedding<'a> {
    /// PrismML `PQ2_0` (ggml 142), and the `d`-first reading of ggml 42.
    Pq2_0(&'a [BlockPQ2_0]),
    /// PrismML `PTQ1_0` (ggml 143).
    Ptq1_0(&'a [BlockPTQ1_0]),
    /// Mainline group-64 `Q2_0`.
    Q2_0G64(&'a [BlockQ2_0G64]),
    /// Legacy OxiBonsai ternary (`qs` first, ggml 42 @ g128).
    Ternary(&'a [BlockTQ2_0_g128]),
    /// 1-bit `Q1_0_g128` (gen-1 `Bonsai-27B-Q1_0`).
    OneBit(&'a [BlockQ1_0G128]),
    /// An unquantized table, already widened to `f32`.
    Dense(Vec<f32>),
}

impl HybridEmbedding<'_> {
    /// Copy the embedding row of `token` into `out[..hidden]`, dequantizing
    /// exactly that row.
    ///
    /// The result is still in the **rotated** basis for a folded model; the
    /// caller applies the inverse Hadamard transform (design §3.5).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `out` is not `hidden` long;
    /// [`ModelError::PositionOutOfRange`] when `token` is past the table;
    /// [`ModelError::Core`] from a block decoder.
    pub fn row(&self, token: u32, hidden: usize, out: &mut [f32]) -> ModelResult<()> {
        if out.len() != hidden {
            return Err(ModelError::ShapeMismatch {
                name: "token embedding row".to_string(),
                expected: vec![hidden],
                actual: vec![out.len()],
            });
        }
        let token = token as usize;
        match self {
            Self::Pq2_0(blocks) => {
                let range = row_blocks(token, hidden, QK_PQ2_0, blocks.len())?;
                BlockPQ2_0::dequant(&blocks[range], out).map_err(ModelError::Core)
            }
            Self::Ptq1_0(blocks) => {
                let range = row_blocks(token, hidden, QK_PTQ1_0, blocks.len())?;
                BlockPTQ1_0::dequant(&blocks[range], out).map_err(ModelError::Core)
            }
            Self::Q2_0G64(blocks) => {
                let range = row_blocks(token, hidden, QK_Q2_0_G64, blocks.len())?;
                BlockQ2_0G64::dequant(&blocks[range], out).map_err(ModelError::Core)
            }
            Self::Ternary(blocks) => {
                let range = row_blocks(token, hidden, QK_TQ2_0_G128, blocks.len())?;
                BlockTQ2_0_g128::dequant(&blocks[range], out).map_err(ModelError::Core)
            }
            Self::OneBit(blocks) => {
                // The one hand-written decode here: `Q1_0_g128` has no
                // `dequant(blocks, out)` in core (only whole-tensor
                // `OneBitTensor::dequantize_all`), so one row is expanded
                // in place. Bit `j` of `qs[j / 8]`, LSB first, selects
                // `+d` or `-d` — transcribed from `dequantize_all`, and
                // pinned by `one_bit_embedding_rows_decode_to_plus_minus_d`.
                let range = row_blocks(token, hidden, QK1_0_G128, blocks.len())?;
                for (b, block) in blocks[range].iter().enumerate() {
                    let d = block.d.to_f32();
                    for j in 0..QK1_0_G128 {
                        let bit = (block.qs[j / 8] >> (j % 8)) & 1;
                        if let Some(slot) = out.get_mut(b * QK1_0_G128 + j) {
                            *slot = if bit != 0 { d } else { -d };
                        }
                    }
                }
                Ok(())
            }
            Self::Dense(table) => {
                let lo = token
                    .checked_mul(hidden)
                    .ok_or(ModelError::PositionOutOfRange {
                        pos: token,
                        max: table.len(),
                    })?;
                let row = table
                    .get(lo..lo + hidden)
                    .ok_or(ModelError::PositionOutOfRange {
                        pos: token,
                        max: table.len() / hidden.max(1),
                    })?;
                out.copy_from_slice(row);
                Ok(())
            }
        }
    }
}

/// The block range covering row `token` of a `hidden`-wide quantized table.
fn row_blocks(
    token: usize,
    hidden: usize,
    group: usize,
    total_blocks: usize,
) -> ModelResult<std::ops::Range<usize>> {
    if group == 0 || !hidden.is_multiple_of(group) {
        return Err(ModelError::ShapeInvariant {
            tensor: names::TOKEN_EMBD.to_string(),
            expected: format!("hidden a whole multiple of the block size {group}"),
            actual: format!("hidden = {hidden}"),
        });
    }
    let per_row = hidden / group;
    let lo = token
        .checked_mul(per_row)
        .ok_or(ModelError::PositionOutOfRange {
            pos: token,
            max: total_blocks,
        })?;
    let hi = lo
        .checked_add(per_row)
        .ok_or(ModelError::PositionOutOfRange {
            pos: token,
            max: total_blocks,
        })?;
    if hi > total_blocks {
        return Err(ModelError::PositionOutOfRange {
            pos: token,
            max: total_blocks / per_row.max(1),
        });
    }
    Ok(lo..hi)
}

// ─────────────────────────────────────────────────────────────────────────
//  ggml wire id 42 resolution
// ─────────────────────────────────────────────────────────────────────────

/// Environment variable name of design SS1.3's layout override.
pub const FORCE_Q2_LAYOUT_ENV: &str = "OXI_FORCE_Q2_LAYOUT";

/// Design SS1.3's escape hatch: an explicit reading for the ambiguous ggml
/// wire id 42, set by the operator when the automatic evidence is
/// inconclusive.
///
/// ggml id 42 is used by three mutually incompatible on-disk layouts (the
/// legacy PrismML `qs`-first group-128 block, PrismML's `d`-first group-128
/// block, and mainline `Q2_0` at group 64). `resolve_type_42_with_sample`
/// decides between them from the tensor offset table and a data sample and
/// is conclusive on every real file; a degenerate sample (all-zero, or a
/// synthetic fixture too small to sample) is the one case where it is not,
/// and a `d`-first file with such a sample would otherwise be decoded as
/// `qs`-first with nothing but a `warn!` to show for it.
///
/// Accepted values, case-insensitive: `d-first` / `dfirst` / `pq2` (PrismML
/// `d`-first, group 128), `qs-first` / `qsfirst` / `tq2` / `legacy` (the
/// historical OxiBonsai reading), `g64` / `q2_0_g64` (mainline group 64).
/// The override applies **before** the sniff, because an operator who has
/// gone to the trouble of setting it knows something the bytes do not say.
///
/// # Errors
///
/// [`ModelError::InvalidTensor`] naming the variable and the accepted
/// values when it is set to anything else -- a typo must not silently
/// degrade to the automatic path, which is the failure mode this hatch
/// exists to prevent.
pub fn forced_q2_layout() -> ModelResult<Option<GgufTensorType>> {
    let Ok(raw) = std::env::var(FORCE_Q2_LAYOUT_ENV) else {
        return Ok(None);
    };
    parse_forced_q2_layout(&raw)
}

/// [`forced_q2_layout`]'s parser, split out so the accepted spellings can
/// be tested without mutating the process environment (which no test can do
/// safely while other tests run in the same process).
///
/// # Errors
///
/// As [`forced_q2_layout`].
pub fn parse_forced_q2_layout(raw: &str) -> ModelResult<Option<GgufTensorType>> {
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Ok(None);
    }
    match value.as_str() {
        "d-first" | "dfirst" | "d_first" | "pq2" | "pq2_0" => {
            Ok(Some(GgufTensorType::Q2_0G128DFirst))
        }
        "qs-first" | "qsfirst" | "qs_first" | "tq2" | "tq2_0" | "legacy" => {
            Ok(Some(GgufTensorType::TQ2_0_g128))
        }
        "g64" | "q2_0_g64" | "q2_0g64" => Ok(Some(GgufTensorType::Q2_0G64)),
        other => Err(ModelError::InvalidTensor(format!(
            "{FORCE_Q2_LAYOUT_ENV}={other:?} is not a recognised layout: use one of \
             d-first|dfirst|pq2, qs-first|qsfirst|tq2|legacy, or g64"
        ))),
    }
}

/// Resolve this file's reading of ggml wire id 42 **once**, or `Ok(None)`
/// when the file has no type-42 tensor.
///
/// A gen-1 `qwen35` file (`Ternary-Bonsai-27B-PQ2_0.gguf`) stores its
/// matrices under id 42, which is three different on-disk layouts; the
/// current files use 142/143 and skip this entirely. The replay walks every
/// tensor offset and sniffs real block bytes, so the answer is computed once
/// per file and threaded through the per-layer loop.
///
/// Mirrors `model::weight_loaders::resolve_id42_once`, which is `pub(super)`
/// to `crate::model` and so unreachable from here (recorded as this
/// package's deviation: the requested change is to widen those three helpers
/// to `pub(crate)`).
///
/// # Errors
///
/// [`ModelError::Core`] when the offset table is internally inconsistent or
/// the sniff actively contradicts a declared legacy tag. A merely
/// *inconclusive* sniff (an all-zero or too-small sample, i.e. a synthetic
/// fixture) falls back to the historical qs-first reading, exactly as the
/// dense loader does.
pub fn resolve_id42(gguf: &GgufFile<'_>) -> ModelResult<Option<GgufTensorType>> {
    // Design SS1.3's operator override, shared with the dense loader so the
    // two paths cannot disagree about what a forced layout means.
    if let Some(forced) = forced_q2_layout()? {
        tracing::debug!(
            layout = ?forced,
            env = FORCE_Q2_LAYOUT_ENV,
            "qwen35: ggml wire id 42 layout forced by the environment"
        );
        return Ok(Some(forced));
    }
    let infos: Vec<TensorInfo> = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let Some(sample_info) = infos
        .iter()
        .find(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
    else {
        return Ok(None);
    };
    let sample_name = sample_info.name.clone();
    let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
    let extents = compute_extents(&infos, data_len).map_err(ModelError::Core)?;
    let alignment = gguf
        .metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;
    let sample = gguf.tensor_data(&sample_name).map_err(ModelError::Core)?;
    let sample = &sample[..sample
        .len()
        .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];
    match resolve_type_42_with_sample(
        &infos,
        alignment,
        gguf.metadata.get("general.quantization_version"),
        Some(&extents),
        sample,
    ) {
        Ok(resolved) => Ok(Some(resolved.tensor_type)),
        Err(BonsaiError::AmbiguousQuantType { hint, .. }) => {
            tracing::warn!(
                tensor = %sample_name,
                reason = %hint,
                env = FORCE_Q2_LAYOUT_ENV,
                "qwen35: ggml wire id 42 could not be conclusively resolved; falling back to \
                 the legacy qs-first TQ2_0_g128 reading -- set OXI_FORCE_Q2_LAYOUT=d-first if \
                 this file is a PrismML d-first checkpoint"
            );
            Ok(None)
        }
        Err(other) => Err(ModelError::Core(other)),
    }
}

/// Apply [`resolve_id42`]'s answer to one tensor's parse-time type.
#[inline]
#[must_use]
pub fn apply_resolved_type(
    raw: GgufTensorType,
    resolved_42: Option<GgufTensorType>,
) -> GgufTensorType {
    if raw.wire_id() != AMBIGUOUS_TYPE_ID {
        return raw;
    }
    resolved_42.unwrap_or(raw)
}

/// [`GgufFile::tensor_data`] sized from a **resolved** type.
///
/// `TensorInfo::data_size` sizes every wire-id-42 tensor as the group-128
/// reading, which under-counts a genuine group-64 tensor by 2 bytes per 128
/// elements.
fn tensor_data_resolved<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    resolved_type: GgufTensorType,
) -> ModelResult<&'a [u8]> {
    let info = gguf.tensors.require(name).map_err(ModelError::Core)?;
    if resolved_type == info.tensor_type {
        return gguf.tensor_data(name).map_err(ModelError::Core);
    }
    let size = row_size_bytes(resolved_type, &info.shape);
    let start = (gguf.data_offset as u64)
        .checked_add(info.offset)
        .ok_or(ModelError::Core(BonsaiError::UnexpectedEof {
            offset: u64::MAX,
        }))?;
    let end = start
        .checked_add(size)
        .ok_or(ModelError::Core(BonsaiError::UnexpectedEof {
            offset: u64::MAX,
        }))?;
    if end > gguf.data.len() as u64 {
        return Err(ModelError::Core(BonsaiError::UnexpectedEof { offset: end }));
    }
    gguf.data
        .get(start as usize..end as usize)
        .ok_or(ModelError::Core(BonsaiError::UnexpectedEof { offset: end }))
}

// ─────────────────────────────────────────────────────────────────────────
//  Shape-checked tensor access
// ─────────────────────────────────────────────────────────────────────────

/// Require `name` and check its GGUF shape against `expected` (`[in, out]`
/// order, `ne[0]` first).
///
/// # Errors
///
/// [`ModelError::MissingTensor`] naming the tensor;
/// [`ModelError::ShapeMismatch`] carrying both shapes.
pub fn expect_shape<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    expected: &[usize],
) -> ModelResult<&'a TensorInfo> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_string(),
        })?;
    let actual: Vec<usize> = info.shape.iter().map(|d| *d as usize).collect();
    if actual != expected {
        return Err(ModelError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual,
        });
    }
    Ok(info)
}

/// Load a small dense tensor (`F32`/`F16`/`BF16`) of exactly `expected_len`
/// elements: the norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`.
///
/// # Errors
///
/// [`ModelError::MissingTensor`], [`ModelError::ShapeMismatch`] for a wrong
/// element count or byte length, or [`ModelError::Core`] naming the type
/// when the tensor is quantized (these tensors never are on a real file).
pub fn load_dense_f32(
    gguf: &GgufFile<'_>,
    name: &str,
    expected_len: usize,
) -> ModelResult<Vec<f32>> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_string(),
        })?;
    let count = usize::try_from(info.element_count()).map_err(|_| ModelError::ShapeInvariant {
        tensor: name.to_string(),
        expected: "element count representable as usize".to_string(),
        actual: info.element_count().to_string(),
    })?;
    if count != expected_len {
        return Err(ModelError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![expected_len],
            actual: vec![count],
        });
    }
    let data = gguf.tensor_data(name).map_err(ModelError::Core)?;
    let (elem_bytes, decode): (usize, fn(&[u8]) -> f32) = match info.tensor_type {
        GgufTensorType::F32 => (4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        GgufTensorType::F16 => (2, |c| {
            half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()
        }),
        GgufTensorType::BF16 => (2, |c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
        other => {
            return Err(ModelError::Core(BonsaiError::non_executable_quant_type(
                name, other,
            )))
        }
    };
    let expected_bytes = expected_len.saturating_mul(elem_bytes);
    if data.len() != expected_bytes {
        return Err(ModelError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![expected_bytes],
            actual: vec![data.len()],
        });
    }
    Ok(data.chunks_exact(elem_bytes).map(decode).collect())
}

/// Load a norm weight and wrap it in an [`RmsNorm`] with the config's eps.
///
/// # Errors
///
/// As [`load_dense_f32`].
pub fn load_norm(gguf: &GgufFile<'_>, name: &str, width: usize, eps: f32) -> ModelResult<RmsNorm> {
    expect_shape(gguf, name, &[width])?;
    Ok(RmsNorm::new(load_dense_f32(gguf, name, width)?, eps))
}

/// Bind one quantized matrix as a [`LinearLayer`], checking its GGUF shape
/// against `[in_features, out_features]` first.
///
/// # Errors
///
/// [`ModelError::MissingTensor`] / [`ModelError::ShapeMismatch`] from
/// [`expect_shape`]; [`ModelError::Core`] with
/// `BonsaiError::NonExecutableQuantType` for a type this build has no
/// hybrid kernel for — never a silent fall-through to another decoder
/// (M-10).
pub fn bind_linear<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    in_features: usize,
    out_features: usize,
    resolved_42: Option<GgufTensorType>,
    kernel: &Arc<KernelDispatcher>,
) -> ModelResult<LinearLayer<'a>> {
    let info = expect_shape(gguf, name, &[in_features, out_features])?;
    let resolved = apply_resolved_type(info.tensor_type, resolved_42);
    let data = tensor_data_resolved(gguf, name, resolved)?;
    let layer = match resolved {
        GgufTensorType::PQ2_0 | GgufTensorType::Q2_0G128DFirst => {
            let blocks = BlockPQ2_0::slice_from_bytes(data).map_err(ModelError::Core)?;
            LinearLayer::PQ2_0(LinearPQ2_0::new(
                blocks,
                out_features,
                in_features,
                kernel.clone(),
            )?)
        }
        GgufTensorType::PTQ1_0 => {
            let blocks = BlockPTQ1_0::slice_from_bytes(data).map_err(ModelError::Core)?;
            LinearLayer::PTQ1_0(LinearPTQ1_0::new(
                blocks,
                out_features,
                in_features,
                kernel.clone(),
            )?)
        }
        GgufTensorType::Q2_0G64 => {
            let blocks = BlockQ2_0G64::slice_from_bytes(data).map_err(ModelError::Core)?;
            LinearLayer::Q2_0G64(LinearQ2_0G64::new(
                blocks,
                out_features,
                in_features,
                kernel.clone(),
            )?)
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)?;
            LinearLayer::Ternary(LinearTernary::new(
                blocks,
                out_features,
                in_features,
                kernel.clone(),
            )?)
        }
        GgufTensorType::Q1_0_g128 => {
            let blocks = BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)?;
            LinearLayer::OneBit(Linear1Bit::new(
                blocks,
                out_features,
                in_features,
                kernel.clone(),
            )?)
        }
        // `dequant_any` CAN decode these, and `HybridEmbedding::Dense`
        // already consumes them for `token_embd.weight`, but `LinearLayer`
        // has no dense variant in this build, so there is no kernel to run
        // the projection through. Deliberately NOT
        // `non_executable_quant_type`: that error's generated "executable
        // types are ..." list names F32 itself, contradicting the refusal
        // it is attached to (gatekeeper REQUIRED #5 / OPTIONAL #O8). The
        // fix that makes these executable is a `LinearLayer::Dense`
        // variant in `layers/linear.rs`; until it exists this says so.
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 => {
            return Err(ModelError::InvalidTensor(format!(
                "{name}: {} (id {}) is stored unquantized, and this build has no dense \
                 LinearLayer variant to run a qwen35 projection through - add \
                 `LinearLayer::Dense` (mirroring `weight_loaders.rs`'s `OutputWeight::Fp32`) \
                 or quantize the file",
                resolved,
                resolved.wire_id(),
            )))
        }
        other => {
            return Err(ModelError::Core(BonsaiError::non_executable_quant_type(
                name, other,
            )))
        }
    };
    Ok(layer)
}

/// Bind `token_embd.weight` for row-wise lookup.
///
/// # Errors
///
/// As [`bind_linear`], plus [`ModelError::Core`] from a block cast.
pub fn bind_embedding<'a>(
    gguf: &'a GgufFile<'a>,
    hidden: usize,
    vocab: usize,
    resolved_42: Option<GgufTensorType>,
) -> ModelResult<HybridEmbedding<'a>> {
    let name = names::TOKEN_EMBD;
    let info = expect_shape(gguf, name, &[hidden, vocab])?;
    let resolved = apply_resolved_type(info.tensor_type, resolved_42);
    let data = tensor_data_resolved(gguf, name, resolved)?;
    embedding_view(name, data, resolved, hidden, vocab)
}

/// A row-wise decoder over a `[row_len, rows]` GGUF matrix.
///
/// Shared by `token_embd.weight` (where the rows are vocabulary entries)
/// and by a gen-1 27B's quantized `ssm_alpha`/`ssm_beta` (where they are
/// v-heads), so one type table serves both and a new quantization format
/// cannot be wired into only one of them.
///
/// # Errors
///
/// [`ModelError::Core`] from a block cast or a dequantizer;
/// [`ModelError::ShapeInvariant`] for an overflowing geometry.
fn embedding_view<'a>(
    name: &str,
    data: &'a [u8],
    resolved: GgufTensorType,
    row_len: usize,
    rows: usize,
) -> ModelResult<HybridEmbedding<'a>> {
    let (hidden, vocab) = (row_len, rows);
    let table = match resolved {
        GgufTensorType::PQ2_0 | GgufTensorType::Q2_0G128DFirst => {
            HybridEmbedding::Pq2_0(BlockPQ2_0::slice_from_bytes(data).map_err(ModelError::Core)?)
        }
        GgufTensorType::PTQ1_0 => {
            HybridEmbedding::Ptq1_0(BlockPTQ1_0::slice_from_bytes(data).map_err(ModelError::Core)?)
        }
        GgufTensorType::Q2_0G64 => HybridEmbedding::Q2_0G64(
            BlockQ2_0G64::slice_from_bytes(data).map_err(ModelError::Core)?,
        ),
        GgufTensorType::TQ2_0_g128 => HybridEmbedding::Ternary(
            BlockTQ2_0_g128::slice_from_bytes(data).map_err(ModelError::Core)?,
        ),
        GgufTensorType::Q1_0_g128 => HybridEmbedding::OneBit(
            BlockQ1_0G128::slice_from_bytes(data).map_err(ModelError::Core)?,
        ),
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 => {
            let n = hidden
                .checked_mul(vocab)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: name.to_string(),
                    expected: "row_len * rows representable as usize".to_string(),
                    actual: format!("{hidden} x {vocab}"),
                })?;
            HybridEmbedding::Dense(widen_dense(name, data, resolved, n)?)
        }
        other => {
            return Err(ModelError::Core(BonsaiError::non_executable_quant_type(
                name, other,
            )))
        }
    };
    Ok(table)
}

/// Widen an unquantized `F32`/`F16`/`BF16` tensor's raw bytes to `f32`.
///
/// # Errors
///
/// [`ModelError::ShapeMismatch`] when `data` is shorter than `n` elements,
/// or [`ModelError::InvalidTensor`] for a type that is not one of the three.
fn widen_dense(
    name: &str,
    data: &[u8],
    resolved: GgufTensorType,
    n: usize,
) -> ModelResult<Vec<f32>> {
    let width = match resolved {
        GgufTensorType::F32 => 4usize,
        GgufTensorType::F16 | GgufTensorType::BF16 => 2,
        other => {
            return Err(ModelError::InvalidTensor(format!(
                "{name}: {other} is not an unquantized type"
            )))
        }
    };
    let needed = n
        .checked_mul(width)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: name.to_string(),
            expected: "n * element width representable as usize".to_string(),
            actual: format!("{n} x {width}"),
        })?;
    let bytes = data
        .get(..needed)
        .ok_or_else(|| ModelError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![needed],
            actual: vec![data.len()],
        })?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(width) {
        let value = match (resolved, chunk) {
            (GgufTensorType::F32, [a, b, c, d]) => f32::from_le_bytes([*a, *b, *c, *d]),
            (GgufTensorType::F16, [a, b]) => {
                half::f16::from_bits(u16::from_le_bytes([*a, *b])).to_f32()
            }
            (_, [a, b]) => bf16_to_f32(u16::from_le_bytes([*a, *b])),
            // Unreachable: `chunks_exact(width)` yields exactly `width`
            // bytes and `width` is 4 for F32, 2 otherwise.
            _ => 0.0,
        };
        out.push(value);
    }
    Ok(out)
}

/// Bind the LM head, refusing a tied head (design §3.5).
///
/// # Errors
///
/// [`ModelError::TiedLmHeadUnsupported`] when `output.weight` is absent;
/// otherwise as [`bind_linear`].
pub fn bind_lm_head<'a>(
    gguf: &'a GgufFile<'a>,
    hidden: usize,
    vocab: usize,
    resolved_42: Option<GgufTensorType>,
    kernel: &Arc<KernelDispatcher>,
) -> ModelResult<LinearLayer<'a>> {
    if gguf.tensors.get(names::OUTPUT).is_none() {
        return Err(ModelError::TiedLmHeadUnsupported {
            lm_head: names::OUTPUT.to_string(),
            embedding: names::TOKEN_EMBD.to_string(),
        });
    }
    bind_linear(gguf, names::OUTPUT, hidden, vocab, resolved_42, kernel)
}

/// Bind one linear-attention layer's Gated-DeltaNet gate pair
/// (`ssm_a` + `ssm_dt.bias`), by name, re-indexed into **grouped** v-head
/// order.
///
/// `ssm_a` and `ssm_dt.bias` are GGUF rows in **tiled** v-head order (design
/// §3.3's table), but [`GdnGateWeights::gates`] hands them to kernel entry
/// points that index by the **grouped** v-head index
/// ([`crate::hybrid::vhead_map::GDN_HEAD_ORDER`]). `vhead_map` is the same
/// map this model's [`crate::hybrid::vhead_map::VHeadMap::gather_grouped`]
/// re-indexes `v`/`z` through, so passing the wrong map here would silently
/// swap in a different layer's — or a differently-shaped model's — v-head
/// permutation; callers thread through the one built once per model in
/// [`crate::hybrid::HybridModel::from_gguf_with`].
///
/// # Errors
///
/// As [`load_dense_f32`]; [`ModelError::InvalidTensor`] from
/// [`validate_a_neg`], naming the **raw GGUF row** of the offending element
/// (checked before the tiled -> grouped gather, deliberately — see below);
/// [`ModelError::ShapeMismatch`] from [`GdnGateWeights::new`] or from
/// [`crate::hybrid::vhead_map::VHeadMap::gather_grouped_scalar`] if
/// `vhead_map`'s v-head count disagrees with `n_v_heads` (never true for a
/// `vhead_map` built from the same config, which every real caller does).
pub fn bind_gdn_gates(
    gguf: &GgufFile<'_>,
    layer: usize,
    n_v_heads: usize,
    vhead_map: &VHeadMap,
) -> ModelResult<GdnGateWeights> {
    let a_name = block_tensor(layer, names::SSM_A);
    let dt_name = block_tensor(layer, names::SSM_DT_BIAS);
    expect_shape(gguf, &a_name, &[n_v_heads])?;
    expect_shape(gguf, &dt_name, &[n_v_heads])?;
    // Named bindings, never positional: `ssm_a` IS `a_neg`, `ssm_dt.bias`
    // IS `dt_bias`. Both come off disk in TILED v-head order.
    let a_neg_tiled = load_dense_f32(gguf, &a_name, n_v_heads)?;
    let dt_bias_tiled = load_dense_f32(gguf, &dt_name, n_v_heads)?;
    // Validate the RAW (tiled) vector, before the gather below, so a bad
    // checkpoint's error names the GGUF row a person can go look up in the
    // file (`ssm_a[17]`) rather than a grouped index that only means
    // something after mentally reversing `vhead_map` (B2-05's load-time
    // check must name "the offending index and value immediately" — an
    // index in the wrong space is not that). `GdnGateWeights::new` still
    // runs the same check on the grouped vector as its own invariant, but
    // never fires first once this passes, since a permutation cannot turn a
    // valid vector into an invalid one or vice versa.
    validate_a_neg(&a_neg_tiled)
        .map_err(|e| ModelError::InvalidTensor(format!("{a_name}: {e}")))?;
    // Re-index tiled -> grouped, exactly like `v`/`z`
    // (`VHeadMap::gather_grouped`) and `ssm_alpha`/`ssm_beta`'s per-token
    // output (`LinearScratch::{alpha_grouped,beta_grouped}`). Skipping this
    // would leave `GdnGateWeights` in tiled order while every other
    // v-indexed quantity the block forward builds is grouped, which is
    // exactly the silent 45-of-48-wrong-heads failure this function exists
    // to close.
    let mut a_neg_grouped = vec![0.0f32; n_v_heads];
    let mut dt_bias_grouped = vec![0.0f32; n_v_heads];
    vhead_map
        .gather_grouped_scalar(&a_neg_tiled, &mut a_neg_grouped)
        .map_err(|e| annotate(e, &a_name))?;
    vhead_map
        .gather_grouped_scalar(&dt_bias_tiled, &mut dt_bias_grouped)
        .map_err(|e| annotate(e, &dt_name))?;
    GdnGateWeights::new(a_neg_grouped, dt_bias_grouped, n_v_heads).map_err(|e| annotate(e, &a_name))
}

/// Prefix a tensor name onto an error whose own message does not carry one.
fn annotate(error: ModelError, name: &str) -> ModelError {
    match error {
        ModelError::InvalidTensor(message) if !message.contains(name) => {
            ModelError::InvalidTensor(format!("{name}: {message}"))
        }
        other => other,
    }
}

/// Bind `ssm_conv1d.weight`, `[conv_kernel, conv_dim]` in GGUF order.
///
/// The GGUF `ne = [4, 10240]` is channel-major with 4 taps per channel
/// (`w[c * 4 + t]`), which is exactly the layout
/// `oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode` expects — so the
/// bytes are used as they are stored and never transposed.
///
/// # Errors
///
/// As [`load_dense_f32`].
pub fn bind_conv1d(
    gguf: &GgufFile<'_>,
    layer: usize,
    config: &HybridConfig,
) -> ModelResult<Arc<[f32]>> {
    let name = block_tensor(layer, names::SSM_CONV1D);
    let conv_dim = config.conv_dim();
    expect_shape(gguf, &name, &[config.ssm_conv_kernel, conv_dim])?;
    let values = load_dense_f32(gguf, &name, config.ssm_conv_kernel * conv_dim)?;
    Ok(values.into())
}

/// Bind one gate projection (`ssm_alpha.weight` / `ssm_beta.weight`).
///
/// BF16 (every Bonsai 2 file) is a zero-copy view; anything else -- a gen-1
/// 27B's quantized gate, or a dequantized/converted file -- is widened to
/// `f32` once, here.
///
/// # Errors
///
/// [`ModelError::MissingTensor`] / [`ModelError::ShapeMismatch`];
/// [`ModelError::Core`] from a dequantizer for a genuinely non-executable
/// type.
pub fn bind_gate_projection<'a>(
    gguf: &'a GgufFile<'a>,
    layer: usize,
    suffix: &str,
    in_features: usize,
    out_features: usize,
    resolved_42: Option<GgufTensorType>,
) -> ModelResult<Bf16Matrix<'a>> {
    let name = block_tensor(layer, suffix);
    let info = expect_shape(gguf, &name, &[in_features, out_features])?;
    if info.tensor_type == GgufTensorType::BF16 {
        let data = gguf.tensor_data(&name).map_err(ModelError::Core)?;
        return Bf16Matrix::new(&name, data, out_features, in_features);
    }
    // Gen-1 27B files (`Bonsai-27B-Q1_0`, `Ternary-Bonsai-27B-{PQ2_0,Q2_0}`)
    // store these two tensors quantized (ids 41/142/42). Dequantize the
    // whole 48 x hidden matrix once at load rather than refusing a variant
    // `ModelVariant::Bonsai27B` advertises as supported (gatekeeper
    // REQUIRED #16); the id-42 ambiguity is resolved the same way every
    // other tensor's is.
    let resolved = apply_resolved_type(info.tensor_type, resolved_42);
    let n = out_features
        .checked_mul(in_features)
        .ok_or_else(|| ModelError::ShapeInvariant {
            tensor: name.clone(),
            expected: "out_features * in_features representable as usize".to_string(),
            actual: format!("{out_features} x {in_features}"),
        })?;
    let data = tensor_data_resolved(gguf, &name, resolved)?;
    let table = embedding_view(&name, data, resolved, in_features, out_features)?;
    let mut values = vec![0.0f32; n];
    for row in 0..out_features {
        let lo = row * in_features;
        let slot =
            values
                .get_mut(lo..lo + in_features)
                .ok_or_else(|| ModelError::ShapeMismatch {
                    name: name.clone(),
                    expected: vec![n],
                    actual: vec![lo + in_features],
                })?;
        let index = u32::try_from(row).map_err(|_| ModelError::ShapeInvariant {
            tensor: name.clone(),
            expected: "row index representable as u32".to_string(),
            actual: row.to_string(),
        })?;
        table.row(index, in_features, slot)?;
    }
    Bf16Matrix::from_dense(&name, values, out_features, in_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_q2_layout_parses_every_documented_spelling_bonsai2() {
        for (raw, expected) in [
            ("d-first", GgufTensorType::Q2_0G128DFirst),
            ("DFirst", GgufTensorType::Q2_0G128DFirst),
            ("pq2", GgufTensorType::Q2_0G128DFirst),
            ("qs-first", GgufTensorType::TQ2_0_g128),
            ("legacy", GgufTensorType::TQ2_0_g128),
            ("TQ2", GgufTensorType::TQ2_0_g128),
            ("g64", GgufTensorType::Q2_0G64),
            ("  q2_0_g64  ", GgufTensorType::Q2_0G64),
        ] {
            assert_eq!(
                parse_forced_q2_layout(raw).expect("valid spelling"),
                Some(expected),
                "spelling {raw:?}"
            );
        }
        assert_eq!(parse_forced_q2_layout("").expect("empty is unset"), None);
        assert_eq!(parse_forced_q2_layout("   ").expect("blank is unset"), None);
    }

    #[test]
    fn forced_q2_layout_refuses_a_typo_rather_than_falling_back_bonsai2() {
        // A typo must be loud: silently degrading to the automatic sniff is
        // the exact failure this hatch exists to prevent (design SS1.3).
        let err = parse_forced_q2_layout("d_frist").expect_err("typo rejected");
        let message = err.to_string();
        assert!(message.contains("OXI_FORCE_Q2_LAYOUT"), "{message}");
        assert!(message.contains("d-first"), "{message}");
    }

    #[test]
    fn bf16_matrix_dense_and_bf16_storage_agree_bonsai2() {
        // Two rows of three columns, chosen so every value is exactly
        // representable in bf16 (so the two storages must agree bitwise).
        let values = vec![1.0f32, -2.0, 0.5, 4.0, -0.25, 8.0];
        let mut bytes = Vec::new();
        for v in &values {
            let bits = v.to_bits();
            let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
            bytes.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
        let view = Bf16Matrix::new("test", &bytes, 2, 3).expect("bf16 view");
        let dense = Bf16Matrix::from_dense("test", values.clone(), 2, 3).expect("dense");
        assert!(!view.is_dequantized());
        assert!(dense.is_dequantized());
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(view.at(row, col).to_bits(), dense.at(row, col).to_bits());
            }
        }
        let x = [1.0f32, 2.0, 3.0];
        let mut a = vec![0.0f32; 2];
        let mut b = vec![0.0f32; 2];
        view.forward_vec(&x, &mut a).expect("bf16 gemv");
        dense.forward_vec(&x, &mut b).expect("dense gemv");
        assert_eq!(a, b);
        assert_eq!(a, vec![1.0 - 4.0 + 1.5, 4.0 - 0.5 + 24.0]);
    }

    #[test]
    fn bf16_matrix_from_dense_rejects_a_wrong_sized_vector_bonsai2() {
        assert!(Bf16Matrix::from_dense("test", vec![0.0; 5], 2, 3).is_err());
    }

    use crate::hybrid::tests_support::{
        synthetic_default, synthetic_gguf, FixtureOptions, FixtureShape,
    };
    use oxibonsai_kernels::gated_delta_net::{softplus, GdnDims, GdnHeadOrder, GdnPath};

    fn dispatcher() -> Arc<KernelDispatcher> {
        Arc::new(KernelDispatcher::with_tier(
            oxibonsai_kernels::KernelTier::Reference,
        ))
    }

    #[test]
    fn binds_a_quantized_matrix_with_the_gguf_in_out_order() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let kernel = dispatcher();
        let layer = bind_linear(
            &gguf,
            &block_tensor(0, names::ATTN_QKV),
            shape.hidden,
            shape.conv_dim(),
            None,
            &kernel,
        )
        .expect("attn_qkv binds");
        assert_eq!(layer.in_features(), shape.hidden);
        assert_eq!(layer.out_features(), shape.conv_dim());
        assert!(matches!(layer, LinearLayer::PQ2_0(_)));
    }

    #[test]
    fn a_mis_shaped_tensor_names_both_shapes() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let kernel = dispatcher();
        let err = bind_linear(
            &gguf,
            &block_tensor(0, names::ATTN_QKV),
            shape.hidden,
            shape.conv_dim() + 64,
            None,
            &kernel,
        )
        .expect_err("a wrong out_features must be refused");
        assert_eq!(err.error_code(), "SHAPE_MISMATCH");
        let message = err.to_string();
        assert!(message.contains(&shape.conv_dim().to_string()), "{message}");
        assert!(
            message.contains(&(shape.conv_dim() + 64).to_string()),
            "{message}"
        );
    }

    #[test]
    fn a_missing_tensor_is_an_error_not_a_warning() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        // Layer 3 is a full-attention layer: it has no `attn_qkv`.
        let err = bind_linear(
            &gguf,
            &block_tensor(3, names::ATTN_QKV),
            256,
            640,
            None,
            &dispatcher(),
        )
        .expect_err("a full layer has no attn_qkv");
        assert_eq!(err.error_code(), "MISSING_TENSOR");
        assert!(err.to_string().contains("blk.3.attn_qkv.weight"));
    }

    #[test]
    fn bf16_gate_projection_is_zero_copy_and_multiplies() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let alpha = bind_gate_projection(
            &gguf,
            0,
            names::SSM_ALPHA,
            shape.hidden,
            shape.n_v_heads,
            None,
        )
        .expect("ssm_alpha binds");
        assert_eq!(alpha.out_features(), shape.n_v_heads);
        assert_eq!(alpha.in_features(), shape.hidden);

        // A one-hot input selects one column, so the product must equal the
        // decoded element itself.
        let mut x = vec![0.0f32; shape.hidden];
        x[7] = 1.0;
        let mut y = vec![0.0f32; shape.n_v_heads];
        alpha.forward_vec(&x, &mut y).expect("forward");
        for (row, value) in y.iter().enumerate() {
            assert_eq!(*value, alpha.at(row, 7), "row {row}");
        }
    }

    /// A [`VHeadMap`] matching a fixture's own `(n_k_heads, n_v_heads)`.
    fn fixture_vhead_map(shape: FixtureShape) -> VHeadMap {
        VHeadMap::new(shape.n_k_heads, shape.n_v_heads).expect("valid fixture geometry")
    }

    #[test]
    fn ssm_a_is_validated_at_load() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let vhead_map = fixture_vhead_map(shape);
        let gates = bind_gdn_gates(&gguf, 0, shape.n_v_heads, &vhead_map).expect("gates bind");
        assert!(gates.a_neg().iter().all(|a| *a <= 0.0));
        assert_eq!(gates.a_neg().len(), shape.n_v_heads);
        assert_eq!(gates.dt_bias().len(), shape.n_v_heads);

        // A positive `ssm_a` is refused with the index and value named.
        let err = GdnGateWeights::new(vec![-1.0, 0.5], vec![-1.0, -1.0], 2)
            .expect_err("positive ssm_a must be refused at load");
        assert_eq!(err.error_code(), "INVALID_TENSOR");
        assert!(err.to_string().contains("ssm_a[1]"), "{err}");
    }

    /// The gatekeeper's blocking finding: `ssm_a` and `ssm_dt.bias` are GGUF
    /// rows in **tiled** v-head order (design §3.3's table) and
    /// [`bind_gdn_gates`] must re-index them into **grouped** order before
    /// [`GdnGateWeights`] hands them to the kernel — exactly the property
    /// `vhead_map::tests::gather_scalar_follows_the_same_map` pins for
    /// [`VHeadMap::gather_grouped_scalar`] itself, checked here one layer up
    /// at the loader boundary. Uses the real 27B v-head/k-head ratio (48
    /// over 16) rather than the fixture default so every one of the 44
    /// non-trivial `tiled(m) != m` heads is exercised, not just a few.
    #[test]
    fn bound_gdn_gates_are_reindexed_from_tiled_to_grouped_order() {
        let shape = FixtureShape {
            n_k_heads: 16,
            n_v_heads: 48,
            ..FixtureShape::default()
        };
        let bytes = synthetic_gguf(shape, FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let vhead_map = fixture_vhead_map(shape);
        assert!(
            !vhead_map.is_identity(),
            "48-over-16 must be a real permutation"
        );

        // The raw GGUF row order (tiled) for layer 0, read directly —
        // bypassing the re-index this test exists to prove.
        let raw_a = load_dense_f32(&gguf, &block_tensor(0, names::SSM_A), shape.n_v_heads)
            .expect("raw ssm_a reads");
        let raw_dt = load_dense_f32(&gguf, &block_tensor(0, names::SSM_DT_BIAS), shape.n_v_heads)
            .expect("raw ssm_dt.bias reads");

        let gates = bind_gdn_gates(&gguf, 0, shape.n_v_heads, &vhead_map).expect("gates bind");

        for grouped in 0..shape.n_v_heads {
            let tiled = vhead_map.tiled(grouped);
            assert_eq!(
                gates.a_neg()[grouped],
                raw_a[tiled],
                "grouped v-head {grouped} (raw GGUF row {tiled}): a_neg"
            );
            assert_eq!(
                gates.dt_bias()[grouped],
                raw_dt[tiled],
                "grouped v-head {grouped} (raw GGUF row {tiled}): dt_bias"
            );
        }
        // The old (broken) binding would have been `gates.a_neg() ==
        // raw_a` verbatim; assert the two genuinely differ so this test
        // cannot pass by coincidence.
        assert_ne!(
            gates.a_neg().to_vec(),
            raw_a,
            "a real 48-over-16 permutation must reorder the array"
        );
    }

    /// The tiled -> grouped gather must not swallow the load-time
    /// `validate_a_neg` diagnostic into the wrong index space: a positive
    /// `ssm_a` must be named by its **raw GGUF row**, the number a person
    /// debugging a bad checkpoint can actually go look up, not by a grouped
    /// index that only means something after mentally reversing
    /// `vhead_map`. Poisons a row whose grouped index differs from itself,
    /// so a regression that validates the gathered (grouped) vector instead
    /// of the raw one names a different, wrong row and this test catches it.
    #[test]
    fn bind_gdn_gates_names_the_raw_gguf_row_not_a_grouped_index() {
        let shape = FixtureShape {
            n_k_heads: 16,
            n_v_heads: 48,
            ..FixtureShape::default()
        };
        let mut bytes = synthetic_gguf(shape, FixtureOptions::default());
        let vhead_map = fixture_vhead_map(shape);

        let bad_row = 17usize;
        assert_ne!(
            vhead_map.grouped(bad_row),
            bad_row,
            "fixture must pick a row whose grouped index is not a fixed point"
        );
        {
            // Overwrite `blk.0.ssm_a`'s raw row `bad_row` with a positive
            // value, in place, the same "flip known bytes" technique
            // `model::tests::refuses_a_dense_qwen3_file` uses for the
            // architecture string.
            let gguf = GgufFile::parse(&bytes).expect("fixture parses");
            let info = gguf
                .tensors
                .get(&block_tensor(0, names::SSM_A))
                .expect("ssm_a exists");
            let byte_pos = (gguf.data_offset as u64 + info.offset) as usize + bad_row * 4;
            bytes[byte_pos..byte_pos + 4].copy_from_slice(&1.5f32.to_le_bytes());
        }

        let gguf = GgufFile::parse(&bytes).expect("poisoned fixture still parses");
        let err = bind_gdn_gates(&gguf, 0, shape.n_v_heads, &vhead_map)
            .expect_err("a positive ssm_a must be refused at load");
        assert_eq!(err.error_code(), "INVALID_TENSOR");
        let message = err.to_string();
        assert!(
            message.contains(&format!("ssm_a[{bad_row}]")),
            "error must name the raw GGUF row {bad_row}: {message}"
        );
        let grouped_row = vhead_map.grouped(bad_row);
        assert!(
            !message.contains(&format!("ssm_a[{grouped_row}]")),
            "error must not name the grouped index {grouped_row} instead: {message}"
        );
    }

    /// The gatekeeper's second requested pin: [`LinearAttnBlock::gates`]
    /// must stay order-consistent with `LinearScratch::alpha_grouped` /
    /// `beta_grouped` — the two are built by different code paths
    /// (load-time binding vs. per-token gather) but must agree on which
    /// physical v-head lives at grouped index `m`, or `GdnGates::log_decay_at`
    /// silently pairs the wrong `a_neg`/`dt_bias` with the wrong
    /// `alpha`/`beta`.
    #[test]
    fn block_gates_stay_order_consistent_with_scratch_alpha_grouped() {
        use crate::hybrid::{HybridBlock, HybridModel, LinearScratch};

        let shape = FixtureShape {
            n_k_heads: 4,
            n_v_heads: 12,
            ..FixtureShape::default()
        };
        let bytes = synthetic_gguf(shape, FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        let vhead_map = model.vhead_map();
        let linear = model
            .block(0)
            .and_then(HybridBlock::as_linear)
            .expect("layer 0 is linear");

        // One token's raw (tiled) alpha/beta projection output, gathered
        // into `LinearScratch`'s grouped buffers exactly as a B2-11 forward
        // will.
        let x = vec![0.37f32; shape.hidden];
        let mut alpha_tiled = vec![0.0f32; shape.n_v_heads];
        let mut beta_tiled = vec![0.0f32; shape.n_v_heads];
        linear
            .ssm_alpha()
            .forward_vec(&x, &mut alpha_tiled)
            .expect("alpha");
        linear
            .ssm_beta()
            .forward_vec(&x, &mut beta_tiled)
            .expect("beta");
        let mut scratch = LinearScratch::new(model.config());
        vhead_map
            .gather_grouped_scalar(&alpha_tiled, &mut scratch.alpha_grouped)
            .expect("gather alpha");
        vhead_map
            .gather_grouped_scalar(&beta_tiled, &mut scratch.beta_grouped)
            .expect("gather beta");

        let raw_a = load_dense_f32(&gguf, &block_tensor(0, names::SSM_A), shape.n_v_heads)
            .expect("raw ssm_a");
        let raw_dt = load_dense_f32(&gguf, &block_tensor(0, names::SSM_DT_BIAS), shape.n_v_heads)
            .expect("raw ssm_dt.bias");

        let gates = linear
            .gates()
            .gates(&scratch.alpha_grouped, &scratch.beta_grouped);
        for m in 0..shape.n_v_heads {
            let tiled = vhead_map.tiled(m);
            let expected = raw_a[tiled] * softplus(alpha_tiled[tiled] + raw_dt[tiled]);
            let got = gates.log_decay_at(m, m).expect("scalar decay");
            assert!(
                (got - expected).abs() < 1e-4,
                "grouped v-head {m} (raw GGUF row {tiled}): got {got}, want {expected}"
            );
        }
        assert!(!vhead_map.is_identity());
    }

    /// The gatekeeper's REQUIRED #8: a transposed `(a_neg, dt_bias)` pair
    /// must be caught by the arithmetic, not by luck.
    ///
    /// Both vectors are negative here (so `validate_a_neg` accepts either
    /// assignment) and their values differ per head (so a swap is
    /// numerically visible). The reference below is design §2.3's decay,
    /// `g = a_neg[h] * softplus(alpha_raw[h] + dt_bias[h])`.
    #[test]
    fn gates_are_bound_by_name_not_position() {
        let n_v = 4usize;
        let a_neg = vec![-0.5f32, -1.5, -0.25, -3.0];
        let dt_bias = vec![-2.0f32, -0.25, -1.0, -0.5];
        let alpha_raw = vec![0.75f32, -0.5, 0.25, 1.5];
        let beta_raw = vec![0.1f32, 0.2, -0.3, 0.4];

        let bound = GdnGateWeights::new(a_neg.clone(), dt_bias.clone(), n_v).expect("gates");
        let gates = bound.gates(&alpha_raw, &beta_raw);
        for h in 0..n_v {
            let expected = a_neg[h] * softplus(alpha_raw[h] + dt_bias[h]);
            let got = gates.log_decay_at(h, h).expect("scalar decay");
            assert!(
                (got - expected).abs() < 1e-6,
                "head {h}: got {got}, want {expected}"
            );
        }

        // The swapped binding is accepted by `validate_a_neg` (both vectors
        // are negative) and produces different decays — which is exactly why
        // the named binding above has to exist.
        let swapped = GdnGateWeights::new(dt_bias.clone(), a_neg.clone(), n_v)
            .expect("both vectors are negative, so the swap is NOT caught by validate_a_neg");
        let swapped_gates = swapped.gates(&alpha_raw, &beta_raw);
        let differing = (0..n_v)
            .filter(|&h| {
                let correct = gates.log_decay_at(h, h).unwrap_or(0.0);
                let wrong = swapped_gates.log_decay_at(h, h).unwrap_or(0.0);
                (correct - wrong).abs() > 1e-6
            })
            .count();
        assert_eq!(
            differing, n_v,
            "every head's decay must change under a swap"
        );
    }

    /// The same swap, driven through the real kernel: the outputs diverge,
    /// so a fixture-level golden catches a transposition even when both
    /// vectors are negative.
    ///
    /// **Two** steps, not one: the state starts at zero, so the very first
    /// token's output is `q · (k ⊗ v·β)` — the decay multiplies an all-zero
    /// state and cancels out. Any golden that stops after one token cannot
    /// see a decay-gate bug at all.
    #[test]
    fn a_swapped_gate_binding_changes_the_kernel_output() {
        let dims = GdnDims::new(2, 4, 8, 8);
        let (n_v, hk, hv) = (dims.n_v_heads, dims.head_k_dim, dims.head_v_dim);
        let a_neg = vec![-0.5f32, -1.5, -0.25, -3.0];
        let dt_bias = vec![-2.0f32, -0.25, -1.0, -0.5];
        let alpha_raw = vec![0.75f32, -0.5, 0.25, 1.5];
        let beta_raw = vec![0.1f32, 0.2, -0.3, 0.4];
        let q: Vec<f32> = (0..dims.qk_len())
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..dims.qk_len())
            .map(|i| (i as f32 * 0.11).cos())
            .collect();
        let v: Vec<f32> = (0..n_v * hv).map(|i| (i as f32 * 0.03).sin()).collect();

        let run = |gate_weights: &GdnGateWeights| {
            let mut state = vec![0.0f32; n_v * hv * hk];
            let mut out = vec![0.0f32; n_v * hv];
            let gates = gate_weights.gates(&alpha_raw, &beta_raw);
            for _ in 0..2 {
                oxibonsai_kernels::gated_delta_net::gdn_step_with(
                    &mut state,
                    &q,
                    &k,
                    &v,
                    &gates,
                    &mut out,
                    &dims,
                    GdnHeadOrder::Grouped,
                    GdnPath::Fused,
                )
                .expect("gdn step");
            }
            out
        };

        let correct = run(&GdnGateWeights::new(a_neg.clone(), dt_bias.clone(), n_v).expect("ok"));
        let swapped = run(&GdnGateWeights::new(dt_bias, a_neg, n_v).expect("ok"));
        let max_delta = correct
            .iter()
            .zip(swapped.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-4,
            "a transposed gate binding must change the output (max delta {max_delta})"
        );
    }

    #[test]
    fn conv1d_is_bound_in_the_kernel_s_own_channel_major_layout() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let config = crate::hybrid::HybridModel::config_from_gguf(&gguf).expect("config");
        let conv = bind_conv1d(&gguf, 0, &config).expect("conv1d binds");
        assert_eq!(conv.len(), shape.conv_kernel * shape.conv_dim());

        // The kernel reads `w[c * KC + t]`; feeding it this slice must not
        // trip its own length contract.
        let mut state = vec![0.0f32; shape.conv_dim() * (shape.conv_kernel - 1)];
        let x = vec![1.0f32; shape.conv_dim()];
        let mut out = vec![0.0f32; shape.conv_dim()];
        oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode(&mut state, &x, &conv, &mut out)
            .expect("conv step accepts the bound layout");
        // The freshest tap is the last of each channel's 4.
        for c in 0..shape.conv_dim() {
            assert!(
                (out[c] - conv[c * shape.conv_kernel + 3]).abs() < 1e-6,
                "channel {c}"
            );
        }
    }

    #[test]
    fn embedding_rows_decode_independently() {
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let table = bind_embedding(&gguf, shape.hidden, shape.vocab, None).expect("embedding");
        let mut row0 = vec![0.0f32; shape.hidden];
        let mut row1 = vec![0.0f32; shape.hidden];
        table.row(0, shape.hidden, &mut row0).expect("row 0");
        table.row(1, shape.hidden, &mut row1).expect("row 1");
        assert_ne!(row0, row1, "distinct tokens must give distinct rows");

        // Out of range is an error, not a panic or a wrapped read.
        let err = table
            .row(shape.vocab as u32, shape.hidden, &mut row0)
            .expect_err("past the table");
        assert_eq!(err.error_code(), "POSITION_OUT_OF_RANGE");
    }

    /// `Bonsai-27B-Q1_0.gguf` is a real gen-1 `qwen35` file: 1-bit weights,
    /// no Hadamard fold. Its embedding is the one layout whose row decode is
    /// hand-written here (core has no `dequant(blocks, out)` for
    /// `Q1_0_g128`), so a wrong bit order would give plausible garbage with
    /// no error anywhere.
    ///
    /// The structural invariant that catches it: every element of a
    /// 128-element block is `±d`, so `|value|` is constant within a block
    /// and the block boundaries fall exactly where `QK1_0_G128` says.
    #[test]
    fn one_bit_embedding_rows_decode_to_plus_minus_d() {
        let shape = FixtureShape::default();
        let bytes = synthetic_gguf(
            shape,
            FixtureOptions {
                quant: oxibonsai_core::gguf::writer::TensorType::Q1_0G128,
                hadamard: false,
                ..FixtureOptions::default()
            },
        );
        let gguf = GgufFile::parse(&bytes).expect("gen-1 fixture parses");
        let table = bind_embedding(&gguf, shape.hidden, shape.vocab, None).expect("embedding");
        assert!(matches!(table, HybridEmbedding::OneBit(_)));

        let mut row = vec![0.0f32; shape.hidden];
        table.row(3, shape.hidden, &mut row).expect("row 3");
        assert_eq!(shape.hidden % QK1_0_G128, 0);
        for (block_index, block) in row.chunks_exact(QK1_0_G128).enumerate() {
            let scale = block[0].abs();
            assert!(scale > 0.0, "block {block_index} decoded to all zeros");
            for (j, value) in block.iter().enumerate() {
                assert_eq!(
                    value.abs(),
                    scale,
                    "block {block_index} element {j}: a 1-bit weight must be +d or -d"
                );
            }
            // A sign-only code must produce both signs somewhere in the
            // fixture's pseudo-random row; an all-one-sign block would mean
            // the bit test collapsed.
            assert!(
                block.iter().any(|v| *v > 0.0) && block.iter().any(|v| *v < 0.0),
                "block {block_index} has only one sign"
            );
        }

        // Distinct tokens still give distinct rows, and the whole model
        // loads over this quantization too.
        let mut other = vec![0.0f32; shape.hidden];
        table.row(4, shape.hidden, &mut other).expect("row 4");
        assert_ne!(row, other);

        let model = crate::hybrid::HybridModel::from_gguf(&gguf, 32)
            .expect("a gen-1 1-bit qwen35 file must load");
        assert!(model.hadamard().is_none());
        assert_eq!(model.quant_type(), GgufTensorType::Q1_0_g128);
        assert!(model
            .block(0)
            .and_then(crate::hybrid::HybridBlock::as_linear)
            .is_some());
    }

    #[test]
    fn a_tied_lm_head_is_refused() {
        let shape = FixtureShape::default();
        let bytes = synthetic_default();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        bind_lm_head(&gguf, shape.hidden, shape.vocab, None, &dispatcher())
            .expect("the default fixture has an explicit head");

        let tied = synthetic_gguf(
            shape,
            FixtureOptions {
                omit_lm_head: true,
                ..FixtureOptions::default()
            },
        );
        let gguf = GgufFile::parse(&tied).expect("tied fixture parses");
        let err = bind_lm_head(&gguf, shape.hidden, shape.vocab, None, &dispatcher())
            .expect_err("a tied head must be refused, never silently reuse token_embd");
        assert_eq!(err.error_code(), "TIED_LM_HEAD_UNSUPPORTED");
        assert!(err.to_string().contains("token_embd.weight"), "{err}");
    }

    #[test]
    fn an_unsupported_quant_type_is_named_never_silently_decoded() {
        let bytes = synthetic_gguf(
            FixtureShape::default(),
            FixtureOptions {
                quant: oxibonsai_core::gguf::writer::TensorType::Q4_0,
                ..FixtureOptions::default()
            },
        );
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let shape = FixtureShape::default();
        let err = bind_linear(
            &gguf,
            &block_tensor(0, names::ATTN_QKV),
            shape.hidden,
            shape.conv_dim(),
            None,
            &dispatcher(),
        )
        .expect_err("Q4_0 has no hybrid kernel");
        assert_eq!(err.error_code(), "CORE_ERROR");
        assert!(err.to_string().contains("blk.0.attn_qkv.weight"), "{err}");
    }
}
