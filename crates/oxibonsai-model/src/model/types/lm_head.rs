//! FP32 LM-head projection (M-23 / K-12).
//!
//! The FP32 output projection used to be a naive scalar triple loop inside
//! `BonsaiModel::forward`:
//!
//! ```rust
//! # let (out_features, in_features) = (2usize, 3usize);
//! # let weights = vec![1.0f32; out_features * in_features];
//! # let normed = vec![2.0f32; in_features];
//! # let mut logits = vec![0.0f32; out_features];
//! for i in 0..out_features {          // vocab
//!     let mut sum = 0.0;
//!     for j in 0..in_features {       // hidden
//!         sum += weights[i * in_features + j] * normed[j];
//!     }
//!     logits[i] = sum;
//! }
//! # assert_eq!(logits, vec![6.0, 6.0]);
//! ```
//!
//! One scalar FMA chain per row, single-threaded, over `vocab × hidden`
//! elements — 151 669 × 4096 for the shipped 8B and **248 320 × 5120 for
//! Bonsai 2 27B**, where it dominates decode. M-23 replaced it with a
//! lane-parallel dot product plus Rayon across output rows.
//!
//! # The hoist (K-12)
//!
//! That replacement lived *here*, private to this crate, which made the FP32
//! LM head the one GEMV in the tree that could not be reached through
//! [`KernelDispatcher`](oxibonsai_kernels::KernelDispatcher): the Metal and
//! CUDA tiers had no way to claim the single largest GEMV in the model, even
//! in principle. The kernel body now lives in
//! [`oxibonsai_kernels::gemv_f32`], and this module is the thin model-side
//! adapter over [`KernelDispatcher::gemv_f32`], converting the kernel crate's
//! named shape errors into [`ModelError::ShapeMismatch`].
//!
//! The move was **verbatim** — same lane count, same accumulator layout, same
//! fold order, same Rayon row threshold — because the parity gate
//! measures these logits: the dispatched kernel produces byte-identical
//! logits to the pre-hoist body on a real model. See
//! `oxibonsai_kernels::gemv_f32`'s module docs for why the lane structure is
//! a genuine NEON-`fmla` / AVX2-`vfmadd` tier without any `unsafe`.

use oxibonsai_kernels::KernelDispatcher;

use crate::error::{ModelError, ModelResult};

use super::{BonsaiModel, OutputWeight};

/// Translate a kernel-crate shape error into the model-crate equivalent.
///
/// [`oxibonsai_kernels::gemv_f32`] names the offending operand (`"output"` /
/// `"input"` / `"weights"`); this keeps the `lm_head <operand>` names the
/// pre-hoist errors used, so callers and tests see the same diagnostics.
fn shape_error(
    err: oxibonsai_kernels::KernelError,
    out_features: usize,
    in_features: usize,
) -> ModelError {
    use oxibonsai_kernels::KernelError as K;
    match err {
        K::NamedBufferTooSmall {
            name: "output",
            available,
            ..
        } => ModelError::ShapeMismatch {
            name: "lm_head output".to_string(),
            expected: vec![out_features],
            actual: vec![available],
        },
        K::NamedDimensionMismatch {
            name: "input", got, ..
        } => ModelError::ShapeMismatch {
            name: "lm_head input".to_string(),
            expected: vec![in_features],
            actual: vec![got],
        },
        K::NamedDimensionMismatch {
            name: "weights",
            got,
            ..
        } => ModelError::ShapeMismatch {
            name: "lm_head weights".to_string(),
            expected: vec![out_features, in_features],
            actual: vec![got],
        },
        other => ModelError::Internal(format!("lm_head: {other}")),
    }
}

/// Dot product of `a[..n]` and `b[..n]` (`n = min(a.len(), b.len())`) with 8
/// independent accumulator lanes.
///
/// A thin re-export of [`oxibonsai_kernels::dot_f32`], kept under this name
/// and visibility because `model/types/tests.rs` asserts directly on it
/// (including the shorter-operand truncation). Production reaches the same
/// kernel through [`forward_f32_with`], so this exists only for those tests.
#[cfg(test)]
#[inline]
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    oxibonsai_kernels::dot_f32(a, b)
}

/// `out[..out_features] = weights[out_features × in_features] · input[..in_features]`,
/// dispatched through `kernel`.
///
/// An **empty** `weights` slice denotes the all-zero LM head of the weightless
/// config-only constructors (M-33): `out` is filled with zeros without ever
/// materializing the `vocab × hidden` zero matrix (4.9 GiB together with the
/// embedding for the 27B config). This is not a special numeric case — it is
/// exactly what multiplying by a zero matrix produces.
///
/// # Errors
///
/// [`ModelError::ShapeMismatch`] if `weights`, `input` or `out` is too short
/// for the declared shape.
pub(super) fn forward_f32_with(
    kernel: &KernelDispatcher,
    weights: &[f32],
    input: &[f32],
    out: &mut [f32],
    out_features: usize,
    in_features: usize,
) -> ModelResult<()> {
    kernel
        .gemv_f32(weights, input, out, out_features, in_features)
        .map_err(|e| shape_error(e, out_features, in_features))
}

/// [`forward_f32_with`] against the CPU-tier dispatcher.
///
/// Retained at its pre-hoist name, signature and visibility because
/// `model/types/tests.rs` (a different package's file this wave) calls it
/// directly. Production goes through [`forward_f32_with`] with the model's own
/// dispatcher, so this wrapper is test-only — the kernel it reaches is
/// nevertheless bit-identical.
#[cfg(test)]
pub(super) fn forward_f32(
    weights: &[f32],
    input: &[f32],
    out: &mut [f32],
    out_features: usize,
    in_features: usize,
) -> ModelResult<()> {
    let kernel = KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier());
    forward_f32_with(&kernel, weights, input, out, out_features, in_features)
}

// ──────────────────────────────────────────────────────────────────────────
// The output-projection weight itself
// ──────────────────────────────────────────────────────────────────────────
//
// `OutputWeight`'s own inherent methods, the fused-GPU logits copy and the
// weight-less constructors' shared dispatcher all live here rather than in
// `model/types/mod.rs`: they are LM-head concerns, and mod.rs is at the
// project's 2000-line ceiling. The enum itself stays declared in mod.rs, so
// every `super::OutputWeight` / `crate::model::types::OutputWeight` path
// elsewhere in the crate resolves exactly as before.

impl OutputWeight<'_> {
    // NOTE on visibility: these four are `pub(in crate::model)`, not
    // `pub(super)`. They were `pub(super)` while they lived in
    // `model/types/mod.rs`, where that *meant* `pub(in crate::model)`; spelling
    // it `pub(super)` here would silently narrow them to
    // `pub(in crate::model::types)` and break `crate::model`'s other children
    // -- including `k_quant_format`'s CUDA-only callers, which no macOS build
    // compiles.
    /// All-zero FP32 LM head of shape `[out_features × in_features]`, stored in
    /// O(1) memory.
    pub(in crate::model) fn zero_fp32(out_features: usize, in_features: usize) -> Self {
        Self::Fp32 {
            weights: Vec::new(),
            out_features,
            in_features,
        }
    }

    /// Short, stable name of this variant, for diagnostics.
    pub(in crate::model) fn kind(&self) -> &'static str {
        match self {
            Self::OneBit(_) => "Q1_0_g128",
            Self::Ternary(_) => "TQ2_0_g128",
            Self::FP8E4M3(_) => "F8_E4M3",
            Self::FP8E5M2(_) => "F8_E5M2",
            Self::Q4_0(_) => "Q4_0",
            Self::Q8_0(_) => "Q8_0",
            Self::Q5K(_) => "Q5_K",
            Self::Q6K(_) => "Q6_K",
            Self::Q2K(_) => "Q2_K",
            Self::Q3K(_) => "Q3_K",
            Self::Q4K(_) => "Q4_K",
            Self::Q8K(_) => "Q8_K",
            Self::Fp32 { .. } => "F32",
        }
    }

    /// The K-quant format of this LM head, or `None` if it is not a K-quant.
    #[cfg(all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(in crate::model) fn k_quant_format(&self) -> Option<oxibonsai_kernels::KQuantFormat> {
        use oxibonsai_kernels::KQuantFormat;
        match self {
            Self::Q2K(_) => Some(KQuantFormat::Q2K),
            Self::Q3K(_) => Some(KQuantFormat::Q3K),
            Self::Q4K(_) => Some(KQuantFormat::Q4K),
            Self::Q5K(_) => Some(KQuantFormat::Q5K),
            Self::Q6K(_) => Some(KQuantFormat::Q6K),
            Self::Q8K(_) => Some(KQuantFormat::Q8K),
            _ => None,
        }
    }

    /// Heap bytes this LM head owns (quantized weights are borrowed from the
    /// memory-mapped GGUF and cost nothing).
    pub(in crate::model) fn resident_bytes(&self) -> usize {
        match self {
            Self::Fp32 { weights, .. } => weights.len() * std::mem::size_of::<f32>(),
            _ => 0,
        }
    }
}

/// Copy a fused-GPU logits buffer into the caller's `[vocab_size]` slice.
///
/// Gated to exactly the configurations that have a fused GPU entry point to
/// copy from.
///
/// The GPU entry points own their output `Vec` and size it from the LM head's
/// `out_features`; if that ever disagrees with `vocab_size`, copy the overlap
/// and zero the rest instead of panicking on a slice index (M-29).
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    )
))]
pub(super) fn copy_logits(produced: &[f32], out: &mut [f32]) {
    let n = produced.len().min(out.len());
    out[..n].copy_from_slice(&produced[..n]);
    if n < out.len() {
        tracing::warn!(
            produced = produced.len(),
            expected = out.len(),
            "fused GPU path produced fewer logits than the vocabulary; zero-filling the tail"
        );
        out[n..].fill(0.0);
    }
}

/// The shared CPU-tier dispatcher the weight-less constructors give their
/// [`OutputWeight::Fp32`] head (K-12/M-23).
///
/// A config-only model's LM head is the *empty* all-zero head, so the tier can
/// have no effect on its logits; what matters is that `BonsaiModel::new` stays
/// as cheap as it was before the field existed. Building one process-global
/// CPU-tier dispatcher keeps it that way: no GPU probe (`auto_detect` opens the
/// device), and no per-model feature detection in the hundreds of tests that
/// construct these fixtures.
pub(super) fn weightless_lm_head_kernel() -> std::sync::Arc<oxibonsai_kernels::KernelDispatcher> {
    static KERNEL: std::sync::OnceLock<std::sync::Arc<oxibonsai_kernels::KernelDispatcher>> =
        std::sync::OnceLock::new();
    KERNEL
        .get_or_init(|| {
            std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::with_tier(
                oxibonsai_kernels::cpu_kernel_tier(),
            ))
        })
        .clone()
}

impl BonsaiModel<'_> {
    /// Project the final hidden state through the LM head into `logits`.
    ///
    /// Every quantized variant already dispatches through the
    /// `Arc<KernelDispatcher>` its [`LinearLayer`] holds; the dense-FP32 head
    /// now does too, via [`Self::lm_head_kernel`] (K-12/M-23) instead of a
    /// crate-private copy of the kernel.
    pub(super) fn apply_lm_head(&self, normed: &[f32], logits: &mut [f32]) -> ModelResult<()> {
        match &self.output_weight {
            OutputWeight::OneBit(linear) => linear.forward_vec(normed, logits),
            OutputWeight::Ternary(linear) => linear.forward(normed, logits),
            OutputWeight::FP8E4M3(linear) => linear.forward(normed, logits),
            OutputWeight::FP8E5M2(linear) => linear.forward(normed, logits),
            OutputWeight::Q4_0(linear) => linear.forward(normed, logits),
            OutputWeight::Q8_0(linear) => linear.forward(normed, logits),
            OutputWeight::Q5K(linear) => linear.forward(normed, logits),
            OutputWeight::Q6K(linear) => linear.forward(normed, logits),
            OutputWeight::Q2K(linear) => linear.forward(normed, logits),
            OutputWeight::Q3K(linear) => linear.forward(normed, logits),
            OutputWeight::Q4K(linear) => linear.forward(normed, logits),
            OutputWeight::Q8K(linear) => linear.forward(normed, logits),
            OutputWeight::Fp32 {
                weights,
                out_features,
                in_features,
            } => forward_f32_with(
                &self.lm_head_kernel,
                weights,
                normed,
                logits,
                *out_features,
                *in_features,
            ),
        }
    }
}
