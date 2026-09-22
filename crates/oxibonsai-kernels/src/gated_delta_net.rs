//! Gated DeltaNet recurrence — types, state and the decode step.
//!
//! 48 of the 64 layers of PrismML **Bonsai 2 27B** (`qwen35`) are linear
//! attention: instead of a growing KV cache they carry a fixed-size recurrent
//! state `S` per v-head and update it with the *gated delta rule*
//!
//! ```text
//! S_t = decay_t · (I − β_t · k_t k_tᵀ) · S_{t−1} + β_t · k_t v_tᵀ
//! o_t = S_tᵀ q_t · 1/sqrt(head_v_dim)
//! ```
//!
//! The authoritative CPU reference is the PrismML llama.cpp fork,
//! `ggml_compute_forward_gated_delta_net_one_chunk`
//! (`fork/models/gated_delta_net_cpu.cpp.txt:108-160`); this module is a
//! faithful port of it, including the two details that are easy to miss:
//!
//! * the output scale is `1/sqrt(S_v)` (`:82`, applied at `:158`) — for
//!   Bonsai 2 that is `1/sqrt(128)`, **not** `1`;
//! * the softplus used by the gate has the ggml cutoff `x > 20.0` (`:121-123`)
//!   — strictly greater, so `x == 20.0` still takes the `ln(1 + exp(x))`
//!   branch. See [`softplus`].
//!
//! # State layout (do not change)
//!
//! Per v-head the `head_v_dim × head_k_dim` state is stored **transposed**:
//!
//! ```text
//! slab[j * head_k_dim + i] == S[i][j]        i: key channel, j: value channel
//! ```
//!
//! Row `j` of the slab is therefore *column* `j` of the mathematical state, and
//! every inner loop of the recurrence (`k·S`, `S += k⊗δ`, `q·S`) becomes a
//! contiguous, vectorisable run over `head_k_dim` floats. For Bonsai 2 one
//! layer holds `48 × 128 × 128 × 4 B = 3.0 MiB`, and all 48 linear layers
//! `144 MiB` per sequence — which is why [`GdnState`] is allocated **once** per
//! sequence (see [`GdnState::with_layers`]) and never per token, and why the
//! decode step is bandwidth-bound rather than FLOP-bound (113 MFLOP/token
//! against 144 MiB/token of state traffic). The `S *= exp(g)` decay is
//! consequently *fused* into the following `k·S` dot so the slab streams once
//! instead of three times; see [`GdnPath`].
//!
//! # Head sharing
//!
//! `n_v_heads` v-heads share `n_k_heads` k/q-heads (Bonsai 2: 48 over 16). The
//! repeat is an **index**, never a materialised copy — see [`GdnHeadOrder`].
//!
//! # Entry points
//!
//! | function | shape |
//! |---|---|
//! | [`gdn_step_f32`] | one token, raw Bonsai 2 gates, state as `&mut [f32]` |
//! | [`gdn_step`] | one token, state as `&mut GdnState` (per-sequence allocation) |
//! | [`gdn_step_with`] | one token, any [`GdnGates`] / [`GdnHeadOrder`] / [`GdnPath`] |
//! | [`gdn_prefill_f32`] | `t_len` tokens, raw Bonsai 2 gates |
//! | [`gdn_chunk`] | `t_len` tokens into a [`GdnState`] |
//! | [`gdn_prefill_with`] | `t_len` tokens, any variant |
//!
//! Decode is literally prefill with `t_len == 1`: [`gdn_step_with`] forwards to
//! [`gdn_prefill_with`], so "prefill over `T` tokens == `T` successive steps"
//! holds **bitwise** by construction, not by convention.

use crate::error::{KernelError, KernelResult};
pub use crate::gated_delta_net_chunk::{
    gdn_chunk, gdn_chunk_f32, gdn_prefill_f32, gdn_prefill_with,
};

/// Geometry of one Gated DeltaNet layer.
///
/// `n_v_heads` must be a whole multiple of `n_k_heads`; the ratio is the
/// v-head repeat [`GdnDims::v_per_k`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDims {
    /// Number of k/q heads (`ssm.group_count`, Bonsai 2: 16).
    pub n_k_heads: usize,
    /// Number of v heads (`ssm.time_step_rank`, Bonsai 2: 48).
    pub n_v_heads: usize,
    /// Key/query channels per head (`ssm.state_size`, Bonsai 2: 128).
    pub head_k_dim: usize,
    /// Value channels per head (Bonsai 2: 128).
    pub head_v_dim: usize,
}

impl GdnDims {
    /// Construct a geometry; validate it with [`GdnDims::validate`].
    #[must_use]
    pub const fn new(
        n_k_heads: usize,
        n_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Self {
        Self {
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
        }
    }

    /// PrismML Bonsai 2 27B: 16 k-heads, 48 v-heads, 128 × 128 state.
    #[must_use]
    pub const fn bonsai2() -> Self {
        Self::new(16, 48, 128, 128)
    }

    /// v-heads per k-head (Bonsai 2: 3).
    ///
    /// Returns 0 for the degenerate `n_k_heads == 0`, which
    /// [`GdnDims::validate`] rejects.
    #[must_use]
    pub const fn v_per_k(&self) -> usize {
        match self.n_v_heads.checked_div(self.n_k_heads) {
            Some(rep) => rep,
            None => 0,
        }
    }

    /// Floats in one layer's recurrent state: `n_v_heads · head_v_dim · head_k_dim`.
    #[must_use]
    pub const fn state_len(&self) -> usize {
        self.n_v_heads * self.head_v_dim * self.head_k_dim
    }

    /// Floats in one token's `q` (and `k`): `n_k_heads · head_k_dim`.
    #[must_use]
    pub const fn qk_len(&self) -> usize {
        self.n_k_heads * self.head_k_dim
    }

    /// Floats in one token's `v` (and the output): `n_v_heads · head_v_dim`.
    #[must_use]
    pub const fn v_len(&self) -> usize {
        self.n_v_heads * self.head_v_dim
    }

    /// Output scale `1/sqrt(head_v_dim)`.
    ///
    /// Transcribed from the fork, where it is `1.0f / sqrtf((float) S_v)` with
    /// `S_v = src_v->ne[0]`, i.e. the **value** channel count. Bonsai 2 has
    /// `head_k_dim == head_v_dim == 128`, so the distinction is unobservable on
    /// the real model; for a non-square geometry this crate's documented
    /// semantics are row length = `head_k_dim`, row count = `head_v_dim`,
    /// scale = `1/sqrt(head_v_dim)`.
    #[must_use]
    pub fn out_scale(&self) -> f32 {
        1.0 / (self.head_v_dim as f32).sqrt()
    }

    /// Reject a geometry the kernels cannot honour.
    ///
    /// # Errors
    ///
    /// [`KernelError::DimensionMismatch`] if any dimension is zero, or
    /// [`KernelError::NamedDimensionMismatch`] if `n_v_heads` is not a whole
    /// multiple of `n_k_heads` (the v-head repeat would not be an integer).
    pub fn validate(&self) -> KernelResult<()> {
        for (name, value) in [
            ("n_k_heads", self.n_k_heads),
            ("n_v_heads", self.n_v_heads),
            ("head_k_dim", self.head_k_dim),
            ("head_v_dim", self.head_v_dim),
        ] {
            if value == 0 {
                return Err(KernelError::dimension_mismatch(name, 1, 0));
            }
        }
        if !self.n_v_heads.is_multiple_of(self.n_k_heads) {
            return Err(KernelError::dimension_mismatch(
                "n_v_heads",
                self.n_k_heads * self.v_per_k().max(1),
                self.n_v_heads,
            ));
        }
        Ok(())
    }
}

/// Which k/q-head a v-head reads.
///
/// The GGUF stores v-indexed *rows* tiled (`j ↔ j % n_k_heads`) while
/// `ssm_out`'s *columns* are grouped (`m ↔ m / v_per_k`); the model layer keeps
/// the weights byte-identical and re-indexes activations at slice time
/// (design §3.3), so the kernel works in **grouped** order by default and the
/// caller hands it already-re-indexed `v`/gate slices.
///
/// [`GdnHeadOrder::Tiled`] exists so the kernel can also be driven in the
/// fork's own index space — that is what the golden dump from
/// `gated_delta_net_cpu.cpp.txt` uses (`ik1 = iv1 % nek1`). The two orders
/// coincide when `v_per_k == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnHeadOrder {
    /// `k_head = v_head / v_per_k` — the order `ssm_out` consumes (default).
    #[default]
    Grouped,
    /// `k_head = v_head % n_k_heads` — the fork's raw GGUF row order.
    Tiled,
}

impl GdnHeadOrder {
    /// The k/q-head that v-head `v_head` reads under this order.
    #[must_use]
    pub const fn k_head(self, v_head: usize, dims: &GdnDims) -> usize {
        match self {
            Self::Grouped => match v_head.checked_div(dims.v_per_k()) {
                Some(k_head) => k_head,
                None => 0,
            },
            Self::Tiled => match v_head.checked_rem(dims.n_k_heads) {
                Some(k_head) => k_head,
                None => 0,
            },
        }
    }
}

/// How the `S *= decay` scale is applied.
///
/// The fused form is the production path: it folds the decay into the first
/// dot and the delta-rule update into the output dot, so the 3.0 MiB slab is
/// read once and written once per token instead of three times each.
///
/// The three-pass form is the fork's literal `scale → dots → mads → dots`
/// ordering, kept as the correctness baseline the fused path is validated
/// against, and used unconditionally for [`GdnDecay::PerChannelLog`] (kda),
/// where the scale is per key-channel and cannot be hoisted the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnPath {
    /// Single streaming pass over the state slab (default).
    #[default]
    Fused,
    /// Fork-faithful three-pass form.
    ThreePass,
}

/// The `β` gate, per token and v-head (`[t_len][n_v_heads]`).
#[derive(Debug, Clone, Copy)]
pub enum GdnBeta<'a> {
    /// Raw projection; `β = sigmoid(raw)` is applied by the kernel
    /// (`raw_gates = true` in the fork).
    Raw(&'a [f32]),
    /// Already activated; used as-is.
    Activated(&'a [f32]),
}

impl<'a> GdnBeta<'a> {
    /// Underlying slice, whichever activation state it is in.
    #[must_use]
    pub const fn values(&self) -> &'a [f32] {
        match self {
            Self::Raw(v) | Self::Activated(v) => v,
        }
    }

    /// Activated β for flat index `t * n_v_heads + h`.
    ///
    /// The caller has already proven the index is in bounds (see
    /// `validate_gdn_call`); an out-of-range index yields `0.0` rather than a
    /// panic so the hot loop never carries a bounds-check branch of its own.
    #[inline]
    #[must_use]
    pub fn at(&self, idx: usize) -> f32 {
        match self {
            Self::Raw(v) => v.get(idx).map_or(0.0, |x| sigmoid(*x)),
            Self::Activated(v) => v.get(idx).copied().unwrap_or(0.0),
        }
    }
}

/// The decay gate.
///
/// Bonsai 2 uses [`GdnDecay::ScalarRaw`]: `ssm_a` and `ssm_dt.bias` are
/// `[n_v_heads]`, one value per v-head. The other two variants keep the
/// signature able to express the fork's remaining modes without a rewrite —
/// `raw_gates = false` ([`GdnDecay::ScalarLog`]) and the per-channel `kda`
/// branch ([`GdnDecay::PerChannelLog`], `gated_delta_net_cpu.cpp.txt:129-139`).
#[derive(Debug, Clone, Copy)]
pub enum GdnDecay<'a> {
    /// Bonsai 2: `g = a_neg[h] · softplus(alpha_raw[t][h] + dt_bias[h])`.
    ///
    /// * `alpha_raw`: `[t_len][n_v_heads]` raw `ssm_alpha` projection.
    /// * `dt_bias`: `[n_v_heads]` (`ssm_dt.bias`).
    /// * `a_neg`: `[n_v_heads]` (`ssm_a`, `A = -exp(A_log)`, negative), used
    ///   **as-is** — see [`validate_a_neg`].
    ScalarRaw {
        /// Raw `ssm_alpha` projection, `[t_len][n_v_heads]`.
        alpha_raw: &'a [f32],
        /// `ssm_dt.bias`, `[n_v_heads]`.
        dt_bias: &'a [f32],
        /// `ssm_a`, `[n_v_heads]`, negative.
        a_neg: &'a [f32],
    },
    /// Pre-computed log-decay `g`, `[t_len][n_v_heads]`; decay is `exp(g)`.
    ScalarLog(&'a [f32]),
    /// kda: per key-channel log-decay, `[t_len][n_v_heads][head_k_dim]`;
    /// channel `i` of the state is scaled by `exp(g[i])`.
    PerChannelLog(&'a [f32]),
}

/// The gate pair consumed by one Gated DeltaNet call.
#[derive(Debug, Clone, Copy)]
pub struct GdnGates<'a> {
    /// `β` gate.
    pub beta: GdnBeta<'a>,
    /// Decay gate.
    pub decay: GdnDecay<'a>,
}

impl<'a> GdnGates<'a> {
    /// The Bonsai 2 gate set: raw `β` and raw scalar decay.
    #[must_use]
    pub const fn bonsai2(
        alpha_raw: &'a [f32],
        beta_raw: &'a [f32],
        dt_bias: &'a [f32],
        a_neg: &'a [f32],
    ) -> Self {
        Self {
            beta: GdnBeta::Raw(beta_raw),
            decay: GdnDecay::ScalarRaw {
                alpha_raw,
                dt_bias,
                a_neg,
            },
        }
    }

    /// `true` when the decay is per key-channel (kda), which forces the
    /// three-pass path.
    #[must_use]
    pub const fn is_per_channel(&self) -> bool {
        matches!(self.decay, GdnDecay::PerChannelLog(_))
    }

    /// The path actually used for these gates: [`GdnPath::ThreePass`] whenever
    /// the decay is per-channel, otherwise the requested one.
    #[must_use]
    pub const fn effective_path(&self, requested: GdnPath) -> GdnPath {
        if self.is_per_channel() {
            GdnPath::ThreePass
        } else {
            requested
        }
    }

    /// Scalar log-decay for flat index `t * n_v_heads + h`, or `None` for the
    /// per-channel variant.
    #[inline]
    #[must_use]
    pub fn log_decay_at(&self, idx: usize, head: usize) -> Option<f32> {
        match self.decay {
            GdnDecay::ScalarRaw {
                alpha_raw,
                dt_bias,
                a_neg,
            } => {
                let alpha = alpha_raw.get(idx).copied().unwrap_or(0.0);
                let bias = dt_bias.get(head).copied().unwrap_or(0.0);
                let a = a_neg.get(head).copied().unwrap_or(0.0);
                Some(a * softplus(alpha + bias))
            }
            GdnDecay::ScalarLog(g) => Some(g.get(idx).copied().unwrap_or(0.0)),
            GdnDecay::PerChannelLog(_) => None,
        }
    }

    /// Raw per-channel log-decay slice for flat index `t * n_v_heads + h`, or
    /// `None` for the scalar variants.
    #[inline]
    #[must_use]
    pub fn per_channel_at(&self, idx: usize, head_k_dim: usize) -> Option<&'a [f32]> {
        match self.decay {
            GdnDecay::PerChannelLog(g) => g.get(idx * head_k_dim..(idx + 1) * head_k_dim),
            _ => None,
        }
    }
}

/// Logistic sigmoid, `1 / (1 + exp(-x))` — the fork's `beta` activation
/// (`gated_delta_net_cpu.cpp.txt:121`).
///
/// Computed in scalar `f32` libm, never a SIMD polynomial: the gate is three
/// transcendental evaluations per head per token (about 2 300 per token for
/// the whole 27B model), and a polynomial `exp` here would multiply the entire
/// 128 × 128 slab by a slightly wrong decay.
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus with ggml's cutoff, `if x > 20 { x } else { ln(1 + exp(x)) }`
/// (`gated_delta_net_cpu.cpp.txt:123`).
///
/// The comparison is **strictly greater**: `x == 20.0` still evaluates
/// `ln(1 + exp(x))`. `ln(1 + exp(x))` is also deliberately *not* rewritten as
/// `ln_1p(exp(x))` — the fork computes `logf(1.0f + expf(x))`, and rounding
/// `1 + exp(x)` before the logarithm is part of the reference result.
#[inline]
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        #[allow(
            clippy::imprecise_flops,
            reason = "fork parity: logf(1.0f + expf(x)), not log1pf(expf(x))"
        )]
        (1.0 + x.exp()).ln()
    }
}

/// Reject an `ssm_a` vector that is not `A = -exp(A_log)`.
///
/// `A_log` is real, so `exp(A_log) > 0` and every element of `ssm_a` must be
/// `<= 0` (`-0.0` and `0.0` pass; an underflowed `exp` legitimately produces
/// them). A positive `a` flips the sign of `g = a · softplus(...)`, making
/// `decay = exp(g) > 1`, which makes the recurrence diverge instead of decay —
/// a model loaded that way produces noise, so the condition is an error, never
/// a log line. The weight loader calls this at load time; the kernels call it
/// once per entry, over the `[n_v_heads]` slice, never inside the per-head loop.
///
/// # Errors
///
/// [`KernelError::UnsupportedOperation`] naming the first offending index and
/// value, for a positive or NaN element.
pub fn validate_a_neg(a_neg: &[f32]) -> KernelResult<()> {
    for (idx, value) in a_neg.iter().enumerate() {
        if value.is_nan() || *value > 0.0 {
            return Err(KernelError::UnsupportedOperation(format!(
                "ssm_a[{idx}] = {value} is not a valid A = -exp(A_log): \
                 every element must be <= 0 (a positive or NaN decay rate \
                 makes exp(g) > 1 and the recurrence diverge)"
            )));
        }
    }
    Ok(())
}

/// Recurrent Gated DeltaNet state for one sequence.
///
/// Holds `n_layers` slabs of `[n_v_heads][head_v_dim][head_k_dim]` f32 in the
/// transposed layout documented at the module level. Allocate it **once** per
/// sequence — for Bonsai 2 the 48 linear layers are 144 MiB, so a per-token
/// allocation would dominate decode.
#[derive(Debug, Clone)]
pub struct GdnState {
    dims: GdnDims,
    n_layers: usize,
    s: Vec<f32>,
}

impl GdnState {
    /// Allocate a zeroed state for a single layer.
    ///
    /// # Errors
    ///
    /// Propagates [`GdnDims::validate`].
    pub fn new(dims: GdnDims) -> KernelResult<Self> {
        Self::with_layers(dims, 1)
    }

    /// Allocate a zeroed state for `n_layers` linear-attention layers.
    ///
    /// # Errors
    ///
    /// Propagates [`GdnDims::validate`]; [`KernelError::DimensionMismatch`] if
    /// `n_layers == 0`.
    pub fn with_layers(dims: GdnDims, n_layers: usize) -> KernelResult<Self> {
        dims.validate()?;
        if n_layers == 0 {
            return Err(KernelError::dimension_mismatch("n_layers", 1, 0));
        }
        Ok(Self {
            dims,
            n_layers,
            s: vec![0.0; n_layers * dims.state_len()],
        })
    }

    /// Geometry this state was allocated for.
    #[must_use]
    pub const fn dims(&self) -> GdnDims {
        self.dims
    }

    /// Number of layers held.
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Resident size in bytes.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.n_layers * self.dims.state_len() * core::mem::size_of::<f32>()
    }

    /// Immutable view of one layer's slab.
    ///
    /// # Errors
    ///
    /// [`KernelError::DimensionMismatch`] if `layer >= n_layers`.
    pub fn layer(&self, layer: usize) -> KernelResult<&[f32]> {
        let len = self.dims.state_len();
        self.s
            .get(layer * len..(layer + 1) * len)
            .ok_or_else(|| KernelError::dimension_mismatch("layer", self.n_layers, layer))
    }

    /// Mutable view of one layer's slab.
    ///
    /// # Errors
    ///
    /// [`KernelError::DimensionMismatch`] if `layer >= n_layers`.
    pub fn layer_mut(&mut self, layer: usize) -> KernelResult<&mut [f32]> {
        let len = self.dims.state_len();
        let n_layers = self.n_layers;
        self.s
            .get_mut(layer * len..(layer + 1) * len)
            .ok_or_else(|| KernelError::dimension_mismatch("layer", n_layers, layer))
    }

    /// One v-head's `[head_v_dim][head_k_dim]` slab inside a layer.
    ///
    /// # Errors
    ///
    /// [`KernelError::DimensionMismatch`] if `layer` or `head` is out of range.
    pub fn head(&self, layer: usize, head: usize) -> KernelResult<&[f32]> {
        let per_head = self.dims.head_v_dim * self.dims.head_k_dim;
        let n_v_heads = self.dims.n_v_heads;
        self.layer(layer)?
            .get(head * per_head..(head + 1) * per_head)
            .ok_or_else(|| KernelError::dimension_mismatch("head", n_v_heads, head))
    }

    /// Whole allocation, layer-major.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.s
    }

    /// Whole allocation, layer-major.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.s
    }

    /// Zero every layer — a new sequence reuses the same allocation.
    pub fn reset(&mut self) {
        self.s.fill(0.0);
    }
}

/// Validate one Gated DeltaNet call's buffers against `dims` and `t_len`.
///
/// Always on (never a `debug_assert!`), following this crate's K-02 length
/// contract: the public entry points prove every slice long enough here, and
/// the SIMD inner kernels then run unchecked.
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] naming the first short buffer,
/// [`KernelError::NamedDimensionMismatch`] for a bad geometry, or the error
/// from [`validate_a_neg`].
#[allow(
    clippy::too_many_arguments,
    reason = "validates every buffer of one call: five slices, the gates, chunk length and geometry"
)]
pub fn validate_gdn_call(
    state: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &[f32],
    t_len: usize,
    dims: &GdnDims,
) -> KernelResult<()> {
    dims.validate()?;
    let checks = [
        ("state", dims.state_len(), state.len()),
        ("q", t_len * dims.qk_len(), q.len()),
        ("k", t_len * dims.qk_len(), k.len()),
        ("v", t_len * dims.v_len(), v.len()),
        ("out", t_len * dims.v_len(), out.len()),
        ("beta", t_len * dims.n_v_heads, gates.beta.values().len()),
    ];
    for (name, needed, available) in checks {
        if available < needed {
            return Err(KernelError::buffer_too_small(name, needed, available));
        }
    }
    match gates.decay {
        GdnDecay::ScalarRaw {
            alpha_raw,
            dt_bias,
            a_neg,
        } => {
            let decay_checks = [
                ("alpha_raw", t_len * dims.n_v_heads, alpha_raw.len()),
                ("dt_bias", dims.n_v_heads, dt_bias.len()),
                ("a_neg", dims.n_v_heads, a_neg.len()),
            ];
            for (name, needed, available) in decay_checks {
                if available < needed {
                    return Err(KernelError::buffer_too_small(name, needed, available));
                }
            }
            validate_a_neg(&a_neg[..dims.n_v_heads])?;
        }
        GdnDecay::ScalarLog(g) => {
            let needed = t_len * dims.n_v_heads;
            if g.len() < needed {
                return Err(KernelError::buffer_too_small("g", needed, g.len()));
            }
        }
        GdnDecay::PerChannelLog(g) => {
            let needed = t_len * dims.n_v_heads * dims.head_k_dim;
            if g.len() < needed {
                return Err(KernelError::buffer_too_small("g", needed, g.len()));
            }
        }
    }
    Ok(())
}

/// One decode step, Bonsai 2 gate shape, state as a plain slice.
///
/// This is the design §2.3 signature. `q`/`k` are `[n_k_heads · head_k_dim]`
/// (already L2-normalised by the caller), `v` and `out` are
/// `[n_v_heads · head_v_dim]`, and the four gate vectors are `[n_v_heads]` in
/// **grouped** v-head order ([`GdnHeadOrder::Grouped`]). `state` is updated in
/// place.
///
/// # Head order (integration contract)
///
/// This entry point assumes **grouped** v-head order
/// ([`GdnHeadOrder::Grouped`]): v-head `h` reads k/q-head `h / v_per_k`, and
/// `v`, the gate vectors and the state slabs are indexed by the same grouped
/// `h`. The GGUF stores v-indexed rows **tiled** (`j ↔ j % n_k_heads`), so a
/// caller holding raw GGUF order must either re-index its slices through the
/// §3.3 v-head map (`hybrid/vhead_map.rs`) or call [`gdn_step_with`] /
/// [`gdn_prefill_with`] with [`GdnHeadOrder::Tiled`]. Passing tiled buffers
/// here is **not** detectable by the kernel — it simply pairs each v-head with
/// the wrong k-head.
///
/// # Argument order
///
/// `dt_bias` comes **before** `a_neg` here (design §2.3). The
/// [`GdnState`]-shaped [`gdn_step`] / [`gdn_chunk`] take them the other way
/// round (work-order shape). Both are `[n_v_heads]` `f32`, so swapping them
/// compiles; [`validate_a_neg`] rejects the swap whenever `dt_bias` holds a
/// positive element, but do not rely on that — check the order at the call
/// site.
///
/// # Errors
///
/// Propagates [`validate_gdn_call`] — including the rejection of a positive
/// `a_neg` ([`validate_a_neg`]).
#[allow(
    clippy::too_many_arguments,
    reason = "design §2.3 signature: four activation buffers plus four gate vectors plus geometry"
)]
pub fn gdn_step_f32(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    dt_bias: &[f32],
    a_neg: &[f32],
    out: &mut [f32],
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> KernelResult<()> {
    let dims = GdnDims::new(n_k_heads, n_v_heads, head_k_dim, head_v_dim);
    let gates = GdnGates::bonsai2(alpha_raw, beta_raw, dt_bias, a_neg);
    gdn_step_with(
        state,
        q,
        k,
        v,
        &gates,
        out,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
}

/// One decode step into a per-sequence [`GdnState`].
///
/// The work-order API shape: the caller owns one [`GdnState`] for the whole
/// sequence and names the layer, so nothing is allocated per token.
///
/// # Head order (integration contract)
///
/// This entry point assumes **grouped** v-head order
/// ([`GdnHeadOrder::Grouped`]): v-head `h` reads k/q-head `h / v_per_k`, and
/// `v`, the gate vectors and the state slabs are indexed by the same grouped
/// `h`. The GGUF stores v-indexed rows **tiled** (`j ↔ j % n_k_heads`), so a
/// caller holding raw GGUF order must either re-index its slices through the
/// §3.3 v-head map (`hybrid/vhead_map.rs`) or call [`gdn_step_with`] /
/// [`gdn_prefill_with`] with [`GdnHeadOrder::Tiled`]. Passing tiled buffers
/// here is **not** detectable by the kernel — it simply pairs each v-head with
/// the wrong k-head.
///
/// # Argument order
///
/// `a_neg` comes **before** `dt_bias` here (work-order shape). The plain-slice
/// [`gdn_step_f32`] / [`gdn_prefill_f32`] take them the other way round
/// (design §2.3). See the note on [`gdn_step_f32`].
///
/// # Errors
///
/// [`KernelError::DimensionMismatch`] for an out-of-range `layer`, otherwise as
/// [`gdn_step_f32`].
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors gdn_step_f32 with an explicit layer index"
)]
pub fn gdn_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    a_neg: &[f32],
    dt_bias: &[f32],
    state: &mut GdnState,
    layer: usize,
    out: &mut [f32],
) -> KernelResult<()> {
    let dims = state.dims();
    let gates = GdnGates::bonsai2(alpha_raw, beta_raw, dt_bias, a_neg);
    let slab = state.layer_mut(layer)?;
    gdn_step_with(
        slab,
        q,
        k,
        v,
        &gates,
        out,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
}

/// One decode step with an explicit gate variant, head order and path.
///
/// Forwards to [`gdn_prefill_with`] with `t_len = 1`, which is what makes
/// "prefill over `T` tokens == `T` successive steps" bitwise rather than
/// approximate.
///
/// # Errors
///
/// Propagates [`validate_gdn_call`].
#[allow(
    clippy::too_many_arguments,
    reason = "explicit-variant entry point: buffers, gates, geometry and both mode selectors"
)]
pub fn gdn_step_with(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &mut [f32],
    dims: &GdnDims,
    order: GdnHeadOrder,
    path: GdnPath,
) -> KernelResult<()> {
    gdn_prefill_with(state, q, k, v, gates, out, 1, dims, order, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_cutoff_is_strictly_greater_than_20() {
        // At exactly 20.0 the fork takes the ln(1+exp(x)) branch.
        let at_20 = softplus(20.0);
        #[allow(clippy::imprecise_flops)]
        let reference_20 = (1.0f32 + 20.0f32.exp()).ln();
        assert_eq!(at_20.to_bits(), reference_20.to_bits());

        let just_above = f32::from_bits(20.0f32.to_bits() + 1);
        assert_eq!(softplus(just_above).to_bits(), just_above.to_bits());

        let just_below = f32::from_bits(20.0f32.to_bits() - 1);
        #[allow(clippy::imprecise_flops)]
        let reference_below = (1.0f32 + just_below.exp()).ln();
        assert_eq!(softplus(just_below).to_bits(), reference_below.to_bits());
    }

    #[test]
    fn dims_reject_non_multiple_head_counts() {
        assert!(GdnDims::new(5, 48, 128, 128).validate().is_err());
        assert!(GdnDims::new(0, 48, 128, 128).validate().is_err());
        // Fully degenerate geometry: every accessor must answer without
        // dividing by zero, and `validate` must reject rather than panic.
        let zero = GdnDims::new(0, 0, 0, 0);
        assert!(zero.validate().is_err());
        assert_eq!(zero.v_per_k(), 0);
        assert_eq!(zero.state_len(), 0);
        assert_eq!(GdnHeadOrder::Grouped.k_head(3, &zero), 0);
        assert_eq!(GdnHeadOrder::Tiled.k_head(3, &zero), 0);
        assert!(GdnDims::bonsai2().validate().is_ok());
        assert_eq!(GdnDims::bonsai2().v_per_k(), 3);
        assert_eq!(GdnDims::bonsai2().state_len(), 48 * 128 * 128);
    }

    #[test]
    fn head_orders_agree_when_the_repeat_is_one() {
        let dims = GdnDims::new(4, 4, 8, 8);
        for h in 0..dims.n_v_heads {
            assert_eq!(
                GdnHeadOrder::Grouped.k_head(h, &dims),
                GdnHeadOrder::Tiled.k_head(h, &dims)
            );
        }
        let dims = GdnDims::new(2, 4, 8, 8);
        assert_eq!(GdnHeadOrder::Grouped.k_head(3, &dims), 1);
        assert_eq!(GdnHeadOrder::Tiled.k_head(3, &dims), 1);
        assert_eq!(GdnHeadOrder::Grouped.k_head(1, &dims), 0);
        assert_eq!(GdnHeadOrder::Tiled.k_head(1, &dims), 1);
    }

    #[test]
    fn state_allocates_once_and_resets() {
        let dims = GdnDims::new(2, 4, 8, 8);
        let mut state = GdnState::with_layers(dims, 3).expect("valid dims");
        assert_eq!(state.n_layers(), 3);
        assert_eq!(state.bytes(), 3 * 4 * 8 * 8 * 4);
        state.layer_mut(1).expect("layer 1")[0] = 1.0;
        assert_eq!(state.head(1, 0).expect("head 0")[0], 1.0);
        assert!(state.layer(3).is_err());
        state.reset();
        assert!(state.as_slice().iter().all(|x| *x == 0.0));
    }

    #[test]
    fn positive_ssm_a_is_rejected() {
        assert!(validate_a_neg(&[-1.0, -0.5, 0.0, -0.0]).is_ok());
        let err = validate_a_neg(&[-1.0, 0.25]).expect_err("positive a must be rejected");
        assert!(err.to_string().contains("ssm_a[1]"));
        assert!(validate_a_neg(&[f32::NAN]).is_err());
    }
}
