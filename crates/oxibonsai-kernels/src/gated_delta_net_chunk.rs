//! Gated DeltaNet chunked prefill and the SIMD row kernels it shares with the
//! decode step.
//!
//! The recurrence itself lives here for both shapes: [`gdn_prefill_with`] runs
//! `t_len` tokens, and
//! [`gdn_step_with`](crate::gated_delta_net::gdn_step_with) is the same call
//! with `t_len == 1`. Decode and prefill therefore execute *the same* per-head,
//! per-token function with the same accumulator layout, which is what makes
//! "prefill over `T` tokens == `T` successive decode steps" hold **bitwise**.
//!
//! Parallelism follows the fork's `one_chunk(ir0, ir1)` split over
//! `n_v_heads × n_seqs`: the recurrence is strictly serial in time, so the
//! chunk is serial per head and rayon-parallel **across** heads. Head `h` only
//! ever touches its own `head_v_dim × head_k_dim` slab, its own output rows and
//! read-only activations, so the parallel and sequential forms are bitwise
//! identical.
//!
//! # Why not the UT-transform
//!
//! The fork also has a `build_delta_net_chunking` path that batches the
//! recurrence through a WY/UT representation. It needs `solve_tri`, `cumsum`,
//! `tri` and `diag` primitives this crate does not have; the sequential
//! recurrence below is `O(t_len · head_v_dim · head_k_dim)` with the whole
//! 64 KiB slab resident in L2 across the chunk, which is the right trade for
//! v1. The UT form is recorded as a v2 performance item
//! (reference: `fork/models/delta-net-base.cpp:16-280`).
//!
//! # Row kernels
//!
//! Three of the six primitives accumulate a dot product; the other three are
//! pure element-wise passes. The fused path pairs them so a state row is
//! visited exactly once per token:
//!
//! | primitive | operation |
//! |---|---|
//! | `scale_dot` | `row *= decay` then `Σ row·k` |
//! | `mad_dot` | `row += k·δ` then `Σ row·q` |
//! | `dot` | `Σ row·x` |
//! | `scale_in_place` | `xs *= decay` (whole slab, three-pass form) |
//! | `mul_in_place` | `row *= exp_g` (kda, per key-channel) |
//! | `mad_in_place` | `row += k·δ` (three-pass form) |
//!
//! `scale_dot` and `mad_dot` are written with the *same* accumulator layout and
//! the same element-wise arithmetic as `scale_in_place`/`mad_in_place` followed
//! by `dot`, so [`GdnPath::Fused`] and [`GdnPath::ThreePass`] agree bitwise on
//! every tier, not merely to a tolerance.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::KernelResult;
use crate::gated_delta_net::{
    validate_gdn_call, GdnDecay, GdnDims, GdnGates, GdnHeadOrder, GdnPath, GdnState,
};

/// Minimum state size (in f32) before the per-head loop engages rayon.
///
/// One Bonsai 2 layer is 786 432 floats, far above this; the unit-test
/// geometries are far below it and stay sequential, which keeps their timing
/// (and any failure) deterministic. The two forms compute identical bits
/// either way.
#[cfg(not(target_arch = "wasm32"))]
const GDN_PAR_MIN_STATE: usize = 1 << 15;

/// Floats of per-head scratch kept on the stack before falling back to a heap
/// allocation (`head_v_dim + head_k_dim`; 256 for Bonsai 2).
const STACK_SCRATCH_FLOATS: usize = 512;

/// SIMD tier used for the Gated DeltaNet row kernels.
///
/// Selected once per head by [`GdnTier::detect`]. Every tier computes the same
/// operations; they differ only in vector width and therefore in summation
/// order, which is why cross-tier parity is a tolerance (≤1e-6 on a 128-wide
/// dot) while same-tier fused-vs-three-pass parity is bitwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnTier {
    /// Portable scalar reference.
    #[default]
    Scalar,
    /// AArch64 NEON (`vfmaq_f32`).
    Neon,
    /// x86-64 AVX2 + FMA.
    Avx2,
    /// x86-64 AVX-512F.
    Avx512,
}

impl GdnTier {
    /// The best tier available on this CPU.
    #[must_use]
    pub fn detect() -> Self {
        detect_tier()
    }

    /// Map a requested tier onto one this build and CPU can actually run,
    /// degrading to [`GdnTier::Scalar`].
    #[must_use]
    pub fn resolve(self) -> Self {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => Self::Neon,
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 if is_x86_feature_detected!("avx512f") => Self::Avx512,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") => {
                Self::Avx2
            }
            _ => Self::Scalar,
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn detect_tier() -> GdnTier {
    GdnTier::Neon
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn detect_tier() -> GdnTier {
    if is_x86_feature_detected!("avx512f") {
        GdnTier::Avx512
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        GdnTier::Avx2
    } else {
        GdnTier::Scalar
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn detect_tier() -> GdnTier {
    GdnTier::Scalar
}

/// Where one head writes its `head_v_dim`-wide output row for token `t`.
///
/// Decode (`t_len == 1`) hands each head a contiguous row, so nothing is
/// allocated. Prefill gathers the head's rows out of the token-major output
/// buffer as disjoint mutable slices — pointers only, never a copy of the
/// activations.
enum OutView<'a, 'b> {
    /// `[t_len][head_v_dim]` contiguous for this head.
    Contiguous(&'a mut [f32]),
    /// One slice per token.
    Rows(&'a mut [&'b mut [f32]]),
}

impl OutView<'_, '_> {
    #[inline]
    fn row(&mut self, t: usize, head_v_dim: usize) -> &mut [f32] {
        match self {
            // Bounds proven by `validate_gdn_call` before any head runs (K-02).
            Self::Contiguous(buf) => &mut buf[t * head_v_dim..(t + 1) * head_v_dim],
            Self::Rows(rows) => rows[t],
        }
    }
}

/// Immutable per-call parameters handed to every head.
#[derive(Clone, Copy)]
struct HeadJob<'a> {
    dims: &'a GdnDims,
    order: GdnHeadOrder,
    path: GdnPath,
    tier: GdnTier,
    t_len: usize,
    scale: f32,
}

/// Prefill `t_len` tokens with the Bonsai 2 gate shape.
///
/// This is the design §2.3 signature. `q`/`k` are `[t_len][n_k_heads ·
/// head_k_dim]`, `v` and `out` are `[t_len][n_v_heads · head_v_dim]`,
/// `alpha_raw`/`beta_raw` are `[t_len][n_v_heads]` and `dt_bias`/`a_neg` are
/// `[n_v_heads]`, all in **grouped** v-head order. `state` carries across
/// chunks and is updated in place.
///
/// # Head order (integration contract)
///
/// Assumes **grouped** v-head order ([`GdnHeadOrder::Grouped`]): v-head `h`
/// reads k/q-head `h / v_per_k`, and `v`, the gate vectors and the state slabs
/// are indexed by that same grouped `h`. The GGUF stores v-indexed rows
/// **tiled** (`j ↔ j % n_k_heads`), so a caller holding raw GGUF order must
/// re-index through the design §3.3 v-head map or call [`gdn_prefill_with`]
/// with [`GdnHeadOrder::Tiled`]. Tiled buffers passed here are **not**
/// detectable by the kernel — each v-head is simply paired with the wrong
/// k-head.
///
/// # Argument order
///
/// `dt_bias` comes **before** `a_neg` (design §2.3); [`gdn_chunk`] takes them
/// the other way round (work-order shape). Both are `[n_v_heads]` `f32`, so a
/// swap compiles — check the order at the call site.
///
/// # Errors
///
/// Propagates [`validate_gdn_call`], including the rejection of a positive
/// `a_neg`.
#[allow(
    clippy::too_many_arguments,
    reason = "design §2.3 signature: activations, four gate vectors, chunk length and geometry"
)]
pub fn gdn_prefill_f32(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    dt_bias: &[f32],
    a_neg: &[f32],
    out: &mut [f32],
    t_len: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> KernelResult<()> {
    let dims = GdnDims::new(n_k_heads, n_v_heads, head_k_dim, head_v_dim);
    let gates = GdnGates::bonsai2(alpha_raw, beta_raw, dt_bias, a_neg);
    gdn_prefill_with(
        state,
        q,
        k,
        v,
        &gates,
        out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
}

/// Alias of [`gdn_prefill_f32`] under the work-order's name for the same call.
///
/// Head order and argument order are [`gdn_prefill_f32`]'s.
///
/// # Errors
///
/// As [`gdn_prefill_f32`].
#[allow(
    clippy::too_many_arguments,
    reason = "alias of gdn_prefill_f32; same parameter list by definition"
)]
pub fn gdn_chunk_f32(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    dt_bias: &[f32],
    a_neg: &[f32],
    out: &mut [f32],
    t_len: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> KernelResult<()> {
    gdn_prefill_f32(
        state, q, k, v, alpha_raw, beta_raw, dt_bias, a_neg, out, t_len, n_k_heads, n_v_heads,
        head_k_dim, head_v_dim,
    )
}

/// Prefill a chunk into a per-sequence [`GdnState`].
///
/// # Head order (integration contract)
///
/// Assumes **grouped** v-head order ([`GdnHeadOrder::Grouped`]): v-head `h`
/// reads k/q-head `h / v_per_k`, and `v`, the gate vectors and the state slabs
/// are indexed by that same grouped `h`. The GGUF stores v-indexed rows
/// **tiled** (`j ↔ j % n_k_heads`), so a caller holding raw GGUF order must
/// re-index through the design §3.3 v-head map or call [`gdn_prefill_with`]
/// with [`GdnHeadOrder::Tiled`]. Tiled buffers passed here are **not**
/// detectable by the kernel — each v-head is simply paired with the wrong
/// k-head.
///
/// # Argument order
///
/// `a_neg` comes **before** `dt_bias` (work-order shape); [`gdn_prefill_f32`]
/// takes them the other way round (design §2.3).
///
/// # Errors
///
/// [`crate::error::KernelError::DimensionMismatch`] for an out-of-range
/// `layer`, otherwise as [`gdn_prefill_f32`].
#[allow(
    clippy::too_many_arguments,
    reason = "GdnState-shaped mirror of gdn_prefill_f32 with an explicit layer index"
)]
pub fn gdn_chunk(
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
    t_len: usize,
) -> KernelResult<()> {
    let dims = state.dims();
    let gates = GdnGates::bonsai2(alpha_raw, beta_raw, dt_bias, a_neg);
    let slab = state.layer_mut(layer)?;
    gdn_prefill_with(
        slab,
        q,
        k,
        v,
        &gates,
        out,
        t_len,
        &dims,
        GdnHeadOrder::Grouped,
        GdnPath::Fused,
    )
}

/// Prefill with an explicit gate variant, head order and path, on the best
/// SIMD tier this CPU offers.
///
/// # Errors
///
/// Propagates [`validate_gdn_call`].
#[allow(
    clippy::too_many_arguments,
    reason = "explicit-variant entry point: buffers, gates, chunk length, geometry, both mode selectors"
)]
pub fn gdn_prefill_with(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &mut [f32],
    t_len: usize,
    dims: &GdnDims,
    order: GdnHeadOrder,
    path: GdnPath,
) -> KernelResult<()> {
    gdn_prefill_with_tier(
        state,
        q,
        k,
        v,
        gates,
        out,
        t_len,
        dims,
        order,
        path,
        GdnTier::detect(),
    )
}

/// Prefill, pinning the SIMD tier.
///
/// The tier is [`GdnTier::resolve`]d first, so requesting a tier this CPU does
/// not have degrades to [`GdnTier::Scalar`] instead of executing an illegal
/// instruction. Intended for cross-tier parity tests and for callers that must
/// reproduce another machine's summation order.
///
/// # Errors
///
/// Propagates [`validate_gdn_call`].
#[allow(
    clippy::too_many_arguments,
    reason = "gdn_prefill_with plus an explicit SIMD tier"
)]
pub fn gdn_prefill_with_tier(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &mut [f32],
    t_len: usize,
    dims: &GdnDims,
    order: GdnHeadOrder,
    path: GdnPath,
    tier: GdnTier,
) -> KernelResult<()> {
    validate_gdn_call(state, q, k, v, gates, out, t_len, dims)?;
    if t_len == 0 {
        return Ok(());
    }

    let job = HeadJob {
        dims,
        order,
        path: gates.effective_path(path),
        tier: tier.resolve(),
        t_len,
        scale: dims.out_scale(),
    };
    let per_head = dims.head_v_dim * dims.head_k_dim;
    let state = &mut state[..dims.state_len()];
    #[cfg(not(target_arch = "wasm32"))]
    let parallel = dims.state_len() >= GDN_PAR_MIN_STATE && dims.n_v_heads > 1;

    if t_len == 1 {
        // Decode: head `h` owns `out[h * head_v_dim ..]`, already contiguous.
        let out = &mut out[..dims.v_len()];
        #[cfg(not(target_arch = "wasm32"))]
        if parallel {
            state
                .par_chunks_mut(per_head)
                .zip(out.par_chunks_mut(dims.head_v_dim))
                .enumerate()
                .for_each(|(head, (slab, out_head))| {
                    run_head(
                        slab,
                        q,
                        k,
                        v,
                        gates,
                        &mut OutView::Contiguous(out_head),
                        head,
                        &job,
                    );
                });
            return Ok(());
        }
        state
            .chunks_mut(per_head)
            .zip(out.chunks_mut(dims.head_v_dim))
            .enumerate()
            .for_each(|(head, (slab, out_head))| {
                run_head(
                    slab,
                    q,
                    k,
                    v,
                    gates,
                    &mut OutView::Contiguous(out_head),
                    head,
                    &job,
                );
            });
        return Ok(());
    }

    // Prefill: the output is token-major, so head `h` owns the strided rows
    // `t * n_v_heads + h`. Gather them as disjoint mutable slices (pointers
    // only — the activations are never copied).
    let n_v_heads = dims.n_v_heads;
    let mut rows: Vec<Vec<&mut [f32]>> =
        (0..n_v_heads).map(|_| Vec::with_capacity(t_len)).collect();
    for (idx, row) in out[..t_len * dims.v_len()]
        .chunks_mut(dims.head_v_dim)
        .enumerate()
    {
        rows[idx % n_v_heads].push(row);
    }

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        state
            .par_chunks_mut(per_head)
            .zip(rows.par_iter_mut())
            .enumerate()
            .for_each(|(head, (slab, head_rows))| {
                run_head(
                    slab,
                    q,
                    k,
                    v,
                    gates,
                    &mut OutView::Rows(head_rows.as_mut_slice()),
                    head,
                    &job,
                );
            });
        return Ok(());
    }
    state
        .chunks_mut(per_head)
        .zip(rows.iter_mut())
        .enumerate()
        .for_each(|(head, (slab, head_rows))| {
            run_head(
                slab,
                q,
                k,
                v,
                gates,
                &mut OutView::Rows(head_rows.as_mut_slice()),
                head,
                &job,
            );
        });
    Ok(())
}

/// Run the whole chunk for one v-head, allocating its scratch once.
#[allow(
    clippy::too_many_arguments,
    reason = "per-head entry: four activation buffers, gates, output view, head index, job"
)]
fn run_head(
    slab: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &mut OutView<'_, '_>,
    head: usize,
    job: &HeadJob<'_>,
) {
    if job.path != GdnPath::ThreePass {
        // The fused path keeps `delta` in a register; no scratch at all.
        gdn_head_chunk(slab, q, k, v, gates, out, &mut [], head, job);
        return;
    }
    let need = job.dims.head_v_dim + job.dims.head_k_dim;
    let mut stack = [0.0f32; STACK_SCRATCH_FLOATS];
    let mut heap: Vec<f32> = Vec::new();
    let scratch: &mut [f32] = if need <= STACK_SCRATCH_FLOATS {
        &mut stack[..need]
    } else {
        heap.resize(need, 0.0);
        &mut heap[..]
    };
    gdn_head_chunk(slab, q, k, v, gates, out, scratch, head, job);
}

/// The recurrence for one v-head over `t_len` tokens.
///
/// Every slice index below is inside a range `validate_gdn_call` has already
/// proven (K-02): the public entry points validate, the inner kernels run
/// unchecked.
#[allow(
    clippy::too_many_arguments,
    reason = "inner per-head kernel: activations, gates, output view, scratch, head index, job"
)]
fn gdn_head_chunk(
    slab: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gates: &GdnGates<'_>,
    out: &mut OutView<'_, '_>,
    scratch: &mut [f32],
    head: usize,
    job: &HeadJob<'_>,
) {
    let dims = job.dims;
    let (hk, hv) = (dims.head_k_dim, dims.head_v_dim);
    let k_head = job.order.k_head(head, dims);
    let (delta, exp_g) = scratch.split_at_mut(scratch.len().min(hv));

    for t in 0..job.t_len {
        let idx = t * dims.n_v_heads + head;
        let qk_off = (t * dims.n_k_heads + k_head) * hk;
        let q_t = &q[qk_off..qk_off + hk];
        let k_t = &k[qk_off..qk_off + hk];
        let v_off = idx * hv;
        let v_t = &v[v_off..v_off + hv];
        let beta = gates.beta.at(idx);
        let out_t = out.row(t, hv);

        match gates.log_decay_at(idx, head) {
            Some(g) => {
                let decay = g.exp();
                if job.path == GdnPath::Fused {
                    gdn_token_fused(slab, q_t, k_t, v_t, out_t, decay, beta, job.scale, job.tier);
                } else {
                    scale_in_place(job.tier, &mut slab[..hv * hk], decay);
                    gdn_token_three_pass(
                        slab, q_t, k_t, v_t, out_t, delta, beta, job.scale, job.tier,
                    );
                }
            }
            None => {
                // kda: per key-channel decay. `exp(g)` is materialised once per
                // token (the fork does the same, into its `delta` scratch) and
                // the rows are scaled column-wise, which the fused form cannot
                // express — hence ThreePass unconditionally.
                let g_off = idx * hk;
                let g_t = match gates.decay {
                    GdnDecay::PerChannelLog(g) => &g[g_off..g_off + hk],
                    _ => &[],
                };
                let exp_g = &mut exp_g[..hk];
                for (dst, src) in exp_g.iter_mut().zip(g_t.iter()) {
                    *dst = src.exp();
                }
                for j in 0..hv {
                    mul_in_place(job.tier, &mut slab[j * hk..(j + 1) * hk], exp_g);
                }
                gdn_token_three_pass(slab, q_t, k_t, v_t, out_t, delta, beta, job.scale, job.tier);
            }
        }
    }
}

/// Fused single-pass token update: each state row is visited exactly once.
///
/// Rows are independent — row `j` depends only on itself, `k`, `q`, `v[j]`,
/// `beta` and `decay` — so interleaving the delta-rule update with the output
/// dot produces exactly the bits the three-pass form produces, while the
/// 3.0 MiB slab is streamed once instead of four times.
#[allow(
    clippy::too_many_arguments,
    reason = "token kernel: four activation slices, two gates, scale and tier"
)]
#[inline]
fn gdn_token_fused(
    slab: &mut [f32],
    q_t: &[f32],
    k_t: &[f32],
    v_t: &[f32],
    out_t: &mut [f32],
    decay: f32,
    beta: f32,
    scale: f32,
    tier: GdnTier,
) {
    let hk = k_t.len();
    for (j, (v_j, out_j)) in v_t.iter().zip(out_t.iter_mut()).enumerate() {
        let row = &mut slab[j * hk..(j + 1) * hk];
        let sum = scale_dot(tier, row, k_t, decay);
        let delta = (v_j - sum) * beta;
        *out_j = mad_dot(tier, row, k_t, delta, q_t) * scale;
    }
}

/// Fork-faithful token update: dots, then mads, then output dots.
///
/// The `S *= decay` (or per-channel `S *= exp(g)`) pass has already been
/// applied by the caller, exactly as `gated_delta_net_cpu.cpp.txt:126-139`
/// applies it before the three `j` loops.
#[allow(
    clippy::too_many_arguments,
    reason = "token kernel: four activation slices, delta scratch, two gates, scale and tier"
)]
#[inline]
fn gdn_token_three_pass(
    slab: &mut [f32],
    q_t: &[f32],
    k_t: &[f32],
    v_t: &[f32],
    out_t: &mut [f32],
    delta: &mut [f32],
    beta: f32,
    scale: f32,
    tier: GdnTier,
) {
    let hk = k_t.len();
    for (j, (v_j, delta_j)) in v_t.iter().zip(delta.iter_mut()).enumerate() {
        let sum = dot(tier, &slab[j * hk..(j + 1) * hk], k_t);
        *delta_j = (v_j - sum) * beta;
    }
    for (j, delta_j) in delta.iter().enumerate().take(v_t.len()) {
        mad_in_place(tier, &mut slab[j * hk..(j + 1) * hk], k_t, *delta_j);
    }
    for (j, out_j) in out_t.iter_mut().enumerate().take(v_t.len()) {
        *out_j = dot(tier, &slab[j * hk..(j + 1) * hk], q_t) * scale;
    }
}

// ─── Row kernels: dispatch ───────────────────────────────────────

#[inline]
fn dot(tier: GdnTier, a: &[f32], b: &[f32]) -> f32 {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; the kernel reads only
        // `a.len().min(b.len())` elements of each slice.
        GdnTier::Neon => unsafe { dot_neon(a, b) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { dot_avx2(a, b) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { dot_avx512(a, b) },
        _ => dot_scalar(a, b),
    }
}

#[inline]
fn scale_dot(tier: GdnTier, row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; bounded by the shorter slice.
        GdnTier::Neon => unsafe { scale_dot_neon(row, k, decay) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { scale_dot_avx2(row, k, decay) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { scale_dot_avx512(row, k, decay) },
        _ => scale_dot_scalar(row, k, decay),
    }
}

#[inline]
fn mad_dot(tier: GdnTier, row: &mut [f32], k: &[f32], delta: f32, q: &[f32]) -> f32 {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; bounded by the shortest slice.
        GdnTier::Neon => unsafe { mad_dot_neon(row, k, delta, q) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { mad_dot_avx2(row, k, delta, q) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { mad_dot_avx512(row, k, delta, q) },
        _ => mad_dot_scalar(row, k, delta, q),
    }
}

#[inline]
fn scale_in_place(tier: GdnTier, xs: &mut [f32], decay: f32) {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; bounded by `xs.len()`.
        GdnTier::Neon => unsafe { scale_in_place_neon(xs, decay) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { scale_in_place_avx2(xs, decay) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { scale_in_place_avx512(xs, decay) },
        _ => scale_in_place_scalar(xs, decay),
    }
}

#[inline]
fn mul_in_place(tier: GdnTier, row: &mut [f32], g: &[f32]) {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; bounded by the shorter slice.
        GdnTier::Neon => unsafe { mul_in_place_neon(row, g) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { mul_in_place_avx2(row, g) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { mul_in_place_avx512(row, g) },
        _ => mul_in_place_scalar(row, g),
    }
}

#[inline]
fn mad_in_place(tier: GdnTier, row: &mut [f32], k: &[f32], delta: f32) {
    match tier {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; bounded by the shorter slice.
        GdnTier::Neon => unsafe { mad_in_place_neon(row, k, delta) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX2+FMA are present.
        GdnTier::Avx2 => unsafe { mad_in_place_avx2(row, k, delta) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `GdnTier::resolve` proved AVX-512F is present.
        GdnTier::Avx512 => unsafe { mad_in_place_avx512(row, k, delta) },
        _ => mad_in_place_scalar(row, k, delta),
    }
}

// ─── Row kernels: scalar ─────────────────────────────────────────

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline]
fn scale_dot_scalar(row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    let mut sum = 0.0f32;
    for (r, k_i) in row.iter_mut().zip(k.iter()) {
        *r *= decay;
        sum += *r * k_i;
    }
    sum
}

#[inline]
fn mad_dot_scalar(row: &mut [f32], k: &[f32], delta: f32, q: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for ((r, k_i), q_i) in row.iter_mut().zip(k.iter()).zip(q.iter()) {
        *r += k_i * delta;
        sum += *r * q_i;
    }
    sum
}

#[inline]
fn scale_in_place_scalar(xs: &mut [f32], decay: f32) {
    for x in xs.iter_mut() {
        *x *= decay;
    }
}

#[inline]
fn mul_in_place_scalar(row: &mut [f32], g: &[f32]) {
    for (r, g_i) in row.iter_mut().zip(g.iter()) {
        *r *= g_i;
    }
}

#[inline]
fn mad_in_place_scalar(row: &mut [f32], k: &[f32], delta: f32) {
    for (r, k_i) in row.iter_mut().zip(k.iter()) {
        *r += k_i * delta;
    }
}

// ─── Row kernels: NEON ───────────────────────────────────────────

/// # Safety
///
/// Reads `a.len().min(b.len())` floats from both pointers; NEON is baseline on
/// aarch64 so no feature detection is required.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 16 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 4;
            *acc_l = vfmaq_f32(*acc_l, vld1q_f32(pa.add(off)), vld1q_f32(pb.add(off)));
        }
        i += 16;
    }
    while i + 4 <= n {
        acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(acc[0], acc[1]),
        vaddq_f32(acc[2], acc[3]),
    ));
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// Reads/writes `row.len().min(k.len())` floats; see [`dot_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scale_dot_neon(row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = vdupq_n_f32(decay);
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 16 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 4;
            let scaled = vmulq_f32(vld1q_f32(pr.add(off)), d);
            vst1q_f32(pr.add(off), scaled);
            *acc_l = vfmaq_f32(*acc_l, scaled, vld1q_f32(pk.add(off)));
        }
        i += 16;
    }
    while i + 4 <= n {
        let scaled = vmulq_f32(vld1q_f32(pr.add(i)), d);
        vst1q_f32(pr.add(i), scaled);
        acc[0] = vfmaq_f32(acc[0], scaled, vld1q_f32(pk.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(acc[0], acc[1]),
        vaddq_f32(acc[2], acc[3]),
    ));
    while i < n {
        let scaled = *pr.add(i) * decay;
        *pr.add(i) = scaled;
        sum += scaled * *pk.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// Reads/writes `row.len().min(k.len()).min(q.len())` floats; see [`dot_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mad_dot_neon(row: &mut [f32], k: &[f32], delta: f32, q: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = row.len().min(k.len()).min(q.len());
    let (pr, pk, pq) = (row.as_mut_ptr(), k.as_ptr(), q.as_ptr());
    let d = vdupq_n_f32(delta);
    let mut acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0;
    while i + 16 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 4;
            let updated = vfmaq_f32(vld1q_f32(pr.add(off)), vld1q_f32(pk.add(off)), d);
            vst1q_f32(pr.add(off), updated);
            *acc_l = vfmaq_f32(*acc_l, updated, vld1q_f32(pq.add(off)));
        }
        i += 16;
    }
    while i + 4 <= n {
        let updated = vfmaq_f32(vld1q_f32(pr.add(i)), vld1q_f32(pk.add(i)), d);
        vst1q_f32(pr.add(i), updated);
        acc[0] = vfmaq_f32(acc[0], updated, vld1q_f32(pq.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(
        vaddq_f32(acc[0], acc[1]),
        vaddq_f32(acc[2], acc[3]),
    ));
    while i < n {
        let updated = (*pk.add(i)).mul_add(delta, *pr.add(i));
        *pr.add(i) = updated;
        sum += updated * *pq.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// Writes `xs.len()` floats; see [`dot_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scale_in_place_neon(xs: &mut [f32], decay: f32) {
    use std::arch::aarch64::*;
    let n = xs.len();
    let p = xs.as_mut_ptr();
    let d = vdupq_n_f32(decay);
    let mut i = 0;
    while i + 16 <= n {
        for lane in 0..4 {
            let off = i + lane * 4;
            vst1q_f32(p.add(off), vmulq_f32(vld1q_f32(p.add(off)), d));
        }
        i += 16;
    }
    while i + 4 <= n {
        vst1q_f32(p.add(i), vmulq_f32(vld1q_f32(p.add(i)), d));
        i += 4;
    }
    while i < n {
        *p.add(i) *= decay;
        i += 1;
    }
}

/// # Safety
///
/// Reads/writes `row.len().min(g.len())` floats; see [`dot_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mul_in_place_neon(row: &mut [f32], g: &[f32]) {
    use std::arch::aarch64::*;
    let n = row.len().min(g.len());
    let (pr, pg) = (row.as_mut_ptr(), g.as_ptr());
    let mut i = 0;
    while i + 4 <= n {
        vst1q_f32(
            pr.add(i),
            vmulq_f32(vld1q_f32(pr.add(i)), vld1q_f32(pg.add(i))),
        );
        i += 4;
    }
    while i < n {
        *pr.add(i) *= *pg.add(i);
        i += 1;
    }
}

/// # Safety
///
/// Reads/writes `row.len().min(k.len())` floats; see [`dot_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mad_in_place_neon(row: &mut [f32], k: &[f32], delta: f32) {
    use std::arch::aarch64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = vdupq_n_f32(delta);
    let mut i = 0;
    while i + 16 <= n {
        for lane in 0..4 {
            let off = i + lane * 4;
            vst1q_f32(
                pr.add(off),
                vfmaq_f32(vld1q_f32(pr.add(off)), vld1q_f32(pk.add(off)), d),
            );
        }
        i += 16;
    }
    while i + 4 <= n {
        vst1q_f32(
            pr.add(i),
            vfmaq_f32(vld1q_f32(pr.add(i)), vld1q_f32(pk.add(i)), d),
        );
        i += 4;
    }
    while i < n {
        *pr.add(i) = (*pk.add(i)).mul_add(delta, *pr.add(i));
        i += 1;
    }
}

// ─── Row kernels: AVX2 ───────────────────────────────────────────

/// # Safety
///
/// Caller must have proven AVX2 and FMA are available
/// ([`GdnTier::resolve`]); reads `a.len().min(b.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 8;
            *acc_l = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(off)),
                _mm256_loadu_ps(pb.add(off)),
                *acc_l,
            );
        }
        i += 32;
    }
    while i + 8 <= n {
        acc[0] = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i)),
            _mm256_loadu_ps(pb.add(i)),
            acc[0],
        );
        i += 8;
    }
    let mut sum = hsum_avx2(_mm256_add_ps(
        _mm256_add_ps(acc[0], acc[1]),
        _mm256_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx2`]; writes `row.len().min(k.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn scale_dot_avx2(row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = _mm256_set1_ps(decay);
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 8;
            let scaled = _mm256_mul_ps(_mm256_loadu_ps(pr.add(off)), d);
            _mm256_storeu_ps(pr.add(off), scaled);
            *acc_l = _mm256_fmadd_ps(scaled, _mm256_loadu_ps(pk.add(off)), *acc_l);
        }
        i += 32;
    }
    while i + 8 <= n {
        let scaled = _mm256_mul_ps(_mm256_loadu_ps(pr.add(i)), d);
        _mm256_storeu_ps(pr.add(i), scaled);
        acc[0] = _mm256_fmadd_ps(scaled, _mm256_loadu_ps(pk.add(i)), acc[0]);
        i += 8;
    }
    let mut sum = hsum_avx2(_mm256_add_ps(
        _mm256_add_ps(acc[0], acc[1]),
        _mm256_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        let scaled = *pr.add(i) * decay;
        *pr.add(i) = scaled;
        sum += scaled * *pk.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx2`]; writes `row.len().min(k.len()).min(q.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn mad_dot_avx2(row: &mut [f32], k: &[f32], delta: f32, q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len()).min(q.len());
    let (pr, pk, pq) = (row.as_mut_ptr(), k.as_ptr(), q.as_ptr());
    let d = _mm256_set1_ps(delta);
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 8;
            let updated = _mm256_fmadd_ps(
                _mm256_loadu_ps(pk.add(off)),
                d,
                _mm256_loadu_ps(pr.add(off)),
            );
            _mm256_storeu_ps(pr.add(off), updated);
            *acc_l = _mm256_fmadd_ps(updated, _mm256_loadu_ps(pq.add(off)), *acc_l);
        }
        i += 32;
    }
    while i + 8 <= n {
        let updated = _mm256_fmadd_ps(_mm256_loadu_ps(pk.add(i)), d, _mm256_loadu_ps(pr.add(i)));
        _mm256_storeu_ps(pr.add(i), updated);
        acc[0] = _mm256_fmadd_ps(updated, _mm256_loadu_ps(pq.add(i)), acc[0]);
        i += 8;
    }
    let mut sum = hsum_avx2(_mm256_add_ps(
        _mm256_add_ps(acc[0], acc[1]),
        _mm256_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        let updated = (*pk.add(i)).mul_add(delta, *pr.add(i));
        *pr.add(i) = updated;
        sum += updated * *pq.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx2`]; writes `xs.len()` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn scale_in_place_avx2(xs: &mut [f32], decay: f32) {
    use std::arch::x86_64::*;
    let n = xs.len();
    let p = xs.as_mut_ptr();
    let d = _mm256_set1_ps(decay);
    let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), d));
        i += 8;
    }
    while i < n {
        *p.add(i) *= decay;
        i += 1;
    }
}

/// # Safety
///
/// As [`dot_avx2`]; writes `row.len().min(g.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn mul_in_place_avx2(row: &mut [f32], g: &[f32]) {
    use std::arch::x86_64::*;
    let n = row.len().min(g.len());
    let (pr, pg) = (row.as_mut_ptr(), g.as_ptr());
    let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(
            pr.add(i),
            _mm256_mul_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pg.add(i))),
        );
        i += 8;
    }
    while i < n {
        *pr.add(i) *= *pg.add(i);
        i += 1;
    }
}

/// # Safety
///
/// As [`dot_avx2`]; writes `row.len().min(k.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn mad_in_place_avx2(row: &mut [f32], k: &[f32], delta: f32) {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = _mm256_set1_ps(delta);
    let mut i = 0;
    while i + 32 <= n {
        for lane in 0..4 {
            let off = i + lane * 8;
            _mm256_storeu_ps(
                pr.add(off),
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(pk.add(off)),
                    d,
                    _mm256_loadu_ps(pr.add(off)),
                ),
            );
        }
        i += 32;
    }
    while i + 8 <= n {
        _mm256_storeu_ps(
            pr.add(i),
            _mm256_fmadd_ps(_mm256_loadu_ps(pk.add(i)), d, _mm256_loadu_ps(pr.add(i))),
        );
        i += 8;
    }
    while i < n {
        *pr.add(i) = (*pk.add(i)).mul_add(delta, *pr.add(i));
        i += 1;
    }
}

/// Horizontal sum of a 256-bit lane, same reduction order as `simd_avx2.rs`.
///
/// # Safety
///
/// Caller must have proven AVX is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

// ─── Row kernels: AVX-512 ────────────────────────────────────────

/// # Safety
///
/// Caller must have proven AVX-512F is available ([`GdnTier::resolve`]); reads
/// `a.len().min(b.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 16;
            *acc_l = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(off)),
                _mm512_loadu_ps(pb.add(off)),
                *acc_l,
            );
        }
        i += 64;
    }
    while i + 16 <= n {
        acc[0] = _mm512_fmadd_ps(
            _mm512_loadu_ps(pa.add(i)),
            _mm512_loadu_ps(pb.add(i)),
            acc[0],
        );
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc[0], acc[1]),
        _mm512_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx512`]; writes `row.len().min(k.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn scale_dot_avx512(row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = _mm512_set1_ps(decay);
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 16;
            let scaled = _mm512_mul_ps(_mm512_loadu_ps(pr.add(off)), d);
            _mm512_storeu_ps(pr.add(off), scaled);
            *acc_l = _mm512_fmadd_ps(scaled, _mm512_loadu_ps(pk.add(off)), *acc_l);
        }
        i += 64;
    }
    while i + 16 <= n {
        let scaled = _mm512_mul_ps(_mm512_loadu_ps(pr.add(i)), d);
        _mm512_storeu_ps(pr.add(i), scaled);
        acc[0] = _mm512_fmadd_ps(scaled, _mm512_loadu_ps(pk.add(i)), acc[0]);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc[0], acc[1]),
        _mm512_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        let scaled = *pr.add(i) * decay;
        *pr.add(i) = scaled;
        sum += scaled * *pk.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx512`]; writes `row.len().min(k.len()).min(q.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mad_dot_avx512(row: &mut [f32], k: &[f32], delta: f32, q: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len()).min(q.len());
    let (pr, pk, pq) = (row.as_mut_ptr(), k.as_ptr(), q.as_ptr());
    let d = _mm512_set1_ps(delta);
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (lane, acc_l) in acc.iter_mut().enumerate() {
            let off = i + lane * 16;
            let updated = _mm512_fmadd_ps(
                _mm512_loadu_ps(pk.add(off)),
                d,
                _mm512_loadu_ps(pr.add(off)),
            );
            _mm512_storeu_ps(pr.add(off), updated);
            *acc_l = _mm512_fmadd_ps(updated, _mm512_loadu_ps(pq.add(off)), *acc_l);
        }
        i += 64;
    }
    while i + 16 <= n {
        let updated = _mm512_fmadd_ps(_mm512_loadu_ps(pk.add(i)), d, _mm512_loadu_ps(pr.add(i)));
        _mm512_storeu_ps(pr.add(i), updated);
        acc[0] = _mm512_fmadd_ps(updated, _mm512_loadu_ps(pq.add(i)), acc[0]);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(
        _mm512_add_ps(acc[0], acc[1]),
        _mm512_add_ps(acc[2], acc[3]),
    ));
    while i < n {
        let updated = (*pk.add(i)).mul_add(delta, *pr.add(i));
        *pr.add(i) = updated;
        sum += updated * *pq.add(i);
        i += 1;
    }
    sum
}

/// # Safety
///
/// As [`dot_avx512`]; writes `xs.len()` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn scale_in_place_avx512(xs: &mut [f32], decay: f32) {
    use std::arch::x86_64::*;
    let n = xs.len();
    let p = xs.as_mut_ptr();
    let d = _mm512_set1_ps(decay);
    let mut i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(p.add(i), _mm512_mul_ps(_mm512_loadu_ps(p.add(i)), d));
        i += 16;
    }
    while i < n {
        *p.add(i) *= decay;
        i += 1;
    }
}

/// # Safety
///
/// As [`dot_avx512`]; writes `row.len().min(g.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mul_in_place_avx512(row: &mut [f32], g: &[f32]) {
    use std::arch::x86_64::*;
    let n = row.len().min(g.len());
    let (pr, pg) = (row.as_mut_ptr(), g.as_ptr());
    let mut i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(
            pr.add(i),
            _mm512_mul_ps(_mm512_loadu_ps(pr.add(i)), _mm512_loadu_ps(pg.add(i))),
        );
        i += 16;
    }
    while i < n {
        *pr.add(i) *= *pg.add(i);
        i += 1;
    }
}

/// # Safety
///
/// As [`dot_avx512`]; writes `row.len().min(k.len())` floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mad_in_place_avx512(row: &mut [f32], k: &[f32], delta: f32) {
    use std::arch::x86_64::*;
    let n = row.len().min(k.len());
    let (pr, pk) = (row.as_mut_ptr(), k.as_ptr());
    let d = _mm512_set1_ps(delta);
    let mut i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(
            pr.add(i),
            _mm512_fmadd_ps(_mm512_loadu_ps(pk.add(i)), d, _mm512_loadu_ps(pr.add(i))),
        );
        i += 16;
    }
    while i < n {
        *pr.add(i) = (*pk.add(i)).mul_add(delta, *pr.add(i));
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The detected tier must be one this build can actually execute.
    #[test]
    fn detected_tier_resolves_to_itself() {
        let tier = GdnTier::detect();
        assert_eq!(tier, tier.resolve());
    }

    /// A tier the current architecture does not have degrades to scalar
    /// instead of executing an illegal instruction.
    #[test]
    fn foreign_tiers_degrade_to_scalar() {
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(GdnTier::Avx2.resolve(), GdnTier::Scalar);
            assert_eq!(GdnTier::Avx512.resolve(), GdnTier::Scalar);
            assert_eq!(GdnTier::Neon.resolve(), GdnTier::Neon);
        }
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(GdnTier::Neon.resolve(), GdnTier::Scalar);
        }
        assert_eq!(GdnTier::Scalar.resolve(), GdnTier::Scalar);
    }

    /// Max absolute difference between two equal-length slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Every row primitive must agree with the scalar reference on the
    /// detected tier, including the non-multiple-of-vector-width tails.
    ///
    /// Element-wise results that are pure multiplies (`scale_in_place`,
    /// `mul_in_place`) are bitwise equal across tiers; the `mad` forms are not,
    /// because the vector tiers fuse the multiply-add and the scalar tier does
    /// not, so they are compared to 1e-6.
    #[test]
    fn row_primitives_match_scalar_on_every_length() {
        let tier = GdnTier::detect();
        for n in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128] {
            let row: Vec<f32> = (0..n).map(|i| 0.5 - (i % 7) as f32 * 0.1).collect();
            let k: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
            let q: Vec<f32> = (0..n).map(|i| 0.125 * (i % 9) as f32 - 0.5).collect();

            let mut a = row.clone();
            let mut b = row.clone();
            let sa = scale_dot(tier, &mut a, &k, 0.75);
            let sb = scale_dot_scalar(&mut b, &k, 0.75);
            assert!(
                (sa - sb).abs() <= 1e-6 * (1.0 + sb.abs()),
                "scale_dot n={n}"
            );
            assert_eq!(a, b, "scale_dot state n={n}");

            let ma = mad_dot(tier, &mut a, &k, 0.3, &q);
            let mb = mad_dot_scalar(&mut b, &k, 0.3, &q);
            assert!((ma - mb).abs() <= 1e-6 * (1.0 + mb.abs()), "mad_dot n={n}");
            assert!(max_abs_diff(&a, &b) <= 1e-6, "mad_dot state n={n}");

            let da = dot(tier, &a, &q);
            let db = dot_scalar(&a, &q);
            assert!((da - db).abs() <= 1e-6 * (1.0 + db.abs()), "dot n={n}");

            let mut c = row.clone();
            let mut d = row.clone();
            scale_in_place(tier, &mut c, 0.5);
            scale_in_place_scalar(&mut d, 0.5);
            assert_eq!(c, d, "scale_in_place n={n}");

            mul_in_place(tier, &mut c, &k);
            mul_in_place_scalar(&mut d, &k);
            assert_eq!(c, d, "mul_in_place n={n}");

            mad_in_place(tier, &mut c, &k, 0.25);
            mad_in_place_scalar(&mut d, &k, 0.25);
            assert!(max_abs_diff(&c, &d) <= 1e-6, "mad_in_place n={n}");
        }
    }

    /// `scale_dot` must leave exactly what `scale_in_place` leaves, and
    /// `mad_dot` exactly what `mad_in_place` leaves — this identity is what
    /// makes the fused and three-pass recurrences bitwise equal.
    #[test]
    fn fused_primitives_are_bitwise_split_primitives() {
        let tier = GdnTier::detect();
        for n in [8usize, 16, 33, 128] {
            let row: Vec<f32> = (0..n).map(|i| 0.75 - (i % 11) as f32 * 0.07).collect();
            let k: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.03 - 0.2).collect();
            let q: Vec<f32> = (0..n).map(|i| 0.4 - (i % 3) as f32 * 0.31).collect();

            let mut fused = row.clone();
            let fused_sum = scale_dot(tier, &mut fused, &k, 0.9375);

            let mut split = row.clone();
            scale_in_place(tier, &mut split, 0.9375);
            let split_sum = dot(tier, &split, &k);

            assert_eq!(fused, split, "scaled row n={n}");
            assert_eq!(fused_sum.to_bits(), split_sum.to_bits(), "scaled sum n={n}");

            let fused_out = mad_dot(tier, &mut fused, &k, 0.125, &q);
            mad_in_place(tier, &mut split, &k, 0.125);
            let split_out = dot(tier, &split, &q);

            assert_eq!(fused, split, "madded row n={n}");
            assert_eq!(fused_out.to_bits(), split_out.to_bits(), "madded sum n={n}");
        }
    }
}
