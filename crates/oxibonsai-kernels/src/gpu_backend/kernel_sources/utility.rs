//! Utility Metal kernels (activations, normalization, element-wise ops).
//!
//! Contains RMSNorm, SwiGLU, softmax, ReLU, SiLU, residual-add,
//! matrix-vector multiply, and argmax kernels.

/// Numerically-stable softmax.
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → size (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& size        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float max_val = input[0];
    for (uint i = 1u; i < size; i++) {
        max_val = max(max_val, input[i]);
    }

    float my_exp = exp(input[gid] - max_val);

    float sum_exp = 0.0f;
    for (uint i = 0u; i < size; i++) {
        sum_exp += exp(input[i] - max_val);
    }

    output[gid] = (sum_exp > 0.0f) ? (my_exp / sum_exp) : (1.0f / float(size));
}
"#;

/// Element-wise ReLU: y = max(0, x).
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = max(0.0f, input[gid]);
}
"#;

/// RMSNorm: y_i = x_i / sqrt(mean(x²) + eps) * weight_i
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"alpha"` → eps (3), `"n"` → count (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant float& eps        [[buffer(3)]],
    constant uint& n           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (uint i = 0u; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = rsqrt(sum_sq / float(n) + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;

/// SiLU (Sigmoid Linear Unit): y = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Buffers: `"x"` → input (0), `"result"` → output (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SILU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}
"#;

/// SwiGLU fused activation: `output[i] = silu(gate[i]) * up[i]`
///
/// Buffers: `"x"` → gate (0), `"y"` → up (1), `"result"` → output (2)
/// Scalars: `"n"` → count (3)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SWIGLU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void swiglu(
    device const float* gate    [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& n            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * up[gid];
}
"#;

/// Residual add in-place: `a[i] += b[i]`
///
/// Buffer `"x"` is read-write (both input and output).
///
/// Buffers: `"x"` → a (0, read-write), `"y"` → b (1)
/// Scalars: `"n"` → count (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RESIDUAL_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void residual_add(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] += b[gid];
}
"#;

/// Fused SwiGLU reading from concatenated [gate, up] buffer.
///
/// `gate = buffer[0..n]`, `up = buffer[n..2n]`
/// `output[i] = silu(gate[i]) * up[i]`
///
/// Buffers: `"x"` → gate_up (0), `"result"` → output (1)
/// Scalars: `"n"` → n (2)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SWIGLU_FUSED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_fused(
    device const float* gate_up  [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate_up[gid];
    float u = gate_up[n + gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * u;
}
"#;

/// RMSNorm with weight (weighted variant for LLM layers).
///
/// Computes `output[i] = (input[i] / sqrt(mean(input²) + eps)) * weight[i]`.
/// Each thread redundantly computes the full sum-of-squares — correct for
/// typical hidden sizes (e.g. 4096) and avoids shared-memory reduction.
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"n"` → count (3), `"alpha"` → eps (4)
///
/// RMSNorm with weight vector, for use from scirs2-core dispatch.
///
/// **Scalar binding order**: scirs2-core binds scalars in a fixed order:
/// `"alpha"`, `"beta"`, `"n"`, `"m"`, `"k"` — only those that are set.
/// Since we `set_f32("alpha", eps)` and `set_u32("n", h)`, the binding is:
///   buffer(3) = alpha (eps, f32), buffer(4) = n (h, u32).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM_WEIGHTED: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_weighted(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    float sum_sq = 0.0f;
    for (uint i = 0u; i < n; i++) {
        float v = input[i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(n) + eps);

    output[gid] = input[gid] * rms * weight[gid];
}
"#;

/// Optimized RMSNorm with parallel threadgroup reduction.
///
/// Fixes the O(n²) issue in V1 where every thread redundantly computes
/// the full sum-of-squares. V2 uses cooperative threadgroup reduction:
///
/// 1. Each of 256 threads computes a partial sum of `x²` over strided elements
/// 2. Tree reduction in threadgroup shared memory to get total sum
/// 3. All threads compute `rms = rsqrt(sum/n + eps)` from shared result
/// 4. All threads apply `output[i] = input[i] * rms * weight[i]`
///
/// Complexity: O(n) total work instead of O(n²).
///
/// Buffers: `"x"` → input (0), `"y"` → weight (1), `"result"` → output (2)
/// Scalars: `"alpha"` → eps (3), `"n"` → count (4)
///
/// Dispatch: `[1, 1, 1]` threadgroups, `[256, 1, 1]` threads
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_RMSNORM_WEIGHTED_V2: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_weighted_v2(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[256];

    // Step 1: Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = input[i];
        partial_sum = fma(v, v, partial_sum);
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Tree reduction in shared memory
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Compute rms scaling factor (all threads read same value)
    float rms = rsqrt(shared_sum[0] / float(n) + eps);

    // Step 4: Apply scaling to output
    for (uint i = tid; i < n; i += tg_size) {
        output[i] = input[i] * rms * weight[i];
    }
}
"#;

/// FP32 matrix-vector multiply: y = A * x
///
/// Buffers: `"x"` → matrix a (0), `"y"` → vector x (1), `"result"` → output (2)
/// Scalars: `"n"` → m (3), `"k"` → k (4)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_MATVEC_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f32(
    device const float* a      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& m           [[buffer(3)]],
    constant uint& k           [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= m) return;

    float sum = 0.0f;
    uint row_offset = gid * k;
    for (uint j = 0u; j < k; j++) {
        sum += a[row_offset + j] * x[j];
    }
    output[gid] = sum;
}
"#;

/// GPU argmax — finds the index of the **first** maximum in a float array.
///
/// Uses a single threadgroup with 1024 threads (sufficient for vocab ≤ ~500K).
/// Each thread scans every 1024th element, then a tree reduction finds the
/// global maximum's index.
///
/// # Tie-break contract (`perf-11` / `RT-22`)
///
/// On equal values this kernel returns the **smallest original index**, the
/// same rule as `llama.cpp`, the CPU sampler's `argmax_first` and the CUDA
/// `argmax_f32` twin in `cuda_kernels.rs`. That is a hard requirement: the
/// runtime's greedy paths mix CPU and GPU argmax across a single decode, and
/// the 27B temperature-0 parity gate against the PrismML fork would otherwise
/// surface a tie-break disagreement as a phantom kernel bug.
///
/// The original kernel reduced with a strict `>` only, which is a comparison
/// on the *reduction slot*, not on the original index: the surviving maximum
/// depended on the fold path, so it was neither global-first nor even
/// slot-minimal. Measured on an M3 before this fix, equal maxima at indices
/// 1000 and 2000 returned **2000**, and 144 of 320 randomized multi-way ties
/// returned a non-minimal index (`tests/gpu_argmax_tiebreak.rs`). Both
/// comparison sites now carry the index tie-break:
///
/// * the per-thread strided scan keeps `>` — since `i` ascends, that already
///   keeps the smallest index a single thread sees among equal values;
/// * every tree-reduction stage takes the partner slot when it is strictly
///   greater **or** equal-with-a-smaller-index.
///
/// An idle thread (`tid >= count`) seeds its slot with the out-of-range index
/// `count` so it loses every tie against a real element, instead of the `0`
/// it used to carry — which, under an index tie-break, would have won.
///
/// Buffers:
/// - buffer(0) = data    (f32, input values)
/// - buffer(1) = result  (uint32, output index — single element)
/// - buffer(2) = count   (uint32, scalar)
///
/// Dispatch: `[1, 1, 1]` threadgroups, `[1024, 1, 1]` threads. `tpg` is
/// assumed to be a power of two ≤ 1024 (the tree halves `tpg / 2`), which
/// every dispatch site (`metal_dispatch.rs::dispatch_argmax`,
/// `metal_prefill/functions.rs:434`, `:1263`) guarantees.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_ARGMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax(
    device const float* data    [[buffer(0)]],
    device uint* result         [[buffer(1)]],
    constant uint& count        [[buffer(2)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    threadgroup float shared_vals[1024];
    threadgroup uint shared_idxs[1024];

    float best_val = -INFINITY;
    // Out-of-range sentinel: a thread that scans nothing must lose every tie.
    uint best_idx = count;

    // `i` ascends, so strict `>` keeps the SMALLEST index this thread sees
    // among equal values (first-index rule, per-thread half).
    for (uint i = tid; i < count; i += tpg) {
        float v = data[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: strictly-greater value wins; on an exact tie the
    // smaller ORIGINAL index wins (first-index rule, cross-thread half).
    for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float ov = shared_vals[tid + stride];
            uint  oi = shared_idxs[tid + stride];
            float mv = shared_vals[tid];
            uint  mi = shared_idxs[tid];
            if (ov > mv || (ov == mv && oi < mi)) {
                shared_vals[tid] = ov;
                shared_idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        // Only reachable with the sentinel when count == 0 (nothing to scan).
        result[0] = (shared_idxs[0] < count) ? shared_idxs[0] : 0u;
    }
}
"#;

/// GPU partial top-k — the `k` highest `(id, value)` pairs of a float array,
/// sorted descending, ties broken the **same way `argmax` above is**
/// (smallest original index wins).
///
/// # Why this exists (`perf-11`, sampled decode)
///
/// Every sampled (non-greedy) decode step needs `top_k`/`top_p` over the
/// logits row, which today means downloading the **entire** row to the CPU —
/// 993 KB/token at Bonsai 2's 248 320-token vocabulary — purely so the CPU
/// sampler can find perhaps the top 20-100 candidates. This kernel runs that
/// reduction on the GPU and returns only the `k` winners, so a sampled
/// request pays the same "download a handful of floats" cost the greedy
/// (`argmax`) path already enjoys.
///
/// # Algorithm
///
/// This generalises `argmax`'s single-threadgroup / tree-reduction shape to
/// `k` iterations of the *same* reduction: iteration `i` finds the
/// arg-max over every element **not** already selected by iterations
/// `0..i`, using the identical tie-break rule (`argmax`'s doc above spells
/// out why that rule matters — mixed CPU/GPU decode and the PrismML parity
/// gate both depend on `argmax_first` semantics), and records the winner in
/// threadgroup-shared `picked[]` before the next iteration excludes it. This
/// is `O(k)` reductions of the same shape `argmax` already does once, so it
/// reuses the same 1024-thread / power-of-two-stride tree and pays no extra
/// GPU→CPU round trips — one dispatch produces all `k` pairs.
///
/// `k` is capped at `MAX_TOPK_F32` (256, comfortably above
/// any realistic `top_k` sampling value — the recommended Bonsai 2 sampling
/// config uses `top_k = 20`) by both sides: the Rust dispatcher
/// (`metal_dispatch.rs::dispatch_topk_f32`) rejects a larger `k` with a typed
/// error instead of silently truncating the caller's request, and the kernel
/// itself clamps to the same bound so the fixed-size `picked[256]`
/// threadgroup array can never be written out of bounds regardless of what
/// value reaches it.
///
/// If `k` exceeds `count` (fewer real elements than requested), the
/// exhausted slots are padded with id `0` / value `-INFINITY` rather than
/// repeating an already-picked entry or reading out of bounds.
///
/// # `-INFINITY` contract (read before wiring a consumer)
///
/// The exhausted-slot sentinel above is **not** guaranteed to be
/// distinguishable from a real logit: grammar/constrained decoding writes an
/// actual `-INFINITY` into masked-out logits before this kernel ever sees
/// them, and the inner scan's tie-break is a strict `v > best_val` starting
/// from `best_val = -INFINITY` — so an element whose value is exactly
/// `-INFINITY` can never win that comparison (`-inf > -inf` is false) and is
/// therefore never selected, exactly as if it did not exist. Concretely,
/// `data = [3.0, -inf]`, `k = 2` returns `out_ids = [0, 0]` /
/// `out_vals = [3.0, -inf]`, not `out_ids = [0, 1]` — index 1 is skipped, not
/// picked-then-reported.
///
/// A slot in `out_vals`/`out_ids` reading `-INFINITY` is therefore ambiguous
/// BY DESIGN between "no real element was left to fill this slot" and "a
/// real, grammar-masked `-INFINITY` logit was present but could never be
/// chosen". Both cases are harmless to a caller that follows this rule:
/// **treat `out_vals[i] == -INFINITY` as "no candidate" and ignore the
/// paired `out_ids[i]`**, never sample or rank on it. (The consumer that must
/// honour this contract is the sampled top-k route in `resident_logits.rs`.)
///
/// Buffers:
/// - buffer(0) = data     (f32, input values)
/// - buffer(1) = out_ids  (uint32, `k` winning indices, descending by value)
/// - buffer(2) = out_vals (f32, `k` winning values, same order)
/// - buffer(3) = count    (uint32, scalar — length of `data`)
/// - buffer(4) = k        (uint32, scalar — number of winners requested)
///
/// Dispatch: `[1, 1, 1]` threadgroups, `[1024, 1, 1]` threads — identical
/// geometry to `argmax`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_TOPK_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void topk_f32(
    device const float* data    [[buffer(0)]],
    device uint*  out_ids       [[buffer(1)]],
    device float* out_vals      [[buffer(2)]],
    constant uint& count        [[buffer(3)]],
    constant uint& k            [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    threadgroup float shared_vals[1024];
    threadgroup uint shared_idxs[1024];
    // Fixed-size regardless of `k`: `kk` below never exceeds this bound, so
    // `picked[iter]` (iter < kk) can never write out of bounds.
    threadgroup uint picked[256];

    const uint kk = min(k, 256u);

    for (uint iter = 0u; iter < kk; iter++) {
        float best_val = -INFINITY;
        // Out-of-range sentinel: a thread that finds nothing (every element
        // it sees is already picked) must lose every tie, exactly as in
        // `argmax`.
        uint best_idx = count;

        for (uint i = tid; i < count; i += tpg) {
            bool already_picked = false;
            for (uint p = 0u; p < iter; p++) {
                if (picked[p] == i) {
                    already_picked = true;
                    break;
                }
            }
            if (already_picked) {
                continue;
            }
            float v = data[i];
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }

        shared_vals[tid] = best_val;
        shared_idxs[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction: strictly-greater value wins; on an exact tie the
        // smaller ORIGINAL index wins — identical rule to `argmax`.
        for (uint stride = tpg / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float ov = shared_vals[tid + stride];
                uint  oi = shared_idxs[tid + stride];
                float mv = shared_vals[tid];
                uint  mi = shared_idxs[tid];
                if (ov > mv || (ov == mv && oi < mi)) {
                    shared_vals[tid] = ov;
                    shared_idxs[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            uint winner = shared_idxs[0];
            bool valid = winner < count;
            // A sentinel here (rather than repeating `winner`) keeps a
            // padding iteration from ever comparing equal to a real index in
            // a later iteration's `already_picked` scan.
            picked[iter] = valid ? winner : count;
            out_ids[iter] = valid ? winner : 0u;
            out_vals[iter] = valid ? shared_vals[0] : -INFINITY;
        }
        // Publishes this iteration's `picked[iter]` (and `out_*[iter]`)
        // before any thread starts the next iteration's exclusion scan.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Tests — host-only kernel source string assertions
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn topk_f32_has_the_expected_entry_point_and_buffer_layout() {
        use super::MSL_TOPK_F32;
        assert!(MSL_TOPK_F32.contains("kernel void topk_f32"));
        assert!(MSL_TOPK_F32.contains("[[buffer(0)]]"));
        assert!(MSL_TOPK_F32.contains("[[buffer(4)]]"));
    }

    /// The 256 cap in the doc comment, the Rust-side `MAX_TOPK_F32` constant
    /// (`metal_dispatch.rs`) and the kernel's own fixed-size `picked[256]`
    /// threadgroup array must all agree, or a caller-supplied `k` within one
    /// side's bound but not the other's is either rejected wrongly or writes
    /// `picked[]` out of bounds.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn topk_f32_caps_k_at_the_documented_256_bound() {
        use super::MSL_TOPK_F32;
        assert!(MSL_TOPK_F32.contains("threadgroup uint picked[256]"));
        assert!(MSL_TOPK_F32.contains("min(k, 256u)"));
    }

    /// Same tie-break rule as `argmax` (`perf-11` / `RT-22`): on an exact
    /// value tie the smaller original index must win the reduction, in both
    /// the per-thread strided scan (`>` keeps the ascending, hence smallest,
    /// index) and the cross-thread tree reduction (explicit index
    /// tie-break). A regression to a naive strict `>` tree reduction here
    /// would reproduce the exact bug `gpu_argmax_tiebreak.rs` guards against
    /// for `argmax`, just for the sampled decode path instead of the greedy
    /// one.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn topk_f32_uses_the_first_index_tiebreak_rule() {
        use super::MSL_TOPK_F32;
        assert!(MSL_TOPK_F32.contains("ov > mv || (ov == mv && oi < mi)"));
        assert!(MSL_TOPK_F32.contains("if (v > best_val)"));
    }

    /// `k > count` must pad with a value no real logit can produce, not
    /// repeat an already-picked winner or read out of bounds.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn topk_f32_pads_exhausted_slots_with_a_sentinel() {
        use super::MSL_TOPK_F32;
        assert!(MSL_TOPK_F32.contains("valid ? winner : count"));
        assert!(MSL_TOPK_F32.contains("out_vals[iter] = valid ? shared_vals[0] : -INFINITY"));
    }

    /// Every iteration must exclude every prior winner, or a tied/duplicate
    /// value could be selected twice.
    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn topk_f32_excludes_previously_picked_indices_every_iteration() {
        use super::MSL_TOPK_F32;
        assert!(MSL_TOPK_F32.contains("if (picked[p] == i)"));
        assert!(MSL_TOPK_F32.contains("if (already_picked)"));
    }
}
