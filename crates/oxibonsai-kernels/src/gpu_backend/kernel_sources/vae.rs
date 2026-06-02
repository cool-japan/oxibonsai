//! MSL kernel sources for the **FLUX.2 VAE decoder** per-op f32 GPU primitives.
//!
//! These four element/structural kernels accelerate the pure-CPU VAE decode of
//! the FLUX.2 image model (the convs are ~60 % of the VAE FLOPs at 512×512, and
//! the CPU path is im2col-memory-bound). They are purely additive — no existing
//! kernel is touched — and each is parity-validated against the CPU reference in
//! `oxibonsai-image::vae` (`conv.rs`, `norm.rs`, `ops.rs`).
//!
//! All buffers are flat NCHW `[C, H, W]` f32 (batch 1), matching the CPU stack.
//!
//! | Kernel                  | Op | CPU reference |
//! |-------------------------|----|---------------|
//! | `im2col_f32`            | im2col patch extraction `[rows, kH·kW·C_in]` in `(kH,kW,C_in)` order | `conv.rs::build_im2col` |
//! | `groupnorm_f32`         | GroupNorm (32 groups, eps 1e-6, per-channel affine) | `norm.rs::forward_inplace` |
//! | `silu_f32`              | element-wise `x / (1 + exp(-x))` | `ops.rs::silu_inplace` / `math::silu` |
//! | `upsample_nearest_f32`  | nearest ×2 `[C,H,W] → [C,2H,2W]` | `ops.rs::upsample_nearest2x` |
//!
//! The `im2col_f32` output feeds the existing parity-clean `gemm_f32_simdgroup`
//! (the `(kH,kW,C_in)` element order is exactly the MLX conv-weight
//! `[C_out, kH, kW, C_in]` flattening, so the GEMM dot is correct with the
//! weight passed directly as `W = [N=C_out, K=kH·kW·C_in]`).

// ═══════════════════════════════════════════════════════════════════════════
// im2col (k≥3) — patch extraction in (kH, kW, C_in) order
// ═══════════════════════════════════════════════════════════════════════════

/// im2col patch extraction for a **tile of output rows**.
///
/// Produces `patches[local_row * patch_dim + j]` for output spatial rows
/// `[row_start, row_start + tile_rows)`, where `patch_dim = k·k·C_in` and the
/// element index `j` decomposes as `(kh·k + kw)·C_in + ci` — the **`(kh, kw,
/// ci)` order that matches the MLX conv weight `[C_out, kH, kW, C_in]`
/// flattening** (the critical correctness invariant, mirroring
/// `oxibonsai-image::vae::conv::build_im2col`). Zero padding: an out-of-range
/// (padded) source position contributes `0`.
///
/// Buffers:
/// - buffer(0) = `input`   (f32, NCHW `[C_in, H·W]` == `input[ci*H*W + ih*W + iw]`)
/// - buffer(1) = `patches` (f32, `[tile_rows, patch_dim]`, written)
/// - buffer(2) = `c_in`      (u32)
/// - buffer(3) = `h`         (u32)
/// - buffer(4) = `w`         (u32)
/// - buffer(5) = `k`         (u32, kernel edge; square)
/// - buffer(6) = `pad`       (u32)
/// - buffer(7) = `w_out`     (u32)
/// - buffer(8) = `row_start` (u32, first output spatial index of this tile)
/// - buffer(9) = `n_elems`   (u32, `tile_rows * patch_dim`, the grid bound)
///
/// Dispatch: `[ceil(n_elems / 256), 1, 1]` threadgroups, `[256, 1, 1]` threads
/// (one thread per patch element).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_IM2COL_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void im2col_f32(
    device const float* input    [[buffer(0)]],
    device       float* patches  [[buffer(1)]],
    constant uint&      c_in      [[buffer(2)]],
    constant uint&      h         [[buffer(3)]],
    constant uint&      w         [[buffer(4)]],
    constant uint&      k         [[buffer(5)]],
    constant uint&      pad       [[buffer(6)]],
    constant uint&      w_out     [[buffer(7)]],
    constant uint&      row_start [[buffer(8)]],
    constant uint&      n_elems   [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_elems) { return; }

    const uint patch_dim = k * k * c_in;
    const uint local_row = gid / patch_dim;     // 0..tile_rows-1
    const uint j         = gid % patch_dim;     // 0..patch_dim-1

    // Output spatial position (global, across the full H_out*W_out plane).
    const uint out_idx = row_start + local_row;
    const uint oh = out_idx / w_out;
    const uint ow = out_idx % w_out;

    // Decompose j = (kh*k + kw)*c_in + ci  →  (kh, kw, ci).
    const uint kc  = k * c_in;
    const uint kh  = j / kc;
    const uint rem = j % kc;
    const uint kw  = rem / c_in;
    const uint ci  = rem % c_in;

    // Padded source coordinates (same geometry as the CPU build_im2col).
    const uint ih_p = oh + kh;
    const uint iw_p = ow + kw;

    float v = 0.0f;
    if (ih_p >= pad && ih_p < h + pad && iw_p >= pad && iw_p < w + pad) {
        const uint ih = ih_p - pad;
        const uint iw = iw_p - pad;
        v = input[(ulong)ci * (ulong)h * (ulong)w + (ulong)ih * (ulong)w + (ulong)iw];
    }
    patches[gid] = v;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// GroupNorm — 32 groups, eps 1e-6, per-channel affine (one threadgroup/group)
// ═══════════════════════════════════════════════════════════════════════════

/// PyTorch-compatible GroupNorm, in place on NCHW `[C, H·W]`.
///
/// One **threadgroup per group** (`num_groups`, 32 in the VAE). Each group spans
/// `gs = C / num_groups` contiguous channels × `hw` spatial positions; the
/// per-group population mean / variance are reduced over all `gs·hw` elements,
/// then `(x - mean) / sqrt(var + eps)` is scaled by the per-channel affine
/// `weight[c]` / `bias[c]`.
///
/// **Numerics.** The CPU reference (`norm.rs`) accumulates the mean and variance
/// in `f64`. Metal has no `double`, so this kernel uses **Kahan-compensated f32
/// summation** within each thread's strided slice, followed by a plain f32 tree
/// reduction of the 256 partials. Over the VAE's group sizes (up to `gs·hw =
/// 3·512·512 ≈ 786 k` elements) this keeps the relative error of the sum /
/// sum-of-squares at ~`1e-6`–`1e-7`, so the normalized output matches the f64
/// reference to ≪ `1e-4` (the parity gate), and the end-to-end `vae_parity`
/// cosine stays ≥ 0.999. The variance is the classic two-pass form
/// (`mean` first, then `Σ(x-mean)²`) to mirror `norm.rs` exactly.
///
/// Buffers:
/// - buffer(0) = `x`      (f32, NCHW `[C, hw]`, read-write / in-place)
/// - buffer(1) = `weight` (f32, `[C]` per-channel scale)
/// - buffer(2) = `bias`   (f32, `[C]` per-channel shift)
/// - buffer(3) = `channels`   (u32, `C`)
/// - buffer(4) = `hw`         (u32, `H·W`)
/// - buffer(5) = `num_groups` (u32)
/// - buffer(6) = `eps`        (f32)
///
/// Dispatch: `[num_groups, 1, 1]` threadgroups, `[256, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_GROUPNORM_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint GN_THREADS = 256u;

kernel void groupnorm_f32(
    device       float* x          [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device const float* bias       [[buffer(2)]],
    constant uint&      channels   [[buffer(3)]],
    constant uint&      hw         [[buffer(4)]],
    constant uint&      num_groups [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    uint ggid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]])
{
    const uint g  = ggid;                       // group index (one threadgroup/group)
    if (g >= num_groups) { return; }
    const uint gs = channels / num_groups;      // channels per group
    const ulong group_elems = (ulong)gs * (ulong)hw;
    const ulong base = (ulong)(g * gs) * (ulong)hw;

    threadgroup float red[GN_THREADS];
    threadgroup float mean_sh;
    threadgroup float inv_std_sh;

    // ── Pass 1: Kahan-compensated partial sum over this thread's stride. ──
    float sum = 0.0f;
    float comp = 0.0f;                          // Kahan running compensation
    for (ulong i = lid; i < group_elems; i += GN_THREADS) {
        float val = x[base + i];
        float y = val - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    red[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Tree reduction of the 256 partials (plain f32 — only 8 levels).
    for (uint stride = GN_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) { red[lid] += red[lid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        mean_sh = red[0] / (float)group_elems;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = mean_sh;

    // ── Pass 2: Kahan-compensated partial sum of (x - mean)^2. ──
    float vsum = 0.0f;
    float vcomp = 0.0f;
    for (ulong i = lid; i < group_elems; i += GN_THREADS) {
        float d = x[base + i] - mean;
        float dv = d * d;
        float y = dv - vcomp;
        float t = vsum + y;
        vcomp = (t - vsum) - y;
        vsum = t;
    }
    red[lid] = vsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GN_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) { red[lid] += red[lid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        float var = red[0] / (float)group_elems;
        inv_std_sh = 1.0f / sqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = inv_std_sh;

    // ── Pass 3: normalize + per-channel affine, in place. ──
    // Element i (within the group) belongs to channel c0 + i/hw.
    const uint c0 = g * gs;
    for (ulong i = lid; i < group_elems; i += GN_THREADS) {
        const uint ci = (uint)(i / (ulong)hw);  // local channel 0..gs-1
        const uint c  = c0 + ci;
        const float wgt = weight[c];
        const float bia = bias[c];
        x[base + i] = (x[base + i] - mean) * inv_std * wgt + bia;
    }
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// SiLU — element-wise x / (1 + exp(-x))
// ═══════════════════════════════════════════════════════════════════════════

/// Element-wise SiLU (`x · sigmoid(x) = x / (1 + exp(-x))`), in place.
///
/// Mirrors `oxibonsai-image::math::silu` exactly. Buffers:
/// - buffer(0) = `x` (f32, read-write / in-place)
/// - buffer(1) = `n` (u32, element count)
///
/// Dispatch: `[ceil(n / 256), 1, 1]` threadgroups, `[256, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_SILU_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_f32(
    device float*  x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    const float v = x[gid];
    x[gid] = v / (1.0f + exp(-v));
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Upsample nearest ×2 — [C, H, W] → [C, 2H, 2W]
// ═══════════════════════════════════════════════════════════════════════════

/// Nearest-neighbour ×2 upsample of an NCHW buffer `[C, H, W] → [C, 2H, 2W]`
/// (each pixel repeated 2× along H and W). Mirrors
/// `oxibonsai-image::vae::ops::upsample_nearest2x`.
///
/// Buffers:
/// - buffer(0) = `input`  (f32, `[C, H·W]`)
/// - buffer(1) = `output` (f32, `[C, 2H·2W]`, written)
/// - buffer(2) = `c`       (u32)
/// - buffer(3) = `h`       (u32)
/// - buffer(4) = `w`       (u32)
/// - buffer(5) = `n_out`   (u32, `C·2H·2W`, the grid bound)
///
/// Dispatch: `[ceil(n_out / 256), 1, 1]` threadgroups, `[256, 1, 1]` threads
/// (one thread per output element).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_UPSAMPLE_NEAREST_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void upsample_nearest_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint&      c       [[buffer(2)]],
    constant uint&      h       [[buffer(3)]],
    constant uint&      w       [[buffer(4)]],
    constant uint&      n_out   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_out) { return; }
    const uint w_out  = w * 2u;
    const uint hw_out = h * 2u * w_out;         // output elements per channel
    const uint ch  = gid / hw_out;
    const uint r   = gid % hw_out;
    const uint ho  = r / w_out;
    const uint wo  = r % w_out;
    const uint hh  = ho / 2u;
    const uint ww  = wo / 2u;
    output[gid] = input[(ulong)ch * (ulong)h * (ulong)w + (ulong)hh * (ulong)w + (ulong)ww];
}
"#;
