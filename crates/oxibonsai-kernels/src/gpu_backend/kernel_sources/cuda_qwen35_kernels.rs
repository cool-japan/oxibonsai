//! CUDA kernel sources for the `qwen35` hybrid path (PrismML Bonsai 2, finding
//! **F13**) and the `PQ2_0` 2-bit decode table (finding **F16**, CUDA twin of
//! MET-11).
//!
//! **Not yet run on CUDA hardware, except the `PQ2_0` GEMV.** The one CUDA
//! validation run so far (RTX A4000, CUDA 12.0, 2026-10-07) had no Bonsai 2
//! 27B files, so none of the hybrid-layer kernels below ran (CUDA-P01..P08,
//! P10). The `PQ2_0` GEMV is CUDA-P09 **partial**: 450 GEMVs on real `PQ2_0`
//! weights matched the CPU, but those files hold no `+2` codes. Every kernel
//! is transcribed from the CPU/Metal reference math (the `qwen35`
//! GGUF metadata's hybrid-layer formulas, `oxibonsai_core::quant_prism`, and
//! `kernel_sources::qwen35`'s MSL — see each kernel's doc comment for its
//! exact source); apart from that partial GEMV run, each is checked only by
//! `scripts/check_cuda.sh`'s syntax pass and (for the pure host helpers)
//! `cargo test`. A cosine-similarity parity
//! run against the CPU reference, per kernel, on real layer-0 activations
//! (cos >= 0.999) is required before this path is advertised as working;
//! the checklist of those runs is `tests/cuda_hybrid_parity_plan.rs`.
//!
//! Every CUDA construct specific to the `qwen35` hybrid layer, the Hadamard
//! transform, `PTQ1_0` or `PQ2_0` lives in this module or in
//! `crate::gpu_backend::cuda_qwen35` (the host-side wiring, `native-cuda` only).
//!
//! # Scope
//!
//! Bonsai 2's hybrid layer needs seven device-side primitives beyond the
//! plain-transformer kernels this crate already has: a blockwise
//! Hadamard transform with fused sign-flip (`fwht_signed`), depthwise causal
//! conv1d (`conv1d_silu`), the gated delta rule recurrence (`gdn_step`), a
//! gated RMSNorm (`gated_rmsnorm`), L2-normalisation (`l2_normalize`), a
//! sigmoid attention gate (`sigmoid_gate`), and partial RoPE with an
//! explicit rotary width (`partial_rope`) — the existing
//! `fused_qk_rope`/`fused_qk_norm_rope` kernels
//! (`crate::gpu_backend::cuda_attn_kernels`) hardcode
//! `half_dim = head_dim >> 1`, which is wrong for Bonsai 2's `n_rot = 64` on
//! a 256-wide head.
//! `PTQ1_0` (ggml 143) is **not** a device kernel here: like `PQ2_0`, its
//! ternary codes are losslessly representable in the existing 2-bit SoA
//! layout, so `crate::gpu_backend::cuda_qwen35` transcodes it once at weight-load
//! time and the proven `gemv_tq2_g128_v1` kernel
//! (`crate::gpu_backend::cuda_kernels`) serves it — no new GEMV kernel, no
//! new decode table to get wrong on-device.
//!
//! Dispatch shapes follow the rest of this crate's CUDA kernels: one CTA per
//! independent row/head/span, launch configuration chosen by the host
//! wiring. Kernels are written in plain loop form (no hand-unrolled 128-bit
//! vector loads, no warp-shuffle micro-optimisation) — correctness first,
//! as for every CUDA kernel in this crate not yet profiled on hardware;
//! `metal_full_layer::qwen35_encode`'s Metal counterpart documents
//! the same choice for its PQ2 GEMV until real hardware can profile it.

/// Blockwise Hadamard transform with fused sign-flip (finding **F13**;
/// `prism.hadamard.*`, design doc §"Hadamard contract").
///
/// Mirrors `kernel_sources::qwen35`'s `q35_fwht_tg` MSL helper and
/// `q35_fwht_signed` kernel exactly: an **unnormalised** blockwise
/// Walsh–Hadamard transform run in shared memory, `len = 1, 2, 4, ...` pass
/// order (bit-reversal-free, same order the CPU's `oxibonsai_kernels::hadamard`
/// runs), over `n / block` independent `block`-wide transforms per row.
///
/// Forward (`inverse == 0`): `y = FWHT_block(x * signs) * scale`, where
/// `scale = 1/sqrt(block)` folds in the design doc's per-block normalisation
/// (`block == 1024` for every folded matrix in this checkpoint, including the
/// inverse embedding path: `hidden_size == 5120` is not itself a power of two
/// `<= 1024`, so that fold runs as 5 independent `block == 1024` chunks, same
/// as every other folded matrix's width — see
/// `fwht_dims_accept_the_checkpoints_folded_widths` below. The kernel takes
/// `block` as an argument rather than a compile-time constant so the same PTX
/// serves every fold without a recompile).
/// Inverse (`inverse != 0`): `y = FWHT_block(x) * signs * scale` — used only
/// for `token_embd.weight` (`prism.hadamard.inverse_weight_names`).
///
/// Dispatch: grid `(width / block, rows, 1)`, block `(min(block, 1024), 1, 1)`.
/// `block` must be a power of two `<= 1024` (`Q35_MAX_SPAN`); the host-side
/// [`qwen35_fwht_dims_ok`] (pure Rust, unit-tested on every host) enforces
/// this before a launch.
///
/// Buffers: `x` (buffer 0, in-place), `signs` (buffer 1, length >= `width`);
/// scalars: `width` (2), `block` (3), `inverse` (4), `scale` (5).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_FWHT_KERNELS_SRC: &str = r#"
/* Count trailing zeros via `popc((x & -x) - 1)` rather than `__ffs` (real
   NVCC has both; the tier-2 `scripts/check_cuda.sh` stub header, which
   parses these strings as plain host C++ when no CUDA toolkit is present,
   only stands in for `__popc`) — `x` is a block/len half-width here, always
   a power of two and never zero, so the `(x & -x) - 1` isolation is exact. */
static __device__ __forceinline__ unsigned int q35_ctz(unsigned int x) {
    return __popc((x & (~x + 1u)) - 1u);
}

/* In-place unnormalised blockwise FWHT over `n` floats of shared memory,
   `n / block` independent `block`-wide transforms, pass order
   `len = 1, 2, 4, ...` exactly as the CPU/Metal run it. The caller has
   already staged `x * sign * scale` and issued a barrier; every pass ends
   with one. Mirrors `kernel_sources::qwen35`'s `q35_fwht_tg` line for line. */
static __device__ __forceinline__ void q35_fwht_shared(
    float* xs, unsigned int n, unsigned int block,
    unsigned int tid, unsigned int tsz)
{
    if (block < 2u) {
        return;
    }
    const unsigned int half_n = n >> 1u;
    const unsigned int half_b = block >> 1u;
    const unsigned int lg_half_b = q35_ctz(half_b);
    for (unsigned int len = 1u; len < block; len <<= 1u) {
        const unsigned int lg = q35_ctz(len);
        for (unsigned int t = tid; t < half_n; t += tsz) {
            const unsigned int blk = t >> lg_half_b;
            const unsigned int j = t & (half_b - 1u);
            const unsigned int pos = blk * block + ((j >> lg) << (lg + 1u)) + (j & (len - 1u));
            const float a = xs[pos];
            const float b = xs[pos + len];
            xs[pos] = a + b;
            xs[pos + len] = a - b;
        }
        __syncthreads();
    }
}

#define Q35_MAX_SPAN 1024u

extern "C" __global__ void fwht_signed(
    float* __restrict__ x,
    const float* __restrict__ signs,
    unsigned int width,
    unsigned int block,
    unsigned int inverse,
    float scale)
{
    __shared__ float xs[Q35_MAX_SPAN];
    const unsigned int tid = threadIdx.x;
    const unsigned int tsz = blockDim.x;
    const unsigned int col0 = blockIdx.x * block;
    float* row = x + (unsigned long long)blockIdx.y * width + col0;

    for (unsigned int i = tid; i < block; i += tsz) {
        const float v = row[i];
        xs[i] = (inverse == 0u) ? (v * signs[col0 + i] * scale) : (v * scale);
    }
    __syncthreads();
    q35_fwht_shared(xs, block, block, tid, tsz);
    for (unsigned int i = tid; i < block; i += tsz) {
        const float v = xs[i];
        row[i] = (inverse == 0u) ? v : (v * signs[col0 + i]);
    }
}
"#;

/// Depthwise causal 1-D convolution (kernel width 4) fused with `SiLU`, over
/// `t_len` tokens per call (finding **F13**; `ssm_conv1d.weight [4, 10240]`,
/// `ssm.conv_kernel=4`).
///
/// Per channel `c` and token `t`: `y[t][c] = SiLU( sum_{k=0..3} w[c*4 + k] *
/// window_t[k] )`, where `window_t` is the 3 causal history taps followed by
/// this token's new sample, and the window shifts by one after each token
/// (drop the oldest tap, append the new sample) so token `t+1` sees token
/// `t`'s sample as its most recent history. `state` is channel-major, 3
/// taps per channel, oldest tap first — exactly the layout
/// `oxibonsai_model::hybrid::recurrent_cache`'s `RecurrentCache` conv window
/// keeps per linear layer and
/// `oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode` updates in place
/// (`state[c*3 + tap]`), NOT a frame-major `[frame][channel]` buffer with the
/// new sample folded into it. `weight`'s channel-major indexing (`c*4 + k`,
/// not `k*channels + c`) matches GGUF `ssm_conv1d.weight [4, 10240]`'s
/// on-disk layout (ne0=4 contiguous per channel) and the Metal reference
/// `q35_conv1d_silu`'s `w[c*4u + k]` (`kernel_sources::qwen35`) exactly; `x`
/// and `out` are token-major (`x[t*channels + c]`), also matching
/// `q35_conv1d_silu`'s `x[t*channels+c]`/`out[t*channels+c]`, so a caller
/// mirroring the Metal dispatch needs no extra transpose.
///
/// Dispatch: grid `(ceil(channels / block_dim), 1, 1)`, block
/// `(256, 1, 1)`.
///
/// Buffers: `x` (0, `[t_len * channels]`, token-major), `weight` (1,
/// `[4 * channels]`, channel-major `[c][k]`), `state` (2, `[3 * channels]`,
/// channel-major `[c][tap]`, oldest tap first, updated in place), `out` (3,
/// `[t_len * channels]`, token-major); scalars: `channels` (4), `t_len` (5).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_CONV1D_KERNELS_SRC: &str = r#"
static __device__ __forceinline__ float q35_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static __device__ __forceinline__ float q35_silu(float x) {
    return x * q35_sigmoid(x);
}

extern "C" __global__ void conv1d_silu(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int channels,
    unsigned int t_len)
{
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    float s0 = state[c * 3u];
    float s1 = state[c * 3u + 1u];
    float s2 = state[c * 3u + 2u];
    const float w0 = weight[c * 4u];
    const float w1 = weight[c * 4u + 1u];
    const float w2 = weight[c * 4u + 2u];
    const float w3 = weight[c * 4u + 3u];

    for (unsigned int t = 0u; t < t_len; ++t) {
        const float xv = x[t * channels + c];
        const float acc = fmaf(xv, w3, fmaf(s2, w2, fmaf(s1, w1, s0 * w0)));
        out[t * channels + c] = q35_silu(acc);
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }

    state[c * 3u] = s0;
    state[c * 3u + 1u] = s1;
    state[c * 3u + 2u] = s2;
}
"#;

/// Per-row L2-normalisation (finding **F13**; the gated-delta-net's
/// `q,k L2-normalised per head` step, `head_k_dim = 128`).
///
/// `y = x / max(||x||_2, eps)`. `eps` guards the all-zero row that a padded
/// batch or an untouched scratch buffer can produce; the CPU reference uses
/// the same guard (never a bare division).
///
/// Dispatch: grid `(n_rows, 1, 1)`, block `(min(row_width, 256), 1, 1)`.
///
/// Buffers: `x` (0, `[n_rows * row_width]`, in place); scalars: `row_width`
/// (1), `eps` (2).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_L2NORM_KERNELS_SRC: &str = r#"
extern "C" __global__ void l2_normalize(
    float* __restrict__ x,
    unsigned int row_width,
    float eps)
{
    extern __shared__ float l2_partial[];
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tsz = blockDim.x;
    float* row_ptr = x + (unsigned long long)row * row_width;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < row_width; i += tsz) {
        const float v = row_ptr[i];
        sum_sq += v * v;
    }
    l2_partial[tid] = sum_sq;
    __syncthreads();
    for (unsigned int stride = tsz >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            l2_partial[tid] += l2_partial[tid + stride];
        }
        __syncthreads();
    }
    const float norm = sqrtf(l2_partial[0]);
    const float inv = 1.0f / fmaxf(norm, eps);
    for (unsigned int i = tid; i < row_width; i += tsz) {
        row_ptr[i] *= inv;
    }
}
"#;

/// Sigmoid attention gate (finding **F13**; full-attention layers' `out =
/// attn * sigmoid(gate)`, gate from the `[q(256) | gate(256)]` interleave of
/// `attn_q`'s projection).
///
/// `y[i] = attn[i] * sigmoid(gate[i])`, elementwise. `gate` is read
/// contiguously (`gate[i]`, not a strided `[q(256) | gate(256)]` read): the
/// caller must de-interleave `attn_q`'s per-head `[q | gate]` projection into
/// a contiguous gate buffer before this launch — this kernel does no
/// gathering of its own, unlike the Metal reference
/// `q35_sigmoid_gate_rotate` (`kernel_sources::qwen35`), which reads the
/// interleave directly with a strided index.
///
/// Dispatch: grid `(ceil(n / block_dim), 1, 1)`, block `(256, 1, 1)`.
///
/// Buffers: `attn` (0), `gate` (1, contiguous, already de-interleaved),
/// `out` (2), all `[n]`; scalar: `n` (3).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_SIGMOID_GATE_KERNELS_SRC: &str = r#"
static __device__ __forceinline__ float q35_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void sigmoid_gate(
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    float* __restrict__ out,
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn[i] * q35_sigmoid(gate[i]);
}
"#;

/// Gated RMSNorm per v-head (finding **F13**; the Gated-DeltaNet output norm
/// `w * RMSNorm(o_head) * silu(z_head)`, `head_v_dim = 128`).
///
/// One CTA per v-head: `y = weight * (o / rms(o)) * silu(z)`.
///
/// Dispatch: grid `(n_v_heads, 1, 1)`, block `(head_dim, 1, 1)` (`head_dim
/// <= 1024`).
///
/// Buffers: `o` (0, `[n_v_heads * head_dim]`), `z` (1, same shape — the
/// caller has already re-ordered the tiled v-head gate into `o`'s head
/// order, `tiled(m) = (m % rep) * n_k_heads + m / rep`, before this launch),
/// `weight` (2, `[head_dim]`, shared across heads), `out` (3, same shape as
/// `o`); scalars: `head_dim` (4), `eps` (5).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_GATED_RMSNORM_KERNELS_SRC: &str = r#"
static __device__ __forceinline__ float q35_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static __device__ __forceinline__ float q35_silu(float x) {
    return x * q35_sigmoid(x);
}

extern "C" __global__ void gated_rmsnorm(
    const float* __restrict__ o,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned int head_dim,
    float eps)
{
    extern __shared__ float grn_partial[];
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int tsz = blockDim.x;
    const float* o_row = o + (unsigned long long)head * head_dim;
    const float* z_row = z + (unsigned long long)head * head_dim;
    float* out_row = out + (unsigned long long)head * head_dim;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < head_dim; i += tsz) {
        const float v = o_row[i];
        sum_sq += v * v;
    }
    grn_partial[tid] = sum_sq;
    __syncthreads();
    for (unsigned int stride = tsz >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            grn_partial[tid] += grn_partial[tid + stride];
        }
        __syncthreads();
    }
    const float mean_sq = grn_partial[0] / (float)head_dim;
    const float inv_rms = rsqrtf(mean_sq + eps);
    for (unsigned int i = tid; i < head_dim; i += tsz) {
        out_row[i] = weight[i] * (o_row[i] * inv_rms) * q35_silu(z_row[i]);
    }
}
"#;

/// Partial (text-only) RoPE with an explicit rotary width (finding **F13**).
///
/// The existing `fused_qk_rope` / `fused_qk_norm_rope` kernels
/// ([`crate::gpu_backend::cuda_attn_kernels`]) hardcode
/// `half_dim = head_dim >> 1`, i.e. they rotate the *whole* head. Bonsai 2's
/// full-attention
/// layers rotate only the first `n_rot` of `head_dim = 256` dimensions
/// (`rope.dimension_count = 64`; `rope.dimension_sections = [11, 11, 10, 0]`).
/// This kernel takes one `cos`/`sin` table per call and applies it uniformly
/// across the `n_rot/2` half-pairs, i.e. it is text-only: it has no 3-axis
/// section schedule of its own. Text-only inference (a single position per
/// token, no vision/temporal axis) needs exactly that: one table built from
/// the combined `11+11+10 = 32` half-pairs at that position. `<|image_pad|>`
/// / `<|video_pad|>` tokens need the vision tower's 3-axis M-RoPE position
/// schedule instead (design doc §8.2); this kernel can still serve that case,
/// but only if the host builds `cos`/`sin` per pair from the right axis
/// first — the kernel itself has no notion of which axis a pair belongs to.
/// Dimensions `[n_rot..head_dim)` pass through unchanged — this kernel must
/// run on a buffer that already holds them, not a zeroed scratch one.
///
/// NEOX half-split rotation on `[0..n_rot)`: for `i` in `0..n_rot/2`,
/// `(x[i], x[i + n_rot/2])` rotate by `(cos[i], sin[i])` — llama.cpp's
/// `GGML_ROPE_TYPE_NEOX` pairing, which its M-RoPE path uses for `qwen35`
/// (`ggml-cpu/ops.cpp`: `rotate_pairs(n_dims, n_dims / 2)`), and the same
/// pairing as this crate's `cuda_attn_kernels::fused_qk_rope`. It is **not**
/// GPT-J's adjacent-pair `(x[2i], x[2i + 1])` pairing.
///
/// Dispatch: grid `(ceil(n_rot/2 / block_dim), n_heads, 1)`, block
/// `(64, 1, 1)`.
///
/// Buffers: `x` (0, `[n_heads * head_dim]`, in place), `cos` (1, `[n_rot /
/// 2]`), `sin` (2, `[n_rot / 2]`); scalars: `head_dim` (3), `n_rot` (4).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_PARTIAL_ROPE_KERNELS_SRC: &str = r#"
extern "C" __global__ void partial_rope(
    float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    unsigned int head_dim,
    unsigned int n_rot)
{
    const unsigned int half_rot = n_rot >> 1u;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_rot) return;

    float* row = x + (unsigned long long)blockIdx.y * head_dim;
    const float c = cos_table[i];
    const float s = sin_table[i];
    const float a = row[i];
    const float b = row[i + half_rot];
    row[i]            = a * c - b * s;
    row[i + half_rot] = a * s + b * c;
}
"#;

/// Gated-DeltaNet recurrence, one token step (finding **F13**; design §2.3,
/// the gated delta rule per token per v-head, whose CPU reference is
/// [`crate::gated_delta_net::gdn_step_with`] — `S`: `128x128` `f32` state
/// per v-head, `nv = 48` v-heads sharing `nk = 16` k-heads via the tiled
/// `vperm`/`gdn_v_grouped` mapping (host-side gather, not this kernel's
/// concern: the caller hands this kernel `q` / `k` already expanded to one
/// `head_k_dim`-wide vector per v-head)).
///
/// One CTA per v-head, `head_v_dim` (128) threads, **thread `j` owns state
/// column `S[:, j]`** (`S[i][j]`, `i` the row / input-channel index) — every
/// step this thread's column is written and read only by thread `j` itself
/// (`kv[j] = sum_i k[i]*S[i][j]`, `o[j] = sum_i q[i]*S[i][j]` are *local*
/// reductions over the thread's own 128-float column once `k`/`q` are
/// broadcast through shared memory), so `S` never needs a cross-thread
/// reduction or a lock — chosen over a row-major, thread-per-row layout
/// specifically to make `kv`/`o` embarrassingly parallel per thread. `S`
/// stays resident in **global** memory across calls (`128*128*4 B = 64 KiB`
/// per v-head — `48` v-heads is `3 MiB`/layer, far past any GPU's shared
/// memory, hence the design doc's dedicated recurrent-state allocation
/// rather than a threadgroup-local one); this kernel reads its slice in,
/// updates it, and writes it back before returning, so a decode loop that
/// calls it once per token leaves `S` correctly threaded across positions.
///
/// Per-thread step order (must match the CPU reference
/// [`crate::gated_delta_net::gdn_step_with`] exactly — parity depends on
/// it): decay first, *then* read `kv` from the decayed state, *then* the
/// delta update, *then* read `o` from the fully updated state.
///
/// ```text
/// g      = -exp(A_log) * softplus(a + dt_bias)   [host/prep kernel; see below]
/// decay  = exp(g)                                 (scalar per v-head, this token)
/// beta   = sigmoid(b)                              (scalar per v-head, this token)
/// S[:,j] *= decay
/// kv[j]   = sum_i k[i] * S[i][j]
/// delta_j = (v[j] - kv[j]) * beta
/// S[:,j] += k[:] * delta_j
/// o[j]    = (sum_i q[i] * S[i][j]) * out_scale
/// ```
///
/// `out_scale` is `1/sqrt(head_v_dim)` —
/// [`crate::gated_delta_net::GdnDims::out_scale`]'s (the CPU reference) `1.0
/// / (head_v_dim as f32).sqrt()`, `1/sqrt(128) ~= 0.0884` for this checkpoint,
/// **not** `1.0`. Applied once per v-head per token, after the state update,
/// exactly where the CPU reference and the Metal `q35_gdn` kernel
/// (`kernel_sources::qwen35`, `o[j] = acc * out_scale`) apply it — never
/// folded into `q`, `decay` or `beta`, which the state update also reads.
///
/// `g`/`beta` are themselves derived from tiny (`width == n_v_heads == 48`)
/// per-token projections (`ssm_alpha`/`ssm_beta`, `O(n_v_heads)` work) —
/// cheap enough to fold into the host-side per-token prep alongside the
/// `q,k` L2-normalisation ([`CUDA_QWEN35_L2NORM_KERNELS_SRC`]) rather than
/// duplicating a reduction-free elementwise pass on-device; `decay` and
/// `beta` therefore arrive as this kernel's inputs, already computed.
///
/// Dispatch: grid `(n_v_heads, 1, 1)`, block `(head_v_dim, 1, 1)`
/// (`head_v_dim == head_k_dim == 128` for this checkpoint; the kernel does
/// not hardcode `128` so a future checkpoint with a different `ssm.state_size`
/// still dispatches correctly as long as `head_v_dim <= 1024`).
///
/// Buffers: `state` (0, `[n_v_heads * head_k_dim * head_v_dim]`,
/// column-major per head — `state[head*hk*hv + j*hk + i]` — updated in
/// place), `q` (1, `[n_v_heads * head_k_dim]`), `k` (2, same shape as `q`),
/// `v` (3, `[n_v_heads * head_v_dim]`), `decay` (4, `[n_v_heads]`), `beta`
/// (5, `[n_v_heads]`), `out` (6, `[n_v_heads * head_v_dim]`); scalars:
/// `head_k_dim` (7) — `head_v_dim` is `blockDim.x` and needs no separate
/// argument — and `out_scale` (8).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_GDN_KERNELS_SRC: &str = r#"
#define Q35_GDN_MAX_HK 256u

extern "C" __global__ void gdn_step(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ out,
    unsigned int head_k_dim,
    float out_scale)
{
    extern __shared__ float gdn_shared[]; /* [0..head_k_dim) = k, [head_k_dim..2*head_k_dim) = q */
    float* k_sh = gdn_shared;
    float* q_sh = gdn_shared + head_k_dim;

    const unsigned int head = blockIdx.x;
    const unsigned int head_v_dim = blockDim.x;
    const unsigned int j = threadIdx.x;

    const float* q_row = q + (unsigned long long)head * head_k_dim;
    const float* k_row = k + (unsigned long long)head * head_k_dim;
    for (unsigned int i = j; i < head_k_dim; i += head_v_dim) {
        k_sh[i] = k_row[i];
        q_sh[i] = q_row[i];
    }
    __syncthreads();

    float col[Q35_GDN_MAX_HK];
    float* s_col = state + ((unsigned long long)head * head_k_dim * head_v_dim) + (unsigned long long)j * head_k_dim;
    const float d = decay[head];
    const float b = beta[head];

    float kv = 0.0f;
    for (unsigned int i = 0u; i < head_k_dim; ++i) {
        col[i] = s_col[i] * d;
        kv += k_sh[i] * col[i];
    }

    const float delta = (v[(unsigned long long)head * head_v_dim + j] - kv) * b;

    float o = 0.0f;
    for (unsigned int i = 0u; i < head_k_dim; ++i) {
        col[i] += k_sh[i] * delta;
        o += q_sh[i] * col[i];
        s_col[i] = col[i];
    }

    out[(unsigned long long)head * head_v_dim + j] = o * out_scale;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// PQ2_0 decode + GEMV (finding F16, CUDA twin of MET-11)
// ═══════════════════════════════════════════════════════════════════════════

/// `PQ2_0` (PrismML ggml type **142**) 2-bit GEMV, CUDA twin of Metal's
/// `MSL_GEMV_PQ2_G128_V1` / `MSL_DECODE_PQ2_FN`
/// (`kernel_sources/decode_ternary.rs`).
///
/// Consumes the **same** SoA layout as `gemv_tq2_g128_v1`
/// ([`crate::gpu_backend::cuda_kernels`]) — `[all d: N×2 B FP16 LE][all qs:
/// N×32 B]` — produced from `PQ2_0`'s d-first AoS blocks by
/// [`CudaGraph::encode_gemv_pq2_cached`](crate::gpu_backend::cuda_graph::CudaGraph::encode_gemv_pq2_cached)'s
/// `reformat_pq2_aos_bytes_to_soa`. The **only** difference from
/// `gemv_tq2_g128_v1` is the decode table: `decode_pq2(0b11) = +2.0`, where
/// `decode_tq2(0b11) = 0.0` — mirroring the Metal fix exactly, this kernel
/// gets its own decode function and its own entry point rather than a
/// branch inside the shared one, so the proven `TQ2_0_g128` kernel is
/// never touched.
///
/// Kept in loop form (not hand-unrolled / 128-bit-vector-loaded like
/// `gemv_tq2_g128_v1`), matching `MSL_GEMV_PQ2_G128_V1`'s own documented
/// choice to defer that optimisation until real hardware profiles it.
///
/// Dispatch: grid `(ceil(n_rows/8), 1, 1)`, block `(256, 1, 1)` — identical
/// to `gemv_tq2_g128_v1`.
///
/// Buffers: SoA weights (0), input `[k]` (1), output `[n_rows]` (2);
/// scalars: `n_rows` (3), `k` (4).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub const CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC: &str = r#"
static __device__ __forceinline__ float q35_fp16_to_f32(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

static __device__ __forceinline__ float decode_pq2(unsigned int code) {
    /* 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2   (fork: y = ((int)q - 1) * d,
       ggml-quants.c:501-511). Deliberately a *different* table from
       `decode_tq2` (which keeps 0b11 at 0.0, finding K-01) — the two must
       never be substituted for one another. */
    return (float)code - 1.0f;
}

extern "C" __global__ void gemv_pq2_g128_v1(
    const unsigned char* __restrict__ soa_raw,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    unsigned int n_rows,
    unsigned int k)
{
    const unsigned int warp_id = threadIdx.x >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int row     = blockIdx.x * 8u + warp_id;
    if (row >= n_rows) return;

    const unsigned int blocks_per_row = k >> 7u;
    const unsigned long long total_blocks = (unsigned long long)n_rows * blocks_per_row;
    const unsigned long long qs_offset    = total_blocks * 2ull;

    const unsigned short* __restrict__ scales =
        (const unsigned short* __restrict__)soa_raw;

    float partial = 0.0f;

    for (unsigned int b = lane; b < blocks_per_row; b += 32u) {
        const unsigned long long block_idx = (unsigned long long)row * blocks_per_row + b;
        const float scale = q35_fp16_to_f32(scales[block_idx]);
        const unsigned char* qs = soa_raw + qs_offset + block_idx * 32u;
        const float* x = input + b * 128u;

        float block_sum = 0.0f;
        #pragma unroll
        for (unsigned int byte_idx = 0u; byte_idx < 32u; ++byte_idx) {
            const unsigned int byte = (unsigned int)qs[byte_idx];
            const float* xj = x + byte_idx * 4u;
            block_sum +=
                decode_pq2((byte      ) & 3u) * xj[0]
              + decode_pq2((byte >> 2u) & 3u) * xj[1]
              + decode_pq2((byte >> 4u) & 3u) * xj[2]
              + decode_pq2((byte >> 6u) & 3u) * xj[3];
        }
        partial += scale * block_sum;
    }

    partial += __shfl_down_sync(0xffffffffu, partial, 16u);
    partial += __shfl_down_sync(0xffffffffu, partial,  8u);
    partial += __shfl_down_sync(0xffffffffu, partial,  4u);
    partial += __shfl_down_sync(0xffffffffu, partial,  2u);
    partial += __shfl_down_sync(0xffffffffu, partial,  1u);

    if (lane == 0u) output[row] = partial;
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Host-testable pure helpers
//
// Gated on `feature = "native-cuda"` alone (no `target_os` restriction): the
// kernel strings above need cudarc + a CUDA-capable target to ever be
// launched, but the small dimension/decode-table helpers below are ordinary
// arithmetic with no device dependency, so — unlike everything else under
// `cuda_graph` / `cuda_full_layer` / `cuda_prefill` in this crate, which is
// gated on `any(target_os = "linux", target_os = "windows")` and so cannot be
// exercised on macOS at all — these run (and are `cargo test`-checked) right
// here. Each one is the scalar twin of a specific kernel's per-thread
// arithmetic above.
// ═══════════════════════════════════════════════════════════════════════════

/// The scalar twin of `decode_pq2` in `CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC`:
/// `00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2`. Only the low two bits of `code`
/// are used.
#[cfg(feature = "native-cuda")]
#[must_use]
pub const fn decode_pq2_code(code: u8) -> f32 {
    (code & 0b11) as f32 - 1.0
}

/// Weights per 2-bit block of the SoA layout `gemv_tq2_g128_v1` and
/// `gemv_pq2_g128_v1` (`CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC`) read.
#[cfg(feature = "native-cuda")]
const GEMV_2BIT_BLOCK_WEIGHTS: usize = 128;

/// Bytes per 2-bit SoA block: a 2-byte FP16 scale plus 32 bytes of codes.
#[cfg(feature = "native-cuda")]
const GEMV_2BIT_BLOCK_BYTES: usize = 34;

/// Host precondition of the 2-bit GEMV kernels' reduction dimension: `k` a
/// positive multiple of 128 (the kernels compute `blocks_per_row = k >> 7`
/// and have no partial-block encoding) and an `input` of at least `k`
/// elements (the host copies `input[..k]` to the device).
///
/// # Errors
/// A message naming the violated bound.
#[cfg(feature = "native-cuda")]
pub fn gemv_2bit_check_dims(k: usize, input_len: usize) -> Result<(), String> {
    if k == 0 || !k.is_multiple_of(GEMV_2BIT_BLOCK_WEIGHTS) {
        return Err(format!(
            "k={k} must be a positive multiple of {GEMV_2BIT_BLOCK_WEIGHTS} (2-bit block size)"
        ));
    }
    if input_len < k {
        return Err(format!(
            "input has {input_len} elements, need at least k={k}"
        ));
    }
    Ok(())
}

/// Host precondition tying a cached 2-bit SoA weight buffer to the GEMV
/// geometry it is launched with: `weight_len` must be exactly `n_rows * (k
/// / 128) * 34` bytes, `n_rows` positive, and both `n_rows` and `k` must fit
/// the kernels' `unsigned int` arguments.
///
/// The kernels derive the codes section's offset from the launch geometry
/// (`qs_offset = n_rows * blocks_per_row * 2`), so a buffer uploaded for a
/// different `n_rows` is read at the wrong offset and, when it is shorter,
/// past its end — this check is the host-side guard against that.
///
/// # Errors
/// A message naming the violated bound.
#[cfg(feature = "native-cuda")]
pub fn gemv_2bit_check_weight_len(
    weight_len: usize,
    n_rows: usize,
    k: usize,
) -> Result<(), String> {
    if n_rows == 0 {
        return Err("n_rows must be positive".to_string());
    }
    if u32::try_from(n_rows).is_err() || u32::try_from(k).is_err() {
        return Err(format!(
            "n_rows={n_rows} / k={k} exceed the kernels' 32-bit arguments"
        ));
    }
    let expected = n_rows
        .checked_mul(k / GEMV_2BIT_BLOCK_WEIGHTS)
        .and_then(|blocks| blocks.checked_mul(GEMV_2BIT_BLOCK_BYTES))
        .ok_or_else(|| format!("n_rows={n_rows} x k={k} overflows the SoA byte size"))?;
    if weight_len != expected {
        return Err(format!(
            "cached weight holds {weight_len} bytes, expected {expected} for {n_rows} rows x \
             k={k} (2-bit SoA, {GEMV_2BIT_BLOCK_BYTES} B per {GEMV_2BIT_BLOCK_WEIGHTS} weights)"
        ));
    }
    Ok(())
}

/// Whether `(width, block)` are valid dispatch dimensions for
/// `CUDA_QWEN35_FWHT_KERNELS_SRC`'s `fwht_signed`: `block` a power of two
/// no larger than `Q35_MAX_SPAN` (1024, the kernel's shared-memory budget),
/// `width` a positive multiple of `block` (an independent transform per
/// `block`-wide chunk, no partial chunk), and both nonzero.
#[cfg(feature = "native-cuda")]
#[must_use]
pub const fn qwen35_fwht_dims_ok(width: usize, block: usize) -> bool {
    const MAX_SPAN: usize = 1024;
    if width == 0 || block == 0 || block > MAX_SPAN {
        return false;
    }
    if !block.is_power_of_two() {
        return false;
    }
    width.is_multiple_of(block)
}

/// Whether `head_k_dim` fits `CUDA_QWEN35_GDN_KERNELS_SRC`'s fixed-size
/// per-thread `col[Q35_GDN_MAX_HK]` local array (`Q35_GDN_MAX_HK == 256`,
/// double this checkpoint's `head_k_dim == 128` so a slightly wider future
/// `ssm.state_size` still fits without a kernel recompile).
#[cfg(feature = "native-cuda")]
#[must_use]
pub const fn qwen35_gdn_head_k_dim_ok(head_k_dim: usize) -> bool {
    const MAX_HK: usize = 256;
    head_k_dim > 0 && head_k_dim <= MAX_HK
}

/// Whether `n_rot` is a valid rotary width for
/// `CUDA_QWEN35_PARTIAL_ROPE_KERNELS_SRC`'s `partial_rope` against a head
/// of `head_dim`: positive, even (the kernel pairs `i` with `i + n_rot/2`),
/// and no wider than the head itself.
#[cfg(feature = "native-cuda")]
#[must_use]
pub const fn qwen35_partial_rope_dims_ok(head_dim: usize, n_rot: usize) -> bool {
    n_rot > 0 && n_rot.is_multiple_of(2) && n_rot <= head_dim
}

/// The scalar twin of `CUDA_QWEN35_GDN_KERNELS_SRC`'s `gdn_step` output
/// scale: `1/sqrt(head_v_dim)`. Kept as its own function (rather than
/// inlined at the one call site in [`super::super::cuda_qwen35`]'s
/// `launch_gdn_step`) so it has exactly one host-testable definition to pin
/// against the CPU reference below.
#[cfg(feature = "native-cuda")]
#[must_use]
pub fn qwen35_gdn_out_scale(head_v_dim: usize) -> f32 {
    1.0 / (head_v_dim as f32).sqrt()
}

/// The scalar twin of one channel's `t_len`-token loop in
/// `CUDA_QWEN35_CONV1D_KERNELS_SRC`'s `conv1d_silu`: `state` is
/// channel-major, 3 taps, oldest first (`[tap]` for one channel — the kernel
/// indexes `state[c*3 + tap]` across channels); `x`/`out` are `[t_len]` for
/// this one channel (the kernel indexes `x[t*channels + c]` /
/// `out[t*channels + c]`); `w` is this channel's 4 taps (`weight[c*4 + k]`
/// in the kernel). Tested below against
/// [`crate::ssm_ops::causal_conv1d_k4_decode`], the CPU reference this
/// layout must interoperate with (`RecurrentCache`'s conv-state slot).
#[cfg(feature = "native-cuda")]
pub fn qwen35_conv1d_silu_channel_ref(
    state: &mut [f32; 3],
    w: [f32; 4],
    x: &[f32],
    out: &mut [f32],
) {
    let (mut s0, mut s1, mut s2) = (state[0], state[1], state[2]);
    for (xv, o) in x.iter().copied().zip(out.iter_mut()) {
        let acc = w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * xv;
        *o = acc / (1.0 + (-acc).exp());
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    *state = [s0, s1, s2];
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn decode_pq2_code_matches_the_fork_table() {
        assert_eq!(decode_pq2_code(0), -1.0);
        assert_eq!(decode_pq2_code(1), 0.0);
        assert_eq!(decode_pq2_code(2), 1.0);
        assert_eq!(decode_pq2_code(3), 2.0);
        // Only the low two bits matter (defensive against a caller passing
        // an unmasked byte).
        assert_eq!(decode_pq2_code(0b1111_1100), -1.0);
        assert_eq!(decode_pq2_code(0b1111_1111), 2.0);
    }

    #[test]
    fn decode_pq2_and_decode_tq2_disagree_only_on_the_reserved_code() {
        // decode_tq2's table (K-01, oxibonsai_core): 0 -> -1, 1 -> 0, 2 -> 1,
        // 3 -> 0. The two tables must differ *exactly* at code 3 — anywhere
        // else, a PQ2_0 tensor silently routed through the ternary decoder
        // would still (coincidentally) produce the right values, hiding the
        // MET-11-class bug this module exists to prevent.
        let decode_tq2 = |code: u8| -> f32 {
            match code & 0b11 {
                0 => -1.0,
                2 => 1.0,
                _ => 0.0, // 1 and the reserved 3 both decode to 0 for TQ2_0_g128
            }
        };
        for code in 0u8..4 {
            let pq2 = decode_pq2_code(code);
            let tq2 = decode_tq2(code);
            if code == 3 {
                assert_ne!(pq2, tq2, "code 0b11 must differ between the two tables");
                assert_eq!(pq2, 2.0);
                assert_eq!(tq2, 0.0);
            } else {
                assert_eq!(pq2, tq2, "codes 0/1/2 agree between PQ2_0 and TQ2_0_g128");
            }
        }
    }

    #[test]
    fn gemv_2bit_dims_reject_non_block_multiples_and_short_input() {
        assert!(gemv_2bit_check_dims(128, 128).is_ok());
        assert!(gemv_2bit_check_dims(5120, 6000).is_ok());
        assert!(gemv_2bit_check_dims(0, 128).is_err(), "k must be positive");
        assert!(gemv_2bit_check_dims(127, 128).is_err());
        assert!(gemv_2bit_check_dims(129, 129).is_err());
        assert!(
            gemv_2bit_check_dims(128, 127).is_err(),
            "input shorter than k"
        );
    }

    #[test]
    fn gemv_2bit_weight_len_must_match_the_launch_geometry_exactly() {
        // 17408 x 5120 (the 27B's ffn_gate): 17408 * 40 blocks * 34 bytes.
        let ffn_gate = 17_408 * (5120 / 128) * 34;
        assert!(gemv_2bit_check_weight_len(ffn_gate, 17_408, 5120).is_ok());
        assert!(gemv_2bit_check_weight_len(34, 1, 128).is_ok());
        // A Q-only buffer launched as fused Q||K||V (more rows than it holds)
        // is the out-of-bounds read this guard exists for.
        assert!(gemv_2bit_check_weight_len(34 * 4, 6, 128).is_err());
        // A larger buffer is refused too: `qs_offset` is derived from
        // `n_rows`, so every row would decode against the wrong codes.
        assert!(gemv_2bit_check_weight_len(34 * 8, 4, 128).is_err());
        assert!(
            gemv_2bit_check_weight_len(0, 0, 128).is_err(),
            "n_rows == 0"
        );
        assert!(
            gemv_2bit_check_weight_len(usize::MAX, usize::MAX, 128).is_err(),
            "rows past u32::MAX / overflowing byte size"
        );
    }

    #[test]
    fn fwht_dims_accept_the_checkpoints_folded_widths() {
        // Every Hadamard fold in this checkpoint is block 1024
        // (`prism.hadamard.block_size`), the inverse embedding fold
        // included: `h = FWHT_1024(z) * signs[5120]`, i.e. five independent
        // 1024-wide transforms over the 5120-wide row (`width=5120,
        // block=1024` below), exactly like every other folded matrix whose
        // width is a multiple of 1024. A single 5120-wide block is not the
        // contract, and `fwht_signed` refuses it anyway (1024 is also the
        // kernel's shared-memory cap and 5120 is not a power of two).
        assert!(qwen35_fwht_dims_ok(17_408, 1024)); // ffn_gate/up/down
        assert!(qwen35_fwht_dims_ok(6144, 1024)); // attn_output / ssm_out
        assert!(qwen35_fwht_dims_ok(5120, 1024)); // attn_q/k/v/qkv/gate, inverse embedding
        assert!(
            !qwen35_fwht_dims_ok(5120, 5120),
            "5120 > Q35_MAX_SPAN (1024)"
        );
        assert!(!qwen35_fwht_dims_ok(5121, 1024), "not a multiple of block");
        assert!(!qwen35_fwht_dims_ok(1024, 0));
        assert!(
            !qwen35_fwht_dims_ok(1024, 3),
            "block must be a power of two"
        );
    }

    #[test]
    fn gdn_head_k_dim_accepts_the_checkpoints_128_and_rejects_the_unbounded() {
        assert!(qwen35_gdn_head_k_dim_ok(128)); // ssm.state_size (head_k_dim) in this checkpoint
        assert!(qwen35_gdn_head_k_dim_ok(256)); // the kernel's compiled-in ceiling
        assert!(!qwen35_gdn_head_k_dim_ok(257));
        assert!(!qwen35_gdn_head_k_dim_ok(0));
    }

    #[test]
    fn partial_rope_dims_accept_the_checkpoints_64_of_256() {
        assert!(qwen35_partial_rope_dims_ok(256, 64)); // rope.dimension_count on a 256-wide head
        assert!(
            qwen35_partial_rope_dims_ok(64, 64),
            "a full-width rotation is also valid"
        );
        assert!(!qwen35_partial_rope_dims_ok(256, 0));
        assert!(
            !qwen35_partial_rope_dims_ok(256, 257),
            "n_rot cannot exceed head_dim"
        );
        assert!(
            !qwen35_partial_rope_dims_ok(256, 65),
            "n_rot must be even (paired rotation)"
        );
    }

    #[test]
    fn gdn_out_scale_matches_the_cpu_references_gdn_dims() {
        // Pin the CUDA launcher's scale directly against the CPU reference's
        // `GdnDims::out_scale` (`gated_delta_net.rs`) so the two can never
        // silently drift: an unscaled `gdn_step` output is `sqrt(head_v_dim)`
        // times too large (`11.3x` at this checkpoint's `head_v_dim == 128`).
        let dims = crate::gated_delta_net::GdnDims::bonsai2();
        assert_eq!(qwen35_gdn_out_scale(dims.head_v_dim), dims.out_scale());
        assert!((qwen35_gdn_out_scale(128) - 0.088_388_35).abs() < 1e-6);
        assert_eq!(qwen35_gdn_out_scale(1), 1.0);
    }

    #[test]
    fn conv1d_silu_channel_ref_matches_the_cpu_causal_conv1d_over_several_tokens() {
        // Drive the CPU reference `causal_conv1d_k4_decode` one token at a
        // time (its own per-token contract) over a multi-channel, multi-tap
        // state and compare against this module's channel-major,
        // x-separate-from-state scalar twin of the CUDA kernel, run the way
        // the kernel itself runs: one channel, `t_len` tokens in sequence.
        // Agreement here is exactly the layout contract `conv1d_silu` must
        // honour: `RecurrentCache`'s conv window is channel-major, 3 taps,
        // oldest first — not a frame-major buffer with the new sample
        // folded in.
        const CHANNELS: usize = 5;
        const KC_TAPS: usize = 3;
        let w: Vec<f32> = (0..CHANNELS * 4).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let xs: [[f32; CHANNELS]; 4] = [
            [0.5, -0.25, 1.0, 0.0, 2.0],
            [-1.0, 0.75, -0.5, 3.0, -2.0],
            [0.25, 0.25, 0.25, 0.25, 0.25],
            [1.5, -1.5, 0.0, -0.5, 0.5],
        ];

        // CPU reference: channel-major state `[channels][KC_TAPS]`, one
        // `causal_conv1d_k4_decode` call per token. Unlike the fused CUDA/
        // Metal `conv1d_silu` kernels, `causal_conv1d_k4_decode` returns the
        // raw convolution only (`kernel_sources::qwen35`'s own doc table:
        // `q35_conv1d_silu` == `causal_conv1d_k4_*` **+** `silu_simd`), so
        // SiLU is applied by hand here before comparing to the fused kernel
        // twin.
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let mut cpu_state = vec![0.0f32; CHANNELS * KC_TAPS];
        let mut cpu_out = vec![0.0f32; CHANNELS];
        let mut cpu_results = Vec::new();
        for x_t in &xs {
            crate::ssm_ops::causal_conv1d_k4_decode(&mut cpu_state, x_t, &w, &mut cpu_out)
                .expect("valid dims");
            cpu_results.push(cpu_out.iter().copied().map(silu).collect::<Vec<f32>>());
        }

        // This module's scalar twin of the kernel: one channel at a time,
        // `t_len` tokens in one call, exactly the kernel's per-thread loop.
        for c in 0..CHANNELS {
            let mut state = [0.0f32; KC_TAPS];
            let w_c = [w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]];
            let x_c: Vec<f32> = xs.iter().map(|frame| frame[c]).collect();
            let mut out_c = vec![0.0f32; xs.len()];
            qwen35_conv1d_silu_channel_ref(&mut state, w_c, &x_c, &mut out_c);

            for (t, &expected) in out_c.iter().enumerate() {
                assert!(
                    (expected - cpu_results[t][c]).abs() < 1e-6,
                    "channel {c} token {t}: cuda-shaped {expected} vs cpu {cpu_c}",
                    cpu_c = cpu_results[t][c]
                );
            }
            // The two references must also land on the same final window,
            // channel-major 3 taps, oldest first.
            assert!((state[0] - cpu_state[c * KC_TAPS]).abs() < 1e-6);
            assert!((state[1] - cpu_state[c * KC_TAPS + 1]).abs() < 1e-6);
            assert!((state[2] - cpu_state[c * KC_TAPS + 2]).abs() < 1e-6);
        }
    }
}

// The `CUDA_QWEN35_*_KERNELS_SRC` constants above are gated on `native-cuda`
// **and** `any(target_os = "linux", target_os = "windows")` (they hold
// kernel-source text that is only ever compiled for the native-CUDA
// backend), unlike the pure helpers `tests` above checks, which are gated
// on `native-cuda` alone so they also run on macOS.
// A test that names one of these constants needs the same, fuller gate as
// the constants themselves — split into its own module rather than
// widening the one above, so `cargo test --all-features` on macOS keeps
// running every test that *can* run here instead of silently losing this
// whole file's coverage to one over-permissive `use super::*;`.
#[cfg(all(
    test,
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod kernel_source_tests {
    use super::*;

    /// Every `CUDA_QWEN35_*_KERNELS_SRC` constant declares the entry point
    /// its doc comment promises — a cheap guard against a copy-paste rename
    /// that leaves the Rust-side launcher looking up a name NVRTC never
    /// compiled (which surfaces only as a runtime `load_function` error on
    /// real hardware, never at this host's compile time).
    #[test]
    fn every_kernel_source_declares_its_documented_entry_point() {
        let pairs: &[(&str, &str)] = &[
            (CUDA_QWEN35_FWHT_KERNELS_SRC, "fwht_signed"),
            (CUDA_QWEN35_CONV1D_KERNELS_SRC, "conv1d_silu"),
            (CUDA_QWEN35_L2NORM_KERNELS_SRC, "l2_normalize"),
            (CUDA_QWEN35_SIGMOID_GATE_KERNELS_SRC, "sigmoid_gate"),
            (CUDA_QWEN35_GATED_RMSNORM_KERNELS_SRC, "gated_rmsnorm"),
            (CUDA_QWEN35_PARTIAL_ROPE_KERNELS_SRC, "partial_rope"),
            (CUDA_QWEN35_GDN_KERNELS_SRC, "gdn_step"),
            (CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC, "gemv_pq2_g128_v1"),
        ];
        for (src, name) in pairs {
            let needle = format!("__global__ void {name}(");
            assert!(
                src.contains(&needle),
                "expected `{needle}` in the kernel source, found none"
            );
        }
    }

    #[test]
    fn every_kernel_source_defines_the_helpers_it_calls() {
        // NVRTC compiles each `pub const CUDA_*` string as an independent
        // translation unit, and `scripts/check_cuda.sh` extracts and
        // syntax-checks each one in isolation the same way, so a kernel may
        // not rely on a helper defined only in a *different* constant —
        // each of the four below must carry its own copy of every
        // `q35_*` device function it calls.
        for (src, calls, defines) in [
            (
                CUDA_QWEN35_CONV1D_KERNELS_SRC,
                &["q35_silu("][..],
                &["q35_silu", "q35_sigmoid"][..],
            ),
            (
                CUDA_QWEN35_SIGMOID_GATE_KERNELS_SRC,
                &["q35_sigmoid("][..],
                &["q35_sigmoid"][..],
            ),
            (
                CUDA_QWEN35_GATED_RMSNORM_KERNELS_SRC,
                &["q35_silu("][..],
                &["q35_silu", "q35_sigmoid"][..],
            ),
            (
                CUDA_QWEN35_GEMV_PQ2_KERNEL_SRC,
                &["q35_fp16_to_f32("][..],
                &["q35_fp16_to_f32"][..],
            ),
        ] {
            for call in calls {
                assert!(src.contains(call), "expected a call to `{call}`");
            }
            for def in defines {
                let needle = format!("float {def}(");
                assert!(
                    src.contains(&needle),
                    "expected `{def}` to be defined in its own source, not just called"
                );
            }
        }
    }
}
