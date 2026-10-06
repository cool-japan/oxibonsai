//! Metal kernels for the Qwen3.5 / PrismML Bonsai 2 hybrid stack
//! (`general.architecture = "qwen35"`, design §4.1).
//!
//! Every kernel here has a CPU twin in this crate that it is tested against
//! (`metal_full_layer::qwen35`'s tests, and the model-level parity gates):
//!
//! | kernel | CPU reference |
//! |---|---|
//! | `q35_fwht_signed` | `hadamard::fwht_forward_signed` / `fwht_inverse_signed` (bitwise) |
//! | `q35_rmsnorm_rotate` | `rms_norm_simd` + `fwht_forward_signed` |
//! | `q35_swiglu_rotate` | `swiglu_simd` + `fwht_forward_signed` |
//! | `q35_sigmoid_gate_rotate` | `norms::sigmoid_mul_simd` + `fwht_forward_signed` |
//! | `q35_gated_norm_rotate` | `norms::rms_norm_gated_simd` per v-head + `fwht_forward_signed` |
//! | `q35_conv1d_silu` | `ssm_ops::causal_conv1d_k4_*` + `silu_simd` |
//! | `q35_gdn` | `norms::l2_norm_simd` + `gated_delta_net_chunk::gdn_prefill_with` |
//! | `q35_gemv_*` | `LinearLayer::forward_mat` for each weight format |
//! | `q35_kv_store` | `KvCache::try_store_key` / `try_store_value` (f16) |
//!
//! # The Hadamard rotation is fused into the producer of each activation
//!
//! 401 of Bonsai 2's 402 matrices are Hadamard-folded: their input must be
//! rotated (`signs`, then a normalised blockwise FWHT-1024) before the
//! matmul. The rotation is fused into the kernel that *produces* the matmul's
//! input — the RMSNorm, the SwiGLU, the sigmoid output gate and the gated
//! RMSNorm — so no activation ever takes a separate rotation dispatch and
//! each is rotated exactly once however many matrices consume it (q/k/v or
//! qkv/gate share one rotation, as do gate/up). The alternative, rotating
//! inside each GEMV's prologue, makes every threadgroup of every consuming
//! GEMV redo the whole transform; measured on the M3 with the real shapes
//! (`attn_qkv` 5120→10240, `attn_q` 5120→12288, `ssm_out` 6144→5120,
//! `attn_k` 5120→1024) it costs 298/361/195/53 µs against 253/299/160/37 µs
//! for producer-fused rotation plus a plain GEMV, with bit-identical output.
//!
//! The FWHT runs in threadgroup memory with the CPU's exact pass order
//! (`len = 1, 2, 4, …`, `u + v` / `u - v`), after the CPU's fused
//! `x * sign * (1/sqrt(block))` load pass with the scale passed in from the
//! host, so a rotation of identical input is bitwise identical to
//! `hadamard::fwht_forward_signed`.
//!
//! # Weights are read in their on-disk (AoS) block layout
//!
//! The GEMVs read every format's blocks exactly as the GGUF stores them, so
//! a memory-mapped model can be bound zero-copy. Measured on the M3 at the
//! 27B's shapes, the `d`-first 34-byte `PQ2_0` blocks read through 16-bit
//! loads stream at the same 55–59 GB/s (≈ 200–220 G weights/s) as a
//! re-laid-out SoA copy. `PTQ1_0` decodes natively at 41–53 GB/s of its
//! 28-byte blocks (≈ 185–240 G weights/s, on par with `PQ2_0` per weight;
//! see the two-pass lane layout of `q35_ptq1_rows`). The lossless transcode to a
//! 34-byte 2-bit format would move 18 % more bytes per token for no decode
//! gain, cost a load-time pass and give up the zero-copy binding — the
//! whole point of the smaller format on a unified-memory machine — so the
//! native kernel is the `PTQ1_0` path.
//!
//! # `PTQ1_0` decode
//!
//! A `PTQ1_0` byte `q` packs trits `c_n = ((q·3ⁿ mod 256)·3) >> 8`. With
//! `F_n = floor(q·3ⁿ/256)` (a product of an integer and a power-of-two
//! fraction, so exact in `f32`), `c_n = F_{n+1} − 3·F_n` — one multiply, one
//! `floor` and one `fma` per trit plane, all exact — and a lane's dot
//! product is `Σ c_n·x_n − Σ x` (values are `code − 1`). The telescoped
//! form `Σ F_n·(x_{n−1} − 3·x_n) − Σ x` saves the `fma` but sums terms up to
//! `242·|x|`: measured at the real shapes it is ≈ 5 % faster and ≈ 25×
//! less accurate (max |Δ| 7e-6–1.5e-5 against 3e-7–5e-7 for exact codes),
//! so the codes are recovered exactly.

/// Helpers shared by every other Qwen3.5 constant: the threadgroup FWHT,
/// the activation functions and the weight-format tags. Carries no entry
/// point of its own; it is concatenated before the other `MSL_QWEN35_*`
/// sources in the combined library.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_QWEN35_COMMON: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Largest span (Hadamard block, or unfolded chunk) one threadgroup rotates.
constant constexpr uint Q35_MAX_SPAN = 1024u;
// Largest row a single-threadgroup RMSNorm stages in threadgroup memory
// (24 KiB of the 32 KiB budget).
constant constexpr uint Q35_MAX_ROW = 6144u;
// Largest Gated-DeltaNet key/value head width (the state rows each lane
// streams hold at most four elements per lane).
constant constexpr uint Q35_GDN_MAX_HEAD = 128u;

// Weight formats of the GEMV kernels (one 128-element super-block each).
constant constexpr uint Q35_FMT_PQ2 = 0u;    // PQ2_0: 34 B, d first, code - 1 (0b11 = +2)
constant constexpr uint Q35_FMT_TQ2 = 1u;    // TQ2_0_g128: 34 B, qs first, d last, 0b11 = 0
constant constexpr uint Q35_FMT_Q2G64 = 2u;  // Q2_0_g64: two 18 B blocks, d first, code - 1
constant constexpr uint Q35_FMT_Q1 = 3u;     // Q1_0_g128: 18 B, d first, bit set = +d

inline float q35_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float q35_silu(float x) { return x * q35_sigmoid(x); }
inline float q35_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// In-place unnormalised blockwise FWHT over `n` floats of threadgroup
// memory, `n / block` independent `block`-wide transforms, pass order
// `len = 1, 2, 4, ...` exactly as the CPU runs it. The caller has already
// staged `x * sign * scale` and issued a barrier; every pass ends with one.
inline void q35_fwht_tg(threadgroup float* xs, uint n, uint block, uint tid, uint tsz) {
    if (block < 2u) {
        return;
    }
    const uint half_n = n >> 1;
    const uint half_b = block >> 1;
    const uint lg_half_b = ctz(half_b);
    for (uint len = 1u; len < block; len <<= 1) {
        const uint lg = ctz(len);
        for (uint t = tid; t < half_n; t += tsz) {
            const uint blk = t >> lg_half_b;
            const uint j = t & (half_b - 1u);
            const uint pos = blk * block + ((j >> lg) << (lg + 1u)) + (j & (len - 1u));
            const float a = xs[pos];
            const float b = xs[pos + len];
            xs[pos] = a + b;
            xs[pos + len] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// The rotation-fused producers and the standalone rotation:
///
/// * `q35_fwht_signed` — standalone blockwise FWHT of `rows` rows of
///   `width`: forward (`signs`, then FWHT) or inverse (FWHT, then `signs`).
///   Grid `[width / block, rows, 1]`, `block` threads (≤ 1024).
/// * `q35_rmsnorm_rotate` — RMSNorm of each row into `normed`, and when
///   `block != 0` the rotated copy into `rotated`. Grid `[rows, 1, 1]`,
///   up to 1024 threads; `n <= 6144`.
/// * `q35_swiglu_rotate` — `silu(gate) * up`, rotated per `span`.
/// * `q35_sigmoid_gate_rotate` — `attn * sigmoid(gate)` with the gate read
///   from the `[q | gate]` interleave of `attn_q`'s output, rotated per span.
/// * `q35_gated_norm_rotate` — the Gated-DeltaNet output norm
///   `w * RMSNorm(o_head) * silu(z_head)` per v-head, with `z` read in the
///   GGUF's tiled v-head order through the index map
///   `tiled(m) = (m % rep) * n_k_heads + m / rep`, rotated per span.
///
/// The three span kernels run one threadgroup per `(span, row)` with
/// `span` threads: `span == block` when the checkpoint is folded (`block !=
/// 0`), otherwise any chunk width ≤ 1024 and the output is written
/// unrotated. `span` must divide `n`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_QWEN35_ROTATE: &str = r#"
kernel void q35_fwht_signed(
    device float* x              [[buffer(0)]],
    device const float* signs    [[buffer(1)]],
    constant uint& width         [[buffer(2)]],
    constant uint& block         [[buffer(3)]],
    constant uint& inverse       [[buffer(4)]],
    constant float& scale        [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tsz2 [[threads_per_threadgroup]])
{
    threadgroup float xs[Q35_MAX_SPAN];
    const uint tsz = tsz2.x;
    const uint col0 = tg.x * block;
    device float* row = x + tg.y * width + col0;
    for (uint i = tid; i < block; i += tsz) {
        const float v = row[i];
        xs[i] = (inverse == 0u) ? (v * signs[col0 + i] * scale) : (v * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q35_fwht_tg(xs, block, block, tid, tsz);
    for (uint i = tid; i < block; i += tsz) {
        const float v = xs[i];
        row[i] = (inverse == 0u) ? v : (v * signs[col0 + i]);
    }
}

kernel void q35_rmsnorm_rotate(
    device const float* x        [[buffer(0)]],
    device const float* w        [[buffer(1)]],
    device float* normed         [[buffer(2)]],
    device float* rotated        [[buffer(3)]],
    device const float* signs    [[buffer(4)]],
    constant uint& n             [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    constant uint& block         [[buffer(7)]],
    constant float& scale        [[buffer(8)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tsz  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float xs[Q35_MAX_ROW];
    threadgroup float partial[32];
    device const float* xr = x + row * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tsz) {
        const float v = xr[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0u) {
        partial[sgid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (tsz + 31u) / 32u;
    float total = 0.0f;
    for (uint s = 0u; s < nsg; ++s) {
        total += partial[s];
    }
    const float inv = 1.0f / sqrt(total / float(n) + eps);
    device float* nr = normed + row * n;
    for (uint i = tid; i < n; i += tsz) {
        const float v = w[i] * (xr[i] * inv);
        nr[i] = v;
        if (block != 0u) {
            xs[i] = v * signs[i] * scale;
        }
    }
    if (block == 0u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q35_fwht_tg(xs, n, block, tid, tsz);
    device float* rr = rotated + row * n;
    for (uint i = tid; i < n; i += tsz) {
        rr[i] = xs[i];
    }
}

kernel void q35_swiglu_rotate(
    device const float* gate     [[buffer(0)]],
    device const float* up       [[buffer(1)]],
    device float* out            [[buffer(2)]],
    device const float* signs    [[buffer(3)]],
    constant uint& n             [[buffer(4)]],
    constant uint& block         [[buffer(5)]],
    constant float& scale        [[buffer(6)]],
    constant uint& span          [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tsz2 [[threads_per_threadgroup]])
{
    threadgroup float xs[Q35_MAX_SPAN];
    const uint tsz = tsz2.x;
    const uint col0 = tg.x * span;
    const uint base = tg.y * n + col0;
    for (uint i = tid; i < span; i += tsz) {
        const float v = q35_silu(gate[base + i]) * up[base + i];
        if (block == 0u) {
            out[base + i] = v;
        } else {
            xs[i] = v * signs[col0 + i] * scale;
        }
    }
    if (block == 0u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q35_fwht_tg(xs, span, block, tid, tsz);
    for (uint i = tid; i < span; i += tsz) {
        out[base + i] = xs[i];
    }
}

kernel void q35_sigmoid_gate_rotate(
    device const float* attn     [[buffer(0)]],
    device const float* q_all    [[buffer(1)]],
    device float* out            [[buffer(2)]],
    device const float* signs    [[buffer(3)]],
    constant uint& n             [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant uint& block         [[buffer(6)]],
    constant float& scale        [[buffer(7)]],
    constant uint& span          [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tsz2 [[threads_per_threadgroup]])
{
    threadgroup float xs[Q35_MAX_SPAN];
    const uint tsz = tsz2.x;
    const uint col0 = tg.x * span;
    const uint row = tg.y;
    for (uint i = tid; i < span; i += tsz) {
        const uint col = col0 + i;
        const uint h = col / head_dim;
        const uint d = col - h * head_dim;
        const float g = q_all[row * 2u * n + h * 2u * head_dim + head_dim + d];
        const float v = attn[row * n + col] * q35_sigmoid(g);
        if (block == 0u) {
            out[row * n + col] = v;
        } else {
            xs[i] = v * signs[col] * scale;
        }
    }
    if (block == 0u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q35_fwht_tg(xs, span, block, tid, tsz);
    for (uint i = tid; i < span; i += tsz) {
        out[row * n + col0 + i] = xs[i];
    }
}

kernel void q35_gated_norm_rotate(
    device const float* o        [[buffer(0)]],
    device const float* z        [[buffer(1)]],
    device const float* w        [[buffer(2)]],
    device float* out            [[buffer(3)]],
    device const float* signs    [[buffer(4)]],
    constant uint& n             [[buffer(5)]],
    constant uint& head_v_dim    [[buffer(6)]],
    constant uint& n_k_heads     [[buffer(7)]],
    constant uint& rep           [[buffer(8)]],
    constant float& eps          [[buffer(9)]],
    constant uint& block         [[buffer(10)]],
    constant float& scale        [[buffer(11)]],
    constant uint& span          [[buffer(12)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    // One thread per element (`span` threads, `span % head_v_dim == 0`,
    // `head_v_dim % 32 == 0`): every simdgroup lies inside one v-head.
    threadgroup float xs[Q35_MAX_SPAN];
    threadgroup float partial[Q35_MAX_SPAN / 32u];
    const uint col = tg.x * span + tid;
    const uint row = tg.y;
    const float v = o[row * n + col];
    const float sq = simd_sum(v * v);
    if (lane == 0u) {
        partial[sgid] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg_per_head = head_v_dim / 32u;
    const uint first_sg = (sgid / sg_per_head) * sg_per_head;
    float total = 0.0f;
    for (uint s = 0u; s < sg_per_head; ++s) {
        total += partial[first_sg + s];
    }
    const float inv = 1.0f / sqrt(total / float(head_v_dim) + eps);
    const uint m = col / head_v_dim;
    const uint j = col - m * head_v_dim;
    const uint tiled = (m % rep) * n_k_heads + m / rep;
    const float gz = z[row * n + tiled * head_v_dim + j];
    const float r = w[j] * v * inv * q35_silu(gz);
    if (block == 0u) {
        out[row * n + col] = r;
        return;
    }
    xs[tid] = r * signs[col] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q35_fwht_tg(xs, span, block, tid, span);
    out[row * n + col] = xs[tid];
}
"#;

/// Quantized and `f32` GEMV kernels.
///
/// `y[c][row] = Σ_k W[row][k] · x[c][k]` for `c < m_cols` (`x` is `[m][k]`,
/// `y` is `[m][n_rows]`), or `y[c][row] += …` when `accumulate != 0` — the
/// residual add fused into the output projections' epilogue. Each simdgroup
/// owns `R` rows and each lane a 16-element slice of every 128-element
/// super-block (8 lanes per super-block, 4 super-blocks per step), so the
/// `x` slice a lane loads is reused for all `R` rows.
///
/// Entry points (`<fmt>` ∈ `pq2`, `tq2`, `q2g64`, `q1`, `ptq1`, `f32`):
/// `q35_gemv_<fmt>`, 4 simdgroups × 2 rows = 8 rows of **one** column per
/// threadgroup (128 threads); grid `[ceil(n_rows / 8), m_cols, 1]`, so a
/// prefill runs every token's column in one dispatch with exactly the
/// decode arithmetic. Measured at the 27B's shapes, decoding each weight
/// once for 2, 4 or 8 columns per threadgroup instead ran at 0.7–1.4×,
/// 0.5–0.6× and 0.3–0.7× this kernel's per-token rate: the per-column `x`
/// reads, not the weight stream, bound those variants, so blocking columns
/// needs a tiled GEMM rather than a wider GEMV.
/// Buffers: weights (0, at the matrix's byte offset), `x` (1), `y` (2);
/// scalars `n_rows` (3), `k` (4), `m_cols` (5), `accumulate` (6). `k` must be
/// a multiple of 128.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_QWEN35_GEMV: &str = r#"
constant constexpr uint Q35_GEMV_R = 2u;
constant constexpr uint Q35_GEMV_NSG = 4u;

inline float4 q35_codes4(uint b) {
    return float4(float(b & 3u), float((b >> 2) & 3u), float((b >> 4) & 3u), float((b >> 6) & 3u));
}

inline float4 q35_tq2_4(uint b) {
    const float4 c = q35_codes4(b);
    return select(select(float4(0.0f), float4(-1.0f), c == 0.0f), float4(1.0f), c == 2.0f);
}

inline float4 q35_bits4(uint b) {
    return float4(float(b & 1u), float((b >> 1) & 1u), float((b >> 2) & 1u), float((b >> 3) & 1u)) * 2.0f - 1.0f;
}

inline uint q35_load_u32(device const uchar* p) {
    device const ushort* h = (device const ushort*)p;
    return uint(h[0]) | (uint(h[1]) << 16);
}

// Decode lane `sub`'s 16 elements of one 128-element super-block starting
// at `sb` into `v[0..4]`, returning the slice's scale.
template <uint FMT>
inline float q35_decode16(device const uchar* sb, uint sub, thread float4* v) {
    if (FMT == Q35_FMT_PQ2) {
        const uint q = q35_load_u32(sb + 2u + sub * 4u);
        v[0] = q35_codes4(q & 0xFFu) - 1.0f;
        v[1] = q35_codes4((q >> 8) & 0xFFu) - 1.0f;
        v[2] = q35_codes4((q >> 16) & 0xFFu) - 1.0f;
        v[3] = q35_codes4(q >> 24) - 1.0f;
        return float(*(device const half*)sb);
    } else if (FMT == Q35_FMT_TQ2) {
        const uint q = q35_load_u32(sb + sub * 4u);
        v[0] = q35_tq2_4(q & 0xFFu);
        v[1] = q35_tq2_4((q >> 8) & 0xFFu);
        v[2] = q35_tq2_4((q >> 16) & 0xFFu);
        v[3] = q35_tq2_4(q >> 24);
        return float(*(device const half*)(sb + 32u));
    } else if (FMT == Q35_FMT_Q2G64) {
        device const uchar* blk = sb + (sub >> 2) * 18u;
        const uint q = q35_load_u32(blk + 2u + (sub & 3u) * 4u);
        v[0] = q35_codes4(q & 0xFFu) - 1.0f;
        v[1] = q35_codes4((q >> 8) & 0xFFu) - 1.0f;
        v[2] = q35_codes4((q >> 16) & 0xFFu) - 1.0f;
        v[3] = q35_codes4(q >> 24) - 1.0f;
        return float(*(device const half*)blk);
    } else {
        const uint q = uint(*(device const ushort*)(sb + 2u + sub * 2u));
        v[0] = q35_bits4(q & 0xFu);
        v[1] = q35_bits4((q >> 4) & 0xFu);
        v[2] = q35_bits4((q >> 8) & 0xFu);
        v[3] = q35_bits4(q >> 12);
        return float(*(device const half*)sb);
    }
}

template <uint FMT>
inline void q35_gemv_slice16(device const uchar* w, device const float* x, device float* y,
                             uint n_rows, uint k, uint m_cols, uint accumulate,
                             uint2 tg, uint sgid, uint lane) {
    const uint sb_bytes = (FMT == Q35_FMT_Q2G64) ? 36u : ((FMT == Q35_FMT_Q1) ? 18u : 34u);
    const uint row0 = (tg.x * Q35_GEMV_NSG + sgid) * Q35_GEMV_R;
    const uint col = tg.y;
    if (row0 >= n_rows || col >= m_cols) {
        return;
    }
    const uint nsb = k / 128u;
    const uint blk_in = lane >> 3;
    const uint sub = lane & 7u;
    device const float4* xc = (device const float4*)(x + col * k);
    float acc[Q35_GEMV_R];
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        acc[r] = 0.0f;
    }
    for (uint b0 = 0u; b0 < nsb; b0 += 4u) {
        const uint sb = b0 + blk_in;
        if (sb < nsb) {
            device const float4* xs = xc + sb * 32u + sub * 4u;
            const float4 xa = xs[0], xb = xs[1], xcc = xs[2], xd = xs[3];
            for (uint r = 0u; r < Q35_GEMV_R; ++r) {
                const uint row = min(row0 + r, n_rows - 1u);
                float4 wv[4];
                const float wd = q35_decode16<FMT>(w + (row * nsb + sb) * sb_bytes, sub, wv);
                const float s = dot(wv[0], xa) + dot(wv[1], xb) + dot(wv[2], xcc) + dot(wv[3], xd);
                acc[r] = fma(wd, s, acc[r]);
            }
        }
    }
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        const float t = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < n_rows) {
            device float* dst = y + col * n_rows + row0 + r;
            *dst = (accumulate != 0u) ? (*dst + t) : t;
        }
    }
}

// PTQ1_0 decodes 20 elements from one 4-byte `qs` word (5 trit planes of a
// uchar4) but only 8 from the 2 `qh` bytes, so an 8-lanes-per-block split
// leaves a quarter of the lanes idle or padded. Two passes keep every lane
// on the same code instead:
//
// 1. `qs`: 5 super-blocks per step, lane `l < 30` owns word `l % 6` of block
//    `l / 6` — words 0..3 hold stage 1 (elements `16 n + 4 w + [0, 4)` for
//    trit `n`), words 4..5 stage 2 (elements `80 + 8 n + 4 (w − 4) +
//    [0, 4)`); lanes 30 and 31 idle.
// 2. `qh`: 16 super-blocks per step, lane `l` owns byte `l % 2` of block
//    `l / 2` — elements `120 + 2 n + (l % 2)` for trits `n = 0..3`.
//
// Measured at the 27B's shapes this streams 185–240 G weights/s, on par with
// `PQ2_0` per weight, against 145–180 for the 8-lane split.
inline void q35_ptq1_rows(device const uchar* w, device const float* x, device float* y,
                          uint n_rows, uint k, uint m_cols, uint accumulate,
                          uint2 tg, uint sgid, uint lane) {
    const uint row0 = (tg.x * Q35_GEMV_NSG + sgid) * Q35_GEMV_R;
    const uint col = tg.y;
    if (row0 >= n_rows || col >= m_cols) {
        return;
    }
    const uint nsb = k / 128u;
    const float s1 = 3.0f / 256.0f, s2 = 9.0f / 256.0f, s3 = 27.0f / 256.0f;
    const float s4 = 81.0f / 256.0f, s5 = 243.0f / 256.0f;
    device const float* xc = x + col * k;
    float acc[Q35_GEMV_R];
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        acc[r] = 0.0f;
    }

    // Pass 1: the `qs` words.
    const uint blk_in = lane / 6u;
    const uint word = lane - blk_in * 6u;
    const uint x_first = (word < 4u) ? word : (20u + (word - 4u));
    const uint x_step = (word < 4u) ? 4u : 2u;
    for (uint b0 = 0u; b0 < nsb; b0 += 5u) {
        const uint sb = b0 + blk_in;
        if (lane < 30u && sb < nsb) {
            device const float4* xs = (device const float4*)xc + sb * 32u + x_first;
            const float4 x0 = xs[0];
            const float4 x1 = xs[x_step];
            const float4 x2 = xs[2u * x_step];
            const float4 x3 = xs[3u * x_step];
            const float4 x4 = xs[4u * x_step];
            // Values are code - 1: sum(code * x) - sum(x).
            const float sx = dot(x0 + x1 + x2 + x3 + x4, float4(1.0f));
            for (uint r = 0u; r < Q35_GEMV_R; ++r) {
                const uint row = min(row0 + r, n_rows - 1u);
                device const uint* bp = (device const uint*)(w + (row * nsb + sb) * 28u);
                const float wd = float(as_type<half2>(bp[6]).y);
                const float4 q = float4(as_type<uchar4>(bp[word]));
                // F_{n+1} = floor(q * 3^(n+1) / 256), exact in f32, and the
                // code of trit n is F_{n+1} - 3 F_n (F_0 = 0), an exact small
                // integer.
                const float4 f1 = floor(q * s1);
                const float4 f2 = floor(q * s2);
                const float4 f3 = floor(q * s3);
                const float4 f4 = floor(q * s4);
                const float4 f5 = floor(q * s5);
                const float s = dot(f1, x0) + dot(fma(f1, -3.0f, f2), x1)
                              + dot(fma(f2, -3.0f, f3), x2) + dot(fma(f3, -3.0f, f4), x3)
                              + dot(fma(f4, -3.0f, f5), x4) - sx;
                acc[r] = fma(wd, s, acc[r]);
            }
        }
    }

    // Pass 2: the `qh` bytes (four trits each).
    const uint h_blk = lane >> 1;
    const uint h_byte = lane & 1u;
    for (uint b0 = 0u; b0 < nsb; b0 += 16u) {
        const uint sb = b0 + h_blk;
        if (sb < nsb) {
            device const float* xh = xc + sb * 128u + 120u + h_byte;
            const float4 xv = float4(xh[0], xh[2], xh[4], xh[6]);
            const float sx = (xv.x + xv.y) + (xv.z + xv.w);
            for (uint r = 0u; r < Q35_GEMV_R; ++r) {
                const uint row = min(row0 + r, n_rows - 1u);
                device const uint* bp = (device const uint*)(w + (row * nsb + sb) * 28u);
                const uint w6 = bp[6];
                const float wd = float(as_type<half2>(w6).y);
                const float q = float((w6 >> (8u * h_byte)) & 0xFFu);
                const float4 fq = floor(q * float4(s1, s2, s3, s4));
                const float4 code = fma(float4(0.0f, fq.x, fq.y, fq.z), -3.0f, fq);
                acc[r] = fma(wd, dot(code, xv) - sx, acc[r]);
            }
        }
    }
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        const float t = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < n_rows) {
            device float* dst = y + col * n_rows + row0 + r;
            *dst = (accumulate != 0u) ? (*dst + t) : t;
        }
    }
}

// Dense f32 rows (`k` a multiple of 128): lane `l` reads float4 `l` of each
// 128-wide chunk.
inline void q35_dense_rows(device const uchar* w, device const float* x, device float* y,
                           uint n_rows, uint k, uint m_cols, uint accumulate,
                           uint2 tg, uint sgid, uint lane) {
    const uint row0 = (tg.x * Q35_GEMV_NSG + sgid) * Q35_GEMV_R;
    const uint col = tg.y;
    if (row0 >= n_rows || col >= m_cols) {
        return;
    }
    device const float4* w4 = (device const float4*)w;
    device const float4* x4 = (device const float4*)(x + col * k);
    const uint k4 = k / 4u;
    float acc[Q35_GEMV_R];
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        acc[r] = 0.0f;
    }
    for (uint i = lane; i < k4; i += 32u) {
        const float4 xv = x4[i];
        for (uint r = 0u; r < Q35_GEMV_R; ++r) {
            const uint row = min(row0 + r, n_rows - 1u);
            acc[r] += dot(w4[row * k4 + i], xv);
        }
    }
    for (uint r = 0u; r < Q35_GEMV_R; ++r) {
        const float t = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < n_rows) {
            device float* dst = y + col * n_rows + row0 + r;
            *dst = (accumulate != 0u) ? (*dst + t) : t;
        }
    }
}

kernel void q35_gemv_pq2(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_gemv_slice16<Q35_FMT_PQ2>(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}

kernel void q35_gemv_tq2(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_gemv_slice16<Q35_FMT_TQ2>(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}

kernel void q35_gemv_q2g64(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_gemv_slice16<Q35_FMT_Q2G64>(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}

kernel void q35_gemv_q1(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_gemv_slice16<Q35_FMT_Q1>(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}

kernel void q35_gemv_ptq1(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_ptq1_rows(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}

kernel void q35_gemv_f32(
    device const uchar* w        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant uint& n_rows        [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& m_cols        [[buffer(5)]],
    constant uint& accumulate    [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    q35_dense_rows(w, x, y, n_rows, k, m_cols, accumulate, tg, sgid, lane);
}
"#;

/// The Gated-DeltaNet layer's recurrent kernels and the batched KV store.
///
/// * `q35_conv1d_silu` — depthwise causal conv (kernel 4) over all
///   `conv_dim` channels of `t_len` tokens, carrying the per-channel window
///   `state [conv_dim][3]` (oldest first), then SiLU. One thread per
///   channel, tokens in order; the four taps reduce exactly as the CPU's
///   `conv_dot4` does (one multiply then three fused multiply-adds, tap 0
///   first), so the pre-SiLU value is bitwise the CPU's.
/// * `q35_gdn` — the gated delta rule for one v-head per threadgroup
///   (grouped order `m`, reading k-head `m / rep` and the tiled GGUF rows at
///   `tiled(m)`), `t_len` tokens in order: L2-normalise q and k
///   (`x / max(sqrt(Σx²), eps)`), `β = sigmoid(b)`, `g = a_neg ·
///   softplus(a + dt_bias)`, then for every state row `j` in one pass
///   `s = S_j · exp(g)`, `δ = (v_j − s·k) · β`, `S_j = s + k δ`,
///   `o_j = (S_j · q) · scale`. The host passes `scale` as the CPU
///   recurrence's `GdnDims::out_scale` (`1/sqrt(head_v_dim)`) and admits
///   only square states (`head_k_dim == head_v_dim`, which the reference
///   implementation requires as `S_k == S_v`), where that is also the
///   reference's `1/sqrt(S_k)`.
///
///   **State strategy.** A v-head's `128 × 128` f32 state is 64 KiB, twice
///   the M3's measured 32 KiB `maxThreadgroupMemoryLength`, so it is never
///   staged: it stays in device memory in the CPU's transposed layout
///   (`slab[j * head_k_dim + i] == S[i][j]`) and each simdgroup streams its
///   own rows — row `j` belongs to simdgroup `j % 8` for every token, 32
///   lanes read it as four coalesced 128-byte runs into registers, apply the
///   decay, the delta update and the output dot, and write it back once.
///   Only the normalised q and k (1 KiB) live in threadgroup memory. The
///   slab is therefore read and written exactly once per token, like the
///   CPU's fused path, and no reduction ever crosses simdgroups.
/// * `q35_kv_store` — store `t_len` tokens' post-RoPE keys and raw values
///   into one KV slot of the `f16` cache at positions `pos0 + t`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_QWEN35_SSM: &str = r#"
kernel void q35_conv1d_silu(
    device const float* x        [[buffer(0)]],
    device const float* w        [[buffer(1)]],
    device float* state          [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& channels      [[buffer(4)]],
    constant uint& t_len         [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= channels) {
        return;
    }
    float s0 = state[c * 3u];
    float s1 = state[c * 3u + 1u];
    float s2 = state[c * 3u + 2u];
    const float w0 = w[c * 4u];
    const float w1 = w[c * 4u + 1u];
    const float w2 = w[c * 4u + 2u];
    const float w3 = w[c * 4u + 3u];
    for (uint t = 0u; t < t_len; ++t) {
        const float xv = x[t * channels + c];
        const float yv = fma(xv, w3, fma(s2, w2, fma(s1, w1, s0 * w0)));
        out[t * channels + c] = q35_silu(yv);
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    state[c * 3u] = s0;
    state[c * 3u + 1u] = s1;
    state[c * 3u + 2u] = s2;
}

kernel void q35_gdn(
    device const float* conv_out [[buffer(0)]],
    device const float* ab       [[buffer(1)]],
    device const float* a_neg    [[buffer(2)]],
    device const float* dt_bias  [[buffer(3)]],
    device float* state          [[buffer(4)]],
    device float* out            [[buffer(5)]],
    constant uint& n_k_heads     [[buffer(6)]],
    constant uint& n_v_heads     [[buffer(7)]],
    constant uint& head_k_dim    [[buffer(8)]],
    constant uint& head_v_dim    [[buffer(9)]],
    constant uint& conv_dim      [[buffer(10)]],
    constant uint& t_len         [[buffer(11)]],
    constant float& eps          [[buffer(12)]],
    constant float& out_scale    [[buffer(13)]],
    uint m     [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tsz   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup float qn[Q35_GDN_MAX_HEAD];
    threadgroup float kn[Q35_GDN_MAX_HEAD];
    threadgroup float red_q[32];
    threadgroup float red_k[32];
    const uint nk = n_k_heads;
    const uint nv = n_v_heads;
    const uint hk = head_k_dim;
    const uint hv = head_v_dim;
    const uint rep = nv / nk;
    const uint kh = m / rep;
    const uint jt = (m % rep) * nk + kh;
    const uint nsg = tsz / 32u;
    device float* slab = state + m * hv * hk;
    for (uint t = 0u; t < t_len; ++t) {
        device const float* row = conv_out + t * conv_dim;
        float qv = 0.0f;
        float kv = 0.0f;
        if (tid < hk) {
            qv = row[kh * hk + tid];
            kv = row[nk * hk + kh * hk + tid];
        }
        const float sq = simd_sum(qv * qv);
        const float sk = simd_sum(kv * kv);
        if (lane == 0u) {
            red_q[sgid] = sq;
            red_k[sgid] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tq = 0.0f;
        float tk = 0.0f;
        for (uint s = 0u; s < nsg; ++s) {
            tq += red_q[s];
            tk += red_k[s];
        }
        const float iq = 1.0f / max(sqrt(tq), eps);
        const float ik = 1.0f / max(sqrt(tk), eps);
        if (tid < hk) {
            qn[tid] = qv * iq;
            kn[tid] = kv * ik;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float a = ab[t * 2u * nv + jt];
        const float b = ab[t * 2u * nv + nv + jt];
        const float decay = exp(a_neg[m] * q35_softplus(a + dt_bias[m]));
        const float beta = q35_sigmoid(b);
        device const float* v = row + 2u * nk * hk + jt * hv;
        device float* o = out + t * nv * hv + m * hv;
        for (uint j = sgid; j < hv; j += nsg) {
            device float* srow = slab + j * hk;
            float sv[4];
            float sum = 0.0f;
            for (uint c = 0u; c < 4u; ++c) {
                const uint i = lane + 32u * c;
                sv[c] = 0.0f;
                if (i < hk) {
                    const float s = srow[i] * decay;
                    sv[c] = s;
                    sum = fma(s, kn[i], sum);
                }
            }
            sum = simd_sum(sum);
            const float delta = (v[j] - sum) * beta;
            float acc = 0.0f;
            for (uint c = 0u; c < 4u; ++c) {
                const uint i = lane + 32u * c;
                if (i < hk) {
                    const float s = fma(kn[i], delta, sv[c]);
                    srow[i] = s;
                    acc = fma(s, qn[i], acc);
                }
            }
            acc = simd_sum(acc);
            if (lane == 0u) {
                o[j] = acc * out_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void q35_kv_store(
    device const float* k        [[buffer(0)]],
    device const float* v        [[buffer(1)]],
    device half* k_cache         [[buffer(2)]],
    device half* v_cache         [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& nkv           [[buffer(5)]],
    constant uint& max_seq       [[buffer(6)]],
    constant uint& pos0          [[buffer(7)]],
    constant ulong& slot_offset  [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint head = gid.y;
    const uint t = gid.z;
    if (d >= head_dim || head >= nkv) {
        return;
    }
    const uint src = (t * nkv + head) * head_dim + d;
    const ulong dst = slot_offset
        + (ulong(head) * ulong(max_seq) + ulong(pos0 + t)) * ulong(head_dim) + ulong(d);
    k_cache[dst] = half(k[src]);
    v_cache[dst] = half(v[src]);
}
"#;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;

    /// Every entry point the hybrid encoder resolves by name is declared in
    /// one of this module's constants, at the start of a line (the MET-12
    /// runtime verifier and the build script's scan both depend on it).
    #[test]
    fn every_hybrid_entry_point_is_declared() {
        let all =
            format!("{MSL_QWEN35_COMMON}{MSL_QWEN35_ROTATE}{MSL_QWEN35_GEMV}{MSL_QWEN35_SSM}");
        for name in [
            "q35_fwht_signed",
            "q35_rmsnorm_rotate",
            "q35_swiglu_rotate",
            "q35_sigmoid_gate_rotate",
            "q35_gated_norm_rotate",
            "q35_conv1d_silu",
            "q35_gdn",
            "q35_kv_store",
        ] {
            assert!(
                all.lines()
                    .any(|l| l.trim_start().starts_with(&format!("kernel void {name}("))),
                "{name} is not declared"
            );
        }
        for fmt in ["pq2", "tq2", "q2g64", "q1", "ptq1", "f32"] {
            let entry = format!("kernel void q35_gemv_{fmt}(");
            assert!(
                MSL_QWEN35_GEMV.lines().any(|l| l.starts_with(&entry)),
                "{entry} missing"
            );
        }
        // Helpers only: the common prelude declares no entry point.
        assert!(!MSL_QWEN35_COMMON.contains("kernel void"));
        // The batched-prefill GEMMs: one per GEMV format, plus the
        // half-activation twins of the five quantized formats, each at the
        // start of a line (the runtime verifier's line-anchored scan).
        let gemm = crate::gpu_backend::kernel_sources::MSL_QWEN35_GEMM;
        for fmt in ["pq2", "tq2", "q2g64", "q1", "ptq1", "f32"] {
            let entry = format!("kernel void q35_gemm_{fmt}(");
            assert!(
                gemm.lines().any(|l| l.starts_with(&entry)),
                "{entry} missing"
            );
        }
        for fmt in ["pq2", "tq2", "q2g64", "q1", "ptq1"] {
            let entry = format!("kernel void q35_gemm_{fmt}_ha(");
            assert!(
                gemm.lines().any(|l| l.starts_with(&entry)),
                "{entry} missing"
            );
        }
        assert!(
            !gemm.contains("q35_gemm_f32_ha"),
            "f32 weights are never staged as half"
        );
        // Every `kernel void` of the GEMM source is one of those eleven.
        assert_eq!(gemm.matches("kernel void ").count(), 11);
    }

    /// The raw-string terminator must not appear inside any source (the
    /// build script ends each constant at the first `"#`).
    #[test]
    fn no_source_contains_a_raw_string_terminator() {
        for src in [
            MSL_QWEN35_COMMON,
            MSL_QWEN35_ROTATE,
            MSL_QWEN35_GEMV,
            crate::gpu_backend::kernel_sources::MSL_QWEN35_GEMM,
            MSL_QWEN35_SSM,
        ] {
            assert!(!src.contains("\"#"));
        }
    }
}
