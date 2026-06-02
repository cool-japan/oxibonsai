//! Flash-attention `simdgroup_matrix` Metal kernel for the FLUX.2 DiT joint
//! attention (`joint_attention_flash_f32`).
//!
//! A **flash-attention v2** (online-softmax) kernel driving Apple's
//! `simdgroup_float8x8` hardware matrix units (8×8×8 MACs, f32 accumulate) for
//! **both** attention matmuls — `S = scale·Q·Kᵀ` and `O += P·V` — instead of
//! scalar dot products. A naive scalar variant (one threadgroup per
//! `(query-row, head)`, no matrix units) reached only ~65 GFLOP/s and *lost* to
//! the real rayon+NEON CPU `joint_attention`; this kernel exists to beat the CPU
//! and is the shipping DiT attention path.
//!
//! Matches the CPU reference `oxibonsai-image/src/math.rs::joint_attention`
//! (and the local parity port in `metal_graph/tests_dit_attention.rs`) in
//! behaviour:
//!
//!   per head `h` (H = 24):
//!     S[qi,ki] = (Σ_d q[h,qi,d]·k[h,ki,d]) · scale,  scale = 1/sqrt(head_dim)
//!     softmax over ki (NON-causal, max-subtraction stabilized)
//!     O[qi,d]  = Σ_ki softmax(S)[qi,ki] · v[h,ki,d]
//!   then transpose to token-major: out[qi, h*head_dim + d] = O[h,qi,d]
//!
//! Layout (identical to the naive kernel):
//!   - `q`/`k`/`v`: head-major `[num_heads, seq, head_dim]` f32 (post-RoPE).
//!   - `out`: token-major `[seq, num_heads*head_dim]` f32 (transpose folded in).
//!
//! # Flash-attention v2 structure
//!
//! Per threadgroup = one `(head h, query-tile of FA_BQ rows)`; grid
//! `[ceil(seq/FA_BQ), num_heads, 1]`. `FA_SIMDGROUPS = 8` simdgroups
//! (256 threads) — each owns `FA_BQ / FA_SIMDGROUPS = 8` query rows (one 8×8
//! M-fragment), tuned up from 4 simdgroups (a measured ~1.5× speedup: halving the
//! per-simdgroup O-fragment count relieves register pressure and lifts
//! occupancy). The online-softmax loop streams `FA_BK`-key tiles:
//!
//! 1. Init per-query running max `m = -inf`, running sum `l = 0`, and the output
//!    accumulator `O[FA_BQ, head_dim]` (held in `simdgroup_float8x8` fragments,
//!    **not** threadgroup memory — `64×128×4 = 32 KiB` would exhaust the
//!    threadgroup budget on its own).
//! 2. For each key-tile of `FA_BK` keys, the loop body runs four sub-steps
//!    (a) Stage `Kᵀ` `[head_dim, FA_BK]` (transposed) into threadgroup memory so
//!    the `S = Q·Kᵀ` matmul reads `K` as the `[K=head_dim, N=FA_BK]` operand with
//!    a natural (non-transposed) `simdgroup_load`. `Q` fragments are loaded
//!    straight from **device** memory (the per-head q tensor is row-major
//!    `[seq, head_dim]`, so the `[FA_BQ, head_dim]` Q tile is contiguous with
//!    leading dimension `head_dim`) — no Q staging.
//!    (b) `S[FA_BQ, FA_BK] = scale · (Q · Kᵀ)` via `simdgroup_float8x8` MACs
//!    (`head_dim = 128 = 16` K-fragments of 8), f32 accumulate → threadgroup
//!    `Ssh`.
//!    (c) Online softmax (scalar, one query-row per lane-strided thread):
//!    `m_new = max(m, rowmax(S))`; `P = exp(S - m_new)` (written back over
//!    `Ssh`); `corr = exp(m - m_new)`; `l = l·corr + rowsum(P)`; the O
//!    accumulator is rescaled by `corr` (per query row) via a **diagonal
//!    `simdgroup_multiply`** (`O = diag(corr) · O`); `m = m_new`.
//!    (d) Stage `V` `[FA_BK, head_dim]` (natural row-major) into the same
//!    threadgroup region `Kᵀ` used (K is dead after sub-step b), then
//!    `O += P · V` via `simdgroup_float8x8` MACs (`FA_BK = 32 = 4` K-fragments of
//!    8), f32 accumulate.
//! 3. After all key-tiles: `O[qi,:] /= l[qi]` (folded into the writeback as a
//!    per-row scale), then scatter transposed: `out[qi, h*head_dim + d]`.
//!
//! The online-softmax is mathematically **exact** vs the CPU full-row softmax
//! (just reordered) — modulo f32 reassociation, so parity holds (expect max-abs
//! ≈ 1e-5..1e-6, cos ≥ 0.999). **Everything is f32** (Q,K,V,S,P,O, accumulate):
//! the `v9`/`v10` lesson is that f16 staging of full-range values drifts past the
//! `1e-3` parity bound; only ternary `code×scale` was f16-exact, which does not
//! apply here.
//!
//! # Threadgroup-memory budget (Apple M3 = 32 KiB hard limit)
//!
//!   `KVsh` (Kᵀ `[128,32]` ≡ V `[32,128]`, unioned — K dead before V staged) ... 16 KiB
//!   `Ssh`  (S/P `[64,32]`) ........................................................ 8 KiB
//!   `corr` / `mrow` / `lrow` / diagonal scratch .................................. < 2 KiB
//!   total ....................................................................... ≈ 26 KiB ✓
//!
//! Boundaries: partial query / key tiles are clamped with `min(...)` (out-of-range
//! rows/cols staged as zero so the matrix MACs still produce correct in-range
//! outputs and the softmax ignores the padding), so the kernel is correct for the
//! non-tile-multiple parity shapes (`seq = 8, 40, 50`). `head_dim` is required to
//! be a multiple of 8 (the matrix-unit edge); the DiT `head_dim = 128` and the
//! parity shapes (all 128) satisfy this — the dispatch layer validates it.

/// Query-tile height (rows of `out` per threadgroup). 64 rows = the `M` extent of
/// both matmuls' output tile; split across `FA_SIMDGROUPS` simdgroups.
/// Metal-only (consumed only by the gated dispatch layer).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const DIT_FLASH_BQ: usize = 64;

/// Key-tile width (keys processed per online-softmax step). 32 keys = the inner
/// `N` of `S` and the `K` of `P·V`. Metal-only.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const DIT_FLASH_BK: usize = 32;

/// Compile-time cap on `seq` accepted by the joint-attention dispatch /
/// validation layer (a generous upper bound; the DiT `seq = 1536` and all parity
/// shapes fit well within it). Metal-only (consumed only by the gated dispatch
/// layer in `metal_graph/graph.rs`).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const DIT_ATTN_MAX_SEQ: usize = 2048;

/// Compile-time cap on `head_dim` accepted by the joint-attention dispatch /
/// validation layer. 256 covers the DiT `head_dim = 128` with head-room without
/// exceeding the 1024-thread hardware limit. Metal-only (consumed only by the
/// gated dispatch layer in `metal_graph/graph.rs`).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const DIT_ATTN_MAX_HEAD_DIM: usize = 256;

/// MSL source for the flash-attention DiT joint-attention kernel
/// `joint_attention_flash_f32`.
///
/// Buffers (identical signature to `joint_attention_f32`):
///   - `q`        `[num_heads × seq × head_dim]` (f32, head-major) `[[buffer(0)]]`
///   - `k`        `[num_heads × seq × head_dim]` (f32, head-major) `[[buffer(1)]]`
///   - `v`        `[num_heads × seq × head_dim]` (f32, head-major) `[[buffer(2)]]`
///   - `out`      `[seq × (num_heads*head_dim)]` (f32, token-major) `[[buffer(3)]]`
///   - `num_heads` (u32 scalar) `[[buffer(4)]]`
///   - `seq`       (u32 scalar) `[[buffer(5)]]`
///   - `head_dim`  (u32 scalar) `[[buffer(6)]]`
///   - `scale`     (f32 scalar) `[[buffer(7)]]`
///
/// Dispatch: `[ceil(seq/FA_BQ), num_heads, 1]` threadgroups,
/// `[FA_SIMDGROUPS·32, 1, 1] = [128, 1, 1]` threads.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub const MSL_DIT_JOINT_ATTENTION_FLASH: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tile / simdgroup geometry (must match dispatch_joint_attention_flash in
// metal_dispatch.rs, and the DIT_FLASH_BQ / DIT_FLASH_BK Rust constants).
//
//   query tile (M)        : FA_BQ = 64  (rows of `out` per threadgroup)
//   key tile   (N of S)   : FA_BK = 32  (keys per online-softmax step)
//   simdgroups/threadgroup: FA_SIMDGROUPS = 8  (-> 256 threads)
//   hardware matrix edge   : FA_FRAG = 8
//   head_dim cap           : FA_DMAX = 256 (fragment-array sizing; DiT uses 128)
//
// S = Q·Kᵀ : each simdgroup owns FA_SG_M = FA_BQ/FA_SIMDGROUPS = 8 query rows
//            x FA_BK = 32 cols  ->  (8/8)x(32/8) = 1x4 = 4 S-fragments.
// O += P·V : same FA_SG_M = 8 query rows x head_dim cols
//            ->  1 x (head_dim/8) O-fragments  (16 for head_dim=128).
constant constexpr uint FA_BQ = 64u;
constant constexpr uint FA_BK = 32u;
constant constexpr uint FA_SIMDGROUPS = 8u;
constant constexpr uint FA_THREADS = FA_SIMDGROUPS * 32u;   // 256
constant constexpr uint FA_FRAG = 8u;                       // hardware 8x8 edge
constant constexpr uint FA_SG_M = FA_BQ / FA_SIMDGROUPS;    // 8 query rows / simdgroup
constant constexpr uint FA_MFRAGS = FA_SG_M / FA_FRAG;      // 1 M-fragment / simdgroup
constant constexpr uint FA_NFRAGS = FA_BK / FA_FRAG;        // 4 S-col fragments
constant constexpr uint FA_DMAX = 256u;                     // head_dim cap (fragment array bound)
constant constexpr uint FA_DFRAGS_MAX = FA_DMAX / FA_FRAG;  // 32 O-col fragments cap

// Flash-attention v2 (online softmax) joint multi-head scaled-dot-product
// attention (non-causal), HW matrix units, writing the token-major transposed
// output directly.
//
// One threadgroup computes the FA_BQ output rows out_head[h, q0 .. q0+FA_BQ, :]
// for the (q-tile, h) identified by the threadgroup grid position.
kernel void joint_attention_flash_f32(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float* out       [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& seq       [[buffer(5)]],
    constant uint& head_dim  [[buffer(6)]],
    constant float& scale    [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint q_tile = tgid.x;          // query-tile index   (0..ceil(seq/FA_BQ))
    const uint h      = tgid.y;          // head index         (0..num_heads)
    if (h >= num_heads) {
        return;
    }
    const uint q0 = q_tile * FA_BQ;      // first query row of this tile
    if (q0 >= seq) {
        return;
    }

    // Per-head base offset into the head-major [num_heads, seq, head_dim] tensors.
    const uint head_off = (h * seq) * head_dim;
    // Number of 8-wide K-fragments along head_dim (the contraction of S = Q·Kᵀ).
    const uint d_frags = head_dim / FA_FRAG;        // 16 for head_dim = 128

    // simdgroup -> a horizontal strip of FA_SG_M (= 16) query rows of the tile.
    const uint sg_m0 = sgid * FA_SG_M;              // tile-local first query row of this simdgroup
    const uint q_row0 = q0 + sg_m0;                 // global first query row of this simdgroup

    // ── Threadgroup scratch ────────────────────────────────────────────────
    // KVsh: unioned Kᵀ[head_dim][FA_BK] (staged for Q·Kᵀ) and V[FA_BK][head_dim]
    // (staged for P·V). K is dead before V is staged, so they share storage.
    // Sized to the head_dim cap (FA_DMAX): FA_DMAX*FA_BK = 256*32 = 8192 floats
    // = 32 KiB ... too large. Bound to head_dim at runtime via FA_BK*head_dim
    // (<= FA_BK*FA_DMAX). Allocate to the *actual* worst-case used here: the
    // DiT/parity head_dim is 128 -> FA_BK*128 = 4096 floats = 16 KiB. To keep a
    // fixed compile-time bound we size for head_dim<=128 (the validated cap for
    // this kernel; the dispatch layer enforces head_dim<=128).
    threadgroup float KVsh[FA_BK * 128u];           // 32 * 128 * 4 = 16 KiB
    // Ssh: the S (then P) tile [FA_BQ][FA_BK].
    threadgroup float Ssh[FA_BQ * FA_BK];           // 64 * 32 * 4 = 8 KiB
    // Per-query-row running softmax state and rescale correction.
    threadgroup float mrow[FA_BQ];                  // running max
    threadgroup float lrow[FA_BQ];                  // running sum (normaliser)
    threadgroup float corr[FA_BQ];                  // per-row rescale this tile
    // Diagonal-matrix scratch for the O rescale (one 8x8 per M-fragment row, per
    // simdgroup). FA_SIMDGROUPS * FA_MFRAGS * 64 floats = 4*2*64 = 512 -> 2 KiB.
    threadgroup float diagsh[FA_SIMDGROUPS * FA_MFRAGS * FA_FRAG * FA_FRAG];

    // ── O accumulator: simdgroup fragments (registers), FA_SG_M x head_dim. ──
    // FA_MFRAGS (2) rows x d_frags (<= FA_DFRAGS_MAX) cols of 8x8 f32 fragments.
    simdgroup_float8x8 oacc[FA_MFRAGS][FA_DFRAGS_MAX];
    for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
        for (uint di = 0u; di < d_frags; di++) {
            oacc[mi][di] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Init running softmax state (all FA_BQ rows; FA_THREADS=256 >= FA_BQ=64).
    if (lid < FA_BQ) {
        mrow[lid] = -INFINITY;
        lrow[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Valid query rows in this tile (partial-tile clamp at the seq edge).
    const uint q_valid = (q0 < seq) ? min(FA_BQ, seq - q0) : 0u;

    const uint k_tiles = (seq + FA_BK - 1u) / FA_BK;    // ceil(seq / FA_BK)

    for (uint kt = 0u; kt < k_tiles; kt++) {
        const uint k0 = kt * FA_BK;                      // first key of this tile
        const uint k_valid = (k0 < seq) ? min(FA_BK, seq - k0) : 0u;

        // ── Stage Kᵀ into KVsh: KVsh[d * FA_BK + j] = k[h, k0+j, d]. ─────────
        // Transposed so S = Q·Kᵀ reads it as the [K=head_dim, N=FA_BK] operand
        // with a natural simdgroup_load (ld = FA_BK). All 256 threads cooperate
        // over head_dim*FA_BK elements; out-of-range keys staged as 0.
        for (uint i = lid; i < head_dim * FA_BK; i += FA_THREADS) {
            const uint d = i / FA_BK;                    // 0..head_dim-1
            const uint j = i % FA_BK;                    // 0..FA_BK-1 (tile-local key)
            float val = 0.0f;
            if (j < k_valid) {
                val = k[head_off + (k0 + j) * head_dim + d];
            }
            KVsh[d * FA_BK + j] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── S[FA_BQ, FA_BK] = scale · (Q · Kᵀ) via 8x8 MACs. ────────────────
        // This simdgroup computes its FA_SG_M (16) x FA_BK (32) strip:
        //   A fragment : Q  device  q[head_off + (q_row0 + mi*8 + ..) * head_dim
        //                + kf*8 + ..]   (row-major, ld = head_dim)
        //   B fragment : Kᵀ KVsh[(kf*8 + ..)*FA_BK + (ni*8 + ..)]  (ld = FA_BK)
        simdgroup_float8x8 sacc[FA_MFRAGS][FA_NFRAGS];
        for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
            for (uint ni = 0u; ni < FA_NFRAGS; ni++) {
                sacc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        for (uint kf = 0u; kf < d_frags; kf++) {
            simdgroup_float8x8 qfrag[FA_MFRAGS];
            simdgroup_float8x8 kfrag[FA_NFRAGS];
            for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                const uint qr = q_row0 + mi * FA_FRAG;   // global query row of this 8x8
                // Clamp: out-of-range query rows load from row (seq-1) but their
                // S is never read (q_valid guards the softmax + writeback); the
                // load must stay in-bounds, so clamp the row index.
                const uint qr_c = (qr < seq) ? qr : (seq - 1u);
                const device float* qsrc = q + head_off + qr_c * head_dim + kf * FA_FRAG;
                simdgroup_load(qfrag[mi], qsrc, head_dim);
            }
            for (uint ni = 0u; ni < FA_NFRAGS; ni++) {
                const uint koff = (kf * FA_FRAG) * FA_BK + ni * FA_FRAG;
                simdgroup_load(kfrag[ni], KVsh + koff, FA_BK);
            }
            for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                for (uint ni = 0u; ni < FA_NFRAGS; ni++) {
                    simdgroup_multiply_accumulate(sacc[mi][ni], qfrag[mi], kfrag[ni], sacc[mi][ni]);
                }
            }
        }
        // Store S strip to Ssh (apply `scale` on store). Ssh row = tile-local
        // query row (sg_m0 + mi*8 + r); col = ni*8 + c.
        for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
            for (uint ni = 0u; ni < FA_NFRAGS; ni++) {
                const uint soff = (sg_m0 + mi * FA_FRAG) * FA_BK + ni * FA_FRAG;
                simdgroup_store(sacc[mi][ni], Ssh + soff, FA_BK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Online softmax (scalar, one query row per strided thread). ──────
        // Thread `lid` owns tile-local query rows {lid, lid+FA_THREADS, ...}.
        // (FA_BQ = 64 <= FA_THREADS = 256, so each valid row is owned once.)
        for (uint r = lid; r < FA_BQ; r += FA_THREADS) {
            float c = 1.0f;                              // default correction (no update)
            if (r < q_valid) {
                // Row max over the FA_BK valid keys of this tile.
                float tile_max = -INFINITY;
                for (uint j = 0u; j < k_valid; j++) {
                    const float s = Ssh[r * FA_BK + j] * scale;
                    tile_max = max(tile_max, s);
                }
                const float m_old = mrow[r];
                const float m_new = max(m_old, tile_max);
                // P = exp(scale*S - m_new); accumulate this tile's sum.
                float tile_sum = 0.0f;
                for (uint j = 0u; j < FA_BK; j++) {
                    float p = 0.0f;
                    if (j < k_valid) {
                        p = exp(Ssh[r * FA_BK + j] * scale - m_new);
                    }
                    Ssh[r * FA_BK + j] = p;              // overwrite S with P (padded cols = 0)
                    tile_sum += p;
                }
                // Correction for the running state + the O accumulator.
                c = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                lrow[r] = lrow[r] * c + tile_sum;
                mrow[r] = m_new;
            } else {
                // Padded query row: zero its P so P·V contributes nothing.
                for (uint j = 0u; j < FA_BK; j++) {
                    Ssh[r * FA_BK + j] = 0.0f;
                }
                c = 1.0f;
            }
            corr[r] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Rescale the O accumulator: O = diag(corr) · O (per query row). ──
        // Build, per M-fragment row of this simdgroup, an 8x8 diagonal matrix
        // diag(corr[row]) in threadgroup scratch, then O[mi][di] =
        // diag · O[mi][di] via simdgroup_multiply (overwrite, not accumulate).
        {
            const uint dbase = (sgid * FA_MFRAGS) * (FA_FRAG * FA_FRAG);
            // Fill 8 diagonal entries per M-fragment with corr[row]; the other
            // 7x8 entries are zeroed. The simdgroup's 32 lanes cooperatively fill
            // the FA_MFRAGS*64 floats (= 64 for FA_MFRAGS=1 at 8 simdgroups).
            const uint lane = lid % 32u;
            for (uint idx = lane; idx < FA_MFRAGS * FA_FRAG * FA_FRAG; idx += 32u) {
                const uint mi = idx / (FA_FRAG * FA_FRAG);    // 0..FA_MFRAGS-1
                const uint e  = idx % (FA_FRAG * FA_FRAG);    // 0..63 within 8x8
                const uint rr = e / FA_FRAG;                  // row in 8x8
                const uint cc = e % FA_FRAG;                  // col in 8x8
                float val = 0.0f;
                if (rr == cc) {
                    const uint tile_row = sg_m0 + mi * FA_FRAG + rr;   // tile-local query row
                    val = corr[tile_row];
                }
                diagsh[dbase + idx] = val;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint dbase = (sgid * FA_MFRAGS) * (FA_FRAG * FA_FRAG);
            for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                simdgroup_float8x8 dfrag;
                simdgroup_load(dfrag, diagsh + dbase + mi * (FA_FRAG * FA_FRAG), FA_FRAG);
                for (uint di = 0u; di < d_frags; di++) {
                    simdgroup_float8x8 tmp;
                    simdgroup_multiply(tmp, dfrag, oacc[mi][di]);
                    oacc[mi][di] = tmp;
                }
            }
        }

        // ── Stage V into KVsh (K is dead): KVsh[j*head_dim + d] = v[h,k0+j,d]. ─
        // No barrier needed before this write: KVsh last held K, whose final read
        // (the Q·Kᵀ simdgroup_load) is separated from here by the S-store and
        // softmax threadgroup barriers; the intervening O-rescale touches only the
        // O registers and the simdgroup-private diagsh slice (never KVsh).
        for (uint i = lid; i < FA_BK * head_dim; i += FA_THREADS) {
            const uint j = i / head_dim;                 // 0..FA_BK-1 (tile-local key)
            const uint d = i % head_dim;                 // 0..head_dim-1
            float val = 0.0f;
            if (j < k_valid) {
                val = v[head_off + (k0 + j) * head_dim + d];
            }
            KVsh[j * head_dim + d] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P · V via 8x8 MACs (contract over FA_BK = 4 K-fragments). ──
        //   A fragment : P  Ssh[(sg_m0 + mi*8 + ..)*FA_BK + kf*8 + ..]  (ld=FA_BK)
        //   B fragment : V  KVsh[(kf*8 + ..)*head_dim + di*8 + ..]      (ld=head_dim)
        const uint pk_frags = FA_BK / FA_FRAG;           // 4
        for (uint kf = 0u; kf < pk_frags; kf++) {
            simdgroup_float8x8 pfrag[FA_MFRAGS];
            for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                const uint poff = (sg_m0 + mi * FA_FRAG) * FA_BK + kf * FA_FRAG;
                simdgroup_load(pfrag[mi], Ssh + poff, FA_BK);
            }
            for (uint di = 0u; di < d_frags; di++) {
                simdgroup_float8x8 vfrag;
                const uint voff = (kf * FA_FRAG) * head_dim + di * FA_FRAG;
                simdgroup_load(vfrag, KVsh + voff, head_dim);
                for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                    simdgroup_multiply_accumulate(oacc[mi][di], pfrag[mi], vfrag, oacc[mi][di]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Writeback: out[qi, h*head_dim + d] = O[qi,d] / lrow[qi]. ────────────
    // Stage each O fragment to a threadgroup scratch (reuse KVsh as Csh), then
    // scatter to `out` with the per-row 1/l normaliser and the partial-tile
    // clamp. Done one simdgroup at a time to reuse the scratch region.
    threadgroup float* Csh = KVsh;   // FA_SG_M * head_dim per simdgroup (<= 16*128 = 2048 floats)
    for (uint sg = 0u; sg < FA_SIMDGROUPS; sg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            for (uint mi = 0u; mi < FA_MFRAGS; mi++) {
                for (uint di = 0u; di < d_frags; di++) {
                    const uint coff = (mi * FA_FRAG) * head_dim + di * FA_FRAG;
                    simdgroup_store(oacc[mi][di], Csh + coff, head_dim);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == sgid) {
            const uint inner = num_heads * head_dim;
            for (uint idx = lid % 32u; idx < FA_SG_M * head_dim; idx += 32u) {
                const uint mm = idx / head_dim;          // 0..FA_SG_M-1 local query row
                const uint d  = idx % head_dim;          // 0..head_dim-1 feature
                const uint tile_row = sg_m0 + mm;        // tile-local query row
                if (tile_row < q_valid) {
                    const uint qi = q0 + tile_row;       // global query row
                    const float l = lrow[tile_row];
                    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
                    out[qi * inner + h * head_dim + d] = Csh[mm * head_dim + d] * inv;
                }
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::*;

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn dit_attention_flash_source_contains_entry_point() {
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("kernel void joint_attention_flash_f32"));
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("#include <metal_stdlib>"));
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("#include <metal_simdgroup_matrix>"));
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("simdgroup_multiply_accumulate"));
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn dit_attention_flash_tile_constants_are_consistent() {
        // The MSL tile literals must match the exported Rust constants.
        assert_eq!(DIT_FLASH_BQ, 64);
        assert_eq!(DIT_FLASH_BK, 32);
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("FA_BQ = 64u"));
        assert!(MSL_DIT_JOINT_ATTENTION_FLASH.contains("FA_BK = 32u"));
    }
}
