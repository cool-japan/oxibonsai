//! # CudaGraph — DiT joint flash-attention encode group (imagen prototype)
//!
//! CUDA prototype mirror of the Metal `encode_joint_attention_flash` /
//! `encode_joint_attention_flash_pooled` + `joint_attn_flash_validate`
//! (`gpu_backend/metal_graph/graph.rs`). Dispatches the scalar-FP32 flash
//! attention kernel `joint_attention_flash_f32` (source
//! [`CUDA_IMAGEN_ATTN_SRC`](crate::gpu_backend::cuda_imagen_attn_kernels::CUDA_IMAGEN_ATTN_SRC)),
//! matching the CPU reference `oxibonsai-image::math::joint_attention`.
//!
//! **CUDA prototype**: authored blind on macOS (no CUDA toolkit). The real
//! compile + parity run happens on Linux. The kernel is fully gated off on
//! macOS. Parity-first: everything is f32, no tensor cores.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

/// Compile-time cap on `seq` accepted by the DiT joint-attention dispatch /
/// validation layer. A generous upper bound covering the DiT `seq = 1536` and
/// the VAE mid-block self-attention `seq = H·W = 64·64 = 4096` (the kernel tiles
/// `seq` over `grid.x = seq.div_ceil(BQ)`, so there is no fixed per-`seq`
/// buffer — this is purely a sanity bound).
pub const DIT_ATTN_MAX_SEQ: usize = 8192;

/// Compile-time cap on `head_dim` accepted by the *general* joint-attention
/// validation. The flash kernel caps `head_dim` at [`DIT_FLASH_HEAD_DIM_CAP`]
/// (its `FA_DMAX` register-array bound) and requires `head_dim % 8 == 0` —
/// see [`CudaGraph::encode_joint_attention_flash`].
pub const DIT_ATTN_MAX_HEAD_DIM: usize = 384;

/// Flash-kernel hard cap on `head_dim` — the kernel's `FA_DMAX` per-thread
/// `Q[]`/`O[]` array bound, and the width that sizes the dynamic shared-mem
/// arena (`FA_BK*head_dim*2*4` bytes, opted into at module load). Covers the DiT
/// `head_dim = 128` and the VAE mid-attention `head_dim = 384` (→ 96 KiB shared,
/// within the Ampere 100 KiB per-block limit).
const DIT_FLASH_HEAD_DIM_CAP: usize = 384;

/// Query-tile height (`BQ`) — rows of `out` per block; must match the kernel's
/// `FA_BQ` and the launch grid `seq.div_ceil(BQ)`. The warp-cooperative kernel
/// uses 4 warps × FA_RPW(2) rows = 8 rows/block; each warp's 32 lanes split
/// head_dim (Q/O in registers, no spill) and finish the score by a warp reduce.
const DIT_FLASH_BQ: usize = 8;

/// Key-tile width (`BK`) — keys staged per online-softmax step; must match the
/// kernel's `FA_BK`. Sizes the dynamic shared mem: `FA_BK*head_dim*2` floats
/// (`Ksh ‖ Vsh`).
const DIT_FLASH_BK: usize = 32;

impl CudaGraph {
    /// Validate the joint flash-attention shape against the kernel caps.
    ///
    /// Mirrors the Metal `joint_attn_flash_validate` (the shared
    /// `joint_attn_validate` length/cap checks + the flash-specific `head_dim %
    /// 8 == 0` and `head_dim <= 384` constraints). Returns
    /// `(qkv_len, out_len, scale)` on success. Every failure is a
    /// [`CudaGraphError::DriverError`] carrying a descriptive message (the CUDA
    /// graph error enum has no dedicated dimension variant; `DriverError` is the
    /// shared shape/length-validation channel across the imagen group files —
    /// matching the gemm + vae groups for cross-file consistency).
    #[allow(clippy::too_many_arguments)]
    fn joint_attn_flash_validate(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(usize, usize, f32), CudaGraphError> {
        if num_heads == 0 || seq == 0 || head_dim == 0 {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: dims must be non-zero (num_heads={num_heads}, seq={seq}, head_dim={head_dim})"
            )));
        }
        if seq > DIT_ATTN_MAX_SEQ {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: seq {seq} exceeds kernel cap {DIT_ATTN_MAX_SEQ}"
            )));
        }
        if head_dim > DIT_ATTN_MAX_HEAD_DIM {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: head_dim {head_dim} exceeds kernel cap {DIT_ATTN_MAX_HEAD_DIM}"
            )));
        }
        if head_dim % 8 != 0 {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: head_dim {head_dim} must be a multiple of 8"
            )));
        }
        if head_dim > DIT_FLASH_HEAD_DIM_CAP {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: head_dim {head_dim} exceeds the flash kernel cap ({DIT_FLASH_HEAD_DIM_CAP})"
            )));
        }

        let qkv_len = num_heads * seq * head_dim;
        let out_len = seq * num_heads * head_dim;
        if q.len() != qkv_len || k.len() != qkv_len || v.len() != qkv_len {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: q/k/v len mismatch (need {qkv_len}, got {}/{}/{})",
                q.len(),
                k.len(),
                v.len()
            )));
        }
        if out.len() != out_len {
            return Err(CudaGraphError::DriverError(format!(
                "joint_attention_flash: out len mismatch (need {out_len}, got {})",
                out.len()
            )));
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        Ok((qkv_len, out_len, scale))
    }

    /// Launch the joint flash-attention on **already-resident** device buffers
    /// (no upload/download) — the building block the resident DiT forward chains.
    /// Dispatches to the head_dim-matched build (lean FA_DMAX=128 ≤ 128, else the
    /// FA_DMAX=384 VAE build). `scale = 1/sqrt(head_dim)`.
    ///
    /// # Safety
    /// `d_q`/`d_k`/`d_v` hold ≥ `num_heads*seq*head_dim` f32, `d_out` ≥
    /// `seq*num_heads*head_dim` f32; `head_dim ≤ 384` and `% 8 == 0`;
    /// `seq.div_ceil(FA_BQ)*num_heads` blocks fit the grid; all on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn launch_joint_attention_flash_resident(
        &self,
        d_q: &CudaSlice<f32>,
        d_k: &CudaSlice<f32>,
        d_v: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        num_heads: u32,
        seq: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), CudaGraphError> {
        let shared_mem_bytes =
            (DIT_FLASH_BK * head_dim as usize * 2 * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((seq as usize).div_ceil(DIT_FLASH_BQ) as u32, num_heads, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };
        let func = if head_dim <= 128 {
            &self.modules.joint_attention_flash_f32
        } else {
            &self.modules.joint_attention_flash_f32_large
        };
        self.stream
            .launch_builder(func)
            .arg(d_q)
            .arg(d_k)
            .arg(d_v)
            .arg(d_out)
            .arg(&num_heads)
            .arg(&seq)
            .arg(&head_dim)
            .arg(&scale)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("joint_attention_flash launch: {e}")))
    }

    /// Execute FLUX.2 DiT joint (txt+img) multi-head scaled-dot-product attention
    /// on the GPU via the scalar-FP32 flash-attention kernel
    /// `joint_attention_flash_f32`, matching the CPU reference
    /// `oxibonsai-image::math::joint_attention` in behaviour.
    ///
    /// `q`, `k`, `v` are head-major `[num_heads × seq × head_dim]` f32 (RoPE
    /// already applied upstream to q,k). `out` receives the token-major
    /// transposed result `[seq × (num_heads*head_dim)]`. Non-causal (full
    /// bidirectional softmax over keys); `scale = 1/sqrt(head_dim)`.
    ///
    /// This is the **standalone** path: fresh q/k/v input buffers and a fresh
    /// zeroed `out` buffer are allocated per call (`// PERF: per-call alloc; pool
    /// in hardware phase`).
    ///
    /// # Errors
    /// Returns [`CudaGraphError::DriverError`] if the slice lengths are
    /// inconsistent with `num_heads*seq*head_dim`, if any dimension is zero, if
    /// `seq`/`head_dim` exceed the kernel's caps, or if `head_dim` is not a
    /// multiple of `8` or exceeds the flash kernel's `384` cap; or if the GPU
    /// work cannot be encoded.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_joint_attention_flash(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), CudaGraphError> {
        let (qkv_len, out_len, scale) =
            Self::joint_attn_flash_validate(q, k, v, out, num_heads, seq, head_dim)?;

        // ── Host → device. PERF: per-call alloc; pool in hardware phase. ──
        // alloc_zeros + memcpy_htod is the proven upload idiom across the CUDA
        // backend (cudagraph_encoding_*). Each q/k/v gets a fresh device slice.
        let mut d_q = self
            .stream
            .alloc_zeros::<f32>(qkv_len)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros joint_attn q: {e}")))?;
        self.stream
            .memcpy_htod(&q[..qkv_len], &mut d_q)
            .map_err(|e| CudaGraphError::DriverError(format!("htod joint_attn q: {e}")))?;
        let mut d_k = self
            .stream
            .alloc_zeros::<f32>(qkv_len)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros joint_attn k: {e}")))?;
        self.stream
            .memcpy_htod(&k[..qkv_len], &mut d_k)
            .map_err(|e| CudaGraphError::DriverError(format!("htod joint_attn k: {e}")))?;
        let mut d_v = self
            .stream
            .alloc_zeros::<f32>(qkv_len)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros joint_attn v: {e}")))?;
        self.stream
            .memcpy_htod(&v[..qkv_len], &mut d_v)
            .map_err(|e| CudaGraphError::DriverError(format!("htod joint_attn v: {e}")))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(out_len)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros joint_attn out: {e}")))?;

        // ── Launch on the freshly-uploaded device buffers via the resident
        // core (dispatch + kernel). ──
        // SAFETY: d_q/d_k/d_v/d_out are freshly `alloc_zeros`'d above with the
        // validated `qkv_len`/`out_len`; `head_dim` is validated ≤ 384 and %8;
        // all live on `self.stream`.
        unsafe {
            self.launch_joint_attention_flash_resident(
                &d_q,
                &d_k,
                &d_v,
                &mut d_out,
                num_heads as u32,
                seq as u32,
                head_dim as u32,
                scale,
            )?;
        }

        // ── Device → host + synchronize. ──
        self.stream
            .memcpy_dtoh(&d_out, &mut out[..out_len])
            .map_err(|e| CudaGraphError::DriverError(format!("dtoh joint_attn out: {e}")))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("joint_attn sync: {e}")))?;
        Ok(())
    }

    /// Pooled variant of [`Self::encode_joint_attention_flash`].
    ///
    /// The image hook calls the `_pooled` name, so it must exist. For the
    /// prototype it simply delegates to the fresh-alloc path.
    ///
    /// `// PERF: pooled variant == fresh for now; add JointAttn pool in hardware phase`
    ///
    /// # Errors
    /// As [`Self::encode_joint_attention_flash`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_joint_attention_flash_pooled(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<(), CudaGraphError> {
        self.encode_joint_attention_flash(q, k, v, out, num_heads, seq, head_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline CPU reference port of oxibonsai-image::math::joint_attention. ──
    // Head-major q/k/v `[num_heads, seq, head_dim]` in; token-major transposed
    // `[seq, num_heads*head_dim]` out; per-head dot, max-subtracted softmax over
    // keys, weighted-V, head→token transpose. scale = 1/sqrt(head_dim).
    fn cpu_joint_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let inner = num_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * inner];
        for h in 0..num_heads {
            let head_off = h * seq * head_dim;
            for qi in 0..seq {
                let q_row = &q[head_off + qi * head_dim..head_off + (qi + 1) * head_dim];
                // scores[ki] = scale * dot(q_row, k_row)
                let mut scores = vec![0.0f32; seq];
                for (ki, score) in scores.iter_mut().enumerate() {
                    let k_row = &k[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q_row[d] * k_row[d];
                    }
                    *score = acc * scale;
                }
                // max-subtracted softmax over keys.
                let mut maxv = f32::NEG_INFINITY;
                for &s in scores.iter() {
                    maxv = maxv.max(s);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - maxv).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                // weighted-V into token-major out[qi, h*head_dim + d].
                let dst = &mut out[qi * inner + h * head_dim..qi * inner + (h + 1) * head_dim];
                for (ki, &w) in scores.iter().enumerate() {
                    let p = w * inv;
                    let v_row = &v[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                    for d in 0..head_dim {
                        dst[d] += p * v_row[d];
                    }
                }
            }
        }
        out
    }

    /// Deterministic LCG fill in `[-1, 1)` (no RNG dep; reproducible).
    fn lcg_fill(buf: &mut [f32], mut state: u64) {
        for x in buf.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32; // top 31 bits
            *x = (bits as f32 / (1u32 << 31) as f32) * 2.0 - 1.0;
        }
    }

    /// Cosine similarity of two equal-length slices.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// Max absolute element-wise difference.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// GPU vs CPU parity for the DiT joint flash-attention, across head counts
    /// and seq lengths that exercise full + partial BQ/BK tiles. Gracefully
    /// skips when no CUDA device is present (the macOS / CI-without-GPU case).
    #[test]
    fn joint_attention_flash_parity() {
        let _serial = super::super::types::gpu_parity_test_guard();
        let graph = match CudaGraph::global() {
            Ok(g) => g,
            Err(_) => {
                eprintln!("joint_attention_flash_parity: no CUDA device — skipping");
                return;
            }
        };

        let head_dim = 128usize;
        // 40/50 exercise partial BQ (64) and BK (32) tiles; 64 is exactly one
        // BQ tile; 8/32 are sub-tile.
        let shapes: &[(usize, usize)] = &[(1, 8), (1, 32), (2, 40), (2, 50), (24, 64)];

        for &(num_heads, seq) in shapes {
            let qkv_len = num_heads * seq * head_dim;
            let out_len = seq * num_heads * head_dim;

            let mut q = vec![0.0f32; qkv_len];
            let mut k = vec![0.0f32; qkv_len];
            let mut v = vec![0.0f32; qkv_len];
            lcg_fill(&mut q, 0x1234_5678_9abc_def0 ^ (seq as u64));
            lcg_fill(&mut k, 0x0fed_cba9_8765_4321 ^ (num_heads as u64));
            lcg_fill(&mut v, 0xdead_beef_cafe_babe ^ ((seq * num_heads) as u64));

            let cpu = cpu_joint_attention(&q, &k, &v, num_heads, seq, head_dim);

            let mut gpu = vec![0.0f32; out_len];
            graph
                .encode_joint_attention_flash(&q, &k, &v, &mut gpu, num_heads, seq, head_dim)
                .expect("encode_joint_attention_flash");

            let cos = cosine(&gpu, &cpu);
            let mad = max_abs_diff(&gpu, &cpu);
            assert!(
                cos >= 0.999,
                "cos {cos} < 0.999 for (num_heads={num_heads}, seq={seq})"
            );
            assert!(
                mad < 1e-3,
                "max-abs {mad} >= 1e-3 for (num_heads={num_heads}, seq={seq})"
            );

            // The pooled variant must produce the identical result.
            let mut gpu_pooled = vec![0.0f32; out_len];
            graph
                .encode_joint_attention_flash_pooled(
                    &q,
                    &k,
                    &v,
                    &mut gpu_pooled,
                    num_heads,
                    seq,
                    head_dim,
                )
                .expect("encode_joint_attention_flash_pooled");
            assert_eq!(
                gpu, gpu_pooled,
                "pooled != fresh for (num_heads={num_heads}, seq={seq})"
            );
        }
    }

    /// Shape validation rejects bad dims without touching the GPU (runs on any
    /// host, including macOS where the whole module is gated out — this test is
    /// only compiled under `native-cuda`).
    #[test]
    fn joint_attn_flash_validate_rejects_bad_shapes() {
        let head_dim = 128usize;
        let (num_heads, seq) = (2usize, 16usize);
        let qkv_len = num_heads * seq * head_dim;
        let out_len = seq * num_heads * head_dim;
        let q = vec![0.0f32; qkv_len];
        let k = vec![0.0f32; qkv_len];
        let v = vec![0.0f32; qkv_len];
        let out = vec![0.0f32; out_len];

        // Good shape validates.
        assert!(
            CudaGraph::joint_attn_flash_validate(&q, &k, &v, &out, num_heads, seq, head_dim)
                .is_ok()
        );
        // head_dim not a multiple of 8.
        assert!(
            CudaGraph::joint_attn_flash_validate(&q, &k, &v, &out, num_heads, seq, 130).is_err()
        );
        // head_dim over the flash cap of 384 (still % 8 == 0). 512 > 384 trips the
        // cap check, which precedes the length check, so the head_dim-128-sized
        // q/k/v buffers don't matter — this exercises the cap-rejection path.
        assert!(
            CudaGraph::joint_attn_flash_validate(&q, &k, &v, &out, num_heads, seq, 512).is_err()
        );
        // Zero dim.
        assert!(CudaGraph::joint_attn_flash_validate(&q, &k, &v, &out, 0, seq, head_dim).is_err());
        // q too short.
        let q_short = vec![0.0f32; qkv_len - 1];
        assert!(CudaGraph::joint_attn_flash_validate(
            &q_short, &k, &v, &out, num_heads, seq, head_dim
        )
        .is_err());
    }
}
