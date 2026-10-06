//! Command encoding of one image through the vision tower (a child module
//! of `metal_vision`): the host rows go in, one command buffer runs the
//! patch embedding, every ViT block, the post-norm and the merger, and the
//! merged rows come back.

use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use super::{
    validate_vit_attention, VisionGpuModel, FLAG_ACCUMULATE, FLAG_BIAS, FLAG_GELU, GEMM_THREADS,
    GEMM_TILE, NORM_THREADS,
};
use crate::gpu_backend::metal_graph::{commit_and_wait, MetalGraphError};

fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, v: u32) {
    enc.set_bytes(index, 4, (&v as *const u32).cast());
}

fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, v: f32) {
    enc.set_bytes(index, 4, (&v as *const f32).cast());
}

/// Copy `values` into the start of a shared buffer that holds at least as
/// many floats (checked by the caller against the scratch geometry).
///
/// # Safety
///
/// `buf` must be a shared-storage buffer of at least `values.len()` floats
/// that no GPU work is using.
unsafe fn upload_into(buf: &Buffer, values: &[f32]) {
    std::ptr::copy_nonoverlapping(values.as_ptr(), buf.contents().cast::<f32>(), values.len());
}

/// One GEMM `c = epilogue(a · wᵀ)` over `m_rows` activation rows.
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    enc: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &Buffer,
    a: &Buffer,
    c: &Buffer,
    bias: &Buffer,
    n_rows: usize,
    k: usize,
    m_rows: usize,
    flags: u32,
) {
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(w), 0);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(c), 0);
    enc.set_buffer(3, Some(bias), 0);
    set_u32(enc, 4, n_rows as u32);
    set_u32(enc, 5, k as u32);
    set_u32(enc, 6, m_rows as u32);
    set_u32(enc, 7, flags);
    enc.dispatch_thread_groups(
        MTLSize::new(
            n_rows.div_ceil(GEMM_TILE) as u64,
            m_rows.div_ceil(GEMM_TILE) as u64,
            1,
        ),
        MTLSize::new(GEMM_THREADS, 1, 1),
    );
}

impl VisionGpuModel {
    /// Encode one image.
    ///
    /// Inputs, all for the same `n` patches in 2 × 2 merge-window order:
    /// `patches` (`n × patch_len`, channel-major within a patch — what the
    /// CPU tower's patch GEMM reads), `pos_rows` (`n × hidden`, the resized
    /// position embedding), `rope_cos` / `rope_sin` (`n × head_dim / 2`,
    /// the 2-D rotary rows). Writes the `n / 4` merged rows (`n / 4 ×
    /// projection_dim`, row-major over the merged grid) into `out`.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a patch count that is not
    /// a positive multiple of 4 within `max_patches`, or buffers of the
    /// wrong length (nothing runs then); a failed command buffer.
    pub fn encode(
        &mut self,
        patches: &[f32],
        pos_rows: &[f32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        out: &mut [f32],
    ) -> Result<(), MetalGraphError> {
        autoreleasepool(|| self.encode_unpooled(patches, pos_rows, rope_cos, rope_sin, out))
    }

    fn encode_unpooled(
        &mut self,
        patches: &[f32],
        pos_rows: &[f32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        out: &mut [f32],
    ) -> Result<(), MetalGraphError> {
        let cfg = self.cfg.clone();
        let bad = |what: String| MetalGraphError::InvalidDimensions(format!("vision GPU: {what}"));
        if patches.is_empty() || !patches.len().is_multiple_of(cfg.patch_len) {
            return Err(bad(format!(
                "{} patch values is not a whole number of {}-value patches",
                patches.len(),
                cfg.patch_len
            )));
        }
        let n = patches.len() / cfg.patch_len;
        if !n.is_multiple_of(4) || n > cfg.max_patches {
            return Err(bad(format!(
                "{n} patches must be a multiple of 4 no larger than {}",
                cfg.max_patches
            )));
        }
        validate_vit_attention(n, cfg.head_dim)?;
        let half = cfg.head_dim / 2;
        let merged = n / 4;
        for (what, got, want) in [
            ("pos_rows", pos_rows.len(), n * cfg.hidden),
            ("rope_cos", rope_cos.len(), n * half),
            ("rope_sin", rope_sin.len(), n * half),
            ("out", out.len(), merged * cfg.projection_dim),
        ] {
            if got != want {
                return Err(bad(format!(
                    "{what} holds {got} floats, {n} patches need {want}"
                )));
            }
        }
        let s = &self.scratch;
        // SAFETY: every scratch buffer was allocated for `max_patches >= n`
        // patches of these widths (shared storage), and no GPU work is in
        // flight: every encode waits for its command buffer.
        unsafe {
            upload_into(&s.patches, patches);
            upload_into(&s.pos, pos_rows);
            upload_into(&s.rope_cos, rope_cos);
            upload_into(&s.rope_sin, rope_sin);
        }

        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_tower(enc, n);
        enc.end_encoding();
        commit_and_wait(cmd, "vision_tower")?;
        // SAFETY: the command buffer has completed, so its timestamps are
        // readable.
        let (start, end): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        self.last_gpu_seconds = (end - start).max(0.0);
        // SAFETY: `out` holds `merged * projection_dim` floats, written by
        // the command buffer that has just completed.
        let src = unsafe {
            std::slice::from_raw_parts(self.scratch.out.contents().cast::<f32>(), out.len())
        };
        out.copy_from_slice(src);
        Ok(())
    }

    /// Every dispatch of one image of `n` patches (see the module docs).
    fn encode_tower(&self, enc: &ComputeCommandEncoderRef, n: usize) {
        let c = &self.cfg;
        let s = &self.scratch;
        let p = &self.pipes;
        let h = c.hidden;

        // Patch embedding: the summed kernel with its bias, then the
        // position rows.
        encode_gemm(
            enc,
            &p.gemm_f32w,
            &self.patch_kernel,
            &s.patches,
            &s.x,
            &self.patch_bias,
            h,
            c.patch_len,
            n,
            FLAG_BIAS,
        );
        enc.set_compute_pipeline_state(&p.add_rows);
        enc.set_buffer(0, Some(&s.x), 0);
        enc.set_buffer(1, Some(&s.pos), 0);
        set_u32(enc, 2, (n * h) as u32);
        enc.dispatch_thread_groups(
            MTLSize::new((n * h).div_ceil(256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        let scale = 1.0f32 / (c.head_dim as f32).sqrt();
        for b in &self.blocks {
            self.encode_norm(enc, &s.x, &b.ln1_w, &b.ln1_b, &s.normed, n, h);
            encode_gemm(
                enc,
                p.gemm_for(b.qkv_w.format),
                &b.qkv_w.buffer,
                &s.normed,
                &s.qkv,
                &b.qkv_b,
                3 * h,
                h,
                n,
                FLAG_BIAS,
            );
            enc.set_compute_pipeline_state(&p.rope_pack);
            enc.set_buffer(0, Some(&s.qkv), 0);
            enc.set_buffer(1, Some(&s.rope_cos), 0);
            enc.set_buffer(2, Some(&s.rope_sin), 0);
            enc.set_buffer(3, Some(&s.q), 0);
            enc.set_buffer(4, Some(&s.k), 0);
            enc.set_buffer(5, Some(&s.v), 0);
            set_u32(enc, 6, n as u32);
            set_u32(enc, 7, c.heads as u32);
            set_u32(enc, 8, c.head_dim as u32);
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (c.head_dim / 2).div_ceil(32) as u64,
                    c.heads as u64,
                    n as u64,
                ),
                MTLSize::new(32, 1, 1),
            );
            self.graph.dispatch_joint_attention_flash(
                enc,
                &s.q,
                &s.k,
                &s.v,
                &s.attn,
                c.heads as u32,
                n as u32,
                c.head_dim as u32,
                scale,
            );
            encode_gemm(
                enc,
                p.gemm_for(b.out_w.format),
                &b.out_w.buffer,
                &s.attn,
                &s.x,
                &b.out_b,
                h,
                h,
                n,
                FLAG_BIAS | FLAG_ACCUMULATE,
            );
            self.encode_norm(enc, &s.x, &b.ln2_w, &b.ln2_b, &s.normed, n, h);
            encode_gemm(
                enc,
                p.gemm_for(b.up_w.format),
                &b.up_w.buffer,
                &s.normed,
                &s.up,
                &b.up_b,
                c.ffn,
                h,
                n,
                FLAG_BIAS | FLAG_GELU,
            );
            encode_gemm(
                enc,
                p.gemm_for(b.down_w.format),
                &b.down_w.buffer,
                &s.up,
                &s.x,
                &b.down_b,
                h,
                c.ffn,
                n,
                FLAG_BIAS | FLAG_ACCUMULATE,
            );
        }

        // Post-norm, then the merger over the merged rows: four consecutive
        // window-order rows are one merged token, so `normed` read as
        // `[n / 4][4 · hidden]` is its input with no copy.
        self.encode_norm(enc, &s.x, &self.post_ln_w, &self.post_ln_b, &s.normed, n, h);
        let merged = n / 4;
        encode_gemm(
            enc,
            p.gemm_for(self.mm0_w.format),
            &self.mm0_w.buffer,
            &s.normed,
            &s.mid,
            &self.mm0_b,
            c.merger_hidden,
            c.merged_width(),
            merged,
            FLAG_BIAS | FLAG_GELU,
        );
        encode_gemm(
            enc,
            p.gemm_for(self.mm2_w.format),
            &self.mm2_w.buffer,
            &s.mid,
            &s.out,
            &self.mm2_b,
            c.projection_dim,
            c.merger_hidden,
            merged,
            FLAG_BIAS,
        );
    }

    /// LayerNorm of `rows` rows of `dim` from `x` into `out`.
    #[allow(clippy::too_many_arguments)]
    fn encode_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        w: &Buffer,
        b: &Buffer,
        out: &Buffer,
        rows: usize,
        dim: usize,
    ) {
        enc.set_compute_pipeline_state(&self.pipes.layer_norm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(b), 0);
        enc.set_buffer(3, Some(out), 0);
        set_u32(enc, 4, dim as u32);
        set_f32(enc, 5, self.cfg.eps);
        enc.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(NORM_THREADS, 1, 1),
        );
    }
}
