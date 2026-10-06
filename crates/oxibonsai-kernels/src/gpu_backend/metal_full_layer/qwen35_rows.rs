//! The forward entry points of the Qwen3.5 hybrid encoder (a child module
//! of `metal_full_layer::qwen35`), with the rotary source decoupled from
//! the KV position.
//!
//! # Why the rope source is its own argument
//!
//! A call over `t` rows at KV positions `kv_start..kv_start + t` stores
//! every key and value at its sequence position and lets row `i` attend to
//! positions `0..=kv_start + i` — but the angle a full-attention layer
//! rotates row `i`'s query and key by need not be the one of position
//! `kv_start + i`:
//!
//! * the rows of an image (3-axis M-RoPE, bonsai2-design.md §6.2) rotate at
//!   `(t, h, w)` positions whose angles are not a row of any single-axis
//!   table ([`Qwen35Rope::PerRow`]: one `n_rot / 2` cos and sin row per
//!   input row);
//! * every text token *after* an image rotates at its sequence position
//!   minus the image's M-RoPE offset, i.e. a contiguous run of the
//!   resident table starting below `kv_start` ([`Qwen35Rope::Contiguous`]
//!   with `rope_start < kv_start`).
//!
//! The kernel (`fused_qk_norm_rope_partial`) already reads one angle row
//! per token (`cos + t * n_rot / 2`), so the source is only a binding: the
//! resident table at `rope_start`, or the scratch rows a
//! [`Qwen35Rope::PerRow`] call uploads. [`Qwen35GpuModel::forward`] is
//! [`Qwen35GpuModel::forward_rows`] with `Contiguous { rope_start:
//! start_pos }` — the same encode, so a text-only sequence is unchanged bit
//! for bit.

use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};

use super::{commit_and_wait, read_buffer, MetalGraphError, Qwen35GpuModel, Qwen35LayerTrace};

/// Where a forward's rotary angles come from (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Qwen35Rope<'r> {
    /// Row `i` rotates at the text position `rope_start + i`: the resident
    /// angle table read from row `rope_start`.
    Contiguous {
        /// Text rotary position of the first row.
        rope_start: usize,
    },
    /// Row `i` rotates by `cos[i * n_rot / 2..][..n_rot / 2]` and the same
    /// `sin` row — each `t_len * n_rot / 2` long.
    PerRow {
        /// Cosines, row-major `[t_len][n_rot / 2]`.
        cos: &'r [f32],
        /// Sines, row-major `[t_len][n_rot / 2]`.
        sin: &'r [f32],
    },
}

/// What [`Qwen35GpuModel::forward_with_dump_rows`] records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Qwen35ForwardDump {
    /// The residual stream layer 0 reads, read back from the device before
    /// any layer runs (`[t_len * hidden]`) — the rows exactly as they
    /// entered the stack.
    pub input: Vec<f32>,
    /// The residual stream after every layer (`[layer][t_len * hidden]`).
    pub layers: Vec<Vec<f32>>,
    /// The last row's `[vocab]` logits.
    pub logits: Vec<f32>,
}

/// The angle buffers the partial-RoPE stage binds for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RopeBinding {
    /// The resident table, from this byte offset.
    Table {
        /// `rope_start * n_rot / 2 * 4`.
        byte_offset: u64,
    },
    /// The call's own rows, uploaded into the scratch angle buffers.
    Scratch,
}

impl Qwen35GpuModel<'_> {
    /// Copy caller rows into the residual stream (growing the scratch to
    /// hold them); returns the row count.
    pub(super) fn load_rows(&mut self, rows: &[f32]) -> Result<usize, MetalGraphError> {
        let h = self.cfg.hidden;
        if rows.is_empty() || !rows.len().is_multiple_of(h) {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: {} input floats is not a whole number of {h}-wide rows",
                rows.len()
            )));
        }
        let t_len = rows.len() / h;
        self.ensure_capacity(t_len)?;
        // SAFETY: `resid` holds `capacity * hidden >= rows.len()` floats and
        // no GPU work is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr(),
                self.scratch.resid.contents().cast::<f32>(),
                rows.len(),
            );
        }
        Ok(t_len)
    }

    /// Resolve `rope` for a call of `t_len` rows: validate it and, for
    /// per-row angles, upload them into the scratch (which must already hold
    /// `t_len` tokens — [`Self::load_rows`] runs first).
    pub(super) fn bind_rope(
        &mut self,
        rope: Qwen35Rope<'_>,
        t_len: usize,
    ) -> Result<RopeBinding, MetalGraphError> {
        let half = self.cfg.n_rot / 2;
        match rope {
            Qwen35Rope::Contiguous { rope_start } => {
                let end = rope_start.checked_add(t_len).ok_or_else(|| {
                    MetalGraphError::InvalidDimensions(
                        "qwen35 GPU: rotary position overflow".to_string(),
                    )
                })?;
                if end > self.cfg.max_seq_len {
                    return Err(MetalGraphError::InvalidDimensions(format!(
                        "qwen35 GPU: rotary positions {rope_start}..{end} run past the angle \
                         table of {}",
                        self.cfg.max_seq_len
                    )));
                }
                Ok(RopeBinding::Table {
                    byte_offset: (rope_start * half * 4) as u64,
                })
            }
            Qwen35Rope::PerRow { cos, sin } => {
                let want = t_len * half;
                if cos.len() != want || sin.len() != want {
                    return Err(MetalGraphError::InvalidDimensions(format!(
                        "qwen35 GPU: per-row angles hold {} / {} floats, {t_len} rows need {want} \
                         each",
                        cos.len(),
                        sin.len()
                    )));
                }
                if t_len > self.scratch.capacity {
                    return Err(MetalGraphError::InvalidDimensions(format!(
                        "qwen35 GPU: {t_len} angle rows exceed the scratch of {} tokens",
                        self.scratch.capacity
                    )));
                }
                for (dst, src) in [(&self.scratch.rope_cos, cos), (&self.scratch.rope_sin, sin)] {
                    // SAFETY: each angle buffer holds `capacity * half >=
                    // want` floats (checked above) and no GPU work is in
                    // flight: every call waits for its command buffer.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            dst.contents().cast::<f32>(),
                            want,
                        );
                    }
                }
                Ok(RopeBinding::Scratch)
            }
        }
    }

    /// Run the whole stack over `t_len = hidden_rows.len() / hidden` tokens
    /// at KV positions `start_pos..start_pos + t_len`, rotating them at the
    /// same text positions (their embeddings, after any inverse rotation, in
    /// `hidden_rows`), advancing the recurrent state once per token and
    /// storing every key/value; when `logits` is `Some`, write the last
    /// token's `[vocab]` logits into it.
    ///
    /// [`Self::forward_rows`] with [`Qwen35Rope::Contiguous`] at
    /// `start_pos`.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_rows`].
    pub fn forward(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
        logits: Option<&mut [f32]>,
    ) -> Result<(), MetalGraphError> {
        self.forward_rows(
            hidden_rows,
            start_pos,
            Qwen35Rope::Contiguous {
                rope_start: start_pos,
            },
            logits,
        )
    }

    /// Run the whole stack over `t_len = hidden_rows.len() / hidden` rows
    /// stored at KV positions `kv_start..kv_start + t_len` and rotated as
    /// `rope` says (see the module docs); when `logits` is `Some`, write the
    /// last row's `[vocab]` logits into it.
    ///
    /// One command buffer, one encoder, one wait, inside one autorelease
    /// pool: `commandBuffer` and `computeCommandEncoder` hand back
    /// autoreleased objects, and without a pool they would pile up on the
    /// calling thread — about 1.8 KiB per call, i.e. per decoded token of a
    /// long-lived server thread — until that thread exits.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a bad row count, a window
    /// past `max_seq_len`, rotary positions past the angle table, per-row
    /// angles of the wrong length or a short `logits`;
    /// [`MetalGraphError::EncodingFailed`] if a layer lacks the cache slot or
    /// sign vector it needs (nothing is committed then); a failed command
    /// buffer.
    pub fn forward_rows(
        &mut self,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), MetalGraphError> {
        autoreleasepool(|| self.forward_rows_unpooled(hidden_rows, kv_start, rope, logits))
    }

    fn forward_rows_unpooled(
        &mut self,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), MetalGraphError> {
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, kv_start)?;
        if let Some(out) = logits.as_ref() {
            if out.len() < self.cfg.vocab {
                return Err(MetalGraphError::InvalidDimensions(format!(
                    "qwen35 GPU: logits buffer holds {} < vocab {}",
                    out.len(),
                    self.cfg.vocab
                )));
            }
        }
        let binding = self.bind_rope(rope, t_len)?;
        let want_logits = logits.is_some();
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let encoded = (0..self.layers.len())
            .try_for_each(|layer| self.encode_layer(enc, layer, t_len, kv_start, binding))
            .and_then(|()| {
                if want_logits {
                    self.encode_head(enc, t_len)
                } else {
                    Ok(())
                }
            });
        // The encoder is closed on every path; an encoding error drops the
        // command buffer uncommitted.
        enc.end_encoding();
        encoded?;
        commit_and_wait(cmd, "qwen35_forward")?;
        // SAFETY: `commit_and_wait` returned after `wait_until_completed`,
        // so the command buffer is in a terminal state and both timestamp
        // properties are readable.
        let (gpu_start, gpu_end): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        self.last_gpu_seconds = (gpu_end - gpu_start).max(0.0);
        if let Some(out) = logits {
            let vocab = self.cfg.vocab;
            // SAFETY: `logits` holds `vocab` floats written by the command
            // buffer that has just completed.
            let src =
                unsafe { std::slice::from_raw_parts(self.logits.contents().cast::<f32>(), vocab) };
            out[..vocab].copy_from_slice(src);
        }
        Ok(())
    }

    /// [`Self::forward`] one layer at a time, returning the residual stream
    /// after every layer (`[layer][t_len * hidden]`) and the last token's
    /// logits — the GPU counterpart of the CPU forward's per-layer dump.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`]; a layer that fails to encode is not committed,
    /// though the layers before it already ran.
    pub fn forward_with_dump(
        &mut self,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), MetalGraphError> {
        let dump = self.forward_with_dump_rows(
            hidden_rows,
            start_pos,
            Qwen35Rope::Contiguous {
                rope_start: start_pos,
            },
        )?;
        Ok((dump.layers, dump.logits))
    }

    /// [`Self::forward_rows`] one layer at a time, returning what layer 0
    /// read, the residual stream after every layer and the last row's
    /// logits ([`Qwen35ForwardDump`]).
    ///
    /// # Errors
    ///
    /// As [`Self::forward_rows`]; a layer that fails to encode is not
    /// committed, though the layers before it already ran.
    pub fn forward_with_dump_rows(
        &mut self,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
    ) -> Result<Qwen35ForwardDump, MetalGraphError> {
        autoreleasepool(|| self.forward_with_dump_unpooled(hidden_rows, kv_start, rope))
    }

    fn forward_with_dump_unpooled(
        &mut self,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
    ) -> Result<Qwen35ForwardDump, MetalGraphError> {
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, kv_start)?;
        let binding = self.bind_rope(rope, t_len)?;
        let n = t_len * self.cfg.hidden;
        let input = read_buffer(&self.scratch.resid, 0, n);
        let mut dump = Vec::with_capacity(self.layers.len());
        for layer in 0..self.layers.len() {
            let cmd = self.graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let encoded = self.encode_layer(enc, layer, t_len, kv_start, binding);
            enc.end_encoding();
            encoded?;
            commit_and_wait(cmd, "qwen35_forward_layer")?;
            dump.push(read_buffer(&self.scratch.resid, 0, n));
        }
        let cmd = self.graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let encoded = self.encode_head(enc, t_len);
        enc.end_encoding();
        encoded?;
        commit_and_wait(cmd, "qwen35_forward_head")?;
        Ok(Qwen35ForwardDump {
            input,
            layers: dump,
            logits: read_buffer(&self.logits, 0, self.cfg.vocab),
        })
    }

    /// Run layer `layer` alone on `hidden_rows` (its input residual rows),
    /// one kernel per command buffer, capturing every intermediate
    /// activation — the per-kernel parity harness. The layer's KV slot or
    /// recurrent state advances exactly as in [`Self::forward`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], plus an out-of-range `layer`.
    pub fn trace_layer(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> Result<Qwen35LayerTrace, MetalGraphError> {
        self.trace_layer_rows(
            layer,
            hidden_rows,
            start_pos,
            Qwen35Rope::Contiguous {
                rope_start: start_pos,
            },
        )
    }

    /// [`Self::trace_layer`] with the rotary source of
    /// [`Self::forward_rows`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward_rows`], plus an out-of-range `layer`.
    pub fn trace_layer_rows(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
    ) -> Result<Qwen35LayerTrace, MetalGraphError> {
        autoreleasepool(|| self.trace_layer_unpooled(layer, hidden_rows, kv_start, rope))
    }

    fn trace_layer_unpooled(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        kv_start: usize,
        rope: Qwen35Rope<'_>,
    ) -> Result<Qwen35LayerTrace, MetalGraphError> {
        if layer >= self.layers.len() {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: layer {layer} of {}",
                self.layers.len()
            )));
        }
        let t_len = self.load_rows(hidden_rows)?;
        self.check_window(t_len, kv_start)?;
        let binding = self.bind_rope(rope, t_len)?;
        let mut trace = Qwen35LayerTrace::default();
        let stages = self.layer_stages(layer);
        for &stage in stages {
            let cmd = self.graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let encoded = self.encode_stage(enc, layer, stage, t_len, kv_start, binding);
            enc.end_encoding();
            encoded?;
            commit_and_wait(cmd, "qwen35_trace_stage")?;
            for (name, buffer, width) in self.stage_outputs(stage) {
                trace
                    .stages
                    .push((name, read_buffer(buffer, 0, t_len * width)));
            }
        }
        Ok(trace)
    }
}
