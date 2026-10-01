//! The Metal runner's resident state: what a runner for a model keeps on
//! the device ([`HybridMetalFootprint`], computed without building one) and
//! exact rollback points of its recurrent state ([`HybridGpuSnapshot`]).
//!
//! # Snapshots
//!
//! A [`HybridGpuSnapshot`] is the device recurrent state — every
//! Gated-DeltaNet slab (144 MiB for the 27B) and conv window (5.6 MiB) —
//! plus the position the runner had consumed. The KV cache is not copied:
//! the encoder writes every position before any query at or after it
//! reads it, so after a restore to position `p` a contiguous sequence only
//! writes at `p` or later and the stored keys and values below `p` are
//! still the ones the snapshot saw. A snapshot is only meaningful for the
//! sequence it was taken from; a reset starts another one.
//!
//! # Footprint
//!
//! [`HybridMetalRunner::footprint`] reports, for a model the runner serves,
//! the weight bytes the device reads, the part of them it copies even when
//! it binds the file mapping in place, and the encoder's own
//! [`Qwen35Footprint`] (fixed state and per-position buffers). A caller
//! budgets a KV window from it — the device ceiling for a runner alone
//! ([`HybridMetalFootprint::device_ceiling`]), or a budget that adds other
//! residents — before any device memory is committed.

use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::{
    qwen35_footprint, Qwen35DeviceLimits, Qwen35Footprint, Qwen35GpuConfig, Qwen35RecurrentSnapshot,
};

use super::{from_gpu, full_matrices, gpu_config, gpu_matrix, linear_matrices, HybridMetalRunner};
use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::HybridBlock;
use crate::hybrid::model::HybridModel;

/// Bytes of one `f32`.
const F32_BYTES: u64 = 4;

/// What a [`HybridMetalRunner`] for one model keeps resident (see the module
/// docs).
#[derive(Debug, Clone, PartialEq)]
pub struct HybridMetalFootprint {
    config: Qwen35GpuConfig,
    n_full: usize,
    n_linear: usize,
    /// Weight bytes the device reads: every projection as stored, the LM
    /// head and the `f32`-widened `ssm_alpha` / `ssm_beta` rows — what a
    /// built runner's [`HybridMetalRunner::weight_bytes`] reports.
    pub weight_bytes: u64,
    /// The part of [`Self::weight_bytes`] the runner copies even when it
    /// binds the file mapping in place: the widened gates and any
    /// unquantized projection.
    pub copied_when_mapped_bytes: u64,
    /// The encoder's recurrent state, logits, activation scratch and
    /// per-position KV buffers.
    pub device: Qwen35Footprint,
}

impl HybridMetalFootprint {
    /// The geometry a runner for the model is built with.
    #[must_use]
    pub fn config(&self) -> &Qwen35GpuConfig {
        &self.config
    }

    /// Full-attention (KV-cached) layers.
    #[must_use]
    pub fn n_full(&self) -> usize {
        self.n_full
    }

    /// Gated-DeltaNet (recurrent) layers.
    #[must_use]
    pub fn n_linear(&self) -> usize {
        self.n_linear
    }

    /// The longest KV window a device with `limits` keeps resident for a
    /// runner that is the only large resident — the bound construction
    /// enforces (`Qwen35GpuModel::max_context`).
    #[must_use]
    pub fn device_ceiling(&self, limits: &Qwen35DeviceLimits) -> usize {
        limits.context_capacity(&self.config, self.n_full, self.n_linear, self.weight_bytes)
    }

    /// Bytes a runner allocates on the device for a KV window of
    /// `positions` besides the weights it reads in place: the copied
    /// weights, the recurrent state, the logits and scratch, and the
    /// per-position buffers — all of it at construction, before the first
    /// token. With `mapped = false` every weight byte is a copy.
    #[must_use]
    pub fn allocated_bytes(&self, positions: usize, mapped: bool) -> u64 {
        let copied = if mapped {
            self.copied_when_mapped_bytes
        } else {
            self.weight_bytes
        };
        copied.saturating_add(self.device.resident_bytes(positions))
    }
}

/// An exact rollback point of a [`HybridMetalRunner`]: its recurrent state
/// and the position it had consumed (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct HybridGpuSnapshot {
    recurrent: Qwen35RecurrentSnapshot,
    position: usize,
}

impl HybridGpuSnapshot {
    /// Positions the runner had consumed when the snapshot was taken.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Bytes of recurrent state the snapshot holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.recurrent.bytes()
    }
}

impl HybridMetalRunner<'_> {
    /// What a runner for `model` would keep resident, computed without
    /// touching the device.
    ///
    /// # Errors
    ///
    /// As [`Self::check_supported`]: a model the Metal kernels do not serve
    /// has no runner footprint.
    pub fn footprint(model: &HybridModel<'_>) -> ModelResult<HybridMetalFootprint> {
        Self::check_supported(model)?;
        let config = gpu_config(model)?;
        let mut weight_bytes = 0u64;
        let mut copied_when_mapped_bytes = 0u64;
        let mut add = |bytes: usize, always_copied: bool| {
            let bytes = bytes as u64;
            weight_bytes = weight_bytes.saturating_add(bytes);
            if always_copied {
                copied_when_mapped_bytes = copied_when_mapped_bytes.saturating_add(bytes);
            }
        };
        // Both widened gates of a linear layer: `[2 * n_v][hidden]` f32.
        let gate_bytes = 2 * config.n_v_heads * config.hidden * F32_BYTES as usize;
        let (mut n_full, mut n_linear) = (0usize, 0usize);
        for block in model.blocks() {
            match block {
                HybridBlock::Full(b) => {
                    n_full += 1;
                    for (name, layer) in full_matrices(b) {
                        let m = gpu_matrix(b.layer_idx(), name, layer)?;
                        add(m.data.byte_len(), m.data.always_copied());
                    }
                }
                HybridBlock::Linear(b) => {
                    n_linear += 1;
                    for (name, layer) in linear_matrices(b) {
                        let m = gpu_matrix(b.layer_idx(), name, layer)?;
                        add(m.data.byte_len(), m.data.always_copied());
                    }
                    add(gate_bytes, true);
                }
            }
        }
        let head = gpu_matrix(usize::MAX, "output", model.lm_head())?;
        add(head.data.byte_len(), head.data.always_copied());
        let device = qwen35_footprint(&config, n_full, n_linear);
        Ok(HybridMetalFootprint {
            config,
            n_full,
            n_linear,
            weight_bytes,
            copied_when_mapped_bytes,
            device,
        })
    }

    /// Copy the recurrent state and the consumed position to the host.
    ///
    /// Takes `&self`: every call that encodes GPU work waits for its command
    /// buffer before it returns, so nothing can be writing the state while a
    /// shared borrow of the runner exists — and a caller holding only a
    /// shared borrow (an engine's `snapshot_sequence(&self)`) can take one.
    ///
    /// # Errors
    ///
    /// None today: the state lives in shared-storage buffers the host reads
    /// directly. The `Result` is the contract for a device-private state,
    /// whose readback is a command buffer that can fail.
    pub fn snapshot_state(&self) -> ModelResult<HybridGpuSnapshot> {
        Ok(HybridGpuSnapshot {
            recurrent: self.gpu.snapshot_state(),
            position: self.token_count,
        })
    }

    /// Return the recurrent state and the position count to `snapshot`.
    ///
    /// The caller is responsible for `snapshot` belonging to this runner's
    /// current sequence (see the module docs); geometry is checked here.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] for a snapshot position past the
    /// KV window, and [`ModelError::Kernel`] carrying
    /// `KernelError::UnsupportedOperation` for a snapshot of another
    /// geometry. Nothing is changed on either refusal.
    pub fn restore_state(&mut self, snapshot: &HybridGpuSnapshot) -> ModelResult<()> {
        if snapshot.position > self.max_seq_len {
            return Err(ModelError::PositionOutOfRange {
                pos: snapshot.position,
                max: self.max_seq_len,
            });
        }
        self.gpu
            .restore_state(&snapshot.recurrent)
            .map_err(from_gpu)?;
        self.token_count = snapshot.position;
        Ok(())
    }

    /// Bytes of the device recurrent state (slabs and conv windows).
    #[must_use]
    pub fn recurrent_state_bytes(&self) -> u64 {
        self.gpu.recurrent_state_bytes()
    }

    /// Bytes the Metal device reports as currently allocated by this
    /// process (`MTLDevice.currentAllocatedSize`: every session's buffers,
    /// including no-copy buffers over a file mapping).
    #[must_use]
    pub fn device_allocated_bytes(&self) -> u64 {
        self.gpu.device_allocated_bytes()
    }

    /// The resident footprint of the encoder this runner built.
    #[must_use]
    pub fn device_footprint(&self) -> Qwen35Footprint {
        self.gpu.footprint()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::TensorType;
    use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::Qwen35MappedRegion;
    use oxibonsai_kernels::gpu_backend::metal_graph::MetalGraph;
    use oxibonsai_kernels::{cpu_kernel_tier, KernelDispatcher, MetalGraphError};

    use super::*;
    use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};

    /// KV window of the fixture models.
    const MAX_SEQ: usize = 64;
    /// Prompt prefilled before the snapshot.
    const PROMPT: [u32; 6] = [3, 9, 27, 81, 243, 5];

    /// `false` only on a host without a Metal device; any other failure to
    /// open it is a test failure, never a skip.
    fn metal_available() -> bool {
        match MetalGraph::global() {
            Ok(_) => true,
            Err(MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the combined Metal library must build on this device: {e}"),
        }
    }

    fn fixture(quant: TensorType) -> Vec<u8> {
        synthetic_gguf(
            FixtureShape::default(),
            FixtureOptions {
                quant,
                ..FixtureOptions::default()
            },
        )
    }

    fn cpu_model<'a>(gguf: &'a GgufFile<'a>) -> HybridModel<'a> {
        let config = HybridModel::config_from_gguf(gguf).expect("fixture config");
        let kernel = Arc::new(KernelDispatcher::with_tier(cpu_kernel_tier()));
        HybridModel::from_gguf_with(gguf, config, MAX_SEQ, &kernel).expect("fixture loads")
    }

    fn argmax(values: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in values.iter().enumerate() {
            if v > values[best] {
                best = i;
            }
        }
        u32::try_from(best).expect("token id")
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Decode `steps` greedy tokens from `logits` at `start`, returning every
    /// logit row.
    fn decode(
        runner: &mut HybridMetalRunner<'_>,
        mut logits: Vec<f32>,
        start: usize,
        steps: usize,
    ) -> Vec<Vec<f32>> {
        let mut rows = Vec::with_capacity(steps);
        for step in 0..steps {
            let token = argmax(&logits);
            runner
                .forward_into(token, start + step, &mut logits)
                .expect("decode");
            rows.push(logits.clone());
        }
        rows
    }

    /// Snapshot at position `p`, decode eight tokens, wander off, restore,
    /// decode the same eight: every logit row is bit-identical, and the
    /// position count follows the snapshot.
    #[test]
    fn a_restored_snapshot_replays_eight_tokens_bit_for_bit() {
        if !metal_available() {
            return;
        }
        for quant in [TensorType::PQ2_0, TensorType::PTQ1_0] {
            let bytes = fixture(quant);
            let gguf = GgufFile::parse(&bytes).expect("fixture parses");
            let model = cpu_model(&gguf);
            let mut runner = HybridMetalRunner::new(&model).expect("runner builds");
            let vocab = runner.vocab_size();
            let mut logits = vec![0.0f32; vocab];
            runner
                .forward_prefill(&PROMPT, 0, &mut logits)
                .expect("prefill");
            assert_eq!(runner.token_count(), PROMPT.len());

            let snapshot = runner.snapshot_state().expect("snapshot");
            assert_eq!(snapshot.position(), PROMPT.len());
            assert_eq!(snapshot.bytes(), runner.recurrent_state_bytes());
            let reference = decode(&mut runner, logits.clone(), PROMPT.len(), 8);
            assert_eq!(runner.token_count(), PROMPT.len() + 8);
            // Wander off past the reference, then come back.
            let mut scratch = vec![0.0f32; vocab];
            for (i, token) in [7u32, 11, 13].into_iter().enumerate() {
                runner
                    .forward_into(token, PROMPT.len() + 8 + i, &mut scratch)
                    .expect("wander");
            }
            runner.restore_state(&snapshot).expect("restore");
            assert_eq!(runner.token_count(), PROMPT.len());
            let replayed = decode(&mut runner, logits.clone(), PROMPT.len(), 8);
            for (step, (a, b)) in replayed.iter().zip(&reference).enumerate() {
                assert_eq!(bits(a), bits(b), "{quant:?} replayed step {step}");
            }

            // A restore past the window is refused and changes nothing.
            let far = HybridGpuSnapshot {
                recurrent: snapshot.recurrent.clone(),
                position: MAX_SEQ + 1,
            };
            let before = runner.snapshot_state().expect("snapshot");
            assert!(matches!(
                runner.restore_state(&far),
                Err(ModelError::PositionOutOfRange { .. })
            ));
            assert_eq!(runner.snapshot_state().expect("snapshot"), before);

            // A reset clears the state and the count.
            runner.reset();
            assert_eq!(runner.token_count(), 0);
        }
    }

    /// Validation refusals happen before any GPU work, so they leave the
    /// sequence (state and count) exactly as it was.
    #[test]
    fn a_refused_call_leaves_the_sequence_untouched() {
        if !metal_available() {
            return;
        }
        let bytes = fixture(TensorType::PQ2_0);
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let model = cpu_model(&gguf);
        let mut runner = HybridMetalRunner::new(&model).expect("runner builds");
        let vocab = runner.vocab_size();
        let mut logits = vec![0.0f32; vocab];
        runner
            .forward_prefill(&PROMPT, 0, &mut logits)
            .expect("prefill");
        let before = runner.snapshot_state().expect("snapshot");
        let bad = u32::try_from(vocab).expect("vocab fits u32");
        assert!(matches!(
            runner.forward_prefill(&[1, 2, bad], PROMPT.len(), &mut logits),
            Err(ModelError::PositionOutOfRange { .. })
        ));
        assert!(runner.forward_into(1, MAX_SEQ, &mut logits).is_err());
        assert!(runner
            .forward_into(1, PROMPT.len(), &mut [0.0f32; 3])
            .is_err());
        assert_eq!(runner.snapshot_state().expect("snapshot"), before);
        assert_eq!(runner.token_count(), PROMPT.len());
    }

    /// The footprint computed from the model alone is what a built runner
    /// holds, and `new_in_place` binds a page-aligned image in place and a
    /// heap image by copy, both computing the same logits.
    #[test]
    fn the_footprint_matches_a_built_runner_and_in_place_residency_follows_alignment() {
        if !metal_available() {
            return;
        }
        let bytes = fixture(TensorType::PTQ1_0);
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let model = cpu_model(&gguf);
        let footprint = HybridMetalRunner::footprint(&model).expect("footprint");
        let runner = HybridMetalRunner::new(&model).expect("runner builds");
        assert_eq!(footprint.weight_bytes, runner.weight_bytes());
        assert_eq!(footprint.device, runner.device_footprint());
        assert_eq!(
            footprint.device.recurrent_bytes,
            runner.recurrent_state_bytes()
        );
        assert_eq!(
            footprint.device.kv_bytes_per_position * MAX_SEQ as u64,
            runner.kv_cache_bytes()
        );
        assert!(
            footprint.copied_when_mapped_bytes > 0,
            "the gates are copied"
        );
        assert!(footprint.copied_when_mapped_bytes < footprint.weight_bytes);
        assert_eq!(
            footprint.allocated_bytes(MAX_SEQ, false),
            footprint.weight_bytes + footprint.device.resident_bytes(MAX_SEQ)
        );
        assert!(runner.device_allocated_bytes() >= runner.kv_cache_bytes());
        let limits = Qwen35DeviceLimits::of_shared_device().expect("device limits");
        assert!(footprint.device_ceiling(&limits) >= MAX_SEQ);

        // A page-aligned copy of the same file: bound in place.
        let page = 16_384usize;
        let layout = std::alloc::Layout::from_size_align(bytes.len().div_ceil(page) * page, page)
            .expect("layout");
        // SAFETY: non-zero size; freed at the end of the test.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        // SAFETY: the allocation is at least `bytes.len()` long and does not
        // overlap `bytes`; it lives until the dealloc below, after every
        // borrow of it is gone.
        let aligned: &[u8] = unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
            std::slice::from_raw_parts(base, bytes.len())
        };
        let mut heap_logits = vec![0.0f32; runner.vocab_size()];
        let mut mapped_logits = heap_logits.clone();
        {
            let aligned_gguf = GgufFile::parse(aligned).expect("aligned copy parses");
            let aligned_model = cpu_model(&aligned_gguf);
            let mut in_place =
                HybridMetalRunner::new_in_place(&aligned_model, aligned).expect("in place");
            assert!(
                in_place.is_mapped(),
                "a page-aligned image is bound in place"
            );
            in_place
                .forward_prefill(&PROMPT, 0, &mut mapped_logits)
                .expect("mapped prefill");
        }
        // An unaligned view can never be bound in place: the weights are
        // copied, whatever buffer they live in.
        let heap = &bytes[1..];
        assert!(Qwen35MappedRegion::page_aligned(heap).is_none());
        let mut copied = HybridMetalRunner::new_in_place(&model, heap).expect("copied");
        assert!(!copied.is_mapped(), "an unaligned image is copied");
        copied
            .forward_prefill(&PROMPT, 0, &mut heap_logits)
            .expect("copied prefill");
        assert_eq!(bits(&heap_logits), bits(&mapped_logits));
        // SAFETY: allocated above with this layout; nothing borrows it now.
        unsafe { std::alloc::dealloc(base, layout) };
    }
}
