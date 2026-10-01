//! The engine side of the Metal hybrid runner: which executor a `qwen35`
//! (PrismML Bonsai 2) engine decodes on, the KV window a Metal-backed engine
//! is wired with, and the glue the seam's forward, prefill, reset and
//! snapshot arms call.
//!
//! # Backend choice
//!
//! [`hybrid_backend_plan`] decides, for a hybrid model already bound on the
//! CPU, whether a [`HybridMetalRunner`](oxibonsai_model::hybrid::metal::HybridMetalRunner)
//! can serve it on this host: a Metal build, a Metal device (probed
//! explicitly — "no device" is `MetalGraphError::DeviceNotFound` and nothing
//! else), a geometry and weight formats the kernels serve
//! (`HybridMetalRunner::check_supported`), and a KV window that fits
//! ([`plan_hybrid_metal_window`]). `--backend metal` turns a "no" into the
//! typed `HYBRID_GPU_BACKEND_UNSUPPORTED` refusal naming the reason;
//! `--backend auto` falls back to the CPU tier with one `info` line naming
//! it; `--backend cpu` never asks.
//!
//! # Residents and the KV window
//!
//! The engine keeps the CPU [`HybridModel`] beside the runner: the runner
//! borrows its weight slices (read in place from the same file mapping, so
//! the weights are resident once), and the embedding pass runs on the CPU
//! model's own layers. Its KV cache and recurrent state therefore stay
//! allocated next to the runner's — two KV caches and two recurrent states
//! per engine. Unified memory makes that a host-RAM question, so the window
//! is budgeted with the runtime's Appendix A.3 formula
//! ([`crate::config::max_context_for_budget`]'s reserves) over **both**
//! residents: `min(requested, declared context, RAM guard of the CPU model
//! alone, budget for the residents kept, the runner's device ceiling)`.
//! For the 27B on a 24 GiB M3 the resident budget is 83 968 positions
//! (`PQ2_0`) / 93 184 (`PTQ1_0`), against 171 008 / 190 464 for a runner
//! alone and a device ceiling of 178 034 / 196 246; the shipped default of
//! 8192 is under all of them.
//!
//! The runner allocates its `f16` KV cache for the **whole** window at
//! construction (64 KiB per position for the 27B — a 32 768-position window
//! is 2 GiB of device memory before the first token), unlike the CPU
//! model's cache, which grows as positions are reached.
//! [`HybridMetalWindow::runner_allocated_bytes`] reports that up-front
//! figure.

use std::fmt;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_model::hybrid::HybridModel;

use crate::config::{ACTIVATION_RESERVE_BYTES, OS_RESERVE_DIVISOR, OS_RESERVE_MIN_BYTES};

/// How a Metal-backed hybrid engine names its executor wherever a kernel
/// tier is reported (`kernel_label`, the tier reason, `oxibonsai info`).
pub const HYBRID_RUNNER_LABEL: &str = "Metal (hybrid runner)";

/// Every RAM-derived window is a whole number of these (Appendix A.3).
const WINDOW_GRANULE: usize = 1024;

// ─────────────────────────────────────────────────────────────────────────
// Which executor a hybrid engine decodes on
// ─────────────────────────────────────────────────────────────────────────

/// The executor a hybrid (`qwen35`) engine decodes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HybridBackend {
    /// The CPU [`HybridModel`] on the best CPU SIMD tier.
    Cpu,
    /// The Metal hybrid runner.
    Metal,
}

impl HybridBackend {
    /// Stable lower-case name (`"cpu"`, `"metal"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

impl fmt::Display for HybridBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The KV window of a Metal-backed hybrid engine
// ─────────────────────────────────────────────────────────────────────────

/// Which per-sequence buffers share the host's memory with the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridResidents {
    /// The CPU model keeps its KV cache and recurrent state beside the
    /// runner's (what an engine does: the embedding pass runs on it).
    CpuModelAndRunner,
    /// Only the runner's buffers.
    RunnerOnly,
}

impl HybridResidents {
    /// Human-readable name of the residents.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::CpuModelAndRunner => {
                "the CPU model's KV cache and recurrent state beside the Metal runner's"
            }
            Self::RunnerOnly => "the Metal runner alone",
        }
    }
}

/// One limit that can bound a Metal-backed hybrid engine's KV window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridWindowLimit {
    /// The model's declared context (`qwen35.context_length`).
    Declared,
    /// The Appendix A.3 guard for the CPU model alone.
    RamGuard,
    /// The Appendix A.3 budget over the residents actually kept.
    ResidentBudget,
    /// The runner's device ceiling (`Qwen35GpuModel::max_context`).
    DeviceCeiling,
}

impl HybridWindowLimit {
    /// Stable name of the limit.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Declared => "declared context",
            Self::RamGuard => "RAM guard (CPU model alone)",
            Self::ResidentBudget => "RAM budget for the residents kept",
            Self::DeviceCeiling => "Metal device ceiling",
        }
    }
}

/// Every input of the window decision as plain numbers, so it is testable
/// without a device or a model (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridWindowInputs {
    /// The window asked for (`--ctx`, or the default).
    pub requested: usize,
    /// The model's declared context.
    pub declared: usize,
    /// Physical RAM, `None` when the host cannot report it (no RAM-derived
    /// limit applies then).
    pub total_ram_bytes: Option<u64>,
    /// Bytes the weights occupy: the GGUF image both executors read.
    pub file_bytes: u64,
    /// The CPU model's recurrent state.
    pub cpu_recurrent_bytes: u64,
    /// The CPU model's `f16` KV bytes per position.
    pub cpu_kv_bytes_per_position: u64,
    /// The CPU model's other per-position bytes (its rope angle table).
    pub cpu_other_bytes_per_position: u64,
    /// The runner's bytes independent of the window: copied weights,
    /// recurrent state, logits, activation scratch and host staging.
    pub runner_fixed_bytes: u64,
    /// The runner's bytes per position (KV, rope angles, a score row).
    pub runner_bytes_per_position: u64,
    /// The runner's `f16` KV bytes per position (a subset of
    /// [`Self::runner_bytes_per_position`]).
    pub runner_kv_bytes_per_position: u64,
    /// The runner's device ceiling.
    pub device_ceiling: usize,
    /// The residents the engine keeps.
    pub residents: HybridResidents,
}

/// The KV window a Metal-backed hybrid engine is wired with, and every limit
/// that went into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMetalWindow {
    /// The wired window.
    pub window: usize,
    /// The window asked for.
    pub requested: usize,
    /// The model's declared context.
    pub declared: usize,
    /// The Appendix A.3 guard for the CPU model alone (what `oxibonsai
    /// info` calls the RAM-derived max `--ctx`); `None` when RAM is unknown.
    pub ram_guard: Option<usize>,
    /// The Appendix A.3 budget for [`Self::residents`]; `None` when RAM is
    /// unknown.
    pub resident_budget: Option<usize>,
    /// The Appendix A.3 budget for a runner alone — the figure that would
    /// apply without the CPU model's buffers; `None` when RAM is unknown.
    pub runner_alone_budget: Option<usize>,
    /// The runner's device ceiling.
    pub device_ceiling: usize,
    /// The residents the budget counts.
    pub residents: HybridResidents,
    /// Every limit that clamps [`Self::requested`] on its own (a RAM-derived
    /// limit that only inherits the declared-context cap is not listed).
    pub limits_applied: Vec<HybridWindowLimit>,
    /// Bytes the runner allocates beside the mapping at construction for
    /// [`Self::window`]: copied weights, recurrent state, logits, scratch
    /// and the per-position buffers.
    pub runner_allocated_bytes: u64,
    /// The runner's `f16` KV cache for [`Self::window`] (a subset of
    /// [`Self::runner_allocated_bytes`]).
    pub runner_kv_bytes: u64,
}

impl HybridMetalWindow {
    /// Whether any limit clamped the request.
    #[must_use]
    pub fn clamped(&self) -> bool {
        self.window < self.requested
    }

    /// Every limit with its value, e.g. `declared context 262144, RAM guard
    /// (CPU model alone) 178176, ...` — the text of the one warning a
    /// clamped window logs, and of `oxibonsai info`.
    #[must_use]
    pub fn describe_limits(&self) -> String {
        let show = |v: Option<usize>| v.map_or_else(|| "unknown".to_string(), |v| v.to_string());
        format!(
            "{} {}, {} {}, {} {} ({}), {} {}; a runner alone would fit {}",
            HybridWindowLimit::Declared.as_str(),
            self.declared,
            HybridWindowLimit::RamGuard.as_str(),
            show(self.ram_guard),
            HybridWindowLimit::ResidentBudget.as_str(),
            show(self.resident_budget),
            self.residents.describe(),
            HybridWindowLimit::DeviceCeiling.as_str(),
            self.device_ceiling,
            show(self.runner_alone_budget),
        )
    }

    /// One-line report: the wired window, its limits and the up-front
    /// device allocation.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "KV window {} positions (requested {}; {}); the runner allocates {} at load, {} of it \
             f16 KV",
            self.window,
            self.requested,
            self.describe_limits(),
            gib(self.runner_allocated_bytes),
            gib(self.runner_kv_bytes),
        )
    }
}

/// Bytes rendered as GiB with two decimals.
fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Appendix A.3's usable budget for `fixed` window-independent bytes and
/// `per_position` bytes per position, floored to a whole granule and capped
/// at `declared`.
fn ram_budget(
    total_ram: u64,
    file_bytes: u64,
    fixed: u64,
    per_position: u64,
    declared: usize,
) -> usize {
    let os_reserve = (total_ram / OS_RESERVE_DIVISOR).max(OS_RESERVE_MIN_BYTES);
    let usable = total_ram
        .saturating_sub(file_bytes)
        .saturating_sub(fixed)
        .saturating_sub(ACTIVATION_RESERVE_BYTES)
        .saturating_sub(os_reserve);
    let positions = match usable.checked_div(per_position) {
        Some(n) => usize::try_from(n).unwrap_or(usize::MAX),
        None => declared,
    };
    (positions.min(declared) / WINDOW_GRANULE) * WINDOW_GRANULE
}

/// Resolve the KV window of a Metal-backed hybrid engine (see the module
/// docs): `min(requested, declared, RAM guard, resident budget, device
/// ceiling)`, with every limit recorded.
#[must_use]
pub fn plan_hybrid_metal_window(inputs: &HybridWindowInputs) -> HybridMetalWindow {
    let declared = inputs.declared.max(1);
    let ram_guard = inputs.total_ram_bytes.map(|total| {
        crate::config::max_context_for_budget(
            total,
            inputs.file_bytes,
            inputs.cpu_recurrent_bytes,
            inputs.cpu_kv_bytes_per_position,
            declared,
        )
    });
    let runner_alone_budget = inputs.total_ram_bytes.map(|total| {
        ram_budget(
            total,
            inputs.file_bytes,
            inputs.runner_fixed_bytes,
            inputs.runner_bytes_per_position,
            declared,
        )
    });
    let resident_budget = match inputs.residents {
        HybridResidents::RunnerOnly => runner_alone_budget,
        HybridResidents::CpuModelAndRunner => inputs.total_ram_bytes.map(|total| {
            ram_budget(
                total,
                inputs.file_bytes,
                inputs
                    .runner_fixed_bytes
                    .saturating_add(inputs.cpu_recurrent_bytes),
                inputs
                    .runner_bytes_per_position
                    .saturating_add(inputs.cpu_kv_bytes_per_position)
                    .saturating_add(inputs.cpu_other_bytes_per_position),
                declared,
            )
        }),
    };
    // The RAM-derived limits are capped at the declared context by
    // construction, so one that merely inherits that cap is not a limit of
    // its own: it counts as applied only below the declared context too.
    let candidates = [
        (HybridWindowLimit::Declared, Some(declared), usize::MAX),
        (HybridWindowLimit::RamGuard, ram_guard, declared),
        (HybridWindowLimit::ResidentBudget, resident_budget, declared),
        (
            HybridWindowLimit::DeviceCeiling,
            Some(inputs.device_ceiling),
            usize::MAX,
        ),
    ];
    let mut window = inputs.requested;
    let mut limits_applied = Vec::new();
    for (limit, value, own_below) in candidates {
        if let Some(value) = value {
            if value < inputs.requested && value < own_below {
                limits_applied.push(limit);
            }
            window = window.min(value);
        }
    }
    let runner_kv_bytes = (window as u64).saturating_mul(inputs.runner_kv_bytes_per_position);
    let runner_allocated_bytes = inputs
        .runner_fixed_bytes
        .saturating_add((window as u64).saturating_mul(inputs.runner_bytes_per_position));
    HybridMetalWindow {
        window,
        requested: inputs.requested,
        declared,
        ram_guard,
        resident_budget,
        runner_alone_budget,
        device_ceiling: inputs.device_ceiling,
        residents: inputs.residents,
        limits_applied,
        runner_allocated_bytes,
        runner_kv_bytes,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Whether a runner serves this model on this host
// ─────────────────────────────────────────────────────────────────────────

/// What `--backend auto` does with a hybrid model on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridBackendPlan {
    /// A Metal runner serves it, with this window.
    Metal {
        /// The KV window and its limits.
        window: HybridMetalWindow,
        /// Whether the runner reads the weights in place from the file
        /// mapping (`false`: it copies them — a GGUF image in a heap
        /// buffer).
        mapped: bool,
    },
    /// It runs on the CPU tier, for this reason.
    Cpu {
        /// Why no runner serves it (the device, the build, the geometry or
        /// format, or the window).
        reason: String,
    },
}

impl HybridBackendPlan {
    /// The executor the plan selects.
    #[must_use]
    pub fn backend(&self) -> HybridBackend {
        match self {
            Self::Metal { .. } => HybridBackend::Metal,
            Self::Cpu { .. } => HybridBackend::Cpu,
        }
    }
}

/// Decide whether a Metal runner serves `model` (bound from `gguf`) on this
/// host with a window of `requested` positions — without building one. The
/// device is only opened to read its limits.
///
/// This is the decision the engine's `load_hybrid` acts on and what
/// `oxibonsai info` reports.
#[must_use]
pub fn hybrid_backend_plan(
    gguf: &GgufFile<'_>,
    model: &HybridModel<'_>,
    requested: usize,
) -> HybridBackendPlan {
    #[cfg(test)]
    if let Some(reason) = FORCED_CPU_PLAN.with(|forced| forced.borrow().clone()) {
        return HybridBackendPlan::Cpu { reason };
    }
    metal::plan(gguf, model, requested)
}

#[cfg(test)]
thread_local! {
    /// Test-only: the reason [`hybrid_backend_plan`] answers with on this
    /// thread while a [`ForcedCpuPlan`] lives.
    static FORCED_CPU_PLAN: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Test-only: while it lives, [`hybrid_backend_plan`] answers "no runner,
/// for `reason`" on this thread, so the refusal and fallback paths run on a
/// host that does have a Metal device. Thread-local, so tests running in
/// parallel are unaffected.
#[cfg(test)]
pub(crate) struct ForcedCpuPlan;

#[cfg(test)]
impl ForcedCpuPlan {
    pub(crate) fn enter(reason: &str) -> Self {
        FORCED_CPU_PLAN.with(|forced| *forced.borrow_mut() = Some(reason.to_string()));
        Self
    }
}

#[cfg(test)]
impl Drop for ForcedCpuPlan {
    fn drop(&mut self) {
        FORCED_CPU_PLAN.with(|forced| *forced.borrow_mut() = None);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::{
        Qwen35DeviceLimits, Qwen35MappedRegion,
    };
    use oxibonsai_kernels::MetalGraphError;
    use oxibonsai_model::hybrid::metal::{HybridGpuSnapshot, HybridMetalRunner};
    use oxibonsai_model::hybrid::HybridModel;

    use super::{
        plan_hybrid_metal_window, HybridBackendPlan, HybridMetalWindow, HybridResidents,
        HybridWindowInputs,
    };
    use crate::error::{RuntimeError, RuntimeResult};

    /// Bytes of one `f16` element (the hybrid model's KV element).
    const KV_ELEM_BYTES: u64 = 2;
    /// Bytes of one `f32` element (rope angles, the runner's host staging).
    const F32_BYTES: u64 = 4;

    /// The CPU model's per-position bytes: `(f16 KV, rope angles)`.
    fn cpu_bytes_per_position(model: &HybridModel<'_>) -> (u64, u64) {
        let c = model.config();
        let kv = (c.num_full_layers() as u64)
            .saturating_mul(c.base.num_kv_heads as u64)
            .saturating_mul((c.base.head_dim + c.base.value_length) as u64)
            .saturating_mul(KV_ELEM_BYTES);
        // `RopeTables`: a cos and a sin row of `n_rot / 2` angles per
        // position.
        let rope = (c.rope_dimension_count as u64).saturating_mul(F32_BYTES);
        (kv, rope)
    }

    pub(super) fn plan(
        gguf: &GgufFile<'_>,
        model: &HybridModel<'_>,
        requested: usize,
    ) -> HybridBackendPlan {
        let limits = match Qwen35DeviceLimits::of_shared_device() {
            Ok(limits) => limits,
            Err(MetalGraphError::DeviceNotFound) => {
                return HybridBackendPlan::Cpu {
                    reason: "no Metal device was found on this host".to_string(),
                }
            }
            Err(e) => {
                return HybridBackendPlan::Cpu {
                    reason: format!("the Metal device could not be opened: {e}"),
                }
            }
        };
        let footprint = match HybridMetalRunner::footprint(model) {
            Ok(footprint) => footprint,
            Err(e) => {
                return HybridBackendPlan::Cpu {
                    reason: format!("the Metal hybrid runner does not serve this model: {e}"),
                }
            }
        };
        let mapped = Qwen35MappedRegion::page_aligned(gguf.data).is_some();
        let (cpu_kv, cpu_rope) = cpu_bytes_per_position(model);
        let config = footprint.config();
        // The runner's host staging for one call's embedded rows.
        let staging = (config.max_batch * config.hidden) as u64 * F32_BYTES;
        let inputs = HybridWindowInputs {
            requested,
            declared: model.config().base.max_context_length,
            total_ram_bytes: crate::config::total_ram_bytes(),
            file_bytes: gguf.data.len() as u64,
            cpu_recurrent_bytes: model.recurrent().memory_bytes() as u64,
            cpu_kv_bytes_per_position: cpu_kv,
            cpu_other_bytes_per_position: cpu_rope,
            runner_fixed_bytes: footprint.allocated_bytes(0, mapped).saturating_add(staging),
            runner_bytes_per_position: footprint.device.per_position_bytes,
            runner_kv_bytes_per_position: footprint.device.kv_bytes_per_position,
            device_ceiling: footprint.device_ceiling(&limits),
            residents: HybridResidents::CpuModelAndRunner,
        };
        let window = plan_hybrid_metal_window(&inputs);
        if window.window == 0 {
            return HybridBackendPlan::Cpu {
                reason: format!(
                    "the Metal hybrid runner has no room for a KV window on this host: {}",
                    window.describe_limits()
                ),
            };
        }
        HybridBackendPlan::Metal { window, mapped }
    }

    /// A Metal-backed hybrid engine's runner and the window it was wired
    /// with.
    pub(crate) struct HybridGpu<'a> {
        runner: HybridMetalRunner<'a>,
        window: HybridMetalWindow,
    }

    impl std::fmt::Debug for HybridGpu<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HybridGpu")
                .field("runner", &self.runner)
                .field("window", &self.window.window)
                .finish()
        }
    }

    /// The seam's snapshot of a Metal-backed hybrid sequence.
    pub(crate) type HybridGpuState = HybridGpuSnapshot;

    impl<'a> HybridGpu<'a> {
        /// Build the runner for `model` (bound at `window.window`) over the
        /// GGUF image it was loaded from.
        ///
        /// # Errors
        ///
        /// The reason the runner could not be built, for the typed refusal
        /// or the CPU fallback's log line.
        pub(crate) fn build(
            gguf: &'a GgufFile<'a>,
            model: &HybridModel<'a>,
            window: HybridMetalWindow,
        ) -> Result<Self, String> {
            let runner = HybridMetalRunner::new_in_place(model, gguf.data)
                .map_err(|e| format!("the Metal hybrid runner could not be built: {e}"))?;
            if runner.max_seq_len() != window.window {
                return Err(format!(
                    "the Metal hybrid runner was built with a {}-position window, not the planned \
                     {}",
                    runner.max_seq_len(),
                    window.window
                ));
            }
            Ok(Self { runner, window })
        }

        pub(crate) fn window(&self) -> &HybridMetalWindow {
            &self.window
        }

        pub(crate) fn token_count(&self) -> usize {
            self.runner.token_count()
        }

        pub(crate) fn is_mapped(&self) -> bool {
            self.runner.is_mapped()
        }

        pub(crate) fn kv_cache_bytes(&self) -> u64 {
            self.runner.kv_cache_bytes()
        }

        pub(crate) fn recurrent_bytes(&self) -> u64 {
            self.runner.recurrent_state_bytes()
        }

        pub(crate) fn device_allocated_bytes(&self) -> u64 {
            self.runner.device_allocated_bytes()
        }

        pub(crate) fn reset(&mut self) {
            self.runner.reset();
        }

        /// Single-token decode at `pos`.
        pub(crate) fn forward_logits(&mut self, token: u32, pos: usize) -> RuntimeResult<Vec<f32>> {
            let mut logits = vec![0.0f32; self.runner.vocab_size()];
            self.runner.forward_into(token, pos, &mut logits)?;
            Ok(logits)
        }

        /// Prefill `tokens` from `pos_start` in calls of at most `chunk`
        /// tokens (the runner splits further at its own batch size),
        /// returning the last token's logits.
        pub(crate) fn prefill_logits(
            &mut self,
            tokens: &[u32],
            pos_start: usize,
            chunk: usize,
        ) -> RuntimeResult<Vec<f32>> {
            let mut logits = vec![0.0f32; self.runner.vocab_size()];
            let chunk = chunk.clamp(1, self.runner.max_batch());
            for (i, window) in tokens.chunks(chunk).enumerate() {
                self.runner
                    .forward_prefill(window, pos_start + i * chunk, &mut logits)?;
            }
            Ok(logits)
        }

        pub(crate) fn snapshot(&self) -> RuntimeResult<HybridGpuState> {
            Ok(self.runner.snapshot_state()?)
        }

        pub(crate) fn restore(&mut self, snapshot: &HybridGpuState) -> RuntimeResult<()> {
            self.runner
                .restore_state(snapshot)
                .map_err(RuntimeError::Model)
        }
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod metal {
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_model::hybrid::HybridModel;

    use super::{HybridBackendPlan, HybridMetalWindow};
    use crate::error::RuntimeResult;

    pub(super) fn plan(
        _gguf: &GgufFile<'_>,
        _model: &HybridModel<'_>,
        _requested: usize,
    ) -> HybridBackendPlan {
        HybridBackendPlan::Cpu {
            reason: "this build has no Metal backend compiled in (macOS with the `metal` feature \
                     is required)"
                .to_string(),
        }
    }

    /// Uninhabited without the Metal backend: no hybrid engine holds one.
    pub(crate) struct HybridGpu<'a> {
        never: std::convert::Infallible,
        _borrow: std::marker::PhantomData<&'a ()>,
    }

    impl std::fmt::Debug for HybridGpu<'_> {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.never {}
        }
    }

    /// Uninhabited without the Metal backend.
    #[derive(Debug, Clone)]
    pub(crate) enum HybridGpuState {}

    impl<'a> HybridGpu<'a> {
        pub(crate) fn build(
            _gguf: &'a GgufFile<'a>,
            _model: &HybridModel<'a>,
            _window: HybridMetalWindow,
        ) -> Result<Self, String> {
            Err(
                "this build has no Metal backend compiled in (macOS with the `metal` feature is \
                 required)"
                    .to_string(),
            )
        }

        pub(crate) fn window(&self) -> &HybridMetalWindow {
            match self.never {}
        }

        pub(crate) fn token_count(&self) -> usize {
            match self.never {}
        }

        pub(crate) fn is_mapped(&self) -> bool {
            match self.never {}
        }

        pub(crate) fn kv_cache_bytes(&self) -> u64 {
            match self.never {}
        }

        pub(crate) fn recurrent_bytes(&self) -> u64 {
            match self.never {}
        }

        pub(crate) fn device_allocated_bytes(&self) -> u64 {
            match self.never {}
        }

        pub(crate) fn reset(&mut self) {
            match self.never {}
        }

        pub(crate) fn forward_logits(
            &mut self,
            _token: u32,
            _pos: usize,
        ) -> RuntimeResult<Vec<f32>> {
            match self.never {}
        }

        pub(crate) fn prefill_logits(
            &mut self,
            _tokens: &[u32],
            _pos_start: usize,
            _chunk: usize,
        ) -> RuntimeResult<Vec<f32>> {
            match self.never {}
        }

        pub(crate) fn snapshot(&self) -> RuntimeResult<HybridGpuState> {
            match self.never {}
        }

        pub(crate) fn restore(&mut self, _snapshot: &HybridGpuState) -> RuntimeResult<()> {
            match self.never {}
        }
    }
}

pub(crate) use metal::{HybridGpu, HybridGpuState};

#[cfg(test)]
mod tests {
    use super::*;

    /// The documented 24 GiB M3 (`hw.memsize`).
    const M3_RAM: u64 = 25_769_803_776;
    /// The two release files.
    const PQ2_0_FILE: u64 = 7_206_168_928;
    const PTQ1_0_FILE: u64 = 5_946_648_928;
    /// The runner's device ceilings on that M3 (`Qwen35GpuModel::max_context`).
    const PQ2_0_DEVICE_CEILING: usize = 178_034;
    const PTQ1_0_DEVICE_CEILING: usize = 196_246;
    /// Recurrent state of one 27B sequence (either executor).
    const RECURRENT: u64 = 156_893_184;
    /// The runner's window-independent bytes for the 27B: the widened
    /// `ssm_alpha`/`ssm_beta` rows, its recurrent state, the logits row, the
    /// activation scratch at a 512-token chunk and the host staging rows.
    const RUNNER_FIXED: u64 =
        48 * 2 * 48 * 5120 * 4 + RECURRENT + 248_320 * 4 + 140_384 * 4 * 512 + 512 * 5120 * 4;

    fn inputs_27b(file_bytes: u64, device_ceiling: usize, requested: usize) -> HybridWindowInputs {
        HybridWindowInputs {
            requested,
            declared: 262_144,
            total_ram_bytes: Some(M3_RAM),
            file_bytes,
            cpu_recurrent_bytes: RECURRENT,
            cpu_kv_bytes_per_position: 65_536,
            cpu_other_bytes_per_position: 256,
            runner_fixed_bytes: RUNNER_FIXED,
            runner_bytes_per_position: 65_888,
            runner_kv_bytes_per_position: 65_536,
            device_ceiling,
            residents: HybridResidents::CpuModelAndRunner,
        }
    }

    /// The budget reproduces the documented 27B figures on the 24 GiB M3:
    /// 83 968 / 93 184 positions with the CPU model beside the runner,
    /// 171 008 / 190 464 for a runner alone, and the CPU-alone guard of
    /// 178 176 / 197 632 that `oxibonsai info` has always printed.
    #[test]
    fn the_27b_budgets_match_the_documented_m3_figures() {
        for (file, ceiling, resident, alone, guard) in [
            (PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 83_968, 171_008, 178_176),
            (PTQ1_0_FILE, PTQ1_0_DEVICE_CEILING, 93_184, 190_464, 197_632),
        ] {
            let plan = plan_hybrid_metal_window(&inputs_27b(file, ceiling, 262_144));
            assert_eq!(plan.resident_budget, Some(resident));
            assert_eq!(plan.runner_alone_budget, Some(alone));
            assert_eq!(plan.ram_guard, Some(guard));
            assert_eq!(plan.device_ceiling, ceiling);
            // The declared context is not below the request; every other
            // limit is, and the resident budget binds.
            assert_eq!(plan.window, resident);
            assert_eq!(
                plan.limits_applied,
                vec![
                    HybridWindowLimit::RamGuard,
                    HybridWindowLimit::ResidentBudget,
                    HybridWindowLimit::DeviceCeiling,
                ]
            );
            assert!(plan.clamped());

            // Without the CPU model's buffers the runner-alone budget binds.
            let alone_plan = plan_hybrid_metal_window(&HybridWindowInputs {
                residents: HybridResidents::RunnerOnly,
                ..inputs_27b(file, ceiling, 262_144)
            });
            assert_eq!(alone_plan.window, alone.min(ceiling));
            assert_eq!(alone_plan.resident_budget, Some(alone));
        }
    }

    /// The shipped default window is under every limit: nothing clamps it and
    /// the runner's up-front allocation is its KV plus the fixed part.
    #[test]
    fn the_default_8192_window_is_unaffected() {
        for (file, ceiling) in [
            (PQ2_0_FILE, PQ2_0_DEVICE_CEILING),
            (PTQ1_0_FILE, PTQ1_0_DEVICE_CEILING),
        ] {
            let plan = plan_hybrid_metal_window(&inputs_27b(file, ceiling, 8192));
            assert_eq!(plan.window, 8192);
            assert!(plan.limits_applied.is_empty());
            assert!(!plan.clamped());
            assert_eq!(plan.runner_kv_bytes, 8192 * 65_536);
            assert_eq!(plan.runner_allocated_bytes, RUNNER_FIXED + 8192 * 65_888);
        }
    }

    /// A 32 768-position window costs 2 GiB of runner KV before the first
    /// token, and a window between the resident budget and the other
    /// limits is clamped by the resident budget alone.
    #[test]
    fn up_front_bytes_and_a_single_binding_limit_are_reported() {
        let plan = plan_hybrid_metal_window(&inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 32_768));
        assert_eq!(plan.window, 32_768);
        assert_eq!(plan.runner_kv_bytes, 2 << 30);
        let plan = plan_hybrid_metal_window(&inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 100_000));
        assert_eq!(plan.window, 83_968);
        assert_eq!(plan.limits_applied, vec![HybridWindowLimit::ResidentBudget]);
        let text = plan.summary();
        for needle in ["83968", "100000", "178034", "178176", "171008"] {
            assert!(text.contains(needle), "{needle} missing from {text}");
        }
    }

    /// With the host's RAM unknown only the declared context and the device
    /// ceiling apply; a budget with no room at all yields a zero window.
    #[test]
    fn unknown_ram_and_an_exhausted_budget() {
        let plan = plan_hybrid_metal_window(&HybridWindowInputs {
            total_ram_bytes: None,
            ..inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 500_000)
        });
        assert_eq!(plan.ram_guard, None);
        assert_eq!(plan.resident_budget, None);
        assert_eq!(plan.window, 262_144.min(PQ2_0_DEVICE_CEILING));
        assert_eq!(
            plan.limits_applied,
            vec![
                HybridWindowLimit::Declared,
                HybridWindowLimit::DeviceCeiling
            ]
        );
        assert!(plan.describe_limits().contains("unknown"));

        let starved = plan_hybrid_metal_window(&HybridWindowInputs {
            total_ram_bytes: Some(8 << 30),
            ..inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 8192)
        });
        assert_eq!(starved.window, 0);
    }

    #[test]
    fn backend_names_are_stable() {
        assert_eq!(HybridBackend::Cpu.as_str(), "cpu");
        assert_eq!(HybridBackend::Metal.to_string(), "metal");
        let cpu = HybridBackendPlan::Cpu { reason: "r".into() };
        assert_eq!(cpu.backend(), HybridBackend::Cpu);
    }
}
