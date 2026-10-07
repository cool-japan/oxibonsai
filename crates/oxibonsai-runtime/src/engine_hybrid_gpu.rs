//! The engine side of the Metal hybrid runner: which executor a `qwen35`
//! (PrismML Bonsai 2) engine decodes on, the KV window a Metal-backed engine
//! is wired with, and the glue the seam's forward, prefill, reset and
//! snapshot arms call.
//!
//! # Backend choice
//!
//! [`hybrid_backend_plan`] decides, for a hybrid model already bound on the
//! CPU, whether a `HybridMetalRunner` (`oxibonsai_model::hybrid::metal`, Metal builds)
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
//! alone and a device ceiling of 178 032 / 196 244; the shipped default of
//! 8192 is under all of them.
//!
//! The runner allocates its `f16` KV cache for the **whole** window at
//! construction (64 KiB per position for the 27B — a 32 768-position window
//! is 2 GiB of device memory before the first token), unlike the CPU
//! model's cache, which grows as positions are reached.
//! [`HybridMetalWindow::runner_allocated_bytes`] reports that up-front
//! figure.
//!
//! # What else the window leaves room for
//!
//! A vision tower loaded beside the engine (`--mmproj`) is one more
//! resident: [`HybridLoadOptions::vision_resident_bytes`] come off the RAM
//! budgets of the window as fixed bytes
//! ([`plan_hybrid_metal_window_with_vision`]; the Metal tower of the Bonsai
//! 2 projector keeps its `Q8_0` / `F16` weights as stored, its scratch and
//! a host position grid resident, 0.87 GiB). The runner's activation
//! scratch grows with the most tokens one call takes, so the prefill chunk
//! ([`HybridLoadOptions::prefill_chunk`], `--prefill-chunk`) is honoured
//! only as far as the window the default chunk would get stays in place:
//! a larger chunk is granted up to the largest call size that keeps it,
//! with one warning naming both numbers. The same bound applies when the
//! chunk is changed after the engine is built (the runner's batch follows
//! the model's prefill chunk at the next prefill).
//!
//! The options reach every constructor through a [`HybridLoadScope`] on
//! the constructing thread; outside one they are the defaults (no vision
//! tower, the model's own 512-token chunk) and every figure above is the
//! one a text-only engine always had.

use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;

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
// What a hybrid engine is sized for besides its KV window
// ─────────────────────────────────────────────────────────────────────────

/// What a caller knows before a hybrid engine is built that sizes it
/// besides the KV window it asks for (see the module docs). The default —
/// no vision tower, the model's own prefill chunk — builds exactly the
/// engine every constructor always built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HybridLoadOptions {
    /// Bytes a vision tower keeps resident beside the engine — for a
    /// Metal-backed engine the Metal tower's
    /// `oxibonsai_model::vision::metal::VisionTowerMetal::footprint`
    /// (`crate::vision_prefill::VisionService::metal_footprint` reads it
    /// from a projector file); `0` without one. A Metal-backed engine's KV
    /// window leaves room for them.
    pub vision_resident_bytes: u64,
    /// The prefill chunk (`--prefill-chunk`); `None` keeps the model's
    /// default (512 tokens). The CPU model takes it, and a Metal runner is
    /// built to take calls of that many tokens as far as the KV window's
    /// memory budget allows.
    pub prefill_chunk: Option<usize>,
}

/// [`HybridLoadOptions::default`] as a constant (the thread-local's initial
/// value).
const NO_LOAD_OPTIONS: HybridLoadOptions = HybridLoadOptions {
    vision_resident_bytes: 0,
    prefill_chunk: None,
};

thread_local! {
    /// The [`HybridLoadOptions`] in force on this thread (see
    /// [`HybridLoadScope`]).
    static HYBRID_LOAD_OPTIONS: Cell<HybridLoadOptions> = const { Cell::new(NO_LOAD_OPTIONS) };
}

/// While it lives, every hybrid engine constructed **on this thread** is
/// built with its [`HybridLoadOptions`] — how the options reach every
/// existing constructor (one engine, or every replica of a pool, which the
/// pool builders construct on the calling thread) without a new signature,
/// the way `oxibonsai_core::config::RopeScalingOverrideScope` reaches the
/// model constructors. Scopes nest: dropping one restores the options that
/// were in force when it was entered. A dense engine ignores them.
#[derive(Debug)]
pub struct HybridLoadScope {
    previous: HybridLoadOptions,
    _not_send: PhantomData<*const ()>,
}

impl HybridLoadScope {
    /// Install `options` for this thread until the returned guard drops.
    #[must_use]
    pub fn enter(options: HybridLoadOptions) -> Self {
        let previous = HYBRID_LOAD_OPTIONS.with(|cell| cell.replace(options));
        Self {
            previous,
            _not_send: PhantomData,
        }
    }

    /// The options in force on this thread (the defaults outside any
    /// scope).
    #[must_use]
    pub fn active() -> HybridLoadOptions {
        HYBRID_LOAD_OPTIONS.with(Cell::get)
    }
}

impl Drop for HybridLoadScope {
    fn drop(&mut self) {
        let previous = self.previous;
        HYBRID_LOAD_OPTIONS.with(|cell| cell.set(previous));
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
    /// Bytes a vision tower keeps resident beside the engine, taken off the
    /// resident and runner-alone budgets (`0`: no vision tower).
    pub vision_resident_bytes: u64,
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
        let vision = if self.vision_resident_bytes == 0 {
            String::new()
        } else {
            format!(
                ", plus a vision tower keeping {} resident",
                gib(self.vision_resident_bytes)
            )
        };
        format!(
            "{} {}, {} {}, {} {} ({}{}), {} {}; a runner alone would fit {}",
            HybridWindowLimit::Declared.as_str(),
            self.declared,
            HybridWindowLimit::RamGuard.as_str(),
            show(self.ram_guard),
            HybridWindowLimit::ResidentBudget.as_str(),
            show(self.resident_budget),
            self.residents.describe(),
            vision,
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
    plan_hybrid_metal_window_with_vision(inputs, 0)
}

/// [`plan_hybrid_metal_window`] for an engine with a vision tower keeping
/// `vision_resident_bytes` resident beside it: the tower's bytes come off
/// the budget for the residents kept and off the runner-alone budget as
/// fixed bytes (they do not grow with the window). The RAM guard is the CPU
/// model's own figure and does not change. With `0` this is exactly
/// [`plan_hybrid_metal_window`].
#[must_use]
pub fn plan_hybrid_metal_window_with_vision(
    inputs: &HybridWindowInputs,
    vision_resident_bytes: u64,
) -> HybridMetalWindow {
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
    let runner_fixed = inputs
        .runner_fixed_bytes
        .saturating_add(vision_resident_bytes);
    let runner_alone_budget = inputs.total_ram_bytes.map(|total| {
        ram_budget(
            total,
            inputs.file_bytes,
            runner_fixed,
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
                runner_fixed.saturating_add(inputs.cpu_recurrent_bytes),
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
        vision_resident_bytes,
    }
}

/// The largest call size in `floor..=wanted` that keeps a KV window of at
/// least `window` positions, where `window_at(batch)` is the window the
/// memory budget allows a runner whose activation scratch is sized for
/// `batch`-token calls — non-increasing in `batch`, since every token of a
/// call costs scratch. `floor` is a call size known to keep the window (the
/// runner's current one); a `wanted` at or below it is returned as it is
/// (a smaller call only frees scratch). This is how a Metal-backed engine
/// sizes its runner's calls for a prefill chunk (see the module docs).
#[must_use]
pub fn largest_batch_keeping_window(
    floor: usize,
    wanted: usize,
    window: usize,
    window_at: impl Fn(usize) -> usize,
) -> usize {
    if wanted <= floor || window_at(wanted) >= window {
        return wanted;
    }
    // `lo` keeps the window, `hi` does not.
    let (mut lo, mut hi) = (floor, wanted);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if window_at(mid) >= window {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─────────────────────────────────────────────────────────────────────────
// The Metal sessions a hybrid engine and a vision tower hold
// ─────────────────────────────────────────────────────────────────────────

/// The Metal sessions of this process against the ceiling engine pools are
/// sized by — what `oxibonsai info` reports. A Metal-backed hybrid engine
/// opens one session for its runner and a Metal vision tower one more
/// ([`MetalSessionReport::sessions_for`]); every engine-pool replica adds a
/// runner session of its own, while the tower is shared. The ceiling
/// (`MetalGraph::max_sessions`, `OXIBONSAI_METAL_MAX_SESSIONS`) bounds pool
/// sizes but is not enforced when a session opens, so a process can hold
/// more live sessions than it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalSessionReport {
    /// Sessions alive on the process-shared device now
    /// (`MetalGraph::live_session_count`).
    pub live: usize,
    /// The pool ceiling (`MetalGraph::max_sessions`).
    pub max: usize,
}

impl MetalSessionReport {
    /// This process's figures; `None` on a build without the Metal
    /// backend, where no session exists.
    #[must_use]
    pub fn current() -> Option<Self> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            Some(Self {
                live: oxibonsai_kernels::MetalGraph::live_session_count(),
                max: oxibonsai_kernels::MetalGraph::max_sessions(),
            })
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            None
        }
    }

    /// Sessions one Metal-backed hybrid engine opens: its runner's, plus
    /// the Metal vision tower's when it serves images.
    #[must_use]
    pub const fn sessions_for(with_vision_tower: bool) -> usize {
        if with_vision_tower {
            2
        } else {
            1
        }
    }

    /// The one-line report: the live count, what an engine (and a vision
    /// tower) adds, and the ceiling with whether that engine stays under it.
    #[must_use]
    pub fn describe(&self, with_vision_tower: bool) -> String {
        let needed = Self::sessions_for(with_vision_tower);
        let opens = if with_vision_tower {
            "2 (its runner's and the Metal vision tower's)".to_string()
        } else {
            "1 (its runner's)".to_string()
        };
        let fits = if self.live.saturating_add(needed) > self.max {
            "past it"
        } else {
            "within it"
        };
        format!(
            "{} live in this process; a Metal-backed engine opens {opens}, {fits}: the ceiling \
             of {} (OXIBONSAI_METAL_MAX_SESSIONS) sizes engine pools and is not enforced when a \
             session opens",
            self.live, self.max
        )
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
        /// The most tokens one runner prefill call takes: the prefill chunk
        /// of the [`HybridLoadOptions`] (the model's own 512 without one),
        /// or the largest call size that keeps `window` when the window's
        /// memory budget cannot hold calls that large (see the module docs).
        call_tokens: usize,
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
/// `oxibonsai info` reports, under the [`HybridLoadOptions`] in force on
/// this thread ([`HybridLoadScope`]).
#[must_use]
pub fn hybrid_backend_plan(
    gguf: &GgufFile<'_>,
    model: &HybridModel<'_>,
    requested: usize,
) -> HybridBackendPlan {
    hybrid_backend_plan_with(gguf, model, requested, &HybridLoadScope::active())
}

/// [`hybrid_backend_plan`] under explicit `options`: the window leaves room
/// for the vision tower's bytes, and the runner's activation scratch is
/// counted at the call size it will be built with for the requested
/// prefill chunk (see the module docs).
#[must_use]
pub fn hybrid_backend_plan_with(
    gguf: &GgufFile<'_>,
    model: &HybridModel<'_>,
    requested: usize,
    options: &HybridLoadOptions,
) -> HybridBackendPlan {
    #[cfg(test)]
    if let Some(reason) = FORCED_CPU_PLAN.with(|forced| forced.borrow().clone()) {
        return HybridBackendPlan::Cpu { reason };
    }
    metal::plan(gguf, model, requested, options)
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
    use oxibonsai_model::hybrid::metal::{
        HybridGpuSnapshot, HybridMetalFootprint, HybridMetalRunner,
    };
    use oxibonsai_model::hybrid::HybridModel;
    use oxibonsai_model::layers::rope_mrope::MropePos;

    use super::{
        largest_batch_keeping_window, plan_hybrid_metal_window_with_vision, HybridBackendPlan,
        HybridLoadOptions, HybridMetalWindow, HybridResidents, HybridWindowInputs,
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

    /// Everything the window and call-size decisions read about one model
    /// on this host: the runner's footprint at the model's own prefill
    /// chunk (the call size a runner is built with by default), the
    /// device's limits, and the window inputs at that call size.
    #[derive(Debug, Clone)]
    pub(crate) struct RunnerSizing {
        footprint: HybridMetalFootprint,
        limits: Qwen35DeviceLimits,
        mapped: bool,
        /// The window inputs for calls of [`Self::base_batch`] tokens.
        base: HybridWindowInputs,
        base_batch: usize,
        vision_resident_bytes: u64,
    }

    impl RunnerSizing {
        /// Read the sizing of a runner for `model` (bound from `gguf`) with
        /// a window of `requested` positions and a vision tower of
        /// `vision_resident_bytes` beside it.
        ///
        /// # Errors
        ///
        /// Why no runner serves the model here: no Metal device, one that
        /// cannot be opened, or a model the runner does not serve.
        pub(crate) fn of(
            gguf: &GgufFile<'_>,
            model: &HybridModel<'_>,
            requested: usize,
            vision_resident_bytes: u64,
        ) -> Result<Self, String> {
            let limits = match Qwen35DeviceLimits::of_shared_device() {
                Ok(limits) => limits,
                Err(MetalGraphError::DeviceNotFound) => {
                    return Err("no Metal device was found on this host".to_string())
                }
                Err(e) => return Err(format!("the Metal device could not be opened: {e}")),
            };
            let footprint = HybridMetalRunner::footprint(model)
                .map_err(|e| format!("the Metal hybrid runner does not serve this model: {e}"))?;
            let mapped = Qwen35MappedRegion::page_aligned(gguf.data).is_some();
            let (cpu_kv, cpu_rope) = cpu_bytes_per_position(model);
            let base = HybridWindowInputs {
                requested,
                declared: model.config().base.max_context_length,
                total_ram_bytes: crate::config::total_ram_bytes(),
                file_bytes: gguf.data.len() as u64,
                cpu_recurrent_bytes: model.recurrent().memory_bytes() as u64,
                cpu_kv_bytes_per_position: cpu_kv,
                cpu_other_bytes_per_position: cpu_rope,
                runner_fixed_bytes: fixed_bytes(&footprint, mapped),
                runner_bytes_per_position: footprint.device.per_position_bytes,
                runner_kv_bytes_per_position: footprint.device.kv_bytes_per_position,
                device_ceiling: footprint.device_ceiling(&limits),
                residents: HybridResidents::CpuModelAndRunner,
            };
            Ok(Self {
                base_batch: footprint.config().max_batch,
                footprint,
                limits,
                mapped,
                base,
                vision_resident_bytes,
            })
        }

        /// Whether the runner reads the weights in place from the mapping.
        pub(crate) fn mapped(&self) -> bool {
            self.mapped
        }

        /// The window inputs for a runner taking calls of `batch` tokens:
        /// its fixed bytes (activation scratch and host staging) and its
        /// device ceiling follow the call size.
        fn inputs_at(&self, batch: usize) -> HybridWindowInputs {
            if batch == self.base_batch {
                return self.base;
            }
            let footprint = self.footprint.with_max_batch(batch);
            HybridWindowInputs {
                runner_fixed_bytes: fixed_bytes(&footprint, self.mapped),
                device_ceiling: footprint.device_ceiling(&self.limits),
                ..self.base
            }
        }

        /// The window plan for a runner taking calls of `batch` tokens.
        pub(crate) fn plan_at(&self, batch: usize) -> HybridMetalWindow {
            plan_hybrid_metal_window_with_vision(&self.inputs_at(batch), self.vision_resident_bytes)
        }

        /// The call size a runner whose calls take `floor` tokens today (and
        /// keep `window`) grows to for a `wanted`-token prefill chunk.
        pub(crate) fn batch_from(&self, floor: usize, wanted: usize, window: usize) -> usize {
            largest_batch_keeping_window(floor, wanted.max(1), window, |batch| {
                self.plan_at(batch).window
            })
        }

        /// The call size a runner built for a `window`-position KV window
        /// takes for `chunk` (`None`: the model's own chunk).
        pub(crate) fn batch_for(&self, chunk: Option<usize>, window: usize) -> usize {
            self.batch_from(self.base_batch, chunk.unwrap_or(self.base_batch), window)
        }

        /// The plan for a runner built for `chunk`: the window a runner at
        /// the model's own call size gets, with the call size grown towards
        /// `chunk` only as far as that window stays (a smaller chunk only
        /// frees scratch, which the plan then counts).
        pub(crate) fn plan_for(&self, chunk: Option<usize>) -> (HybridMetalWindow, usize) {
            let base = self.plan_at(self.base_batch);
            let batch = self.batch_for(chunk, base.window);
            if batch == self.base_batch {
                (base, batch)
            } else {
                (self.plan_at(batch), batch)
            }
        }
    }

    /// The runner's bytes independent of the window for `footprint`'s call
    /// size: the copied weights, the recurrent state, the logits, the
    /// activation scratch, and the host staging of one call's embedded rows.
    fn fixed_bytes(footprint: &HybridMetalFootprint, mapped: bool) -> u64 {
        let config = footprint.config();
        let staging = (config.max_batch * config.hidden) as u64 * F32_BYTES;
        footprint.allocated_bytes(0, mapped).saturating_add(staging)
    }

    pub(super) fn plan(
        gguf: &GgufFile<'_>,
        model: &HybridModel<'_>,
        requested: usize,
        options: &HybridLoadOptions,
    ) -> HybridBackendPlan {
        let sizing = match RunnerSizing::of(gguf, model, requested, options.vision_resident_bytes) {
            Ok(sizing) => sizing,
            Err(reason) => return HybridBackendPlan::Cpu { reason },
        };
        let (window, call_tokens) = sizing.plan_for(options.prefill_chunk);
        if window.window == 0 {
            return HybridBackendPlan::Cpu {
                reason: format!(
                    "the Metal hybrid runner has no room for a KV window on this host: {}",
                    window.describe_limits()
                ),
            };
        }
        HybridBackendPlan::Metal {
            window,
            mapped: sizing.mapped(),
            call_tokens,
        }
    }

    /// A Metal-backed hybrid engine's runner, the window it was wired with
    /// and what sizes its calls.
    pub(crate) struct HybridGpu<'a> {
        runner: HybridMetalRunner<'a>,
        window: HybridMetalWindow,
        sizing: RunnerSizing,
        /// The last prefill chunk the call size could not grow to, and the
        /// call size granted for it (warned about once).
        capped: Option<(usize, usize)>,
    }

    impl std::fmt::Debug for HybridGpu<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HybridGpu")
                .field("runner", &self.runner)
                .field("window", &self.window.window)
                .field("capped", &self.capped)
                .finish()
        }
    }

    /// The seam's snapshot of a Metal-backed hybrid sequence.
    pub(crate) type HybridGpuState = HybridGpuSnapshot;

    /// What the runner's next prefill does with its call size for a prefill
    /// chunk (see `HybridGpu::fit_batch`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum BatchPlan {
        /// Keep the current call size.
        Keep(usize),
        /// Fit the calls to the chunk: from `current` to `batch` tokens.
        Fit { current: usize, batch: usize },
    }

    impl<'a> HybridGpu<'a> {
        /// Build the runner for `model` (bound at `window.window`) over the
        /// GGUF image it was loaded from, taking calls of the prefill chunk
        /// in `options` as far as the window's memory budget allows (one
        /// warning names both numbers when it cannot take all of it).
        ///
        /// # Errors
        ///
        /// The reason the runner could not be built, for the typed refusal
        /// or the CPU fallback's log line.
        pub(crate) fn build(
            gguf: &'a GgufFile<'a>,
            model: &HybridModel<'a>,
            window: HybridMetalWindow,
            options: &HybridLoadOptions,
        ) -> Result<Self, String> {
            let sizing =
                RunnerSizing::of(gguf, model, window.requested, options.vision_resident_bytes)?;
            let batch = sizing.batch_for(options.prefill_chunk, window.window);
            let runner = HybridMetalRunner::new_in_place_with_batch(model, gguf.data, batch)
                .map_err(|e| format!("the Metal hybrid runner could not be built: {e}"))?;
            if runner.max_seq_len() != window.window {
                return Err(format!(
                    "the Metal hybrid runner was built with a {}-position window, not the planned \
                     {}",
                    runner.max_seq_len(),
                    window.window
                ));
            }
            let mut gpu = Self {
                runner,
                window,
                sizing,
                capped: None,
            };
            if let Some(chunk) = options.prefill_chunk {
                if chunk > batch {
                    gpu.note_capped(chunk, batch);
                }
            }
            Ok(gpu)
        }

        /// Record (and warn once about) a prefill chunk the runner's calls
        /// cannot grow to.
        fn note_capped(&mut self, chunk: usize, batch: usize) {
            tracing::warn!(
                prefill_chunk = chunk,
                call_tokens = batch,
                window = self.window.window,
                "the Metal hybrid runner takes prefill calls of at most {batch} tokens, not the \
                 requested prefill chunk of {chunk}: larger calls would not leave room for its \
                 {}-position KV window ({})",
                self.window.window,
                self.window.describe_limits()
            );
            self.capped = Some((chunk, batch));
        }

        /// What [`Self::fit_batch`] does with a `chunk`-token prefill chunk,
        /// decided without changing anything: keep the current call size
        /// (the chunk already is the call size, or was already capped to
        /// it), or fit the calls to the chunk within the window's memory
        /// budget.
        fn batch_plan(&self, chunk: usize) -> BatchPlan {
            let chunk = chunk.max(1);
            let current = self.runner.max_batch();
            if chunk == current {
                return BatchPlan::Keep(current);
            }
            if let Some((capped_chunk, granted)) = self.capped {
                if capped_chunk == chunk && granted == current {
                    return BatchPlan::Keep(current);
                }
            }
            BatchPlan::Fit {
                current,
                batch: self.sizing.batch_from(current, chunk, self.window.window),
            }
        }

        /// The most tokens one call of the runner's next prefill takes for
        /// a `chunk`-token prefill chunk: `chunk` itself, or the largest
        /// call size the window's memory budget allows. Reads the decision
        /// [`Self::fit_batch`] applies, without applying it.
        pub(crate) fn planned_batch(&self, chunk: usize) -> usize {
            match self.batch_plan(chunk) {
                BatchPlan::Keep(current) => current,
                BatchPlan::Fit { batch, .. } => batch,
            }
        }

        /// Size the runner's calls for a `chunk`-token prefill chunk and
        /// return the call size: `chunk` itself when the runner's calls can
        /// take it within the window's memory budget, else the largest call
        /// size that budget allows (warned about once per chunk). A smaller
        /// chunk shrinks the calls, and their scratch, to it.
        fn fit_batch(&mut self, chunk: usize) -> usize {
            let chunk = chunk.max(1);
            let (current, batch) = match self.batch_plan(chunk) {
                BatchPlan::Keep(current) => return current,
                BatchPlan::Fit { current, batch } => (current, batch),
            };
            if batch != current {
                if let Err(e) = self.runner.set_max_batch(batch) {
                    tracing::warn!(
                        prefill_chunk = chunk,
                        call_tokens = current,
                        "the Metal hybrid runner keeps prefill calls of {current} tokens: {e}"
                    );
                    self.capped = Some((chunk, current));
                    return current;
                }
            }
            if batch < chunk {
                self.note_capped(chunk, batch);
            }
            batch
        }

        pub(crate) fn window(&self) -> &HybridMetalWindow {
            &self.window
        }

        pub(crate) fn token_count(&self) -> usize {
            self.runner.token_count()
        }

        /// The most tokens one runner call takes now: the call size the
        /// runner was built with until a prefill fits it to the model's
        /// prefill chunk (`fit_batch`).
        pub(crate) fn max_batch(&self) -> usize {
            self.runner.max_batch()
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

        /// The M-RoPE offset in force at sequence position `pos`.
        pub(crate) fn rope_offset_at(&self, pos: usize) -> usize {
            self.runner.rope_offset_at(pos)
        }

        /// The text rotary position a prompt starting at sequence position
        /// `start_pos` begins at.
        pub(crate) fn rope_start_for(&self, start_pos: usize) -> RuntimeResult<usize> {
            Ok(self.runner.rope_start_for(start_pos)?)
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

        /// Prefill `tokens` from `pos_start` in calls of `chunk` tokens (as
        /// far as the window's memory budget allows, see `fit_batch`),
        /// returning the last token's logits.
        pub(crate) fn prefill_logits(
            &mut self,
            tokens: &[u32],
            pos_start: usize,
            chunk: usize,
        ) -> RuntimeResult<Vec<f32>> {
            self.fit_batch(chunk);
            let mut logits = vec![0.0f32; self.runner.vocab_size()];
            self.runner
                .forward_prefill(tokens, pos_start, &mut logits)?;
            Ok(logits)
        }

        /// Prefill caller-built rows (a multimodal prompt: text rows
        /// embedded, image rows verbatim) at their 3-axis rotary
        /// `positions` from sequence position `pos_start`, in calls of
        /// `chunk` rows as [`Self::prefill_logits`] does, returning the last
        /// row's logits.
        pub(crate) fn prefill_rows(
            &mut self,
            rows: &[f32],
            positions: &[MropePos],
            pos_start: usize,
            chunk: usize,
        ) -> RuntimeResult<Vec<f32>> {
            self.fit_batch(chunk);
            let mut logits = vec![0.0f32; self.runner.vocab_size()];
            self.runner
                .forward_prefill_rows(rows, positions, pos_start, Some(&mut logits))?;
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
    use oxibonsai_model::layers::rope_mrope::MropePos;

    use super::{HybridBackendPlan, HybridLoadOptions, HybridMetalWindow};
    use crate::error::RuntimeResult;

    pub(super) fn plan(
        _gguf: &GgufFile<'_>,
        _model: &HybridModel<'_>,
        _requested: usize,
        _options: &HybridLoadOptions,
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
            _options: &HybridLoadOptions,
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

        pub(crate) fn max_batch(&self) -> usize {
            match self.never {}
        }

        pub(crate) fn planned_batch(&self, _chunk: usize) -> usize {
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

        pub(crate) fn rope_offset_at(&self, _pos: usize) -> usize {
            match self.never {}
        }

        pub(crate) fn rope_start_for(&self, _start_pos: usize) -> RuntimeResult<usize> {
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

        pub(crate) fn prefill_rows(
            &mut self,
            _rows: &[f32],
            _positions: &[MropePos],
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
    const PQ2_0_DEVICE_CEILING: usize = 178_032;
    const PTQ1_0_DEVICE_CEILING: usize = 196_244;
    /// Recurrent state of one 27B sequence (either executor).
    const RECURRENT: u64 = 156_893_184;
    /// The runner's window-independent bytes for the 27B: the widened
    /// `ssm_alpha`/`ssm_beta` rows, its recurrent state, the logits row, the
    /// activation scratch at a 512-token chunk and the host staging rows.
    const RUNNER_FIXED: u64 =
        48 * 2 * 48 * 5120 * 4 + RECURRENT + 248_320 * 4 + 140_448 * 4 * 512 + 512 * 5120 * 4;

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
        for needle in ["83968", "100000", "178032", "178176", "171008"] {
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

    /// A vision tower's resident bytes come off the resident and
    /// runner-alone budgets as fixed bytes — exactly as if the runner's own
    /// fixed bytes were that much larger — while the CPU-alone RAM guard is
    /// unchanged; with no tower the plan is the plain one, figure for
    /// figure, and its text does not mention one.
    #[test]
    fn the_window_plan_leaves_room_for_a_vision_tower() {
        const TOWER: u64 = 1 << 30;
        for (file, ceiling) in [
            (PQ2_0_FILE, PQ2_0_DEVICE_CEILING),
            (PTQ1_0_FILE, PTQ1_0_DEVICE_CEILING),
        ] {
            let inputs = inputs_27b(file, ceiling, 262_144);
            let plain = plan_hybrid_metal_window(&inputs);
            assert_eq!(plan_hybrid_metal_window_with_vision(&inputs, 0), plain);
            assert_eq!(plain.vision_resident_bytes, 0);
            assert!(!plain.describe_limits().contains("vision"));

            let with_tower = plan_hybrid_metal_window_with_vision(&inputs, TOWER);
            let as_fixed = plan_hybrid_metal_window(&HybridWindowInputs {
                runner_fixed_bytes: RUNNER_FIXED + TOWER,
                ..inputs
            });
            assert_eq!(with_tower.resident_budget, as_fixed.resident_budget);
            assert_eq!(with_tower.runner_alone_budget, as_fixed.runner_alone_budget);
            assert_eq!(with_tower.ram_guard, plain.ram_guard);
            assert_eq!(with_tower.vision_resident_bytes, TOWER);
            // 1 GiB is about 8 000 positions of the two residents' KV.
            let shrink =
                plain.resident_budget.unwrap_or(0) - with_tower.resident_budget.unwrap_or(0);
            assert!((7 * 1024..=9 * 1024).contains(&shrink), "{shrink}");
            assert_eq!(with_tower.window, with_tower.resident_budget.unwrap_or(0));
            assert!(with_tower.window < plain.window);
            // The runner's own allocation does not include the tower.
            assert_eq!(
                with_tower.runner_allocated_bytes,
                RUNNER_FIXED + with_tower.window as u64 * 65_888
            );
            let text = with_tower.describe_limits();
            assert!(
                text.contains("plus a vision tower keeping 1.00 GiB resident"),
                "{text}"
            );

            // The shipped default window is still under every limit.
            let default =
                plan_hybrid_metal_window_with_vision(&inputs_27b(file, ceiling, 8192), TOWER);
            assert_eq!(default.window, 8192);
            assert!(default.limits_applied.is_empty());
        }
    }

    /// The call-size search: the largest batch whose window stays, never
    /// below the floor, and a smaller request taken as it is.
    #[test]
    fn the_largest_batch_keeps_the_window() {
        // A budget of 10 000 positions minus 3 per batch token.
        let window_at = |batch: usize| 10_000usize.saturating_sub(3 * batch);
        assert_eq!(
            largest_batch_keeping_window(512, 2048, 8192, window_at),
            602
        );
        assert!(window_at(602) >= 8192 && window_at(603) < 8192);
        assert_eq!(largest_batch_keeping_window(512, 600, 8192, window_at), 600);
        assert_eq!(largest_batch_keeping_window(512, 64, 8192, window_at), 64);
        assert_eq!(largest_batch_keeping_window(512, 512, 8192, window_at), 512);
        // Even the floor does not keep it: the floor stays.
        assert_eq!(
            largest_batch_keeping_window(512, 4096, 9_999, window_at),
            512
        );
        // Nothing binds: the whole request.
        assert_eq!(largest_batch_keeping_window(512, 4096, 0, window_at), 4096);

        // The 27B at its resident budget: a larger call's scratch costs
        // positions, so the call grows only as far as the budget allows.
        let inputs = inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 83_968);
        let per_call_token = 140_448 * 4 + 5120 * 4;
        let at = |batch: usize| {
            plan_hybrid_metal_window(&HybridWindowInputs {
                runner_fixed_bytes: RUNNER_FIXED + (batch as u64 - 512) * per_call_token,
                ..inputs
            })
            .window
        };
        assert_eq!(at(512), 83_968);
        let batch = largest_batch_keeping_window(512, 8192, 83_968, at);
        assert!((512..8192).contains(&batch), "{batch}");
        assert!(at(batch) >= 83_968 && at(batch + 1) < 83_968, "{batch}");
        // At the shipped 8192-position window every chunk up to 8192 fits.
        let small = inputs_27b(PQ2_0_FILE, PQ2_0_DEVICE_CEILING, 8192);
        let at_small = |batch: usize| {
            plan_hybrid_metal_window(&HybridWindowInputs {
                runner_fixed_bytes: RUNNER_FIXED + (batch as u64 - 512) * per_call_token,
                ..small
            })
            .window
        };
        assert_eq!(
            largest_batch_keeping_window(512, 8192, 8192, at_small),
            8192
        );
    }

    /// Scopes install their options on this thread only, nest, and restore
    /// the options that were in force when they were entered.
    #[test]
    fn load_scopes_nest_and_restore() {
        assert_eq!(HybridLoadScope::active(), HybridLoadOptions::default());
        let outer = HybridLoadOptions {
            vision_resident_bytes: 7,
            prefill_chunk: Some(64),
        };
        {
            let _outer = HybridLoadScope::enter(outer);
            assert_eq!(HybridLoadScope::active(), outer);
            let inner = HybridLoadOptions {
                prefill_chunk: None,
                ..outer
            };
            {
                let _inner = HybridLoadScope::enter(inner);
                assert_eq!(HybridLoadScope::active(), inner);
                let other = std::thread::spawn(HybridLoadScope::active)
                    .join()
                    .expect("thread");
                assert_eq!(other, HybridLoadOptions::default());
            }
            assert_eq!(HybridLoadScope::active(), outer);
        }
        assert_eq!(HybridLoadScope::active(), HybridLoadOptions::default());
    }

    /// Whether this build and host give a hybrid engine the Metal runner.
    fn metal_available() -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match oxibonsai_kernels::MetalGraph::shared_device() {
                Ok(_) => true,
                Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
                Err(e) => panic!("the Metal device must open on this host: {e}"),
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            false
        }
    }

    fn greedy() -> crate::sampling::SamplingParams {
        crate::sampling::SamplingParams {
            temperature: 0.0,
            ..crate::sampling::SamplingParams::default()
        }
    }

    /// A Metal-backed engine over the synthetic hybrid built under `options`.
    fn metal_engine<'a>(
        gguf: &'a GgufFile<'a>,
        options: HybridLoadOptions,
    ) -> crate::engine::InferenceEngine<'a> {
        let _scope = HybridLoadScope::enter(options);
        let engine = crate::engine::InferenceEngine::from_gguf_with_backend(
            gguf,
            greedy(),
            7,
            64,
            crate::engine_seam::Backend::Metal,
        )
        .expect("a Metal hybrid engine");
        assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
        engine
    }

    /// The scoped options reach the engine: the window counts the vision
    /// tower, the CPU model takes the prefill chunk and the runner's calls
    /// are sized for it (smaller or larger than the default), and a text
    /// prompt decodes bit for bit as on an engine built without them.
    #[test]
    fn scoped_options_size_a_metal_engine_and_leave_text_unchanged() {
        if !metal_available() {
            return;
        }
        let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let prompt = [7u32, 11, 13, 17, 19];
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        let mut plain = metal_engine(&gguf, HybridLoadOptions::default());
        let plain_window = plain.hybrid_metal_window().cloned().expect("window");
        assert_eq!(plain_window.vision_resident_bytes, 0);
        let default_chunk = plain.hybrid_model().map(|m| m.prefill_chunk());
        assert_eq!(
            plain.hybrid_gpu.as_ref().map(HybridGpu::max_batch),
            default_chunk
        );
        let want = plain.prefill_from_pos(&prompt, 0).expect("prefill");
        let want_ids = plain.generate(&prompt, 6).expect("generate");

        for (chunk, vision) in [(4usize, 3u64 << 20), (1024, 0)] {
            let mut engine = metal_engine(
                &gguf,
                HybridLoadOptions {
                    vision_resident_bytes: vision,
                    prefill_chunk: Some(chunk),
                },
            );
            let window = engine.hybrid_metal_window().cloned().expect("window");
            assert_eq!(window.vision_resident_bytes, vision);
            assert_eq!(window.window, plain_window.window);
            assert_eq!(
                engine.hybrid_model().map(|m| m.prefill_chunk()),
                Some(chunk)
            );
            assert_eq!(
                engine.hybrid_gpu.as_ref().map(HybridGpu::max_batch),
                Some(chunk)
            );
            let got = engine.prefill_from_pos(&prompt, 0).expect("prefill");
            assert_eq!(bits(&got), bits(&want), "chunk {chunk}");
            assert_eq!(engine.generate(&prompt, 6).expect("generate"), want_ids);
        }

        // A chunk set after construction resizes the runner's calls at the
        // next prefill.
        plain
            .hybrid_model_mut()
            .expect("hybrid")
            .set_prefill_chunk(2048)
            .expect("chunk");
        let again = plain.prefill_from_pos(&prompt, 0).expect("prefill");
        assert_eq!(bits(&again), bits(&want));
        assert_eq!(
            plain.hybrid_gpu.as_ref().map(HybridGpu::max_batch),
            Some(2048)
        );
    }

    /// `hybrid_metal_call_tokens` is the call size the runner holds, not the
    /// plan for the model's chunk: the one it was built with, which a chunk
    /// set on the model afterwards changes only once a prefill fits the
    /// calls to it. An engine without a runner has none.
    #[test]
    fn the_runner_call_size_is_the_built_one_until_a_prefill_fits_it() {
        let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");

        let cpu = crate::engine::InferenceEngine::from_gguf_with_backend(
            &gguf,
            greedy(),
            7,
            64,
            crate::engine_seam::Backend::Cpu,
        )
        .expect("a CPU hybrid engine");
        assert_eq!(cpu.hybrid_backend(), Some(HybridBackend::Cpu));
        assert_eq!(cpu.hybrid_metal_call_tokens(), None);

        if !metal_available() {
            return;
        }
        let default_chunk = oxibonsai_model::hybrid::DEFAULT_PREFILL_CHUNK;
        let prompt = [7u32, 11, 13, 17, 19];

        // Built for a chunk: the runner holds it, and it is the call size the
        // engine reports in effect.
        assert_ne!(8, default_chunk, "the chunk differs from the default");
        let for_chunk = metal_engine(
            &gguf,
            HybridLoadOptions {
                vision_resident_bytes: 0,
                prefill_chunk: Some(8),
            },
        );
        assert_eq!(for_chunk.hybrid_metal_call_tokens(), Some(8));
        assert_eq!(for_chunk.prefill_chunk_in_effect(), 8);

        // Built without one: the model's own chunk. A chunk set on the model
        // afterwards moves the plan, not the runner ...
        let mut engine = metal_engine(&gguf, HybridLoadOptions::default());
        assert_eq!(engine.hybrid_metal_call_tokens(), Some(default_chunk));
        engine
            .hybrid_model_mut()
            .expect("hybrid")
            .set_prefill_chunk(2048)
            .expect("chunk");
        assert_eq!(engine.prefill_chunk_in_effect(), 2048);
        assert_eq!(engine.hybrid_metal_call_tokens(), Some(default_chunk));

        // ... until a prefill fits the runner's calls to it.
        engine.prefill_from_pos(&prompt, 0).expect("prefill");
        assert_eq!(engine.hybrid_metal_call_tokens(), Some(2048));
    }

    /// The session report: what one engine (and its vision tower) opens,
    /// whether that stays under the ceiling, and the fact that the ceiling
    /// is not enforced; on a Metal build the live figures are the
    /// process's own.
    #[test]
    fn the_metal_session_report_counts_the_runner_and_the_tower() {
        assert_eq!(MetalSessionReport::sessions_for(false), 1);
        assert_eq!(MetalSessionReport::sessions_for(true), 2);
        let idle = MetalSessionReport { live: 0, max: 4 };
        let text = idle.describe(true);
        assert!(text.starts_with("0 live in this process"), "{text}");
        assert!(
            text.contains("opens 2 (its runner's and the Metal vision tower's)"),
            "{text}"
        );
        assert!(text.contains("within it"), "{text}");
        assert!(
            text.contains("ceiling of 4 (OXIBONSAI_METAL_MAX_SESSIONS)"),
            "{text}"
        );
        assert!(text.contains("not enforced when a session opens"), "{text}");
        assert!(idle.describe(false).contains("opens 1 (its runner's)"));
        let full = MetalSessionReport { live: 3, max: 4 };
        assert!(full.describe(false).contains("within it"));
        assert!(full.describe(true).contains("past it"));
        let current = MetalSessionReport::current();
        assert_eq!(
            current.is_some(),
            cfg!(all(feature = "metal", target_os = "macos")),
            "a report exactly when the Metal backend is built"
        );
        if let Some(report) = current {
            assert!(report.max >= 1, "{report:?}");
        }
    }
}
