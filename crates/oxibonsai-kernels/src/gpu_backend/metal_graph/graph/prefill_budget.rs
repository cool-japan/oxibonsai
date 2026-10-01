//! Wait budgets of the batched Metal prefill (M-18).
//!
//! Every command buffer the prefill runner commits is waited on with
//! [`super::super::buffers::commit_and_wait_bounded`] against a deadline:
//!
//! * the **caller's** deadline, when the calling thread opened a
//!   [`PrefillDeadlineScope`] — the model's prefill router does, with a
//!   budget derived from its measured per-token cost model; else
//! * a **default** deadline from [`default_prefill_budget`]: the request's
//!   floating-point work at a deliberately pessimistic throughput floor, plus
//!   a fixed slack. It is not a tuning knob — it only bounds a prefill that
//!   has stopped making progress, so a server thread can never park forever
//!   in `waitUntilCompleted`.
//!
//! The scope and the fault-injection counter are per thread: the runner waits
//! on the thread that called it, synchronously, however deep the call came
//! from (the cached ternary entry points in `metal_full_layer` call the
//! encoder with fixed arguments, so a budget cannot be threaded through them
//! as a parameter).

use std::cell::Cell;
use std::time::{Duration, Instant};

use super::MetalGraph;

thread_local! {
    /// Deadline of the innermost [`PrefillDeadlineScope`] on this thread.
    static PREFILL_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };

    /// Bounded prefill waits on this thread still to be reported as timed out
    /// ([`MetalGraph::force_prefill_timeouts`]).
    static FORCED_PREFILL_TIMEOUTS: Cell<u32> = const { Cell::new(0) };
}

/// Throughput floor of the default budget, in floating-point operations per
/// second: about 1 % of an M3 GPU's f32 peak, ten times below the slowest
/// fused prefill ever measured on a real model here (the row-wise Q1 GEMM
/// before M-18, ~0.2 TFLOP/s). A prefill slower than this has stalled.
const DEFAULT_BUDGET_FLOOR_FLOPS: f64 = 25.0e9;

/// Fixed slack of the default budget: queueing behind other work on a shared
/// GPU, first-use buffer faults and scheduler noise.
const DEFAULT_BUDGET_SLACK: Duration = Duration::from_secs(10);

/// Geometry of a dense transformer the default prefill budget is computed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillWorkShape {
    /// Transformer layers.
    pub n_layers: usize,
    /// Hidden width.
    pub hidden: usize,
    /// FFN intermediate width.
    pub intermediate: usize,
    /// Query heads.
    pub nq: usize,
    /// Key/value heads.
    pub nkv: usize,
    /// Per-head width.
    pub head_dim: usize,
}

impl PrefillWorkShape {
    /// Multiply-adds of the layer projections for one token.
    #[must_use]
    pub fn projection_macs_per_token(&self) -> f64 {
        let qkv = (self.nq + 2 * self.nkv) * self.head_dim;
        let attn = self.nq * self.head_dim;
        let per_layer = self.hidden * qkv
            + attn * self.hidden
            + 2 * self.hidden * self.intermediate
            + self.intermediate * self.hidden;
        (per_layer * self.n_layers) as f64
    }

    /// Floating-point operations of a `tokens`-token prefill starting at
    /// `pos_start`: the projections, plus causal attention (`QKᵀ` and `PV`)
    /// over every earlier position.
    #[must_use]
    pub fn prefill_flops(&self, tokens: usize, pos_start: usize) -> f64 {
        let n = tokens as f64;
        let mean_context = pos_start as f64 + (n + 1.0) / 2.0;
        let attention = 4.0 * n * mean_context * (self.nq * self.head_dim * self.n_layers) as f64;
        2.0 * n * self.projection_macs_per_token() + attention
    }
}

/// The default wait budget of a `tokens`-token prefill at `pos_start`: its
/// floating-point work at a floor rate of 25 GFLOP/s, plus 10 s of slack
/// (`DEFAULT_BUDGET_FLOOR_FLOPS`, `DEFAULT_BUDGET_SLACK`).
///
/// For Bonsai-8B that is ~13 s for a 10-token prompt and ~40 minutes for a
/// 4096-token one: a stall detector, not a performance expectation. A caller
/// with a measured cost model sets a tighter deadline through
/// [`MetalGraph::prefill_deadline_scope`].
#[must_use]
pub fn default_prefill_budget(
    shape: &PrefillWorkShape,
    tokens: usize,
    pos_start: usize,
) -> Duration {
    let seconds = shape.prefill_flops(tokens, pos_start) / DEFAULT_BUDGET_FLOOR_FLOPS;
    DEFAULT_BUDGET_SLACK + Duration::from_secs_f64(seconds.clamp(0.0, 30.0 * 24.0 * 3600.0))
}

/// RAII scope bounding every batched-prefill command-buffer wait this thread
/// performs until it drops (see the module docs).
///
/// Scopes nest: an inner scope's deadline applies until it drops, then the
/// outer one's is restored.
#[must_use = "the deadline applies only while the scope is alive"]
pub struct PrefillDeadlineScope {
    previous: Option<Instant>,
}

impl Drop for PrefillDeadlineScope {
    fn drop(&mut self) {
        let previous = self.previous;
        let _ = PREFILL_DEADLINE.try_with(|cell| cell.set(previous));
    }
}

impl MetalGraph {
    /// Bound every batched-prefill command-buffer wait on this thread by
    /// `deadline` until the returned scope drops (M-18).
    ///
    /// A prefill that misses the deadline returns a timeout
    /// ([`super::super::MetalGraphError::is_command_buffer_timeout`]) instead
    /// of blocking; its in-flight command buffer finishes on its own and is
    /// drained before the session's prefill buffers are reused.
    pub fn prefill_deadline_scope(deadline: Instant) -> PrefillDeadlineScope {
        let previous = PREFILL_DEADLINE
            .try_with(|cell| cell.replace(Some(deadline)))
            .unwrap_or(None);
        PrefillDeadlineScope { previous }
    }

    /// Fault injection (M-18): report the next `count` bounded batched-prefill
    /// waits **on this thread** as timed out, whatever the GPU's progress.
    ///
    /// The command buffers still run to completion; only the wait gives up,
    /// exactly as it does when a real deadline passes. `0` cancels any
    /// pending injection.
    pub fn force_prefill_timeouts(count: u32) {
        let _ = FORCED_PREFILL_TIMEOUTS.try_with(|cell| cell.set(count));
    }
}

/// The deadline of the innermost [`PrefillDeadlineScope`] on this thread.
pub(crate) fn current_prefill_deadline() -> Option<Instant> {
    PREFILL_DEADLINE.try_with(Cell::get).unwrap_or(None)
}

/// Consume one injected timeout, if any is pending on this thread.
pub(crate) fn take_forced_prefill_timeout() -> bool {
    FORCED_PREFILL_TIMEOUTS
        .try_with(|cell| {
            let pending = cell.get();
            if pending > 0 {
                cell.set(pending - 1);
                true
            } else {
                false
            }
        })
        .unwrap_or(false)
}

/// The deadline a prefill starting now waits against: the caller's scope when
/// one is open, else `now + default` — whichever is earlier when both exist is
/// the caller's call, so the scope wins outright.
pub(crate) fn effective_prefill_deadline(now: Instant, default: Duration) -> Instant {
    current_prefill_deadline().unwrap_or(now + default)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BONSAI_8B: PrefillWorkShape = PrefillWorkShape {
        n_layers: 36,
        hidden: 4096,
        intermediate: 12288,
        nq: 32,
        nkv: 8,
        head_dim: 128,
    };

    #[test]
    fn projection_macs_match_the_8b_parameter_count() {
        // 36 layers x (4096*6144 + 4096*4096 + 3*4096*12288) = 6.95e9.
        let macs = BONSAI_8B.projection_macs_per_token();
        assert!((macs - 6.946e9).abs() / 6.946e9 < 1e-3, "{macs}");
    }

    #[test]
    fn default_budget_grows_with_the_work_and_never_drops_below_the_slack() {
        let short = default_prefill_budget(&BONSAI_8B, 10, 0);
        let long = default_prefill_budget(&BONSAI_8B, 4096, 0);
        let later = default_prefill_budget(&BONSAI_8B, 4096, 4096);
        assert!(short >= DEFAULT_BUDGET_SLACK);
        assert!(long > short * 10, "{long:?} vs {short:?}");
        assert!(later > long, "attention over a longer history costs more");
        // ~57 TFLOP at the floor: tens of minutes, a stall bound.
        assert!(long > Duration::from_secs(600) && long < Duration::from_secs(4 * 3600));
    }

    #[test]
    fn deadline_scopes_nest_and_restore() {
        assert!(current_prefill_deadline().is_none());
        let now = Instant::now();
        let outer_deadline = now + Duration::from_secs(5);
        let inner_deadline = now + Duration::from_secs(1);
        {
            let _outer = MetalGraph::prefill_deadline_scope(outer_deadline);
            assert_eq!(current_prefill_deadline(), Some(outer_deadline));
            {
                let _inner = MetalGraph::prefill_deadline_scope(inner_deadline);
                assert_eq!(current_prefill_deadline(), Some(inner_deadline));
                assert_eq!(
                    effective_prefill_deadline(now, Duration::from_secs(99)),
                    inner_deadline
                );
            }
            assert_eq!(current_prefill_deadline(), Some(outer_deadline));
        }
        assert!(current_prefill_deadline().is_none());
        assert_eq!(
            effective_prefill_deadline(now, Duration::from_secs(3)),
            now + Duration::from_secs(3)
        );
    }

    #[test]
    fn forced_timeouts_are_counted_per_thread() {
        MetalGraph::force_prefill_timeouts(2);
        let other = std::thread::spawn(take_forced_prefill_timeout)
            .join()
            .expect("probe thread");
        assert!(!other, "an injection never leaks to another thread");
        assert!(take_forced_prefill_timeout());
        assert!(take_forced_prefill_timeout());
        assert!(!take_forced_prefill_timeout());
        MetalGraph::force_prefill_timeouts(1);
        MetalGraph::force_prefill_timeouts(0);
        assert!(
            !take_forced_prefill_timeout(),
            "0 cancels a pending injection"
        );
    }
}
