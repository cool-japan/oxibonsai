//! The measured prefill route of a loaded model (M-18): fused Metal batch
//! prefill, or the sequential single-token decode it replaces.
//!
//! # Why a route decision exists at all
//!
//! The fused batch prefill exists to be faster per token than decoding the
//! prompt one token at a time — and on a healthy kernel set it is, by a wide
//! margin. When it is not (a slow GEMM tile at some batch size, a device
//! under memory pressure, a pathological attention shape), every prompt of
//! that model would be prefilled slower than plain decode, and a long prompt
//! could take many minutes to its first token. The policy keeps the evidence
//! and routes around it.
//!
//! # The cost model
//!
//! Per model (keyed by the GGUF mapping's weight epoch, shared by every
//! replica of that mapping — they run the same kernels on the same device):
//!
//! * **decode cost** `D` (seconds per token): an exponentially weighted mean
//!   of measured single-token fused decode steps, seeded by a prior the
//!   caller derives from the model's weight bytes (decode streams every
//!   weight once per token);
//! * **fused cost** `F(b)` (seconds per token) per batch-size bucket
//!   `b = ⌊log₂ n⌋`: the same kind of mean over measured fused prefill calls
//!   of `n` tokens. A timed-out call records its wait as a lower bound.
//!
//! A prompt of `n` tokens takes the fused route unless the bucket of `n` —
//! or, with none measured, the nearest measured bucket — says the fused path
//! costs more per token than `D`, **both measured**. No evidence means
//! fused: a freshly loaded model always tries the fused path (so the strict
//! parity tests that call it directly exercise it), and so does a model whose
//! decode has only its prior — the prior prices weight streaming alone,
//! which on a small model is far below any command buffer's fixed cost.
//!
//! The fused route runs under a deadline of [`FUSED_BUDGET_FACTOR`] times the
//! predicted sequential time (plus [`FUSED_BUDGET_SLACK`]), so even a first
//! call on a broken kernel set costs at most a few times what decoding the
//! prompt would have, and then records the evidence that routes the next
//! prompt around it.
//!
//! Evidence against the fused path is not final: a stall on a shared GPU
//! looks exactly like a slow kernel. Every [`PROBE_EVERY`]th prompt the
//! measurements route sequentially is sent down the fused route anyway, and a
//! fused measurement far below the recorded one replaces it outright instead
//! of being averaged in — so a model recovers its fused prefill once the
//! stall is over, while a genuinely slow fused path costs at most one probe
//! in [`PROBE_EVERY`] prompts.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// How one prompt is prefilled on the Metal route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillRoute {
    /// The fused batch prefill: one GEMM per projection per micro-batch.
    Fused,
    /// One fused single-token decode step per prompt token.
    Sequential,
}

/// A route decision and the wait budget that goes with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillDecision {
    /// Where the prompt goes.
    pub route: PrefillRoute,
    /// Wait budget of the fused route (meaningless for the sequential one).
    pub fused_budget: Duration,
    /// Predicted sequential time for the prompt, seconds.
    pub predicted_sequential_s: f64,
    /// Predicted fused time for the prompt, seconds, when any fused call of
    /// this model has been measured.
    pub predicted_fused_s: Option<f64>,
}

/// Snapshot of one model's cost model, for diagnostics and tests.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillCostSnapshot {
    /// Decode cost per token (seconds) and how many steps it averages.
    pub decode_s_per_token: f64,
    /// Number of decode steps measured.
    pub decode_samples: u64,
    /// `(bucket, seconds per token, samples)` of every measured fused bucket.
    pub fused_buckets: Vec<(u32, f64, u64)>,
    /// A route override, when one is set.
    pub forced: Option<PrefillRoute>,
}

/// Predicted fused time over predicted sequential time beyond which a prompt
/// takes the sequential route. `1.0`: the fused path is kept exactly as long
/// as it is not measured slower.
const SEQUENTIAL_WHEN_FUSED_OVER: f64 = 1.0;

/// Wait budget of a fused prefill, as a multiple of the predicted sequential
/// time of the same prompt.
pub const FUSED_BUDGET_FACTOR: f64 = 3.0;

/// Fixed slack added to every fused budget.
pub const FUSED_BUDGET_SLACK: Duration = Duration::from_secs(5);

/// Weight of the newest sample in both running means.
const EWMA_ALPHA: f64 = 0.25;

/// Batch-size buckets: `⌊log₂ n⌋` clamped to this range.
const MAX_BUCKET: u32 = 20;

/// One prompt in this many that the measurements route sequentially takes
/// the fused route instead, to re-measure it.
pub const PROBE_EVERY: u64 = 16;

/// A fused sample below this fraction of the recorded mean replaces the mean
/// instead of being averaged in (the recorded cost was a stall).
const RECOVERY_FRACTION: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
struct Mean {
    value: f64,
    samples: u64,
}

impl Mean {
    fn push(&mut self, sample: f64) {
        self.value = if self.samples == 0 {
            sample
        } else {
            self.value + EWMA_ALPHA * (sample - self.value)
        };
        self.samples += 1;
    }

    /// Like `push`, except that a sample far below the mean replaces it.
    fn push_recovering(&mut self, sample: f64) {
        if self.samples > 0 && sample < self.value * RECOVERY_FRACTION {
            self.value = sample;
            self.samples += 1;
        } else {
            self.push(sample);
        }
    }
}

#[derive(Debug, Default)]
struct CostModel {
    decode: Option<Mean>,
    fused: HashMap<u32, Mean>,
    forced: Option<PrefillRoute>,
    /// Prompts the measurements have routed sequentially so far.
    sequential_decisions: u64,
}

fn bucket_of(tokens: usize) -> u32 {
    (usize::BITS - 1 - tokens.max(1).leading_zeros()).min(MAX_BUCKET)
}

impl CostModel {
    fn decode_s_per_token(&self, prior: f64) -> f64 {
        self.decode.map_or(prior, |m| m.value)
    }

    /// Fused seconds per token for `tokens`: its own bucket, else the nearest
    /// measured one (ties to the larger bucket, the costlier evidence).
    fn fused_s_per_token(&self, tokens: usize) -> Option<f64> {
        let wanted = bucket_of(tokens);
        self.fused
            .iter()
            .min_by_key(|(bucket, _)| (bucket.abs_diff(wanted), u32::MAX - **bucket))
            .map(|(_, mean)| mean.value)
    }
}

fn registry() -> &'static Mutex<HashMap<u64, CostModel>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, CostModel>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_model<R>(epoch: u64, f: impl FnOnce(&mut CostModel) -> R) -> R {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(guard.entry(epoch).or_default())
}

/// The prefill cost model of every loaded model, keyed by weight epoch.
pub struct MetalPrefillPolicy;

impl MetalPrefillPolicy {
    /// Decide the route of a `tokens`-token prompt for the model keyed by
    /// `epoch`, given the caller's decode-cost prior (seconds per token).
    #[must_use]
    pub fn decide(epoch: u64, tokens: usize, decode_prior_s_per_token: f64) -> PrefillDecision {
        with_model(epoch, |model| {
            let decode = model.decode_s_per_token(decode_prior_s_per_token.max(0.0));
            let predicted_sequential_s = decode * tokens as f64;
            let predicted_fused_s = model
                .fused_s_per_token(tokens)
                .map(|per_token| per_token * tokens as f64);
            // Divert only on two measurements: a fused cost compared with a
            // decode *prior* says nothing (on a tiny model the prior is a few
            // microseconds, far below any command buffer's fixed cost).
            let mut measured_route = match (predicted_fused_s, model.decode.is_some()) {
                (Some(fused), true)
                    if fused > predicted_sequential_s * SEQUENTIAL_WHEN_FUSED_OVER =>
                {
                    PrefillRoute::Sequential
                }
                _ => PrefillRoute::Fused,
            };
            if measured_route == PrefillRoute::Sequential && model.forced.is_none() {
                model.sequential_decisions += 1;
                if model.sequential_decisions.is_multiple_of(PROBE_EVERY) {
                    measured_route = PrefillRoute::Fused;
                }
            }
            let fused_budget = FUSED_BUDGET_SLACK
                + Duration::from_secs_f64(
                    (predicted_sequential_s * FUSED_BUDGET_FACTOR).clamp(0.0, 86_400.0),
                );
            PrefillDecision {
                route: model.forced.unwrap_or(measured_route),
                fused_budget,
                predicted_sequential_s,
                predicted_fused_s,
            }
        })
    }

    /// Record a fused prefill of `tokens` tokens that completed in `elapsed`.
    pub fn record_fused(epoch: u64, tokens: usize, elapsed: Duration) {
        if tokens == 0 {
            return;
        }
        let per_token = elapsed.as_secs_f64() / tokens as f64;
        with_model(epoch, |model| {
            model
                .fused
                .entry(bucket_of(tokens))
                .or_insert(Mean {
                    value: 0.0,
                    samples: 0,
                })
                .push_recovering(per_token);
        });
    }

    /// Record a fused prefill of `tokens` tokens that timed out after
    /// `waited`: its true cost is at least that, so the bucket is raised to
    /// it (never lowered).
    pub fn record_fused_timeout(epoch: u64, tokens: usize, waited: Duration) {
        if tokens == 0 {
            return;
        }
        let per_token = waited.as_secs_f64() / tokens as f64;
        with_model(epoch, |model| {
            let mean = model.fused.entry(bucket_of(tokens)).or_insert(Mean {
                value: 0.0,
                samples: 0,
            });
            mean.value = mean.value.max(per_token);
            mean.samples += 1;
        });
    }

    /// Record one fused single-token decode step that took `elapsed`.
    pub fn record_decode(epoch: u64, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        with_model(epoch, |model| match model.decode.as_mut() {
            Some(mean) => mean.push(seconds),
            None => {
                model.decode = Some(Mean {
                    value: seconds,
                    samples: 1,
                })
            }
        });
    }

    /// Pin the route of the model keyed by `epoch` (`None` returns it to the
    /// measured decision). Shared by every replica of the mapping.
    pub fn force_route(epoch: u64, route: Option<PrefillRoute>) {
        with_model(epoch, |model| model.forced = route);
    }

    /// Drop everything recorded for `epoch` (its mapping was released).
    pub fn forget(epoch: u64) {
        let mut guard = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.remove(&epoch);
    }

    /// The cost model of `epoch`, for diagnostics.
    #[must_use]
    pub fn snapshot(epoch: u64, decode_prior_s_per_token: f64) -> PrefillCostSnapshot {
        with_model(epoch, |model| {
            let mut fused_buckets: Vec<(u32, f64, u64)> = model
                .fused
                .iter()
                .map(|(bucket, mean)| (*bucket, mean.value, mean.samples))
                .collect();
            fused_buckets.sort_by_key(|(bucket, _, _)| *bucket);
            PrefillCostSnapshot {
                decode_s_per_token: model.decode_s_per_token(decode_prior_s_per_token),
                decode_samples: model.decode.map_or(0, |m| m.samples),
                fused_buckets,
                forced: model.forced,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Epochs no loaded model uses (the model counter starts near 1).
    fn test_epoch() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(u64::MAX / 2);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn no_evidence_means_fused() {
        let epoch = test_epoch();
        let decision = MetalPrefillPolicy::decide(epoch, 4096, 0.05);
        assert_eq!(decision.route, PrefillRoute::Fused);
        assert!(decision.predicted_fused_s.is_none());
        assert!((decision.predicted_sequential_s - 4096.0 * 0.05).abs() < 1e-9);
        let budget = decision.fused_budget.as_secs_f64();
        assert!(
            (budget - (5.0 + 3.0 * 4096.0 * 0.05)).abs() < 1e-6,
            "{budget}"
        );
        MetalPrefillPolicy::forget(epoch);
    }

    #[test]
    fn a_fused_path_measured_slower_than_decode_routes_sequential() {
        let epoch = test_epoch();
        // M-18 as measured on Bonsai-8B: 62 ms/token fused vs 52 ms decode.
        MetalPrefillPolicy::record_decode(epoch, Duration::from_millis(52));
        MetalPrefillPolicy::record_fused(epoch, 256, Duration::from_millis(62 * 256));
        let decision = MetalPrefillPolicy::decide(epoch, 256, 1.0);
        assert_eq!(decision.route, PrefillRoute::Sequential);
        // The nearest measured bucket speaks for an unmeasured size.
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 4096, 1.0).route,
            PrefillRoute::Sequential
        );
        // A healthy measurement in the same bucket brings it back.
        for _ in 0..16 {
            MetalPrefillPolicy::record_fused(epoch, 300, Duration::from_millis(7 * 300));
        }
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 256, 1.0).route,
            PrefillRoute::Fused
        );
        MetalPrefillPolicy::forget(epoch);
    }

    #[test]
    fn a_decode_prior_alone_never_diverts_the_fused_route() {
        let epoch = test_epoch();
        MetalPrefillPolicy::record_fused(epoch, 12, Duration::from_millis(12));
        // 1 ms/token fused against a 2 us/token prior: no decode measured.
        let decision = MetalPrefillPolicy::decide(epoch, 12, 2e-6);
        assert_eq!(decision.route, PrefillRoute::Fused);
        MetalPrefillPolicy::record_decode(epoch, Duration::from_micros(300));
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 12, 2e-6).route,
            PrefillRoute::Sequential,
            "measured on both sides, the slower fused path is routed around"
        );
        MetalPrefillPolicy::forget(epoch);
    }

    #[test]
    fn a_timeout_raises_its_bucket_and_forced_routes_win() {
        let epoch = test_epoch();
        MetalPrefillPolicy::record_decode(epoch, Duration::from_millis(20));
        MetalPrefillPolicy::record_fused(epoch, 128, Duration::from_millis(128));
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 128, 1.0).route,
            PrefillRoute::Fused
        );
        MetalPrefillPolicy::record_fused_timeout(epoch, 128, Duration::from_secs(60));
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 128, 1.0).route,
            PrefillRoute::Sequential
        );
        MetalPrefillPolicy::force_route(epoch, Some(PrefillRoute::Fused));
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 128, 1.0).route,
            PrefillRoute::Fused
        );
        let snapshot = MetalPrefillPolicy::snapshot(epoch, 1.0);
        assert_eq!(snapshot.forced, Some(PrefillRoute::Fused));
        assert_eq!(snapshot.decode_samples, 1);
        assert_eq!(snapshot.fused_buckets.len(), 1);
        MetalPrefillPolicy::forget(epoch);
        let fresh = MetalPrefillPolicy::snapshot(epoch, 1.0);
        assert!(fresh.fused_buckets.is_empty() && fresh.forced.is_none());
        MetalPrefillPolicy::forget(epoch);
    }

    #[test]
    fn a_sequential_verdict_is_re_probed_and_a_healthy_probe_recovers() {
        let epoch = test_epoch();
        MetalPrefillPolicy::record_decode(epoch, Duration::from_millis(50));
        // A stall: 60 s for 256 tokens.
        MetalPrefillPolicy::record_fused_timeout(epoch, 256, Duration::from_secs(60));
        let routes: Vec<PrefillRoute> = (0..PROBE_EVERY)
            .map(|_| MetalPrefillPolicy::decide(epoch, 256, 1.0).route)
            .collect();
        let probes = routes.iter().filter(|r| **r == PrefillRoute::Fused).count();
        assert_eq!(
            probes, 1,
            "exactly one probe in {PROBE_EVERY} sequential verdicts"
        );
        assert_eq!(routes.last(), Some(&PrefillRoute::Fused));
        // The probe measures a healthy 6 ms/token: the stall is forgotten.
        MetalPrefillPolicy::record_fused(epoch, 256, Duration::from_millis(6 * 256));
        assert_eq!(
            MetalPrefillPolicy::decide(epoch, 256, 1.0).route,
            PrefillRoute::Fused
        );
        MetalPrefillPolicy::forget(epoch);
    }

    #[test]
    fn buckets_are_log2_of_the_batch() {
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(1), 0);
        assert_eq!(bucket_of(2), 1);
        assert_eq!(bucket_of(255), 7);
        assert_eq!(bucket_of(256), 8);
        assert_eq!(bucket_of(4096), 12);
        assert_eq!(bucket_of(usize::MAX), MAX_BUCKET);
    }
}
