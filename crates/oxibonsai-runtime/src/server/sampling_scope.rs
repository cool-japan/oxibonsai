//! One request's sampler configuration on a leased engine replica.
//!
//! Every OpenAI endpoint (`/v1/chat/completions`, `/extended`,
//! `/v1/completions`) runs its generation through [`RequestSampling::run`]:
//! the request's resolved [`SamplingParams`], its frequency/presence
//! penalties, its `min_p` and its `seed` are installed on the replica's
//! crate-visible [`crate::sampling::Sampler`] for the duration of that one
//! generation and the replica's own state is restored afterwards — also
//! when the generation panics, because the restore lives in a guard's
//! `Drop`. This is the single place the server reads or writes a replica
//! sampler's `min_p`:
//!
//! * the **baseline** `min_p` is whatever the replica's sampler carries (the
//!   serve binaries set it per replica at start-up); a request that omits
//!   `min_p` samples with it, a request that sends one overrides it for that
//!   request only, and `min_p: 0` disables it for that request;
//! * a **seeded** request runs on a fresh [`crate::sampling::Sampler`]
//!   seeded with the request's seed, carrying the request's params, its
//!   penalties and the effective `min_p` (the request's, else the
//!   replica's baseline) — the replica's own sampler, PRNG state included,
//!   is put back untouched afterwards;
//! * an **unseeded** request swaps only the configuration (params,
//!   penalties, `min_p`) onto the replica's sampler, whose PRNG keeps
//!   advancing exactly as before.

use crate::engine::InferenceEngine;
use crate::sampling::{PenaltyParams, Sampler, SamplingParams};

/// How one request samples — see the module doc.
#[derive(Debug, Clone)]
pub(crate) struct RequestSampling {
    /// The request's resolved sampling parameters.
    pub(crate) params: SamplingParams,
    /// The request's frequency/presence penalties; `None` keeps the
    /// replica's own.
    pub(crate) penalties: Option<PenaltyParams>,
    /// The request's `min_p`; `None` keeps the replica's baseline.
    pub(crate) min_p: Option<f32>,
    /// A per-request seed: the generation runs on a fresh sampler seeded
    /// with it.
    pub(crate) seed: Option<u64>,
}

/// What [`SamplerScope`] puts back when it drops.
enum Restore {
    /// A seeded run replaced the whole sampler.
    Sampler(Sampler),
    /// An unseeded run overrode the configuration only.
    Config {
        params: SamplingParams,
        penalties: PenaltyParams,
        min_p: f32,
    },
}

/// The engine with a request's sampler configuration installed; the
/// replica's own configuration comes back when this drops (see the module
/// doc).
pub(crate) struct SamplerScope<'e, 'a> {
    engine: &'e mut InferenceEngine<'a>,
    restore: Option<Restore>,
}

impl<'a> std::ops::Deref for SamplerScope<'_, 'a> {
    type Target = InferenceEngine<'a>;

    fn deref(&self) -> &Self::Target {
        self.engine
    }
}

impl std::ops::DerefMut for SamplerScope<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.engine
    }
}

impl Drop for SamplerScope<'_, '_> {
    fn drop(&mut self) {
        match self.restore.take() {
            Some(Restore::Sampler(sampler)) => self.engine.sampler = sampler,
            Some(Restore::Config {
                params,
                penalties,
                min_p,
            }) => {
                self.engine.sampler.set_params(params);
                self.engine.sampler.set_penalties(penalties);
                self.engine.sampler.set_min_p(min_p);
            }
            None => {}
        }
    }
}

impl RequestSampling {
    /// The `min_p` this request samples with on `engine`: its own, else the
    /// replica's baseline.
    pub(crate) fn effective_min_p(&self, engine: &InferenceEngine<'_>) -> f32 {
        self.min_p.unwrap_or_else(|| engine.sampler.min_p())
    }

    /// Install this configuration on `engine` until the returned scope
    /// drops.
    pub(crate) fn enter<'e, 'a>(
        &self,
        engine: &'e mut InferenceEngine<'a>,
    ) -> SamplerScope<'e, 'a> {
        let restore = match self.seed {
            Some(seed) => {
                let mut fresh = Sampler::new(self.params.clone(), seed);
                let penalties = match self.penalties {
                    Some(penalties) => penalties,
                    None => *engine.sampler.penalties(),
                };
                fresh.set_penalties(penalties);
                fresh.set_min_p(self.effective_min_p(engine));
                Restore::Sampler(std::mem::replace(&mut engine.sampler, fresh))
            }
            None => {
                let restore = Restore::Config {
                    params: engine.sampler.params().clone(),
                    penalties: *engine.sampler.penalties(),
                    min_p: engine.sampler.min_p(),
                };
                engine.sampler.set_params(self.params.clone());
                if let Some(penalties) = self.penalties {
                    engine.sampler.set_penalties(penalties);
                }
                if let Some(min_p) = self.min_p {
                    engine.sampler.set_min_p(min_p);
                }
                restore
            }
        };
        SamplerScope {
            engine,
            restore: Some(restore),
        }
    }

    /// Run `generate` on `engine` under this configuration (see the module
    /// doc), restoring the replica's own sampler state afterwards.
    pub(crate) fn run<'a, R>(
        &self,
        engine: &mut InferenceEngine<'a>,
        generate: impl FnOnce(&mut InferenceEngine<'a>) -> R,
    ) -> R {
        let mut scope = self.enter(engine);
        generate(&mut scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> InferenceEngine<'static> {
        let params = SamplingParams {
            temperature: 0.9,
            ..SamplingParams::default()
        };
        let mut engine =
            InferenceEngine::new(oxibonsai_core::config::Qwen3Config::tiny_test(), params, 7);
        engine.sampler.set_min_p(0.25);
        engine.sampler.set_penalties(PenaltyParams::new(0.5, 0.0));
        engine
    }

    fn request(seed: Option<u64>, min_p: Option<f32>) -> RequestSampling {
        RequestSampling {
            params: SamplingParams {
                temperature: 0.3,
                ..SamplingParams::default()
            },
            penalties: Some(PenaltyParams::new(1.0, 1.5)),
            min_p,
            seed,
        }
    }

    #[test]
    fn an_unseeded_request_overrides_then_restores_the_configuration() {
        let mut engine = engine();
        let seen = request(None, Some(0.6)).run(&mut engine, |e| {
            (
                e.sampler.params().temperature,
                e.sampler.min_p(),
                *e.sampler.penalties(),
            )
        });
        assert_eq!(seen, (0.3, 0.6, PenaltyParams::new(1.0, 1.5)));
        assert_eq!(engine.sampler.params().temperature, 0.9);
        assert_eq!(engine.sampler.min_p(), 0.25);
        assert_eq!(*engine.sampler.penalties(), PenaltyParams::new(0.5, 0.0));
    }

    #[test]
    fn a_request_without_min_p_samples_with_the_replica_baseline() {
        let mut engine = engine();
        for seed in [None, Some(3)] {
            let seen = request(seed, None).run(&mut engine, |e| e.sampler.min_p());
            assert_eq!(seen, 0.25, "seed={seed:?}");
            assert_eq!(engine.sampler.min_p(), 0.25);
        }
    }

    #[test]
    fn min_p_zero_disables_the_baseline_for_that_request_only() {
        let mut engine = engine();
        let seen = request(None, Some(0.0)).run(&mut engine, |e| e.sampler.min_p());
        assert_eq!(seen, 0.0);
        assert_eq!(engine.sampler.min_p(), 0.25);
    }

    #[test]
    fn a_seeded_request_runs_on_a_fresh_sampler_and_hands_the_replicas_back() {
        let mut engine = engine();
        let logits = vec![0.0f32; 64];
        let first = request(Some(11), Some(0.5)).run(&mut engine, |e| {
            assert_eq!(e.sampler.min_p(), 0.5);
            assert_eq!(e.sampler.params().temperature, 0.3);
            e.sampler.sample(&logits).unwrap_or_default()
        });
        let again = request(Some(11), Some(0.5)).run(&mut engine, |e| {
            e.sampler.sample(&logits).unwrap_or_default()
        });
        assert_eq!(first, again, "the same seed draws the same token");
        assert_eq!(engine.sampler.params().temperature, 0.9);
        assert_eq!(engine.sampler.min_p(), 0.25);
        assert_eq!(*engine.sampler.penalties(), PenaltyParams::new(0.5, 0.0));
    }

    /// A seeded request leaves the replica's own PRNG exactly where it was:
    /// the next unseeded draw on a replica that served a seeded request
    /// equals the next draw on an identical replica that served none.
    #[test]
    fn a_seeded_request_leaves_the_replicas_prng_untouched() {
        let logits = vec![0.0f32; 64];
        let mut served = engine();
        let mut untouched = engine();
        let _ = request(Some(99), None).run(&mut served, |e| {
            (0..8)
                .map(|_| e.sampler.sample(&logits).unwrap_or_default())
                .collect::<Vec<u32>>()
        });
        for _ in 0..8 {
            assert_eq!(
                served.sampler.sample(&logits).unwrap_or_default(),
                untouched.sampler.sample(&logits).unwrap_or_default()
            );
        }
    }

    /// Table-style proof of every property this module's doc comment
    /// promises for a fresh seeded *or* unseeded sampler, in one pass: a
    /// request's own `min_p` wins over the replica's baseline, an omitted
    /// `min_p` falls back to it, penalties are carried the same way (the
    /// request's own, or the replica's when the request sends none), and a
    /// seeded request reproduces the same ids across two separate calls.
    #[test]
    fn min_p_and_penalties_follow_the_table_seeded_and_unseeded() {
        struct Row {
            seed: Option<u64>,
            min_p: Option<f32>,
            penalties: Option<PenaltyParams>,
            want_min_p: f32,
            want_penalties: PenaltyParams,
        }
        // The replica baseline (see `engine()`, below): min_p 0.25,
        // penalties (0.5, 0.0).
        let request_penalties = PenaltyParams::new(1.0, 1.5);
        let rows = [
            Row {
                seed: None,
                min_p: Some(0.6),
                penalties: Some(request_penalties),
                want_min_p: 0.6,
                want_penalties: request_penalties,
            },
            Row {
                seed: None,
                min_p: None,
                penalties: Some(request_penalties),
                want_min_p: 0.25,
                want_penalties: request_penalties,
            },
            Row {
                seed: Some(11),
                min_p: Some(0.6),
                penalties: Some(request_penalties),
                want_min_p: 0.6,
                want_penalties: request_penalties,
            },
            Row {
                seed: Some(11),
                min_p: None,
                penalties: Some(request_penalties),
                want_min_p: 0.25,
                want_penalties: request_penalties,
            },
            Row {
                seed: Some(11),
                min_p: Some(0.6),
                penalties: None,
                want_min_p: 0.6,
                want_penalties: PenaltyParams::new(0.5, 0.0),
            },
        ];
        // Non-uniform, non-degenerate logits: a flat row would make min_p
        // and the seeded draw vacuous checks.
        let logits: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.37).sin() * 5.0 + (i as f32) * 0.01)
            .collect();

        for (row_idx, row) in rows.iter().enumerate() {
            let mut replica = engine();
            let sampling = RequestSampling {
                params: SamplingParams {
                    temperature: 0.7,
                    ..SamplingParams::default()
                },
                penalties: row.penalties,
                min_p: row.min_p,
                seed: row.seed,
            };
            let (seen_min_p, seen_penalties, draws) = sampling.run(&mut replica, |e| {
                let seen_min_p = e.sampler.min_p();
                let seen_penalties = *e.sampler.penalties();
                let draws: Vec<u32> = (0..8)
                    .map(|_| e.sampler.sample(&logits).unwrap_or_default())
                    .collect();
                (seen_min_p, seen_penalties, draws)
            });
            assert_eq!(seen_min_p, row.want_min_p, "row {row_idx}: effective min_p");
            assert_eq!(
                seen_penalties, row.want_penalties,
                "row {row_idx}: effective penalties"
            );
            // The replica must always come back exactly as it started.
            assert_eq!(
                replica.sampler.min_p(),
                0.25,
                "row {row_idx}: replica min_p not restored"
            );
            assert_eq!(
                *replica.sampler.penalties(),
                PenaltyParams::new(0.5, 0.0),
                "row {row_idx}: replica penalties not restored"
            );

            if let Some(seed) = row.seed {
                // A fresh replica with the identical baseline `engine()`
                // builds — a seeded run's PRNG state must not carry over
                // from the first call, so this is a new engine, not a
                // second draw on the same one.
                let mut replica_again = engine();
                let draws_again = sampling.run(&mut replica_again, |e| {
                    (0..8)
                        .map(|_| e.sampler.sample(&logits).unwrap_or_default())
                        .collect::<Vec<u32>>()
                });
                assert_eq!(
                    draws, draws_again,
                    "row {row_idx}: seed {seed} must reproduce the same ids on two calls"
                );
            }
        }
    }

    #[test]
    fn the_replica_state_is_restored_even_when_the_generation_panics() {
        let mut engine = engine();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            request(None, Some(0.9)).run(&mut engine, |_e| -> u32 {
                panic!("generation failed");
            })
        }));
        assert!(outcome.is_err());
        assert_eq!(engine.sampler.min_p(), 0.25);
        assert_eq!(engine.sampler.params().temperature, 0.9);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            request(Some(1), Some(0.9)).run(&mut engine, |_e| -> u32 {
                panic!("generation failed");
            })
        }));
        assert!(outcome.is_err());
        assert_eq!(engine.sampler.min_p(), 0.25);
    }
}
