//! Decode-attention criterion benchmark (K-M1).
//!
//! Benchmarks [`fused_attention_head_contiguous`] against a **local copy**
//! of the frozen pre-change algorithm ([`old_algorithm`], below), sweeping
//! `seq_len` in `{128, 256, 512, 1024, 4096, 8192}` x `head_dim` in
//! `{64, 128, 256}` — not just `seq_len=8192`, which maximally *dilutes* the
//! allocation-churn win the change
//! targets (one accumulator amortised over `8192*256` FLOPs) and is what
//! produced the sub-1.0x readings that withdrew the original `>= 1.5x`
//! target. This benchmark asserts **no ratio threshold** — it only records
//! numbers, per that withdrawal (the four
//! release-run readings were 0.87x / 0.94x / 1.05x / 1.23x, two of them
//! below 1.0x, and `attention_fused.rs`'s own doc comment now reads "a wash
//! to modestly faster at this configuration").
//!
//! ## Why `old_algorithm` is a local copy, not `crate::layers::attention_fused::tests::old_algorithm`
//!
//! The real frozen baseline lives in
//! `crates/oxibonsai-model/src/layers/attention_fused_tests.rs`'s
//! `mod old_algorithm`, included into `attention_fused.rs` only under
//! `#[cfg(test)] #[path = "attention_fused_tests.rs"] mod tests;`. Two
//! independent things make it unreachable from an external `benches/`
//! crate, and both live in that file:
//!
//! 1. `mod tests` is gated `#[cfg(test)]`, so it is compiled only for this
//!    crate's own `cargo test`/`cargo nextest` run, never for a dependent
//!    crate such as a `[[bench]]` target (which links `oxibonsai_model` as
//!    an ordinary external dependency, built without `--cfg test`).
//! 2. Even if that gate were widened, `mod old_algorithm` and its
//!    `old_fused_attention_head_contiguous` function are `pub(super)`
//!    relative to `tests` — visible to `tests` and its descendants, never
//!    to `tests`'s *ancestors* (`attention_fused`, and everything outside
//!    it, including this bench crate). Rust's module privacy does not
//!    offer a way to widen that from the *call site* of `mod tests;`
//!    alone; the function's own visibility annotation would have to
//!    change, and that annotation lives in the un-owned file.
//!
//! Editing `attention_fused.rs`'s `#[cfg(test)]` gate to also compile under
//! a bench feature was considered and rejected: this workspace's release
//! gate runs `cargo clippy --workspace --all-features --all-targets`, and
//! `--all-features` would then drag the ~1200-line, 28-test-function
//! `attention_fused_tests.rs` into the *library* build for the first time
//! ever (previously only compiled under `--cfg test`), an untested and
//! unnecessary blast radius for a benchmark. `attention_fused.rs` is left
//! byte-for-byte unedited.
//!
//! Instead, [`old_algorithm`] below is a verbatim copy of the same frozen
//! algorithm (same doc rationale: deliberately NOT deduplicated against the
//! real implementation, so it keeps computing the OLD algorithm even as the
//! real one evolves), built only from [`fused_attention_head_contiguous`]'s
//! genuinely `pub` siblings ([`dot_f32`], [`ATTENTION_BLOCK_SIZE`]). Each
//! swept configuration is parity-checked against the current implementation
//! (max-abs-diff < 1e-5, the project's own established frozen-old tolerance
//! — see `attention_fused_tests.rs`'s `*_frozen_old_*` tests) before it is
//! timed, so a silent copy/paste drift between this file and the original
//! would fail loudly instead of quietly comparing against the wrong
//! baseline.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use oxibonsai_model::layers::attention_fused::{
    dot_f32, fused_attention_head_contiguous, ATTENTION_BLOCK_SIZE,
};
use oxibonsai_model::ModelResult;

/// Verbatim copy of `attention_fused_tests.rs::old_algorithm` — see this
/// file's module doc for why it is copied rather than reused.
mod old_algorithm {
    use super::{dot_f32, ModelResult, ATTENTION_BLOCK_SIZE};

    pub(super) struct OldOnlineSoftmaxState {
        max_val: f32,
        sum_exp: f32,
        output: Vec<f32>,
    }

    impl OldOnlineSoftmaxState {
        fn new(head_dim: usize) -> Self {
            Self {
                max_val: f32::NEG_INFINITY,
                sum_exp: 0.0,
                output: vec![0.0f32; head_dim],
            }
        }

        fn update(&mut self, scores: &[f32], values: &[&[f32]], head_dim: usize) {
            for (idx, &score) in scores.iter().enumerate() {
                let v = values[idx];
                if score > self.max_val {
                    let rescale = if self.max_val == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (self.max_val - score).exp()
                    };
                    self.sum_exp *= rescale;
                    for d in 0..head_dim {
                        self.output[d] *= rescale;
                    }
                    self.max_val = score;
                }
                let exp_score = (score - self.max_val).exp();
                self.sum_exp += exp_score;
                for (out_d, &v_d) in self.output[..head_dim].iter_mut().zip(v.iter()) {
                    *out_d += exp_score * v_d;
                }
            }
        }

        fn finalize(&mut self) {
            if self.sum_exp > 0.0 {
                let inv_sum = 1.0 / self.sum_exp;
                for d in self.output.iter_mut() {
                    *d *= inv_sum;
                }
            }
        }
    }

    /// Pre-change algorithm: scalar V-accumulate, rescale on every new max
    /// within a block, one `Vec` allocation per block.
    pub(super) fn old_fused_attention_head_contiguous(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) -> ModelResult<()> {
        if seq_len == 0 {
            for d in output.iter_mut() {
                *d = 0.0;
            }
            return Ok(());
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut state = OldOnlineSoftmaxState::new(head_dim);
        let mut pos = 0;
        while pos < seq_len {
            let block_end = (pos + ATTENTION_BLOCK_SIZE).min(seq_len);
            let block_len = block_end - pos;
            let mut block_scores = Vec::with_capacity(block_len);
            let mut block_values: Vec<&[f32]> = Vec::with_capacity(block_len);
            for t in pos..block_end {
                let k_slice = &keys[t * head_dim..(t + 1) * head_dim];
                block_scores.push(dot_f32(query, k_slice) * scale);
                block_values.push(&values[t * head_dim..(t + 1) * head_dim]);
            }
            state.update(&block_scores, &block_values, head_dim);
            pos = block_end;
        }
        state.finalize();
        output[..head_dim].copy_from_slice(&state.output[..head_dim]);
        Ok(())
    }
}

/// Deterministic (fixed-seed xorshift) query/keys/values fixture so the
/// benchmark is reproducible across runs without adding a `rand`
/// dependency.
fn make_fixture(seq_len: usize, head_dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut state = seed | 1;
    let mut next_f32 = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // `state >> 40` is a 24-bit value, so dividing by `1 << 24` alone
        // gives [0, 1) (never negative, always feeding the online-softmax
        // path all-same-sign dot products); scale by 2 and re-center for
        // an actually signed [-1.0, 1.0) fixture.
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    let query: Vec<f32> = (0..head_dim).map(|_| next_f32()).collect();
    let keys: Vec<f32> = (0..seq_len * head_dim).map(|_| next_f32()).collect();
    let values: Vec<f32> = (0..seq_len * head_dim).map(|_| next_f32()).collect();
    (query, keys, values)
}

/// Fail loudly (before any timing) if this file's [`old_algorithm`] copy
/// has drifted from the current implementation's numerical behaviour,
/// using the project's own established frozen-old tolerance (1e-5).
fn assert_parity(query: &[f32], keys: &[f32], values: &[f32], seq_len: usize, head_dim: usize) {
    let mut out_current = vec![0.0f32; head_dim];
    let mut out_old = vec![0.0f32; head_dim];
    fused_attention_head_contiguous(query, keys, values, &mut out_current, seq_len, head_dim)
        .expect("current fused_attention_head_contiguous should not error on a valid fixture");
    old_algorithm::old_fused_attention_head_contiguous(
        query,
        keys,
        values,
        &mut out_old,
        seq_len,
        head_dim,
    )
    .expect("old_algorithm copy should not error on a valid fixture");

    let max_diff = out_current
        .iter()
        .zip(out_old.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "attention_bench's old_algorithm copy disagrees with the current \
         fused_attention_head_contiguous at seq_len={seq_len} head_dim={head_dim} \
         (max diff {max_diff}) -- the frozen-baseline copy may have drifted \
         from attention_fused_tests.rs, or the current algorithm's real \
         behaviour has changed; either way this benchmark's comparison \
         would be against the wrong baseline"
    );
}

const SEQ_LENS: &[usize] = &[128, 256, 512, 1024, 4096, 8192];
const HEAD_DIMS: &[usize] = &[64, 128, 256];

fn bench_decode_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_attention_fused_vs_old_algorithm");

    for &head_dim in HEAD_DIMS {
        for &seq_len in SEQ_LENS {
            let (query, keys, values) = make_fixture(seq_len, head_dim, 0xA5A5_5A5A_1234_9876);
            assert_parity(&query, &keys, &values, seq_len, head_dim);

            let mut output = vec![0.0f32; head_dim];
            let label = format!("seq{seq_len}_hd{head_dim}");

            group.bench_with_input(BenchmarkId::new("current", &label), &label, |b, _| {
                b.iter(|| {
                    fused_attention_head_contiguous(
                        black_box(&query),
                        black_box(&keys),
                        black_box(&values),
                        black_box(&mut output),
                        seq_len,
                        head_dim,
                    )
                    .expect("current fused_attention_head_contiguous should not error");
                });
            });

            group.bench_with_input(BenchmarkId::new("old_algorithm", &label), &label, |b, _| {
                b.iter(|| {
                    old_algorithm::old_fused_attention_head_contiguous(
                        black_box(&query),
                        black_box(&keys),
                        black_box(&values),
                        black_box(&mut output),
                        seq_len,
                        head_dim,
                    )
                    .expect("old_algorithm copy should not error");
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_decode_attention);
criterion_main!(benches);
