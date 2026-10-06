//! Acceptance suite for the INT8 dot-product tier (K-14).
//!
//! Covers, in order:
//!
//! 1. **Cross-tier bit-identity** — every `Int8Tier` forms the same `i32`
//!    per block (integer arithmetic is exact), so their `f32` outputs must
//!    be bit-for-bit equal, not merely close.
//! 2. **Accuracy vs the f32 tier** — `cos >= 0.999`, on synthetic weights
//!    *and* on the real 1.7B / 8B / Bonsai-8B / 27B GGUFs when they are
//!    present — for the kernels directly **and** through every native-format
//!    entry point the model calls (`gemv_adaptive*`, `gemm_adaptive_ternary`,
//!    the dispatcher's `OneBitKernel` / `TernaryKernel` methods and the
//!    `parallel` / `parallel_tiled` drivers), selected the way a user selects
//!    it: through `OXIBONSAI_KERNEL_TIER`.
//! 3. **The HARD CONSTRAINT** — the tier is never selected implicitly, the
//!    f32 tiers are untouched by it, and a `KernelTier::Gpu` dispatcher is
//!    never diverted even when the variable is set.
//! 4. **Throughput** — reported always, asserted only under
//!    `OXIBONSAI_INT8_BENCH=1` (a wall-clock ratio is not a sound pass/fail
//!    signal on a shared build machine): the GEMV inner loop, the native
//!    GEMV/GEMM at the 1.7B / 8B shapes (`m = 1` and `m = 64`), the 1-bit
//!    tier against the f32 parallel default, and the 27B `ffn_up` GEMM.
//!
//! ## Why one `mod int8_tier_parity`
//!
//! A gate that selects this file with a **substring filter over test
//! names** (`cargo test ... int8_tier_parity`) instead of `--test` matches
//! a test's full in-binary path, which for a top-level `#[test] fn foo` in
//! an integration test is just `foo` — the filter would select **zero tests
//! and exit 0** (systemic finding S-1). Wrapping everything in a module
//! named after the file makes every path start with `int8_tier_parity::`,
//! so either spelling really runs them.
//!
//! Real-model tests self-skip when `models/` is empty — the normal state in
//! a fresh clone or an isolated worktree — and print why, so a skip is
//! never mistaken for a pass. Point them at a checkout with weights via
//! `OXIBONSAI_MODELS_DIR` (no path is ever hardcoded).

mod int8_tier_parity {
    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_core::gguf::types::GgufTensorType;
    use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
    use oxibonsai_core::{BlockPQ2_0, BlockTQ2_0_g128, QK_PQ2_0, QK_TQ2_0_G128};
    use oxibonsai_kernels::dispatch_int8::{
        gemm_1bit_g128_int8, gemm_two_bit_int8, gemv_1bit_g128_int8, gemv_two_bit_int8, Int8Tier,
        KERNEL_TIER_ENV,
    };
    use oxibonsai_kernels::quant_activation::{Int8Activation, Int8Layout};
    use oxibonsai_kernels::traits::{OneBitKernel, TernaryKernel};
    use oxibonsai_kernels::{KernelDispatcher, KernelResult, KernelTier};
    use std::time::Instant;

    // ─── helpers ─────────────────────────────────────────────────────────

    /// Serializes every test in *this* binary that reads or writes an
    /// environment variable: the tier tests mutate [`KERNEL_TIER_ENV`], the
    /// throughput tests read `OXIBONSAI_INT8_BENCH`, and the real-model tests
    /// read `OXIBONSAI_MODELS_DIR` via `oxibonsai_testkit::workspace` — all
    /// in the same `cargo test --test int8_tier_parity` process.
    /// `std::env::set_var`/`remove_var` are `unsafe fn` (edition 2024)
    /// precisely because a concurrent `std::env::var` on *any* key can
    /// observe a torn `environ` while another thread mutates it (a
    /// whole-process hazard, not a same-key one), and `cargo test` runs
    /// this file's tests as threads in one process by default.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard: acquires [`ENV_LOCK`], then snapshots and clears
    /// [`KERNEL_TIER_ENV`] so every guarded test starts from a known-clean
    /// environment regardless of the ambient shell, restoring the snapshot
    /// on drop (including on panic — `Drop` still runs while unwinding).
    ///
    /// Without it, a developer's own `OXIBONSAI_KERNEL_TIER=neon-dot` export
    /// would silently route every "f32 default" leg below — the dispatcher
    /// and driver entry points all read the variable — through the INT8
    /// tier as well, comparing int8 against int8 while still printing a
    /// `cos` / `ratio` line that looks like a real f32-vs-int8 comparison.
    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        prior: Option<String>,
    }

    impl EnvGuard {
        fn acquire() -> Self {
            let lock = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let prior = std::env::var(KERNEL_TIER_ENV).ok();
            // SAFETY: `lock` is held for the lifetime of the returned `Self`,
            // serializing every reader/writer of `KERNEL_TIER_ENV` in this
            // test binary (see `ENV_LOCK`'s doc comment).
            unsafe {
                std::env::remove_var(KERNEL_TIER_ENV);
            }
            Self { _lock: lock, prior }
        }

        /// Select `tier` through the environment exactly as a user would
        /// (`None` clears the variable).
        fn select(&self, tier: Option<Int8Tier>) {
            // SAFETY: `self._lock` is held (see `acquire`).
            unsafe {
                match tier {
                    Some(t) => std::env::set_var(KERNEL_TIER_ENV, t.name()),
                    None => std::env::remove_var(KERNEL_TIER_ENV),
                }
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: `self._lock` is held for the entire body of `drop`.
            unsafe {
                match &self.prior {
                    Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                    None => std::env::remove_var(KERNEL_TIER_ENV),
                }
            }
        }
    }

    /// Generators, comparisons and the timing harness the tests share.
    mod support;
    use support::*;

    /// The accuracy gate every acceptance assertion in this file uses.
    const COS_GATE: f32 = 0.999;

    // ─── native-format entry points ──────────────────────────────────────

    /// A native-format entry point, called as `(dispatcher, blocks, input,
    /// output, m, n_rows, k)` — a GEMV entry ignores `m` (it is always 1).
    type EntryFn<B> =
        fn(&KernelDispatcher, &[B], &[f32], &mut [f32], usize, usize, usize) -> KernelResult<()>;

    /// One named native-format entry point.
    struct Entry<B> {
        name: &'static str,
        batched: bool,
        run: EntryFn<B>,
    }

    /// Every public `TQ2_0_g128` GEMV/GEMM entry point that honours
    /// `OXIBONSAI_KERNEL_TIER`.
    fn ternary_entries() -> Vec<Entry<BlockTQ2_0_g128>> {
        vec![
            Entry {
                name: "gemv_adaptive_ternary",
                batched: false,
                run: |d, b, x, y, _m, n, k| {
                    oxibonsai_kernels::gemv_adaptive_ternary(d, b, x, y, n, k)
                },
            },
            Entry {
                name: "TernaryKernel::gemv_ternary_g128",
                batched: false,
                run: |d, b, x, y, _m, n, k| d.gemv_ternary_g128(b, x, y, n, k),
            },
            Entry {
                name: "parallel::gemv_ternary_g128_par",
                batched: false,
                run: |d, b, x, y, _m, n, k| {
                    oxibonsai_kernels::parallel::gemv_ternary_g128_par(d, b, x, y, n, k)
                },
            },
            Entry {
                name: "parallel_tiled::gemv_parallel_tiled_ternary",
                batched: false,
                run: |d, b, x, y, _m, n, k| {
                    oxibonsai_kernels::parallel_tiled::gemv_parallel_tiled_ternary(d, b, x, y, n, k)
                },
            },
            Entry {
                name: "gemm_adaptive_ternary",
                batched: true,
                run: |d, b, x, y, m, n, k| {
                    oxibonsai_kernels::gemm_adaptive_ternary(d, b, x, y, m, n, k)
                },
            },
            Entry {
                name: "TernaryKernel::gemm_ternary_g128",
                batched: true,
                run: |d, b, x, y, m, n, k| d.gemm_ternary_g128(b, x, y, m, n, k),
            },
            Entry {
                name: "parallel::gemm_ternary_g128_par",
                batched: true,
                run: |d, b, x, y, m, n, k| {
                    oxibonsai_kernels::parallel::gemm_ternary_g128_par(d, b, x, y, m, n, k)
                },
            },
        ]
    }

    /// Every public `Q1_0_g128` GEMV/GEMM entry point that honours
    /// `OXIBONSAI_KERNEL_TIER`.
    fn one_bit_entries() -> Vec<Entry<BlockQ1_0G128>> {
        vec![
            Entry {
                name: "gemv_adaptive",
                batched: false,
                run: |d, b, x, y, _m, n, k| oxibonsai_kernels::gemv_adaptive(d, b, x, y, n, k),
            },
            Entry {
                name: "OneBitKernel::gemv",
                batched: false,
                run: |d, b, x, y, _m, n, k| d.gemv(b, x, y, n, k),
            },
            Entry {
                name: "parallel::gemv_1bit_g128_par",
                batched: false,
                run: |d, b, x, y, _m, n, k| {
                    oxibonsai_kernels::parallel::gemv_1bit_g128_par(d, b, x, y, n, k)
                },
            },
            Entry {
                name: "parallel_tiled::gemv_parallel_tiled",
                batched: false,
                run: |d, b, x, y, _m, n, k| {
                    oxibonsai_kernels::parallel_tiled::gemv_parallel_tiled(d, b, x, y, n, k)
                },
            },
            Entry {
                name: "OneBitKernel::gemm",
                batched: true,
                run: |d, b, x, y, m, n, k| d.gemm(b, x, y, m, n, k),
            },
            Entry {
                name: "parallel::gemm_1bit_g128_par",
                batched: true,
                run: |d, b, x, y, m, n, k| {
                    oxibonsai_kernels::parallel::gemm_1bit_g128_par(d, b, x, y, m, n, k)
                },
            },
            Entry {
                name: "parallel_tiled::gemm_parallel_tiled",
                batched: true,
                run: |d, b, x, y, m, n, k| {
                    oxibonsai_kernels::parallel_tiled::gemm_parallel_tiled(d, b, x, y, m, n, k)
                },
            },
        ]
    }

    /// The direct INT8 kernel an entry point must reproduce bit for bit
    /// when `tier` is selected.
    fn direct_int8_ternary(
        tier: Int8Tier,
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n_rows];
        if m == 1 {
            gemv_two_bit_int8(tier, blocks, input, &mut out, n_rows, k).expect("direct int8 gemv");
        } else {
            gemm_two_bit_int8(tier, blocks, input, &mut out, m, n_rows, k)
                .expect("direct int8 gemm");
        }
        out
    }

    fn direct_int8_one_bit(
        tier: Int8Tier,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n_rows];
        if m == 1 {
            gemv_1bit_g128_int8(tier, blocks, input, &mut out, n_rows, k)
                .expect("direct int8 gemv");
        } else {
            gemm_1bit_g128_int8(tier, blocks, input, &mut out, m, n_rows, k)
                .expect("direct int8 gemm");
        }
        out
    }

    /// The per-(entry point, tier) evidence one weight matrix yields.
    struct EntryVerdict {
        worst_cos: f32,
        worst_entry: &'static str,
        entries: usize,
    }

    /// Drive every entry point in `entries` on `dispatcher` with each INT8
    /// tier selected through the environment and then with it cleared, and
    /// assert, per entry and per tier:
    ///
    /// - the selected run is bit-identical to calling the INT8 kernel for
    ///   that tier directly (`direct`) — the routing reached exactly it;
    /// - the selected run differs from the cleared run somewhere — the
    ///   variable really changed which kernel ran;
    /// - `cos(cleared, selected) >= COS_GATE` — the accuracy gate.
    ///
    /// Returns the worst cosine per tier.
    #[allow(clippy::too_many_arguments)]
    fn check_entries<B>(
        guard: &EnvGuard,
        dispatcher: &KernelDispatcher,
        entries: &[Entry<B>],
        blocks: &[B],
        n_rows: usize,
        k: usize,
        m_batched: usize,
        seed: u32,
        direct: impl Fn(Int8Tier, &[B], &[f32], usize, usize, usize) -> Vec<f32>,
        what: &str,
    ) -> Vec<(Int8Tier, EntryVerdict)> {
        let gemv_input = activations(k, seed);
        let gemm_input = activations(m_batched * k, seed ^ 0x5A5A);
        let mut verdicts = Vec::new();
        for tier in supported_tiers() {
            let mut verdict = EntryVerdict {
                worst_cos: 1.0,
                worst_entry: "",
                entries: 0,
            };
            let expect_gemv = direct(tier, blocks, &gemv_input, 1, n_rows, k);
            let expect_gemm = direct(tier, blocks, &gemm_input, m_batched, n_rows, k);
            for entry in entries {
                let (m, input, expect) = if entry.batched {
                    (m_batched, &gemm_input, &expect_gemm)
                } else {
                    (1, &gemv_input, &expect_gemv)
                };
                guard.select(Some(tier));
                let mut selected = vec![0.0f32; m * n_rows];
                (entry.run)(dispatcher, blocks, input, &mut selected, m, n_rows, k)
                    .unwrap_or_else(|e| panic!("{what} {} {tier}: {e}", entry.name));
                guard.select(None);
                let mut cleared = vec![0.0f32; m * n_rows];
                (entry.run)(dispatcher, blocks, input, &mut cleared, m, n_rows, k)
                    .unwrap_or_else(|e| panic!("{what} {} f32: {e}", entry.name));

                assert_bits_eq(
                    expect,
                    &selected,
                    &format!(
                        "{what} {} with {KERNEL_TIER_ENV}={tier} vs the direct {tier} kernel",
                        entry.name
                    ),
                );
                assert!(
                    bits_differ(&cleared, &selected),
                    "{what} {}: {KERNEL_TIER_ENV}={tier} produced the f32 bits — the \
                     variable did not change which kernel ran",
                    entry.name
                );
                let cos = cosine(&cleared, &selected);
                assert!(
                    cos >= COS_GATE,
                    "{what} {} {tier} (m={m}): cos {cos} < {COS_GATE}",
                    entry.name
                );
                if cos < verdict.worst_cos {
                    verdict.worst_cos = cos;
                    verdict.worst_entry = entry.name;
                }
                verdict.entries += 1;
            }
            verdicts.push((tier, verdict));
        }
        verdicts
    }

    // ─── 1. cross-tier bit-identity ──────────────────────────────────────

    #[test]
    fn every_int8_tier_produces_bit_identical_output() {
        let (n_rows, k) = (37usize, 3 * QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x11);
        let input = activations(k, 0x12);
        let mut reference = vec![0.0f32; n_rows];
        gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut reference, n_rows, k)
            .expect("scalar int8 gemv");
        for tier in supported_tiers() {
            let mut got = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut got, n_rows, k).expect("int8 gemv");
            assert_bits_eq(&reference, &got, &format!("{tier} vs the scalar int8 tier"));
        }
    }

    #[test]
    fn every_int8_tier_produces_bit_identical_one_bit_output() {
        let (n_rows, k) = (23usize, 2 * QK1_0_G128);
        let blocks = q1_blocks(n_rows * (k / QK1_0_G128), 0x21);
        let input = activations(k, 0x22);
        let mut reference = vec![0.0f32; n_rows];
        gemv_1bit_g128_int8(Int8Tier::Scalar, &blocks, &input, &mut reference, n_rows, k)
            .expect("scalar int8 gemv");
        for tier in supported_tiers() {
            let mut got = vec![0.0f32; n_rows];
            gemv_1bit_g128_int8(tier, &blocks, &input, &mut got, n_rows, k).expect("int8 gemv");
            assert_bits_eq(&reference, &got, &format!("{tier} on the 1-bit format"));
        }
    }

    /// Every tier's GEMM — including [`Int8Tier::NeonI8mm`]'s `SMMLA` path —
    /// must agree with the GEMV applied per batch row, odd-`m`/odd-`n` tails
    /// included.
    #[test]
    fn int8_gemm_matches_the_int8_gemv_per_batch_row_on_every_tier() {
        let (m, n_rows, k) = (5usize, 7usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x31);
        let input = activations(m * k, 0x32);
        for tier in supported_tiers() {
            let mut expect = vec![0.0f32; m * n_rows];
            for mi in 0..m {
                gemv_two_bit_int8(
                    tier,
                    &blocks,
                    &input[mi * k..(mi + 1) * k],
                    &mut expect[mi * n_rows..(mi + 1) * n_rows],
                    n_rows,
                    k,
                )
                .expect("int8 gemv");
            }
            let mut got = vec![0.0f32; m * n_rows];
            gemm_two_bit_int8(tier, &blocks, &input, &mut got, m, n_rows, k).expect("int8 gemm");
            assert_bits_eq(&expect, &got, &format!("{tier} gemm vs the gemv sweep"));
        }
    }

    /// `n_rows = 300` crosses `dispatch_int8::INT8_PAR_MIN_ROWS`, so the
    /// Rayon fan-out (the `par_chunks_mut` split and its block-slice
    /// indexing) runs; the three chunk sizes below are deliberately uneven,
    /// so the comparison exercises ragged chunk boundaries.
    ///
    /// Integer accumulation is exact and a Rayon split only changes *which*
    /// thread computes a given output row, never in what order a row's own
    /// terms are summed — so the parallel 300-row call must be bit-for-bit
    /// identical (`to_bits()`) to the same rows computed as sequential
    /// sub-300-row calls on the matching block slices, on every tier.
    #[test]
    fn gemv_two_bit_int8_rayon_split_matches_sequential_sub_calls() {
        let (n_rows, k) = (300usize, 2 * QK_TQ2_0_G128);
        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks = tq2_blocks(n_rows * blocks_per_row, 0xC0DE_0001);
        let input = activations(k, 0xFACE_0001);

        for tier in supported_tiers() {
            let mut whole = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut whole, n_rows, k)
                .expect("whole 300-row gemv");

            let chunk_sizes = [113usize, 90, 97];
            assert_eq!(chunk_sizes.iter().sum::<usize>(), n_rows);
            let mut sequential = vec![0.0f32; n_rows];
            let mut row_start = 0usize;
            for &rows in &chunk_sizes {
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                gemv_two_bit_int8(
                    tier,
                    &blocks[block_start..block_end],
                    &input,
                    &mut sequential[row_start..row_start + rows],
                    rows,
                    k,
                )
                .expect("sequential sub-call gemv");
                row_start += rows;
            }
            assert_bits_eq(
                &whole,
                &sequential,
                &format!("{tier}: single 300-row call vs the sequential sub-call split"),
            );
        }
    }

    // ─── 2. accuracy vs the f32 tier ─────────────────────────────────────

    #[test]
    fn int8_tier_matches_the_f32_tier_on_synthetic_ternary_weights() {
        let (n_rows, k) = (64usize, 8 * QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x41);
        let input = activations(k, 0x42);
        let mut f32_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut f32_out, n_rows, k)
            .expect("f32 reference gemv");
        for tier in supported_tiers() {
            let mut int8_out = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");
            let cos = cosine(&f32_out, &int8_out);
            assert!(
                cos >= COS_GATE,
                "{tier} vs f32 tier on TQ2_0_g128: cos {cos} < {COS_GATE}"
            );
        }
    }

    #[test]
    fn int8_tier_matches_the_f32_tier_on_synthetic_pq2_0_weights() {
        let (n_rows, k) = (64usize, 8 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x51);
        let input = activations(k, 0x52);
        let mut f32_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::dequant_prism::gemv_pq2_0(&blocks, &input, &mut f32_out, n_rows, k)
            .expect("f32 reference gemv");
        for tier in supported_tiers() {
            let mut int8_out = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");
            let cos = cosine(&f32_out, &int8_out);
            assert!(
                cos >= COS_GATE,
                "{tier} vs f32 tier on PQ2_0: cos {cos} < {COS_GATE}"
            );
        }
    }

    #[test]
    fn int8_tier_matches_the_f32_tier_on_synthetic_one_bit_weights() {
        let (n_rows, k) = (64usize, 8 * QK1_0_G128);
        let blocks = q1_blocks(n_rows * (k / QK1_0_G128), 0x61);
        let input = activations(k, 0x62);
        let mut f32_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv::gemv_1bit_g128(&blocks, &input, &mut f32_out, n_rows, k)
            .expect("f32 reference gemv");
        for tier in supported_tiers() {
            let mut int8_out = vec![0.0f32; n_rows];
            gemv_1bit_g128_int8(tier, &blocks, &input, &mut int8_out, n_rows, k)
                .expect("int8 gemv");
            let cos = cosine(&f32_out, &int8_out);
            assert!(
                cos >= COS_GATE,
                "{tier} vs f32 tier on Q1_0_g128: cos {cos} < {COS_GATE}"
            );
        }
    }

    /// The native entry points on synthetic weights, every tier, every
    /// entry — with `n_rows = 300` so the adaptive drivers' parallel
    /// strategies and the INT8 kernels' Rayon splits both run.
    #[test]
    fn native_entry_points_route_to_the_selected_int8_tier_on_synthetic_weights() {
        let guard = EnvGuard::acquire();
        let dispatcher = cpu_dispatcher();
        let (n_rows, k, m) = (300usize, 2 * QK_TQ2_0_G128, 5usize);

        let tq2 = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x7A01);
        let verdicts = check_entries(
            &guard,
            &dispatcher,
            &ternary_entries(),
            &tq2,
            n_rows,
            k,
            m,
            0x7A02,
            direct_int8_ternary,
            "synthetic TQ2_0_g128",
        );
        for (tier, v) in &verdicts {
            println!(
                "synthetic TQ2_0_g128 [{k}x{n_rows}] {tier} via {} native entry points: \
                 min cos = {:.6} ({})",
                v.entries, v.worst_cos, v.worst_entry
            );
        }

        let q1 = q1_blocks(n_rows * (k / QK1_0_G128), 0x7A03);
        let verdicts = check_entries(
            &guard,
            &dispatcher,
            &one_bit_entries(),
            &q1,
            n_rows,
            k,
            m,
            0x7A04,
            direct_int8_one_bit,
            "synthetic Q1_0_g128",
        );
        for (tier, v) in &verdicts {
            println!(
                "synthetic Q1_0_g128 [{k}x{n_rows}] {tier} via {} native entry points: \
                 min cos = {:.6} ({})",
                v.entries, v.worst_cos, v.worst_entry
            );
        }
    }

    // ─── 2b. accuracy on REAL model weights ──────────────────────────────

    /// Rows of a real weight matrix each **debug-build** acceptance run
    /// measures.
    ///
    /// Bounded deliberately: a gate's plain `cargo test` leg runs this file
    /// in a **debug** build, and a full 12288-row 8B matrix there would be
    /// minutes of unoptimized arithmetic per tier. 192 rows over the
    /// matrix's full `k` is several hundred thousand real MACs per tier —
    /// ample for a cosine gate, and it exercises the genuine weight
    /// distribution. Release builds run the whole matrix.
    const REAL_ROWS: usize = 192;

    /// Weight rows a real-matrix leg uses in this build.
    fn real_rows(total_rows: usize) -> usize {
        if cfg!(debug_assertions) {
            REAL_ROWS.min(total_rows)
        } else {
            total_rows
        }
    }

    /// The first tensor of `want` type in `file`, as its name plus its
    /// `[k, n_rows]` GGUF shape.
    fn first_tensor_of(gguf: &GgufFile<'_>, want: GgufTensorType) -> Option<(String, Vec<u64>)> {
        let mut names: Vec<&str> = gguf
            .tensors
            .iter()
            .filter(|(_, info)| info.tensor_type == want && info.shape.len() == 2)
            .map(|(name, _)| name.as_str())
            .collect();
        names.sort_unstable();
        let name = names.first()?;
        let info = gguf.tensors.get(name)?;
        Some((name.to_string(), info.shape.clone()))
    }

    /// Run the kernel-level cosine gate over a real `TQ2_0_g128` file.
    fn check_real_ternary_model(file_name: &str) {
        let _guard = EnvGuard::acquire();
        let Some(path) = oxibonsai_testkit::workspace::find_model(file_name) else {
            eprintln!(
                "skipping real-model int8 parity for {file_name}: not present under \
                 {:?} (set OXIBONSAI_MODELS_DIR to a checkout that has it)",
                oxibonsai_testkit::workspace::models_dir()
            );
            return;
        };
        let mmap = mmap_gguf_file(&path).expect("mmap gguf");
        let gguf = GgufFile::parse(&mmap).expect("parse gguf");
        let Some((name, shape)) = first_tensor_of(&gguf, GgufTensorType::TQ2_0_g128) else {
            eprintln!("skipping {file_name}: no 2-D TQ2_0_g128 tensor");
            return;
        };
        let k = shape[0] as usize;
        let total_rows = shape[1] as usize;
        assert!(
            k.is_multiple_of(QK_TQ2_0_G128),
            "{file_name}: {name} has k={k}, not a multiple of 128"
        );
        let n_rows = REAL_ROWS.min(total_rows);
        let blocks_per_row = k / QK_TQ2_0_G128;
        let bytes = gguf.tensor_data(&name).expect("tensor data");
        let all = BlockTQ2_0_g128::slice_from_bytes(bytes).expect("block slice");
        let blocks = &all[..n_rows * blocks_per_row];

        let input = activations(k, 0x7E51);
        let mut f32_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(blocks, &input, &mut f32_out, n_rows, k)
            .expect("f32 reference gemv");
        for tier in supported_tiers() {
            let mut int8_out = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");
            let cos = cosine(&f32_out, &int8_out);
            println!("{file_name} {name} [{k}x{n_rows}] {tier}: cos = {cos:.6}");
            assert!(
                cos >= COS_GATE,
                "{file_name} {name} {tier}: cos {cos} < {COS_GATE}"
            );
        }
    }

    #[test]
    fn int8_tier_matches_the_f32_tier_on_the_real_1_7b_weights() {
        check_real_ternary_model("Ternary-Bonsai-1.7B.gguf");
    }

    #[test]
    fn int8_tier_matches_the_f32_tier_on_the_real_8b_weights() {
        check_real_ternary_model("Ternary-Bonsai-8B.gguf");
    }

    #[test]
    fn int8_tier_matches_the_f32_tier_on_the_real_27b_pq2_0_weights() {
        let _guard = EnvGuard::acquire();
        let file_name = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
        let Some(path) = oxibonsai_testkit::workspace::find_model(file_name) else {
            eprintln!(
                "skipping real-model int8 parity for {file_name}: not present under {:?}",
                oxibonsai_testkit::workspace::models_dir()
            );
            return;
        };
        let mmap = mmap_gguf_file(&path).expect("mmap gguf");
        let gguf = GgufFile::parse(&mmap).expect("parse gguf");
        let Some((name, shape)) = first_tensor_of(&gguf, GgufTensorType::PQ2_0) else {
            eprintln!("skipping {file_name}: no 2-D PQ2_0 tensor");
            return;
        };
        let k = shape[0] as usize;
        let total_rows = shape[1] as usize;
        assert!(k.is_multiple_of(QK_PQ2_0), "{file_name}: {name} k={k}");
        let n_rows = REAL_ROWS.min(total_rows);
        let blocks_per_row = k / QK_PQ2_0;
        let bytes = gguf.tensor_data(&name).expect("tensor data");
        let all = BlockPQ2_0::slice_from_bytes(bytes).expect("block slice");
        let blocks = &all[..n_rows * blocks_per_row];

        let input = activations(k, 0x27B0);
        let mut f32_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::dequant_prism::gemv_pq2_0(blocks, &input, &mut f32_out, n_rows, k)
            .expect("f32 reference gemv");
        for tier in supported_tiers() {
            let mut int8_out = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");
            let cos = cosine(&f32_out, &int8_out);
            println!("{file_name} {name} [{k}x{n_rows}] {tier}: cos = {cos:.6}");
            assert!(
                cos >= COS_GATE,
                "{file_name} {name} {tier}: cos {cos} < {COS_GATE}"
            );
        }
    }

    /// The real-weight tensors each native-format entry-point leg covers:
    /// one attention projection, and both FFN shapes (`k = hidden` and
    /// `k = intermediate`).
    const REAL_TENSORS: [&str; 3] = [
        "blk.0.attn_q.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];

    /// Batch size of the GEMM legs on real weights.
    const REAL_GEMM_M: usize = 4;

    /// The native-format entry-point legs on one real GGUF: every tensor in
    /// [`REAL_TENSORS`], every entry point, every supported INT8 tier,
    /// selected through `OXIBONSAI_KERNEL_TIER` on a CPU-tier dispatcher. A
    /// missing model is a self-skip recorded as `legacy-models` / `executed:
    /// false`; `executed: true` (timed from the model's mapping) is written
    /// only after every tensor, entry point and tier above has passed.
    fn check_real_native_entry_points(file_name: &str, format: GgufTensorType, test_name: &str) {
        use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

        let guard = EnvGuard::acquire();
        let Some(path) = oxibonsai_testkit::workspace::find_model(file_name) else {
            eprintln!(
                "skipping native entry-point int8 parity for {file_name}: not present under \
                 {:?} (set OXIBONSAI_MODELS_DIR to a checkout that has it)",
                oxibonsai_testkit::workspace::models_dir()
            );
            record_skipped(Capability::LegacyModels, test_name);
            return;
        };
        let gate_start = std::time::Instant::now();
        let mmap = mmap_gguf_file(&path).expect("mmap gguf");
        let gguf = GgufFile::parse(&mmap).expect("parse gguf");
        let dispatcher = cpu_dispatcher();
        for (ti, tensor) in REAL_TENSORS.iter().enumerate() {
            let info = gguf
                .tensors
                .get(tensor)
                .unwrap_or_else(|| panic!("{file_name}: no {tensor}"));
            assert_eq!(
                info.tensor_type, format,
                "{file_name}: {tensor} is {:?}, not {format:?}",
                info.tensor_type
            );
            let k = info.shape[0] as usize;
            let n_rows = real_rows(info.shape[1] as usize);
            let bytes = gguf.tensor_data(tensor).expect("tensor data");
            let seed = 0x5EA1_0000 + ti as u32;
            let verdicts = match format {
                GgufTensorType::TQ2_0_g128 => {
                    let all = BlockTQ2_0_g128::slice_from_bytes(bytes).expect("block slice");
                    let blocks = &all[..n_rows * (k / QK_TQ2_0_G128)];
                    check_entries(
                        &guard,
                        &dispatcher,
                        &ternary_entries(),
                        blocks,
                        n_rows,
                        k,
                        REAL_GEMM_M,
                        seed,
                        direct_int8_ternary,
                        &format!("{file_name} {tensor}"),
                    )
                }
                GgufTensorType::Q1_0_g128 => {
                    let all = BlockQ1_0G128::slice_from_bytes(bytes).expect("block slice");
                    let blocks = &all[..n_rows * (k / QK1_0_G128)];
                    check_entries(
                        &guard,
                        &dispatcher,
                        &one_bit_entries(),
                        blocks,
                        n_rows,
                        k,
                        REAL_GEMM_M,
                        seed,
                        direct_int8_one_bit,
                        &format!("{file_name} {tensor}"),
                    )
                }
                other => panic!("no native INT8 entry points for {other:?}"),
            };
            for (tier, v) in &verdicts {
                println!(
                    "{file_name} {tensor} [{k}x{n_rows}] {tier} via {} native entry points \
                     (gemv + gemm m={REAL_GEMM_M}): min cos = {:.6} ({})",
                    v.entries, v.worst_cos, v.worst_entry
                );
            }
        }
        record_executed_timed(Capability::LegacyModels, test_name, gate_start.elapsed());
    }

    #[test]
    fn native_entry_points_match_the_f32_tier_on_the_real_1_7b_ternary_weights() {
        check_real_native_entry_points(
            "Ternary-Bonsai-1.7B.gguf",
            GgufTensorType::TQ2_0_g128,
            "oxibonsai-kernels::int8_tier_parity::\
             native_entry_points_match_the_f32_tier_on_the_real_1_7b_ternary_weights",
        );
    }

    #[test]
    fn native_entry_points_match_the_f32_tier_on_the_real_8b_ternary_weights() {
        check_real_native_entry_points(
            "Ternary-Bonsai-8B.gguf",
            GgufTensorType::TQ2_0_g128,
            "oxibonsai-kernels::int8_tier_parity::\
             native_entry_points_match_the_f32_tier_on_the_real_8b_ternary_weights",
        );
    }

    #[test]
    fn native_entry_points_match_the_f32_tier_on_the_real_bonsai_8b_one_bit_weights() {
        check_real_native_entry_points(
            "Bonsai-8B.gguf",
            GgufTensorType::Q1_0_g128,
            "oxibonsai-kernels::int8_tier_parity::\
             native_entry_points_match_the_f32_tier_on_the_real_bonsai_8b_one_bit_weights",
        );
    }

    // ─── 3. the HARD CONSTRAINT ──────────────────────────────────────────

    /// The tier must never be reachable without being named.
    ///
    /// Independent guards: the environment selector returns `None` with a
    /// clean environment; no `KernelTier` the auto-detector can produce is
    /// an INT8 tier (they are different types, so a "silent upgrade of
    /// `KernelTier::Neon`" cannot even be written); no dispatcher reports a
    /// native INT8 tier; and every native-format entry point on the scalar
    /// `Reference` tier reproduces the f32 reference kernels **bit for
    /// bit** — i.e. an unset variable leaves the native paths exactly where
    /// they were.
    #[test]
    fn the_int8_tier_is_never_selected_implicitly() {
        let _guard = EnvGuard::acquire();
        assert_eq!(
            Int8Tier::from_env(),
            None,
            "OXIBONSAI_KERNEL_TIER unset must select no INT8 tier — making this \
             tier a default would change results bit-for-bit and break the \
             CPU-vs-Metal determinism guard"
        );

        let auto = KernelDispatcher::auto_detect();
        let tier = auto.tier();
        println!("auto-detected KernelTier = {tier}");
        assert!(
            !format!("{tier}").contains("int8")
                && !format!("{tier}").contains("dot")
                && !format!("{tier}").contains("vnni"),
            "auto-detection must never land on an INT8 tier, got {tier}"
        );
        assert_eq!(auto.native_int8_tier(), None);
        assert_eq!(cpu_dispatcher().native_int8_tier(), None);
        let reference = KernelDispatcher::with_tier(KernelTier::Reference);
        assert_eq!(reference.native_int8_tier(), None);

        let (n_rows, k, m) = (300usize, 2 * QK_TQ2_0_G128, 3usize);
        let gemv_input = activations(k, 0x91A1);
        let gemm_input = activations(m * k, 0x91A2);

        let tq2 = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x91A3);
        let mut gemv_ref = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(
            &tq2,
            &gemv_input,
            &mut gemv_ref,
            n_rows,
            k,
        )
        .expect("f32 ternary gemv reference");
        let mut gemm_ref = vec![0.0f32; m * n_rows];
        oxibonsai_kernels::gemm_ternary::gemm_tq2_0_g128(
            &tq2,
            &gemm_input,
            &mut gemm_ref,
            m,
            n_rows,
            k,
        )
        .expect("f32 ternary gemm reference");
        for entry in ternary_entries() {
            let (mm, input, expect) = if entry.batched {
                (m, &gemm_input, &gemm_ref)
            } else {
                (1, &gemv_input, &gemv_ref)
            };
            let mut got = vec![0.0f32; mm * n_rows];
            (entry.run)(&reference, &tq2, input, &mut got, mm, n_rows, k).expect(entry.name);
            assert_bits_eq(
                expect,
                &got,
                &format!(
                    "{} on the Reference tier with the variable unset",
                    entry.name
                ),
            );
        }

        let q1 = q1_blocks(n_rows * (k / QK1_0_G128), 0x91A4);
        let mut gemv_ref = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv::gemv_1bit_g128(&q1, &gemv_input, &mut gemv_ref, n_rows, k)
            .expect("f32 1-bit gemv reference");
        let mut gemm_ref = vec![0.0f32; m * n_rows];
        oxibonsai_kernels::gemm::gemm_1bit_g128(&q1, &gemm_input, &mut gemm_ref, m, n_rows, k)
            .expect("f32 1-bit gemm reference");
        for entry in one_bit_entries() {
            let (mm, input, expect) = if entry.batched {
                (m, &gemm_input, &gemm_ref)
            } else {
                (1, &gemv_input, &gemv_ref)
            };
            let mut got = vec![0.0f32; mm * n_rows];
            (entry.run)(&reference, &q1, input, &mut got, mm, n_rows, k).expect(entry.name);
            assert_bits_eq(
                expect,
                &got,
                &format!(
                    "{} on the Reference tier with the variable unset",
                    entry.name
                ),
            );
        }
    }

    /// `KernelTier::Neon` (the host's best f32 CPU tier) must still produce
    /// exactly what it produced before K-14, on the Prism *and* the native
    /// formats: agree with the scalar reference to float noise, and **not**
    /// be the int8 result.
    #[test]
    fn the_f32_neon_tier_is_untouched_by_the_int8_tier() {
        let _guard = EnvGuard::acquire();
        let (n_rows, k) = (16usize, 2 * QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x81);
        let input = activations(k, 0x82);
        let dispatcher = cpu_dispatcher();

        let mut scalar_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(
            &blocks,
            &input,
            &mut scalar_out,
            n_rows,
            k,
        )
        .expect("scalar f32 gemv");

        let mut tier_out = vec![0.0f32; n_rows];
        TernaryKernel::gemv_ternary_g128(&dispatcher, &blocks, &input, &mut tier_out, n_rows, k)
            .expect("dispatcher f32 gemv");
        for (a, b) in scalar_out.iter().zip(tier_out.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * a.abs().max(1.0),
                "the f32 tier drifted from the scalar reference: {a} vs {b}"
            );
        }

        // And it is *not* the int8 result: int8 activation quantization is
        // lossy, so the two must differ somewhere (otherwise the tier is
        // silently the same code path).
        let mut int8_out = vec![0.0f32; n_rows];
        gemv_two_bit_int8(
            Int8Tier::best_available(),
            &blocks,
            &input,
            &mut int8_out,
            n_rows,
            k,
        )
        .expect("int8 gemv");
        assert!(
            bits_differ(&tier_out, &int8_out),
            "the int8 tier produced bit-identical output to the f32 tier — it is \
             not actually quantizing, so this suite proves nothing"
        );

        // Native formats through every public entry point, f32 CPU tier,
        // variable unset: within float noise of the scalar reference, never
        // the int8 bits.
        let (n_rows, k, m) = (300usize, 3 * QK_TQ2_0_G128, 3usize);
        let gemv_input = activations(k, 0x83);
        let gemm_input = activations(m * k, 0x84);
        let tq2 = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x85);
        let q1 = q1_blocks(n_rows * (k / QK1_0_G128), 0x86);
        let tier = Int8Tier::best_available();

        let mut ref_gemv = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(
            &tq2,
            &gemv_input,
            &mut ref_gemv,
            n_rows,
            k,
        )
        .expect("scalar ternary gemv");
        let mut ref_gemm = vec![0.0f32; m * n_rows];
        oxibonsai_kernels::gemm_ternary::gemm_tq2_0_g128(
            &tq2,
            &gemm_input,
            &mut ref_gemm,
            m,
            n_rows,
            k,
        )
        .expect("scalar ternary gemm");
        for entry in ternary_entries() {
            let (mm, input, reference) = if entry.batched {
                (m, &gemm_input, &ref_gemm)
            } else {
                (1, &gemv_input, &ref_gemv)
            };
            let mut got = vec![0.0f32; mm * n_rows];
            (entry.run)(&dispatcher, &tq2, input, &mut got, mm, n_rows, k).expect(entry.name);
            for (a, b) in reference.iter().zip(got.iter()) {
                assert!(
                    (a - b).abs() <= 1e-3 * a.abs().max(1.0),
                    "{}: the f32 tier drifted from the scalar reference: {a} vs {b}",
                    entry.name
                );
            }
            let int8 = direct_int8_ternary(tier, &tq2, input, mm, n_rows, k);
            assert!(
                bits_differ(&got, &int8),
                "{}: the f32 path produced the int8 bits with the variable unset",
                entry.name
            );
        }

        let mut ref_gemv = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv::gemv_1bit_g128(&q1, &gemv_input, &mut ref_gemv, n_rows, k)
            .expect("scalar 1-bit gemv");
        let mut ref_gemm = vec![0.0f32; m * n_rows];
        oxibonsai_kernels::gemm::gemm_1bit_g128(&q1, &gemm_input, &mut ref_gemm, m, n_rows, k)
            .expect("scalar 1-bit gemm");
        for entry in one_bit_entries() {
            let (mm, input, reference) = if entry.batched {
                (m, &gemm_input, &ref_gemm)
            } else {
                (1, &gemv_input, &ref_gemv)
            };
            let mut got = vec![0.0f32; mm * n_rows];
            (entry.run)(&dispatcher, &q1, input, &mut got, mm, n_rows, k).expect(entry.name);
            for (a, b) in reference.iter().zip(got.iter()) {
                assert!(
                    (a - b).abs() <= 1e-3 * a.abs().max(1.0),
                    "{}: the f32 tier drifted from the scalar reference: {a} vs {b}",
                    entry.name
                );
            }
            let int8 = direct_int8_one_bit(tier, &q1, input, mm, n_rows, k);
            assert!(
                bits_differ(&got, &int8),
                "{}: the f32 path produced the int8 bits with the variable unset",
                entry.name
            );
        }
    }

    /// A `KernelTier::Gpu` dispatcher is never diverted, even with the
    /// variable set: its native GEMV/GEMM (and their CPU fallbacks) belong
    /// to the GPU path, which the CPU-vs-Metal determinism guard pins.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_gpu_tier_dispatcher_is_never_diverted() {
        let guard = EnvGuard::acquire();
        let gpu = KernelDispatcher::with_tier(KernelTier::Gpu);
        assert_eq!(gpu.tier(), KernelTier::Gpu);
        let tier = Int8Tier::best_available();

        let (n_rows, k, m) = (64usize, 2 * QK_TQ2_0_G128, 3usize);
        let gemv_input = activations(k, 0x6901);
        let gemm_input = activations(m * k, 0x6902);
        let tq2 = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x6903);
        let q1 = q1_blocks(n_rows * (k / QK1_0_G128), 0x6904);

        let run_all = |label: &str| -> Vec<Vec<f32>> {
            let mut outs = Vec::new();
            for entry in ternary_entries() {
                let (mm, input) = if entry.batched {
                    (m, &gemm_input)
                } else {
                    (1, &gemv_input)
                };
                let mut out = vec![0.0f32; mm * n_rows];
                (entry.run)(&gpu, &tq2, input, &mut out, mm, n_rows, k)
                    .unwrap_or_else(|e| panic!("{label} {}: {e}", entry.name));
                outs.push(out);
            }
            for entry in one_bit_entries() {
                let (mm, input) = if entry.batched {
                    (m, &gemm_input)
                } else {
                    (1, &gemv_input)
                };
                let mut out = vec![0.0f32; mm * n_rows];
                (entry.run)(&gpu, &q1, input, &mut out, mm, n_rows, k)
                    .unwrap_or_else(|e| panic!("{label} {}: {e}", entry.name));
                outs.push(out);
            }
            outs
        };

        let cleared = run_all("cleared");
        guard.select(Some(tier));
        assert_eq!(
            gpu.native_int8_tier(),
            None,
            "a Gpu-tier dispatcher must never report a native INT8 tier"
        );
        assert_eq!(
            cpu_dispatcher().native_int8_tier(),
            Some(tier),
            "while a CPU-tier dispatcher does, with the same environment"
        );
        let selected = run_all("selected");
        guard.select(None);
        for (i, (a, b)) in cleared.iter().zip(selected.iter()).enumerate() {
            assert_bits_eq(
                a,
                b,
                &format!("Gpu-tier entry point #{i} with {KERNEL_TIER_ENV}={tier}"),
            );
        }
    }

    // ─── 4. throughput ───────────────────────────────────────────────────

    /// Measure the GEMV inner loop: NEON f32 (today's tier) vs the fastest
    /// INT8 tier, single-threaded on both sides.
    ///
    /// `n_rows` stays under the INT8 GEMV's Rayon threshold so this compares
    /// *kernels*, not thread counts. The assertion is opt-in
    /// (`OXIBONSAI_INT8_BENCH=1`) because a gate builds and runs under
    /// concurrent compilation, where a wall-clock ratio is not a sound
    /// pass/fail signal; the measurement itself always prints.
    #[test]
    fn int8_gemv_is_faster_than_the_f32_gemv() {
        let _guard = EnvGuard::acquire();
        let (n_rows, k) = (200usize, 64 * QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x91);
        let input = activations(k, 0x92);
        let tier = Int8Tier::best_available();
        let iters = if cfg!(debug_assertions) { 1 } else { 20 };

        let mut f32_out = vec![0.0f32; n_rows];
        let mut int8_out = vec![0.0f32; n_rows];

        // Warm up both paths (page-in, branch predictors).
        f32_gemv(&blocks, &input, &mut f32_out, n_rows, k);
        gemv_two_bit_int8(tier, &blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");

        let t0 = Instant::now();
        for _ in 0..iters {
            f32_gemv(&blocks, &input, &mut f32_out, n_rows, k);
        }
        let f32_time = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..iters {
            gemv_two_bit_int8(tier, &blocks, &input, &mut int8_out, n_rows, k).expect("int8 gemv");
        }
        let int8_time = t1.elapsed();

        let speedup = f32_time.as_secs_f64() / int8_time.as_secs_f64().max(1e-12);
        println!(
            "int8 GEMV inner loop [{k}x{n_rows}] {tier}: f32 {f32_time:?} -> int8 \
             {int8_time:?} = {speedup:.2}x ({iters} iters, \
             debug_assertions={})",
            cfg!(debug_assertions)
        );
        // Accuracy is still asserted here, unconditionally — the number
        // above is only meaningful if the fast path is also correct.
        let cos = cosine(&f32_out, &int8_out);
        assert!(cos >= COS_GATE, "int8 bench output diverged: cos {cos}");

        if bench_asserts() {
            assert!(
                speedup >= 2.5,
                "INT8 GEMV must be >= 2.5x the f32 GEMV (got {speedup:.2}x)"
            );
        }
    }

    /// The INT8 tiers the throughput legs measure: `neon-dot` — the tier a
    /// user is most likely to name, whose GEMM is the row kernel with the
    /// small-`m` fan-out — and the fastest this host supports (`neon-i8mm`
    /// on an i8mm host, whose GEMM is the `SMMLA` kernel), deduplicated.
    fn bench_tiers() -> Vec<Int8Tier> {
        let mut tiers = Vec::new();
        #[cfg(target_arch = "aarch64")]
        if Int8Tier::NeonDot.is_supported() {
            tiers.push(Int8Tier::NeonDot);
        }
        let best = Int8Tier::best_available();
        if !tiers.contains(&best) {
            tiers.push(best);
        }
        tiers
    }

    /// Re-measure a throughput point that misses its bound, up to three
    /// attempts in all, printing every attempt (release builds only — a
    /// debug build's numbers mean nothing). A point fails only if the miss
    /// reproduces on every attempt: a sibling process's load spike can sink
    /// one interleaved measurement on this shared machine, but not three in
    /// a row by chance. Returns the attempt with the best ratio.
    fn measure_point(
        bound: f64,
        what: &str,
        mut measure: impl FnMut() -> (Paired, f32),
    ) -> (Paired, f32) {
        let attempts = if cfg!(debug_assertions) { 1 } else { 3 };
        let (mut best, mut best_cos) = measure();
        let mut attempt = 1;
        while best.ratio() <= bound && attempt < attempts {
            println!(
                "  {what}: {:.2}x is not above {bound:.2}x on attempt {attempt} \
                 (load average {}); re-measuring",
                best.ratio(),
                load_average()
            );
            let (again, cos) = measure();
            attempt += 1;
            if again.ratio() > best.ratio() {
                (best, best_cos) = (again, cos);
            }
        }
        (best, best_cos)
    }

    /// `gemm_two_bit_int8` is Rayon-parallel, so *opting in* to the INT8
    /// tier at a real prefill batch size (27B `ffn_up`, M=64) must not be
    /// slower than the tiled + Rayon f32 default the Prism dispatcher runs —
    /// on `neon-dot` and on the fastest tier alike.
    #[test]
    fn int8_gemm_at_m64_is_not_slower_than_the_f32_default() {
        use oxibonsai_kernels::traits::PrismKernel;

        let _guard = EnvGuard::acquire();
        let (m, n_rows, k) = (64usize, 17408usize, 5120usize);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0xC0DE_0064);
        let input = activations(m * k, 0xFACE_0064);
        let dispatcher = KernelDispatcher::auto_detect();
        let rounds = if cfg!(debug_assertions) { 1 } else { 9 };

        let mut f32_out = vec![0.0f32; m * n_rows];
        let mut int8_out = vec![0.0f32; m * n_rows];
        // Warm the f32 path (page-in, branch predictors, Rayon pool).
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut f32_out, m, n_rows, k)
            .expect("f32 default gemm warmup");
        for tier in bench_tiers() {
            gemm_two_bit_int8(tier, &blocks, &input, &mut int8_out, m, n_rows, k)
                .expect("int8 gemm warmup");
            let what = format!("int8 GEMM at M={m} ffn_up[{k}x{n_rows}] {tier}");
            // One interleaved measurement, no re-measurement: this bound has
            // always been asserted on a single measurement.
            let timing = paired(
                rounds,
                || {
                    dispatcher
                        .gemm_pq2_0(&blocks, &input, &mut f32_out, m, n_rows, k)
                        .expect("f32 default gemm");
                },
                || {
                    gemm_two_bit_int8(tier, &blocks, &input, &mut int8_out, m, n_rows, k)
                        .expect("int8 gemm");
                },
            );
            let cos = cosine(&f32_out, &int8_out);
            let ratio = timing.ratio();
            // `cos` shares the `ratio` line so a line-oriented filter over
            // the output shows both together — it must stay a real
            // f32-vs-int8 comparison (not int8-vs-int8) even when the
            // variable is exported in the ambient shell, which `EnvGuard`
            // guarantees.
            println!(
                "{what}: f32-default {:?} -> int8-opt-in {:?} = {ratio:.2}x, cos = {cos:.6} \
                 ({rounds} interleaved rounds, debug_assertions={})",
                timing.baseline_min,
                timing.candidate_min,
                cfg!(debug_assertions)
            );
            assert!(
                cos >= COS_GATE,
                "{tier}: int8 GEMM output diverged: cos {cos}"
            );
            if bench_asserts() {
                assert!(
                    ratio >= 0.8,
                    "opting into {tier} at M=64 must not be meaningfully slower than the \
                     tiled + Rayon f32 default (got {ratio:.2}x)"
                );
            }
        }
    }

    /// One native-format throughput shape: `(label, k, n_rows)`.
    type Shape = (&'static str, usize, usize);

    /// The real Ternary-Bonsai-1.7B / -8B (Qwen3) projection shapes.
    const NATIVE_SHAPES: [Shape; 6] = [
        ("1.7B attn_q", 2048, 2048),
        ("1.7B ffn_up", 2048, 6144),
        ("1.7B ffn_down", 6144, 2048),
        ("8B attn_q", 4096, 4096),
        ("8B ffn_up", 4096, 12288),
        ("8B ffn_down", 12288, 4096),
    ];

    /// The Bonsai-8B (`Q1_0_g128`) projection shapes.
    const ONE_BIT_8B_SHAPES: [Shape; 3] = [
        ("8B attn_q", 4096, 4096),
        ("8B ffn_up", 4096, 12288),
        ("8B ffn_down", 12288, 4096),
    ];

    /// Paired timing rounds, the whole-shape row count and the GEMM batch
    /// size for this build: the whole shape and `m = 64` in release; a 1/16
    /// row slice, one round and `m = 8` in a debug build (correctness still
    /// runs, the numbers just stop meaning anything).
    fn bench_plan(n_rows: usize) -> (usize, usize, usize) {
        if cfg!(debug_assertions) {
            (1, (n_rows / 16).max(1), 8)
        } else {
            (9, n_rows, 64)
        }
    }

    /// Time `entry` on the f32 default (variable unset) against `tier`
    /// (variable set) in paired rounds; returns the pairing and the cosine
    /// between the two outputs.
    #[allow(clippy::too_many_arguments)]
    fn time_entry<B>(
        guard: &EnvGuard,
        dispatcher: &KernelDispatcher,
        entry: &Entry<B>,
        tier: Int8Tier,
        blocks: &[B],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
        rounds: usize,
    ) -> (Paired, f32) {
        let mut f32_out = vec![0.0f32; m * n_rows];
        let mut int8_out = vec![0.0f32; m * n_rows];
        let run = |out: &mut Vec<f32>| {
            (entry.run)(dispatcher, blocks, input, out, m, n_rows, k).expect(entry.name);
        };
        // Warm both paths (page-in, Rayon pool, branch predictors).
        guard.select(None);
        run(&mut f32_out);
        guard.select(Some(tier));
        run(&mut int8_out);
        let pairing = paired(
            rounds,
            || {
                guard.select(None);
                run(&mut f32_out);
            },
            || {
                guard.select(Some(tier));
                run(&mut int8_out);
            },
        );
        guard.select(None);
        (pairing, cosine(&f32_out, &int8_out))
    }

    /// One format's decode GEMV (`m = 1`, through `gemv_entry`) and batched
    /// GEMM (`m = 64`, through `gemm_entry`) at `shapes`, f32 default
    /// against every [`bench_tiers`] tier selected through the environment,
    /// on a CPU-tier dispatcher. Prints one line per point; returns the
    /// worst ratio.
    fn native_throughput<B>(
        guard: &EnvGuard,
        format: &str,
        entries: &[Entry<B>],
        (gemv_entry, gemm_entry): (&str, &str),
        shapes: &[Shape],
        blocks_for: impl Fn(usize, usize, u32) -> Vec<B>,
    ) -> f64 {
        let dispatcher = cpu_dispatcher();
        let find = |name: &str| {
            entries
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("no entry point {name}"))
        };
        let (gemv, gemm) = (find(gemv_entry), find(gemm_entry));
        println!(
            "native {format} throughput, load average {}",
            load_average()
        );
        let mut worst = f64::INFINITY;
        for tier in bench_tiers() {
            for (si, (label, k, full_rows)) in shapes.iter().enumerate() {
                let (rounds, n_rows, m_gemm) = bench_plan(*full_rows);
                let blocks = blocks_for(n_rows, *k, 0xBE00 + si as u32);
                for (entry, m) in [(gemv, 1usize), (gemm, m_gemm)] {
                    let input = activations(m * k, 0xBE80 + si as u32);
                    let kind = if m == 1 { "GEMV" } else { "GEMM" };
                    let what = format!(
                        "int8 {kind} {format} {label} [{k}x{n_rows}] m={m} {tier} via {}",
                        entry.name
                    );
                    let (timing, cos) = measure_point(0.8, &what, || {
                        time_entry(
                            guard,
                            &dispatcher,
                            entry,
                            tier,
                            &blocks,
                            &input,
                            m,
                            n_rows,
                            *k,
                            rounds,
                        )
                    });
                    worst = worst.min(timing.ratio());
                    println!(
                        "{what}: f32-default {:?} -> int8 {:?} = {:.2}x over f32, ratio = {:.2} \
                         (median {:.2}x), cos = {cos:.6}",
                        timing.baseline_min,
                        timing.candidate_min,
                        timing.ratio(),
                        timing.ratio(),
                        timing.median_ratio,
                    );
                    assert!(cos >= COS_GATE, "{what}: cos {cos}");
                }
            }
        }
        worst
    }

    /// Native `TQ2_0_g128` GEMV (`gemv_adaptive_ternary` — the decode path)
    /// and GEMM (`gemm_adaptive_ternary` — the batched path) at the real
    /// 1.7B / 8B shapes: opting in must never be meaningfully slower than
    /// the f32 default.
    #[test]
    fn int8_native_ternary_gemv_and_gemm_are_faster_than_the_f32_default() {
        let guard = EnvGuard::acquire();
        let worst = native_throughput(
            &guard,
            "TQ2_0_g128",
            &ternary_entries(),
            ("gemv_adaptive_ternary", "gemm_adaptive_ternary"),
            &NATIVE_SHAPES,
            |n_rows, k, seed| tq2_blocks(n_rows * (k / QK_TQ2_0_G128), seed),
        );
        if bench_asserts() {
            assert!(
                worst >= 0.8,
                "opting into the INT8 tier must never be meaningfully slower than the \
                 f32 default on the native ternary shapes (worst ratio {worst:.2}x)"
            );
        }
    }

    /// The 1-bit INT8 kernels are Rayon-parallel, so opting in must not be
    /// slower than the f32 **parallel** default at the Bonsai-8B shapes:
    /// `gemv_adaptive` at `m = 1` (decode) and `parallel::gemm_1bit_g128_par`
    /// at `m = 64` (the model's batched CPU prefill driver).
    #[test]
    fn int8_one_bit_gemv_and_gemm_are_not_slower_than_the_f32_default_at_8b_shapes() {
        let guard = EnvGuard::acquire();
        let worst = native_throughput(
            &guard,
            "Q1_0_g128",
            &one_bit_entries(),
            ("gemv_adaptive", "parallel::gemm_1bit_g128_par"),
            &ONE_BIT_8B_SHAPES,
            |n_rows, k, seed| q1_blocks(n_rows * (k / QK1_0_G128), seed),
        );
        if bench_asserts() {
            assert!(
                worst >= 0.8,
                "opting into the 1-bit INT8 tier must not be slower than the f32 \
                 parallel default at the 8B shapes (worst ratio {worst:.2}x)"
            );
        }
    }

    /// `Int8Tier::NeonI8mm`'s decode-reuse `SMMLA` GEMM against
    /// `Int8Tier::NeonDot`'s `SDOT` row kernel (per-row GEMV fan-out below
    /// the thread count, batch slabs above it) at every batch size from 2
    /// to 8 and a spread up to the CPU prefill's 128-row micro-batch, on the
    /// 27B `ffn_up` and the 8B projection shapes (2-bit), and on the
    /// Bonsai-8B shapes (1-bit). The two tiers must also agree bit for bit —
    /// they form the same integers.
    ///
    /// The asserted comparison is the one-thread one: it measures the two
    /// kernels and nothing else. The whole-pool figure is what a caller
    /// sees, but at the load averages this shared machine runs at it is
    /// decided as much by the scheduler as by the kernels — it is reported
    /// (and re-measured on a miss), not asserted.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i8mm_gemm_is_faster_than_neon_dot_at_every_m() {
        let _guard = EnvGuard::acquire();
        if !Int8Tier::NeonI8mm.is_supported() {
            eprintln!("skip: this host has no i8mm+dotprod");
            return;
        }
        let debug = cfg!(debug_assertions);
        // `(label, k, n_rows, one_bit)`.
        let shapes: [(&str, usize, usize, bool); 7] = [
            ("27B ffn_up TQ2_0_g128", 5120, 17408, false),
            ("8B attn_q TQ2_0_g128", 4096, 4096, false),
            ("8B ffn_up TQ2_0_g128", 4096, 12288, false),
            ("8B ffn_down TQ2_0_g128", 12288, 4096, false),
            ("8B attn_q Q1_0_g128", 4096, 4096, true),
            ("8B ffn_up Q1_0_g128", 4096, 12288, true),
            ("8B ffn_down Q1_0_g128", 12288, 4096, true),
        ];
        let ms: Vec<usize> = if debug {
            vec![2, 3, 8]
        } else {
            vec![2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        };
        let threads = rayon::current_num_threads();
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a one-thread Rayon pool");
        println!(
            "i8mm vs sdot GEMM ({threads} threads), load average {}",
            load_average()
        );
        let (mut worst_single, mut worst_parallel) = (f64::INFINITY, f64::INFINITY);
        for (si, (label, k, full_rows, one_bit)) in shapes.iter().enumerate() {
            let n_rows = if debug { full_rows / 16 } else { *full_rows };
            let seed = 0x1880 + si as u32;
            let tq2 = tq2_blocks(
                if *one_bit {
                    0
                } else {
                    n_rows * (k / QK_TQ2_0_G128)
                },
                seed,
            );
            let q1 = q1_blocks(
                if *one_bit {
                    n_rows * (k / QK1_0_G128)
                } else {
                    0
                },
                seed,
            );
            for &m in &ms {
                let input = activations(m * k, 0x18C0 + m as u32);
                // More rounds where runs are short (and the margin is
                // thinnest, at `m = 2..3`): each side's minimum needs at
                // least one round the scheduler left alone.
                let (single_rounds, parallel_rounds) = if debug {
                    (1, 1)
                } else if m <= 3 {
                    (63, 15)
                } else if m <= 8 {
                    (31, 15)
                } else {
                    (15, 7)
                };
                let mut sdot = vec![0.0f32; m * n_rows];
                let mut mmla = vec![0.0f32; m * n_rows];
                let run = |tier: Int8Tier, out: &mut Vec<f32>| {
                    if *one_bit {
                        gemm_1bit_g128_int8(tier, &q1, &input, out, m, n_rows, *k).expect("gemm");
                    } else {
                        gemm_two_bit_int8(tier, &tq2, &input, out, m, n_rows, *k).expect("gemm");
                    }
                };
                run(Int8Tier::NeonDot, &mut sdot);
                run(Int8Tier::NeonI8mm, &mut mmla);
                assert_bits_eq(&sdot, &mmla, &format!("{label} M={m}: i8mm vs sdot"));
                let what = format!("int8 GEMM {label} [{k}x{n_rows}] M={m}");
                // The kernels themselves, on one core.
                let (single, _) = measure_point(1.0, &format!("{what} on 1 thread"), || {
                    let timing = paired(
                        single_rounds,
                        || single_thread.install(|| run(Int8Tier::NeonDot, &mut sdot)),
                        || single_thread.install(|| run(Int8Tier::NeonI8mm, &mut mmla)),
                    );
                    (timing, 1.0)
                });
                // What a caller sees: the whole Rayon pool.
                let (parallel, _) =
                    measure_point(1.0, &format!("{what} on {threads} threads"), || {
                        let timing = paired(
                            parallel_rounds,
                            || run(Int8Tier::NeonDot, &mut sdot),
                            || run(Int8Tier::NeonI8mm, &mut mmla),
                        );
                        (timing, 1.0)
                    });
                worst_single = worst_single.min(single.ratio());
                worst_parallel = worst_parallel.min(parallel.ratio());
                println!(
                    "{what}: neon-dot {:?} -> neon-i8mm {:?} = {:.2}x over neon-dot on 1 thread \
                     (median {:.2}x); {:?} -> {:?} = {:.2}x over neon-dot on {threads} threads \
                     (median {:.2}x); {single_rounds}/{parallel_rounds} interleaved rounds",
                    single.baseline_min,
                    single.candidate_min,
                    single.ratio(),
                    single.median_ratio,
                    parallel.baseline_min,
                    parallel.candidate_min,
                    parallel.ratio(),
                    parallel.median_ratio,
                );
            }
        }
        println!(
            "int8 GEMM neon-i8mm over neon-dot, worst over every shape and M: \
             {worst_single:.2}x on 1 thread, {worst_parallel:.2}x on {threads} threads \
             (load average {})",
            load_average()
        );
        if bench_asserts() {
            assert!(
                worst_single > 1.0,
                "neon-i8mm's SMMLA GEMM kernel must beat neon-dot's at every M >= 2 \
                 (worst {worst_single:.2}x on 1 thread)"
            );
        }
    }

    /// The best f32 CPU GEMV available here: the NEON tier on AArch64, the
    /// scalar reference elsewhere.
    fn f32_gemv(
        blocks: &[BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is the AArch64 ISA baseline.
            unsafe {
                oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon(
                    blocks, input, output, n_rows, k,
                )
                .expect("neon f32 gemv");
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(blocks, input, output, n_rows, k)
                .expect("scalar f32 gemv");
        }
    }

    // ─── 5. activation-quantization contract ─────────────────────────────

    /// The permuted layout must be a pure re-indexing: dotting through it
    /// gives the same integer as dotting the sequential one.
    #[test]
    fn both_activation_layouts_dot_to_the_same_integer() {
        let input = activations(256, 0xA1);
        let seq = Int8Activation::quantize(&input, 1, 256, 128, Int8Layout::Sequential)
            .expect("quantize");
        let perm =
            Int8Activation::quantize(&input, 1, 256, 128, Int8Layout::Stride4).expect("quantize");
        assert_eq!(seq.scales_row(0), perm.scales_row(0));
        assert_eq!(seq.sums_row(0), perm.sums_row(0));
        for j in 0..256usize {
            let block = j / 128;
            let within = j % 128;
            let pos = block * 128 + Int8Layout::Stride4.position_of(within);
            assert_eq!(
                seq.codes_row(0)[j],
                perm.codes_row(0)[pos],
                "element {j} landed in the wrong permuted slot"
            );
        }
    }

    /// The tier's accuracy comes from the activation quantization alone —
    /// the weights are exact — so the relative activation error bounds the
    /// whole tier's error.
    #[test]
    fn activation_quantization_error_stays_under_one_percent() {
        let input = activations(4096, 0xA2);
        let act =
            Int8Activation::quantize(&input, 1, 4096, 128, Int8Layout::Stride4).expect("quantize");
        let err = act.relative_error(&input).expect("relative_error");
        println!("int8 activation relative error: {err:.6}");
        assert!(err < 0.01, "activation quantization error {err} too large");
    }
}
