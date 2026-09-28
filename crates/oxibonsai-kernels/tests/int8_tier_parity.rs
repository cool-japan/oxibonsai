//! Acceptance suite for the INT8 dot-product tier (K-14 / K-INT8).
//!
//! Covers, in order:
//!
//! 1. **Cross-tier bit-identity** — every `Int8Tier` forms the same `i32`
//!    per block (integer arithmetic is exact), so their `f32` outputs must
//!    be bit-for-bit equal, not merely close.
//! 2. **Accuracy vs the f32 tier** — `cos >= 0.999`, on synthetic weights
//!    *and* on the real 1.7B / 8B / 27B GGUFs when they are present.
//! 3. **The HARD CONSTRAINT** — the tier is never selected implicitly, and
//!    `KernelTier::Neon` is untouched by it.
//! 4. **Throughput** — the GEMV inner loop against the NEON f32 kernel it
//!    replaces (reported always, asserted only under
//!    `OXIBONSAI_INT8_BENCH=1`; see [`int8_tier_parity::speedup`]).
//!
//! ## Why one `mod int8_tier_parity`
//!
//! The package gate selects this file with a **substring filter over test
//! names** (`cargo test ... int8_tier_parity`), not `--test`. libtest
//! matches that substring against a test's full in-binary path, which for a
//! top-level `#[test] fn foo` in an integration test is just `foo` — the
//! filter would select **zero tests and exit 0** (systemic finding S-1;
//! `gemv_ptq1.rs`'s `prism_gemv_tests` module exists for exactly this
//! reason). Wrapping everything in a module named after the file makes
//! every path start with `int8_tier_parity::`, so the filter really runs
//! them.
//!
//! Real-model tests self-skip when `models/` is empty — the normal state in
//! a fresh clone or an isolated worktree — and print why, so a skip is
//! never mistaken for a pass. Point them at a checkout with weights via
//! `OXIBONSAI_MODELS_DIR` (no path is ever hardcoded).

mod int8_tier_parity {
    use half::f16;
    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_core::gguf::types::GgufTensorType;
    use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
    use oxibonsai_core::{BlockPQ2_0, BlockTQ2_0_g128, QK_PQ2_0, QK_TQ2_0_G128};
    use oxibonsai_kernels::dispatch_int8::{
        gemm_two_bit_int8, gemv_1bit_g128_int8, gemv_two_bit_int8, Int8Tier, KERNEL_TIER_ENV,
    };
    use oxibonsai_kernels::quant_activation::{Int8Activation, Int8Layout};

    // ─── helpers ─────────────────────────────────────────────────────────

    /// Serializes every test in *this* binary that reads or writes an
    /// environment variable: `the_int8_tier_is_never_selected_implicitly`
    /// mutates [`KERNEL_TIER_ENV`], `int8_gemv_is_faster_than_the_f32_gemv`
    /// reads `OXIBONSAI_INT8_BENCH`, and the real-model tests read
    /// `OXIBONSAI_MODELS_DIR` via `oxibonsai_testkit::workspace::find_model`
    /// — all in the same `cargo test --test int8_tier_parity` process.
    /// `std::env::set_var`/`remove_var` are `unsafe fn` (edition 2024)
    /// precisely because a concurrent `std::env::var` on *any* key can
    /// observe a torn `environ` while another thread mutates it (a
    /// whole-process hazard, not a same-key one), and `cargo test` runs
    /// this file's tests as threads in one process by default. Mirrors
    /// `oxibonsai_testkit::workspace`'s own `ENV_LOCK` pattern.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
        fn next_u8(&mut self) -> u8 {
            (self.next() >> 19) as u8
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next() >> 8) as i32 % 2001 - 1000) as f32 / 512.0
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        if na == 0.0 || nb == 0.0 {
            return if na == nb { 1.0 } else { 0.0 };
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn tq2_blocks(n: usize, seed: u32) -> Vec<BlockTQ2_0_g128> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                }
            })
            .collect()
    }

    fn pq2_blocks(n: usize, seed: u32) -> Vec<BlockPQ2_0> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                BlockPQ2_0 {
                    d: f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                    qs,
                }
            })
            .collect()
    }

    fn q1_blocks(n: usize, seed: u32) -> Vec<BlockQ1_0G128> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; QK1_0_G128 / 8];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                BlockQ1_0G128 {
                    d: f16::from_f32(0.125 + (rng.next_u8() % 16) as f32 / 256.0),
                    qs,
                }
            })
            .collect()
    }

    fn activations(len: usize, seed: u32) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..len).map(|_| rng.next_f32()).collect()
    }

    /// Every INT8 tier this host can execute.
    fn supported_tiers() -> Vec<Int8Tier> {
        Int8Tier::ALL
            .iter()
            .copied()
            .filter(|t| t.is_supported())
            .collect()
    }

    /// The accuracy gate every acceptance assertion in this file uses.
    const COS_GATE: f32 = 0.999;

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
            for (i, (e, g)) in reference.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    e.to_bits(),
                    g.to_bits(),
                    "{tier} diverged from the scalar int8 tier at row {i}: {e} vs {g}"
                );
            }
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
            assert_eq!(
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{tier} diverged on the 1-bit format"
            );
        }
    }

    /// The `SMMLA` 2x2 GEMM tiler (and every other tier's GEMM) must agree
    /// with the GEMV applied per batch row — including the odd-`m`/odd-`n`
    /// tails the tiler handles separately.
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
            for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    e.to_bits(),
                    g.to_bits(),
                    "{tier} gemm diverged from the gemv sweep at cell {i}: {e} vs {g}"
                );
            }
        }
    }

    /// Every other case in this file uses `n_rows` in
    /// `{5,7,16,23,37,64,192,200}` — all below `dispatch_int8.rs`'s
    /// `INT8_PAR_MIN_ROWS = 256` — so `two_bit_gemv_int8`'s Rayon fan-out
    /// (the `par_chunks_mut` split and its `blocks[row_start *
    /// blocks_per_row .. (row_start + rows) * blocks_per_row]` slicing) has
    /// never executed under test (verifier finding, wave-4 review). `n_rows
    /// = 300` crosses the threshold; the three chunk sizes below are
    /// deliberately uneven and not aligned to any power of two, so the
    /// comparison exercises real, ragged chunk boundaries rather than a
    /// suspiciously round split.
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

            // The exact same rows, split into three uneven sequential
            // sub-300 calls -- each on its own block slice, exactly as
            // `two_bit_gemv_int8`'s Rayon chunks would see them, but
            // computed one at a time on this thread.
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

            for (i, (a, b)) in whole.iter().zip(sequential.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tier}: row {i} differs between the single 300-row call \
                     (which crosses INT8_PAR_MIN_ROWS=256 and Rayon-splits) \
                     and the sequential sub-call split"
                );
            }
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

    // ─── 2b. accuracy on REAL model weights ──────────────────────────────

    /// Rows of a real weight matrix each acceptance run measures.
    ///
    /// Bounded deliberately: the package gate's first leg runs this file in
    /// a **debug** build, and a full 17408-row 27B matrix there would be
    /// minutes of scalar arithmetic per tier. 192 rows over the matrix's
    /// full `k` is several hundred thousand real MACs per tier — ample for
    /// a cosine gate, and it exercises the genuine weight distribution
    /// (which is the whole point of not using synthetic data).
    const REAL_ROWS: usize = 192;

    /// The first tensor of `want` type in `file`, as raw bytes plus its
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

    /// Run the cosine gate over a real `TQ2_0_g128` file.
    fn check_real_ternary_model(file_name: &str) {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    // ─── 3. the HARD CONSTRAINT ──────────────────────────────────────────

    /// The tier must never be reachable without being named.
    ///
    /// Two independent guards: the environment selector returns `None` with
    /// a clean environment, and no `KernelTier` the auto-detector can
    /// produce is an INT8 tier — they are different types, so a "silent
    /// upgrade of `KernelTier::Neon`" cannot even be written.
    #[test]
    fn the_int8_tier_is_never_selected_implicitly() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: serialized by `ENV_LOCK` above against every other test
        // in this file that reads or writes an environment variable.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        assert_eq!(
            Int8Tier::from_env(),
            None,
            "OXIBONSAI_KERNEL_TIER unset must select no INT8 tier — making this \
             tier a default would change results bit-for-bit and break the \
             CPU-vs-Metal determinism guard"
        );

        let auto = oxibonsai_kernels::KernelDispatcher::auto_detect();
        let tier = auto.tier();
        println!("auto-detected KernelTier = {tier}");
        assert!(
            !format!("{tier}").contains("int8")
                && !format!("{tier}").contains("dot")
                && !format!("{tier}").contains("vnni"),
            "auto-detection must never land on an INT8 tier, got {tier}"
        );
    }

    /// `KernelTier::Neon` must still produce exactly what it produced before
    /// K-14: the f32 tier is not routed through anything in this package.
    #[test]
    fn the_f32_neon_tier_is_untouched_by_the_int8_tier() {
        let (n_rows, k) = (16usize, 2 * QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x81);
        let input = activations(k, 0x82);

        let mut scalar_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128(
            &blocks,
            &input,
            &mut scalar_out,
            n_rows,
            k,
        )
        .expect("scalar f32 gemv");

        // The dispatcher's own best CPU tier, which must agree with the
        // scalar reference to float noise and NOT to int8 precision.
        let dispatcher = oxibonsai_kernels::KernelDispatcher::auto_detect();
        let mut tier_out = vec![0.0f32; n_rows];
        oxibonsai_kernels::traits::TernaryKernel::gemv_ternary_g128(
            &dispatcher,
            &blocks,
            &input,
            &mut tier_out,
            n_rows,
            k,
        )
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
            tier_out
                .iter()
                .zip(int8_out.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the int8 tier produced bit-identical output to the f32 tier — it is \
             not actually quantizing, so this suite proves nothing"
        );
    }

    // ─── 4. throughput ───────────────────────────────────────────────────

    /// Measure the GEMV inner loop: NEON f32 (today's tier) vs the fastest
    /// INT8 tier, single-threaded on both sides.
    ///
    /// `n_rows` stays under the INT8 GEMV's Rayon threshold so this compares
    /// *kernels*, not thread counts. The assertion is opt-in
    /// (`OXIBONSAI_INT8_BENCH=1`) because the package gate builds and runs
    /// under concurrent compilation, where a wall-clock ratio is not a
    /// sound pass/fail signal; the measurement itself always prints.
    #[test]
    fn int8_gemv_is_faster_than_the_f32_gemv() {
        use std::time::Instant;

        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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

        if std::env::var("OXIBONSAI_INT8_BENCH").as_deref() == Ok("1") {
            assert!(
                speedup >= 2.5,
                "INT8 GEMV must be >= 2.5x the f32 GEMV (got {speedup:.2}x)"
            );
        }
    }

    /// K-INT8 wave-4b minors[1]/[2]: `gemm_two_bit_int8` used to run
    /// single-threaded regardless of `m`, and `Int8Tier::NeonI8mm` used the
    /// `SMMLA` 2x2 tile for GEMM, which on this M3 measured *slower* than
    /// the plain `SDOT` row loop at every `M >= 2` — together those made
    /// *opting in* to the INT8 tier at a real prefill batch size (27B
    /// `ffn_up`, M=64) slower than the tiled + Rayon f32 default (326ms
    /// `neon-i8mm` vs 98.6ms f32). This checks that gap is closed: the
    /// opt-in path, parallelized and no longer routed through the tile, is
    /// no longer slower than the default.
    #[test]
    fn int8_gemm_at_m64_is_not_slower_than_the_f32_default() {
        use oxibonsai_kernels::dispatch::KernelDispatcher;
        use oxibonsai_kernels::traits::PrismKernel;
        use std::time::Instant;

        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (m, n_rows, k) = (64usize, 17408usize, 5120usize);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0xC0DE_0064);
        let input = activations(m * k, 0xFACE_0064);
        let tier = Int8Tier::best_available();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut f32_out = vec![0.0f32; m * n_rows];
        let mut int8_out = vec![0.0f32; m * n_rows];

        // Warm up both paths (page-in, branch predictors, Rayon pool).
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut f32_out, m, n_rows, k)
            .expect("f32 default gemm warmup");
        gemm_two_bit_int8(tier, &blocks, &input, &mut int8_out, m, n_rows, k)
            .expect("int8 gemm warmup");

        // Min of a few trials on each side: this dev machine runs under
        // heavy, variable background load from sibling sessions, and a
        // single wall-clock sample is dominated by that noise.
        let trials = if cfg!(debug_assertions) { 1 } else { 9 };
        let mut f32_time = std::time::Duration::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            dispatcher
                .gemm_pq2_0(&blocks, &input, &mut f32_out, m, n_rows, k)
                .expect("f32 default gemm");
            f32_time = f32_time.min(t0.elapsed());
        }
        let mut int8_time = std::time::Duration::MAX;
        for _ in 0..trials {
            let t1 = Instant::now();
            gemm_two_bit_int8(tier, &blocks, &input, &mut int8_out, m, n_rows, k)
                .expect("int8 gemm");
            int8_time = int8_time.min(t1.elapsed());
        }

        let ratio = f32_time.as_secs_f64() / int8_time.as_secs_f64().max(1e-12);
        println!(
            "int8 GEMM at M={m} ffn_up[{k}x{n_rows}] {tier}: f32-default {f32_time:?} -> \
             int8-opt-in {int8_time:?} = {ratio:.2}x ({trials} trials each, \
             debug_assertions={})",
            cfg!(debug_assertions)
        );

        let cos = cosine(&f32_out, &int8_out);
        assert!(cos >= COS_GATE, "int8 GEMM output diverged: cos {cos}");

        if std::env::var("OXIBONSAI_INT8_BENCH").as_deref() == Ok("1") {
            assert!(
                ratio >= 0.8,
                "opting into the INT8 tier at M=64 must not be meaningfully \
                 slower than the tiled + Rayon f32 default any more (got \
                 {ratio:.2}x: f32 {f32_time:?} vs int8 {int8_time:?})"
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
