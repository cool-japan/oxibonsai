//! HOTFIX-TESTMEM regression test: constructing an `InferenceEngine` + router
//! from a config that carries no real weights must never approach the
//! multi-GB blowup filed against `server::tests::create_router_builds_without_tokenizer`
//! and `server::tests::create_router_with_shared_metrics`.
//!
//! Those two unit tests built their engine from `Qwen3Config::bonsai_8b()`,
//! which sent `InferenceEngine::new` -> `BonsaiModel::new`
//! (`crates/oxibonsai-model/src/model/types/mod.rs`) allocating two
//! `vocab_size * hidden_size` f32 tables (`token_embd`, `output_weight`; the
//! `token_embd` one goes through an `Arc::from(vec![..])` conversion that
//! forces every page of the destination resident) plus a KV cache sized off a
//! hardcoded `max_seq_len = 4096` — roughly 5 GB total, measured (see below),
//! just to build a router that never runs a forward pass. Both tests are now
//! `Qwen3Config::tiny_test()`-based; this file is the standing regression
//! test the package spec asked for so a future edit can't reintroduce a
//! large config here unnoticed.
//!
//! Measured with `/usr/bin/time -l` against the compiled test binary
//! (`--exact --test-threads=1`, one process per test, this worktree, before
//! this package's fix): `create_router_builds_without_tokenizer` peaked at
//! 5 006 376 960 bytes (~4.66 GiB) maximum resident set size;
//! `create_router_with_shared_metrics` peaked at 5 007 622 144 bytes
//! (~4.66 GiB). Both numbers are recorded in this package's `notes` field.
//!
//! ## Why this test asserts on config-derived sizes, not an allocation count
//!
//! `oxibonsai-model` has a counting `#[global_allocator]` for exactly this
//! kind of assertion (`crates/oxibonsai-model/src/test_alloc.rs`,
//! `count_allocations`), but it is `#![cfg(test)]` for that crate *and*
//! `pub(crate)`: a dependency crate is compiled without `--cfg test` when a
//! *different* crate's integration test binary links it, so the module does
//! not even exist in this binary, and it would not be visible outside
//! `oxibonsai-model` if it did. Reusing it from here is not possible without
//! making it a public, always-compiled API of `oxibonsai-model` — a product
//! change outside this test-only package's `owned_files`. A libc/RSS
//! before-after check from inside the test process (`std::process` /
//! `/proc`) was considered and rejected per the package spec (it is noisy
//! across platforms and measures the whole process, not the allocation under
//! test). Defining a second, independent `#[global_allocator]` scoped to
//! just this integration-test binary was also considered — since each
//! `tests/*.rs` file compiles to its own binary, it would not physically
//! collide with `oxibonsai-model`'s — but the package spec is explicit
//! ("never add a second `#[global_allocator]`"), so this test instead
//! bounds the one config-derived quantity that dominates
//! `BonsaiModel::new`'s footprint: the size of a single
//! `vocab_size * hidden_size` f32 table (`token_embd` and `output_weight`
//! are both exactly this size).

#![cfg(feature = "server")]

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::create_router;

/// `BonsaiModel::new` allocates exactly two `vocab_size * hidden_size` f32
/// tables (`token_embd`, `output_weight`) plus a KV cache that, for any sane
/// `num_layers`/`num_kv_heads`/`head_dim`, is smaller than either of those
/// tables alone. Bounding one table's size below 64 MiB keeps both tables
/// combined (~128 MiB) plus the KV cache/rope/norm overhead safely inside
/// the 256 MiB engine-construction ceiling this package's spec sets, with
/// wide margin (`tiny_test()` actually measures at ~37 MiB per table, i.e.
/// ~74 MiB + ~2 MiB KV cache — see `notes`).
///
/// That margin is real but not huge (~1.7x): `tiny_test()`'s
/// `hidden_size = 64` isn't a multiple of 128, which already makes it
/// incompatible with `BonsaiModel::new_for_testing_with_blocks`'s
/// `is_multiple_of(128)` requirement for a *populated*-block fixture. If a
/// future change raises `tiny_test().hidden_size` to 128 (e.g. to reuse that
/// fixture), the table becomes 74 MiB and this assertion starts failing —
/// intentionally: that would be a real doubling of engine-construction
/// memory, not a false positive, and should be re-evaluated against a raised
/// `MAX_EMBED_TABLE_BYTES` rather than silenced.
const MAX_EMBED_TABLE_BYTES: usize = 64 * 1024 * 1024;

#[test]
fn engine_and_router_construction_stays_within_memory_bound() {
    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);

    // Assert on the config the already-constructed engine actually carries
    // (`engine.dense_model()?.config()`), not on a separately re-constructed
    // `Qwen3Config::tiny_test()` value. That makes the bound self-checking:
    // if this test is ever edited to build the engine from a larger config
    // (e.g. `bonsai_8b()`), the assertion below fails loudly instead of
    // silently passing against a stale, independently computed number.
    let carried = engine
        .dense_model()
        .expect("a config-built engine holds a dense model")
        .config();
    // The engine-level seam reports the same geometry.
    assert_eq!(engine.vocab_size(), carried.vocab_size);
    assert_eq!(engine.hidden_size(), carried.hidden_size);
    let embed_table_bytes = carried
        .vocab_size
        .saturating_mul(carried.hidden_size)
        .saturating_mul(std::mem::size_of::<f32>());
    assert!(
        embed_table_bytes < MAX_EMBED_TABLE_BYTES,
        "engine-construction embedding table is {embed_table_bytes} bytes \
         (vocab_size={}, hidden_size={}); HOTFIX-TESTMEM requires test \
         fixtures to stay under {MAX_EMBED_TABLE_BYTES} bytes per table \
         (server::tests::create_router_* used to allocate ~2.3 GB per table \
         from Qwen3Config::bonsai_8b(), ~5 GB total measured RSS)",
        carried.vocab_size,
        carried.hidden_size,
    );

    // The actual construction path the original bug report named: build a
    // router from this engine. If this ever regresses to a multi-GB config,
    // the test process gets OOM-killed here exactly as the bug described,
    // rather than silently passing.
    let _router = create_router(engine, None);
}

/// Replica sharing on a **real** model (verify:METAL-CONCURRENCY blocking
/// #1, ENGINE-SEAM item 6): three replicas of the dense model at
/// `$OXI_MODEL`, built exactly the way `build_pool_from_gguf_parts` builds a
/// pool (replica 1 maps and leaks the file, replicas 2 and 3 borrow the same
/// `&'static GgufFile` and the shared token-embedding table), must place the
/// model's weights on the GPU **once**:
///
/// * `MetalGraph::bytes_uploaded` — the Metal weight cache's resident gauge —
///   grows with replica 1 and not at all with replicas 2 and 3;
/// * the scirs2 backend's cumulative fresh-upload counter likewise (for a
///   `Q1_0_g128` model every replica runs `upload_weights_to_gpu`, and each
///   used to mint fresh handles — N replicas, N copies — while the ternary
///   fused route skips that upload outright, so there it stays at zero);
/// * each later replica reports zero fresh buffers and exactly replica 1's
///   buffers as shared, and decodes the same greedy tokens.
///
/// Point `OXI_MODEL` at `models/Bonsai-8B.gguf` (the `Q1_0_g128` case this
/// sharing was built for) or `models/Ternary-Bonsai-1.7B.gguf`. Unset, the
/// test prints a capability report and records a skip.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn real_model_replicas_place_their_weights_on_the_gpu_once() {
    use oxibonsai_kernels::gpu_backend::weight_bytes_uploaded_total;
    use oxibonsai_kernels::{KernelDispatcher, KernelTier, MetalGraph};
    use oxibonsai_runtime::engine::Backend;
    use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};

    const TEST: &str =
        "engine_memory_bound::real_model_replicas_place_their_weights_on_the_gpu_once";
    const REPLICAS: usize = 3;
    const MAX_SEQ_LEN: usize = 256;
    const MIB: f64 = 1024.0 * 1024.0;

    let Some(path) = std::env::var_os("OXI_MODEL").filter(|p| !p.is_empty()) else {
        eprintln!(
            "capability report: {TEST} SKIPPED -- $OXI_MODEL is not set (point it at \
             models/Bonsai-8B.gguf or models/Ternary-Bonsai-1.7B.gguf)"
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    let path = std::path::PathBuf::from(path);
    assert!(
        path.is_file(),
        "$OXI_MODEL={} is not a file",
        path.display()
    );

    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 4,
    };
    // "The capital of Japan is" in the Qwen3 vocabulary.
    let prompt = [785u32, 6722, 315, 6323, 374];

    let graph = MetalGraph::global().expect("an accelerated Metal device");
    let probe = KernelDispatcher::auto_detect();
    let graph_start = graph.bytes_uploaded();
    let scirs2_start = weight_bytes_uploaded_total(&probe);

    let (mut first, gguf) = InferenceEngine::from_gguf_path_leaked_with_backend(
        &path,
        params.clone(),
        42,
        MAX_SEQ_LEN,
        Backend::Auto,
    )
    .expect("replica 1 loads");
    if first.is_hybrid() || first.kernel_tier() != KernelTier::Gpu {
        eprintln!(
            "capability report: {TEST} SKIPPED -- {} runs on {} (only a dense model on the GPU \
             tier uploads weights to share)",
            first.model_description(),
            first.kernel_tier()
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let reference = first.generate(&prompt, 4).expect("replica 1 decodes");
    let graph_one = graph.bytes_uploaded().saturating_sub(graph_start);
    let scirs2_one = weight_bytes_uploaded_total(&probe).saturating_sub(scirs2_start);
    let first_uploads = first.gpu_upload_stats();
    assert!(
        graph_one > 0,
        "replica 1 must place the model in the Metal weight cache"
    );
    eprintln!(
        "[{}] replica 1: MetalGraph +{:.1} MiB, scirs2 +{:.1} MiB ({} fresh / {} shared \
         buffers), tokens {reference:?}",
        first.kernel_label(),
        graph_one as f64 / MIB,
        scirs2_one as f64 / MIB,
        first_uploads.fresh_buffers,
        first_uploads.shared_buffers
    );

    let shared_token_embd = first.model_token_embd();
    let mut replicas = vec![first];
    for n in 2..=REPLICAS {
        let mut replica = InferenceEngine::from_gguf_static_with_embd_and_backend(
            gguf,
            params.clone(),
            42,
            MAX_SEQ_LEN,
            std::sync::Arc::clone(&shared_token_embd),
            Backend::Auto,
        )
        .expect("replica loads");
        let tokens = replica.generate(&prompt, 4).expect("replica decodes");
        let uploads = replica.gpu_upload_stats();
        let graph_now = graph.bytes_uploaded().saturating_sub(graph_start);
        let scirs2_now = weight_bytes_uploaded_total(&probe).saturating_sub(scirs2_start);
        eprintln!(
            "replica {n}: MetalGraph total +{:.1} MiB, scirs2 total +{:.1} MiB ({} fresh / {} \
             shared buffers, {:.1} MiB shared), tokens {tokens:?}",
            graph_now as f64 / MIB,
            scirs2_now as f64 / MIB,
            uploads.fresh_buffers,
            uploads.shared_buffers,
            uploads.shared_bytes as f64 / MIB
        );
        assert_eq!(tokens, reference, "replica {n} decodes like replica 1");
        assert_eq!(
            graph_now, graph_one,
            "replica {n} must not grow MetalGraph::bytes_uploaded"
        );
        assert_eq!(
            scirs2_now, scirs2_one,
            "replica {n} must not upload a second scirs2 copy"
        );
        assert_eq!(uploads.fresh_buffers, 0, "replica {n}: fresh uploads");
        assert_eq!(
            uploads.shared_buffers,
            first_uploads.total_buffers(),
            "replica {n} registers every buffer replica 1 retained"
        );
        replicas.push(replica);
    }
    assert_eq!(replicas.len(), REPLICAS);
    record_executed(Capability::LegacyModels, TEST);
}
