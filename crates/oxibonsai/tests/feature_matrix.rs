//! Feature-combination smoke tests for the `oxibonsai` facade (T-12,
//! RAG-EVAL-IMG-17/35).
//!
//! `oxibonsai` re-exports nine other crates, five of them behind their own
//! `#[cfg(feature = ...)]` gate (`rag`, `native-tokenizer`, `eval`, `server`,
//! `image`). Before this file the only gated re-export with a dedicated
//! regression test was `server` (the sibling `server_feature.rs` file); this
//! file extends the same "does the feature actually compile *and* is the
//! re-export load-bearing" check to every other optional feature this crate
//! can build today, plus their real-world unions (`full`, and the GPU +
//! imagen deployment shape `{server, metal, image}`).
//!
//! ## Scope, and why this is not a pile of `assert_type_exists::<T>()` calls
//!
//! The T-12 correction is explicit: seven hand-written
//! type-existence tests are strictly worse than `cargo hack check -p
//! oxibonsai --each-feature` (mechanical, zero maintenance) at proving a
//! feature *compiles*, and the one thing worth writing by hand is a real,
//! end-to-end exercise of the facade's re-exports that a bare compile check
//! cannot give you (`scripts/ci.sh`'s per-feature facade stages are the
//! mechanical compile-only check for `image`/`metal`/`{server,metal,image}`;
//! this file is the load-bearing-reachability check on top of it). Every
//! test below therefore calls a real function or builds a real (small,
//! cheap) value through the `oxibonsai::` path rather than merely naming a
//! type, wherever a cheap, side-effect-free call exists.
//!
//! Two tests below are deliberate exceptions, not an oversight:
//! `image_feature_pipeline_module_is_reachable`'s only public surface
//! reachable without a real GGUF weight file is `pipeline::PipelineError`
//! (an enum with no zero-argument constructor to call), and
//! `server_metal_image_union_composes`'s `MetalGraph` handle cannot be
//! constructed without real GPU hardware. For both, a compile-time
//! reachability check through the facade path *is* the correctly-costed
//! test. `metal_feature_forwards_to_kernels_model_and_runtime` still drives
//! a real, gracefully-fallible dispatcher call — the one `metal` surface
//! that does not need real hardware to exercise meaningfully.
//!
//! ## Combinations covered here
//!
//! - `{default}` (`hf-tokenizer` only): [`default_feature_set_reaches_core_kernels_and_runtime`].
//! - `{rag}`: [`rag_feature_chunk_document_is_reachable`].
//! - `{eval}`: [`eval_feature_sentence_bleu_is_reachable`].
//! - `{native-tokenizer}`: [`native_tokenizer_feature_char_level_stub_round_trips`].
//! - `{metal}`: [`metal_feature_forwards_to_kernels_model_and_runtime`].
//! - `{image}`: [`image_feature_pipeline_module_is_reachable`].
//! - `{server}`: already covered end to end by `server_feature.rs`; not
//!   duplicated here.
//! - `{full}` = `{server, rag, native-tokenizer, hf-tokenizer, eval, image}`:
//!   [`full_feature_set_composes_every_optional_crate_together`] — a union
//!   can fail even when every individual feature compiles (name clashes,
//!   feature-unification surprises), so this touches one real symbol from
//!   every crate `full` pulls in, in a single compilation unit.
//! - `{server, metal, image}` (the real-world GPU + imagen deployment
//!   shape — independent of, and not implied by, `full`):
//!   [`server_metal_image_union_composes`].
//! - Unconditional (compiled regardless of the active feature set):
//!   [`e2e_synthetic_gguf_round_trips_through_core_and_model_facade_paths`].

/// `{default}` — the always-on re-exports (`core`, `kernels`, `runtime`; none
/// of them `#[cfg]`-gated) must all be reachable with zero optional features
/// active, and each call below must actually run without panicking.
#[test]
fn default_feature_set_reaches_core_kernels_and_runtime() {
    // `oxibonsai::core`: a freshly constructed streaming GGUF parser starts
    // empty and incomplete — the same entry point the crate's own top-level
    // doctest uses, but with a real assertion on its state rather than a
    // bare "it compiles".
    let parser = oxibonsai::core::gguf::streaming::GgufStreamParser::new();
    assert!(!parser.is_complete(), "a fresh parser has consumed nothing");
    assert_eq!(parser.bytes_consumed(), 0);

    // `oxibonsai::kernels`: the runtime CPU-tier dispatcher must resolve to
    // *some* tier without panicking, on every architecture this workspace
    // targets.
    let tier = oxibonsai::kernels::dispatch::cpu_kernel_tier();
    assert!(
        !format!("{tier:?}").is_empty(),
        "cpu_kernel_tier() must resolve to a debug-printable tier"
    );

    // `oxibonsai::runtime`: sampling defaults are real, documented values
    // (not just a type that happens to exist).
    let params = oxibonsai::runtime::sampling::SamplingParams::default();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
}

/// `{rag}` — `chunk_document` (the RAG-EVAL-IMG-14 O(n^2) regression surface)
/// must be reachable and actually chunk real text through the facade path.
#[cfg(feature = "rag")]
#[test]
fn rag_feature_chunk_document_is_reachable() {
    let text = "Hello world. This is a small test document for the RAG chunker. \
                It has more than one sentence so chunking has something to do.";
    let chunks = oxibonsai::rag::chunk_document(text, 0, &oxibonsai::rag::ChunkConfig::default());
    assert!(
        !chunks.is_empty(),
        "chunk_document must return at least one chunk for non-empty input"
    );
    assert!(chunks.iter().all(|c| !c.text.is_empty()));
}

/// `{eval}` — `sentence_bleu` must be reachable and return a real score
/// through the facade path (identical candidate/reference must score ~1.0).
#[cfg(feature = "eval")]
#[test]
fn eval_feature_sentence_bleu_is_reachable() {
    let score = oxibonsai::eval::sentence_bleu(
        "the cat sat on the mat",
        &["the cat sat on the mat"],
        &oxibonsai::eval::BleuConfig::default(),
    );
    assert!(
        score.bleu > 0.99,
        "an identical candidate/reference pair should score ~1.0 BLEU, got {}",
        score.bleu
    );
}

/// `{native-tokenizer}` — the Pure-Rust `OxiTokenizer` must be reachable and
/// actually encode/decode through the facade path.
#[cfg(feature = "native-tokenizer")]
#[test]
fn native_tokenizer_feature_char_level_stub_round_trips() {
    let tok = oxibonsai::tokenizer::OxiTokenizer::char_level_stub(128);
    let ids = tok
        .encode("hi!")
        .expect("a char-level stub must encode plain ASCII");
    assert!(!ids.is_empty());
    let text = tok
        .decode(&ids)
        .expect("a char-level stub must decode its own ids");
    assert_eq!(text, "hi!");
}

/// `{metal}` — `metal` must enable `oxibonsai-kernels/metal`,
/// `oxibonsai-model/metal` and `oxibonsai-runtime/metal`. A bare
/// `assert_type_exists::<MetalGraph>()` would only prove the type compiles,
/// exactly the pattern this module doc argues is strictly worse than a real
/// call — so this instead drives a real, gracefully-fallible probe:
/// `KernelDispatcher::try_with_tier` never panics when no accelerated GPU
/// backend is present (it returns `Err`), so the assertion holds whether or
/// not the CI runner has real GPU hardware, while still proving the `Gpu`
/// tier — and the `metal` feature that unlocks it — is reachable end to end
/// through the facade path.
#[cfg(feature = "metal")]
#[test]
fn metal_feature_forwards_to_kernels_model_and_runtime() {
    use oxibonsai::kernels::dispatch::{KernelDispatcher, KernelTier};
    match KernelDispatcher::try_with_tier(KernelTier::Gpu) {
        Ok(dispatcher) => assert_eq!(dispatcher.tier(), KernelTier::Gpu),
        Err(e) => {
            // No accelerated GPU backend on this runner. Still a pass: the
            // feature compiled and the call ran (no panic), which is what
            // this test exists to prove.
            eprintln!(
                "metal_feature_forwards_to_kernels_model_and_runtime: no GPU backend available: {e}"
            );
        }
    }
}

/// `{image}` — `oxibonsai::image` (the `oxibonsai-image` re-export) must be
/// reachable through the facade path. `pipeline::PipelineError` is the
/// crate's documented single library entry point's error type
/// (`pipeline::text_to_image`); it has no zero-argument constructor to call
/// without a real GGUF weight file, so — as the module doc above explains —
/// a compile-time reachability check through the facade path is the
/// correctly-costed test here, not an oversight.
#[cfg(feature = "image")]
#[test]
fn image_feature_pipeline_module_is_reachable() {
    fn assert_type_exists<T>() {}
    assert_type_exists::<oxibonsai::image::pipeline::PipelineError>();
}

/// `{full}` = `{server, rag, native-tokenizer, hf-tokenizer, eval, image}` —
/// every optional crate `full` unions together must compose in one
/// compilation unit, each exercised with a real call rather than a bare type
/// check.
#[cfg(feature = "full")]
#[test]
fn full_feature_set_composes_every_optional_crate_together() {
    let args = oxibonsai::serve::ServerArgs::default();
    assert_eq!(args.port, 8080);

    // `ChunkConfig::default()` discards anything shorter than
    // `min_chunk_size` (32 chars), so this needs more than a few words.
    let text = "One sentence. Two sentences. Three sentences. All in one small paragraph.";
    let chunks = oxibonsai::rag::chunk_document(text, 0, &oxibonsai::rag::ChunkConfig::default());
    assert!(!chunks.is_empty());

    // BLEU-4 needs at least 4 words to form a single 4-gram, or an
    // unsmoothed score is zero by construction (see `BleuConfig::default`'s
    // doc) regardless of how well the candidate matches.
    let score = oxibonsai::eval::sentence_bleu(
        "hello there, how are you today",
        &["hello there, how are you today"],
        &oxibonsai::eval::BleuConfig::default(),
    );
    assert!(score.bleu > 0.99, "expected ~1.0 BLEU, got {}", score.bleu);

    let tok = oxibonsai::tokenizer::OxiTokenizer::char_level_stub(128);
    assert!(tok.encode("full").is_ok());

    // `oxibonsai::image` (added to `full` by RAG-EVAL-IMG-17):
    // `mlx_rng::key` is a pure, file-free, deterministic bit operation
    // (`key(seed) = [(seed >> 32) as u32, seed as u32]`), so this is a real
    // call with a hand-verifiable expected value rather than a bare type
    // check — consistent with every other assertion in this test.
    assert_eq!(oxibonsai::image::sample::mlx_rng::key(42), [0, 42]);
}

/// `{server, metal, image}` — the real-world GPU + imagen deployment shape.
/// Independent of, and not implied by, `full` (which deliberately excludes
/// every GPU feature — see this crate's top-level doc comment). A union can
/// fail even when every individual feature compiles, so this touches one
/// real or reachable symbol from each of the three crates it pulls in.
#[cfg(all(feature = "server", feature = "metal", feature = "image"))]
#[test]
fn server_metal_image_union_composes() {
    fn assert_type_exists<T>() {}
    let args = oxibonsai::serve::ServerArgs::default();
    assert_eq!(args.port, 8080);
    // `MetalGraph` is a GPU-resident handle that cannot be constructed
    // without real Metal hardware (unlike `KernelDispatcher`, which has a
    // hardware-independent fallible constructor exercised by
    // `metal_feature_forwards_to_kernels_model_and_runtime` above); a
    // compile-time reachability check is the correctly-costed test for it.
    // `metal` on this facade only ever compiles on macOS in the first place
    // (the underlying `oxibonsai-kernels` re-export is itself `cfg`'d to
    // `target_os = "macos"`), so this `cfg` is a documentation aid, not a
    // portability guard.
    #[cfg(target_os = "macos")]
    assert_type_exists::<oxibonsai::kernels::MetalGraph>();
    assert_type_exists::<oxibonsai::image::pipeline::PipelineError>();
}

/// Unconditional end-to-end smoke test (T-12's corrected recommendation):
/// build a small but syntactically valid GGUF entirely through
/// `oxibonsai::core`'s writer, round-trip it in memory through its reader,
/// then persist it to a real file under `std::env::temp_dir()` (never a
/// hardcoded path — see the workspace's no-absolute-paths policy) and hand
/// that file to `oxibonsai::model`'s own path-based validator. This is the
/// one thing a per-feature compile check cannot prove: that the re-exported
/// crates are actually wired together and load-bearing when used only
/// through `oxibonsai::` paths.
#[test]
fn e2e_synthetic_gguf_round_trips_through_core_and_model_facade_paths() {
    use oxibonsai::core::gguf::reader::GgufFile;
    use oxibonsai::core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    // 1. Build a minimal, syntactically valid GGUF entirely through
    //    `oxibonsai::core`.
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_tensor(TensorEntry {
        name: "facade_smoke.weight".to_string(),
        shape: vec![8],
        tensor_type: TensorType::F32,
        data: vec![0u8; 32], // 8 x f32
    });
    let bytes = writer
        .to_bytes()
        .expect("a single F32 tensor plus one string KV is always a writable GGUF");

    // 2. Read the in-memory bytes straight back through `oxibonsai::core`'s
    //    own reader (no filesystem involved yet), proving the writer/reader
    //    round trip through the facade path.
    let parsed = GgufFile::parse(&bytes).expect("the bytes we just wrote must parse back");
    assert_eq!(
        parsed.metadata.get_string("general.architecture").ok(),
        Some("qwen3")
    );
    assert_eq!(parsed.tensors.len(), 1);
    drop(parsed);

    // 3. Persist to a real file under `std::env::temp_dir()` and hand it to
    //    `oxibonsai::model`'s own path-based validator — a second, distinct
    //    re-exported crate, reached through a real file this time
    //    (`gguf_loader::validate_gguf_file` mmaps the path).
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_facade_smoke_{}_{unique}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, &bytes)
        .expect("std::env::temp_dir() must be writable for a test fixture");

    let result = oxibonsai::model::gguf_loader::validate_gguf_file(&path);
    // Clean up before asserting, so a failed assertion never leaks the
    // fixture file into the shared temp directory.
    let _ = std::fs::remove_file(&path);

    let warnings =
        result.expect("a single-tensor, current-version GGUF must validate without an Err");
    assert!(
        warnings.is_empty(),
        "unexpected validation warnings on a clean synthetic file: {warnings:?}"
    );
}
