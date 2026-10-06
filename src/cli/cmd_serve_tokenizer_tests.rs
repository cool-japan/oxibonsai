//! The serving tokenizer ladder tests of `cmd_serve.rs`: which tokenizer the
//! server resolves for the three kinds of model file. A sibling of
//! `cmd_serve_tests.rs`, declared there as `mod serving_tokenizer_tests` via
//! `#[path]`, so `super` still names the `tests` module and every test keeps
//! its path.

use super::*;
use crate::cli::test_fixtures::{
    byte_level_tokenizer_json, tokenizer_host_gguf, BYTE_LEVEL_VOCAB, THINKING_TEMPLATE,
    THINK_CLOSE_ID, THINK_OPEN_ID, TOOL_CALL_CLOSE_ID, TOOL_CALL_OPEN_ID,
};
use oxibonsai_runtime::config::ResolvedChatTemplate;

struct Fixture {
    dir: std::path::PathBuf,
    model: String,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A model file (and optionally a `tokenizer.json` beside it) in a
/// fresh directory.
fn fixture(tag: &str, gguf: &[u8], tokenizer_json: bool) -> Fixture {
    let content = tokenizer_json.then(byte_level_tokenizer_json);
    fixture_with_tokenizer_json_content(tag, gguf, content.as_deref())
}

/// [`fixture`] with the on-disk `tokenizer.json`'s exact content
/// controlled by the caller (`None` = write none) — for a fixture whose
/// candidate needs a vocabulary other than [`BYTE_LEVEL_VOCAB`].
fn fixture_with_tokenizer_json_content(
    tag: &str,
    gguf: &[u8],
    tokenizer_json_content: Option<&str>,
) -> Fixture {
    let dir = crate::cli::test_fixtures::scratch_dir(&format!("serve_tok_{tag}"));
    let model = dir.join("Model.gguf");
    std::fs::write(&model, gguf).expect("write model");
    if let Some(content) = tokenizer_json_content {
        std::fs::write(dir.join("tokenizer.json"), content).expect("write tokenizer.json");
    }
    Fixture {
        model: model.to_string_lossy().into_owned(),
        dir,
    }
}

fn load(fix: &Fixture) -> oxibonsai_runtime::TokenizerBridge {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&fix.model)).expect("map");
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
    let (tok, _) = load_serving_tokenizer(None, &fix.model, &gguf).expect("resolve");
    tok.expect("a tokenizer must be found")
}

fn assert_model_ids(tok: &oxibonsai_runtime::TokenizerBridge) {
    assert_eq!(tok.think_open_id(), Some(THINK_OPEN_ID));
    assert_eq!(tok.think_close_id(), Some(THINK_CLOSE_ID));
    assert_eq!(tok.tool_call_open_id(), Some(TOOL_CALL_OPEN_ID));
    assert_eq!(tok.tool_call_close_id(), Some(TOOL_CALL_CLOSE_ID));
}

#[test]
fn embedded_vocabulary_and_template_come_from_the_gguf() {
    // See `tracing_capture_lock`'s doc: every test in this module shares
    // tokenizer-resolution call sites with the capturing test below.
    let _tracing_guard = crate::cli::test_fixtures::tracing_capture_lock();
    let fix = fixture(
        "embedded",
        &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, Some(THINKING_TEMPLATE), true),
        false,
    );
    let tok = load(&fix);
    assert_eq!(tok.vocab_size() as u64, BYTE_LEVEL_VOCAB);
    assert!(matches!(
        tok.resolved_chat_template(),
        ResolvedChatTemplate::Jinja(_)
    ));
    assert_model_ids(&tok);
}

#[test]
fn a_tokenizer_json_gets_the_gguf_template_attached() {
    // See `tracing_capture_lock`'s doc.
    let _tracing_guard = crate::cli::test_fixtures::tracing_capture_lock();
    let fix = fixture(
        "json_plus_template",
        &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, Some(THINKING_TEMPLATE), false),
        true,
    );
    let tok = load(&fix);
    let template = tok.resolved_chat_template();
    assert!(
        matches!(template, ResolvedChatTemplate::Jinja(_)),
        "the GGUF's own template must win over the ChatML fallback"
    );
    // It is THIS template: it opens a think block by default.
    assert!(bonsai2::default_enable_thinking(&template).expect("render"));
    assert_model_ids(&tok);
}

#[test]
fn a_tokenizer_json_with_no_gguf_template_falls_back_to_chatml() {
    // See `tracing_capture_lock`'s doc.
    let _tracing_guard = crate::cli::test_fixtures::tracing_capture_lock();
    let fix = fixture(
        "json_no_template",
        &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, None, false),
        true,
    );
    let tok = load(&fix);
    assert!(matches!(
        tok.resolved_chat_template(),
        ResolvedChatTemplate::Canned(_)
    ));
    assert_model_ids(&tok);
}

#[test]
fn a_vocabulary_mismatch_is_refused_for_an_explicit_tokenizer() {
    // TOK-08 on serve: the model declares 300 rows, the explicit
    // tokenizer has 262 — a different BPE, a hard error.
    // See `tracing_capture_lock`'s doc.
    let _tracing_guard = crate::cli::test_fixtures::tracing_capture_lock();
    let fix = fixture("mismatch", &tokenizer_host_gguf(300, None, false), true);
    let explicit = fix
        .dir
        .join("tokenizer.json")
        .to_string_lossy()
        .into_owned();
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&fix.model)).expect("map");
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
    let err = load_serving_tokenizer(Some(&explicit), &fix.model, &gguf)
        .expect_err("mismatch must be refused");
    assert!(err.to_string().contains("vocabulary mismatch"), "{err}");
}

/// On the 27B-with-a-legacy-`tokenizer.json`-next-to-it shape (an
/// auto-detected candidate that does not fit the model, so
/// the ladder falls back to the GGUF's own embedded vocabulary), one
/// startup that needs THREE independent `TokenizerBridge` instances
/// (router, embedder, RAG — none `Clone`) must still log "resolved chat
/// template" and the tokenizer-source line exactly ONCE each, not once
/// per consumer and not twice within the single resolution itself (the
/// original bug: the losing on-disk candidate's template was resolved
/// and logged, then the embedded fallback's was too).
#[test]
fn one_startup_with_three_tokenizer_consumers_logs_each_line_exactly_once() {
    // Exclusive for the whole test: see `tracing_capture_lock`'s doc.
    let _tracing_guard = crate::cli::test_fixtures::tracing_capture_lock();
    let fix = fixture_with_tokenizer_json_content(
        "resolve_once",
        &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, Some(THINKING_TEMPLATE), true),
        // A legacy on-disk candidate whose vocabulary does NOT match
        // the model's declared (and embedded) BYTE_LEVEL_VOCAB (262) --
        // exactly the shape that sends a real Bonsai 2 serve through
        // the mismatch -> embedded-fallback branch.
        Some(&crate::cli::test_fixtures::tokenizer_json_with_vocab(300)),
    );
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&fix.model)).expect("map");
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");

    let capture = crate::cli::test_fixtures::CapturedEvents::default();
    let tokenizers = tracing::subscriber::with_default(capture.clone(), || {
        // Forces one fresh interest recomputation while `capture` is
        // this thread's current dispatcher; see `tracing_capture_lock`'s
        // doc for what actually makes this deterministic (the lock held
        // above, by this test and every sibling in this module).
        tracing::callsite::rebuild_interest_cache();
        resolve_all_serving_tokenizers(None, &fix.model, &gguf, true, true)
    })
    .expect("resolve");

    // Every consumer (three with the `rag` feature, two without) got its
    // own, independently usable instance of the SAME (embedded-fallback)
    // source.
    let router = tokenizers.router.expect("router tokenizer");
    let embedder = tokenizers.embedder.expect("embedder tokenizer");
    #[cfg(feature = "rag")]
    let rag = tokenizers.rag.expect("rag tokenizer");
    #[cfg(feature = "rag")]
    let rag_ref = Some(&rag);
    #[cfg(not(feature = "rag"))]
    let rag_ref: Option<&oxibonsai_runtime::TokenizerBridge> = None;
    for tok in [Some(&router), Some(&embedder), rag_ref]
        .into_iter()
        .flatten()
    {
        assert_eq!(tok.vocab_size() as u64, BYTE_LEVEL_VOCAB);
        assert!(matches!(
            tok.resolved_chat_template(),
            ResolvedChatTemplate::Jinja(_)
        ));
    }

    let events = capture.events();
    let template_lines = events
        .iter()
        .filter(|e| e.contains("resolved chat template"))
        .count();
    let source_lines = events
        .iter()
        .filter(|e| e.contains("using the tokenizer embedded in the GGUF"))
        .count();
    assert_eq!(
        template_lines, 1,
        "\"resolved chat template\" must log exactly once for the whole startup: {events:?}"
    );
    assert_eq!(
        source_lines, 1,
        "the tokenizer-source line must log exactly once for the whole startup: {events:?}"
    );
}

/// The real Bonsai 2 27B header (`OXI_BONSAI2_PQ2_GGUF`): the serving
/// tokenizer is the GGUF-embedded one, its template is the GGUF's own,
/// and the chat-contract ids are the vocabulary's (design Appendix A.1).
#[test]
fn the_real_27b_file_serves_its_own_template_and_ids() {
    let Some(path) = crate::cli::test_fixtures::env_path(
        "OXI_BONSAI2_PQ2_GGUF",
        "the real Ternary-Bonsai-2-27B-PQ2_0.gguf",
    ) else {
        return;
    };
    let _real = crate::cli::test_fixtures::real_model_lock();
    let model = path.to_string_lossy().into_owned();
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(&path).expect("map");
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
    let (tok, _) = load_serving_tokenizer(None, &model, &gguf).expect("resolve");
    let tok = tok.expect("the 27B embeds its tokenizer");
    assert_eq!(tok.vocab_size(), 248_320);
    assert!(matches!(
        tok.resolved_chat_template(),
        ResolvedChatTemplate::Jinja(_)
    ));
    assert_eq!(tok.think_open_id(), Some(248_068));
    assert_eq!(tok.think_close_id(), Some(248_069));
    assert_eq!(tok.tool_call_open_id(), Some(248_058));
    assert_eq!(tok.tool_call_close_id(), Some(248_059));
    assert!(bonsai2::default_enable_thinking(&tok.resolved_chat_template()).expect("render"));
}
