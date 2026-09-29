//! TOK-08 vocab-aware tokenizer resolution for the standalone
//! `oxibonsai-serve` binary.
//!
//! Loading whatever `tokenizer.json` happens to sit next to a model, with
//! no check that its vocabulary actually matches the model, silently
//! mis-tokenizes every prompt when it does not (the same class of bug
//! TOK-08 also guards against for the CLI's `run`/`chat`/`serve`). This
//! module checks that compatibility before a tokenizer is used, built only
//! on `oxibonsai-runtime`/`oxibonsai-core` public APIs: this crate cannot
//! depend on the `oxibonsai-cli` package (a bin-only crate, and the wrong
//! dependency direction besides).
//!
//! Resolution order, exactly like `run`/`chat`/`oxibonsai serve`:
//!
//! 1. An explicit path (`tokenizer.path`): loaded and hard-checked; a
//!    mismatch is always an error (an explicit override that does not fit
//!    is exactly as dangerous as a bad auto-detected guess).
//! 2. An auto-detected `tokenizer.json` next to the model whose vocabulary
//!    matches: loaded and checked (the check can only succeed here, by
//!    construction).
//! 3. The vocabulary + chat template embedded in the GGUF itself
//!    (`TokenizerBridge::native_from_gguf_metadata`), when it exists and
//!    matches — every Bonsai 2 `qwen35` file carries one.
//! 4. No usable tokenizer at all: [`resolve`] returns `Ok(None)`; the
//!    caller's existing "no tokenizer" handling applies.

use std::path::{Path, PathBuf};

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_runtime::config::ResolvedChatTemplate;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;

/// Where [`resolve`] found its tokenizer — enough to rebuild a further
/// instance of the very same source without repeating the compatibility
/// check (`TokenizerBridge` is not `Clone`; `serve_embedder` needs its own
/// instance of whichever source the router uses).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    /// An on-disk tokenizer file (already validated) at this path.
    File(PathBuf),
    /// The vocabulary + chat template embedded in the GGUF itself.
    GgufEmbedded,
}

/// The model's authoritative vocabulary size: the `token_embd.weight`
/// tensor's row count (the last GGUF dimension), authoritative even when
/// metadata disagrees.
///
/// # Errors
///
/// The GGUF carries no `token_embd.weight` tensor, or its shape is
/// degenerate (a zero-length last dimension).
fn model_vocab_size(gguf: &GgufFile<'_>) -> Result<usize, String> {
    let info = gguf
        .tensors
        .require("token_embd.weight")
        .map_err(|e| format!("cannot determine the model's vocabulary size: {e}"))?;
    match info.shape.last() {
        Some(&last) if last > 0 => Ok(last as usize),
        _ => Err("token_embd.weight has a degenerate (zero-length) shape".to_string()),
    }
}

/// TOK-08: hard-error a loaded tokenizer that does not fit `gguf`'s model.
///
/// * Tokenizer vocab EXCEEDS the model's: always a hard error (it can emit
///   token ids outside the embedding table).
/// * Tokenizer vocab is SMALLER than the model's: also a hard error here —
///   the standalone binary has no `--allow-vocab-mismatch` escape hatch
///   (unlike the CLI's `run`/`chat`), so a smaller vocabulary (a different
///   BPE, not a subset) is never silently accepted.
/// * The model's declared BOS/EOS token ids (when present in GGUF
///   metadata) must be within the tokenizer's vocabulary range.
///
/// # Errors
///
/// A vocabulary mismatch or an out-of-range BOS/EOS id, naming both values.
fn check_compatibility(
    tok: &TokenizerBridge,
    tokenizer_path: &str,
    gguf: &GgufFile<'_>,
) -> Result<(), String> {
    let tok_vocab = tok.vocab_size();
    let model_vocab = model_vocab_size(gguf)?;

    if tok_vocab > model_vocab {
        return Err(format!(
            "tokenizer/model vocabulary mismatch: tokenizer '{tokenizer_path}' has \
             vocab_size={tok_vocab}, which EXCEEDS the model's vocab_size={model_vocab}. It can \
             emit token ids the model's embedding table cannot index."
        ));
    }
    if tok_vocab < model_vocab {
        return Err(format!(
            "tokenizer/model vocabulary mismatch: tokenizer '{tokenizer_path}' has \
             vocab_size={tok_vocab}, smaller than the model's vocab_size={model_vocab}. This is \
             a different BPE, not a subset, and will silently mis-tokenize every prompt."
        ));
    }

    for key in [keys::TOKENIZER_BOS_TOKEN_ID, keys::TOKENIZER_EOS_TOKEN_ID] {
        if let Ok(id) = gguf.metadata.get_u32(key) {
            if id as usize >= tok_vocab {
                return Err(format!(
                    "tokenizer/model incompatibility: the model's GGUF declares {key}={id}, \
                     which is out of range for tokenizer '{tokenizer_path}' \
                     (vocab_size={tok_vocab}); this tokenizer does not match this model."
                ));
            }
        }
    }

    Ok(())
}

/// The tokenizer embedded in `gguf`'s `tokenizer.ggml.*` metadata, with its
/// chat template attached, when it exists and its vocabulary matches
/// `expected_vocab`. `None` for "nothing usable embedded" (no
/// `tokenizer.ggml.tokens` at all, a vocabulary that does not match, or a
/// chat template this engine's Jinja subset cannot compile) — the first
/// case is the common one for a dense model that ships its vocabulary only
/// as a sibling `tokenizer.json`, so only the latter two log a warning.
fn gguf_embedded_tokenizer(gguf: &GgufFile<'_>, expected_vocab: usize) -> Option<TokenizerBridge> {
    gguf.metadata
        .get_string_array(keys::TOKENIZER_TOKENS)
        .ok()?;
    let tok = match TokenizerBridge::native_from_gguf_metadata(&gguf.metadata) {
        Ok(tok) => tok,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "the GGUF carries an embedded tokenizer.ggml.tokens vocabulary that failed to \
                 build"
            );
            return None;
        }
    };
    if tok.vocab_size() == expected_vocab {
        Some(tok)
    } else {
        tracing::warn!(
            embedded_vocab = tok.vocab_size(),
            expected_vocab,
            "the GGUF's embedded tokenizer vocabulary does not match the model's own \
             token_embd row count; ignoring it"
        );
        None
    }
}

/// Resolve the standalone binary's serving tokenizer (TOK-08): `explicit`
/// wins when given (a hard error if it does not fit); otherwise an
/// auto-detected `tokenizer.json` candidate that fits `model_path`; else
/// the GGUF's own embedded vocabulary.
///
/// # Errors
///
/// An explicit tokenizer path that fails to load or does not fit the
/// model, or an auto-detected candidate that loads but fails the
/// compatibility check with no usable embedded fallback.
pub fn resolve(
    explicit: Option<&Path>,
    model_path: &Path,
    gguf: &GgufFile<'_>,
) -> Result<Option<(TokenizerBridge, TokenizerSource)>, String> {
    let expected_vocab = model_vocab_size(gguf).ok();

    if let Some(path) = explicit {
        let path_str = path.display().to_string();
        let tok = TokenizerBridge::native_from_file(&path_str)
            .map_err(|e| format!("failed to load tokenizer '{path_str}': {e}"))?;
        check_compatibility(&tok, &path_str, gguf)?;
        let template = ResolvedChatTemplate::from_gguf(&gguf.metadata)
            .map_err(|e| format!("failed to compile the GGUF's own chat template: {e}"))?;
        tracing::info!(path = %path_str, "tokenizer loaded");
        return Ok(Some((
            tok.with_chat_template(template),
            TokenizerSource::File(path.to_path_buf()),
        )));
    }

    if let Some(candidate) = auto_detect_candidate(model_path, expected_vocab) {
        let candidate_str = candidate.display().to_string();
        match TokenizerBridge::native_from_file(&candidate_str) {
            Ok(tok) => match check_compatibility(&tok, &candidate_str, gguf) {
                Ok(()) => {
                    let template =
                        ResolvedChatTemplate::from_gguf(&gguf.metadata).map_err(|e| {
                            format!("failed to compile the GGUF's own chat template: {e}")
                        })?;
                    tracing::info!(path = %candidate_str, "auto-detected tokenizer alongside model");
                    return Ok(Some((
                        tok.with_chat_template(template),
                        TokenizerSource::File(candidate),
                    )));
                }
                Err(mismatch) => {
                    if let Some(expected) = expected_vocab {
                        if let Some(embedded) = gguf_embedded_tokenizer(gguf, expected) {
                            tracing::info!(
                                skipped = %candidate_str,
                                reason = %mismatch,
                                "the auto-detected tokenizer does not fit this model; using the \
                                 tokenizer embedded in the GGUF instead"
                            );
                            return Ok(Some((embedded, TokenizerSource::GgufEmbedded)));
                        }
                    }
                    return Err(mismatch);
                }
            },
            Err(e) => {
                // An auto-detected candidate that exists but fails to parse
                // is left for the embedded-vocabulary fallback below,
                // exactly like one that does not exist at all -- but this
                // case gets a warning naming the file, since a stray
                // unparsable `tokenizer.json` next to a model is worth the
                // operator's attention.
                tracing::warn!(
                    path = %candidate_str,
                    error = %e,
                    "an auto-detected tokenizer.json exists but failed to parse; ignoring it"
                );
            }
        }
    }

    // No on-disk candidate at all (or it failed to parse): the GGUF's own
    // embedded vocabulary, if it has one that matches.
    match expected_vocab.and_then(|expected| gguf_embedded_tokenizer(gguf, expected)) {
        Some(embedded) => {
            tracing::info!(
                vocab = expected_vocab,
                "using the tokenizer embedded in the GGUF"
            );
            Ok(Some((embedded, TokenizerSource::GgufEmbedded)))
        }
        None => Ok(None),
    }
}

/// Every path [`resolve`] would probe for an auto-detected candidate next
/// to `model_path`, for a "no tokenizer found" message that names exactly
/// where it looked (the caller's job — this module never formats a
/// multi-line message itself).
#[must_use]
pub fn searched_candidates(model_path: &Path) -> Vec<PathBuf> {
    tokenizer_candidates(model_path)
}

/// A further `TokenizerBridge` instance of the very same source [`resolve`]
/// already resolved and validated once (`TokenizerBridge` is not `Clone`;
/// the embedder needs its own instance). Never re-logs and never re-runs
/// the compatibility check — the source already passed it.
///
/// # Errors
///
/// The on-disk file becoming unreadable between the two loads, or (for the
/// embedded source) the GGUF's chat template failing to compile — both
/// already-impossible-in-practice regressions [`resolve`] would have
/// surfaced first.
pub fn rebuild(source: &TokenizerSource, gguf: &GgufFile<'_>) -> Result<TokenizerBridge, String> {
    match source {
        TokenizerSource::File(path) => {
            let path_str = path.display().to_string();
            let tok = TokenizerBridge::native_from_file(&path_str)
                .map_err(|e| format!("failed to load tokenizer '{path_str}': {e}"))?;
            let template = ResolvedChatTemplate::from_gguf(&gguf.metadata)
                .map_err(|e| format!("failed to compile the GGUF's own chat template: {e}"))?;
            Ok(tok.with_chat_template(template))
        }
        TokenizerSource::GgufEmbedded => TokenizerBridge::native_from_gguf_metadata(&gguf.metadata)
            .map_err(|e| format!("failed to rebuild the GGUF-embedded tokenizer: {e}")),
    }
}

/// Strip a trailing GGUF quantization suffix (e.g. `-Q2_0`, `-Q4_K_M`,
/// `-F16`, `-BF16`, `-F32`) from a model basename, so a sibling directory
/// named after the unquantized model is still found.
fn strip_quant_suffix(basename: &str) -> &str {
    let Some(dash_pos) = basename.rfind('-') else {
        return basename;
    };
    let suffix = &basename[dash_pos + 1..];
    if suffix.is_empty() {
        return basename;
    }
    let is_float = matches!(suffix, "F16" | "BF16" | "F32");
    let is_quant = {
        let mut chars = suffix.chars();
        match chars.next() {
            Some('Q') => {
                let rest: String = chars.collect();
                if rest.is_empty() {
                    false
                } else {
                    let mut parts = rest.split('_');
                    let first = parts.next().unwrap_or("");
                    if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
                        false
                    } else {
                        parts.all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric()))
                    }
                }
            }
            _ => false,
        }
    };
    if is_float || is_quant {
        &basename[..dash_pos]
    } else {
        basename
    }
}

/// Every candidate `tokenizer.json` path considered for auto-detection next
/// to `model_path`: its own directory, that directory's parent, a handful
/// of basename-derived sibling directories, and the nearest ancestor
/// literally named `models`. Duplicates are removed so a "not found"
/// message stays compact.
fn tokenizer_candidates(model_path: &Path) -> Vec<PathBuf> {
    let parent = model_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut out: Vec<PathBuf> = Vec::new();
    let mut push_unique = |p: PathBuf| {
        if !out.iter().any(|existing| existing == &p) {
            out.push(p);
        }
    };

    push_unique(parent.join("tokenizer.json"));
    push_unique(parent.join("..").join("tokenizer.json"));

    if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
        let base = strip_quant_suffix(stem);
        for variant in [
            base.to_string(),
            format!("{base}-unpacked"),
            format!("{base}-ONNX"),
        ] {
            push_unique(parent.join(&variant).join("tokenizer.json"));
        }
    }

    for ancestor in model_path.ancestors().skip(1) {
        if ancestor.file_name().and_then(|n| n.to_str()) == Some("models") {
            push_unique(ancestor.join("tokenizer.json"));
            break;
        }
    }

    out
}

/// The first auto-detection candidate that exists on disk and, when
/// `expected_vocab` is known, whose OWN vocabulary matches it (skipping a
/// stray `tokenizer.json` left over from a different model in favor of one
/// that actually fits) — else the first that merely exists, so the
/// downstream compatibility check still produces a concrete, actionable
/// error instead of "not found".
fn auto_detect_candidate(model_path: &Path, expected_vocab: Option<usize>) -> Option<PathBuf> {
    let candidates = tokenizer_candidates(model_path);
    let Some(expected) = expected_vocab else {
        return candidates.into_iter().find(|c| c.exists());
    };
    let mut first_existing = None;
    for candidate in candidates {
        if !candidate.exists() {
            continue;
        }
        if first_existing.is_none() {
            first_existing = Some(candidate.clone());
        }
        if let Ok(tok) = TokenizerBridge::native_from_file(&candidate.display().to_string()) {
            if tok.vocab_size() == expected {
                return Some(candidate);
            }
        }
    }
    first_existing
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_testkit::dense_fixture::{weighted_dense_gguf, VOCAB};
    use oxibonsai_testkit::gguf_fixture::{FixtureQuant, GgufFixtureBuilder};

    /// A tiny GGUF that embeds a real `tokenizer.ggml.*` vocabulary (a
    /// byte-level alphabet, `vocab_size` entries) and, optionally, a chat
    /// template — covers the "embedded vocabulary resolves" leg of the
    /// TOK-08 ladder, since neither shared testkit fixture carries a full
    /// embedded vocabulary ([`weighted_dense_gguf`] embeds none at all;
    /// `qwen35_fixture` carries only an `eos_token_id`). Declares
    /// `general.architecture = "qwen3"` rather than `qwen35`: [`resolve`]
    /// and [`rebuild`] read only the tokenizer/vocabulary metadata, not the
    /// architecture, so the ladder behaves identically on either.
    fn embedded_vocab_gguf(vocab_size: u32, chat_template: Option<&str>) -> Vec<u8> {
        let mut b = GgufFixtureBuilder::new();
        b.metadata_str("general.architecture", "qwen3")
            .metadata_str("general.name", "TokenizerLadderEmbeddedFixture")
            .metadata_u32("qwen3.embedding_length", 8)
            .metadata_u32("qwen3.block_count", 1)
            .metadata_u32("qwen3.vocab_size", vocab_size);
        if let Some(template) = chat_template {
            b.metadata_str("tokenizer.chat_template", template);
        }
        let tokens: Vec<String> = (0..vocab_size).map(|i| format!("t{i}")).collect();
        let types: Vec<i32> = vec![1; tokens.len()];
        b.metadata_str("tokenizer.ggml.model", "gpt2")
            .metadata(
                "tokenizer.ggml.tokens",
                oxibonsai_core::gguf::writer::MetadataWriteValue::ArrayStr(tokens),
            )
            .metadata(
                "tokenizer.ggml.token_type",
                oxibonsai_core::gguf::writer::MetadataWriteValue::ArrayI32(types),
            )
            .metadata(
                "tokenizer.ggml.merges",
                oxibonsai_core::gguf::writer::MetadataWriteValue::ArrayStr(Vec::new()),
            )
            .metadata_u32("tokenizer.ggml.eos_token_id", vocab_size - 1);
        b.tensor(
            "token_embd.weight",
            &[8, vocab_size as u64],
            FixtureQuant::F32,
            1,
        )
        .expect("token_embd.weight");
        b.build().expect("serialize the embedded-vocab fixture")
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_tokenizer_ladder_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// A `tokenizer.json` of exactly `n` plain entries, no added tokens —
    /// enough for a vocab-size compatibility check.
    fn tokenizer_json_with_vocab(n: u64) -> String {
        let mut vocab = serde_json::Map::new();
        for id in 0..n {
            vocab.insert(format!("t{id}"), serde_json::Value::from(id));
        }
        serde_json::json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": [] },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false
            }
        })
        .to_string()
    }

    fn parse<'a>(bytes: &'a [u8]) -> GgufFile<'a> {
        GgufFile::parse(bytes).expect("parse")
    }

    #[test]
    fn an_auto_detected_matching_tokenizer_is_used() {
        let dir = scratch_dir("match");
        let model = dir.join("model.gguf");
        let gguf_bytes = weighted_dense_gguf();
        std::fs::write(&model, &gguf_bytes).expect("write model");
        std::fs::write(
            dir.join("tokenizer.json"),
            tokenizer_json_with_vocab(VOCAB as u64),
        )
        .expect("write tokenizer.json");
        let gguf = parse(&gguf_bytes);

        let (tok, source) = resolve(None, &model, &gguf)
            .expect("resolve")
            .expect("a tokenizer was found");
        assert_eq!(tok.vocab_size(), VOCAB);
        assert!(matches!(source, TokenizerSource::File(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With no on-disk candidate at all, the GGUF's own embedded
    /// vocabulary AND chat template are resolved.
    #[test]
    fn an_embedded_vocab_fixture_resolves_the_embedded_tokenizer_and_its_template() {
        let dir = scratch_dir("embedded");
        let model = dir.join("model.gguf");
        let template = "{{ messages[0].content }}";
        let gguf_bytes = embedded_vocab_gguf(64, Some(template));
        std::fs::write(&model, &gguf_bytes).expect("write model");
        let gguf = parse(&gguf_bytes);

        let (tok, source) = resolve(None, &model, &gguf)
            .expect("resolve")
            .expect("the embedded vocabulary is used");
        assert_eq!(tok.vocab_size(), 64);
        assert_eq!(source, TokenizerSource::GgufEmbedded);
        assert!(
            matches!(tok.resolved_chat_template(), ResolvedChatTemplate::Jinja(_)),
            "the GGUF's own chat_template must be resolved, not the ChatML fallback"
        );

        // A further instance (the embedder's own) rebuilds identically,
        // quietly.
        let second = rebuild(&source, &gguf).expect("rebuild");
        assert_eq!(second.vocab_size(), 64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An explicit `--tokenizer`-equivalent path that does not
    /// fit is a hard error even when the GGUF carries a usable embedded
    /// fallback — an explicit override is never silently second-guessed.
    #[test]
    fn an_explicit_mismatch_stays_a_hard_error_even_with_an_embedded_fallback_available() {
        let dir = scratch_dir("explicit_mismatch_with_fallback");
        let model = dir.join("model.gguf");
        let gguf_bytes = embedded_vocab_gguf(64, None);
        std::fs::write(&model, &gguf_bytes).expect("write model");
        let explicit = dir.join("wrong.json");
        std::fs::write(&explicit, tokenizer_json_with_vocab(65)).expect("write tokenizer.json");
        let gguf = parse(&gguf_bytes);

        let err = resolve(Some(&explicit), &model, &gguf)
            .expect_err("an explicit mismatch is always refused, embedded fallback or not");
        assert!(err.contains("vocabulary mismatch"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_explicit_mismatch_is_a_hard_error() {
        let dir = scratch_dir("explicit_mismatch");
        let model = dir.join("model.gguf");
        let gguf_bytes = weighted_dense_gguf();
        std::fs::write(&model, &gguf_bytes).expect("write model");
        let explicit = dir.join("wrong.json");
        std::fs::write(&explicit, tokenizer_json_with_vocab(VOCAB as u64 + 1))
            .expect("write tokenizer.json");
        let gguf = parse(&gguf_bytes);

        let err = resolve(Some(&explicit), &model, &gguf).expect_err("mismatch must be refused");
        assert!(err.contains("vocabulary mismatch"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_candidate_and_no_embedded_vocabulary_is_none() {
        let dir = scratch_dir("none");
        let model = dir.join("model.gguf");
        let gguf_bytes = weighted_dense_gguf();
        std::fs::write(&model, &gguf_bytes).expect("write model");
        let gguf = parse(&gguf_bytes);

        // `weighted_dense_gguf` carries no embedded `tokenizer.ggml.*`
        // vocabulary and no on-disk candidate exists next to it.
        let resolved = resolve(None, &model, &gguf).expect("resolve");
        assert!(resolved.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_vocab_size_matches_the_fixtures_declared_vocab() {
        let gguf_bytes = weighted_dense_gguf();
        let gguf = parse(&gguf_bytes);
        assert_eq!(model_vocab_size(&gguf).expect("vocab"), VOCAB);
    }

    #[test]
    fn rebuild_of_a_file_source_produces_a_second_usable_instance() {
        let dir = scratch_dir("rebuild_file");
        let model = dir.join("model.gguf");
        let gguf_bytes = weighted_dense_gguf();
        std::fs::write(&model, &gguf_bytes).expect("write model");
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, tokenizer_json_with_vocab(VOCAB as u64)).expect("write tokenizer");
        let gguf = parse(&gguf_bytes);

        let (_first, source) = resolve(Some(&path), &model, &gguf)
            .expect("resolve")
            .expect("found");
        let second = rebuild(&source, &gguf).expect("rebuild");
        assert_eq!(second.vocab_size(), VOCAB);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
