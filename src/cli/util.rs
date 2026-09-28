//! Shared helpers used by several subcommands: stdin prompt reading,
//! tokenizer auto-detection + tokenizer/model compatibility, config-file
//! helpers, memory estimation, and GGUF tensor dequantization for
//! `quantize`.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Maximum prompt size accepted from stdin (`run --prompt -` / `image
/// --prompt -`): 16 MiB. Far larger than any real prompt, but small enough
/// that a misdirected binary stream (e.g. `cat video.mp4 | oxibonsai run
/// --prompt -`) fails fast with a clear message instead of silently
/// truncating (cli-M4) or attempting to tokenize gigabytes of data.
const MAX_STDIN_PROMPT_BYTES: usize = 16 * 1024 * 1024;

/// Read a prompt from stdin in full.
///
/// Reads to EOF rather than silently stopping at the first I/O error or
/// invalid-UTF-8 byte (cli-M4's "silently truncates" defect): any read
/// failure or non-UTF-8 input is a propagated error, and an over-long
/// input is rejected naming the byte limit, instead of returning a
/// confidently-wrong, silently-shortened prompt.
pub(crate) fn read_prompt_stdin() -> anyhow::Result<String> {
    let mut buf = Vec::new();
    io::stdin()
        .lock()
        .take(MAX_STDIN_PROMPT_BYTES as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| anyhow::anyhow!("failed to read prompt from stdin: {e}"))?;

    if buf.len() > MAX_STDIN_PROMPT_BYTES {
        anyhow::bail!(
            "prompt read from stdin exceeds the {MAX_STDIN_PROMPT_BYTES}-byte limit; \
             pass a shorter prompt, or use --prompt <text> directly"
        );
    }

    let text = String::from_utf8(buf)
        .map_err(|e| anyhow::anyhow!("prompt read from stdin is not valid UTF-8: {e}"))?;

    if text.trim().is_empty() {
        anyhow::bail!("prompt is empty: pass --prompt <text>, or pipe non-empty text on stdin");
    }

    Ok(text)
}

/// Result of attempting to locate a `tokenizer.json` for a given model.
///
/// `found` is `Some(path)` when either an explicit override was supplied
/// or auto-detection succeeded.  `searched` lists every candidate path
/// inspected during auto-detection so the user can see exactly where we
/// looked when nothing turned up.
pub(crate) struct TokenizerLookup {
    pub(crate) found: Option<String>,
    pub(crate) searched: Vec<PathBuf>,
}

/// Strip a trailing GGUF quantization suffix (e.g. `-Q2_0`, `-Q4_K_M`,
/// `-F16`, `-BF16`, `-F32`) from a model basename without pulling in the
/// `regex` crate.  Returns the basename unchanged when no recognized
/// suffix is present.
pub(crate) fn strip_quant_suffix(basename: &str) -> &str {
    // Locate the last '-' segment; only that segment is a quant suffix
    // candidate.
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
                // Accept Q<digits>(_<alnum>+)*  e.g. Q1_0, Q2_K, Q4_K_M, Q8_0.
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

/// Build the ordered list of candidate `tokenizer.json` paths to probe
/// for a given model file.  Duplicates (after canonical lexical form)
/// are removed so the warning message stays compact.
pub(crate) fn tokenizer_candidates(model_path: &Path) -> Vec<PathBuf> {
    let parent = model_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut out: Vec<PathBuf> = Vec::new();
    let push_unique = |p: PathBuf, out: &mut Vec<PathBuf>| {
        if !out.iter().any(|existing| existing == &p) {
            out.push(p);
        }
    };

    // 1. Same directory as the model.
    push_unique(parent.join("tokenizer.json"), &mut out);

    // 2. Parent of the model directory.
    push_unique(parent.join("..").join("tokenizer.json"), &mut out);

    // 3. Sibling directories derived from the model basename.
    if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
        let base = strip_quant_suffix(stem);
        for variant in [
            base.to_string(),
            format!("{base}-unpacked"),
            format!("{base}-ONNX"),
        ] {
            push_unique(parent.join(&variant).join("tokenizer.json"), &mut out);
        }
    }

    // 4. Top-level `models/` directory if the model lives anywhere under it.
    for ancestor in model_path.ancestors().skip(1) {
        if ancestor.file_name().and_then(|n| n.to_str()) == Some("models") {
            push_unique(ancestor.join("tokenizer.json"), &mut out);
            break;
        }
    }

    out
}

/// Resolve the tokenizer path: use the explicit path if given, otherwise
/// auto-detect `tokenizer.json` in a small set of conventional locations
/// derived from the model path.
///
/// This is the vocab-*unaware* resolver: it returns the first candidate
/// that exists on disk, full stop. Kept unchanged (rather than folded into
/// [`resolve_tokenizer_vocab_aware`]) because it is still the correct,
/// minimal building block for callers with no model loaded yet to compare
/// against. Callers that DO have a model loaded should prefer
/// [`resolve_tokenizer_vocab_aware`] and, regardless of which resolver was
/// used, must still run [`check_tokenizer_model_compatibility`] once the
/// tokenizer is loaded (TOK-08) — resolution and the hard compatibility
/// check are deliberately separate so an explicit `--tokenizer` override
/// is checked too, not just auto-detected candidates.
pub(crate) fn resolve_tokenizer(tokenizer: Option<&str>, model_path: &str) -> TokenizerLookup {
    if let Some(p) = tokenizer {
        return TokenizerLookup {
            found: Some(p.to_string()),
            searched: Vec::new(),
        };
    }

    let model = Path::new(model_path);
    let candidates = tokenizer_candidates(model);
    for candidate in &candidates {
        if candidate.exists() {
            tracing::info!(
                path = %candidate.display(),
                "auto-detected tokenizer alongside model"
            );
            return TokenizerLookup {
                found: Some(candidate.to_string_lossy().into_owned()),
                searched: candidates,
            };
        }
    }

    TokenizerLookup {
        found: None,
        searched: candidates,
    }
}

/// Vocab-aware tokenizer resolution (TOK-08 / cli-M2).
///
/// Identical to [`resolve_tokenizer`] for an explicit `--tokenizer`
/// override (never second-guessed by auto-detection) or when
/// `expected_vocab_size` is unknown. When auto-detecting with a known
/// expected vocabulary size, walks the same candidate list as
/// [`tokenizer_candidates`] but skips a candidate whose own vocabulary
/// size does not match and continues down the list — so a stray
/// `models/tokenizer.json` left over from a different model does not
/// shadow the correct per-model tokenizer the moment a vocab-matching one
/// is also on the search path. If no candidate matches, falls back to the
/// first one that exists (rather than reporting "not found") so the
/// caller's mandatory [`check_tokenizer_model_compatibility`] call still
/// produces a concrete, actionable error naming the mismatch instead of a
/// vaguer "no tokenizer found" message.
pub(crate) fn resolve_tokenizer_vocab_aware(
    tokenizer: Option<&str>,
    model_path: &str,
    expected_vocab_size: Option<usize>,
) -> TokenizerLookup {
    if tokenizer.is_some() {
        return resolve_tokenizer(tokenizer, model_path);
    }
    let Some(expected) = expected_vocab_size else {
        return resolve_tokenizer(None, model_path);
    };

    let model = Path::new(model_path);
    let candidates = tokenizer_candidates(model);
    let mut first_existing: Option<String> = None;

    for candidate in &candidates {
        if !candidate.exists() {
            continue;
        }
        let candidate_str = candidate.to_string_lossy().into_owned();
        if first_existing.is_none() {
            first_existing = Some(candidate_str.clone());
        }
        // Best-effort vocab peek via the always-available native backend.
        // A candidate that fails to parse here is simply skipped in favor
        // of one that parses and matches; the mandatory hard compatibility
        // check downstream still runs on whatever is ultimately returned.
        match oxibonsai_runtime::TokenizerBridge::native_from_file(&candidate_str) {
            Ok(tok) if tok.vocab_size() == expected => {
                tracing::info!(
                    path = %candidate_str,
                    vocab = tok.vocab_size(),
                    "auto-detected vocab-matching tokenizer"
                );
                return TokenizerLookup {
                    found: Some(candidate_str),
                    searched: candidates,
                };
            }
            Ok(tok) => {
                tracing::warn!(
                    path = %candidate_str,
                    candidate_vocab = tok.vocab_size(),
                    expected_vocab = expected,
                    "skipping auto-detected tokenizer candidate: vocab size does not match \
                     this model; continuing to search"
                );
            }
            Err(e) => {
                tracing::debug!(path = %candidate_str, error = %e, "candidate tokenizer failed to parse");
            }
        }
    }

    match first_existing {
        Some(path) => {
            tracing::warn!(
                path = %path,
                expected_vocab = expected,
                "no auto-detected tokenizer candidate matched the model's vocab size; \
                 using the first one found (the compatibility check will still catch a \
                 real mismatch)"
            );
            TokenizerLookup {
                found: Some(path),
                searched: candidates,
            }
        }
        None => TokenizerLookup {
            found: None,
            searched: candidates,
        },
    }
}

/// TOK-08 for `oxibonsai serve` (wave-3.5 addendum item 6): the SAME
/// vocab-aware resolution + hard tokenizer/model compatibility check `run`
/// and `chat` apply ([`resolve_tokenizer_vocab_aware`] then
/// [`check_tokenizer_model_compatibility`] via
/// `cmd_run::resolve_model_tokenizer`, which also attaches the GGUF's own
/// chat template and falls back to a vocabulary-matching tokenizer embedded
/// in the GGUF when the auto-detected file does not fit), instead of the
/// vocab-unaware first-existing-candidate ladder `serve` used to run. An
/// explicit `--tokenizer` that does not fit the model is a hard error.
///
/// Returns the tokenizer (`None` = there is no usable tokenizer at all) and
/// the lookup, whose `searched` list feeds [`missing_tokenizer_warning`].
///
/// # Errors
///
/// A tokenizer file that cannot be loaded, a vocabulary mismatch, or a GGUF
/// chat template that fails to compile.
#[cfg(feature = "server")]
pub(crate) fn resolve_serving_tokenizer(
    explicit: Option<&str>,
    model_path: &str,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
) -> anyhow::Result<(Option<oxibonsai_runtime::TokenizerBridge>, TokenizerLookup)> {
    let expected_vocab = model_vocab_size(gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(explicit, model_path, expected_vocab);
    let tok = super::cmd_run::resolve_model_tokenizer(
        explicit,
        &lookup,
        gguf,
        expected_vocab,
        super::tokenizer_backend::TokenizerBackendChoice::Auto,
        false,
    )?;
    Ok((tok, lookup))
}

/// Build the multi-line "no tokenizer found" warning shown by the run /
/// chat / serve paths.  Centralized so all three surfaces stay in sync.
pub(crate) fn missing_tokenizer_warning(searched: &[PathBuf]) -> String {
    let mut msg = String::from("no tokenizer found. Searched:\n");
    if searched.is_empty() {
        msg.push_str("  (no candidate paths — model path was not provided)\n");
    } else {
        for path in searched {
            msg.push_str(&format!("  - {}\n", path.display()));
        }
    }
    msg.push_str("To fix:\n");
    msg.push_str("  - Pass --tokenizer <path/to/tokenizer.json>, OR\n");
    msg.push_str(
        "  - Run ./scripts/download_tokenizer.sh to fetch the Qwen3 tokenizer to models/tokenizer.json\n",
    );
    msg.push_str("Continuing with raw token IDs in output.");
    msg
}

// ──────────────────────────────────────────────────────────────────────────
// TOK-08: tokenizer / model vocabulary compatibility
// ──────────────────────────────────────────────────────────────────────────

/// The model's authoritative vocabulary size.
///
/// Prefers the `token_embd.weight` tensor's row count (the last GGUF
/// dimension), which is authoritative even when metadata disagrees;
/// falls back to `Qwen3Config::from_metadata`'s `vocab_size` when the
/// tensor is absent (e.g. a non-Qwen3 file that still exposes usable
/// metadata) or its shape is degenerate.
pub(crate) fn model_vocab_size(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
) -> anyhow::Result<usize> {
    if let Ok(info) = gguf.tensors.require("token_embd.weight") {
        if let Some(&last) = info.shape.last() {
            if last > 0 {
                return Ok(last as usize);
            }
        }
    }
    let config = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)?;
    Ok(config.vocab_size)
}

/// Hard-error when a loaded tokenizer is incompatible with `gguf`'s model
/// (TOK-08). Checked regardless of whether the tokenizer path came from an
/// explicit `--tokenizer` flag or auto-detection — an explicit override
/// that names the wrong tokenizer is exactly as dangerous as a bad guess.
///
/// * Tokenizer vocab EXCEEDS the model's: always a hard error (it can emit
///   token ids outside the embedding table).
/// * Tokenizer vocab is SMALLER than the model's: a hard error unless
///   `allow_vocab_mismatch` is `true` (a smaller vocab is a different BPE,
///   not a subset).
/// * The model's declared BOS/EOS token ids (when present in GGUF
///   metadata) must be within the tokenizer's vocabulary range.
pub(crate) fn check_tokenizer_model_compatibility(
    tok: &oxibonsai_runtime::TokenizerBridge,
    tokenizer_path: &str,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    allow_vocab_mismatch: bool,
) -> anyhow::Result<()> {
    let tok_vocab = tok.vocab_size();
    let model_vocab = model_vocab_size(gguf)?;

    if tok_vocab > model_vocab {
        anyhow::bail!(
            "tokenizer/model vocabulary mismatch: tokenizer '{tokenizer_path}' has \
             vocab_size={tok_vocab}, which EXCEEDS the model's vocab_size={model_vocab}. It can \
             emit token ids the model's embedding table cannot index. Pass the correct \
             --tokenizer for this model."
        );
    }
    if tok_vocab < model_vocab && !allow_vocab_mismatch {
        anyhow::bail!(
            "tokenizer/model vocabulary mismatch: tokenizer '{tokenizer_path}' has \
             vocab_size={tok_vocab}, smaller than the model's vocab_size={model_vocab}. This is \
             a different BPE, not a subset, and will silently mis-tokenize every prompt. Pass \
             the correct --tokenizer for this model, or --allow-vocab-mismatch to proceed \
             anyway at your own risk."
        );
    }

    for key in [
        oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_BOS_TOKEN_ID,
        oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_EOS_TOKEN_ID,
    ] {
        if let Ok(id) = gguf.metadata.get_u32(key) {
            if id as usize >= tok_vocab {
                anyhow::bail!(
                    "tokenizer/model incompatibility: the model's GGUF declares {key}={id}, \
                     which is out of range for tokenizer '{tokenizer_path}' \
                     (vocab_size={tok_vocab}); this tokenizer does not match this model."
                );
            }
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// cli-17: stop-sequence matching for `run`/`chat`
// ──────────────────────────────────────────────────────────────────────────

/// Detects and truncates text at stop sequences.
///
/// Deliberately not `oxibonsai_runtime::api_extensions::StopChecker`: that
/// type is gated behind the runtime's `server` feature, but `--stop` on
/// `run`/`chat` must work in a `--no-default-features` (no `server`)
/// build too. This mirrors its exact algorithm (whole-accumulated-buffer
/// `contains`/`find`, earliest match wins) rather than reusing a second,
/// differently-shaped implementation — the "second naive implementation"
/// concern the cli-17 fix warns against is specifically about a
/// chunk-boundary-unsafe per-token check, which this avoids the same way
/// the server's checker does: callers always pass the *whole* text
/// accumulated so far, never just the newest chunk (see
/// `cmd_run`/`cmd_chat`'s decode loops).
pub(crate) struct StopChecker {
    sequences: Vec<String>,
}

impl StopChecker {
    pub(crate) fn new(sequences: Vec<String>) -> Self {
        Self { sequences }
    }

    /// `true` when a stop sequence is present anywhere in `text`.
    pub(crate) fn check(&self, text: &str) -> bool {
        self.sequences.iter().any(|seq| text.contains(seq.as_str()))
    }

    /// Truncate `text` at the earliest stop-sequence match, if any.
    pub(crate) fn truncate_at_stop(&self, text: &str) -> String {
        let mut earliest: Option<usize> = None;
        for seq in &self.sequences {
            if let Some(pos) = text.find(seq.as_str()) {
                earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
            }
        }
        match earliest {
            Some(pos) => text[..pos].to_string(),
            None => text.to_string(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Shared sampling-parameter construction (orchestrator P0 addendum)
// ──────────────────────────────────────────────────────────────────────────

/// Reject `--grammar`/`--stop` when combined with a non-default sampling
/// penalty.
///
/// The buffered constrained/stop loop (`cmd_run::run_constrained_or_stopped`,
/// shared by `chat`) draws each token either from the grammar's
/// `ConstrainedSampler` chain or from the CLI's own
/// [`oxibonsai_runtime::sampling::Sampler::sample`] — both history-free, so
/// neither applies a repetition, frequency or presence penalty. Only
/// `sample_with_history` (the engine's own `generate`/
/// `generate_streaming_sync` loop, and the CLI's min-p loop) applies them.
/// Previously a non-default penalty combined with `--grammar`/`--stop` was
/// silently dropped while `--help` claimed it was applied on every backend;
/// this fails fast and names exactly which flag(s) conflict instead.
pub(crate) fn reject_penalties_with_constrained_decode(
    use_constrained_or_stop: bool,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) -> anyhow::Result<()> {
    if !use_constrained_or_stop {
        return Ok(());
    }
    let mut offending = Vec::new();
    if repetition_penalty != 1.0 {
        offending.push(format!("--repetition-penalty {repetition_penalty}"));
    }
    if frequency_penalty != 0.0 {
        offending.push(format!("--frequency-penalty {frequency_penalty}"));
    }
    if presence_penalty != 0.0 {
        offending.push(format!("--presence-penalty {presence_penalty}"));
    }
    if offending.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "{} cannot be combined with --grammar or --stop: the constrained/stop-checking \
         decode loop samples history-free and never applies repetition/frequency/presence \
         penalties, so accepting this combination would silently drop the penalty instead \
         of applying it. Drop the penalty flag(s), or drop --grammar/--stop.",
        offending.join(", ")
    );
}

/// The pre-RT-17 sampling literals every command falls back to when neither
/// a flag / `--config` value nor the GGUF's own `general.sampling.*`
/// declares a value — ONE definition shared by `run`/`chat`
/// (`cmd_run::resolve_sampling`, `mod.rs`) and `serve`
/// (`cmd_serve::baseline_sampling_params`), so the two paths cannot drift.
pub(crate) const DEFAULT_TEMPERATURE: f32 = 0.7;
/// See [`DEFAULT_TEMPERATURE`].
pub(crate) const DEFAULT_TOP_K: usize = 40;
/// See [`DEFAULT_TEMPERATURE`].
pub(crate) const DEFAULT_TOP_P: f32 = 0.9;
/// See [`DEFAULT_TEMPERATURE`] (`0.0` = min-p disabled).
pub(crate) const DEFAULT_MIN_P: f32 = 0.0;
/// See [`DEFAULT_TEMPERATURE`] (`1.0` = no penalty: `--temperature 0` is
/// exactly argmax).
pub(crate) const DEFAULT_REPETITION_PENALTY: f32 = 1.0;

/// The single constructor `run`/`chat`/`serve` use to build
/// [`SamplingParams`].
///
/// Every field this function is given is set explicitly, so nothing is
/// silently inherited from [`SamplingParams::default`] (whose
/// `repetition_penalty` is `1.0` workspace-wide since RT-24). With no
/// explicit `--repetition-penalty`, callers resolve to `1.0` (disabled)
/// before calling this, so `--temperature 0` means exactly argmax on every
/// backend, never a hidden non-1.0 penalty perturbing it.
///
/// [`SamplingParams`]: oxibonsai_runtime::sampling::SamplingParams
/// [`SamplingParams::default`]: oxibonsai_runtime::sampling::SamplingParams::default
pub(crate) fn build_sampling_params(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> oxibonsai_runtime::sampling::SamplingParams {
    oxibonsai_runtime::sampling::SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        ..oxibonsai_runtime::sampling::SamplingParams::default()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// deps-08: a real, non-world-writable default data directory (never /tmp)
// ──────────────────────────────────────────────────────────────────────────

/// A per-user data directory this process owns, equivalent in spirit to
/// the `dirs` crate's `data_dir()` (not added as a dependency: the root
/// `Cargo.toml` manifest is outside this package's `owned_files` this
/// wave — recorded in `deviations`). Never `/tmp` or any other
/// world-writable path.
///
/// Resolution: `$XDG_DATA_HOME`, else `~/Library/Application Support` on
/// macOS, `~/.local/share` elsewhere on Unix, `%APPDATA%` on Windows.
/// Returns `None` only when neither the relevant environment variable nor
/// `$HOME`/`%USERPROFILE%` is set.
pub(crate) fn default_data_dir() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg));
        }
    }
    if cfg!(target_os = "windows") {
        return std::env::var("APPDATA").ok().map(PathBuf::from);
    }
    let home = std::env::var("HOME").ok()?;
    if home.is_empty() {
        return None;
    }
    if cfg!(target_os = "macos") {
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support"),
        )
    } else {
        Some(PathBuf::from(home).join(".local").join("share"))
    }
}

/// This binary's own data directory: `<default_data_dir>/oxibonsai`.
/// Falls back to `models` (relative to the current directory) when no
/// data directory can be determined at all, which is still never `/tmp`.
pub(crate) fn oxibonsai_data_dir() -> PathBuf {
    match default_data_dir() {
        Some(dir) => dir.join("oxibonsai"),
        None => PathBuf::from("models"),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// cli-04: honest --config loading (every section, no silently-ignored
// unknown keys, a real error on a bad path)
// ──────────────────────────────────────────────────────────────────────────

/// Validate and load an already-read config file's `content` as an
/// [`oxibonsai_runtime::OxiBonsaiConfig`], propagating a real, path-naming
/// error instead of `load_or_default`'s warn-and-degrade-to-defaults
/// behavior, and additionally rejecting any key this schema does not
/// recognize.
///
/// Takes `content` (rather than re-reading `path` itself) so the caller
/// can pass the exact bytes it already read for [`parse_flat_toml_sections`]
/// — `mod.rs` used to read the file once for that scan and a second time
/// here, a redundant read with a TOCTOU window in which the two reads
/// could observe different file content. One read remains unavoidable
/// within this package alone:
/// [`oxibonsai_runtime::OxiBonsaiConfig::load`] takes a `&Path` and reads
/// the file itself internally (`oxibonsai-runtime/src/config.rs`, outside
/// this package's `owned_files` this wave — recorded in `deviations`); a
/// `from_str`/content-based constructor there, or a `toml` crate
/// dependency edge for this package in the (also out-of-scope) root
/// `Cargo.toml`, would close that second read too.
///
/// `OxiBonsaiConfig`'s own `#[serde(default)]` deserialization silently
/// ignores unknown keys rather than erroring (`#[serde(deny_unknown_fields)]`
/// would fix this at the source, but that struct lives in
/// `oxibonsai-runtime/src/config.rs`, outside this package's
/// `owned_files` this wave — recorded in `deviations`). This performs an
/// equivalent check by hand via [`parse_flat_toml_sections`].
pub(crate) fn load_config_strict(
    content: &str,
    path: &Path,
) -> anyhow::Result<oxibonsai_runtime::OxiBonsaiConfig> {
    parse_flat_toml_sections(content, path)?;

    oxibonsai_runtime::OxiBonsaiConfig::load(path)
        .map_err(|e| anyhow::anyhow!("failed to parse config file {}: {e}", path.display()))
}

/// Known `[section]` names and, per section, known `key` names for
/// [`oxibonsai_runtime::OxiBonsaiConfig`] (mirrors `ServerConfig`,
/// `SamplingConfig`, `ModelConfig`, `ObservabilityConfig` field lists).
const KNOWN_CONFIG_SECTIONS: &[(&str, &[&str])] = &[
    (
        "sampling",
        &[
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ],
    ),
    (
        "model",
        &[
            "model_path",
            "tokenizer_path",
            "max_seq_len",
            "backend",
            "rope_scaling",
            "reasoning_effort",
            "enable_thinking",
            "prefill_chunk",
            "ptq1_transcode",
        ],
    ),
    (
        "server",
        &[
            "host",
            "port",
            "cuda_device",
            "bearer_token_file",
            "rate_limit_rpm",
            "rate_limit_burst",
            "cors_origin",
            "max_body_bytes",
            "max_output_tokens",
            "enable_ui",
        ],
    ),
    ("observability", &["log_level", "json_logs"]),
    (
        "imagen",
        &[
            "model_path",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "seed",
            "output_dir",
        ],
    ),
];

/// A `[section] key = value` pair actually present in a config file, as
/// raw (untyped, comment-stripped, trimmed) TOML value text.
pub(crate) type RawTomlSections =
    std::collections::HashMap<String, std::collections::HashMap<String, String>>;

/// Hand-rolled scan of a flat, 4-section TOML config file (this schema has
/// no nesting, arrays of tables, or multi-line values), returning exactly
/// the `[section] key = value` pairs that were genuinely present in the
/// file text.
///
/// Two jobs in one pass, both needed by cli-04:
///
/// 1. Reject any `[section]` or `key` this schema does not recognize
///    (`#[serde(deny_unknown_fields)]`'s effect, without editing the
///    struct that would need it — see [`load_config_strict`]'s doc).
/// 2. Return which keys were genuinely typed into the file, as raw text.
///    This matters beyond validation: [`oxibonsai_runtime::OxiBonsaiConfig`]
///    deserializes every field with `#[serde(default)]`, so a strongly-typed
///    field reads as that struct's own built-in default even when the file
///    never mentions it — it cannot tell "absent" from "explicitly the
///    default". That matters for every RT-17 sampling value (an absent key
///    must fall through to the GGUF's own `general.sampling.*` default, not
///    to a struct literal) and it is how `repetition_penalty` once
///    reintroduced a hidden non-1.0 penalty (its struct default was `1.1`
///    before RT-24). Consulting the raw text instead of the typed struct
///    closes that gap; `mod.rs`'s merge step is the caller.
pub(crate) fn parse_flat_toml_sections(
    content: &str,
    path: &Path,
) -> anyhow::Result<RawTomlSections> {
    let mut sections: RawTomlSections = std::collections::HashMap::new();
    let mut current_section: Option<&(&str, &[&str])> = None;

    for (line_no, raw_line) in content.lines().enumerate() {
        let line = strip_toml_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(section_name) = parse_section_header(line) {
            match KNOWN_CONFIG_SECTIONS
                .iter()
                .find(|(name, _)| *name == section_name)
            {
                Some(entry) => {
                    current_section = Some(entry);
                    sections.entry(section_name.to_string()).or_default();
                }
                None => anyhow::bail!(
                    "config file {}:{}: unknown section [{section_name}] (known sections: \
                     server, sampling, model, observability, imagen)",
                    path.display(),
                    line_no + 1
                ),
            }
            continue;
        }

        if let Some((key, value)) = parse_key_line(line) {
            match current_section {
                Some((section_name, known_keys)) => {
                    if !known_keys.contains(&key.as_str()) {
                        anyhow::bail!(
                            "config file {}:{}: unknown key '{key}' in section [{section_name}] \
                             (known keys: {})",
                            path.display(),
                            line_no + 1,
                            known_keys.join(", ")
                        );
                    }
                    sections
                        .entry(section_name.to_string())
                        .or_default()
                        .insert(key, value);
                }
                None => anyhow::bail!(
                    "config file {}:{}: key '{key}' is not inside any [section]; this schema \
                     has no top-level keys (known sections: server, sampling, model, \
                     observability, imagen)",
                    path.display(),
                    line_no + 1
                ),
            }
        }
        // Any other line shape (array continuation, multi-line string,
        // inline table, ...) is left unchecked: this scanner only needs
        // to catch the common flat `key = value` shape this schema
        // actually uses, not implement a full TOML grammar.
    }

    Ok(sections)
}

/// Look up a string-valued key (quotes stripped).
pub(crate) fn toml_str(sections: &RawTomlSections, section: &str, key: &str) -> Option<String> {
    let raw = sections.get(section)?.get(key)?;
    Some(unquote_toml_string(raw))
}

/// Look up and parse an `f32`-valued key.
pub(crate) fn toml_f32(sections: &RawTomlSections, section: &str, key: &str) -> Option<f32> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Look up and parse a `usize`-valued key.
pub(crate) fn toml_usize(sections: &RawTomlSections, section: &str, key: &str) -> Option<usize> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Look up and parse a `u16`-valued key. Only `serve`'s `--port` /
/// `[server].port` resolution needs a `u16` (the CLI's `port` type);
/// `#[cfg]`-gated on `server` like that command so a
/// `--no-default-features` build (which excludes it) has no dead code.
#[cfg(feature = "server")]
pub(crate) fn toml_u16(sections: &RawTomlSections, section: &str, key: &str) -> Option<u16> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Look up and parse a `u32`-valued key (`serve --cuda-device` /
/// `--rate-limit-rpm` / `--rate-limit-burst`, `[imagen]` sizes).
pub(crate) fn toml_u32(sections: &RawTomlSections, section: &str, key: &str) -> Option<u32> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Look up and parse a `u64`-valued key (`serve --max-body-bytes`,
/// `[imagen].seed`).
pub(crate) fn toml_u64(sections: &RawTomlSections, section: &str, key: &str) -> Option<u64> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Convert one of `args.rs`'s `validate_*`/`parse_*` `Result<T, String>`
/// outcomes into `anyhow::Result<T>`. The single copy, shared by `mod.rs`
/// and every `cmd_*` module that re-validates a value it resolves itself
/// after the model is loaded (`cmd_run`/`cmd_chat`'s RT-17/REQUIRED #8
/// resolution, done post-GGUF-load rather than in `mod.rs`).
pub(crate) fn validated<T>(result: Result<T, String>) -> anyhow::Result<T> {
    result.map_err(|e| anyhow::anyhow!(e))
}

/// [`validated`] for an optional value: `None` stays `None`, `Some(v)` is run
/// through `validate`. `mod.rs` uses this on every CLI-or-`--config` value
/// that is resolved *before* the model is loaded (the final default, when it
/// depends on the GGUF, is applied later), so a `[sampling].temperature =
/// -5.0` is refused up front, naming the field, exactly like `--temperature
/// -5` — never after an expensive model resolution (cli-12 / cli-04).
pub(crate) fn validated_opt<T>(
    value: Option<T>,
    validate: impl FnOnce(T) -> Result<T, String>,
) -> anyhow::Result<Option<T>> {
    value.map(validate).transpose().map_err(anyhow::Error::msg)
}

/// Look up and parse a `bool`-valued key (`true`/`false`).
pub(crate) fn toml_bool(sections: &RawTomlSections, section: &str, key: &str) -> Option<bool> {
    sections.get(section)?.get(key)?.parse().ok()
}

/// Crate-wide serialization for unit tests that mutate process environment
/// variables (ENGINE-SEAM addendum (3)): every such test in this binary —
/// `cmd_serve`'s, `pull`'s, … — holds [`lock`](test_env::lock) for its whole
/// body and restores what it changed through an [`EnvVarGuard`](test_env::EnvVarGuard),
/// so no two tests ever race on the same variable, and a panicking test still
/// puts the environment back (the guard's `Drop` runs during unwinding while
/// the lock is still held).
#[cfg(test)]
pub(crate) mod test_env {
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Take the process-wide environment lock (poison-tolerant: a failed test
    /// must not cascade into every later one).
    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Sets (or removes) one variable and restores its prior value on drop.
    /// Declare it AFTER the lock guard so it drops first.
    pub(crate) struct EnvVarGuard {
        key: &'static str,
        prior: Option<OsString>,
    }

    impl EnvVarGuard {
        pub(crate) fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let prior = std::env::var_os(key);
            // SAFETY (test-only): every caller holds `lock()`, so no other
            // test thread reads or writes the environment concurrently.
            unsafe { std::env::set_var(key, value) };
            Self { key, prior }
        }

        pub(crate) fn remove(key: &'static str) -> Self {
            let prior = std::env::var_os(key);
            // SAFETY (test-only): see `set`.
            unsafe { std::env::remove_var(key) };
            Self { key, prior }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY (test-only): still under the caller's `lock()` guard.
            match self.prior.take() {
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }
}

// ── Generic "flag > config > hardcoded default" resolvers (cli-04) ──────
//
// `sections` is an empty map when no `--config` was given (see `mod.rs`),
// so these apply uniformly whether or not a config file is in play: a
// present CLI flag always wins, otherwise a genuinely-present config key
// wins, otherwise the caller's own documented default applies. None of
// these ever consult `OxiBonsaiConfig`'s own `#[serde(default)]`-filled
// struct fields, precisely so a config file's mere presence can never
// silently reintroduce a value the file itself never mentioned (the
// `repetition_penalty` hazard documented on `parse_flat_toml_sections`).

pub(crate) fn resolve_str(
    cli: Option<String>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
) -> Option<String> {
    cli.or_else(|| toml_str(sections, section, key))
}

pub(crate) fn resolve_f32(
    cli: Option<f32>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
    default: f32,
) -> f32 {
    cli.or_else(|| toml_f32(sections, section, key))
        .unwrap_or(default)
}

pub(crate) fn resolve_usize(
    cli: Option<usize>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
    default: usize,
) -> usize {
    cli.or_else(|| toml_usize(sections, section, key))
        .unwrap_or(default)
}

#[cfg(feature = "server")]
pub(crate) fn resolve_u16(
    cli: Option<u16>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
    default: u16,
) -> u16 {
    cli.or_else(|| toml_u16(sections, section, key))
        .unwrap_or(default)
}

/// Resolve `--backend`/`[model].backend` (wave-4b ENGINE-SEAM addendum):
/// an explicit flag always wins (already parsed and validated by clap's own
/// `value_parser`, so `cli` arriving `Some` is already a valid [`Backend`]);
/// otherwise a `[model].backend` TOML string is parsed and validated here
/// (config-file values never go through clap's `value_parser`, so cli-04's
/// "every source is checked identically" contract applies just as it does
/// for the numeric sampling flags); otherwise [`Backend::default`] (`Auto`).
///
/// # Errors
///
/// Returns an error naming the bad value when `[model].backend` is present
/// but is not one of `auto`/`cpu`/`metal`.
pub(crate) fn resolve_backend(
    cli: Option<oxibonsai_runtime::engine_seam::Backend>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
) -> anyhow::Result<oxibonsai_runtime::engine_seam::Backend> {
    if let Some(b) = cli {
        return Ok(b);
    }
    match toml_str(sections, section, key) {
        Some(s) => oxibonsai_runtime::engine_seam::Backend::parse(&s).ok_or_else(|| {
            anyhow::anyhow!("invalid [{section}].{key} value '{s}': expected auto, cpu, or metal")
        }),
        None => Ok(oxibonsai_runtime::engine_seam::Backend::default()),
    }
}

/// Resolve `--rope-scaling`/`[model].rope_scaling` (wave-4b orchestrator
/// addendum; see [`oxibonsai_runtime::config::RopeScalingMode`]). Same
/// "flag > config-string > default" precedence as [`resolve_backend`].
///
/// # Errors
///
/// Returns an error naming the bad value when `[model].rope_scaling` is
/// present but is not one of `auto`/`on`/`off`.
pub(crate) fn resolve_rope_scaling(
    cli: Option<oxibonsai_runtime::config::RopeScalingMode>,
    sections: &RawTomlSections,
    section: &str,
    key: &str,
) -> anyhow::Result<oxibonsai_runtime::config::RopeScalingMode> {
    if let Some(m) = cli {
        return Ok(m);
    }
    match toml_str(sections, section, key) {
        Some(s) => s
            .parse::<oxibonsai_runtime::config::RopeScalingMode>()
            .map_err(|e| anyhow::anyhow!("invalid [{section}].{key} value: {e}")),
        None => Ok(oxibonsai_runtime::config::RopeScalingMode::default()),
    }
}

/// Strip a single layer of matching `"`/`'` quotes from a raw TOML string
/// value. Returns the input unchanged (trimmed) when it is not quoted.
fn unquote_toml_string(raw: &str) -> String {
    let trimmed = raw.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Strip a `#`-led TOML comment, honoring `"` and `'` quoting so a `#`
/// inside a string value is not mistaken for a comment marker.
fn strip_toml_comment(line: &str) -> &str {
    let mut in_double = false;
    let mut in_single = false;
    for (i, ch) in line.char_indices() {
        match ch {
            '"' if !in_single => in_double = !in_double,
            '\'' if !in_double => in_single = !in_single,
            '#' if !in_double && !in_single => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Parse a `[section]` header line, returning the section name.
///
/// Deliberately treats `[[array_of_tables]]` and dotted/nested `[a.b]`
/// headers as header candidates too (rather than silently falling through
/// as "not a header at all"): this schema has neither shape, so the
/// caller's `KNOWN_CONFIG_SECTIONS` lookup below correctly rejects both as
/// an unknown section instead of either misattributing a dotted header's
/// `key = value` lines to the last-seen plain section, or a top-level
/// error blaming "not inside any [section]" for what is actually an
/// unsupported header shape.
fn parse_section_header(line: &str) -> Option<&str> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    if inner.is_empty() {
        return None;
    }
    Some(inner.trim())
}

/// Parse a `key = value` line, returning `(key, raw_value)`, both trimmed.
/// The key has its own optional quoting stripped (TOML allows `"key with
/// spaces" = ...`); the value is returned as raw, untyped text for the
/// caller to interpret (see [`toml_str`]/[`toml_f32`]/[`toml_usize`]).
fn parse_key_line(line: &str) -> Option<(String, String)> {
    let eq_pos = line.find('=')?;
    let key = line[..eq_pos].trim();
    if key.is_empty() {
        return None;
    }
    let unquoted_key = key.trim_matches('"').trim_matches('\'').to_string();
    let value = line[eq_pos + 1..].trim().to_string();
    Some((unquoted_key, value))
}

// ──────────────────────────────────────────────────────────────────────────
// cli-14: an up-front, honest memory estimate before `quantize` dequantizes
// every tensor to f32
// ──────────────────────────────────────────────────────────────────────────

/// Best-effort bytes of RAM currently available on this machine.
///
/// Linux: `MemAvailable` from `/proc/meminfo`. macOS: total physical RAM
/// via `sysctl -n hw.memsize` (not "available", but a reasonable
/// conservative stand-in — no portable "available" query exists without
/// an FFI/library dependency this package cannot add, see `deviations`).
/// Other platforms: `None` (callers fall back to a fixed conservative
/// threshold).
pub(crate) fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Conservative fallback ceiling used when [`available_memory_bytes`]
/// cannot determine the real figure (e.g. on an unsupported platform).
pub(crate) const FALLBACK_MEMORY_CEILING_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Fraction of available memory `quantize`'s f32 working set is allowed to
/// consume before the guard refuses to proceed (the process also needs
/// headroom for the OS, the mmap'd input file, and one tensor's encoded
/// output buffer).
const MEMORY_SAFETY_FACTOR: f64 = 0.7;

/// The largest `element_count * 4` bytes over every tensor in `gguf` — the
/// exact size dequantizing this model's single biggest tensor to f32 will
/// require, computed purely from tensor shapes with no dequantization work
/// done (the guard must run before the mmap and before any real work).
///
/// cli-14: `quantize` now streams tensor-by-tensor through
/// [`oxibonsai_model::export::export_to_gguf_streaming`], so the whole
/// model is never resident in RAM at once — only one tensor's f32 buffer
/// (plus its encoded output, always smaller) is live at any time. This
/// replaces the old whole-model *sum*, which over-rejected valid work by
/// as much as the tensor count once streaming made that sum no longer the
/// real peak.
pub(crate) fn peak_streaming_f32_bytes(gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>) -> u128 {
    gguf.tensors
        .iter()
        .map(|(_, info)| {
            let elements: u128 = info.shape.iter().map(|&d| d as u128).product();
            elements.saturating_mul(4)
        })
        .max()
        .unwrap_or(0)
}

/// Return `Err` naming both figures when dequantizing this model's single
/// largest tensor to f32 is estimated to need more memory than this
/// machine can safely spare. `force` bypasses the check entirely.
pub(crate) fn check_quantize_memory_budget(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    force: bool,
) -> anyhow::Result<()> {
    if force {
        return Ok(());
    }
    let estimated = peak_streaming_f32_bytes(gguf);
    let ceiling = match available_memory_bytes() {
        Some(avail) => ((avail as f64) * MEMORY_SAFETY_FACTOR) as u128,
        None => FALLBACK_MEMORY_CEILING_BYTES as u128,
    };
    if estimated > ceiling {
        let estimated_gib = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
        let ceiling_gib = ceiling as f64 / (1024.0 * 1024.0 * 1024.0);
        anyhow::bail!(
            "quantize would need an estimated {estimated_gib:.1} GiB to dequantize this \
             model's single largest tensor to f32, which exceeds this machine's usable \
             memory budget of {ceiling_gib:.1} GiB. Pass --force to proceed anyway, or \
             quantize on a machine with more RAM."
        );
    }
    Ok(())
}

/// Decode a raw IEEE 754 half-precision (binary16) bit pattern to `f32`.
///
/// A small local implementation rather than pulling in the `half` crate
/// as a production dependency of this binary (it is already used
/// pervasively elsewhere in the workspace, just not linked into the CLI
/// binary itself); handles zero, subnormals, normals, infinities and NaN.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as f32;

    let magnitude = if exponent == 0 {
        if mantissa == 0.0 {
            0.0f32
        } else {
            // Subnormal: value = mantissa / 1024 * 2^-14.
            mantissa * 2f32.powi(-24)
        }
    } else if exponent == 0x1F {
        if mantissa == 0.0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let exp = exponent as i32 - 15;
        (1.0 + mantissa / 1024.0) * 2f32.powi(exp)
    };

    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// Dequantize a single named tensor from a parsed GGUF file to `f32`.
///
/// Mirrors the internal tensor loader the inference engine uses
/// (`oxibonsai_model`'s private `load_f32_tensor`), re-implemented here
/// against the same public `oxibonsai_core` block types so the
/// `quantize` subcommand can re-encode a real model's weights through
/// [`oxibonsai_model::export::export_to_gguf`] instead of fabricating a
/// result. Returns an honest error (rather than silently misreading
/// bytes) for any tensor type this CLI does not yet know how to
/// dequantize.
pub(crate) fn dequantize_gguf_tensor(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    name: &str,
) -> anyhow::Result<Vec<f32>> {
    use oxibonsai_core::GgufTensorType;

    let info = gguf.tensors.require(name)?;
    let data = gguf.tensor_data(name)?;

    let out = match info.tensor_type {
        GgufTensorType::F32 => {
            let count = data.len() / 4;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgufTensorType::F16 => {
            let count = data.len() / 2;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(2).enumerate() {
                out[i] = f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgufTensorType::Q1_0_g128 => {
            let blocks = oxibonsai_core::tensor::BlockQ1_0G128::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::tensor::QK1_0_G128;
            let mut out = vec![0.0f32; n];
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let base = i * oxibonsai_core::tensor::QK1_0_G128;
                for j in 0..oxibonsai_core::tensor::QK1_0_G128 {
                    let byte_index = j / 8;
                    let bit_offset = j % 8;
                    let bit = (block.qs[byte_index] >> bit_offset) & 1;
                    out[base + j] = if bit != 0 { d } else { -d };
                }
            }
            out
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_TQ2_0_G128;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockTQ2_0_g128::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q4_0 => {
            let blocks = oxibonsai_core::BlockQ4_0::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_Q4_0;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ4_0::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q8_0 => {
            let blocks = oxibonsai_core::BlockQ8_0::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_Q8_0;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ8_0::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q4_K => {
            let blocks = oxibonsai_core::BlockQ4K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ4K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q5_K => {
            let blocks = oxibonsai_core::BlockQ5K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ5K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q6_K => {
            let blocks = oxibonsai_core::BlockQ6K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ6K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::F8_E4M3 => {
            let blocks = oxibonsai_core::BlockFP8E4M3::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_FP8;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockFP8E4M3::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::F8_E5M2 => {
            let blocks = oxibonsai_core::BlockFP8E5M2::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_FP8;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockFP8E5M2::dequant(blocks, &mut out)?;
            out
        }
        other => anyhow::bail!(
            "tensor '{name}': cannot dequantize source type {other} — `quantize` only reads \
             F32, F16, Q1_0_g128, TQ2_0_g128, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F8_E4M3, or \
             F8_E5M2 tensors"
        ),
    };
    Ok(out)
}

/// Map a CLI `--format` string to the [`oxibonsai_model::export::ExportFormat`]
/// it names, refusing (with an honest error) any format the export
/// pipeline cannot actually produce a loadable GGUF file for.
pub(crate) fn parse_quantize_format(
    format: &str,
) -> anyhow::Result<oxibonsai_model::export::ExportFormat> {
    use oxibonsai_model::export::ExportFormat;
    match format {
        "f32" => Ok(ExportFormat::Float32),
        "q1_0" | "q1_0_g128" => Ok(ExportFormat::Q1_0G128),
        "tq2_0_g128" | "ternary" => Ok(ExportFormat::TernaryG128),
        "fp8_e4m3" => Ok(ExportFormat::FP8E4M3),
        "fp8_e5m2" => Ok(ExportFormat::FP8E5M2),
        "q4_0" => Ok(ExportFormat::Q4_0),
        "q8_0" => Ok(ExportFormat::Q8_0),
        "q4_k" => Ok(ExportFormat::Q4K),
        "q5_k" => Ok(ExportFormat::Q5K),
        "q6_k" => Ok(ExportFormat::Q6K),
        // Wave-3.5 deviation routing: FIX3-GGUF-WRITE's `encode_quantized_tensor`
        // already handles `TensorType::{Q2_K,Q3_K,Q8_K}` (the actual writer
        // support); these three string arms plus `ExportFormat::{Q2K,Q3K,Q8K}`
        // (`crates/oxibonsai-model/src/export.rs`, also owned by this
        // package) are the CLI-side wiring that was left unwired end-to-end.
        "q2_k" => Ok(ExportFormat::Q2K),
        "q3_k" => Ok(ExportFormat::Q3K),
        "q8_k" => Ok(ExportFormat::Q8K),
        other => anyhow::bail!(
            "unsupported quantization format '{other}' — the export pipeline can only \
             produce a loadable GGUF file for: f32, q1_0, tq2_0_g128, fp8_e4m3, fp8_e5m2, \
             q4_0, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k, q8_k (q4_1, f16 output are not \
             supported by the export pipeline; q4_1 has no writer, and f16 has no \
             `ExportFormat` arm — use f32 or a supported quantized format instead)"
        ),
    }
}

#[cfg(test)]
#[path = "util_tests.rs"]
mod tests;
