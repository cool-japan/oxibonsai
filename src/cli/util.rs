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

/// The single constructor `run`/`chat` use to build [`SamplingParams`].
///
/// [`oxibonsai_runtime::sampling::SamplingParams::default`] bakes in
/// `repetition_penalty: 1.1`; every field this function is given is set
/// explicitly, so that value is never silently inherited. With no
/// explicit `--repetition-penalty`, callers resolve to `1.0` (disabled)
/// before calling this, so `--temperature 0` means exactly argmax on
/// every backend, never a hidden non-1.0 penalty perturbing it.
///
/// [`SamplingParams`]: oxibonsai_runtime::sampling::SamplingParams
/// Reject `--grammar`/`--stop` when combined with a non-default sampling
/// penalty.
///
/// `cmd_run::run_constrained_or_stopped` and
/// `cmd_chat::run_constrained_or_stopped_turn` sample via
/// [`oxibonsai_runtime::InferenceEngine::sample`], which delegates to
/// [`oxibonsai_runtime::sampling::Sampler::sample`] — the base, history-free
/// path that applies **no** penalty. Only `sample_with_history` (used
/// internally by the engine's own `generate`/`generate_streaming_sync`, the
/// fast path this loop deliberately does not share) applies repetition,
/// frequency or presence penalties, and it is not on `InferenceEngine`'s
/// public surface. Previously a non-default penalty combined with
/// `--grammar`/`--stop` was silently dropped while `--help` and this
/// module's own doc comments claimed it was applied on every backend; this
/// fails fast and names exactly which flag(s) conflict instead.
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
         decode loop samples via the engine's history-free sampler and never applies \
         repetition/frequency/presence penalties, so accepting this combination would \
         silently drop the penalty instead of applying it. Drop the penalty flag(s), or \
         drop --grammar/--stop.",
        offending.join(", ")
    );
}

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
    ("server", &["host", "port"]),
    (
        "sampling",
        &[
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ],
    ),
    ("model", &["model_path", "tokenizer_path", "max_seq_len"]),
    ("observability", &["log_level", "json_logs"]),
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
///    `config.sampling.repetition_penalty` reads as `1.1` (that struct's
///    own built-in default) even when the file never mentions
///    `repetition_penalty` at all — which would silently reintroduce
///    exactly the hidden non-1.0 penalty the orchestrator's P0 addendum
///    requires this CLI never apply, the moment ANY `--config` file is in
///    use. Consulting the raw text instead of the typed struct for
///    fields with this hazard (repetition_penalty in particular) closes
///    that gap; `mod.rs`'s merge step is the caller.
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
                     server, sampling, model, observability)",
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
                     observability)",
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
        other => anyhow::bail!(
            "unsupported quantization format '{other}' — the export pipeline can only \
             produce a loadable GGUF file for: f32, q1_0, tq2_0_g128, fp8_e4m3, fp8_e5m2, \
             q4_0, q8_0, q4_k, q5_k, q6_k (q2_k, q4_1, f16 output are not supported by the \
             export pipeline; q2_k/q4_1 have no writer, and f16 has no `ExportFormat` \
             arm — use f32 or a supported quantized format instead)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        missing_tokenizer_warning, parse_flat_toml_sections, read_prompt_stdin,
        reject_penalties_with_constrained_decode, resolve_f32, resolve_str, resolve_tokenizer,
        resolve_usize, strip_quant_suffix, tokenizer_candidates, toml_f32, toml_str, toml_usize,
        RawTomlSections,
    };
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Helper: write an empty `tokenizer.json` at the given path, creating
    /// any missing parent directories.
    fn touch_tokenizer(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create_dir_all");
        }
        fs::write(path, b"{}").expect("write tokenizer.json");
    }

    #[test]
    fn resolve_tokenizer_finds_in_same_dir() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Foo-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        touch_tokenizer(&model_dir.join("tokenizer.json"));

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        assert_eq!(
            PathBuf::from(found),
            model_dir.join("tokenizer.json"),
            "should locate tokenizer in the same dir as the model"
        );
    }

    #[test]
    fn resolve_tokenizer_finds_in_parent_dir() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models").join("variant");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Foo-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        // Place tokenizer in the parent directory only.
        let parent_tokenizer = tmp.path().join("models").join("tokenizer.json");
        touch_tokenizer(&parent_tokenizer);

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        // Either the literal `..` candidate or the canonicalized `models/tokenizer.json`
        // candidate is acceptable; both refer to the same file.
        let found_path = PathBuf::from(found);
        let canon_found = fs::canonicalize(&found_path).expect("canonicalize found");
        let canon_target = fs::canonicalize(&parent_tokenizer).expect("canonicalize target");
        assert_eq!(
            canon_found, canon_target,
            "should locate tokenizer in the model's parent directory"
        );
    }

    #[test]
    fn resolve_tokenizer_finds_via_unpacked_sibling() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        // Tokenizer only lives in the sibling unpacked directory.
        let unpacked = model_dir.join("Ternary-Bonsai-8B-unpacked");
        touch_tokenizer(&unpacked.join("tokenizer.json"));

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        assert_eq!(
            PathBuf::from(found),
            unpacked.join("tokenizer.json"),
            "should locate tokenizer via <base>-unpacked sibling directory"
        );
    }

    #[test]
    fn resolve_tokenizer_strips_quant_suffix_for_sibling_lookup() {
        // Verifies the candidate list (without filesystem) for the
        // `Foo-Q2_0.gguf` case includes Foo/, Foo-unpacked/, Foo-ONNX/.
        let model_path = PathBuf::from("models/Foo-Q2_0.gguf");
        let candidates = tokenizer_candidates(&model_path);
        let candidate_strs: Vec<String> = candidates
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let expected = [
            "models/Foo/tokenizer.json",
            "models/Foo-unpacked/tokenizer.json",
            "models/Foo-ONNX/tokenizer.json",
        ];
        for needle in expected {
            assert!(
                candidate_strs.iter().any(|c| c == needle),
                "missing expected candidate {needle}; got {candidate_strs:?}"
            );
        }
    }

    #[test]
    fn resolve_tokenizer_records_searched_paths_when_missing() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        assert!(
            lookup.found.is_none(),
            "should not find tokenizer in empty tree"
        );
        assert!(
            !lookup.searched.is_empty(),
            "searched list must be populated when nothing is found"
        );
        // Confirm at least the "same dir" candidate is recorded.
        assert!(
            lookup
                .searched
                .iter()
                .any(|p| p == &model_dir.join("tokenizer.json")),
            "searched list should include the same-directory candidate"
        );
        // Warning text must mention every searched path and both remedies.
        let warning = missing_tokenizer_warning(&lookup.searched);
        for path in &lookup.searched {
            assert!(
                warning.contains(&path.display().to_string()),
                "warning should list {}, got: {warning}",
                path.display()
            );
        }
        assert!(
            warning.contains("--tokenizer"),
            "warning must mention --tokenizer remedy"
        );
        assert!(
            warning.contains("download_tokenizer.sh"),
            "warning must mention download_tokenizer.sh remedy"
        );
    }

    #[test]
    fn resolve_tokenizer_explicit_override_skips_search() {
        let lookup = resolve_tokenizer(Some("/custom/path/tokenizer.json"), "models/foo.gguf");
        assert_eq!(
            lookup.found.as_deref(),
            Some("/custom/path/tokenizer.json"),
            "explicit override must be returned verbatim"
        );
        assert!(
            lookup.searched.is_empty(),
            "explicit override must not trigger a filesystem search"
        );
    }

    #[test]
    fn strip_quant_suffix_handles_known_formats() {
        assert_eq!(
            strip_quant_suffix("Ternary-Bonsai-8B-Q2_0"),
            "Ternary-Bonsai-8B"
        );
        assert_eq!(strip_quant_suffix("Foo-Q1_0"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-Q4_K_M"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-Q8_0"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-F16"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-BF16"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-F32"), "Foo");
        // Non-quant suffix should be left alone.
        assert_eq!(strip_quant_suffix("Foo-bar"), "Foo-bar");
        assert_eq!(strip_quant_suffix("Foo"), "Foo");
    }

    #[test]
    fn tokenizer_candidates_includes_top_level_models_dir() {
        let model_path = PathBuf::from("models/sub/dir/Foo-Q2_0.gguf");
        let candidates = tokenizer_candidates(&model_path);
        let candidate_strs: Vec<String> = candidates
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert!(
            candidate_strs.iter().any(|c| c == "models/tokenizer.json"),
            "expected top-level models/tokenizer.json candidate, got {candidate_strs:?}"
        );
    }

    // ── read_prompt_stdin (cli-M4) ──────────────────────────────────────

    #[test]
    fn read_prompt_stdin_rejects_empty_input_type_check() {
        // A full stdin-redirection test belongs in an integration test
        // (tests/cli_surface_tests.rs); this just locks in the new
        // `Result` signature so callers must handle the error instead of
        // getting a silently-possibly-truncated `String` back.
        fn assert_is_result(_: fn() -> anyhow::Result<String>) {}
        assert_is_result(read_prompt_stdin);
    }

    // ── config raw-TOML scanner (cli-04) ────────────────────────────────

    #[test]
    fn parse_flat_toml_sections_accepts_well_formed_config() {
        let toml = r#"
            [server]
            host = "0.0.0.0"
            port = 9090

            [sampling]
            temperature = 0.5
            top_k = 20
            top_p = 0.95
            repetition_penalty = 1.05
            max_tokens = 256

            [model]
            model_path = "models/foo.gguf"
            tokenizer_path = "models/tokenizer.json"
            max_seq_len = 8192

            [observability]
            log_level = "debug"
            json_logs = true
        "#;
        let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
            .expect("well-formed config must be accepted");
        assert_eq!(
            toml_str(&sections, "server", "host").as_deref(),
            Some("0.0.0.0")
        );
        assert_eq!(toml_usize(&sections, "server", "port"), Some(9090));
        assert_eq!(toml_f32(&sections, "sampling", "temperature"), Some(0.5));
        assert_eq!(
            toml_f32(&sections, "sampling", "repetition_penalty"),
            Some(1.05)
        );
        assert_eq!(
            toml_str(&sections, "model", "model_path").as_deref(),
            Some("models/foo.gguf")
        );
        assert_eq!(
            toml_str(&sections, "observability", "log_level").as_deref(),
            Some("debug")
        );
    }

    #[test]
    fn parse_flat_toml_sections_rejects_typo_d_key() {
        let toml = "[sampling]\ntemperture = 0.5\n";
        let result = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("temperture"),
            "error should name the bad key: {msg}"
        );
    }

    #[test]
    fn parse_flat_toml_sections_rejects_unknown_section() {
        let toml = "[imagen]\ndit_path = \"x\"\n";
        let result = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("imagen"),
            "error should name the bad section: {msg}"
        );
    }

    #[test]
    fn parse_flat_toml_sections_ignores_comments_and_blank_lines() {
        let toml =
            "# a comment\n\n[server]\n# host is bound here\nhost = \"127.0.0.1\" # trailing\n";
        let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
            .expect("comments must not break parsing");
        assert_eq!(
            toml_str(&sections, "server", "host").as_deref(),
            Some("127.0.0.1")
        );
    }

    #[test]
    fn parse_flat_toml_sections_does_not_false_positive_on_hash_in_string() {
        let toml = "[observability]\nlog_level = \"info#not-a-comment\"\n";
        let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
            .expect("a '#' inside a quoted string must not be treated as a comment");
        assert_eq!(
            toml_str(&sections, "observability", "log_level").as_deref(),
            Some("info#not-a-comment")
        );
    }

    #[test]
    fn parse_flat_toml_sections_absent_key_returns_none_not_a_default() {
        // The whole point of this scanner (vs. the typed, `#[serde(default)]`
        // struct): a key that was never in the file must read back as
        // `None`, not silently produce that struct's own built-in default.
        let toml = "[sampling]\ntemperature = 0.5\n";
        let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
            .expect("valid config");
        assert_eq!(toml_f32(&sections, "sampling", "repetition_penalty"), None);
    }

    // ── resolve_* precedence (cli-04 / orchestrator P0 addendum) ────────

    #[test]
    fn resolve_f32_prefers_explicit_cli_value_over_config() {
        let mut sections = RawTomlSections::new();
        sections
            .entry("sampling".to_string())
            .or_default()
            .insert("repetition_penalty".to_string(), "1.4".to_string());
        let resolved = resolve_f32(Some(2.0), &sections, "sampling", "repetition_penalty", 1.0);
        assert_eq!(resolved, 2.0, "an explicit CLI flag must win over --config");
    }

    #[test]
    fn resolve_f32_falls_back_to_config_value() {
        let mut sections = RawTomlSections::new();
        sections
            .entry("sampling".to_string())
            .or_default()
            .insert("temperature".to_string(), "0.3".to_string());
        let resolved = resolve_f32(None, &sections, "sampling", "temperature", 0.7);
        assert_eq!(resolved, 0.3);
    }

    #[test]
    fn resolve_f32_repetition_penalty_never_silently_becomes_1_1() {
        // The whole point of routing repetition_penalty through the raw
        // scanner instead of `SamplingConfig::default()` (which is 1.1):
        // an empty (or unrelated) config must resolve to the CLI's own
        // safe default (1.0), never that struct's built-in value.
        let sections = RawTomlSections::new();
        let resolved = resolve_f32(None, &sections, "sampling", "repetition_penalty", 1.0);
        assert_eq!(resolved, 1.0);
    }

    #[test]
    fn resolve_usize_and_str_use_hardcoded_default_with_no_config() {
        let sections = RawTomlSections::new();
        assert_eq!(
            resolve_usize(None, &sections, "sampling", "max_tokens", 256),
            256
        );
        assert_eq!(resolve_str(None, &sections, "model", "model_path"), None);
    }

    // ── reject_penalties_with_constrained_decode (the --stop/--grammar +
    // penalty silent-drop finding) ───────────────────────────────────────

    #[test]
    fn constrained_decode_with_default_penalties_is_allowed() {
        reject_penalties_with_constrained_decode(true, 1.0, 0.0, 0.0)
            .expect("all-default penalties never conflict with --grammar/--stop");
    }

    #[test]
    fn non_constrained_decode_allows_any_penalty() {
        // The fast path (`generate`/`generate_streaming_sync`) DOES apply
        // penalties via `sample_with_history`, so outside a
        // grammar/stop-checking loop every value is fine.
        reject_penalties_with_constrained_decode(false, 1.4, 0.5, -0.5)
            .expect("the fast path applies penalties; no combination is rejected");
    }

    #[test]
    fn constrained_decode_rejects_non_default_repetition_penalty() {
        let err = reject_penalties_with_constrained_decode(true, 1.2, 0.0, 0.0)
            .expect_err("a non-default repetition penalty must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("--repetition-penalty"), "got: {msg}");
    }

    #[test]
    fn constrained_decode_rejects_non_default_frequency_penalty() {
        let err = reject_penalties_with_constrained_decode(true, 1.0, 0.3, 0.0)
            .expect_err("a non-default frequency penalty must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("--frequency-penalty"), "got: {msg}");
    }

    #[test]
    fn constrained_decode_rejects_non_default_presence_penalty() {
        let err = reject_penalties_with_constrained_decode(true, 1.0, 0.0, 0.3)
            .expect_err("a non-default presence penalty must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("--presence-penalty"), "got: {msg}");
    }

    #[test]
    fn constrained_decode_rejection_names_every_offending_flag() {
        let err = reject_penalties_with_constrained_decode(true, 1.5, 0.2, 0.1)
            .expect_err("all three penalties are non-default");
        let msg = err.to_string();
        assert!(msg.contains("--repetition-penalty"), "got: {msg}");
        assert!(msg.contains("--frequency-penalty"), "got: {msg}");
        assert!(msg.contains("--presence-penalty"), "got: {msg}");
    }
}
