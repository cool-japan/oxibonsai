//! Tokenizer backend selection (cli-08).
//!
//! The CLI has two tokenizer backends: the Pure-Rust native BPE backend
//! (`oxibonsai-tokenizer`, available on every target) and the HuggingFace
//! `tokenizers` backend, which exists only in a build with the opt-in
//! `hf-tokenizer` Cargo feature (off by default: a default build does not link
//! the `tokenizers` crate at all).
//!
//! `--tokenizer-backend {auto,native,hf}` selects between them. `Auto` (the
//! default) resolves to the native backend in a default build and whenever the
//! binary was compiled with `native-tokenizer`, and to the HuggingFace backend
//! only in a build that has `hf-tokenizer` but not `native-tokenizer`.
//! `Native` forces the Pure-Rust backend; `Hf` forces the HuggingFace one and
//! errors, rather than silently substituting the native backend, when the
//! binary was built without `hf-tokenizer`.

use clap::ValueEnum;

/// Which tokenizer backend to use, selectable via `--tokenizer-backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TokenizerBackendChoice {
    /// Prefer the backend this build's `native-tokenizer` /
    /// `hf-tokenizer` feature combination indicates (see the module doc).
    Auto,
    /// Force the Pure-Rust native BPE backend
    /// ([`oxibonsai_tokenizer::OxiTokenizer`]). Always available, on every
    /// target including `wasm32` and `--no-default-features` builds.
    Native,
    /// Force the HuggingFace `tokenizers` backend. Only available when
    /// this binary was built with the opt-in `hf-tokenizer` Cargo feature
    /// (off by default); otherwise selecting it is an error.
    Hf,
}

impl std::fmt::Display for TokenizerBackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Native => "native",
            Self::Hf => "hf",
        })
    }
}

/// Load a [`oxibonsai_runtime::TokenizerBridge`] from `path`, honoring the
/// requested backend choice.
pub(crate) fn load_tokenizer_bridge(
    path: &str,
    backend: TokenizerBackendChoice,
) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge> {
    match resolve_backend(backend) {
        ResolvedBackend::Native => Ok(oxibonsai_runtime::TokenizerBridge::native_from_file(path)?),
        ResolvedBackend::Hf => load_hf(path),
        ResolvedBackend::HfUnavailable => anyhow::bail!(
            "--tokenizer-backend hf was requested but this binary was built without the \
             `hf-tokenizer` feature; rebuild with --features hf-tokenizer, or pass \
             --tokenizer-backend native (or auto)"
        ),
    }
}

/// The name of the backend [`load_tokenizer_bridge`] will actually use for
/// `choice`, for diagnostics (`build-info`, log lines) — never a hardcoded
/// literal.
pub(crate) fn active_backend_name(backend: TokenizerBackendChoice) -> &'static str {
    match resolve_backend(backend) {
        ResolvedBackend::Native => "native",
        ResolvedBackend::Hf => "hf",
        ResolvedBackend::HfUnavailable => "hf (unavailable in this build)",
    }
}

enum ResolvedBackend {
    Native,
    Hf,
    HfUnavailable,
}

fn resolve_backend(choice: TokenizerBackendChoice) -> ResolvedBackend {
    match choice {
        TokenizerBackendChoice::Native => ResolvedBackend::Native,
        TokenizerBackendChoice::Hf => {
            if hf_tokenizer_compiled_in() {
                ResolvedBackend::Hf
            } else {
                ResolvedBackend::HfUnavailable
            }
        }
        TokenizerBackendChoice::Auto => {
            // The CLI's own `native-tokenizer` feature is otherwise
            // observationally inert (cli-08): compiling it in now flips
            // the default backend `Auto` resolves to.
            if cfg!(feature = "native-tokenizer") {
                ResolvedBackend::Native
            } else if hf_tokenizer_compiled_in() {
                ResolvedBackend::Hf
            } else {
                // Neither `native-tokenizer` nor `hf-tokenizer` compiled
                // in: the native backend is always compiled into
                // `oxibonsai_runtime` regardless of either feature, so
                // this is still a real, working choice, not a fallback
                // that silently does nothing.
                ResolvedBackend::Native
            }
        }
    }
}

fn hf_tokenizer_compiled_in() -> bool {
    cfg!(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))
}

#[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
fn load_hf(path: &str) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge> {
    Ok(oxibonsai_runtime::TokenizerBridge::from_file(path)?)
}

#[cfg(not(all(feature = "hf-tokenizer", not(target_arch = "wasm32"))))]
fn load_hf(_path: &str) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge> {
    anyhow::bail!(
        "--tokenizer-backend hf was requested but this binary was built without the \
         `hf-tokenizer` feature; rebuild with --features hf-tokenizer, or pass \
         --tokenizer-backend native (or auto)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_clap_value_names() {
        assert_eq!(TokenizerBackendChoice::Auto.to_string(), "auto");
        assert_eq!(TokenizerBackendChoice::Native.to_string(), "native");
        assert_eq!(TokenizerBackendChoice::Hf.to_string(), "hf");
    }

    #[test]
    fn native_choice_always_resolves_to_native() {
        assert_eq!(
            active_backend_name(TokenizerBackendChoice::Native),
            "native"
        );
    }

    #[test]
    fn auto_choice_resolves_to_a_real_backend_name() {
        // Whichever this build was compiled with, `Auto` must resolve to
        // one of the two real backends, never a placeholder.
        let name = active_backend_name(TokenizerBackendChoice::Auto);
        assert!(name == "native" || name == "hf", "got {name:?}");
    }

    #[test]
    fn loading_a_nonexistent_file_errors_for_every_backend() {
        for backend in [TokenizerBackendChoice::Auto, TokenizerBackendChoice::Native] {
            let result = load_tokenizer_bridge("/nonexistent/tokenizer.json", backend);
            assert!(
                result.is_err(),
                "backend {backend} should error on a missing file"
            );
        }
    }
}
