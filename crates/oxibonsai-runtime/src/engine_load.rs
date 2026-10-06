//! `'static` / path-based constructors of [`InferenceEngine`]: the ones a
//! server or CLI entry point uses, which memory-map a GGUF and leak it to
//! `'static` so the engine (and any pool replicas) can borrow it for the
//! process lifetime.
//!
//! Split out of `engine.rs` to keep that file under the workspace 2000-line
//! ceiling; these are inherent methods of [`InferenceEngine`].

use oxibonsai_core::config::{RopeScalingOverride, RopeScalingOverrideScope};
use oxibonsai_core::gguf::reader::GgufFile;

use crate::engine::{Backend, InferenceEngine};
use crate::engine_seam::resolve_rope_scaling_at_load;
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling::SamplingParams;

impl<'a> InferenceEngine<'a> {
    /// [`from_gguf_with_backend`](Self::from_gguf_with_backend) with a
    /// `--rope-scaling auto|on|off` override (additive —
    /// every existing constructor is unchanged).
    ///
    /// The override is applied through
    /// [`RopeScalingOverrideScope`], which the model constructors'
    /// internal `Qwen3Config::from_metadata` calls consult on this thread:
    /// [`RopeScalingOverride::Off`] builds a plain (unscaled) RoPE table even
    /// when the file declares YaRN (OxiBonsai <= 0.2.4 behaviour on
    /// Bonsai-8B), [`RopeScalingOverride::On`] refuses a file that declares
    /// no scaling, [`RopeScalingOverride::Auto`] is exactly
    /// [`from_gguf_with_backend`](Self::from_gguf_with_backend).
    ///
    /// # Errors
    ///
    /// As [`from_gguf_with_backend`](Self::from_gguf_with_backend), plus the
    /// refusals of [`resolve_rope_scaling_at_load`].
    pub fn from_gguf_with_backend_and_rope(
        gguf: &'a GgufFile<'a>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        backend: Backend,
        rope: RopeScalingOverride,
    ) -> RuntimeResult<Self> {
        resolve_rope_scaling_at_load(gguf, rope)?;
        let _rope_scope = RopeScalingOverrideScope::enter(rope);
        Self::from_gguf_with_backend(gguf, sampling_params, seed, max_seq_len, backend)
    }
}

impl InferenceEngine<'static> {
    /// Build an engine from an already-`'static` [`GgufFile`].
    ///
    /// This is the shared core used both by [`from_gguf_path`](Self::from_gguf_path)
    /// (after it has leaked the mmap + parsed container to `'static`) and by the
    /// engine pool when constructing additional replicas off a single leaked
    /// GGUF — every replica borrows the *same* `&'static GgufFile` zero-copy, so
    /// only per-replica state (KV cache, light wrappers) is duplicated. The
    /// immutable `token_embd` table is shared across replicas via one
    /// `Arc<[f32]>` when the pool builder uses
    /// [`from_gguf_static_with_embd`](Self::from_gguf_static_with_embd).
    ///
    /// Performs no leaking itself; the caller owns the `'static` lifetime.
    ///
    /// # Errors
    ///
    /// Propagates model-init / GPU-cache errors through [`RuntimeError`].
    pub fn from_gguf_static(
        gguf: &'static GgufFile<'static>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        // `from_gguf` is generic over the GGUF borrow lifetime; instantiating it
        // at `'static` yields an `InferenceEngine<'static>` directly.
        Self::from_gguf(gguf, sampling_params, seed, max_seq_len)
    }

    /// Build an engine from an already-`'static` [`GgufFile`], reusing a
    /// pre-loaded, shared token-embedding table.
    ///
    /// The `'static`-lifetime twin of
    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd). The engine pool calls
    /// this for replicas `2..N`, passing the `Arc<[f32]>` extracted from replica
    /// `#1` (via [`InferenceEngine::model_token_embd`]) so every replica shares a
    /// single token-embedding allocation instead of re-dequantizing its own
    /// copy. KV caches and light wrappers remain per-replica.
    ///
    /// `token_embd` MUST be the dequantized `token_embd.weight` for this exact
    /// GGUF; see
    /// [`BonsaiModel::from_gguf_with_embd`](oxibonsai_model::model::BonsaiModel::from_gguf_with_embd)
    /// for the contract.
    pub fn from_gguf_static_with_embd(
        gguf: &'static GgufFile<'static>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_embd(gguf, sampling_params, seed, max_seq_len, token_embd)
    }

    /// Memory-map + parse a GGUF file and leak both allocations to `'static`,
    /// returning the constructed engine *and* the leaked `&'static GgufFile`.
    ///
    /// The leaked reference lets callers (e.g. the engine pool) build additional
    /// engine replicas off the *same* weights via [`from_gguf_static`](Self::from_gguf_static)
    /// without a second mmap or weight copy. The leaked memory is intentional —
    /// the GGUF is expected to live for the process lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::FileNotFound`] if `path` does not exist.  Other
    /// IO / parse / model-init errors propagate through [`RuntimeError`].
    pub fn from_gguf_path_leaked(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<(Self, &'static GgufFile<'static>)> {
        Self::from_gguf_path_leaked_with_backend(
            path,
            sampling_params,
            seed,
            max_seq_len,
            Backend::Auto,
        )
    }

    /// [`from_gguf_path_leaked`](Self::from_gguf_path_leaked) on an explicit
    /// [`Backend`] — the constructor a `--backend` flag drives.
    ///
    /// # Errors
    ///
    /// As [`from_gguf_path_leaked`](Self::from_gguf_path_leaked), plus the
    /// backend refusals of
    /// [`from_gguf_with_backend`](Self::from_gguf_with_backend).
    pub fn from_gguf_path_leaked_with_backend(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        backend: Backend,
    ) -> RuntimeResult<(Self, &'static GgufFile<'static>)> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(RuntimeError::FileNotFound {
                path: path_ref.display().to_string(),
            });
        }

        // Memory-map and parse, then leak both so the resulting `GgufFile`
        // can live for `'static` without RAII concerns.
        let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(path_ref)?;
        let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(mmap)?;
        let gguf: &'static GgufFile<'static> = Box::leak(Box::new(gguf));

        let engine =
            Self::from_gguf_with_backend(gguf, sampling_params, seed, max_seq_len, backend)?;
        Ok((engine, gguf))
    }

    /// Load an [`InferenceEngine`] directly from a path to a GGUF file.
    ///
    /// This is a convenience wrapper intended for server/CLI entry points that
    /// need an owned, `'static` engine.  It memory-maps the file, parses the
    /// GGUF container, and leaks both allocations so that the borrowed
    /// `GgufFile<'a>` lifetime can be promoted to `'static`.
    ///
    /// The leaked memory is intentional — the engine is expected to live for
    /// the process lifetime.  Do not call this in hot-paths.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::FileNotFound`] if `path` does not exist.  Other
    /// IO / parse / model-init errors propagate through [`RuntimeError`].
    pub fn from_gguf_path(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_path_leaked(path, sampling_params, seed, max_seq_len)
            .map(|(engine, _gguf)| engine)
    }

    /// [`from_gguf_path`](Self::from_gguf_path) on an explicit [`Backend`].
    ///
    /// # Errors
    ///
    /// As [`from_gguf_path_leaked_with_backend`](Self::from_gguf_path_leaked_with_backend).
    pub fn from_gguf_path_with_backend(
        path: impl AsRef<std::path::Path>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        backend: Backend,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_path_leaked_with_backend(path, sampling_params, seed, max_seq_len, backend)
            .map(|(engine, _gguf)| engine)
    }

    /// [`from_gguf_static`](Self::from_gguf_static) on an explicit
    /// [`Backend`] — what the engine pool uses for replicas `2..N` so every
    /// replica runs on the backend replica `#1` was built for.
    ///
    /// # Errors
    ///
    /// As [`from_gguf_with_backend`](Self::from_gguf_with_backend).
    pub fn from_gguf_static_with_embd_and_backend(
        gguf: &'static GgufFile<'static>,
        sampling_params: SamplingParams,
        seed: u64,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
        backend: Backend,
    ) -> RuntimeResult<Self> {
        Self::from_gguf_with_embd_and_backend(
            gguf,
            sampling_params,
            seed,
            max_seq_len,
            token_embd,
            backend,
        )
    }
}
