//! Whether the served engine can take image content parts.
//!
//! An image request is only worth decoding, encoding and prefilling when the
//! engine behind the server can run it, and the engine answers that itself:
//! [`InferenceEngine::multimodal_executor`] names the executor a multimodal
//! prefill runs on, or returns the engine's typed refusal. Nothing here names
//! a backend or decides capability on its own — the refusal a client sees is
//! the engine's, with its stable code and its own words, so an executor that
//! gains (or loses) an image prefill changes this answer by itself.
//!
//! The answer is computed once, with the rest of the served model's
//! description ([`crate::server::ModelDescriptor::image_support`]), and both
//! chat endpoints consult it **before** they render the prompt, decode an
//! image or run the vision tower — and before they look for a vision
//! projector, since a projector cannot help an engine that refuses image
//! turns. An engine that cannot serve images therefore answers the typed
//! refusal below — never a `500` "generation failed" after the tower's
//! seconds of work, and never an SSE error event after a `200`.
//!
//! # The refusals
//!
//! All are `400` with a stable `error.code` naming the reason (the same
//! status `vision_unavailable`, the missing-projector refusal, has):
//!
//! | `error.code` | cause |
//! |---|---|
//! | `NOT_A_HYBRID_MODEL` | the engine holds a dense (`qwen3`-family) model |
//! | `BACKEND_UNAVAILABLE` | reserved: no shipped executor raises it (both hybrid executors, the CPU model and the Metal hybrid runner, prefill image rows); it stays for an executor without an image prefill |
//!
//! The upper-case spelling is the engine's own code
//! ([`crate::engine_seam::engine_error_code`]); the request-level codes
//! (`image_*`, `vision_unavailable`, `too_many_images`) are lower-case.

use crate::engine::InferenceEngine;
use crate::engine_seam::engine_error_code;
use crate::error::RuntimeError;
use crate::server::api_error::ApiError;

/// The code an image request is refused under when the engine's refusal
/// carries no engine code of its own (never the case for the refusals
/// [`InferenceEngine::multimodal_executor`] returns today).
const UNCODED_REFUSAL: &str = "BACKEND_UNAVAILABLE";

/// Whether the engine behind a server can prefill image rows, and if not,
/// the engine's own refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSupport {
    /// The engine prefills image rows.
    Supported,
    /// The engine refuses image turns.
    Refused {
        /// The refusal's stable code (`NOT_A_HYBRID_MODEL`,
        /// `BACKEND_UNAVAILABLE`).
        code: &'static str,
        /// The engine's own message: what it holds or decodes on, the
        /// constraint, and the way out where there is one.
        message: String,
    },
}

impl ImageSupport {
    /// What `engine` can do with image rows: its own pre-encode check
    /// ([`InferenceEngine::multimodal_executor`]), which reads no state.
    #[must_use]
    pub fn of_engine(engine: &InferenceEngine<'_>) -> Self {
        match engine.multimodal_executor() {
            Ok(_) => Self::Supported,
            Err(refusal) => Self::from_refusal(&refusal),
        }
    }

    /// The support an engine whose pre-encode check returned `refusal` has:
    /// the refusal's stable code and, for a typed engine refusal, the
    /// engine's message without the `engine error: [CODE]` frame (the code
    /// travels in its own field).
    #[must_use]
    pub fn from_refusal(refusal: &RuntimeError) -> Self {
        let code = engine_error_code(refusal).unwrap_or(UNCODED_REFUSAL);
        let message = match refusal {
            RuntimeError::Engine(engine) => engine.to_string(),
            other => other.to_string(),
        };
        Self::Refused { code, message }
    }

    /// `true` when image rows can be prefilled.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// The typed `400` an image request is answered with when the engine
    /// cannot serve it; `None` when it can.
    #[must_use]
    pub fn refusal(&self) -> Option<ApiError> {
        match self {
            Self::Supported => None,
            Self::Refused { code, message } => {
                Some(ApiError::bad_request(message.clone(), "messages").with_code(code))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_seam::Backend;
    use crate::sampling::SamplingParams;
    use crate::vision_prefill::multimodal_backend_refusal;
    use axum::http::StatusCode;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf;

    #[test]
    fn a_supporting_engine_has_nothing_to_refuse() {
        assert!(ImageSupport::Supported.refusal().is_none());
        assert!(ImageSupport::Supported.is_supported());
    }

    /// A dense engine is a `400` named `NOT_A_HYBRID_MODEL` that says which
    /// architecture it holds — the engine's own refusal, read off a real
    /// dense engine.
    #[test]
    fn a_dense_engine_is_refused_as_not_a_hybrid_model() {
        let dense = InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            SamplingParams::default(),
            1,
        );
        let support = ImageSupport::of_engine(&dense);
        assert!(!support.is_supported());
        let refusal = support
            .refusal()
            .expect("a dense engine cannot serve images");
        assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
        let json = refusal.to_json();
        assert_eq!(json["error"]["code"], "NOT_A_HYBRID_MODEL", "{json}");
        assert_eq!(json["error"]["param"], "messages", "{json}");
        let message = refusal.message();
        assert!(message.contains("hybrid"), "{message}");
        assert!(
            message.contains(dense.architecture()),
            "names what it holds: {message}"
        );
        assert!(
            !message.contains("engine error"),
            "the code travels in its own field, not in a frame around the message: {message}"
        );
        let early = dense.multimodal_executor().expect_err("dense");
        assert_eq!(ImageSupport::from_refusal(&early), support);
    }

    /// An executor without an image prefill is a `400` named
    /// `BACKEND_UNAVAILABLE` whose message is the engine's own refusal: the
    /// constraint, the executor and the way out.
    #[test]
    fn an_executor_without_an_image_prefill_is_refused_as_backend_unavailable() {
        let refusal_error = multimodal_backend_refusal(Backend::Metal, "qwen35");
        let support = ImageSupport::from_refusal(&refusal_error);
        assert!(!support.is_supported());
        let refusal = support.refusal().expect("a refusal");
        assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
        let json = refusal.to_json();
        assert_eq!(json["error"]["code"], "BACKEND_UNAVAILABLE", "{json}");
        assert_eq!(json["error"]["param"], "messages", "{json}");
        let message = refusal.message();
        assert!(message.contains("metal"), "names the executor: {message}");
        assert!(
            message.contains("--backend cpu"),
            "names the way out: {message}"
        );
        let RuntimeError::Engine(engine) = &refusal_error else {
            panic!("the refusal is a typed engine error: {refusal_error:?}");
        };
        assert_eq!(message, engine.to_string(), "the engine's own words");
    }

    /// A refusal that is not a typed engine error still answers `400` under
    /// a stable code, with the error's own text.
    #[test]
    fn an_uncoded_refusal_keeps_its_text_under_the_fallback_code() {
        let support = ImageSupport::from_refusal(&RuntimeError::Config("no rows here".into()));
        let refusal = support.refusal().expect("a refusal");
        assert_eq!(refusal.status(), StatusCode::BAD_REQUEST);
        assert_eq!(refusal.to_json()["error"]["code"], UNCODED_REFUSAL);
        assert!(refusal.message().contains("no rows here"));
    }

    /// The decision is the engine's own, read off real engines: a dense
    /// engine refuses, and every hybrid engine this host builds — the CPU
    /// model, and the Metal runner where there is one — answers exactly what
    /// its pre-encode check answers. Nothing in this module knows which
    /// executor that is.
    #[test]
    fn the_support_is_read_from_the_engines_own_pre_encode_check() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
        for backend in [Backend::Cpu, Backend::Auto, Backend::Metal] {
            let Ok(engine) = InferenceEngine::from_gguf_with_backend(
                &gguf,
                SamplingParams::default(),
                1,
                64,
                backend,
            ) else {
                assert_ne!(backend, Backend::Cpu, "a CPU hybrid engine always builds");
                continue;
            };
            let support = ImageSupport::of_engine(&engine);
            match engine.multimodal_executor() {
                Ok(_) => assert_eq!(support, ImageSupport::Supported, "{backend}"),
                Err(refusal) => {
                    assert_eq!(support, ImageSupport::from_refusal(&refusal), "{backend}");
                }
            }
            assert_eq!(
                support.is_supported(),
                engine.prefills_images(),
                "{backend}"
            );
        }
    }
}
