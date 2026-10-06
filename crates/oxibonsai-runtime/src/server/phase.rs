//! Which stage of a generation request the per-request deadline caught it in.
//!
//! A request to any of the three generation endpoints (`/v1/chat/completions`,
//! `/v1/chat/completions/extended`, `/v1/completions`) — or to `/rag/query`
//! (the `rag` feature) — moves through its
//! stages one after another, and on a chat request that carries images they
//! are far from equally long: the vision
//! encode and the prefill of every image row run on the executor the server
//! decodes on, and on the Bonsai 2 27B that prefill takes seconds on the
//! Metal hybrid runner but tens of seconds to minutes on the CPU model
//! (67 rows 54.9 s, 75 rows 32.6 s — measured on separate runs at different
//! host load; order of magnitude only), while a token decodes in a second or
//! two; a remote image adds its fetch. A bare "request exceeded the
//! per-request timeout" leaves the operator guessing which knob to turn;
//! naming the stage does not.
//!
//! The handler records the stage it is entering in a [`RequestPhase`] shared
//! (through the request's cancel slot, [`crate::server::deadline::CancelSlot`])
//! with everything that runs on behalf of the request:
//!
//! | [`Phase`] | what runs | who records the edge |
//! |---|---|---|
//! | `Preparing` | template render (chat), retrieving context (`/rag/query`), tokenizing, image decode + resize | the request's start |
//! | `ImageFetch` | fetching one remote `image_url` (only when the operator enabled remote images) | the request's image fetcher, around each fetch ([`crate::server::image_fetch`]); back to `Preparing` once it ends |
//! | `VisionEncode` | the vision tower over the request's images | the handler, right before the encode |
//! | `WaitingForEngine` | queued on the engine pool | the generation path, before it acquires a replica |
//! | `Prefill` | the engine ingesting the prompt | the generation path, once it holds a replica |
//! | `Decode` | the engine producing tokens (every choice / prompt of a multi-generation request) | the first generated token (a channel receiver, or the logprobs callback) |
//!
//! The engine reports no progress of its own, so "prefill" is exactly "holds a
//! replica and has produced no token yet" and "decode" is "has produced at
//! least one". When the deadline fires, [`PhaseSnapshot::timeout_error`] turns
//! the current stage into the `504 request_timeout` a client sees, with the
//! stage's stable name in `error.phase`.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::server::api_error::ApiError;
use crate::vision_prefill::ChatPrompt;

/// A stage of a chat request (see the module docs for the table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Phase {
    /// Rendering the prompt or retrieving its context, tokenizing, and
    /// reading the request's images.
    Preparing = 0,
    /// The vision tower encoding the request's images.
    VisionEncode = 1,
    /// Queued for a free engine replica.
    WaitingForEngine = 2,
    /// The engine ingesting the prompt: no token generated yet.
    Prefill = 3,
    /// The engine producing tokens.
    Decode = 4,
    /// Fetching one of the request's remote images (a sub-stage of
    /// `Preparing`).
    ImageFetch = 5,
}

impl Phase {
    /// The stage's stable name, as `error.phase` carries it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::VisionEncode => "vision_encode",
            Self::WaitingForEngine => "waiting_for_engine",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::ImageFetch => "image_fetch",
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::VisionEncode,
            2 => Self::WaitingForEngine,
            3 => Self::Prefill,
            4 => Self::Decode,
            5 => Self::ImageFetch,
            _ => Self::Preparing,
        }
    }
}

#[derive(Debug, Default)]
struct PhaseState {
    phase: AtomicU8,
    /// Sequence positions of the prompt (image rows included); `0` until known.
    prompt_rows: AtomicUsize,
    /// Images the request carries.
    images: AtomicUsize,
    /// Tokens generated so far.
    generated: AtomicUsize,
}

/// The shared, lock-free record of the stage a request is in. Cloning yields
/// another handle to the same record.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestPhase(Arc<PhaseState>);

impl RequestPhase {
    /// The request has entered `phase`.
    pub(crate) fn enter(&self, phase: Phase) {
        self.0.phase.store(phase as u8, Ordering::Release);
    }

    /// The stage the request is in.
    pub(crate) fn current(&self) -> Phase {
        Phase::from_u8(self.0.phase.load(Ordering::Acquire))
    }

    /// Record the prompt's size once it is known: its sequence positions
    /// (image rows included) and how many images it carries.
    pub(crate) fn set_workload(&self, prompt_rows: usize, images: usize) {
        self.0.prompt_rows.store(prompt_rows, Ordering::Relaxed);
        self.0.images.store(images, Ordering::Relaxed);
    }

    /// One token was generated. The first one moves the request from prefill
    /// to decode; a token never moves a request backwards or out of a stage
    /// it was not in.
    pub(crate) fn token_generated(&self) {
        self.0.generated.fetch_add(1, Ordering::Relaxed);
        self.decode_started();
    }

    /// The engine has begun producing tokens, without a count: the signal a
    /// path that only sees *that* a token exists (the per-token callback of a
    /// `logprobs` generation, which also runs for alternatives) can give.
    /// `Prefill -> Decode` only: a stray late signal must not resurrect a
    /// finished request's stage.
    pub(crate) fn decode_started(&self) {
        self.advance(Phase::Prefill, Phase::Decode);
    }

    /// A remote image fetch begins: `Preparing -> ImageFetch` only, so a
    /// fetch reported late can never pull a request back from a later stage.
    pub(crate) fn image_fetch_started(&self) {
        self.advance(Phase::Preparing, Phase::ImageFetch);
    }

    /// The remote image fetch ended: `ImageFetch -> Preparing` only (the
    /// request goes on decoding and resizing its images).
    pub(crate) fn image_fetch_finished(&self) {
        self.advance(Phase::ImageFetch, Phase::Preparing);
    }

    /// Move `from -> to`, and nothing else.
    fn advance(&self, from: Phase, to: Phase) {
        let _ = self.0.phase.compare_exchange(
            from as u8,
            to as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// A consistent-enough copy of the record for a message.
    pub(crate) fn snapshot(&self) -> PhaseSnapshot {
        PhaseSnapshot {
            phase: self.current(),
            prompt_rows: self.0.prompt_rows.load(Ordering::Relaxed),
            images: self.0.images.load(Ordering::Relaxed),
            generated: self.0.generated.load(Ordering::Relaxed),
        }
    }
}

/// What a [`RequestPhase`] held at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhaseSnapshot {
    /// The stage.
    pub(crate) phase: Phase,
    /// Prompt positions (`0` when not yet known).
    pub(crate) prompt_rows: usize,
    /// Images in the request.
    pub(crate) images: usize,
    /// Tokens generated so far.
    pub(crate) generated: usize,
}

impl PhaseSnapshot {
    /// The stage in words, to follow "request exceeded the server's
    /// per-request timeout of N ms".
    pub(crate) fn describe(&self) -> String {
        let prompt = if self.prompt_rows > 0 {
            format!("the {}-position prompt", self.prompt_rows)
        } else {
            "the prompt".to_string()
        };
        match self.phase {
            Phase::Preparing => {
                "while preparing the request (rendering the prompt or retrieving its context, tokenizing it, and reading its images)"
                    .to_string()
            }
            Phase::ImageFetch => {
                "while fetching one of the request's remote images (the request's image_url)"
                    .to_string()
            }
            Phase::VisionEncode => format!(
                "during the vision encode of the request's {}",
                plural(self.images, "image")
            ),
            Phase::WaitingForEngine => {
                "while waiting for a free engine replica (every replica stayed busy)".to_string()
            }
            Phase::Prefill => {
                format!("during prefill of {prompt} (no token had been generated yet)")
            }
            Phase::Decode if self.generated > 0 => format!(
                "during decode, after {} of {prompt}",
                plural(self.generated, "generated token")
            ),
            Phase::Decode => format!("during decode of {prompt}"),
        }
    }

    /// The `504 request_timeout` for a deadline that expired at this stage:
    /// the message says which stage (and, for an image request, that the
    /// image work is what makes the request slow), and `error.phase` carries
    /// the stage's stable name (`error.generated_tokens` the decode
    /// progress).
    pub(crate) fn timeout_error(&self, limit_ms: u128) -> ApiError {
        let mut message = format!(
            "request exceeded the server's per-request timeout of {limit_ms} ms {}",
            self.describe()
        );
        if self.images > 0 && matches!(self.phase, Phase::VisionEncode | Phase::Prefill) {
            message.push_str(
                "; the vision encode and the prefill of every image row run on the executor the \
                 server decodes on and grow with the image, so raise the server's per-request \
                 timeout for image requests",
            );
        }
        if self.phase == Phase::ImageFetch {
            message.push_str(
                "; a remote image is fetched within its own per-image deadline, and the fetches \
                 of a request count toward the server's per-request timeout, so a slow image \
                 server needs a larger per-request timeout (or the image sent inline as a data \
                 URI)",
            );
        }
        let error = ApiError::timeout(message).with_field("phase", self.phase.name());
        if self.phase == Phase::Decode && self.generated > 0 {
            error.with_field("generated_tokens", self.generated)
        } else {
            error
        }
    }
}

/// [`ChatPrompt::generate`] that keeps `phase` current while it runs — for a
/// path with no token callback of its own (a plain non-streaming request),
/// so a deadline that expires mid-generation can say whether the engine was
/// still in prefill or had begun to decode, and after how many tokens.
///
/// The ids come back through the engine's streaming primitive
/// (`generate_streaming_sync`), which decodes exactly as `generate` does — the
/// same prefill, the same GPU-argmax and top-k routes, the same sampler draws,
/// stop and cancellation checks; it only also sends each id as it is drawn.
/// A collector thread counts them as they arrive (so the count is live, not
/// known only afterwards) and returns them in order.
///
/// # Errors
///
/// Whatever the generation returns, or [`RuntimeError::GenerationStopped`]
/// when the collector thread itself failed.
pub(crate) fn generate_observed(
    prompt: &ChatPrompt,
    engine: &mut InferenceEngine<'_>,
    max_tokens: usize,
    phase: &RequestPhase,
) -> RuntimeResult<Vec<u32>> {
    observe_generation(phase, |tx| {
        prompt.generate_streaming_sync(engine, max_tokens, tx)
    })
}

/// Run `generate` — a streaming generation that sends each id through the
/// channel it is handed, as `generate_streaming_sync` does — while a
/// collector thread counts the ids into `phase` as they arrive, and return
/// them in order. [`generate_observed`] is this over a [`ChatPrompt`]; the
/// legacy completions endpoint runs it over its own token ids.
///
/// # Errors
///
/// Whatever `generate` returns, or [`RuntimeError::GenerationStopped`] when
/// the collector thread itself failed.
pub(crate) fn observe_generation<G>(phase: &RequestPhase, generate: G) -> RuntimeResult<Vec<u32>>
where
    G: FnOnce(&std::sync::mpsc::Sender<u32>) -> RuntimeResult<usize>,
{
    let (tx, rx) = std::sync::mpsc::channel::<u32>();
    std::thread::scope(|scope| {
        let collector = scope.spawn(move || {
            let mut tokens = Vec::new();
            for token in rx {
                phase.token_generated();
                tokens.push(token);
            }
            tokens
        });
        let outcome = generate(&tx);
        // Closing the channel ends the collector's loop.
        drop(tx);
        let tokens = collector.join();
        match (outcome, tokens) {
            (Err(error), _) => Err(error),
            (Ok(_), Ok(tokens)) => Ok(tokens),
            (Ok(_), Err(_)) => Err(RuntimeError::GenerationStopped {
                reason: "the token collector thread of a non-streaming generation failed"
                    .to_string(),
            }),
        }
    })
}

/// `"1 image"` / `"2 images"` (a count of zero reads as the bare plural).
fn plural(count: usize, noun: &str) -> String {
    match count {
        0 => format!("{noun}s"),
        1 => format!("1 {noun}"),
        n => format!("{n} {noun}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_request_is_preparing() {
        let phase = RequestPhase::default();
        assert_eq!(phase.current(), Phase::Preparing);
        assert_eq!(phase.snapshot().generated, 0);
    }

    #[test]
    fn every_stage_has_a_stable_name_and_round_trips() {
        let phase = RequestPhase::default();
        for (stage, name) in [
            (Phase::Preparing, "preparing"),
            (Phase::VisionEncode, "vision_encode"),
            (Phase::WaitingForEngine, "waiting_for_engine"),
            (Phase::Prefill, "prefill"),
            (Phase::Decode, "decode"),
            (Phase::ImageFetch, "image_fetch"),
        ] {
            phase.enter(stage);
            assert_eq!(phase.current(), stage);
            assert_eq!(stage.name(), name);
        }
    }

    /// A fetch moves a preparing request to `image_fetch` and back, and
    /// never touches a request in any other stage (a late report from the
    /// blocking pool cannot rewind a request that moved on).
    #[test]
    fn an_image_fetch_is_a_sub_stage_of_preparing_only() {
        let phase = RequestPhase::default();
        phase.image_fetch_started();
        assert_eq!(phase.current(), Phase::ImageFetch);
        phase.image_fetch_finished();
        assert_eq!(phase.current(), Phase::Preparing);
        for stage in [
            Phase::VisionEncode,
            Phase::WaitingForEngine,
            Phase::Prefill,
            Phase::Decode,
        ] {
            let other = RequestPhase::default();
            other.enter(stage);
            other.image_fetch_started();
            assert_eq!(other.current(), stage, "{stage:?}");
            other.image_fetch_finished();
            assert_eq!(other.current(), stage, "{stage:?}");
        }
    }

    #[test]
    fn a_deadline_during_an_image_fetch_names_the_fetch() {
        let fetching = message(Phase::ImageFetch, 0, 2, 0);
        assert!(
            fetching.contains("while fetching one of the request's remote images"),
            "{fetching}"
        );
        assert!(
            fetching.contains("per-image deadline") && fetching.contains("data URI"),
            "{fetching}"
        );
        let error = PhaseSnapshot {
            phase: Phase::ImageFetch,
            prompt_rows: 0,
            images: 1,
            generated: 0,
        }
        .timeout_error(1_500)
        .to_json();
        assert_eq!(error["error"]["code"], "request_timeout");
        assert_eq!(error["error"]["phase"], "image_fetch");
    }

    #[test]
    fn the_first_token_moves_prefill_to_decode_and_no_other_stage() {
        let phase = RequestPhase::default();
        phase.enter(Phase::Prefill);
        phase.token_generated();
        assert_eq!(phase.current(), Phase::Decode);
        phase.token_generated();
        assert_eq!(phase.snapshot().generated, 2);

        // A token counted while not in prefill changes nothing about the stage.
        for stage in [
            Phase::Preparing,
            Phase::VisionEncode,
            Phase::WaitingForEngine,
        ] {
            let other = RequestPhase::default();
            other.enter(stage);
            other.token_generated();
            assert_eq!(other.current(), stage, "{stage:?}");
        }
    }

    #[test]
    fn clones_share_one_record() {
        let phase = RequestPhase::default();
        let seen_by_the_timeout_branch = phase.clone();
        phase.set_workload(67, 1);
        phase.enter(Phase::Prefill);
        let snapshot = seen_by_the_timeout_branch.snapshot();
        assert_eq!(snapshot.phase, Phase::Prefill);
        assert_eq!((snapshot.prompt_rows, snapshot.images), (67, 1));
    }

    fn message(phase: Phase, prompt_rows: usize, images: usize, generated: usize) -> String {
        PhaseSnapshot {
            phase,
            prompt_rows,
            images,
            generated,
        }
        .timeout_error(60_000)
        .message()
        .to_string()
    }

    #[test]
    fn each_stage_is_named_in_the_message() {
        let preparing = message(Phase::Preparing, 0, 0, 0);
        assert!(
            preparing.contains("per-request timeout of 60000 ms"),
            "{preparing}"
        );
        assert!(
            preparing.contains("while preparing the request"),
            "{preparing}"
        );

        let encode = message(Phase::VisionEncode, 67, 2, 0);
        assert!(
            encode.contains("during the vision encode of the request's 2 images"),
            "{encode}"
        );

        let queued = message(Phase::WaitingForEngine, 67, 1, 0);
        assert!(
            queued.contains("waiting for a free engine replica"),
            "{queued}"
        );

        let prefill = message(Phase::Prefill, 67, 1, 0);
        assert!(
            prefill.contains("during prefill of the 67-position prompt"),
            "{prefill}"
        );
        assert!(
            prefill.contains("no token had been generated yet"),
            "{prefill}"
        );

        let decode = message(Phase::Decode, 67, 1, 12);
        assert!(
            decode.contains("during decode, after 12 generated tokens"),
            "{decode}"
        );
        let one = message(Phase::Decode, 5, 0, 1);
        assert!(
            one.contains("after 1 generated token of the 5-position prompt"),
            "{one}"
        );
    }

    #[test]
    fn an_image_request_slowed_by_its_images_is_told_to_raise_the_timeout() {
        for stage in [Phase::VisionEncode, Phase::Prefill] {
            let with_images = message(stage, 67, 1, 0);
            assert!(
                with_images.contains("raise the server's per-request timeout"),
                "{with_images}"
            );
            // The image work runs where the server decodes (the Metal hybrid
            // runner or the CPU model), and the message says so instead of
            // naming one executor.
            assert!(
                with_images.contains("run on the executor the server decodes on"),
                "{with_images}"
            );
            let text_only = message(stage, 67, 0, 0);
            assert!(
                !text_only.contains("raise"),
                "a text prompt is not slowed by images: {text_only}"
            );
        }
        assert!(
            !message(Phase::Decode, 67, 1, 3).contains("raise"),
            "decode is not image work"
        );
    }

    #[test]
    fn the_error_is_a_504_request_timeout_carrying_the_stage() {
        let error = PhaseSnapshot {
            phase: Phase::Decode,
            prompt_rows: 67,
            images: 1,
            generated: 12,
        }
        .timeout_error(1_500);
        assert_eq!(error.status(), axum::http::StatusCode::GATEWAY_TIMEOUT);
        let json = error.to_json();
        assert_eq!(json["error"]["code"], "request_timeout");
        assert_eq!(json["error"]["phase"], "decode");
        assert_eq!(json["error"]["generated_tokens"], 12);

        let queued = PhaseSnapshot {
            phase: Phase::WaitingForEngine,
            prompt_rows: 0,
            images: 0,
            generated: 0,
        }
        .timeout_error(1_500)
        .to_json();
        assert_eq!(queued["error"]["phase"], "waiting_for_engine");
        assert!(queued["error"]["generated_tokens"].is_null());
    }

    #[test]
    fn a_decode_signal_without_a_count_moves_only_prefill_to_decode() {
        let phase = RequestPhase::default();
        phase.decode_started();
        assert_eq!(
            phase.current(),
            Phase::Preparing,
            "not in prefill: no change"
        );
        phase.enter(Phase::Prefill);
        phase.decode_started();
        assert_eq!(phase.current(), Phase::Decode);
        let caught = phase.snapshot();
        assert_eq!(caught.generated, 0, "the signal carries no count");
        let message = caught.timeout_error(10).message().to_string();
        assert!(message.contains("during decode of the prompt"), "{message}");
        assert!(!message.contains("after 0"), "{message}");
    }

    // ── generate_observed: the same tokens as `generate`, plus the edge ─────

    use crate::engine_seam::Backend;
    use crate::sampling::SamplingParams;
    use crate::vision_prefill::{EncodedImage, MultimodalPrompt};
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_model::vision::{GridSize, VisionTokenIds};
    use oxibonsai_testkit::qwen35_fixture::{synthetic_qwen35_gguf, HIDDEN};

    const MAX_SEQ: usize = 64;
    const IDS: VisionTokenIds = VisionTokenIds {
        vision_start: 500,
        vision_end: 501,
        image_pad: 502,
    };
    const SEED: u64 = 7;

    fn engine<'a>(gguf: &'a GgufFile<'a>, temperature: f32) -> InferenceEngine<'a> {
        let params = SamplingParams {
            temperature,
            ..SamplingParams::default()
        };
        InferenceEngine::from_gguf_with_backend(gguf, params, SEED, MAX_SEQ, Backend::Cpu)
            .expect("a CPU hybrid engine")
    }

    fn image_prompt() -> ChatPrompt {
        let grid = GridSize { h: 2, w: 2 };
        let rows = (0..grid.n_tokens() * HIDDEN)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 16_777_216.0 - 0.5)
            .collect();
        let image = EncodedImage {
            rows,
            grid,
            source: (64, 64),
        };
        let tokens = vec![
            10,
            11,
            IDS.vision_start,
            IDS.image_pad,
            IDS.vision_end,
            12,
            13,
        ];
        ChatPrompt::Multimodal(
            MultimodalPrompt::new(tokens, vec![image], IDS).expect("the prompt splices"),
        )
    }

    /// The observed generation is `generate` — every token — for a greedy and
    /// for a seeded sampled request, text and image prompts alike.
    #[test]
    fn observed_generation_returns_exactly_the_plain_generations_tokens() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
        let prompts = [ChatPrompt::Text(vec![5, 9, 13, 21, 34]), image_prompt()];
        for temperature in [0.0f32, 0.9] {
            for prompt in &prompts {
                let plain = prompt
                    .generate(&mut engine(&gguf, temperature), 6)
                    .expect("plain generation");
                assert!(!plain.is_empty(), "a comparison over at least one token");
                let phase = RequestPhase::default();
                phase.enter(Phase::Prefill);
                let observed =
                    generate_observed(prompt, &mut engine(&gguf, temperature), 6, &phase)
                        .expect("observed generation");
                assert_eq!(
                    observed,
                    plain,
                    "temperature {temperature}, {} image(s)",
                    prompt.image_count()
                );
                let caught = phase.snapshot();
                assert_eq!(caught.phase, Phase::Decode);
                assert_eq!(caught.generated, plain.len(), "the live count is the total");
            }
        }
    }

    /// The same equivalence on a dense engine: `generate` and the streaming
    /// primitive the observed generation drives share their routes (prefill,
    /// GPU argmax and top-k on a fused Metal engine, the classic loop
    /// otherwise), so a plain non-streaming request answers what it did
    /// before the stage record existed — greedy and seeded-sampled alike.
    #[test]
    fn observed_generation_matches_the_plain_generation_on_a_dense_engine() {
        for temperature in [0.0f32, 0.9] {
            let dense = || {
                InferenceEngine::new(
                    oxibonsai_core::config::Qwen3Config::tiny_test(),
                    SamplingParams {
                        temperature,
                        ..SamplingParams::default()
                    },
                    SEED,
                )
            };
            let prompt = ChatPrompt::Text(vec![5, 9, 13, 21, 34]);
            let plain = prompt.generate(&mut dense(), 6).expect("plain generation");
            let phase = RequestPhase::default();
            phase.enter(Phase::Prefill);
            let observed =
                generate_observed(&prompt, &mut dense(), 6, &phase).expect("observed generation");
            assert_eq!(observed, plain, "temperature {temperature}");
            assert_eq!(phase.snapshot().generated, plain.len());
        }
    }

    #[test]
    fn a_generation_cancelled_during_prefill_stays_in_prefill() {
        let bytes = synthetic_qwen35_gguf();
        let gguf = GgufFile::parse(&bytes).expect("the fixture parses");
        let mut engine = engine(&gguf, 0.0);
        engine.arm_cancellation().cancel();
        let phase = RequestPhase::default();
        phase.enter(Phase::Prefill);
        let tokens = generate_observed(&ChatPrompt::Text(vec![5, 9, 13]), &mut engine, 6, &phase)
            .expect("a cancelled generation is not an error");
        assert!(
            tokens.is_empty(),
            "cancelled before the prompt was ingested"
        );
        assert_eq!(
            phase.current(),
            Phase::Prefill,
            "no token was produced, so the request never left prefill"
        );
        assert_eq!(phase.snapshot().generated, 0);
    }
}
