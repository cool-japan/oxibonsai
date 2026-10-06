//! `oxibonsai chat` — interactive multi-turn conversation.
//!
//! Every turn renders the WHOLE conversation through the model's own chat
//! template ([`generate::render_prompt`]; cli-11): the GGUF's
//! `tokenizer.chat_template` (or the ChatML/Qwen3 fallback for a file that
//! ships none), with `--think`/`--no-think`, `--reasoning-effort` and
//! `--tools` applied. The reply is split on the `</think>` token id into
//! reasoning (printed to stderr, or dropped with `--hide-reasoning`) and
//! content, and the assistant turn is appended to the history with its
//! `reasoning_content`, so the next turn re-renders it exactly the way the
//! PrismML fork's own reference renderer does for a prior assistant turn
//! carrying reasoning content. When the rendered conversation no longer
//! fits the context
//! window, the oldest turns are dropped (system messages are kept).
//!
//! With `--mmproj` and `--image` (a Bonsai 2 model), the images are encoded
//! once when the session starts and attached as content parts to the first
//! user message (ahead of its text), so every turn that still holds that
//! message re-prefills them in place of their `<|image_pad|>` placeholders
//! (design §6.2). Context accounting counts the image rows; once the image
//! message has to be dropped to fit, the conversation continues as text.
//! `/reset` starts over with the images attached to the next message.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxibonsai_model::vision::{GridSize, VisionTokenIds};
use oxibonsai_runtime::config::{RenderMessage, ResolvedChatTemplate};
use oxibonsai_runtime::sampling::PenaltyParams;
use oxibonsai_runtime::vision_prefill::{ChatPrompt, EncodedImage, MultimodalPrompt};

use super::bonsai2;
use super::cmd_run::{self, ConstrainedSampling, EngineLoad};
use super::generate::{self, ChatContract, ReasoningDisplay, TokenPrinter};
use super::model_source::ModelSource;
use super::tokenizer_backend::TokenizerBackendChoice;
use super::util::{
    build_sampling_params, missing_tokenizer_warning, model_vocab_size,
    reject_penalties_with_constrained_decode, resolve_tokenizer_vocab_aware, StopChecker,
};
use super::vision;

/// Resolved arguments for `oxibonsai chat`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04). See [`cmd_run::RunArgs`] for the field
/// semantics this mirrors.
pub(crate) struct ChatArgs {
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) seed: u64,
    pub(crate) max_seq_len: Option<usize>,
    pub(crate) tokenizer: Option<String>,
    pub(crate) tokenizer_backend: TokenizerBackendChoice,
    pub(crate) grammar: Option<String>,
    pub(crate) stop: Vec<String>,
    pub(crate) backend: oxibonsai_runtime::engine_seam::Backend,
    pub(crate) rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) tools: Option<String>,
    pub(crate) show_reasoning: bool,
    pub(crate) hide_reasoning: bool,
    pub(crate) ptq1_transcode: bool,
    pub(crate) prefill_chunk: Option<usize>,
    pub(crate) vision: bonsai2::VisionRequest,
    /// `--allow-image-url-fetch`, `--image-url-timeout-ms` and
    /// `--image-url-allow-host` (and their `OXI_*` fallbacks): which image
    /// references resolve, and how a remote one is fetched.
    pub(crate) image_sources: bonsai2::ImageSourceFlags,
    pub(crate) allow_vocab_mismatch: bool,
}

/// One rendered turn, ready to generate from.
pub(crate) struct RenderedTurn {
    /// The prompt token ids.
    pub(crate) tokens: Vec<u32>,
    /// Whether the rendered prompt left the model inside an open `<think>`.
    pub(crate) started_in_think: bool,
    /// How many of the oldest history messages had to be dropped to fit.
    pub(crate) dropped_messages: usize,
    /// The generation budget to actually decode with: equal to the
    /// requested `max_tokens` unless
    /// even dropping every droppable message left no room for the full
    /// request, in which case it is clamped to exactly what remains (a
    /// stderr notice was already printed) — never an error by itself, since
    /// the PROMPT does fit.
    pub(crate) effective_max_tokens: usize,
}

/// Render `history` through `template` under `contract` and encode it,
/// dropping the oldest non-system messages until the rendered prompt itself
/// fits `max_context` (a hard error once nothing more can be dropped: the
/// graceful "context full" clamp below applies only when the prompt fits
/// and just the requested `max_tokens` does not). The dropped messages are
/// removed from `history` itself, so the conversation stays consistent with
/// what the model saw.
///
/// `prompt_rows` counts the sequence positions an encoded prompt occupies:
/// its token count for a text-only session ([`text_rows`]), and — when the
/// session carries images — the count with every `<|image_pad|>` expanded
/// to its image's rows ([`SessionImages::rows`]), so the context check sees
/// what the model will.
///
/// When the prompt fits but `max_tokens` would still carry it past
/// `max_context` after every droppable message is gone,
/// [`RenderedTurn::effective_max_tokens`] is clamped to exactly what
/// remains instead of erroring (a stderr notice is printed).
///
/// # Errors
///
/// A render/encode error, a `prompt_rows` error (placeholders that do not
/// match the session's images), or the latest message ALONE (nothing left
/// to drop) exceeding `max_context`.
pub(crate) fn render_turn(
    tok: &oxibonsai_runtime::TokenizerBridge,
    template: &ResolvedChatTemplate,
    history: &mut Vec<RenderMessage>,
    contract: &ChatContract,
    max_tokens: usize,
    max_context: usize,
    prompt_rows: &dyn Fn(&[u32]) -> anyhow::Result<usize>,
) -> anyhow::Result<RenderedTurn> {
    let mut dropped = 0usize;
    loop {
        let rendered = generate::render_prompt(template, history, contract)?;
        let tokens = tok.encode(&rendered)?;
        let prompt_len = prompt_rows(&tokens)?;
        let oldest = || {
            history
                .iter()
                .position(|m| m.role != "system")
                .filter(|&i| i + 1 < history.len())
        };

        if prompt_len > max_context {
            match oldest() {
                Some(i) => {
                    history.remove(i);
                    dropped += 1;
                    continue;
                }
                None => anyhow::bail!(
                    "sequence length {prompt_len} exceeds max context {max_context}: this \
                     message alone does not fit; shorten it or raise --ctx"
                ),
            }
        }

        if prompt_len.saturating_add(max_tokens) <= max_context {
            return Ok(RenderedTurn {
                tokens,
                started_in_think: bonsai2::prompt_opens_think_block(&rendered),
                dropped_messages: dropped,
                effective_max_tokens: max_tokens,
            });
        }
        // The prompt fits, but the full requested budget would not: drop
        // more history first (frees room for the SAME full budget); only
        // once nothing more can be dropped does the graceful clamp apply.
        match oldest() {
            Some(i) => {
                history.remove(i);
                dropped += 1;
            }
            None => {
                let clamped = max_context.saturating_sub(prompt_len);
                eprintln!(
                    "[context window full: {prompt_len} prompt token(s) + {max_tokens} \
                     requested would exceed --ctx {max_context}; generating {clamped} more \
                     token(s) instead]"
                );
                return Ok(RenderedTurn {
                    tokens,
                    started_in_think: bonsai2::prompt_opens_think_block(&rendered),
                    dropped_messages: dropped,
                    effective_max_tokens: clamped,
                });
            }
        }
    }
}

/// [`render_turn`]'s row count for a text-only session: the token count.
///
/// # Errors
///
/// Never; the signature matches [`SessionImages::rows`].
pub(crate) fn text_rows(tokens: &[u32]) -> anyhow::Result<usize> {
    Ok(tokens.len())
}

/// The images of a `chat --image` session, encoded once at start-up.
pub(crate) struct SessionImages {
    /// The encoded images, in `--image` order.
    images: Vec<EncodedImage>,
    /// Their merged grids (the splice geometry).
    grids: Vec<GridSize>,
    /// The vision marker ids the splice keys on.
    ids: VisionTokenIds,
}

impl SessionImages {
    /// Wrap already-encoded images.
    pub(crate) fn new(images: Vec<EncodedImage>, ids: VisionTokenIds) -> Self {
        let grids = images.iter().map(|image| image.grid).collect();
        Self { images, grids, ids }
    }

    /// Images in the session.
    pub(crate) fn len(&self) -> usize {
        self.images.len()
    }

    /// Sequence positions a rendered conversation occupies with the images
    /// expanded (the plain token count once the image message is gone).
    ///
    /// # Errors
    ///
    /// Placeholders that do not match the session's images.
    pub(crate) fn rows(&self, tokens: &[u32]) -> anyhow::Result<usize> {
        bonsai2::prompt_rows(tokens, &self.grids, self.ids)
    }

    /// Whether a rendered conversation still holds the image message.
    pub(crate) fn placed_in(&self, tokens: &[u32]) -> bool {
        tokens.contains(&self.ids.image_pad)
    }

    /// The prompt for one rendered conversation: multimodal while it holds
    /// the image message, text once that message was dropped.
    ///
    /// # Errors
    ///
    /// Placeholders that do not match the session's images
    /// (`[<code>] <reason>`).
    pub(crate) fn prompt(&self, tokens: Vec<u32>) -> anyhow::Result<ChatPrompt> {
        if !self.placed_in(&tokens) {
            return Ok(ChatPrompt::Text(tokens));
        }
        MultimodalPrompt::new(tokens, self.images.clone(), self.ids)
            .map(ChatPrompt::Multimodal)
            .map_err(|e| anyhow::anyhow!("[{}] {e}", e.code()))
    }
}

/// Prepare, context-check and encode every `--image` of a `chat` session
/// (`None` when the session has none).
///
/// # Errors
///
/// An image that cannot be prepared or encoded, or images whose rows alone
/// exceed `max_context`.
fn encode_session_images(
    vision: &bonsai2::VisionRequest,
    service: Option<&oxibonsai_runtime::vision_prefill::VisionService>,
    max_context: usize,
) -> anyhow::Result<Option<SessionImages>> {
    let Some(service) = service.filter(|_| !vision.images.is_empty()) else {
        return Ok(None);
    };
    let prepared = vision.prepare_images(service)?;
    let rows: usize = prepared.iter().map(|p| p.grid.n_tokens()).sum();
    if rows > max_context {
        anyhow::bail!(
            "the {} image(s) alone occupy {rows} positions, more than max context \
             {max_context}; lower --image-max-tokens or raise --ctx",
            prepared.len()
        );
    }
    let images = vision.encode_prepared(service, &prepared)?;
    eprintln!(
        "[{} image(s) ({rows} image tokens) will be attached to your first message]",
        images.len()
    );
    Ok(Some(SessionImages::new(images, service.token_ids())))
}

pub(crate) fn run(args: ChatArgs) -> anyhow::Result<()> {
    let ChatArgs {
        model,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        max_seq_len,
        tokenizer,
        tokenizer_backend,
        grammar,
        stop,
        backend,
        rope_scaling,
        enable_thinking,
        reasoning_effort,
        tools,
        show_reasoning,
        hide_reasoning,
        ptq1_transcode,
        prefill_chunk,
        vision,
        image_sources,
        allow_vocab_mismatch,
    } = args;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    vision.validate(true, &image_sources)?;
    let contract = ChatContract::from_flags(enable_thinking, reasoning_effort, tools.as_deref())?;
    let (display, _) = ReasoningDisplay::from_flags(show_reasoning, hide_reasoning);

    let source = ModelSource::open(&model, ptq1_transcode)?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(source.bytes())?;
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .unwrap_or("")
        .to_string();

    let max_seq_len = cmd_run::apply_bonsai2_load_time_guards(
        &gguf,
        &arch,
        source.weight_bytes(),
        max_seq_len,
        rope_scaling,
    )?;
    // The vision projector (design §6.2) is checked before any
    // language-model weight is bound — a wrong architecture or a projector
    // the towers refuse fails fast — and its header sizes the engine: a
    // Metal-backed engine's KV window leaves room for the Metal tower. The
    // tower itself is built once the engine exists, for the executor the
    // engine decodes on.
    let hybrid_options = vision.hybrid_load_options(&arch, prefill_chunk)?;
    vision.check_image_tokens(&bonsai2::ModelVocabulary::of_gguf(&gguf))?;
    let sampling = cmd_run::resolve_sampling(temperature, top_k, top_p, min_p, &gguf.metadata)?;

    // cli-12: same shared constructor as `run`,
    // with the same 1.0/0.0/0.0 no-hidden-penalty defaults.
    let params = build_sampling_params(
        sampling.temperature,
        sampling.top_k,
        sampling.top_p,
        repetition_penalty,
    );
    let load = EngineLoad {
        params: params.clone(),
        seed,
        max_seq_len,
        backend,
        rope_scaling,
        prefill_chunk,
        penalties: PenaltyParams::new(frequency_penalty, presence_penalty),
        min_p: sampling.min_p,
        wants_vision: vision.mmproj.is_some(),
        vision_resident_bytes: hybrid_options.vision_resident_bytes,
    };
    let mut engine = cmd_run::load_engine(&gguf, &load, source.transcoded_tensors())?;
    // An engine that cannot serve image turns is refused here with its typed
    // error, before the session's images are encoded.
    if !vision.images.is_empty() {
        vision::require_image_capable_engine(&engine)?;
    }
    // The projector, for the executor the engine decodes on. The image
    // policy is built only when a projector is loaded, so a text-only
    // session never reads the remote-image settings (a stale value in the
    // shell or `.env` cannot stop it).
    let vision_service = vision.load_service_for(
        &arch,
        &bonsai2::ModelVocabulary::of_gguf(&gguf),
        || image_sources.cli_policy(),
        &engine,
    )?;
    if vision_service.is_some() && vision.images.is_empty() {
        tracing::warn!(
            "--mmproj without --image: the projector is loaded but no image is attached"
        );
    }

    // TOK-08: vocab-aware resolution + a hard compatibility check, the
    // GGUF's own template attached, the GGUF-embedded tokenizer as the
    // fallback.
    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), &model, expected_vocab);
    let tok = cmd_run::resolve_model_tokenizer(
        tokenizer.as_deref(),
        &lookup,
        &gguf,
        expected_vocab,
        tokenizer_backend,
        allow_vocab_mismatch,
    )?;
    let Some(tok) = tok else {
        tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
        anyhow::bail!(
            "`oxibonsai chat` renders every turn through the model's chat template and needs a \
             tokenizer to do it; none was found. Pass --tokenizer <path/to/tokenizer.json>."
        );
    };
    let template = tok.resolved_chat_template();
    tracing::info!(
        template = cmd_run::chat_template_source(&template),
        template_thinks_by_default = ?bonsai2::default_enable_thinking(&template).ok(),
        enable_thinking = ?contract.enable_thinking,
        reasoning_effort = ?contract.reasoning_effort,
        tools = contract.tools_json.is_some(),
        "chat contract resolved"
    );

    // Loaded once for the whole session; a fresh `ConstrainedSampler`
    // (fresh recognizer state) is built from it every turn.
    let cached_grammar = match &grammar {
        Some(path) => Some(cmd_run::load_grammar(path)?),
        None => None,
    };
    let use_constrained_or_stop = cached_grammar.is_some() || !stop.is_empty();
    reject_penalties_with_constrained_decode(
        use_constrained_or_stop,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
    )?;
    // Every `--image`, encoded once for the whole session.
    let session_images =
        encode_session_images(&vision, vision_service.as_deref(), engine.max_context())?;
    let prompt_rows = |tokens: &[u32]| match &session_images {
        Some(images) => images.rows(tokens),
        None => text_rows(tokens),
    };
    println!("OxiBonsai Interactive Chat (type 'quit' or Ctrl-D to exit, '/reset' to clear)");
    println!("Tip: press Ctrl-C during generation to interrupt output without exiting.");
    println!("---");

    // Shared cancellation flag. The ctrlc handler sets it; the decode
    // loops check it and stop within one token step.
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        })
        .map_err(|e| anyhow::anyhow!("failed to install Ctrl-C handler: {e}"))?;
    }

    let mut history: Vec<RenderMessage> = Vec::new();
    // Whether the next user message carries the session's images (the
    // first one, and the first one after `/reset`), and whether the history
    // currently holds the message that does.
    let mut attach_images = session_images.is_some();
    let mut images_in_history = false;
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl-D)
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                interrupted.store(false, Ordering::SeqCst);
                println!();
                eprintln!("[Ctrl-C: type 'quit' or press Ctrl-D to exit]");
                continue;
            }
            Err(e) => return Err(e.into()),
        }
        let input = input.trim();
        if input.is_empty() {
            interrupted.store(false, Ordering::SeqCst);
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }
        if input == "/reset" {
            engine.reset();
            history.clear();
            attach_images = session_images.is_some();
            images_in_history = false;
            if attach_images {
                println!("[context cleared; the image(s) will be attached to your next message]");
            } else {
                println!("[context cleared]");
            }
            continue;
        }

        let images_now = match &session_images {
            Some(images) if attach_images => images.len(),
            _ => 0,
        };
        history.push(cmd_run::user_turn(input, images_now));
        let turn = match render_turn(
            &tok,
            &template,
            &mut history,
            &contract,
            max_tokens,
            engine.max_context(),
            &prompt_rows,
        ) {
            Ok(turn) => turn,
            Err(e) => {
                history.pop();
                eprintln!("[{e}]");
                continue;
            }
        };
        if turn.dropped_messages > 0 {
            eprintln!(
                "[dropped the {} oldest message(s) to fit the context window]",
                turn.dropped_messages
            );
        }
        let prompt = match &session_images {
            Some(images) => images.prompt(turn.tokens),
            None => Ok(ChatPrompt::Text(turn.tokens)),
        };
        let prompt = prompt.and_then(|prompt| {
            if images_now > 0 && prompt.image_count() == 0 {
                anyhow::bail!(
                    "[image_placeholder_count_mismatch] the chat template rendered no \
                     <|image_pad|> placeholder for the {images_now} attached image(s); this \
                     model's template does not support image content"
                );
            }
            Ok(prompt)
        });
        let prompt = match prompt {
            Ok(prompt) => prompt,
            Err(e) => {
                history.pop();
                eprintln!("[{e}]");
                continue;
            }
        };
        if images_now > 0 {
            attach_images = false;
            images_in_history = true;
        } else if images_in_history && prompt.image_count() == 0 {
            images_in_history = false;
            eprintln!(
                "[the message carrying the image(s) was dropped to fit the context window; the \
                 conversation continues without them ('/reset' re-attaches them)]"
            );
        }

        interrupted.store(false, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let is_interrupted = || interrupted.load(Ordering::SeqCst);

        let (output_count, reasoning, content) = if use_constrained_or_stop {
            let mut printer = TokenPrinter::new(Some(&tok), turn.started_in_think, display, false);
            let count = cmd_run::run_constrained_or_stopped_with(
                &mut engine,
                &prompt,
                turn.effective_max_tokens,
                cached_grammar.as_ref(),
                &stop,
                Some(&tok),
                &ConstrainedSampling {
                    params: params.clone(),
                    seed,
                    min_p: sampling.min_p,
                },
                &mut printer,
                &is_interrupted,
            )?;
            let stop_checker = StopChecker::new(stop.clone());
            let truncate = |text: &str| stop_checker.truncate_at_stop(text);
            let (reasoning, content) = printer.finish(Some(&truncate));
            (count, reasoning, content)
        } else {
            let mut printer = TokenPrinter::new(Some(&tok), turn.started_in_think, display, true);
            // The engine's own decode path applies `min_p` directly
            // (`load_engine` already called `set_min_p` above),
            // token-for-token identical to the former CLI sampler loop.
            let count = run_streaming_turn(
                &mut engine,
                &prompt,
                turn.effective_max_tokens,
                &mut printer,
                &interrupted,
            )?;
            let (reasoning, content) = printer.finish(None);
            (count, reasoning, content)
        };

        let elapsed = start.elapsed();
        println!(); // newline after streamed output

        let mut reply = RenderMessage::new("assistant", content);
        if let Some(reasoning) = reasoning {
            reply = reply.with_reasoning_content(reasoning);
        }
        history.push(reply);

        if interrupted.swap(false, Ordering::SeqCst) {
            eprintln!("[interrupted after {output_count} tokens]");
        } else {
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                output_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "[{output_count} tokens in {:.2}s, {tok_per_sec:.1} tok/s]",
                elapsed.as_secs_f64()
            );
        }
    }

    Ok(())
}

/// One turn of the engine's own streaming decode loop (no `--grammar` /
/// `--stop` / min-p): `generate_streaming_sync` (its multimodal form for a
/// prompt with images) on a worker thread, tokens printed as they arrive.
/// Breaking out on Ctrl-C drops the receiver, which stops the generation
/// thread within one token step.
fn run_streaming_turn(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt: &ChatPrompt,
    max_tokens: usize,
    printer: &mut TokenPrinter<'_>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<usize> {
    let (tx, rx) = std::sync::mpsc::channel::<u32>();
    std::thread::scope(|s| -> anyhow::Result<usize> {
        let thread_tx = tx.clone();
        let gen_handle =
            s.spawn(move || prompt.generate_streaming_sync(engine, max_tokens, &thread_tx));
        drop(tx);

        let mut count = 0usize;
        for token_id in rx {
            if interrupted.load(Ordering::SeqCst) {
                break;
            }
            count += 1;
            printer.push(token_id)?;
        }
        match gen_handle.join() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err(anyhow::anyhow!("generation thread panicked")),
        }
        Ok(count)
    })
}

#[cfg(test)]
#[path = "cmd_chat_tests.rs"]
mod tests;
