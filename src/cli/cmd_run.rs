//! `oxibonsai run` — single-shot inference on a GGUF model.
//!
//! Decode routing (every path honours `--backend`):
//!
//! * `--grammar` / `--stop` → [`run_constrained_or_stopped`], a buffered
//!   token-by-token loop sampling with the CLI's own
//!   [`oxibonsai_runtime::sampling::Sampler`] (or the grammar-constrained
//!   chain, which includes min-p).
//! * everything else (including a sampled request with `--min-p > 0`) →
//!   the engine's own `generate_streaming_sync` / `generate`
//!   ([`run_engine_generation`]), which applies `--min-p` directly
//!   (`InferenceEngine::set_min_p`, applied once at construction —
//!   [`load_engine`]) and routes a temperature-0, penalty-free
//!   request through the GPU argmax **only** when
//!   `InferenceEngine::greedy_gpu_eligible` holds — i.e. only on a
//!   fused-Metal *GPU-tier* engine. The CLI no longer has a greedy-GPU
//!   shortcut of its own: that shortcut ignored `--backend cpu` and decoded
//!   a CPU-tier engine against the GPU-resident KV cache (garbage output),
//!   and it was also the last reason for this file to gate on the `metal`
//!   Cargo feature (cli-09).

use oxibonsai_runtime::config::RenderMessage;
use oxibonsai_runtime::sampling::{PenaltyParams, SamplingParams};
use oxibonsai_runtime::vision_prefill::{ChatPrompt, RenderContentPart};

use super::args;
use super::bonsai2;
use super::generate::{self, ChatContract, ReasoningDisplay, TokenPrinter};
use super::model_desc;
use super::model_source::ModelSource;
use super::tokenizer_backend::{self, TokenizerBackendChoice};
use super::util::{
    build_sampling_params, check_tokenizer_model_compatibility, clamp_generation_budget,
    missing_tokenizer_warning, model_vocab_size, read_prompt_stdin,
    reject_penalties_with_constrained_decode, resolve_tokenizer_vocab_aware, validated,
    StopChecker, TokenizerLookup,
};
use super::vision;

/// Attach the GGUF's own chat template to `tok`:
/// every production deployment must render prompts through the SHIPPED
/// model's template, not always the hardcoded ChatML fallback.
/// `ResolvedChatTemplate::from_gguf` falls back to the built-in ChatML/Qwen3
/// template only when the file ships no `tokenizer.chat_template` at all.
pub(crate) fn attach_gguf_chat_template(
    tok: oxibonsai_runtime::TokenizerBridge,
    md: &oxibonsai_core::MetadataStore,
) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge> {
    let template = oxibonsai_runtime::config::ResolvedChatTemplate::from_gguf(md)
        .map_err(|e| anyhow::anyhow!("failed to compile the GGUF's own chat template: {e}"))?;
    log_chat_template_source(&template);
    Ok(tok.with_chat_template(template))
}

/// Where a resolved chat template came from, for the info log (the choice
/// must be visible).
pub(crate) fn chat_template_source(
    template: &oxibonsai_runtime::config::ResolvedChatTemplate,
) -> &'static str {
    match template {
        oxibonsai_runtime::config::ResolvedChatTemplate::Jinja(_) => {
            "the GGUF's own tokenizer.chat_template"
        }
        oxibonsai_runtime::config::ResolvedChatTemplate::Canned(_) => {
            "the built-in ChatML/Qwen3 fallback (the GGUF ships no tokenizer.chat_template)"
        }
    }
}

fn log_chat_template_source(template: &oxibonsai_runtime::config::ResolvedChatTemplate) {
    tracing::info!(
        template = chat_template_source(template),
        "resolved chat template"
    );
}

/// The context guard (design §5.6 / Appendix A.3) plus the `--rope-scaling`
/// pre-flight check, run once the GGUF is parsed and before any weights are
/// touched. Returns the resolved `max_seq_len` (the explicit value, or the
/// per-architecture default: 8192 for `qwen35`, 4096 otherwise).
///
/// `weight_bytes` is what the weights occupy (the file, or a
/// `--ptq1-transcode` image). Shared by `run`, `chat` and `serve`.
///
/// # Errors
///
/// A refused context (naming both limits and the GiB), or `--rope-scaling
/// on` on a file that declares no scaling.
pub(crate) fn apply_bonsai2_load_time_guards(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    arch: &str,
    weight_bytes: u64,
    max_seq_len: Option<usize>,
    rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
) -> anyhow::Result<usize> {
    let max_seq_len = validated(args::validate_max_seq_len(
        max_seq_len.unwrap_or_else(|| bonsai2::default_max_seq_len(arch)),
    ))?;
    bonsai2::validate_context_for_model(gguf, arch, weight_bytes, max_seq_len)?;

    // `--rope-scaling` pre-flight: fail before any weight is touched, with a
    // clean message. The engine constructor applies (and logs) the same
    // resolution when it builds the model.
    let mode = oxibonsai_core::config::RopeScalingOverride::from(rope_scaling);
    let declared = oxibonsai_core::config::RopeScaling::from_metadata(&gguf.metadata, arch)
        .map_err(|e| anyhow::anyhow!("failed to read this model's RoPE scaling metadata: {e}"))?;
    mode.apply(declared, arch)
        .map_err(|e| anyhow::anyhow!("--rope-scaling {mode}: {e}"))?;
    Ok(max_seq_len)
}

/// The sampling values a request actually runs with (RT-17).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedSampling {
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
}

/// RT-17's precedence for every sampling default: an explicit value (a CLI
/// flag, or a `--config` value — merged and validated in `mod.rs`) wins;
/// otherwise the model's own `general.sampling.*` declaration; otherwise the
/// shared pre-RT-17 literal (`util::DEFAULT_*`: 0.7 / 40 / 0.9 / 0.0). The
/// model's value is range checked like a flag would be.
///
/// # Errors
///
/// A GGUF-declared value outside the flag's accepted range.
pub(crate) fn resolve_sampling(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    md: &oxibonsai_core::MetadataStore,
) -> anyhow::Result<ResolvedSampling> {
    use super::util::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P};
    use oxibonsai_runtime::sampling::{
        resolve_sampling_default_f32, resolve_sampling_default_usize, GgufSamplingDefaults,
    };
    let declared = GgufSamplingDefaults::from_metadata(md);
    Ok(ResolvedSampling {
        temperature: validated(args::validate_temperature(resolve_sampling_default_f32(
            temperature,
            declared.temperature,
            DEFAULT_TEMPERATURE,
        )))?,
        top_k: resolve_sampling_default_usize(top_k, declared.top_k, DEFAULT_TOP_K),
        top_p: validated(args::validate_top_p(resolve_sampling_default_f32(
            top_p,
            declared.top_p,
            DEFAULT_TOP_P,
        )))?,
        min_p: validated(args::validate_min_p(resolve_sampling_default_f32(
            min_p,
            declared.min_p,
            DEFAULT_MIN_P,
        )))?,
    })
}

/// Everything `run`/`chat` need to build an engine from a parsed GGUF.
pub(crate) struct EngineLoad {
    pub(crate) params: SamplingParams,
    pub(crate) seed: u64,
    pub(crate) max_seq_len: usize,
    pub(crate) backend: oxibonsai_runtime::engine_seam::Backend,
    pub(crate) rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
    pub(crate) prefill_chunk: Option<usize>,
    pub(crate) penalties: PenaltyParams,
    /// The resolved `--min-p`: the engine's own decode path applies it
    /// directly (`InferenceEngine::set_min_p`), so `run`/`chat` need no
    /// separate CLI-owned sampler loop for a sampled request with
    /// `min_p > 0`.
    pub(crate) min_p: f32,
    /// A projector (`--mmproj`) was loaded: under `--backend auto` the engine
    /// must be one that can prefill image rows ([`vision::plan_vision_backend`]).
    pub(crate) wants_vision: bool,
    /// Bytes the projector's Metal tower keeps resident
    /// ([`bonsai2::VisionRequest::hybrid_load_options`]; `0` without
    /// `--mmproj`): a Metal-backed hybrid engine's KV window leaves room for
    /// them.
    pub(crate) vision_resident_bytes: u64,
}

impl EngineLoad {
    /// The options a hybrid engine is built with: the vision tower's
    /// resident bytes and `--prefill-chunk`, so a Metal runner's window and
    /// call size are planned for both before it allocates anything. A dense
    /// engine ignores them.
    pub(crate) fn hybrid_options(&self) -> oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions {
        oxibonsai_runtime::engine_hybrid_gpu::HybridLoadOptions {
            vision_resident_bytes: self.vision_resident_bytes,
            prefill_chunk: self.prefill_chunk,
        }
    }
}

/// Construct the engine for `backend` — honouring `--rope-scaling`, building
/// a hybrid engine under [`EngineLoad::hybrid_options`], applying
/// `--prefill-chunk`, the penalties and `--min-p` — without reporting on it.
fn construct_engine<'a>(
    gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'a>,
    load: &EngineLoad,
    backend: oxibonsai_runtime::engine_seam::Backend,
) -> anyhow::Result<oxibonsai_runtime::InferenceEngine<'a>> {
    let mut engine = {
        let _hybrid =
            oxibonsai_runtime::engine_hybrid_gpu::HybridLoadScope::enter(load.hybrid_options());
        oxibonsai_runtime::InferenceEngine::from_gguf_with_backend_and_rope(
            gguf,
            load.params.clone(),
            load.seed,
            load.max_seq_len,
            backend,
            load.rope_scaling.into(),
        )?
    };
    bonsai2::apply_prefill_chunk(&mut engine, load.prefill_chunk)?;
    engine.set_penalties(load.penalties);
    engine.set_min_p(load.min_p);
    Ok(engine)
}

/// Build the engine (honouring `--backend` and `--rope-scaling`), apply
/// `--prefill-chunk`, the penalties and `--min-p`, and print the
/// resolved-engine summary line (cli-16: from the engine's own accessors).
///
/// With a projector loaded ([`EngineLoad::wants_vision`]) and `--backend
/// auto`, an engine whose executor cannot prefill image rows
/// (`InferenceEngine::prefills_images` is `false`) is dropped and rebuilt on
/// the CPU model ([`vision::plan_vision_backend`]), with one `INFO` line
/// saying why; an explicit backend is never overridden. Both hybrid
/// executors prefill image rows, so `auto` keeps the engine it resolved to.
///
/// # Errors
///
/// Engine construction errors (including the typed backend refusals).
pub(crate) fn load_engine<'a>(
    gguf: &'a oxibonsai_core::gguf::reader::GgufFile<'a>,
    load: &EngineLoad,
    transcoded_tensors: usize,
) -> anyhow::Result<oxibonsai_runtime::InferenceEngine<'a>> {
    let mut engine = construct_engine(gguf, load, load.backend)?;
    // Only a hybrid engine's executor is a question here: a dense engine
    // refuses image turns for its model kind on any backend.
    let wants_vision = load.wants_vision && engine.is_hybrid();
    if vision::plan_vision_backend(load.backend, wants_vision, engine.prefills_images())
        == vision::VisionBackendPlan::RebuildOnCpu
    {
        let executor = vision::executor_name(&engine);
        tracing::info!(executor = %executor, "{}", vision::rebuild_reason(&executor));
        // The first engine (and a Metal runner's device buffers) goes before
        // the second is built: a 27B is never resident twice.
        drop(engine);
        engine = construct_engine(gguf, load, oxibonsai_runtime::engine_seam::Backend::Cpu)?;
    }
    let mut summary = model_desc::engine_summary(&engine);
    if transcoded_tensors > 0 {
        summary.push_str(&format!(
            " | --ptq1-transcode: {transcoded_tensors} PTQ1_0 tensors re-encoded to PQ2_0"
        ));
    }
    eprintln!("{summary}");
    Ok(engine)
}

/// Resolved arguments for `oxibonsai run`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04).
///
/// A struct rather than many positional parameters: named fields let the
/// compiler catch a mismatch instead of silently swapping two `f32`s.
pub(crate) struct RunArgs {
    pub(crate) model: Option<String>,
    pub(crate) prompt: String,
    pub(crate) max_tokens: usize,
    /// RT-17: `None` = no explicit CLI/TOML value — resolved against the
    /// GGUF's own `general.sampling.*` default, then the literal, once the
    /// model is parsed ([`resolve_sampling`]).
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
    /// RT-23, RT-17-style precedence (see `temperature`).
    pub(crate) min_p: Option<f32>,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) seed: u64,
    /// The context guard's default (design §5.6 / Appendix A.3): `None` =
    /// the per-architecture default, applied (and guarded) once the model
    /// is parsed.
    pub(crate) max_seq_len: Option<usize>,
    pub(crate) tokenizer: Option<String>,
    pub(crate) tokenizer_backend: TokenizerBackendChoice,
    /// Render the prompt through the model's chat template.
    pub(crate) chat: bool,
    pub(crate) grammar: Option<String>,
    pub(crate) stop: Vec<String>,
    pub(crate) backend: oxibonsai_runtime::engine_seam::Backend,
    pub(crate) rope_scaling: oxibonsai_runtime::config::RopeScalingMode,
    /// cli-11 chat contract (all require `--chat`).
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) tools: Option<String>,
    pub(crate) show_reasoning: bool,
    pub(crate) hide_reasoning: bool,
    pub(crate) ptq1_transcode: bool,
    pub(crate) prefill_chunk: Option<usize>,
    /// §5.7 vision flags: `--mmproj` loads the projector, every `--image`
    /// is encoded into the (chat-rendered) prompt.
    pub(crate) vision: bonsai2::VisionRequest,
    /// `--allow-image-url-fetch`, `--image-url-timeout-ms` and
    /// `--image-url-allow-host` (and their `OXI_*` fallbacks): which image
    /// references resolve, and how a remote one is fetched.
    pub(crate) image_sources: bonsai2::ImageSourceFlags,
    pub(crate) allow_vocab_mismatch: bool,
    pub(crate) no_stream: bool,
}

/// Refuse chat-contract flags on a raw (non-`--chat`) run: they only mean
/// something when the prompt is rendered through the chat template, and a
/// flag that silently does nothing is never acceptable.
pub(crate) fn require_chat_for_contract_flags(
    chat: bool,
    contract: &ChatContract,
    display_explicit: bool,
) -> anyhow::Result<()> {
    if chat || (contract.is_empty() && !display_explicit) {
        return Ok(());
    }
    let mut flags = Vec::new();
    match contract.enable_thinking {
        Some(true) => flags.push("--think"),
        Some(false) => flags.push("--no-think"),
        None => {}
    }
    if contract.reasoning_effort.is_some() {
        flags.push("--reasoning-effort");
    }
    if contract.tools_json.is_some() {
        flags.push("--tools");
    }
    if display_explicit {
        flags.push("--show-reasoning/--hide-reasoning");
    }
    anyhow::bail!(
        "{} only apply when the prompt is rendered through the model's chat template: add \
         --chat (or use `oxibonsai chat`)",
        flags.join(", ")
    );
}

pub(crate) fn run(args: RunArgs) -> anyhow::Result<()> {
    let RunArgs {
        model,
        prompt,
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
        chat,
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
        no_stream,
    } = args;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    vision.validate(true, &image_sources)?;
    // An image is a part of a chat turn (the template places its
    // placeholder), so `--image` implies the chat contract.
    let chat = chat || !vision.images.is_empty();
    let contract = ChatContract::from_flags(enable_thinking, reasoning_effort, tools.as_deref())?;
    let (display, display_explicit) = ReasoningDisplay::from_flags(show_reasoning, hide_reasoning);
    require_chat_for_contract_flags(chat, &contract, display_explicit)?;

    let prompt_text = if prompt == "-" {
        read_prompt_stdin()?
    } else {
        prompt
    };

    tracing::info!(model = %model, max_tokens, "starting inference");

    let source = ModelSource::open(&model, ptq1_transcode)?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(source.bytes())?;
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .unwrap_or("")
        .to_string();

    let max_seq_len = apply_bonsai2_load_time_guards(
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
    let sampling = resolve_sampling(temperature, top_k, top_p, min_p, &gguf.metadata)?;
    tracing::info!(
        temperature = sampling.temperature,
        top_k = sampling.top_k,
        top_p = sampling.top_p,
        min_p = sampling.min_p,
        "resolved inference"
    );

    // The shared constructor: `--temperature 0`
    // means exactly argmax on every backend — no hidden penalty.
    let params = build_sampling_params(
        sampling.temperature,
        sampling.top_k,
        sampling.top_p,
        repetition_penalty,
    );
    let penalties = PenaltyParams::new(frequency_penalty, presence_penalty);
    let mut engine = load_engine(
        &gguf,
        &EngineLoad {
            params: params.clone(),
            seed,
            max_seq_len,
            backend,
            rope_scaling,
            prefill_chunk,
            penalties,
            min_p: sampling.min_p,
            wants_vision: vision.mmproj.is_some(),
            vision_resident_bytes: hybrid_options.vision_resident_bytes,
        },
        source.transcoded_tensors(),
    )?;
    // An engine that cannot serve image turns refuses the image turn with
    // its typed error — raised here, before any image is encoded, instead of
    // after the tower has run.
    if !vision.images.is_empty() {
        vision::require_image_capable_engine(&engine)?;
    }
    // The projector, for the executor the engine decodes on: the Metal
    // tower beside the Metal hybrid runner, the CPU tower beside the CPU
    // model. The image policy is built only when a projector is loaded, so a
    // text-only run never reads the remote-image settings (a stale value in
    // the shell or `.env` cannot stop it).
    let vision_service = vision.load_service_for(
        &arch,
        &bonsai2::ModelVocabulary::of_gguf(&gguf),
        || image_sources.cli_policy(),
        &engine,
    )?;
    if vision_service.is_some() && vision.images.is_empty() {
        tracing::warn!(
            "--mmproj without --image: the projector is loaded but the prompt has no image"
        );
    }

    // Tokenizer (TOK-08): vocab-aware resolution, a hard
    // compatibility check, the GGUF's own template attached, and the
    // GGUF-embedded tokenizer as the fallback for a Bonsai 2 file.
    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), &model, expected_vocab);
    let tok_bridge = resolve_model_tokenizer(
        tokenizer.as_deref(),
        &lookup,
        &gguf,
        expected_vocab,
        tokenizer_backend,
        allow_vocab_mismatch,
    )?;

    let (prompt_tokens, started_in_think) = match (&tok_bridge, chat) {
        (Some(tok), true) => {
            let template = tok.resolved_chat_template();
            if let Ok(default_thinks) = bonsai2::default_enable_thinking(&template) {
                tracing::info!(
                    template_thinks_by_default = default_thinks,
                    enable_thinking = ?contract.enable_thinking,
                    "chat contract resolved"
                );
            }
            let rendered = generate::render_prompt(
                &template,
                &[user_turn(prompt_text.as_str(), vision.images.len())],
                &contract,
            )?;
            (
                tok.encode(&rendered)?,
                bonsai2::prompt_opens_think_block(&rendered),
            )
        }
        (None, true) => anyhow::bail!(
            "--chat needs a tokenizer to render and encode the chat template, and none was \
             found; pass --tokenizer <path/to/tokenizer.json>"
        ),
        (Some(tok), false) => (tok.encode(&prompt_text)?, false),
        (None, false) => {
            tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
            // cli-07: no hardcoded token id — the model's own declared BOS
            // id, or nothing (refused below).
            let bos = gguf
                .metadata
                .get_u32(oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_BOS_TOKEN_ID)
                .ok();
            (bos.map(|id| vec![id]).unwrap_or_default(), false)
        }
    };
    if prompt_tokens.is_empty() {
        anyhow::bail!(
            "cannot start generation: no tokenizer was found and the model declares no BOS \
             token id, so there is no way to honestly tokenize the prompt. Pass --tokenizer \
             <path/to/tokenizer.json>."
        );
    }
    if grammar.is_some() && tok_bridge.is_none() {
        anyhow::bail!("--grammar requires a tokenizer; none was found (see the warning above)");
    }

    let use_constrained_or_stop = grammar.is_some() || !stop.is_empty();
    reject_penalties_with_constrained_decode(
        use_constrained_or_stop,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
    )?;

    // SV-11 / design §6.2: each `--image` encoded and spliced in place of
    // its `<|image_pad|>`; a text-only prompt stays its token ids.
    let prompt = multimodal_prompt(
        prompt_tokens,
        &vision,
        vision_service.as_deref(),
        engine.max_context(),
        max_tokens,
    )?;
    tracing::info!(
        prompt_tokens = prompt.len(),
        images = prompt.image_count(),
        "prefilling"
    );
    let start = std::time::Instant::now();
    let prompt_len = prompt.len();
    // Clamp the budget to what the
    // context window has left, on every decode path alike, instead of
    // crashing once decode reaches `--ctx`. A prompt that alone overflows
    // stays a hard error (both numbers named).
    let max_tokens = clamp_generation_budget(prompt_len, max_tokens, engine.max_context())?;

    let output_count = if use_constrained_or_stop {
        let mut printer = TokenPrinter::new(tok_bridge.as_ref(), started_in_think, display, false);
        let count = run_constrained_or_stopped(
            &mut engine,
            &prompt,
            max_tokens,
            grammar.as_deref(),
            &stop,
            tok_bridge.as_ref(),
            &ConstrainedSampling {
                params: params.clone(),
                seed,
                min_p: sampling.min_p,
            },
            &mut printer,
        )?;
        let stop_checker = StopChecker::new(stop.clone());
        let truncate = |text: &str| stop_checker.truncate_at_stop(text);
        printer.finish(Some(&truncate));
        count
    } else {
        let mut printer = TokenPrinter::new(tok_bridge.as_ref(), started_in_think, display, true);
        // The engine's own decode path applies `min_p` directly
        // (`load_engine` already called `set_min_p` above) and is
        // token-for-token identical to the CLI's former sampler loop
        // (`min_p_engine_path_matches_the_cli_loop`), so every sampled
        // request routes through it uniformly now.
        let count =
            run_engine_generation(&mut engine, &prompt, max_tokens, no_stream, &mut printer)?;
        printer.finish(None);
        count
    };

    let elapsed = start.elapsed();
    let total_tokens = prompt_len + output_count;
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        output_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!();
    eprintln!(
        "---\n{} prompt + {} generated = {} total tokens in {:.2}s ({:.1} tok/s)",
        prompt_len,
        output_count,
        total_tokens,
        elapsed.as_secs_f64(),
        tok_per_sec
    );

    // GPU profiling summary when OXIBONSAI_PROFILE_GPU=1 was set. The one
    // remaining `feature = "metal"` gate in this file: the kernels crate
    // only defines this function in a Metal build.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let model_size = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
        oxibonsai_kernels::print_gpu_profile_summary(model_size);
    }

    Ok(())
}

/// The user turn of a `run` prompt: the prompt text alone, or — with
/// `images` images — the images as content parts ahead of the text (the
/// order the reference's examples and the golden requests use).
pub(crate) fn user_turn(prompt_text: &str, images: usize) -> RenderMessage {
    if images == 0 {
        return RenderMessage::new("user", prompt_text);
    }
    let mut parts: Vec<RenderContentPart> = (0..images).map(|_| RenderContentPart::Image).collect();
    parts.push(RenderContentPart::Text(prompt_text.to_string()));
    RenderMessage::with_parts("user", parts)
}

/// The prompt to generate from: `tokens` as they are when no image was
/// given, else every `--image` encoded by `service` and spliced in place of
/// its `<|image_pad|>` (design §6.2).
///
/// The images are prepared (resolved, decoded, resized) first and the
/// expanded prompt checked against `max_context` — with the room it leaves
/// for `max_tokens` new tokens, [`vision::check_generation_room`], the same
/// check a text prompt gets — BEFORE the tower encodes anything, so an image
/// prompt that cannot be answered costs no encode.
///
/// # Errors
///
/// An image that cannot be prepared or encoded, a rendered prompt whose
/// placeholders do not match the images (`[<code>] ...`), or an expanded
/// prompt that leaves nothing to generate into.
pub(crate) fn multimodal_prompt(
    tokens: Vec<u32>,
    vision: &bonsai2::VisionRequest,
    service: Option<&oxibonsai_runtime::vision_prefill::VisionService>,
    max_context: usize,
    max_tokens: usize,
) -> anyhow::Result<ChatPrompt> {
    let Some(service) = service.filter(|_| !vision.images.is_empty()) else {
        vision::check_generation_room(tokens.len(), max_tokens, max_context, 0)?;
        return Ok(ChatPrompt::Text(tokens));
    };
    let prepared = vision.prepare_images(service)?;
    let grids: Vec<oxibonsai_model::vision::GridSize> = prepared.iter().map(|p| p.grid).collect();
    // Strict: every image needs its placeholder (a template that dropped
    // the image parts is refused here, before any encode).
    let rows = oxibonsai_model::vision::plan_splice(&tokens, &grids, service.token_ids())
        .map_err(|e| anyhow::anyhow!("[{}] {e}", e.code()))?
        .total_rows();
    vision::check_generation_room(rows, max_tokens, max_context, prepared.len())?;
    let images = vision.encode_prepared(service, &prepared)?;
    oxibonsai_runtime::vision_prefill::MultimodalPrompt::new(tokens, images, service.token_ids())
        .map(ChatPrompt::Multimodal)
        .map_err(|e| anyhow::anyhow!("[{}] {e}", e.code()))
}

/// The engine's own decode loop — `generate_streaming_sync` on a worker
/// thread (tokens printed as they arrive), or `generate` with `--no-stream`
/// (the multimodal forms for a prompt with images). Either routes a greedy,
/// penalty-free text request through the GPU argmax only when the engine
/// itself says it is eligible (a fused-Metal GPU-tier engine; never under
/// `--backend cpu`, never for a hybrid model).
pub(crate) fn run_engine_generation(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt: &ChatPrompt,
    max_tokens: usize,
    no_stream: bool,
    printer: &mut TokenPrinter<'_>,
) -> anyhow::Result<usize> {
    if engine.greedy_gpu_eligible(false) {
        tracing::info!("greedy request on a fused-Metal GPU-tier engine: GPU argmax decode");
    }
    if no_stream {
        let tokens = prompt.generate(engine, max_tokens)?;
        for &token in &tokens {
            printer.push(token)?;
        }
        return Ok(tokens.len());
    }
    let (tx, rx) = std::sync::mpsc::channel::<u32>();
    std::thread::scope(|s| -> anyhow::Result<usize> {
        let thread_tx = tx.clone();
        let gen_handle =
            s.spawn(move || prompt.generate_streaming_sync(engine, max_tokens, &thread_tx));
        drop(tx);
        let mut count = 0usize;
        for token_id in rx {
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

/// The tokenizer a model-loading subcommand uses (TOK-08).
///
/// An explicit `--tokenizer`, or an auto-detected `tokenizer.json` that
/// passes [`check_tokenizer_model_compatibility`], is used exactly as before.
/// When no on-disk candidate exists, or an *auto-detected* one does not fit
/// the model, a tokenizer embedded in the GGUF itself (`tokenizer.ggml.*`)
/// whose vocabulary matches the model is used instead: every Bonsai 2
/// `qwen35` file carries its `pre = qwen35` tokenizer, while the repository's
/// `models/tokenizer.json` is the legacy Qwen3 one, so without this fallback a
/// Bonsai 2 run either failed the compatibility check or mis-tokenized. An
/// explicit `--tokenizer` that does not fit still fails loudly.
///
/// Returns `Ok(None)` when there is no usable tokenizer at all (the caller's
/// existing "no tokenizer" handling applies).
pub(crate) fn resolve_model_tokenizer(
    explicit: Option<&str>,
    lookup: &TokenizerLookup,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
    backend: TokenizerBackendChoice,
    allow_vocab_mismatch: bool,
) -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
    resolve_model_tokenizer_with(
        explicit,
        lookup,
        gguf,
        expected_vocab,
        allow_vocab_mismatch,
        |path| tokenizer_backend::load_tokenizer_bridge(path, backend),
    )
}

/// [`resolve_model_tokenizer`] with the caller's own loader for the on-disk
/// candidate.
pub(crate) fn resolve_model_tokenizer_with(
    explicit: Option<&str>,
    lookup: &TokenizerLookup,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
    allow_vocab_mismatch: bool,
    load: impl FnOnce(&str) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge>,
) -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
    resolve_model_tokenizer_with_source(
        explicit,
        lookup,
        gguf,
        expected_vocab,
        allow_vocab_mismatch,
        load,
    )
    .map(|resolved| resolved.map(|(tok, _source)| tok))
}

/// Where [`resolve_model_tokenizer_with_source`] found its tokenizer: enough
/// to rebuild a FURTHER instance of the very same resolved source (`serve`'s
/// embedder/RAG consumers need their own `TokenizerBridge`, since it is not
/// `Clone`) without repeating the resolution's own logging or the TOK-08
/// compatibility check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TokenizerSource {
    /// An on-disk tokenizer file (already validated) at this path.
    File(String),
    /// The vocabulary + chat template embedded in the GGUF itself.
    GgufEmbedded,
}

/// [`resolve_model_tokenizer_with`], additionally reporting which source
/// won.
pub(crate) fn resolve_model_tokenizer_with_source(
    explicit: Option<&str>,
    lookup: &TokenizerLookup,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
    allow_vocab_mismatch: bool,
    load: impl FnOnce(&str) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge>,
) -> anyhow::Result<Option<(oxibonsai_runtime::TokenizerBridge, TokenizerSource)>> {
    let Some(path) = &lookup.found else {
        return Ok(gguf_embedded_tokenizer(gguf, expected_vocab)
            .map(|tok| (tok, TokenizerSource::GgufEmbedded)));
    };
    let tok = load(path)?;
    // TOK-08: check compatibility BEFORE attaching (and logging) a
    // chat template. Attaching first and only then discovering the
    // candidate is a vocabulary mismatch used to log "resolved chat
    // template" for a template nothing ends up using, then log it AGAIN
    // for the GGUF-embedded fallback that actually wins -- doubling the
    // log line on every mismatch (the common case for a Bonsai 2 file next
    // to a legacy `tokenizer.json`). Checking first means exactly one
    // candidate is ever template-attached, so it logs exactly once.
    match check_tokenizer_model_compatibility(&tok, path, gguf, allow_vocab_mismatch) {
        Ok(()) => {
            // The GGUF's own template always wins
            // when present (falls back to ChatML/Qwen3 when the file ships
            // none).
            let tok = attach_gguf_chat_template(tok, &gguf.metadata)?;
            Ok(Some((tok, TokenizerSource::File(path.clone()))))
        }
        Err(mismatch) if explicit.is_none() => {
            match gguf_embedded_tokenizer_quiet(gguf, expected_vocab) {
                Some(embedded) => {
                    // Exactly one "resolved chat template" line for the
                    // WINNING source (the embedded fallback), plus one
                    // tokenizer-source line naming why the auto-detected
                    // candidate was skipped -- never the mismatched
                    // candidate's own (about-to-be-discarded) template too.
                    log_chat_template_source(&embedded.resolved_chat_template());
                    tracing::info!(
                        skipped = %path,
                        reason = %mismatch,
                        "the auto-detected tokenizer does not fit this model; using the tokenizer \
                         embedded in the GGUF instead"
                    );
                    Ok(Some((embedded, TokenizerSource::GgufEmbedded)))
                }
                None => Err(mismatch),
            }
        }
        Err(mismatch) => Err(mismatch),
    }
}

/// The tokenizer embedded in `gguf`'s `tokenizer.ggml.*` metadata, when the
/// file carries one whose vocabulary equals the model's (`expected_vocab`),
/// with its chat template attached.
///
/// Returns `None` for "there is nothing usable embedded" — no
/// `expected_vocab`, no `tokenizer.ggml.tokens` at all (the legacy
/// `Ternary-Bonsai-{1.7B,8B}.gguf` files carry neither an embedded
/// vocabulary nor a chat template, so this path correctly falls through for
/// them), a vocabulary that does not match, OR a template this engine's
/// Jinja subset cannot compile (logged at `warn`: every real template
/// compiles today, so that is far more likely a regression than an expected
/// case).
pub(crate) fn gguf_embedded_tokenizer(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
) -> Option<oxibonsai_runtime::TokenizerBridge> {
    let expected = expected_vocab?;
    match gguf_embedded_tokenizer_with_template(gguf) {
        Ok(tok) if tok.vocab_size() == expected => {
            tracing::info!(vocab = expected, "using the tokenizer embedded in the GGUF");
            Some(tok)
        }
        Ok(tok) => {
            tracing::debug!(
                embedded_vocab = tok.vocab_size(),
                model_vocab = expected,
                "the GGUF-embedded tokenizer does not match the model's vocabulary; not used"
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "no usable GGUF-embedded tokenizer (no embedded vocabulary, or its chat \
                 template failed to compile -- check the error above for which)"
            );
            None
        }
    }
}

/// [`gguf_embedded_tokenizer`] without any logging of its own: used only as
/// the fallback after an auto-detected on-disk candidate failed the
/// compatibility check, where the caller already logs ONE combined message
/// naming both the skipped candidate and the resolved template —
/// this avoids that single resolution logging "using the tokenizer embedded
/// in the GGUF" (and "resolved chat template") a second, redundant time.
pub(crate) fn gguf_embedded_tokenizer_quiet(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
) -> Option<oxibonsai_runtime::TokenizerBridge> {
    let expected = expected_vocab?;
    let tok = oxibonsai_runtime::TokenizerBridge::native_from_gguf_metadata(&gguf.metadata).ok()?;
    (tok.vocab_size() == expected).then_some(tok)
}

/// Build the GGUF-embedded tokenizer WITH its chat template attached:
/// vocabulary from `tokenizer.ggml.*`, template
/// from `tokenizer.chat_template` (a compile failure is an error, per
/// `ResolvedChatTemplate::from_gguf`'s own contract).
pub(crate) fn gguf_embedded_tokenizer_with_template(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge> {
    let tok = oxibonsai_runtime::TokenizerBridge::native_from_gguf_metadata(&gguf.metadata)?;
    log_chat_template_source(&tok.resolved_chat_template());
    Ok(tok)
}

/// Sampling configuration of the buffered constrained/stop loop.
pub(crate) struct ConstrainedSampling {
    pub(crate) params: SamplingParams,
    pub(crate) seed: u64,
    pub(crate) min_p: f32,
}

/// Grammar-constrained and/or stop-sequence-aware decode loop (cli-17).
///
/// A single, portable, token-by-token loop rather than threading
/// `--grammar`/`--stop` through the streaming fast paths: both features
/// change the sampling procedure itself (masked logits for a grammar; early
/// stop on a matched sequence). Output is buffered in `printer` and printed
/// once generation ends, which sidesteps the stop-sequence chunk-boundary
/// leak entirely (`StopChecker` always sees the whole accumulated text).
///
/// Tokens are drawn by the constrained sampler's own chain
/// (temperature/top-k/min-p/top-p) under `--grammar`, else by a fresh
/// [`oxibonsai_runtime::sampling::Sampler`] seeded exactly like the
/// engine's (same params, same seed, plus `min_p`) — never the engine's
/// history-free `sample`, which cannot apply min-p. Penalties are refused
/// on this path by the caller
/// ([`reject_penalties_with_constrained_decode`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_constrained_or_stopped(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt: &ChatPrompt,
    max_tokens: usize,
    grammar_path: Option<&str>,
    stop: &[String],
    tok_bridge: Option<&oxibonsai_runtime::TokenizerBridge>,
    sampling: &ConstrainedSampling,
    printer: &mut TokenPrinter<'_>,
) -> anyhow::Result<usize> {
    let grammar = grammar_path.map(load_grammar).transpose()?;
    run_constrained_or_stopped_with(
        engine,
        prompt,
        max_tokens,
        grammar.as_ref(),
        stop,
        tok_bridge,
        sampling,
        printer,
        &|| false,
    )
}

/// [`run_constrained_or_stopped`] with an already-loaded grammar and a
/// cancellation probe (`chat`'s Ctrl-C).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_constrained_or_stopped_with(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt: &ChatPrompt,
    max_tokens: usize,
    grammar: Option<&oxibonsai_runtime::Grammar>,
    stop: &[String],
    tok_bridge: Option<&oxibonsai_runtime::TokenizerBridge>,
    sampling: &ConstrainedSampling,
    printer: &mut TokenPrinter<'_>,
    interrupted: &dyn Fn() -> bool,
) -> anyhow::Result<usize> {
    engine.reset();
    let prompt_len = prompt.len();
    let mut logits = prompt.prefill(engine)?;

    let mut constrained = grammar.map(|grammar| {
        build_constrained_sampler_from_grammar(
            grammar.clone(),
            tok_bridge,
            tok_bridge.map(|t| t.vocab_size()).unwrap_or(logits.len()),
            sampling.seed,
            sampling.params.temperature,
            sampling.params.top_k,
            sampling.params.top_p,
            sampling.min_p,
        )
    });
    let mut plain = generate::cli_sampler(
        sampling.params.clone(),
        sampling.seed,
        PenaltyParams::default(),
        sampling.min_p,
    );

    let stop_checker = StopChecker::new(stop.to_vec());
    let mut generated = 0usize;
    let mut pos = prompt_len;
    while generated < max_tokens {
        if interrupted() || logits.is_empty() {
            break;
        }
        let token = match constrained.as_mut() {
            Some(cs) => cs.sample(&mut logits),
            None => plain.sample(&logits)?,
        };
        if engine.is_eos(token) {
            break;
        }
        generated += 1;
        printer.push(token)?;
        if let Some(cs) = constrained.as_ref() {
            if cs.is_complete() {
                break;
            }
        }
        if !stop_checker.is_empty() && stop_checker.check(&printer.full_text()) {
            break;
        }
        logits = engine.decode_step(token, pos)?;
        pos += 1;
    }
    Ok(generated)
}

/// Load a grammar file: `.gbnf` is parsed as GBNF, any other extension is
/// compiled as a JSON Schema. Shared with `chat`, which loads it once per
/// session but builds a fresh constrained sampler (fresh recognizer state)
/// every turn.
pub(crate) fn load_grammar(path: &str) -> anyhow::Result<oxibonsai_runtime::Grammar> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read grammar file '{path}': {e}"))?;
    if path.ends_with(".gbnf") {
        oxibonsai_runtime::parse_gbnf(&content)
            .map_err(|e| anyhow::anyhow!("failed to parse GBNF grammar '{path}': {e}"))
    } else {
        oxibonsai_runtime::compile_json_schema_str(&content)
            .map_err(|e| anyhow::anyhow!("failed to compile JSON Schema grammar '{path}': {e}"))
    }
}

/// Build a [`oxibonsai_runtime::ConstrainedSampler`] from an already-loaded
/// [`oxibonsai_runtime::Grammar`]: temperature → top-k → min-p → top-p
/// (RT-23's llama.cpp/vLLM order), or greedy at temperature 0.
///
/// `GrammarConstraint::new` eagerly decodes every id in `0..vocab_size` once
/// at construction; a per-session cache of that table would need
/// `GrammarConstraint` to expose a recognizer reset, which it does not.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_constrained_sampler_from_grammar(
    grammar: oxibonsai_runtime::Grammar,
    tok: Option<&oxibonsai_runtime::TokenizerBridge>,
    vocab_size: usize,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
) -> oxibonsai_runtime::ConstrainedSampler {
    // An owned lookup table satisfies the constraint's `Send + Sync +
    // 'static` bound and avoids re-decoding on every `allowed_tokens` call.
    let id_to_bytes: Vec<Vec<u8>> = (0..vocab_size as u32)
        .map(|id| match tok {
            Some(t) => t
                .inner()
                .id_to_token(id)
                .map(String::into_bytes)
                .unwrap_or_default(),
            None => Vec::new(),
        })
        .collect();
    let decode_fn =
        move |id: u32| -> Vec<u8> { id_to_bytes.get(id as usize).cloned().unwrap_or_default() };
    let constraint = oxibonsai_runtime::GrammarConstraint::new(grammar, decode_fn, vocab_size);

    let mut chain = oxibonsai_runtime::SamplerChain::new(seed);
    chain = if temperature <= 0.0 {
        chain.add(oxibonsai_runtime::SamplerStep::Greedy)
    } else {
        chain = chain.add(oxibonsai_runtime::SamplerStep::Temperature(temperature));
        if top_k > 0 {
            chain = chain.add(oxibonsai_runtime::SamplerStep::TopK(top_k));
        }
        if min_p > 0.0 {
            chain = chain.add(oxibonsai_runtime::SamplerStep::MinP(min_p));
        }
        if top_p < 1.0 {
            chain = chain.add(oxibonsai_runtime::SamplerStep::TopP(top_p));
        }
        chain
    };

    oxibonsai_runtime::ConstrainedSampler::new(chain, Box::new(constraint), vocab_size)
}

#[cfg(test)]
#[path = "cmd_run_tests.rs"]
mod tests;
