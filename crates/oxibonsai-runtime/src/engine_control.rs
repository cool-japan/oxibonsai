//! Engine control-plane types: cooperative cancellation, end-of-sequence
//! token **sets**, the recurrent-state reset seam, speculative-decode
//! configuration, and the GPU weight-upload decision.
//!
//! These types are deliberately kept out of [`crate::engine`] so that file
//! stays under the workspace 2000-line ceiling, and so the control plane can
//! be unit-tested without constructing a model.
//!
//! | type | finding |
//! |---|---|
//! | [`CancellationToken`] | `SV-09` — cancellation covered only the SSE path |
//! | [`EosTokenSet`] | `RT-18` — `EOS_TOKEN_ID` was Qwen3-only and single-valued |
//! | [`RecurrentState`] | `RT-28` — no recurrent-state reset seam (hybrid `M-05`) |
//! | [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`] | `perf-11` — GPU argmax kernels now tie-break to the first index (`FIX2-KERN`); gate flipped on |
//! | [`SpeculativeConfig`] | `RT-27` / `perf-16` — speculation behind an undocumented env var |
//! | [`FusedMetalRoute`] | `MET-M1` — duplicate GPU-resident weight copy |

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxibonsai_core::gguf::metadata::MetadataValue;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::{keys, tensor_names};

// ─────────────────────────────────────────────────────────────────────────────
// Cancellation (SV-09 + the wave-1 SRV-HARDEN addendum)
// ─────────────────────────────────────────────────────────────────────────────

/// A cooperative cancellation flag shared between the caller and a running
/// generation.
///
/// Cloning yields another handle to the *same* flag, so a server handler can
/// keep one clone, arm the engine with another, and
/// [`cancel`](Self::cancel) it from a timeout branch or a disconnect
/// watcher. Every decode loop in [`InferenceEngine`](crate::engine::InferenceEngine)
/// checks the armed token once per step (and, when
/// [`set_prefill_chunk_tokens`](crate::engine::InferenceEngine::set_prefill_chunk_tokens)
/// is configured, once per prefill chunk), then stops and returns the tokens
/// produced so far — the same shape as the EOS break, so a cancelled
/// generation is never an error and never loses work already done.
///
/// ## Why this exists
///
/// Before `SV-09`, only the SSE path could be cancelled, and then only
/// implicitly: `generate_streaming` breaks when `tx.send` fails because the
/// receiver was dropped. The non-streaming path had no seam at all — the
/// wave-1 review recorded that dropping the handler future on a per-request
/// timeout leaves the `spawn_blocking` generation running to completion,
/// holding the engine replica (and, on the GPU tier, the *only* replica)
/// long past the deadline. Arming a token makes that deadline real.
///
/// The flag is `Relaxed`-ordered on purpose: it guards a control decision,
/// not data hand-off. The worst case of a late-observed store is one extra
/// decode step.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a fresh, un-cancelled token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation. Idempotent, and callable from any thread.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Clear the flag so the token can be reused for another request.
    pub fn reset(&self) {
        self.flag.store(false, Ordering::Relaxed);
    }

    /// Number of live handles to this flag (the caller's plus the engine's).
    ///
    /// Exposed for tests and for an operator-facing "is anything still
    /// holding this request's token" check.
    pub fn handle_count(&self) -> usize {
        Arc::strong_count(&self.flag)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recurrent state reset seam (RT-28 / M-05)
// ─────────────────────────────────────────────────────────────────────────────

/// The per-sequence recurrent state of a hybrid (linear-attention) model.
///
/// Bonsai 2's 48 Gated-DeltaNet layers each carry a causal-conv window
/// (`[3 × 10240]`) and a `S[48][128][128]` recurrence matrix — ~157 MB of
/// state that, unlike a KV cache, is **not** masked by position: a stale `S`
/// silently contaminates the next request rather than being overwritten.
/// `RT-28` therefore requires a reset seam in the runtime *before* the state
/// itself exists.
///
/// `B2-10` supplies the concrete `RecurrentCache`; it only has to
/// `impl RecurrentState for RecurrentCache` and hand the engine one via
/// [`InferenceEngine::set_recurrent_state`](crate::engine::InferenceEngine::set_recurrent_state).
/// Everything else — [`InferenceEngine::reset`](crate::engine::InferenceEngine::reset)
/// calling [`InferenceEngine::reset_recurrent`](crate::engine::InferenceEngine::reset_recurrent),
/// which the server's per-request reset (`RT-03`) already goes through — is
/// wired here, so no follow-up edit to the runtime is needed.
///
/// `Send` is required because engine replicas are moved between threads by
/// the pool (`spawn_blocking`).
pub trait RecurrentState: Send {
    /// Clear every recurrent tensor back to its start-of-sequence value.
    ///
    /// Must be idempotent: the engine calls it on every request reset,
    /// including for a sequence that never ran.
    fn reset_recurrent(&mut self);

    /// Bytes of recurrent state held, for the memory gauge and
    /// `/admin/status`. Defaults to `0` for implementations that do not
    /// track it.
    fn recurrent_memory_bytes(&self) -> usize {
        0
    }

    /// Short human-readable name for logs (e.g. `"gated-delta-net"`).
    fn recurrent_name(&self) -> &str {
        "recurrent-state"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EOS token set (RT-18)
// ─────────────────────────────────────────────────────────────────────────────

/// Token strings that terminate generation in every model family OxiBonsai
/// loads, resolved against the GGUF's own `tokenizer.ggml.tokens` array so
/// the *ids* never have to be hardcoded.
///
/// `<|im_end|>` is the ChatML turn terminator (Qwen3 `151645`, Bonsai 2
/// `248046`); `<|endoftext|>` terminates a base/completion generation and is
/// Bonsai 2's `248044`, which its chat contract also treats as a stop.
/// `<|eot_id|>` / `<|end_of_text|>` are the Llama-3-family spellings, present
/// only in those vocabularies.
const TERMINATOR_TOKEN_STRINGS: [&str; 4] = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end_of_text|>",
];

/// A non-empty, de-duplicated set of end-of-sequence token ids.
///
/// `RT-18`: the engine used a single `EOS_TOKEN_ID` constant fixed to Qwen3's
/// `151645`. Bonsai 2's eos is `248046` — an ordinary word id in a
/// 248 320-entry vocabulary would have been compared instead — and its chat
/// contract terminates on `<|endoftext|>` (`248044`) as well, which a single
/// id cannot express.
///
/// Ordering is meaningful: the first element is the **primary** id, the one
/// [`InferenceEngine::eos_token_id`](crate::engine::InferenceEngine::eos_token_id)
/// returns for callers (and single-id consumers such as `BeamSearchConfig`)
/// that have not yet been widened to the set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EosTokenSet {
    /// Invariant: non-empty, no duplicates, `ids[0]` is the primary id.
    ids: Vec<u32>,
}

impl EosTokenSet {
    /// A set containing exactly one id.
    pub fn single(id: u32) -> Self {
        Self { ids: vec![id] }
    }

    /// Build a set from `primary` plus any number of additional ids,
    /// preserving first-seen order and dropping duplicates.
    pub fn with_extras<I: IntoIterator<Item = u32>>(primary: u32, extras: I) -> Self {
        let mut set = Self::single(primary);
        for id in extras {
            set.insert(id);
        }
        set
    }

    /// Add `id` to the set. Returns `true` if it was not already present.
    pub fn insert(&mut self, id: u32) -> bool {
        if self.ids.contains(&id) {
            return false;
        }
        self.ids.push(id);
        true
    }

    /// The primary end-of-sequence id (the GGUF's `tokenizer.ggml.eos_token_id`
    /// when it had one).
    pub fn primary(&self) -> u32 {
        // The invariant is established by every constructor: `single` seeds
        // one element and nothing removes elements.
        self.ids.first().copied().unwrap_or_default()
    }

    /// Whether `token` terminates generation.
    pub fn contains(&self, token: u32) -> bool {
        self.ids.contains(&token)
    }

    /// Every terminator id, primary first.
    pub fn as_slice(&self) -> &[u32] {
        &self.ids
    }

    /// Number of distinct terminator ids (always ≥ 1).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Always `false` — the set is non-empty by construction. Present
    /// because clippy requires it alongside [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl From<u32> for EosTokenSet {
    fn from(id: u32) -> Self {
        Self::single(id)
    }
}

/// Resolve the end-of-sequence token **set** from a loaded GGUF.
///
/// Resolution order (`RT-18`):
/// 1. [`keys::TOKENIZER_EOS_TOKEN_ID`] → the primary id; `fallback` when
///    absent.
/// 2. [`keys::TOKENIZER_EOT_TOKEN_ID`] → added when present (models that
///    distinguish "end of turn" from "end of text").
/// 3. Every [`TERMINATOR_TOKEN_STRINGS`] entry found in the GGUF's own
///    `tokenizer.ggml.tokens` array, looked up **by name**, so the ids come
///    from the file rather than from a per-model table.
///
/// The vocabulary scan reads `tokenizer.ggml.tokens` through
/// [`MetadataValue::as_array`] and compares `&str` in place: the sibling
/// `get_string_array` would clone all 248 320 entries of a Bonsai 2
/// vocabulary into owned `String`s just to find at most four of them. The
/// scan stops as soon as every terminator has been located.
pub fn resolve_eos_token_set(gguf: &GgufFile<'_>, fallback: u32) -> EosTokenSet {
    let primary = gguf
        .metadata
        .get_u32(keys::TOKENIZER_EOS_TOKEN_ID)
        .unwrap_or(fallback);
    let mut set = EosTokenSet::single(primary);

    if let Ok(eot) = gguf.metadata.get_u32(keys::TOKENIZER_EOT_TOKEN_ID) {
        set.insert(eot);
    }

    for id in terminator_ids_from_vocab(gguf) {
        set.insert(id);
    }

    set
}

/// Ids of the [`TERMINATOR_TOKEN_STRINGS`] present in the GGUF vocabulary,
/// in vocabulary order. Empty when the file carries no token list.
fn terminator_ids_from_vocab(gguf: &GgufFile<'_>) -> Vec<u32> {
    let Some(MetadataValue::Array(tokens)) = gguf.metadata.get(keys::TOKENIZER_TOKENS) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for (idx, entry) in tokens.iter().enumerate() {
        let Some(text) = entry.as_str() else { continue };
        if TERMINATOR_TOKEN_STRINGS.contains(&text) {
            // A GGUF vocabulary is indexed by token id; ids past `u32::MAX`
            // cannot exist in a file this reader accepted.
            if let Ok(id) = u32::try_from(idx) {
                found.push(id);
            }
            if found.len() == TERMINATOR_TOKEN_STRINGS.len() {
                break;
            }
        }
    }
    found
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-argmax tie-break dependency gate (wave-2 verifier finding, RT-24/perf-11)
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the GPU argmax kernels reachable from the fused decode route break
/// ties toward the **global first** index, matching
/// [`argmax_first`](crate::engine_greedy::argmax_first) and the CPU sampler's
/// convention.
///
/// [`InferenceEngine::greedy_gpu_eligible`](crate::engine::InferenceEngine::greedy_gpu_eligible)
/// ANDs this in, and
/// [`InferenceEngine::generate_greedy_gpu`](crate::engine::InferenceEngine::generate_greedy_gpu)
/// — the CLI's direct, unconditional `--temperature 0` fast path, which does
/// not go through `greedy_gpu_eligible` at all — checks it too. Now that this
/// is `true`, every path in this crate that reaches the GPU argmax kernel
/// gets the same first-index tie-break the CPU decode loop always had, so
/// the GPU-argmax fast path (4-byte readback instead of the full logits row)
/// is reachable again.
///
/// ## Why this was `false` (fixed by `perf-11` / `FIX2-KERN`)
///
/// A wave-2 verifier review read the MSL kernel
/// (`crates/oxibonsai-kernels/src/gpu_backend/kernel_sources/utility.rs`,
/// `argmax`) and its CUDA twin (`cuda_kernels.rs`, `argmax_f32`) and
/// confirmed — by tracing the reduction directly, not just reading the
/// finding — that neither tied toward the global first index. Each thread
/// first scanned a strided subset of the input (`i = tid, tid +
/// threads_per_group, ...`) and kept the first index *within that subset* on
/// a tie (a strict `>` comparison). A pairwise tree reduction then combined
/// the per-thread winners, but its tie rule ("keep the current slot's
/// payload when the challenger is only *equal*, never strictly greater") was
/// a slot-position rule, not an original-index rule, and a payload's slot
/// changes as it advances through the tree — so the two were not
/// equivalent. Traced by hand for the finding's own example
/// (`threads_per_group = 1024`, equal maxima at input indices `1000` and
/// `2000`): index `1000` maps to thread `1000`, index `2000` maps to thread
/// `2000 mod 1024 = 976`; the reduction converged thread `976`'s payload
/// down to slot `0` one round before thread `1000`'s payload reached slot
/// `8`, so the final equal-value comparison kept slot `0`'s payload —
/// **index `2000`, not the smaller `1000`**. That regressed the
/// "temperature 0, all penalties at 1.0 ⇒ every kernel tier produces
/// identical tokens" invariant on any model whose logits contain exact
/// duplicates, which is exactly the regime a ternary/1-bit quantized model
/// (the *only* family that reaches this route — see
/// [`FusedMetalRoute::is_fused`]) can hit.
///
/// ## The fix (landed; this gate now reflects it)
///
/// Both kernels now compare the *payload's original index* on a value-tie
/// instead of only the value. `utility.rs`'s `MSL_ARGMAX` tree-reduction
/// step:
///
/// ```text
/// float ov = shared_vals[tid + stride];
/// uint oi = shared_idxs[tid + stride];
/// if (ov > shared_vals[tid] || (ov == shared_vals[tid] && oi < shared_idxs[tid])) {
///     shared_vals[tid] = ov;
///     shared_idxs[tid] = oi;
/// }
/// ```
///
/// mirrored in `cuda_kernels.rs`'s `argmax_f32`. This resolves both the
/// example above and a second, independently-traced case (two ties landing
/// in different tree brackets could otherwise let the higher original index
/// win even when the *slot* tie-break would look like it favoured the lower
/// one). The dependency this gate existed for is now satisfied by
/// `crates/oxibonsai-kernels/tests/gpu_argmax_tiebreak.rs`, a kernel-level
/// harness that compiles the *shipped* `MSL_ARGMAX` source and dispatches it
/// with the *shipped* geometry (one threadgroup, 1024 threads): the worked
/// 1000/2000 example, a unique-maximum sanity sweep, and ≥ 300 randomized
/// multi-way ties (tie multiplicity 2..=16, placed across threadgroup slots,
/// within one slot, at the edges, and over vocab sizes including Bonsai 2's
/// non-power-of-two 248 320) all resolve to the minimal tied index — 320/320
/// passing with 258/320 cases actually crossing a threadgroup slot boundary
/// (the shape that discriminates the old bug from a correct first-index
/// rule, so the test is not vacuously green). Every other part of the
/// `perf-11` GPU-argmax route (penalty refusal, EOS-set handling,
/// cancellation, speculative drafting) was already correct before this flip
/// and needed no further change.
///
/// ## Reverting this gate
///
/// If a future regression is found in either kernel, set this back to
/// `false` and update
/// [`gpu_argmax_tiebreak_gate_is_on_after_the_kernel_fix`](tests::gpu_argmax_tiebreak_gate_is_on_after_the_kernel_fix)'s
/// const-assert back to its original (negated) form — the tripwire is
/// deliberately kept in that test (currently checking `true`) so the crate
/// fails to *compile*, not merely fails a test, the moment the two drift out
/// of sync again.
pub const GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX: bool = true;

// ─────────────────────────────────────────────────────────────────────────────
// Speculative decoding configuration (RT-27 / perf-16)
// ─────────────────────────────────────────────────────────────────────────────

/// Legacy environment switch that enabled n-gram speculation.
///
/// Kept as an **override** for the documented [`SpeculativeConfig`] so
/// existing scripts keep working; `RT-27` is closed by the config field, not
/// by removing the variable.
pub const SPECULATIVE_ENV_VAR: &str = "OXIBONSAI_SPEC";

/// Legacy environment switch for
/// [`SpeculativeConfig::force_cpu_decode_after`].
///
/// A wave-2 verifier review found this was the one decode-loop environment
/// switch `RT-27` left behind: it used to be read with
/// `std::env::var("OXIBONSAI_FORCE_CPU_DECODE_AFTER")` directly inside
/// `generate_greedy_gpu_unchecked`, alongside (but not part of) the
/// documented [`SpeculativeConfig`]. It is now folded into the same config
/// struct and resolved by the same [`SpeculativeConfig::with_env_override`]
/// call `RT-27` already made for [`SPECULATIVE_ENV_VAR`].
pub const FORCE_CPU_DECODE_AFTER_ENV_VAR: &str = "OXIBONSAI_FORCE_CPU_DECODE_AFTER";

/// Which drafting strategy the GPU greedy decode loop uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpeculativeMode {
    /// No drafting: one verified token per forward pass.
    #[default]
    Off,
    /// Prompt/­history n-gram drafting verified by a single batched forward
    /// (`crate::ngram_cache`), with the adaptive accept-rate gate below.
    Ngram,
}

/// Speculative-decode settings for the greedy GPU path.
///
/// `RT-27` / `perf-16`: the n-gram speculative decoder was fully implemented
/// and wired, but reachable only through the undocumented `OXIBONSAI_SPEC=1`
/// environment variable read *inside* the decode function. It is now a
/// documented configuration field on the engine
/// ([`InferenceEngine::set_speculative`](crate::engine::InferenceEngine::set_speculative)),
/// read once at the start of a generation.
///
/// ## Default: [`SpeculativeMode::Off`]
///
/// Verification uses `forward_prefill_verify` (a *batched* forward) while the
/// non-speculative step uses the single-token fused decode. The two are not
/// proven bit-identical on the Metal path — batch-shaped kernels take
/// different reduction orders — so speculation can, in principle, change a
/// token at a near-tie. Greedy output equality across tiers is a release gate
/// (`cross_backend_determinism_tests`), so the default stays `Off` until the
/// real-model parity harness measures both the accept rate and bit-equality
/// on this hardware. `OXIBONSAI_SPEC=1` (or `set_speculative`) opts in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeculativeConfig {
    /// Drafting strategy.
    pub mode: SpeculativeMode,
    /// Draft length per speculation attempt (`k`).
    pub draft_len: usize,
    /// Number of committed tokens to decode normally before the first
    /// speculation attempt, so the n-gram cache has content to draft from.
    pub warmup_tokens: usize,
    /// Minimum measured accept rate to keep speculating, once
    /// [`min_attempts_before_gating`](Self::min_attempts_before_gating)
    /// attempts have been made.
    pub min_accept_rate: f64,
    /// Attempts made optimistically before the accept-rate gate applies.
    pub min_attempts_before_gating: u64,
    /// Even below `min_accept_rate`, re-probe every `retry_interval`
    /// attempts so a sequence that becomes repetitive later can recover the
    /// speedup.
    pub retry_interval: u64,
    /// Debug / test seam: after this many committed tokens, force the
    /// remainder of a GPU-greedy decode onto the CPU fallback path,
    /// exercising the Metal→CPU KV-cache rebuild
    /// (`InferenceEngine::greedy_decode_token_with_fallback`) without a real
    /// Metal dispatch failure. `None` (the default) never forces it.
    ///
    /// Folded in from the legacy [`FORCE_CPU_DECODE_AFTER_ENV_VAR`]
    /// environment switch (a wave-2 verifier finding): this was the last
    /// undocumented decode-loop environment variable `RT-27` left behind.
    pub force_cpu_decode_after: Option<usize>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            mode: SpeculativeMode::Off,
            draft_len: 4,
            warmup_tokens: 15,
            min_accept_rate: 0.6,
            min_attempts_before_gating: 5,
            retry_interval: 20,
            force_cpu_decode_after: None,
        }
    }
}

impl SpeculativeConfig {
    /// The default configuration with n-gram drafting enabled.
    pub fn ngram() -> Self {
        Self {
            mode: SpeculativeMode::Ngram,
            ..Self::default()
        }
    }

    /// Whether drafting is enabled and can produce a non-empty draft.
    pub fn is_enabled(&self) -> bool {
        self.mode == SpeculativeMode::Ngram && self.draft_len > 0
    }

    /// Apply the legacy [`SPECULATIVE_ENV_VAR`] and
    /// [`FORCE_CPU_DECODE_AFTER_ENV_VAR`] overrides, if set.
    ///
    /// `OXIBONSAI_SPEC=1` turns n-gram drafting on; `OXIBONSAI_SPEC=0`
    /// (or any other value) forces it off. `OXIBONSAI_FORCE_CPU_DECODE_AFTER=<n>`
    /// sets [`force_cpu_decode_after`](Self::force_cpu_decode_after) to
    /// `Some(n)`; unset or unparseable leaves it untouched. Either variable
    /// being absent leaves the corresponding field untouched, so a
    /// configuration set through the API always wins over an absent
    /// variable.
    pub fn with_env_override(self) -> Self {
        let after_spec = match std::env::var(SPECULATIVE_ENV_VAR) {
            Ok(v) if v == "1" => Self {
                mode: SpeculativeMode::Ngram,
                ..self
            },
            Ok(_) => Self {
                mode: SpeculativeMode::Off,
                ..self
            },
            Err(_) => self,
        };
        match std::env::var(FORCE_CPU_DECODE_AFTER_ENV_VAR)
            .ok()
            .and_then(|v| v.parse().ok())
        {
            Some(n) => Self {
                force_cpu_decode_after: Some(n),
                ..after_spec
            },
            None => after_spec,
        }
    }

    /// Whether another speculation attempt is worthwhile given the running
    /// accept statistics.
    ///
    /// Optimistic for the first [`min_attempts_before_gating`](Self::min_attempts_before_gating)
    /// attempts, then gated on the measured accept rate with a periodic
    /// re-probe.
    pub fn should_attempt(&self, attempts: u64, accepted_total: u64) -> bool {
        if attempts < self.min_attempts_before_gating {
            return true;
        }
        let capacity = (attempts as f64 * self.draft_len as f64).max(1.0);
        let accuracy = accepted_total as f64 / capacity;
        accuracy > self.min_accept_rate
            || (self.retry_interval > 0 && attempts.is_multiple_of(self.retry_interval))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU weight-upload decision (MET-M1)
// ─────────────────────────────────────────────────────────────────────────────

/// Which fused Metal decode route a GGUF's quantization layout selects.
///
/// `forward_metal.rs` runs the fused `MetalGraph` path only for a **Ternary**
/// or **OneBit** LM head; an FP8 / K-quant / Q4_0 / Q8_0 / FP32 head returns
/// an error and the model falls back to block dispatch. `gpu_cache.rs`'s
/// one-bit branch additionally requires every block to carry Q1 blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FusedMetalRoute {
    /// Not served by the fused graph: block dispatch, which reads the
    /// `Scirs2Backend` weight cache.
    #[default]
    None,
    /// All-ternary (`TQ2_0_g128` family) — `build_ternary_gpu_cache`.
    Ternary,
    /// All-1-bit (`Q1_0_g128`) — `build_cached_weights` with per-layer GPU
    /// handles.
    OneBit,
}

impl FusedMetalRoute {
    /// Whether decode goes through the fused Metal graph at all.
    ///
    /// Precondition for routing greedy generation through the GPU argmax
    /// (`perf-11`): `forward_greedy_gpu` maintains only the GPU-resident KV
    /// cache, so a model that is not fused would decode against an all-zero
    /// cache.
    pub fn is_fused(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether `BonsaiModel::upload_weights_to_gpu` is **redundant** for this
    /// route (`MET-M1`).
    ///
    /// That call fills `Scirs2Backend::weight_cache` — a second,
    /// never-evicted, GPU-resident copy of every quantized tensor, measured
    /// at 197 tensors / **435.69 MB** (the whole quantized model) on the
    /// ternary 1.7B.
    ///
    /// Only the **ternary** route can skip it. `build_ternary_gpu_cache`
    /// copies raw ternary bytes and mints its own handle ids ("no Q1 GPU
    /// handles exist on the ternary path"), and *every* ternary Metal entry
    /// point — `try_metal_full_forward_with_lm_head_ternary` (decode),
    /// `try_metal_prefill_with_lm_head_ternary` (prefill),
    /// `try_metal_prefill_verify_ternary_path` (speculative verify) and
    /// `try_metal_full_forward_ternary_inner` — derives its handles from the
    /// fixed `5_000_000` / `6_000_000` namespaces rather than calling
    /// `*_gpu_handle()`. Nothing on that route reads the uploaded buffers.
    ///
    /// The **one-bit** route is the opposite: `build_cached_weights` keys the
    /// `MetalGraph` cache on the `fused_qkv` / `attn_proj` / `gate_up` /
    /// `down` handles that `upload_weights_to_gpu` produces, and the Q1
    /// forward/prefill/verify paths all bail out when any of them is
    /// `None`. Skipping the upload there makes the fused cache fail to build
    /// outright (`missing GPU handle for layer 0 fused_qkv`, reproduced by
    /// `engine_pool`'s shared-embedding test, whose fixture is Q1). The
    /// upload is load-bearing on that route, not a duplicate.
    ///
    /// Note what skipping costs on the ternary route: should the fused path
    /// fail at runtime, the block-dispatch fallback now runs on CPU SIMD
    /// GEMV rather than cached GPU GEMV. That is the intended `MET-M1`
    /// exchange — the fallback is an error path, the 435 MB was permanent.
    pub fn gpu_weight_upload_redundant(&self) -> bool {
        matches!(self, Self::Ternary)
    }
}

/// Classify a GGUF's quantization layout into a [`FusedMetalRoute`].
///
/// Conservative in exactly one direction: anything it cannot prove — a
/// mixed-quant file, a missing `output.weight`, a K-quant head — is
/// [`FusedMetalRoute::None`], which preserves today's behaviour exactly
/// (upload the weights, no GPU-argmax routing). CUDA builds never consult it:
/// there the uploaded weights are live for FP8 / Q4_0 / Q8_0 block dispatch.
pub fn gguf_fused_metal_route(gguf: &GgufFile<'_>) -> FusedMetalRoute {
    let Some(lm_head) = gguf.tensors.get(tensor_names::OUTPUT) else {
        // No LM head tensor: `BonsaiModel::from_gguf` will either fail or
        // build a non-fused head. Either way, change nothing.
        return FusedMetalRoute::None;
    };

    let route = if lm_head.tensor_type.is_ternary() {
        FusedMetalRoute::Ternary
    } else if lm_head.tensor_type.is_one_bit() {
        FusedMetalRoute::OneBit
    } else {
        return FusedMetalRoute::None;
    };

    // Every quantized *block* matrix must belong to the same family; a mixed
    // file takes the block-dispatch path for at least one layer. `blk.*
    // .weight` covers every transformer-layer matrix; `token_embd.weight`
    // is checked explicitly alongside them (a wave-2 verifier finding: a
    // quantized-but-non-uniform embedding table was previously invisible to
    // this scan, since embedding lookup is CPU-side and the scan only
    // walked `blk.*`/`output.weight`, so a GGUF with all-ternary blocks but
    // e.g. a Q8_0 `token_embd.weight` would still classify as uniformly
    // fused here — harmless today only because nothing on the fused route
    // reads that tensor, which is not a property this function should rely
    // on silently).
    let uniform = gguf.tensors.iter().all(|(name, info)| {
        let is_relevant_block_matrix = (name.starts_with("blk.") && name.ends_with(".weight"))
            || name == tensor_names::TOKEN_EMBD;
        if !is_relevant_block_matrix {
            return true;
        }
        // Norms and other non-blocked tensors (F32/F16/BF16) have block size
        // 1 and are irrelevant to the GPU weight cache.
        if info.tensor_type.block_size() <= 1 {
            return true;
        }
        match route {
            FusedMetalRoute::Ternary => info.tensor_type.is_ternary(),
            FusedMetalRoute::OneBit => info.tensor_type.is_one_bit(),
            FusedMetalRoute::None => false,
        }
    });

    if uniform {
        route
    } else {
        FusedMetalRoute::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    // ── CancellationToken ────────────────────────────────────────────────

    #[test]
    fn cancellation_token_starts_unset_and_latches() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled(), "cancel must be idempotent");
        token.reset();
        assert!(!token.is_cancelled(), "reset re-arms the token");
    }

    #[test]
    fn cancellation_token_clone_shares_one_flag() {
        let token = CancellationToken::new();
        let engine_side = token.clone();
        assert_eq!(token.handle_count(), 2);
        token.cancel();
        assert!(
            engine_side.is_cancelled(),
            "a clone must observe the original's cancellation"
        );
    }

    #[test]
    fn cancellation_token_crosses_threads() {
        let token = CancellationToken::new();
        let worker = token.clone();
        let handle = std::thread::spawn(move || {
            for _ in 0..1_000_000 {
                if worker.is_cancelled() {
                    return true;
                }
                std::hint::spin_loop();
            }
            false
        });
        token.cancel();
        assert!(handle.join().unwrap_or(false), "worker must observe cancel");
    }

    // ── RecurrentState seam ──────────────────────────────────────────────

    #[derive(Default)]
    struct FakeRecurrent {
        resets: usize,
        dirty: bool,
    }

    impl RecurrentState for FakeRecurrent {
        fn reset_recurrent(&mut self) {
            self.resets += 1;
            self.dirty = false;
        }

        fn recurrent_memory_bytes(&self) -> usize {
            157_286_400 + 5_898_240
        }
    }

    #[test]
    fn recurrent_state_trait_defaults_and_reset() {
        let mut state = FakeRecurrent {
            resets: 0,
            dirty: true,
        };
        assert_eq!(state.recurrent_name(), "recurrent-state");
        state.reset_recurrent();
        assert_eq!(state.resets, 1);
        assert!(!state.dirty);
        state.reset_recurrent();
        assert_eq!(state.resets, 2, "reset must be idempotent, not a toggle");
        assert_eq!(state.recurrent_memory_bytes(), 163_184_640);
    }

    // ── EosTokenSet ──────────────────────────────────────────────────────

    #[test]
    fn eos_set_dedups_and_keeps_primary_first() {
        let set = EosTokenSet::with_extras(248_046, [248_044, 248_046, 248_044]);
        assert_eq!(set.primary(), 248_046);
        assert_eq!(set.as_slice(), &[248_046, 248_044]);
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.contains(248_046));
        assert!(set.contains(248_044));
        assert!(!set.contains(151_645));
    }

    #[test]
    fn eos_set_single_is_non_empty() {
        let set = EosTokenSet::single(7);
        assert_eq!(set.primary(), 7);
        assert_eq!(set.as_slice(), &[7]);
        assert_eq!(EosTokenSet::from(7u32), set);
    }

    /// RT-18's headline case, reproduced against a synthetic GGUF built with
    /// Bonsai 2's key/vocabulary shape: eos `248046` (`<|im_end|>`) **and**
    /// `<|endoftext|>` `248044` must both terminate, and the primary id must
    /// remain the GGUF's own `eos_token_id`.
    #[test]
    fn resolve_eos_set_from_bonsai2_shaped_gguf() {
        // A miniature vocabulary whose last entries mirror Bonsai 2's
        // special-token block ordering (`<|endoftext|>` immediately before
        // `<|im_end|>`).
        let mut vocab: Vec<String> = (0..8).map(|i| format!("tok{i}")).collect();
        vocab.push("<|endoftext|>".to_string()); // id 8
        vocab.push("<|im_start|>".to_string()); // id 9
        vocab.push("<|im_end|>".to_string()); // id 10

        let mut w = GgufWriter::new();
        w.add_metadata("tokenizer.ggml.tokens", MetadataWriteValue::ArrayStr(vocab));
        w.add_metadata("tokenizer.ggml.eos_token_id", MetadataWriteValue::U32(10));
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let set = resolve_eos_token_set(&gguf, 151_645);
        assert_eq!(set.primary(), 10, "primary must be the GGUF's eos id");
        assert!(set.contains(10), "<|im_end|> terminates");
        assert!(
            set.contains(8),
            "<|endoftext|> must terminate too (Bonsai 2 chat contract): {:?}",
            set.as_slice()
        );
        assert!(!set.contains(9), "<|im_start|> is not a terminator");
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn resolve_eos_set_honours_eot_key() {
        let mut w = GgufWriter::new();
        w.add_metadata("tokenizer.ggml.eos_token_id", MetadataWriteValue::U32(100));
        w.add_metadata(
            oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_EOT_TOKEN_ID,
            MetadataWriteValue::U32(101),
        );
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let set = resolve_eos_token_set(&gguf, 151_645);
        assert_eq!(set.as_slice(), &[100, 101]);
    }

    #[test]
    fn resolve_eos_set_falls_back_when_key_absent() {
        let w = GgufWriter::new();
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let set = resolve_eos_token_set(&gguf, 151_645);
        assert_eq!(set.as_slice(), &[151_645]);
    }

    #[test]
    fn resolve_eos_set_ignores_a_vocab_without_terminators() {
        let vocab: Vec<String> = (0..4).map(|i| format!("tok{i}")).collect();
        let mut w = GgufWriter::new();
        w.add_metadata("tokenizer.ggml.tokens", MetadataWriteValue::ArrayStr(vocab));
        w.add_metadata("tokenizer.ggml.eos_token_id", MetadataWriteValue::U32(3));
        let bytes = w.to_bytes().expect("write synthetic gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");

        let set = resolve_eos_token_set(&gguf, 151_645);
        assert_eq!(set.as_slice(), &[3]);
    }

    // ── GPU-argmax tie-break gate ────────────────────────────────────────

    /// A tripwire, not a behavioural test: this constant must not flip back
    /// to `false` as a side effect of an unrelated edit. `perf-11` /
    /// `FIX2-KERN` landed the `utility.rs` / `cuda_kernels.rs` fix this gate
    /// was waiting on, plus the kernel-level randomized tie test
    /// (`crates/oxibonsai-kernels/tests/gpu_argmax_tiebreak.rs`, 320/320
    /// passing, 258/320 cases crossing a threadgroup slot so the coverage is
    /// not vacuous), so the constant is now `true`. Inverted from its
    /// original (negated) form -- see [`GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX`]'s
    /// "Reverting this gate" doc section for what to do if a future kernel
    /// regression requires flipping it back. If this assertion is the only
    /// thing standing between you and turning the GPU-argmax route back
    /// *off*, read that doc comment before deleting it.
    #[test]
    fn gpu_argmax_tiebreak_gate_is_on_after_the_kernel_fix() {
        // A compile-time assertion (clippy correctly flags a runtime
        // `assert!` on a `const` as pointless), which is a *stronger*
        // tripwire than a test failure: this crate will not build at all
        // once the constant flips back, until whoever flips it also touches
        // this block. Kept inside a `#[test]` fn purely so it shows up next
        // to this module's other GPU-argmax-gate coverage.
        const {
            assert!(
                GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX,
                "the MSL/CUDA argmax kernels were fixed to tie-break toward \
                 the global first index and crates/oxibonsai-kernels/tests/\
                 gpu_argmax_tiebreak.rs verifies it (perf-11 / FIX2-KERN); \
                 if this constant is being reverted to false, a kernel \
                 regression must have been found -- update this assertion \
                 back to `!GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX` and its \
                 message together with the revert, do not just delete it"
            );
        }
    }

    /// Flipping the gate off is a live, correctly-routing code path, not
    /// dead code the flip to `true` left behind: [`SpeculativeConfig`] (and
    /// every caller that configures a repetition/frequency/presence penalty)
    /// forces the exact same "decode the full logit row, argmax on the CPU"
    /// fallback this gate used to force unconditionally, and that fallback
    /// must keep working regardless of the gate's value -- it is what a
    /// future revert would fall back onto. This is
    /// [`greedy_penalties_active`](crate::engine::InferenceEngine::greedy_penalties_active),
    /// exercised without needing to touch the (now-`true`) constant at all:
    /// a penalised configuration must be refused by the *other*
    /// [`crate::engine::InferenceEngine::greedy_gpu_eligible`] conditions
    /// independently of this gate.
    #[test]
    fn the_non_gate_conditions_still_refuse_eligibility_on_their_own() {
        use crate::engine::InferenceEngine;
        use crate::sampling::{PenaltyParams, SamplingParams};
        use oxibonsai_core::config::Qwen3Config;

        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 16,
        };
        let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 42);
        // A synthetic engine never uses the fused Metal graph, so it is
        // refused regardless of the tie-break gate's value.
        assert!(!engine.uses_fused_gpu_decode());
        assert!(!engine.greedy_gpu_eligible(false));

        // With a penalty configured, `greedy_penalties_active` must still
        // say so -- this is the condition that would keep the GPU-argmax
        // route closed if the tie-break gate were ever reverted to `false`
        // and then reopened once some *other* eligibility condition failed.
        engine.set_penalties(PenaltyParams::new(0.5, 0.0));
        assert!(engine.greedy_penalties_active());
        assert!(!engine.greedy_gpu_eligible(false));
    }

    // ── SpeculativeConfig ────────────────────────────────────────────────

    #[test]
    fn speculative_defaults_are_off_but_configured() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.mode, SpeculativeMode::Off);
        assert!(!cfg.is_enabled());
        assert_eq!(cfg.draft_len, 4);
        assert_eq!(cfg.warmup_tokens, 15);
        assert_eq!(
            cfg.force_cpu_decode_after, None,
            "the debug/test CPU-fallback seam must default to never-force"
        );

        let on = SpeculativeConfig::ngram();
        assert!(on.is_enabled());
        assert_eq!(on.draft_len, cfg.draft_len, "only the mode differs");
        assert_eq!(on.force_cpu_decode_after, None);
    }

    #[test]
    fn force_cpu_decode_after_is_independently_settable() {
        // Struct-update syntax must still work now that the field exists,
        // and setting it must not perturb `is_enabled`/`draft_len`.
        let cfg = SpeculativeConfig {
            force_cpu_decode_after: Some(3),
            ..SpeculativeConfig::ngram()
        };
        assert_eq!(cfg.force_cpu_decode_after, Some(3));
        assert!(cfg.is_enabled());
        assert_eq!(cfg.draft_len, 4);
    }

    #[test]
    fn speculative_zero_draft_len_is_disabled() {
        let cfg = SpeculativeConfig {
            draft_len: 0,
            ..SpeculativeConfig::ngram()
        };
        assert!(!cfg.is_enabled());
    }

    #[test]
    fn speculative_gate_is_optimistic_then_accuracy_driven() {
        let cfg = SpeculativeConfig::ngram();
        // First attempts are optimistic.
        assert!(cfg.should_attempt(0, 0));
        assert!(cfg.should_attempt(4, 0));
        // 8 attempts × 4 drafted = 32 capacity; 30 accepted ≈ 94 % → keep going.
        assert!(cfg.should_attempt(8, 30));
        // 6 accepted of 32 ≈ 19 % → stop, except on the retry probe.
        assert!(!cfg.should_attempt(8, 6));
        assert!(
            cfg.should_attempt(20, 6),
            "attempt 20 is the periodic re-probe"
        );
    }

    // ── MET-M1 fused-route scan ──────────────────────────────────────────

    fn gguf_with_tensors(entries: &[(&str, TensorType, Vec<u64>)]) -> Vec<u8> {
        let mut w = GgufWriter::new();
        for (name, ty, shape) in entries {
            let elems: u64 = shape.iter().product();
            let bytes = ty.expected_bytes(elems) as usize;
            w.add_tensor(TensorEntry {
                name: (*name).to_string(),
                tensor_type: *ty,
                shape: shape.clone(),
                data: vec![0u8; bytes],
            });
        }
        w.to_bytes().expect("write synthetic gguf")
    }

    #[test]
    fn all_ternary_file_is_fused_and_makes_the_upload_redundant() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.attn_q.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.ffn_up.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.attn_norm.weight", TensorType::F32, vec![128]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let route = gguf_fused_metal_route(&gguf);
        assert_eq!(route, FusedMetalRoute::Ternary);
        assert!(route.is_fused());
        assert!(
            route.gpu_weight_upload_redundant(),
            "the ternary GPU cache copies raw bytes and mints its own handles"
        );
    }

    /// The one-bit route is fused, but its `MetalGraph` cache is *keyed on*
    /// the handles `upload_weights_to_gpu` produces, so the upload must not
    /// be skipped — skipping it fails the cache build with "missing GPU
    /// handle for layer 0 fused_qkv".
    #[test]
    fn all_one_bit_file_is_fused_but_still_needs_the_upload() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::Q1_0G128, vec![128, 4]),
            ("blk.0.attn_q.weight", TensorType::Q1_0G128, vec![128, 4]),
            ("blk.0.attn_norm.weight", TensorType::F32, vec![128]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let route = gguf_fused_metal_route(&gguf);
        assert_eq!(route, FusedMetalRoute::OneBit);
        assert!(route.is_fused());
        assert!(
            !route.gpu_weight_upload_redundant(),
            "the Q1 fused cache is built from the uploaded handles"
        );
    }

    #[test]
    fn k_quant_lm_head_is_not_fused() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::Q4_K, vec![256, 4]),
            ("blk.0.attn_q.weight", TensorType::Q4_K, vec![256, 4]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let route = gguf_fused_metal_route(&gguf);
        assert_eq!(route, FusedMetalRoute::None);
        assert!(!route.is_fused());
        assert!(!route.gpu_weight_upload_redundant());
    }

    #[test]
    fn mixed_quant_file_is_not_fused() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.attn_q.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.1.attn_q.weight", TensorType::Q8_0, vec![128, 4]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(
            gguf_fused_metal_route(&gguf),
            FusedMetalRoute::None,
            "one non-ternary block matrix means at least one layer is not fused"
        );
    }

    /// Wave-2 verifier finding: a quantized `token_embd.weight` that does
    /// not match the block family must break fused classification exactly
    /// like a mismatched `blk.*` matrix would, even though nothing on the
    /// fused route reads the embedding table today.
    #[test]
    fn mixed_quant_token_embd_is_not_fused() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.attn_q.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("token_embd.weight", TensorType::Q8_0, vec![128, 4]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(
            gguf_fused_metal_route(&gguf),
            FusedMetalRoute::None,
            "a non-ternary token_embd.weight must not classify as uniformly ternary"
        );
    }

    /// The companion positive case: a `token_embd.weight` that *does* match
    /// the block family must not itself block fused classification.
    #[test]
    fn uniform_quant_token_embd_stays_fused() {
        let bytes = gguf_with_tensors(&[
            ("output.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("blk.0.attn_q.weight", TensorType::TQ2_0_g128, vec![128, 4]),
            ("token_embd.weight", TensorType::TQ2_0_g128, vec![128, 4]),
        ]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(gguf_fused_metal_route(&gguf), FusedMetalRoute::Ternary);
    }

    #[test]
    fn file_without_an_output_weight_is_not_fused() {
        let bytes =
            gguf_with_tensors(&[("blk.0.attn_q.weight", TensorType::TQ2_0_g128, vec![128, 4])]);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(gguf_fused_metal_route(&gguf), FusedMetalRoute::None);
    }

    #[test]
    fn default_route_is_the_conservative_one() {
        assert_eq!(FusedMetalRoute::default(), FusedMetalRoute::None);
    }
}
