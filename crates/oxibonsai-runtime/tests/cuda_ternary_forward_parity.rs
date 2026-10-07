//! CUDA ternary (TQ2_0_g128) CPU↔GPU parity gate.
//!
//! The cross-backend determinism guard (`cross_backend_determinism_tests.rs`)
//! only covers Metal; the CUDA ternary path was historically "deferred, needs
//! hw". This test fills that gap on real CUDA hardware: it drives the real
//! ternary GGUF on the scalar CPU reference (`KernelTier::Reference`) and the
//! CUDA GPU path (`KernelTier::Gpu`) and asserts identical greedy output.
//!
//! It is what surfaced the prefill→decode KV-cache handoff bug: with the CUDA
//! TQ2 batch prefill enabled, prompts longer than ~16 tokens diverged from CPU
//! by a large margin (decode logit Δ ≈ 7.3) because batch prefill writes the
//! prompt KV into a prefill-private cache that the per-token decode path never
//! reads. With that path disabled (the fix), the sequential per-token prefill
//! shares the decode KV cache and CPU↔CUDA agree (decode Δ ≈ 0.002).
//!
//! Both cases below always exist (this file carries no `native-cuda`/
//! `target_os` gate of its own) and self-skip, recording the miss under
//! [`Capability::CudaHardware`], on a build with no CUDA target or no
//! accessible device, and when no usable model file is found (see "The model
//! file" below); the real comparison lives in [`cuda_impl`] and compiles
//! only with `--features native-cuda` on Linux/Windows. Neither branch ever
//! writes `executed: true` from file presence alone — only after a genuine
//! device probe *and* the comparison it names has passed. Run for real (with
//! a `TQ2_0_g128` file, see below) with:
//!   OXI_MODEL=/abs/path/Ternary-Bonsai-8B.gguf \
//!     cargo test -p oxibonsai-runtime --features native-cuda \
//!     --test cuda_ternary_forward_parity
//!
//! # The model file
//!
//! `OXI_MODEL`, when set and non-empty, names the file, which is used as
//! given. Otherwise the testkit fallback (`$OXIBONSAI_MODELS_DIR`, else
//! `<workspace>/models`) is searched for `Ternary-Bonsai-8B.gguf` and then,
//! when that file is absent or stored as `PQ2_0`, for
//! `Ternary-Bonsai-8B-tq2_0_g128.gguf` (the two names the F-M1 harness
//! `cuda_graph_slot_two_models.rs` looks for). The chosen path is printed.
//!
//! The chosen file's stored format is probed before anything loads it: its
//! LM-head tensor type (`output.weight`, else `token_embd.weight`, the
//! loader's own rule) is read from the GGUF header through a memory map.
//! Since 3c9993a, `oxibonsai convert --quant tq2_0_g128` (what
//! `scripts/download_ternary.sh` runs) writes `TQ2_0_g128` (ggml id 42);
//! pre-release 0.2.4 builds before that commit wrote PrismML `PQ2_0` (id
//! 142) instead, which the `qwen3` runtime cannot load:
//! `BonsaiModel::from_gguf` refuses a `PQ2_0` LM head, for which no
//! output-projection wrapper exists. On such a file both cases self-skip
//! with a `skip:` line that names the file and says how to get an id-42 one
//! (re-convert it with the 0.2.4 converter, or point `OXI_MODEL` /
//! `OXIBONSAI_MODELS_DIR` at an id-42 file), and record `executed: false`.
//! Unlike the P14/P15 and F-M1 harnesses, this test never re-encodes a
//! `PQ2_0` file. A file that cannot be opened self-skips as well; one that
//! does not parse as GGUF fails the test, as loading it would. Every other
//! stored type goes to the loader as before.

// Only the non-CUDA arm's `#[test]` bodies call these directly (the CUDA arm
// calls through `cuda_impl`, which imports its own copy) — gated the same
// way those bodies are, so a native-cuda/Linux-or-Windows build never sees
// an unused import here.
#[cfg(not(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
)))]
use oxibonsai_testkit::capability::{record_skipped, Capability};

/// Capability-record name of [`real_ternary_cpu_cuda_parity`].
const REAL_TERNARY_CPU_CUDA_PARITY: &str =
    "oxibonsai-runtime::cuda_ternary_forward_parity::real_ternary_cpu_cuda_parity";
/// Capability-record name of [`real_ternary_decode_logit_delta`].
const REAL_TERNARY_DECODE_LOGIT_DELTA: &str =
    "oxibonsai-runtime::cuda_ternary_forward_parity::real_ternary_decode_logit_delta";

/// The real CUDA-hardware implementation: only compiled where CUDA can
/// actually be exercised. Kept as its own module (rather than inline
/// `#[cfg]` blocks inside the `#[test]` bodies) so the non-CUDA arm below
/// never has to reference a symbol that does not exist on this build.
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod cuda_impl {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_core::gguf::types::GgufTensorType;
    use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
    use oxibonsai_kernels::CudaGraph;
    use oxibonsai_model::model::BonsaiModel;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::sampling::SamplingParams;
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
    use oxibonsai_testkit::workspace::{find_model, models_dir};

    const MAX_SEQ: usize = 512;
    /// The ternary GGUF the testkit fallback looks for first when `OXI_MODEL`
    /// is unset.
    const TERNARY_FILE: &str = "Ternary-Bonsai-8B.gguf";
    /// The id-42 (`TQ2_0_g128`) file's name, looked for when `OXI_MODEL` is
    /// unset and [`TERNARY_FILE`] is absent or stored as `PQ2_0` (the F-M1
    /// harness's `TQ2_FILE_NATIVE`).
    const TERNARY_FILE_NATIVE: &str = "Ternary-Bonsai-8B-tq2_0_g128.gguf";

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        }
    }

    /// Greedily generate `n` tokens from `prompt` on the given kernel tier.
    fn run(gguf_bytes: &[u8], tier: KernelTier, prompt: &[u32], n: usize) -> Vec<u32> {
        let gguf = GgufFile::parse(gguf_bytes).expect("parse gguf");
        let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
        let mut engine = InferenceEngine::from_model_with_tier(model, tier, greedy_params(), 42);
        engine.generate(prompt, n).expect("generate")
    }

    /// Returns `true` when a CUDA device is genuinely accessible from this
    /// process — never inferred from the feature/target combination alone,
    /// so a build with `native-cuda` compiled in but no physical device
    /// still self-skips honestly.
    fn cuda_available() -> bool {
        CudaGraph::global().is_ok()
    }

    /// The format probe: the stored GGUF type of the LM head of the file at
    /// `path`, read before anything loads the file — `output.weight`, or
    /// `token_embd.weight` for a tied model (the loader's own rule), `None`
    /// when the file has neither. The file is memory-mapped and only its
    /// header is parsed, so probing a multi-GB GGUF stays cheap. `Err` when
    /// the file cannot be opened or mapped (a skip, as an unreadable file
    /// always was); a file that does not parse as GGUF panics, as [`run`]
    /// would on it.
    fn lm_head_type(path: &Path) -> Result<Option<GgufTensorType>, String> {
        let mmap = mmap_gguf_file(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
        let gguf = GgufFile::parse(&mmap)
            .unwrap_or_else(|e| panic!("cuda_ternary_forward_parity: parse {path:?}: {e}"));
        Ok(gguf
            .tensors
            .get("output.weight")
            .or_else(|| gguf.tensors.get("token_embd.weight"))
            .map(|info| info.tensor_type))
    }

    /// The stored LM-head type, as the chosen-path line prints it.
    fn lm_head_label(head: Option<GgufTensorType>) -> String {
        match head {
            Some(t) => format!("LM head stored as {t}, ggml id {}", t.wire_id()),
            None => "no output.weight or token_embd.weight tensor".to_owned(),
        }
    }

    /// The `skip:` reason for a file whose LM head is stored as `PQ2_0`.
    fn pq2_0_skip_reason(path: &Path) -> String {
        format!(
            "{path:?} is stored as PQ2_0 (ggml id 142), which the qwen3 runtime cannot load \
             (BonsaiModel::from_gguf refuses a PQ2_0 LM head); re-convert it with the 0.2.4 \
             converter (`oxibonsai convert --quant tq2_0_g128` at or after 3c9993a writes \
             TQ2_0_g128, ggml id 42), or point OXI_MODEL / OXIBONSAI_MODELS_DIR at an id-42 \
             file such as {TERNARY_FILE_NATIVE}"
        )
    }

    /// The ternary GGUF to drive, chosen and probed with [`lm_head_type`]
    /// before anything loads it (module docs, "The model file"); the chosen
    /// path is printed. `OXI_MODEL` is used as given; without it,
    /// [`TERNARY_FILE_NATIVE`] is tried when [`TERNARY_FILE`] is absent or
    /// stored as `PQ2_0`. `Err` carries the reason for the caller's `skip:`
    /// line: no file found, a file that cannot be opened, or a chosen file
    /// stored as `PQ2_0` (ggml id 142), which the `qwen3` runtime cannot
    /// load and this test does not re-encode.
    fn resolve_model(test: &str) -> Result<PathBuf, String> {
        if let Some(path) = std::env::var("OXI_MODEL")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
        {
            let head = lm_head_type(&path)?;
            if head == Some(GgufTensorType::PQ2_0) {
                return Err(pq2_0_skip_reason(&path));
            }
            eprintln!(
                "{test}: model {path:?} (OXI_MODEL; {})",
                lm_head_label(head)
            );
            return Ok(path);
        }
        let plain = find_model(TERNARY_FILE);
        if let Some(path) = &plain {
            let head = lm_head_type(path)?;
            if head != Some(GgufTensorType::PQ2_0) {
                eprintln!("{test}: model {path:?} ({})", lm_head_label(head));
                return Ok(path.clone());
            }
            eprintln!(
                "{test}: {path:?} is stored as PQ2_0 (ggml id 142), which the qwen3 runtime \
                 cannot load; looking for {TERNARY_FILE_NATIVE} instead"
            );
        }
        let Some(native) = find_model(TERNARY_FILE_NATIVE) else {
            return Err(match plain {
                Some(path) => format!(
                    "{} (no {TERNARY_FILE_NATIVE} under {:?} to fall back to)",
                    pq2_0_skip_reason(&path),
                    models_dir()
                ),
                None => format!(
                    "set OXI_MODEL or OXIBONSAI_MODELS_DIR to a real ternary GGUF (neither \
                     {TERNARY_FILE} nor {TERNARY_FILE_NATIVE} found under {:?})",
                    models_dir()
                ),
            });
        };
        let head = lm_head_type(&native)?;
        if head == Some(GgufTensorType::PQ2_0) {
            return Err(pq2_0_skip_reason(&native));
        }
        eprintln!("{test}: model {native:?} ({})", lm_head_label(head));
        Ok(native)
    }

    /// The file [`resolve_model`] chose, read into memory; `Err` carries the
    /// reason for the caller's `skip:` line.
    fn read_model(test: &str) -> Result<Vec<u8>, String> {
        let path = resolve_model(test)?;
        std::fs::read(&path).map_err(|e| format!("cannot read {path:?}: {e}"))
    }

    /// Greedy CPU-reference vs CUDA-Gpu output must match on the real ternary
    /// model. The prompt length is the regression trigger (the bug appeared
    /// above ~16 tokens); override the count via `OXI_PROMPT_LEN` (default
    /// 20, i.e. >16).
    pub(super) fn real_ternary_cpu_cuda_parity(test: &str) {
        if !cuda_available() {
            eprintln!("skip: {test} — no CUDA device accessible on this host");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
        let gguf = match read_model(test) {
            Ok(bytes) => bytes,
            Err(why) => {
                eprintln!("skip: {test} — {why}");
                record_skipped(Capability::CudaHardware, test);
                return;
            }
        };
        let start = Instant::now();
        let plen: usize = std::env::var("OXI_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let prompt: Vec<u32> = (0..plen as u32).map(|i| 1000 + i * 37).collect();
        let n = 4;
        let cpu = run(&gguf, KernelTier::Reference, &prompt, n);
        let cuda = run(&gguf, KernelTier::Gpu, &prompt, n);
        eprintln!("── real model (plen={plen}) ──\n  cpu  = {cpu:?}\n  cuda = {cuda:?}");
        let first = cpu
            .iter()
            .zip(cuda.iter())
            .position(|(a, b)| a != b)
            .map(|i| i as i32)
            .unwrap_or(-1);
        assert_eq!(
            cpu, cuda,
            "real ternary CPU vs CUDA diverge (first diff at index {first})"
        );
        record_executed_timed(Capability::CudaHardware, test, start.elapsed());
    }

    /// Diagnostic: magnitude of the CPU-vs-CUDA logit divergence at the
    /// first decode step. Benign FP non-associativity is ~1e-2; the
    /// KV-handoff bug produced ~7.
    pub(super) fn real_ternary_decode_logit_delta(test: &str) {
        if !cuda_available() {
            eprintln!("skip: {test} — no CUDA device accessible on this host");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
        let gguf = match read_model(test) {
            Ok(bytes) => bytes,
            Err(why) => {
                eprintln!("skip: {test} — {why}");
                record_skipped(Capability::CudaHardware, test);
                return;
            }
        };
        let start = Instant::now();
        let plen: usize = std::env::var("OXI_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let prompt: Vec<u32> = (0..plen as u32).map(|i| 1000 + i * 37).collect();

        let drive = |tier: KernelTier| -> (Vec<f32>, Vec<f32>) {
            let parsed = GgufFile::parse(&gguf).expect("parse");
            let mut model = BonsaiModel::from_gguf(&parsed, MAX_SEQ).expect("from_gguf");
            let kernel = KernelDispatcher::with_tier(tier);
            let p = model.forward_prefill(&prompt, 0, &kernel).expect("prefill");
            let tok0 = p
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap();
            let d = model.forward(tok0, plen, &kernel).expect("decode");
            (p, d)
        };
        let (cpu_p, cpu_d) = drive(KernelTier::Reference);
        let (cuda_p, cuda_d) = drive(KernelTier::Gpu);
        let maxd = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        eprintln!("── real logit delta (plen={plen}) ──");
        eprintln!("  PREFILL max|Δ|={:.4}", maxd(&cpu_p, &cuda_p));
        eprintln!("  DECODE  max|Δ|={:.4}", maxd(&cpu_d, &cuda_d));
        // Decode must stay within FP-noise of the CPU reference.
        assert!(
            maxd(&cpu_d, &cuda_d) < 0.5,
            "decode logit divergence too large: {:.4}",
            maxd(&cpu_d, &cuda_d)
        );
        record_executed_timed(Capability::CudaHardware, test, start.elapsed());
    }
}

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
fn real_ternary_cpu_cuda_parity() {
    cuda_impl::real_ternary_cpu_cuda_parity(REAL_TERNARY_CPU_CUDA_PARITY);
}

#[cfg(not(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
)))]
#[test]
fn real_ternary_cpu_cuda_parity() {
    eprintln!(
        "skip: {REAL_TERNARY_CPU_CUDA_PARITY} — this build has no CUDA target (needs \
         --features native-cuda on Linux/Windows)"
    );
    record_skipped(Capability::CudaHardware, REAL_TERNARY_CPU_CUDA_PARITY);
}

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[test]
fn real_ternary_decode_logit_delta() {
    cuda_impl::real_ternary_decode_logit_delta(REAL_TERNARY_DECODE_LOGIT_DELTA);
}

#[cfg(not(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
)))]
#[test]
fn real_ternary_decode_logit_delta() {
    eprintln!(
        "skip: {REAL_TERNARY_DECODE_LOGIT_DELTA} — this build has no CUDA target (needs \
         --features native-cuda on Linux/Windows)"
    );
    record_skipped(Capability::CudaHardware, REAL_TERNARY_DECODE_LOGIT_DELTA);
}
