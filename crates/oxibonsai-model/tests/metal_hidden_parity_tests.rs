//! Real-model gate of the head-free Metal hidden-state prefill (the dense
//! embedding pass on the GPU tier): `BonsaiModel::forward_hidden` on a Metal
//! engine against the per-token reference `forward_hidden_sequential`, on
//! the shipped `Ternary-Bonsai-1.7B`, `Ternary-Bonsai-8B` and `Bonsai-8B`
//! GGUFs, for 10-, 300- and 2000-token inputs.
//!
//! # What is asserted, per model and input
//!
//! * the Metal route really ran: `forward_hidden`'s rows are bit-identical to
//!   a direct `try_metal_forward_hidden` call (the batched CPU pass computes
//!   the same rows in another arithmetic order and would not be);
//! * every row agrees with the per-token reference to cosine >= 0.999, and
//!   the mean-pooled, L2-normalised embedding to cosine >= 0.9999;
//! * the pass leaves the model's MET-05 device-KV latch clear.
//!
//! The reference is computed **once per model**, over the 2000-token input:
//! its rows are causal (row `i` depends on tokens `0..=i` only) and the
//! shorter inputs are prefixes of the longer one, so the first `n` rows of
//! that one pass are the reference of the `n`-token input. It runs on the
//! GPU-tier dispatcher with every block's weights uploaded, so each of its
//! per-token block steps binds GPU-resident projections — the fastest the
//! per-token host-KV loop gets on this hardware. Measured per token over the
//! 2000-token input: Bonsai-8B 60 ms with its handles uploaded, against
//! 867 ms for Ternary-Bonsai-8B and 234 ms for Ternary-Bonsai-1.7B without
//! (an all-ternary engine skips that upload, so its per-block GEMVs fall back
//! to the CPU), and ~0.7 s for the 1.7B on the NEON tier under load.
//!
//! # Model files
//!
//! `OXI_MODEL` names one GGUF to gate; without it the three files are looked
//! up under the testkit models directory (`OXIBONSAI_MODELS_DIR`, else the
//! workspace `models/`). The test self-skips — writing an `executed: false`
//! record — when no file is found. Run it in `--release`, one real-model
//! process at a time.
//!
//! # Capability record
//!
//! Every run appends one `metal-hidden` line to the capability report
//! (`executed: true` only after every assertion above passed on every
//! resolved model).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;
use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;

const TEST_NAME: &str =
    "oxibonsai-model::metal_hidden_parity_tests::metal_hidden_prefill_matches_the_sequential_reference_on_real_models";

/// The legacy dense models this gate covers.
const MODEL_FILES: [&str; 3] = [
    "Ternary-Bonsai-1.7B.gguf",
    "Ternary-Bonsai-8B.gguf",
    "Bonsai-8B.gguf",
];

/// Input lengths, in tokens.
const LENGTHS: [usize; 3] = [10, 300, 2000];

/// Context the models are loaded with: the longest input plus headroom.
const MAX_SEQ: usize = 2048;

/// `"The quick brown fox jumps over the lazy dog."` (Qwen3 tokenizer).
const TEXT_FOX: [u32; 10] = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];

/// `"Embeddings map a piece of text to a vector, so that passages with
/// similar meanings land near each other."`
const TEXT_EMBEDDINGS: [u32; 22] = [
    25486, 24602, 2415, 264, 6573, 315, 1467, 311, 264, 4621, 11, 773, 429, 46769, 448, 4428,
    49700, 4268, 3143, 1817, 1008, 13,
];

/// `"Rivers carve valleys over millions of years. …"` (104 tokens).
const TEXT_RIVERS: [u32; 104] = [
    49, 1945, 79637, 85397, 916, 11728, 315, 1635, 13, 9959, 51207, 304, 279, 1550, 8166, 11,
    85681, 4628, 438, 432, 6560, 1412, 11, 323, 23377, 9278, 323, 9798, 429, 39236, 279, 14796,
    2721, 19117, 448, 1449, 17726, 13, 10967, 279, 4268, 51039, 724, 11, 279, 1482, 69170, 323,
    21025, 1181, 2795, 11, 4752, 69125, 77366, 323, 90587, 429, 614, 22313, 9720, 2474, 279, 1156,
    23429, 82, 13, 48696, 1431, 1936, 82525, 323, 55489, 288, 311, 81823, 429, 4802, 11, 11133,
    279, 2310, 10775, 315, 17726, 323, 42801, 369, 24020, 3015, 323, 17728, 11, 323, 279, 35517,
    4226, 553, 7218, 862, 58032, 14696, 770, 13,
];

/// `len` tokens of real English text: the fox sentence alone for 10, else
/// the three passages cycled.
fn text_tokens(len: usize) -> Vec<u32> {
    if len <= TEXT_FOX.len() {
        return TEXT_FOX[..len].to_vec();
    }
    TEXT_FOX
        .iter()
        .chain(TEXT_EMBEDDINGS.iter())
        .chain(TEXT_RIVERS.iter())
        .copied()
        .cycle()
        .take(len)
        .collect()
}

/// The GGUFs to gate: `$OXI_MODEL`, else every [`MODEL_FILES`] entry the
/// testkit resolver finds.
fn model_paths() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("OXI_MODEL").filter(|p| !p.is_empty()) {
        return vec![PathBuf::from(path)];
    }
    MODEL_FILES
        .iter()
        .filter_map(|file| oxibonsai_testkit::workspace::find_model(file))
        .collect()
}

/// Record the gate's `metal-hidden` capability evidence — executed with the
/// gate's wall time, or skipped — through the testkit's shared writer.
fn record_metal_hidden(executed: bool, duration: std::time::Duration) {
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
    if executed {
        record_executed_timed(Capability::MetalHidden, TEST_NAME, duration);
    } else {
        record_skipped(Capability::MetalHidden, TEST_NAME);
    }
    eprintln!(
        "capability metal-hidden executed={executed} test={TEST_NAME} duration_ms={}",
        duration.as_millis()
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

/// Mean-pool `rows` (`[n x hidden]`) and scale to unit length.
fn pool(rows: &[f32], hidden: usize) -> Vec<f32> {
    let n = rows.len() / hidden;
    let mut pooled = vec![0.0f64; hidden];
    for row in rows.chunks_exact(hidden) {
        for (acc, v) in pooled.iter_mut().zip(row) {
            *acc += f64::from(*v);
        }
    }
    let norm = pooled.iter().map(|v| v * v).sum::<f64>().sqrt() / n as f64;
    pooled
        .iter()
        .map(|v| (v / n as f64 / norm.max(f64::MIN_POSITIVE)) as f32)
        .collect()
}

#[test]
fn metal_hidden_prefill_matches_the_sequential_reference_on_real_models() {
    let started = Instant::now();
    let paths = model_paths();
    if paths.is_empty() {
        eprintln!("{TEST_NAME}: no real model found (set OXI_MODEL) -- skipping");
        record_metal_hidden(false, started.elapsed());
        return;
    }
    let gpu = KernelDispatcher::auto_detect();
    assert_eq!(
        gpu.tier(),
        KernelTier::Gpu,
        "the Metal hidden prefill needs the GPU tier on this host"
    );
    let longest = LENGTHS.iter().copied().max().unwrap_or(0);
    let all_tokens = text_tokens(longest);
    for path in &paths {
        let mmap = mmap_gguf_file(path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
        let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
        let mut model =
            BonsaiModel::from_gguf(&gguf, MAX_SEQ).unwrap_or_else(|e| panic!("load {path:?}: {e}"));
        // Every block's weights resident on the GPU: the sequential
        // reference's per-token block steps bind them (the Metal hidden route
        // binds the fused weight cache below either way).
        model.upload_weights_to_gpu(&gpu);
        model
            .get_or_create_gpu_cache()
            .unwrap_or_else(|e| panic!("{path:?}: fused weight cache: {e}"));
        let hidden = model.hidden_size();
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let ref_started = Instant::now();
        let reference_all = model
            .forward_hidden_sequential(&all_tokens, &gpu)
            .unwrap_or_else(|e| panic!("{name}: sequential reference: {e}"));
        let ref_secs = ref_started.elapsed().as_secs_f64();
        assert_eq!(reference_all.len(), longest * hidden);
        eprintln!(
            "{name}: sequential reference ({:?}) over {longest} tokens: {ref_secs:.1} s \
             ({:.1} ms/token)",
            gpu.tier(),
            ref_secs * 1e3 / longest as f64
        );
        for len in LENGTHS {
            let tokens = text_tokens(len);
            assert_eq!(
                tokens[..],
                all_tokens[..len],
                "every input is a prefix of the longest one"
            );
            let metal_started = Instant::now();
            let rows = model
                .forward_hidden(&tokens, &gpu)
                .unwrap_or_else(|e| panic!("{name} {len}: forward_hidden: {e}"));
            let metal_secs = metal_started.elapsed().as_secs_f64();
            let direct = model
                .try_metal_forward_hidden(&tokens, &gpu)
                .unwrap_or_else(|e| panic!("{name} {len}: Metal hidden prefill: {e}"))
                .unwrap_or_else(|| panic!("{name} {len}: the Metal route declined"));
            assert!(
                rows.iter()
                    .zip(&direct)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} {len}: forward_hidden did not take the Metal route"
            );
            assert!(!model.gpu_path_active(), "{name} {len}: MET-05 latch set");
            let reference = &reference_all[..len * hidden];
            assert_eq!(rows.len(), len * hidden);
            assert!(rows.iter().all(|v| v.is_finite()), "{name} {len}: finite");
            let worst_row = rows
                .chunks_exact(hidden)
                .zip(reference.chunks_exact(hidden))
                .map(|(a, b)| cosine(a, b))
                .fold(f64::INFINITY, f64::min);
            let pooled = cosine(&pool(&rows, hidden), &pool(reference, hidden));
            eprintln!(
                "{name} {len} tokens: pooled cos = {pooled:.9}, worst row cos = {worst_row:.9}; \
                 metal {metal_secs:.3} s"
            );
            assert!(
                worst_row >= 0.999,
                "{name} {len}: a row diverged from the sequential reference: cos {worst_row}"
            );
            assert!(
                pooled >= 0.9999,
                "{name} {len}: pooled embedding diverged: cos {pooled}"
            );
        }
    }
    record_metal_hidden(true, started.elapsed());
}
