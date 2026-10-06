//! The Metal runner's rows prefill and M-RoPE bookkeeping against the CPU
//! [`HybridModel`] on the synthetic `qwen35` fixture: text rows through the
//! rows path equal the token path bit for bit, a prompt mixing text and
//! image rows tracks the CPU's rows prefill layer by layer, decode after an
//! image rotates at the offset position, snapshots carry the offsets, image
//! rows reach layer 0 unrotated, and the raster order the image rows arrive
//! in is what makes sequence-order causality the reference's 2-D mask.

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::gpu_backend::metal_graph::MetalGraph;
use oxibonsai_kernels::{cpu_kernel_tier, KernelDispatcher};

use super::*;
use crate::hybrid::tests_support::{synthetic_gguf, FixtureOptions, FixtureShape};
use crate::hybrid::vision_prefill::{AssembledPrompt, PromptPiece};
use crate::vision::mrope::{image_positions, text_positions};
use crate::vision::patch_embed::window_order;
use crate::vision::GridSize;

/// KV window of the fixture models.
const WINDOW: usize = 128;

fn metal_available() -> bool {
    match MetalGraph::global() {
        Ok(_) => true,
        Err(MetalGraphError::DeviceNotFound) => false,
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

fn fixture() -> Vec<u8> {
    synthetic_gguf(FixtureShape::default(), FixtureOptions::default())
}

fn cpu_model<'a>(gguf: &'a GgufFile<'a>) -> HybridModel<'a> {
    let config = HybridModel::config_from_gguf(gguf).expect("fixture config");
    let kernel = Arc::new(KernelDispatcher::with_tier(cpu_kernel_tier()));
    HybridModel::from_gguf_with(gguf, config, WINDOW, &kernel).expect("fixture loads")
}

fn runner<'a>(model: &HybridModel<'a>, mode: Qwen35PrefillMode) -> HybridMetalRunner<'a> {
    let mut runner = HybridMetalRunner::new(model).expect("runner builds");
    runner.set_prefill_mode(mode);
    runner
}

/// Deterministic, clearly-not-an-embedding rows standing in for a vision
/// tower's output.
fn image_rows(n: usize, hidden: usize, seed: u32) -> Vec<f32> {
    (0..n * hidden)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x >> 8) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

fn worst_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// One logit row of the runner against the CPU's, held to the tolerance of
/// the runner's CPU-tracking tests (cosine ≥ 0.999999, worst relative error
/// ≤ 1e-4), with the same greedy token unless the CPU's own top-2 is a
/// near-tie.
fn check_step(label: &str, gpu: &[f32], cpu: &[f32]) {
    let (cos, rel) = (cosine(gpu, cpu), worst_rel(gpu, cpu));
    assert!(cos >= 0.999_999, "{label}: logit cosine {cos}");
    assert!(rel <= 1e-4, "{label}: worst relative error {rel:e}");
    let (g, c) = (argmax(gpu), argmax(cpu));
    if g != c {
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            cpu[c] - cpu[g] <= 1e-4 * scale,
            "{label}: greedy token {g} (GPU) vs {c} (CPU)"
        );
    }
}

/// `lead` text, one `grid` image, `tail` text, assembled from position 0 by
/// the CPU model (text rows embedded, image rows verbatim).
fn mixed_prompt(
    model: &HybridModel<'_>,
    lead: &[u32],
    image: &[f32],
    grid: GridSize,
    tail: &[u32],
) -> AssembledPrompt {
    let pieces = [
        PromptPiece::Tokens(lead),
        PromptPiece::Image { rows: image, grid },
        PromptPiece::Tokens(tail),
    ];
    model.assemble_prompt(&pieces, 0).expect("assemble")
}

/// A text-only prompt through the rows path (the model's embedded rows,
/// text positions) is bit-identical to the token prefill in both modes:
/// per-row angles of text positions are the resident table's rows.
#[test]
fn text_rows_prefill_is_bitwise_the_token_prefill_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let model = cpu_model(&gguf);
    let hidden = model.config().base.hidden_size;
    let tokens: Vec<u32> = (0..12u32).map(|i| (i * 41 + 3) % 500).collect();
    let mut rows = vec![0.0f32; tokens.len() * hidden];
    model.embed_token_rows(&tokens, &mut rows).expect("embed");
    let positions = text_positions(0, tokens.len()).expect("positions");
    for mode in [Qwen35PrefillMode::Sequential, Qwen35PrefillMode::Batched] {
        let (mut by_ids, mut by_rows) = (runner(&model, mode), runner(&model, mode));
        let vocab = by_ids.vocab_size();
        let (mut want, mut got) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
        by_ids
            .forward_prefill(&tokens, 0, &mut want)
            .expect("token prefill");
        by_rows
            .forward_prefill_rows(&rows, &positions, 0, Some(&mut got))
            .expect("rows prefill");
        assert_eq!(bits(&got), bits(&want), "{mode}: rows vs tokens");
        assert_eq!(by_rows.rope_delta(), 0);
        assert_eq!(by_rows.token_count(), tokens.len());
        let pos = tokens.len();
        let a = by_ids.forward(7, pos).expect("decode");
        let b = by_rows.forward(7, pos).expect("decode");
        assert_eq!(bits(&a), bits(&b), "{mode}: the next decode step");
    }
}

/// Text, an image and more text, prefilled as rows on the runner and on
/// the CPU model: what entered layer 0 is identical, every layer's residual
/// stream tracks the CPU's (cosine ≥ 0.99999) and the last row's logits are
/// within the CPU-tracking tolerance, in both modes; both leave the same
/// M-RoPE offset.
#[test]
fn mixed_text_and_image_rows_track_the_cpu_rows_prefill_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let grid = GridSize { h: 3, w: 4 };
    let image = image_rows(grid.n_tokens(), hidden, 5);
    let prompt = mixed_prompt(&cpu, &[4, 8, 15, 16, 23], &image, grid, &[42, 7, 9, 11]);
    assert_eq!(prompt.len(), 5 + 12 + 4);
    let vocab = cpu.config().base.vocab_size;
    let mut cl = vec![0.0f32; vocab];
    cpu.reset();
    let dump = cpu
        .forward_prefill_rows_with_dump(&prompt.rows, &prompt.positions, 0, Some(&mut cl))
        .expect("cpu rows prefill");
    let cpu_delta = cpu.rope_delta();
    assert_eq!(cpu_delta, 12 - 4);
    for mode in [Qwen35PrefillMode::Sequential, Qwen35PrefillMode::Batched] {
        let mut gpu = runner(&cpu, mode);
        let got = gpu
            .forward_prefill_rows_with_dump(&prompt.rows, &prompt.positions, 0)
            .expect("metal rows prefill");
        assert_eq!(
            bits(&got.embedding),
            bits(&dump.embedding),
            "{mode}: layer-0 input"
        );
        assert_eq!(got.layers.len(), dump.layers.len());
        for (layer, (g, c)) in got.layers.iter().zip(&dump.layers).enumerate() {
            let cos = cosine(g, c);
            assert!(
                cos >= 0.999_99,
                "{mode} layer {layer}: residual cosine {cos}"
            );
        }
        check_step(&format!("{mode} rows prefill"), &got.logits, &cl);
        assert_eq!(gpu.rope_delta(), cpu_delta, "{mode}");
        assert_eq!(gpu.token_count(), prompt.len());

        // The chunked entry point computes what the one-call dump did.
        let mut chunked = runner(&cpu, mode);
        chunked.set_max_batch(8).expect("smaller batch");
        let mut gl = vec![0.0f32; vocab];
        chunked
            .forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut gl))
            .expect("chunked rows prefill");
        check_step(&format!("{mode} chunked rows prefill"), &gl, &cl);
        assert_eq!(chunked.rope_delta(), cpu_delta);
    }
}

/// Decode after an image: the runner and the CPU model, fed the CPU's
/// greedy tokens, keep the same offset and matching logits step after step;
/// and on the runner a decoded token equals the same token placed as the
/// prompt's last row at its explicit text position (bitwise, in the
/// sequential mode), which differs from rotating it at its sequence
/// position.
#[test]
fn decode_after_an_image_rotates_at_the_offset_position_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let vocab = cpu.config().base.vocab_size;
    let grid = GridSize { h: 3, w: 2 };
    let image = image_rows(grid.n_tokens(), hidden, 11);
    let (lead, tail) = ([9u32, 4, 2], [8u32, 31]);
    let prompt = mixed_prompt(&cpu, &lead, &image, grid, &tail);
    let n = prompt.len();

    // Lockstep decode against the CPU, both modes.
    for mode in [Qwen35PrefillMode::Sequential, Qwen35PrefillMode::Batched] {
        let mut gpu = runner(&cpu, mode);
        let (mut cl, mut gl) = (vec![0.0f32; vocab], vec![0.0f32; vocab]);
        cpu.reset();
        cpu.forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut cl))
            .expect("cpu prefill");
        gpu.forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut gl))
            .expect("metal prefill");
        assert_eq!(gpu.rope_delta(), 6 - 3);
        assert_eq!(gpu.rope_delta(), cpu.rope_delta());
        for step in 0..6usize {
            let token = u32::try_from(argmax(&cl)).expect("token id");
            let pos = n + step;
            cpu.forward(token, pos, &mut cl).expect("cpu decode");
            gpu.forward_into(token, pos, &mut gl).expect("metal decode");
            check_step(&format!("{mode} decode step {step}"), &gl, &cl);
            assert_eq!(gpu.rope_delta(), cpu.rope_delta());
        }
    }

    // The decoded token rotates at its offset position, not its sequence one.
    let next = 123u32;
    let mut decoded = runner(&cpu, Qwen35PrefillMode::Sequential);
    decoded
        .forward_prefill_rows(&prompt.rows, &prompt.positions, 0, None)
        .expect("prefill");
    let after = decoded.forward(next, n).expect("decode");
    let with_next = mixed_prompt(&cpu, &lead, &image, grid, &[8, 31, next]);
    let mut in_prompt = runner(&cpu, Qwen35PrefillMode::Sequential);
    let mut want = vec![0.0f32; vocab];
    in_prompt
        .forward_prefill_rows(&with_next.rows, &with_next.positions, 0, Some(&mut want))
        .expect("prefill");
    assert_eq!(
        bits(&after),
        bits(&want),
        "decode vs the same token as a row"
    );
    let mut forgot_positions = with_next.positions.clone();
    let last = forgot_positions.len() - 1;
    forgot_positions[last] = MropePos::text(u32::try_from(last).expect("fits"));
    let mut forgot = runner(&cpu, Qwen35PrefillMode::Sequential);
    let mut wrong = vec![0.0f32; vocab];
    forgot
        .forward_prefill_rows(&with_next.rows, &forgot_positions, 0, Some(&mut wrong))
        .expect("prefill");
    assert_ne!(
        bits(&wrong),
        bits(&want),
        "the offset must reach the attention"
    );
}

/// A snapshot taken after an image restores the recurrent state, the
/// position and the offset: eight decode steps replay bit for bit.
#[test]
fn a_snapshot_across_an_image_replays_bitwise_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let grid = GridSize { h: 2, w: 3 };
    let image = image_rows(grid.n_tokens(), hidden, 3);
    let prompt = mixed_prompt(&cpu, &[1, 2, 3, 4], &image, grid, &[5, 6]);
    let mut gpu = runner(&cpu, Qwen35PrefillMode::Batched);
    let vocab = gpu.vocab_size();
    let mut logits = vec![0.0f32; vocab];
    gpu.forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut logits))
        .expect("prefill");
    let snapshot = gpu.snapshot_state().expect("snapshot");
    assert_eq!(snapshot.rope_delta(), 6 - 3);
    assert_eq!(snapshot.position(), prompt.len());
    let decode = |gpu: &mut HybridMetalRunner<'_>, mut logits: Vec<f32>| {
        let mut rows = Vec::new();
        for step in 0..8usize {
            let token = u32::try_from(argmax(&logits)).expect("token id");
            gpu.forward_into(token, prompt.len() + step, &mut logits)
                .expect("decode");
            rows.push(bits(&logits));
        }
        rows
    };
    let reference = decode(&mut gpu, logits.clone());
    gpu.reset();
    assert_eq!(gpu.rope_delta(), 0, "a reset clears the offset");
    gpu.restore_state(&snapshot).expect("restore");
    assert_eq!(gpu.rope_delta(), 6 - 3, "the restore brings it back");
    assert_eq!(decode(&mut gpu, logits), reference);
}

/// Continue `gpu` at `k` with a text tail and one decode step, returning
/// both logit rows (long enough for the decode step to sit past where an
/// abandoned continuation ended).
fn continue_text(gpu: &mut HybridMetalRunner<'_>, k: usize) -> (Vec<f32>, Vec<f32>) {
    let tail = [8u32, 31, 12, 40, 3, 77, 21, 6, 90, 14];
    let mut logits = vec![0.0f32; gpu.vocab_size()];
    gpu.forward_prefill(&tail, k, &mut logits)
        .expect("text continuation");
    let next = gpu.forward(123, k + tail.len()).expect("decode");
    (logits, next)
}

/// A text sequence snapshotted at `k`, continued by an image that is then
/// rolled back, continues at the offset in force at `k` (none) —
/// bit-identical to a runner that never saw the image (the CPU model's
/// rollback test, on the runner).
#[test]
fn a_rolled_back_text_sequence_forgets_the_abandoned_image_offset_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let lead = [9u32, 4, 2, 7];
    let k = lead.len();
    let mut gpu = runner(&cpu, Qwen35PrefillMode::Batched);
    let mut reference = runner(&cpu, Qwen35PrefillMode::Batched);
    let mut scratch = vec![0.0f32; gpu.vocab_size()];
    gpu.forward_prefill(&lead, 0, &mut scratch)
        .expect("prefill");
    let snapshot = gpu.snapshot_state().expect("snapshot");
    let grid = GridSize { h: 3, w: 2 };
    let image = image_rows(grid.n_tokens(), hidden, 13);
    let continuation = cpu
        .assemble_prompt_at(
            &[
                PromptPiece::Image { rows: &image, grid },
                PromptPiece::Tokens(&[5]),
            ],
            gpu.rope_start_for(k).expect("rope start"),
        )
        .expect("assemble");
    gpu.forward_prefill_rows(&continuation.rows, &continuation.positions, k, None)
        .expect("image continuation");
    assert_eq!(gpu.rope_delta(), 6 - 3);
    gpu.restore_state(&snapshot).expect("restore");
    assert_eq!(gpu.rope_delta(), 0, "the offset in force at k");
    let (got, got_next) = continue_text(&mut gpu, k);

    reference
        .forward_prefill(&lead, 0, &mut scratch)
        .expect("prefill");
    let (want, want_next) = continue_text(&mut reference, k);
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(bits(&got_next), bits(&want_next));
}

/// A sequence whose own image precedes the snapshot keeps that image's
/// offset through the roll-back — neither zero nor the abandoned second
/// image's larger one.
#[test]
fn a_rolled_back_sequence_keeps_the_offset_of_its_own_image_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let first = GridSize { h: 2, w: 2 };
    let image = image_rows(first.n_tokens(), hidden, 17);
    let lead = mixed_prompt(&cpu, &[1, 2], &image, first, &[3]);
    let k = lead.len();
    let mut gpu = runner(&cpu, Qwen35PrefillMode::Batched);
    let mut reference = runner(&cpu, Qwen35PrefillMode::Batched);
    gpu.forward_prefill_rows(&lead.rows, &lead.positions, 0, None)
        .expect("prefill");
    assert_eq!(gpu.rope_delta(), 4 - 2);
    let snapshot = gpu.snapshot_state().expect("snapshot");
    let second = GridSize { h: 3, w: 3 };
    let other = image_rows(second.n_tokens(), hidden, 19);
    let more = cpu
        .assemble_prompt_at(
            &[PromptPiece::Image {
                rows: &other,
                grid: second,
            }],
            gpu.rope_start_for(k).expect("rope start"),
        )
        .expect("assemble");
    gpu.forward_prefill_rows(&more.rows, &more.positions, k, None)
        .expect("second image");
    assert_eq!(gpu.rope_delta(), 2 + (9 - 3));
    gpu.restore_state(&snapshot).expect("restore");
    assert_eq!(gpu.rope_delta(), 2, "the first image's offset");
    let (got, got_next) = continue_text(&mut gpu, k);

    reference
        .forward_prefill_rows(&lead.rows, &lead.positions, 0, None)
        .expect("prefill");
    let (want, want_next) = continue_text(&mut reference, k);
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(bits(&got_next), bits(&want_next));
}

/// The Hadamard bypass (design §3.5 / §6.2) on the runner: an image row
/// reaches layer 0 exactly as the tower produced it — read back from the
/// device — while a text row is the runner's inverse-transformed embedding;
/// and the bypass is load-bearing (either transform moves an image row).
#[test]
fn image_rows_reach_layer_0_unrotated_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let grid = GridSize { h: 2, w: 2 };
    let image = image_rows(grid.n_tokens(), hidden, 7);
    let (lead, tail) = ([5u32, 6], [7u32]);
    let prompt = mixed_prompt(&cpu, &lead, &image, grid, &tail);
    let mut gpu = runner(&cpu, Qwen35PrefillMode::Batched);
    let dump = gpu
        .forward_prefill_rows_with_dump(&prompt.rows, &prompt.positions, 0)
        .expect("rows prefill with dump");
    let row = |t: usize| &dump.embedding[t * hidden..(t + 1) * hidden];
    for r in 0..grid.n_tokens() {
        let original = &image[r * hidden..(r + 1) * hidden];
        assert_eq!(bits(row(2 + r)), bits(original), "image row {r}");
        for inverse in [false, true] {
            let moved = gpu.rotate(original, hidden, inverse).expect("rotate");
            assert_ne!(
                bits(&moved),
                bits(original),
                "image row {r} inverse {inverse}"
            );
        }
    }
    let text = gpu.embed(&[5, 6, 7]).expect("embed");
    for (t, i) in [(0usize, 0usize), (1, 1), (6, 2)] {
        assert_eq!(
            bits(row(t)),
            bits(&text[i * hidden..(i + 1) * hidden]),
            "text row {t}"
        );
    }
    assert_eq!(gpu.rope_delta(), 2);
}

/// The reference masks key `j` for query `i` at equal temporal position
/// when `(y_j > y_i) || (y_j == y_i && x_j > x_i)`; for an image whose rows
/// arrive row-major over its merged grid — the order `window_order` groups
/// patches in and `image_positions` numbers them — that is exactly
/// sequence-order causality (`j > i`), which is what the runner's attention
/// applies. Checked exhaustively on several grids, together with the two
/// orders the equivalence rests on.
#[test]
fn raster_order_makes_sequence_causality_the_2d_mask() {
    for (h, w) in [(1usize, 5usize), (2, 3), (3, 2), (4, 4), (6, 8)] {
        let grid = GridSize { h, w };
        let p0 = 7usize;
        let positions = image_positions(p0, grid).expect("positions");
        for (i, pi) in positions.iter().enumerate() {
            assert_eq!(
                (pi.h as usize, pi.w as usize),
                (p0 + i / w, p0 + i % w),
                "row {i} of a {h}x{w} image is row-major"
            );
            for (j, pj) in positions.iter().enumerate() {
                let masked =
                    pj.t > pi.t || (pj.t == pi.t && (pj.h > pi.h || (pj.h == pi.h && pj.w > pi.w)));
                assert_eq!(masked, j > i, "{h}x{w}: query {i}, key {j}");
            }
        }
        // The tower's merged rows: four consecutive patches per 2 x 2
        // window, windows row-major over the merged grid.
        let order = window_order(2 * h, 2 * w);
        for (j, window) in order.chunks(4).enumerate() {
            for &(py, px) in window {
                assert_eq!((py / 2, px / 2), (j / w, j % w), "merged row {j}");
            }
        }
    }
}

/// Every malformed rows prefill is refused before any GPU work and leaves
/// the sequence — position, offset, recurrent state — as it was.
#[test]
fn malformed_rows_are_refused_before_any_state_changes_bonsai2() {
    if !metal_available() {
        return;
    }
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = cpu_model(&gguf);
    let hidden = cpu.config().base.hidden_size;
    let mut gpu = runner(&cpu, Qwen35PrefillMode::Batched);
    let mut logits = vec![0.0f32; gpu.vocab_size()];
    gpu.forward_prefill(&[1, 2, 3], 0, &mut logits)
        .expect("prefill");
    let before = gpu.snapshot_state().expect("snapshot");

    gpu.forward_prefill_rows(&[], &[], 3, None)
        .expect("empty is a no-op");
    let positions = text_positions(3, 2).expect("positions");
    let err = gpu
        .forward_prefill_rows(&vec![0.0; hidden], &positions, 3, None)
        .expect_err("one row for two positions");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");
    let ahead = vec![MropePos::text(9)];
    let err = gpu
        .forward_prefill_rows(&vec![0.0; hidden], &ahead, 3, None)
        .expect_err("rotary position past the sequence");
    assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err}");
    let long = text_positions(3, WINDOW).expect("positions");
    let err = gpu
        .forward_prefill_rows(&vec![0.0; WINDOW * hidden], &long, 3, None)
        .expect_err("past the window");
    assert!(
        matches!(err, ModelError::PositionOutOfRange { .. }),
        "{err}"
    );
    let err = gpu
        .forward_prefill_rows(
            &vec![0.0; hidden],
            &text_positions(3, 1).expect("positions"),
            3,
            Some(&mut [0.0f32; 3][..]),
        )
        .expect_err("a short logit buffer");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }), "{err}");
    assert_eq!(gpu.snapshot_state().expect("snapshot"), before);
    assert_eq!(gpu.token_count(), 3);
    assert_eq!(gpu.rope_delta(), 0);
}

// ─────────────────────────────────────────────────────────────────────────
//  The process footprint across many prefill calls
// ─────────────────────────────────────────────────────────────────────────
//
// `-[MTLCommandQueue commandBuffer]` and
// `-[MTLCommandBuffer computeCommandEncoder]` return autoreleased objects. A
// thread with no autorelease pool keeps every one of them until it exits, so
// a long-lived server thread whose runner did not drain a pool per call
// would grow by a command buffer and its encoders on every prefill chunk.
// The test below holds the process footprint flat across many batched
// token-prefill calls and many rows-prefill calls (per-row 3-axis angles), in
// a child process of its own so that no other test of this binary allocates
// while it measures.

/// Set in the child process the footprint test re-runs itself in; the child
/// does the measuring.
const FOOTPRINT_CHILD_ENV: &str = "OXIBONSAI_RUNNER_PREFILL_FOOTPRINT_PROBE";

/// Prefix of the one line the child prints per measured phase.
const FOOTPRINT_REPORT: &str = "runner-prefill-footprint:";

/// Growth one measured phase may add. An undrained command buffer and its
/// encoder cost about 1.9 KiB per call, so a phase of [`FOOTPRINT_CALLS`]
/// undrained calls grows by several MiB (measured without the pool: about
/// 8.7 MiB a phase), where a pooled phase stays within a few hundred KiB
/// either way.
const FOOTPRINT_GROWTH_CEILING: i64 = 1 << 20;

/// Runner calls (one command buffer each) per measured phase.
const FOOTPRINT_CALLS: usize = 4800;

/// Rows per runner call: the GEMM threshold, so every call of the batched
/// mode runs its projections as tiled GEMMs.
const FOOTPRINT_CHUNK: usize =
    oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::Q35_GEMM_MIN_COLS;

/// Calls per prefill: each measured prefill is this many chunks.
const FOOTPRINT_CALLS_PER_PREFILL: usize = 4;

/// Measured phases: the batched token prefill and the rows prefill.
const FOOTPRINT_PHASES: usize = 2;

/// `TASK_VM_INFO`, the `task_info` flavor carrying `phys_footprint`.
const TASK_VM_INFO: u32 = 22;

/// `TASK_VM_INFO_REV1_COUNT`: `task_vm_info` up to and including
/// `phys_footprint`, in 32-bit words.
const TASK_VM_INFO_REV1_WORDS: usize = 38;

/// Word offset of `phys_footprint` (a 64-bit field) in `task_vm_info`.
const PHYS_FOOTPRINT_WORD: usize = 36;

// SAFETY (declarations): the Mach `task_info` call and the `mach_task_self_`
// port as `<mach/task.h>` / `<mach/mach_init.h>` declare them; both live in
// libSystem, which every macOS process links.
extern "C" {
    #[link_name = "mach_task_self_"]
    static MACH_TASK_SELF: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

/// This process's `phys_footprint` (`task_info(TASK_VM_INFO)`): dirty
/// anonymous memory, compressed pages and device allocations — what the
/// kernel's memory ledger charges the process, clean file pages excluded.
fn phys_footprint_bytes() -> i64 {
    let mut words = [0i32; TASK_VM_INFO_REV1_WORDS];
    let mut count = TASK_VM_INFO_REV1_WORDS as u32;
    // SAFETY: `words` is a caller-owned buffer of `count` 32-bit words, so
    // the kernel writes only inside it; `MACH_TASK_SELF` is initialised by
    // libSystem before `main`.
    let rc = unsafe { task_info(MACH_TASK_SELF, TASK_VM_INFO, words.as_mut_ptr(), &mut count) };
    assert_eq!(rc, 0, "task_info(TASK_VM_INFO) failed: kern_return_t {rc}");
    assert!(
        count as usize >= TASK_VM_INFO_REV1_WORDS,
        "task_info(TASK_VM_INFO) returned {count} words, fewer than rev1's {TASK_VM_INFO_REV1_WORDS}"
    );
    let low = u64::from(words[PHYS_FOOTPRINT_WORD] as u32);
    let high = u64::from(words[PHYS_FOOTPRINT_WORD + 1] as u32);
    let bytes = i64::try_from(low | (high << 32)).expect("footprint fits i64");
    assert!(
        bytes > 0,
        "task_info(TASK_VM_INFO) reported a zero footprint"
    );
    bytes
}

/// `bytes` as signed MiB.
fn mib(bytes: i64) -> String {
    format!("{:+.3} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// Run `warm` untimed units of `unit`, then `units` measured ones; print the
/// phase's report line and return the footprint growth over the measured
/// units.
fn measure_footprint(
    label: &str,
    calls: usize,
    warm: usize,
    units: usize,
    mut unit: impl FnMut(),
) -> i64 {
    for _ in 0..warm {
        unit();
    }
    let before = phys_footprint_bytes();
    for _ in 0..units {
        unit();
    }
    let after = phys_footprint_bytes();
    let growth = after - before;
    println!(
        "{FOOTPRINT_REPORT} {label}: {calls} runner calls grew the process footprint by {} \
         ({before} -> {after} bytes)",
        mib(growth)
    );
    growth
}

/// The measuring half of
/// [`metal_runner_prefill_does_not_grow_the_process_footprint_bonsai2`], run
/// in the child process: each phase warmed up first (pipelines, scratch,
/// allocator high-water marks), then [`FOOTPRINT_CALLS`] chunk-sized calls
/// measured.
fn prefill_footprint_probe() {
    let bytes = fixture();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let model = cpu_model(&gguf);
    let hidden = model.config().base.hidden_size;
    let mut gpu = runner(&model, Qwen35PrefillMode::Batched);
    gpu.set_max_batch(FOOTPRINT_CHUNK)
        .expect("chunk-sized calls");
    assert!(
        FOOTPRINT_CHUNK >= gpu.gemm_min_cols(),
        "every measured call takes the GEMM route"
    );
    let n = FOOTPRINT_CHUNK * FOOTPRINT_CALLS_PER_PREFILL;
    assert!(n <= WINDOW, "the prompt fits the fixture's window");
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 37 + 5) % 500).collect();
    let mut logits = vec![0.0f32; gpu.vocab_size()];
    let units = FOOTPRINT_CALLS / FOOTPRINT_CALLS_PER_PREFILL;
    let mut failures = Vec::new();

    // Batched token prefill: `FOOTPRINT_CALLS_PER_PREFILL` GEMM chunks a
    // prompt.
    let label = "batched token prefill";
    let growth = measure_footprint(label, FOOTPRINT_CALLS, 10, units, || {
        gpu.reset();
        gpu.forward_prefill(&tokens, 0, &mut logits)
            .expect("batched prefill");
        assert_eq!(gpu.token_count(), n);
    });
    if growth > FOOTPRINT_GROWTH_CEILING {
        failures.push(format!("{label}: {}", mib(growth)));
    }

    // Rows prefill: text, a 4 x 8 image grid and text, every row at its
    // 3-axis position.
    let grid = GridSize { h: 4, w: 8 };
    let image = image_rows(grid.n_tokens(), hidden, 17);
    let lead = n / 4;
    let tail = n - lead - grid.n_tokens();
    let prompt = mixed_prompt(
        &model,
        &tokens[..lead],
        &image,
        grid,
        &tokens[lead..lead + tail],
    );
    assert_eq!(prompt.len(), n);
    let label = "rows prefill";
    let growth = measure_footprint(label, FOOTPRINT_CALLS, 10, units, || {
        gpu.reset();
        gpu.forward_prefill_rows(&prompt.rows, &prompt.positions, 0, Some(&mut logits))
            .expect("rows prefill");
        assert_eq!(gpu.rope_delta(), grid.n_tokens() - grid.h.max(grid.w));
    });
    if growth > FOOTPRINT_GROWTH_CEILING {
        failures.push(format!("{label}: {}", mib(growth)));
    }
    assert!(
        failures.is_empty(),
        "the runner's prefill grew the process footprint past {} per phase — something \
         each call creates outlives it (an undrained autorelease pool?): {failures:?}",
        mib(FOOTPRINT_GROWTH_CEILING)
    );
}

/// The runner's prefill does not grow the process: 4800 chunk-sized calls
/// of the batched token prefill (every one on the tiled GEMMs) and 4800 of
/// the rows prefill (text and image rows at their 3-axis positions) grow
/// the process footprint by at most 1 MiB each. Every call's command buffer
/// and encoders are autoreleased objects the runner drains per call; left
/// to the thread they cost about 1.9 KiB per call until it exits.
///
/// The measurement runs in a child process running only this test (the
/// test binary re-executed with `--exact`), since other tests of this
/// binary allocate on their own threads meanwhile.
#[test]
fn metal_runner_prefill_does_not_grow_the_process_footprint_bonsai2() {
    if std::env::var_os(FOOTPRINT_CHILD_ENV).is_some() {
        prefill_footprint_probe();
        return;
    }
    if !metal_available() {
        return;
    }
    let path = module_path!();
    let module = path.split_once("::").map_or(path, |(_, rest)| rest);
    let name =
        format!("{module}::metal_runner_prefill_does_not_grow_the_process_footprint_bonsai2");
    let exe = std::env::current_exe().expect("the test binary's path");
    let output = std::process::Command::new(exe)
        .args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env(FOOTPRINT_CHILD_ENV, "1")
        .output()
        .expect("the footprint probe process starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `--nocapture` lets the first report share a line with libtest's own
    // `test <name> ... ` prefix, so a report is found anywhere in a line.
    let reports: Vec<&str> = stdout
        .lines()
        .chain(stderr.lines())
        .filter_map(|line| line.find(FOOTPRINT_REPORT).map(|at| &line[at..]))
        .collect();
    for line in &reports {
        eprintln!("{line}");
    }
    assert!(
        output.status.success(),
        "the footprint probe failed ({}):\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("1 passed"),
        "the probe process ran no test named {name}:\n{stdout}"
    );
    assert_eq!(
        reports.len(),
        FOOTPRINT_PHASES,
        "every measured phase reports once:\n{stdout}\n{stderr}"
    );
}
