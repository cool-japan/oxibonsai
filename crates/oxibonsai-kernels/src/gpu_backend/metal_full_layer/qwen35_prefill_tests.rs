//! The batched prefill and the rotary source of the Qwen3.5 hybrid encoder
//! on the tiny random model of `qwen35_tests`: the GEMM-backed prefill
//! against the sequential one, its chunk invariance, decode untouched,
//! per-row angles against the resident table, the batch-capacity control,
//! and the Gated-DeltaNet recurrence's per-token cost at the 27B's shapes.

use std::time::Instant;

use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};

use super::tests::{assert_bits, session, worst_rel, Rng, TinyModel};
use super::*;

/// Cosine similarity in `f64`.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

/// A model of the tiny geometry with a 256-position window and 128-token
/// calls, in `mode`.
fn model(tiny: &TinyModel, mode: Qwen35PrefillMode) -> Qwen35GpuModel<'static> {
    let mut m = Qwen35GpuModel::new(&tiny.weights(), Qwen35Residency::Copied).expect("model");
    m.set_prefill_mode(mode);
    m
}

/// Run `rows` through `m` in calls of `chunk` rows from position 0,
/// returning the last row's logits.
fn prefill_in_chunks(m: &mut Qwen35GpuModel<'_>, rows: &[f32], chunk: usize) -> Vec<f32> {
    let hidden = m.config().hidden;
    let mut logits = vec![0.0f32; m.config().vocab];
    let n = rows.len() / hidden;
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let last = (end == n).then_some(logits.as_mut_slice());
        m.forward(&rows[start * hidden..end * hidden], start, last)
            .expect("prefill chunk");
        start = end;
    }
    logits
}

/// A 100-token prefill in the batched mode (every projection on the GEMM)
/// tracks the sequential one: logit cosine at least 0.99999 and worst
/// relative error at most 1e-4 — then four decode steps from each keep
/// tracking, so the recurrent state and the KV cache the GEMM prefill left
/// behind are the sequential ones to rounding.
#[test]
fn batched_prefill_tracks_the_sequential_prefill() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(7, 256, 128);
    let hidden = tiny.cfg.hidden;
    let rows = Rng::new(11).vec(104 * hidden, 1.0);
    let (prompt, decode) = rows.split_at(100 * hidden);
    let mut seq = model(&tiny, Qwen35PrefillMode::Sequential);
    let mut bat = model(&tiny, Qwen35PrefillMode::Batched);
    let a = prefill_in_chunks(&mut seq, prompt, 128);
    let b = prefill_in_chunks(&mut bat, prompt, 128);
    let (cos, rel) = (cosine(&b, &a), worst_rel(&b, &a));
    eprintln!("batched vs sequential 100-token prefill: cosine {cos:.9}, worst rel {rel:.3e}");
    assert!(cos >= 0.999_99, "prefill logit cosine {cos}");
    assert!(rel <= 1e-4, "prefill worst relative error {rel:e}");
    let (mut la, mut lb) = (vec![0.0f32; tiny.cfg.vocab], vec![0.0f32; tiny.cfg.vocab]);
    for (step, row) in decode.chunks(hidden).enumerate() {
        seq.forward(row, 100 + step, Some(&mut la)).expect("decode");
        bat.forward(row, 100 + step, Some(&mut lb)).expect("decode");
        let (cos, rel) = (cosine(&lb, &la), worst_rel(&lb, &la));
        assert!(cos >= 0.999_99, "decode step {step}: cosine {cos}");
        assert!(
            rel <= 1e-4,
            "decode step {step}: worst relative error {rel:e}"
        );
    }
}

/// The batched prefill does not depend on how a prompt is chunked, as long
/// as every chunk takes the GEMM: a GEMM row's result is independent of its
/// tile position and every other kernel is per row or sequential in the
/// row, so 120 rows in one call, in 64 + 56 and in 40 + 40 + 40 give the
/// same logits and the same recurrent state, bit for bit.
#[test]
fn batched_prefill_is_bitwise_chunk_invariant() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(13, 256, 128);
    let rows = Rng::new(17).vec(120 * tiny.cfg.hidden, 1.0);
    let mut reference = None;
    for chunk in [120usize, 64, 40] {
        let mut m = model(&tiny, Qwen35PrefillMode::Batched);
        let logits = prefill_in_chunks(&mut m, &rows, chunk);
        let state = m.snapshot_state();
        match &reference {
            None => reference = Some((logits, state)),
            Some((want, want_state)) => {
                assert_bits(&logits, want, &format!("logits, chunk {chunk} vs 120"));
                assert!(
                    state == *want_state,
                    "recurrent state after chunk {chunk} differs from one call"
                );
            }
        }
    }
}

/// A single-token call (decode) takes the GEMV in both modes: bit-identical.
#[test]
fn decode_is_the_same_gemv_in_both_modes() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(19, 64, 32);
    let rows = Rng::new(23).vec(6 * tiny.cfg.hidden, 1.0);
    let mut seq = model(&tiny, Qwen35PrefillMode::Sequential);
    let mut bat = model(&tiny, Qwen35PrefillMode::Batched);
    assert_eq!(bat.prefill_mode().as_str(), "batched");
    assert_eq!(seq.prefill_mode().to_string(), "sequential");
    let (mut la, mut lb) = (vec![0.0f32; tiny.cfg.vocab], vec![0.0f32; tiny.cfg.vocab]);
    for (pos, row) in rows.chunks(tiny.cfg.hidden).enumerate() {
        seq.forward(row, pos, Some(&mut la)).expect("decode");
        bat.forward(row, pos, Some(&mut lb)).expect("decode");
        assert_bits(&lb, &la, &format!("decode step {pos}"));
    }
    // Below the GEMM threshold a multi-token call stays on the GEMV too.
    let short = Rng::new(29).vec((Q35_GEMM_MIN_COLS - 1) * tiny.cfg.hidden, 1.0);
    seq.reset();
    bat.reset();
    seq.forward(&short, 0, Some(&mut la))
        .expect("short prefill");
    bat.forward(&short, 0, Some(&mut lb))
        .expect("short prefill");
    assert_bits(&lb, &la, "a call below Q35_GEMM_MIN_COLS");
}

/// `forward` is `forward_rows` with the contiguous source at the KV start,
/// bit for bit; per-row angles copied from the resident table rotate
/// exactly like the table; a contiguous source below the KV start (text
/// after an image) reads the table from there.
#[test]
fn per_row_angles_equal_the_resident_table_bitwise() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(31, 64, 32);
    let c = tiny.cfg.clone();
    let half = c.n_rot / 2;
    let rows = Rng::new(37).vec(12 * c.hidden, 1.0);
    for mode in [Qwen35PrefillMode::Sequential, Qwen35PrefillMode::Batched] {
        let run = |rope: Qwen35Rope<'_>, kv: usize| {
            let mut m = model(&tiny, mode);
            let mut logits = vec![0.0f32; c.vocab];
            if kv > 0 {
                // Occupy the KV positions below the start with the same rows.
                m.forward(&rows[..kv * c.hidden], 0, None).expect("lead");
            }
            m.forward_rows(&rows[kv * c.hidden..], kv, rope, Some(&mut logits))
                .expect("forward_rows");
            logits
        };
        let plain = {
            let mut m = model(&tiny, mode);
            let mut logits = vec![0.0f32; c.vocab];
            m.forward(&rows, 0, Some(&mut logits)).expect("forward");
            logits
        };
        let contiguous = run(Qwen35Rope::Contiguous { rope_start: 0 }, 0);
        assert_bits(
            &contiguous,
            &plain,
            &format!("{mode}: contiguous vs forward"),
        );
        let per_row = run(
            Qwen35Rope::PerRow {
                cos: &tiny.cos_table()[..12 * half],
                sin: &tiny.sin_table()[..12 * half],
            },
            0,
        );
        assert_bits(&per_row, &plain, &format!("{mode}: per-row table angles"));
        // Rows 4.. at KV 4.. but rotating from 1: the table from row 1.
        let shifted = run(Qwen35Rope::Contiguous { rope_start: 1 }, 4);
        let shifted_rows = run(
            Qwen35Rope::PerRow {
                cos: &tiny.cos_table()[half..9 * half],
                sin: &tiny.sin_table()[half..9 * half],
            },
            4,
        );
        assert_bits(
            &shifted_rows,
            &shifted,
            &format!("{mode}: per-row vs contiguous below the KV start"),
        );
        assert!(
            shifted
                .iter()
                .zip(&plain)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "{mode}: the rotary source must reach the attention"
        );
    }
}

/// A rotary source that does not fit the call is refused before any GPU
/// work: per-row angles of the wrong length, a contiguous run past the
/// table.
#[test]
fn a_bad_rope_source_is_refused() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(41, 32, 16);
    let c = tiny.cfg.clone();
    let half = c.n_rot / 2;
    let mut m = model(&tiny, Qwen35PrefillMode::Batched);
    let rows = Rng::new(43).vec(4 * c.hidden, 1.0);
    let short = vec![0.0f32; 3 * half];
    let err = m
        .forward_rows(
            &rows,
            0,
            Qwen35Rope::PerRow {
                cos: &short,
                sin: &short,
            },
            None,
        )
        .expect_err("three angle rows for four input rows");
    assert!(
        matches!(err, MetalGraphError::InvalidDimensions(_)),
        "{err}"
    );
    let err = m
        .forward_rows(&rows, 0, Qwen35Rope::Contiguous { rope_start: 30 }, None)
        .expect_err("rotary positions 30..34 past a 32-row table");
    assert!(err.to_string().contains("angle table"), "{err}");
}

/// The batch capacity moves after construction: a larger batch takes a
/// longer call, a zero batch and one whose scratch cannot fit beside the KV
/// window are refused naming both numbers, and shrinking releases the
/// scratch.
#[test]
fn the_batch_capacity_can_grow_and_is_bounded_by_the_device() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(47, 256, 16);
    let c = tiny.cfg.clone();
    let mut m = model(&tiny, Qwen35PrefillMode::Batched);
    let rows = Rng::new(53).vec(100 * c.hidden, 1.0);
    assert!(
        m.forward(&rows, 0, None).is_err(),
        "100 rows exceed the 16-token batch"
    );
    m.set_max_batch(128).expect("grow the batch");
    assert_eq!(m.config().max_batch, 128);
    m.forward(&rows, 0, None).expect("a 100-row call now fits");
    assert!(matches!(
        m.set_max_batch(0),
        Err(MetalGraphError::InvalidDimensions(_))
    ));
    let err = m
        .set_max_batch(1 << 40)
        .expect_err("a trillion-token scratch does not fit");
    let text = err.to_string();
    assert!(text.contains("256") && text.contains("128"), "{text}");
    assert_eq!(m.config().max_batch, 128, "a refusal changes nothing");
    m.set_max_batch(8).expect("shrink");
    assert_eq!(m.scratch.capacity, 1, "shrinking releases the scratch");
    m.reset();
    m.forward(&rows[..8 * c.hidden], 0, None)
        .expect("an 8-row call after shrinking");
}

/// Per-token cost of the Gated-DeltaNet recurrence inside a prefill chunk,
/// at the 27B's shapes (48 v-heads over 16 k-heads of 128, conv width
/// 10240): one `q35_gdn` dispatch over a 512-token chunk, best of three GPU
/// times, reported per token and per layer and scaled to the 48 recurrent
/// layers of a token. Recorded, not asserted (it depends on the device's
/// load).
#[test]
fn gdn_cost_per_token_at_the_27b_shapes() {
    let Some(graph) = session() else {
        return;
    };
    const T: usize = 512;
    let (nk, nv, hk, hv) = (16usize, 48usize, 128usize, 128usize);
    let conv_dim = 2 * nk * hk + nv * hv;
    let mut rng = Rng::new(59);
    let conv_out = upload_f32(&graph, &rng.vec(T * conv_dim, 1.0)).expect("conv_out");
    let ab = upload_f32(&graph, &rng.vec(T * 2 * nv, 1.0)).expect("ab");
    let a_neg = upload_f32(
        &graph,
        &(0..nv).map(|h| -0.25 - h as f32 * 0.01).collect::<Vec<_>>(),
    )
    .expect("a_neg");
    let dt_bias = upload_f32(&graph, &rng.vec(nv, 0.5)).expect("dt_bias");
    let state = zeroed(&graph, nv * hk * hv * 4).expect("state");
    let out = zeroed(&graph, T * nv * hv * 4).expect("out");
    let pso = graph.pipeline_for("q35_gdn").expect("q35_gdn");
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let seconds = autoreleasepool(|| {
            let cmd = graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&conv_out), 0);
            enc.set_buffer(1, Some(&ab), 0);
            enc.set_buffer(2, Some(&a_neg), 0);
            enc.set_buffer(3, Some(&dt_bias), 0);
            enc.set_buffer(4, Some(&state), 0);
            enc.set_buffer(5, Some(&out), 0);
            for (i, v) in [nk, nv, hk, hv, conv_dim, T].into_iter().enumerate() {
                set_u32(enc, 6 + i as u64, v as u32);
            }
            set_f32(enc, 12, 1e-6);
            set_f32(enc, 13, 1.0 / (hv as f32).sqrt());
            enc.dispatch_thread_groups(
                MTLSize::new(nv as u64, 1, 1),
                MTLSize::new(GDN_THREADS, 1, 1),
            );
            enc.end_encoding();
            let wall = Instant::now();
            commit_and_wait(cmd, "gdn bench").expect("command buffer");
            // SAFETY: the command buffer completed, so its timestamps are
            // readable.
            let (s, e): (f64, f64) =
                unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
            let gpu = (e - s).max(0.0);
            eprintln!(
                "q35_gdn {T} tokens: GPU {:.2} ms (wall {:.2} ms)",
                gpu * 1e3,
                wall.elapsed().as_secs_f64() * 1e3
            );
            gpu
        });
        best = best.min(seconds);
    }
    let per_token_layer = best / T as f64;
    eprintln!(
        "q35_gdn at the 27B shapes: {:.1} us per token per layer, {:.2} ms per token over 48 \
         recurrent layers",
        per_token_layer * 1e6,
        per_token_layer * 48.0 * 1e3
    );
    let values = read_buffer(&out, 0, 16);
    assert!(values.iter().all(|v| v.is_finite()));
}

/// The GEMM threshold is an object setting: lowered to 2, a 4-token call
/// runs on the GEMM (close to, but not bit-identical with, the sequential
/// GEMV), decode stays on the GEMV, and a threshold that would put decode on
/// the GEMM is refused.
#[test]
fn the_gemm_threshold_is_an_object_setting() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::with_window(61, 64, 32);
    let rows = Rng::new(67).vec(4 * tiny.cfg.hidden, 1.0);
    let mut seq = model(&tiny, Qwen35PrefillMode::Sequential);
    let mut bat = model(&tiny, Qwen35PrefillMode::Batched);
    assert_eq!(bat.gemm_min_cols(), Q35_GEMM_MIN_COLS);
    assert!(
        bat.set_gemm_min_cols(1).is_err(),
        "decode must stay on the GEMV"
    );
    bat.set_gemm_min_cols(2).expect("threshold 2");
    let (mut la, mut lb) = (vec![0.0f32; tiny.cfg.vocab], vec![0.0f32; tiny.cfg.vocab]);
    seq.forward(&rows, 0, Some(&mut la)).expect("sequential");
    bat.forward(&rows, 0, Some(&mut lb)).expect("gemm");
    assert!(cosine(&lb, &la) >= 0.999_99);
    assert!(worst_rel(&lb, &la) <= 1e-4);
    assert!(
        lb.iter().zip(&la).any(|(a, b)| a.to_bits() != b.to_bits()),
        "a 4-token call must have taken the GEMM"
    );
    let row = Rng::new(71).vec(tiny.cfg.hidden, 1.0);
    seq.forward(&row, 4, Some(&mut la)).expect("decode");
    let mut again = model(&tiny, Qwen35PrefillMode::Batched);
    again.set_gemm_min_cols(2).expect("threshold 2");
    again.forward(&row, 0, Some(&mut lb)).expect("decode");
    let mut plain = model(&tiny, Qwen35PrefillMode::Sequential);
    plain.forward(&row, 0, Some(&mut la)).expect("decode");
    assert_bits(&lb, &la, "one token is the GEMV whatever the threshold");
}
