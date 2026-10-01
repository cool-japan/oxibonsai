//! Unit tests for the vision tower: the loader's strictness, each re-layout
//! step against the emulated ggml op sequence, the vision RoPE against its
//! `f64` transliteration, and the whole `f32` graph against the independent
//! `f64` reference on the synthetic projector.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::MetadataWriteValue;
use oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd;
use oxibonsai_testkit::gguf_fixture::{deterministic_weights, tiny_dense_qwen3_gguf, FixtureQuant};
use oxibonsai_testkit::mmproj_fixture::{
    assemble, metadata, pattern_rgb8, synthetic_mmproj_gguf, tensors, FixtureTensor,
    MmprojFixtureSpec,
};

use super::f64_reference as reference;
use super::patch_embed::{normalize_planar, resize_position_grid, window_order};
use super::tower::{gelu_tanh, layer_norm_rows, vision_rope_table};
use super::{GridSize, ImageRgb8, VisionTower};
use crate::error::ModelError;

// ─── helpers ────────────────────────────────────────────────────────────

fn spec() -> MmprojFixtureSpec {
    MmprojFixtureSpec::tiny()
}

fn fixture_bytes() -> Vec<u8> {
    synthetic_mmproj_gguf(&spec()).expect("build the synthetic projector")
}

fn tower_from(bytes: &[u8]) -> VisionTower {
    let gguf = GgufFile::parse(bytes).expect("parse the projector");
    VisionTower::from_mmproj(&gguf).expect("load the projector")
}

fn load_error(meta: &[(String, MetadataWriteValue)], tensors: &[FixtureTensor]) -> ModelError {
    let bytes = assemble(meta, tensors).expect("assemble the modified projector");
    let gguf = GgufFile::parse(&bytes).expect("parse the modified projector");
    VisionTower::from_mmproj(&gguf).expect_err("the loader must refuse this projector")
}

fn set_meta(meta: &mut Vec<(String, MetadataWriteValue)>, key: &str, value: MetadataWriteValue) {
    match meta.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => meta.push((key.to_string(), value)),
    }
}

fn image(width: usize, height: usize) -> ImageRgb8 {
    ImageRgb8::new(width, height, pattern_rgb8(width, height)).expect("pattern image")
}

/// The f32 tower against the f64 reference on one image; returns the
/// largest per-row relative L2 error.
fn tower_vs_reference(bytes: &[u8], width: usize, height: usize) -> f64 {
    let tower = tower_from(bytes);
    let img = image(width, height);
    let (rows, grid) = tower.encode(&img, usize::MAX).expect("encode");
    let gguf = GgufFile::parse(bytes).expect("parse");
    let expected = reference::encode_rgb8(&gguf, width, height, &img.data).expect("f64 reference");
    assert_eq!(
        grid,
        GridSize {
            h: expected.grid_h,
            w: expected.grid_w
        }
    );
    assert_eq!(grid.n_tokens(), expected.n_rows);
    assert_eq!(rows.len(), expected.rows.len());
    assert!(rows.iter().all(|v| v.is_finite()), "non-finite output");
    reference::max_row_relative_error(&rows, &expected.rows, expected.dim)
}

// ─── loading ────────────────────────────────────────────────────────────

#[test]
fn synthetic_projector_loads_and_binds_every_tensor() {
    let s = spec();
    let tower = tower_from(&fixture_bytes());
    let cfg = tower.config();
    assert_eq!(cfg.hidden, s.hidden);
    assert_eq!(cfg.heads, s.heads);
    assert_eq!(cfg.head_dim, s.hidden / s.heads);
    assert_eq!(cfg.ffn, s.ffn);
    assert_eq!(cfg.blocks, s.blocks);
    assert_eq!(cfg.patch_size, s.patch);
    assert_eq!(cfg.spatial_merge, 2);
    assert_eq!(cfg.pos_grid, s.pos_side);
    assert_eq!(cfg.merger_hidden, s.merger_hidden);
    assert_eq!(cfg.projection_dim, s.projection_dim);
    assert_eq!(cfg.image_size, s.pos_side * s.patch);
    assert_eq!(cfg.image_mean, [0.5; 3]);
    assert_eq!(cfg.image_std, [0.5; 3]);
    assert!((cfg.eps - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.merge_unit(), 32);
    assert_eq!(cfg.rope_sections(), [4; 4]);
    assert_eq!(tower.block_count(), s.blocks);
    assert_eq!(tower.bound_tensor_count(), 12 * s.blocks + 10);

    // Every resident f32: the summed patch kernel, bias, position grid,
    // twelve tensors per block, post_ln and the merger.
    let h = s.hidden;
    let per_block =
        2 * h + (3 * h * h + 3 * h) + (h * h + h) + 2 * h + (h * s.ffn + s.ffn) + (s.ffn * h + h);
    let floats = (h * 3 * s.patch * s.patch + h + s.pos_side * s.pos_side * h)
        + s.blocks * per_block
        + 2 * h
        + (4 * h * s.merger_hidden + s.merger_hidden)
        + (s.merger_hidden * s.projection_dim + s.projection_dim);
    assert_eq!(tower.resident_bytes(), floats * 4);
}

#[test]
fn loader_names_a_missing_tensor() {
    let s = spec();
    for missing in [
        "v.blk.1.ffn_down.bias",
        "v.patch_embd.bias",
        "v.patch_embd.weight.1",
        "mm.2.weight",
    ] {
        let mut all = tensors(&s);
        all.retain(|t| t.name != missing);
        match load_error(&metadata(&s), &all) {
            ModelError::MissingTensor { name } => assert_eq!(name, missing),
            other => panic!("{missing}: expected MissingTensor, got {other:?}"),
        }
    }
}

#[test]
fn loader_names_expected_and_actual_shapes() {
    let s = spec();
    let mut all = tensors(&s);
    let target = all
        .iter_mut()
        .find(|t| t.name == "v.blk.0.attn_out.weight")
        .expect("attn_out present");
    target.shape = vec![64, 32];
    target.values.truncate(64 * 32);
    match load_error(&metadata(&s), &all) {
        ModelError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "v.blk.0.attn_out.weight");
            assert_eq!(expected, vec![64, 64]);
            assert_eq!(actual, vec![64, 32]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn loader_refuses_tensors_the_graph_does_not_read() {
    let s = spec();
    let mut all = tensors(&s);
    let extra = |name: &str, shape: &[u64]| FixtureTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        quant: FixtureQuant::F32,
        values: deterministic_weights(shape.iter().product::<u64>() as usize, 7),
    };
    all.push(extra("v.blk.0.ffn_gate.weight", &[64, 96]));
    all.push(extra("v.pre_ln.weight", &[64]));
    all.push(extra("v.class_embd", &[64]));
    all.push(extra("v.deepstack.1.fc1.weight", &[256, 64]));
    match load_error(&metadata(&s), &all) {
        ModelError::ShapeInvariant { actual, .. } => {
            for name in [
                "v.blk.0.ffn_gate.weight",
                "v.pre_ln.weight",
                "v.class_embd",
                "v.deepstack.1.fc1.weight",
            ] {
                assert!(actual.contains(name), "{name} not reported in: {actual}");
            }
            assert!(actual.starts_with("4 unexpected"), "{actual}");
        }
        other => panic!("expected ShapeInvariant, got {other:?}"),
    }
}

#[test]
fn loader_refuses_graph_variants_it_does_not_implement() {
    let s = spec();
    let cases: Vec<(&str, MetadataWriteValue)> = vec![
        (
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        ),
        ("clip.has_vision_encoder", MetadataWriteValue::Bool(false)),
        (
            "clip.projector_type",
            MetadataWriteValue::Str("qwen2vl_merger".to_string()),
        ),
        ("clip.use_gelu", MetadataWriteValue::Bool(false)),
        ("clip.use_silu", MetadataWriteValue::Bool(true)),
        ("clip.vision.spatial_merge_size", MetadataWriteValue::U32(4)),
        (
            "clip.vision.is_deepstack_layers",
            MetadataWriteValue::ArrayBool(vec![false, true]),
        ),
        (
            "clip.vision.attention.head_count_kv",
            MetadataWriteValue::U32(2),
        ),
        (
            "clip.vision.attention.head_count",
            MetadataWriteValue::U32(3),
        ),
        // 64 / 32 = 2: not divisible into four rotary sections.
        (
            "clip.vision.attention.head_count",
            MetadataWriteValue::U32(32),
        ),
        (
            "clip.vision.attention.head_dim",
            MetadataWriteValue::U32(32),
        ),
        (
            "clip.vision.image_std",
            MetadataWriteValue::ArrayF32(vec![0.5, 0.0, 0.5]),
        ),
    ];
    for (key, value) in cases {
        let mut meta = metadata(&s);
        set_meta(&mut meta, key, value.clone());
        let err = load_error(&meta, &tensors(&s));
        assert!(
            matches!(err, ModelError::ShapeInvariant { .. }),
            "{key} = {value:?}: expected ShapeInvariant, got {err:?}"
        );
    }
}

#[test]
fn loader_requires_the_metadata_it_reads() {
    let s = spec();
    for key in [
        "clip.vision.embedding_length",
        "clip.vision.block_count",
        "clip.vision.attention.layer_norm_epsilon",
        "clip.vision.image_mean",
        "clip.projector_type",
    ] {
        let mut meta = metadata(&s);
        meta.retain(|(k, _)| k != key);
        let err = load_error(&meta, &tensors(&s));
        assert!(
            matches!(err, ModelError::Core(_)),
            "{key} removed: expected a core metadata error, got {err:?}"
        );
    }
}

#[test]
fn loader_refuses_an_unsupported_storage_type() {
    let s = spec();
    let mut all = tensors(&s);
    all.iter_mut()
        .find(|t| t.name == "v.blk.1.attn_out.weight")
        .expect("attn_out present")
        .quant = FixtureQuant::Q4_0;
    match load_error(&metadata(&s), &all) {
        ModelError::InvalidTensor(msg) => {
            assert!(msg.contains("v.blk.1.attn_out.weight"), "{msg}");
            assert!(msg.contains("Q4_0"), "{msg}");
        }
        other => panic!("expected InvalidTensor, got {other:?}"),
    }
}

#[test]
fn loader_refuses_a_non_square_position_grid() {
    let s = spec();
    let mut all = tensors(&s);
    let pos = all
        .iter_mut()
        .find(|t| t.name == "v.position_embd.weight")
        .expect("position embedding present");
    pos.shape = vec![64, 60];
    pos.values.truncate(64 * 60);
    let err = load_error(&metadata(&s), &all);
    assert!(matches!(err, ModelError::ShapeInvariant { .. }), "{err:?}");
}

#[test]
fn a_text_model_gguf_is_not_a_vision_projector() {
    let bytes = tiny_dense_qwen3_gguf(3).expect("tiny dense qwen3 fixture");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    match VisionTower::from_mmproj(&gguf) {
        Err(ModelError::ShapeInvariant { tensor, actual, .. }) => {
            assert_eq!(tensor, "general.architecture");
            assert!(actual.contains("qwen3"), "{actual}");
        }
        other => panic!("expected an architecture refusal, got {other:?}"),
    }
}

// ─── re-layout steps against the emulated ggml ops ──────────────────────

#[test]
fn window_order_matches_the_ggml_permute_sequence() {
    for (ph, pw) in [
        (2, 2),
        (2, 4),
        (4, 2),
        (4, 6),
        (6, 4),
        (8, 8),
        (6, 8),
        (12, 2),
    ] {
        // A [pw, ph, 1, 1] tensor holding each patch's raster index.
        let raster: Vec<f64> = (0..ph * pw).map(|i| i as f64).collect();
        let t = reference::GTensor::new(raster, [pw, ph, 1, 1]).expect("tensor");
        let laid_out = reference::spatial_merge_layout(&t)
            .expect("layout")
            .to_vec();
        let from_ops: Vec<(usize, usize)> = laid_out
            .iter()
            .map(|&v| {
                let idx = v as usize;
                (idx / pw, idx % pw)
            })
            .collect();
        assert_eq!(window_order(ph, pw), from_ops, "grid {ph} x {pw}");

        // The rotary positions travel with the patches: channel 0 is the
        // row, channel 1 the column of the same token.
        let positions = reference::vision_positions(pw, ph);
        let n = ph * pw;
        for (t, &(y, x)) in window_order(ph, pw).iter().enumerate() {
            assert_eq!(positions[t], y as i64);
            assert_eq!(positions[n + t], x as i64);
            assert_eq!(positions[2 * n + t], y as i64);
            assert_eq!(positions[3 * n + t], x as i64);
        }
    }
}

#[test]
fn position_resize_matches_the_ggml_op_sequence() {
    let side = 8;
    let hidden = 5;
    let table: Vec<f32> = deterministic_weights(side * side * hidden, 11);
    let table64: Vec<f64> = table.iter().map(|&v| f64::from(v)).collect();
    for (ph, pw) in [
        (4, 4),
        (2, 6),
        (6, 2),
        (12, 10),
        (8, 8),
        (16, 16),
        (2, 2),
        (8, 4),
    ] {
        let ours = resize_position_grid(&table, side, hidden, ph, pw).expect("resize");
        let theirs = reference::resize_position_embeddings(&table64, hidden, side, pw, ph)
            .expect("reference resize")
            .to_vec();
        assert_eq!(ours.len(), theirs.len());
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert!(
                (f64::from(*a) - b).abs() <= 1e-5,
                "{ph}x{pw} element {i}: {a} vs {b}"
            );
        }
        // Align-corners: the corners of the output are the stored corners —
        // exactly at the origin, and up to the kernel's own `f32` rounding
        // of `(n - 1) / scale` at the far corner (ggml computes that
        // coordinate in `f32` too, so a fraction of 1 - 2^-23 is faithful).
        let cell = |grid: &[f32], w: usize, y: usize, x: usize| {
            grid[(y * w + x) * hidden..][..hidden].to_vec()
        };
        assert_eq!(cell(&ours, pw, 0, 0), cell(&table, side, 0, 0));
        let far = cell(&ours, pw, ph - 1, pw - 1);
        let stored = cell(&table, side, side - 1, side - 1);
        for (a, b) in far.iter().zip(&stored) {
            assert!((a - b).abs() <= 1e-5, "{ph}x{pw} far corner: {a} vs {b}");
        }
    }
    // The identity resize is a copy.
    assert_eq!(
        resize_position_grid(&table, side, hidden, side, side).expect("identity"),
        table
    );
}

// ─── rotary embedding ───────────────────────────────────────────────────

#[test]
fn vision_rope_matches_the_ggml_transliteration() {
    for head_dim in [72usize, 16] {
        let n_pairs = head_dim / 2;
        let sections = [head_dim / 4; 4];
        let input: Vec<f32> = deterministic_weights(head_dim, head_dim as u64);
        for (y, x) in [(0, 0), (0, 1), (1, 0), (3, 5), (17, 2), (47, 47), (46, 3)] {
            let mut cos = vec![0.0f32; n_pairs];
            let mut sin = vec![0.0f32; n_pairs];
            vision_rope_table(
                [y, x, y, x],
                sections,
                head_dim,
                10_000.0,
                &mut cos,
                &mut sin,
            )
            .expect("table");
            let cache = reference::mrope_vision_cache(
                [i64::from(y), i64::from(x), i64::from(y), i64::from(x)],
                sections,
                head_dim,
                n_pairs,
                reference::ROPE_FREQ_BASE,
            );
            assert_eq!(cache.len(), n_pairs);
            for (j, &(c, s)) in cache.iter().enumerate() {
                assert!(
                    (f64::from(cos[j]) - c).abs() <= 2e-5,
                    "cos[{j}] at ({y},{x})"
                );
                assert!(
                    (f64::from(sin[j]) - s).abs() <= 2e-5,
                    "sin[{j}] at ({y},{x})"
                );
            }
            // Section semantics: the first quarter of the pairs turns by the
            // row position, the second by the column position.
            assert!((f64::from(cos[0]) - f64::from(y).cos()).abs() <= 1e-6);
            assert!((f64::from(cos[head_dim / 4]) - f64::from(x).cos()).abs() <= 1e-6);

            let mut rotated = vec![0.0f32; head_dim];
            rope_partial_splithalf_simd(&input, &mut rotated, head_dim, head_dim, &cos, &sin)
                .expect("rotate");
            let mut expected: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
            reference::rotate_pairs_vision(&mut expected, n_pairs, &cache);
            for (i, (a, b)) in rotated.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(*a) - b).abs() <= 5e-5,
                    "channel {i} at ({y},{x}): {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn vision_rope_rejects_malformed_requests() {
    let mut cos = vec![0.0f32; 8];
    let mut sin = vec![0.0f32; 8];
    assert!(vision_rope_table([0; 4], [4; 4], 18, 1e4, &mut cos, &mut sin).is_err());
    assert!(vision_rope_table([0; 4], [0; 4], 16, 1e4, &mut cos, &mut sin).is_err());
    assert!(vision_rope_table([0; 4], [4; 4], 32, 1e4, &mut cos, &mut sin).is_err());
    assert!(vision_rope_table([0; 4], [4; 4], 16, 1e4, &mut cos, &mut sin).is_ok());
}

// ─── elementwise pieces ─────────────────────────────────────────────────

#[test]
fn gelu_is_the_tanh_approximation() {
    // tanh form at 1.0: 0.841192; the erf form would be 0.841345.
    assert!(
        (gelu_tanh(1.0) - 0.841_192).abs() < 2e-6,
        "{}",
        gelu_tanh(1.0)
    );
    for i in -400..=400 {
        let x = i as f32 / 40.0;
        let expected = reference::gelu_tanh(f64::from(x));
        assert!(
            (f64::from(gelu_tanh(x)) - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
            "gelu({x})"
        );
    }
}

#[test]
fn layer_norm_rows_matches_an_f64_evaluation() {
    let dim = 72;
    let rows = 5;
    let x: Vec<f32> = deterministic_weights(dim * rows, 21)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 3.0 + (i % dim) as f32 * 0.01 + 7.0)
        .collect();
    let w: Vec<f32> = deterministic_weights(dim, 22)
        .iter()
        .map(|v| 1.0 + 0.3 * v)
        .collect();
    let b: Vec<f32> = deterministic_weights(dim, 23);
    let mut out = vec![0.0f32; dim * rows];
    layer_norm_rows(&x, &w, &b, 1e-6, dim, &mut out);
    for (r, (xs, got)) in x.chunks(dim).zip(out.chunks(dim)).enumerate() {
        let row: Vec<f64> = xs.iter().map(|&v| f64::from(v)).collect();
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        for (i, (((v, g), wi), bi)) in row.iter().zip(got).zip(&w).zip(&b).enumerate() {
            let expected = (v - mean) / (var + 1e-6).sqrt() * f64::from(*wi) + f64::from(*bi);
            assert!((f64::from(*g) - expected).abs() <= 1e-5, "row {r} col {i}");
        }
    }
}

// ─── the whole graph against the f64 reference ──────────────────────────

#[test]
fn synthetic_tower_matches_the_f64_reference_64x64() {
    // 4 x 4 patches against an 8 x 8 stored grid: exercises the resize.
    let err = tower_vs_reference(&fixture_bytes(), 64, 64);
    eprintln!("synthetic 64x64: max per-row relative error {err:.3e}");
    assert!(err <= 1e-4, "64x64: {err:.3e}");
}

#[test]
fn synthetic_tower_matches_the_f64_reference_non_square() {
    // 6 x 4 and 4 x 8 patch grids: a swapped axis would show here.
    for (w, h) in [(96, 64), (64, 128)] {
        let err = tower_vs_reference(&fixture_bytes(), w, h);
        eprintln!("synthetic {w}x{h}: max per-row relative error {err:.3e}");
        assert!(err <= 1e-4, "{w}x{h}: {err:.3e}");
    }
}

#[test]
fn synthetic_tower_matches_the_f64_reference_at_the_stored_grid() {
    // 8 x 8 patches == the stored grid: the position table is used as is.
    let err = tower_vs_reference(&fixture_bytes(), 128, 128);
    eprintln!("synthetic 128x128: max per-row relative error {err:.3e}");
    assert!(err <= 1e-4, "128x128: {err:.3e}");
}

#[test]
fn encode_output_shape_and_grid() {
    let s = spec();
    let tower = tower_from(&fixture_bytes());
    let (rows, grid) = tower.encode(&image(96, 64), 64).expect("encode");
    assert_eq!(grid, GridSize { h: 2, w: 3 });
    assert_eq!(grid.n_tokens(), 6);
    assert_eq!(rows.len(), 6 * s.projection_dim);
    assert_eq!(tower.merged_grid(96, 64, 6).expect("grid"), grid);
}

#[test]
fn encode_normalized_matches_encode_bit_for_bit() {
    let tower = tower_from(&fixture_bytes());
    let img = image(64, 96);
    let cfg = tower.config();
    let planar = normalize_planar(&img, cfg.image_mean, cfg.image_std);
    let (a, ga) = tower.encode(&img, 100).expect("encode");
    let (b, gb) = tower
        .encode_normalized(&planar, img.width, img.height, 100)
        .expect("encode_normalized");
    assert_eq!(ga, gb);
    assert_eq!(a, b);
}

#[test]
fn encode_refuses_images_it_cannot_take_as_is() {
    let tower = tower_from(&fixture_bytes());
    for (w, h) in [(48, 64), (64, 48), (0, 64), (64, 0), (33, 32)] {
        let img = ImageRgb8 {
            width: w,
            height: h,
            data: vec![0; w * h * 3],
        };
        let err = tower.encode(&img, usize::MAX).expect_err("bad geometry");
        assert!(
            matches!(err, ModelError::ShapeInvariant { .. }),
            "{w}x{h}: {err:?}"
        );
    }
    // A buffer that disagrees with the declared size.
    let short = ImageRgb8 {
        width: 64,
        height: 64,
        data: vec![0; 64 * 64 * 3 - 1],
    };
    assert!(matches!(
        tower.encode(&short, usize::MAX),
        Err(ModelError::ShapeMismatch { .. })
    ));
    // The merged-token budget: a 64 x 64 image is 2 x 2 = 4 tokens.
    let img = image(64, 64);
    assert!(tower.encode(&img, 4).is_ok());
    match tower.encode(&img, 3) {
        Err(ModelError::ShapeInvariant { tensor, .. }) => assert_eq!(tensor, "image tokens"),
        other => panic!("expected a token-budget refusal, got {other:?}"),
    }
    // Pre-normalised input of the wrong length.
    assert!(matches!(
        tower.encode_normalized(&[0.0; 10], 64, 64, usize::MAX),
        Err(ModelError::ShapeMismatch { .. })
    ));
    // A declared size whose value count overflows is a mismatch, not a
    // panic.
    let huge = (usize::MAX / 64) / 32 * 32;
    assert!(matches!(
        tower.encode_normalized(&[0.0; 3], huge, 64, usize::MAX),
        Err(ModelError::ShapeMismatch { .. })
    ));
    let huge_image = ImageRgb8 {
        width: huge,
        height: 64,
        data: Vec::new(),
    };
    assert!(tower.encode(&huge_image, usize::MAX).is_err());
}

#[test]
fn the_tower_can_be_shared_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VisionTower>();

    let tower = tower_from(&fixture_bytes());
    let img = image(64, 64);
    let (expected, _) = tower.encode(&img, 16).expect("encode");
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3)
            .map(|_| scope.spawn(|| tower.encode(&img, 16).expect("encode on a thread").0))
            .collect();
        for handle in handles {
            assert_eq!(handle.join().expect("thread"), expected);
        }
    });
}

#[test]
fn encode_does_not_depend_on_the_thread_count() {
    let tower = tower_from(&fixture_bytes());
    let img = image(96, 64);
    let (parallel, _) = tower.encode(&img, usize::MAX).expect("encode");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread pool");
    let (serial, _) = pool
        .install(|| tower.encode(&img, usize::MAX))
        .expect("encode on one thread");
    assert_eq!(parallel, serial);
}

#[test]
fn image_rgb8_validates_its_buffer() {
    assert!(ImageRgb8::new(2, 2, vec![0; 12]).is_ok());
    assert!(matches!(
        ImageRgb8::new(2, 2, vec![0; 11]),
        Err(ModelError::ShapeMismatch { .. })
    ));
    let img = ImageRgb8::new(2, 1, vec![1, 2, 3, 4, 5, 6]).expect("image");
    assert_eq!(img.pixel(1, 0), Some([4, 5, 6]));
    assert_eq!(img.pixel(2, 0), None);
    assert_eq!(img.pixel(0, 1), None);
    assert_eq!(GridSize { h: 6, w: 8 }.n_tokens(), 48);
}
