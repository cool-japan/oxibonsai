//! Unit tests for `model_source.rs` (sibling file, declared there via
//! `#[path]`, so `super` still names that module).

use super::*;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::quant_prism::{BlockPQ2_0, BlockPTQ1_0};

/// A ternary-valued f32 row (`{-s, 0, +s}` per 128-block) of `n` weights.
fn ternary_weights(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let block_scale = 0.25 + ((i / 128) as f32) * 0.125;
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 29;
            match h % 3 {
                0 => -block_scale,
                1 => 0.0,
                _ => block_scale,
            }
        })
        .collect()
}

fn ptq1_bytes(weights: &[f32]) -> Vec<u8> {
    let blocks = BlockPTQ1_0::quantize(weights).expect("quantize PTQ1_0");
    let mut bytes = Vec::with_capacity(blocks.len() * 28);
    for block in &blocks {
        bytes.extend_from_slice(&block.qs);
        bytes.extend_from_slice(&block.qh);
        bytes.extend_from_slice(&block.d.to_le_bytes());
    }
    bytes
}

/// A small GGUF: metadata (incl. a string array, like a vocabulary), one
/// F32 tensor and two PTQ1_0 tensors of different widths.
fn fixture(with_ptq1: bool) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata("general.file_type", MetadataWriteValue::U32(40));
    w.add_metadata(
        "tokenizer.ggml.tokens",
        MetadataWriteValue::ArrayStr(vec![
            "a".to_string(),
            "b".to_string(),
            "<|im_end|>".to_string(),
        ]),
    );
    w.add_metadata("prism.hadamard.version", MetadataWriteValue::U32(1));
    let norm: Vec<u8> = (0..64)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![64],
        tensor_type: TensorType::F32,
        data: norm,
    });
    let a = ternary_weights(256 * 4, 7);
    let b = ternary_weights(128 * 3, 11);
    if with_ptq1 {
        w.add_tensor(TensorEntry {
            name: "blk.0.ffn_up.weight".to_string(),
            shape: vec![256, 4],
            tensor_type: TensorType::PTQ1_0,
            data: ptq1_bytes(&a),
        });
        w.add_tensor(TensorEntry {
            name: "token_embd.weight".to_string(),
            shape: vec![128, 3],
            tensor_type: TensorType::PTQ1_0,
            data: ptq1_bytes(&b),
        });
    }
    (w.to_bytes().expect("serialize fixture"), a, b)
}

fn dequant_ptq1(gguf: &GgufFile<'_>, name: &str) -> Vec<f32> {
    let blocks =
        BlockPTQ1_0::slice_from_bytes(gguf.tensor_data(name).expect("data")).expect("ptq1 blocks");
    let mut out = vec![0.0f32; blocks.len() * 128];
    BlockPTQ1_0::dequant(blocks, &mut out).expect("dequant ptq1");
    out
}

fn dequant_pq2(gguf: &GgufFile<'_>, name: &str) -> Vec<f32> {
    let blocks =
        BlockPQ2_0::slice_from_bytes(gguf.tensor_data(name).expect("data")).expect("pq2 blocks");
    let mut out = vec![0.0f32; blocks.len() * 128];
    BlockPQ2_0::dequant(blocks, &mut out).expect("dequant pq2");
    out
}

#[test]
fn transcode_rewrites_every_ptq1_tensor_losslessly_and_keeps_everything_else() {
    let (src, _, _) = fixture(true);
    let original = GgufFile::parse(&src).expect("parse source");
    let (image, transcoded) = transcode_ptq1_image(&src)
        .expect("transcode")
        .expect("the fixture has PTQ1_0 tensors");
    assert_eq!(transcoded, 2);
    let rebuilt = GgufFile::parse(&image).expect("the image parses as GGUF");

    for name in ["blk.0.ffn_up.weight", "token_embd.weight"] {
        let info = rebuilt.tensors.get(name).expect("tensor kept");
        assert_eq!(info.tensor_type, GgufTensorType::PQ2_0, "{name}");
        assert_eq!(
            info.shape,
            original.tensors.get(name).expect("source tensor").shape,
            "{name} keeps its shape"
        );
        // Lossless: the PQ2_0 values equal the PTQ1_0 values bit for bit.
        let before = dequant_ptq1(&original, name);
        let after = dequant_pq2(&rebuilt, name);
        assert_eq!(before.len(), after.len(), "{name}");
        for (i, (x, y)) in before.iter().zip(&after).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}[{i}]");
        }
    }

    // A non-PTQ1_0 tensor is copied byte for byte.
    assert_eq!(
        rebuilt.tensor_data("output_norm.weight").expect("norm"),
        original.tensor_data("output_norm.weight").expect("norm")
    );
    // The metadata section is reproduced verbatim.
    assert_eq!(rebuilt.metadata.len(), original.metadata.len());
    for (key, value) in original.metadata.iter() {
        // `MetadataValue` has no `PartialEq`; its `Debug` form carries the
        // variant and the full value.
        assert_eq!(
            format!("{:?}", rebuilt.metadata.get(key)),
            format!("{:?}", Some(value)),
            "metadata key {key}"
        );
    }
    assert_eq!(
        rebuilt
            .metadata
            .get_string_array("tokenizer.ggml.tokens")
            .expect("tokens"),
        vec!["a".to_string(), "b".to_string(), "<|im_end|>".to_string()]
    );
}

#[test]
fn transcode_is_none_for_a_file_without_ptq1_tensors() {
    let (src, _, _) = fixture(false);
    assert!(transcode_ptq1_image(&src).expect("parse").is_none());
}

#[test]
fn transcode_rejects_bytes_that_are_not_gguf() {
    assert!(transcode_ptq1_image(b"definitely not a gguf file").is_err());
}

#[test]
fn model_source_maps_by_default_and_transcodes_on_request() {
    let (src, _, _) = fixture(true);
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_cli_model_source_{}_{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, &src).expect("write fixture");
    let path_str = path.to_string_lossy().into_owned();

    let mapped = ModelSource::open(&path_str, false).expect("map");
    assert_eq!(mapped.bytes(), src.as_slice());
    assert_eq!(mapped.transcoded_tensors(), 0);
    assert_eq!(mapped.weight_bytes(), src.len() as u64);

    let transcoded = ModelSource::open(&path_str, true).expect("transcode");
    let _ = std::fs::remove_file(&path);
    assert_eq!(transcoded.transcoded_tensors(), 2);
    let gguf = GgufFile::parse(transcoded.bytes()).expect("parse image");
    assert_eq!(
        gguf.tensors
            .get("token_embd.weight")
            .expect("embd")
            .tensor_type,
        GgufTensorType::PQ2_0
    );
    // 34/28 bytes per block: the image is larger than the source.
    assert!(transcoded.weight_bytes() > src.len() as u64);
    #[cfg(feature = "server")]
    {
        let leaked = transcoded.into_static();
        assert!(GgufFile::parse(leaked).is_ok());
    }
}
