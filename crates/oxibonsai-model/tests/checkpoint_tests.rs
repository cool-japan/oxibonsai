//! Integration tests for the `checkpoint` module.
//!
//! Tests cover round-trip serialization, error handling, and interop with
//! [`WeightTensor`].

use oxibonsai_model::{model_merge::WeightTensor, Checkpoint, CheckpointError, CheckpointTensor};

// ─────────────────────────────────────────────────────────────────────────────
// 1. checkpoint_new_empty
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_new_empty() {
    let ck = Checkpoint::new();
    assert_eq!(ck.tensors.len(), 0, "expected 0 tensors");
    assert_eq!(ck.num_params(), 0, "expected 0 params");
    assert_eq!(ck.metadata.len(), 0, "expected empty metadata");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. checkpoint_add_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_add_tensor() {
    let mut ck = Checkpoint::new();
    let t = CheckpointTensor::new("weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    ck.add_tensor(t);
    assert_eq!(ck.tensors.len(), 1);
    assert_eq!(ck.num_params(), 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. checkpoint_set_get_metadata
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_set_get_metadata() {
    let mut ck = Checkpoint::new();
    ck.set_metadata("step", "42");
    ck.set_metadata("loss", "0.123");
    assert_eq!(ck.get_metadata("step"), Some("42"));
    assert_eq!(ck.get_metadata("loss"), Some("0.123"));
    assert_eq!(ck.get_metadata("missing"), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. checkpoint_get_nonexistent_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_get_nonexistent_tensor() {
    let ck = Checkpoint::new();
    assert!(ck.get_tensor("no_such_tensor").is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. checkpoint_total_bytes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_total_bytes() {
    let mut ck = Checkpoint::new();
    ck.add_tensor(CheckpointTensor::new("a", vec![0.0f32; 10], vec![10]));
    ck.add_tensor(CheckpointTensor::new("b", vec![0.0f32; 6], vec![2, 3]));
    assert_eq!(ck.total_bytes(), (10 + 6) * 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. checkpoint_tensor_element_count
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_tensor_element_count() {
    let t = CheckpointTensor::new("x", vec![0.0f32; 24], vec![2, 3, 4]);
    assert_eq!(t.element_count(), 24);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. checkpoint_tensor_size_bytes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_tensor_size_bytes() {
    let t = CheckpointTensor::new("x", vec![0.0f32; 24], vec![2, 3, 4]);
    assert_eq!(t.size_bytes(), 24 * 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. checkpoint_write_read_empty
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_write_read_empty() {
    let ck = Checkpoint::new();

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write_to failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read_from failed");
    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.tensors.len(), 0);
    assert_eq!(loaded.metadata.len(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. checkpoint_write_read_one_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_write_read_one_tensor() {
    let mut ck = Checkpoint::new();
    let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    ck.add_tensor(CheckpointTensor::new(
        "linear.weight",
        original_data.clone(),
        vec![2, 3],
    ));

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.tensors.len(), 1);

    let t = &loaded.tensors[0];
    assert_eq!(t.name, "linear.weight");
    assert_eq!(t.shape, vec![2u64, 3]);
    assert_eq!(t.data, original_data);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. checkpoint_write_read_metadata
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_write_read_metadata() {
    let mut ck = Checkpoint::new();
    ck.set_metadata("epoch", "5");
    ck.set_metadata("lr", "0.0001");
    ck.set_metadata("model", "qwen3-8b");

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.get_metadata("epoch"), Some("5"));
    assert_eq!(loaded.get_metadata("lr"), Some("0.0001"));
    assert_eq!(loaded.get_metadata("model"), Some("qwen3-8b"));
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. checkpoint_write_read_multiple_tensors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_write_read_multiple_tensors() {
    let mut ck = Checkpoint::new();
    ck.add_tensor(CheckpointTensor::new(
        "embed.weight",
        vec![0.1, 0.2],
        vec![1, 2],
    ));
    ck.add_tensor(CheckpointTensor::new(
        "layer.0.attn.q_proj",
        vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
        vec![2, 3],
    ));
    ck.add_tensor(CheckpointTensor::new(
        "lm_head.bias",
        vec![0.0, 0.01],
        vec![2],
    ));

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.tensors.len(), 3);

    assert_eq!(loaded.tensors[0].name, "embed.weight");
    assert_eq!(loaded.tensors[1].name, "layer.0.attn.q_proj");
    assert_eq!(loaded.tensors[2].name, "lm_head.bias");

    // Verify data integrity for the middle tensor.
    assert_eq!(
        loaded.tensors[1].data,
        vec![1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. checkpoint_write_read_large_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_write_read_large_tensor() {
    const N: usize = 10_000;
    // Deterministic pseudo-data: value at index i = i as f32 / N as f32.
    let data: Vec<f32> = (0..N).map(|i| i as f32 / N as f32).collect();

    let mut ck = Checkpoint::new();
    ck.add_tensor(CheckpointTensor::new("big", data.clone(), vec![N as u64]));

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.tensors[0].data.len(), N);
    // Check every value is preserved bit-exactly.
    assert!(loaded.tensors[0].data == data, "large tensor data mismatch");
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. checkpoint_save_load_tempfile
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_save_load_tempfile() {
    let mut ck = Checkpoint::new();
    ck.set_metadata("test", "tempfile");
    let pi = std::f32::consts::PI;
    let e = std::f32::consts::E;
    ck.add_tensor(CheckpointTensor::new("w", vec![pi, e], vec![2]));

    let tmp_path = std::env::temp_dir().join("oxibonsai_ck_test_save_load.bin");

    ck.save(&tmp_path).expect("save failed");
    let loaded = Checkpoint::load(&tmp_path).expect("load failed");

    assert_eq!(loaded.get_metadata("test"), Some("tempfile"));
    let t = loaded.get_tensor("w").expect("tensor 'w' not found");
    assert_eq!(t.data, vec![std::f32::consts::PI, std::f32::consts::E]);

    // Clean up.
    let _ = std::fs::remove_file(&tmp_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. checkpoint_invalid_magic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_invalid_magic() {
    let bad_bytes = b"NOPE\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = Checkpoint::read_from(&mut bad_bytes.as_slice());
    match result {
        Err(CheckpointError::InvalidMagic(m)) => {
            assert_eq!(&m, b"NOPE");
        }
        other => panic!("expected InvalidMagic, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. checkpoint_unsupported_version
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_unsupported_version() {
    // Build a byte stream with the correct magic but version = 99.
    let mut bytes = Vec::<u8>::new();
    bytes.extend_from_slice(b"OXCK");
    bytes.extend_from_slice(&99u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // flags
    bytes.extend_from_slice(&0u64.to_le_bytes()); // num_tensors
    bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata_len

    let result = Checkpoint::read_from(&mut bytes.as_slice());
    match result {
        Err(CheckpointError::UnsupportedVersion(99)) => {}
        other => panic!("expected UnsupportedVersion(99), got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. checkpoint_truncated_data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_truncated_data() {
    // Serialize a real checkpoint then truncate the last few bytes.
    let mut ck = Checkpoint::new();
    ck.add_tensor(CheckpointTensor::new("x", vec![1.0f32, 2.0, 3.0], vec![3]));

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    // Lop off the last 8 bytes (part of the tensor data).
    let mut reader = &buf[..buf.len().saturating_sub(8)];

    let result = Checkpoint::read_from(&mut reader);
    // Must be an error; must not panic.
    assert!(result.is_err(), "expected error on truncated input");
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. checkpoint_from_to_weight_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_from_to_weight_tensor() {
    let wt = WeightTensor::new("layer.weight", vec![0.5f32, -0.5, 1.0, -1.0], vec![2, 2]);

    let ct = CheckpointTensor::from_weight_tensor(&wt);
    assert_eq!(ct.name, "layer.weight");
    assert_eq!(ct.shape, vec![2u64, 2]);
    assert_eq!(ct.data, vec![0.5f32, -0.5, 1.0, -1.0]);

    let wt2 = ct.to_weight_tensor();
    assert_eq!(wt2.name, "layer.weight");
    assert_eq!(wt2.shape, vec![2usize, 2]);
    assert_eq!(wt2.data, wt.data);
}

// ─────────────────────────────────────────────────────────────────────────────
// 19. checkpoint_shape_data_mismatch_undersized_data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_shape_data_mismatch_undersized_data() {
    // Hand-craft a stream whose tensor declares shape [4, 4] (16 elements) but
    // only actually carries 3 f32 values. Without cross-validation this would
    // construct a `CheckpointTensor` whose data is shorter than its shape
    // implies, which later panics with an out-of-bounds slice index in
    // shape-driven consumers (e.g. structured pruning) instead of surfacing a
    // typed error.
    let mut bytes = Vec::<u8>::new();
    bytes.extend_from_slice(b"OXCK");
    bytes.extend_from_slice(&1u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // flags
    bytes.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata_len = 0

    let name = b"bad_tensor";
    bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
    bytes.extend_from_slice(name);

    // shape = [4, 4] -> claims 16 elements
    bytes.extend_from_slice(&2u32.to_le_bytes()); // ndim = 2
    bytes.extend_from_slice(&4u64.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes());

    // data_len = 3, only 3 f32 values actually follow
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for v in [1.0f32, 2.0, 3.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let result = Checkpoint::read_from(&mut bytes.as_slice());
    match result {
        Err(CheckpointError::ShapeDataMismatch {
            name,
            shape,
            data_len,
        }) => {
            assert_eq!(name, "bad_tensor");
            assert_eq!(shape, vec![4u64, 4]);
            assert_eq!(data_len, 3);
        }
        other => panic!("expected ShapeDataMismatch, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 20. checkpoint_shape_data_mismatch_oversized_data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_shape_data_mismatch_oversized_data() {
    // Same idea but in the other direction: shape claims [2] (2 elements) while
    // 4 f32 values actually follow. Must also be rejected, not silently
    // truncated/ignored.
    let mut bytes = Vec::<u8>::new();
    bytes.extend_from_slice(b"OXCK");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata_len = 0

    let name = b"oversized";
    bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
    bytes.extend_from_slice(name);

    bytes.extend_from_slice(&1u32.to_le_bytes()); // ndim = 1
    bytes.extend_from_slice(&2u64.to_le_bytes()); // shape = [2]

    bytes.extend_from_slice(&4u64.to_le_bytes()); // data_len = 4
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let result = Checkpoint::read_from(&mut bytes.as_slice());
    assert!(
        matches!(result, Err(CheckpointError::ShapeDataMismatch { .. })),
        "expected ShapeDataMismatch for oversized data, got {result:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 21. checkpoint_shape_data_mismatch_empty_shape_nonempty_data
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_shape_data_mismatch_empty_shape_nonempty_data() {
    // An empty shape (ndim = 0) denotes zero elements per
    // `CheckpointTensor::element_count()`'s documented convention, so any
    // non-empty data for a shapeless tensor must also be rejected.
    let mut bytes = Vec::<u8>::new();
    bytes.extend_from_slice(b"OXCK");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
    bytes.extend_from_slice(&0u32.to_le_bytes()); // metadata_len = 0

    let name = b"shapeless";
    bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
    bytes.extend_from_slice(name);

    bytes.extend_from_slice(&0u32.to_le_bytes()); // ndim = 0 (no shape dims)

    bytes.extend_from_slice(&1u64.to_le_bytes()); // data_len = 1
    bytes.extend_from_slice(&1.0f32.to_le_bytes());

    let result = Checkpoint::read_from(&mut bytes.as_slice());
    assert!(
        matches!(result, Err(CheckpointError::ShapeDataMismatch { .. })),
        "expected ShapeDataMismatch for empty shape with non-empty data, got {result:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 22. checkpoint_shape_data_match_round_trips_cleanly
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn checkpoint_shape_data_match_round_trips_cleanly() {
    // Sanity check: well-formed tensors (shape product == data.len()) must
    // still read back successfully after adding the cross-validation guard.
    let mut ck = Checkpoint::new();
    ck.add_tensor(CheckpointTensor::new(
        "ok_tensor",
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    ));

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.tensors.len(), 1);
    assert_eq!(loaded.tensors[0].data.len(), 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. metadata_serialize_deserialize
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metadata_serialize_deserialize() {
    // Exercise the internal round-trip indirectly via Checkpoint I/O.
    let mut ck = Checkpoint::new();
    ck.set_metadata("a", "hello");
    ck.set_metadata("b", "world 42");
    ck.set_metadata("c", "3.14");

    let mut buf = Vec::<u8>::new();
    ck.write_to(&mut buf).expect("write failed");

    let loaded = Checkpoint::read_from(&mut buf.as_slice()).expect("read failed");
    assert_eq!(loaded.get_metadata("a"), Some("hello"));
    assert_eq!(loaded.get_metadata("b"), Some("world 42"));
    assert_eq!(loaded.get_metadata("c"), Some("3.14"));
    assert_eq!(loaded.metadata.len(), 3);
}
