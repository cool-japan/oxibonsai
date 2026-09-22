//! Writer layout acceptance: alignment padding, the per-row size rule, and
//! the `wire_id()` audit.
//!
//! Three things this suite pins down:
//!
//! 1. **Alignment padding (core-gguf-10).** ggml lays each tensor at
//!    `GGML_PAD(running_sum, alignment)`; the writer used to lay them
//!    back-to-back, producing offsets llama.cpp hard-rejects
//!    (`gguf.cpp:780`) and that our own `slice_from_bytes` alignment guards
//!    would refuse.
//! 2. **Per-row sizing (core-gguf-N1).** A quantized tensor whose `ne0` is
//!    not a whole number of blocks cannot be expressed in ggml at all and
//!    must be refused at write time, not silently emitted.
//! 3. **`wire_id()` (§8.2).** The `Q2_0_g64` sentinel discriminant
//!    (`0x4000_002A`) must never reach the tensor-info record; the byte on
//!    disk has to be exactly 42, and 142/143 for the PrismML types.

use oxibonsai_core::gguf::metadata::MetadataValue;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::gguf::writer::{
    align_up, GgufWriter, MetadataWriteValue, TensorEntry, TensorType, WriteError,
};

// ─────────────────────────────────────────────────────────────────────────────
// A minimal, independent tensor-info reader
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawTensorInfo {
    name: String,
    shape: Vec<u64>,
    type_id: u32,
    offset: u64,
}

/// Walk the tensor-info directory straight off the bytes, without going
/// through `GgufTensorType`, so the assertion is on what is *literally on
/// disk* rather than on a value the reader re-derived.
fn raw_tensor_infos(bytes: &[u8]) -> Vec<RawTensorInfo> {
    let mut pos = 0usize;
    let rd_u32 = |pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(
            bytes[*pos..*pos + 4]
                .try_into()
                .expect("4 bytes available for a u32 field"),
        );
        *pos += 4;
        v
    };
    let rd_u64 = |pos: &mut usize| -> u64 {
        let v = u64::from_le_bytes(
            bytes[*pos..*pos + 8]
                .try_into()
                .expect("8 bytes available for a u64 field"),
        );
        *pos += 8;
        v
    };

    assert_eq!(rd_u32(&mut pos), 0x4655_4747, "GGUF magic");
    assert_eq!(rd_u32(&mut pos), 3, "GGUF version");
    let tensor_count = rd_u64(&mut pos);
    let kv_count = rd_u64(&mut pos);

    for _ in 0..kv_count {
        let key_len = rd_u64(&mut pos) as usize;
        pos += key_len;
        let vtype = rd_u32(&mut pos);
        skip_value(bytes, &mut pos, vtype);
    }

    let mut out = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name_len = rd_u64(&mut pos) as usize;
        let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
            .expect("tensor name is valid UTF-8");
        pos += name_len;
        let n_dims = rd_u32(&mut pos);
        let shape: Vec<u64> = (0..n_dims).map(|_| rd_u64(&mut pos)).collect();
        let type_id = rd_u32(&mut pos);
        let offset = rd_u64(&mut pos);
        out.push(RawTensorInfo {
            name,
            shape,
            type_id,
            offset,
        });
    }
    out
}

fn skip_value(bytes: &[u8], pos: &mut usize, vtype: u32) {
    match vtype {
        0 | 1 | 7 => *pos += 1,
        2 | 3 => *pos += 2,
        4..=6 => *pos += 4,
        10..=12 => *pos += 8,
        8 => {
            let len = u64::from_le_bytes(
                bytes[*pos..*pos + 8]
                    .try_into()
                    .expect("8 bytes for a string length"),
            ) as usize;
            *pos += 8 + len;
        }
        9 => {
            let elem = u32::from_le_bytes(
                bytes[*pos..*pos + 4]
                    .try_into()
                    .expect("4 bytes for an array element type"),
            );
            *pos += 4;
            let count = u64::from_le_bytes(
                bytes[*pos..*pos + 8]
                    .try_into()
                    .expect("8 bytes for an array count"),
            );
            *pos += 8;
            for _ in 0..count {
                skip_value(bytes, pos, elem);
            }
        }
        other => panic!("unknown metadata value type {other}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Alignment padding — core-gguf-10
// ─────────────────────────────────────────────────────────────────────────────

/// A deliberately ragged set: none of these tensors is a multiple of 32 bytes
/// on its own, so every following offset needs padding.
fn ragged_writer() -> GgufWriter<'static> {
    let mut w = GgufWriter::new();
    // 5 f32 = 20 bytes.
    w.add_tensor(TensorEntry {
        name: "a.weight".to_string(),
        shape: vec![5],
        tensor_type: TensorType::F32,
        data: vec![0u8; 20],
    });
    // 10 f32 = 40 bytes.
    w.add_tensor(TensorEntry {
        name: "b.weight".to_string(),
        shape: vec![10],
        tensor_type: TensorType::F32,
        data: vec![1u8; 40],
    });
    // 3 rows of 1 block of 34 bytes = 102 bytes.
    w.add_tensor(TensorEntry {
        name: "c.weight".to_string(),
        shape: vec![128, 3],
        tensor_type: TensorType::TQ2_0_g128,
        data: vec![2u8; 102],
    });
    // 7 f32 = 28 bytes.
    w.add_tensor(TensorEntry {
        name: "d.weight".to_string(),
        shape: vec![7],
        tensor_type: TensorType::F32,
        data: vec![3u8; 28],
    });
    // 2 rows of 1 block of 28 bytes = 56 bytes.
    w.add_tensor(TensorEntry {
        name: "e.weight".to_string(),
        shape: vec![128, 2],
        tensor_type: TensorType::PTQ1_0,
        data: vec![4u8; 56],
    });
    w
}

#[test]
fn every_tensor_offset_is_alignment_aligned_for_a_ragged_shape_set() {
    let bytes = ragged_writer().to_bytes().expect("write");
    let infos = raw_tensor_infos(&bytes);
    assert_eq!(infos.len(), 5);
    for info in &infos {
        assert_eq!(
            info.offset % 32,
            0,
            "tensor '{}' offset {} is not 32-aligned",
            info.name,
            info.offset
        );
    }
    // Exactly llama.cpp's invariant: offset[i] == Σ GGML_PAD(nbytes, 32).
    let sizes = [20u64, 40, 102, 28, 56];
    let mut running = 0u64;
    for (info, size) in infos.iter().zip(sizes.iter()) {
        assert_eq!(
            info.offset, running,
            "tensor '{}' must sit at the running padded sum",
            info.name
        );
        running = align_up(running + size, 32).expect("align");
    }
    assert_eq!(infos[1].offset, 32);
    assert_eq!(infos[2].offset, 96);
    assert_eq!(infos[3].offset, 224);
    assert_eq!(infos[4].offset, 256);
}

#[test]
fn padded_tensors_still_read_back_byte_for_byte() {
    let bytes = ragged_writer().to_bytes().expect("write");
    let parsed = GgufFile::parse(&bytes).expect("parse");
    for (name, fill, len) in [
        ("a.weight", 0u8, 20usize),
        ("b.weight", 1, 40),
        ("c.weight", 2, 102),
        ("d.weight", 3, 28),
        ("e.weight", 4, 56),
    ] {
        let data = parsed.tensor_data(name).expect("tensor data");
        assert_eq!(data.len(), len, "{name} length");
        assert!(data.iter().all(|&b| b == fill), "{name} contents");
    }
}

#[test]
fn a_non_default_alignment_is_honoured_too() {
    let mut w = ragged_writer();
    w.set_alignment(64);
    let bytes = w.to_bytes().expect("write");
    for info in raw_tensor_infos(&bytes) {
        assert_eq!(info.offset % 64, 0, "tensor '{}' at 64", info.name);
    }
    let parsed = GgufFile::parse(&bytes).expect("parse");
    assert_eq!(parsed.tensor_data("c.weight").expect("data").len(), 102);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-row sizing — core-gguf-N1
// ─────────────────────────────────────────────────────────────────────────────

/// `[100, 2]` with `TQ2_0_g128`: 100 is not a multiple of the 128-wide block,
/// so ggml cannot describe the tensor and the writer must refuse it. Before
/// this it serialised 68 bytes without complaint and `GgufFile::parse` read
/// it back `Ok`.
#[test]
fn writer_refuses_a_100_by_2_tq2_0_g128_tensor() {
    let mut w = GgufWriter::new();
    w.add_tensor(TensorEntry {
        name: "ragged.weight".to_string(),
        shape: vec![100, 2],
        tensor_type: TensorType::TQ2_0_g128,
        data: vec![0u8; 68],
    });
    match w.to_bytes() {
        Err(WriteError::TensorLayout { name, reason }) => {
            assert_eq!(name, "ragged.weight");
            assert!(reason.contains("100"), "reason: {reason}");
        }
        other => panic!("expected TensorLayout, got {other:?}"),
    }
}

/// A `[200, 3]` `TQ2_0_g128` tensor is 204 bytes (3 rows × 2 blocks × 34),
/// not the 170 the flattened formula gives. Note the writer refuses it
/// anyway because 200 is not a block multiple — the size rule and the
/// blocking rule are separate, and both must hold.
#[test]
fn per_row_size_and_blocking_rule_are_independent() {
    assert_eq!(TensorType::TQ2_0_g128.row_bytes(&[200, 3]), 204);
    assert_ne!(TensorType::TQ2_0_g128.row_bytes(&[200, 3]), 170);
    assert_eq!(
        GgufTensorType::TQ2_0_g128.block_size(),
        128,
        "200 % 128 != 0, so the writer must still refuse the shape"
    );

    let mut w = GgufWriter::new();
    w.add_tensor(TensorEntry {
        name: "t".to_string(),
        shape: vec![200, 3],
        tensor_type: TensorType::TQ2_0_g128,
        data: vec![0u8; 204],
    });
    assert!(matches!(w.to_bytes(), Err(WriteError::TensorLayout { .. })));
}

#[test]
fn a_block_aligned_ragged_row_count_round_trips_with_the_per_row_size() {
    // 384 = 3 blocks per row, 5 rows → 15 blocks → 510 bytes.
    let mut w = GgufWriter::new();
    w.add_tensor(TensorEntry {
        name: "w".to_string(),
        shape: vec![384, 5],
        tensor_type: TensorType::PQ2_0,
        data: vec![9u8; 510],
    });
    let bytes = w.to_bytes().expect("write");
    let parsed = GgufFile::parse(&bytes).expect("parse");
    let info = parsed.tensors.require("w").expect("tensor");
    assert_eq!(info.data_size(), 510);
    assert_eq!(parsed.tensor_data("w").expect("data").len(), 510);
}

// ─────────────────────────────────────────────────────────────────────────────
// wire_id() audit — §8.2
// ─────────────────────────────────────────────────────────────────────────────

/// Writing a `Q2_0_g64` tensor must put **exactly 42** in the type field, and
/// a `PQ2_0` / `PTQ1_0` tensor 142 / 143. The sentinel discriminant
/// `0x4000_002A` must appear nowhere in the file.
#[test]
fn wire_ids_round_trip_exactly_through_the_tensor_info_record() {
    let mut w = GgufWriter::new();
    // 64-wide blocks: 128 elements per row = 2 blocks = 36 bytes, 2 rows.
    w.add_tensor(TensorEntry {
        name: "g64.weight".to_string(),
        shape: vec![128, 2],
        tensor_type: TensorType::Q2_0G64,
        data: vec![0u8; 72],
    });
    w.add_tensor(TensorEntry {
        name: "pq2.weight".to_string(),
        shape: vec![128, 2],
        tensor_type: TensorType::PQ2_0,
        data: vec![0u8; 68],
    });
    w.add_tensor(TensorEntry {
        name: "ptq1.weight".to_string(),
        shape: vec![128, 2],
        tensor_type: TensorType::PTQ1_0,
        data: vec![0u8; 56],
    });
    w.add_tensor(TensorEntry {
        name: "legacy.weight".to_string(),
        shape: vec![128, 2],
        tensor_type: TensorType::TQ2_0_g128,
        data: vec![0u8; 68],
    });
    let bytes = w.to_bytes().expect("write");

    let infos = raw_tensor_infos(&bytes);
    let by_name = |n: &str| {
        infos
            .iter()
            .find(|i| i.name == n)
            .unwrap_or_else(|| panic!("tensor {n} missing"))
            .clone()
    };
    assert_eq!(by_name("g64.weight").type_id, 42, "Q2_0_g64 wire id");
    assert_eq!(by_name("pq2.weight").type_id, 142, "PQ2_0 wire id");
    assert_eq!(by_name("ptq1.weight").type_id, 143, "PTQ1_0 wire id");
    assert_eq!(by_name("legacy.weight").type_id, 42, "TQ2_0_g128 wire id");

    // The sentinel must not appear anywhere in the serialised bytes.
    let sentinel = 0x4000_002Au32.to_le_bytes();
    assert!(
        !bytes.windows(4).any(|w| w == sentinel),
        "the Q2_0_g64 sentinel discriminant leaked into the file"
    );
}

/// Every writer type's `wire_id()` must match the reader's view of the same
/// id, so a file this writer produces is read back as the type it declared
/// (modulo the deliberate id-42 ambiguity, which `quant_resolve` settles).
#[test]
fn writer_and_reader_type_tables_agree() {
    for (ty, id) in [
        (TensorType::F32, 0u32),
        (TensorType::F16, 1),
        (TensorType::Q4_0, 2),
        (TensorType::Q8_0, 8),
        (TensorType::Q4_K, 12),
        (TensorType::Q5_K, 13),
        (TensorType::Q6_K, 14),
        (TensorType::BF16, 30),
        (TensorType::TQ1_0, 34),
        (TensorType::TQ2_0, 35),
        (TensorType::MXFP4, 39),
        (TensorType::NVFP4, 40),
        (TensorType::Q1_0G128, 41),
        (TensorType::F8_E4M3, 43),
        (TensorType::F8_E5M2, 44),
        (TensorType::PQ2_0, 142),
        (TensorType::PTQ1_0, 143),
    ] {
        assert_eq!(ty.wire_id(), id, "{ty:?} wire id");
        let read = GgufTensorType::from_id(id)
            .unwrap_or_else(|e| panic!("reader rejects id {id} written for {ty:?}: {e}"));
        assert_eq!(
            read.block_size(),
            ty.block_size(),
            "{ty:?} block size disagrees with the reader"
        );
        assert_eq!(
            read.block_bytes(),
            ty.block_bytes(),
            "{ty:?} block bytes disagree with the reader"
        );
    }
    // The two id-42 writer spellings map onto the reader's two 42 readings.
    assert_eq!(TensorType::Q2_0G64.wire_id(), 42);
    assert_eq!(TensorType::Q2_0G64.block_bytes(), 18);
    assert_eq!(GgufTensorType::Q2_0G64.block_bytes(), 18);
    assert_eq!(TensorType::TQ2_0_g128.block_bytes(), 34);
    assert_eq!(GgufTensorType::TQ2_0_g128.block_bytes(), 34);
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata value coverage — core-gguf-09
// ─────────────────────────────────────────────────────────────────────────────

/// `prism.hadamard.sign_values` is `arr[i32]`; smuggling it through
/// `ArrayU32` turns every `-1` into `4294967295`.
#[test]
fn signed_arrays_survive_the_round_trip_as_signed() {
    // The first six values of the real PQ2_0 file's sign_values.
    let signs: Vec<i32> = vec![-1, -1, -1, 1, -1, 1];
    let mut w = GgufWriter::new();
    w.add_metadata(
        "prism.hadamard.sign_values",
        MetadataWriteValue::ArrayI32(signs.clone()),
    );
    w.add_metadata(
        "prism.hadamard.sign_widths",
        MetadataWriteValue::ArrayI32(vec![5120, 6144, 17408]),
    );
    w.add_metadata(
        "qwen35.rope.dimension_sections",
        MetadataWriteValue::ArrayI32(vec![11, 11, 10, 0]),
    );
    let bytes = w.to_bytes().expect("write");
    let parsed = GgufFile::parse(&bytes).expect("parse");

    match parsed.metadata.get("prism.hadamard.sign_values") {
        Some(MetadataValue::Array(items)) => {
            let got: Vec<i32> = items
                .iter()
                .map(|v| match v {
                    MetadataValue::Int32(i) => *i,
                    other => panic!("expected Int32, got {other:?}"),
                })
                .collect();
            assert_eq!(got, signs, "sign_values must round-trip as signed");
        }
        other => panic!("expected an array, got {other:?}"),
    }

    match parsed.metadata.get("qwen35.rope.dimension_sections") {
        Some(MetadataValue::Array(items)) => assert_eq!(items.len(), 4),
        other => panic!("expected an array, got {other:?}"),
    }
}

#[test]
fn every_new_metadata_variant_round_trips() {
    let mut w = GgufWriter::new();
    w.add_metadata("k.u8", MetadataWriteValue::U8(200));
    w.add_metadata("k.i8", MetadataWriteValue::I8(-100));
    w.add_metadata("k.u16", MetadataWriteValue::U16(65_000));
    w.add_metadata("k.i16", MetadataWriteValue::I16(-30_000));
    w.add_metadata("k.arr_u8", MetadataWriteValue::ArrayU8(vec![1, 2, 3]));
    w.add_metadata("k.arr_i64", MetadataWriteValue::ArrayI64(vec![-1, 2, -3]));
    w.add_metadata("k.arr_u64", MetadataWriteValue::ArrayU64(vec![1, 2, 3]));
    w.add_metadata("k.arr_f64", MetadataWriteValue::ArrayF64(vec![0.5, -0.25]));
    w.add_metadata(
        "k.arr_bool",
        MetadataWriteValue::ArrayBool(vec![true, false, true]),
    );
    let bytes = w.to_bytes().expect("write");
    let parsed = GgufFile::parse(&bytes).expect("parse");

    assert!(matches!(
        parsed.metadata.get("k.u8"),
        Some(MetadataValue::Uint8(200))
    ));
    assert!(matches!(
        parsed.metadata.get("k.i8"),
        Some(MetadataValue::Int8(-100))
    ));
    assert!(matches!(
        parsed.metadata.get("k.u16"),
        Some(MetadataValue::Uint16(65_000))
    ));
    assert!(matches!(
        parsed.metadata.get("k.i16"),
        Some(MetadataValue::Int16(-30_000))
    ));
    for (key, len) in [
        ("k.arr_u8", 3usize),
        ("k.arr_i64", 3),
        ("k.arr_u64", 3),
        ("k.arr_f64", 2),
        ("k.arr_bool", 3),
    ] {
        match parsed.metadata.get(key) {
            Some(MetadataValue::Array(items)) => assert_eq!(items.len(), len, "{key}"),
            other => panic!("{key}: expected an array, got {other:?}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// align_up is the one shared helper
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn writer_and_tensor_info_share_one_align_up() {
    for (offset, alignment, expect) in [
        (0u64, 32u64, Some(0u64)),
        (1, 32, Some(32)),
        (102, 32, Some(128)),
        (256, 32, Some(256)),
        (4, 0, None),
        (4, 3, None),
    ] {
        assert_eq!(
            align_up(offset, alignment),
            expect,
            "{offset} @ {alignment}"
        );
        assert_eq!(
            oxibonsai_core::gguf::tensor_info::align_up(offset, alignment),
            expect,
            "the reader-side spelling must be the same function"
        );
    }
}
