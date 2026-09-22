//! Integration tests for the GGUF metadata array accessors (`core-gguf-08`)
//! and the streaming parser's `feed()` contract, resumable-array parsing,
//! and tensor-dimension cap (`core-gguf-14`, `core-gguf-16`).
//!
//! These exercise the crate's public API the way a real caller would,
//! complementing the unit tests co-located with the implementation in
//! `src/gguf/metadata.rs` and `src/gguf/streaming.rs`.

use std::time::{Duration, Instant};

use oxibonsai_core::{BonsaiError, GgufStreamParser, GgufValue, MetadataStore, StreamedGguf};

/// GGUF magic number "GGUF" in little-endian bytes.
const GGUF_MAGIC_LE: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

fn push_gguf_string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

fn push_scalar_u8(bytes: &mut Vec<u8>, key: &str, value: u8) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&0u32.to_le_bytes()); // GgufValueType::Uint8
    bytes.push(value);
}

fn push_scalar_i8(bytes: &mut Vec<u8>, key: &str, value: i8) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // GgufValueType::Int8
    bytes.push(value.to_le_bytes()[0]);
}

fn push_scalar_u16(bytes: &mut Vec<u8>, key: &str, value: u16) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&2u32.to_le_bytes()); // GgufValueType::Uint16
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_scalar_i16(bytes: &mut Vec<u8>, key: &str, value: i16) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&3u32.to_le_bytes()); // GgufValueType::Int16
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32_array(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&9u32.to_le_bytes()); // GgufValueType::Array
    bytes.extend_from_slice(&5u32.to_le_bytes()); // element type: Int32
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&9u32.to_le_bytes()); // GgufValueType::Array
    bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: String
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        push_gguf_string(bytes, v);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MetadataStore: arr[i32], arr[str], and small-int scalars (core-gguf-08)
// ─────────────────────────────────────────────────────────────────────────

/// Builds a metadata block shaped like the real Bonsai 2 27B GGUF's
/// `prism.hadamard.*` keys (an `arr[i32]` matching the confirmed real
/// `sign_values` prefix `[-1,-1,-1,1,-1,1]`, and an `arr[str]` matching the
/// shape of `weight_names`), plus every small-integer scalar type that had
/// no reader at all before this fix (`Uint8`/`Int8`/`Uint16`/`Int16`), and
/// reads every one back with the correct sign and value.
#[test]
fn metadata_store_reads_i32_array_string_array_and_small_int_scalars() {
    let mut bytes = Vec::new();
    push_i32_array(
        &mut bytes,
        "prism.hadamard.sign_values",
        &[-1, -1, -1, 1, -1, 1],
    );
    push_string_array(
        &mut bytes,
        "prism.hadamard.weight_names",
        &["output.weight", "attn_q.weight"],
    );
    push_scalar_u8(&mut bytes, "u8_val", 250);
    push_scalar_i8(&mut bytes, "i8_val", -120);
    push_scalar_u16(&mut bytes, "u16_val", 65_000);
    push_scalar_i16(&mut bytes, "i16_val", -32_000);

    let (store, _) = MetadataStore::parse(&bytes, 0, 6).expect("metadata block should parse");
    assert_eq!(store.len(), 6);

    assert_eq!(
        store
            .get_i32_array("prism.hadamard.sign_values")
            .expect("sign_values should read as an i32 array"),
        vec![-1, -1, -1, 1, -1, 1]
    );
    assert_eq!(
        store
            .get_string_array("prism.hadamard.weight_names")
            .expect("weight_names should read as a string array"),
        vec!["output.weight".to_string(), "attn_q.weight".to_string()]
    );

    let u8_val = store.get("u8_val").expect("u8_val should exist");
    assert_eq!(u8_val.as_u32(), Some(250));
    assert_eq!(u8_val.as_i32(), Some(250));
    assert_eq!(u8_val.as_f32(), Some(250.0));

    let i8_val = store.get("i8_val").expect("i8_val should exist");
    assert_eq!(
        i8_val.as_i32(),
        Some(-120),
        "sign must survive the round trip"
    );
    assert_eq!(
        i8_val.as_u32(),
        None,
        "a negative value must never silently become a large unsigned one"
    );

    let u16_val = store.get("u16_val").expect("u16_val should exist");
    assert_eq!(u16_val.as_u32(), Some(65_000));
    assert_eq!(u16_val.as_i32(), Some(65_000));

    let i16_val = store.get("i16_val").expect("i16_val should exist");
    assert_eq!(
        i16_val.as_i32(),
        Some(-32_000),
        "sign must survive the round trip"
    );
    assert_eq!(i16_val.as_u32(), None);
    assert_eq!(i16_val.as_u64(), None);
}

// ─────────────────────────────────────────────────────────────────────────
// Streaming parser: resumable arrays, feed() progress contract
// (core-gguf-14)
// ─────────────────────────────────────────────────────────────────────────

/// Builds a full GGUF byte stream: header + one metadata entry whose value
/// is a `count`-element string array, and zero tensors. Mirrors the real
/// `tokenizer.ggml.tokens` entry in the Bonsai 2 27B GGUF (248320 short
/// strings as a single top-level array KV) — the scenario `core-gguf-14`
/// measured as quadratic under the pre-fix restart-at-offset-0 parser.
fn build_string_array_gguf(count: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&GGUF_MAGIC_LE);
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

    push_gguf_string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes()); // GgufValueType::Array
    bytes.extend_from_slice(&8u32.to_le_bytes()); // element type: String
    bytes.extend_from_slice(&(count as u64).to_le_bytes());
    for i in 0..count {
        push_gguf_string(&mut bytes, &format!("tok_{i:06}"));
    }
    bytes
}

/// Feeds `bytes` through a fresh [`GgufStreamParser`] in `chunk_size`-byte
/// pieces and returns the finished result.
fn parse_all_via_chunks(bytes: &[u8], chunk_size: usize) -> StreamedGguf {
    let mut parser = GgufStreamParser::new();
    for chunk in bytes.chunks(chunk_size) {
        parser.feed(chunk).expect("chunked feed should not error");
    }
    assert!(parser.is_complete(), "parser should reach completion");
    parser.finish().expect("finish should succeed")
}

/// (i) A 250000-element string array — the real scale of
/// `tokenizer.ggml.tokens` in the Bonsai 2 27B GGUF — fed through 4 KiB
/// chunks must parse completely and correctly.
#[test]
fn streaming_250k_string_array_in_4kib_chunks_parses_correctly() {
    const COUNT: usize = 250_000;
    let bytes = build_string_array_gguf(COUNT);

    let result = parse_all_via_chunks(&bytes, 4096);
    assert_eq!(result.metadata.len(), 1);
    assert_eq!(result.metadata[0].0, "tokenizer.ggml.tokens");
    match &result.metadata[0].1 {
        GgufValue::Array(arr) => {
            assert_eq!(arr.len(), COUNT);
            match &arr[0] {
                GgufValue::String(s) => assert_eq!(s, "tok_000000"),
                other => panic!("expected String, got: {other:?}"),
            }
            match &arr[COUNT / 2] {
                GgufValue::String(s) => assert_eq!(s, &format!("tok_{:06}", COUNT / 2)),
                other => panic!("expected String, got: {other:?}"),
            }
            match &arr[COUNT - 1] {
                GgufValue::String(s) => assert_eq!(s, &format!("tok_{:06}", COUNT - 1)),
                other => panic!("expected String, got: {other:?}"),
            }
        }
        other => panic!("expected Array, got: {other:?}"),
    }
}

/// (iii) A `feed()` call landing entirely inside an in-progress array
/// element's 8-byte string-length prefix — after an earlier element of the
/// same array already parsed — makes no progress and must return `Ok(0)`.
#[test]
fn feed_returns_zero_when_chunk_completes_no_new_array_element() {
    let mut header = Vec::new();
    header.extend_from_slice(&GGUF_MAGIC_LE);
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes());

    let mut array_header = Vec::new();
    push_gguf_string(&mut array_header, "names");
    array_header.extend_from_slice(&9u32.to_le_bytes()); // Array
    array_header.extend_from_slice(&8u32.to_le_bytes()); // String
    array_header.extend_from_slice(&2u64.to_le_bytes());

    let mut element0 = Vec::new();
    push_gguf_string(&mut element0, "first");
    let mut element1 = Vec::new();
    push_gguf_string(&mut element1, "second-element");

    let mut parser = GgufStreamParser::new();

    let mut first = Vec::new();
    first.extend_from_slice(&header);
    first.extend_from_slice(&array_header);
    first.extend_from_slice(&element0);
    first.extend_from_slice(&element1[..4]); // partial length prefix only
    let consumed = parser.feed(&first).expect("first feed should not error");
    assert_eq!(
        consumed,
        first.len(),
        "real progress (header + array header + one element) was made"
    );

    let consumed = parser
        .feed(&element1[4..6])
        .expect("second feed should not error");
    assert_eq!(
        consumed, 0,
        "a chunk landing inside an incomplete array-element length prefix must return Ok(0)"
    );
    assert!(!parser.is_complete());

    parser
        .feed(&element1[6..])
        .expect("final feed should not error");
    assert!(parser.is_complete());
}

/// (ii) Total work must be `O(n)`, not `O(n^2)`, in the element count: the
/// pre-fix parser re-scanned an in-progress array from its own start
/// (buffer offset 0) on every `feed()` call, so parsing a `2n`-element
/// array cost roughly `4x` a `n`-element one (`O(n)` calls each redoing up
/// to `O(n)` of work), not roughly `2x`. Asserts the generous-but-decisive
/// bound the package spec calls for: `time(2n) < 3 * time(n)`.
///
/// Each attempt takes the minimum of several interleaved samples per size
/// (never batching all of one size before the other) because a load spike
/// from an unrelated process on a shared build machine can only ever make
/// a single run slower, never faster — the minimum across samples is the
/// closest observable proxy for the algorithm's actual cost, and
/// interleaving spreads any such spike over both sizes instead of letting
/// it land entirely on one side of the ratio.
///
/// On top of that, the whole measure-and-compare procedure is retried up
/// to [`MAX_ATTEMPTS`] times, succeeding as soon as one attempt satisfies
/// the bound — a deliberate, evidence-based departure from a single
/// pass/fail measurement (recorded in this package's `deviations`), not a
/// weakening of the property being checked. Be honest about the margin
/// this relies on: a quadratic implementation's ratio for doubling the
/// element count is *exactly* `4x` by definition (`(2n)^2 / n^2 = 4`),
/// which is precisely why the spec's bound sits at `3x` — the midpoint
/// between linear's `~2x` and quadratic's `~4x`. That margin is real but
/// not huge, so an occasional scheduling-noise spike can push a single
/// attempt's ratio just past `3x` even for a correct `O(n)` implementation
/// (observed directly: one attempt hit `3.224x` at load average ~19 on an
/// 8-core machine — see `deviations`), which is exactly the false
/// positive a retry should absorb. A genuinely quadratic implementation,
/// in contrast, would fail every attempt (its ratio sits near `4x`, not
/// marginally above `3x`) and would also take roughly two orders of
/// magnitude longer in absolute wall time at this problem size
/// (`core-gguf-14`'s evidence: ~159 ms for a single 248320-element
/// one-shot parse vs. an estimated ~1.6 s total when the same bytes were
/// re-scanned from scratch on every incoming chunk), so a false negative
/// from this retry design is implausible. If all attempts fail, the panic
/// message reports every attempt's numbers so a real regression is still
/// diagnosable.
#[test]
fn streaming_array_parse_time_is_linear_not_quadratic_in_element_count() {
    const N: usize = 250_000;
    const TWO_N: usize = 500_000;
    const SAMPLES_PER_ATTEMPT: usize = 5;
    const MAX_ATTEMPTS: usize = 5;

    let bytes_n = build_string_array_gguf(N);
    let bytes_two_n = build_string_array_gguf(TWO_N);

    let time_one = |bytes: &[u8], expected: usize| -> Duration {
        let start = Instant::now();
        let result = parse_all_via_chunks(bytes, 4096);
        let elapsed = start.elapsed();
        let actual = match &result.metadata[0].1 {
            GgufValue::Array(arr) => arr.len(),
            other => panic!("expected Array, got: {other:?}"),
        };
        assert_eq!(actual, expected);
        elapsed
    };

    // One discarded warm-up parse per size (page faults / allocator
    // warm-up should not count against any measurement below).
    let _ = time_one(&bytes_n, N);
    let _ = time_one(&bytes_two_n, TWO_N);

    let mut attempts = Vec::with_capacity(MAX_ATTEMPTS);
    for _ in 0..MAX_ATTEMPTS {
        let mut n_times = Vec::with_capacity(SAMPLES_PER_ATTEMPT);
        let mut two_n_times = Vec::with_capacity(SAMPLES_PER_ATTEMPT);
        for _ in 0..SAMPLES_PER_ATTEMPT {
            n_times.push(time_one(&bytes_n, N));
            two_n_times.push(time_one(&bytes_two_n, TWO_N));
        }

        let time_n = n_times
            .into_iter()
            .min()
            .expect("SAMPLES_PER_ATTEMPT samples were just pushed")
            // Floor the baseline so a near-zero measurement (timer
            // granularity, not real cost) cannot amplify noise into a
            // huge ratio on the other side of the comparison.
            .max(Duration::from_millis(1));
        let time_two_n = two_n_times
            .into_iter()
            .min()
            .expect("SAMPLES_PER_ATTEMPT samples were just pushed");

        if time_two_n < time_n * 3 {
            return; // This attempt satisfied the linear-scaling bound.
        }
        attempts.push((time_n, time_two_n));
    }

    let report: Vec<String> = attempts
        .iter()
        .enumerate()
        .map(|(i, (time_n, time_two_n))| {
            format!(
                "  attempt {}: n={N} best={time_n:?}, 2n={TWO_N} best={time_two_n:?}, ratio={:.2}x",
                i + 1,
                time_two_n.as_secs_f64() / time_n.as_secs_f64()
            )
        })
        .collect();
    panic!(
        "expected roughly-linear scaling (2n < 3x n) on at least one of {MAX_ATTEMPTS} attempts, \
         but every attempt exceeded it — a ratio consistently near or above 4x indicates the \
         O(n^2) restart-at-offset-0 regression this test guards against:\n{}",
        report.join("\n")
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Tensor dimension cap (core-gguf-16, streaming-parser half)
// ─────────────────────────────────────────────────────────────────────────

/// `GGML_MAX_DIMS` is 4; a tensor declaring 5 dimensions is invalid input
/// and the streaming parser must reject it outright as `InvalidMetadata`,
/// not silently accept it with dimensions `4..n_dims` dropped from a
/// `[u64; 4]`.
///
/// The batch (`TensorStore`/`GgufFile`) parser's matching half of this fix
/// lives in `crates/oxibonsai-core/src/gguf/tensor_info.rs` (owned by
/// `B2-01`). It originally used different wording than this package's own
/// `streaming.rs`; the two were unified to the identical message ("tensor
/// has {n} dimensions; GGML_MAX_DIMS is {max}") by the wave-2.5 addendum
/// (`B2-16`, item 4), so this now pins the exact text both parsers agree on
/// instead of only a substring of the old, streaming-only wording.
#[test]
fn streaming_parser_rejects_tensor_with_five_dimensions() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&GGUF_MAGIC_LE);
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    push_gguf_string(&mut bytes, "five_d.weight");
    bytes.extend_from_slice(&5u32.to_le_bytes()); // n_dims = 5
    for dim in [2u64, 3, 4, 5, 6] {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes()); // tensor type: F32
    bytes.extend_from_slice(&0u64.to_le_bytes()); // offset

    let mut parser = GgufStreamParser::new();
    match parser.feed(&bytes) {
        Err(BonsaiError::InvalidMetadata { key, reason }) => {
            assert_eq!(key, "five_d.weight");
            assert_eq!(
                reason, "tensor has 5 dimensions; GGML_MAX_DIMS is 4",
                "wording must match tensor_info.rs's identical check exactly"
            );
        }
        other => panic!("expected InvalidMetadata for a 5-D tensor, got: {other:?}"),
    }
}
