//! Fuzz-style tests for GGUF parser robustness using proptest.
//!
//! Ensures the parser handles adversarial and malformed input gracefully
//! (returning errors, never panicking or OOMing).
//!
//! # T-03: this file used to encode a wrong specification
//!
//! `random_tensor_type_id_no_panic` (and two sibling tests) used to hardcode
//! a stale, incomplete `known_ids` list (missing IDs 16-29, 34, 39, 40, 43,
//! 44, 142, 143 — i.e. every `oxibonsai_core::GgufTensorType` variant added
//! since the test was first written), which made it structurally incapable
//! of ever failing on the ids that actually matter (`P(random u32 hits one
//! of ~40 specific values)` is astronomically small either way, so the test
//! never meaningfully exercised the "known" branch at all). All three now
//! derive their expected id set from [`GgufTensorType::ALL`] itself via
//! [`known_wire_ids`], so they can never silently drift out of sync with the
//! real type table again, plus a new exhaustive (non-random) scan over the
//! small, dense id space where every interesting boundary actually lives.
//!
//! # cargo-fuzz target
//!
//! `crates/oxibonsai-core/fuzz/` is a standalone `cargo-fuzz` crate for
//! `GgufFile::parse`. It carries its own one-crate `[workspace]` (the same
//! technique `oxibonsai-testkit`'s `Cargo.toml` uses, so `cargo
//! build/test/clippy --workspace` from the repository root never sees it and
//! cannot regress): `crates/oxibonsai-core/fuzz/Cargo.toml`, a
//! `libfuzzer-sys` dependency, `crates/oxibonsai-core/fuzz/fuzz_targets/
//! gguf_parse.rs` (`fuzz_target!(|data: &[u8]| { let _ =
//! oxibonsai_core::gguf::reader::GgufFile::parse(data); });`), and a seed
//! corpus at `crates/oxibonsai-core/fuzz/corpus/gguf_parse/`. It needs the
//! nightly toolchain and `cargo install cargo-fuzz` to run
//! (`cargo +nightly fuzz run gguf_parse` from `crates/oxibonsai-core/`), so it
//! is not part of `cargo test --workspace`; `cargo check --manifest-path
//! crates/oxibonsai-core/fuzz/Cargo.toml` is the compile-only check.
//!
//! Separately, the crafted byte patterns an exploratory probe found
//! interesting are *also* ported below as permanent, deterministic
//! regression tests (the "── Crafted adversarial files" section):
//! odd/misaligned data offsets, overlapping tensors, a `u64::MAX`-dimension
//! tensor, a non-UTF-8 tensor name, and a `probe_compat` call against a file
//! claiming a 200 MB tensor name it does not contain. These run on every
//! `cargo test` (no nightly toolchain needed).
//!
//! These crafted cases are NOT the fuzz target's seed corpus.
//! `crates/oxibonsai-core/fuzz/corpus/gguf_parse/` holds 8 plain
//! header-shaped seeds (`empty`, `single_byte`, `magic_only`,
//! `bad_magic`, `unsupported_version`, `huge_tensor_count`,
//! `huge_metadata_kv_count`, `valid_empty_header`) and none of the five
//! crafted cases above. The two forms of coverage are complementary
//! (this file's crafted cases run deterministically on every `cargo test`;
//! the fuzz target explores its own, disjoint corpus under `cargo +nightly
//! fuzz run`); copying the five crafted files' exact byte patterns into the
//! corpus directory would let the fuzzer start from them.

use proptest::prelude::*;

use oxibonsai_core::gguf::header::GgufHeader;
use oxibonsai_core::gguf::metadata::MetadataStore;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::TensorStore;
use oxibonsai_core::gguf::types::{GgufTensorType, GgufValueType};

// ── Helper: build a valid GGUF header ────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747;

fn make_valid_header(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_kv_count.to_le_bytes());
    data
}

/// Every wire id [`GgufTensorType::from_id`] accepts, derived from the type
/// table itself (`GgufTensorType::ALL`) rather than hand-copied — so this
/// list cannot go stale the way the three-times-duplicated literal list it
/// replaces did (T-03). `wire_id()` collapses the `Q2_0G64`/`Q2_0G128DFirst`
/// sentinel discriminants back to `42`, matching what `from_id(42)` itself
/// resolves to, so a plain `BTreeSet` (dedup + fast `contains`) is exactly
/// the right shape.
fn known_wire_ids() -> std::collections::BTreeSet<u32> {
    GgufTensorType::ALL.iter().map(|ty| ty.wire_id()).collect()
}

// ── A minimal byte-builder for hand-crafted GGUF files ───────────────────
// (mirrors the exploratory probe tool that first found these cases)

/// Appends little-endian GGUF primitives to a byte buffer. Kept minimal and
/// local to this file rather than promoted to `oxibonsai-testkit`: every
/// method here is a one-line `to_le_bytes` wrapper, so the duplication cost
/// of *not* sharing it is lower than the coupling cost of sharing it.
struct GgufBytes(Vec<u8>);

impl GgufBytes {
    fn new() -> Self {
        let mut v = Vec::new();
        v.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        v.extend_from_slice(&3u32.to_le_bytes()); // version 3
        Self(v)
    }

    fn counts(mut self, tensor_count: u64, metadata_kv_count: u64) -> Self {
        self.0.extend_from_slice(&tensor_count.to_le_bytes());
        self.0.extend_from_slice(&metadata_kv_count.to_le_bytes());
        self
    }

    fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn u64(mut self, v: u64) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// A GGUF string: `u64` byte length, then the raw bytes (not
    /// necessarily valid UTF-8 — see [`Self::raw_str`]).
    fn str(self, s: &str) -> Self {
        self.raw_str(s.as_bytes())
    }

    fn raw_str(mut self, bytes: &[u8]) -> Self {
        self.0
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.0.extend_from_slice(bytes);
        self
    }

    fn raw(mut self, bytes: &[u8]) -> Self {
        self.0.extend_from_slice(bytes);
        self
    }

    /// A `general.alignment = value` (UINT32) metadata entry.
    fn alignment_metadata(self, value: u32) -> Self {
        self.str("general.alignment").u32(4).u32(value)
    }

    /// One tensor-info entry: 1-D shape `[dim]`, ggml type `type_id`, byte
    /// `offset` into the data section.
    fn tensor_1d(self, name: &str, dim: u64, type_id: u32, offset: u64) -> Self {
        self.str(name).u32(1).u64(dim).u32(type_id).u64(offset)
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }
}

// ── 1. Random byte sequences never panic ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn random_bytes_header_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        // Should return Err, never panic
        let _ = GgufHeader::parse(&data, 0);
    }

    #[test]
    fn random_bytes_full_parse_no_panic(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        // Full parser should also never panic on random data
        let _ = GgufFile::parse(&data);
    }

    #[test]
    fn random_bytes_metadata_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        // Metadata parser with random data + random count
        let _ = MetadataStore::parse(&data, 0, 1);
        let _ = MetadataStore::parse(&data, 0, 0);
    }

    #[test]
    fn random_bytes_tensor_store_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = TensorStore::parse(&data, 0, 1);
        let _ = TensorStore::parse(&data, 0, 0);
    }
}

// ── 2. Truncated data at various points returns errors ───────────────────

#[test]
fn truncated_header_empty() {
    let result = GgufHeader::parse(&[], 0);
    assert!(result.is_err(), "empty data should fail");
}

#[test]
fn truncated_header_partial_magic() {
    let data = GGUF_MAGIC.to_le_bytes();
    // Only 4 bytes: magic but no version
    let result = GgufHeader::parse(&data[..3], 0);
    assert!(result.is_err(), "3 bytes should fail");
}

#[test]
fn truncated_header_magic_only() {
    let data = GGUF_MAGIC.to_le_bytes();
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "magic only (4 bytes) should fail");
}

#[test]
fn truncated_header_no_counts() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // version
                                                 // Missing tensor_count and metadata_kv_count
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "header without counts should fail");
}

#[test]
fn truncated_header_partial_tensor_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // partial u64 tensor_count
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "partial tensor_count should fail");
}

// ── 3. Oversized metadata counts don't cause OOM ─────────────────────────

#[test]
fn oversized_metadata_count_returns_error() {
    // Header claims u64::MAX metadata entries, but only 24 bytes of header data
    let data = make_valid_header(3, 0, u64::MAX);
    let result = GgufFile::parse(&data);
    // Should fail because there's no actual metadata to parse, not OOM
    assert!(
        result.is_err(),
        "u64::MAX metadata count should fail gracefully"
    );
}

#[test]
fn oversized_tensor_count_returns_error() {
    // Header claims u64::MAX tensors, zero metadata
    let data = make_valid_header(3, u64::MAX, 0);
    let result = GgufFile::parse(&data);
    // Should fail since no tensor data follows
    assert!(
        result.is_err(),
        "u64::MAX tensor count should fail gracefully"
    );
}

#[test]
fn large_metadata_count_no_oom() {
    // Claim 1 billion metadata entries but provide no data
    let data = make_valid_header(3, 0, 1_000_000_000);
    let result = GgufFile::parse(&data);
    assert!(result.is_err(), "1B metadata entries should fail, not OOM");
}

// ── 4. Invalid UTF-8 in string metadata returns error ────────────────────

#[test]
fn invalid_utf8_string_metadata() {
    // Build a metadata entry with invalid UTF-8 bytes in the key
    let mut data = Vec::new();
    // String length = 4
    data.extend_from_slice(&4u64.to_le_bytes());
    // Invalid UTF-8 bytes (0xFF is never valid in UTF-8)
    data.extend_from_slice(&[0xFF, 0xFE, 0x80, 0x80]);
    // Value type = Uint32 (4)
    data.extend_from_slice(&4u32.to_le_bytes());
    // Value = 42
    data.extend_from_slice(&42u32.to_le_bytes());

    let result = MetadataStore::parse(&data, 0, 1);
    assert!(result.is_err(), "invalid UTF-8 key should return error");
}

#[test]
fn invalid_utf8_string_value() {
    // Valid key, but String value with invalid UTF-8
    let mut data = Vec::new();
    // Key: "test" (valid UTF-8)
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"test");
    // Value type = String (8)
    data.extend_from_slice(&8u32.to_le_bytes());
    // String length = 3
    data.extend_from_slice(&3u64.to_le_bytes());
    // Invalid UTF-8 bytes
    data.extend_from_slice(&[0xFF, 0xFE, 0xFD]);

    let result = MetadataStore::parse(&data, 0, 1);
    assert!(
        result.is_err(),
        "invalid UTF-8 string value should return error"
    );
}

// ── 5. Random tensor type IDs handled ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn random_tensor_type_id_no_panic(id in any::<u32>()) {
        // Should return Ok for known types, Err for unknown, never panic.
        let result = GgufTensorType::from_id(id);
        let expected = known_wire_ids().contains(&id);
        prop_assert_eq!(
            result.is_ok(),
            expected,
            "from_id({}) ok={} disagrees with GgufTensorType::ALL",
            id,
            result.is_ok()
        );
    }

    #[test]
    fn random_value_type_id_no_panic(id in any::<u32>()) {
        let result = GgufValueType::from_id(id);
        if id <= 12 {
            assert!(result.is_ok(), "value type {id} should parse");
        } else {
            assert!(result.is_err(), "value type {id} should fail");
        }
    }
}

/// Exhaustive (not random) scan of every id in the small, dense ranges
/// where a real boundary lives: the whole `u8` range (covers every
/// mainline/upstream ggml id, 0-40, plus the "just past" ids 41-44 this
/// project adds) and the 140-150 window around the PrismML extension ids
/// 142/143. Unlike the proptest above (`P(hit)` for any *specific* id is
/// astronomically small over `any::<u32>()`), this is the test that
/// actually would have caught 142/143 being forgotten — T-03's review
/// singled this out as the fix the finder's own two "already
/// correct" sibling tests both still missed (both omitted 43/44 too).
#[test]
fn exhaustive_type_id_scan_matches_known_wire_ids() {
    let known = known_wire_ids();
    for id in 0u32..=255 {
        assert_eq!(
            GgufTensorType::from_id(id).is_ok(),
            known.contains(&id),
            "id {id} (u8 range)"
        );
    }
    for id in 140u32..150 {
        assert_eq!(
            GgufTensorType::from_id(id).is_ok(),
            known.contains(&id),
            "id {id} (PrismML extension window)"
        );
    }
    // The set itself must be non-trivial and must contain every format this
    // session's target (Bonsai 2 27B) actually ships: PQ2_0 (142) and
    // PTQ1_0 (143).
    assert!(known.contains(&142), "PQ2_0 must be a known id");
    assert!(known.contains(&143), "PTQ1_0 must be a known id");
    assert!(known.contains(&43), "F8_E4M3 must be a known id");
    assert!(known.contains(&44), "F8_E5M2 must be a known id");
}

// ── 6. Alignment calculations with random offsets ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn alignment_never_decreases_offset(offset in 0usize..1_000_000, alignment in 1usize..1024) {
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        prop_assert!(aligned >= offset, "aligned offset must be >= original");
    }

    #[test]
    fn alignment_result_is_multiple(offset in 0usize..1_000_000, alignment_exp in 0u32..10) {
        let alignment = 1usize << alignment_exp; // power of 2
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        prop_assert_eq!(aligned % alignment, 0, "result must be multiple of alignment");
    }
}

// ── 7. Metadata with max u32/u64 values ──────────────────────────────────

#[test]
fn metadata_u32_max_value() {
    let mut data = Vec::new();
    // Key: "max_u32"
    let key = "max_u32";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Value type = Uint32 (4)
    data.extend_from_slice(&4u32.to_le_bytes());
    // Value = u32::MAX
    data.extend_from_slice(&u32::MAX.to_le_bytes());

    let (store, _) = MetadataStore::parse(&data, 0, 1).expect("max u32 value should parse");
    let val = store.get_u32("max_u32").expect("key should exist");
    assert_eq!(val, u32::MAX);
}

#[test]
fn metadata_u64_max_value() {
    let mut data = Vec::new();
    // Key: "max_u64"
    let key = "max_u64";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Value type = Uint64 (10)
    data.extend_from_slice(&10u32.to_le_bytes());
    // Value = u64::MAX
    data.extend_from_slice(&u64::MAX.to_le_bytes());

    let (store, _) = MetadataStore::parse(&data, 0, 1).expect("max u64 value should parse");
    let val = store.get_u64("max_u64").expect("key should exist");
    assert_eq!(val, u64::MAX);
}

#[test]
fn header_version_2_accepted() {
    let data = make_valid_header(2, 0, 0);
    let (header, _) = GgufHeader::parse(&data, 0).expect("version 2 should be accepted");
    assert_eq!(header.version, 2);
}

#[test]
fn header_version_3_accepted() {
    let data = make_valid_header(3, 0, 0);
    let (header, _) = GgufHeader::parse(&data, 0).expect("version 3 should be accepted");
    assert_eq!(header.version, 3);
}

#[test]
fn header_version_0_rejected() {
    let data = make_valid_header(0, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 0 should be rejected");
}

#[test]
fn header_version_1_rejected() {
    let data = make_valid_header(1, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 1 should be rejected");
}

#[test]
fn header_version_4_rejected() {
    let data = make_valid_header(4, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 4 should be rejected");
}

#[test]
fn empty_metadata_store_operations() {
    let store = MetadataStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.get("any_key").is_none());
    assert!(store.get_string("any_key").is_err());
    assert!(store.get_u32("any_key").is_err());
    assert!(store.get_u64("any_key").is_err());
    assert!(store.get_f32("any_key").is_err());
    assert_eq!(store.get_u32_or("any_key", 99), 99);
    assert!(
        (store.get_f32_or("any_key", std::f32::consts::PI) - std::f32::consts::PI).abs()
            < f32::EPSILON
    );
}

#[test]
fn tensor_store_empty_operations() {
    let store = TensorStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.get("any_name").is_none());
    assert!(store.require("any_name").is_err());
    assert!(store.sorted_names().is_empty());
    assert!(store.count_by_type().is_empty());
}

// ── Tensor type properties ───────────────────────────────────────────────

#[test]
fn all_known_tensor_types_have_properties() {
    // Was a stale, hand-copied 18-entry list (missing every id added since
    // it was written, including 43/44/142/143); now derived from the type
    // table itself so it can never silently fall out of date again (T-03).
    for &id in &known_wire_ids() {
        let ty = GgufTensorType::from_id(id).expect("known type should parse");
        assert!(ty.block_size() > 0, "block_size for {ty} must be > 0");
        assert!(ty.block_bytes() > 0, "block_bytes for {ty} must be > 0");
        assert!(!ty.name().is_empty(), "name for {ty} must not be empty");
    }
}

#[test]
fn q1_0_g128_is_only_one_bit() {
    for &id in &known_wire_ids() {
        let ty = GgufTensorType::from_id(id).expect("known type should parse");
        if id == 41 {
            assert!(ty.is_one_bit(), "Q1_0_g128 should be one_bit");
        } else {
            assert!(!ty.is_one_bit(), "{ty} (id {id}) should not be one_bit");
        }
    }
}

// ── 8. Truncated-header proptest (varied byte counts) ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Any slice shorter than a complete GGUF header (24 bytes) must return Err.
    #[test]
    fn prop_test_gguf_truncated_header(
        len in 0usize..24usize,
        fill in any::<u8>(),
    ) {
        let data = vec![fill; len];
        let result = GgufHeader::parse(&data, 0);
        prop_assert!(result.is_err(),
            "truncated header of {len} bytes should fail");
    }

    /// Supplying a wrong (non-GGUF) magic value must cause parse to fail.
    #[test]
    fn prop_test_gguf_random_magic(
        magic in any::<u32>().prop_filter("not GGUF magic", |&m| m != 0x4655_4747u32),
    ) {
        let mut data = Vec::with_capacity(24);
        data.extend_from_slice(&magic.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());         // version = 3
        data.extend_from_slice(&0u64.to_le_bytes());         // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes());         // metadata_kv_count = 0
        let result = GgufFile::parse(&data);
        prop_assert!(result.is_err(),
            "wrong magic {magic:#010x} should be rejected");
    }

    /// Valid GGUF magic with an implausibly large tensor count must return Err.
    #[test]
    fn prop_test_gguf_invalid_tensor_count(
        tensor_count in (1_000_000u64..=u64::MAX),
    ) {
        let data = make_valid_header(3, tensor_count, 0);
        let result = GgufFile::parse(&data);
        prop_assert!(result.is_err(),
            "giant tensor_count {tensor_count} should fail");
    }
}

/// Empty byte slice must return Err immediately without panicking.
#[test]
fn prop_test_gguf_empty_input() {
    let result = GgufFile::parse(&[]);
    assert!(result.is_err(), "empty input must return error");
    // Verify the header parser also rejects empty slices.
    let result2 = GgufHeader::parse(&[], 0);
    assert!(result2.is_err(), "GgufHeader must reject empty input");
}

/// Valid magic + version but random trailing body must return Err without panicking.
#[test]
fn prop_test_gguf_valid_magic_corrupted_body() {
    // Use a large claimed tensor/metadata count so the parser tries to read
    // more data than is available and returns an error.
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // correct magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&1_000u64.to_le_bytes()); // 1000 tensors (won't fit)
    data.extend_from_slice(&1_000u64.to_le_bytes()); // 1000 metadata entries
                                                     // Random trailing bytes that are nowhere near enough to satisfy the counts.
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33]);
    let result = GgufFile::parse(&data);
    assert!(result.is_err(), "corrupted body should return error");
}

// ── 9. Metadata string max-len handling ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// A metadata entry whose string key claims to be very long but provides
    /// insufficient data must return an error (not panic or allocate huge memory).
    #[test]
    fn prop_test_metadata_string_max_len(
        claimed_len in (1_000_000u64..=u64::MAX / 2),
    ) {
        let mut data = Vec::new();
        // Key length = claimed_len (far more than data provides).
        data.extend_from_slice(&claimed_len.to_le_bytes());
        // Only a few bytes of "key" data — far short of claimed_len.
        data.extend_from_slice(b"short");
        // Value type = Uint32, value = 0 (will never be reached).
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let result = MetadataStore::parse(&data, 0, 1);
        prop_assert!(result.is_err(),
            "claimed string len {claimed_len} with only 5 bytes should fail");
    }

    /// A metadata array entry that claims a huge element count must return an
    /// error without allocating enormous memory (OOM prevention).
    #[test]
    fn prop_test_metadata_array_overflow(
        array_count in (1_000_000u64..=u64::MAX / 2),
    ) {
        let mut data = Vec::new();
        // Key: "arr"
        let key = b"arr";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        // Value type = Array (9 in GGUF spec).
        data.extend_from_slice(&9u32.to_le_bytes());
        // Array element type = Uint32 (4).
        data.extend_from_slice(&4u32.to_le_bytes());
        // Element count = array_count (absurdly large).
        data.extend_from_slice(&array_count.to_le_bytes());
        // No actual elements follow — parser must detect truncation.

        let result = MetadataStore::parse(&data, 0, 1);
        prop_assert!(result.is_err(),
            "array with {array_count} claimed elements must fail, not OOM");
    }
}

// ── Crafted adversarial files (T-03 cargo-fuzz-target substitute) ────────
//
// See this file's header doc comment: these port the specific byte patterns
// a prior exploratory probe found interesting into permanent, deterministic
// regression tests.

/// `general.alignment = 1` plus a tensor at byte offset 1: the data section
/// starts at an odd, unaligned address. `validate_tensor_layout`
/// (core-gguf-03 / sec-13) must reject a tensor whose offset does not land
/// on an `alignment`-aligned boundary — this used to be reachable all the
/// way to `BlockQ1_0G128::slice_from_bytes` handing back a live reference
/// into a misaligned `mmap` slice (core-gguf-07 / sec-02, since fixed by
/// `from_bytes`'s own alignment check). Parsing must reject this file
/// outright, before any block-level code ever sees the misaligned pointer.
#[test]
fn crafted_odd_data_offset_with_alignment_one_is_rejected() {
    let data = GgufBytes::new()
        .counts(1, 1)
        .alignment_metadata(1)
        .tensor_1d("w", 256, 41, 1) // Q1_0_g128, 2 blocks, offset=1 (odd)
        .raw(&[0xABu8; 256]) // enough trailing bytes that truncation isn't the cause
        .finish();
    assert!(
        GgufFile::parse(&data).is_err(),
        "an odd, alignment-1 tensor offset must be rejected at parse time"
    );
}

/// Two tensors whose declared byte ranges overlap. `validate_tensor_layout`
/// requires each tensor to start exactly where the previous one's
/// `GGML_PAD(size, alignment)` says it ends, so an overlapping second
/// tensor must be rejected rather than silently accepted (aliased mutable
/// views into the same bytes would otherwise be obtainable via
/// `tensor_data` for two different tensor names).
#[test]
fn crafted_overlapping_tensors_are_rejected() {
    let data = GgufBytes::new()
        .counts(2, 0)
        .tensor_1d("a", 128, 41, 0) // Q1_0_g128, 1 block = 18 bytes, offset 0
        .tensor_1d("bb", 128, 41, 4) // overlaps `a`'s [0, 18) range
        .raw(&[0u8; 64])
        .finish();
    assert!(
        GgufFile::parse(&data).is_err(),
        "overlapping tensor byte ranges must be rejected"
    );
}

/// A 1-D shape of `u64::MAX` (the bit pattern other tools would read as a
/// negative dimension) must not panic, overflow-wrap into a small, plausible
/// element count, or allocate/read out of bounds.
#[test]
fn crafted_u64_max_dimension_does_not_panic_or_oob_read() {
    let data = GgufBytes::new()
        .counts(1, 0)
        .tensor_1d("w", u64::MAX, 41, 0)
        .raw(&[0u8; 64])
        .finish();
    // Never panics (a plain call, not `catch_unwind` — a panic here would
    // already fail the test process); either parse rejects it outright, or
    // (parse tolerates the declared shape and only `tensor_data` refuses
    // to hand back an absurd, out-of-file byte range) both must return Err,
    // never a valid slice into memory the file does not contain.
    match GgufFile::parse(&data) {
        Err(_) => {}
        Ok(file) => {
            assert!(
                file.tensor_data("w").is_err(),
                "a u64::MAX-element tensor cannot possibly fit in a 64-byte file"
            );
        }
    }
}

/// A tensor name containing invalid UTF-8 bytes must be rejected, matching
/// the existing metadata-key/value UTF-8 checks (`invalid_utf8_string_*`
/// above) but exercising `TensorStore::parse`'s own name-reading path
/// instead.
#[test]
fn crafted_tensor_name_with_invalid_utf8_is_rejected() {
    let data = GgufBytes::new()
        .counts(1, 0)
        .raw_str(&[0xFF, 0xFE]) // invalid UTF-8 tensor name, length-prefixed
        .u32(1)
        .u64(128)
        .u32(41)
        .u64(0)
        .raw(&[0u8; 64])
        .finish();
    assert!(
        GgufFile::parse(&data).is_err(),
        "a non-UTF-8 tensor name must be rejected, not panic or silently substitute"
    );
}

/// `probe_compat` must not hang or allocate ~200 MB when a file *claims* an
/// enormous tensor name length it does not actually contain (the file here
/// is a few dozen bytes total). Bounded wall-clock check rather than a
/// memory-allocation check, since the latter cannot be asserted portably.
#[test]
fn probe_compat_huge_claimed_tensor_name_does_not_hang() {
    let data = GgufBytes::new()
        .counts(1, 0)
        .u64(200 * 1024 * 1024) // claims a 200 MB tensor name
        .finish();
    let start = std::time::Instant::now();
    let result = GgufFile::probe_compat(&data);
    let elapsed = start.elapsed();
    assert!(
        result.is_err(),
        "a 200 MB claimed name the file cannot back must fail"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "probe_compat must fail fast on an unbacked huge claimed length, took {elapsed:?}"
    );
}
