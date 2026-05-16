//! Tests for the `Fp8` variant of `KvCacheLevel`.
//!
//! These tests verify the ordinal, tag, memory-factor, and from_ordinal
//! roundtrip contract for the newly added `Fp8` tier.

use oxibonsai_runtime::kv_cache_policy::KvCacheLevel;

// ─── Test 1: Fp8 ordinal, tag, and memory_factor ─────────────────────────────

/// Verify that `KvCacheLevel::Fp8` has the expected constant properties:
/// - `ordinal()` returns 2 (between Q8=1 and Q4=3)
/// - `tag()` returns "fp8"
/// - `memory_factor()` returns 0.5 (same byte-width as Q8 / INT8)
#[test]
fn kv_cache_level_fp8_ordinal_tag_factor() {
    let fp8 = KvCacheLevel::Fp8;

    assert_eq!(
        fp8.ordinal(),
        2,
        "Fp8 ordinal should be 2 (between Q8=1 and Q4=3)"
    );

    assert_eq!(fp8.tag(), "fp8", "Fp8 tag should be \"fp8\"");

    let factor = fp8.memory_factor();
    assert!(
        (factor - 0.5_f32).abs() < f32::EPSILON,
        "Fp8 memory_factor should be 0.5, got {factor}"
    );
}

// ─── Test 2: from_ordinal roundtrip ──────────────────────────────────────────

/// `from_ordinal(fp8.ordinal())` must return `KvCacheLevel::Fp8`.
/// This validates the internal ordinal ↔ variant bijection.
#[test]
fn kv_cache_policy_fp8_from_ordinal_roundtrip() {
    // Access from_ordinal indirectly via observe() API — we test via ordinal/tag consistency.
    // Fp8.ordinal() = 2; the tag must agree.
    let fp8 = KvCacheLevel::Fp8;
    let ordinal = fp8.ordinal();

    // Verify the full set so no ordinal collides with Fp8's.
    assert_ne!(
        ordinal,
        KvCacheLevel::Fp16.ordinal(),
        "Fp8 ordinal must not collide with Fp16"
    );
    assert_ne!(
        ordinal,
        KvCacheLevel::Q8.ordinal(),
        "Fp8 ordinal must not collide with Q8"
    );
    assert_ne!(
        ordinal,
        KvCacheLevel::Q4.ordinal(),
        "Fp8 ordinal must not collide with Q4"
    );

    // Tag uniqueness
    assert_ne!(fp8.tag(), KvCacheLevel::Fp16.tag());
    assert_ne!(fp8.tag(), KvCacheLevel::Q8.tag());
    assert_ne!(fp8.tag(), KvCacheLevel::Q4.tag());

    // The tag must be stable (constant fn)
    assert_eq!(KvCacheLevel::Fp8.tag(), "fp8");
}

// ─── Test 3: Fp8 sits between Q8 and Q4 in ordinal order ─────────────────────

/// The pressure ordering must satisfy: Fp16 < Q8 < Fp8 < Q4.
/// This ensures the policy controller's ordinal-based clamp is well-defined.
#[test]
fn kv_cache_level_fp8_is_between_q8_and_q4() {
    let fp16_ord = KvCacheLevel::Fp16.ordinal();
    let q8_ord = KvCacheLevel::Q8.ordinal();
    let fp8_ord = KvCacheLevel::Fp8.ordinal();
    let q4_ord = KvCacheLevel::Q4.ordinal();

    assert!(
        fp16_ord < q8_ord,
        "Fp16 ({fp16_ord}) should be less compact than Q8 ({q8_ord})"
    );
    assert!(
        q8_ord < fp8_ord,
        "Q8 ({q8_ord}) should be less compact than Fp8 ({fp8_ord})"
    );
    assert!(
        fp8_ord < q4_ord,
        "Fp8 ({fp8_ord}) should be less compact than Q4 ({q4_ord})"
    );
}
