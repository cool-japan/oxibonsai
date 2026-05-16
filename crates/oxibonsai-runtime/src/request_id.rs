//! Structured request identifiers for end-to-end tracing.
//!
//! Each request gets a 128-bit `RequestId` rendered as a 32-hex-character
//! UUIDv4-style string (with the version + variant bits set per RFC 4122).
//!
//! No external `uuid` or `rand` dependency is required: the generator is
//! a thread-safe SplitMix64 stream seeded from the process start time and
//! a per-thread counter, which is sufficient for trace-correlation purposes
//! (uniqueness within a single server lifetime).
//!
//! ## Wire format
//!
//! ```text
//! xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
//! ```
//!
//! The version nibble is fixed at `4` and the variant nibble starts with
//! `8`/`9`/`a`/`b` (RFC 4122 §4.4).
//!
//! ## Usage
//!
//! ```
//! use oxibonsai_runtime::request_id::RequestId;
//!
//! let a = RequestId::new();
//! let b = RequestId::new();
//! assert_ne!(a, b);
//! assert_eq!(a.as_hex().len(), 32);
//! assert_eq!(a.as_uuid().len(), 36);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Core type ─────────────────────────────────────────────────────────────

/// 128-bit request identifier rendered as RFC 4122 UUIDv4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId {
    high: u64,
    low: u64,
}

impl RequestId {
    /// Construct a fresh `RequestId` from the global generator.
    pub fn new() -> Self {
        let high = next_u64();
        let low = next_u64();
        Self::from_pair(high, low)
    }

    /// Construct a `RequestId` from raw 64-bit halves, applying the UUIDv4
    /// version (0x4) and variant (0b10) bits.
    pub fn from_pair(high: u64, low: u64) -> Self {
        // UUIDv4: high nibble of byte 6 = 0x4
        let high = (high & !0x0000_0000_0000_F000) | 0x0000_0000_0000_4000;
        // Variant: top two bits of byte 8 = 0b10 (i.e. nibble in {8,9,a,b})
        let low = (low & !0xC000_0000_0000_0000) | 0x8000_0000_0000_0000;
        Self { high, low }
    }

    /// 32-character lowercase hex (no dashes).
    pub fn as_hex(&self) -> String {
        format!("{:016x}{:016x}", self.high, self.low)
    }

    /// 36-character UUID format with dashes (`8-4-4-4-12`).
    pub fn as_uuid(&self) -> String {
        let h = self.as_hex();
        // Bytes 0-3 (8 hex), 4-5 (4), 6-7 (4), 8-9 (4), 10-15 (12)
        format!(
            "{}-{}-{}-{}-{}",
            &h[0..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..32]
        )
    }

    /// High 64 bits.
    pub fn high(&self) -> u64 {
        self.high
    }

    /// Low 64 bits.
    pub fn low(&self) -> u64 {
        self.low
    }

    /// Parse a 32-char hex string (no dashes) back into a [`RequestId`].
    ///
    /// Returns `None` if the input is malformed.
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 32 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let high = u64::from_str_radix(&s[0..16], 16).ok()?;
        let low = u64::from_str_radix(&s[16..32], 16).ok()?;
        Some(Self { high, low })
    }

    /// Parse a UUID-formatted string (with dashes) back into a [`RequestId`].
    pub fn from_uuid(s: &str) -> Option<Self> {
        if s.len() != 36 {
            return None;
        }
        // Strip the four dashes and dispatch to `from_hex`.
        let mut buf = String::with_capacity(32);
        for (i, c) in s.chars().enumerate() {
            match i {
                8 | 13 | 18 | 23 => {
                    if c != '-' {
                        return None;
                    }
                }
                _ => buf.push(c),
            }
        }
        Self::from_hex(&buf)
    }

    /// Return the raw 16 bytes of this request id in big-endian order
    /// (high half first).
    ///
    /// Equivalent to `[high.to_be_bytes(), low.to_be_bytes()].concat()`,
    /// without an allocation. Useful for binary protocols that store
    /// request ids alongside other binary payloads.
    pub fn as_bytes(&self) -> [u8; 16] {
        let h = self.high.to_be_bytes();
        let l = self.low.to_be_bytes();
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&h);
        out[8..].copy_from_slice(&l);
        out
    }

    /// Reconstruct a [`RequestId`] from its 16-byte big-endian representation.
    ///
    /// Note: this preserves the bytes as-is (UUIDv4 version + variant nibbles
    /// are NOT re-imposed), so round-tripping through `as_bytes -> from_bytes`
    /// yields the same id. To enforce the v4 layout from arbitrary bytes,
    /// pipe through [`RequestId::from_pair`] explicitly.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut h_arr = [0u8; 8];
        let mut l_arr = [0u8; 8];
        h_arr.copy_from_slice(&bytes[..8]);
        l_arr.copy_from_slice(&bytes[8..]);
        Self {
            high: u64::from_be_bytes(h_arr),
            low: u64::from_be_bytes(l_arr),
        }
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

// ─── Thread-safe SplitMix64 generator ──────────────────────────────────────

static GLOBAL_STATE: AtomicU64 = AtomicU64::new(0);

fn ensure_seeded() {
    if GLOBAL_STATE.load(Ordering::Relaxed) == 0 {
        // Seed from process start time; mix in a constant to ensure non-zero.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xa3b1_c4d5_e6f7_8901);
        let seed = nanos ^ 0x9E37_79B9_7F4A_7C15; // golden-ratio constant
        let _ = GLOBAL_STATE.compare_exchange(0, seed, Ordering::Relaxed, Ordering::Relaxed);
    }
}

fn next_u64() -> u64 {
    ensure_seeded();
    // SplitMix64 step on the global counter.
    let prev = GLOBAL_STATE.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    let mut z = prev.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_is_unique() {
        let mut set = HashSet::new();
        for _ in 0..2000 {
            let id = RequestId::new();
            assert!(set.insert(id), "duplicate request id observed");
        }
    }

    #[test]
    fn hex_is_32_chars() {
        let id = RequestId::new();
        let h = id.as_hex();
        assert_eq!(h.len(), 32);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn uuid_format_is_well_formed() {
        let id = RequestId::new();
        let s = id.as_uuid();
        assert_eq!(s.len(), 36);
        let parts: Vec<&str> = s.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
        // Version must be 4
        assert!(parts[2].starts_with('4'));
        // Variant must be 8/9/a/b
        let variant = parts[3].chars().next().expect("non-empty variant nibble");
        assert!(matches!(variant, '8' | '9' | 'a' | 'b'));
    }

    #[test]
    fn from_pair_sets_version_and_variant() {
        let id = RequestId::from_pair(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF);
        let h = id.as_hex();
        // Position 12 is the version nibble
        assert_eq!(&h[12..13], "4");
        // Position 16 is the variant nibble — must be one of 8,9,a,b
        let v = h.chars().nth(16).expect("variant nibble");
        assert!(matches!(v, '8' | '9' | 'a' | 'b'));
    }

    #[test]
    fn round_trip_hex() {
        let id = RequestId::new();
        let s = id.as_hex();
        let parsed = RequestId::from_hex(&s).expect("hex parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn round_trip_uuid() {
        let id = RequestId::new();
        let s = id.as_uuid();
        let parsed = RequestId::from_uuid(&s).expect("uuid parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn rejects_bad_hex() {
        assert!(RequestId::from_hex("").is_none());
        assert!(RequestId::from_hex("too-short").is_none());
        assert!(RequestId::from_hex(&"x".repeat(32)).is_none());
        // Wrong length but valid hex chars
        assert!(RequestId::from_hex(&"a".repeat(31)).is_none());
        assert!(RequestId::from_hex(&"a".repeat(33)).is_none());
    }

    #[test]
    fn rejects_bad_uuid() {
        assert!(RequestId::from_uuid("not-a-uuid").is_none());
        // Right length, wrong dash positions
        assert!(RequestId::from_uuid(&"a".repeat(36)).is_none());
        // Right length and dashes, but non-hex
        assert!(RequestId::from_uuid("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").is_none());
    }

    #[test]
    fn display_uses_uuid_format() {
        let id = RequestId::new();
        let s = format!("{id}");
        assert_eq!(s, id.as_uuid());
    }

    #[test]
    fn as_bytes_round_trip() {
        let id = RequestId::new();
        let bytes = id.as_bytes();
        let recovered = RequestId::from_bytes(bytes);
        assert_eq!(id, recovered);
    }

    #[test]
    fn as_bytes_big_endian_layout() {
        let id = RequestId::from_pair(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
        let bytes = id.as_bytes();
        // First 8 bytes are the high half in big-endian order.
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[1], 0x23);
        assert_eq!(bytes[6], 0x4D); // 0x4 nibble (UUIDv4 version) was set on byte 6
                                    // Byte 8 has the variant nibble set in the high 2 bits.
        let variant = bytes[8] >> 6;
        assert_eq!(variant, 0b10);
    }

    #[test]
    fn from_bytes_preserves_arbitrary_bytes() {
        // from_bytes does NOT re-impose the v4 layout — round-trip is exact.
        let bytes = [0u8; 16];
        let id = RequestId::from_bytes(bytes);
        assert_eq!(id.as_bytes(), bytes);
    }

    #[test]
    fn high_low_recoverable() {
        let id = RequestId::from_pair(0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210);
        // After version/variant masking, high() should still match the
        // exact 64-bit half stored.
        let h_hex = format!("{:016x}", id.high());
        let l_hex = format!("{:016x}", id.low());
        assert_eq!(id.as_hex(), format!("{h_hex}{l_hex}"));
    }

    #[test]
    fn concurrent_generation_is_unique() {
        use std::sync::Arc;
        use std::sync::Mutex;
        use std::thread;

        let collected = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let collected = Arc::clone(&collected);
            handles.push(thread::spawn(move || {
                let mut local = HashSet::new();
                for _ in 0..500 {
                    local.insert(RequestId::new());
                }
                let mut g = collected.lock().expect("lock poisoned");
                for id in local {
                    assert!(g.insert(id), "duplicate id from concurrent generation");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panic");
        }
        assert_eq!(collected.lock().expect("lock").len(), 8 * 500);
    }
}
