//! Shared "serve hardening" primitives used by both the standalone
//! `oxibonsai-serve` binary (`crates/oxibonsai-serve/src/hardening.rs`) and
//! the `oxibonsai serve` CLI subcommand (`src/cli/cmd_serve.rs` /
//! `src/cli/admission.rs`) -- two binaries in separate crates that cannot
//! depend on each other (findings `SV-30` / `sec-M3`). Both already depend
//! on this crate (`oxibonsai_runtime`) for other things (the middleware and
//! rate-limiter modules), so this module needs no new path dependency; it
//! sits next to [`crate::middleware`] and [`crate::rate_limiter`] as the
//! third piece of shared serve-binary plumbing, closing the "verbatim twin
//! implementations across two binaries" finding.
//!
//! One canonical implementation each of:
//! - [`resolve_admission_limit`] -- pool-size-derived admission ceiling
//!   (findings `sec-20` / `perf-M1`).
//! - [`validate_serve_params`] -- the three "serve hardening" CLI/config
//!   invariants (findings `SV-30` / `sec-M3`).
//! - [`lookup_expected_checksum`] / [`verify_model_checksum`] /
//!   [`sha256_file`] / [`compute_sha256_hex`] -- model checksum verification
//!   (finding `sec-12`), including a from-scratch streaming SHA-256. No new
//!   Cargo dependency is needed for this: FIPS 180-4 is ~150 lines of safe,
//!   dependency-free Rust, which is more in line with this workspace's
//!   Pure-Rust policy than routing a `sha2` dependency through `deny.toml`
//!   review for one call site.

use std::fs::File;
use std::io::Read;
use std::path::Path;

// ─── sec-20 / perf-M1: pool-size-derived admission ceiling ─────────────────

/// Derive the real admission ceiling from the engine pool's actual size:
/// measured serve throughput is strictly serialized on the GPU tier (pool
/// size clamped to 1), so admitting the configured
/// `limits.max_concurrent_requests` (default 32) against a single-replica
/// pool queues every excess request behind that one engine until the
/// per-request timeout fires, instead of shedding it immediately with a
/// fast `503` + `Retry-After`.
///
/// A small multiple of the pool size (4x) lets a modest burst queue briefly
/// without reproducing the "32 admitted, 1 engine, most of them time out"
/// pathology; `configured` still wins when it is the *smaller* value, so an
/// operator who deliberately tightened it below the pool-size heuristic is
/// never loosened back up.
///
/// The single canonical implementation for both `oxibonsai-serve`
/// (`crates/oxibonsai-serve/src/hardening.rs::build_router`, via a
/// module-level re-export) and `oxibonsai serve`
/// (`src/cli/cmd_serve.rs::harden_router`, via `src/cli/admission.rs`'s
/// re-export).
pub fn resolve_admission_limit(configured: usize, pool_size: usize) -> usize {
    const POOL_SIZE_MULTIPLE: usize = 4;
    configured
        .min(pool_size.saturating_mul(POOL_SIZE_MULTIPLE))
        .max(1)
}

// ─── SV-30 / sec-M3: shared serve-hardening parameter validation ──────────

/// Minimum bearer-token (and admin-token) length, in UTF-8 bytes.
pub const MIN_BEARER_TOKEN_LEN: usize = 16;

/// Validate the three "serve hardening" parameters both binaries accept
/// (findings `SV-30` / `sec-M3`): an unset-or-short bearer token is fine
/// (auth is optional), but a *configured* token must meet the minimum
/// length, and the two admission knobs must be positive --
/// `max_concurrent_requests == 0` builds a zero-permit semaphore that
/// `load_shed`s every request forever, and `request_timeout_ms == 0` fires
/// the timeout before any handler can complete.
///
/// The single canonical implementation for both `oxibonsai-serve`
/// (`crates/oxibonsai-serve/src/validation.rs::validate_serve_params`, used
/// by `ServerConfig::validate`, via a re-export) and `oxibonsai serve`
/// (`src/cli/admission.rs::validate_serve_args`, a re-export of this
/// function under its historical name).
pub fn validate_serve_params(
    bearer_token: Option<&str>,
    max_concurrent_requests: usize,
    request_timeout_ms: u64,
) -> Result<(), String> {
    if let Some(tok) = bearer_token {
        if tok.len() < MIN_BEARER_TOKEN_LEN {
            return Err(format!(
                "bearer token must be at least {MIN_BEARER_TOKEN_LEN} chars, got {}",
                tok.len()
            ));
        }
    }
    if max_concurrent_requests == 0 {
        return Err(
            "max_concurrent_requests must be >= 1 (0 rejects every request forever)".to_string(),
        );
    }
    if request_timeout_ms == 0 {
        return Err(
            "request_timeout_ms must be >= 1 (0 times out every request immediately)".to_string(),
        );
    }
    Ok(())
}

// ─── sec-12: model checksum verification ───────────────────────────────────

/// Look up `model_path`'s expected SHA-256 hex digest in `checksums_text`,
/// which uses the format both GNU `sha256sum` and BSD/macOS `shasum -a 256`
/// read and write: `<64-hex-chars>  <path-or-bare-filename>` (one or two
/// spaces; `#`-prefixed lines are comments). Matches by bare filename, since
/// `scripts/checksums.sha256` lists paths relative to `models/` while an
/// operator's model path may be absolute or relative to a different working
/// directory.
pub fn lookup_expected_checksum(checksums_text: &str, model_path: &Path) -> Option<String> {
    let file_name = model_path.file_name()?.to_str()?;
    for raw_line in checksums_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let hex = parts.next()?;
        let rest = parts.next()?.trim_start();
        if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let listed_file_name = Path::new(rest).file_name().and_then(|n| n.to_str());
        if listed_file_name == Some(file_name) {
            return Some(hex.to_ascii_lowercase());
        }
    }
    None
}

/// Encode `bytes` as a lowercase hex string, without going through
/// `format!`/`fmt::Write` (avoids both a heap-churning `format!` call per
/// byte and any `Result` to discard -- `fmt::Write` on a `String` never
/// fails, but a lookup table is simpler still).
fn to_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Compute a file's SHA-256 digest, reading it in bounded 64 KiB chunks --
/// never `read_to_string`/`fs::read` the whole file at once, since GGUFs are
/// multi-gigabyte.
pub fn sha256_file(path: &Path) -> std::io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

/// Compute a file's SHA-256 hex digest (finding `sec-12`).
///
/// Backed by [`sha256_file`], a from-scratch streaming FIPS 180-4
/// implementation (see the module docs for why no new Cargo dependency is
/// needed) -- this always returns `Ok(Some(_))` for a readable file. The
/// `Option` stays in the signature because [`verify_model_checksum`]'s
/// contract is "a digest that could not be computed must never block model
/// loading"; keeping the type honest here means a future backend change
/// that legitimately needs to report "unavailable" again would not have to
/// touch every call site.
pub fn compute_sha256_hex(path: &Path) -> std::io::Result<Option<String>> {
    let digest = sha256_file(path)?;
    Ok(Some(to_lower_hex(&digest)))
}

/// Verify `model_path` against the checksums file at `checksums_path`, when
/// one is present and lists a known entry for it (finding `sec-12`).
///
/// Never blocks loading when verification is simply *unavailable* -- no
/// checksums file, no matching entry, an unreadable model file, or (should
/// a future backend regress to it) no computable digest -- only a
/// **confirmed** mismatch is fatal, and only that case returns `Err`.
pub fn verify_model_checksum(model_path: &Path, checksums_path: &Path) -> Result<(), String> {
    let text = match std::fs::read_to_string(checksums_path) {
        Ok(t) => t,
        Err(_) => return Ok(()), // No checksums file at the configured/default location.
    };
    let Some(expected) = lookup_expected_checksum(&text, model_path) else {
        return Ok(()); // No known checksum for this specific file.
    };
    match compute_sha256_hex(model_path) {
        Ok(Some(actual)) if actual.eq_ignore_ascii_case(&expected) => Ok(()),
        Ok(Some(actual)) => Err(format!(
            "checksum mismatch for {}: expected {expected}, computed {actual} -- refusing to \
             load a model whose bytes do not match {}",
            model_path.display(),
            checksums_path.display()
        )),
        Ok(None) => {
            // Not reachable through `compute_sha256_hex`'s current
            // implementation (it always returns `Ok(Some(_))` on success),
            // but this call site still honors the full `Option` contract
            // rather than assuming today's implementation detail.
            tracing::warn!(
                path = %model_path.display(),
                checksums = %checksums_path.display(),
                "a known checksum exists for this model but no digest was returned for it; \
                 loading without verification"
            );
            Ok(())
        }
        Err(e) => {
            tracing::warn!(
                path = %model_path.display(),
                error = %e,
                "failed to read the model file while checksumming; loading without verification"
            );
            Ok(())
        }
    }
}

/// Minimal, dependency-free FIPS 180-4 SHA-256 (see the module docs for why
/// this workspace prefers this over a `sha2` dependency). Processes input in
/// 64-byte blocks behind a small carry buffer so it can be fed arbitrary
/// chunk sizes -- [`sha256_file`] streams 64 KiB `File::read`s through it.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    /// Feed `data` in, of any length, in any number of calls. Full 64-byte
    /// blocks are compressed immediately; a trailing partial block is
    /// carried in `self.buffer` until either a later call completes it or
    /// [`Self::finalize`] pads it.
    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);

        if self.buffer_len > 0 {
            let need = 64 - self.buffer_len;
            let take = need.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                Self::compress(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }

        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.state, &block);
            data = &data[64..];
        }

        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    /// Apply FIPS 180-4 padding (a `0x80` byte, zeros out to 56 mod 64, then
    /// the 8-byte big-endian bit length) to whatever is left in the carry
    /// buffer, compress the resulting one or two final blocks, and emit the
    /// big-endian digest.
    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);

        let mut tail = Vec::with_capacity(128);
        tail.extend_from_slice(&self.buffer[..self.buffer_len]);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bit_len.to_be_bytes());
        debug_assert_eq!(tail.len() % 64, 0);

        for chunk in tail.chunks_exact(64) {
            let mut block = [0u8; 64];
            block.copy_from_slice(chunk);
            Self::compress(&mut self.state, &block);
        }

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// The SHA-256 compression function: message schedule expansion plus 64
    /// rounds, mutating `state` in place.
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];

        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod admission_limit_tests {
    use super::*;

    #[test]
    fn admission_limit_is_clamped_by_a_small_pool() {
        // The exact regression: default 32, single-replica GPU-tier pool.
        assert_eq!(resolve_admission_limit(32, 1), 4);
    }

    #[test]
    fn admission_limit_respects_a_lower_explicit_configuration() {
        assert_eq!(resolve_admission_limit(2, 1), 2);
    }

    #[test]
    fn admission_limit_is_unaffected_by_a_large_enough_pool() {
        assert_eq!(resolve_admission_limit(32, 8), 32);
    }

    #[test]
    fn admission_limit_is_never_zero() {
        assert_eq!(resolve_admission_limit(0, 0), 1);
    }
}

#[cfg(test)]
mod validate_params_tests {
    use super::*;

    #[test]
    fn accepts_sane_defaults() {
        assert!(validate_serve_params(None, 32, 60_000).is_ok());
        assert!(validate_serve_params(Some(&"x".repeat(MIN_BEARER_TOKEN_LEN)), 1, 1).is_ok());
    }

    #[test]
    fn rejects_short_token() {
        let err = validate_serve_params(Some("short"), 32, 60_000).expect_err("must reject");
        assert!(err.contains("bearer token"));
    }

    #[test]
    fn rejects_zero_concurrency() {
        let err = validate_serve_params(None, 0, 60_000).expect_err("must reject");
        assert!(err.contains("max_concurrent_requests"));
    }

    #[test]
    fn rejects_zero_timeout() {
        let err = validate_serve_params(None, 32, 0).expect_err("must reject");
        assert!(err.contains("request_timeout_ms"));
    }
}

#[cfg(test)]
mod lookup_tests {
    use super::*;

    #[test]
    fn matches_by_bare_filename() {
        let text = "\
# comment line
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo-Q2_0.gguf
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  models/Bar.gguf
";
        let got = lookup_expected_checksum(text, Path::new("/somewhere/else/Bar.gguf"));
        assert_eq!(
            got,
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string())
        );
    }

    #[test]
    fn is_none_for_unknown_file() {
        let text =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
        assert!(lookup_expected_checksum(text, Path::new("models/Unknown.gguf")).is_none());
    }

    #[test]
    fn ignores_malformed_lines() {
        let text = "not-a-valid-hex-line models/Foo.gguf\ntoo-short  models/Foo.gguf\n";
        assert!(lookup_expected_checksum(text, Path::new("models/Foo.gguf")).is_none());
    }
}

#[cfg(test)]
mod sha256_tests {
    use super::*;

    /// A process-unique temp path for `label` (never collides across
    /// tests -- each test uses a distinct label, and `nextest` runs each
    /// test in its own process anyway).
    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxibonsai_runtime_serve_shared_sha256_{label}_{}",
            std::process::id()
        ))
    }

    fn hash_bytes(label: &str, data: &[u8]) -> [u8; 32] {
        let path = temp_path(label);
        std::fs::write(&path, data).expect("write temp file");
        let digest = sha256_file(&path).expect("hash temp file");
        let _ = std::fs::remove_file(&path);
        digest
    }

    // Every expected digest below was computed independently with the
    // system `shasum -a 256` (BSD/macOS) / `sha256sum` (GNU) tool, not
    // derived from this implementation -- these are ground-truth FIPS
    // 180-4 test vectors, not self-referential assertions.

    #[test]
    fn empty_input_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash_bytes("empty", b""), expected);
    }

    #[test]
    fn abc_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash_bytes("abc", b"abc"), expected);
    }

    /// The classic NIST two-block boundary vector: 56 bytes is the shortest
    /// message whose padding (a `0x80` byte plus the 8-byte length) no
    /// longer fits in the same 64-byte block as the message itself.
    #[test]
    fn fifty_six_byte_message_matches_the_known_two_block_vector() {
        let expected: [u8; 32] = [
            0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8, 0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e,
            0x60, 0x39, 0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67, 0xf6, 0xec, 0xed, 0xd4,
            0x19, 0xdb, 0x06, 0xc1,
        ];
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(msg.len(), 56);
        assert_eq!(hash_bytes("two_block", msg), expected);
    }

    /// Exactly one block (64 bytes) of message: the padding needs a whole
    /// second, otherwise-empty block -- the boundary case adjacent to the
    /// one above from the other side.
    #[test]
    fn sixty_four_byte_message_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0x7c, 0xe1, 0x00, 0x97, 0x1f, 0x64, 0xe7, 0x00, 0x1e, 0x8f, 0xe5, 0xa5, 0x19, 0x73,
            0xec, 0xdf, 0xe1, 0xce, 0xd4, 0x2b, 0xef, 0xe7, 0xee, 0x8d, 0x5f, 0xd6, 0x21, 0x95,
            0x06, 0xb5, 0x39, 0x3c,
        ];
        let msg = vec![b'x'; 64];
        assert_eq!(hash_bytes("sixty_four", &msg), expected);
    }

    /// 55 bytes: the longest message whose padding (`0x80` + zeros + 8-byte
    /// length) still fits inside a single 64-byte block.
    #[test]
    fn fifty_five_byte_message_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0xd5, 0xe2, 0x85, 0x68, 0x3c, 0xd4, 0xef, 0xc0, 0x2d, 0x02, 0x1a, 0x5c, 0x62, 0x01,
            0x46, 0x94, 0x95, 0x89, 0x01, 0x00, 0x5d, 0x6f, 0x71, 0xe8, 0x9e, 0x09, 0x89, 0xfa,
            0xc7, 0x7e, 0x40, 0x72,
        ];
        let msg = vec![b'x'; 55];
        assert_eq!(hash_bytes("fifty_five", &msg), expected);
    }

    /// One million `'a'` bytes -- the standard NIST "large data" vector,
    /// and also exactly block-aligned (1,000,000 % 64 == 0), so the padding
    /// must still open a fresh block.
    #[test]
    fn one_million_a_bytes_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0xcd, 0xc7, 0x6e, 0x5c, 0x99, 0x14, 0xfb, 0x92, 0x81, 0xa1, 0xc7, 0xe2, 0x84, 0xd7,
            0x3e, 0x67, 0xf1, 0x80, 0x9a, 0x48, 0xa4, 0x97, 0x20, 0x0e, 0x04, 0x6d, 0x39, 0xcc,
            0xc7, 0x11, 0x2c, 0xd0,
        ];
        assert_eq!(1_000_000 % 64, 0);
        let msg = vec![b'a'; 1_000_000];
        assert_eq!(hash_bytes("million_a", &msg), expected);
    }

    /// A 200,003-byte (deliberately *not* block- or chunk-aligned) file
    /// exercises [`sha256_file`]'s real `File::read` loop end to end --
    /// several full 64 KiB reads plus one short final read -- rather than a
    /// single in-memory `Sha256::update` call. This is the exact code path
    /// finding `sec-12` requires ("streaming a 64 KiB chunk loop over
    /// `File::read`").
    #[test]
    fn streamed_pattern_file_matches_the_known_vector() {
        let expected: [u8; 32] = [
            0xa2, 0x82, 0x39, 0x32, 0x36, 0xee, 0x5a, 0xb5, 0x79, 0x78, 0x88, 0xe8, 0xff, 0xa8,
            0xb5, 0x96, 0x56, 0x69, 0x98, 0xb4, 0x2e, 0x76, 0x44, 0xfe, 0x1f, 0xf9, 0x25, 0x6c,
            0x1b, 0xa7, 0xa6, 0x96,
        ];
        let data: Vec<u8> = (0..200_003usize).map(|i| (i % 256) as u8).collect();
        assert_eq!(hash_bytes("pattern", &data), expected);
    }

    #[test]
    fn compute_sha256_hex_lowercase_hex_encodes_the_digest() {
        let path = temp_path("hex_encoding");
        std::fs::write(&path, b"abc").expect("write temp file");
        let hex = compute_sha256_hex(&path)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            hex,
            concat!(
                "ba7816bf", "8f01cfea", "414140de", "5dae2223", "b00361a3", "96177a9c", "b410ff61",
                "f20015ad"
            )
        );
    }

    #[test]
    fn compute_sha256_hex_errors_on_a_missing_file() {
        let path = std::env::temp_dir().join(format!(
            "oxibonsai_runtime_serve_shared_sha256_definitely_missing_{}",
            std::process::id()
        ));
        assert!(compute_sha256_hex(&path).is_err());
    }
}

#[cfg(test)]
mod checksum_tests {
    use super::*;

    fn unique_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxibonsai_runtime_serve_shared_checksum_{label}_{}",
            std::process::id()
        ))
    }

    #[test]
    fn verify_is_a_noop_when_checksums_file_is_absent() {
        let dir = unique_dir("absent");
        let missing_checksums = dir.join("does-not-exist.sha256");
        let model = dir.join("Model.gguf");
        verify_model_checksum(&model, &missing_checksums)
            .expect("missing checksums file must not block loading");
    }

    #[test]
    fn verify_is_a_noop_when_no_entry_matches() {
        let dir = unique_dir("no_match");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let checksums_path = dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Other.gguf\n",
        )
        .expect("write checksums file");
        let model = dir.join("Model.gguf");
        verify_model_checksum(&model, &checksums_path)
            .expect("no matching entry must not block loading");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The equality branch had **zero** coverage before this fix: every
    /// prior test either had no matching entry, no checksums file, or (with
    /// the old `compute_sha256_hex` stub) always fell into the
    /// unavailable-digest branch regardless of whether the real digest
    /// actually matched.
    #[test]
    fn verify_succeeds_when_the_checksum_really_matches() {
        let dir = unique_dir("matches");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Model.gguf");
        std::fs::write(&model, b"the real bytes of a model file").expect("write model");
        let hex = compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        assert_eq!(hex.len(), 64);
        let checksums_path = dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        verify_model_checksum(&model, &checksums_path)
            .expect("a genuinely matching checksum must not block loading");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE regression test finding `sec-12` was missing: a known-good
    /// checksum entry that no longer matches the file on disk (corruption,
    /// truncation, a bad download) must be **fatal**, not a warning -- this
    /// is `verify_model_checksum`'s one fatal branch, and it was
    /// unreachable dead code while `compute_sha256_hex` always returned
    /// `Ok(None)`.
    #[test]
    fn verify_fails_when_a_known_good_entry_no_longer_matches_a_corrupted_file() {
        let dir = unique_dir("corrupted");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Model.gguf");

        // Write the "known-good" content and record its real checksum...
        std::fs::write(&model, b"the original, known-good model bytes").expect("write model");
        let good_hex = compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let checksums_path = dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{good_hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        // ...then corrupt the file in place, simulating a bad download or
        // on-disk bit rot: the checksums file still lists the ORIGINAL
        // (now stale) digest as "known-good".
        std::fs::write(
            &model,
            b"corrupted! these are not the original bytes at all",
        )
        .expect("corrupt model");

        let err = verify_model_checksum(&model, &checksums_path)
            .expect_err("a stale known-good entry against corrupted bytes must be fatal");
        assert!(
            err.contains("checksum mismatch"),
            "error should explain the mismatch: {err}"
        );
        assert!(
            err.contains(&good_hex),
            "error should include the expected digest: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A known checksum entry naming a file that cannot be opened at all
    /// (deleted between config validation and load, permissions, ...) must
    /// degrade to "loading without verification" (the `Err(e)` branch), not
    /// be treated as a mismatch.
    #[test]
    fn verify_degrades_gracefully_when_the_model_file_cannot_be_opened() {
        let dir = unique_dir("missing_model");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Model.gguf"); // Deliberately never created.
        let checksums_path = dir.join("checksums.sha256");
        std::fs::write(&checksums_path, format!("{}  Model.gguf\n", "c".repeat(64)))
            .expect("write checksums file");

        verify_model_checksum(&model, &checksums_path)
            .expect("an unreadable model file must not block loading (degrades to a warning)");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
