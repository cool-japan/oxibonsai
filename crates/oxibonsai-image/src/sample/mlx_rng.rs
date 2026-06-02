//! Byte-exact Pure-Rust port of MLX's Threefry-2x32 random-number generator.
//!
//! This reproduces, bit-for-bit, the CPU random path of MLX so the OxiBonsai
//! text-to-image pipeline can generate its own initial noise (and any other
//! `mx.random.*` draw) without dumping a golden `.npy`. Every function below is
//! a direct transliteration of the corresponding MLX C++ source — file and line
//! references are given inline so the port can be audited against upstream.
//!
//! Ported from the MLX source tree (`mlx/`):
//! - `mlx/random.cpp`
//!   - `key(seed)` (line 30): `{seed >> 32, seed & 0xffffffff}`.
//!   - `uniform` (lines 95-141): `bits / 4294967295.0`, then `min(u, nextafter(1,0))`,
//!     then `lo + (hi - lo) * u`.
//!   - `normal` (lines 174-204): `uniform` over `(nextafter(-1,0), 1.0)`, then
//!     `sqrt(2) * erfinv(u)`.
//! - `mlx/backend/cpu/threefry.cpp` — `threefry2x32_hash` (lines 7-29).
//! - `mlx/backend/cpu/primitives.cpp` — `RandomBits::eval_cpu` (lines 272-333):
//!   the counter→element fill layout for a single key.
//! - `mlx/backend/cpu/simd/math.h` — `erfinv` (lines 150-191): the two-branch
//!   polynomial (`abs(log(1 - a*a)) > 6.125` selects the `lhs`/`rhs` Horner form).
//!
//! All integer arithmetic is `u32` wrapping; all float arithmetic is `f32`,
//! matching MLX's `float32` random path exactly.

use std::f32::consts::SQRT_2;

/// The largest `f32` strictly below `1.0`, i.e. `std::nextafter(1.0f, 0.0f)` in
/// MLX (`mlx/random.cpp:125`). Used to clamp uniform samples so they stay in
/// `[low, high)`.
const NEXT_BELOW_ONE: f32 = 0.99999994_f32;

/// `std::nextafter(-1.0f, 0.0f)` — the lower bound MLX feeds into `erfinv` for
/// `normal` (`mlx/random.cpp:190`). Equal to `-NEXT_BELOW_ONE`.
const NEXT_ABOVE_NEG_ONE: f32 = -0.99999994_f32;

/// `std::numeric_limits<uint32_t>::max()` as the `f32` divisor MLX uses to map
/// raw bits into `[0, 1]` (`mlx/random.cpp:136`).
const U32_MAX_AS_F32: f32 = 4294967295.0_f32;

/// Build the 2-word Threefry key from a 64-bit seed.
///
/// Port of `mlx::core::random::key` (`mlx/random.cpp:30-34`):
/// `k1 = seed >> 32`, `k2 = seed & 0xffffffff`.
#[must_use]
pub fn key(seed: u64) -> [u32; 2] {
    let k1 = (seed >> 32) as u32;
    let k2 = seed as u32;
    [k1, k2]
}

/// The Threefry-2x32 hash: maps a `(key, count)` pair to two pseudo-random u32s.
///
/// Direct port of `threefry2x32_hash` (`mlx/backend/cpu/threefry.cpp:7-29`).
/// Rotation schedule `[[13,15,26,6],[17,29,16,24]]`, parity constant
/// `0x1BD11BDA`, key schedule `ks = {k0, k1, k0 ^ k1 ^ parity}`. All arithmetic
/// is `u32` wrapping; the rotate is a 32-bit left-rotate (`rotate_left`).
#[must_use]
pub fn threefry2x32(key: [u32; 2], count: [u32; 2]) -> [u32; 2] {
    const ROTATIONS: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];
    const PARITY: u32 = 0x1BD1_1BDA;

    let ks: [u32; 3] = [key[0], key[1], key[0] ^ key[1] ^ PARITY];

    // count.first += ks[0]; count.second += ks[1];
    let mut c0 = count[0].wrapping_add(ks[0]);
    let mut c1 = count[1].wrapping_add(ks[1]);

    for i in 0..5usize {
        for &r in &ROTATIONS[i % 2] {
            // count.first += count.second;
            c0 = c0.wrapping_add(c1);
            // count.second = rotl(count.second, r) ^ count.first;
            c1 = c1.rotate_left(r) ^ c0;
        }
        // count.first  += ks[(i + 1) % 3];
        c0 = c0.wrapping_add(ks[(i + 1) % 3]);
        // count.second += ks[(i + 2) % 3] + i + 1;
        c1 = c1
            .wrapping_add(ks[(i + 2) % 3])
            .wrapping_add((i as u32).wrapping_add(1));
    }

    [c0, c1]
}

/// Fill `n` `u32` outputs from a single key, reproducing MLX's
/// `RandomBits::eval_cpu` control flow (`mlx/backend/cpu/primitives.cpp:272-333`)
/// for the single-key, `width == 4` (u32) case.
///
/// For a u32 output the per-element layout is:
/// - `out_skip = ceil(bytes_per_key / 4) = n` (4 bytes per element).
/// - `half = out_skip / 2`, `even = out_skip % 2 == 0`.
/// - `count = {0, half + !even}` (the second counter starts past the midpoint,
///   and one further when `n` is odd).
/// - While `count.first + 1 < half`: one Threefry call writes a *pair*
///   `(out[count.first], out[count.second])`, advancing both counters.
/// - If `count.first < half`: one more call; its `.first` lands at
///   `out[count.first]`, its `.second` is written by `copy_remaining` at
///   `count.second`.
/// - If `!even` (odd `n`): a final call at `count = {half, 0}`; its `.first`
///   lands at `out[half]` (the middle element).
///
/// `copy_remaining(loc, v)` writes the whole `u32` when `4*loc + 4 <= 4*n`
/// (always true for u32 outputs in-bounds), so for this code path it is a plain
/// `out[loc] = v`. This control flow is the #1 byte-match risk: it determines
/// exactly which random word lands at which element.
#[must_use]
pub fn random_bits(n: usize, key: [u32; 2]) -> Vec<u32> {
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    // bytes_per_key = itemsize(=4) * n; out_skip = ceil(bytes_per_key / 4) = n.
    let out_skip = n;
    let half = out_skip / 2;
    let even = out_skip % 2 == 0;

    // copy_remaining for the u32-output case: 4*loc + 4 <= 4*n  <=>  loc < n,
    // which always holds for the indices used below, so it is a direct store.
    let mut c_first: u32 = 0;
    let mut c_second: u32 = half as u32 + u32::from(!even);

    // for (; count.first + 1 < half; count.first++, count.second++)
    while (c_first as usize) + 1 < half {
        let rb = threefry2x32(key, [c_first, c_second]);
        out[c_first as usize] = rb[0];
        out[c_second as usize] = rb[1];
        c_first = c_first.wrapping_add(1);
        c_second = c_second.wrapping_add(1);
    }

    // if (count.first < half) { ... ptr[count.first++] = rb.first; copy_remaining(count.second, rb.second); }
    if (c_first as usize) < half {
        let rb = threefry2x32(key, [c_first, c_second]);
        out[c_first as usize] = rb[0];
        c_first = c_first.wrapping_add(1);
        // copy_remaining(count.second, rb.second)
        if (c_second as usize) < n {
            out[c_second as usize] = rb[1];
        }
    }

    // if (!even) { count.second = 0; copy_remaining(half, threefry(...).first); }
    if !even {
        let rb = threefry2x32(key, [half as u32, 0]);
        // copy_remaining(half, rb.first): 4*half + 4 <= 4*n  <=>  half < n (true for odd n).
        if half < n {
            out[half] = rb[0];
        }
    }

    let _ = c_first;
    out
}

/// Draw `n` uniform `f32` samples in `[lo, hi)` from a single key.
///
/// Port of `random::uniform` (`mlx/random.cpp:95-141`): `u = bits / UINT32_MAX`,
/// `u = min(u, nextafter(1, 0))`, result `= lo + (hi - lo) * u`. The clamp keeps
/// every sample strictly below `hi`.
#[must_use]
pub fn uniform(n: usize, key: [u32; 2], lo: f32, hi: f32) -> Vec<f32> {
    let bits = random_bits(n, key);
    let range = hi - lo;
    bits.into_iter()
        .map(|b| {
            let u = (b as f32) / U32_MAX_AS_F32;
            let u = if u < NEXT_BELOW_ONE {
                u
            } else {
                NEXT_BELOW_ONE
            };
            range * u + lo
        })
        .collect()
}

/// Draw `n` standard-normal `f32` samples from a single key.
///
/// Port of `random::normal` (`mlx/random.cpp:174-204`): uniform over
/// `(nextafter(-1, 0), 1.0)`, then `sqrt(2) * erfinv(u)`. (`loc = 0`,
/// `scale = 1`.)
#[must_use]
pub fn normal(n: usize, key: [u32; 2]) -> Vec<f32> {
    let u = uniform(n, key, NEXT_ABOVE_NEG_ONE, 1.0_f32);
    u.into_iter().map(|x| SQRT_2 * erfinv(x)).collect()
}

/// Inverse error function, `f32`, two-branch Horner polynomial.
///
/// Direct port of `mlx::core::simd::erfinv` (`mlx/backend/cpu/simd/math.h:150-191`)
/// for the scalar (`N == 1`) case:
/// `t = log(fma(a, -a, 1.0)) = log(1 - a*a)`, then if `abs(t) > 6.125` use the
/// `lhs` 9-term polynomial else the `rhs` 9-term polynomial, and return
/// `a * p`. `t` is non-positive here (`1 - a*a <= 1`), so `abs(t) > 6.125`
/// selects the `lhs` branch for `a` very close to the bounds.
///
/// The polynomial coefficients are copied verbatim from the MLX source (the
/// `0x...` hex annotations in `math.h` are the exact `f32` bit patterns).
/// Clippy's `excessive_precision` lint fires because the decimal literals carry
/// more digits than `f32` can hold, but the nearest-`f32` rounding of each
/// literal is exactly the MLX value, so truncating them would risk breaking
/// byte-exactness — hence the targeted `allow`.
#[allow(clippy::excessive_precision)]
#[must_use]
pub fn erfinv(a: f32) -> f32 {
    // t = log(fma(a, -a, 1.0))  =  log(1 - a*a)
    let t = a.mul_add(-a, 1.0_f32).ln();

    // lhs polynomial (maximum ulp error ≈ 2.35793).
    let lhs = |t: f32| -> f32 {
        let mut p = 3.03697567e-10_f32;
        p = p.mul_add(t, 2.93243101e-8_f32);
        p = p.mul_add(t, 1.22150334e-6_f32);
        p = p.mul_add(t, 2.84108955e-5_f32);
        p = p.mul_add(t, 3.93552968e-4_f32);
        p = p.mul_add(t, 3.02698812e-3_f32);
        p = p.mul_add(t, 4.83185798e-3_f32);
        p = p.mul_add(t, -2.64646143e-1_f32);
        p.mul_add(t, 8.40016484e-1_f32)
    };

    // rhs polynomial (maximum ulp error ≈ 2.35002).
    let rhs = |t: f32| -> f32 {
        let mut p = 5.43877832e-9_f32;
        p = p.mul_add(t, 1.43285448e-7_f32);
        p = p.mul_add(t, 1.22774793e-6_f32);
        p = p.mul_add(t, 1.12963626e-7_f32);
        p = p.mul_add(t, -5.61530760e-5_f32);
        p = p.mul_add(t, -1.47697632e-4_f32);
        p = p.mul_add(t, 2.31468678e-3_f32);
        p = p.mul_add(t, 1.15392581e-2_f32);
        p = p.mul_add(t, -2.32015476e-1_f32);
        p.mul_add(t, 8.86226892e-1_f32)
    };

    let thresh = 6.125_f32;
    let p = if t.abs() > thresh { lhs(t) } else { rhs(t) };
    a * p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `random::key` matches the MLX `test random key` vectors
    /// (`mlx/tests/random_tests.cpp:11-24`).
    #[test]
    fn key_vectors() {
        assert_eq!(key(0), [0, 0]);
        assert_eq!(key(1), [0, 1]);
        let seed = 1u64 << 32;
        assert_eq!(key(seed), [1, 0]);
        assert_eq!(key(seed + 1), [1, 1]);
    }

    /// `split(key) = bits({2,2})` reshaped: the first row is `bits([2], key)`'s
    /// pair-fill, so `random_bits(2, key(0))` must reproduce the head of the
    /// `test random split` vector (`mlx/tests/random_tests.cpp:39-43`):
    /// `split(key(0))` → key `{4146024105, 967050713}`,
    /// subkey `{2718843009, 1272950319}`.
    ///
    /// MLX's `split(key, num)` calls `bits({num, 2}, 4, key)` over a single key,
    /// so the full `num*2` u32 stream is exactly `random_bits(num*2, key)`.
    #[test]
    fn split_two_matches_mlx() {
        // split(key(0), 2) = bits({2,2}, key(0)) = random_bits(4, key(0)).
        let v = random_bits(4, key(0));
        assert_eq!(v, vec![4146024105, 967050713, 2718843009, 1272950319]);
    }

    /// `split(key(0), 3) = bits({3,2}, key(0)) = random_bits(6, key(0))`
    /// (`mlx/tests/random_tests.cpp:44-53`).
    #[test]
    fn split_three_matches_mlx() {
        let v = random_bits(6, key(0));
        assert_eq!(
            v,
            vec![2467461003, 428148500, 3186719485, 3840466878, 2562233961, 1946702221]
        );
    }

    /// Scalar `bits({}, key)` = `random_bits(1, key)`. From
    /// `mlx/tests/random_tests.cpp:91-108`: key(0) → `1797259609`,
    /// key(1) → `507451445`.
    #[test]
    fn scalar_bits_matches_mlx() {
        assert_eq!(random_bits(1, key(0)), vec![1797259609]);
        assert_eq!(random_bits(1, key(1)), vec![507451445]);
    }

    /// `bits({3,1}, key(0)) = random_bits(3, key(0))` →
    /// `{4146024105, 1351547692, 2718843009}` (`mlx/tests/random_tests.cpp:121-125`).
    /// This exercises the odd-`n` middle-element path of the fill layout.
    #[test]
    fn three_bits_odd_layout_matches_mlx() {
        assert_eq!(
            random_bits(3, key(0)),
            vec![4146024105, 1351547692, 2718843009]
        );
    }

    /// The 2-element fill (`n == 2`, `half == 1`, `even`) runs the
    /// `count.first < half` branch once with `count == {0, 1}`, so
    /// `random_bits(2, key)` is exactly `threefry(key, {0, 1})`.
    #[test]
    fn threefry_pair_equals_bits_two() {
        let rb = threefry2x32(key(0), [0, 1]);
        assert_eq!(random_bits(2, key(0)), vec![rb[0], rb[1]]);
    }

    /// The 4-element fill writes `out[0]` from `threefry(key, {0, 2})` (the
    /// first `while` iteration with `count == {0, 2}`), confirming the
    /// `count.second` start offset (`half + !even == 2`). This documents the
    /// exact counter→element mapping that makes `split(key(0), 2)` match.
    #[test]
    fn bits_four_first_word_uses_count_zero_two() {
        let rb = threefry2x32(key(0), [0, 2]);
        let v = random_bits(4, key(0));
        assert_eq!(v[0], rb[0]); // out[0]
        assert_eq!(v[2], rb[1]); // out[2]
        assert_eq!(v[0], 4146024105); // MLX split(key(0),2) head
    }

    /// Uniform scalar matches MLX's `to_float(bits)` reference
    /// (`mlx/tests/random_tests.cpp:303-317`): key(0) → `1797259609 / UINT32_MAX`.
    #[test]
    fn uniform_scalar_matches_mlx() {
        let u0 = uniform(1, key(0), 0.0, 1.0)[0];
        let expected0 = 1797259609.0_f32 / U32_MAX_AS_F32;
        assert_eq!(u0, expected0);

        let u1 = uniform(1, key(1), 0.0, 1.0)[0];
        let expected1 = 507451445.0_f32 / U32_MAX_AS_F32;
        assert_eq!(u1, expected1);
    }

    /// Uniform respects the half-open upper bound: every sample is `< hi`.
    #[test]
    fn uniform_upper_bound_respected() {
        let v = uniform(4096, key(128291), -1.0, 1.0);
        assert!(v.iter().all(|&x| x < 1.0));
        assert!(v.iter().all(|&x| x >= -1.0));
    }

    /// `erfinv(0) == 0`, and the function is odd: `erfinv(-x) == -erfinv(x)`.
    #[test]
    fn erfinv_basic() {
        assert_eq!(erfinv(0.0), 0.0);
        let x = 0.3_f32;
        assert!((erfinv(-x) + erfinv(x)).abs() < 1e-6);
    }

    /// Normal samples are finite and use both `erfinv` branches without panic.
    #[test]
    fn normal_finite() {
        let v = normal(131072, key(42));
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
