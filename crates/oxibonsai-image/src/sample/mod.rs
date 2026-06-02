//! FLUX.2 native sampling scaffolding: initial noise, position ids, and the
//! flow-match Euler sigma/timestep schedule — all Pure-Rust and byte-exact
//! against the MLX (`mflux-prism`) FLUX.2 pipeline.
//!
//! Together with [`mlx_rng`] this makes the OxiBonsai text-to-image pipeline
//! self-sufficient: the initial noise, the `img_ids` / `txt_ids` RoPE position
//! grids, and the sampler schedule are reproduced from the model definition
//! rather than loaded from a golden `.npy` dump.
//!
//! Ported from `mflux-prism` (`mflux-prism/src/mflux`):
//! - `models/flux2/latent_creator/flux2_latent_creator.py`
//!   - `prepare_latents` (lines 60-77): `normal(shape=(B, C*4, h/2, w/2), key(seed))`
//!     where `h = 2*(height // 16)`, so for `height == width == 512` the latent
//!     grid is `32 × 32` and the channel count is `num_latents_channels(=32) * 4 = 128`.
//!   - `pack_latents` (lines 19-22): `reshape(B, C, h*w).transpose(0, 2, 1)` →
//!     packed `(B, h*w, C)`.
//!   - `prepare_grid_ids` (lines 44-57): `[t_coord, flat_h, flat_w, layer=0]`.
//! - `models/flux2/model/flux2_text_encoder/prompt_encoder.py`
//!   - `prepare_text_ids` (lines 66-85): `[t=0, h=0, w=0, token_id]`.
//! - `models/common/schedulers/flow_match_euler_discrete_scheduler.py`
//!   - `get_timesteps_and_sigmas` (lines 62-76), `_compute_empirical_mu`
//!     (lines 78-88), `_time_shift_exponential_array` (lines 94-96).
//!
//! ## Precision / byte-exactness
//!
//! [`create_noise`] returns the **f32** packed normal values. The golden
//! `init_latents.npy` (seed 42, 512²) was dumped pre-`bfloat16`-cast, so the
//! f32 output byte-matches it exactly (`max-abs == 0`, cosine `== 1.0`). The
//! real mflux runtime casts the latents to `bfloat16` (`ModelConfig.precision`)
//! before the DiT; callers that need that form should round at the DiT boundary
//! (`half::bf16::from_f32(x).to_f32()`). The f32 form is the correctness oracle
//! and is what the existing DiT golden-input path already consumes.

pub mod mlx_rng;

/// Default FLUX.2 latent channel count *before* the 2×2 patchify pack
/// (`num_latents_channels` in `prepare_latents`).
const NUM_LATENT_CHANNELS: usize = 32;

/// Packed channel count after the `*4` patchify (`num_latents_channels * 4`).
/// This is the per-token feature width of the packed latent and of `img_ids`'
/// companion latent.
pub const PACKED_CHANNELS: usize = NUM_LATENT_CHANNELS * 4;

/// FLUX.2 VAE scale factor (`vae_scale_factor` in `prepare_latents`). The latent
/// grid side is `height / (vae_scale_factor * 2)`.
const VAE_SCALE_FACTOR: usize = 8;

/// Compute the packed latent grid dimensions `(lat_h, lat_w)` for a target
/// image of `height × width` pixels, mirroring `prepare_latents`
/// (`flux2_latent_creator.py:68-71`): the full latent side is
/// `2 * (px // (vae_scale_factor * 2))`, and the packed (patchified) side is
/// half of that.
#[must_use]
pub fn latent_grid(height: usize, width: usize) -> (usize, usize) {
    let full_h = 2 * (height / (VAE_SCALE_FACTOR * 2));
    let full_w = 2 * (width / (VAE_SCALE_FACTOR * 2));
    (full_h / 2, full_w / 2)
}

/// Generate the initial packed latent noise for a seed and target resolution.
///
/// Returns a row-major `[seq, PACKED_CHANNELS]` buffer (`seq = lat_h * lat_w`),
/// matching mflux's packed latent `(1, seq, 128)` with the batch dim squeezed.
///
/// Mirrors `prepare_packed_latents`:
/// 1. `raw = normal(1 * 128 * lat_h * lat_w, key(seed))`, laid out NCHW as
///    `(1, 128, lat_h, lat_w)` row-major.
/// 2. `pack_latents`: `reshape(1, 128, lat_h*lat_w).transpose(0, 2, 1)`, i.e.
///    `packed[s, c] = raw[c, s]` where `s = row * lat_w + col`.
///
/// The values are **f32** (see the module-level precision note); the golden
/// `init_latents.npy` byte-matches this exactly.
#[must_use]
pub fn create_noise(seed: u64, height: usize, width: usize) -> Vec<f32> {
    let (lat_h, lat_w) = latent_grid(height, width);
    let seq = lat_h * lat_w;
    let n = PACKED_CHANNELS * seq; // 1 * 128 * lat_h * lat_w

    // NCHW (1, 128, lat_h, lat_w) row-major standard-normal draw.
    let raw = mlx_rng::normal(n, mlx_rng::key(seed));

    // pack: packed[s * 128 + c] = raw[c * seq + s].
    let mut packed = vec![0.0f32; n];
    for c in 0..PACKED_CHANNELS {
        let row_base = c * seq;
        for s in 0..seq {
            packed[s * PACKED_CHANNELS + c] = raw[row_base + s];
        }
    }
    packed
}

/// Build the image RoPE position ids, `[lat_h * lat_w, 4]` row-major.
///
/// Port of `prepare_grid_ids` with `t_coord == 0` and `layer == 0`
/// (`flux2_latent_creator.py:44-57`): row `i = [0, i / lat_w, i % lat_w, 0]`
/// (h-major over a `lat_w`-wide grid).
#[must_use]
pub fn img_ids(lat_h: usize, lat_w: usize) -> Vec<f32> {
    let seq = lat_h * lat_w;
    let mut ids = vec![0.0f32; seq * 4];
    for i in 0..seq {
        let base = i * 4;
        // ids[base + 0] = 0.0 (t_coord)
        ids[base + 1] = (i / lat_w) as f32; // flat_h
        ids[base + 2] = (i % lat_w) as f32; // flat_w
                                            // ids[base + 3] = 0.0 (layer)
    }
    ids
}

/// Build the text RoPE position ids, `[seq_txt, 4]` row-major.
///
/// Port of `prepare_text_ids` with `t_coord == None`
/// (`prompt_encoder.py:66-85`): row `j = [0, 0, 0, j]` (only the token-id
/// coordinate advances).
#[must_use]
pub fn txt_ids(seq_txt: usize) -> Vec<f32> {
    let mut ids = vec![0.0f32; seq_txt * 4];
    for j in 0..seq_txt {
        ids[j * 4 + 3] = j as f32; // token_id
    }
    ids
}

/// Empirical `mu` for the flow-match schedule.
///
/// Port of `_compute_empirical_mu`
/// (`flow_match_euler_discrete_scheduler.py:78-88`). Computed in `f64` to mirror
/// Python's float arithmetic (the scalar `mu` is a Python `float`).
fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let l = image_seq_len as f64;
    let (a1, b1) = (8.73809524e-05_f64, 1.89833333_f64);
    let (a2, b2) = (0.00016927_f64, 0.45666666_f64);
    if image_seq_len > 4300 {
        return a2 * l + b2;
    }
    let m_200 = a2 * l + b2;
    let m_10 = a1 * l + b1;
    let a = (m_200 - m_10) / 190.0_f64;
    let b = m_200 - 200.0_f64 * a;
    a * (num_steps as f64) + b
}

/// Compute the flow-match Euler `(timesteps[steps], sigmas[steps + 1])`.
///
/// Port of `get_timesteps_and_sigmas`
/// (`flow_match_euler_discrete_scheduler.py:62-76`):
/// 1. `sigmas = linspace(1.0, 1.0 / steps, steps)`.
/// 2. `mu = _compute_empirical_mu(image_seq_len, steps)`.
/// 3. `sigmas = exp(mu) / (exp(mu) + (1/sigma - 1))` (exponential time shift,
///    `sigma_power == 1`).
/// 4. `timesteps = sigmas * 1000`.
/// 5. append `0.0` to `sigmas`.
///
/// The exponential shift is evaluated in `f32` to mirror MLX's `mx.exp` on the
/// `float32` sigma array; this reproduces the golden `timesteps`/`sigmas`
/// (seed-independent) exactly.
#[must_use]
pub fn flow_match_schedule(image_seq_len: usize, steps: usize) -> (Vec<f32>, Vec<f32>) {
    if steps == 0 {
        return (Vec::new(), vec![0.0f32]);
    }

    let mu = compute_empirical_mu(image_seq_len, steps);
    let exp_mu = (mu as f32).exp();

    // linspace(1.0, 1.0/steps, steps) in f32.
    let last = 1.0_f32 / (steps as f32);
    let mut sigmas: Vec<f32> = Vec::with_capacity(steps + 1);
    let mut timesteps: Vec<f32> = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = if steps == 1 {
            1.0_f32
        } else {
            // linspace endpoints inclusive: 1.0 + i*(last - 1.0)/(steps - 1).
            1.0_f32 + (i as f32) * (last - 1.0_f32) / ((steps - 1) as f32)
        };
        // exponential time shift, sigma_power == 1.
        let shifted = exp_mu / (exp_mu + (1.0_f32 / t - 1.0_f32));
        sigmas.push(shifted);
        timesteps.push(shifted * 1000.0_f32);
    }
    // append terminal 0 sigma.
    sigmas.push(0.0_f32);

    (timesteps, sigmas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_grid_512() {
        assert_eq!(latent_grid(512, 512), (32, 32));
    }

    #[test]
    fn latent_grid_1024() {
        assert_eq!(latent_grid(1024, 1024), (64, 64));
    }

    #[test]
    fn create_noise_shape() {
        let v = create_noise(42, 512, 512);
        assert_eq!(v.len(), 1024 * PACKED_CHANNELS);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// The pack transpose: `packed[s, c] = raw[c, s]`. Reconstruct `raw` from a
    /// known seed and check the first column / first row land where expected.
    #[test]
    fn create_noise_pack_direction() {
        let (lat_h, lat_w) = latent_grid(512, 512);
        let seq = lat_h * lat_w;
        let n = PACKED_CHANNELS * seq;
        let raw = mlx_rng::normal(n, mlx_rng::key(42));
        let packed = create_noise(42, 512, 512);
        // packed[s=0, c=0..3] == raw[c*seq + 0]
        for c in 0..4 {
            assert_eq!(packed[c], raw[c * seq]);
        }
        // packed[s=1, c=0] == raw[0*seq + 1]
        assert_eq!(packed[PACKED_CHANNELS], raw[1]);
    }

    #[test]
    fn img_ids_layout() {
        let ids = img_ids(32, 32);
        assert_eq!(ids.len(), 1024 * 4);
        // row 0 = [0,0,0,0]
        assert_eq!(&ids[0..4], &[0.0, 0.0, 0.0, 0.0]);
        // row 1 = [0,0,1,0]
        assert_eq!(&ids[4..8], &[0.0, 0.0, 1.0, 0.0]);
        // row 31 = [0,0,31,0]
        assert_eq!(&ids[31 * 4..31 * 4 + 4], &[0.0, 0.0, 31.0, 0.0]);
        // row 32 = [0,1,0,0]
        assert_eq!(&ids[32 * 4..32 * 4 + 4], &[0.0, 1.0, 0.0, 0.0]);
        // row 33 = [0,1,1,0]
        assert_eq!(&ids[33 * 4..33 * 4 + 4], &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn txt_ids_layout() {
        let ids = txt_ids(512);
        assert_eq!(ids.len(), 512 * 4);
        assert_eq!(&ids[0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&ids[4..8], &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&ids[511 * 4..511 * 4 + 4], &[0.0, 0.0, 0.0, 511.0]);
    }

    /// Schedule for `(image_seq_len=1024, steps=4)` reproduces the golden
    /// `timesteps`/`sigmas` (seed-independent). The expected literals are the
    /// exact golden `f32` values, so the `excessive_precision` lint is allowed
    /// here (truncating them would weaken the check).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn schedule_1024_4() {
        let (timesteps, sigmas) = flow_match_schedule(1024, 4);
        let exp_ts = [
            1000.0_f32,
            958.0853881835938,
            883.9818115234375,
            717.49658203125,
        ];
        let exp_sg = [
            1.0_f32,
            0.9580853581428528,
            0.8839818239212036,
            0.7174965739250183,
            0.0,
        ];
        assert_eq!(timesteps.len(), 4);
        assert_eq!(sigmas.len(), 5);
        for (a, b) in timesteps.iter().zip(exp_ts.iter()) {
            assert!((a - b).abs() <= 1e-3, "timestep {a} vs {b}");
        }
        for (a, b) in sigmas.iter().zip(exp_sg.iter()) {
            assert!((a - b).abs() <= 1e-6, "sigma {a} vs {b}");
        }
    }

    #[test]
    fn schedule_first_last() {
        let (timesteps, sigmas) = flow_match_schedule(1024, 4);
        // first sigma is exactly 1.0 (t == 1.0 → shifted == 1.0).
        assert_eq!(sigmas[0], 1.0);
        assert_eq!(timesteps[0], 1000.0);
        // terminal sigma appended is 0.
        assert_eq!(*sigmas.last().unwrap_or(&-1.0), 0.0);
    }
}
