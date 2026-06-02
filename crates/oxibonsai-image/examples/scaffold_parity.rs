//! Full-chain golden validation of the FLUX.2 native scaffolding generators
//! ([`oxibonsai_image::sample`]) against the golden `.npy` dump (seed 42, 512²,
//! 4 steps).
//!
//! Generates `create_noise(42, 512, 512)`, `img_ids(32, 32)`, `txt_ids(512)`,
//! and `flow_match_schedule(1024, 4)`, loads the golden tensors, squeezes their
//! leading batch dim (`(1, seq, C) -> (seq, C)`), and reports per-tensor: shape
//! match, max-abs, and cosine.
//!
//! Gates:
//! - `init_latents` : cosine ≥ 0.99999 (also reports the bit-exact element count;
//!   the golden is f32 pre-`bfloat16`-cast, so the Rust f32 noise byte-matches it
//!   — this is the proof the Threefry port is exact).
//! - `img_ids`, `txt_ids` : EXACT integer equality.
//! - `sigmas`, `timesteps` : max-abs ≤ 1e-5.
//!
//! Usage (the golden dir defaults to the standard location):
//! ```text
//! cargo run -p oxibonsai-image --release --example scaffold_parity
//! cargo run -p oxibonsai-image --release --example scaffold_parity -- /tmp/bonsai_golden/f32
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::sample::{create_noise, flow_match_schedule, img_ids, txt_ids};

/// A loaded `.npy` tensor (f32, C-order).
struct Npy {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// Reorder a Fortran-stored (column-major) buffer into C (row-major) order.
/// `init_latents.npy` is dumped with `fortran_order == True`; the other goldens
/// are C-order.
fn fortran_to_c(src: &[f32], shape: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let mut f_stride = vec![1usize; ndim];
    for d in 1..ndim {
        f_stride[d] = f_stride[d - 1] * shape[d - 1];
    }
    let mut out = vec![0.0f32; numel];
    for (c_pos, slot) in out.iter_mut().enumerate() {
        let mut rem = c_pos;
        let mut f_off = 0usize;
        for (d, &fs) in f_stride.iter().enumerate() {
            let stride_c: usize = shape[d + 1..].iter().product();
            let idx = rem / stride_c;
            rem %= stride_c;
            f_off += idx * fs;
        }
        *slot = src[f_off];
    }
    out
}

/// Minimal NumPy `.npy` reader: v1.0/2.0 header, `descr == '<f4'`. Handles both
/// C-order and Fortran-order (the latter is reordered to C-order on load).
fn read_npy(path: &Path) -> Result<Npy, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: bad npy magic", path.display()));
    }
    let major = bytes[6];
    let (header_start, header_len) = if major >= 2 {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    } else {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| format!("{}: header utf8: {e}", path.display()))?;
    if !header.contains("'<f4'") {
        return Err(format!("{}: descr is not '<f4': {header}", path.display()));
    }
    let fortran = header.contains("'fortran_order': True");
    let s_idx = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: no shape key", path.display()))?;
    let open = header[s_idx..]
        .find('(')
        .map(|o| s_idx + o + 1)
        .ok_or_else(|| format!("{}: no shape open paren", path.display()))?;
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .ok_or_else(|| format!("{}: no shape close paren", path.display()))?;
    let shape: Vec<usize> = header[open..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| format!("shape parse: {e}")))
        .collect::<Result<_, _>>()?;
    let data_start = header_start + header_len;
    let payload = &bytes[data_start..];
    if payload.len() % 4 != 0 {
        return Err(format!("{}: payload not f32-aligned", path.display()));
    }
    let data: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let numel: usize = shape.iter().product();
    if data.len() < numel {
        return Err(format!(
            "{}: payload short ({} < {})",
            path.display(),
            data.len(),
            numel
        ));
    }
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&data[..numel], &shape)
    } else {
        data[..numel].to_vec()
    };
    Ok(Npy { data, shape })
}

/// Drop a leading batch dim of 1 from a golden shape (e.g. `(1, 1024, 4)` ->
/// `(1024, 4)`). The data buffer is unchanged (batch-1 is a no-op on the
/// row-major layout).
fn squeeze_batch(n: &Npy) -> Vec<usize> {
    if n.shape.len() >= 3 && n.shape[0] == 1 {
        n.shape[1..].to_vec()
    } else {
        n.shape.clone()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn exact_count(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
}

fn main() -> ExitCode {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/bonsai_golden/f32"));

    println!("== FLUX.2 scaffold parity vs golden (seed 42, 512^2, 4 steps) ==");
    println!("   golden dir: {}", dir.display());

    let mut pass = true;

    // ---- init_latents: cosine >= 0.99999, report bit-exact count ----
    match read_npy(&dir.join("init_latents.npy")) {
        Ok(g) => {
            let sq = squeeze_batch(&g);
            let rust = create_noise(42, 512, 512);
            let shape_ok = sq == vec![1024, PACKED_CHANNELS_SEQ.1] && rust.len() == g.data.len();
            let cos = cosine(&rust, &g.data);
            let m = max_abs(&rust, &g.data);
            let exact = exact_count(&rust, &g.data);
            let gate = shape_ok && cos >= 0.99999;
            pass &= gate;
            println!("\n[init_latents]");
            println!(
                "  golden shape (squeezed) = {sq:?}, rust len = {}",
                rust.len()
            );
            println!("  cosine   = {cos:.8}  (gate >= 0.99999)");
            println!("  max-abs  = {m:.6e}");
            println!(
                "  bit-exact = {exact}/{} ({:.4}%)",
                g.data.len(),
                100.0 * exact as f64 / g.data.len().max(1) as f64
            );
            println!("  => {}", if gate { "PASS" } else { "FAIL" });
        }
        Err(e) => {
            pass = false;
            println!("\n[init_latents] load error: {e}");
        }
    }

    // ---- img_ids: EXACT integer equality ----
    match read_npy(&dir.join("img_ids.npy")) {
        Ok(g) => {
            let sq = squeeze_batch(&g);
            let rust = img_ids(32, 32);
            let shape_ok = sq == vec![1024, 4] && rust.len() == g.data.len();
            let exact = exact_count(&rust, &g.data);
            let gate = shape_ok && exact == g.data.len();
            pass &= gate;
            println!("\n[img_ids]");
            println!(
                "  golden shape (squeezed) = {sq:?}, rust len = {}",
                rust.len()
            );
            println!(
                "  exact = {exact}/{} {}",
                g.data.len(),
                if gate { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            pass = false;
            println!("\n[img_ids] load error: {e}");
        }
    }

    // ---- txt_ids: EXACT integer equality ----
    match read_npy(&dir.join("txt_ids.npy")) {
        Ok(g) => {
            let sq = squeeze_batch(&g);
            let seq_txt = sq.first().copied().unwrap_or(0);
            let rust = txt_ids(seq_txt);
            let shape_ok = sq == vec![seq_txt, 4] && rust.len() == g.data.len();
            let exact = exact_count(&rust, &g.data);
            let gate = shape_ok && exact == g.data.len();
            pass &= gate;
            println!("\n[txt_ids]");
            println!(
                "  golden shape (squeezed) = {sq:?}, rust len = {}",
                rust.len()
            );
            println!(
                "  exact = {exact}/{} {}",
                g.data.len(),
                if gate { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            pass = false;
            println!("\n[txt_ids] load error: {e}");
        }
    }

    // ---- sigmas / timesteps: max-abs <= 1e-5 ----
    // image_seq_len == 1024 (the packed image sequence), steps == 4.
    let (rust_timesteps, rust_sigmas) = flow_match_schedule(1024, 4);

    match read_npy(&dir.join("sigmas.npy")) {
        Ok(g) => {
            let len_ok = rust_sigmas.len() == g.data.len();
            let m = max_abs(&rust_sigmas, &g.data);
            let gate = len_ok && m <= 1e-5;
            pass &= gate;
            println!("\n[sigmas]");
            println!(
                "  golden len = {}, rust len = {}",
                g.data.len(),
                rust_sigmas.len()
            );
            println!(
                "  max-abs = {m:.6e} (gate <= 1e-5) {}",
                if gate { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            pass = false;
            println!("\n[sigmas] load error: {e}");
        }
    }

    match read_npy(&dir.join("timesteps.npy")) {
        Ok(g) => {
            let len_ok = rust_timesteps.len() == g.data.len();
            let m = max_abs(&rust_timesteps, &g.data);
            let gate = len_ok && m <= 1e-5;
            pass &= gate;
            println!("\n[timesteps]");
            println!(
                "  golden len = {}, rust len = {}",
                g.data.len(),
                rust_timesteps.len()
            );
            println!(
                "  max-abs = {m:.6e} (gate <= 1e-5) {}",
                if gate { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            pass = false;
            println!("\n[timesteps] load error: {e}");
        }
    }

    println!(
        "\n== SCAFFOLD PARITY {} ==",
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `(_, packed_channels)` — the expected `[seq, C]` feature width of the packed
/// latent, exposed by the library as `sample::PACKED_CHANNELS` (128).
const PACKED_CHANNELS_SEQ: (usize, usize) = (1024, oxibonsai_image::sample::PACKED_CHANNELS);
