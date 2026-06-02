//! Layered validation of the Pure-Rust MLX RNG port ([`oxibonsai_image::sample::mlx_rng`])
//! against a *live* MLX oracle dumped from an MLX-capable python.
//!
//! Requires an MLX-capable python (i.e. `import mlx.core` works); point at it
//! with `OXI_MLX_PYTHON=/path/to/.venv/bin/python` (defaults to `python3`).
//!
//! For each seed it compares three layers:
//! 1. **bits** (u32-exact): `mx.random.split(key, num)` equals
//!    `random_bits(num * 2, key)` — `split` is `bits({num, 2}, key)`, so this is
//!    a direct u32-stream check of the Threefry + fill layout. (This MLX build
//!    does not expose `mx.random.bits`; `split` is the exposed equivalent.)
//! 2. **uniform** (f32): `mx.random.uniform(shape, key)` vs `uniform(n, key, 0, 1)`.
//! 3. **normal** (f32): `mx.random.normal(shape, key)` vs `normal(n, key)`.
//!
//! Usage:
//! ```text
//! cargo run -p oxibonsai-image --release --example mlx_rng_check
//! # optional: point at a different python
//! OXI_MLX_PYTHON=/path/to/python cargo run ... --example mlx_rng_check
//! ```
//!
//! Reports per-seed/per-layer: u32 mismatch count (bits), and f32 max-abs error
//! (uniform/normal). Exit code 0 iff every layer matches within tolerance.

use std::process::{Command, ExitCode};

use oxibonsai_image::sample::mlx_rng;

/// Default python interpreter. Override with `OXI_MLX_PYTHON` to point at an
/// MLX-capable interpreter (e.g. a project `.venv/bin/python`).
const DEFAULT_PYTHON: &str = "python3";

/// Tolerance for the uniform layer (should be 0 — it is a pure `bits / UINT32_MAX`
/// reciprocal, identical in MLX and Rust; a tiny slack absorbs print round-trip).
const UNIFORM_TOL: f32 = 2e-7;

/// Tolerance for the normal layer. `erfinv` is a 2.36-ULP polynomial; MLX and
/// Rust evaluate the identical FMA sequence, so the error is dominated by the
/// text round-trip of the dump, not the math.
const NORMAL_TOL: f32 = 5e-4;

fn python() -> String {
    std::env::var("OXI_MLX_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON.to_string())
}

/// Run a python snippet, return stdout as a string.
fn run_py(src: &str) -> Result<String, String> {
    let out = Command::new(python())
        .arg("-c")
        .arg(src)
        .output()
        .map_err(|e| format!("spawn python: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "python exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Parse one whitespace/newline-separated line of values.
fn parse_line<T: std::str::FromStr>(line: &str) -> Result<Vec<T>, String> {
    line.split_whitespace()
        .map(|t| t.parse::<T>().map_err(|_| format!("parse {t:?}")))
        .collect()
}

/// Dump `split(key(seed), num)` flattened to a u32 stream.
fn mlx_bits(seed: u64, num: usize) -> Result<Vec<u32>, String> {
    let src = format!(
        "import mlx.core as mx\n\
         a = mx.random.split(mx.random.key({seed}), {num})\n\
         print(' '.join(str(int(x)) for x in a.reshape(-1).tolist()))\n"
    );
    let out = run_py(&src)?;
    parse_line::<u32>(out.trim())
}

/// Dump `uniform(shape=[n], key(seed))`.
fn mlx_uniform(seed: u64, n: usize) -> Result<Vec<f32>, String> {
    let src = format!(
        "import mlx.core as mx\n\
         a = mx.random.uniform(shape=[{n}], key=mx.random.key({seed}))\n\
         print(' '.join(repr(float(x)) for x in a.tolist()))\n"
    );
    let out = run_py(&src)?;
    parse_line::<f32>(out.trim())
}

/// Dump `normal(shape=[n], key(seed))`.
fn mlx_normal(seed: u64, n: usize) -> Result<Vec<f32>, String> {
    let src = format!(
        "import mlx.core as mx\n\
         a = mx.random.normal(shape=[{n}], key=mx.random.key({seed}))\n\
         print(' '.join(repr(float(x)) for x in a.tolist()))\n"
    );
    let out = run_py(&src)?;
    parse_line::<f32>(out.trim())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> ExitCode {
    let seeds: [u64; 3] = [0, 42, 9909];
    // bits uses `num` key-pairs (=> num*2 u32); the f32 layers use `n` elements.
    let bits_nums: [usize; 2] = [8, 131072];
    let elem_ns: [usize; 2] = [8, 131072];

    let mut all_ok = true;

    println!("== MLX RNG layered validation (python: {}) ==", python());

    for &seed in &seeds {
        println!("\n-- seed {seed} --");

        // bits (u32-exact) via split.
        for &num in &bits_nums {
            match mlx_bits(seed, num) {
                Ok(mlx) => {
                    let rust = mlx_rng::random_bits(num * 2, mlx_rng::key(seed));
                    let mismatches = mlx.iter().zip(rust.iter()).filter(|(a, b)| a != b).count();
                    let len_ok = mlx.len() == rust.len();
                    let ok = len_ok && mismatches == 0;
                    all_ok &= ok;
                    println!(
                        "  bits[num={num} -> {} u32]: mismatches={mismatches} {}",
                        num * 2,
                        if ok { "OK" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    all_ok = false;
                    println!("  bits[num={num}]: oracle error: {e}");
                }
            }
        }

        // uniform (f32) and normal (f32).
        for &n in &elem_ns {
            match mlx_uniform(seed, n) {
                Ok(mlx) => {
                    let rust = mlx_rng::uniform(n, mlx_rng::key(seed), 0.0, 1.0);
                    let m = max_abs(&mlx, &rust);
                    let ok = mlx.len() == rust.len() && m <= UNIFORM_TOL;
                    all_ok &= ok;
                    println!(
                        "  uniform[n={n}]: max-abs={m:.3e} (tol {UNIFORM_TOL:.0e}) {}",
                        if ok { "OK" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    all_ok = false;
                    println!("  uniform[n={n}]: oracle error: {e}");
                }
            }
            match mlx_normal(seed, n) {
                Ok(mlx) => {
                    let rust = mlx_rng::normal(n, mlx_rng::key(seed));
                    let m = max_abs(&mlx, &rust);
                    let ok = mlx.len() == rust.len() && m <= NORMAL_TOL;
                    all_ok &= ok;
                    println!(
                        "  normal [n={n}]: max-abs={m:.3e} (tol {NORMAL_TOL:.0e}) {}",
                        if ok { "OK" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    all_ok = false;
                    println!("  normal [n={n}]: oracle error: {e}");
                }
            }
        }
    }

    println!(
        "\n== MLX RNG CHECK {} ==",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if all_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
