//! Where the Q4_0 / Q8_0 / K-quant linear layers may run their GEMV, decided
//! once, when the layer is constructed.
//!
//! [`LinearQ4_0`](super::linear_standard::LinearQ4_0),
//! [`LinearQ8_0`](super::linear_standard::LinearQ8_0) and the six K-quant
//! layers ([`super::linear_kquant_full`], [`super::linear_kquant_ext`]) are
//! constructed without a `KernelDispatcher` of their own. On a `native-cuda`
//! build their `forward` used to try the CUDA GEMV whenever
//! `CudaGraph::global()` succeeded, and the K-quant layers ran their CPU
//! kernels through one process-wide dispatcher auto-detected on first use.
//! Neither honoured the engine's explicit CPU request: the CUDA attempt never
//! consulted it, and the shared dispatcher was detected on whichever thread
//! first needed it, normally one outside the request. So a model loaded with
//! `Backend::Cpu` (the CLI's `--backend cpu`) still ran every Q4_0 / Q8_0 /
//! K-quant GEMV on the GPU, although the engine constructs such a model inside
//! a [`CpuOnlyBackendScope`] precisely so that the model's own dispatchers
//! land on the CPU tier.
//!
//! The decision is now taken at construction, as an FP8 layer's is (an FP8
//! layer keeps the dispatcher the model was loaded with): a layer constructed
//! while a [`CpuOnlyBackendScope`] is active on the constructing thread never
//! tries a CUDA GEMV, and a K-quant layer constructed there runs on a CPU-tier
//! dispatcher that never probes the GPU. The scope is thread-local and the
//! engine decodes on other threads, so a per-call check would see no request
//! at all; the captured decision does not depend on the thread `forward` runs
//! on. A layer constructed with no scope active runs exactly the code it ran
//! before, including the lazily detected shared K-quant dispatcher.
//!
//! The decision is backend-agnostic. On a Metal build the Q4_0 / Q8_0 layers
//! apply it to their Metal GEMV in the same way, and a K-quant layer
//! constructed under the scope runs its CPU kernel through the CPU-tier
//! dispatcher instead of the Metal K-quant GEMV that the shared auto-detected
//! dispatcher selects on a Metal host.
//!
//! [`CpuOnlyBackendScope`]: oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope

use oxibonsai_kernels::KernelDispatcher;

/// Whether a layer constructed on this thread right now is bound to the CPU:
/// `true` while a [`CpuOnlyBackendScope`] is active (the engine's
/// `Backend::Cpu`), `false` otherwise.
///
/// [`CpuOnlyBackendScope`]: oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope
pub(crate) fn cpu_only_at_load() -> bool {
    oxibonsai_kernels::gpu_backend::cpu_only_backend_active()
}

/// Whether a layer constructed on this thread right now may try a GPU GEMV
/// kernel (CUDA on Linux / Windows, Metal on macOS): the negation of
/// [`cpu_only_at_load`].
#[cfg(any(
    all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ),
    all(feature = "metal", target_os = "macos")
))]
pub(crate) fn gpu_gemv_allowed_at_load() -> bool {
    !cpu_only_at_load()
}

/// The dispatcher a K-quant layer runs its GEMV through whenever no CUDA
/// kernel answers the call, given the layer's [`cpu_only_at_load`].
///
/// Each is detected once per process rather than once per `forward` call:
/// `KernelDispatcher::auto_detect()` allocates a fresh backend handle and logs
/// its selection on every call, and the hardware tier is a property of the
/// machine, not of any one layer or token.
///
/// * A layer constructed with no CPU-only request shares the dispatcher every
///   K-quant layer shared before, auto-detected on first use. Unlike
///   `with_tier(cpu_kernel_tier())`, `auto_detect` keeps the GPU-aware tier
///   selection the K-quant `StandardQuantKernel` methods can use.
/// * A layer constructed under the request shares a dispatcher pinned to the
///   CPU's own tier, which never probes the GPU and does not depend on the
///   thread that first uses it.
pub(crate) fn kquant_dispatcher(cpu_only: bool) -> &'static KernelDispatcher {
    if cpu_only {
        static CPU_ONLY: std::sync::OnceLock<KernelDispatcher> = std::sync::OnceLock::new();
        CPU_ONLY.get_or_init(|| KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier()))
    } else {
        static AUTO: std::sync::OnceLock<KernelDispatcher> = std::sync::OnceLock::new();
        AUTO.get_or_init(KernelDispatcher::auto_detect)
    }
}

#[cfg(all(
    test,
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
thread_local! {
    /// CUDA GEMV calls the quantized linear layers made on this thread.
    ///
    /// Thread-local because the tests run in parallel and a CUDA GEMV runs
    /// synchronously on the thread that called `forward`.
    static CUDA_GEMV_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Note that a quantized linear layer is about to call a CUDA GEMV kernel on
/// this thread, so the tests can tell which path a `forward` took.
#[cfg(all(
    test,
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub(crate) fn note_cuda_gemv_call() {
    CUDA_GEMV_CALLS.with(|calls| calls.set(calls.get() + 1));
}

/// Outside tests there is nothing to note: this compiles to nothing.
#[cfg(all(
    not(test),
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
#[inline(always)]
pub(crate) fn note_cuda_gemv_call() {}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
thread_local! {
    /// Metal GEMV calls the Q4_0 / Q8_0 linear layers made on this thread.
    ///
    /// Thread-local for the same reason as `CUDA_GEMV_CALLS`: the tests run in
    /// parallel and a Metal GEMV runs synchronously on the calling thread.
    static METAL_GEMV_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Note that a Q4_0 / Q8_0 linear layer is about to call the Metal GEMV
/// kernel on this thread, so the tests can tell which path a `forward` took.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
pub(crate) fn note_metal_gemv_call() {
    METAL_GEMV_CALLS.with(|calls| calls.set(calls.get() + 1));
}

/// Outside tests there is nothing to note: this compiles to nothing.
#[cfg(all(not(test), feature = "metal", target_os = "macos"))]
#[inline(always)]
pub(crate) fn note_metal_gemv_call() {}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope;

    #[test]
    fn linear_kquant_dispatcher_follows_the_request_captured_at_load() {
        assert!(!cpu_only_at_load(), "no CPU-only request outside the scope");
        let captured = {
            let _scope = CpuOnlyBackendScope::enter();
            cpu_only_at_load()
        };
        assert!(
            captured,
            "a layer constructed under CpuOnlyBackendScope is CPU-only"
        );

        // The CPU-only dispatcher runs the CPU's own tier and is the same
        // object whichever thread asks for it, inside the scope or not.
        let cpu_only = kquant_dispatcher(true);
        assert_eq!(
            cpu_only.tier(),
            oxibonsai_kernels::cpu_kernel_tier(),
            "a K-quant layer constructed under CpuOnlyBackendScope must run on the CPU tier"
        );
        let in_scope = {
            let _scope = CpuOnlyBackendScope::enter();
            kquant_dispatcher(true)
        };
        assert!(
            std::ptr::eq(cpu_only, in_scope),
            "detected once per process"
        );
        let elsewhere = std::thread::spawn(|| kquant_dispatcher(true))
            .join()
            .expect("probe thread");
        assert!(std::ptr::eq(cpu_only, elsewhere));

        // Every other layer keeps the auto-detected dispatcher.
        let auto = kquant_dispatcher(false);
        assert!(!std::ptr::eq(cpu_only, auto));
        assert_eq!(auto.tier(), KernelDispatcher::auto_detect().tier());
    }

    /// The regression test for `Backend::Cpu` running Q4_0 / Q8_0 / K-quant
    /// GEMVs on the GPU: every layer constructed under `CpuOnlyBackendScope`
    /// must take its CPU path, with the forward run on another thread after
    /// the scope is gone (as the engine's decode does), and with no scope at
    /// construction a CUDA host must still take the CUDA GEMV.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    mod cuda {
        use super::super::CUDA_GEMV_CALLS;
        use crate::error::ModelResult;
        use crate::layers::linear::{
            LinearLayer, LinearQ2K, LinearQ3K, LinearQ4K, LinearQ4_0, LinearQ5K, LinearQ6K,
            LinearQ8K, LinearQ8_0,
        };
        use oxibonsai_core::{
            BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
        };
        use oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope;
        use oxibonsai_kernels::KernelResult;

        /// Constructs a layer over a test's synthetic tensor.
        type BuildLayer<'f, 'a> = dyn Fn() -> ModelResult<LinearLayer<'a>> + 'f;

        /// The CPU kernel a layer's CPU path runs, as `(input, output)`.
        type CpuGemv<'f> = dyn Fn(&[f32], &mut [f32]) -> KernelResult<()> + 'f;

        fn cuda_gemv_calls() -> u64 {
            CUDA_GEMV_CALLS.with(std::cell::Cell::get)
        }

        /// Deterministic mixed-sign weight matrix (row-major).
        fn weights(n_rows: usize, in_features: usize) -> Vec<f32> {
            (0..n_rows * in_features)
                .map(|idx| {
                    let r = (idx / in_features) as f32;
                    let c = (idx % in_features) as f32;
                    (r * 0.13 + c * 0.07).sin() * 2.0 + (c * 0.031).cos() + 0.2
                })
                .collect()
        }

        /// Deterministic input vector of length `in_features`.
        fn input(in_features: usize) -> Vec<f32> {
            (0..in_features)
                .map(|i| 0.5 + 0.4 * ((i as f32) * 0.05).sin())
                .collect()
        }

        fn bits(values: &[f32]) -> Vec<u32> {
            values.iter().map(|v| v.to_bits()).collect()
        }

        /// Check one layer type: `build` constructs it over a synthetic
        /// tensor, `reference` is the CPU kernel its CPU path runs.
        fn assert_cpu_only_at_load<'a>(
            label: &str,
            n_rows: usize,
            in_features: usize,
            build: &BuildLayer<'_, 'a>,
            reference: &CpuGemv<'_>,
        ) {
            let input = input(in_features);

            // Constructed under the explicit CPU request, exactly as
            // `Backend::Cpu` constructs a model.
            let cpu_only = {
                let _scope = CpuOnlyBackendScope::enter();
                build().expect("construct the layer under CpuOnlyBackendScope")
            };
            // The scope is gone and `forward` runs on a thread that never had
            // one, as the engine's decode thread does.
            let input_ref = input.as_slice();
            let (calls, output) = std::thread::scope(|scope| {
                scope
                    .spawn(move || {
                        let before = cuda_gemv_calls();
                        let mut output = vec![0.0f32; n_rows];
                        cpu_only
                            .forward_vec(input_ref, &mut output)
                            .expect("forward of the CPU-only layer");
                        (cuda_gemv_calls() - before, output)
                    })
                    .join()
                    .expect("forward thread")
            });
            assert_eq!(
                calls, 0,
                "{label}: a layer constructed under CpuOnlyBackendScope called a CUDA GEMV"
            );
            let mut expected = vec![0.0f32; n_rows];
            reference(&input, &mut expected).expect("CPU reference GEMV");
            assert_eq!(
                bits(&output),
                bits(&expected),
                "{label}: the CPU-only layer must return its CPU kernel's result bit for bit"
            );

            // Positive control: with no scope at construction a CUDA host
            // takes the CUDA GEMV, so the zero above is an observation and
            // not a counter that never moves.
            if oxibonsai_kernels::CudaGraph::global().is_ok() {
                let auto = build().expect("construct the layer with no scope active");
                let before = cuda_gemv_calls();
                let mut output = vec![0.0f32; n_rows];
                auto.forward_vec(&input, &mut output)
                    .expect("forward of the auto layer");
                assert_eq!(
                    cuda_gemv_calls() - before,
                    1,
                    "{label}: with no CPU-only request a CUDA host must take the CUDA GEMV"
                );
            } else {
                eprintln!("{label}: no CUDA device, so the positive control did not run");
            }
        }

        #[test]
        fn linear_q4_0_and_q8_0_constructed_under_cpu_only_scope_never_call_cuda() {
            let (n_rows, in_features) = (33, 1056);
            let w = weights(n_rows, in_features);

            let q4_0 = BlockQ4_0::quantize(&w).expect("Q4_0 quantize");
            assert_cpu_only_at_load(
                "Q4_0",
                n_rows,
                in_features,
                &|| Ok(LinearQ4_0::new(&q4_0, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q4_0(&q4_0, x, y, n_rows, in_features),
            );

            let q8_0 = BlockQ8_0::quantize(&w).expect("Q8_0 quantize");
            assert_cpu_only_at_load(
                "Q8_0",
                n_rows,
                in_features,
                &|| Ok(LinearQ8_0::new(&q8_0, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q8_0(&q8_0, x, y, n_rows, in_features),
            );
        }

        #[test]
        fn linear_kquants_constructed_under_cpu_only_scope_never_call_cuda() {
            let (n_rows, in_features) = (5, 512);
            let w = weights(n_rows, in_features);

            let q2k = BlockQ2K::quantize(&w).expect("Q2_K quantize");
            assert_cpu_only_at_load(
                "Q2_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ2K::new(&q2k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q2k(&q2k, x, y, n_rows, in_features),
            );

            let q3k = BlockQ3K::quantize(&w).expect("Q3_K quantize");
            assert_cpu_only_at_load(
                "Q3_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ3K::new(&q3k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q3k(&q3k, x, y, n_rows, in_features),
            );

            let q4k = BlockQ4K::quantize(&w).expect("Q4_K quantize");
            assert_cpu_only_at_load(
                "Q4_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ4K::new(&q4k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q4k(&q4k, x, y, n_rows, in_features),
            );

            let q5k = BlockQ5K::quantize(&w).expect("Q5_K quantize");
            assert_cpu_only_at_load(
                "Q5_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ5K::new(&q5k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q5k(&q5k, x, y, n_rows, in_features),
            );

            let q6k = BlockQ6K::quantize(&w).expect("Q6_K quantize");
            assert_cpu_only_at_load(
                "Q6_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ6K::new(&q6k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q6k(&q6k, x, y, n_rows, in_features),
            );

            let q8k = BlockQ8K::quantize(&w).expect("Q8_K quantize");
            assert_cpu_only_at_load(
                "Q8_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ8K::new(&q8k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q8k(&q8k, x, y, n_rows, in_features),
            );
        }
    }

    /// The Metal half of the same regression: a Q4_0 / Q8_0 layer constructed
    /// under `CpuOnlyBackendScope` takes its CPU kernel (no Metal GEMV call,
    /// output bit-identical to the CPU kernel) with the forward run on another
    /// thread after the scope is gone, while with no scope at construction a
    /// Metal host still takes the Metal GEMV; a K-quant layer constructed under
    /// the scope runs on the CPU-tier dispatcher and matches the CPU kernel bit
    /// for bit, while the shared dispatcher of an unscoped layer is the GPU
    /// tier on a Metal host.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod metal {
        use super::super::{kquant_dispatcher, METAL_GEMV_CALLS};
        use crate::error::ModelResult;
        use crate::layers::linear::{
            LinearLayer, LinearQ2K, LinearQ3K, LinearQ4K, LinearQ4_0, LinearQ5K, LinearQ6K,
            LinearQ8K, LinearQ8_0,
        };
        use oxibonsai_core::{
            BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
        };
        use oxibonsai_kernels::gpu_backend::CpuOnlyBackendScope;
        use oxibonsai_kernels::{KernelResult, KernelTier, MetalGraph};

        /// Constructs a layer over a test's synthetic tensor.
        type BuildLayer<'f, 'a> = dyn Fn() -> ModelResult<LinearLayer<'a>> + 'f;

        /// The CPU kernel a layer's CPU path runs, as `(input, output)`.
        type CpuGemv<'f> = dyn Fn(&[f32], &mut [f32]) -> KernelResult<()> + 'f;

        fn metal_gemv_calls() -> u64 {
            METAL_GEMV_CALLS.with(std::cell::Cell::get)
        }

        /// Deterministic mixed-sign weight matrix (row-major).
        fn weights(n_rows: usize, in_features: usize) -> Vec<f32> {
            (0..n_rows * in_features)
                .map(|idx| {
                    let r = (idx / in_features) as f32;
                    let c = (idx % in_features) as f32;
                    (r * 0.13 + c * 0.07).sin() * 2.0 + (c * 0.031).cos() + 0.2
                })
                .collect()
        }

        /// Deterministic input vector of length `in_features`.
        fn input(in_features: usize) -> Vec<f32> {
            (0..in_features)
                .map(|i| 0.5 + 0.4 * ((i as f32) * 0.05).sin())
                .collect()
        }

        fn bits(values: &[f32]) -> Vec<u32> {
            values.iter().map(|v| v.to_bits()).collect()
        }

        /// Construct the layer under the scope, forward it on a thread that
        /// never had one, and check that the thread made no Metal GEMV call
        /// and that the output is the CPU kernel's, bit for bit.
        fn assert_cpu_only_matches_cpu_kernel<'a>(
            label: &str,
            n_rows: usize,
            in_features: usize,
            build: &BuildLayer<'_, 'a>,
            reference: &CpuGemv<'_>,
        ) {
            let input = input(in_features);
            let cpu_only = {
                let _scope = CpuOnlyBackendScope::enter();
                build().expect("construct the layer under CpuOnlyBackendScope")
            };
            let input_ref = input.as_slice();
            let (calls, output) = std::thread::scope(|scope| {
                scope
                    .spawn(move || {
                        let before = metal_gemv_calls();
                        let mut output = vec![0.0f32; n_rows];
                        cpu_only
                            .forward_vec(input_ref, &mut output)
                            .expect("forward of the CPU-only layer");
                        (metal_gemv_calls() - before, output)
                    })
                    .join()
                    .expect("forward thread")
            });
            assert_eq!(
                calls, 0,
                "{label}: a layer constructed under CpuOnlyBackendScope called the Metal GEMV"
            );
            let mut expected = vec![0.0f32; n_rows];
            reference(&input, &mut expected).expect("CPU reference GEMV");
            assert_eq!(
                bits(&output),
                bits(&expected),
                "{label}: the CPU-only layer must return its CPU kernel's result bit for bit"
            );
        }

        /// Positive control for Q4_0 / Q8_0: with no scope at construction a
        /// Metal host takes the Metal GEMV, so the zero above is an
        /// observation and not a counter that never moves.
        fn assert_unscoped_takes_metal<'a>(
            label: &str,
            n_rows: usize,
            in_features: usize,
            build: &BuildLayer<'_, 'a>,
        ) {
            if MetalGraph::global().is_err() {
                eprintln!("{label}: no Metal device, so the positive control did not run");
                return;
            }
            let auto = build().expect("construct the layer with no scope active");
            let input = input(in_features);
            let before = metal_gemv_calls();
            let mut output = vec![0.0f32; n_rows];
            auto.forward_vec(&input, &mut output)
                .expect("forward of the auto layer");
            assert_eq!(
                metal_gemv_calls() - before,
                1,
                "{label}: with no CPU-only request a Metal host must take the Metal GEMV"
            );
        }

        #[test]
        fn linear_q4_0_and_q8_0_constructed_under_cpu_only_scope_never_call_metal() {
            let (n_rows, in_features) = (33, 1056);
            let w = weights(n_rows, in_features);

            let q4_0 = BlockQ4_0::quantize(&w).expect("Q4_0 quantize");
            let build_q4_0: &BuildLayer<'_, '_> =
                &|| Ok(LinearQ4_0::new(&q4_0, n_rows, in_features)?.into());
            assert_cpu_only_matches_cpu_kernel("Q4_0", n_rows, in_features, build_q4_0, &|x, y| {
                oxibonsai_kernels::gemv_q4_0(&q4_0, x, y, n_rows, in_features)
            });
            assert_unscoped_takes_metal("Q4_0", n_rows, in_features, build_q4_0);

            let q8_0 = BlockQ8_0::quantize(&w).expect("Q8_0 quantize");
            let build_q8_0: &BuildLayer<'_, '_> =
                &|| Ok(LinearQ8_0::new(&q8_0, n_rows, in_features)?.into());
            assert_cpu_only_matches_cpu_kernel("Q8_0", n_rows, in_features, build_q8_0, &|x, y| {
                oxibonsai_kernels::gemv_q8_0(&q8_0, x, y, n_rows, in_features)
            });
            assert_unscoped_takes_metal("Q8_0", n_rows, in_features, build_q8_0);
        }

        #[test]
        fn linear_kquants_constructed_under_cpu_only_scope_run_the_cpu_kernel() {
            let (n_rows, in_features) = (5, 512);
            let w = weights(n_rows, in_features);

            let q2k = BlockQ2K::quantize(&w).expect("Q2_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q2_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ2K::new(&q2k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q2k(&q2k, x, y, n_rows, in_features),
            );

            let q3k = BlockQ3K::quantize(&w).expect("Q3_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q3_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ3K::new(&q3k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q3k(&q3k, x, y, n_rows, in_features),
            );

            let q4k = BlockQ4K::quantize(&w).expect("Q4_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q4_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ4K::new(&q4k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q4k(&q4k, x, y, n_rows, in_features),
            );

            let q5k = BlockQ5K::quantize(&w).expect("Q5_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q5_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ5K::new(&q5k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q5k(&q5k, x, y, n_rows, in_features),
            );

            let q6k = BlockQ6K::quantize(&w).expect("Q6_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q6_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ6K::new(&q6k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q6k(&q6k, x, y, n_rows, in_features),
            );

            let q8k = BlockQ8K::quantize(&w).expect("Q8_K quantize");
            assert_cpu_only_matches_cpu_kernel(
                "Q8_K",
                n_rows,
                in_features,
                &|| Ok(LinearQ8K::new(&q8k, n_rows, in_features)?.into()),
                &|x, y| oxibonsai_kernels::gemv_q8k(&q8k, x, y, n_rows, in_features),
            );

            // Positive control: the shared dispatcher every unscoped K-quant
            // layer runs through is the GPU tier on a Metal host, so the
            // CPU-tier dispatcher above is a decision and not the only option.
            if MetalGraph::global().is_ok() {
                assert_eq!(
                    kquant_dispatcher(false).tier(),
                    KernelTier::Gpu,
                    "with no CPU-only request the shared K-quant dispatcher is the GPU tier on a Metal host"
                );
            } else {
                eprintln!("K-quant: no Metal device, so the positive control did not run");
            }
        }
    }
}
