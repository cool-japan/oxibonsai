//! Intermediate buffer set for the FFN pipeline, plus low-level allocation,
//! upload/download, and dispatch helpers shared across the directory module.

use metal::{Buffer, CommandBufferRef, Device, MTLCommandBufferStatus, MTLResourceOptions};
use std::ffi::c_void;

use super::error::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the FFN pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Lazily allocated intermediate buffers used by `encode_ffn_phase`.
pub(super) struct MetalBuffers {
    pub(super) hidden_buf: Buffer,
    pub(super) attn_out_buf: Buffer,
    pub(super) norm_weight_buf: Buffer,
    pub(super) proj_buf: Buffer,
    pub(super) normed_buf: Buffer,
    pub(super) swiglu_buf: Buffer,
    pub(super) down_buf: Buffer,
    /// Hidden dimension these buffers were allocated for.
    pub(super) hidden_size: usize,
    /// Intermediate dimension (gate/up half size).
    pub(super) intermediate_size: usize,
}

impl MetalBuffers {
    /// Allocate all intermediate buffers for the given dimensions.
    pub(super) fn allocate(
        device: &Device,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, MetalGraphError> {
        let h_bytes = (hidden_size * std::mem::size_of::<f32>()) as u64;
        let inter_bytes = (intermediate_size * std::mem::size_of::<f32>()) as u64;
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;

        Ok(Self {
            hidden_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload/download
            attn_out_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            norm_weight_buf: alloc_buf(device, h_bytes, shared)?, // CPU upload
            proj_buf: alloc_buf(device, h_bytes, private)?,  // GPU-only intermediate
            normed_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            swiglu_buf: alloc_buf(device, inter_bytes, private)?, // GPU-only intermediate

            down_buf: alloc_buf(device, h_bytes, private)?, // GPU-only intermediate
            hidden_size,
            intermediate_size,
        })
    }

    /// Check whether existing buffers match the requested dimensions.
    pub(super) fn matches(&self, hidden_size: usize, intermediate_size: usize) -> bool {
        self.hidden_size == hidden_size && self.intermediate_size == intermediate_size
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-allocated GPU buffers for the attention pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Reject an allocation request before it ever reaches `Device::new_buffer`.
///
/// `-[MTLDevice newBufferWithLength:options:]` returns `nil` above
/// `maxBufferLength` (or under GPU memory pressure). The `metal` crate's
/// `foreign_types`-based `Buffer` then wraps that pointer via `from_ptr`,
/// which is undefined behavior on a null pointer (a debug-build panic,
/// release-build UB) — checking the size **before** calling `new_buffer` is
/// the only place this can be turned into an ordinary `Result::Err` (MET-06).
///
/// `what` is a short static label identifying the call site, threaded into
/// [`MetalGraphError::BufferTooLarge`] for diagnostics.
fn check_buffer_length(
    byte_len: u64,
    device: &Device,
    what: &'static str,
) -> Result<(), MetalGraphError> {
    let max = device.max_buffer_length();
    if byte_len > max {
        return Err(MetalGraphError::BufferTooLarge {
            what,
            requested: byte_len,
            max,
        });
    }
    Ok(())
}

/// Helper: allocate a Metal buffer, converting a null pointer into an error.
///
/// Checks the requested size against `MTLDevice::maxBufferLength` **before**
/// calling `Device::new_buffer` (see [`check_buffer_length`]); the
/// post-hoc null/length checks below remain as belt-and-braces for any other
/// allocation failure (e.g. transient GPU OOM under the limit).
pub(crate) fn alloc_buf(
    device: &Device,
    byte_len: u64,
    opts: MTLResourceOptions,
) -> Result<Buffer, MetalGraphError> {
    if byte_len == 0 {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    check_buffer_length(byte_len, device, "alloc_buf")?;
    let buf = device.new_buffer(byte_len, opts);
    // StorageModePrivate buffers have contents() == null by design
    if opts.contains(MTLResourceOptions::StorageModePrivate) {
        // For private buffers, just check length as a sanity proxy
        if buf.length() < byte_len {
            return Err(MetalGraphError::BufferCreationFailed);
        }
    } else if buf.contents().is_null() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Upload / download helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Copy a host `f32` slice into a shared Metal buffer.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must be
/// large enough to hold `data.len()` floats.
pub(crate) unsafe fn upload_f32(buf: &Buffer, data: &[f32]) {
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
}

/// Copy from a shared Metal buffer into a host `f32` slice.
///
/// # Safety
///
/// The buffer must have been allocated with `StorageModeShared` and must
/// contain at least `out.len()` floats of valid data.
pub(crate) unsafe fn download_f32(buf: &Buffer, out: &mut [f32]) {
    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), out.len());
}

/// Upload raw bytes (weight data) into a GPU-accessible Metal buffer.
///
/// Uses `StorageModeShared` so the CPU can write directly and the GPU
/// can read without an explicit blit copy.
///
/// Routed through [`alloc_buf`] (rather than calling `Device::new_buffer`
/// directly) so the `maxBufferLength` guard in [`check_buffer_length`] runs
/// before allocation and `buf.contents()` is already known non-null by the
/// time this function's own `copy_nonoverlapping` runs (MET-06).
pub(super) fn upload_bytes(device: &Device, data: &[u8]) -> Result<Buffer, MetalGraphError> {
    if data.is_empty() {
        return Err(MetalGraphError::BufferCreationFailed);
    }
    let opts = MTLResourceOptions::StorageModeShared;
    let buf = alloc_buf(device, data.len() as u64, opts)?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len());
    }
    Ok(buf)
}

// ═══════════════════════════════════════════════════════════════════════════
// Command buffer submission
// ═══════════════════════════════════════════════════════════════════════════

/// Commit a command buffer, wait for GPU completion, and turn any
/// non-`Completed` status into a named [`MetalGraphError`] instead of
/// silently returning whatever bytes happen to be in the output buffer as
/// `Ok(())` (MET-04).
///
/// `-[MTLCommandBuffer waitUntilCompleted]` returns when the command buffer
/// reaches ANY terminal state, including `MTLCommandBufferStatus::Error`
/// (GPU page fault, timeout, device-removed, out-of-memory, ...) — on error
/// the encoded compute never ran, so a caller that unconditionally downloads
/// the output buffer and returns `Ok(())` hands back stale or undefined
/// data. Every `commit()` / `wait_until_completed()` pair in this crate's
/// Metal dispatch code must go through this function instead of calling
/// them directly.
///
/// `what` is a short static label identifying the call site (e.g.
/// `"encode_full_layer"`), threaded into
/// [`MetalGraphError::CommandBufferFailed`] for diagnostics.
///
/// # Contract
///
/// The caller must have already called `end_encoding()` on every encoder
/// attached to `cmd`, and must not call `commit()` on `cmd` again after this
/// returns (Metal command buffers are single-use).
pub(crate) fn commit_and_wait(
    cmd: &CommandBufferRef,
    what: &'static str,
) -> Result<(), MetalGraphError> {
    cmd.commit();
    cmd.wait_until_completed();
    let status = cmd.status();
    let error = if status == MTLCommandBufferStatus::Completed {
        None
    } else {
        // SAFETY: `wait_until_completed()` has already returned above, so
        // the command buffer has reached a terminal state and reading its
        // `error` property is safe (mirrors the GPUStartTime / GPUEndTime
        // pattern at `metal_full_layer/functions.rs`).
        unsafe { command_buffer_error_description(cmd) }
    };
    map_command_buffer_status(status, what, error)
}

/// Map a terminal `MTLCommandBufferStatus` to `Ok(())` (only for
/// `Completed`) or a named [`MetalGraphError::CommandBufferFailed`] for
/// anything else. This is the exact check that was missing before MET-04:
/// without it, every status — including `Error` — fell through to `Ok(())`
/// and the caller downloaded/returned whatever bytes already happened to be
/// in the output buffer.
///
/// Split out of [`commit_and_wait`] so this decision is unit-testable with
/// synthetic statuses: neither an oversized-threadgroup dispatch nor a wildly
/// out-of-bounds GPU write reliably produces a non-`Completed` status on
/// Apple Silicon (both were tried against this function's caller and both
/// silently completed — Apple GPUs tolerate a lot before actually faulting),
/// so a real hardware-triggered `Error` status cannot be produced
/// deterministically in this test suite; see `commit_and_wait`'s own tests
/// for the real end-to-end happy path this pairs with.
fn map_command_buffer_status(
    status: MTLCommandBufferStatus,
    what: &'static str,
    error: Option<String>,
) -> Result<(), MetalGraphError> {
    match status {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(MetalGraphError::CommandBufferFailed {
            what,
            status,
            error,
        }),
    }
}

/// Read `-[MTLCommandBuffer error].localizedDescription`, if the driver
/// supplied an `NSError`.
///
/// `metal` 0.33 does not expose `-[MTLCommandBuffer error]` as a safe
/// binding, so this reads it directly via the `objc` crate — the same
/// pattern the project already uses for `GPUStartTime`/`GPUEndTime` at
/// `metal_full_layer/functions.rs`. Returns `None` (rather than failing)
/// whenever any step of the `NSError` → `NSString` → `CStr` chain yields a
/// null pointer, so a missing description never masks the real
/// [`MetalGraphError::CommandBufferFailed { status, .. }`] this backs.
///
/// # Safety
/// Must be called only after `wait_until_completed()` (or an equivalent
/// synchronization point) has returned for `cmd`.
unsafe fn command_buffer_error_description(cmd: &CommandBufferRef) -> Option<String> {
    let err_obj: *mut objc::runtime::Object = msg_send![cmd, error];
    if err_obj.is_null() {
        return None;
    }
    let desc_obj: *mut objc::runtime::Object = msg_send![err_obj, localizedDescription];
    if desc_obj.is_null() {
        return None;
    }
    let bytes: *const std::os::raw::c_char = msg_send![desc_obj, UTF8String];
    if bytes.is_null() {
        return None;
    }
    Some(
        std::ffi::CStr::from_ptr(bytes)
            .to_string_lossy()
            .into_owned(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute threadgroup count: `ceil(n / divisor)`, guaranteed >= 1.
#[inline]
pub(crate) fn div_ceil(n: usize, divisor: usize) -> usize {
    n.div_ceil(divisor)
}

/// Convenience: `set_bytes` for a single scalar value at a given buffer index.
///
/// # Safety
///
/// The encoder must be in a valid state and `index` must not collide with
/// any buffer binding.
pub(crate) unsafe fn set_scalar<T: Copy>(
    encoder: &metal::ComputeCommandEncoderRef,
    index: u64,
    value: &T,
) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const c_void,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (MET-04 / MET-06) — real Metal hardware required; skip cleanly when
// no device is present (headless / non-Apple-Silicon CI).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests {
    use super::*;
    use metal::{CompileOptions, MTLSize};

    // ── MET-04: commit_and_wait ─────────────────────────────────────────

    /// Real end-to-end happy path: an empty command buffer (no encoded
    /// work) commits, completes, and `commit_and_wait` returns `Ok(())`.
    /// This is the simplest possible proof that the added status check does
    /// not spuriously reject a normal, successful submission.
    #[test]
    fn commit_and_wait_empty_command_buffer_is_ok() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return, // no Metal device on this host — skip
        };
        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let result = commit_and_wait(cmd, "test_empty_command_buffer");
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    }

    /// Real end-to-end happy path with an actual compute dispatch: data
    /// written by the GPU kernel is visible after `commit_and_wait` returns,
    /// proving the success path still waits for and surfaces real work
    /// (not just a no-op command buffer).
    #[test]
    fn commit_and_wait_real_dispatch_completes_with_correct_data() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let queue = device.new_command_queue();
        let src = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_double(device float* buf [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
                buf[tid] = buf[tid] * 2.0;
            }
        "#;
        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(src, &opts)
            .expect("compile test_double kernel");
        let func = lib
            .get_function("test_double", None)
            .expect("get_function test_double");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline test_double");

        let n: usize = 8;
        let buf = alloc_buf(
            &device,
            (n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("alloc test buffer");
        let input: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        unsafe { upload_f32(&buf, &input) };

        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buf), 0);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(n as u64, 1, 1));
        encoder.end_encoding();

        let result = commit_and_wait(cmd, "test_real_dispatch");
        assert!(result.is_ok(), "expected Ok, got {result:?}");

        let mut output = vec![0f32; n];
        unsafe { download_f32(&buf, &mut output) };
        for (i, &v) in output.iter().enumerate() {
            let expected = i as f32 * 2.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: got {v}, expected {expected}"
            );
        }
    }

    /// Direct unit test of the status→`Result` mapping that `commit_and_wait`
    /// delegates to.
    ///
    /// A real hardware-triggered non-`Completed` status could not be
    /// produced deterministically on this hardware: neither an
    /// oversized-threadgroup dispatch (4x `max_total_threads_per_threadgroup`)
    /// nor a wildly out-of-bounds GPU write (~4 GB past a 4-byte buffer) made
    /// `cmd.status()` come back as anything other than `Completed` — Apple
    /// Silicon GPUs tolerate both without faulting. Rather than ship a test
    /// that would pass today for the wrong reason (and silently stop
    /// covering anything if the driver's tolerance ever changes), this
    /// exercises the exact decision `commit_and_wait` makes with every
    /// status Metal can report, via [`MTLCommandBufferStatus`]'s own public,
    /// freely-constructible variants — no GPU required.
    #[test]
    fn map_command_buffer_status_completed_is_ok() {
        assert!(map_command_buffer_status(MTLCommandBufferStatus::Completed, "t", None).is_ok());
    }

    /// The MET-04 regression itself: before the fix, `commit_and_wait`'s
    /// predecessor (a bare `commit()` + `wait_until_completed()`) never
    /// looked at the status at all, so a GPU fault fell through to `Ok(())`
    /// and the caller returned whatever bytes were already sitting in the
    /// output buffer (the previous token's contents) as a "successful"
    /// result. This asserts the fixed mapping instead returns a named `Err`
    /// carrying the real status and driver message.
    #[test]
    fn map_command_buffer_status_error_is_err_not_ok() {
        let result = map_command_buffer_status(
            MTLCommandBufferStatus::Error,
            "test_site",
            Some("Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)".to_string()),
        );
        match result {
            Err(MetalGraphError::CommandBufferFailed {
                what,
                status,
                error,
            }) => {
                assert_eq!(what, "test_site");
                assert_eq!(status, MTLCommandBufferStatus::Error);
                assert_eq!(
                    error.as_deref(),
                    Some("Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)")
                );
            }
            other => panic!("expected Err(CommandBufferFailed), got {other:?}"),
        }
    }

    /// Every non-`Completed` status must map to `Err`, not just `Error` —
    /// including the pre-terminal ones (`NotEnqueued`/`Enqueued`/
    /// `Committed`/`Scheduled`), which should never be observed after
    /// `wait_until_completed()` returns, but must not be silently treated as
    /// success if they somehow are.
    #[test]
    fn map_command_buffer_status_every_non_completed_status_is_err() {
        for status in [
            MTLCommandBufferStatus::NotEnqueued,
            MTLCommandBufferStatus::Enqueued,
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
            MTLCommandBufferStatus::Error,
        ] {
            let result = map_command_buffer_status(status, "t", None);
            assert!(result.is_err(), "status {status:?} should map to Err");
        }
    }

    // ── MET-06: buffer-size guard ────────────────────────────────────────

    /// A request exactly at the device's `maxBufferLength` is accepted.
    #[test]
    fn check_buffer_length_accepts_at_max() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let max = device.max_buffer_length();
        assert!(check_buffer_length(max, &device, "t").is_ok());
    }

    /// A request one byte over `maxBufferLength` is rejected with a named,
    /// diagnosable error — not silently passed through to `new_buffer`
    /// (which would return `nil`, and whose `foreign_types` wrapper then
    /// treats a `nil` pointer as UB, per MET-06).
    #[test]
    fn check_buffer_length_rejects_one_byte_over_max() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let max = device.max_buffer_length();
        let result = check_buffer_length(max + 1, &device, "test_alloc");
        match result {
            Err(MetalGraphError::BufferTooLarge {
                what,
                requested,
                max: reported_max,
            }) => {
                assert_eq!(what, "test_alloc");
                assert_eq!(requested, max + 1);
                assert_eq!(reported_max, max);
            }
            other => panic!("expected Err(BufferTooLarge), got {other:?}"),
        }
    }

    /// `alloc_buf` — the shared allocator every weight/KV/scratch buffer in
    /// this crate goes through — gets a named `Err`, not a null-pointer
    /// panic/UB, for a request above `maxBufferLength`. This is the ACCEPTANCE
    /// criterion "a test requesting a buffer larger than `max_buffer_length()`
    /// gets a named Err, not SIGSEGV": the guard runs before `new_buffer` is
    /// ever called, so no huge (multi-GB) allocation is needed to exercise it.
    #[test]
    fn alloc_buf_rejects_over_max_buffer_length() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let max = device.max_buffer_length();
        let result = alloc_buf(&device, max + 1, MTLResourceOptions::StorageModeShared);
        assert!(
            matches!(result, Err(MetalGraphError::BufferTooLarge { .. })),
            "expected Err(BufferTooLarge), got {result:?}"
        );
    }

    /// `upload_bytes` now routes through `alloc_buf` (see its own doc
    /// comment), so the over-`maxBufferLength` guard applies to it too. A
    /// literal `&[u8]` longer than `maxBufferLength` (tens of GB) cannot be
    /// constructed here without either actually allocating that much host
    /// memory or invoking undefined behavior (a slice must be backed by
    /// real, valid memory for its entire declared length) — `alloc_buf`'s
    /// test above exercises the identical guard on the identical code path
    /// `upload_bytes` reduces to. This test instead locks down
    /// `upload_bytes`'s own realistic behavior: a normal small upload
    /// round-trips, and an empty slice is still rejected.
    #[test]
    fn upload_bytes_happy_path_roundtrips() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let buf = upload_bytes(&device, &data).expect("upload_bytes should succeed");
        let mut readback = vec![0u8; data.len()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents() as *const u8,
                readback.as_mut_ptr(),
                data.len(),
            );
        }
        assert_eq!(readback, data);
    }

    #[test]
    fn upload_bytes_rejects_empty_slice() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        assert!(matches!(
            upload_bytes(&device, &[]),
            Err(MetalGraphError::BufferCreationFailed)
        ));
    }
}
