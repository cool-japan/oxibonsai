//! Shared test-only counting allocator (`#[cfg(test)]`, this crate's
//! `--lib` unit-test binary only — never compiled into the published
//! library).
//!
//! A `GlobalAlloc` wrapper that counts allocations made while a
//! thread-local "measuring" flag is set on the calling thread, and
//! otherwise behaves exactly like `System`. Scoped per-thread (not one
//! shared atomic): `cargo test`'s default harness runs many tests
//! concurrently on separate threads, and a single global counter would be
//! corrupted by unrelated tests allocating at the same time.
//!
//! Rust permits exactly one `#[global_allocator]` per binary, and this
//! crate's unit-test binary needs one for more than one independent
//! zero-allocation regression test (`layers::attention_fused` here, and
//! `model::types` separately) — declaring it in a single shared module
//! that every such test imports from avoids an
//! `error: cannot define multiple global allocators` merge hazard between
//! them. Every method delegates directly to `System` unconditionally and
//! only the calling thread's counter moves while that thread opted in, so
//! it is behaviorally transparent to every other test sharing this binary.
#![cfg(test)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn note_alloc_on_this_thread() {
    // `try_with` (never panics, e.g. during thread teardown) rather than
    // `with`: an allocator must never panic from inside `alloc`.
    let _ = MEASURING.try_with(|m| {
        if m.get() {
            let _ = ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

// SAFETY: every method delegates directly to `std::alloc::System`; this
// wrapper only adds thread-local bookkeeping around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc_on_this_thread();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc_on_this_thread();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_alloc_on_this_thread();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Run `f`, returning its result and the number of heap allocations made by
/// *this thread* while it ran.
pub(crate) fn count_allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOC_COUNT.with(|c| c.set(0));
    MEASURING.with(|m| m.set(true));
    let result = f();
    MEASURING.with(|m| m.set(false));
    let count = ALLOC_COUNT.with(|c| c.get());
    (result, count)
}
