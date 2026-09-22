//! Pre-allocated scratch buffers for a single `TransformerBlock`'s forward
//! pass.  Eliminates per-token heap allocations in the hot path.
//!
//! Visibility: `pub(super)` so the sibling forward modules can destructure
//! the struct without going through accessor methods.
//!
//! ## `clear()` semantics (M-20)
//!
//! Every buffer here is overwritten by its producer before it is ever read
//! again — either unconditionally on every forward pass (e.g. `normed`,
//! `q_all`), or in lock-step with a runtime condition that gates both the
//! write *and* the read together (e.g. `fused_qkv` is written only when a
//! fused GPU/CPU handle exists for this layer, and is read only inside that
//! same branch; `attn_proj`/`gate_out`/`up_out`/`swiglu_out`/`down_out` are
//! written and read only when the fused-FFN path did *not* fire). Because
//! of that, zeroing all sixteen buffers before every token — roughly 0.2 MB
//! per layer on Ternary-Bonsai-1.7B, summed to ~5.9 MB/token across all 28
//! layers — is pure waste in a release build: nothing ever observes the
//! zero, since every read is preceded by a write on the same pass.
//!
//! `clear()` therefore does nothing in a release build. In a debug build it
//! instead NaN-poisons every buffer and, on the *next* call, asserts that no
//! buffer was left "partially poisoned" — the one shape that indicates a
//! real bug (a producer that only overwrote part of a buffer that then got
//! read, e.g. from a mis-sized loop bound). A buffer that is entirely
//! untouched this pass (still fully poisoned, matching the "written iff
//! read" buffers above when their guard condition didn't fire) or entirely
//! overwritten (no poison left) both satisfy the invariant — see
//! [`ScratchBuffers::assert_buffer_all_or_nothing`].

pub(super) struct ScratchBuffers {
    pub(super) normed: Vec<f32>,
    pub(super) q_all: Vec<f32>,
    pub(super) k_all: Vec<f32>,
    pub(super) v_all: Vec<f32>,
    pub(super) q_normed: Vec<f32>,
    pub(super) k_normed: Vec<f32>,
    pub(super) q_rope: Vec<f32>,
    pub(super) k_rope: Vec<f32>,
    pub(super) attn_out: Vec<f32>,
    pub(super) attn_proj: Vec<f32>,
    pub(super) gate_out: Vec<f32>,
    pub(super) up_out: Vec<f32>,
    pub(super) swiglu_out: Vec<f32>,
    pub(super) down_out: Vec<f32>,
    pub(super) fused_qkv: Vec<f32>,
    pub(super) fused_gate_up: Vec<f32>,
}

impl ScratchBuffers {
    pub(super) fn new(h: usize, nq: usize, nkv: usize, hd: usize, inter: usize) -> Self {
        Self {
            normed: vec![0.0; h],
            q_all: vec![0.0; nq * hd],
            k_all: vec![0.0; nkv * hd],
            v_all: vec![0.0; nkv * hd],
            q_normed: vec![0.0; nq * hd],
            k_normed: vec![0.0; nkv * hd],
            q_rope: vec![0.0; nq * hd],
            k_rope: vec![0.0; nkv * hd],
            attn_out: vec![0.0; nq * hd],
            attn_proj: vec![0.0; h],
            gate_out: vec![0.0; inter],
            up_out: vec![0.0; inter],
            swiglu_out: vec![0.0; inter],
            down_out: vec![0.0; h],
            fused_qkv: vec![0.0; nq * hd + nkv * hd + nkv * hd],
            fused_gate_up: vec![0.0; inter * 2],
        }
    }

    /// Release build: a no-op (every buffer is fully overwritten by its
    /// producer before it is read — see the module doc comment). Debug
    /// build: NaN-poisons every buffer after first checking that the
    /// *previous* pass left none of them partially overwritten.
    pub(super) fn clear(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.assert_previous_pass_not_partially_poisoned();
            self.poison_fill();
        }
    }

    /// Fill every buffer with NaN. Debug-only: this is the poison value
    /// [`Self::assert_previous_pass_not_partially_poisoned`] looks for on
    /// the next call, and NaN propagates loudly through any downstream
    /// float math instead of silently reading a plausible-looking zero.
    #[cfg(debug_assertions)]
    fn poison_fill(&mut self) {
        self.normed.fill(f32::NAN);
        self.q_all.fill(f32::NAN);
        self.k_all.fill(f32::NAN);
        self.v_all.fill(f32::NAN);
        self.q_normed.fill(f32::NAN);
        self.k_normed.fill(f32::NAN);
        self.q_rope.fill(f32::NAN);
        self.k_rope.fill(f32::NAN);
        self.attn_out.fill(f32::NAN);
        self.attn_proj.fill(f32::NAN);
        self.gate_out.fill(f32::NAN);
        self.up_out.fill(f32::NAN);
        self.swiglu_out.fill(f32::NAN);
        self.down_out.fill(f32::NAN);
        self.fused_qkv.fill(f32::NAN);
        self.fused_gate_up.fill(f32::NAN);
    }

    /// Check every buffer for a partial overwrite left by the pass that ran
    /// since the last `clear()`. Called at the *top* of `clear()`, i.e.
    /// after that pass's writes and reads have already happened — this is
    /// the natural place to observe the invariant without needing a hook in
    /// `forward.rs`/`forward_sw.rs`/`forward_stats.rs` (all three already
    /// call `scratch.clear()` unconditionally at the top of every forward).
    #[cfg(debug_assertions)]
    fn assert_previous_pass_not_partially_poisoned(&self) {
        Self::assert_buffer_all_or_nothing(&self.normed, "normed");
        Self::assert_buffer_all_or_nothing(&self.q_all, "q_all");
        Self::assert_buffer_all_or_nothing(&self.k_all, "k_all");
        Self::assert_buffer_all_or_nothing(&self.v_all, "v_all");
        Self::assert_buffer_all_or_nothing(&self.q_normed, "q_normed");
        Self::assert_buffer_all_or_nothing(&self.k_normed, "k_normed");
        Self::assert_buffer_all_or_nothing(&self.q_rope, "q_rope");
        Self::assert_buffer_all_or_nothing(&self.k_rope, "k_rope");
        Self::assert_buffer_all_or_nothing(&self.attn_out, "attn_out");
        Self::assert_buffer_all_or_nothing(&self.attn_proj, "attn_proj");
        Self::assert_buffer_all_or_nothing(&self.gate_out, "gate_out");
        Self::assert_buffer_all_or_nothing(&self.up_out, "up_out");
        Self::assert_buffer_all_or_nothing(&self.swiglu_out, "swiglu_out");
        Self::assert_buffer_all_or_nothing(&self.down_out, "down_out");
        Self::assert_buffer_all_or_nothing(&self.fused_qkv, "fused_qkv");
        Self::assert_buffer_all_or_nothing(&self.fused_gate_up, "fused_gate_up");
    }

    /// A buffer must either be entirely poisoned (untouched this pass — the
    /// "written iff read" guard condition didn't fire) or entirely clean
    /// (fully overwritten). Anything in between means some producer wrote
    /// less than the whole buffer before it — or a sibling buffer sharing
    /// its guard condition — was read.
    ///
    /// Deliberately an O(1) endpoint probe rather than an O(len) scan: this
    /// runs at the top of every forward pass in every debug/test build
    /// (~53K floats per layer x 28 layers x every token on
    /// Ternary-Bonsai-1.7B), so a full scan would measurably slow the test
    /// suite. Every producer on this path writes a buffer via either a
    /// `copy_from_slice` starting at offset 0 or a `for head in 0..n` loop
    /// covering every head starting from head 0, so a bug that leaves a
    /// suffix un-overwritten disagrees with position 0 at the first (and
    /// every later) poisoned index — the two endpoints are as sensitive to
    /// that failure shape as a full scan. Set `OXI_SCRATCH_POISON_FULL_SCAN=1`
    /// to run the exhaustive version while investigating a specific
    /// regression.
    #[cfg(debug_assertions)]
    fn assert_buffer_all_or_nothing(buf: &[f32], label: &str) {
        Self::check_buffer_all_or_nothing(buf, label, Self::full_scan_enabled());
    }

    /// The actual check, with `full_scan` taken as a parameter rather than
    /// read from the environment — split out so tests can exercise each
    /// branch directly and deterministically instead of depending on
    /// `OXI_SCRATCH_POISON_FULL_SCAN` being (un)set in the ambient test
    /// environment.
    #[cfg(debug_assertions)]
    fn check_buffer_all_or_nothing(buf: &[f32], label: &str, full_scan: bool) {
        if buf.is_empty() {
            return;
        }
        if full_scan {
            let nan_count = buf.iter().filter(|x| x.is_nan()).count();
            debug_assert!(
                nan_count == 0 || nan_count == buf.len(),
                "ScratchBuffers.{label}: partially overwritten ({nan_count}/{len} \
                 elements still NaN-poisoned from the previous clear()) -- a producer \
                 on this buffer's write path wrote less than the whole buffer before \
                 it (or a sibling sharing its guard condition) was read",
                len = buf.len(),
            );
        } else {
            let first_nan = buf[0].is_nan();
            let last_nan = buf[buf.len() - 1].is_nan();
            debug_assert!(
                first_nan == last_nan,
                "ScratchBuffers.{label}: endpoints disagree on poison state \
                 (first.is_nan()={first_nan}, last.is_nan()={last_nan}) -- a producer \
                 on this buffer's write path wrote less than the whole buffer before \
                 it (or a sibling sharing its guard condition) was read",
            );
        }
    }

    /// Reads `OXI_SCRATCH_POISON_FULL_SCAN` once per process and caches the
    /// result — cheap enough to call from the hot path.
    #[cfg(debug_assertions)]
    fn full_scan_enabled() -> bool {
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("OXI_SCRATCH_POISON_FULL_SCAN").as_deref() == Ok("1"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(h: usize, nq: usize, nkv: usize, hd: usize, inter: usize) -> ScratchBuffers {
        ScratchBuffers::new(h, nq, nkv, hd, inter)
    }

    #[test]
    fn new_buffers_are_zeroed_and_correctly_sized() {
        // h=8, nq=4, nkv=2, hd=2, inter=6
        let s = make(8, 4, 2, 2, 6);
        assert_eq!(s.normed, vec![0.0; 8]);
        assert_eq!(s.q_all.len(), 4 * 2); // nq * hd
        assert_eq!(s.k_all.len(), 2 * 2); // nkv * hd
        assert_eq!(s.fused_qkv.len(), 4 * 2 + 2 * 2 + 2 * 2); // q_rows + k_rows + v_rows
        assert_eq!(s.fused_gate_up.len(), 6 * 2); // inter * 2
        assert!(s.q_all.iter().all(|&x| x == 0.0));
    }

    /// `clear()` on a debug build must NaN-poison every buffer. This is the
    /// direct behavioral proof that `clear()` is not simply the old
    /// zero-fill under a new name.
    #[cfg(debug_assertions)]
    #[test]
    fn clear_poisons_every_buffer_in_debug() {
        let mut s = make(8, 4, 2, 2, 6);
        s.clear();
        assert!(s.normed.iter().all(|x| x.is_nan()));
        assert!(s.q_all.iter().all(|x| x.is_nan()));
        assert!(s.k_all.iter().all(|x| x.is_nan()));
        assert!(s.v_all.iter().all(|x| x.is_nan()));
        assert!(s.q_normed.iter().all(|x| x.is_nan()));
        assert!(s.k_normed.iter().all(|x| x.is_nan()));
        assert!(s.q_rope.iter().all(|x| x.is_nan()));
        assert!(s.k_rope.iter().all(|x| x.is_nan()));
        assert!(s.attn_out.iter().all(|x| x.is_nan()));
        assert!(s.attn_proj.iter().all(|x| x.is_nan()));
        assert!(s.gate_out.iter().all(|x| x.is_nan()));
        assert!(s.up_out.iter().all(|x| x.is_nan()));
        assert!(s.swiglu_out.iter().all(|x| x.is_nan()));
        assert!(s.down_out.iter().all(|x| x.is_nan()));
        assert!(s.fused_qkv.iter().all(|x| x.is_nan()));
        assert!(s.fused_gate_up.iter().all(|x| x.is_nan()));
    }

    /// A buffer that is fully overwritten (simulating a real producer)
    /// between two `clear()` calls must NOT trip the invariant.
    #[cfg(debug_assertions)]
    #[test]
    fn fully_overwritten_buffer_passes_invariant() {
        let mut s = make(8, 4, 2, 2, 6);
        s.clear(); // first call: initial state is all-zero, trivially passes
        s.normed.fill(1.0); // simulate a producer fully overwriting `normed`
        s.clear(); // must not panic: normed went from all-poison to all-clean
        assert!(s.normed.iter().all(|x| x.is_nan())); // re-poisoned for the next pass
    }

    /// A buffer left entirely untouched (simulating a guard condition that
    /// didn't fire, e.g. `fused_qkv` with no fused handle) must also NOT
    /// trip the invariant.
    #[cfg(debug_assertions)]
    #[test]
    fn entirely_untouched_buffer_passes_invariant() {
        let mut s = make(8, 4, 2, 2, 6);
        s.clear();
        s.clear(); // fused_qkv untouched between the two calls: still all-poison
        assert!(s.fused_qkv.iter().all(|x| x.is_nan()));
    }

    /// A buffer partially overwritten (simulating exactly the K-M1/M-20 bug
    /// class this mechanism exists to catch) MUST trip the invariant via the
    /// real `clear()` path. The `expected` substring is the suffix shared by
    /// both the full-scan and endpoint-probe panic messages (see
    /// `check_buffer_all_or_nothing`), so this test's outcome does not
    /// depend on whether `OXI_SCRATCH_POISON_FULL_SCAN` happens to be set in
    /// the ambient environment -- either branch panics with it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "wrote less than the whole buffer before it")]
    fn partially_overwritten_buffer_fails_invariant() {
        let mut s = make(8, 4, 2, 2, 6);
        s.clear();
        s.gate_out[0] = 1.0; // only the first element gets "written"; rest stays poisoned
        s.clear(); // must panic: gate_out is neither all-poison nor all-clean
    }

    /// The full-scan mode must catch a partial overwrite in the *interior*
    /// of a buffer that the default O(1) endpoint probe cannot see (both
    /// endpoints still agree). Calls `check_buffer_all_or_nothing` directly
    /// with an explicit `full_scan` argument rather than going through the
    /// env-var-cached `assert_buffer_all_or_nothing`, so this is
    /// deterministic regardless of test-execution order or the ambient
    /// environment.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "partially overwritten")]
    fn full_scan_catches_interior_partial_overwrite() {
        let mut buf = [0.0f32; 8];
        buf[0] = f32::NAN;
        buf[7] = f32::NAN;
        // Endpoints agree (both poisoned) but the interior (indices 1..7)
        // is clean -- only the exhaustive scan can see this.
        ScratchBuffers::check_buffer_all_or_nothing(&buf, "test_buf", true);
    }

    /// The default O(1) endpoint probe does NOT catch the same interior
    /// corruption -- the documented, accepted limitation traded for running
    /// at the top of every forward pass instead of only under
    /// `OXI_SCRATCH_POISON_FULL_SCAN=1`. Companion to the test above: same
    /// buffer, opposite `full_scan` argument, opposite outcome.
    #[cfg(debug_assertions)]
    #[test]
    fn endpoint_probe_misses_interior_partial_overwrite() {
        let mut buf = [0.0f32; 8];
        buf[0] = f32::NAN;
        buf[7] = f32::NAN;
        ScratchBuffers::check_buffer_all_or_nothing(&buf, "test_buf", false); // must not panic
    }
}
