//! `MET-08` acceptance: concurrent Metal sessions must produce **byte-identical**
//! results to the same work run sequentially, and a GPU engine pool must
//! actually scale.
//!
//! Before `MET-08` every GPU dispatch in the process went through one
//! `MetalGraph`: one command queue, one device KV cache, one set of full-layer
//! and prefill scratch buffers. The `MutexGuard`s taken at the top of a fused
//! forward were held through every layer's encode plus `commit()` and
//! `wait_until_completed()`, so two inference replicas could not overlap — and
//! because `GpuKvCache::matches` compares only the *shape*
//! `(n_layers, n_kv, max_seq, head_dim)`, two replicas of the same model would
//! have silently trampled each other's attention state. `engine_pool` clamped
//! GPU pools to a single replica for exactly that reason.
//!
//! The type is now split: a process-global `MetalDevice` (device, compiled
//! pipelines, weight cache) and a per-session `MetalGraph` (its own command
//! queue and its own workspace, KV cache included). Replicas that key a weight
//! to the same slot share one copy of it — the ternary route does (1x weights
//! for any number of replicas), the Q1 route does not yet (see
//! `metal_graph/session.rs`, *Sizing reality*). This file is the evidence:
//!
//! 1. two sessions are distinct objects that nonetheless share the weight
//!    cache;
//! 2. binding is scoped, re-entrant and thread-local;
//! 3. **per-session state is really per session**: two threads decode two
//!    different 50-token sequences through the fused ternary path (device KV
//!    cache, full-layer buffers, logits buffer — the state `MET-08` split) in
//!    strict lock-step and fully concurrently, each in its own session, and
//!    reproduce the sequential baseline bit for bit — while the *same* run
//!    with both threads in **one** session diverges, which is what proves the
//!    test can see a shared KV cache at all;
//! 4. GEMM dispatch from concurrently bound sessions stays correct (a
//!    dispatch-correctness check only — see the note on those two tests);
//! 5. an unbound thread still gets the process-default session, so
//!    single-session behaviour is unchanged;
//! 6. the engine pool hands every replica its own session and concurrent
//!    generation matches isolated baselines token for token;
//! 7. **measured** on the real model (`OXI_MODEL`): a GPU pool of 1, 2 and 3
//!    replicas serving 8 concurrent requests — distinct sessions, 1x resident
//!    weights, byte-identical outputs, aggregate throughput that does not fall
//!    as replicas are added.
//!
//! Every GPU test no-ops on a host without a Metal device, and the real-model
//! test prints a skip report when `OXI_MODEL` is unset, so the suite is green
//! on CI machines with no GPU. All test names start with `metal_concurrency_`
//! so `cargo test -p oxibonsai-runtime --features metal metal_concurrency`
//! selects exactly this file.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use oxibonsai_kernels::{MetalDevice, MetalGraph, MetalWeightHandle};

/// Iterations of the concurrency stress tests: decode steps per sequence in
/// the per-session-state test, GEMMs per worker in the dispatch test.
const STRESS_ITERATIONS: usize = 50;

/// GEMM shape for the dispatch test: `m x k` input against a `n_rows x k` weight.
const M: usize = 8;
/// Rows of the weight matrix (output width).
const N_ROWS: usize = 64;
/// Reduction dimension.
const K: usize = 128;

/// `true` when this host has no Metal device, so the GPU tests should no-op.
fn no_gpu() -> bool {
    MetalGraph::new_session().is_err()
}

/// A deterministic f32 weight matrix, distinct per `seed`.
fn weight_bytes(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(N_ROWS * K * 4);
    for row in 0..N_ROWS {
        for col in 0..K {
            let v = ((row * 7 + col * 13 + seed * 31) % 19) as f32 * 0.125 - 1.0;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes
}

/// A deterministic `m x k` input, distinct per `seed`.
fn input(seed: usize) -> Vec<f32> {
    (0..M * K)
        .map(|i| ((i + seed * 17) % 23) as f32 * 0.0625 - 0.5)
        .collect()
}

/// Run one GEMM in `session` and return the output row-major.
fn gemm(session: &MetalGraph, weight: &MetalWeightHandle, seed: usize) -> Vec<f32> {
    let mut out = vec![0f32; M * N_ROWS];
    session
        .encode_gemm_f32(weight, &input(seed), &mut out, M, N_ROWS, K)
        .expect("encode_gemm_f32");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Two sessions are distinct, and share the device + weight cache
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_sessions_are_distinct_but_share_the_weight_cache() {
    // Two sessions on one (isolated) device — the relationship every
    // `MetalGraph::new_session()` has to the process-global device — so the
    // counts below are this test's alone even when the suite runs in
    // parallel (the real-model pool test reads the global device's gauge).
    let Ok(device) = MetalDevice::isolated() else {
        return;
    };
    let a = MetalGraph::new_session_on(&device);
    let b = MetalGraph::new_session_on(&device);
    assert_ne!(
        a.session_id(),
        b.session_id(),
        "each session must be individually identifiable"
    );

    // Sharing the weight cache is what lets replicas that key a weight to one
    // slot hold one copy of it: an upload made through session `a` is visible
    // in session `b`'s accounting, and asking `b` for the same key hands back
    // the *same* GPU buffer rather than uploading a second copy.
    let key = 0x4d45_5430_3800_0001; // "MET08" + a slot nobody else uses
    let bytes = weight_bytes(1);
    let before = b.cached_weight_count().expect("cached count");
    let from_a = a.get_or_upload_weight(key, &bytes).expect("upload via a");
    let after_upload = b.cached_weight_count().expect("cached count");
    assert_eq!(
        after_upload,
        before + 1,
        "an upload in one session must land in the shared cache"
    );

    let uploads_before = b.weight_upload_count();
    let from_b = b.get_or_upload_weight(key, &bytes).expect("lookup via b");
    assert_eq!(
        b.weight_upload_count(),
        uploads_before,
        "a sibling session must hit the cache, not upload a second copy"
    );
    assert!(
        Arc::ptr_eq(&from_a, &from_b),
        "both sessions must be handed the same GPU buffer"
    );

    a.evict_f32_weight(key).expect("evict");
    assert_eq!(b.cached_weight_count().expect("cached count"), before);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Binding is scoped, re-entrant, and thread-local
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_binding_is_scoped_and_reentrant() {
    if no_gpu() {
        return;
    }
    let outer = MetalGraph::new_session().expect("outer session");
    let inner = MetalGraph::new_session().expect("inner session");

    assert_eq!(
        MetalGraph::current_session_id(),
        None,
        "a fresh thread starts unbound"
    );

    MetalGraph::with_session(&outer, || {
        assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
        assert_eq!(
            MetalGraph::global().expect("global").session_id(),
            outer.session_id(),
            "global() must resolve to the bound session"
        );

        MetalGraph::with_session(&inner, || {
            assert_eq!(MetalGraph::current_session_id(), Some(inner.session_id()));
        });

        assert_eq!(
            MetalGraph::current_session_id(),
            Some(outer.session_id()),
            "leaving an inner scope must restore the outer binding"
        );

        // Re-binding the session that is already bound must not clear it when
        // that redundant scope ends.
        MetalGraph::with_session(&outer, || {
            assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
        });
        assert_eq!(MetalGraph::current_session_id(), Some(outer.session_id()));
    });

    assert_eq!(
        MetalGraph::current_session_id(),
        None,
        "the binding must not outlive its scope"
    );

    // Bindings are per thread: a sibling thread sees none of this one's.
    let seen = std::thread::spawn(MetalGraph::current_session_id)
        .join()
        .expect("join");
    assert_eq!(seen, None, "a binding must not leak across threads");
}

/// The lease shape (acquire + `Deref` on a tokio worker, then used and
/// dropped on a `spawn_blocking` thread): once the using thread binds the
/// session, the worker's earlier binding is stale — an unleased dispatch left
/// on the worker resolves to the process-default session, never into the
/// replica the other thread is running (a Metal concurrency finding).
#[test]
fn metal_concurrency_a_moved_binding_goes_stale_on_the_old_thread() {
    if no_gpu() {
        return;
    }
    let replica = MetalGraph::new_session().expect("replica session");
    let id = replica.session_id();
    MetalGraph::bind_current(&replica);
    assert_eq!(MetalGraph::current_session_id(), Some(id));

    let moved = Arc::clone(&replica);
    let used_there = std::thread::spawn(move || {
        MetalGraph::bind_current(&moved);
        let bound = MetalGraph::global().expect("global there").session_id();
        MetalGraph::unbind_current_if(moved.session_id());
        bound
    })
    .join()
    .expect("using thread");
    assert_eq!(used_there, id);

    assert_eq!(MetalGraph::current_session_id(), None);
    assert_ne!(
        MetalGraph::global().expect("global here").session_id(),
        id,
        "the old thread must not keep dispatching into the moved replica"
    );
}

#[test]
fn metal_concurrency_unbound_threads_use_the_process_default_session() {
    if no_gpu() {
        return;
    }
    // The single-session invariant: with nobody binding anything, every caller
    // — on any thread — resolves to one session, exactly as before `MET-08`.
    // This is what makes a single-request run byte-identical to the old code.
    let here = MetalGraph::global().expect("global here").session_id();
    let there = std::thread::spawn(|| MetalGraph::global().expect("global there").session_id())
        .join()
        .expect("join");
    assert_eq!(
        here, there,
        "unbound threads must share the process-default session"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Per-session state: two decodes in lock-step and concurrently == sequential
// ─────────────────────────────────────────────────────────────────────────────

mod ternary_fixture {
    //! A synthetic, fully-ternary 2-layer model (h = 128, 64-token context),
    //! small enough to decode 50 steps in milliseconds and large enough to go
    //! through the fused Metal ternary path end to end.

    use half::f16;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    pub const HIDDEN: usize = 128;
    pub const INTER: usize = 256;
    pub const LAYERS: usize = 2;
    pub const NQ: usize = 4;
    pub const NKV: usize = 2;
    pub const HD: usize = 32;
    pub const VOCAB: usize = 32;
    pub const MAX_SEQ: usize = 64;

    /// `TQ2_0_g128` blocks emitting only the three ternary codes.
    fn tq2(num_weights: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out = Vec::with_capacity(num_weights / 128 * 34);
        for _ in 0..num_weights / 128 {
            for _ in 0..32 {
                let mut byte = 0u8;
                for lane in 0..4 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    byte |= (((state >> 33) % 3) as u8) << (2 * lane);
                }
                out.push(byte);
            }
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let scale = 0.2 + ((state >> 33) % 1000) as f32 * 0.0004;
            out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        out
    }

    fn f32s(n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (scale * (1.0 + 0.25 * (i as f32 * 0.013).sin())).to_le_bytes())
            .collect()
    }

    /// The model as GGUF bytes.
    pub fn gguf_bytes() -> Vec<u8> {
        let mut w = GgufWriter::new();
        let meta: [(&str, MetadataWriteValue); 11] = [
            (
                "general.architecture",
                MetadataWriteValue::Str("qwen3".into()),
            ),
            (
                "general.name",
                MetadataWriteValue::Str("Met08Stress".into()),
            ),
            (
                "qwen3.embedding_length",
                MetadataWriteValue::U32(HIDDEN as u32),
            ),
            ("qwen3.block_count", MetadataWriteValue::U32(LAYERS as u32)),
            (
                "qwen3.attention.head_count",
                MetadataWriteValue::U32(NQ as u32),
            ),
            (
                "qwen3.attention.head_count_kv",
                MetadataWriteValue::U32(NKV as u32),
            ),
            (
                "qwen3.feed_forward_length",
                MetadataWriteValue::U32(INTER as u32),
            ),
            ("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32)),
            ("qwen3.context_length", MetadataWriteValue::U32(512)),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            ),
            ("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0)),
        ];
        for (key, value) in meta {
            w.add_metadata(key, value);
        }
        let mut tensor = |name: String, shape: Vec<u64>, tensor_type: TensorType, data: Vec<u8>| {
            w.add_tensor(TensorEntry {
                name,
                shape,
                tensor_type,
                data,
            });
        };
        let (h, inter) = (HIDDEN as u64, INTER as u64);
        tensor(
            "token_embd.weight".into(),
            vec![h, VOCAB as u64],
            TensorType::F32,
            f32s(VOCAB * HIDDEN, 0.5),
        );
        tensor(
            "output_norm.weight".into(),
            vec![h],
            TensorType::F32,
            f32s(HIDDEN, 1.0),
        );
        tensor(
            "output.weight".into(),
            vec![h, VOCAB as u64],
            TensorType::TQ2_0_g128,
            tq2(VOCAB * HIDDEN, 0xCAFE),
        );
        for l in 0..LAYERS {
            let p = format!("blk.{l}");
            for (n, len) in [
                ("attn_norm", HIDDEN),
                ("ffn_norm", HIDDEN),
                ("attn_q_norm", HD),
                ("attn_k_norm", HD),
            ] {
                tensor(
                    format!("{p}.{n}.weight"),
                    vec![len as u64],
                    TensorType::F32,
                    f32s(len, 1.0),
                );
            }
            let s = 0x1000 + (l as u64) * 16;
            let (q, kv) = ((NQ * HD) as u64, (NKV * HD) as u64);
            for (n, rows, cols, bump) in [
                ("attn_q", h, q, 0),
                ("attn_k", h, kv, 1),
                ("attn_v", h, kv, 2),
                ("attn_output", q, h, 3),
                ("ffn_gate", h, inter, 4),
                ("ffn_up", h, inter, 5),
                ("ffn_down", inter, h, 6),
            ] {
                tensor(
                    format!("{p}.{n}.weight"),
                    vec![rows, cols],
                    TensorType::TQ2_0_g128,
                    tq2((rows * cols) as usize, s + bump),
                );
            }
        }
        w.to_bytes().expect("GgufWriter::to_bytes")
    }
}

/// The two teacher-forced token sequences the workers decode (different, so
/// a shared KV cache would mix them).
fn sequence(worker: usize) -> Vec<u32> {
    (0..STRESS_ITERATIONS)
        .map(|p| ((p * (7 + 4 * worker) + 3 * worker + 1) % ternary_fixture::VOCAB) as u32)
        .collect()
}

/// Logits of every step as raw bits.
type Trace = Vec<Vec<u32>>;

/// Decode one step through the fused ternary path and return its logit bits.
fn step(model: &oxibonsai_model::model::BonsaiModel<'_>, token: u32, pos: usize) -> Vec<u32> {
    model
        .forward_logits_gpu_ternary_cached(token, pos)
        .expect("fused ternary decode step")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// How the two workers of [`run_two_workers`] are scheduled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Schedule {
    /// A's step `p`, then B's step `p`, then A's step `p + 1`, …
    LockStep,
    /// Both workers run step `p` at the same time.
    Concurrent,
}

/// A two-party rendezvous that **fails instead of hanging**: if the peer
/// thread panicked (it breaks the rendezvous while unwinding) or nothing
/// arrives within [`Rendezvous::TIMEOUT`], `wait` panics with a message. A
/// plain `std::sync::Barrier` would leave the surviving worker parked forever
/// and turn a test failure into a hung test binary.
struct Rendezvous {
    /// `(arrived, generation, broken)`.
    state: std::sync::Mutex<(usize, u64, bool)>,
    cv: std::sync::Condvar,
}

impl Rendezvous {
    /// Far above one fused decode step of the fixture (~ms), far below a CI
    /// timeout.
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

    fn new() -> Self {
        Self {
            state: std::sync::Mutex::new((0, 0, false)),
            cv: std::sync::Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        assert!(!state.2, "the peer worker failed before this rendezvous");
        state.0 += 1;
        if state.0 == 2 {
            state.0 = 0;
            state.1 += 1;
            self.cv.notify_all();
            return;
        }
        let generation = state.1;
        let deadline = std::time::Instant::now() + Self::TIMEOUT;
        while state.1 == generation && !state.2 {
            let now = std::time::Instant::now();
            assert!(
                now < deadline,
                "the peer worker never reached the rendezvous"
            );
            state = self
                .cv
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        assert!(
            !state.2,
            "the peer worker failed while this one was waiting"
        );
    }

    /// Wake every waiter with a failure (called while unwinding).
    fn break_all(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.2 = true;
        self.cv.notify_all();
    }
}

/// Breaks the rendezvous points it guards if its thread unwinds.
struct BreakOnPanic<'r>(&'r [&'r Rendezvous]);

impl Drop for BreakOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            for rendezvous in self.0 {
                rendezvous.break_all();
            }
        }
    }
}

/// Run `work` on worker 0 and then on worker 1 — never both at once — each
/// with its own session bound immediately before its turn; both workers
/// return together once worker 1's turn is over.
fn in_turn<R>(
    worker: usize,
    session: &Arc<MetalGraph>,
    before: &Rendezvous,
    after: &Rendezvous,
    work: impl FnOnce() -> R,
) -> R {
    if worker == 1 {
        before.wait();
    }
    MetalGraph::bind_current(session);
    let out = work();
    if worker == 0 {
        before.wait();
    }
    after.wait();
    out
}

/// Decode [`sequence`]`(0)` and `(1)` on two threads, each through its own
/// `BonsaiModel` replica of one GGUF, worker `w` dispatching in `sessions[w]`.
///
/// A worker binds its session immediately before each piece of GPU work,
/// exactly as `EngineLease::deref` does before every use, and at no other
/// time: a session belongs to the thread that bound it last, so in the control
/// run — **one** session shared by both threads — any bind outside a worker's
/// own turn would claim the session away from its peer in the middle of the
/// peer's work. Everything that is not a decode step therefore runs
/// [`in_turn`]: the GPU-cache warm-up, and the final drop of the replicas
/// (whose refcounted `Drop` releases the weights from the bound session's
/// device). Loading a replica is host-only and needs no binding. The decode
/// loop is where the schedules differ.
fn run_two_workers(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    sessions: [Arc<MetalGraph>; 2],
    schedule: Schedule,
) -> [Trace; 2] {
    let before = Rendezvous::new();
    let after = Rendezvous::new();
    let [session_0, session_1] = sessions;
    std::thread::scope(|scope| {
        let spawn = |worker: usize, session: Arc<MetalGraph>| {
            let (before, after) = (&before, &after);
            scope.spawn(move || {
                let guarded = [before, after];
                let _break_on_panic = BreakOnPanic(&guarded);
                let model =
                    oxibonsai_model::model::BonsaiModel::from_gguf(gguf, ternary_fixture::MAX_SEQ)
                        .expect("load replica");
                in_turn(worker, &session, before, after, || {
                    model.get_or_create_gpu_cache().expect("warm replica");
                });
                let tokens = sequence(worker);
                let mut trace = Trace::with_capacity(tokens.len());
                for (pos, &token) in tokens.iter().enumerate() {
                    let logits = match schedule {
                        Schedule::LockStep => {
                            in_turn(worker, &session, before, after, || step(&model, token, pos))
                        }
                        Schedule::Concurrent => {
                            before.wait();
                            MetalGraph::bind_current(&session);
                            let logits = step(&model, token, pos);
                            after.wait();
                            logits
                        }
                    };
                    trace.push(logits);
                }
                in_turn(worker, &session, before, after, move || drop(model));
                MetalGraph::unbind_current_if(session.session_id());
                trace
            })
        };
        let worker_0 = spawn(0, session_0);
        let worker_1 = spawn(1, session_1);
        [
            worker_0.join().expect("worker 0"),
            worker_1.join().expect("worker 1"),
        ]
    })
}

/// The per-session-state acceptance (`MET-08`): two sessions keep two device
/// KV caches (and two full-layer / logits workspaces).
///
/// Two different 50-token sequences are decoded through the fused ternary
/// path — which reads every earlier position's K/V back out of the session's
/// device KV cache at each step — on two threads:
///
/// - in **strict lock-step** (A's step `p`, then B's step `p`, …), so any
///   shared KV cache would have B's writes land between A's writes and A's
///   next read;
/// - fully **concurrently** (both threads submit step `p` at once), so any
///   shared queue, buffer or lock-free scratch would race.
///
/// Both must reproduce the sequential single-session baseline **bit for bit**.
/// The control then runs the same lock-step with both threads in **one**
/// session — the pre-`MET-08` world — and worker 0 must diverge. Without the
/// control this test could not tell a per-session KV cache from a shared one.
///
/// Every session sits on one isolated device, so this test's weight uploads
/// never touch the process-global device's accounting.
#[test]
fn metal_concurrency_two_sessions_keep_separate_kv_caches_under_lockstep_and_concurrency() {
    let Ok(device) = MetalDevice::isolated() else {
        return;
    };
    let new_session = || MetalGraph::new_session_on(&device);
    let bytes = ternary_fixture::gguf_bytes();
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse fixture");

    // Sequential baseline: sequence 0, then sequence 1, in one session. Each
    // sequence starts at position 0 and rewrites every KV slot it reads.
    let baseline: [Trace; 2] = {
        let session = new_session();
        MetalGraph::with_session(&session, || {
            let model =
                oxibonsai_model::model::BonsaiModel::from_gguf(&gguf, ternary_fixture::MAX_SEQ)
                    .expect("load baseline model");
            model.get_or_create_gpu_cache().expect("warm baseline");
            [0usize, 1].map(|worker| {
                sequence(worker)
                    .iter()
                    .enumerate()
                    .map(|(pos, &token)| step(&model, token, pos))
                    .collect::<Trace>()
            })
        })
    };
    assert_eq!(baseline[0].len(), STRESS_ITERATIONS);
    assert_ne!(baseline[0], baseline[1], "the two sequences must differ");

    for schedule in [Schedule::LockStep, Schedule::Concurrent] {
        let sessions = [new_session(), new_session()];
        let got = run_two_workers(&gguf, sessions, schedule);
        for worker in 0..2 {
            let first_bad = got[worker]
                .iter()
                .zip(&baseline[worker])
                .position(|(a, b)| a != b);
            assert_eq!(
                first_bad, None,
                "{schedule:?}: worker {worker} diverged from the sequential baseline at step \
                 {first_bad:?} — per-session KV / workspace state leaked across sessions"
            );
        }
    }

    // Control: one session shared by both threads, strict lock-step.
    let shared = new_session();
    let got = run_two_workers(&gguf, [Arc::clone(&shared), shared], Schedule::LockStep);
    assert_ne!(
        got[0], baseline[0],
        "with ONE shared session, worker 0 must read worker 1's KV writes and diverge — if it \
         does not, this test cannot detect a shared KV cache and the assertions above prove \
         nothing"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. GEMM dispatch from concurrently bound sessions
// ─────────────────────────────────────────────────────────────────────────────
//
// Scope note: `encode_gemm_f32` stages its
// input and output in the **process-wide** DiT I/O pool, whose mutex is held
// across the whole upload → dispatch → wait → download. The two tests below
// therefore do not overlap their GEMMs, touch none of the per-session state
// `MET-08` split, and would pass even if that state were shared again. They
// are kept as what they are — a dispatch-correctness check: every session's
// command queue, its thread binding and the shared pipelines produce the
// right floats when sessions are created, bound and used from several threads
// at once. The per-session-state evidence is the lock-step test above.

#[test]
fn metal_concurrency_two_sessions_match_two_sequential_runs_bitwise() {
    if no_gpu() {
        return;
    }

    // ── Baseline: both workloads run one after the other, in one session ──
    let baseline_session = MetalGraph::new_session().expect("baseline session");
    let mut baselines = Vec::new();
    for worker in 0..2usize {
        let handle = baseline_session
            .upload_weight(&weight_bytes(worker))
            .expect("upload baseline weight");
        let mut per_iteration = Vec::with_capacity(STRESS_ITERATIONS);
        for iteration in 0..STRESS_ITERATIONS {
            per_iteration.push(gemm(&baseline_session, &handle, worker * 1000 + iteration));
        }
        baselines.push(per_iteration);
    }

    // ── Concurrent: one session per worker, bound on its own thread ──
    let baselines = Arc::new(baselines);
    let mut threads = Vec::new();
    for worker in 0..2usize {
        let baselines = Arc::clone(&baselines);
        threads.push(std::thread::spawn(move || {
            let session = MetalGraph::new_session().expect("worker session");
            MetalGraph::with_session(&session, || {
                // Every `MetalGraph::global()` inside this closure — including
                // the ones the model and kernel stack make on its behalf — is
                // this worker's session.
                let bound = MetalGraph::global().expect("bound global");
                assert_eq!(bound.session_id(), session.session_id());

                let handle = session
                    .upload_weight(&weight_bytes(worker))
                    .expect("upload worker weight");
                for iteration in 0..STRESS_ITERATIONS {
                    let got = gemm(&session, &handle, worker * 1000 + iteration);
                    assert_eq!(
                        got, baselines[worker][iteration],
                        "worker {worker} iteration {iteration} diverged from the sequential \
                         baseline"
                    );
                }
            });
            session.session_id()
        }));
    }

    let ids: Vec<u64> = threads
        .into_iter()
        .map(|t| t.join().expect("worker thread"))
        .collect();
    assert_ne!(ids[0], ids[1], "the two workers must not share a session");
}

#[test]
fn metal_concurrency_many_sessions_stay_independent_under_load() {
    if no_gpu() {
        return;
    }
    // Five workers, one above the default session ceiling
    // (`OXIBONSAI_METAL_MAX_SESSIONS` unset = 4): creating sessions is never
    // refused — the ceiling is a *pool-sizing* policy, not a hard limit — and
    // each one still computes its own workload correctly while the others
    // hammer the same device.
    let workers = MetalGraph::max_sessions() + 1;
    let reference = MetalGraph::new_session().expect("reference session");
    let expected: Vec<Vec<f32>> = (0..workers)
        .map(|w| {
            let handle = reference
                .upload_weight(&weight_bytes(w + 7))
                .expect("upload reference weight");
            gemm(&reference, &handle, w + 7)
        })
        .collect();

    let expected = Arc::new(expected);
    let mut threads = Vec::new();
    for worker in 0..workers {
        let expected = Arc::clone(&expected);
        threads.push(std::thread::spawn(move || {
            let session = MetalGraph::new_session().expect("worker session");
            let handle = session
                .upload_weight(&weight_bytes(worker + 7))
                .expect("upload worker weight");
            for _ in 0..STRESS_ITERATIONS / 5 {
                assert_eq!(
                    gemm(&session, &handle, worker + 7),
                    expected[worker],
                    "worker {worker} diverged under concurrent load"
                );
            }
            session.session_id()
        }));
    }
    let mut ids: Vec<u64> = threads
        .into_iter()
        .map(|t| t.join().expect("worker thread"))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), workers, "every worker must get its own session");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Pool sizing follows the session ceiling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn metal_concurrency_gpu_pool_sizing_follows_the_session_ceiling() {
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_runtime::engine_pool::{gpu_max_replicas, resolve_pool_sizing_with_gpu_max};

    // The GPU tier is no longer clamped to one replica: it is capped by how
    // many sessions the host can afford (one device KV cache each).
    let sizing = resolve_pool_sizing_with_gpu_max(Some(3), KernelTier::Gpu, 3);
    assert_eq!(sizing.effective, 3);
    assert!(!sizing.clamped_by_gpu_tier);

    // Above the ceiling the request is capped *and reported* as capped, so an
    // admission controller can shed rather than queue invisibly (`perf-M1`).
    let capped = resolve_pool_sizing_with_gpu_max(Some(9), KernelTier::Gpu, 2);
    assert_eq!(capped.effective, 2);
    assert!(capped.clamped_by_gpu_tier);
    assert_eq!(capped.gpu_max, Some(2));

    assert!(
        gpu_max_replicas() >= 1,
        "a pool must have at least a replica"
    );
    assert_eq!(
        gpu_max_replicas(),
        MetalGraph::max_sessions().max(1),
        "pool sizing must follow the kernels crate's session ceiling"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. The engine pool: a session per replica, and identical output under load
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn metal_concurrency_pool_replicas_hold_their_own_sessions_and_agree_with_baselines() {
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::engine_pool::EnginePool;
    use oxibonsai_runtime::sampling::SamplingParams;

    // This mirrors the server's real shape: acquire a lease on the async
    // runtime, then move it into `spawn_blocking` and generate there — which
    // is precisely why the lease binds its session in `Deref` (on the thread
    // that *uses* it) rather than in `acquire` (on the thread that took it).
    //
    // Two things are asserted: (a) inside the blocking thread the bound
    // session is the lease's own, and it is released again when the lease
    // drops; (b) three replicas generating simultaneously each reproduce the
    // token sequence they produce alone. On a Metal host `auto_detect` still
    // resolves this synthetic config to a CPU tier, so (a) is vacuous here
    // (`None == None`); the GPU-tier half — distinct `Some` sessions per live
    // replica — is asserted on a real GPU pool by
    // `metal_concurrency_real_model_pool_scaling_and_byte_identity`. (b)
    // holds on every tier.
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let seed = 42u64;
    let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    let max_tokens = 4usize;

    // Isolated baselines: one fresh engine per prompt, run alone.
    let baselines: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let mut engine = InferenceEngine::new(config.clone(), params.clone(), seed);
            engine.generate(p, max_tokens).expect("baseline generate")
        })
        .collect();

    let engines: Vec<InferenceEngine<'static>> = (0..prompts.len())
        .map(|_| InferenceEngine::new(config.clone(), params.clone(), seed))
        .collect();
    let pool = EnginePool::new(engines);

    // Lease every replica *before* any generation starts, so each task holds a
    // distinct, untouched replica for its whole life — otherwise a fast task
    // could return its replica and a later one reuse it, comparing a second
    // run against a first-run baseline.
    let mut leases = Vec::new();
    for _ in 0..prompts.len() {
        leases.push(pool.acquire().await.expect("acquire"));
    }
    let ids: Vec<Option<u64>> = leases.iter().map(|l| l.gpu_session_id()).collect();
    for (i, a) in ids.iter().enumerate() {
        for b in ids.iter().skip(i + 1) {
            if let (Some(a), Some(b)) = (a, b) {
                assert_ne!(a, b, "two live replicas share one Metal session");
            }
        }
    }

    let mut handles = Vec::new();
    for (lease, prompt) in leases.into_iter().zip(prompts.clone()) {
        handles.push(tokio::task::spawn_blocking(move || {
            let mut lease = lease;
            let expected_session = lease.gpu_session_id();
            let out = lease.generate(&prompt, max_tokens).expect("generate");
            // The lease bound its session on the blocking thread, not on the
            // async worker that acquired it.
            let bound_during = MetalGraph::current_session_id();
            drop(lease);
            // ...and released it again on the way out, so an unleased dispatch
            // on this thread falls back to the process-default session.
            let bound_after = MetalGraph::current_session_id();
            (out, expected_session, bound_during, bound_after)
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let (got, expected_session, bound_during, bound_after) = h.await.expect("join");
        assert_eq!(
            bound_during, expected_session,
            "request {i} did not run in its replica's session"
        );
        assert_eq!(
            bound_after, None,
            "request {i} left its session bound after the lease dropped"
        );
        assert_eq!(
            got, baselines[i],
            "concurrent request {i} diverged from its isolated baseline"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Measured on the real model: pool size 1, 2, 3 × 8 concurrent requests
// ─────────────────────────────────────────────────────────────────────────────

mod pool_measurement {
    //! Shared by the real-model pool test: run a batch of concurrent greedy
    //! requests through a pool the way the server does (acquire on the async
    //! runtime, generate inside `spawn_blocking`).

    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use oxibonsai_runtime::engine_pool::EnginePool;

    /// Outcome of one batch.
    pub struct Batch {
        /// Wall time from the first request's submission to the last reply.
        pub elapsed: Duration,
        /// Generated tokens per request, in submission order.
        pub outputs: Vec<Vec<u32>>,
        /// The session each request ran in (`None` off the GPU tier).
        pub sessions: Vec<Option<u64>>,
        /// Per request: submission of the batch → that request's reply.
        pub latencies: Vec<Duration>,
    }

    impl Batch {
        /// Generated tokens across the batch.
        pub fn tokens(&self) -> usize {
            self.outputs.iter().map(Vec::len).sum()
        }

        /// Aggregate generated tokens per second.
        pub fn tokens_per_second(&self) -> f64 {
            self.tokens() as f64 / self.elapsed.as_secs_f64().max(f64::EPSILON)
        }

        /// Mean request completion latency, in milliseconds.
        pub fn mean_latency_ms(&self) -> f64 {
            let n = self.latencies.len().max(1) as f64;
            self.latencies
                .iter()
                .map(Duration::as_secs_f64)
                .sum::<f64>()
                * 1e3
                / n
        }

        /// Fastest request completion latency, in milliseconds — the request
        /// that waited least behind the others.
        pub fn min_latency_ms(&self) -> f64 {
            self.latencies
                .iter()
                .map(Duration::as_secs_f64)
                .fold(f64::INFINITY, f64::min)
                * 1e3
        }
    }

    /// Submit every prompt at once and wait for all of them.
    pub async fn run(pool: &Arc<EnginePool>, prompts: &[Vec<u32>], max_tokens: usize) -> Batch {
        let start = Instant::now();
        let mut handles = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let pool = Arc::clone(pool);
            let prompt = prompt.clone();
            handles.push(tokio::spawn(async move {
                let lease = pool.acquire().await.expect("acquire a replica");
                let (out, session) = tokio::task::spawn_blocking(move || {
                    let mut lease = lease;
                    let session = lease.gpu_session_id();
                    let out = lease
                        .generate(&prompt, max_tokens)
                        .expect("greedy generation on the GPU pool");
                    (out, session)
                })
                .await
                .expect("blocking generation task");
                (out, session, start.elapsed())
            }));
        }
        let mut outputs = Vec::with_capacity(prompts.len());
        let mut sessions = Vec::with_capacity(prompts.len());
        let mut latencies = Vec::with_capacity(prompts.len());
        for handle in handles {
            let (out, session, latency) = handle.await.expect("request task");
            outputs.push(out);
            sessions.push(session);
            latencies.push(latency);
        }
        Batch {
            elapsed: start.elapsed(),
            outputs,
            sessions,
            latencies,
        }
    }
}

/// ACCEPTANCE "measured serve throughput scales with the pool size", on the
/// real **Ternary-Bonsai-1.7B** (`OXI_MODEL`).
///
/// Three GPU pools — 1, 2 and 3 replicas — each serve the same 8 concurrent
/// short greedy requests, the way the server does. Measured and asserted:
///
/// - every live replica of a GPU pool dispatches in its **own** session
///   (`Some`, pairwise distinct — the GPU-tier half of the pool test above);
/// - **1x weights**: warming a pool of 1, 2 or 3 replicas adds the same
///   resident weight bytes to the shared device (ternary slots are
///   address-derived, so replicas share them);
/// - **byte identity**: every request's tokens equal the pool-of-1 baseline,
///   whichever replica served it, in every round;
/// - **scaling**, as a relative invariant only (never absolute ms, the host
///   is shared): every round serves the same batch through the 1-, 2- and
///   3-replica pools back to back, and the **median over rounds** of the
///   2-replica pool's aggregate tokens/s divided by the 1-replica pool's *in
///   the same round* is at least 1.0. Pairing within a round cancels drifting
///   background load, which a best-of-rounds comparison does not (one quiet
///   round for the 1-replica pool alone could decide it).
///
/// The measured numbers are printed. Skips with a capability report when
/// `OXI_MODEL` is unset or the model does not resolve to the GPU tier.
///
/// # What the numbers mean (measured on an M3, 2026-09-23)
///
/// Aggregate throughput rises only a few percent from 1 to 2 or 3 replicas
/// (median paired speed-ups of 1.03–1.05× over several runs on a shared host;
/// single rounds swing with background load), and the reason is the GPU, not
/// a lock. With `OXIBONSAI_PROFILE_GPU=1` a single stream shows `wall ≈
/// gpu_exec ≈ 13-15 ms` per token (under 0.5 ms of CPU encode), while two
/// concurrent streams show `gpu_exec` unchanged per command buffer and
/// `wall ≈ 2 × gpu_exec`: each session's command buffer is committed while
/// the other's is executing (so no host-side lock is holding it back —
/// `MET-08` removed those) and the GPU then runs the two back to back.
/// Batch-1 decode already occupies this GPU for the whole token, so replicas
/// add request **concurrency** — one in-flight request per replica progresses
/// side by side instead of queueing behind the others, at the price of each
/// finishing later (the printed first-reply latency grows with the pool) —
/// and overlap the host-side work, but not GPU throughput; multiplying that
/// would take batched decode — one command buffer serving several sequences
/// per weight read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metal_concurrency_real_model_pool_scaling_and_byte_identity() {
    use oxibonsai_kernels::KernelTier;
    use oxibonsai_runtime::engine_pool::build_pool_from_gguf;
    use oxibonsai_runtime::sampling::SamplingParams;

    let Some(path) = std::env::var_os("OXI_MODEL") else {
        eprintln!(
            "metal_concurrency_real_model_pool_scaling_and_byte_identity: OXI_MODEL not set — \
             skipping (set OXI_MODEL=<Ternary-Bonsai-1.7B.gguf> to measure GPU pool scaling)"
        );
        return;
    };
    if no_gpu() {
        eprintln!("metal_concurrency_real_model_pool_scaling_and_byte_identity: no Metal device");
        return;
    }
    const REQUESTS: usize = 8;
    const MAX_TOKENS: usize = 24;
    const MAX_SEQ: usize = 256;
    // Odd, so the median paired speed-up is one measured round.
    const ROUNDS: usize = 7;
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: MAX_TOKENS,
    };
    // Eight short prompts (token ids well inside every Qwen3 vocabulary).
    let prompts: Vec<Vec<u32>> = (0..REQUESTS)
        .map(|r| (0..6).map(|i| (1000 + r * 977 + i * 131) as u32).collect())
        .collect();
    let shared = MetalGraph::global().expect("process-default session");

    let mut pools = Vec::new();
    for replicas in 1..=3usize {
        let resident_before = shared.bytes_uploaded();
        let (pool, tier, size) =
            build_pool_from_gguf(&path, params.clone(), 42, MAX_SEQ, Some(replicas))
                .expect("build the pool");
        if tier != KernelTier::Gpu {
            eprintln!(
                "metal_concurrency_real_model_pool_scaling_and_byte_identity: OXI_MODEL resolved \
                 to {tier}, not the GPU tier — skipping"
            );
            return;
        }
        assert_eq!(size, replicas, "the pool must hold what was asked for");

        // Distinct sessions for every live replica (GPU tier: `Some` each).
        let mut leases = Vec::new();
        for _ in 0..replicas {
            leases.push(pool.acquire().await.expect("acquire"));
        }
        let mut ids: Vec<u64> = leases
            .iter()
            .map(|l| {
                l.gpu_session_id()
                    .expect("a GPU-tier replica must own a session")
            })
            .collect();
        drop(leases);
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            replicas,
            "live replicas must not share a session"
        );

        // Warm-up: one request per replica (weights upload, KV allocation).
        let warm = pool_measurement::run(&pool, &prompts[..replicas], MAX_TOKENS).await;
        assert!(warm.tokens() > 0, "the warm-up generated nothing");
        let weight_bytes = shared.bytes_uploaded().saturating_sub(resident_before);
        pools.push((replicas, pool, weight_bytes));
    }

    // 1x weights: the resident weight bytes a pool adds do not grow with its
    // replica count.
    let (_, _, one_replica_bytes) = &pools[0];
    assert!(*one_replica_bytes > 0, "warming a pool uploaded no weights");
    for (replicas, _, bytes) in &pools {
        assert_eq!(
            bytes, one_replica_bytes,
            "a {replicas}-replica ternary pool holds {bytes} weight bytes vs {one_replica_bytes} \
             for one replica — replicas must share one copy"
        );
    }

    // Timed rounds: each round runs the three pools back to back, so the
    // pools of one round see the same background load and are compared with
    // each other only.
    let mut tps_by_round: Vec<[f64; 3]> = Vec::with_capacity(ROUNDS);
    let mut baseline: Option<Vec<Vec<u32>>> = None;
    let mut table = String::new();
    for round in 0..ROUNDS {
        let mut this_round = [0f64; 3];
        for (index, (replicas, pool, _)) in pools.iter().enumerate() {
            let batch = pool_measurement::run(pool, &prompts, MAX_TOKENS).await;
            let tps = batch.tokens_per_second();
            this_round[index] = tps;
            let mut used: Vec<u64> = batch.sessions.iter().flatten().copied().collect();
            used.sort_unstable();
            used.dedup();
            table.push_str(&format!(
                "  round {round}  pool {replicas}: {} tokens in {:>7.1} ms = {tps:>6.1} tok/s; \
                 request latency first/mean {:>6.1}/{:>6.1} ms ({} sessions served the {} \
                 requests)\n",
                batch.tokens(),
                batch.elapsed.as_secs_f64() * 1e3,
                batch.min_latency_ms(),
                batch.mean_latency_ms(),
                used.len(),
                prompts.len()
            ));
            match &baseline {
                None => baseline = Some(batch.outputs),
                Some(expected) => {
                    for (i, (got, want)) in batch.outputs.iter().zip(expected).enumerate() {
                        assert_eq!(
                            got, want,
                            "pool {replicas} round {round}: request {i} is not byte-identical \
                             to the pool-1 baseline"
                        );
                    }
                }
            }
        }
        tps_by_round.push(this_round);
    }

    // Best-of-rounds aggregate per pool (printed), and the paired per-round
    // speed-up over the 1-replica pool, whose median is what is asserted.
    let best = |index: usize| {
        tps_by_round
            .iter()
            .map(|round| round[index])
            .fold(0f64, f64::max)
    };
    let paired_ratios = |index: usize| {
        let mut ratios: Vec<f64> = tps_by_round
            .iter()
            .map(|round| round[index] / round[0].max(f64::EPSILON))
            .collect();
        ratios.sort_by(f64::total_cmp);
        ratios
    };
    let (ratios_2, ratios_3) = (paired_ratios(1), paired_ratios(2));
    let median_2 = ratios_2[ROUNDS / 2];
    let median_3 = ratios_3[ROUNDS / 2];
    eprintln!(
        "MET-08 pool scaling on {} ({REQUESTS} concurrent greedy requests x {MAX_TOKENS} \
         tokens):\n{table}  best-of-{ROUNDS} aggregate: pool 1 = {:.1} tok/s, pool 2 = {:.1} \
         tok/s, pool 3 = {:.1} tok/s\n  paired per-round speed-up over pool 1 (median of \
         {ROUNDS}, range): pool 2 = {median_2:.3}x ({:.3}-{:.3}), pool 3 = {median_3:.3}x \
         ({:.3}-{:.3})\n  resident weights per pool = {:.2} MB with 1, 2 and 3 replicas",
        std::path::Path::new(&path).display(),
        best(0),
        best(1),
        best(2),
        ratios_2[0],
        ratios_2[ROUNDS - 1],
        ratios_3[0],
        ratios_3[ROUNDS - 1],
        *one_replica_bytes as f64 / 1e6,
    );
    assert!(
        median_2 >= 1.0,
        "a 2-replica GPU pool must not serve 8 concurrent requests slower than a 1-replica pool: \
         median paired speed-up {median_2:.3}x over {ROUNDS} rounds"
    );
}
