#!/usr/bin/env bash
# OxiBonsai — the release gate: scripts/ci.sh in strict mode, plus the
# hardware-capability enforcement that a plain CI run cannot provide (T-05).
#
# T-05: every hardware-dependent test in this workspace historically
# self-skipped to GREEN (metal_k_quant_gemv_parity.rs, the cuda_* tests,
# rag_real_generation_tests.rs, image/tests/parity_gguf.rs) by returning
# `Ok(())`/early-`return`ing from inside the test body when the needed
# hardware/fixture was absent — indistinguishable, from nextest's point of
# view, from the test actually having validated something. No amount of
# looking at `cargo nextest`'s exit code can tell the two apart; only the
# test body itself knows which one happened. This script is the consumer
# half of the fix: it enforces that the capability manifest below actually
# contains fresh, real evidence for every capability this HOST requires.
#
# ── THE CAPABILITY MANIFEST CONTRACT (read this before writing a producer) ──
# Path:   the ABSOLUTE path in the `OXIBONSAI_CAPABILITY_REPORT` environment
#         variable. Both this script and scripts/ci.sh export it, before
#         running anything that might produce a record, as:
#           export OXIBONSAI_CAPABILITY_REPORT="<absolute>/capability-report.json"
#         (the absolute directory is `${CARGO_TARGET_DIR:-target}`, resolved
#         against the project root if it was a relative path).
#         PRODUCERS MUST READ `OXIBONSAI_CAPABILITY_REPORT` AND WRITE THERE —
#         a producer test that instead resolves `${CARGO_TARGET_DIR:-target}`
#         itself, relative to its OWN process, gets the WRONG path and the
#         write is silently lost: `cargo nextest run` spawns each test
#         binary with its current directory set to that test's OWN crate
#         root (e.g. `crates/oxibonsai-kernels`), not the workspace root, so
#         a relative `target/capability-report.json` resolves to
#         `crates/oxibonsai-kernels/target/capability-report.json` — a file
#         this gate never reads — instead of the manifest at the workspace
#         root's target dir. The result looks exactly like "the capability
#         genuinely didn't execute": a FALSE "NO manifest records" verdict
#         for a test that ran correctly. A producer invoked completely
#         outside either gate script (e.g. `cargo test` by hand, with
#         neither script's `export` in its environment) should fall back to
#         `env!("CARGO_MANIFEST_DIR")`-relative `../../target/capability-report.json`
#         (or read `CARGO_TARGET_DIR` itself) only in that case, and say so
#         in a doc comment next to the fallback.
# Format: JSON LINES — one compact JSON object per line, APPENDED (never
#         rewritten) by each test that exercises a gated capability. JSONL,
#         not a single JSON array, because `cargo nextest` runs test
#         binaries as separate parallel OS processes: a shared JSON array
#         would need a read-modify-write cycle that concurrent processes
#         would corrupt or silently lose writes to. Every write must be a
#         single `write` syscall of a complete, newline-terminated line
#         (e.g. open the file with `OpenOptions::new().create(true).append(true)`
#         and write one `serde_json::to_string(&record)? + "\n"` per call) so
#         concurrent appends interleave safely.
# Each line's object:
#   {"capability": "metal", "executed": true, "test": "oxibonsai-kernels::metal_k_quant_gemv_parity::metal_gemv_q2k_matches_scalar"}
#   - "capability" (string, required): one of "metal", "cuda", "rag-real-generation",
#     "image-parity", "legacy-models", "bonsai2-models", "bonsai2-metal-engine",
#     "cuda-hardware", "bonsai2-mmproj", "bonsai2-vision", "metal-hidden",
#     "bonsai2-vision-metal"
#     (extend this list here, in the same edit that adds a new gated
#     capability, so this file stays the single source of truth).
#     "metal-hidden" is the head-free Metal hidden-state prefill behind dense
#     embeddings, checked against the per-token reference on the real legacy
#     models — see `Capability::MetalHidden`'s doc comment in
#     `crates/oxibonsai-testkit/src/capability.rs`.
#     "bonsai2-mmproj" is the real Bonsai 2 vision projector
#     (`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`) bound and checked against the
#     f64 reference (`Capability::Bonsai2Mmproj`); "bonsai2-vision" is the
#     27B plus that projector end to end on the CPU path — the model-level
#     golden gate and the chat-endpoint round trip (`Capability::Bonsai2Vision`).
#     "bonsai2-vision-metal" is the same image path on Metal
#     (`Capability::Bonsai2VisionMetal`): the 27B image prompts on the Metal
#     hybrid runner and the Metal vision tower against the CPU path and the
#     fork's image goldens (both bands), the image chat round trip on a Metal
#     engine, and the shipped CLI end to end (`oxibonsai run --mmproj
#     --backend auto` stays on the Metal runner and prints the golden answer,
#     the same with `--backend cpu`, and `oxibonsai serve` answers the image
#     request). The Metal tower on the real projector records
#     "bonsai2-mmproj".
#     "cuda-hardware" is distinct from "cuda": it names validation whose own
#     claim is stronger than "the CUDA code path did not panic" (an
#     end-to-end numeric parity result, or a checklist item never yet
#     exercised on real hardware) — see
#     `crates/oxibonsai-testkit/src/capability.rs`'s
#     `Capability::CudaHardware` doc comment. No `REQUIRED_CAPS` entry
#     currently gates on it, since no host that has ever produced this
#     project's capability manifest has had a CUDA device, but a future
#     host that does have one can require it the same way `--require-cuda`
#     requires "cuda".
#     "cuda-syntax" appears only in waiver records ("waived", below); nothing requires it.
#   - "executed" (bool, required): true iff the test body actually reached
#     and ran the hardware/fixture-dependent code path this run — NOT
#     merely "the process exited 0". A test that detects the capability is
#     absent and takes the self-skip path must still write a record, with
#     "executed": false, so an all-skipped run is visibly distinct in the
#     manifest from a manifest that is simply missing (the latter means the
#     producer test was never converted to use this contract at all).
#   - "test" (string, required): the fully-qualified test name, for
#     diagnostics when a required capability's evidence is missing/stale.
#   - "duration_ms" (integer, OPTIONAL): the record's own measured
#     wall-clock cost in milliseconds (`oxibonsai_testkit::capability::
#     record_timed`/`record_executed_timed`), omitted entirely by any call
#     site that has not opted into measuring it. `check_capability_manifest`
#     below prints it on each capability's own "OK" line when at least one
#     matching record carries it, so the real-model gates' relative cost is
#     visible in this script's own output without re-running under a
#     stopwatch. Every real-model gate records it; a required test
#     (`--require-tests`) whose executed record has no `duration_ms` earns a
#     `WARN` line naming it — the report is then incomplete, but a missing
#     timer never turns the gate red.
#   - "waived" (string, OPTIONAL): marks a record of an explicit, flag-granted
#     WAIVER instead of hardware evidence; only THIS script writes one, never a
#     test. The one value is "approximate-accepted", with "capability":
#     "cuda-syntax", "executed": false, "test": "scripts/check_cuda.sh", appended
#     once ci.sh passed under `--accept-approximate-cuda-syntax` and its
#     cuda-syntax stage reported `CUDA_SYNTAX_RESULT=approximate-accepted` (a real
#     nvcc pass waives nothing and writes none). `check_capability_manifest`
#     ignores it (it is no "cuda" evidence): it is there for the reader.
# This script MOVES ASIDE any manifest a previous run left (to
# `capability-report.json.<UTC timestamp>.<pid>` beside it, so that earlier
# evidence is kept) before invoking scripts/ci.sh, and only trusts entries
# written *during this run* (mtime check) — a leftover file from a previous
# invocation, or one for a different feature set, must never be read as this
# run's evidence. The run therefore starts from no manifest and the final
# check reads this run's records alone. Producers do not need to truncate the
# file themselves; appending is always correct.
#
# Required-on-this-host capabilities (matches this script's own spec):
#   - "metal" is required whenever `uname -s` is Darwin (this project
#     targets Apple Silicon as a first-class backend).
#   - "cuda" is required only when --require-cuda is passed (the development
#     machine has no CUDA hardware at all; do not require it by
#     default — see CONTEXT: "CUDA code is compile-blind").
#   - ci.sh's `cuda-syntax` stage needs a real nvcc, which a host without the CUDA
#     toolkit (every macOS host) cannot supply: --accept-approximate-cuda-syntax is
#     the explicit, never-implied opt-out (a clean APPROXIMATE check, no nvcc
#     compile; recorded as a waiver and named in the verdict; see "waived"). It
#     never waives a syntax error and is refused with --require-cuda.
#   - "legacy-models" is required on Darwin unless --skip-legacy-models is
#     passed. It is the evidence that the real-model CPU/NEON/Metal greedy
#     parity gate — the strongest correctness signal this project can
#     produce, and the one that guards README's "byte-identical output at
#     --temperature 0" claim — actually RAN against the shipped GGUFs. The
#     producer tests are not `#[ignore]`d (design decision D-6(4)), so on a
#     host that has models/ they run and record; on a host that does not
#     they record executed=false, and a release must not be cut from such a
#     run without the operator saying so out loud. --skip-legacy-models is
#     that explicit statement; it is visible in this script's own output.
#   - "metal-hidden" is required on Darwin under the same condition as
#     "legacy-models" (it is evidence about the same three legacy GGUFs, and
#     --skip-legacy-models drops it too): stage 1b runs
#     `oxibonsai-model::metal_hidden_parity_tests` once per legacy model file
#     (`OXI_MODEL` names one GGUF per run) and the check requires that test's
#     own name to have an `executed: true` record, so a run that self-skipped
#     — no model resolved, or a build without the Metal feature — fails the
#     gate instead of passing on the strength of some lighter Metal test.
#   - "legacy-models" ALSO covers the real-model gates stage 1b adds beyond
#     the cross-tier matrix, each required by name: the speculative gate
#     (`oxibonsai-model::speculative_ternary_metal_gates`, one test per
#     ternary model: speculative greedy against plain greedy on the Metal
#     route, at draft lengths below and above the 8-row tiled-GEMM boundary
#     and across the adaptive lookahead's range), the embedding leg
#     (`oxibonsai-runtime::embeddings_model_backed`'s `embed_bench_short_and_long`
#     and `real_model_places_queen_nearer_to_king_than_banana`) and the M-18
#     fused-prefill chunk-size sweep (`oxibonsai-model`'s
#     `real_model_metal_prefill_chunk_size_sweep`, run once on Bonsai-8B and
#     once on the Ternary-Bonsai-1.7B). The benchmark opts in explicitly —
#     `OXIBONSAI_EMBED_BENCH=1` and `OXI_MODEL` are both set by this script —
#     so the leg never self-skips; it never sets
#     `OXIBONSAI_EMBED_BENCH_PER_TOKEN` (and removes an inherited one), so at
#     2000 tokens the benchmark compares the production Metal embedding with
#     the batched CPU pass and asserts the 20 s target instead of paying the
#     per-token reference loop, which stays a manual measurement.
#   - "bonsai2-models" is required on Darwin unless --skip-bonsai2-models is
#     passed, same shape as "legacy-models" above, for the Bonsai 2 27B
#     target instead of the 1.7B/8B/1-bit legacy models: the evidence that
#     `oxibonsai-model::hybrid_forward_parity_tests`'s three real-27B gates
#     (PQ2_0, PTQ1_0, the f64-layer reference),
#     `oxibonsai-runtime::bonsai2_engine_tests`'s two real-27B gates and
#     `oxibonsai-runtime::bonsai2_runtime_tests`'s two real-vocabulary/
#     chat-template gates (G6/G7) and two `oxibonsai info --json` gates (G9,
#     one per quant band) actually RAN against the shipped
#     `Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf` files, not merely that they
#     self-skipped cleanly. --skip-bonsai2-models is the explicit "this
#     release carries no such evidence" statement. This script additionally
#     requires each of those nine gates' OWN test name to have an
#     `executed: true` record (`check_capability_manifest`'s
#     `--require-tests` option, below) — a single record from any one of them
#     is not enough, unlike a plain capability name.
#   - "bonsai2-models" ALSO covers, on Darwin unless --skip-bonsai2-metal is
#     passed, `oxibonsai-model::hybrid_metal_gates`'s two Metal 27B gates
#     (CPU-vs-Metal token parity and decode throughput) — required by name,
#     same as the five gates above — and "bonsai2-metal-engine" is required
#     alongside it: `oxibonsai-runtime::bonsai2_metal_engine_tests`'s four
#     Metal engine gates (the engine on the Metal hybrid runner vs the CPU
#     engine and the fork for each band, engine decode throughput and
#     memory, and a `/v1/chat/completions` round trip), also by name.
#     --skip-bonsai2-metal is the narrower opt out: it drops only the
#     Metal-specific evidence requirements, not the whole "bonsai2-models"
#     capability. Under the same condition "bonsai2-models" also requires
#     `oxibonsai-model::hybrid_metal_prefill_gates`'s two gates by name (the
#     runner's batched prefill against its decode rate on both bands, and the
#     token and logprob gates with every prefill on the GEMM).
#   - "bonsai2-vision" and "bonsai2-mmproj" are required on Darwin, by name,
#     whenever the real vision projector is present next to the 27B and
#     neither --skip-bonsai2-models nor --skip-bonsai2-vision was passed:
#     `oxibonsai-model::bonsai2_vision_tests` (the 27B and its projector
#     against the fork's image goldens, CPU path),
#     `oxibonsai-runtime::bonsai2_vision_runtime_tests`'s real-27B chat round
#     trip and `oxibonsai-model::vision_mmproj_tests`'s real-projector gate
#     all ran for real. A host that has the 27B but NOT the projector fails
#     closed — a release must not ship `--mmproj` without that evidence —
#     unless --skip-bonsai2-vision says so out loud.
#   - Wherever the vision leg runs, its Metal half runs and is required too,
#     unless --skip-bonsai2-metal is passed: "bonsai2-vision-metal" by the
#     names of `oxibonsai-model::bonsai2_vision_metal_tests`'s gate (both
#     bands), `oxibonsai-runtime::bonsai2_vision_metal_runtime_tests`'s round
#     trip and `oxibonsai-cli::bonsai2_vision_cli_tests`'s CLI gate, and
#     "bonsai2-mmproj" additionally by the name of
#     `oxibonsai-model::vision_metal_tests`'s Metal-tower gate. It needs both
#     27B bands beside the projector and fails closed, before invoking
#     cargo, when the second band is missing.
#
# ── STAGE 0: THE RELEASE CLI BINARY IS BUILT ONCE, BEFORE ANYTHING ELSE ─────
# Two real-model gates (`bonsai2_runtime_tests`'s `oxibonsai info --json` cases
# and `legacy_parity_tests`'s `oxibonsai eval --task mmlu` case) run the
# shipped `oxibonsai` binary as a subprocess. A test must not build that
# binary itself: an in-test `cargo build` (a) compiles for minutes inside the
# real-model critical section, (b) with a narrower feature set than the
# release ships, overwrites a wider binary in a shared target directory and
# silently voids every later leg that expected e.g. the Metal backend, and
# (c) leaves the test executing `<manifest>/../../target/release/oxibonsai`
# even when `CARGO_TARGET_DIR` put the build elsewhere. Stage 0 therefore
# builds the binary ONCE — `cargo build --release -p oxibonsai-cli --bin
# oxibonsai --all-features --message-format=json`, in the caller's
# `CARGO_TARGET_DIR` —
# takes the executable path from cargo's own `compiler-artifact` record for
# the `oxibonsai` binary (never a computed path), checks it starts, and
# exports it as `OXIBONSAI_CLI_BIN`, which every later stage (ci.sh's nextest
# run, stages 1b-1d) inherits and `oxibonsai_testkit::cli_bin::
# resolve_cli_binary` honours. The build FAILS CLOSED: if it fails, reports
# no executable, or the executable is missing or will not run `--version`,
# the gate stops here. An `OXIBONSAI_CLI_BIN` already in the environment is
# deliberately overridden: the gate must test the binary of THIS tree.
#
# ── THE REAL-MODEL PARITY STAGE MUST BE SERIALIZED ──────────────────────────
# Stage 1b's real legacy-model gates (the six cross-tier parity tests, three
# in oxibonsai-model and three in oxibonsai-runtime, plus the M-18 / M-02 /
# M-08 measurement gates, the `oxibonsai eval` CLI gate, the ONNX-export round
# trip and the engine-level speculative gates, all in the two
# `legacy_parity_tests` binaries; the Metal hidden-state parity gate
# `metal_hidden_parity_tests`, run once per legacy model file; the speculative
# gate `speculative_ternary_metal_gates` on the two ternary models; the
# embedding leg `embeddings_model_backed`; and the M-18 fused-prefill
# chunk-size sweep, a unit test of `oxibonsai-model` run on Bonsai-8B and on
# the Ternary-Bonsai-1.7B), stage 1c's nine real Bonsai 2 27B
# gates (`hybrid_forward_parity_tests`'s three `hybrid_real_27b_*_bonsai2`
# cases, `bonsai2_engine_tests`'s two
# `bonsai2_*_engine_greedy_matches_the_fork_goldens` cases, and
# `bonsai2_runtime_tests`'s four `real_27b_*_bonsai2` G6/G7/G9 cases) plus, on
# Darwin, `hybrid_metal_gates`'s two Metal 27B cases,
# `bonsai2_metal_engine_tests`'s four `real_27b_metal_engine_*` cases and
# `hybrid_metal_prefill_gates`'s two batched-prefill cases, then its vision
# leg (`bonsai2_vision_tests`, `bonsai2_vision_runtime_tests` and
# `vision_mmproj_tests`) and that leg's Metal half (`vision_metal_tests`,
# `bonsai2_vision_metal_tests`, `bonsai2_vision_metal_runtime_tests` and the
# CLI's `bonsai2_vision_cli_tests`), and stage 1d's real-model lib/bin cases each load a
# multi-GB GGUF and decode through it. Run concurrently on an 8-core/24 GB M3
# they have driven the machine's load average past 90. Stages 1b-1d therefore
# invoke every real-model test binary directly with `--test-threads=1`, one
# binary at a time, and the heavier tests additionally carry an in-binary mutex
# (`real_model_serial`/`gpu_serial`/`REAL_MODEL_LOCK`) because the Metal
# decode path is a process-global singleton. `.config/nextest.toml`
# separately excludes the real-27B and legacy real-model cases from a plain
# `cargo nextest run` via `default-filter`, and pins the ones it does list
# through nextest to a single-threaded `real-model-gate` test group — a
# nextest invocation is never how this gate itself collects real-model
# evidence; stages 1b-1d's direct `cargo test --release` calls are.
#
# A capability with zero "executed": true records for this run — including
# because the manifest has zero records for it at all, e.g. because no
# producer test has adopted this contract yet — is INCOMPLETE, and this
# script fails the release gate. It is not this script's job to weaken
# that; it is the producer tests' job to write to the manifest.
#
# Usage:
#   ./scripts/release-gate.sh                        # metal + legacy-models +
#                                                     # bonsai2-models required
#                                                     # iff macOS
#   ./scripts/release-gate.sh --require-cuda         # also require cuda evidence
#   ./scripts/release-gate.sh --skip-legacy-models   # release WITHOUT real legacy-
#                                                     # model parity evidence (state
#                                                     # it in the release notes)
#   ./scripts/release-gate.sh --skip-bonsai2-models  # release WITHOUT real Bonsai 2
#                                                     # 27B evidence (state it in the
#                                                     # release notes)
#   ./scripts/release-gate.sh --skip-bonsai2-vision  # release WITHOUT real Bonsai 2
#                                                     # vision-projector evidence
#                                                     # (state it in the release notes)
#   ./scripts/release-gate.sh --accept-approximate-cuda-syntax
#                                                     # no CUDA toolkit on this host: accept an approximate
#                                                     # CUDA syntax check, loudly (state it in the release notes)
#   OXIBONSAI_M08_RUN_LONG=1 ./scripts/release-gate.sh
#                                                     # MANDATORY SEPARATE RELEASE STEP:
#                                                     # also run and require M-08's
#                                                     # 20000-token YaRN gate (below)
#   ./scripts/release-gate.sh --self-test            # run this script's own offline
#                                                     # scenarios (stub cargo, no
#                                                     # build, no hardware) and exit
#
# ── M-08: the 20000-token YaRN gate is a separate, mandatory release step ──
# `oxibonsai-runtime::legacy_parity_tests::
# m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens` proves that
# `Bonsai-8B.gguf`'s declared YaRN scaling (factor 4, original context 16384)
# moves the real model's logits beyond float noise after a 20000-token
# natural-text prompt, against a copy of the same file patched to factor 1.0.
# It takes two real decodes of roughly an hour each, so a default run of this
# script leaves it out (it self-skips with `executed: false`). Every release
# still needs one passing run of it on the release commit: run this script
# once with `OXIBONSAI_M08_RUN_LONG=1` exported. Stage 1b then runs the gate
# (serialized, release profile — `scripts/ci.sh` keeps it out of its own
# parallel nextest stages) and the capability check requires its
# `executed: true` record by name (`legacy_models_require_tests_arg`); a
# self-skip, a timeout or a failure is a failed release gate.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

# shellcheck disable=SC2034  # several constants below are read only by release-gate-selftest.sh (sourced for --self-test), which shellcheck does not follow
{
# Release file names of the two Bonsai 2 27B bands, shared by the real
# Bonsai 2 27B stage below and its `--self-test` stub scenarios.
BONSAI2_PQ2_FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
BONSAI2_PTQ1_FILE="Ternary-Bonsai-2-27B-PTQ1_0.gguf"

# The three legacy dense models the CPU cross-tier gates and the Metal
# hidden-state gate all run on, in the order the metal-hidden stage runs them.
LEGACY_MODEL_FILES=(
    "Ternary-Bonsai-1.7B.gguf"
    "Ternary-Bonsai-8B.gguf"
    "Bonsai-8B.gguf"
)

# The capability-record name of the Metal hidden-state parity gate
# (`TEST_NAME` in `crates/oxibonsai-model/tests/metal_hidden_parity_tests.rs`),
# shared by the required-tests list and the `--self-test` scenarios.
METAL_HIDDEN_TEST_NAME="oxibonsai-model::metal_hidden_parity_tests::metal_hidden_prefill_matches_the_sequential_reference_on_real_models"

# The capability-record names of the stage-1b real-model gates that are not
# part of the cross-tier matrix. Each is the exact `TEST_NAME` the producing
# test writes (the `--self-test` scenarios check that, so a rename cannot
# silently turn a requirement into one nothing can satisfy).
#   - the speculative gate: one test per ternary legacy model;
#   - the embedding leg: the opt-in benchmark and the semantic smoke test;
#   - the Metal-vs-CPU greedy fallback gate of stage 1d.
SPECULATIVE_TEST_FILE="crates/oxibonsai-model/tests/speculative_ternary_metal_gates.rs"
SPECULATIVE_TEST_NAMES=(
    "oxibonsai-model::speculative_ternary_metal_gates::ternary_1_7b_speculative_greedy_equals_plain_greedy_on_metal"
    "oxibonsai-model::speculative_ternary_metal_gates::ternary_8b_speculative_greedy_equals_plain_greedy_on_metal"
)
SPECULATIVE_ENGINE_TEST_FILE="crates/oxibonsai-runtime/tests/legacy_parity_tests/speculative.rs"
SPECULATIVE_ENGINE_TEST_NAMES=(
    "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_ngram_speculation_equals_plain_greedy_on_metal"
    "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_adaptive_two_engine_speculation_equals_plain_greedy_on_metal"
)
EMBED_TEST_FILE="crates/oxibonsai-runtime/tests/embeddings_model_backed.rs"
EMBED_BENCH_TEST_NAME="oxibonsai-runtime::embeddings_model_backed::embed_bench_short_and_long"
EMBED_SMOKE_TEST_NAME="oxibonsai-runtime::embeddings_model_backed::real_model_places_queen_nearer_to_king_than_banana"
FALLBACK_TEST_FILE="crates/oxibonsai-runtime/tests/metal_greedy_cpu_fallback_tests.rs"
FALLBACK_TEST_NAME="oxibonsai-runtime::metal_greedy_cpu_fallback_tests::real_model_greedy_gpu_fallback_byte_identical"

# The M-18 fused-prefill chunk-size sweep: a unit test of `oxibonsai-model`'s
# `chunked_prefill` module (reached with `--lib`), run once per model in
# `SWEEP_MODEL_FILES` order — the 1-bit Bonsai-8B first (the model whose
# per-token prefill cost grew super-linearly before the tiled Q1 GEMM), then
# the Ternary-Bonsai-1.7B. `SWEEP_TEST_FILTER` is the libtest name filter that
# selects it; `SWEEP_TEST_NAME` is the capability-record name it writes.
SWEEP_TEST_FILE="crates/oxibonsai-model/src/chunked_prefill.rs"
SWEEP_TEST_FILTER="real_model_metal_prefill_chunk_size_sweep"
SWEEP_TEST_NAME="oxibonsai-model::lib::real_model_metal_prefill_chunk_size_sweep"
SWEEP_MODEL_FILES=(
    "Bonsai-8B.gguf"
    "Ternary-Bonsai-1.7B.gguf"
)

# The Bonsai 2 vision projector and the three real-weight vision gates (stage
# 1c's vision leg): the two record "bonsai2-vision", the projector gate
# "bonsai2-mmproj".
BONSAI2_MMPROJ_FILE="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
BONSAI2_VISION_TEST_FILES=(
    "crates/oxibonsai-model/tests/bonsai2_vision_tests.rs"
    "crates/oxibonsai-runtime/tests/bonsai2_vision_runtime_tests.rs"
)
BONSAI2_VISION_TEST_NAMES=(
    "oxibonsai-model::bonsai2_vision_tests::bonsai2_vision_real_27b_matches_the_fork_goldens_on_the_cpu_path"
    "oxibonsai-runtime::bonsai2_vision_runtime_tests::real_27b_server_round_trip_reproduces_the_golden_prompt_one"
)
BONSAI2_MMPROJ_TEST_FILE="crates/oxibonsai-model/tests/vision_mmproj_tests.rs"
BONSAI2_MMPROJ_TEST_NAME="oxibonsai-model::vision_mmproj_tests::real_mmproj_binds_every_block_and_matches_the_f64_reference"

# The Metal half of the vision leg (on Darwin unless --skip-bonsai2-metal):
# the projector's Metal tower records "bonsai2-mmproj"; the 27B image prompts
# on the Metal runner, the image chat round trip on a Metal engine and the
# shipped CLI end to end record "bonsai2-vision-metal". Each array entry pairs
# with the file at the same index.
BONSAI2_MMPROJ_METAL_TEST_FILE="crates/oxibonsai-model/tests/vision_metal_tests.rs"
BONSAI2_MMPROJ_METAL_TEST_NAME="oxibonsai-model::vision_metal_tests::real_mmproj_metal_tower_tracks_the_cpu_tower_and_the_f64_reference"
BONSAI2_VISION_METAL_TEST_FILES=(
    "crates/oxibonsai-model/tests/bonsai2_vision_metal_tests.rs"
    "crates/oxibonsai-runtime/tests/bonsai2_vision_metal_runtime_tests.rs"
    "tests/bonsai2_vision_cli_tests.rs"
)
BONSAI2_VISION_METAL_TEST_NAMES=(
    "oxibonsai-model::bonsai2_vision_metal_tests::hybrid_real_27b_metal_vision_matches_the_cpu_path_and_the_fork_goldens_bonsai2"
    "oxibonsai-runtime::bonsai2_vision_metal_runtime_tests::real_27b_metal_engine_serves_the_golden_image_prompt_over_http_bonsai2"
    "oxibonsai-cli::bonsai2_vision_cli_tests::real_27b_cli_image_turns_stay_on_metal_and_answer_the_golden_bonsai2"
)

# The Metal runner's batched (GEMM) prefill gates (stage 1c's last leg, on
# Darwin unless --skip-bonsai2-metal), both of which record "bonsai2-models".
BONSAI2_PREFILL_TEST_FILE="crates/oxibonsai-model/tests/hybrid_metal_prefill_gates.rs"
BONSAI2_PREFILL_TEST_NAMES=(
    "oxibonsai-model::hybrid_metal_prefill_gates::hybrid_real_27b_metal_prefill_throughput_bonsai2"
    "oxibonsai-model::hybrid_metal_prefill_gates::hybrid_real_27b_metal_batched_prefill_matches_cpu_tokens_bonsai2"
)

# The exact arguments of the stage-0 CLI build; the `--self-test` stand-in
# `cargo` checks that it receives exactly these.
RELEASE_CLI_BUILD_ARGS=(build --release -p oxibonsai-cli --bin oxibonsai --all-features --message-format=json)
}

# Parses the JSONL capability manifest at `report_path` and verifies that
# every capability named in the remaining arguments has at least one
# `"executed": true` record written no earlier than `run_start_epoch`.
# Exit 0 = every required capability has real, fresh evidence; exit 1 = at
# least one does not, OR the manifest itself is unusable (missing, stale,
# or contains a malformed line). Factored into its own function (rather
# than an inline heredoc at the call site) so `--self-test` can
# exercise this exact parsing logic against synthetic manifests, with no
# cargo run and no hardware involved.
#
# Each remaining argument is either a bare capability name (the check
# above: at least one `executed: true` record, from ANY test, is enough),
# or `--require-tests=<capability>:<name1>,<name2>,...` (repeatable, one
# per capability that needs it), which ADDITIONALLY requires an
# `executed: true` record for EACH named test individually — a capability
# several tests can satisfy on their own (e.g. any one of
# `legacy_parity_tests`'s cases) is not the same guarantee as "these five
# SPECIFIC gates all ran"; `--require-tests` is how this script asks for
# the stronger one without weakening the plain check for capabilities that
# do not need it. A required test whose executed record carries no
# `duration_ms` is reported with a `WARN` line and does not change the
# verdict.
check_capability_manifest() {
    local report_path="$1" run_start_epoch="$2"
    shift 2
    # python3 does the JSONL parse + freshness/executed check; bash only
    # routes its verdict. Malformed lines are reported AND fail the gate
    # (see `malformed` below): a manifest a human/tool can partially write
    # garbage into must not look like a clean pass just because the
    # well-formed lines happened to say "executed": true.
    python3 - "$report_path" "$run_start_epoch" "$@" <<'PYEOF'
import json, sys, os

report_path = sys.argv[1]
run_start_epoch = int(sys.argv[2])
args = sys.argv[3:]

required = []          # capability names, in first-seen order
required_tests = {}    # capability -> [test name, ...]
for arg in args:
    if arg.startswith("--require-tests="):
        rest = arg[len("--require-tests="):]
        cap, sep, names = rest.partition(":")
        if not sep:
            print(f"FAIL: malformed --require-tests argument (expected cap:name1,name2): {arg!r}")
            sys.exit(2)
        names_list = [n for n in names.split(",") if n]
        required_tests.setdefault(cap, []).extend(names_list)
        if cap not in required:
            required.append(cap)
    else:
        if arg not in required:
            required.append(arg)

if not os.path.isfile(report_path):
    print(f"FAIL: capability manifest does not exist: {report_path}")
    print("No producer test wrote to it during this run.")
    sys.exit(1)

mtime = os.path.getmtime(report_path)
if mtime < run_start_epoch - 1:  # 1s grace for filesystem timestamp granularity
    print(f"FAIL: capability manifest predates this run "
          f"(mtime={mtime:.0f}, run_start={run_start_epoch}) — stale evidence.")
    sys.exit(1)

executed_by_cap = {}
malformed = 0
with open(report_path, "r", encoding="utf-8", errors="replace") as f:
    for lineno, raw in enumerate(f, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"WARN: manifest line {lineno} is not valid JSON ({e}); ignoring: {line!r}")
            malformed += 1
            continue
        cap = rec.get("capability")
        executed = rec.get("executed")
        test = rec.get("test", "<unnamed test>")
        duration_ms = rec.get("duration_ms")
        if not isinstance(cap, str) or not isinstance(executed, bool):
            print(f"WARN: manifest line {lineno} missing required fields "
                  f"'capability'(str)/'executed'(bool); ignoring: {line!r}")
            malformed += 1
            continue
        if duration_ms is not None and not isinstance(duration_ms, (int, float)):
            duration_ms = None
        executed_by_cap.setdefault(cap, []).append((executed, test, duration_ms))

ok = True
for cap in required:
    entries = executed_by_cap.get(cap, [])
    ran = [(t, d) for (executed, t, d) in entries if executed]
    if ran:
        total_ms = sum(d for (_, d) in ran if d is not None)
        timed_count = sum(1 for (_, d) in ran if d is not None)
        if timed_count:
            print(f"OK: '{cap}' executed by {len(ran)} test(s), e.g. {ran[0][0]} "
                  f"({timed_count} with duration_ms, totalling {total_ms:.0f}ms)")
        else:
            print(f"OK: '{cap}' executed by {len(ran)} test(s), e.g. {ran[0][0]}")
    else:
        ok = False
        if entries:
            print(f"FAIL: '{cap}' has {len(entries)} manifest record(s) but NONE with "
                  f"executed=true (all self-skipped this run).")
        else:
            print(f"FAIL: '{cap}' has NO manifest records at all — either no producer "
                  f"test has adopted the capability-manifest contract yet, or none ran.")

    names_needed = required_tests.get(cap)
    if names_needed:
        have = {t for (t, _d) in ran}
        missing_names = [n for n in names_needed if n not in have]
        if missing_names:
            ok = False
            print(f"FAIL: '{cap}' is missing executed=true evidence for {len(missing_names)} "
                  f"required test(s):")
            for name in missing_names:
                print(f"  - {name}")
        else:
            print(f"OK: '{cap}' has executed=true for all {len(names_needed)} required test(s).")
        # An executed record without `duration_ms` is reported, never failed:
        # the evidence is real, only the report of what it cost is incomplete.
        untimed_names = [n for n in dict.fromkeys(names_needed)
                         if any(t == n and d is None for (t, d) in ran)]
        if untimed_names:
            print(f"WARN: '{cap}': {len(untimed_names)} required test(s) have an executed "
                  f"record without duration_ms (not a failure; the gate's cost report is "
                  f"incomplete):")
            for name in untimed_names:
                print(f"  - {name}")

if malformed:
    print(f"FAIL: {malformed} malformed manifest line(s) found — a partially-corrupted "
          f"manifest cannot be trusted as evidence for the capabilities that DID parse, "
          f"so this fails the gate even if every required capability otherwise looks "
          f"satisfied.")
    ok = False

sys.exit(0 if ok else 1)
PYEOF
}

# The `--require-tests=legacy-models:...` argument the capability check
# below passes for the "legacy-models" capability: every real-model gate
# stage 1b and stage 1d run against the legacy GGUFs, by name, so a release
# can never be cut when only some of the legacy models' files (or the ONNX
# export, or the tokenizer) are present — without naming them all, a host
# with only Ternary-Bonsai-1.7B.gguf present still satisfies the plain
# "legacy-models" capability check (that model's own gates run and record
# `executed: true`) while every other gate silently self-skips. The list:
# the six core cross-tier greedy gates (three models x {numeric parity in
# oxibonsai-model, golden text in oxibonsai-runtime}), the three stage-1d
# cases, the M-18 / M-02 / M-08-control real-model measurement gates, the
# `oxibonsai eval --task mmlu` CLI gate and the ONNX-export round trip (ONNX
# export -> GGUF -> real tokenizer), the speculative gates (one model-level test
# per ternary model, plus the engine-level n-gram and two-engine adaptive tests
# in oxibonsai-runtime's `legacy_parity_tests`), the embedding leg's two tests
# and the M-18 fused-prefill chunk-size sweep (`oxibonsai-model`'s
# `chunked_prefill` unit test, run once on Bonsai-8B and once on the
# Ternary-Bonsai-1.7B by stage 1b; the numerics gate named
# `prefill_chunk_size_is_a_dispatch_knob_not_a_numerics_knob_on_a_real_long_prompt`
# is a different test). With `m08_long` = 1
# (`OXIBONSAI_M08_RUN_LONG=1` in this
# script's environment) it also names M-08's 20000-token YaRN gate, which
# only runs when that variable is set — see the usage notes at the top of
# this file for why that run is a separate, mandatory release step.
#   legacy_models_require_tests_arg <m08_long:0|1>
legacy_models_require_tests_arg() {
    local m08_long="$1"
    local names="\
oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::lib::temperature_zero_completion_takes_the_metal_greedy_gpu_path,\
oxibonsai-runtime::lib::real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop,\
oxibonsai-runtime::metal_greedy_cpu_fallback_tests::real_model_greedy_gpu_fallback_byte_identical,\
oxibonsai-model::speculative_ternary_metal_gates::ternary_1_7b_speculative_greedy_equals_plain_greedy_on_metal,\
oxibonsai-model::speculative_ternary_metal_gates::ternary_8b_speculative_greedy_equals_plain_greedy_on_metal,\
oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_ngram_speculation_equals_plain_greedy_on_metal,\
oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_adaptive_two_engine_speculation_equals_plain_greedy_on_metal,\
oxibonsai-runtime::embeddings_model_backed::embed_bench_short_and_long,\
oxibonsai-runtime::embeddings_model_backed::real_model_places_queen_nearer_to_king_than_banana,\
oxibonsai-model::lib::real_model_metal_prefill_chunk_size_sweep,\
oxibonsai-model::legacy_parity_tests::prefill_chunk_size_is_a_dispatch_knob_not_a_numerics_knob_on_a_real_long_prompt,\
oxibonsai-model::legacy_parity_tests::footprint_and_measured_rss_confirm_the_embedding_table_stays_unmaterialized_bonsai_8b,\
oxibonsai-model::legacy_parity_tests::m08_yarn_scaling_wiring_takes_effect_on_the_real_bonsai_8b_file,\
oxibonsai-runtime::legacy_parity_tests::eval_cli_scores_a_real_mmlu_style_dataset_through_score_choices_logprob,\
oxibonsai-runtime::legacy_parity_tests::onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip"
    if [[ "$m08_long" -eq 1 ]]; then
        names="$names,oxibonsai-runtime::legacy_parity_tests::m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens"
    fi
    printf '%s' "--require-tests=legacy-models:$names"
}

# Runs the real Bonsai 2 27B gates, one after another, never in parallel, so
# this script never maps more than one multi-GB 27B GGUF at a time:
# oxibonsai-model's `hybrid_forward_parity_tests` (three gates: PQ2_0,
# PTQ1_0, the f64-layer reference), oxibonsai-runtime's `bonsai2_engine_tests`
# (two gates), then — on Darwin, unless `skip_metal`, and only once the test
# file exists in this checkout — oxibonsai-model's `hybrid_metal_gates`
# (CPU-vs-Metal token parity + decode throughput), oxibonsai-runtime's
# `bonsai2_metal_engine_tests` (the same on the product path: the Metal
# engine vs the CPU engine and the fork, engine decode throughput and
# memory, and a `/v1/chat/completions` round trip) and oxibonsai-model's
# `hybrid_metal_prefill_gates` (the runner's batched prefill against its
# decode rate, and the token and logprob gates with every prefill on the
# GEMM — both bands).
#
# Fails CLOSED, before invoking cargo at all, when either release GGUF is
# missing from `models_dir` and `skip` is not set: a partial 27B evidence
# run is refused outright (naming what is missing and both ways out) rather
# than left to run for several minutes and self-skip, which the capability
# check at the end of this script would catch anyway but only after paying
# the wall-clock cost of everything that DID have its file.
#
# Every leg runs with `OXIBONSAI_KERNEL_TIER` removed from its environment
# (`env -u`): the CPU model's `PQ2_0` GEMV honours that INT8 tier selector
# (K-14), so a value a developer exported would move the CPU legs off the
# fork's ids and break CPU-vs-Metal identity. The release evidence is the
# default configuration's.
#
# All inputs are explicit arguments — never a global — so `--self-test`
# can drive this exact function against a scratch directory and a
# PATH-shimmed `cargo` that only logs its own invocations, with no real
# build, hardware or GGUF involved.
#   run_bonsai2_models_stage <skip:0|1> <skip_metal:0|1> <models_dir> \
#       <golden_dir> <metal_test_file> <is_darwin:0|1>
run_bonsai2_models_stage() {
    local skip="$1" skip_metal="$2" models_dir="$3" golden_dir="$4" \
        metal_test_file="$5" is_darwin="$6"

    if [[ "$skip" -eq 1 ]]; then
        echo "--skip-bonsai2-models was passed: this release carries NO real Bonsai 2"
        echo "27B parity evidence. Say so in the release notes."
        return 0
    fi
    if [[ "$is_darwin" -ne 1 ]]; then
        echo "Both gates target the Metal-capable CPU/GPU tiers on Apple Silicon;"
        echo "there is no non-macOS run of them on this machine. Nothing to run."
        return 0
    fi

    local pq2_path="$models_dir/$BONSAI2_PQ2_FILE" ptq1_path="$models_dir/$BONSAI2_PTQ1_FILE"
    local missing=()
    [[ -s "$pq2_path" ]] || missing+=("$BONSAI2_PQ2_FILE")
    [[ -s "$ptq1_path" ]] || missing+=("$BONSAI2_PTQ1_FILE")
    if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "MISSING fixtures: ${missing[*]}"
        echo "Refusing to run a partial Bonsai 2 27B gate: put the missing file(s) under"
        echo "$models_dir (or point OXIBONSAI_MODELS_DIR at them), or pass"
        echo "--skip-bonsai2-models to release deliberately without this evidence."
        return 1
    fi

    echo "── oxibonsai-model::hybrid_forward_parity_tests (real 27B gates) ──────"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        cargo test --release -p oxibonsai-model --all-features \
        --test hybrid_forward_parity_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-runtime::bonsai2_engine_tests (real 27B gates) ───────────"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_engine_tests -- --test-threads=1 --nocapture || return $?

    # G6/G7/G9 all live in one binary: G6 (tokenization) and G7 (chat
    # template) are header-only (`mmap` + metadata parse, never a tensor
    # byte) and self-locate under `models_dir` on their own; G9 (`oxibonsai
    # info --json`, one case per band) needs its own per-band env var (see
    # `locate_named_27b_gguf`'s doc comment: it runs the CLI as a fresh
    # process per case, the binary stage 0 exported as OXIBONSAI_CLI_BIN, so
    # it deliberately has no `models/` fallback).
    # One invocation with every var set covers all of it — this also avoids
    # a G9 case ever reaching its `OXI_REQUIRE_MODEL_FILES=1` hard-failure
    # branch just because a *different* leg forgot to export its env var.
    echo "── oxibonsai-runtime::bonsai2_runtime_tests (G6/G7/G9) ─────────────────"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_runtime_tests -- --test-threads=1 --nocapture || return $?

    if [[ "$skip_metal" -eq 1 ]]; then
        echo "--skip-bonsai2-metal was passed: this release carries NO Metal 27B"
        echo "CPU-vs-Metal parity / decode-throughput evidence. Say so in the release notes."
        return 0
    fi
    if [[ ! -f "$metal_test_file" ]]; then
        echo "NOTE: $metal_test_file is not present in this checkout — the Metal 27B leg"
        echo "(CPU-vs-Metal token parity, decode throughput) cannot run yet. The capability"
        echo "check below still requires its two gates by name unless --skip-bonsai2-metal"
        echo "is passed, so this is visible as a failure there, not a silent pass here."
        return 0
    fi
    echo "── oxibonsai-model::hybrid_metal_gates (real 27B Metal gates) ─────────"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        cargo test --release -p oxibonsai-model --features metal \
        --test hybrid_metal_gates -- --test-threads=1 --nocapture || return $?
    # The same runner behind the product path: `InferenceEngine` on the Metal
    # hybrid runner (what `run`/`chat`/`serve --backend auto` pick on this
    # host) against the CPU engine and the fork, its decode throughput and
    # memory, and one `/v1/chat/completions` round trip — both bands.
    echo "── oxibonsai-runtime::bonsai2_metal_engine_tests (real 27B Metal engine gates) ─"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_metal_engine_tests -- --test-threads=1 --nocapture || return $?
    # The runner's batched (GEMM) prefill: a 512-token prefill against the
    # same process's decode rate, and the token and logprob gates with every
    # prefill on the GEMM — both bands. The gates find the files through
    # their two variables only (no models-directory fallback), and
    # `OXI_REQUIRE_MODEL_FILES=1` turns a band that is not found into a
    # failure rather than a self-skip.
    echo "── oxibonsai-model::hybrid_metal_prefill_gates (real 27B batched prefill gates) ─"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-model --all-features \
        --test hybrid_metal_prefill_gates -- --test-threads=1 --nocapture || return $?
    return 0
}

# Builds the `oxibonsai` CLI once with the full feature set the release ships
# and prints the path of the executable cargo reports for it (stage 0; see the
# header). Everything else goes to stderr, so a caller can capture the path
# with `$(...)` and still see cargo's own diagnostics.
#
# Fails CLOSED (non-zero, nothing on stdout) when cargo cannot be started or
# exits non-zero, when its JSON stream carries no `compiler-artifact` with an
# executable for the `oxibonsai` binary target (the `oxibonsai` library crate
# has a record of the same name with `executable: null`, which is skipped),
# when that file does not exist, or when it does not answer `--version` with
# exit 0. The environment's `OXIBONSAI_CLI_BIN` is never read here: the gate
# tests the binary of this tree. The path is cargo's own record, never a
# computed `<target dir>/release/oxibonsai`, so a custom `CARGO_TARGET_DIR`
# is honoured by construction.
#
# The `cargo` it runs is whatever `cargo` resolves to on `PATH`, so
# `--self-test` can put a stand-in in front and exercise this exact function.
build_release_cli_binary() {
    command -v python3 >/dev/null 2>&1 || {
        echo "FATAL: python3 not found — cannot read cargo's JSON build stream." >&2
        return 2
    }
    local stream cargo_rc=0 cli_path
    stream="$(mktemp "${TMPDIR:-/tmp}/oxibonsai_release_gate_cli_stream.XXXXXX")" || {
        echo "FAIL: could not create a scratch file for cargo's JSON stream" >&2
        return 1
    }
    cargo "${RELEASE_CLI_BUILD_ARGS[@]}" >"$stream" || cargo_rc=$?
    if [[ "$cargo_rc" -ne 0 ]]; then
        echo "FAIL: 'cargo ${RELEASE_CLI_BUILD_ARGS[*]}' exited $cargo_rc" >&2
        rm -f "$stream"
        return "$cargo_rc"
    fi
    cli_path="$(python3 - "$stream" <<'PYEOF'
import json, sys

path = None
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        reason = rec.get("reason")
        if reason == "build-finished" and rec.get("success") is False:
            sys.exit(1)
        if reason != "compiler-artifact":
            continue
        target = rec.get("target") or {}
        if target.get("name") != "oxibonsai":
            continue
        kinds = target.get("kind")
        if isinstance(kinds, list) and "bin" not in kinds:
            continue
        exe = rec.get("executable")
        if isinstance(exe, str) and exe:
            path = exe
if path is None:
    sys.exit(1)
print(path)
PYEOF
    )" || {
        echo "FAIL: cargo's JSON stream names no executable for the 'oxibonsai' binary of oxibonsai-cli" >&2
        rm -f "$stream"
        return 1
    }
    rm -f "$stream"
    if [[ ! -f "$cli_path" ]]; then
        echo "FAIL: cargo reported the CLI executable at $cli_path, but no file exists there" >&2
        return 1
    fi
    if ! "$cli_path" --version >/dev/null 2>&1; then
        echo "FAIL: the built CLI $cli_path does not answer --version with exit 0" >&2
        return 1
    fi
    printf '%s\n' "$cli_path"
}

# Runs `oxibonsai-model`'s `metal_hidden_parity_tests` — the head-free Metal
# hidden-state prefill behind dense embeddings, checked against the per-token
# reference on a real model at 10, 300 and 2000 tokens — once per legacy model
# file, in `LEGACY_MODEL_FILES` order, one real-model process at a time, in
# `--release` with `--nocapture`. `OXI_MODEL` names exactly one GGUF per run,
# so a run can never quietly cover fewer models than the release names. The
# test is not `#[ignore]`d and records `metal-hidden`; the capability check
# requires `METAL_HIDDEN_TEST_NAME` to have an `executed: true` record.
#
# Fails CLOSED before invoking cargo at all when any of the model files is
# missing from `models_dir` — a partial run would otherwise pass on the
# strength of the models that WERE present — and stops at the first leg whose
# cargo invocation fails.
#   run_metal_hidden_stage <models_dir>
run_metal_hidden_stage() {
    local models_dir="$1" file
    local missing=()
    for file in "${LEGACY_MODEL_FILES[@]}"; do
        [[ -s "$models_dir/$file" ]] || missing+=("$file")
    done
    if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "MISSING fixtures: ${missing[*]}"
        echo "Refusing to run a partial metal-hidden gate: put the missing file(s) under"
        echo "$models_dir (or point OXIBONSAI_MODELS_DIR at them), or pass"
        echo "--skip-legacy-models to release deliberately without this evidence."
        return 1
    fi
    for file in "${LEGACY_MODEL_FILES[@]}"; do
        echo "── oxibonsai-model::metal_hidden_parity_tests ($file) ──"
        env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
            "OXI_MODEL=$models_dir/$file" \
            cargo test --release -p oxibonsai-model --features metal \
            --test metal_hidden_parity_tests -- --test-threads=1 --nocapture || return $?
    done
    return 0
}

# Fails (returns 1, printing what is missing and both ways out) unless every
# named file exists under `models_dir`. The shared head of the real-model stage
# functions below: a partial run would otherwise start cargo, spend minutes
# and self-skip, which the capability check at the end would only catch after
# paying for everything that DID have its file.
#   require_model_files <stage label> <skip flag name> <models_dir> <file>...
require_model_files() {
    local label="$1" skip_flag="$2" models_dir="$3"
    shift 3
    local file missing=()
    for file in "$@"; do
        [[ -s "$models_dir/$file" ]] || missing+=("$file")
    done
    if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "MISSING fixtures: ${missing[*]}"
        echo "Refusing to run a partial $label gate: put the missing file(s) under"
        echo "$models_dir (or point OXIBONSAI_MODELS_DIR at them), or pass"
        echo "$skip_flag to release deliberately without this evidence."
        return 1
    fi
    return 0
}

# Runs `oxibonsai-model`'s `speculative_ternary_metal_gates` — speculative
# greedy decoding against plain greedy decoding, token for token, on the real
# Ternary-Bonsai-1.7B and Ternary-Bonsai-8B on the Metal route (draft lengths
# below and above the 8-row tiled-GEMM boundary and the adaptive lookahead's
# whole range) — in `--release`, `--nocapture`, one real-model process. The
# test finds its two files through `OXIBONSAI_MODELS_DIR`, is not `#[ignore]`d
# and records `legacy-models` per model; the capability check requires both
# names. Fails CLOSED before invoking cargo when either ternary file is
# missing.
#   run_speculative_stage <models_dir>
run_speculative_stage() {
    local models_dir="$1"
    require_model_files "speculative-decoding" "--skip-legacy-models" "$models_dir" \
        "${LEGACY_MODEL_FILES[0]}" "${LEGACY_MODEL_FILES[1]}" || return 1
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        cargo test --release -p oxibonsai-model --features metal \
        --test speculative_ternary_metal_gates -- --test-threads=1 --nocapture
}

# Runs `oxibonsai-runtime`'s `embeddings_model_backed` against the real
# Ternary-Bonsai-1.7B: the semantic smoke test and the embedding benchmark
# (`embed_bench_short_and_long`: the production Metal hidden prefill against the
# batched CPU pass at 10 / 200 / 2000 tokens and against the per-token loop at
# 10 / 200 tokens, 20 s target at 2000). The benchmark runs only on an explicit
# opt-in, so this passes `OXIBONSAI_EMBED_BENCH=1` AND `OXI_MODEL` itself — the
# leg can never self-skip into a green — and the capability check requires both
# tests by name.
#
# The per-token reference loop at 2000 tokens (about five to six minutes on an
# M3 at a load average of about 10, longer under a heavier load) has its own
# opt-in, `OXIBONSAI_EMBED_BENCH_PER_TOKEN=1`,
# for manual runs. This leg never sets it and removes an inherited one
# (`env -u`): at 2000 tokens the gate compares the production Metal embedding
# with the batched CPU pass (pooled cosine >= 0.9999) and asserts the 20 s
# Metal target, so the leg's cost and verdict do not depend on a developer's
# exported variable. Fails CLOSED before invoking cargo when the model or the
# tokenizer is missing.
#   run_embedding_stage <models_dir>
run_embedding_stage() {
    local models_dir="$1"
    require_model_files "embedding" "--skip-legacy-models" "$models_dir" \
        "${LEGACY_MODEL_FILES[0]}" "tokenizer.json" || return 1
    env -u OXIBONSAI_KERNEL_TIER -u OXIBONSAI_EMBED_BENCH_PER_TOKEN \
        "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_MODEL=$models_dir/${LEGACY_MODEL_FILES[0]}" OXIBONSAI_EMBED_BENCH=1 \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test embeddings_model_backed -- --test-threads=1 --nocapture
}

# Runs the M-18 acceptance gate: `oxibonsai-model`'s fused Metal prefill
# chunk-size sweep (`real_model_metal_prefill_chunk_size_sweep`, a unit test of
# the `chunked_prefill` module, hence `--lib`) on each model in
# `SWEEP_MODEL_FILES` — Bonsai-8B, then the Ternary-Bonsai-1.7B — one real-model
# process at a time, in `--release` with `--nocapture`. `OXI_MODEL` names
# exactly one GGUF per run (without it the test self-skips), so a run can never
# quietly cover fewer models than the release names. For each model the test
# prefills a 4096-token prompt at six chunk sizes with the fused route pinned,
# decodes it once token by token, and asserts the fused path actually ran, the
# 256-token speed-up over sequential decode, that the whole prompt prefills
# faster fused than decoded, and the bound on the fused per-token cost's growth
# from the short prompt to the full one. It records `legacy-models` under
# `SWEEP_TEST_NAME`, which the capability check requires by name.
#
# `OXIBONSAI_KERNEL_TIER` and `OXIBONSAI_PREFILL_SWEEP_TOKENS` are removed from
# each leg's environment (`env -u`): the sweep measures the default Metal
# route at its default prompt length, and a value a developer exported would
# silently move the release evidence off it.
#
# Fails CLOSED before invoking cargo at all when either model file is missing
# from `models_dir`, and stops at the first leg whose cargo invocation fails.
#   run_prefill_sweep_stage <models_dir>
run_prefill_sweep_stage() {
    local models_dir="$1" file
    require_model_files "M-18 prefill sweep" "--skip-legacy-models" "$models_dir" \
        "${SWEEP_MODEL_FILES[@]}" || return 1
    for file in "${SWEEP_MODEL_FILES[@]}"; do
        echo "── oxibonsai-model::lib::$SWEEP_TEST_FILTER ($file) ──"
        env -u OXIBONSAI_KERNEL_TIER -u OXIBONSAI_PREFILL_SWEEP_TOKENS \
            "OXI_MODEL=$models_dir/$file" \
            cargo test --release -p oxibonsai-model --features metal --lib \
            -- --test-threads=1 --nocapture "$SWEEP_TEST_FILTER" || return $?
    done
    return 0
}

# Runs the three real-model lib/bin cases of stage 1d that decode the legacy
# Ternary-Bonsai-1.7B through the Metal engine, one after another:
# `temperature_zero_completion_takes_the_metal_greedy_gpu_path` and
# `real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop`
# (both `oxibonsai-runtime --lib`, env-gated on `OXI_MODEL` + `OXI_TOKENIZER`)
# and `metal_greedy_cpu_fallback_tests`'s
# `real_model_greedy_gpu_fallback_byte_identical` (Metal against pure-CPU greedy
# plus the forced mid-stream CPU fallback). Every one self-skips with a record
# when `OXI_MODEL` is unset, so each leg is handed it explicitly: a leg that
# ran without it would exit 0 having validated nothing. Fails CLOSED before
# invoking cargo when the model or the tokenizer is missing.
#   run_legacy_lib_bin_stage <models_dir>
run_legacy_lib_bin_stage() {
    local models_dir="$1"
    require_model_files "legacy lib/bin" "--skip-legacy-models" "$models_dir" \
        "${LEGACY_MODEL_FILES[0]}" "tokenizer.json" || return 1
    local model="$models_dir/${LEGACY_MODEL_FILES[0]}" tokenizer="$models_dir/tokenizer.json"
    echo "── oxibonsai-runtime::lib (legacy-models: temperature-0 GPU path, stream stop) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_MODEL=$model" "OXI_TOKENIZER=$tokenizer" \
        cargo test --release -p oxibonsai-runtime --all-features --lib \
        -- --test-threads=1 \
        temperature_zero_completion_takes_the_metal_greedy_gpu_path \
        real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop \
        || return $?
    echo ""
    echo "── oxibonsai-runtime::metal_greedy_cpu_fallback_tests (legacy-models: Metal↔CPU byte parity + mid-stream fallback) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_MODEL=$model" \
        cargo test --release -p oxibonsai-runtime --features metal \
        --test metal_greedy_cpu_fallback_tests -- --test-threads=1 \
        real_model_greedy_gpu_fallback_byte_identical
}

# Whether this run requires — and therefore runs — the Bonsai 2 vision
# evidence: prints 1 on a Darwin host that has the real projector next to the
# 27B when neither --skip-bonsai2-models (no 27B evidence at all, so none of
# the vision legs that load it) nor --skip-bonsai2-vision was passed, else 0.
# The one predicate the stage and the capability requirement both read, so
# they can never disagree.
#   bonsai2_vision_required <skip_vision:0|1> <skip_models:0|1> <models_dir> <is_darwin:0|1>
bonsai2_vision_required() {
    local skip_vision="$1" skip_models="$2" models_dir="$3" is_darwin="$4"
    if [[ "$skip_vision" -eq 1 || "$skip_models" -eq 1 || "$is_darwin" -ne 1 ]]; then
        echo 0
    elif [[ -s "$models_dir/$BONSAI2_MMPROJ_FILE" ]]; then
        echo 1
    else
        echo 0
    fi
}

# Runs the real Bonsai 2 vision gates on the CPU path, in `--release`, one real
# 27B process at a time: oxibonsai-model's `bonsai2_vision_tests` (the 27B
# `PQ2_0` and its projector against the fork's image goldens), oxibonsai-runtime's
# `bonsai2_vision_runtime_tests` (its `real_27b_` leg: the same weights behind
# the chat endpoint) and oxibonsai-model's `vision_mmproj_tests` (the real
# projector bound block by block and checked against the f64 reference).
# `OXI_REQUIRE_MODEL_FILES=1` turns a missing file into a failure instead of a
# self-skip, and `env -u OXIBONSAI_KERNEL_TIER` keeps the CPU legs on the
# default configuration the release ships.
#
# Skipped (return 0, saying so) with --skip-bonsai2-vision, with
# --skip-bonsai2-models (the 27B evidence is not collected at all) and on a
# non-Darwin host. Fails CLOSED, before invoking cargo, when the 27B is
# present but the projector is not: a release that ships `--mmproj` must carry
# this evidence, and the refusal names both ways out. Also fails closed when
# the 27B itself is missing.
#   run_bonsai2_vision_stage <skip_vision:0|1> <skip_models:0|1> <models_dir> <is_darwin:0|1>
run_bonsai2_vision_stage() {
    local skip_vision="$1" skip_models="$2" models_dir="$3" is_darwin="$4"

    if [[ "$skip_vision" -eq 1 ]]; then
        echo "--skip-bonsai2-vision was passed: this release carries NO real Bonsai 2"
        echo "vision-projector evidence (no image answer, projector or round-trip"
        echo "gate ran). Say so in the release notes."
        return 0
    fi
    if [[ "$skip_models" -eq 1 ]]; then
        echo "--skip-bonsai2-models was passed: the Bonsai 2 vision gates load the 27B,"
        echo "so they are not run either; this release carries NO vision evidence."
        return 0
    fi
    if [[ "$is_darwin" -ne 1 ]]; then
        echo "The vision gates are gated to the Apple Silicon host this release is cut on;"
        echo "there is no non-macOS run of them on this machine. Nothing to run."
        return 0
    fi

    local pq2_path="$models_dir/$BONSAI2_PQ2_FILE" mmproj_path="$models_dir/$BONSAI2_MMPROJ_FILE"
    if [[ ! -s "$pq2_path" ]]; then
        echo "MISSING fixtures: $BONSAI2_PQ2_FILE"
        echo "Refusing to run a partial Bonsai 2 vision gate: it needs the 27B beside its"
        echo "projector. Put the file under $models_dir (or point OXIBONSAI_MODELS_DIR at"
        echo "it), or pass --skip-bonsai2-vision / --skip-bonsai2-models."
        return 1
    fi
    if [[ ! -s "$mmproj_path" ]]; then
        echo "MISSING fixtures: $BONSAI2_MMPROJ_FILE"
        echo "The 27B is present but its vision projector is not: refusing to release"
        echo "without the real image-path evidence. Put the file under $models_dir (or"
        echo "point OXIBONSAI_MODELS_DIR at it), or pass --skip-bonsai2-vision to release"
        echo "deliberately without it."
        return 1
    fi

    echo "── oxibonsai-model::bonsai2_vision_tests (real 27B + projector, CPU path) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" \
        OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-model --all-features \
        --test bonsai2_vision_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-runtime::bonsai2_vision_runtime_tests (real 27B chat round trip) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" \
        OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_vision_runtime_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-model::vision_mmproj_tests (real projector) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXIBONSAI_MODELS_DIR=$models_dir" \
        "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-model --all-features \
        --test vision_mmproj_tests -- --test-threads=1 --nocapture
}

# Prints 1 when this run requires (and so runs) the Metal half of the vision
# evidence — wherever `bonsai2_vision_required` holds and --skip-bonsai2-metal
# was not passed — else 0; the stage and the capability check both read it.
#   bonsai2_vision_metal_required <skip_vision> <skip_models> <skip_metal> <models_dir> <is_darwin>
bonsai2_vision_metal_required() {
    local skip_vision="$1" skip_models="$2" skip_metal="$3" models_dir="$4" is_darwin="$5"
    if [[ "$skip_metal" -eq 1 ]]; then
        echo 0
    else
        bonsai2_vision_required "$skip_vision" "$skip_models" "$models_dir" "$is_darwin"
    fi
}

# Runs the Metal half of the real Bonsai 2 vision gates, in `--release`, one
# real-model process at a time: oxibonsai-model's `vision_metal_tests` (the
# Metal tower on the real projector against the CPU tower and the f64
# reference) and `bonsai2_vision_metal_tests` (the 27B image prompts on the
# Metal runner and tower against the CPU path and the vendored image goldens,
# both bands), oxibonsai-runtime's `bonsai2_vision_metal_runtime_tests` (the
# image chat round trip on a Metal engine) and the CLI's
# `bonsai2_vision_cli_tests` (stage 0's `OXIBONSAI_CLI_BIN`: `run --mmproj
# --backend auto` stays on the Metal runner and prints the golden answer, as
# `--backend cpu` does, `serve` answers the image request, and a dense model
# is refused before any image is decoded), each with OXI_REQUIRE_MODEL_FILES=1.
# Skipped (saying so) under --skip-bonsai2-vision, --skip-bonsai2-models,
# --skip-bonsai2-metal and on a non-Darwin host; otherwise it fails CLOSED
# before invoking cargo when the projector or either 27B band is missing.
#   run_bonsai2_vision_metal_stage <skip_vision> <skip_models> <skip_metal> <models_dir> <is_darwin>
run_bonsai2_vision_metal_stage() {
    local skip_vision="$1" skip_models="$2" skip_metal="$3" models_dir="$4" is_darwin="$5"

    if [[ "$skip_vision" -eq 1 || "$skip_models" -eq 1 || "$is_darwin" -ne 1 ]]; then
        echo "The vision leg does not run here (see above), so neither does its Metal half."
        return 0
    fi
    if [[ "$skip_metal" -eq 1 ]]; then
        echo "--skip-bonsai2-metal was passed: this release carries NO Metal vision"
        echo "evidence (tower, runner rows prefill, Metal chat round trip, CLI on the"
        echo "Metal runner). Say so in the release notes."
        return 0
    fi

    local pq2_path="$models_dir/$BONSAI2_PQ2_FILE" ptq1_path="$models_dir/$BONSAI2_PTQ1_FILE" \
        mmproj_path="$models_dir/$BONSAI2_MMPROJ_FILE"
    require_model_files "Bonsai 2 Metal vision" "--skip-bonsai2-metal (or --skip-bonsai2-vision)" \
        "$models_dir" "$BONSAI2_PQ2_FILE" "$BONSAI2_PTQ1_FILE" "$BONSAI2_MMPROJ_FILE" || return 1

    echo "── oxibonsai-model::vision_metal_tests (real projector on the Metal tower) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" \
        OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-model --all-features \
        --test vision_metal_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-model::bonsai2_vision_metal_tests (real 27B vision on Metal, both bands) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" \
        OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-model --all-features \
        --test bonsai2_vision_metal_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-runtime::bonsai2_vision_metal_runtime_tests (image chat on Metal) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_vision_metal_runtime_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-cli::bonsai2_vision_cli_tests (the shipped CLI: run and serve with --mmproj) ──"
    env -u OXIBONSAI_KERNEL_TIER "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_MMPROJ_GGUF=$mmproj_path" OXI_REQUIRE_MODEL_FILES=1 \
        cargo test --release -p oxibonsai-cli --all-features \
        --test bonsai2_vision_cli_tests -- --test-threads=1 --nocapture
}

# ── --self-test ──────────────────────────────────────────────────────────
# The offline scenarios live in scripts/release-gate-selftest.sh, which defines
# `release_gate_self_test` (and the helpers only it uses). It is sourced only
# when `--self-test` is asked for (see the argument loop below), so a gate run
# never reads it.

REQUIRE_CUDA=0
SKIP_LEGACY_MODELS=0
SKIP_BONSAI2_MODELS=0
SKIP_BONSAI2_METAL=0
SKIP_BONSAI2_VISION=0
# CUDA_WAIVER_USED is 1 only when ci.sh reports having used the waiver (never
# from the flag alone); the verdict and the manifest record are driven by it.
ACCEPT_APPROX_CUDA=0
CUDA_WAIVER_USED=0
# M-08's 20000-token YaRN gate runs (and is then required by name) only when
# this is exactly "1" — see the usage notes at the top of this file.
M08_RUN_LONG=0
[[ "${OXIBONSAI_M08_RUN_LONG:-}" == "1" ]] && M08_RUN_LONG=1
for arg in "$@"; do
    case "$arg" in
        --require-cuda) REQUIRE_CUDA=1 ;;
        --accept-approximate-cuda-syntax) ACCEPT_APPROX_CUDA=1 ;;
        --skip-legacy-models) SKIP_LEGACY_MODELS=1 ;;
        --skip-bonsai2-models) SKIP_BONSAI2_MODELS=1 ;;
        --skip-bonsai2-metal) SKIP_BONSAI2_METAL=1 ;;
        --skip-bonsai2-vision) SKIP_BONSAI2_VISION=1 ;;
        --self-test)
            # A missing sibling file fails closed: no verdict is a failure, never a pass.
            # shellcheck source=release-gate-selftest.sh
            source "$SCRIPT_DIR/release-gate-selftest.sh" || exit 2
            release_gate_self_test
            exit $?
            ;;
        --help|-h)
            cat <<'HELP_EOF'
Usage: release-gate.sh [--require-cuda] [--accept-approximate-cuda-syntax]
                        [--skip-legacy-models] [--skip-bonsai2-models] [--skip-bonsai2-metal]
                        [--skip-bonsai2-vision] [--self-test]

  --require-cuda        Also require fresh "cuda" capability-manifest
                         evidence (see the capability-manifest contract at
                         the top of this file). Off by default: this
                         project treats CUDA as compile-blind on hosts with
                         no CUDA hardware, so a default run never demands
                         CUDA evidence.

  --accept-approximate-cuda-syntax
                         On a release host WITHOUT the CUDA toolkit (every macOS
                         host), accept an APPROXIMATE CUDA kernel-syntax check
                         (parsed as C++ with CUDA builtins stubbed; NOT an nvcc
                         compile) instead of failing the gate. Off by default.
                         Loud: the header, a waiver record in the manifest and
                         the verdict "PASSED with a waiver"; state it in the
                         release notes. Never waives a syntax error; REFUSED
                         with --require-cuda (exit 2).

  --skip-legacy-models   Opt out of the "legacy-models" capability, which
                         is otherwise REQUIRED on Darwin (design decision
                         D-6(4)). "legacy-models" is
                         the evidence that the real-model CPU/NEON/Metal
                         greedy parity gate (oxibonsai-model and
                         oxibonsai-runtime's legacy_parity_tests, driven
                         against the shipped GGUFs under ./models) actually
                         ran on this host, guarding README's "byte-identical
                         output at --temperature 0" claim. Passing this flag
                         releases WITHOUT that real-model evidence — the
                         gate skips stage 1b entirely (including the Metal
                         hidden-state gate, oxibonsai-model's
                         metal_hidden_parity_tests, the speculative gate,
                         oxibonsai-model's speculative_ternary_metal_gates,
                         the embedding leg, oxibonsai-runtime's
                         embeddings_model_backed, and the M-18 fused-prefill
                         chunk-size sweep, oxibonsai-model's
                         real_model_metal_prefill_chunk_size_sweep, which
                         run on the same GGUFs) and does not add "legacy-models" or
                         "metal-hidden" to the required-capability list —
                         and must be stated explicitly in the release notes
                         when used; it is never applied implicitly.

  --skip-bonsai2-models  Opt out of the "bonsai2-models" capability, which
                         is otherwise REQUIRED on Darwin (same shape as
                         --skip-legacy-models above, for the
                         Bonsai 2 27B target). "bonsai2-models" is the
                         evidence that the real Bonsai 2 27B gates
                         (oxibonsai-model's hybrid_forward_parity_tests and
                         oxibonsai-runtime's bonsai2_engine_tests, driven
                         against the shipped Ternary-Bonsai-2-27B-{PQ2_0,
                         PTQ1_0}.gguf files) actually ran on this host.
                         Passing this flag releases WITHOUT that real-model
                         evidence — the gate skips stage 1c entirely and
                         does not add "bonsai2-models" to the
                         required-capability list — and must be stated
                         explicitly in the release notes when used; it is
                         never applied implicitly.

  --skip-bonsai2-metal   Opt out only of the Metal-specific Bonsai 2 27B
                         evidence: oxibonsai-model's hybrid_metal_gates
                         (CPU-vs-Metal token parity and decode throughput)
                         and hybrid_metal_prefill_gates (the runner's
                         batched prefill), oxibonsai-runtime's
                         bonsai2_metal_engine_tests ("bonsai2-metal-engine"),
                         and the Metal half of the vision leg
                         ("bonsai2-vision-metal": vision_metal_tests,
                         bonsai2_vision_metal_tests,
                         bonsai2_vision_metal_runtime_tests and the CLI's
                         bonsai2_vision_cli_tests). The non-Metal Bonsai 2
                         27B gates (stage 1c's first three legs) and the CPU
                         vision leg are still required. Passing this flag
                         releases WITHOUT Metal 27B evidence and must be
                         stated explicitly in the release notes when used.

  --skip-bonsai2-vision  Opt out of the real Bonsai 2 vision evidence
                         ("bonsai2-vision" and "bonsai2-mmproj"): oxibonsai-model's
                         bonsai2_vision_tests and vision_mmproj_tests and
                         oxibonsai-runtime's bonsai2_vision_runtime_tests, driven
                         against the 27B PQ2_0 GGUF and its Qwen3-VL projector
                         (Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf), CPU path. The
                         evidence is REQUIRED on Darwin whenever the projector is
                         present beside the 27B; a host with the 27B but WITHOUT
                         the projector fails closed instead of silently releasing
                         without it. Passing this flag releases WITHOUT real
                         image-path evidence and must be stated explicitly in the
                         release notes when used. --skip-bonsai2-models implies
                         it (the vision legs load the 27B). It also drops the
                         Metal half of the vision leg ("bonsai2-vision-metal",
                         see --skip-bonsai2-metal).

  --self-test            Run this script's own offline self-test scenarios
                         (capability-manifest parsing, the stage functions,
                         the stage-0 CLI build, the main flow and the CUDA
                         waiver across ci.sh/check_cuda.sh in a scratch copy,
                         all against stand-in tools; defined in
                         scripts/release-gate-selftest.sh) and exit (no
                         release gate).

  --help, -h             Show this message and exit.

Stage 0 (always, before anything else): builds the oxibonsai CLI once
(`cargo build --release -p oxibonsai-cli --bin oxibonsai --all-features`),
exports the executable cargo reports as OXIBONSAI_CLI_BIN for every later
stage, and fails closed if the build fails or reports no runnable binary.
An OXIBONSAI_CLI_BIN already in the environment is overridden.

Environment:
  OXIBONSAI_M08_RUN_LONG=1
                         MANDATORY SEPARATE RELEASE STEP. Also runs M-08's
                         20000-token YaRN gate on Bonsai-8B.gguf
                         (oxibonsai-runtime legacy_parity_tests::
                         m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens,
                         two real decodes of roughly an hour each) inside
                         stage 1b, and requires its executed=true record by
                         name. A default run leaves the gate out only because
                         of that runtime: every release needs one passing run
                         of this script with the variable set, on the release
                         commit.
HELP_EOF
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# --require-cuda asks for real CUDA evidence, --accept-approximate-cuda-syntax says
# there is no real CUDA check: a contradiction, in either argument order. Refused
# here, after the whole loop and before the manifest is rotated or stage 0 builds
# anything, so a mistaken combination can never start a long release run.
if [[ "$ACCEPT_APPROX_CUDA" -eq 1 && "$REQUIRE_CUDA" -eq 1 ]]; then
    echo "ERROR: --accept-approximate-cuda-syntax cannot be combined with --require-cuda:" >&2
    echo "a release that requires CUDA evidence must run on a host with the CUDA toolkit." >&2
    exit 2
fi

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
# Absolute, exported for producer tests to read directly — see the
# capability-manifest contract note above. Resolved by string prefix
# (never `cd`) so this cannot fail or create a directory just from
# computing a path; the actual directory is created below by `mkdir -p`
# before anything is written into it.
case "$TARGET_DIR" in
    /*) ABS_TARGET_DIR="$TARGET_DIR" ;;
    *) ABS_TARGET_DIR="$PROJECT_ROOT/$TARGET_DIR" ;;
esac
export OXIBONSAI_CAPABILITY_REPORT="$ABS_TARGET_DIR/capability-report.json"
CAPABILITY_REPORT="$OXIBONSAI_CAPABILITY_REPORT"

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai release gate"
echo "═══════════════════════════════════════════════════════════════"
echo "project root: $PROJECT_ROOT"
echo "capability manifest: $CAPABILITY_REPORT"
echo "require-cuda: $([[ "$REQUIRE_CUDA" -eq 1 ]] && echo yes || echo no)"
echo "cuda-syntax: $([[ "$ACCEPT_APPROX_CUDA" -eq 1 ]] && echo 'approximate accepted by flag' || echo 'nvcc required')"
if [[ "$ACCEPT_APPROX_CUDA" -eq 1 ]]; then
    echo "  -> without nvcc the CUDA kernel sources are only parsed as C++ with CUDA builtins stubbed (NOT an"
    echo "     nvcc compile): the verdict will say \"PASSED with a waiver\"; state this in the release notes."
fi
echo "skip-legacy-models: $([[ "$SKIP_LEGACY_MODELS" -eq 1 ]] && echo yes || echo no)"
if [[ "$SKIP_LEGACY_MODELS" -eq 1 ]]; then
    echo "  -> real-model CPU/NEON/Metal greedy parity evidence (\"legacy-models\")"
    echo "     will NOT be required or collected this run; state this explicitly"
    echo "     in the release notes (run with --help for the full explanation)."
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> real-model CPU/NEON/Metal greedy parity evidence (\"legacy-models\")"
    echo "     is REQUIRED on this host; pass --skip-legacy-models to opt out."
fi
echo "skip-bonsai2-models: $([[ "$SKIP_BONSAI2_MODELS" -eq 1 ]] && echo yes || echo no)"
if [[ "$SKIP_BONSAI2_MODELS" -eq 1 ]]; then
    echo "  -> real Bonsai 2 27B parity evidence (\"bonsai2-models\") will NOT be"
    echo "     required or collected this run; state this explicitly in the"
    echo "     release notes (run with --help for the full explanation)."
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> real Bonsai 2 27B parity evidence (\"bonsai2-models\") is REQUIRED"
    echo "     on this host; pass --skip-bonsai2-models to opt out."
fi
echo "m08-long (OXIBONSAI_M08_RUN_LONG=1): $([[ "$M08_RUN_LONG" -eq 1 ]] && echo yes || echo no)"
if [[ "$M08_RUN_LONG" -eq 0 ]]; then
    echo "  -> M-08's 20000-token YaRN gate is NOT run this time. It is a separate,"
    echo "     mandatory release step: rerun with OXIBONSAI_M08_RUN_LONG=1 exported."
fi
echo "skip-bonsai2-metal: $([[ "$SKIP_BONSAI2_METAL" -eq 1 ]] && echo yes || echo no)"
if [[ "$SKIP_BONSAI2_METAL" -eq 1 ]]; then
    echo "  -> the Metal-specific Bonsai 2 evidence (CPU-vs-Metal token parity,"
    echo "     decode throughput, batched prefill, the Metal engine and the Metal"
    echo "     half of the vision leg, \"bonsai2-vision-metal\") will NOT be"
    echo "     required or collected this run; state this explicitly in the"
    echo "     release notes."
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> the Metal-specific Bonsai 2 evidence (including the Metal half of the"
    echo "     vision leg) is REQUIRED on this host; pass --skip-bonsai2-metal to"
    echo "     opt out."
fi
echo "skip-bonsai2-vision: $([[ "$SKIP_BONSAI2_VISION" -eq 1 ]] && echo yes || echo no)"
if [[ "$SKIP_BONSAI2_VISION" -eq 1 ]]; then
    echo "  -> real Bonsai 2 vision-projector evidence (\"bonsai2-vision\", \"bonsai2-mmproj\")"
    echo "     will NOT be required or collected this run; state this explicitly in"
    echo "     the release notes."
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> real Bonsai 2 vision-projector evidence is REQUIRED on this host whenever"
    echo "     the projector is present beside the 27B (and fails closed when it is"
    echo "     not); pass --skip-bonsai2-vision to opt out."
fi
echo ""

# Fresh-evidence guarantee: move any leftover manifest aside (to a name with
# a UTC timestamp and this process's pid, so no earlier run's evidence is
# lost and two runs never share a name) and record the moment this run
# started, so a stale file from a previous invocation (or a differently-
# configured one) can never be mistaken for this run's proof: the run starts
# from no manifest at all, and the capability check below reads only the file
# this run's producers wrote. A manifest that cannot be moved aside fails the
# gate rather than being left in place to be read as this run's.
mkdir -p "$TARGET_DIR"
if [[ -e "$CAPABILITY_REPORT" ]]; then
    ROTATED_REPORT="$CAPABILITY_REPORT.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv -f "$CAPABILITY_REPORT" "$ROTATED_REPORT" || {
        echo "FATAL: could not move the previous capability manifest $CAPABILITY_REPORT aside" >&2
        exit 2
    }
    echo "previous capability manifest kept as: $ROTATED_REPORT"
fi
RUN_START_EPOCH="$(date +%s)"

# ── 0. The release CLI binary, built once ────────────────────────────────
# The binary every later stage runs (see STAGE 0 in the header) is built here,
# once, with the full release feature set, and exported as OXIBONSAI_CLI_BIN,
# which ci.sh's nextest run and stages 1b-1d inherit. This stage FAILS CLOSED:
# no later stage runs without it. The capture is `VAR="$(cmd)" || { rc=$?; }`
# for the reason given at stage 1 below.
echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 0: release CLI binary (cargo build --release --all-features)"
echo "═══════════════════════════════════════════════════════════════"
if [[ -n "${OXIBONSAI_CLI_BIN:-}" ]]; then
    echo "NOTE: OXIBONSAI_CLI_BIN=$OXIBONSAI_CLI_BIN is set in the environment; the gate"
    echo "builds its own binary from this tree and overrides it."
fi
RELEASE_CLI_BIN="$(build_release_cli_binary)" || {
    rc=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: the release CLI binary could not be built (exit $rc)."
    echo "See cargo's output above; no later stage runs without it."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$rc"
}
export OXIBONSAI_CLI_BIN="$RELEASE_CLI_BIN"
echo "OXIBONSAI_CLI_BIN=$OXIBONSAI_CLI_BIN (exported to every later stage)"
echo ""

# ── 1. The full gate, in strict mode ─────────────────────────────────────
# This script has already moved any earlier manifest aside (above); ci.sh
# then truncates the same path right before its nextest stages
# (belt-and-braces: it must produce a clean slate whether invoked directly by
# a developer or, as here, via release-gate.sh), so the run still starts from
# an empty manifest at the same path — no earlier evidence is lost, and the
# two steps never race.
#
# NOTE: `rc=$?` is captured via `cmd || { rc=$?; ... }`, not
# `if ! cmd; then rc=$?; ...`. Inside `if ! cmd; then`, `$?` reports the
# *if-condition's own* (negated) truth value — always 0 in the then-branch
# — never cmd's real exit code. Verified empirically: an earlier version
# of this exact idiom printed "did not pass (exit 0)" for a run that had
# actually failed with a non-zero cargo-deny exit code, and — far worse —
# would have gone on to `exit 0`, so a caller under `set -e`
# (scripts/publish.sh) would NOT have stopped and would have published
# despite the gate failing. Do not "simplify" this back to `if ! cmd`.
#
# --accept-approximate-cuda-syntax is forwarded only when it was passed here, and
# counts as USED only when ci.sh's cuda-syntax stage really took it (a host with
# nvcc passes the real check and waives nothing). So with the flag, ci.sh's stdout
# is also copied to a scratch log whose `CUDA_SYNTAX_RESULT=` line is read
# afterwards (stderr stays attached); without it ci.sh runs exactly as before.
# `pipefail` makes the pipeline's status ci.sh's own, so the idiom above holds.
CI_ARGS=(--release)
CI_STDOUT_LOG=""
if [[ "$ACCEPT_APPROX_CUDA" -eq 1 ]]; then
    CI_ARGS+=(--accept-approximate-cuda-syntax)
    CI_STDOUT_LOG="$(mktemp "${TMPDIR:-/tmp}/oxibonsai_release_gate_ci_stdout.XXXXXX")" ||
        { echo "FATAL: could not create a scratch file for scripts/ci.sh's output." >&2; exit 2; }
fi
CI_RC=0
if [[ -n "$CI_STDOUT_LOG" ]]; then
    "$SCRIPT_DIR/ci.sh" "${CI_ARGS[@]}" | tee "$CI_STDOUT_LOG" || CI_RC=$?
else
    "$SCRIPT_DIR/ci.sh" "${CI_ARGS[@]}" || CI_RC=$?
fi
if [[ "$CI_RC" -ne 0 ]]; then
    [[ -n "$CI_STDOUT_LOG" ]] && rm -f "$CI_STDOUT_LOG"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: scripts/ci.sh --release did not pass (exit $CI_RC)."
    echo "See its output above for the failing stage."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$CI_RC"
fi

# With the flag: ci.sh passed, so its cuda-syntax stage passed with a real nvcc
# check or the accepted approximate one, and its CUDA_SYNTAX_RESULT line says
# which; anything else is a broken contract and fails closed.
if [[ -n "$CI_STDOUT_LOG" ]]; then
    CUDA_VERDICT="none"
    grep -q -F -x 'CUDA_SYNTAX_RESULT=nvcc-ok' "$CI_STDOUT_LOG" && CUDA_VERDICT="nvcc-ok"
    grep -q -F -x 'CUDA_SYNTAX_RESULT=approximate-accepted' "$CI_STDOUT_LOG" && CUDA_VERDICT="approximate-accepted"
    rm -f "$CI_STDOUT_LOG"
    case "$CUDA_VERDICT" in
        nvcc-ok) echo "cuda-syntax: a real nvcc check passed; the waiver was not needed and is not used." ;;
        approximate-accepted)
            CUDA_WAIVER_USED=1
            # Appended AFTER ci.sh, which truncates the manifest before its nextest stages.
            printf '%s\n' '{"capability":"cuda-syntax","executed":false,"waived":"approximate-accepted","test":"scripts/check_cuda.sh"}' \
                >>"$CAPABILITY_REPORT" ||
                { echo "FATAL: could not record the CUDA waiver in $CAPABILITY_REPORT." >&2; exit 2; }
            echo ""
            echo "WAIVER IN EFFECT: --accept-approximate-cuda-syntax was given and used. The CUDA kernel sources were"
            echo "only parsed as C++ with CUDA builtins stubbed, NOT compiled by nvcc: the CUDA backend is not certified"
            echo "by this run. Recorded in the capability manifest (capability \"cuda-syntax\", waived \"approximate-accepted\")."
            ;;
        *)
            echo "RELEASE GATE FAILED: ci.sh passed under --accept-approximate-cuda-syntax but printed no" >&2
            echo "recognised CUDA_SYNTAX_RESULT line, so the gate cannot say whether the waiver was used." >&2
            exit 2
            ;;
    esac
fi

# ── 1b. Real-model legacy parity gate, SERIALIZED (D-6(4)) ───────────────
# Invoked directly, and with `--test-threads=1`, for the reason documented in
# THE REAL-MODEL PARITY STAGE MUST BE SERIALIZED at the top of this file.
# These tests are NOT `#[ignore]`d: on a host with the GGUFs they run the full
# Reference/NEON/Metal matrix; on a host without them they self-skip and
# record `executed: false`, which the capability check below turns into a
# failed gate unless --skip-legacy-models was passed.
if [[ "$SKIP_LEGACY_MODELS" -eq 1 ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Real-model legacy parity gate: SKIPPED BY REQUEST"
    echo "═══════════════════════════════════════════════════════════════"
    echo "--skip-legacy-models was passed: this release carries NO real-model"
    echo "CPU/NEON/Metal greedy parity evidence. Say so in the release notes."
elif [[ "$(uname -s)" != "Darwin" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Real-model legacy parity gate: NOT AVAILABLE ON THIS HOST"
    echo "═══════════════════════════════════════════════════════════════"
    echo "Both parity test files are #![cfg(all(feature = \"metal\", target_os ="
    echo "\"macos\"))] — there is no non-macOS GPU tier to compare against on"
    echo "this machine (CUDA is compile-blind here). Nothing to run."
else
    LEGACY_MODELS_DIR="${OXIBONSAI_MODELS_DIR:-$PROJECT_ROOT/models}"
    LEGACY_FIXTURES=(
        "${LEGACY_MODEL_FILES[@]}"
        "tokenizer.json"
        "Ternary-Bonsai-1.7B-ONNX/onnx/model_q2.onnx"
    )
    LEGACY_MISSING=()
    for fixture in "${LEGACY_FIXTURES[@]}"; do
        [[ -s "$LEGACY_MODELS_DIR/$fixture" ]] || LEGACY_MISSING+=("$fixture")
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Real-model legacy parity gate (--test-threads=1)"
    echo "═══════════════════════════════════════════════════════════════"
    echo "models dir: $LEGACY_MODELS_DIR"
    if [[ "${#LEGACY_MISSING[@]}" -gt 0 ]]; then
        echo "MISSING fixtures: ${LEGACY_MISSING[*]}"
        echo "The gates below will self-skip and record executed=false, which the"
        echo "capability check will report as a FAILED release gate. Put the GGUFs"
        echo "in place, point OXIBONSAI_MODELS_DIR at them, or pass"
        echo "--skip-legacy-models deliberately."
    fi

    if [[ "$M08_RUN_LONG" -eq 1 ]]; then
        echo "OXIBONSAI_M08_RUN_LONG=1: oxibonsai-runtime's legacy_parity_tests also"
        echo "runs M-08's 20000-token YaRN gate on Bonsai-8B.gguf (two child"
        echo "processes of roughly an hour each) and the capability check requires it."
    fi

    # Both crates, both halves: oxibonsai-model is the tokenizer-free
    # cross-tier numeric gate, oxibonsai-runtime is the golden-TEXT half
    # (the captured Metal reference output, embedded in the runtime crate's
    # own test file, compared through the real TokenizerBridge and the real
    # SamplingParams).
    for legacy_pkg in oxibonsai-model oxibonsai-runtime; do
        echo ""
        echo "── $legacy_pkg::legacy_parity_tests ──────────────────────────"
        OXIBONSAI_MODELS_DIR="$LEGACY_MODELS_DIR" \
            cargo test --release -p "$legacy_pkg" --features metal \
            --test legacy_parity_tests -- --test-threads=1 --nocapture || {
            rc=$?
            echo ""
            echo "═══════════════════════════════════════════════════════════════"
            echo "RELEASE GATE FAILED: $legacy_pkg's real-model parity gate did"
            echo "not pass (exit $rc). This is the release blocker class: read the"
            echo "printed per-step delta series and top-5s above before touching"
            echo "any bound — a token-chain divergence and a numeric-bound breach"
            echo "are very different findings."
            echo "═══════════════════════════════════════════════════════════════"
            exit "$rc"
        }
    done

    # The head-free Metal hidden-state prefill behind dense embeddings, on the
    # same three legacy GGUFs, once per file (`run_metal_hidden_stage`). It
    # fails closed on a missing model file and on the first failing leg, and
    # the capability check below requires its `metal-hidden` record by name.
    echo ""
    echo "── oxibonsai-model::metal_hidden_parity_tests (once per legacy model) ──"
    run_metal_hidden_stage "$LEGACY_MODELS_DIR" || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: the Metal hidden-state parity gate did not"
        echo "pass (exit $rc). The output above names the model whose leg failed,"
        echo "or the legacy GGUF that is missing."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }

    # Speculative greedy against plain greedy on the two ternary models, on
    # the Metal route: a verify of 8 rows or more (draft length 7 and up, and
    # the adaptive lookahead's upper range) runs the tiled GEMM family while
    # the decode it must equal runs the single-token one. Required by name.
    echo ""
    echo "── oxibonsai-model::speculative_ternary_metal_gates (speculative == plain greedy, ternary Metal) ──"
    run_speculative_stage "$LEGACY_MODELS_DIR" || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: the speculative-decoding gate did not pass"
        echo "(exit $rc). A divergence names the model, prompt, draft length, token"
        echo "index and the verify batch size that produced it: that is a finding"
        echo "about the tiled-GEMM kernel-family switch, not a bound to loosen."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }

    # The embedding leg: the semantic smoke test and the benchmark, which
    # opts in explicitly (this stage sets OXIBONSAI_EMBED_BENCH=1 and OXI_MODEL
    # itself, so the leg never self-skips) and never asks for the per-token
    # reference at 2000 tokens (OXIBONSAI_EMBED_BENCH_PER_TOKEN stays unset).
    echo ""
    echo "── oxibonsai-runtime::embeddings_model_backed (embedding smoke + benchmark, explicit opt-in) ──"
    run_embedding_stage "$LEGACY_MODELS_DIR" || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: the real-model embedding leg did not pass"
        echo "(exit $rc)."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }

    # M-18: the fused Metal prefill's chunk-size sweep against sequential
    # decode, once on Bonsai-8B and once on the Ternary-Bonsai-1.7B
    # (`run_prefill_sweep_stage`). It fails closed on a missing model file and
    # on the first failing leg, and the capability check below requires the
    # sweep's own record by name.
    echo ""
    echo "── oxibonsai-model::lib::$SWEEP_TEST_FILTER (M-18 fused-prefill chunk-size sweep, once per model) ──"
    run_prefill_sweep_stage "$LEGACY_MODELS_DIR" || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: the M-18 fused-prefill chunk-size sweep did not"
        echo "pass (exit $rc). The output above names the model whose leg failed, or"
        echo "the GGUF that is missing; the per-chunk-size timings and the load"
        echo "average they were measured under are printed beside the verdict."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }
fi

# ── 1c. Real Bonsai 2 27B gates, SERIALIZED, RELEASE ─────────────────────
# `run_bonsai2_models_stage` (defined above `check_capability_manifest`)
# runs the model-crate gates, then the runtime-crate engine gates, then —
# on Darwin, unless --skip-bonsai2-metal, and once the test file exists —
# the Metal gates, one after another, never in parallel, so this script
# never maps more than one 27B GGUF (5.9-7.2 GB) at a time. It fails CLOSED
# before invoking cargo at all when either release GGUF is missing and
# --skip-bonsai2-models was not passed.
BONSAI2_MODELS_DIR="${OXIBONSAI_MODELS_DIR:-$PROJECT_ROOT/models}"
BONSAI2_GOLDEN_DIR="$PROJECT_ROOT/crates/oxibonsai-model/tests/fixtures/bonsai2_golden"
BONSAI2_METAL_TEST_FILE="$PROJECT_ROOT/crates/oxibonsai-model/tests/hybrid_metal_gates.rs"
BONSAI2_IS_DARWIN=0
[[ "$(uname -s)" == "Darwin" ]] && BONSAI2_IS_DARWIN=1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real Bonsai 2 27B gate (--test-threads=1, one 27B process at a time)"
echo "═══════════════════════════════════════════════════════════════"
echo "models dir: $BONSAI2_MODELS_DIR"
echo "golden dir: $BONSAI2_GOLDEN_DIR (vendored)"
run_bonsai2_models_stage "$SKIP_BONSAI2_MODELS" "$SKIP_BONSAI2_METAL" \
    "$BONSAI2_MODELS_DIR" "$BONSAI2_GOLDEN_DIR" "$BONSAI2_METAL_TEST_FILE" \
    "$BONSAI2_IS_DARWIN" || {
    rc=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: the real Bonsai 2 27B gate did not pass (exit $rc)."
    echo "See its output above for which leg failed, or which file is missing."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$rc"
}

# The vision leg of stage 1c: the 27B with its Qwen3-VL projector on the CPU
# path (`run_bonsai2_vision_stage`), required by name below wherever the
# projector exists. It fails CLOSED when the 27B is present without the
# projector, unless --skip-bonsai2-vision says so.
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real Bonsai 2 vision gate (CPU path, --test-threads=1)"
echo "═══════════════════════════════════════════════════════════════"
run_bonsai2_vision_stage "$SKIP_BONSAI2_VISION" "$SKIP_BONSAI2_MODELS" \
    "$BONSAI2_MODELS_DIR" "$BONSAI2_IS_DARWIN" || {
    rc=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: the real Bonsai 2 vision gate did not pass (exit $rc)."
    echo "See its output above for which leg failed, or which file is missing."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$rc"
}

# Its Metal half (`run_bonsai2_vision_metal_stage`), required by name below
# wherever the CPU vision leg ran, unless --skip-bonsai2-metal.
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real Bonsai 2 vision gate (Metal tower, runner and CLI, --test-threads=1)"
echo "═══════════════════════════════════════════════════════════════"
run_bonsai2_vision_metal_stage "$SKIP_BONSAI2_VISION" "$SKIP_BONSAI2_MODELS" \
    "$SKIP_BONSAI2_METAL" "$BONSAI2_MODELS_DIR" "$BONSAI2_IS_DARWIN" || {
    rc=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: the real Bonsai 2 Metal vision gate did not pass (exit $rc)."
    echo "See its output above for which leg failed, or which file is missing."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$rc"
}

# ── 1d. Real-model lib/bin acceptance cases, SERIALIZED, RELEASE ────────
# The remaining real-model acceptance cases live as ordinary `#[test]`s
# inside library/binary crates rather than their own integration-test
# binary, so they cannot be selected by `--test <name>` the way stages 1b/1c
# select a whole file: each is run by its own name, `--test-threads=1`,
# strictly after every earlier real-model stage has fully exited (one
# real-model process at a time). Every case here is env-gated and
# self-skips with a capability record when its own variable is unset —
# never `#[ignore]`d — so this stage's own exit code is meaningful even
# before every model file is in place; the required-test-name check in the
# capability-manifest stage below is what turns "self-skipped" into a
# failed release gate when the evidence is supposed to be required.
LEGACY_MODELS_DIR_1D="${OXIBONSAI_MODELS_DIR:-$PROJECT_ROOT/models}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Real-model lib/bin acceptance cases (--test-threads=1)"
echo "═══════════════════════════════════════════════════════════════"
# Same three-way gate as stage 1b (skip flag / non-Darwin / run): these
# three cases (two `--lib`, one `--test metal_greedy_cpu_fallback_tests`)
# are the Metal-fused-decode, stream-stop and Metal↔CPU byte-parity
# legacy-models evidence, so they must not run — or be required — under the
# exact conditions stage 1b itself does not run.
if [[ "$SKIP_LEGACY_MODELS" -eq 1 ]]; then
    echo ""
    echo "── oxibonsai-runtime::lib (legacy-models): SKIPPED BY REQUEST ──"
    echo "--skip-legacy-models was passed: these three real-model cases are not run."
elif [[ "$(uname -s)" != "Darwin" ]]; then
    echo ""
    echo "── oxibonsai-runtime::lib (legacy-models): NOT AVAILABLE ON THIS HOST ──"
    echo "Both cases need the fused Metal greedy-decode path; nothing to run here."
else
    run_legacy_lib_bin_stage "$LEGACY_MODELS_DIR_1D" || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: oxibonsai-runtime's real-model lib / Metal-fallback"
        echo "cases did not pass (exit $rc)."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }
fi

if [[ "$SKIP_BONSAI2_MODELS" -eq 1 ]]; then
    echo ""
    echo "── oxibonsai-runtime::lib / oxibonsai-cli::bin (bonsai2-models): SKIPPED ──"
    echo "--skip-bonsai2-models was passed: these two real-27B lib/bin cases are not run."
else
    BONSAI2_PQ2_PATH_1D="$BONSAI2_MODELS_DIR/$BONSAI2_PQ2_FILE"
    echo ""
    echo "── oxibonsai-runtime::lib (bonsai2-models: real_27b_ embedding case) ───"
    OXI_BONSAI2_PQ2_GGUF="$BONSAI2_PQ2_PATH_1D" \
        cargo test --release -p oxibonsai-runtime --all-features --lib \
        -- --test-threads=1 real_27b_ || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: oxibonsai-runtime's real_27b_ lib case did not"
        echo "pass (exit $rc)."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }

    echo ""
    echo "── oxibonsai-cli::bin (bonsai2-models: real_27b_ embedding-serving case) ──"
    OXI_BONSAI2_PQ2_GGUF="$BONSAI2_PQ2_PATH_1D" \
        cargo test --release -p oxibonsai-cli --all-features --bin oxibonsai \
        -- --test-threads=1 real_27b_ || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: oxibonsai-cli's real_27b_ bin case did not pass"
        echo "(exit $rc)."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }
fi

# The closing block of a PASSED run; both PASS exits below use it, so a waived
# run (CUDA_WAIVER_USED=1) can never read as a full one.
#   print_gate_passed <the verdict line of a run that waived nothing>
print_gate_passed() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    if [[ "$CUDA_WAIVER_USED" -eq 1 ]]; then
        echo "RELEASE GATE PASSED with a waiver: CUDA kernel syntax was checked approximately (no nvcc)."
        echo "The kernels were parsed as C++ with CUDA builtins stubbed, not compiled by nvcc, and nothing ran"
        echo "on CUDA hardware: the CUDA backend is not certified by this run. State this in the release notes."
        echo "scripts/publish.sh runs this gate again and needs --accept-approximate-cuda-syntax as well."
    else
        echo "$1"
    fi
    echo "═══════════════════════════════════════════════════════════════"
}

# ── 2. Hardware-capability enforcement (T-05) ────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Capability manifest check"
echo "═══════════════════════════════════════════════════════════════"

REQUIRED_CAPS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    REQUIRED_CAPS+=("metal")
    # A capability that self-skipped must not read as covered. "metal" alone
    # cannot express this one — `metal_k_quant_gemv_parity` writes 30+
    # metal/executed=true records on any Metal host regardless of whether
    # the real models were ever touched — so the real-model gate has its own
    # name. See the --skip-legacy-models note in this file's header.
    if [[ "$SKIP_LEGACY_MODELS" -eq 0 ]]; then
        REQUIRED_CAPS+=("legacy-models")
        # "legacy-models" has records from other, lighter tests too (the
        # gguf_loader.rs/model_registry.rs detection-only cases) — require
        # every real legacy-model gate individually, by name; see
        # `legacy_models_require_tests_arg` for the list and why. The M-08
        # 20000-token YaRN gate is on it exactly when
        # OXIBONSAI_M08_RUN_LONG=1 (the separate release step described at
        # the top of this file).
        REQUIRED_CAPS+=("$(legacy_models_require_tests_arg "$M08_RUN_LONG")")
        # The Metal hidden-state gate runs on the same legacy models (stage
        # 1b, once per file), so it is required under the same condition. A
        # bare "metal-hidden" would be satisfied by any one record, so the
        # gate is named as well: a run that self-skipped writes
        # executed=false for it and fails here.
        REQUIRED_CAPS+=("metal-hidden")
        REQUIRED_CAPS+=("--require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME")
    fi
    # Same reasoning, for the Bonsai 2 27B target: see the
    # --skip-bonsai2-models note in this file's header.
    if [[ "$SKIP_BONSAI2_MODELS" -eq 0 ]]; then
        REQUIRED_CAPS+=("bonsai2-models")
        # "bonsai2-models" has records from OTHER, lighter tests too (e.g.
        # header-only metadata checks) — require these five core gates (the
        # three model-crate gates, the two runtime-engine gates) and the
        # two stage-1d lib/bin cases individually, so a release can never be
        # cut on the strength of only the cheaper ones.
        BONSAI2_REQUIRED_NAMES="\
oxibonsai-model::hybrid_forward_parity_tests::hybrid_real_27b_pq2_0_matches_the_fork_goldens_bonsai2,\
oxibonsai-model::hybrid_forward_parity_tests::hybrid_real_27b_ptq1_0_greedy_matches_the_fork_goldens_bonsai2,\
oxibonsai-model::hybrid_forward_parity_tests::hybrid_real_27b_layers_match_the_f64_reference_bonsai2,\
oxibonsai-runtime::bonsai2_engine_tests::bonsai2_pq2_engine_greedy_matches_the_fork_goldens,\
oxibonsai-runtime::bonsai2_engine_tests::bonsai2_ptq1_engine_greedy_matches_the_fork_goldens,\
oxibonsai-runtime::bonsai2_runtime_tests::real_27b_tokenize_matches_the_fork_for_all_five_golden_texts_bonsai2,\
oxibonsai-runtime::bonsai2_runtime_tests::real_27b_chat_template_matches_the_fork_for_all_five_golden_cases_bonsai2,\
oxibonsai-runtime::bonsai2_runtime_tests::real_27b_info_reports_pq2_0_variant_and_layer_split_bonsai2,\
oxibonsai-runtime::bonsai2_runtime_tests::real_27b_info_reports_ptq1_0_variant_and_layer_split_bonsai2,\
oxibonsai-runtime::lib::real_27b_hybrid_embed_returns_a_unit_vector,\
oxibonsai-cli::bin::real_27b_hybrid_model_serves_embeddings"
        if [[ "$SKIP_BONSAI2_METAL" -eq 0 ]]; then
            # The two Metal 27B gates (CPU-vs-Metal token parity, decode
            # throughput), required on Darwin unless explicitly opted out —
            # see the --skip-bonsai2-metal note above.
            BONSAI2_REQUIRED_NAMES="$BONSAI2_REQUIRED_NAMES,\
oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_matches_cpu_tokens_bonsai2,\
oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_decode_throughput_bonsai2"
            # The runner's two batched-prefill gates, under the same opt out.
            BONSAI2_REQUIRED_NAMES="$BONSAI2_REQUIRED_NAMES,$(IFS=,; echo "${BONSAI2_PREFILL_TEST_NAMES[*]}")"
        fi
        REQUIRED_CAPS+=("--require-tests=bonsai2-models:$BONSAI2_REQUIRED_NAMES")
        if [[ "$SKIP_BONSAI2_METAL" -eq 0 ]]; then
            # The four Metal engine gates on the product path record under
            # their own name.
            REQUIRED_CAPS+=("bonsai2-metal-engine")
            REQUIRED_CAPS+=("--require-tests=bonsai2-metal-engine:\
oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_pq2_0_matches_cpu_and_fork_bonsai2,\
oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_ptq1_0_matches_cpu_and_fork_bonsai2,\
oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_decode_throughput_and_memory_bonsai2,\
oxibonsai-runtime::bonsai2_metal_engine_tests::real_27b_metal_engine_serves_chat_like_the_cpu_engine_bonsai2")
        fi
    fi
    # The vision evidence, wherever the vision leg ran: the same predicate
    # that decides whether `run_bonsai2_vision_stage` runs the legs. Both
    # capabilities are required by name — the lighter synthetic cases of the
    # same files also write records, and a bare capability name would accept
    # any one of them.
    if [[ "$(bonsai2_vision_required "$SKIP_BONSAI2_VISION" "$SKIP_BONSAI2_MODELS" \
        "$BONSAI2_MODELS_DIR" "$BONSAI2_IS_DARWIN")" -eq 1 ]]; then
        REQUIRED_CAPS+=("bonsai2-vision")
        REQUIRED_CAPS+=("--require-tests=bonsai2-vision:$(IFS=,; echo "${BONSAI2_VISION_TEST_NAMES[*]}")")
        REQUIRED_CAPS+=("bonsai2-mmproj")
        REQUIRED_CAPS+=("--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME")
    fi
    # The Metal half of the vision evidence, wherever it ran: the Metal
    # tower's gate is a second "bonsai2-mmproj" name, the rest are
    # "bonsai2-vision-metal", each by name.
    if [[ "$(bonsai2_vision_metal_required "$SKIP_BONSAI2_VISION" "$SKIP_BONSAI2_MODELS" \
        "$SKIP_BONSAI2_METAL" "$BONSAI2_MODELS_DIR" "$BONSAI2_IS_DARWIN")" -eq 1 ]]; then
        REQUIRED_CAPS+=("--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_METAL_TEST_NAME")
        REQUIRED_CAPS+=("bonsai2-vision-metal")
        REQUIRED_CAPS+=("--require-tests=bonsai2-vision-metal:$(IFS=,; echo "${BONSAI2_VISION_METAL_TEST_NAMES[*]}")")
    fi
fi
if [[ "$REQUIRE_CUDA" -eq 1 ]]; then
    REQUIRED_CAPS+=("cuda")
fi

if [[ "${#REQUIRED_CAPS[@]}" -eq 0 ]]; then
    echo "No hardware capability is required on this host/configuration. Skipping."
    print_gate_passed "RELEASE GATE PASSED."
    exit 0
fi
echo "Required on this host: ${REQUIRED_CAPS[*]}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "FATAL: python3 not found — cannot parse the capability manifest." >&2
    exit 2
fi

# See check_capability_manifest() (defined near the top of this file, next
# to the capability-manifest contract comment) for what "malformed" does to
# the exit code and why it is factored out this way.
if ! check_capability_manifest "$CAPABILITY_REPORT" "$RUN_START_EPOCH" "${REQUIRED_CAPS[@]}"; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: a required hardware capability did not"
    echo "actually execute this run (see above): its producer test either"
    echo "self-skipped (executed=false), never ran, or has not adopted the"
    echo "manifest contract documented at the top of this script."
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi

print_gate_passed "RELEASE GATE PASSED. scripts/publish.sh is safe to run next."
exit 0
