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
#     "image-parity", "legacy-models", "bonsai2-models" (extend this list
#     here, in the same edit that adds a new gated capability, so this file
#     stays the single source of truth).
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
#     stopwatch.
# This script DELETES the manifest before invoking scripts/ci.sh and only
# trusts entries written *during this run* (mtime check) — a leftover file
# from a previous invocation, or one for a different feature set, must
# never be read as this run's evidence. Producers therefore do not need to
# truncate the file themselves; appending is always correct.
#
# Required-on-this-host capabilities (matches this script's own spec):
#   - "metal" is required whenever `uname -s` is Darwin (this project
#     targets Apple Silicon as a first-class backend).
#   - "cuda" is required only when --require-cuda is passed (this session's
#     dev machine has no CUDA hardware at all; do not require it by
#     default — see CONTEXT: "CUDA code is compile-blind").
#   - "legacy-models" is required on Darwin unless --skip-legacy-models is
#     passed. It is the evidence that the real-model CPU/NEON/Metal greedy
#     parity gate — the strongest correctness signal this project can
#     produce, and the one that guards README's "byte-identical output at
#     --temperature 0" claim — actually RAN against the shipped GGUFs. The
#     producer tests are no longer `#[ignore]`d (orchestrator ruling D-6(4)),
#     so on a host that has models/ they run and record; on a host that does
#     not they record executed=false, and a release must not be cut from such
#     a run without the operator saying so out loud. --skip-legacy-models is
#     that explicit statement; it is visible in this script's own output.
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
#     same as the five gates above. --skip-bonsai2-metal is the narrower opt
#     out: it drops only the Metal-specific evidence requirement, not the
#     whole "bonsai2-models" capability.
#
# ── THE REAL-MODEL PARITY STAGE MUST BE SERIALIZED ──────────────────────────
# Stage 1b's real legacy-model gates (the six cross-tier parity tests, three
# in oxibonsai-model and three in oxibonsai-runtime, plus the M-18 / M-02 /
# M-08 / CLI-CORE / CONVERT-EXPORT measurement gates, all in the two
# `legacy_parity_tests` binaries), stage 1c's nine real
# Bonsai 2 27B gates (`hybrid_forward_parity_tests`'s three
# `hybrid_real_27b_*_bonsai2` cases, `bonsai2_engine_tests`'s two
# `bonsai2_*_engine_greedy_matches_the_fork_goldens` cases, and
# `bonsai2_runtime_tests`'s four `real_27b_*_bonsai2` G6/G7/G9 cases) plus, on
# Darwin, `hybrid_metal_gates`'s two Metal 27B cases, and stage 1d's real-model
# lib/bin cases each load a multi-GB GGUF and decode through it. Run
# concurrently on an 8-core/24 GB M3 they have driven the machine's load
# average past 90. Stages 1b-1d therefore invoke every real-model test
# binary directly with `--test-threads=1`, one binary at a time, and the
# heavier tests additionally carry an in-binary mutex
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
# that; it is the producer tests' job to start writing to the manifest
# (tracked as CI-GATE's T-05 deviation until TESTS-INFRA lands it).
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
#   OXIBONSAI_M08_RUN_LONG=1 ./scripts/release-gate.sh
#                                                     # MANDATORY SEPARATE RELEASE STEP:
#                                                     # also run and require M-08's
#                                                     # 20000-token YaRN gate (below)
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

# Release file names of the two Bonsai 2 27B bands, shared by the real
# Bonsai 2 27B stage below and its `--self-test` stub scenarios.
BONSAI2_PQ2_FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
BONSAI2_PTQ1_FILE="Ternary-Bonsai-2-27B-PTQ1_0.gguf"

# Parses the JSONL capability manifest at `report_path` and verifies that
# every capability named in the remaining arguments has at least one
# `"executed": true` record written no earlier than `run_start_epoch`.
# Exit 0 = every required capability has real, fresh evidence; exit 1 = at
# least one does not, OR the manifest itself is unusable (missing, stale,
# or contains a malformed line). Factored into its own function (rather
# than an inline heredoc at the call site) so `--self-test` below can
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
# do not need it.
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
# cases, the M-18 / M-02 / M-08-control / CLI-CORE real-model measurement
# gates and the CONVERT-EXPORT round trip (ONNX export -> GGUF -> real
# tokenizer). With `m08_long` = 1 (`OXIBONSAI_M08_RUN_LONG=1` in this
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
# (CPU-vs-Metal token parity + decode throughput).
#
# Fails CLOSED, before invoking cargo at all, when either release GGUF is
# missing from `models_dir` and `skip` is not set: a partial 27B evidence
# run is refused outright (naming what is missing and both ways out) rather
# than left to run for several minutes and self-skip, which the capability
# check at the end of this script would catch anyway but only after paying
# the wall-clock cost of everything that DID have its file.
#
# All inputs are explicit arguments — never a global — so `--self-test`
# below can drive this exact function against a scratch directory and a
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
    OXIBONSAI_MODELS_DIR="$models_dir" \
        cargo test --release -p oxibonsai-model --all-features \
        --test hybrid_forward_parity_tests -- --test-threads=1 --nocapture || return $?

    echo "── oxibonsai-runtime::bonsai2_engine_tests (real 27B gates) ───────────"
    env "OXIBONSAI_MODELS_DIR=$models_dir" "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_engine_tests -- --test-threads=1 --nocapture || return $?

    # G6/G7/G9 all live in one binary: G6 (tokenization) and G7 (chat
    # template) are header-only (`mmap` + metadata parse, never a tensor
    # byte) and self-locate under `models_dir` on their own; G9 (`oxibonsai
    # info --json`, one case per band) needs its own per-band env var (see
    # `locate_named_27b_gguf`'s doc comment: it spawns a `cargo build` and a
    # fresh process per case, so it deliberately has no `models/` fallback).
    # One invocation with every var set covers all of it — this also avoids
    # a G9 case ever reaching its `OXI_REQUIRE_MODEL_FILES=1` hard-failure
    # branch just because a *different* leg forgot to export its env var.
    echo "── oxibonsai-runtime::bonsai2_runtime_tests (G6/G7/G9) ─────────────────"
    env "OXIBONSAI_MODELS_DIR=$models_dir" "OXI_BONSAI2_PQ2_GGUF=$pq2_path" \
        "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" \
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
    env "OXI_BONSAI2_PQ2_GGUF=$pq2_path" "OXI_BONSAI2_PTQ1_GGUF=$ptq1_path" \
        "OXI_BONSAI2_GOLDEN_DIR=$golden_dir" \
        cargo test --release -p oxibonsai-model --features metal \
        --test hybrid_metal_gates -- --test-threads=1 --nocapture || return $?
    return 0
}

# ── --self-test: exercise check_capability_manifest() offline ───────────
# No cargo, no hardware, no network: builds synthetic JSONL manifests under
# a throwaway `mktemp -d` (honours $TMPDIR; never a hardcoded /tmp path)
# and asserts check_capability_manifest()'s verdict on each scenario. This
# is what "pins" the malformed-line-fails-closed behaviour documented
# above with a test, per item (d): it cannot silently regress back to the
# old "reported, not silently ignored, but still exits 0" shape without
# this failing.
release_gate_self_test() {
    local failures=0
    local total=0
    local work_dir
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch directory" >&2
        return 1
    }

    local now future_epoch
    now="$(date +%s)"
    # A run_start_epoch far in the future makes every real-filesystem mtime
    # "predate" it, exercising the staleness branch without needing a
    # platform-specific `touch -t` (BSD and GNU disagree on that flag's
    # syntax) to backdate a file instead.
    future_epoch=$((now + 1000000))

    check() {
        local label="$1" expect_rc="$2"
        shift 2
        echo ""
        echo "-- self-test: $label --"
        total=$((total + 1))
        local rc=0
        check_capability_manifest "$@" || rc=$?
        if [[ "$rc" -ne "$expect_rc" ]]; then
            echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
            failures=$((failures + 1))
        else
            echo "OK: self-test '$label' (exit $rc, as expected)"
        fi
    }

    # 1. Manifest file does not exist at all -> FAIL.
    check "missing manifest" 1 "$work_dir/does-not-exist.jsonl" "$now" metal

    # 2. Manifest exists but is empty -> FAIL (no records for the required cap).
    : >"$work_dir/empty.jsonl"
    check "empty manifest" 1 "$work_dir/empty.jsonl" "$now" metal

    # 3. Only executed:false records -> FAIL (self-skipped is not proof).
    printf '%s\n' '{"capability":"metal","executed":false,"test":"t::skip"}' \
        >"$work_dir/all_false.jsonl"
    check "all executed:false" 1 "$work_dir/all_false.jsonl" "$now" metal

    # 4. One executed:true record -> PASS.
    printf '%s\n' '{"capability":"metal","executed":true,"test":"t::ran"}' \
        >"$work_dir/one_true.jsonl"
    check "one executed:true" 0 "$work_dir/one_true.jsonl" "$now" metal

    # 5. Manifest predates this "run" -> FAIL (stale evidence), proven via
    #    a run_start_epoch set in the future rather than a backdated mtime.
    check "stale manifest (future run_start)" 1 "$work_dir/one_true.jsonl" "$future_epoch" metal

    # 6. A malformed line alongside an otherwise-good executed:true record
    #    -> FAIL. This is the exact behaviour item (d) pins.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::ran"}' \
        'not valid json at all' \
        >"$work_dir/malformed_plus_good.jsonl"
    check "malformed line + good record" 1 "$work_dir/malformed_plus_good.jsonl" "$now" metal

    # 7. A required capability with zero records, even though the manifest
    #    is otherwise well-formed and fresh -> FAIL (checked per-capability,
    #    not "the manifest has at least one good line anywhere").
    check "required cap absent from otherwise-good manifest" 1 \
        "$work_dir/one_true.jsonl" "$now" cuda

    # 8. Two required capabilities, both satisfied -> PASS.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::ran"}' \
        '{"capability":"cuda","executed":true,"test":"t::ran_cuda"}' \
        >"$work_dir/both_true.jsonl"
    check "two required capabilities, both satisfied" 0 \
        "$work_dir/both_true.jsonl" "$now" metal cuda

    # 9. HANDOVER-INFRA: the "bonsai2-models" capability is checked by the
    #    exact same generic logic as every other capability name above — this
    #    scenario exercises it by name so a future rename/typo of the string
    #    this script and `Capability::Bonsai2Models::as_str()` must agree on
    #    fails a test here, not only in production.
    printf '%s\n' '{"capability":"bonsai2-models","executed":true,"test":"t::bonsai2_ran"}' \
        >"$work_dir/bonsai2_models_true.jsonl"
    check "bonsai2-models executed:true" 0 "$work_dir/bonsai2_models_true.jsonl" "$now" bonsai2-models

    # 10/11. `--require-tests`: a capability can have `executed: true`
    # records yet still be missing one of the SPECIFIC gates this script
    # requires by name (item 1(b)'s per-capability required-test-name list).
    printf '%s\n%s\n' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::a"}' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::b"}' \
        >"$work_dir/require_tests_two.jsonl"
    check "require-tests: every named test present" 0 \
        "$work_dir/require_tests_two.jsonl" "$now" \
        "--require-tests=bonsai2-models:gate::a,gate::b"
    check "require-tests: one named test missing" 1 \
        "$work_dir/require_tests_two.jsonl" "$now" \
        "--require-tests=bonsai2-models:gate::a,gate::c"

    # `duration_ms`: an optional field (`oxibonsai_testkit::capability::
    # record_executed_timed`). A record that carries it must still pass the
    # ordinary check (and print a total on the "OK" line, verified by eye
    # above, not asserted here); a non-numeric value must not crash the
    # parser, just be treated as if the field were absent.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::timed","duration_ms":1500}' \
        '{"capability":"metal","executed":true,"test":"t::bad_duration","duration_ms":"not-a-number"}' \
        >"$work_dir/duration_ms.jsonl"
    check "duration_ms: numeric and non-numeric values both still pass" 0 \
        "$work_dir/duration_ms.jsonl" "$now" metal

    # 12-... : `run_bonsai2_models_stage`'s fail-closed and leg-ordering
    # behaviour, via a PATH-shimmed `cargo` that only logs its own
    # invocations (one line per call) instead of building or running
    # anything — no real 27B GGUF, build or hardware involved.
    local fake_bin fake_log stub_output
    fake_bin="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest_bin.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch bin directory" >&2
        failures=$((failures + 1))
        fake_bin=""
    }
    if [[ -n "$fake_bin" ]]; then
        fake_log="$work_dir/fake_cargo_invocations.log"
        stub_output="$work_dir/stage_stub_output.log"
        cat >"$fake_bin/cargo" <<'CARGO_STUB_EOF'
#!/usr/bin/env bash
echo "$@" >>"$OXIBONSAI_SELFTEST_CARGO_LOG"
exit 0
CARGO_STUB_EOF
        chmod +x "$fake_bin/cargo"

        run_stage_stub() {
            local label="$1" expect_rc="$2" expect_calls="$3"
            shift 3
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            : >"$fake_log"
            local rc=0
            OXIBONSAI_SELFTEST_CARGO_LOG="$fake_log" PATH="$fake_bin:$PATH" \
                run_bonsai2_models_stage "$@" >"$stub_output" 2>&1 || rc=$?
            local calls
            calls="$(wc -l <"$fake_log" | tr -d ' ')"
            if [[ "$rc" -ne "$expect_rc" ]]; then
                echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
                echo "  stage output: $(cat "$stub_output")"
                failures=$((failures + 1))
            elif [[ "$calls" -ne "$expect_calls" ]]; then
                echo "FAIL: self-test '$label': expected $expect_calls cargo invocation(s), got $calls"
                echo "  invocation log: $(cat "$fake_log")"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, $calls cargo invocation(s), as expected)"
            fi
        }

        local models_half models_full golden_stub metal_present metal_absent
        models_half="$work_dir/models_half"
        mkdir -p "$models_half"
        printf 'x' >"$models_half/$BONSAI2_PQ2_FILE"   # PTQ1_0 deliberately absent
        models_full="$work_dir/models_full"
        mkdir -p "$models_full"
        printf 'x' >"$models_full/$BONSAI2_PQ2_FILE"
        printf 'x' >"$models_full/$BONSAI2_PTQ1_FILE"
        golden_stub="$work_dir/golden_stub"
        mkdir -p "$golden_stub"
        metal_present="$work_dir/hybrid_metal_gates.rs"
        printf '// self-test stub\n' >"$metal_present"
        metal_absent="$work_dir/does-not-exist/hybrid_metal_gates.rs"

        run_stage_stub "stage1c stub: half-populated models dir refuses before any cargo call" \
            1 0 0 0 "$models_half" "$golden_stub" "$metal_present" 1
        run_stage_stub "stage1c stub: --skip-bonsai2-models never invokes cargo" \
            0 0 1 0 "$models_half" "$golden_stub" "$metal_present" 1
        run_stage_stub "stage1c stub: non-Darwin host runs nothing" \
            0 0 0 0 "$models_full" "$golden_stub" "$metal_present" 0
        run_stage_stub "stage1c stub: fully populated + metal file present runs all four legs" \
            0 4 0 0 "$models_full" "$golden_stub" "$metal_present" 1
        total=$((total + 1))
        if [[ -f "$fake_log" ]] && [[ "$(grep -o 'hybrid_forward_parity_tests\|bonsai2_engine_tests\|bonsai2_runtime_tests\|hybrid_metal_gates' "$fake_log" | tr '\n' ',')" \
            == "hybrid_forward_parity_tests,bonsai2_engine_tests,bonsai2_runtime_tests,hybrid_metal_gates," ]]; then
            echo "OK: self-test 'stage1c stub: the four legs run in the documented order (bonsai2_runtime_tests covers G6/G7/G9 in one invocation)'"
        else
            echo "FAIL: self-test 'stage1c stub: the four legs run in the documented order'"
            failures=$((failures + 1))
        fi
        run_stage_stub "stage1c stub: fully populated, metal test file absent, only three legs run" \
            0 3 0 0 "$models_full" "$golden_stub" "$metal_absent" 1
        run_stage_stub "stage1c stub: --skip-bonsai2-metal runs only the first three legs" \
            0 3 0 1 "$models_full" "$golden_stub" "$metal_present" 1
    fi

    # "legacy-models" through `--require-tests`, exercised by name (same
    # reasoning as scenario 9's bonsai2-models check): a host with only
    # Ternary-Bonsai-1.7B.gguf present writes executed=true for that model's
    # two gates but executed=false for Ternary-Bonsai-8B.gguf's and
    # Bonsai-8B.gguf's four — the exact partial-fixture shape this script's
    # REQUIRED_CAPS entry for "legacy-models" must fail on.
    printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
        '{"capability":"legacy-models","executed":true,"test":"oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":true,"test":"oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers"}' \
        >"$work_dir/legacy_models_partial.jsonl"
    check "legacy-models require-tests: only the 1.7B pair ran -> FAIL" 1 \
        "$work_dir/legacy_models_partial.jsonl" "$now" \
        "--require-tests=legacy-models:\
oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers"

    # `legacy_models_require_tests_arg`'s own list, both ways: every legacy
    # gate has an executed=true record except M-08's 20000-token YaRN gate,
    # which self-skipped. Without OXIBONSAI_M08_RUN_LONG=1 that gate is not
    # required (PASS); with it, its self-skip fails the gate; once it has
    # really run, the gate passes. A self-skipped CONVERT-EXPORT round trip
    # (no ONNX export on the host) fails either way.
    local m08_long_name="oxibonsai-runtime::legacy_parity_tests::m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens"
    local onnx_name="oxibonsai-runtime::legacy_parity_tests::onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip"
    local legacy_arg legacy_names legacy_name
    legacy_arg="$(legacy_models_require_tests_arg 0)"
    IFS=',' read -r -a legacy_names <<<"${legacy_arg#--require-tests=legacy-models:}"
    : >"$work_dir/legacy_m08_skipped.jsonl"
    : >"$work_dir/legacy_onnx_skipped.jsonl"
    for legacy_name in "${legacy_names[@]}"; do
        printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$legacy_name" \
            >>"$work_dir/legacy_m08_skipped.jsonl"
        if [[ "$legacy_name" != "$onnx_name" ]]; then
            printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$legacy_name" \
                >>"$work_dir/legacy_onnx_skipped.jsonl"
        fi
    done
    printf '{"capability":"legacy-models","executed":false,"test":"%s"}\n' "$m08_long_name" \
        >>"$work_dir/legacy_m08_skipped.jsonl"
    printf '{"capability":"legacy-models","executed":false,"test":"%s"}\n' "$onnx_name" \
        >>"$work_dir/legacy_onnx_skipped.jsonl"
    cp "$work_dir/legacy_m08_skipped.jsonl" "$work_dir/legacy_m08_ran.jsonl"
    printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$m08_long_name" \
        >>"$work_dir/legacy_m08_ran.jsonl"
    check "legacy-models: M-08 long gate self-skipped, OXIBONSAI_M08_RUN_LONG unset -> PASS" 0 \
        "$work_dir/legacy_m08_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    check "legacy-models: M-08 long gate self-skipped under OXIBONSAI_M08_RUN_LONG=1 -> FAIL" 1 \
        "$work_dir/legacy_m08_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 1)"
    check "legacy-models: M-08 long gate ran under OXIBONSAI_M08_RUN_LONG=1 -> PASS" 0 \
        "$work_dir/legacy_m08_ran.jsonl" "$now" "$(legacy_models_require_tests_arg 1)"
    check "legacy-models: CONVERT-EXPORT round trip self-skipped -> FAIL" 1 \
        "$work_dir/legacy_onnx_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"

    rm -rf "$work_dir" "$fake_bin"
    if [[ "$failures" -eq 0 ]]; then
        echo ""
        echo "OK: release-gate.sh self-test — all $total scenarios matched their expected verdict."
        return 0
    fi
    echo ""
    echo "FAILED: release-gate.sh self-test — $failures/$total scenario(s) did not match."
    return 1
}

REQUIRE_CUDA=0
SKIP_LEGACY_MODELS=0
SKIP_BONSAI2_MODELS=0
SKIP_BONSAI2_METAL=0
# M-08's 20000-token YaRN gate runs (and is then required by name) only when
# this is exactly "1" — see the usage notes at the top of this file.
M08_RUN_LONG=0
[[ "${OXIBONSAI_M08_RUN_LONG:-}" == "1" ]] && M08_RUN_LONG=1
for arg in "$@"; do
    case "$arg" in
        --require-cuda) REQUIRE_CUDA=1 ;;
        --skip-legacy-models) SKIP_LEGACY_MODELS=1 ;;
        --skip-bonsai2-models) SKIP_BONSAI2_MODELS=1 ;;
        --skip-bonsai2-metal) SKIP_BONSAI2_METAL=1 ;;
        --self-test)
            release_gate_self_test
            exit $?
            ;;
        --help|-h)
            cat <<'HELP_EOF'
Usage: release-gate.sh [--require-cuda] [--skip-legacy-models]
                        [--skip-bonsai2-models] [--skip-bonsai2-metal]
                        [--self-test]

  --require-cuda        Also require fresh "cuda" capability-manifest
                         evidence (see the capability-manifest contract at
                         the top of this file). Off by default: this
                         project treats CUDA as compile-blind on hosts with
                         no CUDA hardware, so a default run never demands
                         CUDA evidence.

  --skip-legacy-models   Opt out of the "legacy-models" capability, which
                         is otherwise REQUIRED on Darwin (FIX3-PARITY /
                         orchestrator ruling D-6(4)). "legacy-models" is
                         the evidence that the real-model CPU/NEON/Metal
                         greedy parity gate (oxibonsai-model and
                         oxibonsai-runtime's legacy_parity_tests, driven
                         against the shipped GGUFs under ./models) actually
                         ran on this host, guarding README's "byte-identical
                         output at --temperature 0" claim. Passing this flag
                         releases WITHOUT that real-model evidence — the
                         gate skips stage 1b entirely and does not add
                         "legacy-models" to the required-capability list —
                         and must be stated explicitly in the release notes
                         when used; it is never applied implicitly.

  --skip-bonsai2-models  Opt out of the "bonsai2-models" capability, which
                         is otherwise REQUIRED on Darwin (HANDOVER-INFRA,
                         same shape as --skip-legacy-models above, for the
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

  --skip-bonsai2-metal   Opt out only of the Metal-specific half of
                         "bonsai2-models" evidence: oxibonsai-model's
                         hybrid_metal_gates (CPU-vs-Metal token parity and
                         decode throughput on the real Bonsai 2 27B). The
                         non-Metal Bonsai 2 27B gates (stage 1c's first two
                         legs) are still required. Passing this flag
                         releases WITHOUT Metal 27B evidence and must be
                         stated explicitly in the release notes when used.

  --self-test            Run this script's own capability-manifest-parsing
                         self-test scenarios and exit (no release gate).

  --help, -h             Show this message and exit.

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
    echo "  -> the Metal-specific half of \"bonsai2-models\" evidence (CPU-vs-Metal"
    echo "     token parity, decode throughput) will NOT be required or collected"
    echo "     this run; state this explicitly in the release notes."
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> the Metal-specific half of \"bonsai2-models\" evidence is REQUIRED on"
    echo "     this host; pass --skip-bonsai2-metal to opt out."
fi
echo ""

# Fresh-evidence guarantee: delete any leftover manifest and record the
# moment this run started, so a stale file from a previous invocation (or
# a differently-configured one) can never be mistaken for this run's proof.
mkdir -p "$TARGET_DIR"
rm -f "$CAPABILITY_REPORT"
RUN_START_EPOCH="$(date +%s)"

# ── 1. The full gate, in strict mode ─────────────────────────────────────
# ci.sh itself also truncates the manifest right before its nextest stages
# (belt-and-braces: it must produce a clean slate whether invoked directly
# by a developer or, as here, via release-gate.sh) — both truncations
# target the same path, so this is idempotent, not a double-reset race.
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
"$SCRIPT_DIR/ci.sh" --release || {
    rc=$?
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE FAILED: scripts/ci.sh --release did not pass (exit $rc)."
    echo "See its output above for the failing stage."
    echo "═══════════════════════════════════════════════════════════════"
    exit "$rc"
}

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
        "Ternary-Bonsai-1.7B.gguf"
        "Ternary-Bonsai-8B.gguf"
        "Bonsai-8B.gguf"
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
    echo "── oxibonsai-runtime::lib (legacy-models: temperature-0 GPU path, stream stop) ──"
    OXI_MODEL="$LEGACY_MODELS_DIR_1D/Ternary-Bonsai-1.7B.gguf" \
        OXI_TOKENIZER="$LEGACY_MODELS_DIR_1D/tokenizer.json" \
        cargo test --release -p oxibonsai-runtime --all-features --lib \
        -- --test-threads=1 \
        temperature_zero_completion_takes_the_metal_greedy_gpu_path \
        real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: oxibonsai-runtime's real-model lib cases did not"
        echo "pass (exit $rc)."
        echo "═══════════════════════════════════════════════════════════════"
        exit "$rc"
    }
    echo ""
    echo "── oxibonsai-runtime::metal_greedy_cpu_fallback_tests (legacy-models: Metal↔CPU byte parity + mid-stream fallback) ──"
    OXI_MODEL="$LEGACY_MODELS_DIR_1D/Ternary-Bonsai-1.7B.gguf" \
        cargo test --release -p oxibonsai-runtime --features metal \
        --test metal_greedy_cpu_fallback_tests -- --test-threads=1 \
        real_model_greedy_gpu_fallback_byte_identical || {
        rc=$?
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "RELEASE GATE FAILED: real_model_greedy_gpu_fallback_byte_identical did"
        echo "not pass (exit $rc)."
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
            # throughput), required on Darwin unless
            # explicitly opted out — see the --skip-bonsai2-metal note above.
            BONSAI2_REQUIRED_NAMES="$BONSAI2_REQUIRED_NAMES,\
oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_matches_cpu_tokens_bonsai2,\
oxibonsai-model::hybrid_metal_gates::hybrid_real_27b_metal_decode_throughput_bonsai2"
        fi
        REQUIRED_CAPS+=("--require-tests=bonsai2-models:$BONSAI2_REQUIRED_NAMES")
    fi
fi
if [[ "$REQUIRE_CUDA" -eq 1 ]]; then
    REQUIRED_CAPS+=("cuda")
fi

if [[ "${#REQUIRED_CAPS[@]}" -eq 0 ]]; then
    echo "No hardware capability is required on this host/configuration. Skipping."
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "RELEASE GATE PASSED."
    echo "═══════════════════════════════════════════════════════════════"
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
    echo "actually execute this run (see above). This is expected today for"
    echo "any capability whose producer test has not yet adopted the"
    echo "manifest contract documented at the top of this script — see"
    echo "CI-GATE's recorded deviations for the exact per-file change."
    echo "═══════════════════════════════════════════════════════════════"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "RELEASE GATE PASSED. scripts/publish.sh is safe to run next."
echo "═══════════════════════════════════════════════════════════════"
exit 0
