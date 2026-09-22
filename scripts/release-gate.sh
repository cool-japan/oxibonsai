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
#     "image-parity", "legacy-models" (extend this list here, in the same
#     edit that adds a new gated capability, so this file stays the single
#     source of truth).
#   - "executed" (bool, required): true iff the test body actually reached
#     and ran the hardware/fixture-dependent code path this run — NOT
#     merely "the process exited 0". A test that detects the capability is
#     absent and takes the self-skip path must still write a record, with
#     "executed": false, so an all-skipped run is visibly distinct in the
#     manifest from a manifest that is simply missing (the latter means the
#     producer test was never converted to use this contract at all).
#   - "test" (string, required): the fully-qualified test name, for
#     diagnostics when a required capability's evidence is missing/stale.
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
#
# ── THE REAL-MODEL PARITY STAGE MUST BE SERIALIZED ──────────────────────────
# The six real-model parity tests (three in oxibonsai-model, three in
# oxibonsai-runtime) each load a multi-GB GGUF per kernel tier and decode 64
# greedy tokens through it. Run in parallel on an 8-core/24 GB M3 they drove
# the machine to load average 96 (measured, wave-3.5 triage). Stage 1b below
# therefore invokes them directly with `--test-threads=1`, and they carry an
# in-binary mutex ([`gpu_serial`]) because the Metal decode path is a
# process-global singleton.
#
# NOTE for whoever is granted `.config/nextest.toml` (wave3.5.md SS6: it has
# no wave-3.5 owner — CI-GATE (wave 1) and FIX-02-CI (wave 1.5), its only
# prior owners, are both closed, and this exact gap was measured but is
# OUT OF SCOPE for this file's package to fix): `scripts/ci.sh`'s
# `cargo nextest run --workspace [--all-features] --profile ci`/`--profile
# default` stages (both invoked regardless of `--release`; ci.sh's
# `--release` only affects missing-tool strictness and CUDA_ARGS, NOT the
# cargo profile of these two nextest stages, which build DEBUG) run each
# test in its OWN PROCESS, which neither `gpu_serial`'s in-binary mutex nor
# this script's `--test-threads=1` (stage 1b, above) constrains. On a host
# whose checkout has a populated `models/` (unlike the isolated worktree
# this exact gap was measured in, whose `models/` carries only a
# `.gitkeep` — that is the only reason this has not yet redded a real
# `--workspace` nextest run) those two stages will schedule the six
# real-model `legacy_parity_tests`
# gates CONCURRENTLY, in a DEBUG build, which measured a load average of 96
# on this 8-core/24 GB M3 (wave-3.5 triage). Two fixes were drafted and
# verified not to regress `cargo nextest list`'s parse, but NEITHER was
# applied here because `.config/nextest.toml` is not this package's file to
# edit (edit-only-owned-files is a hard rule this session runs under) —
# whoever is granted it should pick ONE:
#
#   (A, preferred) exclude the binary from these nextest stages entirely,
#   via `default-filter` (confirmed supported: `cargo nextest run --help`
#   lists `--ignore-default-filter` on nextest 0.9.108, the version on this
#   host) on `[profile.default]` and `[profile.ci]`:
#       default-filter = 'not binary(legacy_parity_tests)'
#   This does NOT reintroduce `#[ignore]` / D-6(4)'s hole: the tests still
#   run, fully, serialized, in release, right here in stage 1b — they are
#   just never SCHEDULED by nextest, which has no way to serialize them
#   across its own process-per-test model anyway. A developer who wants to
#   run them through nextest locally still can, explicitly, with
#   `cargo nextest run -E 'binary(legacy_parity_tests)' --ignore-default-filter`.
#   This sidesteps fix (B)'s timeout-calibration problem entirely.
#
#   (B, the alternative, if (A) is rejected for some reason) a `test-group`
#   with `max-threads = 1` over `binary(legacy_parity_tests)`, e.g.:
#       [test-groups]
#       real-model-gate = { max-threads = 1 }
#       [[profile.default.overrides]]
#       filter = 'binary(legacy_parity_tests)'
#       test-group = 'real-model-gate'
#       slow-timeout = { period = "300s", terminate-after = 8 }
#       [[profile.ci.overrides]]
#       filter = 'binary(legacy_parity_tests)'
#       test-group = 'real-model-gate'
#       slow-timeout = { period = "300s", terminate-after = 8 }
#   CAVEAT (measured, wave-3.5 verifier, not previously stated): the 300s x
#   8 = 2400s ceiling above is calibrated on the RELEASE-profile timings this
#   triage measured (776.92s for the model crate's 3 gates, 1109.20s for the
#   runtime crate's 3 gates) — but ci.sh's nextest stages build DEBUG (see
#   above), where these gates run measurably slower and 2400s is UNVALIDATED
#   and likely insufficient; measure the debug timing before trusting it.
#   TOML FOOTGUN: `.config/nextest.toml` already has bare-key
#   `[profile.default]`/`[profile.ci]`/`[profile.fast]` tables; append BOTH
#   new array-of-tables (`[[profile.default.overrides]]`,
#   `[[profile.ci.overrides]]`) and `[test-groups]` at the END of the file —
#   an array-of-tables placed mid-file silently re-parents every bare key
#   that follows it into the array's own table. `cargo nextest list` is the
#   cheap way to validate the result parses as intended.
# A capability with zero "executed": true records for this run — including
# because the manifest has zero records for it at all, e.g. because no
# producer test has adopted this contract yet — is INCOMPLETE, and this
# script fails the release gate. It is not this script's job to weaken
# that; it is the producer tests' job to start writing to the manifest
# (tracked as CI-GATE's T-05 deviation until TESTS-INFRA lands it).
#
# Usage:
#   ./scripts/release-gate.sh                       # metal + legacy-models
#                                                   # required iff macOS
#   ./scripts/release-gate.sh --require-cuda        # also require cuda evidence
#   ./scripts/release-gate.sh --skip-legacy-models  # release WITHOUT real-model
#                                                   # parity evidence (state it
#                                                   # in the release notes)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

# Parses the JSONL capability manifest at `report_path` and verifies that
# every capability named in the remaining arguments has at least one
# `"executed": true` record written no earlier than `run_start_epoch`.
# Exit 0 = every required capability has real, fresh evidence; exit 1 = at
# least one does not, OR the manifest itself is unusable (missing, stale,
# or contains a malformed line). Factored into its own function (rather
# than an inline heredoc at the call site) so `--self-test` below can
# exercise this exact parsing logic against synthetic manifests, with no
# cargo run and no hardware involved.
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
required = sys.argv[3:]

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
        if not isinstance(cap, str) or not isinstance(executed, bool):
            print(f"WARN: manifest line {lineno} missing required fields "
                  f"'capability'(str)/'executed'(bool); ignoring: {line!r}")
            malformed += 1
            continue
        executed_by_cap.setdefault(cap, []).append((executed, test))

ok = True
for cap in required:
    entries = executed_by_cap.get(cap, [])
    ran = [t for (executed, t) in entries if executed]
    if ran:
        print(f"OK: '{cap}' executed by {len(ran)} test(s), e.g. {ran[0]}")
    else:
        ok = False
        if entries:
            print(f"FAIL: '{cap}' has {len(entries)} manifest record(s) but NONE with "
                  f"executed=true (all self-skipped this run).")
        else:
            print(f"FAIL: '{cap}' has NO manifest records at all — either no producer "
                  f"test has adopted the capability-manifest contract yet, or none ran.")

if malformed:
    print(f"FAIL: {malformed} malformed manifest line(s) found — a partially-corrupted "
          f"manifest cannot be trusted as evidence for the capabilities that DID parse, "
          f"so this fails the gate even if every required capability otherwise looks "
          f"satisfied.")
    ok = False

sys.exit(0 if ok else 1)
PYEOF
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

    rm -rf "$work_dir"
    if [[ "$failures" -eq 0 ]]; then
        echo ""
        echo "OK: release-gate.sh self-test — all 8 scenarios matched their expected verdict."
        return 0
    fi
    echo ""
    echo "FAILED: release-gate.sh self-test — $failures scenario(s) did not match."
    return 1
}

REQUIRE_CUDA=0
SKIP_LEGACY_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --require-cuda) REQUIRE_CUDA=1 ;;
        --skip-legacy-models) SKIP_LEGACY_MODELS=1 ;;
        --self-test)
            release_gate_self_test
            exit $?
            ;;
        --help|-h)
            cat <<'HELP_EOF'
Usage: release-gate.sh [--require-cuda] [--skip-legacy-models] [--self-test]

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

  --self-test            Run this script's own capability-manifest-parsing
                         self-test scenarios and exit (no release gate).

  --help, -h             Show this message and exit.
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

    # Both crates, both halves: oxibonsai-model is the tokenizer-free
    # cross-tier numeric gate, oxibonsai-runtime is the golden-TEXT half
    # (scratchpad/golden_legacy/legacy_golden.json's captured Metal output,
    # through the real TokenizerBridge and the real SamplingParams).
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

# ── 2. Hardware-capability enforcement (T-05) ────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Capability manifest check"
echo "═══════════════════════════════════════════════════════════════"

REQUIRED_CAPS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    REQUIRED_CAPS+=("metal")
    # T-05's whole point: a capability that self-skipped must not read as
    # covered. "metal" alone cannot express this one — `metal_k_quant_gemv_parity`
    # writes 30+ metal/executed=true records on any Metal host regardless of
    # whether the real models were ever touched — so the real-model gate has
    # its own name. See the --skip-legacy-models note in this file's header.
    if [[ "$SKIP_LEGACY_MODELS" -eq 0 ]]; then
        REQUIRED_CAPS+=("legacy-models")
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
