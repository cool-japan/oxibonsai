#!/usr/bin/env bash
# OxiBonsai — the single, repository-local CI quality gate.
#
# T-01: CI used to be `.github/workflows/ci.yml.disabled` — a file that runs
# on no machine, automatically, ever. Per this project's own policy
# (CLAUDE.md: only `pypi-publish.yml` / `npm-publish.yml` may live in
# `.github/workflows`), the gate cannot come back as a workflow file. This
# script IS the gate. It is meant to be invoked:
#   - by hand, during development (`./scripts/ci.sh`);
#   - by the git pre-push hook installed via `./scripts/preflight.sh --install-hook`
#     (preflight.sh runs a fast subset itself, then this on request / in CI-like
#     contexts — see preflight.sh's header);
#   - by `./scripts/release-gate.sh`, which always runs this in `--release`
#     mode and layers the hardware-capability check (T-05) on top.
#
# Runs every stage below IN ORDER, failing fast (stopping) at the first
# stage that genuinely fails, and printing a clearly named
# "Stage N/TOTAL: <name>" header before each one so a failure is
# unambiguous about where it happened.
#
# Modes:
#   (default)   dev mode — a stage whose TOOL is simply not installed on this
#               machine (no cargo-nextest, no wasm32 target, no CUDA
#               toolkit, …) is reported SKIPPED and does not fail the run.
#               A stage whose tool IS present but reports a real error
#               always fails the run, in both modes.
#   --release   release mode — a missing tool is never a silent pass: the
#               stage is reported INCOMPLETE and the whole run fails
#               (non-zero exit). This is what `scripts/release-gate.sh`
#               always uses. "A missing tool must SKIP loudly ... never
#               pass silently" applies only to release mode; dev mode is
#               meant to be usable on a laptop that does not have every
#               optional toolchain installed.
#
# Usage:
#   ./scripts/ci.sh                 # dev mode, all stages
#   ./scripts/ci.sh --release       # release mode (see above)
#   ./scripts/ci.sh --only build    # run a single stage by short name (see STAGE names below), for iterating
#   ./scripts/ci.sh --list          # list stage short names and exit
#
# Every `cargo` invocation below picks up `CARGO_TARGET_DIR` /
# `CARGO_BUILD_JOBS` from the environment as usual; this script never sets
# them itself so it is safe to run several copies against isolated target
# dirs concurrently.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

RELEASE_MODE=0
ONLY_STAGE=""
LIST_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --release) RELEASE_MODE=1 ;;
        --only)    : ;; # value consumed below
        --only=*)  ONLY_STAGE="${arg#--only=}" ;;
        --list)    LIST_ONLY=1 ;;
        --help|-h)
            echo "Usage: $0 [--release] [--only=<stage>] [--list]"
            exit 0
            ;;
        *)
            if [[ "${PREV_ARG:-}" == "--only" ]]; then
                ONLY_STAGE="$arg"
            else
                echo "ERROR: unknown argument: $arg" >&2
                exit 2
            fi
            ;;
    esac
    PREV_ARG="$arg"
done

TARGET_DIR="${CARGO_TARGET_DIR:-target}"

# Producers of the capability manifest (T-05 — see release-gate.sh's header
# for the full contract) are hardware-gated *tests*, and `cargo nextest run`
# spawns each test binary with its CWD set to that test's OWN crate root
# (e.g. crates/oxibonsai-kernels), not the workspace root. A producer that
# tried to resolve `${CARGO_TARGET_DIR:-target}` itself would therefore
# write to `crates/<pkg>/target/capability-report.json` whenever
# CARGO_TARGET_DIR is unset (the common case), instead of the manifest this
# gate (and release-gate.sh) actually reads — silently producing a false
# "NO manifest records" for a capability that genuinely ran. Export the
# resolved ABSOLUTE path once, here, so every child process (nextest test
# binaries, doctests, …) can read it from the environment instead.
# Resolved by string prefix rather than `$(cd "$TARGET_DIR" && pwd)`: the
# target dir need not exist yet (e.g. on `./scripts/ci.sh --list`, or a
# fresh checkout before the first build), and `cd`-ing into a nonexistent
# directory would either fail loudly (under `-e`, which this script does
# not use) or, worse, silently resolve to the wrong directory — this
# script must not gain a directory-creating side effect just from
# computing a path.
case "$TARGET_DIR" in
    /*) ABS_TARGET_DIR="$TARGET_DIR" ;;
    *) ABS_TARGET_DIR="$PROJECT_ROOT/$TARGET_DIR" ;;
esac
export OXIBONSAI_CAPABILITY_REPORT="$ABS_TARGET_DIR/capability-report.json"
CAPABILITY_REPORT="$OXIBONSAI_CAPABILITY_REPORT"

# Canonical stage-name list, defined once and shared by --list and the
# --only validation below, so the two can never drift out of sync with
# each other (or silently disagree about what a valid stage name is).
STAGE_NAMES=(
    fmt build-default build-all-features
    facade-image facade-metal facade-server-metal-image
    clippy-all-features clippy-default
    nextest-all-features nextest-default doctests docs-build deny
    pure-rust cuda-syntax wasm-tokenizer llvm-cov tmp-hardcode-advisory
)

if [[ -n "$ONLY_STAGE" ]]; then
    KNOWN=0
    for s in "${STAGE_NAMES[@]}"; do
        [[ "$s" == "$ONLY_STAGE" ]] && { KNOWN=1; break; }
    done
    if [[ "$KNOWN" -eq 0 ]]; then
        # A typo'd --only value must not silently skip every stage and
        # report a green summary having run nothing — that is the exact
        # "reports pass while doing nothing" failure shape this whole
        # package exists to eliminate (see release-gate.sh's header for
        # the sibling bug that shape caused elsewhere in this package).
        echo "ERROR: --only '$ONLY_STAGE' is not a known stage name." >&2
        echo "Known stages:" >&2
        printf '  %s\n' "${STAGE_NAMES[@]}" >&2
        exit 2
    fi
    if [[ "$RELEASE_MODE" -eq 1 ]]; then
        # Same failure shape as the unknown-stage check above, but for a
        # *valid* --only in --release mode: `--release --only=fmt` would
        # otherwise run 1 of N stages, record the rest "SKIPPED(--only)",
        # and still print "ALL STAGES COMPLETE." with exit 0 — a release
        # gate reporting green after doing almost nothing. Neither
        # release-gate.sh nor publish.sh ever passes --only alongside
        # --release (both invoke a full `ci.sh --release`), so refusing
        # the combination here cannot break either caller.
        echo "ERROR: --only is incompatible with --release: a partial run must never" >&2
        echo "be reported through the release gate's pass/fail signal. Run the single" >&2
        echo "stage in dev mode instead, or omit --only for a full --release run." >&2
        exit 2
    fi
fi

# ── Small helpers ────────────────────────────────────────────────────────

STAGE_NUM=0
TOTAL_STAGES=${#STAGE_NAMES[@]}   # derived, not hardcoded — cannot drift when a stage is added/removed
STAGE_RESULTS=()   # "name:status" entries for the final summary

banner() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Stage $STAGE_NUM/$TOTAL_STAGES: $1"
    echo "═══════════════════════════════════════════════════════════════"
}

# Returns 0 (true) when this stage should be skipped because --only names a
# *different* stage. Silent on purpose: with --only set, a developer wants
# just that one stage's output, not 13 "skipping" banners around it. Still
# advances STAGE_NUM and records the result so the final summary and the
# Stage N/TOTAL numbering both stay accurate.
only_filter_skips() {
    local short="$1"
    if [[ -n "$ONLY_STAGE" && "$ONLY_STAGE" != "$short" ]]; then
        STAGE_RESULTS+=("$short:SKIPPED(--only)")
        return 0
    fi
    return 1
}

# Runs a stage whose tool might legitimately be absent (nextest, clippy
# component, cargo-deny, wasm target, cargo-llvm-cov, a CUDA toolchain).
#   record_stage <short-name> <tool-presence-check-exit-status> <install-hint> -- <cmd...>
# `tool_present` must already have been computed by the caller (0 = present).
run_optional_stage() {
    local short="$1" tool_present="$2" install_hint="$3"
    shift 3
    STAGE_NUM=$((STAGE_NUM + 1))
    only_filter_skips "$short" && return 0
    banner "$short"
    if [[ "$tool_present" -ne 0 ]]; then
        if [[ "$RELEASE_MODE" -eq 1 ]]; then
            echo "INCOMPLETE: required tool is not installed. $install_hint"
            STAGE_RESULTS+=("$short:INCOMPLETE")
            echo ""
            echo "FATAL: stage '$short' is INCOMPLETE in --release mode. Aborting."
            print_summary
            exit 2
        else
            echo "SKIPPED: tool not installed (dev mode). $install_hint"
            STAGE_RESULTS+=("$short:SKIPPED")
            return 0
        fi
    fi
    echo "+ $*"
    if "$@"; then
        echo "OK: $short"
        STAGE_RESULTS+=("$short:OK")
        return 0
    else
        local rc=$?
        echo "FAILED ($rc): $short"
        STAGE_RESULTS+=("$short:FAILED")
        print_summary
        exit "$rc"
    fi
}

# Runs a mandatory stage (no missing-tool leniency — cargo itself, or a
# script we ship, must simply work).
run_required_stage() {
    local short="$1"
    shift
    STAGE_NUM=$((STAGE_NUM + 1))
    only_filter_skips "$short" && return 0
    banner "$short"
    echo "+ $*"
    if "$@"; then
        echo "OK: $short"
        STAGE_RESULTS+=("$short:OK")
        return 0
    else
        local rc=$?
        echo "FAILED ($rc): $short"
        STAGE_RESULTS+=("$short:FAILED")
        print_summary
        exit "$rc"
    fi
}

# Runs an advisory stage: prints its own PASS/WARN, never fails the gate.
run_advisory_stage() {
    local short="$1"
    shift
    STAGE_NUM=$((STAGE_NUM + 1))
    only_filter_skips "$short" && return 0
    banner "$short (advisory — does not fail the gate)"
    if "$@"; then
        STAGE_RESULTS+=("$short:OK")
    else
        STAGE_RESULTS+=("$short:WARN")
    fi
    return 0
}

print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Summary"
    echo "═══════════════════════════════════════════════════════════════"
    for entry in "${STAGE_RESULTS[@]+"${STAGE_RESULTS[@]}"}"; do
        printf '  %-45s %s\n' "${entry%%:*}" "${entry#*:}"
    done
    echo "  mode: $([[ "$RELEASE_MODE" -eq 1 ]] && echo release || echo dev)"
}

tool_ok() { command -v "$1" >/dev/null 2>&1; echo $?; }

if [[ "$LIST_ONLY" -eq 1 ]]; then
    printf '%s\n' "${STAGE_NAMES[@]}"
    exit 0
fi

echo "OxiBonsai CI gate — mode: $([[ "$RELEASE_MODE" -eq 1 ]] && echo release || echo dev)"
echo "project root: $PROJECT_ROOT"
echo "CARGO_TARGET_DIR: ${CARGO_TARGET_DIR:-<unset, defaults to ./target>}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "FATAL: cargo not found on PATH. Install Rust (https://rustup.rs) first." >&2
    exit 2
fi

# ── Stage 0: cargo fmt ───────────────────────────────────────────────────
# Carried over from the pre-existing scripts/publish.sh CI checks (step
# 1/6 there) so that delegating publish.sh's gate to this script does not
# silently drop the formatting check.
FMT_PRESENT="$(tool_ok cargo-fmt)"
run_optional_stage "fmt" "$FMT_PRESENT" \
    "install the rustfmt component: rustup component add rustfmt" \
    cargo fmt --all -- --check

# ── Stage 1: default-feature build (T-11) ───────────────────────────────
run_required_stage "build-default" cargo build --workspace

# ── Stage 2: all-features build ─────────────────────────────────────────
run_required_stage "build-all-features" cargo build --workspace --all-features

# ── Stage 2b: oxibonsai facade per-feature checks (T-12/RAG-EVAL-IMG-17/35) ──
# `build-all-features` above proves the *union* of every feature compiles;
# it cannot prove any individual combination does, because a stray
# `#[cfg(feature = "x")]` block can silently depend on some other feature
# always being present alongside it in an all-on build and never get
# noticed. These three checks are exactly the combinations RAG-EVAL-IMG-17/35
# added GPU/image coverage for and that `--all-features`/the default build
# do not exercise on their own: `image` alone, `metal` alone, and their
# real-world union with `server` (the combination
# `tests/feature_matrix.rs`'s `server_metal_image_union_composes` test
# exercises at the API level; these three are the mechanical per-feature
# compile check underneath it — deliberately three explicit stages rather
# than a `cargo hack --each-feature` dependency, since `cargo-hack` is not
# already an installed/documented tool for this repo (`rg 'cargo-hack|cargo
# hack'` finds no such usage anywhere outside this comment and
# `tests/feature_matrix.rs`'s own doc comment), and this package's spec
# says explicitly: "do not add a new tool requirement silently").
run_required_stage "facade-image" cargo check -p oxibonsai --features image
run_required_stage "facade-metal" cargo check -p oxibonsai --features metal
run_required_stage "facade-server-metal-image" \
    cargo check -p oxibonsai --features server,metal,image

# ── Stage 3: clippy, all features ───────────────────────────────────────
CLIPPY_PRESENT="$(tool_ok cargo-clippy)"
run_optional_stage "clippy-all-features" "$CLIPPY_PRESENT" \
    "install the clippy component: rustup component add clippy" \
    cargo clippy --workspace --all-features --all-targets -- -D warnings

# ── Stage 4: clippy, default features (T-11) ────────────────────────────
run_optional_stage "clippy-default" "$CLIPPY_PRESENT" \
    "install the clippy component: rustup component add clippy" \
    cargo clippy --workspace --all-targets -- -D warnings

# ── Stage 5/6 prep: fresh capability manifest (T-05) ────────────────────
# Hardware-gated tests (metal/cuda/rag-real-generation/image-parity) are
# meant to append one JSON-line record per execution to this file — see
# scripts/release-gate.sh's header comment for the exact contract. Truncate
# it before the nextest stages so a stale manifest from a previous run (or
# a different feature set) can never be read as "this run's" evidence.
#
# Guarded by the same condition that decides whether either nextest stage
# will actually run below: an unguarded truncation here means
# `ci.sh --only=<anything, even a stage with nothing to do with nextest>`
# silently wipes a previous run's manifest without producing a new one —
# exactly the kind of evidence-destroying side effect a partial run must
# not have.
if [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "nextest-all-features" || "$ONLY_STAGE" == "nextest-default" ]]; then
    mkdir -p "$TARGET_DIR"
    : >"$CAPABILITY_REPORT"
fi

# ── Stage 5: nextest, all features ──────────────────────────────────────
NEXTEST_PRESENT="$(tool_ok cargo-nextest)"
run_optional_stage "nextest-all-features" "$NEXTEST_PRESENT" \
    "install nextest: cargo install cargo-nextest --locked" \
    cargo nextest run --workspace --all-features --profile ci

# ── Stage 6: nextest, default features (T-11) ───────────────────────────
run_optional_stage "nextest-default" "$NEXTEST_PRESENT" \
    "install nextest: cargo install cargo-nextest --locked" \
    cargo nextest run --workspace --profile ci

# ── Stage 7: doctests (T-02 — nextest cannot run these at all) ──────────
run_required_stage "doctests" cargo test --doc --workspace --all-features

# ── Stage 7b: docs build ─────────────────────────────────────────────────
# Carried over from the pre-existing scripts/publish.sh CI checks (step
# 5/6 there — "cargo doc"), so that delegating publish.sh's gate to this
# script does not silently drop the docs-build check crates.io itself
# relies on. Not part of the T-01 12-stage list verbatim, same reasoning
# as the fmt stage above.
run_required_stage "docs-build" cargo doc --workspace --all-features --no-deps

# ── Stage 8: cargo-deny ─────────────────────────────────────────────────
DENY_PRESENT="$(tool_ok cargo-deny)"
run_optional_stage "deny" "$DENY_PRESENT" \
    "install cargo-deny: cargo install cargo-deny --locked" \
    cargo deny check bans licenses sources advisories

# ── Stage 9: Pure Rust policy (deps-01/03) ──────────────────────────────
run_required_stage "pure-rust" "$SCRIPT_DIR/check_pure_rust.sh"

# ── Stage 10: CUDA kernel-source syntax (F12) ───────────────────────────
STAGE_NUM=$((STAGE_NUM + 1))
if only_filter_skips "cuda-syntax"; then
    :
else
    banner "cuda-syntax"
    CUDA_ARGS=()
    [[ "$RELEASE_MODE" -eq 1 ]] && CUDA_ARGS+=(--release)
    # ${arr[@]+"${arr[@]}"}, not a bare "${arr[@]}": macOS's default
    # /bin/bash (3.2) treats an empty array's "${arr[@]}" as unbound under
    # `set -u`. CUDA_ARGS is empty in dev mode (the common case), so this
    # is not a hypothetical.
    if "$SCRIPT_DIR/check_cuda.sh" "${CUDA_ARGS[@]+"${CUDA_ARGS[@]}"}"; then
        STAGE_RESULTS+=("cuda-syntax:OK")
    else
        rc=$?
        if [[ "$rc" -eq 2 ]]; then
            # INCOMPLETE — check_cuda.sh only returns this in --release mode
            # (dev mode always exits 0 even when skipped/approximate).
            echo "FAILED (incomplete, $rc): cuda-syntax"
            STAGE_RESULTS+=("cuda-syntax:INCOMPLETE")
            print_summary
            exit "$rc"
        fi
        echo "FAILED ($rc): cuda-syntax"
        STAGE_RESULTS+=("cuda-syntax:FAILED")
        print_summary
        exit "$rc"
    fi
fi

# ── Stage 11: wasm32 build for oxibonsai-tokenizer (TOK-11) ─────────────
WASM_PRESENT=1
if rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
    WASM_PRESENT=0
elif rustc --print target-list 2>/dev/null | grep -q '^wasm32-unknown-unknown$' \
    && rustc --target wasm32-unknown-unknown --print cfg >/dev/null 2>&1; then
    # Toolchains without rustup (e.g. a distro-packaged rustc) can still
    # have the target installed; probe rustc directly as a fallback.
    WASM_PRESENT=0
fi
run_optional_stage "wasm-tokenizer" "$WASM_PRESENT" \
    "install the target: rustup target add wasm32-unknown-unknown" \
    cargo build --target wasm32-unknown-unknown -p oxibonsai-tokenizer

# ── Stage 12: coverage baseline (T-18 — record it, don't just print it) ──
# T-18's fix asks for a *recorded* baseline, not merely a number that
# scrolls off the terminal: tee the exact same `--summary-only` output that
# used to be the whole stage into $TARGET_DIR/coverage-baseline.txt so a
# later run (or a human) can diff today's numbers against it. `run_optional_stage`
# invokes its trailing argv directly (`"$@"`, no shell), so a `|` cannot be
# embedded there — wrap the pipeline in a function instead.
# shellcheck disable=SC2329  # invoked indirectly via run_optional_stage "$@"
llvm_cov_stage() {
    mkdir -p "$TARGET_DIR"
    echo "+ cargo llvm-cov --workspace --summary-only | tee $TARGET_DIR/coverage-baseline.txt"
    # This script runs under `set -uo pipefail` (no `-e`): pipefail is what
    # makes the pipeline's exit status `cargo llvm-cov`'s real one instead
    # of `tee`'s, so a genuine coverage-run failure still fails this stage.
    cargo llvm-cov --workspace --summary-only | tee "$TARGET_DIR/coverage-baseline.txt"
}
LLVM_COV_PRESENT="$(tool_ok cargo-llvm-cov)"
run_optional_stage "llvm-cov" "$LLVM_COV_PRESENT" \
    "install cargo-llvm-cov: cargo install cargo-llvm-cov --locked" \
    llvm_cov_stage

# ── Stage 13: repo-wide hardcoded /tmp advisory (deps-08 follow-up) ─────
# NON-FATAL, for now. deps-08's CLI-side fix (src/cli/cmd_image.rs,
# args.rs, examples/mlx_image_parity.rs — owned by other packages, not this
# one) has not necessarily landed yet, so this stays advisory rather than
# gating; it exists to keep the remaining hardcoded-/tmp sites visible
# until it does.
#
# FLIP-TO-FATAL TRIGGER: once src/cli/cmd_image.rs and src/cli/args.rs no
# longer default any path to `/tmp` (deps-08's CLI half, routed to
# CLI-CORE, wave 2) and a repo-wide `hits` run comes back empty, change the
# call below from `run_advisory_stage` to `run_required_stage` (or make
# this function's `return 1` fail the gate some other way) so a future
# regression is caught, not just displayed. Do not flip it before that —
# the CLI half's hits are real, expected, and not this package's to fix.
#
# There used to be a second `grep -v` here excluding
# scripts/{benchmark,cli_ternary,download_ternary,bench_ternary}.sh on the
# theory that CI-GATE had already fixed those four scripts' `/tmp` usage.
# That filter is now dead weight, not a real exclusion: `rg -c '/tmp/'
# scripts/benchmark.sh scripts/cli_ternary.sh scripts/download_ternary.sh
# scripts/bench_ternary.sh` returns zero for all four, so the filter never
# matches anything — it can only ever hide a *regression* reintroduced into
# exactly the files wave 1 already cleaned up, which is the opposite of
# what an advisory grep gate is for. Removed; if one of those four scripts
# ever needs `/tmp` again, honouring `${TMPDIR:-/tmp}` (already excluded
# below) is the correct fix, not a per-file carve-out.
# shellcheck disable=SC2329  # invoked indirectly via run_advisory_stage "$@"
tmp_hardcode_advisory() {
    local hits
    hits="$(grep -rn '/tmp/' --include='*.rs' --include='*.sh' \
        --exclude-dir=target --exclude-dir=.git \
        -- src crates scripts 2>/dev/null \
        | grep -v -E '\$\{TMPDIR:-/tmp\}' \
        || true)"
    if [[ -n "$hits" ]]; then
        echo "WARN: hardcoded /tmp found outside \${TMPDIR:-/tmp} usage (deps-08):"
        echo "$hits" | while IFS= read -r line; do echo "  $line"; done
        return 1
    fi
    echo "OK: no hardcoded /tmp found outside \${TMPDIR:-/tmp} usage."
    return 0
}
run_advisory_stage "tmp-hardcode-advisory" tmp_hardcode_advisory

# ── Done ─────────────────────────────────────────────────────────────────
print_summary
echo ""
echo "ALL STAGES COMPLETE."
exit 0
