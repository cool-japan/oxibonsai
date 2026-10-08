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
#   --accept-approximate-cuda-syntax
#               the one explicit opt-out of that rule, for a release host
#               without the CUDA toolkit (every macOS host): in --release
#               mode the `cuda-syntax` stage then accepts a CLEAN approximate
#               check (clang++/g++ parse with CUDA builtins stubbed; see
#               scripts/check_cuda.sh) instead of requiring nvcc. The stage's
#               Summary line reads `OK (approximate, accepted by flag)` and a
#               waiver line follows the table, so the run can never be read
#               as a full CUDA check. A syntax error still fails, and a host
#               with no C++ compiler at all is still INCOMPLETE. Without the
#               flag the stage is exactly the strict one described above; in
#               dev mode the flag is accepted and changes nothing.
#
# --with-models: two more stages (`real-model-legacy`, `real-model-bonsai2`,
#   SKIPPED by default without it) additionally run the real-model gates
#   this workspace ships (`legacy_parity_tests`/`hybrid_forward_parity_tests`/
#   `bonsai2_engine_tests`/`bonsai2_runtime_tests`) directly, once each,
#   `--release --test-threads=1`.
#   This script is not itself the serialization boundary those gates need
#   across MULTIPLE real-model binaries — `scripts/release-gate.sh`'s
#   stages 1b-1d are (they run one real-model binary completely before
#   starting the next) — so `--with-models` is the "I want to see the
#   real-model gates run as part of an ordinary CI pass" convenience for a
#   single unattended host, not a substitute for the release gate's own
#   stronger sequencing guarantees.
#
# Usage:
#   ./scripts/ci.sh                 # dev mode, all stages
#   ./scripts/ci.sh --release       # release mode (see above)
#   ./scripts/ci.sh --release --accept-approximate-cuda-syntax
#                                   # release mode on a host without the CUDA toolkit (see above)
#   ./scripts/ci.sh --with-models   # also run the real-model gates (see above)
#   ./scripts/ci.sh --only build    # run a single stage by short name (see STAGE names below), for iterating
#   ./scripts/ci.sh --list          # list stage short names and exit
#
# `--only` is refused together with `--release` (a partial run must never be
# reported through the release pass/fail signal), with ONE exception:
# `--release --only cuda-syntax` runs just that stage, whose release verdict is
# decided entirely by scripts/check_cuda.sh and depends on no other stage, so it
# is the one single-stage release run that is a faithful probe (it is how the
# `cuda-syntax` waiver is checked on its own). Such a run ends with a PARTIAL RUN
# line instead of ALL STAGES COMPLETE. and is never a release verdict.
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
WITH_MODELS=0
ACCEPT_APPROX_CUDA=0
# Set once the cuda-syntax stage passed ONLY approximately and the flag accepted it.
CUDA_WAIVER_USED=0
for arg in "$@"; do
    case "$arg" in
        --release) RELEASE_MODE=1 ;;
        --with-models) WITH_MODELS=1 ;;
        --accept-approximate-cuda-syntax) ACCEPT_APPROX_CUDA=1 ;;
        --only)    : ;; # value consumed below
        --only=*)  ONLY_STAGE="${arg#--only=}" ;;
        --list)    LIST_ONLY=1 ;;
        --help|-h)
            echo "Usage: $0 [--release] [--accept-approximate-cuda-syntax] [--with-models] [--only=<stage>] [--list]"
            echo ""
            echo "  --release                        strict mode: a missing tool is INCOMPLETE and fails the run"
            echo "  --accept-approximate-cuda-syntax with --release and no nvcc: accept a clean approximate"
            echo "                                   CUDA kernel-syntax check (labelled in the Summary and after"
            echo "                                   it; never the default; no effect without --release)"
            echo "  --with-models                    also run the real-model stages"
            echo "  --only <stage>                   run one stage (refused with --release, except cuda-syntax)"
            echo "  --list                           list the stage names and exit"
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
    nextest-all-features nextest-default doctests docs-build docs-strict deny
    pure-rust cuda-syntax wasm-tokenizer llvm-cov tmp-hardcode-advisory
    real-model-legacy real-model-bonsai2
)

if [[ -n "$ONLY_STAGE" ]]; then
    KNOWN=0
    for s in "${STAGE_NAMES[@]}"; do
        [[ "$s" == "$ONLY_STAGE" ]] && { KNOWN=1; break; }
    done
    if [[ "$KNOWN" -eq 0 ]]; then
        # A typo'd --only value must not silently skip every stage and
        # report a green summary having run nothing — that is the exact
        # "reports pass while doing nothing" failure shape this gate
        # exists to eliminate (see release-gate.sh's header for the
        # sibling bug that shape caused there).
        echo "ERROR: --only '$ONLY_STAGE' is not a known stage name." >&2
        echo "Known stages:" >&2
        printf '  %s\n' "${STAGE_NAMES[@]}" >&2
        exit 2
    fi
    if [[ "$RELEASE_MODE" -eq 1 && "$ONLY_STAGE" != "cuda-syntax" ]]; then
        # Same failure shape as the unknown-stage check above, but for a
        # *valid* --only in --release mode: `--release --only=fmt` would
        # otherwise run 1 of N stages, record the rest "SKIPPED(--only)",
        # and still print "ALL STAGES COMPLETE." with exit 0 — a release
        # gate reporting green after doing almost nothing. Neither
        # release-gate.sh nor publish.sh ever passes --only alongside
        # --release (both invoke a full `ci.sh --release`), so refusing
        # the combination here cannot break either caller.
        #
        # The one exception is `cuda-syntax`, which stays allowed: its
        # release verdict (an nvcc pass, a waived approximate pass, or
        # INCOMPLETE) is decided entirely by scripts/check_cuda.sh and
        # depends on no other stage, so running it alone is a faithful
        # probe of exactly that verdict — the check of the
        # --accept-approximate-cuda-syntax waiver relies on it. Such a run
        # is labelled PARTIAL at its end (never "ALL STAGES COMPLETE.").
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

# Runs a stage that only executes with `--with-models`, in EITHER dev or
# release mode: unlike `run_optional_stage`'s "missing tool" trichotomy,
# skipping this stage is never a release-mode failure — real-model GGUFs
# are gitignored working-tree data, not an installable tool, and
# `scripts/release-gate.sh` (which always invokes `ci.sh --release` without
# `--with-models`) is what actually enforces they ran with real evidence
# before a release, through its own stages 1b-1d. Without `--with-models`
# this always reports SKIPPED and returns 0, regardless of `$RELEASE_MODE`.
run_with_models_stage() {
    local short="$1" fn="$2"
    STAGE_NUM=$((STAGE_NUM + 1))
    only_filter_skips "$short" && return 0
    banner "$short"
    if [[ "$WITH_MODELS" -ne 1 ]]; then
        echo "SKIPPED: --with-models was not passed."
        STAGE_RESULTS+=("$short:SKIPPED")
        return 0
    fi
    echo "+ $fn"
    if "$fn"; then
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
    echo "  mode: $([[ "$RELEASE_MODE" -eq 1 ]] && echo release || echo dev)$([[ -n "$ONLY_STAGE" && "$RELEASE_MODE" -eq 1 ]] && echo " (PARTIAL run, --only)")"
    if [[ "$CUDA_WAIVER_USED" -eq 1 ]]; then
        echo "  WAIVER: cuda-syntax was checked APPROXIMATELY (no nvcc), accepted by --accept-approximate-cuda-syntax."
        echo "          The CUDA backend is not certified by this run."
    fi
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
# `tests/feature_matrix.rs`'s own doc comment), and a gate must not add a
# new tool requirement silently).
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
# `env -u OXIBONSAI_M08_RUN_LONG`: M-08's opt-in 20000-token YaRN gate (two
# real Bonsai-8B decodes of roughly an hour each) belongs to the serialized,
# release-profile real-model stage of `scripts/release-gate.sh` (stage 1b),
# never to these parallel nextest runs — with the variable unset here it
# self-skips in well under a second, as in any ordinary run.
NEXTEST_PRESENT="$(tool_ok cargo-nextest)"
run_optional_stage "nextest-all-features" "$NEXTEST_PRESENT" \
    "install nextest: cargo install cargo-nextest --locked" \
    env -u OXIBONSAI_M08_RUN_LONG cargo nextest run --workspace --all-features --profile ci

# ── Stage 6: nextest, default features (T-11) ───────────────────────────
run_optional_stage "nextest-default" "$NEXTEST_PRESENT" \
    "install nextest: cargo install cargo-nextest --locked" \
    env -u OXIBONSAI_M08_RUN_LONG cargo nextest run --workspace --profile ci

# ── Stage 7: doctests (T-02 — nextest cannot run these at all) ──────────
run_required_stage "doctests" cargo test --doc --workspace --all-features

# ── Stage 7b: docs build ─────────────────────────────────────────────────
# Carried over from the pre-existing scripts/publish.sh CI checks (step
# 5/6 there — "cargo doc"), so that delegating publish.sh's gate to this
# script does not silently drop the docs-build check crates.io itself
# relies on. Not part of the T-01 12-stage list verbatim, same reasoning
# as the fmt stage above.
run_required_stage "docs-build" cargo doc --workspace --all-features --no-deps

# ── Stage 7c: docs build, rustdoc warnings fail-closed ──────────────────
# `docs-build` above only fails on a hard rustdoc ERROR (broken intra-doc
# links, invalid syntax); a rustdoc WARNING (e.g. a bare URL that should be
# a markdown link) is silent there. This stage re-runs the same build with
# `RUSTDOCFLAGS="-D warnings"`, so a warning fails the gate exactly like a
# clippy warning does. Cargo's own build cache means this rebuilds only the
# doc pass, not the whole workspace, when stage 7b already ran.
# shellcheck disable=SC2329  # invoked indirectly via run_required_stage "$@"
docs_strict_stage() {
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
}
run_required_stage "docs-strict" docs_strict_stage

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
    if [[ "$RELEASE_MODE" -eq 1 ]]; then
        CUDA_ARGS+=(--release)
        [[ "$ACCEPT_APPROX_CUDA" -eq 1 ]] && CUDA_ARGS+=(--accept-approximate)
    elif [[ "$ACCEPT_APPROX_CUDA" -eq 1 ]]; then
        echo "NOTE: --accept-approximate-cuda-syntax has no effect without --release"
        echo "(dev mode never fails this stage for a missing nvcc)."
    fi
    # The stage's label comes from check_cuda.sh's own machine-readable verdict
    # line (CUDA_SYNTAX_RESULT=...), never from its exit code alone: exit 0 in
    # --release mode means "a real nvcc pass" OR "a waived approximate pass", and
    # the Summary must say which. The output is teed so it still streams live.
    CUDA_OUT_LOG="$(mktemp "${TMPDIR:-/tmp}/oxibonsai_ci_cuda_syntax.XXXXXX")" || {
        echo "FATAL: could not create a scratch file for the cuda-syntax verdict." >&2
        exit 2
    }
    # ${arr[@]+"${arr[@]}"}, not a bare "${arr[@]}": macOS's default
    # /bin/bash (3.2) treats an empty array's "${arr[@]}" as unbound under
    # `set -u`. CUDA_ARGS is empty in dev mode (the common case), so this
    # is not a hypothetical. `pipefail` is set, so PIPESTATUS[0] — the
    # checker's own exit status, read before anything else runs — is what counts.
    "$SCRIPT_DIR/check_cuda.sh" "${CUDA_ARGS[@]+"${CUDA_ARGS[@]}"}" | tee "$CUDA_OUT_LOG"
    rc=${PIPESTATUS[0]}
    CUDA_RESULT=""
    while IFS= read -r cuda_line; do
        case "$cuda_line" in
            CUDA_SYNTAX_RESULT=*) CUDA_RESULT="${cuda_line#CUDA_SYNTAX_RESULT=}" ;;
        esac
    done <"$CUDA_OUT_LOG"
    rm -f "$CUDA_OUT_LOG"
    if [[ "$rc" -eq 0 ]]; then
        if [[ "$RELEASE_MODE" -ne 1 || "$CUDA_RESULT" == "nvcc-ok" ]]; then
            STAGE_RESULTS+=("cuda-syntax:OK")
        elif [[ "$CUDA_RESULT" == "approximate-accepted" ]]; then
            CUDA_WAIVER_USED=1
            echo "WAIVER IN EFFECT: cuda-syntax passed only APPROXIMATELY (no nvcc), because"
            echo "--accept-approximate-cuda-syntax was given. The CUDA backend is not certified by this run."
            STAGE_RESULTS+=("cuda-syntax:OK (approximate, accepted by flag)")
        else
            # check_cuda.sh exited 0 in --release mode without saying which tier
            # passed: its contract is broken, and a pass that cannot be labelled
            # must not be recorded as OK.
            echo "FAILED (incomplete): cuda-syntax exited 0 in --release mode without a recognised"
            echo "CUDA_SYNTAX_RESULT line (got '${CUDA_RESULT:-<none>}'); refusing to record an unlabelled pass."
            STAGE_RESULTS+=("cuda-syntax:INCOMPLETE")
            print_summary
            exit 2
        fi
    else
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

# ── Stage 11: wasm32 checks (TOK-11) ─────────────────────────────────────
# The stage keeps its name and is two commands, both for
# `wasm32-unknown-unknown`, run in this order, the stage failing on the first
# that fails:
#   1. `cargo build -p oxibonsai-tokenizer`: the tokenizer builds for the
#      target with its default features.
#   2. `cargo check -p oxibonsai-runtime -p oxibonsai-tokenizer
#      --no-default-features` with every warning denied: the runtime (its
#      `server` feature off) and the tokenizer check clean for the target.
#      Nothing else compiles the runtime for wasm32, and its
#      `cfg(target_arch = "wasm32")` code is invisible to the host builds,
#      clippy and the tests — where a stray unused import or a function that
#      lost its last caller on that target hides until someone checks by hand.
# The warnings are denied through `RUSTFLAGS=-D warnings` on that one command,
# the way the docs-strict stage denies rustdoc's through RUSTDOCFLAGS: the
# value is the caller's `RUSTFLAGS` (if any) plus the flag, set for that
# command only. With an explicit `--target` the flag reaches the wasm32
# artifacts and not the host's build scripts and proc macros, and, being part
# of cargo's fingerprint, it only ever rebuilds wasm32 artifacts.
# `CARGO_ENCODED_RUSTFLAGS`, which cargo prefers to RUSTFLAGS and which would
# so drop the denial without a word, is unset for it. A warning in any
# workspace crate the two packages pull in fails the stage; cargo caps the
# lints of registry dependencies, so theirs never do.
WASM_PRESENT=1
if rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
    WASM_PRESENT=0
elif rustc --print target-list 2>/dev/null | grep -q '^wasm32-unknown-unknown$' \
    && rustc --target wasm32-unknown-unknown --print cfg >/dev/null 2>&1; then
    # Toolchains without rustup (e.g. a distro-packaged rustc) can still
    # have the target installed; probe rustc directly as a fallback.
    WASM_PRESENT=0
fi
# shellcheck disable=SC2329  # invoked indirectly via run_optional_stage "$@"
wasm_stage() {
    echo "+ cargo build --target wasm32-unknown-unknown -p oxibonsai-tokenizer"
    cargo build --target wasm32-unknown-unknown -p oxibonsai-tokenizer || return $?
    echo "+ RUSTFLAGS='${RUSTFLAGS:+$RUSTFLAGS }-D warnings' cargo check -p oxibonsai-runtime -p oxibonsai-tokenizer --target wasm32-unknown-unknown --no-default-features"
    env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" \
        cargo check -p oxibonsai-runtime -p oxibonsai-tokenizer \
        --target wasm32-unknown-unknown --no-default-features
}
run_optional_stage "wasm-tokenizer" "$WASM_PRESENT" \
    "install the target (the tokenizer build and the runtime check both need it): rustup target add wasm32-unknown-unknown" \
    wasm_stage

# ── Stage 12: coverage baseline (T-18 — record it, don't just print it) ──
# T-18's fix asks for a *recorded* baseline, not merely a number that
# scrolls off the terminal: tee the `--summary-only` output into
# $TARGET_DIR/coverage-baseline.txt so a later run (or a human) can diff
# today's numbers against it. There is no coverage *threshold* yet (a floor
# set against today's numbers would institutionalise today's blind spots),
# so the stage fails only when a test fails, never on a percentage.
#
# What the stage measures: the coverage of the tests that need no real model
# file, run instrumented. It is HERMETIC with respect to real models — by
# construction, and it says so before it runs — so its figures do not depend on
# whether `models/` holds real models on the host. How the real models behave
# is evidenced by the nextest stages (5 and 6, uninstrumented) and by the
# release gate's serialised real-model legs, never by this stage: under
# coverage counters a real-model test takes many times the minutes it needs
# without them, and its timing assertions (the batched prefill must outrun the
# sequential one, ...) measure the instrumentation, not the code.
#
# Every way a test finds a real model file is closed for the run:
#   - the variables that point a test, or the code it drives, at a model,
#     tokenizer, projector or weight file are unset (COVERAGE_MODEL_VARS),
#     OXI_REQUIRE_MODEL_FILES among them: it turns a missing model into a
#     failure, and none can be found;
#   - OXIBONSAI_MODELS_DIR names a directory that does not exist (the stage
#     removes anything at that path first), so the `<workspace>/models` default
#     of `oxibonsai_testkit::workspace` is never consulted and every
#     `find_model` comes back empty. It is a path that does not exist, not an
#     empty directory, because the tests that read the variable through
#     `test_fixtures::env_path` skip on a missing path but treat an existing
#     directory as the real-model directory and fail when it holds no `.gguf`
#     (`validate_and_run_agree_on_every_real_model`);
#   - no test reaches a model past that variable: every lookup goes through
#     `oxibonsai_testkit::workspace::{models_dir, find_model, find_model_as_named}`
#     or reads
#     OXIBONSAI_MODELS_DIR itself before it falls back to the workspace's
#     `models/`. A test that built a model path from its crate's compile-time
#     `CARGO_MANIFEST_DIR` alone would be out of reach of every variable and
#     would still open a populated `models/` here — do not write one. (The
#     tests that read `models/tokenizer.json` that way load a tokenizer, not
#     a model.)
# With every route closed, every real-model test self-skips in well under a
# second.
#
# What the stage records is kept apart from the release evidence:
# OXIBONSAI_CAPABILITY_REPORT names a file of its own under the target dir,
# truncated when the stage starts. The skip records of the self-skipping
# real-model tests, and any record a test writes under instrumentation,
# therefore never enter the manifest that release-gate.sh checks, and no
# coverage run can satisfy a required capability. The variable is set for the
# one command (`env`), so no other stage sees it and nothing here truncates or
# redirects the manifest of the others; the stage also fails if that manifest
# changes while it runs.
#
# The tests run under `cargo llvm-cov nextest` with the same `ci` profile as
# stages 5 and 6, not under plain `cargo llvm-cov` (which drives `cargo test`):
#   - nextest runs every test in its own process, as every other test stage
#     does. Under `cargo test` all tests of a binary share one instrumented
#     process, whose coverage counters are contended by every concurrent test
#     thread, so a test sees a much slower machine under coverage than outside
#     it; that distorted timing-sensitive tests (the deadline tests of
#     `crates/oxibonsai-runtime/tests/server_hardening_round4.rs` failed about
#     half the runs that way before they were made deterministic).
#   - the `ci` profile's `default-filter` excludes the full-weight 27B tests
#     and its `real-model-gate` group keeps the other real-model tests to one
#     at a time, as in stages 5 and 6; `cargo test` has neither.
# `run_optional_stage` invokes its trailing argv directly (`"$@"`, no shell), so
# the pipeline lives in a function; `pipefail` (set at the top of this script)
# makes the pipeline's exit status `cargo llvm-cov`'s real one instead of
# `tee`'s.
#
# The variables that make a test find, or insist on finding, a real model.
# `OXIBONSAI_M08_RUN_LONG` (the 20 000-token real-model gate) is unset for the
# reason stage 5 gives. Add a variable here in the edit that introduces one.
COVERAGE_MODEL_VARS=(
    OXI_MODEL OXI_TOKENIZER
    OXI_BONSAI2_PQ2_GGUF OXI_BONSAI2_PTQ1_GGUF OXI_BONSAI2_MMPROJ_GGUF
    OXIBONSAI_MODEL_PATH OXIBONSAI_TOKENIZER_PATH OXIBONSAI_BONSAI2_HEADERS_DIR
    OXI_DIT_GGUF OXIBONSAI_DIT_GGUF OXI_VAE_WEIGHTS OXI_VAE_SAFETENSORS
    OXI_TE_WEIGHTS OXI_TE_4BIT OXI_TE_TOKENIZER_DIR
    OXI_REQUIRE_MODEL_FILES OXIBONSAI_M08_RUN_LONG
)
# shellcheck disable=SC2329  # invoked indirectly via run_optional_stage "$@"
llvm_cov_stage() {
    local no_models="$ABS_TARGET_DIR/coverage-no-models"
    local capability_report="$ABS_TARGET_DIR/coverage-capability-report.json"
    local started_marker="$ABS_TARGET_DIR/coverage-stage-started"
    local unset_args=() var rc records executed
    for var in "${COVERAGE_MODEL_VARS[@]}"; do
        unset_args+=(-u "$var")
    done
    mkdir -p "$TARGET_DIR" || return $?
    rm -rf -- "${no_models:?}" || return $?
    : >"$capability_report" || return $?
    : >"$started_marker" || return $?
    echo "HERMETIC: no real model can be found by this stage's tests — the model, tokenizer and weight variables are unset, OXIBONSAI_MODELS_DIR names a directory that does not exist ($no_models), and capability records go to $capability_report, not to the release manifest; real-model behaviour is evidenced by the nextest stages and the release gate's serialised legs, not by coverage."
    echo "+ cargo llvm-cov nextest --workspace --profile ci --summary-only | tee $TARGET_DIR/coverage-baseline.txt"
    env "${unset_args[@]}" \
        OXIBONSAI_MODELS_DIR="$no_models" \
        OXIBONSAI_CAPABILITY_REPORT="$capability_report" \
        cargo llvm-cov nextest --workspace --profile ci --summary-only \
        | tee "$TARGET_DIR/coverage-baseline.txt"
    rc=$?
    records=$(grep -c '"capability"' "$capability_report" 2>/dev/null) || records=0
    executed=$(grep -c '"executed":true' "$capability_report" 2>/dev/null) || executed=0
    echo "coverage capability records: $records ($executed executed), all in $capability_report"
    # The redirect above keeps this stage's records out of the release
    # manifest by construction; this checks it. Nothing else runs while a stage
    # does, so a manifest written since the stage started was written by it.
    if [[ "$CAPABILITY_REPORT" -nt "$started_marker" ]]; then
        echo "FAILED: the release manifest $CAPABILITY_REPORT was modified while the coverage stage ran; coverage must never write release evidence." >&2
        return 1
    fi
    echo "the release manifest $CAPABILITY_REPORT was not modified by this stage"
    return "$rc"
}
LLVM_COV_PRESENT=$(( $(tool_ok cargo-llvm-cov) | $(tool_ok cargo-nextest) ))
run_optional_stage "llvm-cov" "$LLVM_COV_PRESENT" \
    "install cargo-llvm-cov and cargo-nextest: cargo install cargo-llvm-cov cargo-nextest --locked" \
    llvm_cov_stage

# ── Stage 13: repo-wide hardcoded temp-directory advisory (deps-08) ─────
# NON-FATAL by design: a hit is reported as a WARN so a developer sees it, but
# it does not fail the gate. The shipped tree is clean (every default scratch
# path comes from `std::env::temp_dir()` in Rust and `${TMPDIR:-/tmp}` in
# shell), so promoting the stage to a hard failure is a one-word change —
# `run_advisory_stage` to `run_required_stage` below — for a release owner who
# wants a regression to stop the gate rather than only be displayed.
#
# There is deliberately no per-file carve-out: a script or example that needs
# a scratch directory honours `${TMPDIR:-/tmp}` (excluded below), which is the
# fix for a hit. The search pattern is written `/tm[p]/` so that this script's
# own text does not match itself.
# shellcheck disable=SC2329  # invoked indirectly via run_advisory_stage "$@"
tmp_hardcode_advisory() {
    local hits
    hits="$(grep -rn '/tm[p]/' --include='*.rs' --include='*.sh' \
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

# ── Stage 14/15: real-model gates (--with-models only) ──────────────────
# Off by default (`WITH_MODELS=0`): these gates each map a real, multi-GB
# GGUF under `models/` and take real minutes even when they self-skip a
# missing file cleanly is fast, but a present file means a full decode.
# `--with-models` is the explicit "yes, actually run them here" request;
# without it these two stages report SKIPPED, in both dev and release mode
# (a missing MODEL FILE is a different situation from a missing TOOL, so
# release mode's usual "a skip must be loud" rule does not apply — the
# model files are gitignored and not expected to exist on every host).
# `scripts/release-gate.sh` remains the place that enforces they MUST have
# run with real evidence before a release; this is the "run them as part of
# an ordinary CI pass, if you have the weights" convenience.
# shellcheck disable=SC2329  # invoked indirectly via run_optional_stage "$@"
real_model_legacy_stage() {
    local models_dir="${OXIBONSAI_MODELS_DIR:-$PROJECT_ROOT/models}"
    for pkg in oxibonsai-model oxibonsai-runtime; do
        echo "+ OXIBONSAI_MODELS_DIR=$models_dir cargo test --release -p $pkg --features metal --test legacy_parity_tests -- --test-threads=1 --nocapture"
        OXIBONSAI_MODELS_DIR="$models_dir" \
            cargo test --release -p "$pkg" --features metal \
            --test legacy_parity_tests -- --test-threads=1 --nocapture || return $?
    done
}
run_with_models_stage "real-model-legacy" real_model_legacy_stage

# shellcheck disable=SC2329  # invoked indirectly via run_optional_stage "$@"
real_model_bonsai2_stage() {
    local models_dir="${OXIBONSAI_MODELS_DIR:-$PROJECT_ROOT/models}"
    local golden_dir="$PROJECT_ROOT/crates/oxibonsai-model/tests/fixtures/bonsai2_golden"
    echo "+ OXIBONSAI_MODELS_DIR=$models_dir cargo test --release -p oxibonsai-model --all-features --test hybrid_forward_parity_tests -- --test-threads=1 --nocapture"
    OXIBONSAI_MODELS_DIR="$models_dir" \
        cargo test --release -p oxibonsai-model --all-features \
        --test hybrid_forward_parity_tests -- --test-threads=1 --nocapture || return $?
    local pq2="$models_dir/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    local ptq1="$models_dir/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    local engine_env=("OXIBONSAI_MODELS_DIR=$models_dir" "OXI_BONSAI2_GOLDEN_DIR=$golden_dir")
    [[ -s "$pq2" ]] && engine_env+=("OXI_BONSAI2_PQ2_GGUF=$pq2")
    [[ -s "$ptq1" ]] && engine_env+=("OXI_BONSAI2_PTQ1_GGUF=$ptq1")
    echo "+ env ${engine_env[*]} cargo test --release -p oxibonsai-runtime --all-features --test bonsai2_engine_tests -- --test-threads=1 --nocapture"
    env "${engine_env[@]}" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_engine_tests -- --test-threads=1 --nocapture || return $?
    # G6/G7 (tokenization, chat-template rendering) and G9 (`oxibonsai info
    # --json`) against the real 27B files.
    local runtime_env=("OXIBONSAI_MODELS_DIR=$models_dir")
    [[ -s "$pq2" ]] && runtime_env+=("OXI_BONSAI2_PQ2_GGUF=$pq2")
    [[ -s "$ptq1" ]] && runtime_env+=("OXI_BONSAI2_PTQ1_GGUF=$ptq1")
    echo "+ env ${runtime_env[*]} cargo test --release -p oxibonsai-runtime --all-features --test bonsai2_runtime_tests -- --test-threads=1 --nocapture"
    env "${runtime_env[@]}" \
        cargo test --release -p oxibonsai-runtime --all-features \
        --test bonsai2_runtime_tests -- --test-threads=1 --nocapture || return $?
}
run_with_models_stage "real-model-bonsai2" real_model_bonsai2_stage

# ── Done ─────────────────────────────────────────────────────────────────
print_summary
echo ""
if [[ -n "$ONLY_STAGE" && "$RELEASE_MODE" -eq 1 ]]; then
    # Only reachable for `--release --only cuda-syntax` (see the --only check
    # above): never worded like the verdict of a full release run.
    echo "PARTIAL RUN COMPLETE: only '$ONLY_STAGE' ran (--only). This is NOT a release-gate pass:"
    echo "every other stage was skipped. scripts/release-gate.sh is the release gate."
else
    echo "ALL STAGES COMPLETE."
fi
exit 0
