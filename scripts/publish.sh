#!/usr/bin/env bash
# OxiBonsai — publish crates to crates.io in dependency order.
#
# Usage:
#   ./scripts/publish.sh                # dry-run (default): one `cargo publish --workspace
#                                       # --dry-run`, which packages every crate and verifies
#                                       # each against the others' local packages — a per-crate
#                                       # dry run cannot resolve a sibling at a version that is
#                                       # not on crates.io yet
#   ./scripts/publish.sh --for-real     # actually publish, crate by crate, in dependency order
#   ./scripts/publish.sh --require-cuda # also require CUDA capability evidence (see release-gate.sh)
#   ./scripts/publish.sh --accept-approximate-cuda-syntax
#                                       # release from a host WITHOUT the CUDA toolkit: forwarded to
#                                       # release-gate.sh, which then accepts an approximate CUDA
#                                       # syntax check, loudly (see its --help; the gate refuses it
#                                       # together with --require-cuda). Forwarded ONLY when you pass
#                                       # it here, never by default.
#
# T-01: there is no `--skip-ci` any more, in either mode. The prior escape
# hatch is exactly the defect this project's production-release audit
# flagged ("the only gate ... has a --skip-ci escape hatch") — a release
# script that can silently skip its own quality gate is not a gate. The
# full gate (scripts/release-gate.sh, which runs scripts/ci.sh --release
# plus the T-05 hardware-capability check) always runs first, for a
# dry-run exactly as for a real publish, because a dry-run is a rehearsal
# for the real thing and should reflect the same rigor.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ── Parse flags ──────────────────────────────────────────────────────────────

DRY_RUN_FLAG=(--dry-run --allow-dirty)
IS_DRY_RUN=1
RELEASE_GATE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --for-real)
            DRY_RUN_FLAG=()
            IS_DRY_RUN=0
            ;;
        --require-cuda)
            RELEASE_GATE_ARGS+=(--require-cuda)
            ;;
        --accept-approximate-cuda-syntax)
            RELEASE_GATE_ARGS+=(--accept-approximate-cuda-syntax)
            ;;
        --help|-h)
            echo "Usage: $0 [--for-real] [--require-cuda] [--accept-approximate-cuda-syntax]"
            echo ""
            echo "  --for-real      Actually publish to crates.io (default: dry-run)"
            echo "  --require-cuda  Forwarded to scripts/release-gate.sh"
            echo "  --accept-approximate-cuda-syntax"
            echo "                  Forwarded to scripts/release-gate.sh, ONLY when you pass it here (never by"
            echo "                  default): for a host without the CUDA toolkit, accept an approximate CUDA"
            echo "                  kernel-syntax check, loudly; the gate refuses it together with --require-cuda."
            echo ""
            echo "There is no --skip-ci: scripts/release-gate.sh always runs first."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if [[ "$IS_DRY_RUN" -eq 0 ]]; then
    echo "============================================"
    echo "  REAL PUBLISH MODE — crates.io upload"
    echo "============================================"
    echo ""
    read -r -p "Are you sure? Type 'yes' to continue: " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "============================================"
    echo "  DRY-RUN MODE (no actual publish)"
    echo "============================================"
fi

echo ""

# ── Release gate (mandatory — no --skip-ci) ─────────────────────────────────
# ${arr[@]+"${arr[@]}"}, not a bare "${arr[@]}": macOS's default /bin/bash
# (3.2) treats an empty array's "${arr[@]}" as unbound under `set -u`, and
# RELEASE_GATE_ARGS is empty unless --require-cuda or
# --accept-approximate-cuda-syntax was passed.
#
# The failure branch is an explicit `|| { rc=$?; ...; exit "$rc"; }`, not a
# bare call relying on this script's own `set -e` to stop on a non-zero
# exit. A publish script whose entire safety property depends on an
# implicit shell option holding all the way down a call chain is exactly
# the class of fragility this gate exists to remove (see
# release-gate.sh's own header for the concrete bug that shape caused
# there before it was fixed) — make the abort explicit and say plainly
# that nothing was published.
echo ">> Running scripts/release-gate.sh (scripts/ci.sh --release + T-05 capability check)"
echo ""
"$SCRIPT_DIR/release-gate.sh" "${RELEASE_GATE_ARGS[@]+"${RELEASE_GATE_ARGS[@]}"}" || {
    rc=$?
    echo "" >&2
    echo "ABORTING: scripts/release-gate.sh failed (exit $rc). Nothing was published." >&2
    exit "$rc"
}
echo ""
echo ">> Release gate passed."
echo ""

# ── Publish crates in dependency order ───────────────────────────────────────

CRATES=(
    oxibonsai-core
    oxibonsai-tokenizer
    oxibonsai-kernels
    oxibonsai-image
    oxibonsai-rag
    oxibonsai-model
    oxibonsai-runtime
    oxibonsai-eval
    oxibonsai-serve
    oxibonsai
    oxibonsai-cli
)

echo ">> Publishing crates"
echo ""

# A dry run packages and verifies the whole workspace in ONE cargo invocation:
# cargo then resolves each crate's workspace siblings from the packages it has
# just built, through a temporary registry overlay keyed by version. Cargo
# treats those sources as immutable, so run the dry run with CARGO_TARGET_DIR
# pointing at an EMPTY directory (or after `cargo clean`): a target directory
# that served an earlier dry run of this same version verifies the new tree
# against the stale rlibs of that run and fails on whatever changed since. Run crate by crate, as the real publish below must be, a dry run
# of any crate with a workspace dependency fails with "failed to select a
# version for the requirement `oxibonsai-core = \"^<this version>\"`" — that
# version is not on crates.io until the real publish has uploaded it.
# `oxibonsai-testkit` is `publish = false` and is named only to say so.
if [[ "$IS_DRY_RUN" -eq 1 ]]; then
    echo "  cargo publish --workspace --exclude oxibonsai-testkit ${DRY_RUN_FLAG[*]}"
    cargo publish --workspace --exclude oxibonsai-testkit "${DRY_RUN_FLAG[@]}" 2>&1
    echo ""
    echo "============================================"
    echo "  Dry-run complete — no crates published."
    echo "============================================"
    exit 0
fi

for crate in "${CRATES[@]}"; do
    echo "  Publishing $crate ..."
    if [[ "$crate" == "oxibonsai-cli" ]]; then
        cargo publish "${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"}" 2>&1
    else
        cargo publish "${DRY_RUN_FLAG[@]+"${DRY_RUN_FLAG[@]}"}" -p "$crate" 2>&1
    fi
    echo "  $crate — done"
    echo ""

    # Wait for crates.io index to update between real publishes
    if [[ "$IS_DRY_RUN" -eq 0 && "$crate" != "oxibonsai-cli" ]]; then
        echo "  Waiting 30s for crates.io index ..."
        sleep 30
    fi
done

echo "============================================"
if [[ "$IS_DRY_RUN" -eq 0 ]]; then
    echo "  All crates published to crates.io!"
else
    echo "  Dry-run complete — no crates published."
fi
echo "============================================"
