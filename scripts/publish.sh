#!/usr/bin/env bash
# OxiBonsai — publish crates to crates.io in dependency order.
#
# Usage:
#   ./scripts/publish.sh            # dry-run (default)
#   ./scripts/publish.sh --for-real # actually publish
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Parse flags ──────────────────────────────────────────────────────────────

DRY_RUN="--dry-run --allow-dirty"
SKIP_CI=""

for arg in "$@"; do
    case "$arg" in
        --for-real)
            DRY_RUN=""
            ;;
        --skip-ci)
            SKIP_CI="1"
            ;;
        --help|-h)
            echo "Usage: $0 [--for-real] [--skip-ci]"
            echo ""
            echo "  --for-real   Actually publish to crates.io (default: dry-run)"
            echo "  --skip-ci    Skip CI checks (fmt, clippy, test, doc)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

if [[ -z "$DRY_RUN" ]]; then
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

# ── CI checks ────────────────────────────────────────────────────────────────

if [[ -z "$SKIP_CI" ]]; then
    echo ">> Step 1/6: cargo fmt --check"
    cargo fmt --all -- --check
    echo "   OK"
    echo ""

    echo ">> Step 2/6: cargo deny check bans licenses sources"
    cargo deny check bans licenses sources
    echo "   OK"
    echo ""

    echo ">> Step 3/6: cargo clippy (all targets, deny warnings)"
    cargo clippy --all-features --all-targets --workspace -- -D warnings
    echo "   OK"
    echo ""

    echo ">> Step 4/6: cargo nextest run"
    cargo nextest run --all-features --workspace
    echo "   OK"
    echo ""

    echo ">> Step 5/6: cargo doc"
    cargo doc --all-features --workspace --no-deps
    echo "   OK"
    echo ""
else
    echo ">> Skipping CI checks (--skip-ci)"
    echo ""
fi

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

echo ">> Step 6/6: Publishing crates"
echo ""

for crate in "${CRATES[@]}"; do
    echo "  Publishing $crate ..."
    if [[ "$crate" == "oxibonsai-cli" ]]; then
        cargo publish $DRY_RUN 2>&1
    else
        cargo publish $DRY_RUN -p "$crate" 2>&1
    fi
    echo "  $crate — done"
    echo ""

    # Wait for crates.io index to update between real publishes
    if [[ -z "$DRY_RUN" && "$crate" != "oxibonsai-cli" ]]; then
        echo "  Waiting 30s for crates.io index ..."
        sleep 30
    fi
done

echo "============================================"
if [[ -z "$DRY_RUN" ]]; then
    echo "  All crates published to crates.io!"
else
    echo "  Dry-run complete — no crates published."
fi
echo "============================================"
