#!/usr/bin/env bash
# Download a Ternary/Bonsai GGUF for OxiBonsai.
#
# Usage:
#   ./scripts/download_ternary.sh [8b|4b|1.7b]                     # unpacked safetensors -> local GGUF convert
#   ./scripts/download_ternary.sh 27b[-ptq1_0|-pq2_0|-q2_0|-all]   # Bonsai 2 27B: pre-built GGUF, direct download
#   ./scripts/download_ternary.sh --help
#   ./scripts/download_ternary.sh --self-test                      # offline dispatch test, see below
#
# 8b/4b/1.7b: PrismML publishes the original Ternary Bonsai models as
# *unpacked safetensors* — this downloads the shards and runs
# `oxibonsai convert --quant tq2_0_g128` locally to produce the GGUF.
#
# 27b[-ptq1_0|-pq2_0|-q2_0|-all]: PrismML's newer Bonsai 2 27B release
# (2026-09-17) ships as pre-quantized GGUF directly — three language-model
# variants (PTQ1_0 1-bit, PQ2_0 2-bit, and a legacy-layout Q2_0 that NEEDS
# PrismML's `prism` llama.cpp fork to read correctly: it reuses ggml type
# id 42 with a group-128 block layout, which mainline llama.cpp treats as
# the official group-64 Q2_0 and would silently misdecode — see
# docs/models.md, "ggml type id 42: three layouts under one id") plus a shared
# Q8_0 mmproj vision projector — this downloads the GGUF(s) verbatim, no local
# conversion.
#
# REPO PROVENANCE (sec-12) — two separate sources, kept separate below because
# they carry different authority; do not merge them into one unsourced claim:
#   - CONFIRMED BY THE PROJECT OWNER: the repo is
#     `prism-ml/Ternary-Bonsai-2-27B-gguf`, holding
#     Ternary-Bonsai-2-27B-PTQ1_0.gguf (5946648928 B),
#     Ternary-Bonsai-2-27B-PQ2_0.gguf (7206168928 B) and
#     Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf (629246976 B) — independently
#     corroborated by the file list in the README of PrismML's Bonsai-demo
#     repository. This is OXI_BONSAI2_REPO's default below.
#   - FROM `MODEL-FORMATS.md` in PrismML's Bonsai-demo repository (its table
#     of model files): the third language GGUF,
#     Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf, is deliberately kept
#     OUT of that repo — mainline llama.cpp already knows ggml type id 42 and
#     would silently misdecode this file's group-128 layout as the official
#     group-64 Q2_0 rather than refusing it outright — and is published
#     separately, for testing only, in `prism-ml/Ternary-Bonsai-2-27B-gguf-dev`
#     (~7.6 GB). This is OXI_BONSAI2_DEV_REPO's default below.
# Override either env var if you need a different revision or mirror.
# `oxibonsai pull` (cli-05) is the native equivalent with the same validation
# baked in; this script remains the scripted fallback.
#
# Every downloaded/converted GGUF is verified against
# `scripts/checksums.sha256` afterwards (sec-12): a mismatch on a directly
# downloaded (authoritative) file aborts loudly; a mismatch on a locally
# converted (advisory) file is a warning, since the converter's own
# behavior — not just the upstream source — determines those bytes.
#
# --self-test runs an isolated, offline check of the variant-dispatch logic
# only (resolve_variant(), below): every `27b-*` variant resolves to the
# expected repo:file pairs (in particular, that `27b-q2_0`/`27b-all` pull
# the legacy file from OXI_BONSAI2_DEV_REPO, never OXI_BONSAI2_REPO), and
# an unknown variant is rejected. It never invokes the `hf` CLI, touches
# the network, or requires `hf` to even be installed.
#
# Requires: hf CLI (pip install huggingface_hub  — new command is `hf`, not `huggingface-cli`)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ── Bonsai 2 27B repo/file constants — see the provenance comment above ──
OXI_BONSAI2_REPO="${OXI_BONSAI2_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf}"
OXI_BONSAI2_DEV_REPO="${OXI_BONSAI2_DEV_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf-dev}"
MMPROJ_FILE="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
PTQ1_0_FILE="Ternary-Bonsai-2-27B-PTQ1_0.gguf"
PQ2_0_FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
# Legacy group-128 layout (ggml type id 42, general.file_type 41) — needs
# PrismML's `prism` fork to read; see the header comment above. Lives in
# OXI_BONSAI2_DEV_REPO, NOT OXI_BONSAI2_REPO — that split is the whole
# point of the structural fix below (deps-08: resolving every direct file
# against a single $REPO would fetch this one from the wrong repository).
PRISM_Q2_0_FILE="Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf"

MODE=""                 # convert (unpack+build+convert) | direct (fetch GGUF verbatim)
REPO=""                 # convert mode only: the single unpacked-safetensors repo
OUT=""                  # convert mode only: local GGUF output path
DIRECT_FILES=()         # direct mode only: "repo:file" pairs (see resolve_variant)

# Resolves a `$1` size/variant argument into MODE plus either (REPO, OUT)
# for convert mode or DIRECT_FILES for direct mode. Pure argument mapping —
# no network, no `hf` CLI, no filesystem writes — which is exactly what
# lets --self-test below exercise it in isolation. `DIRECT_FILES` entries
# are "repo:file" pairs, not bare filenames: the three direct-mode
# artifacts do NOT all come from the same repo (see the provenance comment
# at the top of this file), so a single global $REPO used for all of them
# would be wrong for 27b-q2_0/27b-all.
resolve_variant() {
    local size="$1"
    MODE="convert"
    REPO=""
    OUT=""
    DIRECT_FILES=()
    case "$size" in
        8b|8B)
            REPO="prism-ml/Ternary-Bonsai-8B-unpacked"
            OUT="models/Ternary-Bonsai-8B.gguf"
            ;;
        4b|4B)
            REPO="prism-ml/Ternary-Bonsai-4B-unpacked"
            OUT="models/Ternary-Bonsai-4B.gguf"
            ;;
        1.7b|1.7B)
            REPO="prism-ml/Ternary-Bonsai-1.7B-unpacked"
            OUT="models/Ternary-Bonsai-1.7B.gguf"
            ;;
        27b|27B|bonsai2|Bonsai2|27b-ptq1_0)
            MODE="direct"
            DIRECT_FILES=("$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE")
            ;;
        27b-pq2_0)
            MODE="direct"
            DIRECT_FILES=("$OXI_BONSAI2_REPO:$PQ2_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE")
            ;;
        27b-q2_0)
            MODE="direct"
            DIRECT_FILES=("$OXI_BONSAI2_DEV_REPO:$PRISM_Q2_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE")
            ;;
        27b-all)
            MODE="direct"
            DIRECT_FILES=(
                "$OXI_BONSAI2_REPO:$PTQ1_0_FILE"
                "$OXI_BONSAI2_REPO:$PQ2_0_FILE"
                "$OXI_BONSAI2_DEV_REPO:$PRISM_Q2_0_FILE"
                "$OXI_BONSAI2_REPO:$MMPROJ_FILE"
            )
            ;;
        *)
            echo "ERROR: unknown size/variant: $size" >&2
            echo "Usage: $0 [8b|4b|1.7b|27b|27b-ptq1_0|27b-pq2_0|27b-q2_0|27b-all]" >&2
            exit 1
            ;;
    esac
}

assert_direct_files() {
    local label="$1"
    shift
    local expected=("$@")
    if [[ "${#DIRECT_FILES[@]}" -ne "${#expected[@]}" ]]; then
        echo "FAIL: $label: expected ${#expected[@]} entries, got ${#DIRECT_FILES[@]}"
        DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
        return
    fi
    local i
    for i in "${!expected[@]}"; do
        if [[ "${DIRECT_FILES[$i]}" != "${expected[$i]}" ]]; then
            echo "FAIL: $label: entry $i expected '${expected[$i]}', got '${DIRECT_FILES[$i]}'"
            DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
        fi
    done
}

# ── --self-test: isolated, offline dispatch test (no network, no `hf`) ──
# deps-08: every `27b-*` variant must resolve to the
# correct repo:file pairs (in particular, that only 27b-q2_0/27b-all pull
# from OXI_BONSAI2_DEV_REPO), and an unknown variant must be rejected. Only
# calls resolve_variant() — the download loop that actually shells out to
# `hf` lives entirely below the `if [[ "$MODE" == "direct" ]]` gate later in
# this script, which self-test never reaches.
run_dispatch_self_test() {
    DISPATCH_TEST_FAILURES=0

    resolve_variant "27b"
    [[ "$MODE" == "direct" ]] || { echo "FAIL: 27b: MODE='$MODE', want direct"; DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1)); }
    assert_direct_files "27b" "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "27B"
    assert_direct_files "27B (case alias)" "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "bonsai2"
    assert_direct_files "bonsai2 (alias of 27b)" "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "Bonsai2"
    assert_direct_files "Bonsai2 (case alias)" "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "27b-ptq1_0"
    assert_direct_files "27b-ptq1_0" "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "27b-pq2_0"
    assert_direct_files "27b-pq2_0" "$OXI_BONSAI2_REPO:$PQ2_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "27b-q2_0"
    assert_direct_files "27b-q2_0 (must come from the DEV repo, not the main one)" \
        "$OXI_BONSAI2_DEV_REPO:$PRISM_Q2_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    resolve_variant "27b-all"
    assert_direct_files "27b-all" \
        "$OXI_BONSAI2_REPO:$PTQ1_0_FILE" "$OXI_BONSAI2_REPO:$PQ2_0_FILE" \
        "$OXI_BONSAI2_DEV_REPO:$PRISM_Q2_0_FILE" "$OXI_BONSAI2_REPO:$MMPROJ_FILE"

    # A convert-mode variant must NOT be MODE=direct and must not touch
    # DIRECT_FILES (regression guard: a copy-paste into the wrong case arm
    # would otherwise silently turn a convert variant into a direct one).
    resolve_variant "1.7b"
    if [[ "$MODE" != "convert" ]]; then
        echo "FAIL: 1.7b: MODE='$MODE', want convert"
        DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
    fi
    if [[ "${#DIRECT_FILES[@]}" -ne 0 ]]; then
        echo "FAIL: 1.7b: DIRECT_FILES should be empty, has ${#DIRECT_FILES[@]} entries"
        DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
    fi
    if [[ "$REPO" != "prism-ml/Ternary-Bonsai-1.7B-unpacked" || "$OUT" != "models/Ternary-Bonsai-1.7B.gguf" ]]; then
        echo "FAIL: 1.7b: REPO/OUT unexpected: REPO='$REPO' OUT='$OUT'"
        DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
    fi

    # Unknown-variant rejection: resolve_variant's `*)` arm calls `exit 1`,
    # so it MUST run in a subshell here — otherwise that `exit` would kill
    # this self-test (and the whole script), not just report a failure.
    # Inside the `else` of a plain (non-negated) `if CMD; then ... else`,
    # `$?` correctly holds CMD's own exit status (unlike `if ! CMD; then`,
    # where the `!` negation loses it — see release-gate.sh's header for
    # the concrete bug that idiom causes).
    if (resolve_variant "not-a-real-variant") >/dev/null 2>&1; then
        echo "FAIL: unknown variant 'not-a-real-variant' did not reject (expected exit 1)"
        DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
    else
        rc=$?
        if [[ "$rc" -ne 1 ]]; then
            echo "FAIL: unknown variant exited $rc, want 1"
            DISPATCH_TEST_FAILURES=$((DISPATCH_TEST_FAILURES + 1))
        fi
    fi

    if [[ "$DISPATCH_TEST_FAILURES" -eq 0 ]]; then
        echo "OK: dispatch self-test — every 27b-* variant resolves to the correct"
        echo "    repo:file pairs, a convert-mode variant is unaffected, and an"
        echo "    unknown variant is rejected."
        return 0
    fi
    echo "FAILED: dispatch self-test — $DISPATCH_TEST_FAILURES assertion(s) failed."
    return 1
}

SIZE="${1:-1.7b}"

case "$SIZE" in
    --help|-h)
        awk 'NR==1{next} /^#/{print; next} {exit}' "$0"
        exit 0
        ;;
    --self-test)
        run_dispatch_self_test
        exit $?
        ;;
esac

resolve_variant "$SIZE"

# ── Validate prerequisites ───────────────────────────────────────────────────
HF_CMD=""
if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
elif [[ -x "$HOME/.local/bin/hf" ]]; then
    HF_CMD="$HOME/.local/bin/hf"
elif [[ -x "$HOME/.local/bin/huggingface-cli" ]]; then
    HF_CMD="$HOME/.local/bin/huggingface-cli"
else
    echo "ERROR: hf CLI not found." >&2
    echo "Install it with: pip install huggingface_hub" >&2
    echo "or add ~/.local/bin to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
fi

mkdir -p models

# ── Checksum verification helper (sec-12) ───────────────────────────────────
# `kind` is "authoritative" (a byte-for-byte published artifact — a
# mismatch aborts) or "advisory" (a locally converted artifact — a
# mismatch only warns, since the converter's own version affects the
# bytes). See scripts/checksums.sha256's header for the full rationale.
verify_checksum() {
    local file="$1" kind="$2"
    local checksums_file="$SCRIPT_DIR/checksums.sha256"
    if [[ ! -f "$checksums_file" ]]; then
        echo "WARN: $checksums_file not found; skipping checksum verification for $file." >&2
        return 0
    fi
    local line
    line="$(awk -v f="models/$file" '$2 == f {print; exit}' "$checksums_file")"
    if [[ -z "$line" ]]; then
        echo "NOTE: no checksum recorded yet for 'models/$file' in scripts/checksums.sha256;" >&2
        echo "      skipping verification. Add one with:" >&2
        echo "        shasum -a 256 models/$file | awk '{print \$1 \"  \" \$2}'" >&2
        return 0
    fi
    local verify_out rc=0
    if command -v sha256sum >/dev/null 2>&1; then
        verify_out="$(cd "$PROJECT_ROOT" && printf '%s\n' "$line" | sha256sum -c - 2>&1)" || rc=$?
    elif command -v shasum >/dev/null 2>&1; then
        verify_out="$(cd "$PROJECT_ROOT" && printf '%s\n' "$line" | shasum -a 256 -c - 2>&1)" || rc=$?
    else
        echo "WARN: neither sha256sum nor shasum found on PATH; cannot verify $file." >&2
        return 0
    fi
    if [[ "$rc" -eq 0 ]]; then
        echo "Checksum OK: models/$file"
        return 0
    fi
    if [[ "$kind" == "authoritative" ]]; then
        echo "ERROR: checksum MISMATCH for 'models/$file' (authoritative artifact):" >&2
        echo "$verify_out" >&2
        echo "Refusing to continue. Delete the file and retry the download, or update" >&2
        echo "scripts/checksums.sha256 if this is a confirmed, intentional upstream revision." >&2
        return 1
    fi
    echo "WARNING: checksum drift for 'models/$file' (advisory — this is a locally" >&2
    echo "converted artifact, so drift can reflect an intentional converter change" >&2
    echo "rather than corruption; see scripts/checksums.sha256):" >&2
    echo "$verify_out" >&2
    return 0
}

if [[ "$MODE" == "direct" ]]; then
    # ── Bonsai 2 27B: fetch pre-built GGUF(s) verbatim, no conversion ───────
    # Each DIRECT_FILES entry is a "repo:file" pair (see resolve_variant):
    # PTQ1_0/PQ2_0/mmproj come from OXI_BONSAI2_REPO, but the legacy Q2_0
    # file comes from OXI_BONSAI2_DEV_REPO — do not collapse this back to a
    # single $REPO for the whole loop.
    echo "Downloading Bonsai 2 27B artifact(s) (using: $HF_CMD)..."
    echo "NOTE: OXI_BONSAI2_REPO=$OXI_BONSAI2_REPO (PTQ1_0 / PQ2_0 / mmproj)" >&2
    echo "      OXI_BONSAI2_DEV_REPO=$OXI_BONSAI2_DEV_REPO (Q2_0, testing only)" >&2
    echo "      Override either env var if you need a different revision or mirror." >&2
    HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
    HF_PARALLEL_ARGS=()
    if "$HF_CMD" download --help 2>&1 | grep -q -- "--max-workers"; then
        HF_PARALLEL_ARGS=(--max-workers "$HF_MAX_WORKERS")
    fi
    DONE_FILES=()
    for entry in "${DIRECT_FILES[@]}"; do
        repo="${entry%%:*}"
        f="${entry#*:}"
        if [[ -f "models/$f" ]]; then
            echo "Already present: models/$f (skipping download; delete it to re-fetch)"
        else
            echo "Downloading $f from $repo ..."
            "$HF_CMD" download "$repo" "$f" \
                --local-dir models \
                "${HF_PARALLEL_ARGS[@]+"${HF_PARALLEL_ARGS[@]}"}"
        fi
        verify_checksum "$f" "authoritative"
        DONE_FILES+=("models/$f")
    done
    echo "Done: ${DONE_FILES[*]}"
    exit 0
fi

# ── convert mode (1.7b/4b/8b): unpacked safetensors -> local GGUF convert ──

LOCAL_DIR="models/${REPO##*/}"  # e.g. models/Ternary-Bonsai-1.7B-unpacked

# On mounted filesystems (e.g. /mnt/g), hf's lock file creation can race.
# Pre-create lock/cache dirs and use a conservative worker count by default.
mkdir -p "$LOCAL_DIR/.huggingface/download"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
HF_PARALLEL_ARGS=()
if "$HF_CMD" download --help 2>&1 | grep -q -- "--max-workers"; then
    HF_PARALLEL_ARGS=(--max-workers "$HF_MAX_WORKERS")
else
    echo "Note: $HF_CMD does not support --max-workers; continuing with CLI defaults." >&2
fi

# ── Download from HuggingFace ────────────────────────────────────────────────
echo "Downloading $REPO  (using: $HF_CMD)..."
hf_download() {
    # The new `hf` CLI's argparse requires positional `filenames` to appear
    # before optional flags; older `huggingface-cli` was order-agnostic.
    # ${arr[@]+"${arr[@]}"} rather than a bare "${arr[@]}": macOS ships bash
    # 3.2 as /bin/bash, which treats "${arr[@]}" on a still-empty array as
    # an unbound-variable error under `set -u` (fixed in bash 4.4+, never
    # backported to 3.2). This is the portable no-op-when-empty idiom —
    # needed here because HF_PARALLEL_ARGS is legitimately empty whenever
    # the installed hf/huggingface-cli predates --max-workers.
    "$HF_CMD" download "$REPO" \
        "$@" \
        --local-dir "$LOCAL_DIR" \
        "${HF_PARALLEL_ARGS[@]+"${HF_PARALLEL_ARGS[@]}"}"
}

# Download metadata files first (always present).
hf_download "model.safetensors.index.json" "config.json" "tokenizer.json" 2>/dev/null || \
hf_download --include "*.safetensors.index.json" --include "config.json" --include "tokenizer.json"

# If a shard index exists, download each shard listed in it; otherwise try single-file.
INDEX_FILE="$LOCAL_DIR/model.safetensors.index.json"
if [[ -f "$INDEX_FILE" ]]; then
    # Sharded layout: collect unique shard filenames from the weight_map.
    SHARDS=$(python3 -c "
import json, sys
with open('$INDEX_FILE') as f:
    d = json.load(f)
print('\n'.join(sorted(set(d['weight_map'].values()))))
")
    echo "Downloading $(echo "$SHARDS" | wc -l | tr -d ' ') shard(s)..."
    while IFS= read -r shard; do
        echo "  $shard"
        hf_download "$shard"
    done <<< "$SHARDS"
else
    # Single-file layout.
    hf_download "model.safetensors"
fi

# ── Build the converter ──────────────────────────────────────────────────────
echo "Building converter..."
cargo build --release 2>&1 | tail -5

# ── Convert to GGUF ─────────────────────────────────────────────────────────
echo "Converting $LOCAL_DIR -> $OUT..."
./target/release/oxibonsai convert \
    --from "$LOCAL_DIR" \
    --to "$OUT" \
    --quant tq2_0_g128

echo "Done: $OUT"
verify_checksum "$(basename "$OUT")" "advisory"

# ── Copy tokenizer if not already present ────────────────────────────────────
if [[ -f "$LOCAL_DIR/tokenizer.json" && ! -f "models/tokenizer.json" ]]; then
    cp "$LOCAL_DIR/tokenizer.json" models/tokenizer.json
    echo "Copied tokenizer.json to models/"
fi

# ── Remove downloaded source files ───────────────────────────────────────────
echo "Removing $LOCAL_DIR..."
rm -rf "$LOCAL_DIR"
