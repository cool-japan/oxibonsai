#!/usr/bin/env bash
# Download Ternary-Bonsai unpacked safetensors and convert to OxiBonsai GGUF.
#
# Usage: ./scripts/download_ternary.sh [8b|4b|1.7b]  (default: 1.7b)
#
# Requires: hf CLI (pip install huggingface_hub  — new command is `hf`, not `huggingface-cli`)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SIZE="${1:-1.7b}"

case "$SIZE" in
  8b|8B)
    REPO="prism-ml/Ternary-Bonsai-8B-unpacked"
    OUT="models/Ternary-Bonsai-8B.gguf"
    ;;
  4b|4B)
    REPO="prism-ml/Ternary-Bonsai-4B-unpacked"
    OUT="models/Ternary-Bonsai-4B.gguf"
    ;;
  1.7b|1.7B|*)
    REPO="prism-ml/Ternary-Bonsai-1.7B-unpacked"
    OUT="models/Ternary-Bonsai-1.7B.gguf"
    ;;
esac

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
    "$HF_CMD" download "$REPO" \
        --local-dir "$LOCAL_DIR" \
        "${HF_PARALLEL_ARGS[@]}" \
        "$@"
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

# ── Copy tokenizer if not already present ────────────────────────────────────
if [[ -f "$LOCAL_DIR/tokenizer.json" && ! -f "models/tokenizer.json" ]]; then
    cp "$LOCAL_DIR/tokenizer.json" models/tokenizer.json
    echo "Copied tokenizer.json to models/"
fi

# ── Remove downloaded source files ───────────────────────────────────────────
echo "Removing $LOCAL_DIR..."
rm -rf "$LOCAL_DIR"
