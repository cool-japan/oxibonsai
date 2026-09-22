#!/usr/bin/env bash
# OxiBonsai — Ternary Bonsai CLI smoke + performance test.
#
# Runs inference on a Ternary Bonsai GGUF (TQ2_0_g128 weights) and prints
# a concise throughput summary at the end. On macOS with `metal` the
# whole forward pass runs in a single Metal command buffer via the fused
# TQ2 path — expect 40+ tok/s on Apple Silicon for the 1.7B model.
#
# Usage:
#   ./scripts/cli_ternary.sh                      # CPU SIMD (NEON / AVX2 / AVX-512)
#   ./scripts/cli_ternary.sh metal                # Metal GPU (macOS) — fused TQ2 path
#   ./scripts/cli_ternary.sh cuda                 # native CUDA backend (Linux/Windows)
#   ./scripts/cli_ternary.sh cuda-scirs           # scirs2-core CUDA backend
#   ./scripts/cli_ternary.sh --model <path>       # custom GGUF (default: 1.7B)
#   ./scripts/cli_ternary.sh --max-tokens 200     # override token count
#   ./scripts/cli_ternary.sh --prompt "…"         # override prompt
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────

MODEL="models/Ternary-Bonsai-1.7B.gguf"
TOKENIZER="models/tokenizer.json"
PROMPT="Explain 2+2 = ??"
MAX_TOKENS=100
TEMPERATURE=0.7
TOP_P=0.9
SEED=42

# ── Parse arguments ──────────────────────────────────────────────────────────

FEATURES=""
FEATURE_LABEL="default"

for arg in "$@"; do
    case "$arg" in
        metal)
            FEATURES="--features metal"
            FEATURE_LABEL="metal"
            ;;
        cuda)
            FEATURES="--features native-cuda,simd-avx2"
            FEATURE_LABEL="cuda"
            ;;
        cuda-scirs)
            FEATURES="--features cuda"
            FEATURE_LABEL="cuda-scirs"
            ;;
        --model)
            # handled below via shift; getopts not used to keep it simple
            ;;
        --model=*)
            MODEL="${arg#--model=}"
            ;;
        --help|-h)
            echo "Usage: $0 [metal|cuda|cuda-scirs] [--model <path>] [--prompt <str>] [--max-tokens <n>]"
            echo ""
            echo "  (no args)         Build & run with default features (CPU SIMD)"
            echo "  metal             Build & run with Metal GPU acceleration (macOS)"
            echo "  cuda              Build & run with native CUDA backend (Linux/Windows)"
            echo "  cuda-scirs        Build & run with scirs2-core CUDA backend"
            echo "  --model <path>    Path to Ternary Bonsai GGUF (default: $MODEL)"
            echo "  --prompt <str>    Prompt to generate from (default: \"$PROMPT\")"
            echo "  --max-tokens <n>  Max tokens to generate (default: $MAX_TOKENS)"
            exit 0
            ;;
        *)
            # Check if previous arg was --model/--prompt/--max-tokens (positional value)
            ;;
    esac
done

# Re-parse for two-token flags.
SKIP_KIND=""
for arg in "$@"; do
    if [[ -n "$SKIP_KIND" ]]; then
        case "$SKIP_KIND" in
            model)       MODEL="$arg" ;;
            prompt)      PROMPT="$arg" ;;
            max-tokens)  MAX_TOKENS="$arg" ;;
        esac
        SKIP_KIND=""
        continue
    fi
    case "$arg" in
        --model)       SKIP_KIND="model" ;;
        --prompt)      SKIP_KIND="prompt" ;;
        --max-tokens)  SKIP_KIND="max-tokens" ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────────────────────────

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    echo "" >&2
    echo "Place a Ternary Bonsai GGUF at:  $MODEL" >&2
    echo "Or pass a custom path:           $0 --model /path/to/Ternary-Bonsai-*.gguf" >&2
    echo "" >&2
    echo "See the project README for download / conversion instructions." >&2
    exit 1
fi

TOKENIZER_FLAG=""
if [[ -f "$TOKENIZER" ]]; then
    TOKENIZER_FLAG="--tokenizer $TOKENIZER"
else
    echo "WARN: tokenizer not found ($TOKENIZER), running without tokenizer"
fi

# ── Build ────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  OxiBonsai Ternary CLI smoke test  (features: $FEATURE_LABEL)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Building release binary..."
# shellcheck disable=SC2086
cargo build --release $FEATURES
echo "Build OK"
echo ""

# ── Run inference ────────────────────────────────────────────────────────────

BIN="./target/release/oxibonsai"
RUN_LOG=$(mktemp "${TMPDIR:-/tmp}/oxibonsai_ternary_run.XXXXXX")
trap 'rm -f "$RUN_LOG"' EXIT

echo "── run: basic inference (ternary) ──────────────────────────"
echo "  model:       $MODEL"
echo "  prompt:      \"$PROMPT\""
echo "  max_tokens:  $MAX_TOKENS"
echo "  temperature: $TEMPERATURE"
echo "  top_p:       $TOP_P"
echo "  seed:        $SEED"
echo ""

# shellcheck disable=SC2086
RUST_LOG=${RUST_LOG:-warn} "$BIN" run \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    $TOKENIZER_FLAG \
    2>&1 | tee "$RUN_LOG"

# Extract tok/s from the standard summary line emitted by the CLI:
#   "8 prompt + 100 generated = 108 total tokens in 1.85s (53.9 tok/s)"
TOKS_PER_SEC=$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$RUN_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)

echo ""
echo "── info ─────────────────────────────────────────────────────"
"$BIN" info --model "$MODEL"

echo ""
echo "── validate ─────────────────────────────────────────────────"
"$BIN" validate --model "$MODEL"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Ternary CLI summary  (features: $FEATURE_LABEL)"
echo "═══════════════════════════════════════════════════════════════"
if [[ -n "$TOKS_PER_SEC" ]]; then
    printf "  Throughput: %s tok/s  (model: %s)\n" "$TOKS_PER_SEC" "$(basename "$MODEL")"
else
    echo "  Throughput: (tok/s not captured — check run output above)"
fi
echo "═══════════════════════════════════════════════════════════════"
