#!/usr/bin/env bash
# OxiBonsai — Ternary Bonsai CPU vs Metal throughput benchmark.
#
# Builds the release binary for each target (CPU-only, then Metal on macOS)
# and reports the tok/s measured by the CLI's standard summary line. Intended
# for quick regression checks after kernel / dispatch changes.
#
# Usage:
#   ./scripts/bench_ternary.sh                    # default 1.7B, 100 tokens, 3 runs each
#   ./scripts/bench_ternary.sh --model <path>     # override model
#   ./scripts/bench_ternary.sh --prompt "…"       # override prompt
#   ./scripts/bench_ternary.sh --max-tokens 200   # override token count
#   ./scripts/bench_ternary.sh --runs 1           # faster, less noise tolerance
#   ./scripts/bench_ternary.sh --metal-only       # skip CPU run
#   ./scripts/bench_ternary.sh --cpu-only         # skip Metal run
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# NOTE on "${arr[@]+"${arr[@]}"}" below: macOS ships bash 3.2 as /bin/bash
# (and `#!/usr/bin/env bash` resolves to it unless a newer bash is on PATH
# first). Under `set -u`, bash 3.2 treats "${arr[@]}" on a still-empty array
# as an unbound-variable error — bash 4.4+ fixed this, 3.2 never will. The
# `${arr[@]+"${arr[@]}"}` form is the portable workaround: it expands to
# nothing when the array is empty/unset and to the normal argument list
# otherwise. Do not "simplify" it back to a bare "${arr[@]}" — that
# reintroduces a crash on the default macOS shell whenever the guarded
# array is legitimately empty (no tokenizer.json, --runs 0, etc).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

MODEL="models/Ternary-Bonsai-1.7B.gguf"
TOKENIZER="models/tokenizer.json"
PROMPT="Explain quantum computing in simple terms"
MAX_TOKENS=100
SEED=42
RUNS=3
DO_CPU=1
DO_METAL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --prompt)      PROMPT="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --runs)        RUNS="$2"; shift 2 ;;
        --cpu-only)    DO_METAL=0; shift ;;
        --metal-only)  DO_CPU=0; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

TOKENIZER_FLAG=()
if [[ -f "$TOKENIZER" ]]; then
    TOKENIZER_FLAG=(--tokenizer "$TOKENIZER")
fi

# ── Helpers ─────────────────────────────────────────────────────────────────

run_one() {
    # $1 = feature label (for printing), $2 = seed
    local label="$1"
    local seed="$2"
    local out
    out=$(RUST_LOG=warn ./target/release/oxibonsai run \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --seed "$seed" \
        "${TOKENIZER_FLAG[@]+"${TOKENIZER_FLAG[@]}"}" 2>&1 | tail -2 | tr -d '\r')
    local toks
    toks=$(echo "$out" | grep -oE '[0-9]+\.[0-9]+ tok/s' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || true)
    if [[ -z "$toks" ]]; then toks="—"; fi
    printf "  [%s] run seed=%-3d  %s tok/s\n" "$label" "$seed" "$toks"
    echo "$toks"
}

summary() {
    # $1 = label, remaining args are numeric tok/s values
    local label="$1"; shift
    local sum=0 cnt=0 best="" val
    for val in "$@"; do
        if [[ "$val" != "—" ]]; then
            sum=$(awk -v s="$sum" -v v="$val" 'BEGIN{print s + v}')
            cnt=$((cnt + 1))
            if [[ -z "$best" ]] || awk -v a="$val" -v b="$best" 'BEGIN{exit !(a > b)}'; then
                best="$val"
            fi
        fi
    done
    if [[ $cnt -gt 0 ]]; then
        local avg
        avg=$(awk -v s="$sum" -v c="$cnt" 'BEGIN{printf "%.1f", s/c}')
        printf "  %-14s avg=%s tok/s   best=%s tok/s   (n=%d)\n" "$label" "$avg" "$best" "$cnt"
    else
        printf "  %-14s (no successful runs)\n" "$label"
    fi
}

# ── CPU SIMD run ────────────────────────────────────────────────────────────

CPU_RESULTS=()
if [[ "$DO_CPU" -eq 1 ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  CPU SIMD build (default features)"
    echo "═══════════════════════════════════════════════════════════════"
    cargo build --release --quiet
    for i in $(seq 1 "$RUNS"); do
        result=$(run_one "cpu " $((SEED + i - 1)) | tail -1)
        CPU_RESULTS+=("$result")
    done
    echo ""
fi

# ── Metal run ───────────────────────────────────────────────────────────────

METAL_RESULTS=()
if [[ "$DO_METAL" -eq 1 ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Metal build (--features metal)"
    echo "═══════════════════════════════════════════════════════════════"
    cargo build --release --features metal --quiet
    # Warm Metal shader cache so the first timed run isn't penalised.
    RUST_LOG=warn ./target/release/oxibonsai run \
        --model "$MODEL" \
        --prompt "hi" \
        --max-tokens 1 \
        --seed 1 \
        "${TOKENIZER_FLAG[@]+"${TOKENIZER_FLAG[@]}"}" >/dev/null 2>&1 || true
    for i in $(seq 1 "$RUNS"); do
        result=$(run_one "metal" $((SEED + i - 1)) | tail -1)
        METAL_RESULTS+=("$result")
    done
    echo ""
fi

# ── Summary ─────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  Throughput summary  (model: $(basename "$MODEL"), tokens: $MAX_TOKENS)"
echo "═══════════════════════════════════════════════════════════════"
if [[ "$DO_CPU"   -eq 1 ]]; then summary "CPU (SIMD)"   "${CPU_RESULTS[@]+"${CPU_RESULTS[@]}"}";   fi
if [[ "$DO_METAL" -eq 1 ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    summary "Metal (GPU)"  "${METAL_RESULTS[@]+"${METAL_RESULTS[@]}"}"
fi
echo "═══════════════════════════════════════════════════════════════"
