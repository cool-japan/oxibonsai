#!/usr/bin/env bash
#
# benchmark.sh - Compare OxiBonsai vs llama.cpp performance
# Usage: ./scripts/benchmark.sh
#

OXIBONSAI="./target/release/oxibonsai"
# Optional llama.cpp comparison; point LLAMA_CLI at your llama-cli binary
# (skipped automatically if not found / not executable).
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"
MODEL="models/Bonsai-8B.gguf"
TOKENIZER="models/tokenizer.json"
TOKENS=50

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# Prompts
PROMPT_NAMES=("Short" "Medium" "Long")
PROMPTS=(
    "Explain 2+2 = ??"
    "Write a detailed explanation of how transformer neural networks work, including self-attention mechanism"
    "Write a comprehensive guide to Rust programming language covering ownership, borrowing, lifetimes, traits, generics, error handling, async programming, and macros"
)

# Results arrays
declare -a OXI_TOKS
declare -a LLAMA_EVAL_TOKS
declare -a LLAMA_PROMPT_TOKS
declare -a OXI_STATUS
declare -a LLAMA_STATUS

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Error: Model not found at ${MODEL}${RESET}"
    echo "Download it first or place it in the models/ directory."
    exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
    echo -e "${RED}Error: Tokenizer not found at ${TOKENIZER}${RESET}"
    exit 1
fi

if [[ ! -x "$LLAMA_CLI" ]]; then
    echo -e "${YELLOW}Warning: llama-cli not found at ${LLAMA_CLI}${RESET}"
    echo "llama.cpp benchmarks will be skipped."
    SKIP_LLAMA=1
fi

# ---------------------------------------------------------------------------
# Build release if needed
# ---------------------------------------------------------------------------

if [[ ! -f "$OXIBONSAI" ]] || [[ "$(find src crates -name '*.rs' -newer "$OXIBONSAI" 2>/dev/null | head -1)" ]]; then
    echo -e "${CYAN}Building OxiBonsai (release + metal)...${RESET}"
    cargo build --release --features metal 2>&1
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Build failed!${RESET}"
        exit 1
    fi
    echo -e "${GREEN}Build complete.${RESET}"
    echo
fi

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

run_oxibonsai() {
    local prompt="$1"
    local tmpfile
    tmpfile=$(mktemp /tmp/oxibonsai_bench.XXXXXX)

    RUST_LOG=info "$OXIBONSAI" run \
        --model "$MODEL" \
        --prompt "$prompt" \
        --max-tokens "$TOKENS" \
        --tokenizer "$TOKENIZER" \
        2>&1 | tee "$tmpfile"

    # Parse: "8 prompt + 50 generated = 58 total tokens in 8.70s (5.7 tok/s)"
    local toks
    toks=$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$tmpfile" | tail -1 | grep -oE '[0-9]+\.[0-9]+')

    rm -f "$tmpfile"

    if [[ -n "$toks" ]]; then
        echo "$toks"
    else
        echo "FAIL"
    fi
}

run_llama() {
    local prompt="$1"
    local tmpfile
    tmpfile=$(mktemp /tmp/llama_bench.XXXXXX)

    "$LLAMA_CLI" \
        -m "$MODEL" \
        -ngl 99 \
        -c 4096 \
        -p "$prompt" \
        -n "$TOKENS" \
        --no-display-prompt \
        2>&1 | tee "$tmpfile"

    # Parse prompt eval: "prompt eval time = ... tokens/s)"
    local prompt_toks
    prompt_toks=$(grep "prompt eval time" "$tmpfile" | grep -oE '[0-9]+\.[0-9]+ tokens/s' | grep -oE '[0-9]+\.[0-9]+')

    # Parse eval (generation): "eval time = ... tokens/s)" — the line that does NOT say "prompt"
    local eval_toks
    eval_toks=$(grep "eval time" "$tmpfile" | grep -v "prompt" | grep -oE '[0-9]+\.[0-9]+ tokens/s' | grep -oE '[0-9]+\.[0-9]+')

    rm -f "$tmpfile"

    if [[ -n "$eval_toks" ]]; then
        echo "${eval_toks}|${prompt_toks}"
    else
        echo "FAIL|FAIL"
    fi
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

echo
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       OxiBonsai vs llama.cpp Benchmark (${TOKENS} tokens)          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo
echo -e "Model: ${CYAN}${MODEL}${RESET}"
echo -e "Date:  $(date '+%Y-%m-%d %H:%M:%S')"
echo

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    name="${PROMPT_NAMES[$i]}"

    echo -e "${BOLD}━━━ ${name} Prompt ━━━${RESET}"
    echo -e "  \"${prompt:0:80}$([ ${#prompt} -gt 80 ] && echo '...')\""
    echo

    # --- OxiBonsai ---
    echo -e "  ${GREEN}▶ OxiBonsai (Metal)${RESET}"
    oxi_result=$(run_oxibonsai "$prompt")
    if [[ "$oxi_result" == "FAIL" ]]; then
        OXI_TOKS[$i]="—"
        OXI_STATUS[$i]="FAIL"
        echo -e "  ${RED}  Failed to parse tok/s${RESET}"
    else
        OXI_TOKS[$i]="$oxi_result"
        OXI_STATUS[$i]="OK"
        echo -e "  ${GREEN}  ${oxi_result} tok/s${RESET}"
    fi
    echo

    # --- llama.cpp ---
    if [[ -z "$SKIP_LLAMA" ]]; then
        echo -e "  ${CYAN}▶ llama.cpp (Metal, ngl=99)${RESET}"
        llama_result=$(run_llama "$prompt")
        IFS='|' read -r eval_toks prompt_toks <<< "$llama_result"
        if [[ "$eval_toks" == "FAIL" ]]; then
            LLAMA_EVAL_TOKS[$i]="—"
            LLAMA_PROMPT_TOKS[$i]="—"
            LLAMA_STATUS[$i]="FAIL"
            echo -e "  ${RED}  Failed to parse tok/s${RESET}"
        else
            LLAMA_EVAL_TOKS[$i]="$eval_toks"
            LLAMA_PROMPT_TOKS[$i]="$prompt_toks"
            LLAMA_STATUS[$i]="OK"
            echo -e "  ${CYAN}  Generation: ${eval_toks} tok/s | Prompt eval: ${prompt_toks} tok/s${RESET}"
        fi
    else
        LLAMA_EVAL_TOKS[$i]="—"
        LLAMA_PROMPT_TOKS[$i]="—"
        LLAMA_STATUS[$i]="SKIP"
    fi
    echo
done

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

echo
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║                          BENCHMARK RESULTS                                  ║${RESET}"
echo -e "${BOLD}╠══════════╦═══════════════════╦═══════════════════╦═══════════════════════════╣${RESET}"
printf  "${BOLD}║ %-8s ║ %-17s ║ %-17s ║ %-25s ║${RESET}\n" \
        "Prompt" "OxiBonsai (tok/s)" "llama.cpp (tok/s)" "llama.cpp prompt (tok/s)"
echo -e "${BOLD}╠══════════╬═══════════════════╬═══════════════════╬═══════════════════════════╣${RESET}"

for i in "${!PROMPTS[@]}"; do
    name="${PROMPT_NAMES[$i]}"
    oxi="${OXI_TOKS[$i]}"
    llama_eval="${LLAMA_EVAL_TOKS[$i]}"
    llama_prompt="${LLAMA_PROMPT_TOKS[$i]}"

    # Color the comparison
    oxi_display="$oxi"
    llama_display="$llama_eval"

    printf "║ %-8s ║ %17s ║ %17s ║ %25s ║\n" \
           "$name" "$oxi_display" "$llama_display" "$llama_prompt"
done

echo -e "${BOLD}╠══════════╩═══════════════════╩═══════════════════╩═══════════════════════════╣${RESET}"

# Compute averages if we have numeric data
oxi_sum=0; oxi_count=0
llama_sum=0; llama_count=0

for i in "${!PROMPTS[@]}"; do
    if [[ "${OXI_STATUS[$i]}" == "OK" ]]; then
        oxi_sum=$(echo "$oxi_sum + ${OXI_TOKS[$i]}" | bc)
        oxi_count=$((oxi_count + 1))
    fi
    if [[ "${LLAMA_STATUS[$i]}" == "OK" ]]; then
        llama_sum=$(echo "$llama_sum + ${LLAMA_EVAL_TOKS[$i]}" | bc)
        llama_count=$((llama_count + 1))
    fi
done

if [[ $oxi_count -gt 0 ]]; then
    oxi_avg=$(echo "scale=1; $oxi_sum / $oxi_count" | bc)
else
    oxi_avg="—"
fi

if [[ $llama_count -gt 0 ]]; then
    llama_avg=$(echo "scale=1; $llama_sum / $llama_count" | bc)
else
    llama_avg="—"
fi

printf "║  Average:   OxiBonsai = %-10s  llama.cpp = %-10s               ║\n" \
       "${oxi_avg} tok/s" "${llama_avg} tok/s"

if [[ "$oxi_avg" != "—" ]] && [[ "$llama_avg" != "—" ]]; then
    ratio=$(echo "scale=1; $oxi_avg * 100 / $llama_avg" | bc)
    printf "║  OxiBonsai is at %-5s%% of llama.cpp generation speed                     ║\n" "$ratio"
fi

echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════════════╝${RESET}"
echo
echo -e "Completed at $(date '+%Y-%m-%d %H:%M:%S')"
