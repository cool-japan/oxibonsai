#!/usr/bin/env bash
# OxiBonsai — regenerate the Bonsai 2 27B golden set from the PrismML
# llama.cpp fork.
#
# This script is DOCUMENTED, not run by CI or by any package's gate: it
# needs a locally built PrismML llama.cpp fork (branch `prism`) and the
# real, multi-GB `Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf` files, neither
# of which this repository ships or can assume are present on a build
# machine. It exists so the exact procedure that produced
# `crates/oxibonsai-model/tests/fixtures/bonsai2_golden/` (and the vendored
# CPU-backend spread comparison under `bonsai2_golden_cpu/`) is reproducible
# from the repository rather than living only in one contributor's shell
# history, and so a future re-capture (e.g. after a genuine upstream model
# or fork change) can be run with the same prompts and settings.
#
# What each stage captures, against the fork's `llama-completion` /
# `llama-server` (`/completion`, `/tokenize`, `/apply-template`,
# `/v1/chat/completions`):
#   - Metal (`-ngl 99`), for both quant bands (PQ2_0, PTQ1_0): the 32-token
#     raw greedy continuation + its prompt-token dump (`llama-completion`),
#     and the first 24 steps' top-10 logprobs (`llama-server /completion`,
#     PQ2_0 only — design SS7.0.2: the fork's two bands agree byte for byte).
#   - `tokenize.json`: 5 texts (incl. an emoji/CJK/whitespace mix and a
#     `<think>`-tagged chat turn) tokenized with `add_special=False`.
#   - `apply_template.json`: 5 chat-template rendering cases (plain,
#     `enable_thinking=False`, `reasoning_effort=low`, a multi-turn history
#     with `reasoning_content`, and a tool-call schema).
#   - `chat.prompt1.server.json`: one full `/v1/chat/completions` round trip
#     with `logprobs`/`top_logprobs`, for the reasoning-split gate.
#   - CPU (`-ngl 0`), PQ2_0 only, same 3 prompts/settings as the Metal
#     server leg: an independent oracle to bound the fork's own
#     Metal-vs-CPU numeric spread (the number this project's gates compare
#     their own CPU/Metal spread against, not just each other).
#
# Usage:
#   OXIBONSAI_FORK_BUILD_DIR=/path/to/prism-llama.cpp/build/bin \
#   OXIBONSAI_MODELS_DIR=/path/to/models \
#   OXIBONSAI_GOLDEN_OUT_DIR=/path/to/output/golden2 \
#     ./scripts/bonsai2_golden.sh metal
#
#   OXIBONSAI_FORK_BUILD_DIR=... OXIBONSAI_MODELS_DIR=... \
#   OXIBONSAI_GOLDEN_CPU_OUT_DIR=/path/to/output/golden_cpu \
#     ./scripts/bonsai2_golden.sh cpu
#
# Neither output directory defaults into this repository: copy the files
# you want to keep into `crates/oxibonsai-model/tests/fixtures/bonsai2_golden{,_cpu}/`
# by hand, after reviewing the diff, exactly as any other vendored fixture
# update would be reviewed.
#
# Requires on PATH: the fork's own `llama-completion`/`llama-server`
# binaries (at `$OXIBONSAI_FORK_BUILD_DIR`), `curl`, `python3`.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage: bonsai2_golden.sh <metal|cpu|help>

  metal   Capture the Metal-backend (-ngl 99) golden set: both quant bands'
          raw greedy continuations + prompt-token dumps, PQ2_0's per-step
          server logprobs, tokenize.json, apply_template.json and
          chat.prompt1.server.json. Writes to $OXIBONSAI_GOLDEN_OUT_DIR.

  cpu     Capture the CPU-backend (-ngl 0) PQ2_0 server logprobs for the
          same 3 prompts, and print the Metal-vs-CPU spread against an
          already-captured 'metal' run at $OXIBONSAI_GOLDEN_OUT_DIR. Writes
          to $OXIBONSAI_GOLDEN_CPU_OUT_DIR.

  help    Show this message and exit.

Required environment variables (no path in this script defaults to
anything under this repository or any session-specific location):
  OXIBONSAI_FORK_BUILD_DIR   Directory holding the PrismML llama.cpp fork's
                             built 'llama-completion'/'llama-server'
                             binaries (branch 'prism').
  OXIBONSAI_MODELS_DIR       Directory holding
                             Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf.
  OXIBONSAI_GOLDEN_OUT_DIR   Output directory for 'metal' (also read by
                             'cpu' for the spread comparison).

Optional:
  OXIBONSAI_GOLDEN_CPU_OUT_DIR   Output directory for 'cpu'
                                 (default: alongside OXIBONSAI_GOLDEN_OUT_DIR,
                                 as a 'golden_cpu' sibling).
  OXIBONSAI_GOLDEN_PORT          Server port for 'metal' (default 18089).
  OXIBONSAI_GOLDEN_CPU_PORT      Server port for 'cpu' (default 18090).
USAGE_EOF
}

require_dir() {
    local var_name="$1" value="${!1:-}"
    if [[ -z "$value" ]]; then
        echo "ERROR: $var_name must be set (see --help)." >&2
        exit 2
    fi
    if [[ ! -d "$value" ]]; then
        echo "ERROR: $var_name=$value is not a directory." >&2
        exit 2
    fi
}

wait_for_health() {
    local port="$1" tries="$2"
    for _ in $(seq 1 "$tries"); do
        curl -s "http://127.0.0.1:$port/health" | grep -q '"ok"' && return 0
        sleep 2
    done
    echo "ERROR: server on port $port never reported healthy." >&2
    return 1
}

# One raw-prompt / greedy-completion capture pair per (model, prompt): the
# `llama-completion` text dump plus its own prompt-token dump (parsed from
# its `--verbose-prompt --log-file` output).
capture_completion() {
    local bin="$1" model_path="$2" model_label="$3" prompt="$4" index="$5" out_dir="$6"
    "$bin/llama-completion" -m "$model_path" -p "$prompt" -n 32 --temp 0 --top-k 1 -ngl 99 \
        --no-warmup --simple-io --verbose-prompt -c 4096 -s 42 -no-cnv --no-display-prompt \
        --log-file "$out_dir/$model_label.prompt$index.log" \
        >"$out_dir/$model_label.prompt$index.txt" 2>"$out_dir/$model_label.prompt$index.err"
    grep -E "[0-9]+ -> '" "$out_dir/$model_label.prompt$index.log" | head -80 \
        >"$out_dir/$model_label.prompt$index.prompt_tokens.txt"
}

capture_server_completion() {
    local prompt="$1" out_file="$2" port="$3"
    python3 - "$prompt" "$out_file" "$port" <<'PY'
import json, sys, urllib.request

prompt, out_file, port = sys.argv[1], sys.argv[2], sys.argv[3]
req = {
    "prompt": prompt, "n_predict": 24, "temperature": 0, "top_k": 1,
    "n_probs": 10, "seed": 42, "cache_prompt": False,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/completion",
    data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=7200) as resp:
    data = json.load(resp)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)
print("content:", repr(data.get("content")))
PY
}

run_metal() {
    require_dir OXIBONSAI_FORK_BUILD_DIR
    require_dir OXIBONSAI_MODELS_DIR
    require_dir OXIBONSAI_GOLDEN_OUT_DIR
    local bin="$OXIBONSAI_FORK_BUILD_DIR" models="$OXIBONSAI_MODELS_DIR" out="$OXIBONSAI_GOLDEN_OUT_DIR"
    local port="${OXIBONSAI_GOLDEN_PORT:-18089}"
    local prompts=(
        "The capital of Japan is"
        "def fibonacci(n):"
        "Once upon a time, in a small village by the sea,"
    )

    for model in Ternary-Bonsai-2-27B-PQ2_0 Ternary-Bonsai-2-27B-PTQ1_0; do
        local i=0
        for prompt in "${prompts[@]}"; do
            i=$((i + 1))
            echo "=== $model prompt$i: $prompt ==="
            capture_completion "$bin" "$models/$model.gguf" "$model" "$prompt" "$i" "$out"
            cat "$out/$model.prompt$i.txt"
        done
    done

    "$bin/llama-server" -m "$models/Ternary-Bonsai-2-27B-PQ2_0.gguf" -ngl 99 -c 4096 \
        --port "$port" --host 127.0.0.1 --no-warmup --jinja >"$out/server.log" 2>&1 &
    local server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null' EXIT
    wait_for_health "$port" 240 || exit 1

    local i=0
    for prompt in "${prompts[@]}"; do
        i=$((i + 1))
        capture_server_completion "$prompt" "$out/PQ2_0.prompt$i.server.json" "$port"
    done

    python3 - "$out/tokenize.json" "$port" <<'PY'
import json, sys, urllib.request

out_file, port = sys.argv[1], sys.argv[2]
texts = [
    "The capital of Japan is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "Hello, world! 日本語のテキストです。 \U0001f363 test123 can't won't  double  space\n\nnewlines\tTab",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
]
results = {}
for text in texts:
    req = {"content": text, "add_special": False, "with_pieces": True}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as resp:
        results[text] = json.load(resp)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
for text, value in results.items():
    ids = [x["id"] if isinstance(x, dict) else x for x in value["tokens"]][:40]
    print(repr(text[:40]), "->", ids)
PY

    python3 - "$out/apply_template.json" "$port" <<'PY'
import json, sys, urllib.request

out_file, port = sys.argv[1], sys.argv[2]
cases = [
    {"messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}]},
    {"messages": [{"role": "user", "content": "What is 2+2?"}],
     "chat_template_kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "What is 2+2?"}],
     "chat_template_kwargs": {"reasoning_effort": "low"}},
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "reasoning_content": "user greets"},
        {"role": "user", "content": "Bye"},
    ]},
    {"messages": [{"role": "user", "content": "weather?"}],
     "tools": [{"type": "function", "function": {
         "name": "get_weather", "description": "Get weather",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                        "required": ["city"]}}}]},
]
results = []
for case in cases:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/apply-template",
        data=json.dumps(case).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            data = json.load(resp)
        results.append({"input": case, "output": data})
        print(repr(data.get("prompt", "")[:400]))
    except Exception as e:  # noqa: BLE001 - this is a capture script, not production code
        results.append({"input": case, "error": str(e)})
        print("ERR", e)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
PY

    python3 - "$out/chat.prompt1.server.json" "$port" <<'PY'
import json, sys, urllib.request

out_file, port = sys.argv[1], sys.argv[2]
req = {
    "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
    "max_tokens": 48, "temperature": 0, "top_k": 1, "seed": 42,
    "logprobs": True, "top_logprobs": 5,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=3600) as resp:
    data = json.load(resp)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)
print(json.dumps(data["choices"][0]["message"], ensure_ascii=False)[:800])
PY

    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    trap - EXIT
    echo "metal golden set written to $out:"
    ls -la "$out"
}

run_cpu() {
    require_dir OXIBONSAI_FORK_BUILD_DIR
    require_dir OXIBONSAI_MODELS_DIR
    require_dir OXIBONSAI_GOLDEN_OUT_DIR
    local bin="$OXIBONSAI_FORK_BUILD_DIR" models="$OXIBONSAI_MODELS_DIR"
    local metal_out="$OXIBONSAI_GOLDEN_OUT_DIR"
    local out="${OXIBONSAI_GOLDEN_CPU_OUT_DIR:-$metal_out/../golden_cpu}"
    mkdir -p "$out"
    local port="${OXIBONSAI_GOLDEN_CPU_PORT:-18090}"
    local prompts=(
        "The capital of Japan is"
        "def fibonacci(n):"
        "Once upon a time, in a small village by the sea,"
    )

    "$bin/llama-server" -m "$models/Ternary-Bonsai-2-27B-PQ2_0.gguf" -ngl 0 -c 4096 \
        --port "$port" --host 127.0.0.1 --no-warmup --jinja >"$out/server_cpu.log" 2>&1 &
    local server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null' EXIT
    wait_for_health "$port" 300 || exit 1

    local i=0
    for prompt in "${prompts[@]}"; do
        i=$((i + 1))
        capture_server_completion "$prompt" "$out/PQ2_0.prompt$i.server.json" "$port"
    done

    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null
    trap - EXIT

    python3 - "$metal_out" "$out" <<'PY'
import json, sys

metal_dir, cpu_dir = sys.argv[1], sys.argv[2]
for i in (1, 2, 3):
    metal = json.load(open(f"{metal_dir}/PQ2_0.prompt{i}.server.json", encoding="utf-8"))
    cpu = json.load(open(f"{cpu_dir}/PQ2_0.prompt{i}.server.json", encoding="utf-8"))
    metal_ids = [t["id"] for t in metal["completion_probabilities"]]
    cpu_ids = [t["id"] for t in cpu["completion_probabilities"]]
    print(f"prompt{i}: greedy ids identical={metal_ids == cpu_ids} "
          f"metal_n={len(metal_ids)} cpu_n={len(cpu_ids)}")
    worst, worst_at, order_mismatch = 0.0, None, 0
    for step, (a, b) in enumerate(zip(metal["completion_probabilities"], cpu["completion_probabilities"])):
        ma = {x["id"]: x["logprob"] for x in a["top_logprobs"]}
        cb = {x["id"]: x["logprob"] for x in b["top_logprobs"]}
        if [x["id"] for x in a["top_logprobs"]] != [x["id"] for x in b["top_logprobs"]]:
            order_mismatch += 1
        common = set(ma) & set(cb)
        delta = max((abs(ma[k] - cb[k]) for k in common), default=0.0)
        print(f"  step {step:2d} tok {a['id']:>7} |dlogprob| top-10 max={delta:.3e} "
              f"(common {len(common)}/10)")
        if delta > worst:
            worst, worst_at = delta, step
    print(f"  => worst={worst:.3e} at step {worst_at}; "
          f"top-10 order mismatches={order_mismatch}/{len(metal_ids)}")
PY
    echo "cpu golden set written to $out"
}

case "${1:-help}" in
    metal) run_metal ;;
    cpu) run_cpu ;;
    help|--help|-h) usage ;;
    *)
        echo "ERROR: unknown mode '${1:-}'." >&2
        usage
        exit 2
        ;;
esac
