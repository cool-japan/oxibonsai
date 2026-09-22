#!/usr/bin/env bash
# OxiBonsai — CUDA kernel-source syntax gate (F12: "31,000 lines of CUDA are
# compiled by nothing").
#
# Extracts every `pub const CUDA_<NAME>: &str = r#"..."#;` kernel-source
# string found anywhere under
# `crates/oxibonsai-kernels/src/gpu_backend/` (recursive — this includes
# `gpu_backend/kernel_sources/archive.rs`, whose 11 `CUDA_*` constants do
# NOT end in `_SRC` and are NOT direct children of `gpu_backend/`; an
# earlier version of this script only globbed top-level `cuda*.rs` files
# and silently checked 13 of the 24 real CUDA_* kernel-source constants
# while printing "PASS: all 13" — see the post-extraction count guard
# below, which turns any future recurrence of that narrowing into a loud
# failure instead of a silent undercount) and runs a syntax pass over each
# one, in priority order:
#
#   1. REAL   — `nvcc --cuda` (NVIDIA CUDA Compiler), if a CUDA toolkit is
#               installed. This is an authoritative CUDA syntax+semantics
#               pass; it needs no physical GPU (nvcc's front end runs on any
#               host), only the toolkit itself.
#   2. APPROX — `clang++ -x c++ -fsyntax-only`, if tier 1 is unavailable.
#               Inline PTX `asm(...)` statements are stripped (they are not
#               valid host C++) and a small stub header defines away
#               CUDA-only builtins (`__global__`, `threadIdx`, `__shfl_*`,
#               …) so the kernel bodies parse as ordinary C++. This CANNOT
#               see CUDA-specific semantic errors (address-space rules, PTX
#               correctness, real NVCC diagnostics) — it only proves the
#               kernel bodies are not gibberish C++. Every line of output
#               this tier produces says "approximate" so it is never
#               mistaken for a real CUDA compile.
#   3. SKIPPED — neither tool is present. Reported loudly, never silently.
#
# `--release` makes tier 2 (or no tool at all) INCOMPLETE rather than a
# silent pass: an approximate check is real evidence for local development,
# but it must never stand in for a real CUDA compile in a release gate (see
# CONTEXT: "CUDA code is compile-blind; do not claim CUDA runtime
# validation"). Without `--release`, a clean tier-2 or tier-3 result exits 0
# so local iteration on a GPU-less dev machine is not blocked.
#
# Usage:
#   ./scripts/check_cuda.sh              # dev mode: approximate/skip is OK
#   ./scripts/check_cuda.sh --release    # release mode: only a real nvcc pass is a PASS
#
# Exit codes:
#   0  PASS       — a real (tier 1) check ran clean, OR (non-release mode
#                   only) tier 2 ran clean / tier 3 skipped
#   1  FAIL       — a syntax error was found (by whichever tier ran)
#   2  INCOMPLETE — no authoritative tier ran and --release was given
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

RELEASE_MODE=0
for arg in "$@"; do
    case "$arg" in
        --release) RELEASE_MODE=1 ;;
        --help|-h)
            echo "Usage: $0 [--release]"
            echo "  --release   only a real nvcc pass counts as complete (approx/skip -> exit 2)"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

SRC_DIR="crates/oxibonsai-kernels/src/gpu_backend"
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: $SRC_DIR not found (run from the repo root)." >&2
    exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_check_cuda.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "  CUDA kernel-source syntax gate"
echo "═══════════════════════════════════════════════════════════════"

# ── Stub header: defines CUDA-only builtins as inert host stand-ins so the
# kernel bodies can be parsed as plain C++ by tier 2. Also read by nvcc in
# tier 1 for the small number of `min`/`max` overloads it does not carry
# for host-side scalars in a `--cuda` frontend-only pass (harmless there:
# real device builtins from <cuda_runtime.h> still win in a real compile).
cat >"$WORK_DIR/cuda_stub.h" <<'STUBEOF'
#pragma once
#include <cmath>
#include <algorithm>
#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
#define __shared__
#define __constant__ const
struct uint3 { unsigned int x,y,z; };
static uint3 threadIdx, blockIdx, blockDim, gridDim;
static inline void __syncthreads() {}
template<typename T> static inline T __shfl_down_sync(unsigned, T v, unsigned, int=32){return v;}
template<typename T> static inline T __shfl_sync(unsigned, T v, unsigned, int=32){return v;}
template<typename T> static inline T __shfl_xor_sync(unsigned, T v, unsigned, int=32){return v;}
static inline float __int_as_float(int i){ float f; __builtin_memcpy(&f,&i,4); return f; }
static inline float __uint_as_float(unsigned i){ float f; __builtin_memcpy(&f,&i,4); return f; }
static inline int __float_as_int(float f){ int i; __builtin_memcpy(&i,&f,4); return i; }
template<typename T> static inline T __ldg(const T* p){ return *p; }
static inline float __fmaf_rn(float a,float b,float c){return a*b+c;}
static inline float __expf(float x){return expf(x);}
static inline float rsqrtf(float x){return 1.0f/sqrtf(x);}
static inline double rsqrt(double x){return 1.0/sqrt(x);}
static inline float rsqrt(float x){return 1.0f/sqrtf(x);}
static inline float __fdividef(float a,float b){return a/b;}
static inline unsigned int __popc(unsigned int x){return __builtin_popcount(x);}
template<typename T> static inline T atomicAdd(T* p,T v){T o=*p;*p+=v;return o;}
static inline unsigned int min(unsigned int a,unsigned int b){return a<b?a:b;}
static inline unsigned int max(unsigned int a,unsigned int b){return a>b?a:b;}
static inline int min(int a,int b){return a<b?a:b;}
static inline int max(int a,int b){return a>b?a:b;}
// `__CUDACC__` is defined by nvcc for every .cu translation unit it compiles
// (including tier 1's `--cuda` frontend-only pass), and nvcc's own implicit
// prelude already defines the real `float4`/`uint4`/`int4` vector types (via
// its auto-included `vector_types.h`) plus `__half` once a kernel includes
// `<cuda_fp16.h>`. Unlike the macro/overload shims above — which the header
// comment already documents as harmless no-ops under a real nvcc compile —
// these are `struct` type definitions: a *second*, stub definition of a name
// nvcc's own headers already declare is an ODR/redefinition error, not a
// harmless shadow. Gating them out under `__CUDACC__` keeps tier 1 (the
// authoritative check) exercising nvcc's real types/headers unchanged, and
// keeps a genuinely missing `#include <cuda_fp16.h>` in a kernel a real,
// visible nvcc failure instead of one this stub quietly papers over; they
// only materialize for tier 2's host C++ compiler, which has no CUDA headers
// at all and needs *some* definition to parse kernel bodies that use them.
#if !defined(__CUDACC__)
struct float4 { float x, y, z, w; };
static inline float4 make_float4(float x, float y, float z, float w){ float4 r{x,y,z,w}; return r; }
struct uint4 { unsigned int x, y, z, w; };
static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w){ uint4 r{x,y,z,w}; return r; }
struct int4 { int x, y, z, w; };
static inline int4 make_int4(int x, int y, int z, int w){ int4 r{x,y,z,w}; return r; }
struct __half { unsigned short bits; };
static inline float __half2float(__half h){ (void)h; return 0.0f; }
static inline __half __float2half(float f){ (void)f; return __half{}; }
#endif
STUBEOF

# ── 1. Extract every `pub const CUDA_<NAME>: &str = r#"…"#;` from every
#      `.rs` file anywhere under gpu_backend/ (recursive — see header),
#      stripping inline `asm(...)` statements (not valid host C++) as it goes.
#      A second, looser pass double-checks that nothing declared as a
#      `pub const CUDA_*` failed to be isolated by the strict raw-string
#      pattern (e.g. a future kernel written with `r##"…"##;` instead of
#      `r#"…"#;`) — a mismatch there is exactly the "prints PASS while
#      quietly ignoring N kernels" failure shape this gate exists to catch,
#      so it is a hard, named failure, not another silent narrowing.
EXTRACT_LOG="$WORK_DIR/extract.log"
if ! python3 - "$SRC_DIR" "$WORK_DIR" >"$EXTRACT_LOG" 2>&1 <<'PYEOF'
import re, sys, pathlib

src_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])

# Strict: only names whose raw-string body this extractor can actually
# isolate (single-hash `r#"..."#;` raw strings — the convention every
# CUDA_* kernel-source constant in this codebase uses today).
strict_pat = re.compile(r'pub const (CUDA_[A-Z0-9_]+)\s*:\s*&str\s*=\s*r#"(.*?)"#;', re.S)

# Loose: matches the declaration itself regardless of raw-string
# hash-delimiter count, purely to detect when the strict pattern above
# has missed a real CUDA_* constant.
loose_pat = re.compile(r'pub const (CUDA_[A-Z0-9_]+)\s*:\s*&str\s*=')

def strip_asm(text: str) -> str:
    out = []
    i = 0
    while True:
        j = text.find('asm', i)
        if j < 0:
            out.append(text[i:])
            break
        # must be a standalone token, not e.g. "basmear", "asmith" or "asm_flag"
        before_ok = j > 0 and (text[j - 1].isalnum() or text[j - 1] == '_')
        after = text[j + 3:j + 4]
        after_ok = after.isalnum() or after == '_'
        if before_ok or after_ok:
            out.append(text[i:j + 3])
            i = j + 3
            continue
        out.append(text[i:j])
        k = text.find('(', j)
        if k < 0:
            out.append(text[j:])
            break
        depth = 0
        m = k
        while m < len(text):
            if text[m] == '(':
                depth += 1
            elif text[m] == ')':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        n = m + 1
        while n < len(text) and text[n] in ' \t\r\n':
            n += 1
        if n < len(text) and text[n] == ';':
            n += 1
        out.append('/*asm*/;')
        i = n
    return ''.join(out)

declared = set()
extracted = set()
n = 0
for f in sorted(src_dir.rglob("*.rs")):
    text = f.read_text()
    for m in loose_pat.finditer(text):
        declared.add(m.group(1))
    for m in strict_pat.finditer(text):
        name, body = m.group(1), m.group(2)
        stripped = strip_asm(body)
        out_path = out_dir / f"{f.stem}__{name}.cu"
        out_path.write_text('#include "cuda_stub.h"\n' + stripped)
        extracted.add(name)
        n += 1

missing = sorted(declared - extracted)
if missing:
    print(f"ERROR: {len(missing)} CUDA_* constant(s) are declared under {src_dir} but", file=sys.stderr)
    print("were NOT extracted by the strict single-hash-raw-string pattern (likely a", file=sys.stderr)
    print("different raw-string delimiter or unusual formatting). Fix the extractor —", file=sys.stderr)
    print("do not let this silently narrow the syntax gate again:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}", file=sys.stderr)
    sys.exit(1)

print(f"extracted {n} CUDA_* kernel-source constant(s)")
PYEOF
then
    cat "$EXTRACT_LOG" >&2
    echo "ERROR: CUDA kernel-source extraction failed (see above)." >&2
    exit 2
fi
cat "$EXTRACT_LOG"

CU_FILES=("$WORK_DIR"/*.cu)
if [[ ! -e "${CU_FILES[0]}" ]]; then
    echo "ERROR: no CUDA_* constants extracted from $SRC_DIR — check the extraction regex" \
         "against the current source layout." >&2
    exit 2
fi
echo "  ${#CU_FILES[@]} kernel-source file(s) extracted to a scratch dir."
echo ""

# ── 2. Pick a checker tier ───────────────────────────────────────────────
FAIL=0
declare -a FAILED_FILES=()

if command -v nvcc >/dev/null 2>&1; then
    echo "── tier 1 (REAL): nvcc --cuda ──"
    NVCC_VERSION="$(nvcc --version 2>&1 | tail -1)"
    echo "  using: $(command -v nvcc)  ($NVCC_VERSION)"
    for f in "${CU_FILES[@]}"; do
        base="$(basename "$f")"
        if nvcc --cuda -I "$WORK_DIR" "$f" -o "$WORK_DIR/${base%.cu}.cu.cpp.ii" >"$WORK_DIR/${base}.log" 2>&1; then
            echo "  OK    $base"
        else
            FAIL=1
            FAILED_FILES+=("$base")
            echo "  FAIL  $base"
            sed 's/^/        /' "$WORK_DIR/${base}.log"
        fi
    done
    TIER="real"
elif command -v clang++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1; then
    CXX="$(command -v clang++ || command -v g++)"
    echo "── tier 2 (APPROXIMATE — NOT a real CUDA compile): $CXX -fsyntax-only ──"
    echo "  No nvcc/NVRTC toolchain found. This tier only proves the kernel bodies"
    echo "  are syntactically valid C++ once CUDA builtins are stubbed out; it"
    echo "  cannot catch CUDA-specific semantic errors or verify PTX asm blocks"
    echo "  (which are stripped, not checked). Do not read a clean run here as"
    echo "  'the CUDA compiles'."
    for f in "${CU_FILES[@]}"; do
        base="$(basename "$f")"
        if "$CXX" -x c++ -std=c++14 -fsyntax-only -Wall -I "$WORK_DIR" "$f" >"$WORK_DIR/${base}.log" 2>&1; then
            echo "  approx-OK    $base"
        else
            FAIL=1
            FAILED_FILES+=("$base")
            echo "  approx-FAIL  $base"
            sed 's/^/        /' "$WORK_DIR/${base}.log"
        fi
    done
    TIER="approx"
else
    echo "── tier 3: SKIPPED ──"
    echo "  Neither nvcc nor a C++ compiler (clang++/g++) was found on PATH."
    echo "  ${#CU_FILES[@]} CUDA_* kernel-source constant(s) were NOT syntax-checked."
    TIER="skipped"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [[ "$FAIL" -ne 0 ]]; then
    echo "FAIL: ${#FAILED_FILES[@]} kernel-source file(s) failed the $TIER syntax check:"
    printf '  - %s\n' "${FAILED_FILES[@]}"
    exit 1
fi

case "$TIER" in
    real)
        echo "PASS: all ${#CU_FILES[@]} kernel-source file(s) passed a real nvcc --cuda syntax check."
        exit 0
        ;;
    approx)
        if [[ "$RELEASE_MODE" -eq 1 ]]; then
            echo "INCOMPLETE (--release): only the approximate clang++/g++ check ran clean;"
            echo "no real CUDA toolchain (nvcc) was available. A release gate requires the"
            echo "authoritative check. Run this on a host with the CUDA toolkit installed."
            exit 2
        fi
        echo "PASS (approximate): all ${#CU_FILES[@]} kernel-source file(s) parsed as valid C++"
        echo "once CUDA builtins were stubbed. This is NOT a substitute for a real nvcc pass."
        exit 0
        ;;
    skipped)
        if [[ "$RELEASE_MODE" -eq 1 ]]; then
            echo "INCOMPLETE (--release): no CUDA toolchain and no C++ compiler were found;"
            echo "${#CU_FILES[@]} kernel-source constant(s) were not checked at all."
            exit 2
        fi
        echo "SKIPPED: no CUDA toolchain and no C++ compiler were found;"
        echo "${#CU_FILES[@]} kernel-source constant(s) were not checked at all."
        echo "(non-release mode: not treated as a failure, but genuinely incomplete)"
        exit 0
        ;;
esac
