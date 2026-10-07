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
#               pass; it needs no physical GPU, only the toolkit itself.
#               It is NOT front-end only: `nvcc --cuda --dryrun` (CUDA 12.0)
#               shows cudafe++ for the host side plus a full device compile
#               for nvcc's default arch (cicc -arch compute_52, ptxas
#               -arch=sm_52, fatbinary). The stub header below expands to
#               nothing under nvcc, and inline PTX `asm(...)` is stripped at
#               extraction for BOTH tiers, so ptxas never sees inline PTX.
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
# `--accept-approximate` is the one explicit, named opt-out of that rule, for a
# release host that has no CUDA toolkit (every macOS host; the CUDA backend is
# feature-gated off by default and is documented as not validated on hardware).
# It only matters together with `--release`, and it only ever accepts ONE
# situation: tier 1 is unavailable, tier 2 ran on EVERY extracted kernel source
# and ALL of them parsed clean. Then the run prints a loud banner naming what
# was and was not checked and exits 0 (CUDA_SYNTAX_RESULT=approximate-accepted).
# It never accepts a syntax error (exit 1, with or without the flag), never
# accepts a run in which no checker ran (exit 2: nothing was checked), and never
# lets a failing nvcc fall back to the approximate tier. Without `--release` it
# is accepted and changes nothing.
#
# Usage:
#   ./scripts/check_cuda.sh                                # dev mode: approximate/skip is OK
#   ./scripts/check_cuda.sh --release                      # release mode: only a real nvcc pass is a PASS
#   ./scripts/check_cuda.sh --release --accept-approximate # release mode, no nvcc: a clean approximate
#                                                          # run is accepted, loudly (see above)
#
# Exit codes:
#   0  PASS       — a real (tier 1) check ran clean, OR (non-release mode
#                   only) tier 2 ran clean / tier 3 skipped, OR (--release
#                   --accept-approximate only) tier 2 ran clean on every kernel
#                   source
#   1  FAIL       — a syntax error was found (by whichever tier ran)
#   2  INCOMPLETE — no authoritative tier ran and --release was given (and,
#                   for tier 2, --accept-approximate was not)
#
# Machine-readable verdict: every run that reaches a verdict ends its stdout
# with exactly one line `CUDA_SYNTAX_RESULT=<value>` (scripts/ci.sh reads it to
# label the stage; never infer the tier from the exit code alone):
#   nvcc-ok               a real nvcc pass (tier 1)
#   approximate-accepted  --release --accept-approximate, tier 2 clean (the waiver)
#   approximate-ok        dev mode, tier 2 clean
#   skipped               dev mode, no tool ran
#   incomplete            --release, nothing authoritative ran (exit 2)
#   failed                a syntax error (exit 1)
#
# Tool selection is by PATH: nvcc for tier 1, then clang++, then g++ for tier 2.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

RELEASE_MODE=0
ACCEPT_APPROXIMATE=0
for arg in "$@"; do
    case "$arg" in
        --release) RELEASE_MODE=1 ;;
        --accept-approximate) ACCEPT_APPROXIMATE=1 ;;
        --help|-h)
            echo "Usage: $0 [--release] [--accept-approximate]"
            echo "  --release             only a real nvcc pass counts as complete (approx/skip -> exit 2)"
            echo "  --accept-approximate  with --release and no nvcc: accept a CLEAN approximate (clang++/g++)"
            echo "                        run as the verdict, loudly (CUDA_SYNTAX_RESULT=approximate-accepted)."
            echo "                        Never accepts an error or a run in which no checker ran;"
            echo "                        accepted and ignored without --release."
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
    echo "CUDA_SYNTAX_RESULT=incomplete"
    exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_check_cuda.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "═══════════════════════════════════════════════════════════════"
echo "  CUDA kernel-source syntax gate"
echo "═══════════════════════════════════════════════════════════════"

# ── Stub header: defines CUDA-only builtins as inert host stand-ins so the
# kernel bodies can be parsed as plain C++ by tier 2. Every extracted file
# includes it in BOTH tiers, but its whole body sits inside
# `#if !defined(__CUDACC__)`, so under nvcc (tier 1) it expands to nothing.
# nvcc defines `__CUDACC__` and force-includes the real <cuda_runtime.h>
# (`-include cuda_runtime.h` in `nvcc --cuda --dryrun`), which already
# declares every name this stub provides. A second, host-side definition of
# any of them is NOT harmless under nvcc: before this gating, every one of
# the 31 kernel sources failed tier 1 inside this header (blocker B2, first
# real-nvcc run, CUDA 12.0) with redefinition errors for `uint3` and
# threadIdx/blockIdx/blockDim/gridDim, "function rsqrt(float) has already
# been defined", "cannot overload functions distinguished by return type
# alone" (`__popc`) and the int/unsigned `min`/`max` overloads, and the empty
# `__global__`/`__device__` macros turned every kernel into a host function
# ("calling a __device__ function from a __host__ function is not allowed").
# Any new shim must go inside the guard too.
cat >"$WORK_DIR/cuda_stub.h" <<'STUBEOF'
#pragma once
// Tier-2-only host shims: the ENTIRE body is gated on `!defined(__CUDACC__)`.
// nvcc defines `__CUDACC__` for every .cu translation unit it compiles
// (including tier 1's `--cuda` pass) and force-includes <cuda_runtime.h>,
// which already provides every name below: the execution-space and
// `__restrict__`/`__forceinline__` macros (crt/host_defines.h), `uint3`,
// `dim3` and the `float4`/`uint4`/`int4` vector types (vector_types.h),
// threadIdx/blockIdx/blockDim/gridDim (device_launch_parameters.h), the
// warp, atomic, conversion and fast-math intrinsics, and the `rsqrt` and
// int/unsigned `min`/`max` overloads (crt/math_functions.h{,pp}). A second,
// stub definition of any of them is a redefinition / overload error under
// nvcc, never a harmless shadow, so tier 1 (the authoritative check) sees
// only nvcc's real headers. `__half` is gated for a different reason:
// <cuda_runtime.h> does NOT declare it (only <cuda_fp16.h> does), so a kernel
// that uses `__half` without including <cuda_fp16.h> must stay a real,
// visible nvcc failure instead of one this stub quietly papers over.
// These definitions only ever materialize for tier 2's host C++ compiler,
// which has no CUDA headers at all and needs *some* definition to parse
// kernel bodies that use them.
#if !defined(__CUDACC__)
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
// glibc's <math.h> already declares `extern float __expf(float)` (its
// __MATHCALL macro declares every libm function a second time under a `__`
// prefix), so on a glibc host this static shim was a hard error in every
// file ("static declaration of '__expf' follows non-static declaration")
// and tier 2 never passed on Linux. Kernel calls then parse against glibc's
// own prototype; macOS and other non-glibc hosts keep the shim unchanged.
#if !defined(__GLIBC__)
static inline float __expf(float x){return expf(x);}
#endif
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
    echo "CUDA_SYNTAX_RESULT=incomplete"
    exit 2
fi
cat "$EXTRACT_LOG"

CU_FILES=("$WORK_DIR"/*.cu)
if [[ ! -e "${CU_FILES[0]}" ]]; then
    echo "ERROR: no CUDA_* constants extracted from $SRC_DIR — check the extraction regex" \
         "against the current source layout." >&2
    echo "CUDA_SYNTAX_RESULT=incomplete"
    exit 2
fi
echo "  ${#CU_FILES[@]} kernel-source file(s) extracted to a scratch dir."
echo ""

# ── 2. Pick a checker tier ───────────────────────────────────────────────
FAIL=0
APPROX_OK=0   # kernel sources the approximate tier parsed clean (the waiver counts these)
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
    # A kernel that uses `__half` must `#include <cuda_fp16.h>` (tier 1 keeps a
    # missing include a real nvcc failure). The host compiler has no CUDA
    # headers, so give that include an empty stand-in here; `__half` itself
    # comes from cuda_stub.h, which every extracted file includes first. It is
    # written ONLY in this tier-2 branch: in tier 1 the same `-I "$WORK_DIR"`
    # would otherwise shadow nvcc's real <cuda_fp16.h>.
    printf '#pragma once\n// tier-2 stand-in: __half is shimmed in cuda_stub.h\n' >"$WORK_DIR/cuda_fp16.h"
    for f in "${CU_FILES[@]}"; do
        base="$(basename "$f")"
        if "$CXX" -x c++ -std=c++14 -fsyntax-only -Wall -I "$WORK_DIR" "$f" >"$WORK_DIR/${base}.log" 2>&1; then
            echo "  approx-OK    $base"
            APPROX_OK=$((APPROX_OK + 1))
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
    echo "CUDA_SYNTAX_RESULT=failed"
    exit 1
fi

case "$TIER" in
    real)
        echo "PASS: all ${#CU_FILES[@]} kernel-source file(s) passed a real nvcc --cuda syntax check."
        if [[ "$RELEASE_MODE" -eq 1 && "$ACCEPT_APPROXIMATE" -eq 1 ]]; then
            echo "NOTE: --accept-approximate was given, but a real nvcc check ran: the waiver was"
            echo "not needed and is not used."
        fi
        echo "CUDA_SYNTAX_RESULT=nvcc-ok"
        exit 0
        ;;
    approx)
        if [[ "$RELEASE_MODE" -eq 1 ]]; then
            # The waiver accepts exactly one situation: the approximate tier parsed
            # EVERY extracted kernel source clean (a failure was already exit 1 above;
            # the count below is the same guarantee, stated as a number).
            if [[ "$ACCEPT_APPROXIMATE" -eq 1 && "$APPROX_OK" -eq "${#CU_FILES[@]}" ]]; then
                echo "APPROXIMATE, ACCEPTED BY FLAG: $APPROX_OK kernel sources parsed as C++ with CUDA builtins stubbed; this is NOT an nvcc validation; the CUDA backend is not certified by this run"
                echo "  --release --accept-approximate was given and no nvcc was available, so a clean"
                echo "  approximate ($CXX -fsyntax-only) run stands in for the authoritative check."
                echo "  CHECKED:     each of the $APPROX_OK CUDA_* kernel sources is valid C++ once the CUDA-only"
                echo "               builtins are stubbed out."
                echo "  NOT CHECKED: CUDA semantics (address spaces, launch bounds, the real signatures of"
                echo "               warp/atomic intrinsics), inline PTX asm (stripped, not parsed), the"
                echo "               host-side launch code, and any execution on a CUDA device."
                echo "  State in the release notes that the CUDA backend's kernels were syntax-checked"
                echo "  approximately and are not hardware-validated."
                echo "CUDA_SYNTAX_RESULT=approximate-accepted"
                exit 0
            fi
            echo "INCOMPLETE (--release): only the approximate clang++/g++ check ran clean;"
            echo "no real CUDA toolchain (nvcc) was available. A release gate requires the"
            echo "authoritative check. Run this on a host with the CUDA toolkit installed."
            echo "CUDA_SYNTAX_RESULT=incomplete"
            exit 2
        fi
        echo "PASS (approximate): all ${#CU_FILES[@]} kernel-source file(s) parsed as valid C++"
        echo "once CUDA builtins were stubbed. This is NOT a substitute for a real nvcc pass."
        echo "CUDA_SYNTAX_RESULT=approximate-ok"
        exit 0
        ;;
    skipped)
        if [[ "$RELEASE_MODE" -eq 1 ]]; then
            echo "INCOMPLETE (--release): no CUDA toolchain and no C++ compiler were found;"
            echo "${#CU_FILES[@]} kernel-source constant(s) were not checked at all."
            if [[ "$ACCEPT_APPROXIMATE" -eq 1 ]]; then
                echo "(--accept-approximate cannot waive this: nothing ran.)"
            fi
            echo "CUDA_SYNTAX_RESULT=incomplete"
            exit 2
        fi
        echo "SKIPPED: no CUDA toolchain and no C++ compiler were found;"
        echo "${#CU_FILES[@]} kernel-source constant(s) were not checked at all."
        echo "(non-release mode: not treated as a failure, but genuinely incomplete)"
        echo "CUDA_SYNTAX_RESULT=skipped"
        exit 0
        ;;
esac
