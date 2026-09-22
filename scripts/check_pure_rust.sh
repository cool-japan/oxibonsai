#!/usr/bin/env bash
# OxiBonsai — Pure Rust policy gate (COOLJAPAN policy: default features must
# be C/C++/Fortran-free).
#
# Greps the RESOLVED DEFAULT dependency set (i.e. exactly what
# `cargo build --workspace` pulls in — no `--all-features`, no
# `--no-default-features`) for any crate that reaches the graph through a
# `cc`, `cmake` or `bindgen` build-dependency, which are the standard Rust
# proxies for "this crate compiles or links a C/C++ toolchain". When one is
# found, prints the exact reverse-dependency path (which workspace crate, via
# which chain) so the offending manifest is obvious, and fails.
#
# As a best-effort second layer, also scans the `build.rs` of every
# default-resolved dependency for a literal `Command::new("cc"|"gcc"|...)`
# style shell-out that would evade the `cc`/`cmake`/`bindgen` crate-name
# check entirely. This is a heuristic (a crate could construct the compiler
# name dynamically to dodge it) and is reported separately from the
# authoritative crate-graph check.
#
# Usage:
#   ./scripts/check_pure_rust.sh              # check the default feature set (this crate's normal build)
#   ./scripts/check_pure_rust.sh --quiet       # only print violations / the final verdict
#
# Exit codes:
#   0  clean — no cc/cmake/bindgen reachable in the default dependency graph
#   1  violation — see stdout for the crate(s) and the path(s) that pull them in
#   2  usage / environment error (cargo itself failed, not a policy verdict)
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

QUIET=0
for arg in "$@"; do
    case "$arg" in
        --quiet) QUIET=1 ;;
        --help|-h)
            echo "Usage: $0 [--quiet]"
            echo "Fails if cc/cmake/bindgen are reachable from the default-feature dependency graph."
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log() { [[ "$QUIET" -eq 1 ]] || echo "$@"; }

log "═══════════════════════════════════════════════════════════════"
log "  Pure Rust policy check (default features, no C/C++/Fortran)"
log "═══════════════════════════════════════════════════════════════"

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found on PATH — cannot resolve the dependency graph." >&2
    exit 2
fi

BANNED_BUILD_HELPERS=(cc cmake bindgen)
VIOLATIONS=0
TREE_OUT="$(mktemp "${TMPDIR:-/tmp}/oxibonsai_pure_rust_tree.XXXXXX")"
trap 'rm -f "$TREE_OUT"' EXIT

for pkg in "${BANNED_BUILD_HELPERS[@]}"; do
    log ""
    log "── checking: is '$pkg' reachable from the default dependency graph? ──"
    # -e normal,build: exclude dev-dependencies (tests/benches never ship).
    # No --all-features / --no-default-features: exactly the default build.
    # --target all: without it, `cargo tree -i` only considers the HOST
    # platform's target-cfg'd dependencies, so a banned build-helper
    # reachable solely through a `[target.'cfg(windows)'.dependencies]` (or
    # any other non-host cfg) section is invisible here even though it is
    # genuinely part of the default-feature dependency SET this multi-
    # platform project resolves — cargo itself prints "warning: nothing to
    # print... try --target all" in exactly that situation (confirmed in
    # this workspace: 'cc' is unreachable without this flag, reachable with
    # it, via chrono's Haiku-only `iana-time-zone-haiku` build-dependency —
    # inert on every platform this project actually ships for, but real).
    if cargo tree -i "$pkg" --workspace -e normal,build --target all >"$TREE_OUT" 2>&1; then
        # `cargo tree -i` exits 0 in TWO different situations, and only one
        # of them is a violation:
        #   (1) the crate IS reachable under these filters -> prints an
        #       actual reverse-dependency tree ("<pkg> v<version>\n...").
        #   (2) the crate exists somewhere in the lockfile's full graph but
        #       is NOT reachable under the requested edge-kind filter (e.g.
        #       it sits only behind a dev-dependency edge that
        #       `-e normal,build` excludes) -> cargo still exits 0, but
        #       prints ONLY "warning: nothing to print." plus a hint, with
        #       no tree body at all. Treating exit 0 alone as "found" makes
        #       (2) a false positive; confirmed empirically for 'cmake' in
        #       this workspace (reachable only via `reqwest`'s dev-dependency
        #       edge on oxibonsai-cli/oxibonsai-serve, which -e normal,build
        #       correctly excludes — there is no real cmake/cc dependency
        #       from that direction).
        if grep -qi "nothing to print" "$TREE_OUT"; then
            log "  OK: '$pkg' is not reachable under -e normal,build --target all"
            log "      (it may still appear elsewhere in the full lockfile graph, e.g."
            log "      behind a dev-dependency edge, which is out of scope for this check)."
        elif [ "$pkg" = "cc" ] && [ "$(grep -E '^[├└]── ' "$TREE_OUT" | grep -c -v -E '^[├└]── iana-time-zone-haiku ')" = "0" ] && grep -qE '^[├└]── iana-time-zone-haiku ' "$TREE_OUT"; then
            # Narrow, documented allowance: `cc` reachable ONLY through
            # `iana-time-zone-haiku`, chrono's Haiku-OS-only time-zone
            # build-dependency (`[target.'cfg(target_os = "haiku")']`).
            # It is never compiled on macOS/Linux/Windows/wasm — the
            # platforms this project ships for — so no C code enters the
            # default build. Any OTHER reverse path to `cc` (a second
            # depth-1 dependent) still counts as a violation.
            log "  OK: 'cc' is reachable only via the Haiku-only 'iana-time-zone-haiku'"
            log "      build-dependency of chrono (inert on every shipped platform)."
        else
            VIOLATIONS=$((VIOLATIONS + 1))
            echo "VIOLATION: '$pkg' is reachable from the default-feature build."
            echo "  Reverse-dependency path (top = the banned crate, chain runs down to the"
            echo "  workspace member that pulls it in):"
            sed 's/^/    /' "$TREE_OUT"
            echo "  Fix: replace the offending dependency with its COOLJAPAN Pure-Rust"
            echo "  equivalent, or move it behind a non-default feature (CLAUDE.md"
            echo "  Dependency Replacements table)."
        fi
    else
        # cargo tree exits non-zero with "did not match any packages" when the
        # crate is genuinely absent from the resolved graph — that is the
        # PASS case for this check. Any other failure is a real cargo error.
        if grep -qi "did not match any packages\|no matching packages\|matched no packages" "$TREE_OUT"; then
            log "  OK: '$pkg' does not appear in the default dependency graph."
        else
            echo "ERROR: 'cargo tree -i $pkg' failed unexpectedly:" >&2
            sed 's/^/    /' "$TREE_OUT" >&2
            exit 2
        fi
    fi
done

log ""
log "── best-effort scan: build.rs literal compiler shell-outs ──"
# Cross-reference `cargo metadata`'s resolved (default-feature) package list
# against each package's on-disk manifest directory, and grep any build.rs
# found there for a literal `Command::new("cc"|"gcc"|"g++"|"clang"|"clang++"|"cmake")`.
# This is a heuristic bonus layer on top of the authoritative crate-name
# check above: it cannot see a dynamically-constructed compiler name, and it
# does report vendored `cc`/`cmake` crates a second time (expected — the
# crate-name check already fails the build in that case).
METADATA_JSON="$(mktemp "${TMPDIR:-/tmp}/oxibonsai_pure_rust_meta.XXXXXX")"
trap 'rm -f "$TREE_OUT" "$METADATA_JSON"' EXIT
if cargo metadata --format-version 1 >"$METADATA_JSON" 2>/dev/null; then
    SHELLOUT_HITS="$(python3 - "$METADATA_JSON" <<'PYEOF'
import json, sys, re, pathlib

with open(sys.argv[1]) as f:
    meta = json.load(f)

pattern = re.compile(
    r'Command::new\(\s*"(cc|gcc|g\+\+|clang|clang\+\+|cmake|c\+\+)"\s*\)'
)
hits = []
seen_manifests = set()
for pkg in meta.get("packages", []):
    manifest = pkg.get("manifest_path")
    if not manifest or manifest in seen_manifests:
        continue
    seen_manifests.add(manifest)
    build_rs = pathlib.Path(manifest).parent / "build.rs"
    if not build_rs.is_file():
        continue
    try:
        text = build_rs.read_text(errors="replace")
    except OSError:
        continue
    m = pattern.search(text)
    if m:
        hits.append(f"{pkg.get('name')} {pkg.get('version')} -> {build_rs} (matches {m.group(1)!r})")

for h in hits:
    print(h)
PYEOF
)"
    if [[ -n "$SHELLOUT_HITS" ]]; then
        VIOLATIONS=$((VIOLATIONS + 1))
        echo "VIOLATION (heuristic): build.rs shells out to a C/C++ compiler directly:"
        while IFS= read -r line; do echo "    $line"; done <<<"$SHELLOUT_HITS"
    else
        log "  OK: no build.rs in the resolved graph shells out to cc/gcc/g++/clang/cmake directly."
    fi
else
    log "  SKIPPED: 'cargo metadata' failed (non-fatal for this heuristic layer)."
fi

log ""
log "═══════════════════════════════════════════════════════════════"
if [[ "$VIOLATIONS" -gt 0 ]]; then
    echo "FAIL: $VIOLATIONS pure-rust policy violation(s) in the default dependency graph."
    exit 1
fi
log "  PASS: default dependency graph is C/C++/Fortran-free."
log "═══════════════════════════════════════════════════════════════"
exit 0
