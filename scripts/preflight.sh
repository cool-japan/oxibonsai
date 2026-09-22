#!/usr/bin/env bash
# OxiBonsai — fast local pre-push sanity check.
#
# T-01: "the CI gate must be a repository-local script... invoked by a
# documented pre-push hook and by the release gate." This is the pre-push
# half: a FAST subset of scripts/ci.sh meant to run on every `git push`
# without becoming annoying — it does not build --all-features, does not
# run the test suite, and does not run the slower gates (deny, CUDA syntax,
# coverage). It catches the mistakes that are cheap to catch early
# (formatting, a default-feature-only compile error, an obvious clippy
# lint) and otherwise gets out of the way. For everything else, run the
# full gate yourself before opening a PR:
#   ./scripts/ci.sh
# and before cutting a release:
#   ./scripts/release-gate.sh
#
# Usage:
#   ./scripts/preflight.sh                 # run the fast checks once
#   ./scripts/preflight.sh --install-hook   # wire this script up as `git push`'s pre-push hook
#   ./scripts/preflight.sh --uninstall-hook # remove a hook this script installed
#
# The installed hook is a two-line forwarding script (see install_hook
# below) written to the actual git hooks directory as resolved by
# `git rev-parse --git-path hooks` — this is correct from inside a git
# worktree too (hooks are shared across a repository's worktrees; that
# directory lives under the *main* checkout's `.git`, not the worktree's).
# `--install-hook` refuses to clobber a pre-push hook it did not itself
# write; `--uninstall-hook` refuses to remove one it does not recognize.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 2

HOOK_MARKER="# installed-by: oxibonsai scripts/preflight.sh --install-hook"

install_hook() {
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git not found on PATH." >&2
        exit 2
    fi
    local hooks_dir
    hooks_dir="$(git rev-parse --git-path hooks 2>/dev/null)" || {
        echo "ERROR: not inside a git repository." >&2
        exit 2
    }
    mkdir -p "$hooks_dir"
    local hook_path="$hooks_dir/pre-push"
    if [[ -e "$hook_path" ]] && ! grep -qF "$HOOK_MARKER" "$hook_path" 2>/dev/null; then
        echo "ERROR: $hook_path already exists and was not installed by this script." >&2
        echo "       Refusing to overwrite it. Remove or back it up manually, then retry." >&2
        exit 1
    fi
    # deps-08 / CLAUDE.md "never hardcode absolute paths": hooks live under
    # the MAIN checkout's `.git` and are shared by every worktree of this
    # repository, but $PROJECT_ROOT here is wherever `--install-hook` was
    # run FROM — often an ephemeral agent worktree. Baking that path into
    # `exec` would pin every future `git push`, from any worktree including
    # the main one, to a directory that disappears the moment this worktree
    # is pruned. The hook body below resolves its own root at RUN time
    # instead (via `git rev-parse --show-toplevel`, evaluated when the hook
    # actually executes, from whichever worktree invoked `git push`), so it
    # keeps working after this worktree is gone. `\$root`/`\$rc` are
    # deliberately escaped so they land in the generated file literally,
    # not expanded now by this heredoc.
    cat >"$hook_path" <<EOF
#!/usr/bin/env bash
$HOOK_MARKER
# Regenerate with: scripts/preflight.sh --install-hook (from any worktree of this repository)
root="\$(git rev-parse --show-toplevel 2>/dev/null)"
rc=\$?
if [[ \$rc -ne 0 || -z "\$root" ]]; then
    echo "preflight pre-push hook: could not resolve the repository root (git rev-parse --show-toplevel failed); skipping fast checks for this push." >&2
    exit 0
fi
exec "\$root/scripts/preflight.sh"
EOF
    chmod +x "$hook_path"
    echo "Installed pre-push hook: $hook_path"
    echo "It runs this script's fast checks before every 'git push' from any"
    echo "worktree of this repository. Uninstall with:"
    echo "  $PROJECT_ROOT/scripts/preflight.sh --uninstall-hook"
}

uninstall_hook() {
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git not found on PATH." >&2
        exit 2
    fi
    local hooks_dir
    hooks_dir="$(git rev-parse --git-path hooks 2>/dev/null)" || {
        echo "ERROR: not inside a git repository." >&2
        exit 2
    }
    local hook_path="$hooks_dir/pre-push"
    if [[ ! -e "$hook_path" ]]; then
        echo "No pre-push hook installed at $hook_path — nothing to do."
        exit 0
    fi
    if ! grep -qF "$HOOK_MARKER" "$hook_path" 2>/dev/null; then
        echo "ERROR: $hook_path exists but was not installed by this script." >&2
        echo "       Refusing to remove a hook this script does not recognize." >&2
        exit 1
    fi
    rm -f "$hook_path"
    echo "Removed pre-push hook: $hook_path"
}

for arg in "$@"; do
    case "$arg" in
        --install-hook)
            install_hook
            exit 0
            ;;
        --uninstall-hook)
            uninstall_hook
            exit 0
            ;;
        --help|-h)
            sed -n '2,24p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# ── The fast checks themselves ──────────────────────────────────────────

echo "OxiBonsai preflight (fast pre-push checks)"
echo "project root: $PROJECT_ROOT"

if ! command -v cargo >/dev/null 2>&1; then
    echo "FATAL: cargo not found on PATH." >&2
    exit 2
fi

FAILED=0

step() {
    local name="$1"
    shift
    echo ""
    echo "-- $name --"
    if "$@"; then
        echo "OK: $name"
    else
        echo "FAILED: $name"
        FAILED=1
    fi
}

if command -v cargo-fmt >/dev/null 2>&1; then
    step "fmt" cargo fmt --all -- --check
else
    echo ""
    echo "-- fmt --"
    echo "SKIPPED: rustfmt component not installed (rustup component add rustfmt)"
fi

step "check (default features)" cargo check --workspace
step "check (all features)" cargo check --workspace --all-features

if command -v cargo-clippy >/dev/null 2>&1; then
    step "clippy (default features)" cargo clippy --workspace --all-targets -- -D warnings
else
    echo ""
    echo "-- clippy --"
    echo "SKIPPED: clippy component not installed (rustup component add clippy)"
fi

echo ""
if [[ "$FAILED" -ne 0 ]]; then
    echo "PREFLIGHT FAILED — fix the above before pushing."
    echo "(This is the fast subset only. Run ./scripts/ci.sh for the full gate.)"
    exit 1
fi
echo "PREFLIGHT OK (fast subset only — run ./scripts/ci.sh before opening a PR)."
exit 0
