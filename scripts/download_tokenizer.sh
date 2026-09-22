#!/usr/bin/env bash
# Download Qwen3 tokenizer.json from HuggingFace.
#
# Usage:
#   ./scripts/download_tokenizer.sh
#
# Verified against scripts/checksums.sha256 after download (sec-12); a
# mismatch is treated as authoritative (this is a direct, unprocessed
# download) and aborts loudly rather than leaving a corrupt/tampered file
# in place silently.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKSUMS_FILE="$SCRIPT_DIR/checksums.sha256"

HF_REPO="Qwen/Qwen3-8B"
OUTPUT="$PROJECT_ROOT/models/tokenizer.json"

if [ -f "$OUTPUT" ]; then
    echo "✓ tokenizer.json already exists at $OUTPUT"
    exit 0
fi

mkdir -p "$PROJECT_ROOT/models"

echo "Downloading tokenizer.json from $HF_REPO ..."
# --connect-timeout/--max-time: this is a released model's tokenizer file on
# a "main" branch pointer, not pinned to a commit — a hung connection should
# fail fast rather than block a CI/dev run indefinitely.
#
# NOTE: the curl exit code is captured via `rc=$?; ... || true` rather than
# `if ! curl ...; then rc=$?; ...`, because inside `if ! cmd; then`, `$?`
# reports the if-condition's own (negated) truth value — always 0 in the
# then-branch — not curl's real exit code. Verified empirically; do not
# "simplify" this back to `if ! curl ...`.
curl -fSL --connect-timeout 15 --max-time 300 \
    "https://huggingface.co/${HF_REPO}/resolve/main/tokenizer.json" \
    -o "$OUTPUT" || {
    rc=$?
    rm -f "$OUTPUT"
    echo "ERROR: download failed (curl exit $rc); removed any partial file." >&2
    exit "$rc"
}

echo "✓ Saved to $OUTPUT ($(wc -c < "$OUTPUT") bytes)"

# ── Checksum verification (sec-12) ──────────────────────────────────────────
if [[ ! -f "$CHECKSUMS_FILE" ]]; then
    echo "WARN: $CHECKSUMS_FILE not found; skipping checksum verification." >&2
    exit 0
fi
LINE="$(awk '$2 == "models/tokenizer.json" {print; exit}' "$CHECKSUMS_FILE")"
if [[ -z "$LINE" ]]; then
    echo "NOTE: no checksum recorded yet for models/tokenizer.json; skipping verification." >&2
    exit 0
fi
VERIFY_OUT=""
VERIFY_RC=0
if command -v sha256sum >/dev/null 2>&1; then
    VERIFY_OUT="$(cd "$PROJECT_ROOT" && printf '%s\n' "$LINE" | sha256sum -c - 2>&1)" || VERIFY_RC=$?
elif command -v shasum >/dev/null 2>&1; then
    VERIFY_OUT="$(cd "$PROJECT_ROOT" && printf '%s\n' "$LINE" | shasum -a 256 -c - 2>&1)" || VERIFY_RC=$?
else
    echo "WARN: neither sha256sum nor shasum found on PATH; cannot verify the download." >&2
    exit 0
fi
if [[ "$VERIFY_RC" -ne 0 ]]; then
    echo "ERROR: checksum MISMATCH for models/tokenizer.json:" >&2
    echo "$VERIFY_OUT" >&2
    echo "Removing the downloaded file — do not trust it. Retry, or if this is a" >&2
    echo "confirmed upstream revision, update scripts/checksums.sha256." >&2
    rm -f "$OUTPUT"
    exit 1
fi
echo "✓ Checksum OK: models/tokenizer.json"
