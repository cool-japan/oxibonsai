# shellcheck shell=bash
# OxiBonsai — the offline self-test of scripts/release-gate.sh.
#
# This file is SOURCED by `release-gate.sh --self-test` and by nothing else: it
# defines `release_gate_self_test`, which drives the gate's own functions
# (`check_capability_manifest`, `legacy_models_require_tests_arg`, the stage
# functions, `build_release_cli_binary`, and the gate's main flow in a scratch
# copy, plus the CUDA-syntax waiver across the real ci.sh, check_cuda.sh and
# publish.sh) against synthetic manifests and stand-in tools. It reads the gate's
# own globals and functions (`SCRIPT_DIR`, `PROJECT_ROOT`, the `*_TEST_NAME`
# constants, ...), so it can only run inside the gate's shell; it lives in its
# own file so that neither file nears the 2000-line size ceiling. The gate's
# real run never loads it, and the scratch copy of `release-gate.sh` the
# self-test runs end to end never needs it.
#
# Every scenario must hold under the macOS system bash (3.2): no associative
# arrays, no `mapfile`, no `${var,,}`, and an array that may be empty is
# expanded as `${arr[@]+"${arr[@]}"}`.
#
# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "release-gate-selftest.sh is sourced by 'release-gate.sh --self-test'; run that instead." >&2
    exit 2
fi

# Whether the Rust source file `file` (relative to the project root) carries
# `name` as a string literal — resolving the `\`-newline continuations a long
# `TEST_NAME` constant is split with. The scenarios below use this to prove
# each name the gate requires is one a test records.
#   test_name_in_source <file> <name>
test_name_in_source() {
    python3 - "$PROJECT_ROOT/$1" "$2" <<'PYEOF'
import re, sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    source = f.read()
# A Rust string continuation (backslash, newline, leading whitespace) is elided.
source = re.sub(r"\\\n[ \t]*", "", source)
sys.exit(0 if '"' + sys.argv[2] + '"' in source else 1)
PYEOF
}

# ── --self-test: exercise the gate's own logic offline ──────────────────
# No real build, no hardware, no network: builds synthetic JSONL manifests
# under a throwaway `mktemp -d` (honours $TMPDIR; never a hardcoded /tmp
# path) and asserts check_capability_manifest()'s verdict on each scenario.
# This is what "pins" the malformed-line-fails-closed behaviour documented
# above with a test: it cannot silently regress back to the old "reported,
# not silently ignored, but still exits 0" shape without this failing. The
# stage functions (`run_bonsai2_models_stage`, `run_metal_hidden_stage`,
# `build_release_cli_binary`) are driven the same way, through a stand-in
# `cargo` placed first on `PATH` that only logs its invocations or prints a
# canned JSON build stream.
release_gate_self_test() {
    local failures=0
    local total=0
    local work_dir
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch directory" >&2
        return 1
    }

    local now future_epoch
    now="$(date +%s)"
    # A run_start_epoch far in the future makes every real-filesystem mtime
    # "predate" it, exercising the staleness branch without needing a
    # platform-specific `touch -t` (BSD and GNU disagree on that flag's
    # syntax) to backdate a file instead.
    future_epoch=$((now + 1000000))

    check() {
        local label="$1" expect_rc="$2"
        shift 2
        echo ""
        echo "-- self-test: $label --"
        total=$((total + 1))
        local rc=0
        check_capability_manifest "$@" || rc=$?
        if [[ "$rc" -ne "$expect_rc" ]]; then
            echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
            failures=$((failures + 1))
        else
            echo "OK: self-test '$label' (exit $rc, as expected)"
        fi
    }

    # 1. Manifest file does not exist at all -> FAIL.
    check "missing manifest" 1 "$work_dir/does-not-exist.jsonl" "$now" metal

    # 2. Manifest exists but is empty -> FAIL (no records for the required cap).
    : >"$work_dir/empty.jsonl"
    check "empty manifest" 1 "$work_dir/empty.jsonl" "$now" metal

    # 3. Only executed:false records -> FAIL (self-skipped is not proof).
    printf '%s\n' '{"capability":"metal","executed":false,"test":"t::skip"}' \
        >"$work_dir/all_false.jsonl"
    check "all executed:false" 1 "$work_dir/all_false.jsonl" "$now" metal

    # 4. One executed:true record -> PASS.
    printf '%s\n' '{"capability":"metal","executed":true,"test":"t::ran"}' \
        >"$work_dir/one_true.jsonl"
    check "one executed:true" 0 "$work_dir/one_true.jsonl" "$now" metal

    # 5. Manifest predates this "run" -> FAIL (stale evidence), proven via
    #    a run_start_epoch set in the future rather than a backdated mtime.
    check "stale manifest (future run_start)" 1 "$work_dir/one_true.jsonl" "$future_epoch" metal

    # 6. A malformed line alongside an otherwise-good executed:true record
    #    -> FAIL. This is the exact behaviour item (d) pins.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::ran"}' \
        'not valid json at all' \
        >"$work_dir/malformed_plus_good.jsonl"
    check "malformed line + good record" 1 "$work_dir/malformed_plus_good.jsonl" "$now" metal

    # 7. A required capability with zero records, even though the manifest
    #    is otherwise well-formed and fresh -> FAIL (checked per-capability,
    #    not "the manifest has at least one good line anywhere").
    check "required cap absent from otherwise-good manifest" 1 \
        "$work_dir/one_true.jsonl" "$now" cuda

    # 8. Two required capabilities, both satisfied -> PASS.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::ran"}' \
        '{"capability":"cuda","executed":true,"test":"t::ran_cuda"}' \
        >"$work_dir/both_true.jsonl"
    check "two required capabilities, both satisfied" 0 \
        "$work_dir/both_true.jsonl" "$now" metal cuda

    # 9. The "bonsai2-models" capability is checked by the
    #    exact same generic logic as every other capability name above — this
    #    scenario exercises it by name so a future rename/typo of the string
    #    this script and `Capability::Bonsai2Models::as_str()` must agree on
    #    fails a test here, not only in production.
    printf '%s\n' '{"capability":"bonsai2-models","executed":true,"test":"t::bonsai2_ran"}' \
        >"$work_dir/bonsai2_models_true.jsonl"
    check "bonsai2-models executed:true" 0 "$work_dir/bonsai2_models_true.jsonl" "$now" bonsai2-models

    # 10/11. `--require-tests`: a capability can have `executed: true`
    # records yet still be missing one of the SPECIFIC gates this script
    # requires by name (item 1(b)'s per-capability required-test-name list).
    printf '%s\n%s\n' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::a"}' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::b"}' \
        >"$work_dir/require_tests_two.jsonl"
    check "require-tests: every named test present" 0 \
        "$work_dir/require_tests_two.jsonl" "$now" \
        "--require-tests=bonsai2-models:gate::a,gate::b"
    check "require-tests: one named test missing" 1 \
        "$work_dir/require_tests_two.jsonl" "$now" \
        "--require-tests=bonsai2-models:gate::a,gate::c"

    # `duration_ms`: an optional field (`oxibonsai_testkit::capability::
    # record_executed_timed`). A record that carries it must still pass the
    # ordinary check (and print a total on the "OK" line, verified by eye
    # above, not asserted here); a non-numeric value must not crash the
    # parser, just be treated as if the field were absent.
    printf '%s\n%s\n' \
        '{"capability":"metal","executed":true,"test":"t::timed","duration_ms":1500}' \
        '{"capability":"metal","executed":true,"test":"t::bad_duration","duration_ms":"not-a-number"}' \
        >"$work_dir/duration_ms.jsonl"
    check "duration_ms: numeric and non-numeric values both still pass" 0 \
        "$work_dir/duration_ms.jsonl" "$now" metal

    # 12-... : `run_bonsai2_models_stage`'s fail-closed and leg-ordering
    # behaviour, via a PATH-shimmed `cargo` that only logs its own
    # invocations (one line per call) instead of building or running
    # anything — no real 27B GGUF, build or hardware involved.
    local fake_bin fake_log stub_output
    fake_bin="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest_bin.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch bin directory" >&2
        failures=$((failures + 1))
        fake_bin=""
    }
    if [[ -n "$fake_bin" ]]; then
        fake_log="$work_dir/fake_cargo_invocations.log"
        stub_output="$work_dir/stage_stub_output.log"
        cat >"$fake_bin/cargo" <<'CARGO_STUB_EOF'
#!/usr/bin/env bash
echo "$* [OXIBONSAI_KERNEL_TIER=${OXIBONSAI_KERNEL_TIER-unset}]" >>"$OXIBONSAI_SELFTEST_CARGO_LOG"
exit 0
CARGO_STUB_EOF
        chmod +x "$fake_bin/cargo"

        run_stage_stub() {
            local label="$1" expect_rc="$2" expect_calls="$3"
            shift 3
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            : >"$fake_log"
            local rc=0
            OXIBONSAI_SELFTEST_CARGO_LOG="$fake_log" PATH="$fake_bin:$PATH" \
                run_bonsai2_models_stage "$@" >"$stub_output" 2>&1 || rc=$?
            local calls
            calls="$(wc -l <"$fake_log" | tr -d ' ')"
            if [[ "$rc" -ne "$expect_rc" ]]; then
                echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
                echo "  stage output: $(cat "$stub_output")"
                failures=$((failures + 1))
            elif [[ "$calls" -ne "$expect_calls" ]]; then
                echo "FAIL: self-test '$label': expected $expect_calls cargo invocation(s), got $calls"
                echo "  invocation log: $(cat "$fake_log")"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, $calls cargo invocation(s), as expected)"
            fi
        }

        local models_half models_full golden_stub metal_present metal_absent
        models_half="$work_dir/models_half"
        mkdir -p "$models_half"
        printf 'x' >"$models_half/$BONSAI2_PQ2_FILE"   # PTQ1_0 deliberately absent
        models_full="$work_dir/models_full"
        mkdir -p "$models_full"
        printf 'x' >"$models_full/$BONSAI2_PQ2_FILE"
        printf 'x' >"$models_full/$BONSAI2_PTQ1_FILE"
        golden_stub="$work_dir/golden_stub"
        mkdir -p "$golden_stub"
        metal_present="$work_dir/hybrid_metal_gates.rs"
        printf '// self-test stub\n' >"$metal_present"
        metal_absent="$work_dir/does-not-exist/hybrid_metal_gates.rs"

        run_stage_stub "stage1c stub: half-populated models dir refuses before any cargo call" \
            1 0 0 0 "$models_half" "$golden_stub" "$metal_present" 1
        run_stage_stub "stage1c stub: --skip-bonsai2-models never invokes cargo" \
            0 0 1 0 "$models_half" "$golden_stub" "$metal_present" 1
        run_stage_stub "stage1c stub: non-Darwin host runs nothing" \
            0 0 0 0 "$models_full" "$golden_stub" "$metal_present" 0
        run_stage_stub "stage1c stub: fully populated + metal file present runs all six legs" \
            0 6 0 0 "$models_full" "$golden_stub" "$metal_present" 1
        total=$((total + 1))
        if [[ -f "$fake_log" ]] && [[ "$(grep -o 'hybrid_forward_parity_tests\|bonsai2_engine_tests\|bonsai2_runtime_tests\|hybrid_metal_gates\|bonsai2_metal_engine_tests\|hybrid_metal_prefill_gates' "$fake_log" | tr '\n' ',')" \
            == "hybrid_forward_parity_tests,bonsai2_engine_tests,bonsai2_runtime_tests,hybrid_metal_gates,bonsai2_metal_engine_tests,hybrid_metal_prefill_gates," ]]; then
            echo "OK: self-test 'stage1c stub: the six legs run in the documented order (bonsai2_runtime_tests covers G6/G7/G9 in one invocation; the batched-prefill leg runs last)'"
        else
            echo "FAIL: self-test 'stage1c stub: the six legs run in the documented order'"
            failures=$((failures + 1))
        fi
        # The batched-prefill leg is a --release run of its own binary with
        # both bands' files and OXI_REQUIRE_MODEL_FILES=1, so a band it cannot
        # find fails the leg instead of self-skipping.
        total=$((total + 1))
        if [[ "$(grep -c -F -- "test --release -p oxibonsai-model --all-features --test hybrid_metal_prefill_gates -- --test-threads=1 --nocapture" "$fake_log")" -eq 1 ]]; then
            echo "OK: self-test 'stage1c stub: the batched-prefill leg is a --release, --nocapture run of hybrid_metal_prefill_gates'"
        else
            echo "FAIL: self-test 'stage1c stub: the batched-prefill leg is a --release, --nocapture run of hybrid_metal_prefill_gates'"
            echo "  invocation log: $(cat "$fake_log")"
            failures=$((failures + 1))
        fi
        # An exported INT8 tier selector must never reach a leg (see
        # `run_bonsai2_models_stage`): the stub logs what each call saw.
        local had_tier=0 prior_tier="${OXIBONSAI_KERNEL_TIER-}"
        [[ -n "${OXIBONSAI_KERNEL_TIER+set}" ]] && had_tier=1
        export OXIBONSAI_KERNEL_TIER=int8-scalar
        run_stage_stub "stage1c stub: an exported OXIBONSAI_KERNEL_TIER still runs all six legs" \
            0 6 0 0 "$models_full" "$golden_stub" "$metal_present" 1
        total=$((total + 1))
        if [[ -f "$fake_log" ]] \
            && [[ "$(grep -c -F 'OXIBONSAI_KERNEL_TIER=unset]' "$fake_log")" -eq 6 ]]; then
            echo "OK: self-test 'stage1c stub: every leg runs with OXIBONSAI_KERNEL_TIER removed from its environment'"
        else
            echo "FAIL: self-test 'stage1c stub: every leg runs with OXIBONSAI_KERNEL_TIER removed from its environment'"
            echo "  invocation log: $(cat "$fake_log")"
            failures=$((failures + 1))
        fi
        if [[ "$had_tier" -eq 1 ]]; then
            export OXIBONSAI_KERNEL_TIER="$prior_tier"
        else
            unset OXIBONSAI_KERNEL_TIER
        fi
        run_stage_stub "stage1c stub: fully populated, metal test file absent, only three legs run" \
            0 3 0 0 "$models_full" "$golden_stub" "$metal_absent" 1
        run_stage_stub "stage1c stub: --skip-bonsai2-metal runs only the first three legs" \
            0 3 0 1 "$models_full" "$golden_stub" "$metal_present" 1
    fi

    # "legacy-models" through `--require-tests`, exercised by name (same
    # reasoning as scenario 9's bonsai2-models check): a host with only
    # Ternary-Bonsai-1.7B.gguf present writes executed=true for that model's
    # two gates but executed=false for Ternary-Bonsai-8B.gguf's and
    # Bonsai-8B.gguf's four — the exact partial-fixture shape this script's
    # REQUIRED_CAPS entry for "legacy-models" must fail on.
    printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
        '{"capability":"legacy-models","executed":true,"test":"oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":true,"test":"oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers"}' \
        '{"capability":"legacy-models","executed":false,"test":"oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers"}' \
        >"$work_dir/legacy_models_partial.jsonl"
    check "legacy-models require-tests: only the 1.7B pair ran -> FAIL" 1 \
        "$work_dir/legacy_models_partial.jsonl" "$now" \
        "--require-tests=legacy-models:\
oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers,\
oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers,\
oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers"

    # `legacy_models_require_tests_arg`'s own list, both ways: every legacy
    # gate has an executed=true record except M-08's 20000-token YaRN gate,
    # which self-skipped. Without OXIBONSAI_M08_RUN_LONG=1 that gate is not
    # required (PASS); with it, its self-skip fails the gate; once it has
    # really run, the gate passes. A self-skipped ONNX-export round trip
    # (no ONNX export on the host) fails either way.
    local m08_long_name="oxibonsai-runtime::legacy_parity_tests::m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens"
    local onnx_name="oxibonsai-runtime::legacy_parity_tests::onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip"
    local legacy_arg legacy_names legacy_name
    legacy_arg="$(legacy_models_require_tests_arg 0)"
    IFS=',' read -r -a legacy_names <<<"${legacy_arg#--require-tests=legacy-models:}"
    : >"$work_dir/legacy_m08_skipped.jsonl"
    : >"$work_dir/legacy_onnx_skipped.jsonl"
    for legacy_name in "${legacy_names[@]}"; do
        printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$legacy_name" \
            >>"$work_dir/legacy_m08_skipped.jsonl"
        if [[ "$legacy_name" != "$onnx_name" ]]; then
            printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$legacy_name" \
                >>"$work_dir/legacy_onnx_skipped.jsonl"
        fi
    done
    printf '{"capability":"legacy-models","executed":false,"test":"%s"}\n' "$m08_long_name" \
        >>"$work_dir/legacy_m08_skipped.jsonl"
    printf '{"capability":"legacy-models","executed":false,"test":"%s"}\n' "$onnx_name" \
        >>"$work_dir/legacy_onnx_skipped.jsonl"
    cp "$work_dir/legacy_m08_skipped.jsonl" "$work_dir/legacy_m08_ran.jsonl"
    printf '{"capability":"legacy-models","executed":true,"test":"%s"}\n' "$m08_long_name" \
        >>"$work_dir/legacy_m08_ran.jsonl"
    check "legacy-models: M-08 long gate self-skipped, OXIBONSAI_M08_RUN_LONG unset -> PASS" 0 \
        "$work_dir/legacy_m08_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    check "legacy-models: M-08 long gate self-skipped under OXIBONSAI_M08_RUN_LONG=1 -> FAIL" 1 \
        "$work_dir/legacy_m08_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 1)"
    check "legacy-models: M-08 long gate ran under OXIBONSAI_M08_RUN_LONG=1 -> PASS" 0 \
        "$work_dir/legacy_m08_ran.jsonl" "$now" "$(legacy_models_require_tests_arg 1)"
    check "legacy-models: ONNX-export round trip self-skipped -> FAIL" 1 \
        "$work_dir/legacy_onnx_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"

    # ── Timed records: an executed required test without `duration_ms` ──
    # is reported with a WARN and never fails the gate; a fully timed
    # manifest prints no WARN; a non-numeric duration counts as missing.
    check_output() {
        local label="$1" expect_rc="$2" must_match="$3" must_not_match="$4"
        shift 4
        echo ""
        echo "-- self-test: $label --"
        total=$((total + 1))
        local rc=0 out
        out="$(check_capability_manifest "$@" 2>&1)" || rc=$?
        printf '%s\n' "$out"
        if [[ "$rc" -ne "$expect_rc" ]]; then
            echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
            failures=$((failures + 1))
        elif [[ -n "$must_match" && "$out" != *"$must_match"* ]]; then
            echo "FAIL: self-test '$label': the report does not contain: $must_match"
            failures=$((failures + 1))
        elif [[ -n "$must_not_match" && "$out" == *"$must_not_match"* ]]; then
            echo "FAIL: self-test '$label': the report must not contain: $must_not_match"
            failures=$((failures + 1))
        else
            echo "OK: self-test '$label' (exit $rc, report as expected)"
        fi
    }
    printf '%s\n%s\n' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::timed","duration_ms":1500}' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::untimed"}' \
        >"$work_dir/untimed_one.jsonl"
    check_output "require-tests: an executed record without duration_ms WARNs and still passes" 0 \
        "WARN: 'bonsai2-models': 1 required test(s) have an executed record without duration_ms" \
        "  - gate::timed" \
        "$work_dir/untimed_one.jsonl" "$now" "--require-tests=bonsai2-models:gate::timed,gate::untimed"
    check_output "require-tests: the WARN names the untimed test" 0 \
        "  - gate::untimed" "" \
        "$work_dir/untimed_one.jsonl" "$now" "--require-tests=bonsai2-models:gate::timed,gate::untimed"
    printf '%s\n%s\n' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::a","duration_ms":10}' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::b","duration_ms":20}' \
        >"$work_dir/timed_all.jsonl"
    check_output "require-tests: every required record timed -> no WARN" 0 \
        "OK: 'bonsai2-models' has executed=true for all 2 required test(s)." "WARN" \
        "$work_dir/timed_all.jsonl" "$now" "--require-tests=bonsai2-models:gate::a,gate::b"
    printf '%s\n' \
        '{"capability":"bonsai2-models","executed":true,"test":"gate::bad_timer","duration_ms":"soon"}' \
        >"$work_dir/timed_non_numeric.jsonl"
    check_output "require-tests: a non-numeric duration_ms counts as missing (WARN, still passes)" 0 \
        "  - gate::bad_timer" "" \
        "$work_dir/timed_non_numeric.jsonl" "$now" "--require-tests=bonsai2-models:gate::bad_timer"
    printf '%s\n' \
        '{"capability":"bonsai2-models","executed":false,"test":"gate::never_ran"}' \
        >"$work_dir/timed_skipped.jsonl"
    check_output "require-tests: a skipped required test still FAILS (a missing timer is only a WARN)" 1 \
        "FAIL: 'bonsai2-models' has 1 manifest record(s) but NONE with executed=true" "" \
        "$work_dir/timed_skipped.jsonl" "$now" "--require-tests=bonsai2-models:gate::never_ran"

    # ── metal-hidden: the Metal hidden-state gate is required by name ────
    # The name this script requires must be the one the test records under
    # (its `TEST_NAME` constant), or the requirement could never be met.
    total=$((total + 1))
    echo ""
    echo "-- self-test: metal-hidden: the required name is the one metal_hidden_parity_tests records --"
    if grep -q -F "\"$METAL_HIDDEN_TEST_NAME\"" \
        "$PROJECT_ROOT/crates/oxibonsai-model/tests/metal_hidden_parity_tests.rs"; then
        echo "OK: self-test 'metal-hidden: the required name is the one metal_hidden_parity_tests records'"
    else
        echo "FAIL: self-test 'metal-hidden: the required name is the one metal_hidden_parity_tests records'"
        echo "  $METAL_HIDDEN_TEST_NAME is not a string literal in metal_hidden_parity_tests.rs"
        failures=$((failures + 1))
    fi
    printf '{"capability":"metal-hidden","executed":false,"test":"%s"}\n' "$METAL_HIDDEN_TEST_NAME" \
        >"$work_dir/metal_hidden_skipped.jsonl"
    check "metal-hidden: the leg self-skipped (executed:false) -> FAIL" 1 \
        "$work_dir/metal_hidden_skipped.jsonl" "$now" \
        metal-hidden "--require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME"
    : >"$work_dir/metal_hidden_absent.jsonl"
    check "metal-hidden: the leg never ran (no record) -> FAIL" 1 \
        "$work_dir/metal_hidden_absent.jsonl" "$now" \
        metal-hidden "--require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME"
    printf '{"capability":"metal-hidden","executed":true,"test":"some::other_metal_hidden_test"}\n' \
        >"$work_dir/metal_hidden_other.jsonl"
    check "metal-hidden: an executed record of a DIFFERENT test does not satisfy the requirement -> FAIL" 1 \
        "$work_dir/metal_hidden_other.jsonl" "$now" \
        metal-hidden "--require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME"
    printf '{"capability":"metal-hidden","executed":true,"test":"%s","duration_ms":90000}\n' \
        "$METAL_HIDDEN_TEST_NAME" "$METAL_HIDDEN_TEST_NAME" "$METAL_HIDDEN_TEST_NAME" \
        >"$work_dir/metal_hidden_ran.jsonl"
    check_output "metal-hidden: three timed executed runs (one per model) -> PASS, no WARN" 0 \
        "OK: 'metal-hidden' has executed=true for all 1 required test(s)." "WARN" \
        "$work_dir/metal_hidden_ran.jsonl" "$now" \
        metal-hidden "--require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME"

    # ── run_metal_hidden_stage: one run per legacy model, fail closed ───
    local fake_mh_bin fake_mh_log mh_output
    fake_mh_bin="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest_mh.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch bin directory" >&2
        failures=$((failures + 1))
        fake_mh_bin=""
    }
    if [[ -n "$fake_mh_bin" ]]; then
        fake_mh_log="$work_dir/fake_mh_cargo.log"
        mh_output="$work_dir/mh_stage_output.log"
        cat >"$fake_mh_bin/cargo" <<'MH_CARGO_STUB_EOF'
#!/usr/bin/env bash
echo "$* [OXI_MODEL=${OXI_MODEL-unset}] [OXIBONSAI_KERNEL_TIER=${OXIBONSAI_KERNEL_TIER-unset}]" \
    >>"$OXIBONSAI_SELFTEST_CARGO_LOG"
if [[ -n "${OXIBONSAI_SELFTEST_FAIL_ON:-}" && "${OXI_MODEL-}" == *"$OXIBONSAI_SELFTEST_FAIL_ON"* ]]; then
    exit 101
fi
exit 0
MH_CARGO_STUB_EOF
        chmod +x "$fake_mh_bin/cargo"

        local models_legacy_full models_legacy_half legacy_file
        models_legacy_full="$work_dir/models_legacy_full"
        models_legacy_half="$work_dir/models_legacy_half"
        mkdir -p "$models_legacy_full" "$models_legacy_half"
        for legacy_file in "${LEGACY_MODEL_FILES[@]}"; do
            printf 'x' >"$models_legacy_full/$legacy_file"
        done
        printf 'x' >"$models_legacy_half/${LEGACY_MODEL_FILES[0]}"
        printf 'x' >"$models_legacy_half/${LEGACY_MODEL_FILES[1]}"

        # run_mh_stub <label> <expect_rc> <expect_calls> <fail_on> <models_dir>
        run_mh_stub() {
            local label="$1" expect_rc="$2" expect_calls="$3" fail_on="$4" dir="$5"
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            : >"$fake_mh_log"
            local rc=0 calls
            OXIBONSAI_SELFTEST_CARGO_LOG="$fake_mh_log" OXIBONSAI_SELFTEST_FAIL_ON="$fail_on" \
                PATH="$fake_mh_bin:$PATH" run_metal_hidden_stage "$dir" >"$mh_output" 2>&1 || rc=$?
            calls="$(wc -l <"$fake_mh_log" | tr -d ' ')"
            if [[ "$rc" -ne "$expect_rc" ]]; then
                echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
                echo "  stage output: $(cat "$mh_output")"
                failures=$((failures + 1))
            elif [[ "$calls" -ne "$expect_calls" ]]; then
                echo "FAIL: self-test '$label': expected $expect_calls cargo invocation(s), got $calls"
                echo "  invocation log: $(cat "$fake_mh_log")"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, $calls cargo invocation(s), as expected)"
            fi
        }

        run_mh_stub "metal-hidden stub: a fully populated models dir runs one leg per legacy model" \
            0 3 "" "$models_legacy_full"
        total=$((total + 1))
        local mh_models_seen mh_models_expected
        mh_models_seen="$(grep -o 'OXI_MODEL=[^]]*' "$fake_mh_log" | tr '\n' ',')"
        mh_models_expected=""
        for legacy_file in "${LEGACY_MODEL_FILES[@]}"; do
            mh_models_expected="${mh_models_expected}OXI_MODEL=$models_legacy_full/$legacy_file,"
        done
        if [[ "$mh_models_seen" == "$mh_models_expected" ]]; then
            echo "OK: self-test 'metal-hidden stub: OXI_MODEL names each legacy model once, in order'"
        else
            echo "FAIL: self-test 'metal-hidden stub: OXI_MODEL names each legacy model once, in order'"
            echo "  expected: $mh_models_expected"
            echo "  seen:     $mh_models_seen"
            failures=$((failures + 1))
        fi
        total=$((total + 1))
        if [[ "$(grep -c -F 'test --release -p oxibonsai-model --features metal --test metal_hidden_parity_tests -- --test-threads=1 --nocapture' "$fake_mh_log")" -eq 3 ]]; then
            echo "OK: self-test 'metal-hidden stub: every leg is a --release, --nocapture run of metal_hidden_parity_tests'"
        else
            echo "FAIL: self-test 'metal-hidden stub: every leg is a --release, --nocapture run of metal_hidden_parity_tests'"
            echo "  invocation log: $(cat "$fake_mh_log")"
            failures=$((failures + 1))
        fi
        local had_tier_mh=0 prior_tier_mh="${OXIBONSAI_KERNEL_TIER-}"
        [[ -n "${OXIBONSAI_KERNEL_TIER+set}" ]] && had_tier_mh=1
        export OXIBONSAI_KERNEL_TIER=int8-scalar
        run_mh_stub "metal-hidden stub: an exported OXIBONSAI_KERNEL_TIER still runs all three legs" \
            0 3 "" "$models_legacy_full"
        total=$((total + 1))
        if [[ "$(grep -c -F 'OXIBONSAI_KERNEL_TIER=unset]' "$fake_mh_log")" -eq 3 ]]; then
            echo "OK: self-test 'metal-hidden stub: every leg runs with OXIBONSAI_KERNEL_TIER removed from its environment'"
        else
            echo "FAIL: self-test 'metal-hidden stub: every leg runs with OXIBONSAI_KERNEL_TIER removed from its environment'"
            echo "  invocation log: $(cat "$fake_mh_log")"
            failures=$((failures + 1))
        fi
        if [[ "$had_tier_mh" -eq 1 ]]; then
            export OXIBONSAI_KERNEL_TIER="$prior_tier_mh"
        else
            unset OXIBONSAI_KERNEL_TIER
        fi
        run_mh_stub "metal-hidden stub: a missing legacy model refuses before any cargo call" \
            1 0 "" "$models_legacy_half"
        run_mh_stub "metal-hidden stub: a failing leg stops the stage (second of three fails)" \
            101 2 "${LEGACY_MODEL_FILES[1]}" "$models_legacy_full"
    fi

    # ── The speculative, embedding, legacy lib/bin and vision stages ─────
    # Driven through a stand-in `cargo` that logs, per invocation, its
    # arguments and the environment each leg is handed (`unset` where a
    # variable is absent), so the scenarios can assert what the leg really
    # saw — above all that `OXI_MODEL` reaches the Metal fallback leg (which
    # self-skips to a green without it) and that the embedding benchmark's
    # opt-in is set by the gate itself.
    local fake_leg_bin fake_leg_log leg_output
    fake_leg_bin="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest_leg.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch bin directory" >&2
        failures=$((failures + 1))
        fake_leg_bin=""
    }
    if [[ -n "$fake_leg_bin" ]]; then
        fake_leg_log="$work_dir/fake_leg_cargo.log"
        leg_output="$work_dir/leg_stage_output.log"
        cat >"$fake_leg_bin/cargo" <<'LEG_CARGO_STUB_EOF'
#!/usr/bin/env bash
echo "$* [OXI_MODEL=${OXI_MODEL-unset}] [OXI_TOKENIZER=${OXI_TOKENIZER-unset}] [OXIBONSAI_EMBED_BENCH=${OXIBONSAI_EMBED_BENCH-unset}] [OXIBONSAI_EMBED_BENCH_PER_TOKEN=${OXIBONSAI_EMBED_BENCH_PER_TOKEN-unset}] [OXIBONSAI_PREFILL_SWEEP_TOKENS=${OXIBONSAI_PREFILL_SWEEP_TOKENS-unset}] [OXIBONSAI_MODELS_DIR=${OXIBONSAI_MODELS_DIR-unset}] [OXI_BONSAI2_PQ2_GGUF=${OXI_BONSAI2_PQ2_GGUF-unset}] [OXI_BONSAI2_MMPROJ_GGUF=${OXI_BONSAI2_MMPROJ_GGUF-unset}] [OXI_BONSAI2_PTQ1_GGUF=${OXI_BONSAI2_PTQ1_GGUF-unset}] [OXI_REQUIRE_MODEL_FILES=${OXI_REQUIRE_MODEL_FILES-unset}] [OXIBONSAI_KERNEL_TIER=${OXIBONSAI_KERNEL_TIER-unset}]" \
    >>"$OXIBONSAI_SELFTEST_CARGO_LOG"
if [[ -n "${OXIBONSAI_SELFTEST_FAIL_ON:-}" ]] \
    && [[ "$*" == *"$OXIBONSAI_SELFTEST_FAIL_ON"* || "${OXI_MODEL-}" == *"$OXIBONSAI_SELFTEST_FAIL_ON"* ]]; then
    exit 101
fi
exit 0
LEG_CARGO_STUB_EOF
        chmod +x "$fake_leg_bin/cargo"

        # run_leg_stub <label> <expect_rc> <expect_calls> <fail_on> <function> <args...>
        run_leg_stub() {
            local label="$1" expect_rc="$2" expect_calls="$3" fail_on="$4" fn="$5"
            shift 5
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            : >"$fake_leg_log"
            local rc=0 calls
            OXIBONSAI_SELFTEST_CARGO_LOG="$fake_leg_log" OXIBONSAI_SELFTEST_FAIL_ON="$fail_on" \
                PATH="$fake_leg_bin:$PATH" "$fn" "$@" >"$leg_output" 2>&1 || rc=$?
            calls="$(wc -l <"$fake_leg_log" | tr -d ' ')"
            if [[ "$rc" -ne "$expect_rc" ]]; then
                echo "FAIL: self-test '$label': expected exit $expect_rc, got $rc"
                echo "  stage output: $(cat "$leg_output")"
                failures=$((failures + 1))
            elif [[ "$calls" -ne "$expect_calls" ]]; then
                echo "FAIL: self-test '$label': expected $expect_calls cargo invocation(s), got $calls"
                echo "  invocation log: $(cat "$fake_leg_log")"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, $calls cargo invocation(s), as expected)"
            fi
        }

        # assert_leg_log <label> <needle> <expected count>: the last run's
        # invocation log contains `needle` on exactly that many lines.
        assert_leg_log() {
            local label="$1" needle="$2" expect="$3" got
            total=$((total + 1))
            got="$(grep -c -F -- "$needle" "$fake_leg_log")"
            if [[ "$got" -eq "$expect" ]]; then
                echo "OK: self-test '$label'"
            else
                echo "FAIL: self-test '$label': '$needle' on $got invocation line(s), expected $expect"
                echo "  invocation log: $(cat "$fake_leg_log")"
                failures=$((failures + 1))
            fi
        }

        local dir_leg_full dir_leg_no_8b dir_leg_no_17b dir_leg_no_b8 dir_leg_no_tok legacy_file_leg
        dir_leg_full="$work_dir/leg_models_full"
        dir_leg_no_8b="$work_dir/leg_models_no_ternary_8b"
        dir_leg_no_17b="$work_dir/leg_models_no_ternary_1_7b"
        dir_leg_no_b8="$work_dir/leg_models_no_bonsai_8b"
        dir_leg_no_tok="$work_dir/leg_models_no_tokenizer"
        mkdir -p "$dir_leg_full" "$dir_leg_no_8b" "$dir_leg_no_17b" "$dir_leg_no_b8" "$dir_leg_no_tok"
        for legacy_file_leg in "${LEGACY_MODEL_FILES[@]}" tokenizer.json; do
            printf 'x' >"$dir_leg_full/$legacy_file_leg"
        done
        printf 'x' >"$dir_leg_no_8b/${LEGACY_MODEL_FILES[0]}"
        printf 'x' >"$dir_leg_no_8b/${LEGACY_MODEL_FILES[2]}"
        printf 'x' >"$dir_leg_no_8b/tokenizer.json"
        printf 'x' >"$dir_leg_no_17b/${LEGACY_MODEL_FILES[1]}"
        printf 'x' >"$dir_leg_no_17b/${LEGACY_MODEL_FILES[2]}"
        printf 'x' >"$dir_leg_no_17b/tokenizer.json"
        printf 'x' >"$dir_leg_no_b8/${LEGACY_MODEL_FILES[0]}"
        printf 'x' >"$dir_leg_no_b8/${LEGACY_MODEL_FILES[1]}"
        printf 'x' >"$dir_leg_no_b8/tokenizer.json"
        for legacy_file_leg in "${LEGACY_MODEL_FILES[@]}"; do
            printf 'x' >"$dir_leg_no_tok/$legacy_file_leg"
        done

        # An exported INT8 tier selector must not reach any of these legs.
        local had_tier_leg=0 prior_tier_leg="${OXIBONSAI_KERNEL_TIER-}"
        [[ -n "${OXIBONSAI_KERNEL_TIER+set}" ]] && had_tier_leg=1
        export OXIBONSAI_KERNEL_TIER=int8-scalar

        # ── speculative gate ──
        run_leg_stub "speculative stub: a full models dir runs the one gate binary" \
            0 1 "" run_speculative_stage "$dir_leg_full"
        assert_leg_log "speculative stub: --release run of speculative_ternary_metal_gates with the metal feature, --nocapture" \
            "test --release -p oxibonsai-model --features metal --test speculative_ternary_metal_gates -- --test-threads=1 --nocapture" 1
        assert_leg_log "speculative stub: the models dir is handed over and OXIBONSAI_KERNEL_TIER is removed" \
            "[OXIBONSAI_MODELS_DIR=$dir_leg_full]" 1
        assert_leg_log "speculative stub: an exported OXIBONSAI_KERNEL_TIER does not reach the leg" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 1
        run_leg_stub "speculative stub: a missing Ternary-Bonsai-8B refuses before any cargo call" \
            1 0 "" run_speculative_stage "$dir_leg_no_8b"
        run_leg_stub "speculative stub: a missing Ternary-Bonsai-1.7B refuses before any cargo call" \
            1 0 "" run_speculative_stage "$dir_leg_no_17b"
        run_leg_stub "speculative stub: a failing gate's exit code is the stage's" \
            101 1 "speculative_ternary_metal_gates" run_speculative_stage "$dir_leg_full"

        # ── embedding leg ──
        run_leg_stub "embedding stub: a full models dir runs the one test binary" \
            0 1 "" run_embedding_stage "$dir_leg_full"
        assert_leg_log "embedding stub: the leg passes the benchmark opt-in OXIBONSAI_EMBED_BENCH=1 explicitly" \
            "[OXIBONSAI_EMBED_BENCH=1]" 1
        assert_leg_log "embedding stub: OXI_MODEL names the Ternary-Bonsai-1.7B file" \
            "[OXI_MODEL=$dir_leg_full/${LEGACY_MODEL_FILES[0]}]" 1
        assert_leg_log "embedding stub: --release run of embeddings_model_backed with every feature, --nocapture" \
            "test --release -p oxibonsai-runtime --all-features --test embeddings_model_backed -- --test-threads=1 --nocapture" 1
        assert_leg_log "embedding stub: an exported OXIBONSAI_KERNEL_TIER does not reach the leg" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 1
        assert_leg_log "embedding stub: the leg does not set the per-token reference opt-in OXIBONSAI_EMBED_BENCH_PER_TOKEN" \
            "[OXIBONSAI_EMBED_BENCH_PER_TOKEN=unset]" 1
        # A developer's exported opt-in must not reach the gate's leg either:
        # the leg's cost and verdict are the gate configuration's.
        local had_per_token_leg=0 prior_per_token_leg="${OXIBONSAI_EMBED_BENCH_PER_TOKEN-}"
        [[ -n "${OXIBONSAI_EMBED_BENCH_PER_TOKEN+set}" ]] && had_per_token_leg=1
        export OXIBONSAI_EMBED_BENCH_PER_TOKEN=1
        run_leg_stub "embedding stub: an exported OXIBONSAI_EMBED_BENCH_PER_TOKEN=1 still runs the one test binary" \
            0 1 "" run_embedding_stage "$dir_leg_full"
        assert_leg_log "embedding stub: an exported OXIBONSAI_EMBED_BENCH_PER_TOKEN=1 does not reach the leg" \
            "[OXIBONSAI_EMBED_BENCH_PER_TOKEN=unset]" 1
        assert_leg_log "embedding stub: the leg still passes the benchmark opt-in OXIBONSAI_EMBED_BENCH=1 with the per-token variable exported" \
            "[OXIBONSAI_EMBED_BENCH=1]" 1
        if [[ "$had_per_token_leg" -eq 1 ]]; then
            export OXIBONSAI_EMBED_BENCH_PER_TOKEN="$prior_per_token_leg"
        else
            unset OXIBONSAI_EMBED_BENCH_PER_TOKEN
        fi
        run_leg_stub "embedding stub: a missing tokenizer refuses before any cargo call" \
            1 0 "" run_embedding_stage "$dir_leg_no_tok"
        run_leg_stub "embedding stub: a missing Ternary-Bonsai-1.7B refuses before any cargo call" \
            1 0 "" run_embedding_stage "$dir_leg_no_17b"

        # ── M-18 fused-prefill chunk-size sweep: once per model ──
        # Bonsai-8B first, then the Ternary-Bonsai-1.7B; an exported
        # OXIBONSAI_PREFILL_SWEEP_TOKENS must not move the evidence off the
        # default prompt length either.
        local had_sweep_tokens_leg=0 prior_sweep_tokens_leg="${OXIBONSAI_PREFILL_SWEEP_TOKENS-}"
        [[ -n "${OXIBONSAI_PREFILL_SWEEP_TOKENS+set}" ]] && had_sweep_tokens_leg=1
        export OXIBONSAI_PREFILL_SWEEP_TOKENS=64
        run_leg_stub "sweep stub: a full models dir runs one leg per model" \
            0 2 "" run_prefill_sweep_stage "$dir_leg_full"
        assert_leg_log "sweep stub: every leg is the --release, --nocapture, single-thread run of the sweep unit test with the metal feature" \
            "test --release -p oxibonsai-model --features metal --lib -- --test-threads=1 --nocapture real_model_metal_prefill_chunk_size_sweep [OXI_MODEL=" 2
        assert_leg_log "sweep stub: the first leg runs Bonsai-8B" \
            "[OXI_MODEL=$dir_leg_full/Bonsai-8B.gguf]" 1
        assert_leg_log "sweep stub: the second leg runs the Ternary-Bonsai-1.7B" \
            "[OXI_MODEL=$dir_leg_full/Ternary-Bonsai-1.7B.gguf]" 1
        total=$((total + 1))
        local sweep_order
        sweep_order="$(grep -o 'OXI_MODEL=[^]]*' "$fake_leg_log" | sed 's#.*/##' | tr '\n' ',')"
        if [[ "$sweep_order" == "Bonsai-8B.gguf,Ternary-Bonsai-1.7B.gguf," ]]; then
            echo "OK: self-test 'sweep stub: Bonsai-8B runs first, then the Ternary-Bonsai-1.7B, each once'"
        else
            echo "FAIL: self-test 'sweep stub: Bonsai-8B runs first, then the Ternary-Bonsai-1.7B, each once' (saw: $sweep_order)"
            failures=$((failures + 1))
        fi
        assert_leg_log "sweep stub: an exported OXIBONSAI_KERNEL_TIER does not reach either leg" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 2
        assert_leg_log "sweep stub: an exported OXIBONSAI_PREFILL_SWEEP_TOKENS does not reach either leg" \
            "[OXIBONSAI_PREFILL_SWEEP_TOKENS=unset]" 2
        run_leg_stub "sweep stub: a missing Bonsai-8B refuses before any cargo call" \
            1 0 "" run_prefill_sweep_stage "$dir_leg_no_b8"
        run_leg_stub "sweep stub: a missing Ternary-Bonsai-1.7B refuses before any cargo call" \
            1 0 "" run_prefill_sweep_stage "$dir_leg_no_17b"
        total=$((total + 1))
        local sweep_refusal=""
        sweep_refusal="$(run_prefill_sweep_stage "$dir_leg_no_b8" 2>&1)" || true
        if [[ "$sweep_refusal" == *"MISSING fixtures: Bonsai-8B.gguf"* && "$sweep_refusal" == *"--skip-legacy-models"* ]]; then
            echo "OK: self-test 'sweep stub: the refusal names the missing model and the --skip-legacy-models opt-out'"
        else
            echo "FAIL: self-test 'sweep stub: the refusal names the missing model and the --skip-legacy-models opt-out' (saw: $sweep_refusal)"
            failures=$((failures + 1))
        fi
        run_leg_stub "sweep stub: a failing first leg's exit code is the stage's and the second leg never runs" \
            101 1 "/Bonsai-8B.gguf" run_prefill_sweep_stage "$dir_leg_full"
        run_leg_stub "sweep stub: a failing second leg's exit code is the stage's after both ran" \
            101 2 "/Ternary-Bonsai-1.7B.gguf" run_prefill_sweep_stage "$dir_leg_full"
        if [[ "$had_sweep_tokens_leg" -eq 1 ]]; then
            export OXIBONSAI_PREFILL_SWEEP_TOKENS="$prior_sweep_tokens_leg"
        else
            unset OXIBONSAI_PREFILL_SWEEP_TOKENS
        fi

        # ── stage 1d: legacy lib/bin legs, above all the Metal fallback leg ──
        run_leg_stub "stage1d stub: a full models dir runs the lib leg then the fallback leg" \
            0 2 "" run_legacy_lib_bin_stage "$dir_leg_full"
        assert_leg_log "stage1d stub: the fallback leg runs metal_greedy_cpu_fallback_tests, release, with the metal feature" \
            "test --release -p oxibonsai-runtime --features metal --test metal_greedy_cpu_fallback_tests -- --test-threads=1 real_model_greedy_gpu_fallback_byte_identical" 1
        assert_leg_log "stage1d stub: OXI_MODEL reaches BOTH legs (without it the real-model case self-skips to green)" \
            "[OXI_MODEL=$dir_leg_full/${LEGACY_MODEL_FILES[0]}]" 2
        assert_leg_log "stage1d stub: the lib leg also gets the tokenizer" \
            "[OXI_TOKENIZER=$dir_leg_full/tokenizer.json]" 1
        assert_leg_log "stage1d stub: no leg sees an exported OXIBONSAI_KERNEL_TIER" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 2
        run_leg_stub "stage1d stub: a missing Ternary-Bonsai-1.7B refuses before any cargo call" \
            1 0 "" run_legacy_lib_bin_stage "$dir_leg_no_17b"
        run_leg_stub "stage1d stub: a missing tokenizer refuses before any cargo call" \
            1 0 "" run_legacy_lib_bin_stage "$dir_leg_no_tok"
        run_leg_stub "stage1d stub: a failing lib leg stops the stage" \
            101 1 "--lib" run_legacy_lib_bin_stage "$dir_leg_full"

        # ── Bonsai 2 vision leg ──
        local dir_vis_full dir_vis_no_proj dir_vis_no_27b
        dir_vis_full="$work_dir/vis_models_full"
        dir_vis_no_proj="$work_dir/vis_models_no_projector"
        dir_vis_no_27b="$work_dir/vis_models_no_27b"
        mkdir -p "$dir_vis_full" "$dir_vis_no_proj" "$dir_vis_no_27b"
        printf 'x' >"$dir_vis_full/$BONSAI2_PQ2_FILE"
        printf 'x' >"$dir_vis_full/$BONSAI2_MMPROJ_FILE"
        printf 'x' >"$dir_vis_no_proj/$BONSAI2_PQ2_FILE"
        printf 'x' >"$dir_vis_no_27b/$BONSAI2_MMPROJ_FILE"

        run_leg_stub "vision stub: --skip-bonsai2-vision never invokes cargo" \
            0 0 "" run_bonsai2_vision_stage 1 0 "$dir_vis_full" 1
        run_leg_stub "vision stub: --skip-bonsai2-models skips the vision legs too" \
            0 0 "" run_bonsai2_vision_stage 0 1 "$dir_vis_full" 1
        run_leg_stub "vision stub: a non-Darwin host runs nothing" \
            0 0 "" run_bonsai2_vision_stage 0 0 "$dir_vis_full" 0
        run_leg_stub "vision stub: the 27B without its projector fails closed before any cargo call" \
            1 0 "" run_bonsai2_vision_stage 0 0 "$dir_vis_no_proj" 1
        total=$((total + 1))
        local vis_refusal=""
        vis_refusal="$(run_bonsai2_vision_stage 0 0 "$dir_vis_no_proj" 1 2>&1)" || true
        if [[ "$vis_refusal" == *"--skip-bonsai2-vision"* && "$vis_refusal" == *"$BONSAI2_MMPROJ_FILE"* ]]; then
            echo "OK: self-test 'vision stub: the refusal names the explicit --skip-bonsai2-vision opt-out'"
        else
            echo "FAIL: self-test 'vision stub: the refusal names the explicit --skip-bonsai2-vision opt-out'"
            failures=$((failures + 1))
        fi
        run_leg_stub "vision stub: the projector without the 27B fails closed before any cargo call" \
            1 0 "" run_bonsai2_vision_stage 0 0 "$dir_vis_no_27b" 1
        run_leg_stub "vision stub: 27B and projector present run the three binaries" \
            0 3 "" run_bonsai2_vision_stage 0 0 "$dir_vis_full" 1
        total=$((total + 1))
        local vis_order
        vis_order="$(grep -o 'bonsai2_vision_tests\|bonsai2_vision_runtime_tests\|vision_mmproj_tests' "$fake_leg_log" | tr '\n' ',')"
        if [[ "$vis_order" == "bonsai2_vision_tests,bonsai2_vision_runtime_tests,vision_mmproj_tests," ]]; then
            echo "OK: self-test 'vision stub: the model gate, the runtime round trip and the projector gate run in that order'"
        else
            echo "FAIL: self-test 'vision stub: the three binaries run in the documented order' (saw: $vis_order)"
            failures=$((failures + 1))
        fi
        assert_leg_log "vision stub: every leg is a --release, --nocapture, single-thread run" \
            "test --release -p " 3
        assert_leg_log "vision stub: every leg runs with OXI_REQUIRE_MODEL_FILES=1" \
            "[OXI_REQUIRE_MODEL_FILES=1]" 3
        assert_leg_log "vision stub: the 27B and projector reach the two language-model legs by path" \
            "[OXI_BONSAI2_PQ2_GGUF=$dir_vis_full/$BONSAI2_PQ2_FILE] [OXI_BONSAI2_MMPROJ_GGUF=$dir_vis_full/$BONSAI2_MMPROJ_FILE]" 2
        assert_leg_log "vision stub: the projector gate gets the projector by path, and no 27B" \
            "[OXI_BONSAI2_PQ2_GGUF=unset] [OXI_BONSAI2_MMPROJ_GGUF=$dir_vis_full/$BONSAI2_MMPROJ_FILE]" 1
        assert_leg_log "vision stub: no leg sees an exported OXIBONSAI_KERNEL_TIER" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 3
        run_leg_stub "vision stub: a failing runtime leg stops the stage after two binaries" \
            101 2 "bonsai2_vision_runtime_tests" run_bonsai2_vision_stage 0 0 "$dir_vis_full" 1

        # The predicate the stage and the capability requirement share:
        # `<skip_vision> <skip_models> <models_dir> <is_darwin>|<want>|<label>`.
        local vis_pred_case vis_pred_args vis_pred_rest vis_pred_got vis_pred_want vis_pred_text
        local vis_a vis_b vis_c vis_d
        for vis_pred_case in \
            "0 0 $dir_vis_full 1|1|a Darwin host with the projector requires the vision evidence" \
            "1 0 $dir_vis_full 1|0|--skip-bonsai2-vision drops the requirement" \
            "0 1 $dir_vis_full 1|0|--skip-bonsai2-models drops the requirement" \
            "0 0 $dir_vis_full 0|0|a non-Darwin host requires nothing" \
            "0 0 $dir_vis_no_proj 1|0|no projector, no by-name requirement (the stage itself fails closed)"; do
            total=$((total + 1))
            vis_pred_args="${vis_pred_case%%|*}"
            vis_pred_rest="${vis_pred_case#*|}"
            vis_pred_want="${vis_pred_rest%%|*}"
            vis_pred_text="${vis_pred_rest#*|}"
            read -r vis_a vis_b vis_c vis_d <<<"$vis_pred_args"
            vis_pred_got="$(bonsai2_vision_required "$vis_a" "$vis_b" "$vis_c" "$vis_d")"
            if [[ "$vis_pred_got" == "$vis_pred_want" ]]; then
                echo "OK: self-test 'bonsai2_vision_required: $vis_pred_text'"
            else
                echo "FAIL: self-test 'bonsai2_vision_required: $vis_pred_text': got $vis_pred_got, want $vis_pred_want"
                failures=$((failures + 1))
            fi
        done

        # ── Bonsai 2 vision leg, Metal half ──
        # Both bands and the projector: the full run; without the projector
        # or the PTQ1_0 band the stage fails closed before any cargo call.
        local dir_vm_full dir_vm_no_proj dir_vm_no_ptq1
        dir_vm_full="$work_dir/vm_models_full"
        dir_vm_no_proj="$work_dir/vm_models_no_projector"
        dir_vm_no_ptq1="$work_dir/vm_models_no_ptq1"
        mkdir -p "$dir_vm_full" "$dir_vm_no_proj" "$dir_vm_no_ptq1"
        printf 'x' >"$dir_vm_full/$BONSAI2_PQ2_FILE"
        printf 'x' >"$dir_vm_full/$BONSAI2_PTQ1_FILE"
        printf 'x' >"$dir_vm_full/$BONSAI2_MMPROJ_FILE"
        printf 'x' >"$dir_vm_no_proj/$BONSAI2_PQ2_FILE"
        printf 'x' >"$dir_vm_no_proj/$BONSAI2_PTQ1_FILE"
        printf 'x' >"$dir_vm_no_ptq1/$BONSAI2_PQ2_FILE"
        printf 'x' >"$dir_vm_no_ptq1/$BONSAI2_MMPROJ_FILE"

        run_leg_stub "vision-metal stub: --skip-bonsai2-vision never invokes cargo" \
            0 0 "" run_bonsai2_vision_metal_stage 1 0 0 "$dir_vm_full" 1
        run_leg_stub "vision-metal stub: --skip-bonsai2-models never invokes cargo" \
            0 0 "" run_bonsai2_vision_metal_stage 0 1 0 "$dir_vm_full" 1
        run_leg_stub "vision-metal stub: --skip-bonsai2-metal never invokes cargo" \
            0 0 "" run_bonsai2_vision_metal_stage 0 0 1 "$dir_vm_full" 1
        total=$((total + 1))
        local vm_skip_note=""
        vm_skip_note="$(run_bonsai2_vision_metal_stage 0 0 1 "$dir_vm_full" 1 2>&1)" || true
        if [[ "$vm_skip_note" == *"--skip-bonsai2-metal was passed"* && "$vm_skip_note" == *"release notes"* ]]; then
            echo "OK: self-test 'vision-metal stub: --skip-bonsai2-metal says out loud that the release carries no Metal vision evidence'"
        else
            echo "FAIL: self-test 'vision-metal stub: --skip-bonsai2-metal says so out loud' (saw: $vm_skip_note)"
            failures=$((failures + 1))
        fi
        run_leg_stub "vision-metal stub: a non-Darwin host runs nothing" \
            0 0 "" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_full" 0
        run_leg_stub "vision-metal stub: the 27B without its projector fails closed before any cargo call" \
            1 0 "" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_no_proj" 1
        run_leg_stub "vision-metal stub: a missing PTQ1_0 band fails closed before any cargo call" \
            1 0 "" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_no_ptq1" 1
        total=$((total + 1))
        local vm_refusal=""
        vm_refusal="$(run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_no_proj" 1 2>&1)" || true
        if [[ "$vm_refusal" == *"MISSING fixtures: $BONSAI2_MMPROJ_FILE"* \
            && "$vm_refusal" == *"--skip-bonsai2-metal"* && "$vm_refusal" == *"--skip-bonsai2-vision"* ]]; then
            echo "OK: self-test 'vision-metal stub: the refusal names the missing projector and both opt-outs'"
        else
            echo "FAIL: self-test 'vision-metal stub: the refusal names the missing projector and both opt-outs' (saw: $vm_refusal)"
            failures=$((failures + 1))
        fi
        run_leg_stub "vision-metal stub: 27B bands and projector present run the four binaries" \
            0 4 "" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_full" 1
        total=$((total + 1))
        local vm_order
        vm_order="$(grep -o -- '--test [a-z0-9_]*' "$fake_leg_log" | sed 's/--test //' | tr '\n' ',')"
        if [[ "$vm_order" == "vision_metal_tests,bonsai2_vision_metal_tests,bonsai2_vision_metal_runtime_tests,bonsai2_vision_cli_tests," ]]; then
            echo "OK: self-test 'vision-metal stub: the Metal tower, the 27B gate, the runtime round trip and the CLI run in that order'"
        else
            echo "FAIL: self-test 'vision-metal stub: the four binaries run in the documented order' (saw: $vm_order)"
            failures=$((failures + 1))
        fi
        assert_leg_log "vision-metal stub: every leg is a --release, --nocapture, single-thread run" \
            "-- --test-threads=1 --nocapture" 4
        assert_leg_log "vision-metal stub: the CLI leg runs oxibonsai-cli's bonsai2_vision_cli_tests in --release with every feature" \
            "test --release -p oxibonsai-cli --all-features --test bonsai2_vision_cli_tests -- --test-threads=1 --nocapture" 1
        assert_leg_log "vision-metal stub: every leg runs with OXI_REQUIRE_MODEL_FILES=1" \
            "[OXI_REQUIRE_MODEL_FILES=1]" 4
        assert_leg_log "vision-metal stub: the projector reaches every leg by path" \
            "[OXI_BONSAI2_MMPROJ_GGUF=$dir_vm_full/$BONSAI2_MMPROJ_FILE]" 4
        assert_leg_log "vision-metal stub: the 27B PQ2_0 reaches the three language-model legs by path" \
            "[OXI_BONSAI2_PQ2_GGUF=$dir_vm_full/$BONSAI2_PQ2_FILE]" 3
        assert_leg_log "vision-metal stub: the PTQ1_0 band reaches the both-band model gate only" \
            "[OXI_BONSAI2_PTQ1_GGUF=$dir_vm_full/$BONSAI2_PTQ1_FILE]" 1
        assert_leg_log "vision-metal stub: the Metal-tower gate gets the projector and no 27B" \
            "[OXI_BONSAI2_PQ2_GGUF=unset] [OXI_BONSAI2_MMPROJ_GGUF=$dir_vm_full/$BONSAI2_MMPROJ_FILE]" 1
        assert_leg_log "vision-metal stub: no leg sees an exported OXIBONSAI_KERNEL_TIER" \
            "[OXIBONSAI_KERNEL_TIER=unset]" 4
        run_leg_stub "vision-metal stub: a failing runtime leg stops the stage before the CLI leg" \
            101 3 "bonsai2_vision_metal_runtime_tests" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_full" 1
        run_leg_stub "vision-metal stub: a failing CLI leg's exit code is the stage's" \
            101 4 "bonsai2_vision_cli_tests" run_bonsai2_vision_metal_stage 0 0 0 "$dir_vm_full" 1

        # The predicate the stage's requirement reads:
        # `<skip_vision> <skip_models> <skip_metal> <models_dir> <is_darwin>|<want>|<label>`.
        local vm_pred_case vm_pred_args vm_pred_rest vm_pred_got vm_pred_want vm_pred_text
        local vm_a vm_b vm_c vm_d vm_e
        for vm_pred_case in \
            "0 0 0 $dir_vm_full 1|1|a Darwin host with the projector requires the Metal vision evidence" \
            "0 0 1 $dir_vm_full 1|0|--skip-bonsai2-metal drops the requirement" \
            "1 0 0 $dir_vm_full 1|0|--skip-bonsai2-vision drops it too" \
            "0 1 0 $dir_vm_full 1|0|--skip-bonsai2-models drops it too" \
            "0 0 0 $dir_vm_full 0|0|a non-Darwin host requires nothing" \
            "0 0 0 $dir_vm_no_proj 1|0|no projector, no by-name requirement (the stages fail closed)"; do
            total=$((total + 1))
            vm_pred_args="${vm_pred_case%%|*}"
            vm_pred_rest="${vm_pred_case#*|}"
            vm_pred_want="${vm_pred_rest%%|*}"
            vm_pred_text="${vm_pred_rest#*|}"
            read -r vm_a vm_b vm_c vm_d vm_e <<<"$vm_pred_args"
            vm_pred_got="$(bonsai2_vision_metal_required "$vm_a" "$vm_b" "$vm_c" "$vm_d" "$vm_e")"
            if [[ "$vm_pred_got" == "$vm_pred_want" ]]; then
                echo "OK: self-test 'bonsai2_vision_metal_required: $vm_pred_text'"
            else
                echo "FAIL: self-test 'bonsai2_vision_metal_required: $vm_pred_text': got $vm_pred_got, want $vm_pred_want"
                failures=$((failures + 1))
            fi
        done

        if [[ "$had_tier_leg" -eq 1 ]]; then
            export OXIBONSAI_KERNEL_TIER="$prior_tier_leg"
        else
            unset OXIBONSAI_KERNEL_TIER
        fi
    fi

    # ── Every name this script requires is one a test records ───────────
    # A requirement for a name no test writes could never be met (and a
    # renamed test would silently stop being gated), so each is checked
    # against the Rust source that records it.
    local name_label name_file name_value
    for name_label in \
        "${SPECULATIVE_TEST_FILE}|${SPECULATIVE_TEST_NAMES[0]}" \
        "${SPECULATIVE_TEST_FILE}|${SPECULATIVE_TEST_NAMES[1]}" \
        "${SPECULATIVE_ENGINE_TEST_FILE}|${SPECULATIVE_ENGINE_TEST_NAMES[0]}" \
        "${SPECULATIVE_ENGINE_TEST_FILE}|${SPECULATIVE_ENGINE_TEST_NAMES[1]}" \
        "${EMBED_TEST_FILE}|${EMBED_BENCH_TEST_NAME}" \
        "${EMBED_TEST_FILE}|${EMBED_SMOKE_TEST_NAME}" \
        "${SWEEP_TEST_FILE}|${SWEEP_TEST_NAME}" \
        "${FALLBACK_TEST_FILE}|${FALLBACK_TEST_NAME}" \
        "${BONSAI2_VISION_TEST_FILES[0]}|${BONSAI2_VISION_TEST_NAMES[0]}" \
        "${BONSAI2_VISION_TEST_FILES[1]}|${BONSAI2_VISION_TEST_NAMES[1]}" \
        "${BONSAI2_MMPROJ_TEST_FILE}|${BONSAI2_MMPROJ_TEST_NAME}" \
        "${BONSAI2_MMPROJ_METAL_TEST_FILE}|${BONSAI2_MMPROJ_METAL_TEST_NAME}" \
        "${BONSAI2_VISION_METAL_TEST_FILES[0]}|${BONSAI2_VISION_METAL_TEST_NAMES[0]}" \
        "${BONSAI2_VISION_METAL_TEST_FILES[1]}|${BONSAI2_VISION_METAL_TEST_NAMES[1]}" \
        "${BONSAI2_VISION_METAL_TEST_FILES[2]}|${BONSAI2_VISION_METAL_TEST_NAMES[2]}" \
        "${BONSAI2_PREFILL_TEST_FILE}|${BONSAI2_PREFILL_TEST_NAMES[0]}" \
        "${BONSAI2_PREFILL_TEST_FILE}|${BONSAI2_PREFILL_TEST_NAMES[1]}"; do
        name_file="${name_label%%|*}"
        name_value="${name_label#*|}"
        total=$((total + 1))
        echo ""
        echo "-- self-test: the required name ${name_value##*::} is one $name_file records --"
        if test_name_in_source "$name_file" "$name_value"; then
            echo "OK: self-test 'the required name ${name_value##*::} is one $name_file records'"
        else
            echo "FAIL: self-test 'the required name ${name_value##*::} is one $name_file records'"
            echo "  $name_value is not a string literal in $name_file"
            failures=$((failures + 1))
        fi
    done

    # Each new name is part of the legacy-models requirement, and the vision
    # names are exactly what the vision requirement carries.
    local legacy_required
    legacy_required=",$(legacy_models_require_tests_arg 0),"
    for name_value in "${SPECULATIVE_TEST_NAMES[@]}" "${SPECULATIVE_ENGINE_TEST_NAMES[@]}" \
        "$EMBED_BENCH_TEST_NAME" "$EMBED_SMOKE_TEST_NAME" "$SWEEP_TEST_NAME" "$FALLBACK_TEST_NAME"; do
        total=$((total + 1))
        echo ""
        echo "-- self-test: legacy-models requires ${name_value##*::} by name --"
        if [[ "$legacy_required" == *",$name_value,"* ]]; then
            echo "OK: self-test 'legacy-models requires ${name_value##*::} by name'"
        else
            echo "FAIL: self-test 'legacy-models requires ${name_value##*::} by name'"
            failures=$((failures + 1))
        fi
    done

    # A leg that self-skipped (executed:false) is a failed gate, one name at a
    # time: no required legacy-models name can be satisfied by another's record.
    local one_off flag
    for one_off in "${legacy_names[@]}"; do
        : >"$work_dir/legacy_each.jsonl"
        for legacy_name in "${legacy_names[@]}"; do
            if [[ "$legacy_name" == "$one_off" ]]; then flag=false; else flag=true; fi
            printf '{"capability":"legacy-models","executed":%s,"test":"%s","duration_ms":1}\n' \
                "$flag" "$legacy_name" >>"$work_dir/legacy_each.jsonl"
        done
        check "legacy-models: only ${one_off##*::} self-skipped -> FAIL" 1 \
            "$work_dir/legacy_each.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    done

    # The libtest filter the sweep leg passes must select the sweep's own test
    # function (a rename would otherwise leave a leg that runs zero tests and
    # exits 0), and the capability name it requires must carry that function's
    # name.
    total=$((total + 1))
    echo ""
    echo "-- self-test: sweep: the libtest filter is the name of the sweep's test function --"
    if grep -q -F "fn $SWEEP_TEST_FILTER()" "$PROJECT_ROOT/$SWEEP_TEST_FILE" \
        && [[ "$SWEEP_TEST_NAME" == *"::$SWEEP_TEST_FILTER" ]]; then
        echo "OK: self-test 'sweep: the libtest filter is the name of the sweep's test function'"
    else
        echo "FAIL: self-test 'sweep: the libtest filter is the name of the sweep's test function'"
        echo "  fn $SWEEP_TEST_FILTER() is not defined in $SWEEP_TEST_FILE, or $SWEEP_TEST_NAME does not end in it"
        failures=$((failures + 1))
    fi

    # The M-18 sweep requirement: it is a legacy-models gate required by its
    # own name, so a manifest in which only the sweep self-skipped (no
    # `OXI_MODEL`, so no model ran) or never ran at all fails, and no other
    # test's record can stand in for it. Every other required name is
    # `executed: true` here, so the sweep is the only reason for a FAIL.
    local sweep_manifest_name
    for sweep_manifest_name in skipped absent other ran; do
        : >"$work_dir/sweep_$sweep_manifest_name.jsonl"
        for legacy_name in "${legacy_names[@]}"; do
            if [[ "$legacy_name" != "$SWEEP_TEST_NAME" ]]; then
                printf '{"capability":"legacy-models","executed":true,"test":"%s","duration_ms":1}\n' \
                    "$legacy_name" >>"$work_dir/sweep_$sweep_manifest_name.jsonl"
            fi
        done
    done
    printf '{"capability":"legacy-models","executed":false,"test":"%s"}\n' "$SWEEP_TEST_NAME" \
        >>"$work_dir/sweep_skipped.jsonl"
    printf '{"capability":"legacy-models","executed":true,"test":"some::other_legacy_models_test","duration_ms":1}\n' \
        >>"$work_dir/sweep_other.jsonl"
    printf '{"capability":"legacy-models","executed":true,"test":"%s","duration_ms":245898}\n' "$SWEEP_TEST_NAME" \
        >>"$work_dir/sweep_ran.jsonl"
    check "sweep: the M-18 sweep self-skipped (executed:false) -> FAIL" 1 \
        "$work_dir/sweep_skipped.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    check "sweep: the M-18 sweep never ran (no record) -> FAIL" 1 \
        "$work_dir/sweep_absent.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    check "sweep: an executed record of a DIFFERENT legacy-models test does not satisfy the sweep -> FAIL" 1 \
        "$work_dir/sweep_other.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"
    check_output "sweep: the M-18 sweep ran (timed) alongside every other required gate -> PASS, no WARN" 0 \
        "OK: 'legacy-models' has executed=true for all" "WARN" \
        "$work_dir/sweep_ran.jsonl" "$now" "$(legacy_models_require_tests_arg 0)"

    # The vision requirement: both names of "bonsai2-vision" and the
    # projector gate's, each individually load-bearing.
    local vis_names_csv
    vis_names_csv="$(IFS=,; echo "${BONSAI2_VISION_TEST_NAMES[*]}")"
    printf '%s\n%s\n%s\n' \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[0]}\",\"duration_ms\":5}" \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[1]}\",\"duration_ms\":5}" \
        "{\"capability\":\"bonsai2-mmproj\",\"executed\":true,\"test\":\"$BONSAI2_MMPROJ_TEST_NAME\",\"duration_ms\":5}" \
        >"$work_dir/vision_all.jsonl"
    check "vision: every required name executed -> PASS" 0 \
        "$work_dir/vision_all.jsonl" "$now" \
        bonsai2-vision "--require-tests=bonsai2-vision:$vis_names_csv" \
        bonsai2-mmproj "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME"
    printf '%s\n%s\n%s\n' \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[0]}\",\"duration_ms\":5}" \
        "{\"capability\":\"bonsai2-vision\",\"executed\":false,\"test\":\"${BONSAI2_VISION_TEST_NAMES[1]}\"}" \
        "{\"capability\":\"bonsai2-mmproj\",\"executed\":true,\"test\":\"$BONSAI2_MMPROJ_TEST_NAME\",\"duration_ms\":5}" \
        >"$work_dir/vision_runtime_skipped.jsonl"
    check "vision: the runtime round trip self-skipped -> FAIL" 1 \
        "$work_dir/vision_runtime_skipped.jsonl" "$now" \
        bonsai2-vision "--require-tests=bonsai2-vision:$vis_names_csv" \
        bonsai2-mmproj "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME"
    printf '%s\n%s\n%s\n' \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[0]}\",\"duration_ms\":5}" \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[1]}\",\"duration_ms\":5}" \
        "{\"capability\":\"bonsai2-mmproj\",\"executed\":false,\"test\":\"$BONSAI2_MMPROJ_TEST_NAME\"}" \
        >"$work_dir/vision_mmproj_skipped.jsonl"
    check "vision: the projector gate self-skipped -> FAIL" 1 \
        "$work_dir/vision_mmproj_skipped.jsonl" "$now" \
        bonsai2-vision "--require-tests=bonsai2-vision:$vis_names_csv" \
        bonsai2-mmproj "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME"

    # The Metal half: every "bonsai2-vision-metal" name and the Metal tower's
    # second "bonsai2-mmproj" name, each individually load-bearing, read the
    # way the gate builds the requirement (both "bonsai2-mmproj" lists
    # accumulate).
    local vm_names_csv vm_name vm_one_off vm_flag
    vm_names_csv="$(IFS=,; echo "${BONSAI2_VISION_METAL_TEST_NAMES[*]}")"
    vm_manifest() {
        # vm_manifest <out> <name whose record is executed:false, or "">
        # <write the Metal-tower record: yes|no>
        local out="$1" skipped="$2" tower="$3" n f
        printf '%s\n' \
            "{\"capability\":\"bonsai2-mmproj\",\"executed\":true,\"test\":\"$BONSAI2_MMPROJ_TEST_NAME\",\"duration_ms\":5}" \
            >"$out"
        if [[ "$tower" == "yes" ]]; then
            printf '%s\n' \
                "{\"capability\":\"bonsai2-mmproj\",\"executed\":true,\"test\":\"$BONSAI2_MMPROJ_METAL_TEST_NAME\",\"duration_ms\":5}" \
                >>"$out"
        fi
        for n in "${BONSAI2_VISION_METAL_TEST_NAMES[@]}"; do
            if [[ "$n" == "$skipped" ]]; then f=false; else f=true; fi
            printf '{"capability":"bonsai2-vision-metal","executed":%s,"test":"%s","duration_ms":7}\n' \
                "$f" "$n" >>"$out"
        done
    }
    vm_manifest "$work_dir/vision_metal_all.jsonl" "" yes
    check_output "vision-metal: every required name executed (timed) -> PASS, no WARN" 0 \
        "OK: 'bonsai2-vision-metal' has executed=true for all" "WARN" \
        "$work_dir/vision_metal_all.jsonl" "$now" \
        "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME" \
        "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_METAL_TEST_NAME" \
        bonsai2-vision-metal "--require-tests=bonsai2-vision-metal:$vm_names_csv"
    for vm_one_off in "${BONSAI2_VISION_METAL_TEST_NAMES[@]}"; do
        vm_manifest "$work_dir/vision_metal_each.jsonl" "$vm_one_off" yes
        check "vision-metal: only ${vm_one_off##*::} self-skipped -> FAIL" 1 \
            "$work_dir/vision_metal_each.jsonl" "$now" \
            bonsai2-vision-metal "--require-tests=bonsai2-vision-metal:$vm_names_csv"
    done
    vm_manifest "$work_dir/vision_metal_no_tower.jsonl" "" no
    check "vision-metal: the Metal tower's gate never ran (only the CPU projector gate did) -> FAIL" 1 \
        "$work_dir/vision_metal_no_tower.jsonl" "$now" \
        "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME" \
        "--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_METAL_TEST_NAME"
    : >"$work_dir/vision_metal_absent.jsonl"
    printf '%s\n' \
        "{\"capability\":\"bonsai2-vision\",\"executed\":true,\"test\":\"${BONSAI2_VISION_TEST_NAMES[0]}\",\"duration_ms\":5}" \
        >"$work_dir/vision_metal_absent.jsonl"
    check "vision-metal: no bonsai2-vision-metal record at all (a CPU vision record cannot stand in) -> FAIL" 1 \
        "$work_dir/vision_metal_absent.jsonl" "$now" \
        bonsai2-vision-metal "--require-tests=bonsai2-vision-metal:$vm_names_csv"
    # The batched-prefill gates are "bonsai2-models" names: a self-skip of
    # either fails the requirement on its own.
    local prefill_csv
    prefill_csv="$(IFS=,; echo "${BONSAI2_PREFILL_TEST_NAMES[*]}")"
    for vm_name in "${BONSAI2_PREFILL_TEST_NAMES[@]}"; do
        : >"$work_dir/prefill_each.jsonl"
        for vm_one_off in "${BONSAI2_PREFILL_TEST_NAMES[@]}"; do
            if [[ "$vm_one_off" == "$vm_name" ]]; then vm_flag=false; else vm_flag=true; fi
            printf '{"capability":"bonsai2-models","executed":%s,"test":"%s","duration_ms":9}\n' \
                "$vm_flag" "$vm_one_off" >>"$work_dir/prefill_each.jsonl"
        done
        check "bonsai2-models: only ${vm_name##*::} self-skipped -> FAIL" 1 \
            "$work_dir/prefill_each.jsonl" "$now" \
            bonsai2-models "--require-tests=bonsai2-models:$prefill_csv"
    done

    # ── build_release_cli_binary: stage 0 fails closed ───────────────────
    local fake_cli_bin fake_cli_log cli_err
    fake_cli_bin="$(mktemp -d "${TMPDIR:-/tmp}/oxibonsai_release_gate_selftest_cli.XXXXXX")" || {
        echo "FAIL: self-test: could not create a scratch bin directory" >&2
        failures=$((failures + 1))
        fake_cli_bin=""
    }
    if [[ -n "$fake_cli_bin" ]]; then
        fake_cli_log="$work_dir/fake_cli_cargo.log"
        cli_err="$work_dir/cli_stage_stderr.log"
        cat >"$fake_cli_bin/cargo" <<'CLI_CARGO_STUB_EOF'
#!/usr/bin/env bash
echo "$*" >>"$OXIBONSAI_SELFTEST_CARGO_LOG"
case "${OXIBONSAI_SELFTEST_CARGO_MODE:-ok}" in
    fail)
        echo "error: could not compile (self-test stand-in, expected)" >&2
        exit 101
        ;;
    no_artifact)
        printf '%s\n' '{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"oxibonsai"},"executable":null}'
        printf '%s\n' '{"reason":"build-finished","success":true}'
        exit 0
        ;;
    build_finished_false)
        printf '{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"oxibonsai"},"executable":"%s"}\n' \
            "$OXIBONSAI_SELFTEST_CLI_PATH"
        printf '%s\n' '{"reason":"build-finished","success":false}'
        exit 0
        ;;
    *)
        printf '%s\n' '{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"oxibonsai"},"executable":null}'
        printf '%s\n' '{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"oxibonsai-serve"},"executable":"target/release/oxibonsai-serve"}'
        printf '{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"oxibonsai"},"executable":"%s"}\n' \
            "$OXIBONSAI_SELFTEST_CLI_PATH"
        printf '%s\n' '{"reason":"build-finished","success":true}'
        exit 0
        ;;
esac
CLI_CARGO_STUB_EOF
        chmod +x "$fake_cli_bin/cargo"
        local cli_good cli_bad_version cli_gone
        cli_good="$fake_cli_bin/oxibonsai_good"
        cli_bad_version="$fake_cli_bin/oxibonsai_bad_version"
        cli_gone="$fake_cli_bin/oxibonsai_deleted"
        printf '#!/bin/sh\necho "oxibonsai 0.0.0"\nexit 0\n' >"$cli_good"
        printf '#!/bin/sh\necho "boom" >&2\nexit 2\n' >"$cli_bad_version"
        chmod +x "$cli_good" "$cli_bad_version"

        # run_cli_stub <label> <expect_ok:0|1> <cargo mode> <reported cli path>
        run_cli_stub() {
            local label="$1" expect_ok="$2" mode="$3" cli_path="$4"
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            : >"$fake_cli_log"
            local rc=0 printed
            printed="$(OXIBONSAI_SELFTEST_CARGO_LOG="$fake_cli_log" OXIBONSAI_SELFTEST_CARGO_MODE="$mode" \
                OXIBONSAI_SELFTEST_CLI_PATH="$cli_path" PATH="$fake_cli_bin:$PATH" \
                OXIBONSAI_CLI_BIN="$fake_cli_bin/a-stale-value-the-gate-must-ignore" \
                build_release_cli_binary 2>"$cli_err")" || rc=$?
            if [[ "$expect_ok" -eq 1 ]]; then
                if [[ "$rc" -ne 0 || "$printed" != "$cli_path" ]]; then
                    echo "FAIL: self-test '$label': expected exit 0 printing $cli_path, got exit $rc printing '$printed'"
                    echo "  stderr: $(cat "$cli_err")"
                    failures=$((failures + 1))
                    return
                fi
            else
                if [[ "$rc" -eq 0 || -n "$printed" ]]; then
                    echo "FAIL: self-test '$label': expected a non-zero exit and nothing on stdout, got exit $rc printing '$printed'"
                    failures=$((failures + 1))
                    return
                fi
            fi
            echo "OK: self-test '$label' (exit $rc, as expected)"
        }

        run_cli_stub "stage 0 stub: the reported --all-features release binary is resolved and printed" \
            1 ok "$cli_good"
        total=$((total + 1))
        if [[ "$(cat "$fake_cli_log")" == "${RELEASE_CLI_BUILD_ARGS[*]}" ]]; then
            echo "OK: self-test 'stage 0 stub: cargo receives exactly the --all-features release build arguments'"
        else
            echo "FAIL: self-test 'stage 0 stub: cargo receives exactly the --all-features release build arguments'"
            echo "  expected: ${RELEASE_CLI_BUILD_ARGS[*]}"
            echo "  got:      $(cat "$fake_cli_log")"
            failures=$((failures + 1))
        fi
        run_cli_stub "stage 0 stub: a failed cargo build fails closed" 0 fail "$cli_good"
        run_cli_stub "stage 0 stub: a build stream without an executable for the CLI fails closed" \
            0 no_artifact "$cli_good"
        run_cli_stub "stage 0 stub: a stream reporting build-finished success=false fails closed" \
            0 build_finished_false "$cli_good"
        run_cli_stub "stage 0 stub: a missing CLI binary fails closed" 0 ok "$cli_gone"
        run_cli_stub "stage 0 stub: a CLI binary that fails --version fails closed" 0 ok "$cli_bad_version"

        # The main flow's own stage 0, end to end: a copy of this script in a
        # scratch tree with a stand-in `ci.sh` (it records the CLI binary the
        # environment carries when stage 1 starts, then fails so the gate
        # stops there) and the stand-in `cargo` first on PATH.
        local gate_tree gate_log ci_marker
        gate_tree="$work_dir/gate_tree"
        gate_log="$work_dir/gate_flow.log"
        ci_marker="$work_dir/ci_marker.txt"
        mkdir -p "$gate_tree/scripts"
        cp "$SCRIPT_DIR/release-gate.sh" "$gate_tree/scripts/release-gate.sh"
        cat >"$gate_tree/scripts/ci.sh" <<'CI_STUB_EOF'
#!/usr/bin/env bash
echo "${OXIBONSAI_CLI_BIN-unset}" >"$OXIBONSAI_SELFTEST_CI_MARKER"
exit 1
CI_STUB_EOF
        chmod +x "$gate_tree/scripts/ci.sh"

        # run_gate_flow_stub <label> <cargo mode> <expected marker: "absent" or the exact CLI path>
        run_gate_flow_stub() {
            local label="$1" mode="$2" expect_marker="$3"
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            rm -f "$ci_marker"
            local rc=0
            env -u CARGO_TARGET_DIR OXIBONSAI_SELFTEST_CI_MARKER="$ci_marker" \
                OXIBONSAI_SELFTEST_CARGO_LOG="$fake_cli_log" OXIBONSAI_SELFTEST_CARGO_MODE="$mode" \
                OXIBONSAI_SELFTEST_CLI_PATH="$cli_good" PATH="$fake_cli_bin:$PATH" \
                OXIBONSAI_CLI_BIN="$fake_cli_bin/a-stale-value-the-gate-must-ignore" \
                bash "$gate_tree/scripts/release-gate.sh" --skip-legacy-models --skip-bonsai2-models \
                >"$gate_log" 2>&1 || rc=$?
            if [[ "$rc" -eq 0 ]]; then
                echo "FAIL: self-test '$label': the gate exited 0 although its stand-in ci.sh fails"
                failures=$((failures + 1))
            elif [[ "$expect_marker" == "absent" && -e "$ci_marker" ]]; then
                echo "FAIL: self-test '$label': ci.sh ran although stage 0 failed (it saw: $(cat "$ci_marker"))"
                failures=$((failures + 1))
            elif [[ "$expect_marker" != "absent" && "$(cat "$ci_marker" 2>/dev/null)" != "$expect_marker" ]]; then
                echo "FAIL: self-test '$label': ci.sh saw OXIBONSAI_CLI_BIN='$(cat "$ci_marker" 2>/dev/null)', expected '$expect_marker'"
                echo "  gate output tail: $(tail -n 12 "$gate_log")"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, as expected)"
            fi
        }
        run_gate_flow_stub "stage 0 wiring: a failed CLI build stops the gate before ci.sh runs" \
            fail absent
        run_gate_flow_stub "stage 0 wiring: no runnable CLI binary stops the gate before ci.sh runs" \
            no_artifact absent
        run_gate_flow_stub "stage 0 wiring: the built binary is exported to ci.sh, overriding a stale OXIBONSAI_CLI_BIN" \
            ok "$cli_good"

        # The previous run's capability manifest is moved aside (kept under a
        # timestamped name), never deleted, and the live path is gone when
        # stage 1 starts: a stand-in ci.sh records whether the manifest path
        # exists at that moment and then fails, so the gate stops there.
        local rot_target rot_report rot_marker rot_log rot_old_line rot_expected=0
        rot_target="$gate_tree/target"
        rot_report="$rot_target/capability-report.json"
        rot_marker="$work_dir/rotation_marker.txt"
        rot_log="$work_dir/rotation_flow.log"
        rot_old_line='{"capability":"metal","executed":true,"test":"earlier::run"}'
        cat >"$gate_tree/scripts/ci.sh" <<'ROT_CI_STUB_EOF'
#!/usr/bin/env bash
if [[ -e "${OXIBONSAI_CAPABILITY_REPORT:?}" ]]; then
    echo "present" >"$OXIBONSAI_SELFTEST_CI_MARKER"
else
    echo "absent" >"$OXIBONSAI_SELFTEST_CI_MARKER"
fi
exit 1
ROT_CI_STUB_EOF
        chmod +x "$gate_tree/scripts/ci.sh"
        rm -f "$rot_report" "$rot_report".*

        # run_rotation_flow <label> <leftover: yes|no>
        # A leftover manifest adds one rotated file holding its line; no
        # leftover adds none. Either way the live path is absent at stage 1
        # and again when the gate has stopped.
        run_rotation_flow() {
            local label="$1" leftover="$2"
            echo ""
            echo "-- self-test: $label --"
            total=$((total + 1))
            rm -f "$rot_marker"
            mkdir -p "$rot_target"
            if [[ "$leftover" == "yes" ]]; then
                printf '%s\n' "$rot_old_line" >"$rot_report"
                rot_expected=$((rot_expected + 1))
            fi
            local rc=0
            env -u CARGO_TARGET_DIR OXIBONSAI_SELFTEST_CI_MARKER="$rot_marker" \
                OXIBONSAI_SELFTEST_CARGO_LOG="$fake_cli_log" OXIBONSAI_SELFTEST_CARGO_MODE=ok \
                OXIBONSAI_SELFTEST_CLI_PATH="$cli_good" PATH="$fake_cli_bin:$PATH" \
                bash "$gate_tree/scripts/release-gate.sh" --skip-legacy-models --skip-bonsai2-models \
                >"$rot_log" 2>&1 || rc=$?
            local rotated_files=("$rot_report".*) rotated_count=0 holding_old=0 rotated_file
            for rotated_file in "${rotated_files[@]}"; do
                [[ -e "$rotated_file" ]] || continue
                rotated_count=$((rotated_count + 1))
                if grep -q -F -- "$rot_old_line" "$rotated_file"; then
                    holding_old=$((holding_old + 1))
                fi
            done
            local kept_note=0
            grep -q -F -- "previous capability manifest kept as:" "$rot_log" && kept_note=1
            if [[ "$rc" -eq 0 ]]; then
                echo "FAIL: self-test '$label': the gate exited 0 although its stand-in ci.sh fails"
                failures=$((failures + 1))
            elif [[ "$(cat "$rot_marker" 2>/dev/null)" != "absent" ]]; then
                echo "FAIL: self-test '$label': the manifest path was '$(cat "$rot_marker" 2>/dev/null)' when ci.sh started, expected absent"
                echo "  gate output tail: $(tail -n 12 "$rot_log")"
                failures=$((failures + 1))
            elif [[ -e "$rot_report" ]]; then
                echo "FAIL: self-test '$label': the live manifest path exists after the run"
                failures=$((failures + 1))
            elif [[ "$rotated_count" -ne "$rot_expected" || "$holding_old" -ne "$rot_expected" ]]; then
                echo "FAIL: self-test '$label': $rotated_count rotated file(s), $holding_old holding the earlier line; expected $rot_expected of each"
                failures=$((failures + 1))
            elif [[ "$leftover" == "yes" && "$kept_note" -ne 1 ]]; then
                echo "FAIL: self-test '$label': the gate did not say where the earlier manifest was kept"
                failures=$((failures + 1))
            elif [[ "$leftover" == "no" && "$kept_note" -ne 0 ]]; then
                echo "FAIL: self-test '$label': the gate reported moving a manifest that did not exist"
                failures=$((failures + 1))
            else
                echo "OK: self-test '$label' (exit $rc, $rotated_count rotated file(s))"
            fi
        }
        run_rotation_flow "manifest rotation: an earlier run's manifest is moved aside with a timestamp, not deleted, and the run starts without one" yes
        run_rotation_flow "manifest rotation: with no earlier manifest nothing is moved and the run still starts without one" no
        run_rotation_flow "manifest rotation: a second earlier manifest gets its own name and the first is kept" yes
        total=$((total + 1))
        echo ""
        echo "-- self-test: manifest rotation: the rotated names carry a UTC timestamp and the gate's pid --"
        local rotated_shape_ok=1 rotated_name
        for rotated_name in "$rot_report".*; do
            if [[ "${rotated_name##*/}" != capability-report.json.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z.[0-9]* ]]; then
                rotated_shape_ok=0
            fi
        done
        if [[ "$rotated_shape_ok" -eq 1 ]]; then
            echo "OK: self-test 'manifest rotation: the rotated names carry a UTC timestamp and the gate's pid'"
        else
            echo "FAIL: self-test 'manifest rotation: the rotated names carry a UTC timestamp and the gate's pid' ($(ls "$rot_target"))"
            failures=$((failures + 1))
        fi
        rm -f "$rot_report".*

        # Stage 1b's Metal hidden-state leg and the capability requirement,
        # end to end (Darwin only: the legacy stages are Darwin-only): with a
        # passing stand-in ci.sh, a stand-in cargo that succeeds at everything
        # and a complete scratch models directory, the gate runs
        # `metal_hidden_parity_tests` once per legacy model, requires
        # "metal-hidden" by name — and then, since no test wrote a manifest,
        # fails at the capability check rather than passing.
        if [[ "$(uname -s)" == "Darwin" ]]; then
            local flow_models flow_file flow_calls flow_required
            flow_models="$work_dir/flow_models"
            mkdir -p "$flow_models/Ternary-Bonsai-1.7B-ONNX/onnx"
            for flow_file in "${LEGACY_MODEL_FILES[@]}" tokenizer.json \
                Ternary-Bonsai-1.7B-ONNX/onnx/model_q2.onnx; do
                printf 'x' >"$flow_models/$flow_file"
            done
            printf '#!/usr/bin/env bash\nexit 0\n' >"$gate_tree/scripts/ci.sh"
            chmod +x "$gate_tree/scripts/ci.sh"
            echo ""
            echo "-- self-test: stage 1b wiring: the Metal hidden-state leg runs once per legacy model and metal-hidden is required by name --"
            total=$((total + 1))
            : >"$fake_cli_log"
            local flow_rc=0
            env -u CARGO_TARGET_DIR OXIBONSAI_MODELS_DIR="$flow_models" \
                OXIBONSAI_SELFTEST_CARGO_LOG="$fake_cli_log" OXIBONSAI_SELFTEST_CARGO_MODE=ok \
                OXIBONSAI_SELFTEST_CLI_PATH="$cli_good" PATH="$fake_cli_bin:$PATH" \
                bash "$gate_tree/scripts/release-gate.sh" --skip-bonsai2-models \
                >"$gate_log" 2>&1 || flow_rc=$?
            flow_calls="$(grep -c -F -- '--test metal_hidden_parity_tests' "$fake_cli_log")"
            flow_required="$(grep -F 'Required on this host:' "$gate_log")"
            if [[ "$flow_rc" -eq 0 ]]; then
                echo "FAIL: the gate passed although no test wrote a capability manifest"
                failures=$((failures + 1))
            elif [[ "$flow_calls" -ne 3 ]]; then
                echo "FAIL: expected 3 metal_hidden_parity_tests invocations, got $flow_calls"
                echo "  invocation log: $(cat "$fake_cli_log")"
                failures=$((failures + 1))
            elif [[ "$flow_required" != *"metal-hidden --require-tests=metal-hidden:$METAL_HIDDEN_TEST_NAME"* ]]; then
                echo "FAIL: metal-hidden is not required by name: $flow_required"
                failures=$((failures + 1))
            else
                echo "OK: self-test 'stage 1b wiring: the Metal hidden-state leg runs once per legacy model and metal-hidden is required by name' (exit $flow_rc, $flow_calls legs)"
            fi

            # The same run's other legacy legs: the speculative gate, the
            # embedding leg (with its explicit opt-in) and stage 1d's Metal
            # fallback leg each ran exactly once, and each is required by name.
            local flow_leg flow_leg_label flow_leg_want flow_leg_got
            for flow_leg in \
                "--test speculative_ternary_metal_gates|the speculative gate runs once|1" \
                "--test embeddings_model_backed|the embedding leg runs once|1" \
                "$SWEEP_TEST_FILTER|the M-18 sweep leg runs once per model (twice)|2" \
                "--test metal_greedy_cpu_fallback_tests|the Metal fallback leg runs once|1"; do
                flow_leg_label="${flow_leg#*|}"
                flow_leg_want="${flow_leg_label#*|}"
                flow_leg_label="${flow_leg_label%%|*}"
                flow_leg_got="$(grep -c -F -- "${flow_leg%%|*}" "$fake_cli_log")"
                total=$((total + 1))
                echo ""
                echo "-- self-test: stage 1b wiring: $flow_leg_label --"
                if [[ "$flow_leg_got" -eq "$flow_leg_want" ]]; then
                    echo "OK: self-test 'stage 1b wiring: $flow_leg_label'"
                else
                    echo "FAIL: self-test 'stage 1b wiring: $flow_leg_label': ${flow_leg%%|*} on $flow_leg_got invocation(s), expected $flow_leg_want"
                    echo "  invocation log: $(cat "$fake_cli_log")"
                    failures=$((failures + 1))
                fi
            done
            local flow_name
            for flow_name in "${SPECULATIVE_TEST_NAMES[@]}" "${SPECULATIVE_ENGINE_TEST_NAMES[@]}" \
                "$EMBED_BENCH_TEST_NAME" "$EMBED_SMOKE_TEST_NAME" "$SWEEP_TEST_NAME" "$FALLBACK_TEST_NAME"; do
                total=$((total + 1))
                echo ""
                echo "-- self-test: stage 1b wiring: ${flow_name##*::} is required by name --"
                if [[ "$flow_required" == *"$flow_name"* ]]; then
                    echo "OK: self-test 'stage 1b wiring: ${flow_name##*::} is required by name'"
                else
                    echo "FAIL: self-test 'stage 1b wiring: ${flow_name##*::} is required by name': $flow_required"
                    failures=$((failures + 1))
                fi
            done
            total=$((total + 1))
            echo ""
            echo "-- self-test: stage 1b wiring: no vision requirement under --skip-bonsai2-models --"
            if [[ "$flow_required" != *"bonsai2-vision"* && "$flow_required" != *"bonsai2-mmproj"* ]]; then
                echo "OK: self-test 'stage 1b wiring: no vision requirement under --skip-bonsai2-models'"
            else
                echo "FAIL: self-test 'stage 1b wiring: no vision requirement under --skip-bonsai2-models': $flow_required"
                failures=$((failures + 1))
            fi

            # Stage 1c's vision leg end to end, on a scratch models directory
            # with both 27B bands: the projector present (the three binaries
            # run and both vision capabilities are required by name), the
            # projector absent (fail closed before any vision binary runs, the
            # refusal naming the opt-out) and --skip-bonsai2-vision (no vision
            # binary, no vision requirement).
            local vflow_models vflow_rc vflow_log
            vflow_models="$work_dir/vflow_models"
            mkdir -p "$vflow_models"
            printf 'x' >"$vflow_models/$BONSAI2_PQ2_FILE"
            printf 'x' >"$vflow_models/$BONSAI2_PTQ1_FILE"
            vflow_log="$work_dir/vflow.log"
            # run_vflow <label> <expect: ok|refused|skipped> <extra gate args...>
            run_vflow() {
                local label="$1" expect="$2"
                shift 2
                total=$((total + 1))
                echo ""
                echo "-- self-test: $label --"
                : >"$fake_cli_log"
                vflow_rc=0
                env -u CARGO_TARGET_DIR OXIBONSAI_MODELS_DIR="$vflow_models" \
                    OXIBONSAI_SELFTEST_CARGO_LOG="$fake_cli_log" OXIBONSAI_SELFTEST_CARGO_MODE=ok \
                    OXIBONSAI_SELFTEST_CLI_PATH="$cli_good" PATH="$fake_cli_bin:$PATH" \
                    bash "$gate_tree/scripts/release-gate.sh" --skip-legacy-models "$@" \
                    >"$vflow_log" 2>&1 || vflow_rc=$?
                local vis_calls metal_calls req
                vis_calls="$(grep -c -E -- '--test (bonsai2_vision_tests|bonsai2_vision_runtime_tests|vision_mmproj_tests) ' "$fake_cli_log")"
                metal_calls="$(grep -c -E -- '--test (vision_metal_tests|bonsai2_vision_metal_tests|bonsai2_vision_metal_runtime_tests|bonsai2_vision_cli_tests) ' "$fake_cli_log")"
                req="$(grep -F 'Required on this host:' "$vflow_log")"
                local names_csv metal_csv metal_req
                names_csv="$(IFS=,; echo "${BONSAI2_VISION_TEST_NAMES[*]}")"
                metal_csv="$(IFS=,; echo "${BONSAI2_VISION_METAL_TEST_NAMES[*]}")"
                metal_req="--require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_METAL_TEST_NAME bonsai2-vision-metal --require-tests=bonsai2-vision-metal:$metal_csv"
                if [[ "$vflow_rc" -eq 0 ]]; then
                    echo "FAIL: self-test '$label': the gate passed although no test wrote a capability manifest"
                    failures=$((failures + 1))
                    return
                fi
                case "$expect" in
                    ok)
                        if [[ "$vis_calls" -ne 3 || "$metal_calls" -ne 4 \
                            || "$req" != *"bonsai2-vision --require-tests=bonsai2-vision:$names_csv bonsai2-mmproj --require-tests=bonsai2-mmproj:$BONSAI2_MMPROJ_TEST_NAME"* \
                            || "$req" != *"$metal_req"* ]]; then
                            echo "FAIL: self-test '$label': $vis_calls CPU and $metal_calls Metal vision binaries ran; required line: $req"
                            failures=$((failures + 1))
                            return
                        fi
                        ;;
                    no_metal)
                        if [[ "$vis_calls" -ne 3 || "$metal_calls" -ne 0 \
                            || "$req" != *"bonsai2-vision --require-tests=bonsai2-vision:$names_csv"* \
                            || "$req" == *"bonsai2-vision-metal"* \
                            || "$req" == *"$BONSAI2_MMPROJ_METAL_TEST_NAME"* ]]; then
                            echo "FAIL: self-test '$label': $vis_calls CPU and $metal_calls Metal vision binaries ran; required line: $req"
                            failures=$((failures + 1))
                            return
                        fi
                        ;;
                    refused)
                        if [[ "$vis_calls" -ne 0 || "$metal_calls" -ne 0 || "$req" != "" \
                            || "$(cat "$vflow_log")" != *"MISSING fixtures: $BONSAI2_MMPROJ_FILE"* \
                            || "$(cat "$vflow_log")" != *"--skip-bonsai2-vision"* ]]; then
                            echo "FAIL: self-test '$label': $vis_calls vision binaries ran; log tail: $(tail -n 12 "$vflow_log")"
                            failures=$((failures + 1))
                            return
                        fi
                        ;;
                    skipped)
                        if [[ "$vis_calls" -ne 0 || "$metal_calls" -ne 0 \
                            || "$req" == *"bonsai2-vision"* || "$req" == *"bonsai2-mmproj"* ]]; then
                            echo "FAIL: self-test '$label': $vis_calls vision binaries ran; required line: $req"
                            failures=$((failures + 1))
                            return
                        fi
                        ;;
                esac
                echo "OK: self-test '$label' (exit $vflow_rc, $vis_calls CPU and $metal_calls Metal vision binaries)"
            }
            printf 'x' >"$vflow_models/$BONSAI2_MMPROJ_FILE"
            run_vflow "stage 1c vision wiring: the projector is present -> the CPU leg's three binaries and the Metal half's four run, and every vision capability is required by name" ok
            run_vflow "stage 1c vision wiring: --skip-bonsai2-metal keeps the CPU vision leg but runs and requires none of its Metal half" no_metal --skip-bonsai2-metal
            run_vflow "stage 1c vision wiring: --skip-bonsai2-vision runs no vision binary and requires no vision capability" skipped --skip-bonsai2-vision
            rm -f "$vflow_models/$BONSAI2_MMPROJ_FILE"
            run_vflow "stage 1c vision wiring: the 27B without its projector fails closed before any vision binary and names --skip-bonsai2-vision" refused
            run_vflow "stage 1c vision wiring: --skip-bonsai2-vision releases without the projector" skipped --skip-bonsai2-vision
        fi
    fi

    # ── --accept-approximate-cuda-syntax: the CUDA-syntax waiver, end to end ──
    # The REAL release-gate.sh, ci.sh, check_cuda.sh and publish.sh run in
    # scratch trees on a PATH built from scratch links alone, so neither a host
    # nvcc nor a host C++ compiler can leak in. Stand-ins: cargo and the cargo-*
    # tools (a `cargo nextest` writes one `metal` record, as a gated test would),
    # nvcc and clang++ (each rejects a source carrying SELFTEST_SYNTAX_ERROR),
    # and a fixture gpu_backend/ with two (or, "broken", three) CUDA_* sources.
    local cu_tree="$work_dir/cuda_tree" cu_pub="$work_dir/cuda_pub_tree" cu_bin="$work_dir/cuda_bin"
    local cu_nvcc="$work_dir/cuda_nvcc" cu_cxx="$work_dir/cuda_cxx" cu_out="$work_dir/cuda_run.log"
    local cu_calls="$work_dir/cuda_calls.log" cu_cli="$work_dir/cuda_bin/oxibonsai_stub" cu_rc=0 cu_tool cu_link
    mkdir -p "$cu_tree/scripts" "$cu_tree/crates/oxibonsai-kernels/src/gpu_backend" \
        "$cu_pub/scripts" "$cu_bin" "$cu_nvcc" "$cu_cxx"
    for cu_tool in env uname date mktemp mv rm mkdir cat grep sed tee dirname basename tail head wc tr cp ls sort python3; do
        cu_link="$(command -v "$cu_tool" 2>/dev/null)" && ln -s "$cu_link" "$cu_bin/$cu_tool"
    done
    ln -s "$BASH" "$cu_bin/bash"
    cp "$SCRIPT_DIR/release-gate.sh" "$SCRIPT_DIR/ci.sh" "$SCRIPT_DIR/check_cuda.sh" "$cu_tree/scripts/"
    cp "$SCRIPT_DIR/publish.sh" "$cu_pub/scripts/"
    printf '#!/usr/bin/env bash\nexit 0\n' >"$cu_tree/scripts/check_pure_rust.sh"
    printf '#!/usr/bin/env bash\necho "gate[$*]" >>"${OXIBONSAI_SELFTEST_CARGO_LOG:?}"\nexit "${OXIBONSAI_SELFTEST_STAND_IN_RC:-0}"\n' \
        >"$cu_pub/scripts/release-gate.sh"
    printf '#!/bin/sh\necho "oxibonsai 0.0.0"\n' >"$cu_cli"
    printf '#!/usr/bin/env bash\necho wasm32-unknown-unknown\n' >"$cu_bin/rustup"
    for cu_tool in cargo-fmt cargo-clippy cargo-nextest cargo-deny cargo-llvm-cov; do
        printf '#!/usr/bin/env bash\nexit 0\n' >"$cu_bin/$cu_tool"
    done
    cat >"$cu_bin/cargo" <<'CU_CARGO_EOF'
#!/usr/bin/env bash
echo "$*" >>"${OXIBONSAI_SELFTEST_CARGO_LOG:?}"
case "${1:-}" in
    nextest) printf '%s\n' '{"capability":"metal","executed":true,"test":"selftest::metal"}' >>"${OXIBONSAI_CAPABILITY_REPORT:?}" ;;
    build)
        if [[ "$*" == *--message-format=json* ]]; then
            printf '{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"oxibonsai"},"executable":"%s"}\n' "${OXIBONSAI_SELFTEST_CLI_PATH:?}"
            printf '%s\n' '{"reason":"build-finished","success":true}'
        fi
        ;;
esac
exit 0
CU_CARGO_EOF
    cat >"$cu_cxx/clang++" <<'CU_CXX_EOF'
#!/usr/bin/env bash
for src; do :; done
if grep -q SELFTEST_SYNTAX_ERROR "$src"; then echo "$src:1:1: error: self-test syntax error" >&2; exit 1; fi
exit 0
CU_CXX_EOF
    cat >"$cu_nvcc/nvcc" <<'CU_NVCC_EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "Cuda compilation tools, release 0.0 (self-test stand-in)"; exit 0; fi
for src in "$@"; do case "$src" in *.cu) break ;; esac; done
if grep -q SELFTEST_SYNTAX_ERROR "$src"; then echo "$src:1:1: error: self-test syntax error" >&2; exit 1; fi
exit 0
CU_NVCC_EOF
    # Only the stand-ins are made executable: the other entries of $cu_bin are links to host tools.
    chmod +x "$cu_bin/cargo" "$cu_bin/rustup" "$cu_bin"/cargo-* "$cu_cli" "$cu_cxx/clang++" "$cu_nvcc/nvcc" \
        "$cu_tree/scripts/check_pure_rust.sh" "$cu_pub/scripts/release-gate.sh"

    # cu_fixture <ok|broken>: the kernel sources check_cuda.sh extracts.
    cu_fixture() {
        local src="$cu_tree/crates/oxibonsai-kernels/src/gpu_backend/selftest_kernels.rs"
        printf 'pub const CUDA_SELFTEST_A: &str = r#"\nextern "C" __global__ void a(float* o) { o[0] = 1.0f; }\n"#;\n' >"$src"
        printf 'pub const CUDA_SELFTEST_B: &str = r#"\nextern "C" __global__ void b(float* o) { o[0] = 2.0f; }\n"#;\n' >>"$src"
        if [[ "$1" == "broken" ]]; then
            printf 'pub const CUDA_SELFTEST_BAD: &str = r#"\nSELFTEST_SYNTAX_ERROR\n"#;\n' >>"$src"
        fi
    }
    # cu_run <none|cxx|nvcc|both: the tools on PATH> <ok|broken: the kernel sources> <script> [args...]:
    # runs a script of the scratch tree; its output goes to $cu_out and its exit status to $cu_rc.
    cu_run() {
        local tools="$1" fixture="$2" script="$3" cu_path="$cu_bin"
        shift 3
        [[ "$tools" == cxx || "$tools" == both ]] && cu_path="$cu_path:$cu_cxx"
        [[ "$tools" == nvcc || "$tools" == both ]] && cu_path="$cu_path:$cu_nvcc"
        cu_fixture "$fixture"
        rm -rf "$cu_tree/target"
        : >"$cu_calls"
        cu_rc=0
        env -u CARGO_TARGET_DIR -u OXIBONSAI_CLI_BIN -u OXIBONSAI_M08_RUN_LONG PATH="$cu_path" \
            OXIBONSAI_SELFTEST_CARGO_LOG="$cu_calls" OXIBONSAI_SELFTEST_CLI_PATH="$cu_cli" \
            "$BASH" "$cu_tree/scripts/$script" "$@" >"$cu_out" 2>&1 || cu_rc=$?
    }
    # cu_pub_run [args...]: publish.sh (a dry run) in its scratch tree, against a stand-in gate that logs its argv.
    cu_pub_run() {
        : >"$cu_calls"
        cu_rc=0
        env PATH="$cu_bin" OXIBONSAI_SELFTEST_CARGO_LOG="$cu_calls" "$BASH" "$cu_pub/scripts/publish.sh" "$@" \
            >"$cu_out" 2>&1 || cu_rc=$?
    }
    # cu_assert <label> <expected exit> <check>...: checks the last run. A check is +text (the output contains it),
    # -text (it does not), ~regex (it matches); prefix m (the capability manifest the run left) or c (the log of
    # cargo / stand-in invocations) to read that file instead; NOCALLS / NOTARGET: nothing was invoked / no target dir.
    cu_assert() {
        local label="$1" expect_rc="$2" want file text bad="" report="$cu_tree/target/capability-report.json"
        shift 2
        total=$((total + 1))
        echo ""
        echo "-- self-test: $label --"
        [[ "$cu_rc" -eq "$expect_rc" ]] || bad="expected exit $expect_rc, got $cu_rc"
        for want in "$@"; do
            [[ -z "$bad" ]] || break
            file="$cu_out"
            case "$want" in
                m[+~-]*) file="$report"; want="${want#m}" ;;
                c[+~-]*) file="$cu_calls"; want="${want#c}" ;;
            esac
            text="${want#?}"
            case "$want" in
                NOCALLS) [[ ! -s "$cu_calls" ]] || bad="something was invoked: $(cat "$cu_calls")" ;;
                NOTARGET) [[ ! -e "$cu_tree/target" ]] || bad="the run created a target dir" ;;
                +*) grep -q -F -- "$text" "$file" 2>/dev/null || bad="lacks: $text" ;;
                -*) ! grep -q -F -- "$text" "$file" 2>/dev/null || bad="must not contain: $text" ;;
                ~*) grep -q -E -- "$text" "$file" 2>/dev/null || bad="no match for: $text" ;;
            esac
        done
        if [[ -n "$bad" ]]; then
            echo "FAIL: self-test '$label': $bad"
            echo "  output tail: $(tail -n 100 "$cu_out")"
            failures=$((failures + 1))
        else
            echo "OK: self-test '$label' (exit $cu_rc, output as expected)"
        fi
    }

    local cu_phrase="parsed as C++ with CUDA builtins stubbed; this is NOT an nvcc validation; the CUDA backend is not certified by this run"
    local cu_verdict="RELEASE GATE PASSED with a waiver: CUDA kernel syntax was checked approximately (no nvcc)"
    local cu_record='"capability":"cuda-syntax","executed":false,"waived":"approximate-accepted"'
    local cu_gate=(release-gate.sh --skip-legacy-models --skip-bonsai2-models)

    # check_cuda.sh by itself: every tier, with and without --release / --accept-approximate.
    cu_run cxx ok check_cuda.sh
    cu_assert "check_cuda dev: a clean approximate run passes (approximate-ok)" 0 "+CUDA_SYNTAX_RESULT=approximate-ok" "-ACCEPTED BY FLAG"
    cu_run none ok check_cuda.sh
    cu_assert "check_cuda dev: no checker is SKIPPED, exit 0" 0 "+CUDA_SYNTAX_RESULT=skipped"
    cu_run cxx ok check_cuda.sh --accept-approximate
    cu_assert "check_cuda dev: --accept-approximate is accepted and changes nothing" 0 "+CUDA_SYNTAX_RESULT=approximate-ok" "-ACCEPTED BY FLAG"
    cu_run cxx ok check_cuda.sh --release
    cu_assert "check_cuda --release, no flag: approximate only is INCOMPLETE, exit 2" 2 \
        "+INCOMPLETE (--release)" "+CUDA_SYNTAX_RESULT=incomplete" "-ACCEPTED BY FLAG"
    cu_run cxx ok check_cuda.sh --release --accept-approximate
    cu_assert "check_cuda --release --accept-approximate: a clean approximate tier is accepted, loudly, exit 0" 0 \
        "+APPROXIMATE, ACCEPTED BY FLAG: 2 kernel sources $cu_phrase" "+NOT CHECKED:" "+CUDA_SYNTAX_RESULT=approximate-accepted"
    cu_run cxx broken check_cuda.sh --release --accept-approximate
    cu_assert "check_cuda --release --accept-approximate: an approximate-tier error is FAILED, exit 1" 1 \
        "+approx-FAIL" "+CUDA_SYNTAX_RESULT=failed" "-approximate-accepted" "-ACCEPTED BY FLAG"
    cu_run none ok check_cuda.sh --release --accept-approximate
    cu_assert "check_cuda --release --accept-approximate: no checker at all stays INCOMPLETE, exit 2" 2 \
        "+nothing ran" "+CUDA_SYNTAX_RESULT=incomplete" "-ACCEPTED BY FLAG"
    cu_run nvcc ok check_cuda.sh --release
    cu_assert "check_cuda --release: a real nvcc pass prints nvcc-ok" 0 "+CUDA_SYNTAX_RESULT=nvcc-ok"
    cu_run both ok check_cuda.sh --release --accept-approximate
    cu_assert "check_cuda --release --accept-approximate with nvcc: the waiver is not used" 0 \
        "+CUDA_SYNTAX_RESULT=nvcc-ok" "+the waiver was" "-ACCEPTED BY FLAG"
    cu_run both broken check_cuda.sh --release --accept-approximate
    cu_assert "check_cuda: a failing nvcc never falls back to the approximate tier, flag or not" 1 \
        "+CUDA_SYNTAX_RESULT=failed" "-tier 2" "-approximate-accepted"

    # ci.sh: the stage label, the waiver line and the single-stage release probe.
    cu_run cxx ok ci.sh --release --only fmt
    cu_assert "ci.sh: --only is still refused with --release for every stage but cuda-syntax" 2 \
        "+--only is incompatible with --release" "-Stage 1/"
    cu_run cxx ok ci.sh --release --only cuda-syntax
    cu_assert "ci.sh --release --only cuda-syntax, no flag: the strict stage is INCOMPLETE; the run is labelled PARTIAL" 2 \
        "+FAILED (incomplete, 2): cuda-syntax" "~cuda-syntax +INCOMPLETE" "+(PARTIAL run, --only)" "-ALL STAGES COMPLETE."
    cu_run cxx ok ci.sh --release --accept-approximate-cuda-syntax --only cuda-syntax
    cu_assert "ci.sh ... --only cuda-syntax with the flag: exit 0, the Summary says approximate, the run ends PARTIAL" 0 \
        "~cuda-syntax +OK \(approximate, accepted by flag\)" "+WAIVER: cuda-syntax was checked APPROXIMATELY" \
        "+PARTIAL RUN COMPLETE" "-ALL STAGES COMPLETE."
    cu_run nvcc ok ci.sh --release --accept-approximate-cuda-syntax --only cuda-syntax
    cu_assert "ci.sh with the flag but a real nvcc: the Summary line is a plain OK, no waiver" 0 \
        "~cuda-syntax +OK$" "-approximate, accepted by flag" "-WAIVER"
    cu_run cxx ok ci.sh --accept-approximate-cuda-syntax --only cuda-syntax
    cu_assert "ci.sh dev mode: the flag is accepted, says it has no effect, and the stage is a plain OK" 0 \
        "+has no effect without --release" "~cuda-syntax +OK$" "+ALL STAGES COMPLETE." "-WAIVER"

    # The usage text of every script that gained the flag.
    cu_run none ok check_cuda.sh --help
    cu_assert "check_cuda --help documents --accept-approximate" 0 "+--accept-approximate"
    cu_run none ok ci.sh --help
    cu_assert "ci.sh --help documents --accept-approximate-cuda-syntax" 0 "+--accept-approximate-cuda-syntax"
    cu_run none ok ci.sh --list
    cu_assert "ci.sh --list still prints bare stage names" 0 "+cuda-syntax" "-accept-approximate"
    cu_run none ok release-gate.sh --help
    cu_assert "release-gate.sh --help documents --accept-approximate-cuda-syntax and its --require-cuda refusal" 0 \
        "+--accept-approximate-cuda-syntax" "+REFUSED"

    # A checker that exits 0 under --release without a CUDA_SYNTAX_RESULT line cannot be labelled.
    printf '#!/usr/bin/env bash\nexit 0\n' >"$cu_tree/scripts/check_cuda.sh"
    cu_run cxx ok ci.sh --release --only cuda-syntax
    cu_assert "ci.sh: a --release pass that names no tier is not recorded as OK, exit 2" 2 \
        "+refusing to record an unlabelled pass" "~cuda-syntax +INCOMPLETE" "-WAIVER"
    cp "$SCRIPT_DIR/check_cuda.sh" "$cu_tree/scripts/check_cuda.sh"

    # The gate, end to end through the real ci.sh and check_cuda.sh.
    cu_run cxx ok "${cu_gate[@]}"
    cu_assert "waiver (a): no flag and no nvcc FAILS at cuda-syntax, exit 2 (the fail-closed default)" 2 \
        "+cuda-syntax: nvcc required" "+INCOMPLETE (--release)" "~cuda-syntax +INCOMPLETE" \
        "+RELEASE GATE FAILED: scripts/ci.sh --release did not pass (exit 2)" "-$cu_verdict" "-WAIVER" "m-cuda-syntax"
    cu_run cxx ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver (b): the flag and a clean approximate tier get past ci.sh and PASS with a waiver that is named and recorded" 0 \
        "+cuda-syntax: approximate accepted by flag" "~cuda-syntax +OK \(approximate, accepted by flag\)" "+WAIVER IN EFFECT" \
        "+$cu_verdict" "-RELEASE GATE FAILED" "m+$cu_record"
    cu_run cxx ok "${cu_gate[@]}" --require-cuda --accept-approximate-cuda-syntax
    cu_assert "waiver (c): the flag with --require-cuda is refused, exit 2, before anything runs" 2 \
        "+a release that requires CUDA evidence must run on a host with the CUDA toolkit" "-OxiBonsai release gate" NOCALLS NOTARGET
    cu_run cxx ok "${cu_gate[@]}" --accept-approximate-cuda-syntax --require-cuda
    cu_assert "waiver (c): the same refusal in the other argument order" 2 \
        "+a release that requires CUDA evidence must run on a host with the CUDA toolkit" "-OxiBonsai release gate" NOCALLS NOTARGET
    cu_run cxx broken "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver (d): the flag never waives an approximate-tier error: FAILS at cuda-syntax, exit 1" 1 \
        "+approx-FAIL" "~cuda-syntax +FAILED" "+RELEASE GATE FAILED: scripts/ci.sh --release did not pass (exit 1)" \
        "-$cu_verdict" "-WAIVER IN EFFECT" "m-cuda-syntax"
    cu_run both ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver (e): the flag with a real nvcc waives nothing: a plain PASSED, no waiver text, no manifest record" 0 \
        "+a real nvcc check passed" "+RELEASE GATE PASSED." "-$cu_verdict" "-WAIVER IN EFFECT" "m-cuda-syntax"
    cu_run both ok "${cu_gate[@]}"
    cu_assert "waiver: without the flag a real nvcc pass is a plain PASSED (the default path is unchanged)" 0 \
        "+cuda-syntax: nvcc required" "+RELEASE GATE PASSED." "-$cu_verdict" "-WAIVER" "m-cuda-syntax"
    cu_run none ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver: the flag cannot waive a host where no checker ran: exit 2" 2 \
        "+nothing ran" "-$cu_verdict" "m-cuda-syntax"
    # A host that requires no capability (not Darwin) takes the gate's other PASS exit: it names the waiver too.
    printf '#!/usr/bin/env bash\necho Linux\n' >"$cu_bin/uname.stand_in"
    chmod +x "$cu_bin/uname.stand_in"
    mv -f "$cu_bin/uname.stand_in" "$cu_bin/uname"   # replaces the link itself, never writes through it
    cu_run cxx ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver: a host with no required capability also names the waiver in its verdict and records it" 0 \
        "+No hardware capability is required" "+$cu_verdict" "m+$cu_record"
    cu_run both ok "${cu_gate[@]}"
    cu_assert "waiver: that same PASS exit without a waiver is a plain PASSED" 0 \
        "+No hardware capability is required" "+RELEASE GATE PASSED." "-$cu_verdict" "m-cuda-syntax"
    rm -f "$cu_bin/uname"
    ln -s "$(command -v uname)" "$cu_bin/uname"
    # The waiver record is ignored by the capability check and is no CUDA evidence.
    printf '%s\n%s\n' '{"capability":"metal","executed":true,"test":"t::ran"}' \
        "{$cu_record,\"test\":\"scripts/check_cuda.sh\"}" >"$work_dir/waiver_record.jsonl"
    check "waiver record: the capability check ignores it (metal is still satisfied)" 0 "$work_dir/waiver_record.jsonl" "$now" metal
    check "waiver record: it is no cuda evidence" 1 "$work_dir/waiver_record.jsonl" "$now" cuda

    # The gate against a stand-in ci.sh that logs its argv (the real one is set aside for good).
    printf '#!/usr/bin/env bash\necho "ci[$*]" >>"${OXIBONSAI_SELFTEST_CARGO_LOG:?}"\nexit "${OXIBONSAI_SELFTEST_STAND_IN_RC:-0}"\n' \
        >"$cu_tree/scripts/ci.sh"
    export OXIBONSAI_SELFTEST_STAND_IN_RC=1
    cu_run cxx ok "${cu_gate[@]}"
    cu_assert "waiver: without the flag ci.sh is invoked with --release alone" 1 "c+ci[--release]" "c-accept-approximate"
    cu_run cxx ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver: with the flag ci.sh gets --accept-approximate-cuda-syntax too" 1 "c+ci[--release --accept-approximate-cuda-syntax]"
    export OXIBONSAI_SELFTEST_STAND_IN_RC=0
    cu_run cxx ok "${cu_gate[@]}" --accept-approximate-cuda-syntax
    cu_assert "waiver: a passing ci.sh that names no CUDA_SYNTAX_RESULT fails the gate closed, exit 2" 2 \
        "+recognised CUDA_SYNTAX_RESULT line" "-$cu_verdict" "m-cuda-syntax"
    unset OXIBONSAI_SELFTEST_STAND_IN_RC

    # publish.sh forwards the flag to the gate only when it was passed.
    cu_pub_run
    cu_assert "publish.sh: no flag, the gate is invoked with no arguments" 0 "c+gate[]" "+Dry-run complete"
    cu_pub_run --accept-approximate-cuda-syntax
    cu_assert "publish.sh: --accept-approximate-cuda-syntax is forwarded to the gate" 0 \
        "c+gate[--accept-approximate-cuda-syntax]" "+Dry-run complete"
    cu_pub_run --require-cuda --accept-approximate-cuda-syntax
    cu_assert "publish.sh: both flags are forwarded as passed (the gate refuses the pair)" 0 \
        "c+gate[--require-cuda --accept-approximate-cuda-syntax]"
    export OXIBONSAI_SELFTEST_STAND_IN_RC=2
    cu_pub_run --accept-approximate-cuda-syntax
    cu_assert "publish.sh: a refusing gate aborts the publish" 2 "+ABORTING: scripts/release-gate.sh failed (exit 2)" "c-publish"
    unset OXIBONSAI_SELFTEST_STAND_IN_RC
    cu_pub_run --help
    cu_assert "publish.sh --help documents the flag and that it is never forwarded by default" 0 \
        "+--accept-approximate-cuda-syntax" "+ONLY when you pass it here"

    rm -rf "$work_dir" "$fake_bin" "${fake_mh_bin:-}" "${fake_leg_bin:-}" "${fake_cli_bin:-}"
    if [[ "$failures" -eq 0 ]]; then
        echo ""
        echo "OK: release-gate.sh self-test — all $total scenarios matched their expected verdict."
        return 0
    fi
    echo ""
    echo "FAILED: release-gate.sh self-test — $failures/$total scenario(s) did not match."
    return 1
}
