//! Tests of the `qwen35` hybrid report `info` and `validate` print: which
//! executor `--backend auto` resolves to, the Metal runner's KV window with
//! every limit, its residents and its up-front device allocation — or the
//! reason no runner serves the model — in both the text and `--json` forms.

use super::*;

use oxibonsai_runtime::engine_hybrid_gpu::{
    plan_hybrid_metal_window, HybridBackendPlan, HybridResidents, HybridWindowInputs,
};

/// Whether this build and host have a Metal device the runner can use.
fn metal_device_present() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

fn fixture_report() -> HybridReport {
    let bytes = oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf();
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("fixture parses");
    hybrid_report(&gguf).expect("hybrid report")
}

/// A Metal plan for the 27B on the documented 24 GiB M3 at `requested`.
fn metal_plan_27b(requested: usize) -> HybridBackendPlan {
    let recurrent = 156_893_184u64;
    let window = plan_hybrid_metal_window(&HybridWindowInputs {
        requested,
        declared: 262_144,
        total_ram_bytes: Some(25_769_803_776),
        file_bytes: 7_206_168_928,
        cpu_recurrent_bytes: recurrent,
        cpu_kv_bytes_per_position: 65_536,
        cpu_other_bytes_per_position: 256,
        runner_fixed_bytes: 48 * 2 * 48 * 5120 * 4
            + recurrent
            + 248_320 * 4
            + 140_384 * 4 * 512
            + 512 * 5120 * 4,
        runner_bytes_per_position: 65_888,
        runner_kv_bytes_per_position: 65_536,
        device_ceiling: 178_034,
        residents: HybridResidents::CpuModelAndRunner,
    });
    HybridBackendPlan::Metal {
        window,
        mapped: true,
    }
}

fn kernel_tier_lines(lines: &[String]) -> Vec<&String> {
    lines
        .iter()
        .filter(|l| l.contains("Kernel tier:"))
        .collect()
}

/// On a Metal host the synthetic hybrid resolves to the Metal runner, and
/// the report says so — one `Kernel tier:` line (still naming the model
/// kind), the window with its limits, the residents and the allocation;
/// without a device the one line names why the CPU tier runs it.
#[test]
fn the_hybrid_report_names_the_backend_auto_resolves_to() {
    let report = fixture_report();
    let plan = report.backend_plan.clone().expect("the fixture binds");
    let lines = report.lines(1 << 20);
    let tiers = kernel_tier_lines(&lines);
    assert_eq!(tiers.len(), 1, "{lines:#?}");
    assert!(tiers[0].contains("hybrid qwen35 model"), "{}", tiers[0]);
    let json = report.to_json(1 << 20);
    if metal_device_present() {
        let HybridBackendPlan::Metal { window, .. } = &plan else {
            panic!("a Metal host must plan the Metal runner: {plan:?}");
        };
        // Default --ctx 8192 against the fixture's declared 4096.
        assert_eq!(window.requested, 8192);
        assert_eq!(window.window, 4096);
        assert!(tiers[0].starts_with("Kernel tier: gpu ("), "{}", tiers[0]);
        assert!(tiers[0].contains("Metal (hybrid runner)"), "{}", tiers[0]);
        assert!(tiers[0].contains("device ceiling"), "{}", tiers[0]);
        let text = lines.join("\n");
        assert!(text.contains("Backend: Metal (hybrid runner)"), "{text}");
        assert!(
            text.contains("the CPU model's KV cache and recurrent state beside"),
            "{text}"
        );
        assert!(text.contains("allocates its f16 KV"), "{text}");
        assert_eq!(json["kernel_tier"], "gpu");
        assert_eq!(json["backend"]["executor"], "metal");
        assert_eq!(json["backend"]["window"], 4096);
        assert_eq!(json["backend"]["limits_applied"][0], "declared context");
    } else {
        let HybridBackendPlan::Cpu { reason } = &plan else {
            panic!("a host without Metal must plan the CPU: {plan:?}");
        };
        assert!(tiers[0].contains(reason.as_str()), "{}", tiers[0]);
        assert_eq!(json["backend"]["executor"], "cpu");
        assert_eq!(json["backend"]["reason"], reason.as_str());
    }
    assert_eq!(json["cpu_kernel_tier"], report.kernel_tier.to_string());
}

/// The 27B at the default `--ctx`: the Metal lines print the wired window,
/// every limit (the resident budget with both residents, the runner-alone
/// figure and the device ceiling), and the up-front 512 MiB of f16 KV.
#[test]
fn the_27b_metal_report_prints_every_limit_and_the_up_front_bytes() {
    let mut report = fixture_report();
    report.backend_plan = Some(metal_plan_27b(8192));
    let (tier, reason) = hybrid_auto_tier(&report);
    assert_eq!(tier, "gpu");
    assert!(reason.contains("KV window 8192 positions"), "{reason}");
    assert!(reason.contains("device ceiling 178034"), "{reason}");
    let text = report.lines(7_206_168_928).join("\n");
    for needle in [
        "KV window 8192 positions",
        "RAM budget for the residents kept 83968",
        "a runner alone would fit 171008",
        "Metal device ceiling 178034",
        "RAM guard (CPU model alone) 178176",
        "65536 bytes/position: 0.50 GiB for the 8192-position window",
    ] {
        assert!(text.contains(needle), "{needle} missing:\n{text}");
    }
    let json = report.to_json(7_206_168_928);
    assert_eq!(json["backend"]["resident_budget"], 83_968);
    assert_eq!(json["backend"]["runner_kv_bytes"], 8192u64 * 65_536);

    // A request past the resident budget is clamped to it, and says so.
    report.backend_plan = Some(metal_plan_27b(262_144));
    let json = report.to_json(7_206_168_928);
    assert_eq!(json["backend"]["window"], 83_968);
    let limits = json["backend"]["limits_applied"]
        .as_array()
        .expect("limits array");
    assert!(limits
        .iter()
        .any(|l| l == "RAM budget for the residents kept"));
}

/// A CPU plan's one tier line carries the reason verbatim, and the report
/// prints no Metal lines.
#[test]
fn a_cpu_plan_names_its_reason_and_prints_no_metal_lines() {
    let mut report = fixture_report();
    report.backend_plan = Some(HybridBackendPlan::Cpu {
        reason: "no Metal device was found on this host".to_string(),
    });
    let lines = report.lines(1 << 20);
    let tiers = kernel_tier_lines(&lines);
    assert_eq!(tiers.len(), 1);
    assert!(
        tiers[0].contains("no Metal device was found on this host"),
        "{}",
        tiers[0]
    );
    assert!(!lines.iter().any(|l| l.starts_with("Backend:")));
    assert_eq!(hybrid_auto_tier(&report).0, report.kernel_tier.to_string());
}
