use std::path::Path;
use cutmesh::catalog::{
    build_model_catalog_entry, detect_local_gpu_capabilities_with_budget, schedule_nodes_for_model,
};
use cutmesh::governance::{
    evaluate_external_cluster_governance, ExternalClusterStateInput, GovernancePolicyConfig,
};
use cutmesh::shield::{execute_laptop_micro_shield_with_input, ExternalShieldInput};

#[test]
fn test_local_gpu_registers_and_power_budget_scaling() {
    let full_watt = detect_local_gpu_capabilities_with_budget("test-node-100", "ep-100", 100);
    let half_watt = detect_local_gpu_capabilities_with_budget("test-node-50", "ep-50", 50);

    assert_eq!(full_watt.user_power_budget_pct, 100);
    assert_eq!(half_watt.user_power_budget_pct, 50);
    assert!(
        (half_watt.allocated_power_budget_watts - (full_watt.allocated_power_budget_watts * 0.5))
            .abs()
            <= 0.5
    );
}

#[test]
fn test_external_json_shield_and_catalog_agnostic() {
    let shield_input_path = Path::new("config/shield_input.json");
    let shield_input = ExternalShieldInput::from_json_file(shield_input_path)
        .expect("Failed loading config/shield_input.json");

    for model_file in [
        "models/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf",
        "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ] {
        let model_path = Path::new(model_file);
        if model_path.exists() {
            let catalog = build_model_catalog_entry(model_path).expect("Catalog build failed");
            assert!(catalog.total_layers > 0);
            assert!(catalog.vocab_size > 0);

            let audit = execute_laptop_micro_shield_with_input(model_path, &shield_input)
                .expect("Micro-Shield execution failed");
            assert!(audit.whitebox_inversion_attack_test.mathematically_non_reversible);
            assert_eq!(
                audit
                    .whitebox_inversion_attack_test
                    .micro_shield_tokens_recovered_by_attacker_pct,
                0.0
            );
            assert!(audit.output_clustered_lm_head_cpu.exact_top1_token_match);
        }
    }
}

#[test]
fn test_external_json_governance_and_water_filling_vram() {
    let policy_path = Path::new("config/governance_policy.json");
    let cluster_path = Path::new("config/cluster_state_example.json");
    let model_path = Path::new("models/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf");

    if !model_path.exists() {
        return;
    }

    let policy = GovernancePolicyConfig::from_json_file(policy_path)
        .expect("Failed loading config/governance_policy.json");
    let cluster_input = ExternalClusterStateInput::from_json_file(cluster_path)
        .expect("Failed loading config/cluster_state_example.json");
    let catalog = build_model_catalog_entry(model_path).expect("Failed building catalog");

    let report = evaluate_external_cluster_governance(
        model_path,
        &catalog,
        &cluster_input,
        &policy,
    )
    .expect("Governance evaluation failed");

    // Verify Byzantine node was quarantined
    assert_eq!(report.quarantined_byzantine_nodes.len(), 1);
    assert!(report.active_parallel_pods_formed >= 2);

    // Verify Water-Filling VRAM cap on Node 1 (1536 MB free - 384 MB reserve = max 1152 MB)
    let raw_caps: Vec<_> = cluster_input
        .candidate_nodes
        .iter()
        .take(2)
        .map(|n| n.capabilities.clone())
        .collect();
    let plan = schedule_nodes_for_model(&catalog, &raw_caps).expect("Schedule failed");
    for w in &plan.selected_workers {
        let usable = w.vram_free_mb.saturating_sub(policy.cuda_vram_reserve_mb) as f64;
        assert!(
            w.assigned_weights_mb <= usable + 64.0,
            "Worker {} assigned {:.2} MB exceeding usable {:.2} MB",
            w.node_name,
            w.assigned_weights_mb,
            usable
        );
    }

    // Verify Canary Trap catches the norm-preserving cheater
    assert!(report.canary_trap_anti_cheat_audit.honest_worker_check.canary_trap_passed);
    assert!(
        !report
            .canary_trap_anti_cheat_audit
            .norm_preserving_cheater_check
            .canary_trap_passed
    );

    // Verify all compute receipts have valid signatures
    assert!(!report.signed_compute_receipts_ledger.is_empty());
    for rcpt in &report.signed_compute_receipts_ledger {
        assert!(rcpt.signature_verified);
    }
}

