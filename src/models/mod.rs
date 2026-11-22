pub mod adapter;
pub mod policy;
pub mod qwen38_27b;
pub mod registry;

pub use adapter::{
    EngineLaunchSpec, InferenceEngineAdapter, LlamaCppEngineAdapter, ModelAdapter,
    PolicyFileAdapter,
};
pub use policy::{
    EnginePolicy, LlamaServerLaunchParams, ModelPolicy, SamplingPolicy, ShardingPolicy,
    SpeculativePolicy,
};
pub use qwen38_27b::Qwen38_27BAdapter;
pub use registry::ModelAdapterRegistry;

use std::collections::BTreeSet;

impl ModelPolicy {
    pub fn load_catalog_policies() -> Vec<Self> {
        ModelAdapterRegistry::load_catalog_policies()
    }

    pub fn qwen38_27b_policy() -> Self {
        Qwen38_27BAdapter::embedded_default().policy().clone()
    }

    /// Resolves the active [`ModelPolicy`] through [`ModelAdapterRegistry`].
    pub fn resolve(model_source: &str) -> Self {
        ModelAdapterRegistry::resolve(model_source).policy().clone()
    }

    pub fn pipeline_layers(&self, header_total_layers: u16) -> u16 {
        ModelAdapterRegistry::wrap_policy(self.clone()).pipeline_layers(header_total_layers)
    }

    pub fn layer_group_alignment(&self) -> u16 {
        ModelAdapterRegistry::wrap_policy(self.clone()).layer_group_alignment()
    }

    pub fn max_nodes_for_rtt(&self, max_rtt_ms: f64, min_nodes: usize) -> usize {
        ModelAdapterRegistry::wrap_policy(self.clone()).max_nodes_for_rtt(max_rtt_ms, min_nodes)
    }

    pub fn stage0_total_reserve_mb(&self, header_non_block_mb: f64) -> f64 {
        ModelAdapterRegistry::wrap_policy(self.clone())
            .stage0_total_reserve_mb(header_non_block_mb)
    }

    pub fn snap_cut_layer(&self, raw_end_excl: u16, min_end_excl: u16, max_end_excl: u16) -> u16 {
        ModelAdapterRegistry::wrap_policy(self.clone()).snap_cut_layer(
            raw_end_excl,
            min_end_excl,
            max_end_excl,
        )
    }

    pub fn stage0_pinned_layers(&self) -> BTreeSet<u16> {
        ModelAdapterRegistry::wrap_policy(self.clone()).stage0_pinned_layers()
    }

    pub fn build_rpc0_tensor_override(&self, primary_buft: &str) -> String {
        ModelAdapterRegistry::wrap_policy(self.clone()).build_stage0_tensor_overrides(primary_buft)
    }

    pub fn build_llama_server_args(&self, params: &LlamaServerLaunchParams<'_>) -> Vec<String> {
        let adapter = ModelAdapterRegistry::wrap_policy(self.clone());
        adapter.build_engine_args(&LlamaCppEngineAdapter, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_adapter_emits_distributed_rpc_args_without_wan_checkpoints() {
        let adapter = ModelAdapterRegistry::resolve("models/Qwen3.8-27B-UD-Q4_K_M.gguf");
        assert_eq!(adapter.model_id(), Qwen38_27BAdapter::MODEL_ID);

        let tmp = std::env::temp_dir().join("cutmesh_qwen_test_model.gguf");
        let args = adapter.build_engine_args(
            &LlamaCppEngineAdapter,
            &LlamaServerLaunchParams {
                model_path: &tmp,
                ordered_bridges: &["127.0.0.1:41000".to_string()],
                devices_vec: &["CUDA0".to_string(), "RPC0".to_string()],
                splits_vec: &["48".to_string(), "52".to_string()],
                primary_buft: "CUDA0",
                requested_ctx_size: 262_144,
                requested_ubatch_size: 512,
                port: 8089,
            },
        );

        let has_pair = |k: &str, v: &str| args.windows(2).any(|w| w[0] == k && w[1] == v);
        assert!(has_pair("--spec-type", "ngram-mod,draft-mtp"));
        assert!(has_pair("--spec-draft-n-max", "5"));
        assert!(has_pair("--spec-draft-p-min", "0.50"));
        assert!(has_pair("-ctxcp", "0"));
        assert!(has_pair("-cram", "0"));
        assert!(has_pair("-fa", "on"));
        assert!(has_pair("-ctk", "q8_0"));
        assert!(has_pair("-ctv", "q8_0"));
        assert!(has_pair("-ctkd", "q8_0"));
        assert!(has_pair("-ctvd", "q8_0"));
        assert!(!args.iter().any(|a| a == "--no-cache-prompt"));
    }

    #[test]
    fn registry_falls_back_to_generic_adapter_for_unknown_models() {
        let adapter = ModelAdapterRegistry::resolve("models/unknown-custom-7b-q4.gguf");
        assert_ne!(adapter.model_id(), Qwen38_27BAdapter::MODEL_ID);
        assert_eq!(adapter.policy().architecture, "generic");
        assert_eq!(adapter.layer_group_alignment(), 1);
    }

    #[test]
    fn policy_file_adapter_emits_draft_filename_and_spec_type_when_file_exists() {
        let mut policy = ModelPolicy::generic_default("custom-model.gguf");
        let dir = std::env::temp_dir().join(format!("cutmesh_draft_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let model_path = dir.join("model.gguf");
        let draft_path = dir.join("draft.gguf");
        std::fs::write(&draft_path, b"GGUF").unwrap();
        policy.speculative.enabled = true;
        policy.speculative.draft_model_filename = "draft.gguf".to_string();
        policy.speculative.spec_type = "draft-dflash".to_string();
        policy.engine.ctx_checkpoints = 8;
        policy.engine.cache_ram_mib = 1024;

        let adapter = PolicyFileAdapter::new(policy);
        // Single-GPU local run preserves local RAM checkpoints
        let local_args = adapter.build_engine_args(
            &LlamaCppEngineAdapter,
            &LlamaServerLaunchParams {
                model_path: &model_path,
                ordered_bridges: &[],
                devices_vec: &["CUDA0".to_string()],
                splits_vec: &["100".to_string()],
                primary_buft: "CUDA0",
                requested_ctx_size: 8192,
                requested_ubatch_size: 512,
                port: 8089,
            },
        );
        // Distributed RPC run disables RAM checkpoints automatically
        let rpc_args = adapter.build_engine_args(
            &LlamaCppEngineAdapter,
            &LlamaServerLaunchParams {
                model_path: &model_path,
                ordered_bridges: &["127.0.0.1:41000".to_string()],
                devices_vec: &["CUDA0".to_string(), "RPC0".to_string()],
                splits_vec: &["50".to_string(), "50".to_string()],
                primary_buft: "CUDA0",
                requested_ctx_size: 8192,
                requested_ubatch_size: 512,
                port: 8089,
            },
        );

        let _ = std::fs::remove_dir_all(&dir);
        assert!(local_args
            .windows(2)
            .any(|w| w[0] == "-md" && w[1].ends_with("draft.gguf")));
        assert!(local_args
            .windows(2)
            .any(|w| w[0] == "--spec-type" && w[1] == "draft-dflash"));
        assert!(local_args.windows(2).any(|w| w[0] == "-ctxcp" && w[1] == "8"));
        assert!(local_args.windows(2).any(|w| w[0] == "-cram" && w[1] == "1024"));
        assert!(rpc_args.windows(2).any(|w| w[0] == "-ctxcp" && w[1] == "0"));
        assert!(rpc_args.windows(2).any(|w| w[0] == "-cram" && w[1] == "0"));
    }
}
