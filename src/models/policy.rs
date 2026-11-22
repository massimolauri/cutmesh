use serde::{Deserialize, Serialize};
use std::path::Path;

fn default_max_wan_nodes() -> usize {
    3
}

fn default_max_lan_nodes() -> usize {
    4
}

fn default_max_context_length() -> u32 {
    262_144
}

fn default_kv_vram_per_1k_mb() -> u64 {
    64
}

/// Declarative per-model configuration loaded from `models/<model_id>/policy.json`
/// or pushed at runtime by the Control Plane (`CUTMESH_MODEL_POLICY_JSON`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPolicy {
    pub model_id: String,
    pub display_name: String,
    pub architecture: String,
    pub gguf_filename: String,
    #[serde(default)]
    pub match_patterns: Vec<String>,
    #[serde(default)]
    pub canonical_hf_url: String,
    pub sharding: ShardingPolicy,
    pub speculative: SpeculativePolicy,
    pub engine: EnginePolicy,
    pub sampling: SamplingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingPolicy {
    pub trunk_layers: u16,
    #[serde(default)]
    pub mtp_layers: Vec<u16>,
    pub layer_group_alignment: u16,
    #[serde(default)]
    pub preferred_cut_layers: Vec<u16>,
    #[serde(default)]
    pub allowed_cut_layers: Vec<u16>,
    #[serde(default)]
    pub dflash_tap_layers: Vec<u16>,
    #[serde(default)]
    pub layer_weights_mib: Vec<f64>,
    #[serde(default)]
    pub non_block_master_weights_mib: f64,
    #[serde(default = "default_kv_vram_per_1k_mb")]
    pub kv_vram_per_1k_tokens_mb: u64,
    pub stage0_pin_output_head: bool,
    pub stage0_pin_mtp_layers: bool,
    pub stage0_extra_vram_reserve_mb: u64,
    pub worker_vram_reserve_mb: u64,
    #[serde(default = "default_max_wan_nodes")]
    pub max_wan_nodes: usize,
    #[serde(default = "default_max_lan_nodes")]
    pub max_lan_nodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativePolicy {
    pub enabled: bool,
    pub spec_type: String,
    #[serde(default)]
    pub draft_model_filename: String,
    #[serde(default)]
    pub draft_model_hf_url: String,
    #[serde(default)]
    pub draft_weights_mib: f64,
    pub draft_max: u32,
    pub draft_min: u32,
    pub draft_p_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePolicy {
    #[serde(default = "default_max_context_length")]
    pub max_context_length: u32,
    #[serde(default = "default_max_context_length")]
    pub default_ctx_size: u32,
    pub default_ubatch_size: u32,
    pub parallel_slots: u32,
    pub continuous_batching: bool,
    pub ctx_checkpoints: u32,
    pub cache_ram_mib: i32,
    pub cache_prompt: bool,
    pub cache_idle_slots: bool,
    pub reasoning_format: String,
    pub no_warmup: bool,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPolicy {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub thinking_default: String,
}

/// Runtime launch parameters supplied by the orchestrator/worker when starting an inference engine.
#[derive(Debug, Clone)]
pub struct LlamaServerLaunchParams<'a> {
    pub model_path: &'a Path,
    pub ordered_bridges: &'a [String],
    pub devices_vec: &'a [String],
    pub splits_vec: &'a [String],
    pub primary_buft: &'a str,
    pub requested_ctx_size: u32,
    pub requested_ubatch_size: u32,
    pub port: u16,
}

impl ModelPolicy {
    /// Safe model-agnostic fallback policy when a GGUF does not match any registered adapter or
    /// `models/*/policy.json` file (avoids silently applying another model's layer/KV geometry).
    pub fn generic_default(model_source: &str) -> Self {
        let file_name = Path::new(model_source)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("generic-model.gguf")
            .to_string();
        let model_id = file_name
            .trim_end_matches(".gguf")
            .to_ascii_lowercase();

        Self {
            model_id,
            display_name: file_name.clone(),
            architecture: "generic".to_string(),
            gguf_filename: file_name,
            match_patterns: Vec::new(),
            canonical_hf_url: String::new(),
            sharding: ShardingPolicy {
                trunk_layers: 0,
                mtp_layers: Vec::new(),
                layer_group_alignment: 1,
                preferred_cut_layers: Vec::new(),
                allowed_cut_layers: Vec::new(),
                dflash_tap_layers: Vec::new(),
                layer_weights_mib: Vec::new(),
                non_block_master_weights_mib: 1024.0,
                kv_vram_per_1k_tokens_mb: 64,
                stage0_pin_output_head: true,
                stage0_pin_mtp_layers: false,
                stage0_extra_vram_reserve_mb: 1024,
                worker_vram_reserve_mb: 1024,
                max_wan_nodes: 3,
                max_lan_nodes: 4,
            },
            speculative: SpeculativePolicy {
                enabled: false,
                spec_type: "none".to_string(),
                draft_model_filename: String::new(),
                draft_model_hf_url: String::new(),
                draft_weights_mib: 0.0,
                draft_max: 0,
                draft_min: 0,
                draft_p_min: 0.5,
            },
            engine: EnginePolicy {
                max_context_length: 32_768,
                default_ctx_size: 32_768,
                default_ubatch_size: 512,
                parallel_slots: 1,
                continuous_batching: true,
                ctx_checkpoints: 0,
                cache_ram_mib: 0,
                cache_prompt: true,
                cache_idle_slots: true,
                reasoning_format: "none".to_string(),
                no_warmup: true,
                extra_args: vec!["-fa".to_string(), "on".to_string()],
            },
            sampling: SamplingPolicy {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 40,
                min_p: 0.0,
                presence_penalty: 0.0,
                thinking_default: "disabled".to_string(),
            },
        }
    }
}
