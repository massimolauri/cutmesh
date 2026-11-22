use crate::models::policy::{LlamaServerLaunchParams, ModelPolicy};
use std::collections::BTreeSet;
use std::path::Path;

/// Engine-neutral launch specification produced by a [`ModelAdapter`] and translated into
/// backend CLI arguments by an [`InferenceEngineAdapter`] (Bridge Pattern).
#[derive(Debug, Clone)]
pub struct EngineLaunchSpec {
    pub model_path: String,
    pub rpc_bridges: Vec<String>,
    pub devices: Vec<String>,
    pub tensor_splits: Vec<String>,
    pub tensor_overrides: String,
    pub parallel_slots: u32,
    pub continuous_batching: bool,
    pub ctx_size: u32,
    pub ubatch_size: u32,
    pub ctx_checkpoints: u32,
    pub cache_ram_mib: i32,
    pub cache_prompt: bool,
    pub cache_idle_slots: bool,
    pub reasoning_format: String,
    pub speculative_enabled: bool,
    pub draft_model_path: Option<String>,
    pub spec_type: Option<String>,
    pub draft_max: u32,
    pub draft_min: u32,
    pub draft_p_min: f32,
    pub port: u16,
    pub no_warmup: bool,
    pub extra_args: Vec<String>,
}

/// Bridge trait decoupling per-model specifications from a concrete inference server binary
/// (`llama-server` today; extensible to other engines without changing model adapters).
pub trait InferenceEngineAdapter: Send + Sync {
    fn build_args(&self, spec: &EngineLaunchSpec) -> Vec<String>;
}

/// Concrete [`InferenceEngineAdapter`] for `llama.cpp`'s `llama-server`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlamaCppEngineAdapter;

impl InferenceEngineAdapter for LlamaCppEngineAdapter {
    fn build_args(&self, spec: &EngineLaunchSpec) -> Vec<String> {
        let mut args = vec!["-m".to_string(), spec.model_path.clone()];

        if !spec.rpc_bridges.is_empty() {
            args.push("--rpc".to_string());
            args.push(spec.rpc_bridges.join(","));
        }

        args.extend([
            "--device".to_string(),
            spec.devices.join(","),
            "-ngl".to_string(),
            "99".to_string(),
            "--tensor-split".to_string(),
            spec.tensor_splits.join(","),
            "-fit".to_string(),
            "off".to_string(),
        ]);

        if !spec.tensor_overrides.is_empty() {
            args.push("-ot".to_string());
            args.push(spec.tensor_overrides.clone());
        }

        args.push("-np".to_string());
        args.push(spec.parallel_slots.max(1).to_string());

        if spec.continuous_batching {
            args.push("-cb".to_string());
        }

        args.push("-c".to_string());
        args.push(spec.ctx_size.to_string());
        args.push("-ub".to_string());
        args.push(spec.ubatch_size.to_string());
        args.push("-ctxcp".to_string());
        args.push(spec.ctx_checkpoints.to_string());
        args.push("-cram".to_string());
        args.push(spec.cache_ram_mib.to_string());

        if !spec.cache_prompt {
            args.push("--no-cache-prompt".to_string());
        }
        if !spec.cache_idle_slots {
            args.push("--no-cache-idle-slots".to_string());
        }
        if !spec.reasoning_format.is_empty() && spec.reasoning_format != "none" {
            args.push("--reasoning-format".to_string());
            args.push(spec.reasoning_format.clone());
        }

        if spec.speculative_enabled {
            if let Some(ref draft_path) = spec.draft_model_path {
                args.push("-md".to_string());
                args.push(draft_path.clone());
            }
            if let Some(ref spec_type) = spec.spec_type {
                args.push("--spec-type".to_string());
                args.push(spec_type.clone());
            }
            if spec.draft_max > 0 {
                args.push("--spec-draft-n-max".to_string());
                args.push(spec.draft_max.to_string());
            }
            if spec.draft_min > 0 {
                args.push("--spec-draft-n-min".to_string());
                args.push(spec.draft_min.to_string());
            }
            if spec.draft_p_min > 0.0 {
                args.push("--spec-draft-p-min".to_string());
                args.push(format!("{:.2}", spec.draft_p_min));
            }
        }

        args.extend([
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            spec.port.to_string(),
        ]);

        if spec.no_warmup {
            args.push("--no-warmup".to_string());
        }

        for extra in &spec.extra_args {
            args.push(extra.clone());
        }

        args
    }
}

/// Core per-model Adapter trait.
///
/// Each supported model implements [`ModelAdapter`] (either via a dedicated per-model struct
/// such as [`crate::models::Qwen38_27BAdapter`] or via [`PolicyFileAdapter`] for declarative
/// `models/<id>/policy.json` definitions). The orchestrator, rig planner, and worker runtime
/// interact exclusively with this abstraction rather than branching on model names.
pub trait ModelAdapter: Send + Sync {
    fn policy(&self) -> &ModelPolicy;

    fn model_id(&self) -> &str {
        &self.policy().model_id
    }

    fn matches(&self, model_source: &str) -> bool {
        let p = self.policy();
        let lower = model_source.to_ascii_lowercase();
        let file_name = Path::new(model_source)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model_source)
            .to_ascii_lowercase();

        (!p.gguf_filename.is_empty() && p.gguf_filename.eq_ignore_ascii_case(&file_name))
            || p.model_id.eq_ignore_ascii_case(&file_name)
            || p.match_patterns
                .iter()
                .any(|pat| !pat.is_empty() && lower.contains(&pat.to_ascii_lowercase()))
    }

    fn pipeline_layers(&self, header_total_layers: u16) -> u16 {
        let trunk = self.policy().sharding.trunk_layers;
        if trunk > 0 && trunk <= header_total_layers {
            trunk
        } else {
            header_total_layers.max(1)
        }
    }

    fn layer_group_alignment(&self) -> u16 {
        self.policy().sharding.layer_group_alignment.max(1)
    }

    fn max_nodes_for_rtt(&self, max_rtt_ms: f64, min_nodes: usize) -> usize {
        let sh = &self.policy().sharding;
        let base_max = if max_rtt_ms <= 2.5 {
            sh.max_lan_nodes.max(1)
        } else {
            sh.max_wan_nodes.max(1)
        };
        base_max.max(min_nodes)
    }

    fn stage0_total_reserve_mb(&self, header_non_block_mb: f64) -> f64 {
        let p = self.policy();
        let base_non_block = if header_non_block_mb > 0.0 {
            header_non_block_mb
        } else {
            p.sharding.non_block_master_weights_mib
        };
        let draft_mb = if p.speculative.enabled {
            p.speculative.draft_weights_mib
        } else {
            0.0
        };
        base_non_block + (p.sharding.stage0_extra_vram_reserve_mb as f64) + draft_mb
    }

    fn snap_cut_layer(&self, raw_end_excl: u16, min_end_excl: u16, max_end_excl: u16) -> u16 {
        let sh = &self.policy().sharding;
        let lo = min_end_excl.min(max_end_excl);
        let hi = max_end_excl.max(lo);

        let valid_preferred: Vec<u16> = sh
            .preferred_cut_layers
            .iter()
            .copied()
            .filter(|&c| c >= lo && c <= hi)
            .collect();
        if let Some(&best_pref) = valid_preferred
            .iter()
            .min_by_key(|&&c| (c as i32 - raw_end_excl as i32).abs())
        {
            if (best_pref as i32 - raw_end_excl as i32).abs() <= 3 {
                return best_pref;
            }
        }

        let valid_allowed: Vec<u16> = sh
            .allowed_cut_layers
            .iter()
            .copied()
            .filter(|&c| c >= lo && c <= hi)
            .collect();
        if let Some(&best_allowed) = valid_allowed
            .iter()
            .min_by_key(|&&c| (c as i32 - raw_end_excl as i32).abs())
        {
            return best_allowed;
        }

        let align = self.layer_group_alignment();
        if align > 1 {
            let snapped = ((raw_end_excl + align / 2) / align) * align;
            snapped.clamp(lo, hi)
        } else {
            raw_end_excl.clamp(lo, hi)
        }
    }

    fn stage0_pinned_layers(&self) -> BTreeSet<u16> {
        let sh = &self.policy().sharding;
        if sh.stage0_pin_mtp_layers {
            sh.mtp_layers.iter().copied().collect()
        } else {
            BTreeSet::new()
        }
    }

    fn build_stage0_tensor_overrides(&self, primary_buft: &str) -> String {
        let sh = &self.policy().sharding;
        let mut rules = Vec::new();
        if sh.stage0_pin_output_head {
            rules.push(format!("^output\\.weight={}", primary_buft));
            rules.push(format!("^output_norm\\.weight={}", primary_buft));
        }
        if sh.stage0_pin_mtp_layers {
            for &l in &sh.mtp_layers {
                rules.push(format!("^blk\\.{}\\..*={}", l, primary_buft));
            }
        }
        rules.join(",")
    }

    /// Produces an engine-neutral [`EngineLaunchSpec`] tailored for this model and runtime topology.
    fn engine_launch_spec(&self, params: &LlamaServerLaunchParams<'_>) -> EngineLaunchSpec {
        let p = self.policy();
        let ctx_size = if params.requested_ctx_size == 0 {
            p.engine.default_ctx_size.max(p.engine.max_context_length)
        } else {
            params.requested_ctx_size
        };
        let ubatch_size = if params.requested_ubatch_size == 0 {
            p.engine.default_ubatch_size
        } else {
            params.requested_ubatch_size
        };

        // On distributed RPC rigs, RAM context checkpoints (`-ctxcp` / `-cram`) trigger full
        // state tensor downloads over WAN (`RPC_CMD_GET_TENSOR`) multiple times per prompt.
        // Always disable RAM checkpoints on multi-node RPC rigs while preserving in-VRAM `cache_prompt`.
        let is_distributed_rpc = !params.ordered_bridges.is_empty();
        let ctx_checkpoints = if is_distributed_rpc {
            0
        } else {
            p.engine.ctx_checkpoints
        };
        let cache_ram_mib = if is_distributed_rpc {
            0
        } else {
            p.engine.cache_ram_mib
        };

        let mut draft_model_path = None;
        let mut spec_type = None;
        if p.speculative.enabled {
            if !p.speculative.draft_model_filename.is_empty() {
                if let Some(parent) = params.model_path.parent() {
                    let candidate = parent.join(&p.speculative.draft_model_filename);
                    if candidate.exists() {
                        draft_model_path = Some(candidate.to_string_lossy().to_string());
                    }
                }
            }
            if !p.speculative.spec_type.is_empty() && p.speculative.spec_type != "none" {
                let mapped = if p.speculative.spec_type == "mtp" {
                    "draft-mtp".to_string()
                } else {
                    p.speculative.spec_type.clone()
                };
                spec_type = Some(mapped);
            }
        }

        EngineLaunchSpec {
            model_path: params.model_path.to_string_lossy().to_string(),
            rpc_bridges: params.ordered_bridges.to_vec(),
            devices: params.devices_vec.to_vec(),
            tensor_splits: params.splits_vec.to_vec(),
            tensor_overrides: self.build_stage0_tensor_overrides(params.primary_buft),
            parallel_slots: p.engine.parallel_slots.max(1),
            continuous_batching: p.engine.continuous_batching,
            ctx_size,
            ubatch_size,
            ctx_checkpoints,
            cache_ram_mib,
            cache_prompt: p.engine.cache_prompt,
            cache_idle_slots: p.engine.cache_idle_slots,
            reasoning_format: p.engine.reasoning_format.clone(),
            speculative_enabled: p.speculative.enabled,
            draft_model_path,
            spec_type,
            draft_max: p.speculative.draft_max,
            draft_min: p.speculative.draft_min,
            draft_p_min: p.speculative.draft_p_min,
            port: params.port,
            no_warmup: p.engine.no_warmup,
            extra_args: p.engine.extra_args.clone(),
        }
    }

    fn build_engine_args(
        &self,
        engine: &dyn InferenceEngineAdapter,
        params: &LlamaServerLaunchParams<'_>,
    ) -> Vec<String> {
        let spec = self.engine_launch_spec(params);
        engine.build_args(&spec)
    }
}

/// Declarative adapter backed directly by any `models/<model_id>/policy.json` file or DB policy.
#[derive(Debug, Clone)]
pub struct PolicyFileAdapter {
    policy: ModelPolicy,
}

impl PolicyFileAdapter {
    pub fn new(policy: ModelPolicy) -> Self {
        Self { policy }
    }
}

impl ModelAdapter for PolicyFileAdapter {
    fn policy(&self) -> &ModelPolicy {
        &self.policy
    }
}
