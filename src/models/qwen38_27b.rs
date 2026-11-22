use crate::models::adapter::{EngineLaunchSpec, ModelAdapter};
use crate::models::policy::{LlamaServerLaunchParams, ModelPolicy};

const EMBEDDED_QWEN38_27B_POLICY: &str = include_str!("../../models/qwen3.8-27b/policy.json");

/// Dedicated per-model adapter for **Qwen 3.8 27B (`qwen3.8-27b`)**.
///
/// Encapsulates Qwen 3.8 27B-specific invariants on top of its declarative `policy.json`:
/// - Atomic `3×DeltaNet + 1×FullAttn` (`group_align = 4`) cut alignment across its 64 trunk layers
/// - Stage-0 pinning for `output.weight`, `output_norm.weight`, and MTP layer `blk.64`
/// - Zero WAN RAM checkpoints (`-ctxcp 0 -cram 0`) on distributed RPC rigs so DeltaNet recurrent
///   states (`~60-96 MiB` per snapshot) are never downloaded across WAN mid-prefill
/// - Q8_0 KV cache flags (`-fa on`, `-ctk q8_0`, `-ctv q8_0`, `-ctkd q8_0`, `-ctvd q8_0`) for full 262k context
#[derive(Debug, Clone)]
pub struct Qwen38_27BAdapter {
    policy: ModelPolicy,
}

impl Qwen38_27BAdapter {
    pub const MODEL_ID: &'static str = "qwen3.8-27b";

    pub fn new(policy: ModelPolicy) -> Self {
        Self { policy }
    }

    pub fn embedded_default() -> Self {
        let policy: ModelPolicy = serde_json::from_str(EMBEDDED_QWEN38_27B_POLICY)
            .expect("valid embedded models/qwen3.8-27b/policy.json");
        Self { policy }
    }
}

impl ModelAdapter for Qwen38_27BAdapter {
    fn policy(&self) -> &ModelPolicy {
        &self.policy
    }

    fn engine_launch_spec(&self, params: &LlamaServerLaunchParams<'_>) -> EngineLaunchSpec {
        let mut spec = self.default_launch_spec(params);
        // Qwen 3.8 27B carries 48 recurrent DeltaNet layers: never allow RAM checkpoints on RPC rigs.
        if !params.ordered_bridges.is_empty() {
            spec.ctx_checkpoints = 0;
            spec.cache_ram_mib = 0;
        }
        // Guarantee Flash Attention + Q8_0 KV quantization are present even if an older DB policy omitted them.
        ensure_arg_pair(&mut spec.extra_args, "-fa", "on");
        ensure_arg_pair(&mut spec.extra_args, "-ctk", "q8_0");
        ensure_arg_pair(&mut spec.extra_args, "-ctv", "q8_0");
        ensure_arg_pair(&mut spec.extra_args, "-ctkd", "q8_0");
        ensure_arg_pair(&mut spec.extra_args, "-ctvd", "q8_0");
        spec
    }
}

impl Qwen38_27BAdapter {
    fn default_launch_spec(&self, params: &LlamaServerLaunchParams<'_>) -> EngineLaunchSpec {
        crate::models::adapter::PolicyFileAdapter::new(self.policy.clone()).engine_launch_spec(params)
    }
}

fn ensure_arg_pair(args: &mut Vec<String>, flag: &str, value: &str) {
    if !args.windows(2).any(|w| w[0] == flag) {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}
