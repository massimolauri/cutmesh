use crate::models::adapter::{ModelAdapter, PolicyFileAdapter};
use crate::models::policy::ModelPolicy;
use crate::models::qwen38_27b::Qwen38_27BAdapter;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Registry Pattern for resolving per-model [`ModelAdapter`] instances dynamically.
///
/// Resolution priority:
/// 1. Runtime Control Plane override (`CUTMESH_MODEL_POLICY_JSON`)
/// 2. Disk policies discovered under `models/*/policy.json` (`CUTMESH_MODELS_DIR`, `./models`,
///    `/data/cutmesh/models`, `/opt/cutmesh/models`)
/// 3. Built-in dedicated per-model adapters (currently [`Qwen38_27BAdapter`])
/// 4. Clean generic fallback (`PolicyFileAdapter(ModelPolicy::generic_default(...))`)
pub struct ModelAdapterRegistry;

impl ModelAdapterRegistry {
    /// Wraps a concrete [`ModelPolicy`] in its dedicated per-model [`ModelAdapter`] when one
    /// exists, otherwise returns a generic [`PolicyFileAdapter`].
    pub fn wrap_policy(policy: ModelPolicy) -> Box<dyn ModelAdapter> {
        if policy.model_id.eq_ignore_ascii_case(Qwen38_27BAdapter::MODEL_ID) {
            Box::new(Qwen38_27BAdapter::new(policy))
        } else {
            Box::new(PolicyFileAdapter::new(policy))
        }
    }

    /// Discovers all catalog policies from disk + built-in adapters, prepending any runtime
    /// `CUTMESH_MODEL_POLICY_JSON` override.
    pub fn load_catalog_policies() -> Vec<ModelPolicy> {
        static BASE_POLICIES: OnceLock<Vec<ModelPolicy>> = OnceLock::new();
        let base = BASE_POLICIES.get_or_init(|| {
            let mut policies: Vec<ModelPolicy> = Vec::new();
            let mut loaded_ids: BTreeSet<String> = BTreeSet::new();

            let search_roots = [
                std::env::var("CUTMESH_MODELS_DIR").ok().map(PathBuf::from),
                Some(PathBuf::from("models")),
                Some(PathBuf::from("/data/cutmesh/models")),
                Some(PathBuf::from("/opt/cutmesh/models")),
            ];

            for root in search_roots.into_iter().flatten() {
                if let Ok(entries) = std::fs::read_dir(&root) {
                    for entry in entries.flatten() {
                        let policy_file = entry.path().join("policy.json");
                        if policy_file.is_file() {
                            if let Ok(raw) = std::fs::read_to_string(&policy_file) {
                                if let Ok(parsed) = serde_json::from_str::<ModelPolicy>(&raw) {
                                    if loaded_ids.insert(parsed.model_id.clone()) {
                                        policies.push(parsed);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Ensure Qwen 3.8 27B dedicated adapter is always registered even if running outside repo root.
            let qwen = Qwen38_27BAdapter::embedded_default().policy().clone();
            if loaded_ids.insert(qwen.model_id.clone()) {
                policies.push(qwen);
            }

            policies
        });

        if let Ok(db_json) = std::env::var("CUTMESH_MODEL_POLICY_JSON") {
            let trimmed = db_json.trim();
            if !trimmed.is_empty() {
                if let Ok(parsed) = serde_json::from_str::<ModelPolicy>(trimmed) {
                    let mut out = vec![parsed.clone()];
                    for p in base {
                        if p.model_id != parsed.model_id {
                            out.push(p.clone());
                        }
                    }
                    return out;
                }
            }
        }

        base.clone()
    }

    /// Resolves the best-matching [`ModelAdapter`] for a GGUF path, HuggingFace URL, or `model_id`.
    pub fn resolve(model_source: &str) -> Box<dyn ModelAdapter> {
        if let Ok(db_json) = std::env::var("CUTMESH_MODEL_POLICY_JSON") {
            let trimmed = db_json.trim();
            if !trimmed.is_empty() {
                if let Ok(parsed) = serde_json::from_str::<ModelPolicy>(trimmed) {
                    return Self::wrap_policy(parsed);
                }
            }
        }

        for policy in Self::load_catalog_policies() {
            let adapter = Self::wrap_policy(policy);
            if adapter.matches(model_source) {
                return adapter;
            }
        }

        Box::new(PolicyFileAdapter::new(ModelPolicy::generic_default(
            model_source,
        )))
    }
}
