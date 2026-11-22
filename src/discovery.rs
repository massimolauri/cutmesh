use crate::catalog::GpuNodeCapabilities;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_SWARM_REGISTRY_FILE: &str = ".cutmesh_swarm_registry.json";

/// Live Heartbeat & Auto-Discovery Lease published by a supervised GPU Worker.
/// Eliminates manual copy-pasting of `iroh://...` tickets and tracks self-healing restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerDiscoveryLease {
    pub node_name: String,
    pub endpoint_id: String,
    pub iroh_ticket: String,
    pub local_rpc_port: u16,
    pub device: String,
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub effective_tflops: f64,
    pub effective_bandwidth_gb_s: f64,
    pub allocated_power_watts: f64,
    pub temperature_c: u32,
    pub gpu_utilization_pct: u32,
    pub worker_pid: u32,
    pub rpc_child_pid: u32,
    pub self_healing_restarts: u32,
    pub health_status: String,
    pub last_heartbeat_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SwarmDiscoveryRegistry {
    pub updated_at_unix_ms: u64,
    pub active_workers: Vec<WorkerDiscoveryLease>,
}

pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl SwarmDiscoveryRegistry {
    pub fn load_or_default(path: &Path) -> Self {
        if let Ok(raw) = std::fs::read_to_string(path) {
            if let Ok(reg) = serde_json::from_str::<SwarmDiscoveryRegistry>(&raw) {
                return reg;
            }
        }
        Self::default()
    }

    pub fn save_atomic(&self, path: &Path) -> Result<(), String> {
        let tmp_path = path.with_extension("json.tmp");
        let serialized = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&tmp_path, serialized).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp_path, path).map_err(|e| e.to_string())
    }

    /// Upserts a Worker's live lease and prunes stale entries older than `max_age_ms`.
    pub fn upsert_worker_lease(
        path: &Path,
        lease: WorkerDiscoveryLease,
        max_age_ms: u64,
    ) -> Result<Self, String> {
        let mut reg = Self::load_or_default(path);
        let now = now_unix_ms();
        reg.active_workers.retain(|w| {
            w.node_name != lease.node_name
                && w.endpoint_id != lease.endpoint_id
                && now.saturating_sub(w.last_heartbeat_unix_ms) <= max_age_ms
        });
        reg.active_workers.push(lease);
        reg.active_workers.sort_by(|a, b| a.node_name.cmp(&b.node_name));
        reg.updated_at_unix_ms = now;
        reg.save_atomic(path)?;
        Ok(reg)
    }

    /// Discovers all currently healthy Worker `iroh://...` tickets without manual copy-paste.
    pub fn discover_healthy_tickets(path: &Path, max_age_ms: u64) -> Vec<WorkerDiscoveryLease> {
        let reg = Self::load_or_default(path);
        let now = now_unix_ms();
        reg.active_workers
            .into_iter()
            .filter(|w| {
                now.saturating_sub(w.last_heartbeat_unix_ms) <= max_age_ms
                    && w.health_status.starts_with("ONLINE")
            })
            .collect()
    }
}

/// Evaluates the live GPU silicon telemetry (`nvml.dll`) to determine the Worker's operational health state.
pub fn classify_worker_health_state(caps: &GpuNodeCapabilities) -> String {
    if caps.temperature_c >= 88 {
        format!(
            "DRAINING_THERMAL_THROTTLE ({}°C >= 88°C)",
            caps.temperature_c
        )
    } else {
        format!(
            "ONLINE_HEALTHY ({}°C, {} MB free VRAM)",
            caps.temperature_c, caps.vram_free_mb
        )
    }
}

/// DTO sent every 2 seconds by a GPU Worker container/process to the Spring Boot Control Plane
/// (`POST /api/v1/internal/workers/heartbeat`) authenticated via `CUTMESH_NODE_TOKEN`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlPlaneHeartbeatRequest {
    pub node_token: String,
    pub node_name: String,
    pub endpoint_id: String,
    pub iroh_ticket: String,
    pub local_rpc_port: u16,
    pub device: String,
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub effective_tflops: f64,
    pub effective_bandwidth_gb_s: f64,
    pub allocated_power_watts: f64,
    pub power_budget_pct: u32,
    pub temperature_c: u32,
    pub gpu_utilization_pct: u32,
    pub self_healing_restarts: u32,
    pub health_status: String,
    pub pool_mode: String,
    pub measured_rtt_ms: f64,
}

/// Response returned by the Spring Boot Control Plane to the GPU Worker upon heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlaneHeartbeatResponse {
    pub accepted: bool,
    pub assignment_state: String,
    pub assigned_rig_id: Option<String>,
    pub assigned_model_id: Option<String>,
    pub assigned_role: Option<String>,
    #[serde(default)]
    pub assigned_model_url: Option<String>,
    #[serde(default)]
    pub assigned_layer_start: Option<u16>,
    #[serde(default)]
    pub assigned_layer_end: Option<u16>,
    pub next_heartbeat_secs: Option<u64>,
}

fn shared_control_plane_http_client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: std::sync::OnceLock<Result<reqwest::Client, String>> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(4))
                .connect_timeout(std::time::Duration::from_secs(2))
                .build()
                .map_err(|e| format!("Failed building Control Plane HTTP client: {}", e))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

fn normalize_hub_url(hub_base_url: &str, suffix: &str) -> String {
    let trimmed = hub_base_url.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        format!("{}{}", trimmed, suffix)
    } else {
        format!("http://{}{}", trimmed, suffix)
    }
}

/// Sends a non-blocking HTTP/HTTPS heartbeat from the Rust GPU Worker to the Spring Boot Control Plane.
pub async fn send_heartbeat_to_control_plane(
    hub_base_url: &str,
    payload: &ControlPlaneHeartbeatRequest,
) -> Result<ControlPlaneHeartbeatResponse, String> {
    let client = shared_control_plane_http_client()?;
    let url = normalize_hub_url(hub_base_url, "/api/v1/internal/workers/heartbeat");

    let max_allocatable_mb = ((payload.vram_free_mb as f64) * 0.85).round() as u64;
    let wire_json = serde_json::json!({
        "iroh_node_id": payload.endpoint_id,
        "iroh_ticket": payload.iroh_ticket,
        "gpu_name": payload.gpu_name,
        "free_vram_mb": payload.vram_free_mb,
        "total_vram_mb": payload.vram_total_mb,
        "max_allocatable_vram_mb": max_allocatable_mb.max(256),
        "pool_mode": payload.pool_mode,
        "rpc_healthy": payload.health_status.starts_with("ONLINE"),
        "watchdog_restarts": payload.self_healing_restarts,
        "version": "1.0.0",
        "node_name": payload.node_name,
        "local_rpc_port": payload.local_rpc_port,
        "device": payload.device,
        "effective_tflops": payload.effective_tflops,
        "effective_bandwidth_gb_s": payload.effective_bandwidth_gb_s,
        "allocated_power_watts": payload.allocated_power_watts,
        "power_budget_pct": payload.power_budget_pct,
        "temperature_c": payload.temperature_c,
        "gpu_utilization_pct": payload.gpu_utilization_pct
    });

    let mut req = client.post(&url).json(&wire_json);
    if !payload.node_token.trim().is_empty() {
        req = req.bearer_auth(payload.node_token.trim());
    }

    let resp = tokio::time::timeout(std::time::Duration::from_millis(3500), req.send())
        .await
        .map_err(|_| "Control Plane heartbeat timeout (3500ms)".to_string())?
        .map_err(|e| format!("Control Plane heartbeat request error: {}", e))?;

    let status = resp.status();
    let body_text = resp
        .text()
        .await
        .map_err(|e| format!("Control Plane read error: {}", e))?;
    if !status.is_success() {
        return Err(format!(
            "Control Plane heartbeat rejected (HTTP {}): {}",
            status.as_u16(),
            body_text
        ));
    }

    serde_json::from_str::<ControlPlaneHeartbeatResponse>(&body_text)
        .map_err(|e| format!("Invalid Control Plane heartbeat response JSON: {}", e))
}

/// Sends an immediate graceful disconnect notification to the Spring Boot Control Plane
/// (`POST /api/v1/internal/workers/disconnect`) when the worker receives Ctrl+C / SIGTERM.
pub async fn send_disconnect_to_control_plane(
    hub_base_url: &str,
    node_token: &str,
    iroh_node_id: &str,
) -> Result<(), String> {
    let client = shared_control_plane_http_client()?;
    let url = normalize_hub_url(hub_base_url, "/api/v1/internal/workers/disconnect");
    let wire_json = serde_json::json!({
        "iroh_node_id": iroh_node_id
    });

    let mut req = client.post(&url).json(&wire_json);
    if !node_token.trim().is_empty() {
        req = req.bearer_auth(node_token.trim());
    }
    let _ = tokio::time::timeout(std::time::Duration::from_millis(1500), req.send()).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_paste_auto_discovery_and_self_healing_lease() {
        let tmp = std::env::temp_dir().join("test_cutmesh_discovery_registry.json");
        let _ = std::fs::remove_file(&tmp);

        let lease1 = WorkerDiscoveryLease {
            node_name: "worker-rtx4060ti-1".to_string(),
            endpoint_id: "ed25519:1111".to_string(),
            iroh_ticket: "iroh://ed25519:1111@127.0.0.1:51001".to_string(),
            local_rpc_port: 50051,
            device: "CUDA0".to_string(),
            gpu_name: "NVIDIA GeForce RTX 4060 Ti".to_string(),
            vram_total_mb: 8188,
            vram_free_mb: 6350,
            effective_tflops: 21.0,
            effective_bandwidth_gb_s: 288.0,
            allocated_power_watts: 160.0,
            temperature_c: 28,
            gpu_utilization_pct: 5,
            worker_pid: 1234,
            rpc_child_pid: 5678,
            self_healing_restarts: 1,
            health_status: "ONLINE_HEALTHY (28°C, 6350 MB free VRAM)".to_string(),
            last_heartbeat_unix_ms: now_unix_ms(),
        };

        SwarmDiscoveryRegistry::upsert_worker_lease(&tmp, lease1, 30_000).unwrap();
        let discovered = SwarmDiscoveryRegistry::discover_healthy_tickets(&tmp, 30_000);
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].iroh_ticket, "iroh://ed25519:1111@127.0.0.1:51001");
        assert_eq!(discovered[0].self_healing_restarts, 1);

        let _ = std::fs::remove_file(&tmp);
    }
}

