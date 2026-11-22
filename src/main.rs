mod cli_runtime;
use cli_runtime::{
    cuda_path_env, ld_library_path_env, maybe_trigger_worker_shard_prewarm, prewarm_workers_parallel,
    probe_tickets_parallel, resolve_llama_bin_dir, shutdown_signal, spawn_ggml_rpc_process,
    wait_for_llama_server_ready, ChildGuard,
};

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tracing::{info, warn};

use cutmesh::catalog::{
    build_model_catalog_entry, detect_local_gpu_capabilities_with_budget, schedule_nodes_for_model,
    GpuNodeCapabilities,
};
use cutmesh::discovery::{
    classify_worker_health_state, now_unix_ms, send_disconnect_to_control_plane,
    send_heartbeat_to_control_plane, ControlPlaneHeartbeatRequest, SwarmDiscoveryRegistry,
    WorkerDiscoveryLease, DEFAULT_SWARM_REGISTRY_FILE,
};
use cutmesh::gguf::slicer::{
    default_ggml_rpc_cache_dir, group_layers, plan_shard_for_vram_mb, populate_ggml_rpc_cache,
};
use cutmesh::governance::{
    evaluate_external_cluster_governance, ExternalClusterStateInput, GovernancePolicyConfig,
};
use cutmesh::process_guard::{
    acquire_single_instance, kill_orphan, remove_pid, tcp_port_in_use, write_pid,
};
use cutmesh::rig::{
    build_sparse_master_gguf, ensure_tensor_hash_manifest, fetch_or_parse_gguf_header,
    form_and_close_rigs_with_min_nodes, pull_rig_shard_to_rpc_cache, resolve_remote_gguf_url,
};
use cutmesh::rpc_tunnel::{
    register_sparse_hash_translations, request_worker_spawn_master_rig,
    spawn_client_iroh_http_bridge, spawn_client_iroh_rpc_bridge, spawn_worker_iroh_rpc_server,
    stop_active_master_llama_server_if_idle, SpawnMasterRigRequest,
};
use cutmesh::shield::{
    compile_covariant_gguf, execute_laptop_micro_shield_with_input, ExternalShieldInput,
};
use cutmesh::swarm::{
    spawn_lightweight_swarm_client_router_with_mode, spawn_prewarmed_swarm_pod_endpoint,
    SwarmPodTicket,
};

#[derive(Parser)]
#[command(
    name = "cutmesh",
    about = "Production Iroh Direct-QUIC P2P Distributed LLM Engine (llama.cpp + ggml-rpc)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a Model Catalog Entry & Schedule Layers from live Iroh Tickets or an external Workers JSON file
    Catalog {
        #[arg(long)]
        model: PathBuf,
        /// Optional external JSON file containing a Vec<GpuNodeCapabilities>
        #[arg(long)]
        workers_json: Option<PathBuf>,
        /// Optional live Iroh worker tickets (iroh://...) separated by comma
        #[arg(long, value_delimiter = ',')]
        tickets: Vec<String>,
        #[arg(long, default_value_t = false)]
        wan_dht: bool,
        /// Local GPU power budget % if scheduling against local hardware
        #[arg(long, default_value_t = 100)]
        power_pct: u32,
    },
    /// Inspect a .gguf model file and compute exact layer byte ranges for an external VRAM budget
    InspectGguf {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        vram_mb: u64,
    },
    /// Start a standalone llama.cpp GPU Worker over Iroh Direct-QUIC (HW Probe + ggml-rpc)
    Worker {
        #[arg(long, default_value = "gpu-worker-auto")]
        name: String,
        #[arg(long, default_value_t = 0)]
        quic_port: u16,
        #[arg(long, default_value_t = 50051)]
        local_rpc_port: u16,
        #[arg(long, default_value = "CUDA0")]
        device: String,
        /// TEST ONLY: deterministic identity. Omit in production: each node gets a unique,
        /// persistent key stored in CUTMESH_DATA_DIR (default ~/.cutmesh).
        #[arg(long)]
        seed: Option<u8>,
        #[arg(long, default_value_t = false)]
        wan_dht: bool,
        /// Percentage of GPU power/load to allocate to P2P inference (10..=100)
        #[arg(long, default_value_t = 100)]
        power_pct: u32,
        #[arg(long, default_value = "bin/llamacpp")]
        llama_bin_dir: PathBuf,
        /// Optional Spring Boot Control Plane URL (or set CUTMESH_HUB_URL env var)
        #[arg(long)]
        hub_url: Option<String>,
        /// Optional Node Enrollment Token generated from the Web Dashboard (or set CUTMESH_NODE_TOKEN env var)
        #[arg(long)]
        node_token: Option<String>,
        /// Pool mode: PUBLIC_AUTOPILOT or PRIVATE_DEDICATED (or set CUTMESH_POOL_MODE env var)
        #[arg(long, default_value = "PUBLIC_AUTOPILOT")]
        pool_mode: String,
    },
    /// Probe P2P Workers over Iroh QUIC, schedule layers by VRAM/TFLOPS/Watts, and start llama-server OpenAI Pod
    Gateway {
        #[arg(long)]
        model: PathBuf,
        /// Optional HuggingFace .gguf URL so Workers download their assigned weights directly (auto-resolved for known models)
        #[arg(long)]
        model_url: Option<String>,
        #[arg(long, value_delimiter = ',')]
        tickets: Vec<String>,
        #[arg(long, default_value_t = 8089)]
        port: u16,
        #[arg(long, default_value_t = 4096)]
        ctx_size: u64,
        #[arg(long, default_value_t = 4)]
        parallel_slots: usize,
        #[arg(long, default_value = "swarm-pod-1")]
        pod_name: String,
        #[arg(long, default_value_t = false)]
        wan_dht: bool,
        #[arg(long, default_value_t = false)]
        local_cuda: bool,
        #[arg(long, default_value_t = 512)]
        ubatch_size: u64,
        #[arg(long, default_value_t = false)]
        no_warmup: bool,
        #[arg(long, default_value = "bin/llamacpp")]
        llama_bin_dir: PathBuf,
    },
    /// Start a Zero-GGUF Lightweight Client Router connected to Pre-Warmed Swarm Pods (pod://...)
    ClientRouter {
        #[arg(long, value_delimiter = ',')]
        pods: Vec<String>,
        /// Enable E2E encrypted QUIC relay fallback for laptop clients on 5G / Hotel Symmetric NAT
        #[arg(long, default_value_t = false)]
        mobile_relay_fallback: bool,
    },
    /// Execute Laptop CPU Micro-Shield using an external JSON input file (token IDs + session keys)
    Shield {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input_json: PathBuf,
    },
    /// Evaluate Decentralized Swarm Governance using external Cluster State JSON and Policy JSON files
    Govern {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cluster_json: PathBuf,
        #[arg(long)]
        policy_json: PathBuf,
    },
    /// Populate local ggml-rpc-server -c FNV-1a tensor cache directory from a .gguf file
    PrewarmCache {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        start_layer: Option<u16>,
        #[arg(long)]
        end_layer: Option<u16>,
    },
    /// Automatically form & close P2P GPU Rigs (`ClosedAndLocked`), elect each Rig Master (`CUDA0`), and compute per-node HTTP Range slices from a local `.gguf` or remote HuggingFace URL
    PlanRig {
        /// Local `.gguf` file path OR remote `https://huggingface.co/.../*.gguf` URL (inspected via 12 MB HTTP Range without downloading weights)
        #[arg(long)]
        model: String,
        /// Optional external JSON file containing a Vec<GpuNodeCapabilities>
        #[arg(long)]
        workers_json: Option<PathBuf>,
        /// Optional live Iroh worker tickets (iroh://...) separated by comma
        #[arg(long, value_delimiter = ',')]
        tickets: Vec<String>,
        /// Enable WAN DHT discovery when probing live tickets
        #[arg(long, default_value_t = false)]
        wan_dht: bool,
        /// Maximum inter-node RTT (ms) allowed when clustering GPUs into the same Rig
        #[arg(long, default_value_t = 15.0)]
        max_rtt_ms: f64,
        /// Safety VRAM reserve (MiB) kept free on each GPU for CUDA compute graphs & buffers
        #[arg(long, default_value_t = 1536)]
        vram_reserve_mb: u64,
        /// Minimum number of GPU nodes required before closing a Rig
        #[arg(long, default_value_t = 1)]
        min_nodes: usize,
        /// Also include the local GPU (`CUDA0`) in the candidate Rig pool alongside remote tickets
        #[arg(long, default_value_t = false)]
        include_local_cuda: bool,
        /// Local GPU power percentage if probing local GPU
        #[arg(long, default_value_t = 100)]
        power_pct: u32,
    },
    /// Download or extract ONLY the assigned layer slice (`start_layer..=end_layer`) via HTTP Range into the `ggml-rpc` cache without downloading the full `.gguf`
    PullRigShard {
        /// Local `.gguf` file path OR remote `https://huggingface.co/.../*.gguf` URL
        #[arg(long)]
        model: String,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        start_layer: u16,
        #[arg(long)]
        end_layer: u16,
        #[arg(long, default_value_t = false)]
        include_non_block: bool,
    },
    /// Zero-Touch Automatic Rig Orchestrator: probes local CUDA0 + remote Iroh workers, closes the Rig,
    /// elects the Rig Master, runs the Orthogonal Residual Cipher (Q in O(d)) Logit Lens audit, and launches the P2P Gateway
    AutoRig {
        #[arg(long)]
        model: PathBuf,
        /// Optional HuggingFace .gguf URL so Workers download their assigned weights directly (auto-resolved for known models)
        #[arg(long)]
        model_url: Option<String>,
        #[arg(long, value_delimiter = ',')]
        tickets: Vec<String>,
        #[arg(long, default_value_t = false)]
        wan_dht: bool,
        #[arg(long, default_value_t = 2)]
        min_nodes: usize,
        #[arg(long, default_value_t = 15.0)]
        max_rtt_ms: f64,
        #[arg(long, default_value_t = 1536)]
        vram_reserve_mb: u64,
        #[arg(long, default_value_t = false)]
        include_local_cuda: bool,
        #[arg(long, default_value_t = 100)]
        power_pct: u32,
        #[arg(long, default_value = "bin/llamacpp")]
        llama_bin_dir: PathBuf,
        #[arg(long, default_value_t = 8089)]
        port: u16,
        /// Context window size (`0` = automatically maximize `-c` to fill 100% of the closed Rig's remaining VRAM)
        #[arg(long, default_value_t = 0)]
        ctx_size: u32,
        #[arg(long, default_value_t = 512)]
        ubatch_size: u32,
    },
    /// Physically compile a Covariant-GL(d) or Strict-Fidelity O(d) transformed .gguf file on disk + Zero-Weight Client key manifest
    CompileCovariantGguf {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "cutmesh-covariant-session-secret-2026")]
        secret_seed: String,
        /// Anisotropy condition number kappa_Lambda (1.0 = Strict-Fidelity O(d), 2.0 = Covariant-GL(d))
        #[arg(long, default_value_t = 2.0)]
        kappa_lambda: f64,
    },
    /// Inspect the Zero-Copy-Paste Auto-Discovery Registry and live Self-Healing Worker leases
    Discover {
        #[arg(long, default_value = DEFAULT_SWARM_REGISTRY_FILE)]
        registry_path: PathBuf,
    },
    /// Execute the live Zero-Weight-Disclosure Covariant Protocol Audit on a real .gguf model and output full JSON metrics
    AuditCovariant {
        #[arg(long, default_value = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")]
        model: PathBuf,
        #[arg(long, default_value_t = 896)]
        dim: usize,
        #[arg(long, default_value_t = 151936)]
        vocab_size: usize,
        #[arg(long, default_value = "cutmesh-covariant-session-secret-2026")]
        secret_seed: String,
    },
}

fn tail_master_llama_log(last_offset: &mut u64) {
    use std::io::{Read, Seek, SeekFrom};
    let log_path = std::env::temp_dir().join("llama-server-master.log");
    let Ok(meta) = std::fs::metadata(&log_path) else {
        return;
    };
    let len = meta.len();
    if len < *last_offset {
        *last_offset = 0;
    }
    if len <= *last_offset {
        return;
    }
    let Ok(mut f) = std::fs::File::open(&log_path) else {
        return;
    };
    if f.seek(SeekFrom::Start(*last_offset)).is_err() {
        return;
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return;
    }
    *last_offset += buf.len() as u64;
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim_end())
        .filter(|l| !l.is_empty())
        .collect();
    if !lines.is_empty() {
        let start = lines.len().saturating_sub(30);
        info!("[llama-server-master]\n{}", lines[start..].join("\n"));
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "cutmesh=info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Catalog {
            model,
            workers_json,
            tickets,
            wan_dht,
            power_pct,
        } => {
            let entry = build_model_catalog_entry(&model)?;
            let mut candidates: Vec<GpuNodeCapabilities> = Vec::new();

            if let Some(json_path) = workers_json {
                let raw = std::fs::read_to_string(&json_path)?;
                let loaded: Vec<GpuNodeCapabilities> = serde_json::from_str(&raw)?;
                candidates.extend(loaded);
            }

            for (_tk, caps) in probe_tickets_parallel(&tickets, wan_dht).await? {
                candidates.push(caps);
            }

            if candidates.is_empty() {
                candidates.push(detect_local_gpu_capabilities_with_budget(
                    "local-probed-gpu",
                    "local-endpoint",
                    power_pct,
                ));
            }

            let plan = schedule_nodes_for_model(&entry, &candidates)?;
            let out = serde_json::json!({
                "model_catalog_entry": entry,
                "swarm_execution_plan": plan,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::InspectGguf { path, vram_mb } => {
            let path_str = path.to_string_lossy().to_string();
            let (hdr, model_name) = fetch_or_parse_gguf_header(&path_str)?;
            let layers = group_layers(&hdr);
            let plan = plan_shard_for_vram_mb(&hdr, 0, vram_mb, 1024);
            let weights_mb = hdr.tensors.iter().map(|t| t.byte_size as f64).sum::<f64>() / (1024.0 * 1024.0);
            let catalog_entry = cutmesh::catalog::build_model_catalog_entry_from_header(&hdr, &model_name).ok();
            let kv_mb = catalog_entry.as_ref().map(|c| c.kv_cache_4k_ctx_mb).unwrap_or(1024);
            let target_rig_vram_mb = (weights_mb.ceil() as u64) + kv_mb + 512;
            let out = serde_json::json!({
                "version": hdr.version,
                "architecture": hdr.architecture,
                "total_layers": hdr.total_layers,
                "embedding_length": hdr.embedding_length,
                "vocab_size": hdr.vocab_size,
                "tensor_count": hdr.tensor_count,
                "layer_count_detected": layers.len(),
                "weights_mb": weights_mb.round() as u64,
                "target_rig_vram_mb": target_rig_vram_mb,
                "catalog_entry": catalog_entry,
                "shard_plan": plan.map(|p| serde_json::json!({
                    "start_layer": p.start_layer,
                    "end_layer": p.end_layer,
                    "weights_mb": p.weights_bytes / (1024 * 1024),
                    "http_range_bytes": format!("bytes={}-{}", p.file_byte_range.0, p.file_byte_range.1),
                }))
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::Worker {
            name,
            quic_port,
            local_rpc_port,
            device,
            seed,
            wan_dht,
            power_pct,
            llama_bin_dir,
            hub_url,
            node_token,
            pool_mode,
        } => {
            let effective_hub_url = hub_url.or_else(|| std::env::var("CUTMESH_HUB_URL").ok());
            let effective_node_token = node_token
                .or_else(|| std::env::var("CUTMESH_NODE_TOKEN").ok())
                .unwrap_or_else(|| "syn_node_dev_local_default".to_string());
            let effective_pool_mode =
                std::env::var("CUTMESH_POOL_MODE").unwrap_or_else(|_| pool_mode.clone());
            let effective_power_pct = std::env::var("CUTMESH_POWER_PCT")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(power_pct);

            let local_rpc_addr = format!("127.0.0.1:{}", local_rpc_port);
            cutmesh::rpc_tunnel::set_active_worker_device(&device);

            // Single instance per QUIC port + cleanup of an orphaned ggml-rpc-server from a previous run
            let worker_lock = format!("worker-{}", quic_port);
            let rpc_lock = format!("rpc-{}", local_rpc_port);
            acquire_single_instance(&worker_lock, "cutmesh")?;
            kill_orphan(&rpc_lock, "rpc-server");
            if tcp_port_in_use(local_rpc_port) {
                return Err(format!(
                    "La porta RPC TCP {} è già occupata da un altro processo: ggml-rpc-server non può partire. Liberala o usa --local-rpc-port diverso.",
                    local_rpc_port
                )
                .into());
            }

            let mut rpc_child =
                spawn_ggml_rpc_process(&llama_bin_dir, local_rpc_port, &device)?;
            write_pid(&rpc_lock, rpc_child.id());
            let mut rpc_started_at = std::time::Instant::now();
            let mut rpc_fast_failures = 0u32;
            let mut rpc_next_respawn_at = std::time::Instant::now();
            let mut rpc_exit_noted = false;
            let mut last_master_llama_log_offset: u64 = 0;
            tokio::time::sleep(Duration::from_millis(800)).await;

            let (ticket, _stats, hw_caps) = spawn_worker_iroh_rpc_server(
                &name,
                quic_port,
                &local_rpc_addr,
                seed,
                wan_dht,
                effective_power_pct,
            )
            .await?;

            info!(
                "IROH DIRECT-QUIC GPU WORKER READY | Node: {} | GPU: {} ({} MB VRAM, Power Budget: {}% = {:.1}W, Eff: {:.1} TFLOPS, {:.0} GB/s)",
                name,
                hw_caps.gpu_name,
                hw_caps.vram_total_mb,
                hw_caps.user_power_budget_pct,
                hw_caps.allocated_power_budget_watts,
                hw_caps.effective_available_tflops,
                hw_caps.effective_available_bandwidth_gb_s
            );
            let ticket_str = ticket.to_string_ticket();
            info!("IROH TICKET (auto-published to {}): {}", DEFAULT_SWARM_REGISTRY_FILE, ticket_str);

            let reg_path = PathBuf::from(DEFAULT_SWARM_REGISTRY_FILE);
            let mut restarts = 0u32;
            let worker_pid = std::process::id();

            // Publish initial Auto-Discovery Lease immediately
            let initial_status = classify_worker_health_state(&hw_caps);
            let initial_lease = WorkerDiscoveryLease {
                node_name: name.clone(),
                endpoint_id: hw_caps.endpoint_id.clone(),
                iroh_ticket: ticket_str.clone(),
                local_rpc_port,
                device: device.clone(),
                gpu_name: hw_caps.gpu_name.clone(),
                vram_total_mb: hw_caps.vram_total_mb,
                vram_free_mb: hw_caps.vram_free_mb,
                effective_tflops: hw_caps.effective_available_tflops,
                effective_bandwidth_gb_s: hw_caps.effective_available_bandwidth_gb_s,
                allocated_power_watts: hw_caps.allocated_power_budget_watts,
                temperature_c: hw_caps.temperature_c,
                gpu_utilization_pct: hw_caps.gpu_utilization_pct,
                worker_pid,
                rpc_child_pid: rpc_child.id(),
                self_healing_restarts: restarts,
                health_status: initial_status.clone(),
                last_heartbeat_unix_ms: now_unix_ms(),
            };
            let _ = SwarmDiscoveryRegistry::upsert_worker_lease(&reg_path, initial_lease, 30_000);

            let mut last_prewarmed_shard: Option<(String, u16, u16)> = None;
            if let Some(ref hub) = effective_hub_url {
                let hb = ControlPlaneHeartbeatRequest {
                    node_token: effective_node_token.clone(),
                    node_name: name.clone(),
                    endpoint_id: hw_caps.endpoint_id.clone(),
                    iroh_ticket: ticket_str.clone(),
                    local_rpc_port,
                    device: device.clone(),
                    gpu_name: hw_caps.gpu_name.clone(),
                    vram_total_mb: hw_caps.vram_total_mb,
                    vram_free_mb: hw_caps.vram_free_mb,
                    effective_tflops: hw_caps.effective_available_tflops,
                    effective_bandwidth_gb_s: hw_caps.effective_available_bandwidth_gb_s,
                    allocated_power_watts: hw_caps.allocated_power_budget_watts,
                    power_budget_pct: effective_power_pct,
                    temperature_c: hw_caps.temperature_c,
                    gpu_utilization_pct: hw_caps.gpu_utilization_pct,
                    self_healing_restarts: restarts,
                    health_status: initial_status,
                    pool_mode: effective_pool_mode.clone(),
                    measured_rtt_ms: 0.0,
                };
                if let Ok(ack) = send_heartbeat_to_control_plane(hub, &hb).await {
                    info!(
                        "Enrolled with Spring Boot Control Plane ({}) -> State: {}, Assigned Rig: {:?}",
                        hub, ack.assignment_state, ack.assigned_rig_id
                    );
                    maybe_trigger_worker_shard_prewarm(&name, &ack, &mut last_prewarmed_shard);
                }
            }

            // Autonomous 5-second Self-Healing Supervisor & Heartbeat Loop
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            let mut was_assigned_to_rig = false;
            let shutdown = shutdown_signal();
            tokio::pin!(shutdown);
            loop {
                tokio::select! {
                    _ = &mut shutdown => {
                        info!("Shutting down supervised Worker {}...", name);
                        if let Some(ref hub) = effective_hub_url {
                            let _ = send_disconnect_to_control_plane(
                                hub,
                                &effective_node_token,
                                &hw_caps.endpoint_id,
                            ).await;
                        }
                        break;
                    }
                    _ = tick.tick() => {
                        cutmesh::rpc_tunnel::supervise_active_master_llama_server();
                        tail_master_llama_log(&mut last_master_llama_log_offset);

                        // 1. Supervise ggml-rpc-server: keep running across auto-rig role changes
                        //    and respawn with exponential backoff (2s..60s) on failure.
                        if let Ok(Some(exit_status)) = rpc_child.try_wait() {
                            let now = std::time::Instant::now();
                            if !rpc_exit_noted {
                                rpc_exit_noted = true;
                                if rpc_started_at.elapsed() < Duration::from_secs(10) {
                                    rpc_fast_failures += 1;
                                } else {
                                    rpc_fast_failures = 0;
                                }
                                if rpc_fast_failures >= 3 {
                                    let hint = if tcp_port_in_use(local_rpc_port) {
                                        format!("la porta TCP {} è occupata da un altro processo", local_rpc_port)
                                    } else {
                                        "controlla driver CUDA / binario ggml-rpc-server".to_string()
                                    };
                                    warn!(
                                        "WATCHDOG: ggml-rpc-server termina subito dopo l'avvio ({} volte di fila, ultimo stato: {}): {}",
                                        rpc_fast_failures, exit_status, hint
                                    );
                                }
                                let backoff_secs = 2u64.saturating_pow(rpc_fast_failures.min(6)).min(60);
                                rpc_next_respawn_at = now + Duration::from_secs(backoff_secs);
                            }
                            if now >= rpc_next_respawn_at {
                                restarts += 1;
                                info!(
                                    "SELF-HEALING WATCHDOG: ggml-rpc-server exited ({}) -> Respawning instance #{} on 127.0.0.1:{}...",
                                    exit_status, restarts, local_rpc_port
                                );
                                match spawn_ggml_rpc_process(&llama_bin_dir, local_rpc_port, &device) {
                                    Ok(new_child) => {
                                        write_pid(&rpc_lock, new_child.id());
                                        rpc_child = new_child;
                                    }
                                    Err(e) => warn!("WATCHDOG: respawn fallito: {}", e),
                                }
                                rpc_started_at = now;
                                rpc_exit_noted = false;
                            }
                        } else if cutmesh::rpc_tunnel::take_worker_rpc_recycle_request() {
                            info!(
                                "RIG SESSION ENDED -> Recycling ggml-rpc-server on 127.0.0.1:{} to release 100% of CUDA VMM pool back to NVML...",
                                local_rpc_port
                            );
                            let _ = rpc_child.kill();
                            let _ = rpc_child.wait();
                            match spawn_ggml_rpc_process(&llama_bin_dir, local_rpc_port, &device) {
                                Ok(new_child) => {
                                    write_pid(&rpc_lock, new_child.id());
                                    rpc_child = new_child;
                                    rpc_started_at = std::time::Instant::now();
                                    rpc_exit_noted = false;
                                }
                                Err(e) => warn!("WATCHDOG: recycle respawn fallito: {}", e),
                            }
                        }

                        // 2. Refresh live GPU silicon telemetry & Auto-Discovery heartbeat
                        let live_caps = cutmesh::catalog::detect_local_gpu_capabilities_with_budget(
                            &name,
                            &hw_caps.endpoint_id,
                            effective_power_pct,
                        );
                        let live_status = classify_worker_health_state(&live_caps);
                        let lease = WorkerDiscoveryLease {
                            node_name: name.clone(),
                            endpoint_id: hw_caps.endpoint_id.clone(),
                            iroh_ticket: ticket_str.clone(),
                            local_rpc_port,
                            device: device.clone(),
                            gpu_name: live_caps.gpu_name.clone(),
                            vram_total_mb: live_caps.vram_total_mb,
                            vram_free_mb: live_caps.vram_free_mb,
                            effective_tflops: live_caps.effective_available_tflops,
                            effective_bandwidth_gb_s: live_caps.effective_available_bandwidth_gb_s,
                            allocated_power_watts: live_caps.allocated_power_budget_watts,
                            temperature_c: live_caps.temperature_c,
                            gpu_utilization_pct: live_caps.gpu_utilization_pct,
                            worker_pid,
                            rpc_child_pid: rpc_child.id(),
                            self_healing_restarts: restarts,
                            health_status: live_status.clone(),
                            last_heartbeat_unix_ms: now_unix_ms(),
                        };
                        let _ = SwarmDiscoveryRegistry::upsert_worker_lease(&reg_path, lease, 60_000);

                        if let Some(ref hub) = effective_hub_url {
                            let hb = ControlPlaneHeartbeatRequest {
                                node_token: effective_node_token.clone(),
                                node_name: name.clone(),
                                endpoint_id: hw_caps.endpoint_id.clone(),
                                iroh_ticket: ticket_str.clone(),
                                local_rpc_port,
                                device: device.clone(),
                                gpu_name: live_caps.gpu_name.clone(),
                                vram_total_mb: live_caps.vram_total_mb,
                                vram_free_mb: live_caps.vram_free_mb,
                                effective_tflops: live_caps.effective_available_tflops,
                                effective_bandwidth_gb_s: live_caps.effective_available_bandwidth_gb_s,
                                allocated_power_watts: live_caps.allocated_power_budget_watts,
                                power_budget_pct: effective_power_pct,
                                temperature_c: live_caps.temperature_c,
                                gpu_utilization_pct: live_caps.gpu_utilization_pct,
                                self_healing_restarts: restarts,
                                health_status: live_status,
                                pool_mode: effective_pool_mode.clone(),
                                measured_rtt_ms: 0.0,
                            };
                            if let Ok(ack) = send_heartbeat_to_control_plane(hub, &hb).await {
                                if ack.assigned_rig_id.is_none() {
                                    stop_active_master_llama_server_if_idle();
                                    if was_assigned_to_rig {
                                        was_assigned_to_rig = false;
                                        cutmesh::rpc_tunnel::request_worker_rpc_recycle();
                                    }
                                } else {
                                    was_assigned_to_rig = true;
                                }
                                maybe_trigger_worker_shard_prewarm(
                                    &name,
                                    &ack,
                                    &mut last_prewarmed_shard,
                                );
                            }
                        }
                    }
                }
            }
            let _ = rpc_child.kill();
            let _ = rpc_child.wait();
            remove_pid(&rpc_lock);
            remove_pid(&worker_lock);
        }
        Commands::Gateway {
            model,
            model_url,
            tickets,
            port,
            ctx_size,
            parallel_slots,
            pod_name,
            wan_dht,
            local_cuda,
            ubatch_size,
            no_warmup,
            llama_bin_dir,
        } => {
            let catalog_entry = build_model_catalog_entry(&model)?;
            let mut worker_caps = Vec::new();
            let mut bridge_by_ep = std::collections::HashMap::new();
            let mut ticket_by_ep = std::collections::HashMap::new();

            if local_cuda {
                let mut local_caps =
                    detect_local_gpu_capabilities_with_budget("local-cuda0", "local-cuda0", 100);
                local_caps.measured_rtt_ms = 0.0;
                local_caps.direct_udp_reachable = true;
                info!(
                    "Probed Local CUDA0 ({}) -> Free VRAM: {} MB, Power: {}% ({:.1}W), Eff: {:.1} TFLOPS, {:.0} GB/s, RTT: 0.00 ms",
                    local_caps.gpu_name,
                    local_caps.vram_free_mb,
                    local_caps.user_power_budget_pct,
                    local_caps.allocated_power_budget_watts,
                    local_caps.effective_available_tflops,
                    local_caps.effective_available_bandwidth_gb_s
                );
                worker_caps.push(local_caps);
            }

            let mut ticket_order_bridges = Vec::new();
            for (parsed, caps) in probe_tickets_parallel(&tickets, wan_dht).await? {
                info!(
                    "Probed Worker {} ({}) -> Free VRAM: {} MB, Power: {}% ({:.1}W), Eff: {:.1} TFLOPS, {:.0} GB/s, RTT: {:.2} ms, Link: {:.1} MB/s",
                    caps.node_name,
                    caps.gpu_name,
                    caps.vram_free_mb,
                    caps.user_power_budget_pct,
                    caps.allocated_power_budget_watts,
                    caps.effective_available_tflops,
                    caps.effective_available_bandwidth_gb_s,
                    caps.measured_rtt_ms,
                    caps.measured_link_mb_s
                );
                let ep_id = caps.endpoint_id.clone();
                ticket_by_ep.insert(ep_id.clone(), (parsed.clone(), caps.node_name.clone()));
                worker_caps.push(caps);
                let (local_addr, _stats) = spawn_client_iroh_rpc_bridge(parsed, wan_dht).await?;
                ticket_order_bridges.push(local_addr.clone());
                bridge_by_ep.insert(ep_id, local_addr);
            }

            let swarm_plan = schedule_nodes_for_model(&catalog_entry, &worker_caps)?;
            info!(
                "Scheduled Swarm Tensor-Split: {} (Estimated Decode: {:.1} tok/s)",
                swarm_plan.llama_cpp_tensor_split_arg, swarm_plan.estimated_cluster_decode_tok_s
            );

            // Autonomous Worker Shard Pull: instruct remote Workers to download their weights directly from HuggingFace
            if let Some(remote_url) = resolve_remote_gguf_url(&model, model_url.as_deref()) {
                let total_l = catalog_entry.total_layers.max(1) as u16;
                let mut prewarm_targets = Vec::new();
                let remote_workers: Vec<_> = swarm_plan
                    .selected_workers
                    .iter()
                    .filter(|w| w.endpoint_id != "local-cuda0")
                    .collect();
                let n_rem = remote_workers.len().max(1);
                let mut cur_l = 0u16;
                for (i, w) in remote_workers.iter().enumerate() {
                    let count = if i + 1 == n_rem {
                        total_l.saturating_sub(cur_l).max(1)
                    } else {
                        ((total_l as f64 * w.tensor_split_ratio).round() as u16)
                            .max(1)
                            .min(total_l.saturating_sub(cur_l + 1))
                    };
                    let s_l = cur_l;
                    let e_l = (s_l + count - 1).min(total_l - 1);
                    cur_l = (e_l + 1).min(total_l - 1);
                    if let Some((tk, n_name)) = ticket_by_ep.get(&w.endpoint_id) {
                        prewarm_targets.push((tk.clone(), n_name.clone(), s_l, e_l));
                    }
                }
                if prewarm_targets.is_empty() {
                    for (tk, n_name) in ticket_by_ep.values() {
                        prewarm_targets.push((tk.clone(), n_name.clone(), 0, total_l - 1));
                    }
                }
                prewarm_workers_parallel(
                    &remote_url,
                    total_l,
                    catalog_entry.total_weights_mb,
                    wan_dht,
                    prewarm_targets,
                )
                .await;
            } else {
                warn!(
                    "Nessun URL remoto HuggingFace rilevato per {:?}: specifica --model-url <https://huggingface.co/.../*.gguf> \
                     affinché i worker scarichino autonomamente i pesi invece di riceverli dal Gateway",
                    model
                );
            }

            let mut ordered_bridges = Vec::new();
            let mut devices_list = Vec::new();
            let mut ordered_splits = Vec::new();

            if let Ok(explicit_split) = std::env::var("CUTMESH_TENSOR_SPLIT") {
                if local_cuda {
                    devices_list.push("CUDA0".to_string());
                }
                for (rpc_idx, addr) in ticket_order_bridges.iter().enumerate() {
                    ordered_bridges.push(addr.clone());
                    devices_list.push(format!("RPC{}", rpc_idx));
                }
                ordered_splits.push(explicit_split);
            } else {
                if local_cuda {
                    if let Some(loc_w) = swarm_plan
                        .selected_workers
                        .iter()
                        .find(|w| w.endpoint_id == "local-cuda0")
                    {
                        devices_list.push("CUDA0".to_string());
                        ordered_splits.push(format!("{:.0}", (loc_w.tensor_split_ratio * 100.0).round()));
                    }
                }

                let mut rpc_idx = 0usize;
                for w in &swarm_plan.selected_workers {
                    if w.endpoint_id == "local-cuda0" {
                        continue;
                    }
                    if let Some(addr) = bridge_by_ep.get(&w.endpoint_id) {
                        ordered_bridges.push(addr.clone());
                        devices_list.push(format!("RPC{}", rpc_idx));
                        ordered_splits.push(format!("{:.0}", (w.tensor_split_ratio * 100.0).round()));
                        rpc_idx += 1;
                    }
                }
                if ordered_bridges.is_empty() && !bridge_by_ep.is_empty() {
                    for addr in bridge_by_ep.into_values() {
                        ordered_bridges.push(addr);
                        devices_list.push(format!("RPC{}", rpc_idx));
                        rpc_idx += 1;
                    }
                }
            }

            let rpc_arg = ordered_bridges.join(",");
            let rpc_devices_arg = devices_list.join(",");
            let tensor_split_final = if !ordered_splits.is_empty() {
                ordered_splits.join(",")
            } else {
                swarm_plan.llama_cpp_tensor_split_arg.clone()
            };

            let effective_llama_dir = resolve_llama_bin_dir(&llama_bin_dir);
            let server_bin = if cfg!(target_os = "windows") {
                effective_llama_dir.join("llama-server.exe")
            } else {
                effective_llama_dir.join("llama-server")
            };
            let mut cmd_args = vec![
                "-m".to_string(),
                model.to_str().unwrap().to_string(),
            ];
            if !rpc_arg.is_empty() {
                cmd_args.push("--rpc".to_string());
                cmd_args.push(rpc_arg);
            }
            cmd_args.extend([
                "--device".to_string(),
                rpc_devices_arg,
                "-ngl".to_string(),
                "99".to_string(),
                "--tensor-split".to_string(),
                tensor_split_final,
                "-np".to_string(),
                parallel_slots.to_string(),
                "-cb".to_string(),
                "-c".to_string(),
                ctx_size.to_string(),
                "-ub".to_string(),
                ubatch_size.to_string(),
                "--port".to_string(),
                port.to_string(),
            ]);
            if let Ok(ot_expr) = std::env::var("CUTMESH_OVERRIDE_TENSOR") {
                for expr in ot_expr.split(';') {
                    if !expr.trim().is_empty() {
                        cmd_args.push("-ot".to_string());
                        cmd_args.push(expr.trim().to_string());
                    }
                }
            }
            if no_warmup {
                cmd_args.push("--no-warmup".to_string());
            }
            let llama_srv_raw = Command::new(server_bin)
                .env("PATH", cuda_path_env())
                .env("LD_LIBRARY_PATH", ld_library_path_env(&effective_llama_dir))
                .args(&cmd_args)
                .spawn()?;
            let _llama_guard = ChildGuard(Some(llama_srv_raw));
            wait_for_llama_server_ready(port).await?;

            let (pod_ticket, _pod_metrics) = spawn_prewarmed_swarm_pod_endpoint(
                &pod_name,
                &catalog_entry.model_name,
                port,
                parallel_slots,
                99,
            )
            .await?;

            info!(
                "OpenAI llama-server Gateway active on http://127.0.0.1:{}/v1/chat/completions | Pre-Warmed Pod Ticket: {}",
                port,
                pod_ticket.to_string_ticket()
            );
            tokio::signal::ctrl_c().await?;
        }
        Commands::ClientRouter {
            pods,
            mobile_relay_fallback,
        } => {
            let mut parsed_pods = Vec::new();
            for p in &pods {
                parsed_pods.push(SwarmPodTicket::parse(p)?);
            }
            let (router_port, statuses) =
                spawn_lightweight_swarm_client_router_with_mode(parsed_pods, mobile_relay_fallback)
                    .await?;
            info!(
                "Zero-GGUF Lightweight OpenAI Client Router listening on http://127.0.0.1:{}/v1/chat/completions across {} Pre-Warmed Pods (Mobile Relay Fallback: {})",
                router_port,
                statuses.len(),
                mobile_relay_fallback
            );
            tokio::signal::ctrl_c().await?;
        }
        Commands::Shield { model, input_json } => {
            let external_input = ExternalShieldInput::from_json_file(&input_json)?;
            let gpu_profile =
                detect_local_gpu_capabilities_with_budget("local-silicon-probe", "local", 100);
            let catalog_entry = build_model_catalog_entry(&model)?;
            let shield_audit = execute_laptop_micro_shield_with_input(&model, &external_input)?;
            let combined = serde_json::json!({
                "local_gpu_silicon_registers_nvcuda": gpu_profile,
                "catalog_laptop_micro_shield_spec": catalog_entry.laptop_cpu_micro_shield,
                "live_cpu_benchmark_and_non_reversibility_audit": shield_audit,
            });
            println!("{}", serde_json::to_string_pretty(&combined)?);
        }
        Commands::Govern {
            model,
            cluster_json,
            policy_json,
        } => {
            let catalog_entry = build_model_catalog_entry(&model)?;
            let cluster_input = ExternalClusterStateInput::from_json_file(&cluster_json)?;
            let policy = GovernancePolicyConfig::from_json_file(&policy_json)?;
            let gov_report = evaluate_external_cluster_governance(
                &model,
                &catalog_entry,
                &cluster_input,
                &policy,
            )?;
            println!("{}", serde_json::to_string_pretty(&gov_report)?);
        }
        Commands::PrewarmCache {
            model,
            cache_dir,
            start_layer,
            end_layer,
        } => {
            let (count, bytes) =
                populate_ggml_rpc_cache(&model, &cache_dir, start_layer, end_layer)?;
            info!(
                "Pre-warmed {} tensors ({:.2} MB) in {:?}",
                count,
                (bytes as f64) / (1024.0 * 1024.0),
                cache_dir
            );
        }
        Commands::PlanRig {
            model,
            workers_json,
            tickets,
            wan_dht,
            max_rtt_ms,
            vram_reserve_mb,
            min_nodes,
            include_local_cuda,
            power_pct,
        } => {
            let model_hdr_arg = model.clone();
            let (hdr, model_name) = tokio::task::spawn_blocking(move || {
                fetch_or_parse_gguf_header(&model_hdr_arg)
            })
            .await
            .map_err(|e| format!("header parse task panicked: {}", e))??;
            let mut candidates: Vec<GpuNodeCapabilities> = Vec::new();

            if include_local_cuda {
                candidates.push(detect_local_gpu_capabilities_with_budget(
                    "local-cuda0-node",
                    "local-cuda0",
                    power_pct,
                ));
            }
            if let Some(json_path) = workers_json {
                let raw = std::fs::read_to_string(&json_path)?;
                let loaded: Vec<GpuNodeCapabilities> = serde_json::from_str(&raw)?;
                candidates.extend(loaded);
            }
            for (_tk, caps) in probe_tickets_parallel(&tickets, wan_dht).await? {
                candidates.push(caps);
            }
            if candidates.is_empty() {
                candidates.push(detect_local_gpu_capabilities_with_budget(
                    "local-probed-gpu",
                    "local-endpoint",
                    power_pct,
                ));
            }

            let report = form_and_close_rigs_with_min_nodes(
                &model,
                &hdr,
                &model_name,
                &candidates,
                max_rtt_ms,
                vram_reserve_mb,
                min_nodes,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::PullRigShard {
            model,
            cache_dir,
            start_layer,
            end_layer,
            include_non_block,
        } => {
            let report = pull_rig_shard_to_rpc_cache(
                &model,
                &cache_dir,
                start_layer,
                end_layer,
                include_non_block,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::AutoRig {
            model,
            model_url,
            tickets,
            wan_dht,
            min_nodes,
            max_rtt_ms,
            vram_reserve_mb,
            include_local_cuda,
            power_pct,
            llama_bin_dir,
            port,
            ctx_size,
            ubatch_size,
        } => {
            let model_str = model.to_string_lossy().to_string();
            let model_hdr_arg = model_str.clone();
            let (hdr, model_name) = tokio::task::spawn_blocking(move || {
                fetch_or_parse_gguf_header(&model_hdr_arg)
            })
            .await
            .map_err(|e| format!("header parse task panicked: {}", e))??;

            let mut candidates: Vec<GpuNodeCapabilities> = Vec::new();
            if include_local_cuda || tickets.is_empty() {
                let mut local_gpu = detect_local_gpu_capabilities_with_budget(
                    "local-cuda0-master-candidate",
                    "local-cuda0",
                    power_pct,
                );
                local_gpu.measured_rtt_ms = 0.0;
                candidates.push(local_gpu);
            }

            let mut effective_tickets = tickets.clone();
            if effective_tickets.is_empty() {
                let discovered = SwarmDiscoveryRegistry::discover_healthy_tickets(
                    std::path::Path::new(DEFAULT_SWARM_REGISTRY_FILE),
                    30_000,
                );
                for lease in discovered {
                    info!(
                        "Zero-Copy-Paste Auto-Discovery: Found healthy Worker {} ({}) -> {}",
                        lease.node_name, lease.gpu_name, lease.iroh_ticket
                    );
                    effective_tickets.push(lease.iroh_ticket);
                }
            }

            let mut parsed_tickets = Vec::new();
            let mut ticket_by_ep = std::collections::HashMap::new();
            for (tk, caps) in probe_tickets_parallel(&effective_tickets, wan_dht).await? {
                info!(
                    "AutoRig Probed Peer {} ({}) -> Free VRAM: {} MB, Eff: {:.1} TFLOPS, {:.0} GB/s, RTT: {:.2} ms, Link: {:.1} MB/s",
                    caps.node_name,
                    caps.gpu_name,
                    caps.vram_free_mb,
                    caps.effective_available_tflops,
                    caps.effective_available_bandwidth_gb_s,
                    caps.measured_rtt_ms,
                    caps.measured_link_mb_s
                );
                ticket_by_ep.insert(caps.endpoint_id.clone(), (tk.clone(), caps.node_name.clone()));
                candidates.push(caps);
                parsed_tickets.push(tk);
            }

            let report = form_and_close_rigs_with_min_nodes(
                &model_str,
                &hdr,
                &model_name,
                &candidates,
                max_rtt_ms,
                vram_reserve_mb,
                min_nodes,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);

            let active_rig = report
                .rigs
                .first()
                .ok_or("No Rig formed by AutoRig")?;
            info!(
                "=== AUTO-RIG [{}] STATE: {:?} | ELECTED MASTER: {} | SPLIT: {} | DEVICES: {} ===",
                active_rig.rig_id,
                active_rig.state,
                active_rig.elected_master_node,
                active_rig.llama_cpp_tensor_split_arg,
                active_rig.llama_cpp_devices_arg
            );
            if active_rig.state != cutmesh::rig::RigLifecycleState::ClosedAndLocked {
                warn!(
                    "[RIG_PHASE:CLOSED_REJECTED] Impossibile avviare il Rig {}: {}",
                    active_rig.rig_id, active_rig.closing_reason
                );
                return Err(active_rig.closing_reason.clone().into());
            }

            let mut worker_reported_hashes = std::collections::BTreeMap::new();
            if let Some(remote_url) = resolve_remote_gguf_url(&model, model_url.as_deref()) {
                let mut prewarm_targets = Vec::new();
                let mut local_prewarm_handle = None;
                for m in &active_rig.members {
                    if m.endpoint_id == "local-cuda0" {
                        let url_c = remote_url.clone();
                        let s_l = m.start_layer;
                        let e_l = m.end_layer_inclusive;
                        let c_dir = default_ggml_rpc_cache_dir();
                        local_prewarm_handle = Some(tokio::task::spawn_blocking(move || {
                            pull_rig_shard_to_rpc_cache(&url_c, &c_dir, s_l, e_l, true)
                        }));
                        continue;
                    }
                    if let Some((tk, n_name)) = ticket_by_ep.get(&m.endpoint_id) {
                        prewarm_targets.push((
                            tk.clone(),
                            n_name.clone(),
                            m.start_layer,
                            m.end_layer_inclusive,
                        ));
                    }
                }
                worker_reported_hashes = prewarm_workers_parallel(
                    &remote_url,
                    hdr.total_layers.max(1),
                    report.total_weights_mb,
                    wan_dht,
                    prewarm_targets,
                )
                .await;
                if let Some(h) = local_prewarm_handle {
                    if let Ok(Ok(rep)) = h.await {
                        info!(
                            "Download autonomo CUDA0 locale completato: {} tensori ({:.2} MB, risparmiato {:.1}% banda) in {}",
                            rep.cached_tensor_count,
                            rep.downloaded_mb,
                            rep.bandwidth_saved_pct,
                            rep.cache_dir
                        );
                        worker_reported_hashes.extend(rep.tensor_hashes);
                    }
                }
            }

            let effective_ctx = if ctx_size == 0 {
                active_rig.auto_maximized_context_tokens
            } else {
                ctx_size.min(active_rig.auto_maximized_context_tokens)
            };

            // Master-on-GPU Delegation (Zero-Load VPS / 0 MB WAN Logit Traffic):
            // When the Elected Stage-0 Master is a remote GPU Worker (e.g. a 24GB GPU node), instruct that GPU
            // Worker over `ALPN_MASTER_RIG` to spawn `llama-server` directly on the Master GPU node
            // (so output.weight + sampling run locally on the Master GPU node and inter-GPU activations
            // flow P2P between workers without bouncing through the VPS), and tunnel `127.0.0.1:<port>`
            // on the VPS to the Master GPU's `llama-server` over `ALPN_OPENAI_HTTP`.
            if active_rig.elected_master_endpoint_id != "local-cuda0"
                && std::env::var("CUTMESH_MASTER_ON_VPS").as_deref() != Ok("1")
            {
                if let Some(remote_url) = resolve_remote_gguf_url(&model, model_url.as_deref()) {
                    if let Some((master_tk, master_name)) =
                        ticket_by_ep.get(&active_rig.elected_master_endpoint_id).cloned()
                    {
                        let fname = remote_url
                            .rsplit('/')
                            .next()
                            .unwrap_or("model.gguf")
                            .split('?')
                            .next()
                            .unwrap_or("model.gguf");
                        let _ = std::fs::create_dir_all("models");
                        let manifest_file = PathBuf::from("models").join(format!("{}.hashes.json", fname));
                        let _ = ensure_tensor_hash_manifest(
                            &remote_url,
                            &hdr,
                            &manifest_file,
                            &worker_reported_hashes,
                        );
                        if let Ok(raw_manifest) = std::fs::read_to_string(&manifest_file) {
                            if let Ok(full_map) =
                                serde_json::from_str::<std::collections::BTreeMap<String, u64>>(&raw_manifest)
                            {
                                worker_reported_hashes.extend(full_map);
                            }
                        }

                        let mut peer_tickets_str = Vec::new();
                        let mut split_ratios_u32 = Vec::new();
                        for (idx, m) in active_rig.members.iter().enumerate() {
                            split_ratios_u32.push(m.tensor_split_ratio);
                            if idx > 0 {
                                if let Some((peer_tk, _)) = ticket_by_ep.get(&m.endpoint_id) {
                                    peer_tickets_str.push(peer_tk.to_string_ticket());
                                }
                            }
                        }

                        info!(
                            "[RIG_PHASE:STARTING_LLAMA] Avvio Master-on-GPU llama-server direttamente su {} (Zero-Load VPS, 0 MB WAN logits, peer P2P: {}, split: {:?}, -c {} tokens)...",
                            master_name,
                            peer_tickets_str.len(),
                            split_ratios_u32,
                            effective_ctx
                        );

                        let resolved_policy = cutmesh::models::ModelPolicy::resolve(&remote_url);
                        let spawn_req = SpawnMasterRigRequest {
                            model_url: remote_url.clone(),
                            peer_worker_tickets: peer_tickets_str,
                            tensor_split_ratios: split_ratios_u32,
                            ctx_size: effective_ctx,
                            ubatch_size,
                            port,
                            worker_reported_hashes: worker_reported_hashes.clone(),
                            policy_json: serde_json::to_string(&resolved_policy).ok(),
                        };

                        match request_worker_spawn_master_rig(&master_tk, wan_dht, &spawn_req).await {
                            Ok(resp) => {
                                let _ = spawn_client_iroh_http_bridge(master_tk.clone(), port, wan_dht).await?;
                                wait_for_llama_server_ready(port).await?;
                                info!(
                                    "[RIG_PHASE:READY] AutoRig Master-on-GPU Gateway READY (200 OK) on http://127.0.0.1:{}/v1/chat/completions ({})",
                                    port, resp.message
                                );
                                shutdown_signal().await;
                                return Ok(());
                            }
                            Err(e) => {
                                if !include_local_cuda {
                                    warn!(
                                        "[RIG_PHASE:ERROR] Delega Master-on-GPU su {} non riuscita dopo i tentativi di ripristino: {}",
                                        master_name, e
                                    );
                                    return Err(format!(
                                        "Delega Master-on-GPU su {} non riuscita: {}",
                                        master_name, e
                                    )
                                    .into());
                                }
                                warn!(
                                    "Delega Master-on-GPU su {} non riuscita ({}), fallback su coordinatore locale...",
                                    master_name, e
                                );
                            }
                        }
                    }
                }
            }

            // Spawn Iroh Direct-UDP bridges for all remote workers in the Rig
            let mut bridge_by_ep = std::collections::HashMap::new();
            for tk in parsed_tickets {
                let ep_id = tk.endpoint_addr.id.to_string();
                let (addr, _stats) = spawn_client_iroh_rpc_bridge(tk, wan_dht).await?;
                bridge_by_ep.insert(ep_id, addr);
            }

            let mut ordered_bridges = Vec::new();
            let mut devices_vec = Vec::new();
            let mut splits_vec = Vec::new();
            let mut rpc_i = 0usize;
            for m in &active_rig.members {
                if m.endpoint_id == "local-cuda0" {
                    devices_vec.push("CUDA0".to_string());
                    splits_vec.push(m.tensor_split_ratio.to_string());
                } else if let Some(addr) = bridge_by_ep.get(&m.endpoint_id) {
                    ordered_bridges.push(addr.clone());
                    devices_vec.push(format!("RPC{}", rpc_i));
                    splits_vec.push(m.tensor_split_ratio.to_string());
                    rpc_i += 1;
                }
            }
            if devices_vec.is_empty() {
                devices_vec.push("CUDA0".to_string());
                splits_vec.push("100".to_string());
            }

            let effective_llama_dir = resolve_llama_bin_dir(&llama_bin_dir);
            let server_bin = if cfg!(target_os = "windows") {
                effective_llama_dir.join("llama-server.exe")
            } else {
                effective_llama_dir.join("llama-server")
            };

            let resolved_model_path = if let Some(remote_url) = resolve_remote_gguf_url(&model, model_url.as_deref()) {
                let fname = remote_url
                    .rsplit('/')
                    .next()
                    .unwrap_or("model.gguf")
                    .split('?')
                    .next()
                    .unwrap_or("model.gguf");
                let _ = std::fs::create_dir_all("models");
                let local_cache_file = PathBuf::from("models").join(fname);
                let manifest_file = PathBuf::from("models").join(format!("{}.hashes.json", fname));

                info!(
                    "[RIG_PHASE:PREWARMING] Allineamento manifesto hash FNV-1a e costruzione Sparse GGUF Master (Zero-Layer-Download) in {:?}...",
                    local_cache_file
                );
                let translations = ensure_tensor_hash_manifest(
                    &remote_url,
                    &hdr,
                    &manifest_file,
                    &worker_reported_hashes,
                )?;
                register_sparse_hash_translations(translations);

                let dl_bytes = build_sparse_master_gguf(&remote_url, &hdr, &local_cache_file)?;
                info!(
                    "[RIG_PHASE:PREWARMING] Sparse GGUF Master pronto in {:?} (scaricati solo {:.1} MB di header/embedding, 0 MB di layer GPU)",
                    local_cache_file,
                    (dl_bytes as f64) / (1024.0 * 1024.0)
                );
                local_cache_file
            } else {
                PathBuf::from(&model_str)
            };
            let effective_ctx = if ctx_size == 0 {
                active_rig.auto_maximized_context_tokens
            } else {
                ctx_size.min(active_rig.auto_maximized_context_tokens)
            };
            let primary_gpu_buft = if devices_vec.first().map(|s| s.as_str()) == Some("CUDA0") {
                "CUDA0".to_string()
            } else if let Some(first_bridge) = ordered_bridges.first() {
                format!("RPC0[{}]", first_bridge)
            } else {
                "CPU".to_string()
            };
            let model_policy = cutmesh::models::ModelPolicy::resolve(&model_str);
            let ot_arg = model_policy.build_rpc0_tensor_override(&primary_gpu_buft);
            info!(
                "[RIG_PHASE:STARTING_LLAMA] Avvio llama-server su porta {} (policy='{}', --device {} --tensor-split {} -ot {} -c {} tokens, KV = {} MiB)...",
                port,
                model_policy.model_id,
                devices_vec.join(","),
                splits_vec.join(","),
                ot_arg,
                effective_ctx,
                active_rig.allocated_kv_cache_mb
            );

            let cmd_args = model_policy.build_llama_server_args(
                &cutmesh::models::LlamaServerLaunchParams {
                    model_path: &resolved_model_path,
                    ordered_bridges: &ordered_bridges,
                    devices_vec: &devices_vec,
                    splits_vec: &splits_vec,
                    primary_buft: &primary_gpu_buft,
                    requested_ctx_size: effective_ctx,
                    requested_ubatch_size: ubatch_size,
                    port,
                },
            );

            let llama_srv_raw = Command::new(server_bin)
                .env("PATH", cuda_path_env())
                .env("LD_LIBRARY_PATH", ld_library_path_env(&effective_llama_dir))
                .args(&cmd_args)
                .spawn()?;
            let _llama_guard = ChildGuard(Some(llama_srv_raw));
            wait_for_llama_server_ready(port).await?;
            info!(
                "[RIG_PHASE:READY] AutoRig Master Gateway READY (200 OK) on http://127.0.0.1:{}/v1/chat/completions",
                port
            );
            tokio::signal::ctrl_c().await?;
        }
        Commands::CompileCovariantGguf {
            input,
            output,
            secret_seed,
            kappa_lambda,
        } => {
            let report = compile_covariant_gguf(&input, &output, &secret_seed, kappa_lambda)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Discover { registry_path } => {
            let reg = SwarmDiscoveryRegistry::load_or_default(&registry_path);
            let healthy = SwarmDiscoveryRegistry::discover_healthy_tickets(&registry_path, 30_000);
            let out = serde_json::json!({
                "registry_file": registry_path,
                "total_registered_workers": reg.active_workers.len(),
                "healthy_auto_discoverable_workers": healthy.len(),
                "workers": reg.active_workers,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Commands::AuditCovariant {
            model,
            dim,
            vocab_size,
            secret_seed,
        } => {
            let mut seed_bytes = [0u8; 32];
            let raw = secret_seed.as_bytes();
            for (i, byte) in seed_bytes.iter_mut().enumerate() {
                *byte = raw.get(i % raw.len()).copied().unwrap_or(0x42) ^ ((i as u8).wrapping_mul(31));
            }
            let report = cutmesh::shield::run_zero_weight_covariant_protocol_audit(
                &model,
                dim,
                vocab_size,
                &seed_bytes,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }

    Ok(())
}

