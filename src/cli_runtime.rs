use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use cutmesh::catalog::GpuNodeCapabilities;
use cutmesh::gguf::slicer::default_ggml_rpc_cache_dir;
use cutmesh::rig::pull_rig_shard_to_rpc_cache;
use cutmesh::rpc_tunnel::{
    probe_worker_hardware_iroh, request_worker_prewarm_shard, PrewarmShardRequest, RpcNodeTicket,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{info, warn};

/// RAII guard ensuring child `ggml-rpc-server` / `llama-server` processes are terminated on drop.
pub struct ChildGuard(pub Option<std::process::Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn cuda_path_env() -> String {
    let current = std::env::var("PATH").unwrap_or_default();
    if cfg!(target_os = "windows") {
        if let Ok(cuda_root) = std::env::var("CUDA_PATH") {
            return format!("{}\\bin;{}", cuda_root, current);
        }
        let cuda_base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
        if let Ok(entries) = std::fs::read_dir(&cuda_base) {
            let mut versions: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path().join("bin"))
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            if let Some(latest_bin) = versions.pop() {
                return format!("{};{}", latest_bin.to_string_lossy(), current);
            }
        }
        current
    } else {
        format!("/usr/local/cuda/bin:{}", current)
    }
}

pub fn resolve_llama_bin_dir(llama_bin_dir: &Path) -> PathBuf {
    if !cfg!(target_os = "windows") && !llama_bin_dir.exists() {
        let opt = PathBuf::from("/opt/cutmesh/llamacpp");
        if opt.exists() {
            return opt;
        }
    }
    llama_bin_dir.to_path_buf()
}

pub fn ld_library_path_env(llama_bin_dir: &Path) -> String {
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let filtered: Vec<&str> = current
        .split(':')
        .filter(|p| !p.is_empty() && !p.contains("/compat"))
        .collect();
    let dir_str = llama_bin_dir.to_string_lossy();
    if filtered.is_empty() {
        format!("{}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64", dir_str)
    } else {
        format!(
            "{}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:{}",
            dir_str,
            filtered.join(":")
        )
    }
}

pub fn spawn_ggml_rpc_process(
    llama_bin_dir: &Path,
    port: u16,
    device: &str,
) -> Result<std::process::Child, String> {
    let rpc_bin = if cfg!(target_os = "windows") {
        llama_bin_dir.join("ggml-rpc-server.exe")
    } else {
        let candidate = llama_bin_dir.join("ggml-rpc-server");
        if candidate.exists() {
            candidate
        } else {
            llama_bin_dir.join("rpc-server")
        }
    };
    let log_path = std::env::temp_dir().join("ggml-rpc-server.log");
    let stdout_cfg = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());
    let stderr_cfg = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());

    Command::new(&rpc_bin)
        .env("PATH", cuda_path_env())
        .env("LD_LIBRARY_PATH", ld_library_path_env(llama_bin_dir))
        .args([
            "-H",
            "127.0.0.1",
            "-p",
            &port.to_string(),
            "-d",
            device,
            "-c",
        ])
        .stdout(stdout_cfg)
        .stderr(stderr_cfg)
        .spawn()
        .map_err(|e| format!("Failed starting {:?} on port {}: {}", rpc_bin, port, e))
}

pub async fn wait_for_llama_server_ready(port: u16) -> Result<(), String> {
    let addr = format!("127.0.0.1:{}", port);
    for _ in 0..1800 {
        if let Ok(mut stream) = TcpStream::connect(&addr).await {
            let req = format!(
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
                port
            );
            if stream.write_all(req.as_bytes()).await.is_ok() {
                let mut buf = vec![0u8; 1024];
                if let Ok(n) = stream.read(&mut buf).await {
                    let s = String::from_utf8_lossy(&buf[..n]);
                    if s.contains("200 OK") && s.contains("\"ok\"") {
                        return Ok(());
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err("Timeout waiting for llama-server /health to become ready".into())
}

/// Resolves on Ctrl-C (SIGINT) or, on Unix, on SIGTERM (what `docker stop` sends).
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Probes all workers **in parallel** over direct QUIC paths, with 3 attempts each.
pub async fn probe_tickets_parallel(
    raw_tickets: &[String],
    wan_dht: bool,
) -> Result<Vec<(RpcNodeTicket, GpuNodeCapabilities)>, String> {
    let mut handles = Vec::new();
    for (idx, raw) in raw_tickets.iter().enumerate() {
        let tk =
            RpcNodeTicket::parse(raw).map_err(|e| format!("Ticket #{} non valido: {}", idx, e))?;
        handles.push(tokio::spawn(async move {
            let mut last_err = String::new();
            for attempt in 1..=3u64 {
                match probe_worker_hardware_iroh(&tk, wan_dht).await {
                    Ok(caps) => return Ok((tk, caps)),
                    Err(e) => {
                        warn!("Worker #{} tentativo {}/3 fallito: {}", idx, attempt, e);
                        last_err = e;
                        if attempt < 3 {
                            tokio::time::sleep(Duration::from_secs(attempt * 2)).await;
                        }
                    }
                }
            }
            Err(format!(
                "Worker #{} non raggiungibile in diretta: {}",
                idx, last_err
            ))
        }));
    }

    let mut ok = Vec::new();
    let mut errors = Vec::new();
    for h in handles {
        match h.await {
            Ok(Ok(v)) => ok.push(v),
            Ok(Err(e)) => errors.push(e),
            Err(e) => errors.push(format!("probe task panicked: {}", e)),
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    Ok(ok)
}

/// Triggered automatically when the Control Plane assigns a Worker to a Rig (`LOCKED_IN_RIG`).
pub fn maybe_trigger_worker_shard_prewarm(
    worker_name: &str,
    ack: &cutmesh::discovery::ControlPlaneHeartbeatResponse,
    last_prewarmed_shard: &mut Option<(String, u16, u16)>,
) {
    if ack.assignment_state != "LOCKED_IN_RIG" {
        return;
    }
    let Some(ref model_url) = ack.assigned_model_url else {
        return;
    };
    let trimmed = model_url.trim();
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return;
    }
    let s_l = ack.assigned_layer_start.unwrap_or(0);
    let e_l = ack.assigned_layer_end.unwrap_or(64).max(s_l);
    let key = (trimmed.to_string(), s_l, e_l);
    if last_prewarmed_shard.as_ref() == Some(&key) {
        return;
    }
    *last_prewarmed_shard = Some(key);

    let url_owned = trimmed.to_string();
    let name_owned = worker_name.to_string();
    let cache_dir = default_ggml_rpc_cache_dir();
    info!(
        worker = %name_owned,
        "Control Plane ha assegnato il Rig {:?}: avvio automatico download shard da HuggingFace ({}) layer [{}..={}] in {:?}",
        ack.assigned_rig_id,
        url_owned,
        s_l,
        e_l,
        cache_dir
    );
    tokio::task::spawn_blocking(move || {
        match pull_rig_shard_to_rpc_cache(&url_owned, &cache_dir, s_l, e_l, true) {
            Ok(rep) => info!(
                worker = %name_owned,
                "Download automatico shard da Control Plane completato: {} tensori ({:.2} MB, risparmiato {:.1}% banda) in {}",
                rep.cached_tensor_count,
                rep.downloaded_mb,
                rep.bandwidth_saved_pct,
                rep.cache_dir
            ),
            Err(e) => warn!(
                worker = %name_owned,
                "Download automatico shard da Control Plane non riuscito: {}",
                e
            ),
        }
    });
}

/// Instructs all selected remote Workers in parallel over QUIC (`ALPN_PREWARM_SHARD`)
/// to download their assigned GGUF layer slices into their local tensor cache.
pub async fn prewarm_workers_parallel(
    model_url: &str,
    total_layers: u16,
    total_weights_mb: f64,
    wan_dht: bool,
    targets: Vec<(RpcNodeTicket, String, u16, u16)>,
) -> std::collections::BTreeMap<String, u64> {
    let mut merged_hashes = std::collections::BTreeMap::new();
    if targets.is_empty() {
        return merged_hashes;
    }
    let full_prewarm = total_weights_mb <= 1024.0
        || std::env::var("CUTMESH_PREWARM_FULL").as_deref() == Ok("1");
    let max_layer = total_layers.saturating_sub(1);

    info!(
        "Avvio download autonomo dei pesi da HuggingFace su {} worker in parallelo ({})...",
        targets.len(),
        model_url
    );

    let mut handles = Vec::new();
    for (tk, node_name, s_l, e_l) in targets {
        let (eff_start, eff_end) = if full_prewarm {
            (0, max_layer)
        } else {
            (s_l.saturating_sub(2), (e_l + 2).min(max_layer))
        };
        let req = PrewarmShardRequest {
            model_url: model_url.to_string(),
            start_layer: eff_start,
            end_layer_inclusive: eff_end,
            include_non_block_tensors: true,
        };
        handles.push(tokio::spawn(async move {
            let res = request_worker_prewarm_shard(&tk, wan_dht, &req).await;
            (node_name, req, res)
        }));
    }

    for h in handles {
        if let Ok((node_name, req, res)) = h.await {
            match res {
                Ok(rep) => {
                    info!(
                        worker = %node_name,
                        "Download autonomo completato: {} tensori ({:.2} MB, layer {}..={}, risparmiato {:.1}% banda) in {}",
                        rep.cached_tensor_count,
                        rep.downloaded_mb,
                        req.start_layer,
                        req.end_layer_inclusive,
                        rep.bandwidth_saved_pct,
                        rep.cache_dir
                    );
                    merged_hashes.extend(rep.tensor_hashes);
                }
                Err(e) => {
                    warn!(
                        worker = %node_name,
                        "Pre-warm autonomo non riuscito ({}). Verifica che l'immagine del worker sia aggiornata.",
                        e
                    );
                }
            }
        }
    }
    merged_hashes
}
