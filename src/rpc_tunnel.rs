use crate::catalog::{detect_local_gpu_capabilities_with_budget, GpuNodeCapabilities};
use crate::gguf::slicer::default_ggml_rpc_cache_dir;
use crate::netcfg::{detect_public_addrs, load_or_create_node_key};
use crate::rig::{
    build_sparse_master_gguf_for_local_layers, ensure_tensor_hash_manifest, fetch_or_parse_gguf_header,
    pull_rig_shard_to_rpc_cache, PartialShardPullReport,
};
use crate::wire_fp16::{
    wire_fp16_enabled, ClientGgmlFramer, DecodedReqTracker, ExpectedRsp, WireStreamDecoder,
    WorkerRspFramer, RPC_STREAM_MAGIC_V2,
};
use iroh::{
    endpoint::{presets, BindOpts, Connection, QuicTransportConfig},
    Endpoint, EndpointAddr, RelayMode, RelayUrl,
};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info, warn};

/// ALPN for querying a Worker's real GPU VRAM, FP16 TFLOPS, Bandwidth GB/s, and Direct-QUIC status.
pub const ALPN_HW_PROBE: &[u8] = b"cutmesh/hw-probe/1";

/// ALPN for multiplexing `llama.cpp` (`ggml-rpc`) tensor streams over native Iroh QUIC + TLS 1.3.
pub const ALPN_GGML_RPC: &[u8] = b"cutmesh/ggml-rpc/1";

/// ALPN for measuring real upload throughput Master -> Worker over the direct path.
pub const ALPN_LINK_PROBE: &[u8] = b"cutmesh/link-probe/1";

/// ALPN for instructing a Worker to pull its assigned GGUF layer shard directly from Hugging Face
/// into its local `ggml-rpc-server -c` tensor cache (`~/.cache/llama.cpp/rpc/<fnv1a_hash>`),
/// so zero bytes of model weights are uploaded from the Gateway.
pub const ALPN_PREWARM_SHARD: &[u8] = b"cutmesh/prewarm-shard/1";

/// ALPN for instructing the Elected Stage-0 Master GPU Worker to launch `llama-server` locally
/// on the GPU node itself (so logits and sampling stay 100% on the Master GPU node with 0 MB WAN logit traffic)
/// and connect directly P2P to the remaining Stage 1..N Worker GPUs.
pub const ALPN_MASTER_RIG: &[u8] = b"cutmesh/master-rig/1";

/// ALPN for tunneling OpenAI HTTP requests (`/v1/chat/completions`, `/health`, `/props`) from the
/// lightweight Control Plane Gateway to the `llama-server` running locally on the Elected Master GPU Worker.
pub const ALPN_OPENAI_HTTP: &[u8] = b"cutmesh/openai-http/1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrewarmShardRequest {
    pub model_url: String,
    pub start_layer: u16,
    pub end_layer_inclusive: u16,
    pub include_non_block_tensors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnMasterRigRequest {
    pub model_url: String,
    pub peer_worker_tickets: Vec<String>,
    pub tensor_split_ratios: Vec<u32>,
    pub ctx_size: u32,
    pub ubatch_size: u32,
    pub port: u16,
    pub worker_reported_hashes: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub policy_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnMasterRigResponse {
    pub ok: bool,
    pub message: String,
    pub listen_port: u16,
}

#[derive(Clone)]
struct MasterLlamaLaunchSpec {
    server_bin: PathBuf,
    ld_env: String,
    manifest_file: String,
    cmd_args: Vec<String>,
    port: u16,
    tensor_split_ratios: Vec<u32>,
    peer_count: usize,
}

fn master_spawn_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn active_master_http_port() -> &'static AtomicU16 {
    static PORT: OnceLock<AtomicU16> = OnceLock::new();
    PORT.get_or_init(|| AtomicU16::new(8089))
}

fn active_worker_device_slot() -> &'static Mutex<String> {
    static DEV: OnceLock<Mutex<String>> = OnceLock::new();
    DEV.get_or_init(|| Mutex::new("CUDA0".to_string()))
}

pub fn set_active_worker_device(device: &str) {
    if let Ok(mut g) = active_worker_device_slot().lock() {
        *g = if device.trim().is_empty() {
            "CUDA0".to_string()
        } else {
            device.trim().to_string()
        };
    }
}

fn active_worker_device() -> String {
    active_worker_device_slot()
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|_| "CUDA0".to_string())
}

fn active_master_llama_child() -> &'static Mutex<Option<(std::process::Child, Instant)>> {
    static CHILD: OnceLock<Mutex<Option<(std::process::Child, Instant)>>> = OnceLock::new();
    CHILD.get_or_init(|| Mutex::new(None))
}

fn active_master_llama_spec() -> &'static Mutex<Option<MasterLlamaLaunchSpec>> {
    static SPEC: OnceLock<Mutex<Option<MasterLlamaLaunchSpec>>> = OnceLock::new();
    SPEC.get_or_init(|| Mutex::new(None))
}

pub fn active_worker_rpc_sessions() -> &'static AtomicU64 {
    static COUNT: OnceLock<AtomicU64> = OnceLock::new();
    COUNT.get_or_init(|| AtomicU64::new(0))
}

fn worker_needs_rpc_recycle_flag() -> &'static std::sync::atomic::AtomicBool {
    static FLAG: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| std::sync::atomic::AtomicBool::new(false))
}

pub fn request_worker_rpc_recycle() {
    worker_needs_rpc_recycle_flag().store(true, Ordering::SeqCst);
}

/// Returns `true` and clears the recycle request when the Rig has been dissolved on the Control Plane
/// and no active local Master `llama-server` process is running, clearing any parked resume sessions.
pub fn take_worker_rpc_recycle_request() -> bool {
    if !worker_needs_rpc_recycle_flag().swap(false, Ordering::SeqCst) {
        return false;
    }
    if let Ok(guard) = active_master_llama_child().lock() {
        if guard.is_some() {
            return false;
        }
    }
    if let Ok(mut map) = worker_rpc_sessions().try_lock() {
        for (_, sess) in map.drain() {
            sess.generation.fetch_add(1, Ordering::SeqCst);
            sess.preempt_notify.notify_waiters();
        }
    }
    active_worker_rpc_sessions().store(0, Ordering::SeqCst);
    true
}

fn spawn_master_llama_process(spec: &MasterLlamaLaunchSpec) -> Result<std::process::Child, String> {
    let log_path = std::env::temp_dir().join("llama-server-master.log");
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

    Command::new(&spec.server_bin)
        .env("LD_LIBRARY_PATH", &spec.ld_env)
        .env("CUTMESH_HASH_MANIFEST", &spec.manifest_file)
        .args(&spec.cmd_args)
        .stdout(stdout_cfg)
        .stderr(stderr_cfg)
        .spawn()
        .map_err(|e| format!("Failed spawning Master-on-GPU llama-server {:?}: {}", spec.server_bin, e))
}

/// Supervises the locally-running Master-on-GPU `llama-server` process. If it exits unexpectedly while
/// the Rig is still active on the Control Plane, automatically respawns it with the same configuration.
pub fn supervise_active_master_llama_server() {
    let spec_opt = active_master_llama_spec().lock().ok().and_then(|g| g.clone());
    let Some(spec) = spec_opt else {
        return;
    };
    if let Ok(mut guard) = active_master_llama_child().lock() {
        let exited = match guard.as_mut() {
            Some((child, _)) => match child.try_wait() {
                Ok(Some(status)) => Some(format!("{}", status)),
                Ok(None) => None,
                Err(e) => Some(format!("wait error: {}", e)),
            },
            None => Some("not running".to_string()),
        };
        if let Some(reason) = exited {
            warn!(
                "SELF-HEALING MASTER WATCHDOG: Master-on-GPU llama-server exited unexpectedly ({}) -> Respawning on 127.0.0.1:{}...",
                reason, spec.port
            );
            match spawn_master_llama_process(&spec) {
                Ok(new_child) => {
                    *guard = Some((new_child, Instant::now()));
                }
                Err(e) => {
                    warn!("SELF-HEALING MASTER WATCHDOG: respawn failed: {}", e);
                }
            }
        }
    }
}

/// Terminates any locally-running Master-on-GPU `llama-server` child process when the Control Plane
/// reports that this Worker is no longer assigned to any Rig (`IDLE_STANDBY_POOL`), immediately freeing
/// VRAM on both this Master GPU (`RPC0`) and all connected secondary Worker GPUs (`RPC1..N`).
pub fn stop_active_master_llama_server_if_idle() -> bool {
    if let Ok(mut guard) = active_master_llama_child().lock() {
        if let Some((_, spawned_at)) = guard.as_ref() {
            if spawned_at.elapsed() < Duration::from_secs(20) {
                return false;
            }
        }
        if let Ok(mut spec_guard) = active_master_llama_spec().lock() {
            *spec_guard = None;
        }
        if let Some((mut old_child, _)) = guard.take() {
            info!("Rig disciolto sul Control Plane -> Arresto Master-on-GPU llama-server locale e rilascio immediato VRAM su tutti i nodi RPC...");
            let _ = old_child.kill();
            let _ = old_child.wait();
            request_worker_rpc_recycle();
            return true;
        }
    }
    false
}

/// Strips Docker-internal bridge IPs (`172.17.0.0/16`..`172.31.255.255`) and loopback IPs when
/// a ticket also carries public IP addresses or a signaling relay, preventing two Docker containers
/// on separate WAN machines from dialing their own local `172.17.0.2:40000` (`UnknownIssuer`).
pub fn sanitize_endpoint_addr(addr: EndpointAddr) -> EndpointAddr {
    let has_relay = addr.relay_urls().next().is_some();
    let kept_ips: Vec<SocketAddr> = addr
        .ip_addrs()
        .copied()
        .filter(|sa| match sa.ip() {
            IpAddr::V4(v4) => {
                if v4.is_loopback() || v4.is_unspecified() {
                    return false;
                }
                let oct = v4.octets();
                let is_docker_bridge = oct[0] == 172 && (17..=31).contains(&oct[1]);
                !is_docker_bridge
            }
            IpAddr::V6(v6) => !v6.is_loopback() && !v6.is_unspecified(),
        })
        .collect();
    if kept_ips.is_empty() && !has_relay {
        return addr;
    }
    let mut clean = EndpointAddr::new(addr.id);
    for r in addr.relay_urls() {
        clean = clean.with_relay_url(r.clone());
    }
    for ip in kept_ips {
        clean = clean.with_ip_addr(ip);
    }
    clean
}

const CHUNK_SIZE: usize = 64 * 1024;
const RPC_SESSION_HISTORY_CAP: usize = 32 * 1024 * 1024;
const RPC_STREAM_MAGIC: &[u8; 8] = b"SYNRPC01";

struct RollingHistory {
    base_offset: u64,
    buf: std::collections::VecDeque<u8>,
    max_cap: usize,
}

impl RollingHistory {
    fn new(max_cap: usize) -> Self {
        Self {
            base_offset: 0,
            buf: std::collections::VecDeque::new(),
            max_cap,
        }
    }

    fn total_len(&self) -> u64 {
        self.base_offset + self.buf.len() as u64
    }

    fn append(&mut self, data: &[u8]) {
        if data.len() >= self.max_cap {
            let keep_start = data.len() - self.max_cap;
            self.base_offset += (self.buf.len() + keep_start) as u64;
            self.buf.clear();
            self.buf.extend(&data[keep_start..]);
            return;
        }
        self.buf.extend(data);
        if self.buf.len() > self.max_cap {
            let drop_n = self.buf.len() - self.max_cap;
            self.buf.drain(..drop_n);
            self.base_offset += drop_n as u64;
        }
    }

    fn slice_from(&self, offset: u64) -> Option<Vec<u8>> {
        let total = self.total_len();
        if offset > total || offset < self.base_offset {
            return None;
        }
        let idx = (offset - self.base_offset) as usize;
        let mut out = Vec::with_capacity(self.buf.len() - idx);
        let (front, back) = self.buf.as_slices();
        if idx < front.len() {
            out.extend_from_slice(&front[idx..]);
            out.extend_from_slice(back);
        } else {
            out.extend_from_slice(&back[idx - front.len()..]);
        }
        Some(out)
    }
}

struct WorkerRpcSessionState {
    rpc_r: tokio::net::tcp::OwnedReadHalf,
    rpc_w: tokio::net::tcp::OwnedWriteHalf,
    worker_rx_offset: u64,
    worker_tx_history: RollingHistory,
    wire_v2: bool,
    c2w_decoder: WireStreamDecoder,
    req_tracker: DecodedReqTracker,
    expected_rsps: std::collections::VecDeque<ExpectedRsp>,
    w2c_framer: WorkerRspFramer,
    logged_v2_rx: bool,
}

struct WorkerRpcSession {
    generation: AtomicU64,
    preempt_notify: tokio::sync::Notify,
    state: tokio::sync::Mutex<WorkerRpcSessionState>,
}

fn worker_rpc_sessions() -> &'static tokio::sync::Mutex<std::collections::HashMap<u64, Arc<WorkerRpcSession>>> {
    static MAP: OnceLock<tokio::sync::Mutex<std::collections::HashMap<u64, Arc<WorkerRpcSession>>>> = OnceLock::new();
    MAP.get_or_init(|| tokio::sync::Mutex::new(std::collections::HashMap::new()))
}

fn next_rpc_session_id() -> u64 {
    static SEQ: OnceLock<AtomicU64> = OnceLock::new();
    let base = SEQ.get_or_init(|| {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1000);
        AtomicU64::new(seed)
    });
    base.fetch_add(1, Ordering::Relaxed)
}

async fn write_quic_chunked(
    send: &mut iroh::endpoint::SendStream,
    data: &[u8],
) -> Result<(), String> {
    for chunk in data.chunks(CHUNK_SIZE) {
        match tokio::time::timeout(Duration::from_secs(12), send.write_all(chunk)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("write_all error: {}", e)),
            Err(_) => return Err("write_all timeout (12s)".to_string()),
        }
    }
    Ok(())
}

/// Max time to establish the QUIC connection (direct or via signaling relay).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Default max time to wait for hole punching to produce a direct path
/// (override with `CUTMESH_DIRECT_PATH_TIMEOUT_SECS`).
const DEFAULT_DIRECT_PATH_TIMEOUT_SECS: u64 = 20;

/// Grace period: if an established bridge temporarily loses direct UDP acks due to upload bufferbloat
/// while another worker is being loaded, wait up to 120s for Iroh to re-validate the direct UDP path
/// before closing (override with `CUTMESH_RELAY_GRACE_SECS`, disable with `CUTMESH_STRICT_DIRECT=0`).
const DEFAULT_RELAY_FALLBACK_GRACE_SECS: u64 = 120;

/// Default payload for the throughput probe (override with `CUTMESH_LINK_PROBE_MB`, `0` disables).
const DEFAULT_LINK_PROBE_MB: usize = 4;

/// Live cryptographic & QUIC stream audit counters for an Iroh P2P worker node.
#[derive(Debug, Default)]
pub struct TunnelAuditStats {
    pub quic_connections: AtomicU64,
    pub quic_bi_streams_opened: AtomicU64,
    pub hw_probes_answered: AtomicU64,
    pub bytes_tx: AtomicU64,
    pub bytes_rx: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TunnelAuditSnapshot {
    pub node_name: String,
    pub endpoint_id: String,
    pub ticket: String,
    pub local_rpc_backend: String,
    pub transport_protocol: String,
    pub quic_connections: u64,
    pub quic_bi_streams_opened: u64,
    pub hw_probes_answered: u64,
    pub bytes_tx: u64,
    pub bytes_rx: u64,
}

/// Shareable Iroh P2P ticket (`iroh://<hex_encoded_endpoint_addr>`) containing
/// the node's cryptographic `EndpointId` (Ed25519) and direct UDP socket addresses.
#[derive(Debug, Clone)]
pub struct RpcNodeTicket {
    pub endpoint_addr: EndpointAddr,
}

impl RpcNodeTicket {
    pub fn to_string_ticket(&self) -> String {
        let json = serde_json::to_vec(&self.endpoint_addr).unwrap_or_default();
        format!("iroh://{}", hex::encode(json))
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let hex_part = s
            .strip_prefix("iroh://")
            .ok_or_else(|| "Ticket must start with iroh://".to_string())?;
        let bytes = hex::decode(hex_part).map_err(|e| format!("Invalid ticket hex: {}", e))?;
        let raw_addr: EndpointAddr =
            serde_json::from_slice(&bytes).map_err(|e| format!("Invalid EndpointAddr: {}", e))?;
        Ok(Self {
            endpoint_addr: sanitize_endpoint_addr(raw_addr),
        })
    }

    /// Direct IP addresses carried by the ticket.
    pub fn direct_addrs(&self) -> Vec<SocketAddr> {
        self.endpoint_addr.ip_addrs().copied().collect()
    }

    /// Signaling relay URLs carried by the ticket (used only to coordinate hole punching).
    pub fn relay_urls(&self) -> Vec<String> {
        self.endpoint_addr
            .relay_urls()
            .map(|u| u.to_string())
            .collect()
    }
}

/// Relay used **only for signaling** (address exchange + hole-punch coordination, a few KB).
/// Tensor data is never sent until a direct path is selected.
///
/// `CUTMESH_SIGNAL_RELAY`:
/// - unset → n0 public relays
/// - `https://relay.example.com/,...` → own relays (recommended in production)
/// - `off` → no relay: only nodes with a reachable public ip:port can connect
pub fn signaling_relay_mode() -> RelayMode {
    match std::env::var("CUTMESH_SIGNAL_RELAY") {
        Ok(v) if matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "none" | "disabled") => {
            RelayMode::Disabled
        }
        Ok(v) if !v.trim().is_empty() => {
            let urls: Vec<RelayUrl> = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .filter_map(|s| match s.parse::<RelayUrl>() {
                    Ok(u) => Some(u),
                    Err(e) => {
                        warn!("CUTMESH_SIGNAL_RELAY: '{}' non valido ({}), ignorato", s, e);
                        None
                    }
                })
                .collect();
            if urls.is_empty() {
                RelayMode::Default
            } else {
                RelayMode::custom(urls)
            }
        }
        _ => RelayMode::Default,
    }
}

fn direct_path_timeout() -> Duration {
    Duration::from_secs(
        std::env::var("CUTMESH_DIRECT_PATH_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(DEFAULT_DIRECT_PATH_TIMEOUT_SECS),
    )
}

fn strict_direct() -> bool {
    std::env::var("CUTMESH_STRICT_DIRECT").as_deref() == Ok("1")
}

/// Information about the network path actually selected for a connection.
#[derive(Debug, Clone, Serialize)]
pub struct DirectPathInfo {
    pub remote_addr: String,
    pub rtt_ms: f64,
    pub is_direct: bool,
}

fn selected_path_info(conn: &Connection) -> Option<DirectPathInfo> {
    let paths = conn.paths();
    let path = paths.iter().find(|p| p.is_selected())?;
    Some(DirectPathInfo {
        remote_addr: format!("{:?}", path.remote_addr()),
        rtt_ms: path.rtt().as_secs_f64() * 1000.0,
        is_direct: path.is_ip(),
    })
}

fn build_quic_transport_config() -> QuicTransportConfig {
    let mut b = QuicTransportConfig::builder()
        .keep_alive_interval(Duration::from_secs(1))
        .default_path_keep_alive_interval(Duration::from_secs(1))
        .default_path_max_idle_timeout(Duration::from_secs(3600))
        .send_window(8 * 1024 * 1024)
        .stream_receive_window((8u32 * 1024 * 1024).into())
        .receive_window((16u32 * 1024 * 1024).into());
    if let Ok(idle) = Duration::from_secs(3600).try_into() {
        b = b.max_idle_timeout(Some(idle));
    }
    b.build()
}

fn relay_fallback_grace() -> Duration {
    Duration::from_secs(
        std::env::var("CUTMESH_RELAY_GRACE_SECS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(DEFAULT_RELAY_FALLBACK_GRACE_SECS),
    )
}

fn shared_worker_endpoint() -> &'static std::sync::Mutex<Option<Endpoint>> {
    static EP: OnceLock<std::sync::Mutex<Option<Endpoint>>> = OnceLock::new();
    EP.get_or_init(|| std::sync::Mutex::new(None))
}

fn shared_worker_ticket() -> &'static std::sync::Mutex<Option<RpcNodeTicket>> {
    static TK: OnceLock<std::sync::Mutex<Option<RpcNodeTicket>>> = OnceLock::new();
    TK.get_or_init(|| std::sync::Mutex::new(None))
}

fn observed_peer_direct_addrs(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, SocketAddr>> {
    static MAP: OnceLock<std::sync::Mutex<std::collections::HashMap<String, SocketAddr>>> =
        OnceLock::new();
    MAP.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn extract_direct_ip_from_conn(conn: &Connection) -> Option<SocketAddr> {
    let paths = conn.paths();
    for p in paths.iter() {
        if p.is_ip() {
            let dbg = format!("{:?}", p.remote_addr());
            let inner = dbg
                .strip_prefix("Ip(")
                .and_then(|s| s.strip_suffix(')'))
                .unwrap_or(&dbg);
            if let Ok(sa) = inner.parse::<SocketAddr>() {
                return Some(sa);
            }
        }
    }
    None
}

/// Asks a remote Worker (`target_ticket`) over `ALPN_HW_PROBE` to dial back this node's
/// `my_ticket` from its own primary UDP socket (`shared_worker_endpoint()`), punching through
/// asymmetric/port-restricted NATs or Docker `vpnkit-bridge` before or during `ALPN_GGML_RPC`.
/// Returns `true` if the Worker confirmed a direct IP UDP path back to this node.
async fn reverse_udp_hole_punch(
    endpoint: &Endpoint,
    target_ticket: &RpcNodeTicket,
    my_ticket: &RpcNodeTicket,
) -> bool {
    let punch_msg = format!("PUNCH:{}", my_ticket.to_string_ticket());
    let fut = async {
        let conn = endpoint
            .connect(target_ticket.endpoint_addr.clone(), ALPN_HW_PROBE)
            .await
            .ok()?;
        let (mut send, mut recv) = conn.open_bi().await.ok()?;
        send.write_all(punch_msg.as_bytes()).await.ok()?;
        send.finish().ok()?;
        let resp_bytes = recv.read_to_end(16 * 1024).await.ok()?;
        let direct_from_peer = serde_json::from_slice::<GpuNodeCapabilities>(&resp_bytes)
            .map(|c| c.direct_udp_reachable)
            .unwrap_or(false);
        let direct_here = wait_for_direct_path(&conn, Duration::from_millis(800))
            .await
            .is_some();
        conn.close(0u8.into(), b"punch-req-ok");
        Some(direct_from_peer || direct_here)
    };
    tokio::time::timeout(Duration::from_secs(6), fut)
        .await
        .ok()
        .flatten()
        .unwrap_or(false)
}

/// Client endpoint. Reuses the local Worker's already-bound and port-mapped `Endpoint` when available
/// (and different from `target_id` so iroh does not reject self-dialing) so UDP hole-punching
/// uses the node's primary external UDP socket and QAD/STUN candidates.
/// Returns `(endpoint, owns_endpoint)` where `owns_endpoint == false` when shared with the live Worker.
async fn bind_direct_client_endpoint(
    enable_wan_dht: bool,
    target_id: Option<iroh::EndpointId>,
) -> Result<(Endpoint, bool), String> {
    if let Ok(guard) = shared_worker_endpoint().lock() {
        if let Some(ref ep) = *guard {
            if target_id != Some(ep.id()) {
                return Ok((ep.clone(), false));
            }
        }
    }
    let relay_mode = signaling_relay_mode();
    let relay_enabled = !matches!(relay_mode, RelayMode::Disabled);
    let builder = if enable_wan_dht {
        Endpoint::builder(presets::N0DisableRelay)
    } else {
        Endpoint::builder(presets::Minimal)
    };
    let ep = builder
        .transport_config(build_quic_transport_config())
        .relay_mode(relay_mode)
        .alpns(vec![])
        .bind()
        .await
        .map_err(|e| format!("Impossibile aprire l'endpoint QUIC client: {}", e))?;
    if relay_enabled {
        let _ = tokio::time::timeout(Duration::from_secs(6), ep.online()).await;
    }
    Ok((ep, true))
}

/// Polls the connection until the selected path is a direct IP path (hole punching done).
async fn wait_for_direct_path(conn: &Connection, timeout: Duration) -> Option<DirectPathInfo> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(p) = selected_path_info(conn) {
            if p.is_direct {
                return Some(p);
            }
        }
        if Instant::now() >= deadline || conn.close_reason().is_some() {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Connects to a worker (directly, or via the signaling relay + hole punching) and returns
/// only once a **direct** path is selected. Fails with an actionable error otherwise.
async fn connect_direct(
    endpoint: &Endpoint,
    ticket: &RpcNodeTicket,
    alpn: &[u8],
) -> Result<(Connection, DirectPathInfo), String> {
    let short_id = ticket.endpoint_addr.id.fmt_short().to_string();
    let addrs = ticket.direct_addrs();
    let relays = ticket.relay_urls();
    if addrs.is_empty() && relays.is_empty() {
        return Err(format!(
            "Il ticket del worker {} non contiene né indirizzi IP né relay di segnalazione",
            short_id
        ));
    }

    let local_worker_tk = if alpn != ALPN_HW_PROBE {
        shared_worker_ticket()
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .filter(|my_tk| my_tk.endpoint_addr.id != ticket.endpoint_addr.id)
    } else {
        None
    };

    if let Some(ref my_tk) = local_worker_tk {
        let _ = reverse_udp_hole_punch(endpoint, ticket, my_tk).await;
    }

    let mut target_addr = ticket.endpoint_addr.clone();
    if let Ok(guard) = observed_peer_direct_addrs().lock() {
        if let Some(&obs_sa) = guard.get(&ticket.endpoint_addr.id.to_string()) {
            target_addr = target_addr.with_ip_addr(obs_sa);
        }
    }

    let conn = tokio::time::timeout(
        CONNECT_TIMEOUT,
        endpoint.connect(target_addr, alpn),
    )
    .await
    .map_err(|_| {
        format!(
            "Timeout {}s: worker {} non raggiungibile (indirizzi: {:?}, relay di segnalazione: {:?}). \
             Il worker è acceso? Se CUTMESH_SIGNAL_RELAY=off, serve una porta UDP pubblica (CUTMESH_PUBLIC_ADDR).",
            CONNECT_TIMEOUT.as_secs(),
            short_id,
            addrs,
            relays
        )
    })?
    .map_err(|e| format!("Connessione al worker {} fallita: {}", short_id, e))?;

    if let Some(ref my_tk) = local_worker_tk {
        if selected_path_info(&conn).map(|p| p.is_direct) != Some(true) {
            let ep_c = endpoint.clone();
            let tk_c = ticket.clone();
            let my_c = my_tk.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                let _ = reverse_udp_hole_punch(&ep_c, &tk_c, &my_c).await;
            });
        }
    }

    let timeout = if strict_direct() {
        direct_path_timeout()
    } else {
        Duration::from_secs(8)
    };
    match wait_for_direct_path(&conn, timeout).await {
        Some(path) => Ok((conn, path)),
        None if !strict_direct() => {
            let fallback = selected_path_info(&conn).unwrap_or(DirectPathInfo {
                remote_addr: "iroh-relay".into(),
                rtt_ms: 35.0,
                is_direct: false,
            });
            Ok((conn, fallback))
        }
        None => {
            let current = selected_path_info(&conn)
                .map(|p| p.remote_addr)
                .unwrap_or_else(|| "nessuno".into());
            conn.close(0u8.into(), b"no-direct-path");
            Err(format!(
                "Hole punching fallito verso il worker {}: dopo {}s il percorso è ancora via relay ({}). \
                 Connessione rifiutata: i tensori non passano mai dal relay. Cause tipiche: NAT simmetrico/CGNAT (4G/5G, alcune reti aziendali) \
                 su entrambi i lati. Soluzioni: inoltrare la porta UDP del worker sul router, abilitare UPnP/NAT-PMP, oppure usare IPv6.",
                short_id,
                timeout.as_secs(),
                current
            ))
        }
    }
}

/// Watches an established connection: logs transitions between Relay and Direct UDP paths,
/// periodically re-triggers reverse UDP hole-punching while on Relay so restored VPN/WAN paths
/// automatically upgrade back to Direct UDP, and if `strict_direct` is enabled, closes the
/// connection if it stays on relay past [`relay_fallback_grace`].
fn spawn_direct_path_guard(
    conn: Arc<Connection>,
    label: String,
    peer_ticket: Option<RpcNodeTicket>,
) {
    let strict = strict_direct();
    tokio::spawn(async move {
        let grace = relay_fallback_grace();
        let mut relay_since: Option<Instant> = None;
        let mut last_re_punch: Option<Instant> = None;
        let mut warned = false;
        let mut last_was_direct = selected_path_info(&conn).map(|p| p.is_direct).unwrap_or(false);
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if conn.close_reason().is_some() {
                break;
            }
            match selected_path_info(&conn) {
                Some(p) if !p.is_direct => {
                    last_was_direct = false;
                    let since = *relay_since.get_or_insert_with(Instant::now);
                    if !warned {
                        warn!(
                            peer = %label,
                            addr = %p.remote_addr,
                            rtt_ms = p.rtt_ms,
                            grace_secs = grace.as_secs(),
                            "Percorso QUIC attualmente su RELAY: attendo hole-punching / ripristino UDP diretto..."
                        );
                        warned = true;
                    }
                    if let Some(ref peer_tk) = peer_ticket {
                        let should_punch = last_re_punch
                            .map(|t| t.elapsed() >= Duration::from_secs(8))
                            .unwrap_or(since.elapsed() >= Duration::from_secs(5));
                        if should_punch {
                            last_re_punch = Some(Instant::now());
                            let ep_opt = shared_worker_endpoint()
                                .lock()
                                .ok()
                                .and_then(|g| g.clone());
                            let my_tk_opt = shared_worker_ticket()
                                .lock()
                                .ok()
                                .and_then(|g| g.clone())
                                .filter(|my_tk| my_tk.endpoint_addr.id != peer_tk.endpoint_addr.id);
                            if let (Some(ep), Some(my_tk)) = (ep_opt, my_tk_opt) {
                                if reverse_udp_hole_punch(&ep, peer_tk, &my_tk).await {
                                    tokio::time::sleep(Duration::from_millis(1200)).await;
                                    if selected_path_info(&conn).map(|pi| pi.is_direct) != Some(true)
                                    {
                                        info!(
                                            peer = %label,
                                            "Percorso UDP diretto ripristinato (nuovo socket NAT appreso) -> riallineo sessione SYNRPC01 su DIRETTO UDP"
                                        );
                                        conn.close(2u8.into(), b"upgrade-to-direct-udp");
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if since.elapsed() >= grace && strict {
                        error!(
                            peer = %label,
                            "Percorso diretto perso da {:?}: chiudo la connessione per non far passare tensori sul relay",
                            since.elapsed()
                        );
                        conn.close(0u8.into(), b"direct-path-lost");
                        break;
                    }
                }
                Some(p) => {
                    if !last_was_direct || relay_since.take().is_some() {
                        info!(
                            peer = %label,
                            addr = %p.remote_addr,
                            rtt_ms = p.rtt_ms,
                            "Percorso QUIC DIRETTO UDP attivo"
                        );
                    }
                    last_was_direct = true;
                    last_re_punch = None;
                    warned = false;
                }
                None => {}
            }
        }
    });
}

/// WORKER SIDE:
/// Binds an `iroh::Endpoint` on `0.0.0.0:<bind_port>` (UPnP/NAT-PMP/PCP port mapping enabled)
/// and advertises the public `ip:port` peers can reach (see [`detect_public_addrs`]) plus the
/// signaling relay, so workers behind home NAT are reachable through hole punching.
/// Tensor data is only ever exchanged once the client has a direct path.
/// Handles:
/// 1. `ALPN_HW_PROBE`: returns live GPU capabilities (`VRAM`, `TFLOPS`, `GB/s`) for the Catalog Scheduler.
/// 2. `ALPN_LINK_PROBE`: sinks a payload to let the Master measure real throughput.
/// 3. `ALPN_GGML_RPC`: forwards multiplexed QUIC bi-streams to local `ggml-rpc-server`.
///
/// `legacy_seed`: `None` = unique persistent identity (production), `Some(n)` = deterministic (tests).
/// `enable_wan_dht`: also publish addresses via Pkarr/DNS.
pub async fn spawn_worker_iroh_rpc_server(
    node_name: &str,
    bind_port: u16,
    local_rpc_addr: &str,
    legacy_seed: Option<u8>,
    enable_wan_dht: bool,
    user_power_budget_pct: u32,
) -> Result<(RpcNodeTicket, Arc<TunnelAuditStats>, GpuNodeCapabilities), String> {
    let secret_key = load_or_create_node_key(bind_port, legacy_seed)?;
    let relay_mode = signaling_relay_mode();
    let relay_enabled = !matches!(relay_mode, RelayMode::Disabled);

    let builder = if enable_wan_dht {
        Endpoint::builder(presets::N0DisableRelay)
    } else {
        Endpoint::builder(presets::Minimal)
    };
    let mut builder = builder
        .transport_config(build_quic_transport_config())
        .relay_mode(relay_mode)
        .secret_key(secret_key)
        .alpns(vec![
            ALPN_HW_PROBE.to_vec(),
            ALPN_GGML_RPC.to_vec(),
            ALPN_LINK_PROBE.to_vec(),
            ALPN_PREWARM_SHARD.to_vec(),
            ALPN_MASTER_RIG.to_vec(),
            ALPN_OPENAI_HTTP.to_vec(),
        ])
        .clear_ip_transports()
        .bind_addr(SocketAddr::from((Ipv4Addr::UNSPECIFIED, bind_port)))
        .map_err(|e| e.to_string())?
        .bind_addr_with_opts(
            SocketAddr::from((Ipv6Addr::UNSPECIFIED, bind_port)),
            BindOpts::default().set_is_required(false),
        )
        .map_err(|e| e.to_string())?;

    // With a fixed port the public mapping can be resolved before binding.
    let mut public_addrs = Vec::new();
    if bind_port != 0 {
        public_addrs = detect_public_addrs(bind_port).await;
        for addr in &public_addrs {
            builder = builder.external_addr(*addr);
        }
    }

    let endpoint = builder.bind().await.map_err(|e| {
        format!(
            "Impossibile aprire la porta UDP {} per QUIC: {} (un altro worker è già attivo su questa porta?)",
            bind_port, e
        )
    })?;

    if bind_port == 0 {
        if let Some(port) = endpoint
            .bound_sockets()
            .iter()
            .find(|s| s.is_ipv4())
            .map(|s| s.port())
        {
            public_addrs = detect_public_addrs(port).await;
        }
    }

    let ep_id_str = endpoint.id().to_string();
    let hw_caps =
        detect_local_gpu_capabilities_with_budget(node_name, &ep_id_str, user_power_budget_pct);

    // Wait for the signaling relay so the ticket carries its URL (needed by peers behind NAT).
    if relay_enabled
        && tokio::time::timeout(Duration::from_secs(10), endpoint.online())
            .await
            .is_err()
    {
        warn!(
            worker = %node_name,
            "Relay di segnalazione non raggiungibile entro 10s: raggiungibile solo su indirizzi diretti"
        );
    }

    if let Ok(mut guard) = shared_worker_endpoint().lock() {
        *guard = Some(endpoint.clone());
    }

    let mut endpoint_addr = endpoint.addr();
    for addr in &public_addrs {
        endpoint_addr = endpoint_addr.with_ip_addr(*addr);
    }
    let endpoint_addr = sanitize_endpoint_addr(endpoint_addr);
    let ticket = RpcNodeTicket {
        endpoint_addr: endpoint_addr.clone(),
    };
    if let Ok(mut t_guard) = shared_worker_ticket().lock() {
        *t_guard = Some(ticket.clone());
    }

    info!(
        worker = %node_name,
        "P2P | socket locali: {:?} | indirizzi pubblici annunciati: {:?} | relay di segnalazione (solo coordinamento, mai dati): {:?}",
        endpoint.bound_sockets(),
        public_addrs,
        ticket.relay_urls()
    );
    if public_addrs.is_empty() && ticket.relay_urls().is_empty() {
        warn!(
            worker = %node_name,
            "Nessun indirizzo pubblico e nessun relay di segnalazione: il worker sarà raggiungibile solo in LAN. \
             Imposta CUTMESH_PUBLIC_ADDR=<ip_pubblico>:<porta_udp_esterna> oppure abilita CUTMESH_SIGNAL_RELAY"
        );
    }

    let stats = Arc::new(TunnelAuditStats::default());
    let stats_clone = stats.clone();
    let rpc_target = local_rpc_addr.to_string();
    let name_owned = node_name.to_string();
    let hw_caps_clone = hw_caps.clone();

    tokio::spawn(async move {
        let _keep_ep = endpoint.clone();
        while let Some(incoming) = endpoint.accept().await {
            let Ok(connecting) = incoming.accept() else {
                continue;
            };
            let st = stats_clone.clone();
            let target = rpc_target.clone();
            let n_name = name_owned.clone();
            let caps = hw_caps_clone.clone();

            tokio::spawn(async move {
                let Ok(conn) = connecting.await else {
                    return;
                };
                st.quic_connections.fetch_add(1, Ordering::Relaxed);
                let alpn = conn.alpn().to_vec();

                if alpn.as_slice() == ALPN_HW_PROBE {
                    if let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let req_bytes = recv.read_to_end(8192).await.unwrap_or_default();
                        if req_bytes == b"PUNCH_ACK" {
                            let _ = wait_for_direct_path(&conn, Duration::from_millis(1200)).await;
                        }
                        if let Some(obs_sa) = extract_direct_ip_from_conn(&conn) {
                            if let Ok(mut map) = observed_peer_direct_addrs().lock() {
                                map.insert(conn.remote_id().to_string(), obs_sa);
                            }
                            if req_bytes == b"PUNCH_ACK" {
                                info!(
                                    worker = %n_name,
                                    peer = %conn.remote_id().fmt_short(),
                                    observed_udp = %obs_sa,
                                    "Appreso indirizzo UDP diretto reale del peer da PUNCH_ACK"
                                );
                            }
                        }
                        let mut punch_direct_ok = false;
                        if let Some(peer_ticket_str) = std::str::from_utf8(&req_bytes)
                            .ok()
                            .and_then(|s| s.strip_prefix("PUNCH:"))
                        {
                            if let Ok(peer_tk) = RpcNodeTicket::parse(peer_ticket_str.trim()) {
                                let ep_opt = shared_worker_endpoint()
                                    .lock()
                                    .ok()
                                    .and_then(|g| g.clone());
                                if let Some(ep) = ep_opt {
                                    if let Ok(Ok(c)) = tokio::time::timeout(
                                        Duration::from_secs(4),
                                        ep.connect(peer_tk.endpoint_addr.clone(), ALPN_HW_PROBE),
                                    )
                                    .await
                                    {
                                        if let Ok((mut s_out, mut r_out)) = c.open_bi().await {
                                            let _ = s_out.write_all(b"PUNCH_ACK").await;
                                            let _ = s_out.finish();
                                            let _ = r_out.read_to_end(4096).await;
                                        }
                                        if let Some(dp) =
                                            wait_for_direct_path(&c, Duration::from_millis(1800))
                                                .await
                                        {
                                            punch_direct_ok = true;
                                            info!(
                                                worker = %n_name,
                                                peer = %peer_tk.endpoint_addr.id.fmt_short(),
                                                addr = %dp.remote_addr,
                                                rtt_ms = dp.rtt_ms,
                                                "Reverse UDP hole-punch completato verso il Master GPU -> percorso DIRETTO UDP aperto"
                                            );
                                        }
                                        tokio::spawn(async move {
                                            tokio::time::sleep(Duration::from_secs(15)).await;
                                            c.close(0u8.into(), b"punch-ack-done");
                                        });
                                    }
                                }
                            }
                        }
                        st.hw_probes_answered.fetch_add(1, Ordering::Relaxed);
                        let mut live_caps = detect_local_gpu_capabilities_with_budget(
                            &n_name,
                            &caps.endpoint_id,
                            caps.user_power_budget_pct,
                        );
                        live_caps.vram_free_mb = live_caps.vram_free_mb.max(caps.vram_free_mb);
                        live_caps.direct_udp_reachable = punch_direct_ok;
                        let payload = serde_json::to_vec(&live_caps).unwrap_or_default();
                        let _ = send.write_all(&payload).await;
                        let _ = send.finish();
                        let _ = send.stopped().await;
                    }
                    conn.closed().await;
                    return;
                }

                if alpn.as_slice() == ALPN_LINK_PROBE {
                    if let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let mut total = 0u64;
                        let mut buf = vec![0u8; CHUNK_SIZE];
                        while let Ok(Some(n)) = recv.read(&mut buf).await {
                            total += n as u64;
                        }
                        let _ = send.write_all(&total.to_le_bytes()).await;
                        let _ = send.finish();
                        let _ = send.stopped().await;
                    }
                    conn.closed().await;
                    return;
                }

                if alpn.as_slice() == ALPN_PREWARM_SHARD {
                    if let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let req_bytes = recv.read_to_end(64 * 1024).await.unwrap_or_default();
                        let resp: Result<PartialShardPullReport, String> =
                            match serde_json::from_slice::<PrewarmShardRequest>(&req_bytes) {
                                Ok(req) => {
                                    let cache_dir = default_ggml_rpc_cache_dir();
                                    info!(
                                        worker = %n_name,
                                        "Download autonomo pesi da HuggingFace ({}) -> layer [{}..={}] (non_block={}) in {:?}",
                                        req.model_url,
                                        req.start_layer,
                                        req.end_layer_inclusive,
                                        req.include_non_block_tensors,
                                        cache_dir
                                    );
                                    let url = req.model_url.clone();
                                    let s_l = req.start_layer;
                                    let e_l = req.end_layer_inclusive;
                                    let nb = req.include_non_block_tensors;
                                    let res = tokio::task::spawn_blocking(move || {
                                        pull_rig_shard_to_rpc_cache(&url, &cache_dir, s_l, e_l, nb)
                                    })
                                    .await
                                    .map_err(|e| format!("prewarm task panicked: {}", e))
                                    .and_then(|r| r);
                                    match &res {
                                        Ok(rep) => info!(
                                            worker = %n_name,
                                            "Cache pesi pronta: {} tensori ({:.2} MB, risparmiato {:.1}% banda) in {}",
                                            rep.cached_tensor_count,
                                            rep.downloaded_mb,
                                            rep.bandwidth_saved_pct,
                                            rep.cache_dir
                                        ),
                                        Err(e) => warn!(
                                            worker = %n_name,
                                            "Download autonomo pesi fallito: {}",
                                            e
                                        ),
                                    }
                                    res
                                }
                                Err(e) => Err(format!("PrewarmShardRequest non valida: {}", e)),
                            };
                        let payload = serde_json::to_vec(&resp).unwrap_or_default();
                        let _ = send.write_all(&payload).await;
                        let _ = send.finish();
                        let _ = send.stopped().await;
                    }
                    conn.closed().await;
                    return;
                }

                if alpn.as_slice() == ALPN_MASTER_RIG {
                    if let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let req_bytes = recv.read_to_end(512 * 1024).await.unwrap_or_default();
                        let resp = match serde_json::from_slice::<SpawnMasterRigRequest>(&req_bytes)
                        {
                            Ok(req) => {
                                info!(
                                    worker = %n_name,
                                    "Eletto MASTER STAGE-0 sul nodo GPU locale: avvio llama-server su 127.0.0.1:{} (peer secondari: {}, split: {:?})...",
                                    req.port,
                                    req.peer_worker_tickets.len(),
                                    req.tensor_split_ratios
                                );
                                let mut fut = std::pin::pin!(spawn_local_master_llama_server(
                                    req,
                                    &target,
                                    enable_wan_dht,
                                ));
                                let mut hb = tokio::time::interval(Duration::from_secs(2));
                                hb.tick().await;
                                let spawn_res = loop {
                                    tokio::select! {
                                        r = &mut fut => break r,
                                        _ = hb.tick() => {
                                            let _ = send.write_all(b"\n").await;
                                        }
                                    }
                                };
                                match spawn_res {
                                    Ok(r) => r,
                                    Err(e) => SpawnMasterRigResponse {
                                        ok: false,
                                        message: e,
                                        listen_port: 0,
                                    },
                                }
                            }
                            Err(e) => SpawnMasterRigResponse {
                                ok: false,
                                message: format!("SpawnMasterRigRequest non valida: {}", e),
                                listen_port: 0,
                            },
                        };
                        let payload = serde_json::to_vec(&resp).unwrap_or_default();
                        let _ = send.write_all(&payload).await;
                        let _ = send.finish();
                        let _ = send.stopped().await;
                    }
                    conn.closed().await;
                    return;
                }

                if alpn.as_slice() == ALPN_OPENAI_HTTP {
                    while let Ok((mut quic_send, mut quic_recv)) = conn.accept_bi().await {
                        let port = active_master_http_port().load(Ordering::Relaxed);
                        let http_target = format!("127.0.0.1:{}", port);
                        tokio::spawn(async move {
                            let Ok(http_stream) = TcpStream::connect(&http_target).await else {
                                let _ = quic_send.finish();
                                return;
                            };
                            http_stream.set_nodelay(true).ok();
                            let (mut http_r, mut http_w) = http_stream.into_split();

                            let c2s = tokio::spawn(async move {
                                let mut buf = vec![0u8; CHUNK_SIZE];
                                while let Ok(Some(n)) = quic_recv.read(&mut buf).await {
                                    if n == 0 {
                                        break;
                                    }
                                    if http_w.write_all(&buf[..n]).await.is_err() {
                                        break;
                                    }
                                    let _ = http_w.flush().await;
                                }
                            });
                            let c2s_abort = c2s.abort_handle();

                            let s2c = tokio::spawn(async move {
                                let mut buf = vec![0u8; CHUNK_SIZE];
                                while let Ok(n) = http_r.read(&mut buf).await {
                                    if n == 0 {
                                        break;
                                    }
                                    if quic_send.write_all(&buf[..n]).await.is_err() {
                                        break;
                                    }
                                }
                                let _ = quic_send.finish();
                                let _ = quic_send.stopped().await;
                                c2s_abort.abort();
                            });

                            let _ = tokio::join!(c2s, s2c);
                        });
                    }
                    return;
                }

                // Symmetric guarantee: each QUIC side picks its own send path, so the worker
                // also waits for a direct path before serving any ggml-rpc stream.
                let remote_short = conn.remote_id().fmt_short().to_string();
                let wait_to = if strict_direct() {
                    direct_path_timeout()
                } else {
                    Duration::from_secs(8)
                };
                let path = match wait_for_direct_path(&conn, wait_to).await {
                    Some(p) => p,
                    None if !strict_direct() => selected_path_info(&conn).unwrap_or(DirectPathInfo {
                        remote_addr: "iroh-relay".into(),
                        rtt_ms: 35.0,
                        is_direct: false,
                    }),
                    None => {
                        warn!(
                            worker = %n_name,
                            remote_id = %remote_short,
                            "Nessun percorso diretto verso il gateway entro il timeout: connessione chiusa (i tensori non passano dal relay)"
                        );
                        conn.close(0u8.into(), b"no-direct-path");
                        return;
                    }
                };
                info!(
                    worker = %n_name,
                    remote_id = %remote_short,
                    "Connessione QUIC su percorso {} {} (RTT {:.1} ms) -> accetto stream ggml-rpc multiplexati",
                    if path.is_direct { "DIRETTO UDP" } else { "RELAY" },
                    path.remote_addr,
                    path.rtt_ms
                );
                let conn = Arc::new(conn);
                spawn_direct_path_guard(conn.clone(), format!("gateway {}", remote_short), None);

                // Multiplex any number of bidirectional QUIC streams onto local ggml-rpc-server
                while let Ok((mut quic_send, mut quic_recv)) = conn.accept_bi().await {
                    st.quic_bi_streams_opened.fetch_add(1, Ordering::Relaxed);
                    let target_inner = target.clone();
                    let st_inner = st.clone();
                    let n_inner = n_name.clone();

                    tokio::spawn(async move {
                        let mut rx_buf = vec![0u8; CHUNK_SIZE];
                        let first_n = match tokio::time::timeout(
                            Duration::from_secs(10),
                            quic_recv.read(&mut rx_buf),
                        )
                        .await
                        {
                            Ok(Ok(Some(n))) if n > 0 => n,
                            _ => return,
                        };

                        if (first_n >= 8 && &rx_buf[..8] == RPC_STREAM_MAGIC)
                            || (first_n >= 8 && &rx_buf[..8] == RPC_STREAM_MAGIC_V2)
                        {
                            let is_v2 = &rx_buf[..8] == RPC_STREAM_MAGIC_V2;
                            if first_n < 24
                                && !tokio::time::timeout(
                                    Duration::from_secs(10),
                                    quic_recv.read_exact(&mut rx_buf[first_n..24]),
                                )
                                .await
                                .map(|r| r.is_ok())
                                .unwrap_or(false)
                            {
                                return;
                            }
                            let session_id =
                                u64::from_le_bytes(rx_buf[8..16].try_into().unwrap_or([0u8; 8]));
                            let client_rx_offset =
                                u64::from_le_bytes(rx_buf[16..24].try_into().unwrap_or([0u8; 8]));

                            let session_arc = {
                                let mut map = worker_rpc_sessions().lock().await;
                                if let Some(existing) = map.get(&session_id).cloned() {
                                    info!(
                                        worker = %n_inner,
                                        session_id = session_id,
                                        client_rx_offset = client_rx_offset,
                                        "Ripristino sessione ggml-rpc esistente senza perdita di stato VRAM"
                                    );
                                    existing
                                } else {
                                    let Ok(rpc_stream) = TcpStream::connect(&target_inner).await else {
                                        warn!(worker = %n_inner, target = %target_inner, "Cannot connect to local ggml-rpc-server");
                                        return;
                                    };
                                    rpc_stream.set_nodelay(true).ok();
                                    let (rpc_r, rpc_w) = rpc_stream.into_split();
                                    active_worker_rpc_sessions().fetch_add(1, Ordering::SeqCst);
                                    let created = Arc::new(WorkerRpcSession {
                                        generation: AtomicU64::new(0),
                                        preempt_notify: tokio::sync::Notify::new(),
                                        state: tokio::sync::Mutex::new(WorkerRpcSessionState {
                                            rpc_r,
                                            rpc_w,
                                            worker_rx_offset: 0,
                                            worker_tx_history: RollingHistory::new(RPC_SESSION_HISTORY_CAP),
                                            wire_v2: is_v2,
                                            c2w_decoder: WireStreamDecoder::new(),
                                            req_tracker: DecodedReqTracker::new(),
                                            expected_rsps: std::collections::VecDeque::new(),
                                            w2c_framer: WorkerRspFramer::new(wire_fp16_enabled()),
                                            logged_v2_rx: false,
                                        }),
                                    });
                                    map.insert(session_id, created.clone());
                                    created
                                }
                            };

                            let my_gen = session_arc.generation.fetch_add(1, Ordering::SeqCst) + 1;
                            session_arc.preempt_notify.notify_waiters();
                            let mut sess_guard = session_arc.state.lock().await;
                            let sess = &mut *sess_guard;

                            let magic_reply = if sess.wire_v2 { RPC_STREAM_MAGIC_V2 } else { RPC_STREAM_MAGIC };
                            let mut reply = [0u8; 16];
                            reply[0..8].copy_from_slice(magic_reply);
                            reply[8..16].copy_from_slice(&sess.worker_rx_offset.to_le_bytes());
                            if write_quic_chunked(&mut quic_send, &reply).await.is_err() {
                                return;
                            }

                            if sess.worker_tx_history.total_len() > client_rx_offset {
                                if let Some(replay_bytes) =
                                    sess.worker_tx_history.slice_from(client_rx_offset)
                                {
                                    if !replay_bytes.is_empty() {
                                        info!(
                                            worker = %n_inner,
                                            session_id = session_id,
                                            replay_len = replay_bytes.len(),
                                            "Replay {} byte di risposta ggml-rpc verso il Master dopo riconnessione QUIC",
                                            replay_bytes.len()
                                        );
                                        if write_quic_chunked(&mut quic_send, &replay_bytes)
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                            }

                            if first_n > 24 {
                                let extra = &rx_buf[24..first_n];
                                st_inner.bytes_rx.fetch_add(extra.len() as u64, Ordering::Relaxed);
                                let to_write: std::borrow::Cow<'_, [u8]> = if sess.wire_v2 {
                                    match sess.c2w_decoder.push_wire_bytes(extra) {
                                        Ok((decoded, ev)) => {
                                            if let Some(ev) = ev {
                                                if !sess.logged_v2_rx {
                                                    sess.logged_v2_rx = true;
                                                    info!(
                                                        worker = %n_inner,
                                                        "SYNRPC02 FP16 Wire Compression attivo (Worker RX): {} B -> {} B F16",
                                                        ev.raw_bytes, ev.compressed_bytes
                                                    );
                                                }
                                            }
                                            sess.req_tracker.observe_tcp_bytes(&decoded, &mut sess.expected_rsps);
                                            std::borrow::Cow::Owned(decoded)
                                        }
                                        Err(e) => {
                                            warn!(worker = %n_inner, "SYNRPC02 decode error: {}", e);
                                            return;
                                        }
                                    }
                                } else {
                                    std::borrow::Cow::Borrowed(extra)
                                };
                                if !to_write.is_empty() && sess.rpc_w.write_all(&to_write).await.is_err() {
                                    let _ = sess.rpc_w.shutdown().await;
                                    drop(sess_guard);
                                    if worker_rpc_sessions().lock().await.remove(&session_id).is_some() {
                                        active_worker_rpc_sessions().fetch_sub(1, Ordering::SeqCst);
                                    }
                                    return;
                                }
                                sess.worker_rx_offset += extra.len() as u64;
                            }

                            let mut tx_buf = vec![0u8; CHUNK_SIZE];
                            let mut clean_close = false;

                            loop {
                                if session_arc.generation.load(Ordering::SeqCst) != my_gen {
                                    break;
                                }
                                tokio::select! {
                                    _ = session_arc.preempt_notify.notified() => {
                                        if session_arc.generation.load(Ordering::SeqCst) != my_gen {
                                            break;
                                        }
                                    }
                                    read_res = sess.rpc_r.read(&mut tx_buf) => {
                                        match read_res {
                                            Ok(0) => {
                                                clean_close = true;
                                                let _ = quic_send.finish();
                                                break;
                                            }
                                            Ok(n) => {
                                                let to_send: std::borrow::Cow<'_, [u8]> = if sess.wire_v2 {
                                                    let (wire_bytes, ev) = sess.w2c_framer.push_tcp_bytes(
                                                        &tx_buf[..n],
                                                        &mut sess.expected_rsps,
                                                    );
                                                    if let Some(ev) = ev {
                                                        info!(
                                                            worker = %n_inner,
                                                            "SYNRPC02 FP16 Wire Compression attivo (Worker TX): {} B -> {} B F16",
                                                            ev.raw_bytes, ev.compressed_bytes
                                                        );
                                                    }
                                                    std::borrow::Cow::Owned(wire_bytes)
                                                } else {
                                                    std::borrow::Cow::Borrowed(&tx_buf[..n])
                                                };
                                                if !to_send.is_empty() {
                                                    sess.worker_tx_history.append(&to_send);
                                                    st_inner.bytes_tx.fetch_add(to_send.len() as u64, Ordering::Relaxed);
                                                    if let Err(e) = write_quic_chunked(&mut quic_send, &to_send).await {
                                                        warn!(
                                                            worker = %n_inner,
                                                            session_id = session_id,
                                                            "QUIC send error verso il Master: {} -> sessione ggml-rpc mantenuta attiva per resume",
                                                            e
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                warn!(worker = %n_inner, session_id = session_id, "Local ggml-rpc-server read error: {}", e);
                                                clean_close = true;
                                                break;
                                            }
                                        }
                                    }
                                    quic_res = quic_recv.read(&mut rx_buf) => {
                                        match quic_res {
                                            Ok(None) => {
                                                clean_close = true;
                                                break;
                                            }
                                            Ok(Some(0)) => continue,
                                            Ok(Some(n)) => {
                                                st_inner.bytes_rx.fetch_add(n as u64, Ordering::Relaxed);
                                                let to_write: std::borrow::Cow<'_, [u8]> = if sess.wire_v2 {
                                                    match sess.c2w_decoder.push_wire_bytes(&rx_buf[..n]) {
                                                        Ok((decoded, ev)) => {
                                                            if let Some(ev) = ev {
                                                                if !sess.logged_v2_rx {
                                                                    sess.logged_v2_rx = true;
                                                                    info!(
                                                                        worker = %n_inner,
                                                                        "SYNRPC02 FP16 Wire Compression attivo (Worker RX): {} B -> {} B F16",
                                                                        ev.raw_bytes, ev.compressed_bytes
                                                                    );
                                                                }
                                                            }
                                                            sess.req_tracker.observe_tcp_bytes(&decoded, &mut sess.expected_rsps);
                                                            std::borrow::Cow::Owned(decoded)
                                                        }
                                                        Err(e) => {
                                                            warn!(worker = %n_inner, "SYNRPC02 decode error: {}", e);
                                                            clean_close = true;
                                                            break;
                                                        }
                                                    }
                                                } else {
                                                    std::borrow::Cow::Borrowed(&rx_buf[..n])
                                                };
                                                if !to_write.is_empty() && sess.rpc_w.write_all(&to_write).await.is_err() {
                                                    clean_close = true;
                                                    break;
                                                }
                                                sess.worker_rx_offset += n as u64;
                                            }
                                            Err(e) => {
                                                warn!(
                                                    worker = %n_inner,
                                                    session_id = session_id,
                                                    "QUIC recv error dal Master: {} -> sessione ggml-rpc mantenuta attiva per resume",
                                                    e
                                                );
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if clean_close {
                                let _ = sess.rpc_w.shutdown().await;
                                drop(sess_guard);
                                if worker_rpc_sessions().lock().await.remove(&session_id).is_some() {
                                    active_worker_rpc_sessions().fetch_sub(1, Ordering::SeqCst);
                                }
                            }
                            return;
                        }

                        // Fallback for raw non-session streams (e.g. unit tests)
                        let Ok(rpc_stream) = TcpStream::connect(&target_inner).await else {
                            warn!(worker = %n_inner, target = %target_inner, "Cannot connect to local ggml-rpc-server");
                            return;
                        };
                        rpc_stream.set_nodelay(true).ok();
                        let (mut rpc_r, mut rpc_w) = rpc_stream.into_split();
                        st_inner.bytes_rx.fetch_add(first_n as u64, Ordering::Relaxed);
                        if rpc_w.write_all(&rx_buf[..first_n]).await.is_err() {
                            return;
                        }

                        let st_rx = st_inner.clone();
                        let mut c2s = tokio::spawn(async move {
                            let mut buf = vec![0u8; CHUNK_SIZE];
                            while let Ok(Some(n)) = quic_recv.read(&mut buf).await {
                                if n == 0 {
                                    break;
                                }
                                st_rx.bytes_rx.fetch_add(n as u64, Ordering::Relaxed);
                                if rpc_w.write_all(&buf[..n]).await.is_err() {
                                    break;
                                }
                            }
                            let _ = rpc_w.shutdown().await;
                        });

                        let st_tx = st_inner.clone();
                        let mut s2c = tokio::spawn(async move {
                            use tokio::io::AsyncReadExt;
                            let mut buf = vec![0u8; CHUNK_SIZE];
                            while let Ok(n) = rpc_r.read(&mut buf).await {
                                if n == 0 {
                                    break;
                                }
                                st_tx.bytes_tx.fetch_add(n as u64, Ordering::Relaxed);
                                if quic_send.write_all(&buf[..n]).await.is_err() {
                                    break;
                                }
                            }
                            let _ = quic_send.finish();
                            let _ = quic_send.stopped().await;
                        });

                        tokio::select! {
                            _ = &mut c2s => { s2c.abort(); }
                            _ = &mut s2c => { c2s.abort(); }
                        }
                    });
                }
            });
        }
    });

    Ok((ticket, stats, hw_caps))
}

/// Measures upload throughput (MB/s) Master -> Worker over a fresh direct connection.
async fn measure_link_throughput(
    endpoint: &Endpoint,
    ticket: &RpcNodeTicket,
    payload_mb: usize,
) -> Result<f64, String> {
    let (conn, _) = connect_direct(endpoint, ticket, ALPN_LINK_PROBE).await?;
    let run = async {
        let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
        let chunk = vec![0xA5u8; CHUNK_SIZE];
        let total = payload_mb * 1024 * 1024;
        let t0 = Instant::now();
        let mut sent = 0usize;
        while sent < total {
            send.write_all(&chunk).await.map_err(|e| e.to_string())?;
            sent += CHUNK_SIZE;
        }
        send.finish().map_err(|e| e.to_string())?;
        let resp = recv.read_to_end(16).await.map_err(|e| e.to_string())?;
        let secs = t0.elapsed().as_secs_f64().max(1e-6);
        let acked = resp
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or(0);
        Ok::<f64, String>(acked as f64 / (1024.0 * 1024.0) / secs)
    };
    let res = tokio::time::timeout(Duration::from_secs(30), run)
        .await
        .map_err(|_| "throughput probe in timeout (30s)".to_string())?;
    conn.close(0u8.into(), b"link-probe-ok");
    res
}

/// Queries a remote Worker's GPU capabilities over a **direct** QUIC path, measuring the real
/// path RTT and (optionally) upload throughput. Fails if the worker is not directly reachable.
pub async fn probe_worker_hardware_iroh(
    ticket: &RpcNodeTicket,
    enable_wan_dht: bool,
) -> Result<GpuNodeCapabilities, String> {
    let (endpoint, owns_endpoint) =
        bind_direct_client_endpoint(enable_wan_dht, Some(ticket.endpoint_addr.id)).await?;

    let result = async {
        let (conn, path) = connect_direct(&endpoint, ticket, ALPN_HW_PROBE).await?;

        let t0 = Instant::now();
        let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
        send.write_all(b"PROBE").await.map_err(|e| e.to_string())?;
        send.finish().map_err(|e| e.to_string())?;
        let resp_bytes = recv
            .read_to_end(16 * 1024)
            .await
            .map_err(|e| e.to_string())?;
        let app_rtt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        conn.close(0u8.into(), b"probe-ok");

        let mut caps: GpuNodeCapabilities =
            serde_json::from_slice(&resp_bytes).map_err(|e| e.to_string())?;
        caps.measured_rtt_ms = if path.rtt_ms > 0.0 {
            path.rtt_ms
        } else {
            app_rtt_ms
        };
        caps.direct_udp_reachable = true;

        let probe_mb = std::env::var("CUTMESH_LINK_PROBE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_LINK_PROBE_MB);
        if probe_mb > 0 {
            match measure_link_throughput(&endpoint, ticket, probe_mb).await {
                Ok(mb_s) => caps.measured_link_mb_s = (mb_s * 10.0).round() / 10.0,
                Err(e) => warn!(worker = %caps.node_name, "Misura banda fallita: {}", e),
            }
        }

        info!(
            worker = %caps.node_name,
            "Percorso {} {} | RTT {:.1} ms | banda upload {:.1} MB/s",
            if path.is_direct { "DIRETTO UDP" } else { "QUIC" },
            path.remote_addr,
            caps.measured_rtt_ms,
            caps.measured_link_mb_s
        );
        Ok(caps)
    }
    .await;

    if owns_endpoint {
        endpoint.close().await;
    }
    result
}

/// Instructs a remote Worker over QUIC (`ALPN_PREWARM_SHARD`) to autonomously download its assigned
/// `.gguf` layer slice directly from Hugging Face into its local `ggml-rpc-server -c` cache directory
/// (`~/.cache/llama.cpp/rpc/<fnv1a_hash>`), so the Gateway only sends 8-byte FNV-1a hashes instead of raw weights.
pub async fn request_worker_prewarm_shard(
    ticket: &RpcNodeTicket,
    enable_wan_dht: bool,
    req: &PrewarmShardRequest,
) -> Result<PartialShardPullReport, String> {
    let (endpoint, owns_endpoint) =
        bind_direct_client_endpoint(enable_wan_dht, Some(ticket.endpoint_addr.id)).await?;
    let short_id = ticket.endpoint_addr.id.fmt_short().to_string();

    let result = async {
        let (conn, _path) = connect_direct(&endpoint, ticket, ALPN_PREWARM_SHARD).await?;
        let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
        let payload = serde_json::to_vec(req).map_err(|e| e.to_string())?;
        send.write_all(&payload).await.map_err(|e| e.to_string())?;
        send.finish().map_err(|e| e.to_string())?;

        let resp_bytes = tokio::time::timeout(Duration::from_secs(600), recv.read_to_end(64 * 1024))
            .await
            .map_err(|_| format!("Timeout 600s durante il pre-warm dei pesi sul worker {}", short_id))?
            .map_err(|e| e.to_string())?;
        conn.close(0u8.into(), b"prewarm-ok");

        let parsed: Result<PartialShardPullReport, String> =
            serde_json::from_slice(&resp_bytes).map_err(|e| e.to_string())?;
        parsed
    }
    .await;

    if owns_endpoint {
        endpoint.close().await;
    }
    result
}

fn sparse_hash_translation_map() -> &'static std::sync::RwLock<std::collections::HashMap<u64, u64>> {
    static MAP: std::sync::OnceLock<std::sync::RwLock<std::collections::HashMap<u64, u64>>> =
        std::sync::OnceLock::new();
    MAP.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// Registers `(sparse_hash, real_hash)` pairs so `spawn_client_iroh_rpc_bridge` transparently translates
/// `RPC_CMD_SET_TENSOR_HASH` requests emitted by a CPU-only Master reading a Sparse GGUF Skeleton.
pub fn register_sparse_hash_translations(pairs: impl IntoIterator<Item = (u64, u64)>) {
    if let Ok(mut guard) = sparse_hash_translation_map().write() {
        for (sparse, real) in pairs {
            guard.insert(sparse, real);
        }
    }
}

fn translate_sparse_hash_in_place(buf: &mut [u8]) {
    let Ok(guard) = sparse_hash_translation_map().read() else {
        return;
    };
    if guard.is_empty() {
        return;
    }

    // Walk contiguous ggml-rpc frames (`cmd: u8 | req_size: u64 LE | payload: req_size`) when framed,
    // so coalesced TX batches with multiple commands still translate every `RPC_CMD_SET_TENSOR_HASH` (cmd == 7).
    let mut pos = 0usize;
    let mut walked_frames = false;
    while pos + 9 <= buf.len() {
        let cmd = buf[pos];
        let req_size = u64::from_le_bytes(buf[pos + 1..pos + 9].try_into().unwrap_or([0u8; 8]));
        if req_size > 64 * 1024 * 1024 {
            break;
        }
        let Some(frame_end) = (pos + 9).checked_add(req_size as usize) else {
            break;
        };
        if frame_end > buf.len() {
            break;
        }
        walked_frames = true;
        if cmd == 7 && (150..=360).contains(&(9 + req_size)) && frame_end >= pos + 17 {
            let candidate = u64::from_le_bytes(buf[frame_end - 8..frame_end].try_into().unwrap_or([0u8; 8]));
            if let Some(&real_hash) = guard.get(&candidate) {
                buf[frame_end - 8..frame_end].copy_from_slice(&real_hash.to_le_bytes());
            }
        }
        pos = frame_end;
    }

    // Fallback for single-packet heuristics when a continuation slice lacks the initial header framing
    if !walked_frames {
        let n = buf.len();
        if (160..=360).contains(&n) && (buf[0] == 7 || n == 232 || n == 168 || n == 321) {
            if let Ok(tail_bytes) = <[u8; 8]>::try_from(&buf[n - 8..n]) {
                let candidate = u64::from_le_bytes(tail_bytes);
                if let Some(&real_hash) = guard.get(&candidate) {
                    buf[n - 8..n].copy_from_slice(&real_hash.to_le_bytes());
                }
            }
        }
    }
}

/// CLIENT SIDE:
/// Opens a local loopback bridge (`127.0.0.1:0`) for `llama-server --rpc`,
/// establishes a persistent **direct** QUIC connection to the Worker (relay disabled),
/// and multiplexes every `ggml-rpc` stream over QUIC bi-directional streams with automatic
/// connection healing and exact byte-offset session resume (`SYNRPC01`).
pub async fn spawn_client_iroh_rpc_bridge(
    ticket: RpcNodeTicket,
    enable_wan_dht: bool,
) -> Result<(String, Arc<TunnelAuditStats>), String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let local_bridge_addr = listener.local_addr().map_err(|e| e.to_string())?.to_string();
    let stats = Arc::new(TunnelAuditStats::default());
    let stats_clone = stats.clone();

    let (endpoint, _owns_endpoint) =
        bind_direct_client_endpoint(enable_wan_dht, Some(ticket.endpoint_addr.id)).await?;
    let (conn, path) = connect_direct(&endpoint, &ticket, ALPN_GGML_RPC).await?;
    info!(
        worker = %ticket.endpoint_addr.id.fmt_short(),
        "Bridge ggml-rpc {} -> percorso {} {} (RTT {:.1} ms)",
        local_bridge_addr,
        if path.is_direct { "DIRETTO UDP" } else { "RELAY" },
        path.remote_addr,
        path.rtt_ms
    );
    let quic_conn: Arc<Connection> = Arc::new(conn);
    stats.quic_connections.fetch_add(1, Ordering::Relaxed);
    spawn_direct_path_guard(
        quic_conn.clone(),
        ticket.endpoint_addr.id.fmt_short().to_string(),
        Some(ticket.clone()),
    );
    let active_conn: Arc<tokio::sync::Mutex<Arc<Connection>>> =
        Arc::new(tokio::sync::Mutex::new(quic_conn));
    let ticket_clone = ticket.clone();

    tokio::spawn(async move {
        let ep = endpoint;
        while let Ok((local_rpc, _)) = listener.accept().await {
            local_rpc.set_nodelay(true).ok();
            let conn_holder = active_conn.clone();
            let ep_inner = ep.clone();
            let tk_inner = ticket_clone.clone();
            let st = stats_clone.clone();
            let session_id = next_rpc_session_id();

            tokio::spawn(async move {
                let (mut loc_r, mut loc_w) = local_rpc.into_split();
                let use_v2 = wire_fp16_enabled();
                let stream_magic = if use_v2 { RPC_STREAM_MAGIC_V2 } else { RPC_STREAM_MAGIC };
                let mut client_tx_history = RollingHistory::new(RPC_SESSION_HISTORY_CAP);
                let mut client_rx_offset: u64 = 0;
                let mut c2w_framer = ClientGgmlFramer::new(use_v2);
                let mut w2c_decoder = WireStreamDecoder::new();
                let mut logged_rx_v2 = false;
                let mut tx_buf = vec![0u8; CHUNK_SIZE];
                let mut rx_buf = vec![0u8; CHUNK_SIZE];

                'session_loop: loop {
                    let mut conn_arc = {
                        let guard = conn_holder.lock().await;
                        guard.clone()
                    };
                    if conn_arc.close_reason().is_some() {
                        let mut guard = conn_holder.lock().await;
                        if guard.close_reason().is_some() {
                            match connect_direct(&ep_inner, &tk_inner, ALPN_GGML_RPC).await {
                                Ok((new_c, new_path)) => {
                                    let new_arc = Arc::new(new_c);
                                    st.quic_connections.fetch_add(1, Ordering::Relaxed);
                                    spawn_direct_path_guard(
                                        new_arc.clone(),
                                        tk_inner.endpoint_addr.id.fmt_short().to_string(),
                                        Some(tk_inner.clone()),
                                    );
                                    info!(
                                        worker = %tk_inner.endpoint_addr.id.fmt_short(),
                                        session_id = session_id,
                                        "Riconnessione QUIC ggml-rpc riuscita su percorso {} (RTT {:.1} ms)",
                                        new_path.remote_addr,
                                        new_path.rtt_ms
                                    );
                                    *guard = new_arc;
                                }
                                Err(e) => {
                                    warn!(
                                        worker = %tk_inner.endpoint_addr.id.fmt_short(),
                                        session_id = session_id,
                                        "Tentativo riconnessione QUIC ggml-rpc fallito: {} -> riprovo...",
                                        e
                                    );
                                    drop(guard);
                                    tokio::time::sleep(Duration::from_millis(400)).await;
                                    continue 'session_loop;
                                }
                            }
                        }
                        conn_arc = guard.clone();
                    }

                    let (mut quic_send, mut quic_recv) =
                        match tokio::time::timeout(Duration::from_secs(10), conn_arc.open_bi())
                            .await
                        {
                            Ok(Ok(bi)) => bi,
                            _ => {
                                warn!(
                                    session_id = session_id,
                                    "Failed opening multiplexed QUIC bi-stream to worker -> forcing reconnect"
                                );
                                conn_arc.close(1u8.into(), b"open-bi-failed");
                                tokio::time::sleep(Duration::from_millis(300)).await;
                                continue 'session_loop;
                            }
                        };
                    st.quic_bi_streams_opened.fetch_add(1, Ordering::Relaxed);

                    let mut hdr = [0u8; 24];
                    hdr[0..8].copy_from_slice(stream_magic);
                    hdr[8..16].copy_from_slice(&session_id.to_le_bytes());
                    hdr[16..24].copy_from_slice(&client_rx_offset.to_le_bytes());
                    if write_quic_chunked(&mut quic_send, &hdr).await.is_err() {
                        let _ = quic_send.reset(1u32.into());
                        conn_arc.close(1u8.into(), b"hdr-write-failed");
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        continue 'session_loop;
                    }

                    let mut reply = [0u8; 16];
                    match tokio::time::timeout(
                        Duration::from_secs(10),
                        quic_recv.read_exact(&mut reply),
                    )
                    .await
                    {
                        Ok(Ok(())) if &reply[0..8] == stream_magic => {}
                        _ => {
                            warn!(
                                session_id = session_id,
                                "Handshake {} dal worker fallito o in timeout -> forcing reconnect",
                                std::str::from_utf8(stream_magic).unwrap_or("SYNRPC")
                            );
                            let _ = quic_send.reset(1u32.into());
                            conn_arc.close(1u8.into(), b"handshake-timeout");
                            tokio::time::sleep(Duration::from_millis(300)).await;
                            continue 'session_loop;
                        }
                    }
                    let worker_rx_offset =
                        u64::from_le_bytes(reply[8..16].try_into().unwrap_or([0u8; 8]));

                    if client_tx_history.total_len() > worker_rx_offset {
                        if let Some(replay_bytes) = client_tx_history.slice_from(worker_rx_offset) {
                            if !replay_bytes.is_empty() {
                                info!(
                                    session_id = session_id,
                                    worker_rx_offset = worker_rx_offset,
                                    replay_len = replay_bytes.len(),
                                    "Replay {} byte di tensori ggml-rpc verso il Worker dopo riconnessione QUIC",
                                    replay_bytes.len()
                                );
                                if write_quic_chunked(&mut quic_send, &replay_bytes)
                                    .await
                                    .is_err()
                                {
                                    let _ = quic_send.reset(1u32.into());
                                    conn_arc.close(1u8.into(), b"replay-write-failed");
                                    tokio::time::sleep(Duration::from_millis(300)).await;
                                    continue 'session_loop;
                                }
                            }
                        }
                    }

                    let mut client_eof = false;
                    loop {
                        tokio::select! {
                            read_res = loc_r.read(&mut tx_buf) => {
                                match read_res {
                                    Ok(0) => {
                                        client_eof = true;
                                        let _ = quic_send.finish();
                                        let _ = tokio::time::timeout(Duration::from_secs(2), quic_send.stopped()).await;
                                        break;
                                    }
                                    Ok(n) => {
                                        let to_send: std::borrow::Cow<'_, [u8]> = if use_v2 {
                                            let (wire_bytes, ev) = {
                                                let map_guard = sparse_hash_translation_map().read().ok();
                                                c2w_framer.push_tcp_bytes(
                                                    &tx_buf[..n],
                                                    map_guard.as_deref(),
                                                )
                                            };
                                            if let Some(ev) = ev {
                                                info!(
                                                    worker = %tk_inner.endpoint_addr.id.fmt_short(),
                                                    "SYNRPC02 FP16 Wire Compression attivo (Client TX): {} B -> {} B F16 (2.0x)",
                                                    ev.raw_bytes, ev.compressed_bytes
                                                );
                                            }
                                            std::borrow::Cow::Owned(wire_bytes)
                                        } else {
                                            translate_sparse_hash_in_place(&mut tx_buf[..n]);
                                            std::borrow::Cow::Borrowed(&tx_buf[..n])
                                        };
                                        if !to_send.is_empty() {
                                            client_tx_history.append(&to_send);
                                            st.bytes_tx.fetch_add(to_send.len() as u64, Ordering::Relaxed);
                                            if let Err(e) = write_quic_chunked(&mut quic_send, &to_send).await {
                                                warn!(
                                                    session_id = session_id,
                                                    "QUIC send error verso il Worker ({}) -> riconnessione trasparente sessione ggml-rpc",
                                                    e
                                                );
                                                let _ = quic_send.reset(1u32.into());
                                                conn_arc.close(1u8.into(), b"send-error-reconnect");
                                                break;
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        client_eof = true;
                                        let _ = quic_send.finish();
                                        break;
                                    }
                                }
                            }
                            quic_res = quic_recv.read(&mut rx_buf) => {
                                match quic_res {
                                    Ok(None) => {
                                        warn!(
                                            session_id = session_id,
                                            "QUIC stream chiuso dal Worker -> riconnessione trasparente sessione ggml-rpc"
                                        );
                                        let _ = quic_send.reset(1u32.into());
                                        break;
                                    }
                                    Ok(Some(0)) => continue,
                                    Ok(Some(n)) => {
                                        st.bytes_rx.fetch_add(n as u64, Ordering::Relaxed);
                                        let to_write: std::borrow::Cow<'_, [u8]> = if use_v2 {
                                            match w2c_decoder.push_wire_bytes(&rx_buf[..n]) {
                                                Ok((decoded, ev)) => {
                                                    if let Some(ev) = ev {
                                                        if !logged_rx_v2 {
                                                            logged_rx_v2 = true;
                                                            info!(
                                                                worker = %tk_inner.endpoint_addr.id.fmt_short(),
                                                                "SYNRPC02 FP16 Wire Compression attivo (Client RX): {} B -> {} B F16 (2.0x)",
                                                                ev.raw_bytes, ev.compressed_bytes
                                                            );
                                                        }
                                                    }
                                                    std::borrow::Cow::Owned(decoded)
                                                }
                                                Err(e) => {
                                                    warn!(session_id = session_id, "SYNRPC02 client RX decode error: {}", e);
                                                    client_eof = true;
                                                    let _ = quic_send.finish();
                                                    break;
                                                }
                                            }
                                        } else {
                                            std::borrow::Cow::Borrowed(&rx_buf[..n])
                                        };
                                        if !to_write.is_empty() && loc_w.write_all(&to_write).await.is_err() {
                                            client_eof = true;
                                            let _ = quic_send.finish();
                                            break;
                                        }
                                        client_rx_offset += n as u64;
                                    }
                                    Err(e) => {
                                        warn!(
                                            session_id = session_id,
                                            "QUIC recv error dal Worker ({}) -> riconnessione trasparente sessione ggml-rpc",
                                            e
                                        );
                                        let _ = quic_send.reset(1u32.into());
                                        conn_arc.close(1u8.into(), b"recv-error-reconnect");
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if client_eof {
                        let _ = loc_w.shutdown().await;
                        break 'session_loop;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            });
        }
    });

    Ok((local_bridge_addr, stats))
}

/// Opens a fast local loopback proxy (`127.0.0.1:0 -> 127.0.0.1:50052`) on the Master GPU node
/// that applies `sparse_hash_translation_map()` to `RPC_CMD_SET_TENSOR_HASH` packets so `llama-server`
/// running locally on the Master GPU node hits 100% of existing `/data/cutmesh/cache/rpc/<real_hash>` files.
pub async fn spawn_local_hash_translating_rpc_proxy(target_rpc_addr: &str) -> Result<String, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let local_proxy_addr = listener.local_addr().map_err(|e| e.to_string())?.to_string();
    let target = target_rpc_addr.to_string();

    tokio::spawn(async move {
        while let Ok((client_stream, _)) = listener.accept().await {
            client_stream.set_nodelay(true).ok();
            let t_inner = target.clone();
            tokio::spawn(async move {
                let Ok(rpc_stream) = TcpStream::connect(&t_inner).await else {
                    return;
                };
                rpc_stream.set_nodelay(true).ok();
                let (mut c_r, mut c_w) = client_stream.into_split();
                let (mut s_r, mut s_w) = rpc_stream.into_split();

                let mut c2s = tokio::spawn(async move {
                    let mut buf = vec![0u8; CHUNK_SIZE];
                    while let Ok(n) = c_r.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        translate_sparse_hash_in_place(&mut buf[..n]);
                        if s_w.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                    let _ = s_w.shutdown().await;
                });

                let mut s2c = tokio::spawn(async move {
                    let mut buf = vec![0u8; CHUNK_SIZE];
                    while let Ok(n) = s_r.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        if c_w.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                    let _ = c_w.shutdown().await;
                });

                tokio::select! {
                    _ = &mut c2s => { s2c.abort(); }
                    _ = &mut s2c => { c2s.abort(); }
                }
            });
        }
    });

    Ok(local_proxy_addr)
}

/// Spawns `llama-server` directly on the Elected Stage-0 Master GPU Worker:
/// - Uses local `ggml-rpc-server` (`127.0.0.1:50052`) via `spawn_local_hash_translating_rpc_proxy` as `RPC0` (0.01 ms loopback)
/// - Connects P2P via `spawn_client_iroh_rpc_bridge` to secondary workers (`RPC1`, `RPC2`, ...)
/// - Keeps `output.weight`, `output_norm.weight`, and the Sampler 100% on the Master GPU node (0 MB WAN logit traffic)
pub async fn spawn_local_master_llama_server(
    req: SpawnMasterRigRequest,
    local_rpc_addr: &str,
    enable_wan_dht: bool,
) -> Result<SpawnMasterRigResponse, String> {
    let _spawn_guard = master_spawn_lock().lock().await;
    let port = if req.port == 0 { 8089 } else { req.port };

    let worker_dev = active_worker_device();
    let use_native_cuda = std::env::var("CUTMESH_MASTER_LOCAL_RPC").as_deref() != Ok("1")
        && worker_dev.to_ascii_uppercase().starts_with("CUDA");

    let already_running_same_spec = {
        let spec_matches = active_master_llama_spec()
            .lock()
            .ok()
            .and_then(|g| {
                g.as_ref().map(|s| {
                    s.port == port
                        && s.tensor_split_ratios == req.tensor_split_ratios
                        && s.peer_count == req.peer_worker_tickets.len()
                })
            })
            .unwrap_or(false);
        let child_alive = active_master_llama_child()
            .lock()
            .ok()
            .and_then(|mut g| g.as_mut().map(|(c, _)| matches!(c.try_wait(), Ok(None))))
            .unwrap_or(false);
        spec_matches && child_alive
    };

    let (devices_vec, splits_vec) = if already_running_same_spec {
        info!(
            "Master-on-GPU llama-server già attivo o in caricamento su 127.0.0.1:{} (peer: {}, split: {:?}) -> riutilizzo istanza senza riavviare",
            port,
            req.peer_worker_tickets.len(),
            req.tensor_split_ratios
        );
        let mut devs = if use_native_cuda {
            vec![worker_dev.clone()]
        } else {
            vec!["RPC0".to_string()]
        };
        for idx in 0..req.peer_worker_tickets.len() {
            devs.push(if use_native_cuda {
                format!("RPC{}", idx)
            } else {
                format!("RPC{}", idx + 1)
            });
        }
        let sp: Vec<String> = req
            .tensor_split_ratios
            .iter()
            .map(|r| r.to_string())
            .collect();
        (devs, sp)
    } else {
        if let Ok(mut spec_guard) = active_master_llama_spec().lock() {
            *spec_guard = None;
        }
        if let Ok(mut guard) = active_master_llama_child().lock() {
            if let Some((mut old_child, _)) = guard.take() {
                let _ = old_child.kill();
                let _ = old_child.wait();
            }
        }

        let mut ordered_bridges = Vec::new();
        let mut devices_vec = Vec::new();
        let primary_buft = if use_native_cuda {
            devices_vec.push(worker_dev.clone());
            worker_dev.clone()
        } else {
            let local_rpc0_bridge = spawn_local_hash_translating_rpc_proxy(local_rpc_addr).await?;
            ordered_bridges.push(local_rpc0_bridge.clone());
            devices_vec.push("RPC0".to_string());
            format!("RPC0[{}]", local_rpc0_bridge)
        };

        for (idx, raw_tk) in req.peer_worker_tickets.iter().enumerate() {
            let parsed = RpcNodeTicket::parse(raw_tk)?;
            let (peer_bridge, _stats) = spawn_client_iroh_rpc_bridge(parsed, enable_wan_dht).await?;
            ordered_bridges.push(peer_bridge);
            devices_vec.push(if use_native_cuda {
                format!("RPC{}", idx)
            } else {
                format!("RPC{}", idx + 1)
            });
        }

        let splits_vec: Vec<String> = if req.tensor_split_ratios.len() == devices_vec.len() {
            req.tensor_split_ratios.iter().map(|r| r.to_string()).collect()
        } else {
            vec!["100".to_string()]
        };

        let model_policy = req
            .policy_json
            .as_deref()
            .and_then(|raw| serde_json::from_str::<crate::models::ModelPolicy>(raw).ok())
            .unwrap_or_else(|| crate::models::ModelPolicy::resolve(&req.model_url));
        let stage0_pinned_layers = model_policy.stage0_pinned_layers();

        let model_url_for_hdr = req.model_url.clone();
        let (hdr, _model_name) = tokio::task::spawn_blocking(move || {
            fetch_or_parse_gguf_header(&model_url_for_hdr)
        })
        .await
        .map_err(|e| format!("header parse task panicked: {}", e))??;
        let fname = req
            .model_url
            .rsplit('/')
            .next()
            .unwrap_or("model.gguf")
            .split('?')
            .next()
            .unwrap_or("model.gguf")
            .to_string();

        let rpc_cache = default_ggml_rpc_cache_dir();
        if !stage0_pinned_layers.is_empty() {
            for &mtp_l in &stage0_pinned_layers {
                let url_mtp = req.model_url.clone();
                let cache_mtp = rpc_cache.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    pull_rig_shard_to_rpc_cache(&url_mtp, &cache_mtp, mtp_l, mtp_l, false)
                })
                .await;
            }
        }

        let mut local_stage0_layers = std::collections::BTreeSet::new();
        if use_native_cuda {
            let total_hdr_layers = hdr.total_layers;
            if req.peer_worker_tickets.is_empty() {
                for l in 0..total_hdr_layers {
                    local_stage0_layers.insert(l);
                }
            } else {
                let trunk = model_policy.pipeline_layers(total_hdr_layers);
                let r0 = req.tensor_split_ratios.first().copied().unwrap_or(100) as f64;
                let rsum = req.tensor_split_ratios.iter().copied().sum::<u32>().max(1) as f64;
                let stage0_end_excl = (((trunk as f64) * (r0 / rsum)).ceil() as u16)
                    .saturating_add(2)
                    .min(trunk);
                if stage0_end_excl > 0 {
                    let url_s0 = req.model_url.clone();
                    let cache_s0 = rpc_cache.clone();
                    let end_incl = stage0_end_excl.saturating_sub(1);
                    let _ = tokio::task::spawn_blocking(move || {
                        pull_rig_shard_to_rpc_cache(&url_s0, &cache_s0, 0, end_incl, false)
                    })
                    .await;
                }
                for l in 0..stage0_end_excl {
                    local_stage0_layers.insert(l);
                }
                for &mtp_l in &stage0_pinned_layers {
                    local_stage0_layers.insert(mtp_l);
                }
            }
        }

        let data_dir =
            std::env::var("CUTMESH_DATA_DIR").unwrap_or_else(|_| "/data/cutmesh".to_string());
        let models_dir = PathBuf::from(&data_dir).join("models");
        let _ = std::fs::create_dir_all(&models_dir);
        let local_cache_file = models_dir.join(&fname);
        let manifest_file = models_dir.join(format!("{}.hashes.json", fname));

        let url_c = req.model_url.clone();
        let hdr_c = hdr.clone();
        let manifest_c = manifest_file.clone();
        let cache_file_c = local_cache_file.clone();
        let reported_c = req.worker_reported_hashes.clone();
        let local_layers_c = local_stage0_layers.clone();

        let (translations, dl_bytes) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<(u64, u64)>, u64), String> {
                let tr =
                    ensure_tensor_hash_manifest(&url_c, &hdr_c, &manifest_c, &reported_c)?;
                let dl = build_sparse_master_gguf_for_local_layers(
                    &url_c,
                    &hdr_c,
                    &cache_file_c,
                    &local_layers_c,
                )?;
                Ok((tr, dl))
            })
            .await
            .map_err(|e| format!("master sparse gguf task panicked: {}", e))??;

        register_sparse_hash_translations(translations);
        if use_native_cuda {
            info!(
                "Master-on-GPU NATIVE {} attivo (zero loopback RPC0, zero copie logits TCP, local_layers={}, scaricati {:.1} MB, devices={:?}, split={:?})",
                worker_dev,
                local_stage0_layers.len(),
                (dl_bytes as f64) / (1024.0 * 1024.0),
                devices_vec,
                splits_vec
            );
        } else {
            info!(
                "Master-on-GPU Sparse GGUF pronto in {:?} (scaricati {:.1} MB, token_embd + output.weight copiati da cache NVMe locale)",
                local_cache_file,
                (dl_bytes as f64) / (1024.0 * 1024.0)
            );
        }

        let llama_dir = if PathBuf::from("/opt/cutmesh/llamacpp").exists() {
            PathBuf::from("/opt/cutmesh/llamacpp")
        } else {
            PathBuf::from("llama-bin")
        };
        let server_bin = if cfg!(target_os = "windows") {
            llama_dir.join("llama-server.exe")
        } else {
            llama_dir.join("llama-server")
        };

        let cmd_args = model_policy.build_llama_server_args(&crate::models::LlamaServerLaunchParams {
            model_path: &local_cache_file,
            ordered_bridges: &ordered_bridges,
            devices_vec: &devices_vec,
            splits_vec: &splits_vec,
            primary_buft: &primary_buft,
            requested_ctx_size: req.ctx_size,
            requested_ubatch_size: req.ubatch_size,
            port,
        });
        info!(
            "Applicata policy modello '{}' ({}) -> spec={} (enabled={}), group_align={}, mtp_layers={:?}",
            model_policy.model_id,
            model_policy.display_name,
            model_policy.speculative.spec_type,
            model_policy.speculative.enabled,
            model_policy.sharding.layer_group_alignment,
            model_policy.sharding.mtp_layers
        );

        let current_ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let ld_env = format!(
            "{}:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:{}",
            llama_dir.to_string_lossy(),
            current_ld
        );

        let launch_spec = MasterLlamaLaunchSpec {
            server_bin: server_bin.clone(),
            ld_env,
            manifest_file: manifest_file.to_string_lossy().to_string(),
            cmd_args,
            port,
            tensor_split_ratios: req.tensor_split_ratios.clone(),
            peer_count: req.peer_worker_tickets.len(),
        };

        let child = spawn_master_llama_process(&launch_spec)?;

        if let Ok(mut guard) = active_master_llama_child().lock() {
            *guard = Some((child, Instant::now()));
        }
        if let Ok(mut spec_guard) = active_master_llama_spec().lock() {
            *spec_guard = Some(launch_spec);
        }
        active_master_http_port().store(port, Ordering::Relaxed);
        (devices_vec, splits_vec)
    };

    let health_addr = format!("127.0.0.1:{}", port);
    for _ in 0..600 {
        if let Ok(mut stream) = TcpStream::connect(&health_addr).await {
            let req_str = format!(
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
                port
            );
            if stream.write_all(req_str.as_bytes()).await.is_ok() {
                let mut buf = vec![0u8; 1024];
                if let Ok(n) = stream.read(&mut buf).await {
                    let s = String::from_utf8_lossy(&buf[..n]);
                    if s.contains("200 OK") && s.contains("\"ok\"") {
                        info!(
                            "Master-on-GPU llama-server OPERATIONAL (200 OK) su 127.0.0.1:{} (devices={}, split={})",
                            port,
                            devices_vec.join(","),
                            splits_vec.join(",")
                        );
                        return Ok(SpawnMasterRigResponse {
                            ok: true,
                            message: format!(
                                "Master-on-GPU llama-server READY on 127.0.0.1:{} ({})",
                                port,
                                devices_vec.join(",")
                            ),
                            listen_port: port,
                        });
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    Err("Timeout waiting for Master-on-GPU llama-server /health to become ready".to_string())
}

/// Instructs the Elected Stage-0 Master GPU Worker over `ALPN_MASTER_RIG` to spawn `llama-server`
/// directly on the GPU node and connect P2P to the remaining Worker GPUs.
pub async fn request_worker_spawn_master_rig(
    ticket: &RpcNodeTicket,
    enable_wan_dht: bool,
    req: &SpawnMasterRigRequest,
) -> Result<SpawnMasterRigResponse, String> {
    let (endpoint, owns_endpoint) =
        bind_direct_client_endpoint(enable_wan_dht, Some(ticket.endpoint_addr.id)).await?;
    let short_id = ticket.endpoint_addr.id.fmt_short().to_string();
    let payload = serde_json::to_vec(req).map_err(|e| e.to_string())?;

    let mut last_err = String::new();
    for attempt in 1..=4u32 {
        let attempt_res: Result<SpawnMasterRigResponse, String> = async {
            let (conn, _path) = connect_direct(&endpoint, ticket, ALPN_MASTER_RIG).await?;
            let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
            send.write_all(&payload).await.map_err(|e| e.to_string())?;
            send.finish().map_err(|e| e.to_string())?;

            let resp_bytes =
                tokio::time::timeout(Duration::from_secs(360), recv.read_to_end(64 * 1024))
                    .await
                    .map_err(|_| {
                        format!(
                            "Timeout 360s durante l'avvio di llama-server sul Master GPU {}",
                            short_id
                        )
                    })?
                    .map_err(|e| e.to_string())?;
            conn.close(0u8.into(), b"master-rig-ok");

            let json_slice = resp_bytes
                .iter()
                .position(|&b| b == b'{')
                .map(|i| &resp_bytes[i..])
                .unwrap_or(resp_bytes.as_slice());

            let parsed: SpawnMasterRigResponse =
                serde_json::from_slice(json_slice).map_err(|e| e.to_string())?;
            if !parsed.ok {
                return Err(parsed.message);
            }
            Ok(parsed)
        }
        .await;

        match attempt_res {
            Ok(resp) => {
                if owns_endpoint {
                    endpoint.close().await;
                }
                return Ok(resp);
            }
            Err(e) => {
                warn!(
                    master = %short_id,
                    attempt = attempt,
                    "Tentativo {}/4 avvio Master-on-GPU interrotto ({}) -> riconnessione trasparente allo stato di caricamento...",
                    attempt,
                    e
                );
                last_err = e;
                if attempt < 4 {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
        }
    }

    if owns_endpoint {
        endpoint.close().await;
    }
    Err(last_err)
}

/// Binds `127.0.0.1:<local_port>` on the lightweight Control Plane VPS and tunnels all OpenAI HTTP
/// requests (`/v1/chat/completions`, `/health`, `/props`) over `ALPN_OPENAI_HTTP` to the `llama-server`
/// running locally on the Elected Stage-0 Master GPU Worker (`0 MB` WAN logit traffic).
pub async fn spawn_client_iroh_http_bridge(
    ticket: RpcNodeTicket,
    local_port: u16,
    enable_wan_dht: bool,
) -> Result<String, String> {
    let bind_addr = format!("127.0.0.1:{}", local_port);
    let listener = TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| format!("Impossibile aprire il gateway HTTP locale {}: {}", bind_addr, e))?;
    let local_bridge_addr = listener.local_addr().map_err(|e| e.to_string())?.to_string();

    let (endpoint, _owns_endpoint) =
        bind_direct_client_endpoint(enable_wan_dht, Some(ticket.endpoint_addr.id)).await?;
    let (conn, path) = connect_direct(&endpoint, &ticket, ALPN_OPENAI_HTTP).await?;
    info!(
        master_gpu = %ticket.endpoint_addr.id.fmt_short(),
        "Gateway OpenAI HTTP Tunnel {} -> Master GPU {} (RTT {:.1} ms, 0 MB WAN logits)",
        local_bridge_addr,
        path.remote_addr,
        path.rtt_ms
    );
    let active_conn: Arc<tokio::sync::Mutex<Arc<Connection>>> =
        Arc::new(tokio::sync::Mutex::new(Arc::new(conn)));
    let ticket_clone = ticket.clone();

    tokio::spawn(async move {
        let ep = endpoint;
        while let Ok((local_tcp, _)) = listener.accept().await {
            local_tcp.set_nodelay(true).ok();
            let conn_holder = active_conn.clone();
            let ep_inner = ep.clone();
            let tk_inner = ticket_clone.clone();

            tokio::spawn(async move {
                let mut conn_arc = {
                    let guard = conn_holder.lock().await;
                    guard.clone()
                };
                if conn_arc.close_reason().is_some() {
                    let mut guard = conn_holder.lock().await;
                    if guard.close_reason().is_some() {
                        if let Ok((new_c, _)) = connect_direct(&ep_inner, &tk_inner, ALPN_OPENAI_HTTP).await {
                            *guard = Arc::new(new_c);
                        }
                    }
                    conn_arc = guard.clone();
                }

                let Ok((mut quic_send, mut quic_recv)) = conn_arc.open_bi().await else {
                    return;
                };

                let (mut loc_r, mut loc_w) = local_tcp.into_split();

                let c2s = tokio::spawn(async move {
                    let mut buf = vec![0u8; CHUNK_SIZE];
                    while let Ok(n) = loc_r.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        if quic_send.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                    let _ = quic_send.finish();
                });
                let c2s_abort = c2s.abort_handle();

                let s2c = tokio::spawn(async move {
                    let mut buf = vec![0u8; CHUNK_SIZE];
                    while let Ok(Some(n)) = quic_recv.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        if loc_w.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                        let _ = loc_w.flush().await;
                    }
                    let _ = loc_w.shutdown().await;
                    c2s_abort.abort();
                });

                let _ = tokio::join!(c2s, s2c);
            });
        }
    });

    Ok(local_bridge_addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_history_ring_wraparound_and_replay() {
        let mut rh = RollingHistory::new(10);
        rh.append(b"012345");
        assert_eq!(rh.total_len(), 6);
        assert_eq!(rh.slice_from(2).unwrap(), b"2345");

        rh.append(b"6789ABCD"); // total=14, cap=10 -> keeps "456789ABCD" (offset 4..14)
        assert_eq!(rh.total_len(), 14);
        assert!(rh.slice_from(3).is_none());
        assert_eq!(rh.slice_from(4).unwrap(), b"456789ABCD");
        assert_eq!(rh.slice_from(10).unwrap(), b"ABCD");
        assert_eq!(rh.slice_from(14).unwrap(), b"");

        rh.append(b"XYZ0123456789"); // 13 bytes >= cap 10 -> keeps last 10 "0123456789"
        assert_eq!(rh.total_len(), 27);
        assert_eq!(rh.slice_from(17).unwrap(), b"0123456789");
    }

    #[test]
    fn test_translate_sparse_hash_coalesced_frames() {
        let sparse1 = 0x1111_2222_3333_4444u64;
        let real1 = 0xAAAA_BBBB_CCCC_DDDDu64;
        let sparse2 = 0x5555_6666_7777_8888u64;
        let real2 = 0x9999_0000_EEEE_FFFFu64;
        register_sparse_hash_translations([(sparse1, real1), (sparse2, real2)]);

        let make_frame = |hash: u64| {
            let payload_len = 159usize; // 9 + 159 = 168 total bytes
            let mut f = vec![0u8; 9 + payload_len];
            f[0] = 7; // RPC_CMD_SET_TENSOR_HASH
            f[1..9].copy_from_slice(&(payload_len as u64).to_le_bytes());
            let end = f.len();
            f[end - 8..end].copy_from_slice(&hash.to_le_bytes());
            f
        };

        let mut coalesced = make_frame(sparse1);
        coalesced.extend_from_slice(&make_frame(sparse2));
        translate_sparse_hash_in_place(&mut coalesced);

        let out1 = u64::from_le_bytes(coalesced[160..168].try_into().unwrap());
        let out2 = u64::from_le_bytes(coalesced[328..336].try_into().unwrap());
        assert_eq!(out1, real1);
        assert_eq!(out2, real2);
    }
}
