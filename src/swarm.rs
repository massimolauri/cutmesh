use iroh::{
    endpoint::{presets, Connection},
    Endpoint, EndpointAddr, SecretKey,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

/// ALPN for querying a Pre-Warmed Swarm Pod's live load, free continuous-batching slots, and model info.
pub const ALPN_SWARM_HEALTH: &[u8] = b"cutmesh/swarm-health/1";

/// ALPN for zero-GGUF Lightweight Clients to send concurrent OpenAI requests over Iroh Direct-QUIC
/// to a Pre-Warmed Swarm Pod (running `llama-server --parallel N --cont-batching`).
pub const ALPN_OPENAI_SWARM: &[u8] = b"cutmesh/openai-swarm/1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPodStatus {
    pub pod_name: String,
    pub endpoint_id: String,
    pub model_name: String,
    pub max_parallel_slots: usize,
    pub active_requests: usize,
    pub total_requests_served: u64,
    pub healthy: bool,
    pub measured_rtt_ms: f64,
}

#[derive(Debug, Default)]
pub struct SwarmPodMetrics {
    pub active_requests: AtomicUsize,
    pub total_requests_served: AtomicU64,
    pub total_bytes_in: AtomicU64,
    pub total_bytes_out: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct SwarmPodTicket {
    pub pod_name: String,
    pub endpoint_addr: EndpointAddr,
}

impl SwarmPodTicket {
    pub fn to_string_ticket(&self) -> String {
        let json = serde_json::to_vec(&self.endpoint_addr).unwrap_or_default();
        format!("pod://{}:{}", self.pod_name, hex::encode(json))
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let rest = s
            .strip_prefix("pod://")
            .ok_or_else(|| "Pod ticket must start with pod://".to_string())?;
        let (name, hex_part) = rest
            .split_once(':')
            .ok_or_else(|| "Invalid pod ticket format".to_string())?;
        let bytes = hex::decode(hex_part).map_err(|e| e.to_string())?;
        let endpoint_addr: EndpointAddr =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        Ok(Self {
            pod_name: name.to_string(),
            endpoint_addr,
        })
    }
}

/// POD SIDE:
/// Exposes a Pre-Warmed `llama-server.exe` (`--parallel N --cont-batching`) onto the Iroh Direct-QUIC P2P network
/// so that any number of Zero-GGUF Clients can send concurrent requests without ever downloading or reloading weights.
pub async fn spawn_prewarmed_swarm_pod_endpoint(
    pod_name: &str,
    model_name: &str,
    local_llama_http_port: u16,
    max_parallel_slots: usize,
    seed_byte: u8,
) -> Result<(SwarmPodTicket, Arc<SwarmPodMetrics>), String> {
    let mut seed = [0u8; 32];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = seed_byte.wrapping_add(i as u8).wrapping_mul(71);
    }
    let secret_key = SecretKey::from_bytes(&seed);

    let endpoint = Endpoint::builder(presets::Minimal)
        .secret_key(secret_key)
        .alpns(vec![ALPN_SWARM_HEALTH.to_vec(), ALPN_OPENAI_SWARM.to_vec()])
        .clear_ip_transports()
        .bind_addr("127.0.0.1:0".parse::<SocketAddr>().unwrap())
        .map_err(|e| e.to_string())?
        .bind()
        .await
        .map_err(|e| e.to_string())?;

    let ep_id = endpoint.id().to_string();
    let ticket = SwarmPodTicket {
        pod_name: pod_name.to_string(),
        endpoint_addr: endpoint.addr(),
    };
    let metrics = Arc::new(SwarmPodMetrics::default());
    let metrics_clone = metrics.clone();
    let p_name = pod_name.to_string();
    let m_name = model_name.to_string();
    let llama_addr = format!("127.0.0.1:{}", local_llama_http_port);

    tokio::spawn(async move {
        let _keep_ep = endpoint.clone();
        while let Some(incoming) = endpoint.accept().await {
            let Ok(connecting) = incoming.accept() else {
                continue;
            };
            let met = metrics_clone.clone();
            let pn = p_name.clone();
            let mn = m_name.clone();
            let eid = ep_id.clone();
            let target_http = llama_addr.clone();

            tokio::spawn(async move {
                let Ok(conn) = connecting.await else {
                    return;
                };
                let alpn = conn.alpn().to_vec();

                if alpn.as_slice() == ALPN_SWARM_HEALTH {
                    if let Ok((mut send, mut recv)) = conn.accept_bi().await {
                        let _ = recv.read_to_end(64).await;
                        let status = SwarmPodStatus {
                            pod_name: pn,
                            endpoint_id: eid,
                            model_name: mn,
                            max_parallel_slots,
                            active_requests: met.active_requests.load(Ordering::Relaxed),
                            total_requests_served: met
                                .total_requests_served
                                .load(Ordering::Relaxed),
                            healthy: true,
                            measured_rtt_ms: 0.0,
                        };
                        let payload = serde_json::to_vec(&status).unwrap_or_default();
                        let _ = send.write_all(&payload).await;
                        let _ = send.finish();
                        let _ = send.stopped().await;
                    }
                    conn.closed().await;
                    return;
                }

                // Multiplex concurrent OpenAI HTTP requests over Iroh QUIC streams onto pre-warmed llama-server
                while let Ok((mut quic_send, mut quic_recv)) = conn.accept_bi().await {
                    let met_inner = met.clone();
                    let http_target = target_http.clone();

                    tokio::spawn(async move {
                        met_inner.active_requests.fetch_add(1, Ordering::Relaxed);
                        let req_bytes = quic_recv.read_to_end(4 * 1024 * 1024).await.unwrap_or_default();
                        met_inner
                            .total_bytes_in
                            .fetch_add(req_bytes.len() as u64, Ordering::Relaxed);

                        if let Ok(mut llama_tcp) = TcpStream::connect(&http_target).await {
                            llama_tcp.set_nodelay(true).ok();
                            if llama_tcp.write_all(&req_bytes).await.is_ok() {
                                let mut resp_bytes = Vec::new();
                                let _ = llama_tcp.read_to_end(&mut resp_bytes).await;
                                met_inner
                                    .total_bytes_out
                                    .fetch_add(resp_bytes.len() as u64, Ordering::Relaxed);
                                let _ = quic_send.write_all(&resp_bytes).await;
                            }
                        }

                        let _ = quic_send.finish();
                        let _ = quic_send.stopped().await;
                        met_inner.active_requests.fetch_sub(1, Ordering::Relaxed);
                        met_inner
                            .total_requests_served
                            .fetch_add(1, Ordering::Relaxed);
                    });
                }
            });
        }
    });

    Ok((ticket, metrics))
}

/// Probes a Swarm Pod's health, active continuous-batching slots, and RTT over Iroh Direct-QUIC.
pub async fn probe_swarm_pod(
    ep: &Endpoint,
    ticket: &SwarmPodTicket,
) -> Result<SwarmPodStatus, String> {
    let t0 = Instant::now();
    let conn = ep
        .connect(ticket.endpoint_addr.clone(), ALPN_SWARM_HEALTH)
        .await
        .map_err(|e| e.to_string())?;
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| e.to_string())?;
    send.write_all(b"HEALTH").await.map_err(|e| e.to_string())?;
    send.finish().map_err(|e| e.to_string())?;
    let bytes = recv.read_to_end(16 * 1024).await.map_err(|e| e.to_string())?;
    let rtt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut status: SwarmPodStatus = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    status.measured_rtt_ms = (rtt_ms * 100.0).round() / 100.0;
    conn.close(0u8.into(), b"ok");
    Ok(status)
}

/// ZERO-GGUF LIGHTWEIGHT CLIENT ROUTER:
/// Connects to a pool of Pre-Warmed Swarm Pods over Iroh QUIC,
/// load-balances concurrent `/v1/chat/completions` requests across available Continuous-Batching slots,
/// and automatically fails over if any Pod disconnects.
/// Supports `allow_mobile_relay_fallback` for laptop users on 5G / Hotel Symmetric NAT where direct UDP is blocked,
/// while GPU Workers inside the Pod always enforce strict Direct-QUIC.
pub async fn spawn_lightweight_swarm_client_router(
    pod_tickets: Vec<SwarmPodTicket>,
) -> Result<(u16, Vec<SwarmPodStatus>), String> {
    spawn_lightweight_swarm_client_router_with_mode(pod_tickets, false).await
}

pub async fn spawn_lightweight_swarm_client_router_with_mode(
    pod_tickets: Vec<SwarmPodTicket>,
    allow_mobile_relay_fallback: bool,
) -> Result<(u16, Vec<SwarmPodStatus>), String> {
    let builder = if allow_mobile_relay_fallback {
        Endpoint::builder(presets::N0).alpns(vec![])
    } else {
        Endpoint::builder(presets::Minimal)
            .alpns(vec![])
            .clear_ip_transports()
            .bind_addr("127.0.0.1:0".parse::<SocketAddr>().unwrap())
            .map_err(|e| e.to_string())?
    };
    let endpoint = builder.bind().await.map_err(|e| e.to_string())?;

    // 1. Probe all Pods and open persistent multiplexed QUIC connections to healthy Pods
    let mut initial_statuses = Vec::new();
    let mut active_conns: Vec<(String, Arc<Connection>)> = Vec::new();

    for tk in &pod_tickets {
        match probe_swarm_pod(&endpoint, tk).await {
            Ok(st) => {
                if let Ok(conn) = endpoint
                    .connect(tk.endpoint_addr.clone(), ALPN_OPENAI_SWARM)
                    .await
                {
                    info!(
                        "Swarm Router connected to Pre-Warmed Pod '{}' (Model: {}, Parallel Slots: {}, RTT: {:.2} ms)",
                        st.pod_name, st.model_name, st.max_parallel_slots, st.measured_rtt_ms
                    );
                    initial_statuses.push(st);
                    active_conns.push((tk.pod_name.clone(), Arc::new(conn)));
                }
            }
            Err(e) => {
                warn!("Pod '{}' unreachable during discovery: {}", tk.pod_name, e);
            }
        }
    }

    if active_conns.is_empty() {
        return Err("No healthy Pre-Warmed Swarm Pods available".into());
    }

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let router_port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let conns_arc = Arc::new(active_conns);
    let rr_counter = Arc::new(AtomicUsize::new(0));

    tokio::spawn(async move {
        let _keep_ep = endpoint;
        while let Ok((mut client_tcp, _)) = listener.accept().await {
            client_tcp.set_nodelay(true).ok();
            let pool = conns_arc.clone();
            let rr = rr_counter.clone();

            tokio::spawn(async move {
                // Read full HTTP request (headers + optional Content-Length body) from local OpenAI caller
                let req_bytes = match read_full_http_request(&mut client_tcp).await {
                    Some(b) if !b.is_empty() => b,
                    _ => return,
                };
                let req_slice = req_bytes.as_slice();

                // Try routing across available Pods with automatic failover
                let start_idx = rr.fetch_add(1, Ordering::Relaxed);
                for attempt in 0..pool.len() {
                    let idx = (start_idx + attempt) % pool.len();
                    let (pod_name, conn) = &pool[idx];
                    if let Ok((mut q_send, mut q_recv)) = conn.open_bi().await {
                        if q_send.write_all(req_slice).await.is_ok() && q_send.finish().is_ok() {
                            if let Ok(resp) = q_recv.read_to_end(8 * 1024 * 1024).await {
                                if !resp.is_empty() {
                                    let _ = client_tcp.write_all(&resp).await;
                                    return;
                                }
                            }
                        }
                    }
                    warn!("Pod '{}' failed mid-request -> failing over to next Pod in Swarm!", pod_name);
                }
            });
        }
    });

    Ok((router_port, initial_statuses))
}

async fn read_full_http_request(stream: &mut TcpStream) -> Option<Vec<u8>> {
    const MAX_HTTP_REQ_BYTES: usize = 16 * 1024 * 1024;
    let mut buf = Vec::with_capacity(8192);
    let mut tmp = [0u8; 8192];
    let mut header_end: Option<usize> = None;
    let mut expected_total: Option<usize> = None;

    loop {
        if let Some(total) = expected_total {
            if buf.len() >= total {
                break;
            }
        }
        let n = tokio::time::timeout(std::time::Duration::from_secs(10), stream.read(&mut tmp))
            .await
            .ok()?
            .ok()?;
        if n == 0 {
            break;
        }
        if buf.len() + n > MAX_HTTP_REQ_BYTES {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);

        if header_end.is_none() {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let h_end = pos + 4;
                header_end = Some(h_end);
                let hdr_str = String::from_utf8_lossy(&buf[..pos]);
                let content_len = hdr_str
                    .lines()
                    .find_map(|line| {
                        let (k, v) = line.split_once(':')?;
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            v.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if h_end.saturating_add(content_len) > MAX_HTTP_REQ_BYTES {
                    return None;
                }
                expected_total = Some(h_end + content_len);
                if buf.len() >= h_end + content_len {
                    break;
                }
            }
        }
    }
    Some(buf)
}

