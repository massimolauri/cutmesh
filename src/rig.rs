use crate::catalog::{build_model_catalog_entry_from_header, GpuNodeCapabilities, ModelCatalogEntry};
use crate::gguf::slicer::{
    ggml_rpc_fnv1a_hash, ggml_rpc_sparse_tensor_hash, ggml_rpc_sparse_tensor_tag, group_layers,
    link_sparse_rpc_cache_entry, RPC_CACHE_HASH_THRESHOLD,
};
use crate::gguf::{GgufHeader, GgufTensorDescriptor};
use crate::shield::{run_orthogonal_residual_cipher_audit, OrthogonalResidualCipherAudit};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Lifecycle state of a P2P GPU Rig.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigLifecycleState {
    /// The Rig has accumulated enough usable VRAM to hold 100% of the model weights + KV cache + compute buffers.
    /// No additional GPUs are admitted to this Rig (preventing unnecessary P2P network hops);
    /// extra GPUs are routed to form the next horizontal Rig replica (`rig-02`, `rig-03`, ...).
    ClosedAndLocked,
    /// Waiting for additional P2P GPUs to join until `accumulated_usable_vram_mb >= target_rig_vram_mb`.
    OpenCollectingVram { deficit_vram_mb: u64 },
}

/// Role automatically assigned to each node inside a Rig via deterministic hardware + RTT election.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigNodeRole {
    /// Elected Rig Master (Pod Leader): runs `llama-server --local-cuda` (`CUDA0`) for the initial layer slice
    /// and coordinates the Iroh Direct-QUIC pipeline across the Rig Workers.
    Master,
    /// Rig Worker (`RPC0`, `RPC1`, ...): runs `ggml-rpc-server` behind the Iroh Direct-UDP tunnel
    /// and downloads ONLY its assigned layer byte-range (`http_range_bytes`) into the RPC tensor cache.
    Worker { rpc_index: usize },
}

/// Exact layer slice and HTTP `Range:` byte window assigned to a single member of a Rig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigMemberShard {
    pub node_name: String,
    pub endpoint_id: String,
    pub gpu_name: String,
    pub role: RigNodeRole,
    pub llama_device_name: String,
    pub vram_free_mb: u64,
    pub usable_vram_mb: u64,
    pub user_power_budget_pct: u32,
    pub effective_tflops: f64,
    pub effective_bandwidth_gb_s: f64,
    pub measured_rtt_ms: f64,
    pub master_election_score: f64,
    pub start_layer: u16,
    pub end_layer_inclusive: u16,
    pub includes_token_embd: bool,
    pub includes_output_head: bool,
    pub assigned_tensor_count: usize,
    pub assigned_weights_mb: f64,
    pub tensor_split_ratio: u32,
    /// Exact contiguous HTTP `Range: bytes=start-end` inside the `.gguf` file covering this node's transformer blocks.
    pub block_http_range_bytes: (u64, u64),
    /// Percentage of the `.gguf` file that this node skips downloading thanks to partial shard pull.
    pub download_savings_pct: f64,
}

/// A self-contained, geo-clustered P2P GPU Rig formed and closed automatically for a target model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmRig {
    pub rig_id: String,
    pub state: RigLifecycleState,
    pub closing_reason: String,
    pub elected_master_node: String,
    pub elected_master_endpoint_id: String,
    pub target_rig_vram_mb: u64,
    pub accumulated_usable_vram_mb: u64,
    pub total_assigned_weights_mb: f64,
    /// Automatically maximized context window (`-c`) computed to saturate the Rig's remaining usable VRAM up to `model.max_context_length`.
    pub auto_maximized_context_tokens: u32,
    /// Exact KV cache VRAM (`MiB`) allocated for `auto_maximized_context_tokens`.
    pub allocated_kv_cache_mb: u64,
    pub max_inter_node_rtt_ms: f64,
    pub llama_cpp_devices_arg: String,
    pub llama_cpp_tensor_split_arg: String,
    pub estimated_rig_decode_tok_s: f64,
    pub estimated_rig_prefill_tok_s: f64,
    pub members: Vec<RigMemberShard>,
}

/// Complete output of the Rig Formation & Closing Engine across an arbitrary pool of candidate P2P GPUs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigFormationReport {
    pub model_source: String,
    pub model_name: String,
    pub architecture: String,
    pub total_layers: u16,
    pub total_weights_mb: f64,
    pub kv_cache_4k_ctx_mb: u64,
    pub target_rig_vram_mb: u64,
    pub per_node_vram_safety_reserve_mb: u64,
    pub max_allowed_rig_rtt_ms: f64,
    pub min_nodes_per_rig: usize,
    pub closed_ready_rigs_count: usize,
    pub aggregate_swarm_decode_tok_s: f64,
    pub orthogonal_residual_stream_cipher: OrthogonalResidualCipherAudit,
    pub rigs: Vec<SwarmRig>,
}

/// Summary returned after executing a Partial HTTP-Range (or local seek) Shard Pull into the `ggml-rpc` cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialShardPullReport {
    pub model_source: String,
    pub start_layer: u16,
    pub end_layer_inclusive: u16,
    pub include_non_block_tensors: bool,
    pub cached_tensor_count: usize,
    pub downloaded_bytes: u64,
    pub downloaded_mb: f64,
    pub total_model_bytes: u64,
    pub bandwidth_saved_pct: f64,
    pub cache_dir: String,
    /// Exact 64-bit FNV-1a hashes (`tensor_name -> real_hash`) computed by the Worker for its cached tensors,
    /// enabling the CPU-only Master to use a Sparse GGUF Skeleton (`0 Bytes` of layer weights downloaded on the Master).
    #[serde(default)]
    pub tensor_hashes: BTreeMap<String, u64>,
}

/// Fetches ONLY a single GGUF file's header + tensor descriptor table (`0..range_mib MiB`).
pub fn fetch_single_gguf_header(shard_source: &str, range_mib: u64) -> Result<GgufHeader, String> {
    if shard_source.starts_with("http://") || shard_source.starts_with("https://") {
        let range_end: u64 = range_mib * 1024 * 1024 - 1;
        let header_bytes = fetch_http_byte_range(shard_source, 0, range_end)?;
        let mut cursor = Cursor::new(header_bytes);
        let mut hdr = GgufHeader::parse_reader(&mut cursor)?;
        for t in &mut hdr.tensors {
            t.shard_source = Some(shard_source.to_string());
        }
        Ok(hdr)
    } else {
        GgufHeader::parse_file(Path::new(shard_source))
    }
}

/// Fetches ONLY the GGUF header + tensor descriptor table (`0..header_bytes`) from either:
/// - a remote `http://` or `https://` URL (using HTTP `Range: bytes=0-12582911` so a 20+ GB or 186+ GB multi-shard model is inspected in <1.5s without downloading weights), or
/// - a local `.gguf` file on disk.
pub fn fetch_or_parse_gguf_header(model_source: &str) -> Result<(GgufHeader, String), String> {
    if model_source.starts_with("http://") || model_source.starts_with("https://") {
        let shards = crate::gguf::expand_split_gguf_sources(model_source);
        // Fetch the first 12 MiB of the primary file via HTTP Range request (covers metadata, 250k+ vocab, and tensor descriptors)
        let mut hdr = fetch_single_gguf_header(&shards[0], 12)?;

        if shards.len() > 1 {
            let mut max_layer: Option<u16> = if hdr.total_layers > 0 {
                Some(hdr.total_layers - 1)
            } else {
                None
            };
            for sib_url in &shards[1..] {
                let sib_hdr = fetch_single_gguf_header(sib_url, 2)?;
                if sib_hdr.total_layers > 0 {
                    let l = sib_hdr.total_layers - 1;
                    max_layer = Some(max_layer.map_or(l, |m| m.max(l)));
                }
                if hdr.embedding_length == 0 && sib_hdr.embedding_length > 0 {
                    hdr.embedding_length = sib_hdr.embedding_length;
                }
                if hdr.vocab_size == 0 && sib_hdr.vocab_size > 0 {
                    hdr.vocab_size = sib_hdr.vocab_size;
                }
                hdr.tensor_count += sib_hdr.tensor_count;
                hdr.tensors.extend(sib_hdr.tensors);
            }
            if let Some(m) = max_layer {
                hdr.total_layers = m + 1;
            }
        }

        let model_name = model_source
            .rsplit('/')
            .next()
            .unwrap_or("remote_model.gguf")
            .split('?')
            .next()
            .unwrap_or("remote_model.gguf")
            .to_string();
        Ok((hdr, model_name))
    } else {
        let path = Path::new(model_source);
        let hdr = GgufHeader::parse_file(path)?;
        let model_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model.gguf")
            .to_string();
        Ok((hdr, model_name))
    }
}

struct HttpRangeJob {
    url: String,
    range_header: String,
    resp_tx: std::sync::mpsc::Sender<Result<Vec<u8>, String>>,
}

fn http_range_dispatcher() -> Result<&'static std::sync::mpsc::Sender<HttpRangeJob>, String> {
    static TX: OnceLock<Result<std::sync::mpsc::Sender<HttpRangeJob>, String>> = OnceLock::new();
    TX.get_or_init(|| {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<HttpRangeJob>();
        std::thread::Builder::new()
            .name("cutmesh-http-range".into())
            .spawn(move || {
                let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                else {
                    return;
                };
                let Ok(client) = reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(10))
                    .read_timeout(std::time::Duration::from_secs(30))
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                else {
                    return;
                };
                while let Ok(job) = job_rx.recv() {
                    let c = client.clone();
                    rt.spawn(async move {
                        let mut last_err = String::new();
                        for attempt in 0..3u8 {
                            let res = async {
                                let resp = c
                                    .get(&job.url)
                                    .header(reqwest::header::RANGE, &job.range_header)
                                    .send()
                                    .await
                                    .map_err(|e| {
                                        format!(
                                            "HTTP Range request ({}) to {} failed: {}",
                                            job.range_header, job.url, e
                                        )
                                    })?;
                                let status = resp.status();
                                if !status.is_success() {
                                    return Err(format!(
                                        "HTTP Range request ({}) to {} failed with status {}",
                                        job.range_header, job.url, status
                                    ));
                                }
                                let bytes = resp.bytes().await.map_err(|e| {
                                    format!(
                                        "Failed reading HTTP Range body ({}) from {}: {}",
                                        job.range_header, job.url, e
                                    )
                                })?;
                                Ok(bytes.to_vec())
                            }
                            .await;
                            match res {
                                Ok(bytes) => {
                                    let _ = job.resp_tx.send(Ok(bytes));
                                    return;
                                }
                                Err(e) => {
                                    last_err = e;
                                    if attempt < 2 {
                                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                    }
                                }
                            }
                        }
                        let _ = job.resp_tx.send(Err(last_err));
                    });
                }
            })
            .map_err(|e| format!("Failed spawning HTTP Range runtime thread: {}", e))?;
        Ok(job_tx)
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Coalesces already-sorted tensor descriptors within the same shard file into contiguous byte windows
/// (gap <= 64 KiB, total window <= `max_window_bytes`).
fn coalesce_byte_windows(
    descriptors: &[GgufTensorDescriptor],
    default_source: &str,
    max_window_bytes: u64,
) -> Vec<(String, u64, u64, Vec<GgufTensorDescriptor>)> {
    let mut windows = Vec::new();
    let mut idx = 0usize;
    while idx < descriptors.len() {
        let win_shard = descriptors[idx]
            .shard_source
            .as_deref()
            .unwrap_or(default_source)
            .to_string();
        let win_start = descriptors[idx].absolute_file_offset;
        let mut win_end = win_start + descriptors[idx].byte_size;
        let mut batch_end_idx = idx + 1;

        while batch_end_idx < descriptors.len() {
            let next_t = &descriptors[batch_end_idx];
            let next_shard = next_t.shard_source.as_deref().unwrap_or(default_source);
            let next_end = next_t.absolute_file_offset + next_t.byte_size;
            if next_shard == win_shard
                && next_t.absolute_file_offset <= win_end + 65_536
                && (next_end - win_start) <= max_window_bytes
            {
                win_end = next_end;
                batch_end_idx += 1;
            } else {
                break;
            }
        }

        windows.push((
            win_shard,
            win_start,
            win_end,
            descriptors[idx..batch_end_idx].to_vec(),
        ));
        idx = batch_end_idx;
    }
    windows
}

/// Executes an HTTP `Range: bytes=start-end` request via a persistent keep-alive `reqwest::Client`.
pub fn fetch_http_byte_range(url: &str, start_byte: u64, end_byte: u64) -> Result<Vec<u8>, String> {
    let tx = http_range_dispatcher()?;
    let (resp_tx, resp_rx) = std::sync::mpsc::channel();
    tx.send(HttpRangeJob {
        url: url.to_string(),
        range_header: format!("bytes={}-{}", start_byte, end_byte),
        resp_tx,
    })
    .map_err(|e| format!("HTTP Range dispatcher unavailable: {}", e))?;
    resp_rx
        .recv()
        .map_err(|e| format!("HTTP Range dispatcher dropped response: {}", e))?
}

/// Deterministic Master Election Score for a node inside a Rig.
pub fn compute_master_election_score(node: &GpuNodeCapabilities, reserve_mb: u64) -> f64 {
    compute_stage0_master_score(node, reserve_mb, 0.0)
}

/// Deterministic Stage-0 Master Election Score:
/// Rewards usable VRAM capacity, memory bandwidth, sqrt(TFLOPS), and low RTT,
/// while applying a strong viability penalty if headroom after `stage0_overhead_mb` < 4096 MiB
/// so small 8 GB cards never displace 12–24 GB cards as Stage-0 Master.
pub fn compute_stage0_master_score(
    node: &GpuNodeCapabilities,
    reserve_mb: u64,
    stage0_overhead_mb: f64,
) -> f64 {
    // Stage-0 Master always needs at least 2560 MiB reserve for CUDA primary context,
    // HTTP server, and RPC ingress buffers when stage0_overhead_mb > 0.
    let effective_reserve = if stage0_overhead_mb > 0.0 {
        reserve_mb.max(2560)
    } else {
        reserve_mb
    };
    let usable_vram = node.vram_free_mb.saturating_sub(effective_reserve) as f64;
    let vram_factor = 1.0 + (usable_vram / 8192.0);
    let perf = node.effective_available_bandwidth_gb_s * node.effective_available_tflops.max(0.5).sqrt();
    let rtt_penalty = 1.0 + node.measured_rtt_ms.max(0.0) * 0.004;
    let mut score = (perf * vram_factor) / rtt_penalty;
    if stage0_overhead_mb > 0.0 {
        let headroom = usable_vram - stage0_overhead_mb;
        if headroom < 4096.0 {
            let viability = 0.05 * (headroom.max(0.0) / 4096.0).max(0.05);
            score *= viability;
        }
    }
    (score * 100.0).round() / 100.0
}

/// Automatically groups candidate P2P GPUs into geo-clustered **Rigs**, decides the exact moment
/// to **CLOSE each Rig** (`ClosedAndLocked`) as soon as accumulated usable VRAM covers the model
/// (`weights + KV cache`), elects the **Rig Master (`CUDA0`)**, and calculates each member's
/// exact layer slice and HTTP `Range:` byte offsets for partial downloading.
pub fn form_and_close_rigs(
    model_source: &str,
    hdr: &GgufHeader,
    model_name: &str,
    candidates: &[GpuNodeCapabilities],
    max_rig_rtt_ms: f64,
    per_node_vram_reserve_mb: u64,
) -> Result<RigFormationReport, String> {
    form_and_close_rigs_with_min_nodes(
        model_source,
        hdr,
        model_name,
        candidates,
        max_rig_rtt_ms,
        per_node_vram_reserve_mb,
        1,
    )
}

pub fn form_and_close_rigs_with_min_nodes(
    model_source: &str,
    hdr: &GgufHeader,
    model_name: &str,
    candidates: &[GpuNodeCapabilities],
    max_rig_rtt_ms: f64,
    per_node_vram_reserve_mb: u64,
    min_nodes_per_rig: usize,
) -> Result<RigFormationReport, String> {
    let min_nodes = min_nodes_per_rig.max(1);
    let catalog: ModelCatalogEntry = build_model_catalog_entry_from_header(hdr, model_name)?;
    let total_tensor_bytes: u64 = hdr.tensors.iter().map(|t| t.byte_size).sum();
    let total_weights_mb = (total_tensor_bytes as f64) / (1024.0 * 1024.0);
    let target_rig_vram_mb = (total_weights_mb.ceil() as u64) + catalog.kv_cache_4k_ctx_mb + 512;

    // Filter eligible nodes with Direct-UDP reachability and enough free VRAM beyond safety reserve
    let mut pool: Vec<GpuNodeCapabilities> = candidates
        .iter()
        .filter(|n| n.direct_udp_reachable && n.vram_free_mb > per_node_vram_reserve_mb + 512)
        .cloned()
        .collect();

    if pool.is_empty() {
        return Err("Nessun nodo GPU idoneo trovato per formare un Rig".into());
    }

    // Sort candidate pool: prioritize nodes within max_rig_rtt_ms first, then by highest Master/Bandwidth score
    pool.sort_by(|a, b| {
        let a_in_rtt = a.measured_rtt_ms <= max_rig_rtt_ms;
        let b_in_rtt = b.measured_rtt_ms <= max_rig_rtt_ms;
        match (a_in_rtt, b_in_rtt) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                let sa = compute_master_election_score(a, per_node_vram_reserve_mb);
                let sb = compute_master_election_score(b, per_node_vram_reserve_mb);
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            }
        }
    });

    let model_policy = crate::models::ModelPolicy::resolve(model_source);
    let max_nodes_per_rig: usize = model_policy.max_nodes_for_rtt(max_rig_rtt_ms, min_nodes);

    // Partition candidates into Rigs, closing each Rig immediately when accumulated_usable_vram_mb >= target_rig_vram_mb
    // (with 1, 2, or up to 3 GPUs). If 3 GPUs still do not reach target_rig_vram_mb, the Rig remains unclosed/rejected.
    let mut raw_rig_groups: Vec<(Vec<GpuNodeCapabilities>, u64)> = Vec::new();
    let mut current_group: Vec<GpuNodeCapabilities> = Vec::new();
    let mut current_usable_vram: u64 = 0;

    for node in pool {
        let usable = node.vram_free_mb.saturating_sub(per_node_vram_reserve_mb);
        current_usable_vram += usable;
        current_group.push(node);

        // RIG CLOSING DECISION:
        // Close the Rig immediately once accumulated usable VRAM covers 100% of weights + KV cache AND min_nodes is reached,
        // or stop adding nodes once max_nodes_per_rig (3 GPUs) is reached.
        if (current_usable_vram >= target_rig_vram_mb && current_group.len() >= min_nodes)
            || current_group.len() >= max_nodes_per_rig
        {
            raw_rig_groups.push((current_group, current_usable_vram));
            current_group = Vec::new();
            current_usable_vram = 0;
        }
    }
    if !current_group.is_empty() {
        raw_rig_groups.push((current_group, current_usable_vram));
    }

    let layers_map = group_layers(hdr);
    let non_block_bytes: u64 = hdr
        .tensors
        .iter()
        .filter(|t| t.layer_index.is_none())
        .map(|t| t.byte_size)
        .sum();
    let non_block_mb = (non_block_bytes as f64) / (1024.0 * 1024.0);
    let total_block_bytes = total_tensor_bytes.saturating_sub(non_block_bytes).max(1);
    let total_block_mb = (total_block_bytes as f64) / (1024.0 * 1024.0);
    let stage0_overhead_mb = model_policy.stage0_total_reserve_mb(non_block_mb);

    let mut rigs = Vec::new();
    let mut closed_ready_rigs_count = 0usize;
    let mut aggregate_swarm_decode_tok_s = 0.0f64;

    for (rig_idx, (mut group, acc_usable_vram)) in raw_rig_groups.into_iter().enumerate() {
        let rig_id = format!("rig-{:02}", rig_idx + 1);
        let is_closed = acc_usable_vram >= target_rig_vram_mb
            && group.len() >= min_nodes
            && group.len() <= max_nodes_per_rig;
        let state = if is_closed {
            closed_ready_rigs_count += 1;
            RigLifecycleState::ClosedAndLocked
        } else {
            RigLifecycleState::OpenCollectingVram {
                deficit_vram_mb: target_rig_vram_mb.saturating_sub(acc_usable_vram).max(1),
            }
        };

        // AUTOMATIC MASTER ELECTION inside this Rig:
        // Highest compute_stage0_master_score (tie-broken deterministically by endpoint_id) becomes Rig Master (CUDA0).
        group.sort_by(|a, b| {
            let sa = compute_stage0_master_score(a, per_node_vram_reserve_mb, stage0_overhead_mb);
            let sb = compute_stage0_master_score(b, per_node_vram_reserve_mb, stage0_overhead_mb);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });

        let master_node = group[0].clone();
        // Order remaining workers: nodes on the same LAN (`RTT <= 2.5 ms`) stay contiguous with Stage 0,
        // otherwise keep descending hardware capacity (`compute_master_election_score`) so larger GPUs
        // precede smaller GPUs in the pipeline.
        if group.len() > 2 {
            let master_rtt = master_node.measured_rtt_ms;
            group[1..].sort_by(|a, b| {
                let a_lan = (a.measured_rtt_ms - master_rtt).abs() <= 2.5 && a.measured_rtt_ms <= 5.0;
                let b_lan = (b.measured_rtt_ms - master_rtt).abs() <= 2.5 && b.measured_rtt_ms <= 5.0;
                match (a_lan, b_lan) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => {
                        let sa = compute_master_election_score(a, per_node_vram_reserve_mb);
                        let sb = compute_master_election_score(b, per_node_vram_reserve_mb);
                        sb.partial_cmp(&sa)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
                    }
                }
            });
        }

        let closing_reason = if is_closed {
            format!(
                "CLOSED: {} GPU raggiungono {} MiB di VRAM utilizzabile (>= {} MiB richiesti per pesi + KV/SSM cache, max {} GPU). Rig pronto ed efficiente.",
                group.len(),
                acc_usable_vram,
                target_rig_vram_mb,
                max_nodes_per_rig
            )
        } else if group.len() > max_nodes_per_rig {
            format!(
                "REJECTED_TOO_MANY_GPUS: {} GPU superano il limite massimo efficiente ({} GPU) per questo modello.",
                group.len(),
                max_nodes_per_rig
            )
        } else if group.len() < min_nodes {
            format!(
                "WAITING_FOR_MIN_NODES: {} GPU pronte su {} richieste dal Rig (min_nodes={}). In attesa degli altri nodi.",
                group.len(),
                min_nodes,
                min_nodes
            )
        } else {
            format!(
                "INSUFFICIENT_VRAM_CLOSED: {} GPU forniscono {} MiB utilizzabili su {} MiB richiesti (mancano {} MiB). Il Rig non può partire e resta chiuso.",
                group.len(),
                acc_usable_vram,
                target_rig_vram_mb,
                target_rig_vram_mb.saturating_sub(acc_usable_vram)
            )
        };

        // Water-Filling layer allocation across the Rig members (1..max_wan_nodes GPUs: Master Stage 0 + Workers RPC0..RPCk)
        let n_members = group.len();

        let caps_mb: Vec<f64> = group
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let node_reserve = if i == 0 {
                    per_node_vram_reserve_mb.max(2560).max(model_policy.sharding.worker_vram_reserve_mb)
                } else {
                    per_node_vram_reserve_mb.max(model_policy.sharding.worker_vram_reserve_mb)
                };
                let raw_usable = n.vram_free_mb.saturating_sub(node_reserve) as f64;
                let stage_reserve = if i == 0 && n_members > 1 {
                    stage0_overhead_mb
                } else {
                    0.0
                };
                (raw_usable - stage_reserve).max(256.0)
            })
            .collect();

        // Weight layer allocation directly by usable VRAM capacity (`caps_mb`) when the model fills >=50%
        // of cluster VRAM so no GPU overflows while keeping balanced KV-cache headroom across nodes.
        let total_caps_mb: f64 = caps_mb.iter().sum::<f64>().max(1.0);
        let high_vram_pressure = total_block_mb / total_caps_mb >= 0.50;
        let scores: Vec<f64> = group
            .iter()
            .enumerate()
            .map(|(i, n)| {
                if high_vram_pressure {
                    caps_mb[i].max(256.0)
                } else {
                    let vram_factor = 1.0 + (n.vram_total_mb as f64) / 24576.0;
                    let perf = n.effective_available_bandwidth_gb_s * n.effective_available_tflops.max(0.5).sqrt();
                    perf * vram_factor
                }
            })
            .collect();

        let mut alloc_mb = vec![0.0f64; n_members];
        let mut frozen = vec![false; n_members];
        let mut remaining_mb = total_block_mb;

        for _ in 0..n_members {
            let unfrozen_sum: f64 = scores
                .iter()
                .enumerate()
                .filter(|(i, _)| !frozen[*i])
                .map(|(_, &s)| s)
                .sum();
            if unfrozen_sum <= 0.0 || remaining_mb <= 0.0 {
                break;
            }
            let mut capped_any = false;
            for i in 0..n_members {
                if !frozen[i] {
                    let desired = remaining_mb * (scores[i] / unfrozen_sum);
                    if desired > caps_mb[i] {
                        alloc_mb[i] = caps_mb[i];
                        frozen[i] = true;
                        remaining_mb = (remaining_mb - caps_mb[i]).max(0.0);
                        capped_any = true;
                    }
                }
            }
            if !capped_any {
                for i in 0..n_members {
                    if !frozen[i] {
                        alloc_mb[i] = remaining_mb * (scores[i] / unfrozen_sum);
                        frozen[i] = true;
                    }
                }
                break;
            }
        }

        let alloc_sum: f64 = alloc_mb.iter().sum::<f64>().max(1.0);
        let total_layers = hdr.total_layers.max(1);
        let pipeline_layers = model_policy.pipeline_layers(total_layers);
        let group_align: u16 = model_policy.layer_group_alignment();
        let mtp_extra = if model_policy.speculative.enabled && !model_policy.sharding.mtp_layers.is_empty() {
            model_policy.sharding.mtp_layers.len() as f64
        } else {
            0.0
        };
        let llama_act_gpu_layers = (pipeline_layers as f64) + 1.0 + mtp_extra;
        let mut prev_cum_pct: u32 = 0;

        let mut current_layer: u16 = 0;
        let mut members = Vec::with_capacity(n_members);
        let mut devices_list = Vec::with_capacity(n_members);
        let mut split_list = Vec::with_capacity(n_members);
        let mut max_rtt = 0.0f64;
        let mut total_assigned_mb = 0.0f64;

        for (idx, node) in group.iter().enumerate() {
            let ratio = (alloc_mb[idx] / alloc_sum).max(0.05);
            let start_l = current_layer;
            let end_l_incl = if idx + 1 == n_members {
                pipeline_layers.saturating_sub(1)
            } else {
                let remaining_peers = (n_members - idx - 1) as u16;
                let min_reserved_after = remaining_peers * group_align;
                let max_end_exclusive = pipeline_layers.saturating_sub(min_reserved_after).max(start_l + group_align);
                let target_slice_bytes = ((total_block_bytes as f64) * ratio) as u64;

                let mut acc_b = 0u64;
                let mut raw_end_excl = start_l + group_align;
                for l in start_l..max_end_exclusive {
                    let lb_bytes = layers_map
                        .get(&l)
                        .map(|lb| lb.total_bytes)
                        .or_else(|| {
                            model_policy
                                .sharding
                                .layer_weights_mib
                                .get(l as usize)
                                .map(|&mib| (mib * 1024.0 * 1024.0) as u64)
                        })
                        .unwrap_or(0);
                    if acc_b + lb_bytes / 2 > target_slice_bytes && l > start_l {
                        raw_end_excl = l;
                        break;
                    }
                    acc_b += lb_bytes;
                    raw_end_excl = l + 1;
                }
                let aligned_end_excl = model_policy.snap_cut_layer(
                    raw_end_excl,
                    start_l + group_align,
                    max_end_exclusive,
                );
                aligned_end_excl - 1
            };
            current_layer = end_l_incl + 1;

            let (role, dev_name) = if idx == 0 {
                (RigNodeRole::Master, "CUDA0".to_string())
            } else {
                (
                    RigNodeRole::Worker { rpc_index: idx - 1 },
                    format!("RPC{}", idx - 1),
                )
            };

            // Stage 0 (Master / fastest GPU) always holds both token_embd.weight and output.weight
            let includes_token_embd = idx == 0;
            let includes_output_head = idx == 0;

            // Compute exact tensor count, weight bytes, and contiguous block HTTP Range [start_byte, end_byte]
            let mut tensor_count = 0usize;
            let mut slice_bytes = 0u64;
            let mut min_offset = u64::MAX;
            let mut max_offset = 0u64;

            for l in start_l..=end_l_incl {
                if let Some(lb) = layers_map.get(&l) {
                    tensor_count += lb.tensor_count;
                    slice_bytes += lb.total_bytes;
                    min_offset = min_offset.min(lb.start_byte);
                    max_offset = max_offset.max(lb.end_byte);
                }
            }
            for t in &hdr.tensors {
                if t.layer_index.is_none() {
                    let is_embd = t.name.contains("token_embd") || t.name.contains("wte");
                    if (is_embd && includes_token_embd) || (!is_embd && includes_output_head) {
                        tensor_count += 1;
                        slice_bytes += t.byte_size;
                    }
                }
            }
            if min_offset == u64::MAX {
                min_offset = hdr.data_section_offset;
            }

            // Pass the exact layer-index midpoint percentage of `llama.cpp`'s `act_gpu_layers` to `--tensor-split`
            // so `std::upper_bound(splits, float(il)/act_gpu_layers)` cuts at `aligned_end_excl`!
            let cum_pct = if idx + 1 == n_members {
                100u32
            } else {
                ((((end_l_incl as f64) + 0.5) / llama_act_gpu_layers) * 100.0)
                    .round()
                    .clamp((prev_cum_pct + 1) as f64, 99.0) as u32
            };
            let split_pct = cum_pct.saturating_sub(prev_cum_pct).max(1);
            prev_cum_pct = cum_pct;

            let slice_mb = (slice_bytes as f64) / (1024.0 * 1024.0);
            total_assigned_mb += slice_mb;
            max_rtt = max_rtt.max(node.measured_rtt_ms);

            let savings_pct = if total_tensor_bytes > 0 {
                ((1.0 - (slice_bytes as f64 / total_tensor_bytes as f64)).max(0.0) * 1000.0).round()
                    / 10.0
            } else {
                0.0
            };

            devices_list.push(dev_name.clone());
            split_list.push(split_pct.to_string());

            members.push(RigMemberShard {
                node_name: node.node_name.clone(),
                endpoint_id: node.endpoint_id.clone(),
                gpu_name: node.gpu_name.clone(),
                role,
                llama_device_name: dev_name,
                vram_free_mb: node.vram_free_mb,
                usable_vram_mb: node.vram_free_mb.saturating_sub(per_node_vram_reserve_mb),
                user_power_budget_pct: node.user_power_budget_pct,
                effective_tflops: node.effective_available_tflops,
                effective_bandwidth_gb_s: node.effective_available_bandwidth_gb_s,
                measured_rtt_ms: node.measured_rtt_ms,
                master_election_score: scores[idx],
                start_layer: start_l,
                end_layer_inclusive: end_l_incl,
                includes_token_embd,
                includes_output_head,
                assigned_tensor_count: tensor_count,
                assigned_weights_mb: (slice_mb * 100.0).round() / 100.0,
                tensor_split_ratio: split_pct,
                block_http_range_bytes: (min_offset, max_offset.saturating_sub(1)),
                download_savings_pct: savings_pct,
            });
        }

        // Physics-derived speed estimate for this Rig (calibrated against our 3x RTX A4000 Estonia Direct-UDP benchmark: 17.14 tok/s)
        let bottleneck_bw = members
            .iter()
            .map(|m| m.effective_bandwidth_gb_s)
            .fold(f64::INFINITY, f64::min)
            .max(50.0);
        let active_weights_gb = catalog.active_weights_per_token_mb / 1024.0;
        let gpu_mem_read_ms = (active_weights_gb / bottleneck_bw) * 1000.0;
        let kernel_launch_ms = (hdr.total_layers as f64) * 0.11;
        let rpc_hops = members.len().saturating_sub(1) as f64;
        let total_rtt_ms: f64 = members.iter().skip(1).map(|m| m.measured_rtt_ms).sum();
        let rpc_serialization_ms = rpc_hops * 6.5 + total_rtt_ms * 2.0;
        let total_decode_ms = gpu_mem_read_ms + kernel_launch_ms + rpc_serialization_ms;
        let est_decode_tok_s = ((1000.0 / total_decode_ms.max(1.0)) * 100.0).round() / 100.0;
        let est_prefill_tok_s = ((est_decode_tok_s * 0.88) * 100.0).round() / 100.0;

        if is_closed {
            aggregate_swarm_decode_tok_s += est_decode_tok_s;
        }

        // AUTO-MAX CONTEXT (Driven by ModelPolicy):
        // Use all remaining usable VRAM inside the Rig (after model weights + policy reserve buffer)
        // to maximize the context window `-c` up to `model_policy.engine.max_context_length` (262,144).
        let policy_max_ctx = model_policy.engine.max_context_length.max(4096);
        let total_kv_per_1k_mb = (model_policy.sharding.kv_vram_per_1k_tokens_mb.max(8)) as f64;
        let mut min_gpu_max_ctx: u32 = policy_max_ctx;
        for (idx, m) in members.iter().enumerate() {
            let assigned_layers = (m.end_layer_inclusive.saturating_sub(m.start_layer) + 1) as f64
                + if idx == 0 {
                    model_policy.sharding.mtp_layers.len() as f64
                } else {
                    0.0
                };
            let layer_frac = (assigned_layers / (total_layers.max(1) as f64)).clamp(0.05, 1.0);
            let gpu_kv_per_1k_mb = (total_kv_per_1k_mb * layer_frac).max(1.0);
            let reserve_mb = if idx == 0 {
                (model_policy.sharding.stage0_extra_vram_reserve_mb as f64)
                    .max(2560.0)
                    .max(per_node_vram_reserve_mb as f64)
            } else {
                (model_policy.sharding.worker_vram_reserve_mb as f64)
                    .max(1024.0)
                    .max(per_node_vram_reserve_mb as f64)
            };
            let gpu_surplus_mb = ((m.vram_free_mb as f64) - m.assigned_weights_mb - reserve_mb).max(128.0);
            let gpu_max_ctx = ((gpu_surplus_mb / gpu_kv_per_1k_mb) * 1024.0).floor() as u32;
            min_gpu_max_ctx = min_gpu_max_ctx.min(gpu_max_ctx);
        }
        let rounded_ctx = (min_gpu_max_ctx / 1024).max(4) * 1024;
        let auto_maximized_context_tokens = rounded_ctx.min(policy_max_ctx);
        let allocated_kv_cache_mb =
            (((auto_maximized_context_tokens as f64 / 1024.0) * total_kv_per_1k_mb).ceil() as u64)
                .max(catalog.kv_cache_4k_ctx_mb);

        rigs.push(SwarmRig {
            rig_id,
            state,
            closing_reason,
            elected_master_node: master_node.node_name,
            elected_master_endpoint_id: master_node.endpoint_id,
            target_rig_vram_mb,
            accumulated_usable_vram_mb: acc_usable_vram,
            total_assigned_weights_mb: (total_assigned_mb * 100.0).round() / 100.0,
            auto_maximized_context_tokens,
            allocated_kv_cache_mb,
            max_inter_node_rtt_ms: max_rtt,
            llama_cpp_devices_arg: devices_list.join(","),
            llama_cpp_tensor_split_arg: split_list.join(","),
            estimated_rig_decode_tok_s: est_decode_tok_s,
            estimated_rig_prefill_tok_s: est_prefill_tok_s,
            members,
        });
    }

    let hidden_dim = (hdr.embedding_length as usize).max(128);
    let rig_secret: [u8; 32] = *blake3::hash(format!("rig-q-key-{}", model_name).as_bytes()).as_bytes();
    let mut synth_vocab: Vec<Vec<f32>> = Vec::with_capacity(512);
    for v_idx in 0..512 {
        let mut h = blake3::Hasher::new();
        h.update(b"vocab-row");
        h.update(&(v_idx as u64).to_le_bytes());
        let mut xof = h.finalize_xof();
        let mut raw = vec![0u8; hidden_dim * 2];
        xof.fill(&mut raw);
        let vec_f32: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| ((u16::from_le_bytes(*c) as f32) / 32767.5) - 1.0)
            .collect();
        synth_vocab.push(vec_f32);
    }
    let test_targets: Vec<usize> = (0..16).map(|i| i * 31 % 512).collect();
    let ortho_audit =
        run_orthogonal_residual_cipher_audit(hidden_dim, &rig_secret, &synth_vocab, &test_targets);

    Ok(RigFormationReport {
        model_source: model_source.to_string(),
        model_name: model_name.to_string(),
        architecture: hdr.architecture.clone(),
        total_layers: hdr.total_layers,
        total_weights_mb: (total_weights_mb * 100.0).round() / 100.0,
        kv_cache_4k_ctx_mb: catalog.kv_cache_4k_ctx_mb,
        target_rig_vram_mb,
        per_node_vram_safety_reserve_mb: per_node_vram_reserve_mb,
        max_allowed_rig_rtt_ms: max_rig_rtt_ms,
        min_nodes_per_rig: min_nodes,
        closed_ready_rigs_count,
        aggregate_swarm_decode_tok_s: (aggregate_swarm_decode_tok_s * 100.0).round() / 100.0,
        orthogonal_residual_stream_cipher: ortho_audit,
        rigs,
    })
}

/// Process-wide mutex preventing duplicate concurrent shard downloads on a single Worker
/// when both Control Plane Heartbeat and `ALPN_PREWARM_SHARD` arrive simultaneously.
fn prewarm_shard_mutex() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Downloads or extracts ONLY the assigned layer slice (`start_layer ..= end_layer_inclusive`)
/// from a remote HuggingFace `.gguf` URL (via HTTP `Range: bytes=start-end`) or local `.gguf` file
/// directly into the `ggml-rpc-server -c` cache directory (`~/.cache/llama.cpp/rpc/<fnv1a_hash>`),
/// completely avoiding downloading the full `.gguf` model on Worker nodes!
pub fn pull_rig_shard_to_rpc_cache(
    model_source: &str,
    cache_dir: &Path,
    start_layer: u16,
    end_layer_inclusive: u16,
    include_non_block_tensors: bool,
) -> Result<PartialShardPullReport, String> {
    let _guard = prewarm_shard_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;

    let model_key = blake3::hash(format!("model_idx_v1|{}", model_source).as_bytes());
    let index_path = cache_dir.join(format!(
        ".cutmesh_tensor_hashes_{}.json",
        &model_key.to_hex()[..16]
    ));
    let mut known_hashes: BTreeMap<String, u64> = std::fs::read_to_string(&index_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let marker_key = blake3::hash(
        format!(
            "v2|{}|{}|{}|{}|{}",
            model_source,
            start_layer,
            end_layer_inclusive,
            include_non_block_tensors,
            RPC_CACHE_HASH_THRESHOLD
        )
        .as_bytes(),
    );
    let marker_path = cache_dir.join(format!(
        ".cutmesh_shard_{}.json",
        &marker_key.to_hex()[..16]
    ));
    if let Ok(raw_marker) = std::fs::read_to_string(&marker_path) {
        if let Ok(mut cached_report) = serde_json::from_str::<PartialShardPullReport>(&raw_marker) {
            if !cached_report.tensor_hashes.is_empty() {
                cached_report.cache_dir = cache_dir.to_string_lossy().to_string();
                return Ok(cached_report);
            }
        }
    }

    let (hdr, _model_name) = fetch_or_parse_gguf_header(model_source)?;
    let total_model_bytes: u64 = hdr.tensors.iter().map(|t| t.byte_size).sum();
    let model_policy = crate::models::ModelPolicy::resolve(model_source);
    let stage0_pinned_layers = model_policy.stage0_pinned_layers();

    // Filter tensors belonging to this node's assigned layer slice (`start_layer..=end_layer_inclusive`)
    // plus any layers explicitly pinned to Stage 0 by `models/<model_id>/policy.json`.
    let mut selected_tensors: Vec<GgufTensorDescriptor> = hdr
        .tensors
        .iter()
        .filter(|t| {
            if t.byte_size <= RPC_CACHE_HASH_THRESHOLD {
                return false;
            }
            match t.layer_index {
                Some(l) => {
                    (l >= start_layer && l <= end_layer_inclusive)
                        || (include_non_block_tensors && stage0_pinned_layers.contains(&l))
                }
                None => include_non_block_tensors,
            }
        })
        .cloned()
        .collect();

    selected_tensors.sort_by(|a, b| {
        a.shard_source
            .cmp(&b.shard_source)
            .then(a.absolute_file_offset.cmp(&b.absolute_file_offset))
    });

    let mut report_hashes: BTreeMap<String, u64> = BTreeMap::new();
    let mut pending_tensors: Vec<GgufTensorDescriptor> = Vec::new();
    let mut cached_count = 0usize;
    let mut downloaded_bytes = 0u64;

    for t in &selected_tensors {
        if let Some(&h) = known_hashes.get(&t.name) {
            let out_path = cache_dir.join(format!("{:016x}", h));
            let exists = std::fs::metadata(&out_path)
                .map(|m| m.len() == t.byte_size)
                .unwrap_or(false);
            if exists {
                link_sparse_rpc_cache_entry(cache_dir, &out_path, &t.name, t.byte_size);
                report_hashes.insert(t.name.clone(), h);
                cached_count += 1;
                downloaded_bytes += t.byte_size;
                continue;
            }
        }
        pending_tensors.push(t.clone());
    }

    let is_remote = model_source.starts_with("http://") || model_source.starts_with("https://");

    // Group remaining uncached tensors into contiguous byte windows (up to 64 MiB per HTTP Range chunk)
    const MAX_CHUNK_WINDOW_BYTES: u64 = 64 * 1024 * 1024;
    let windows = coalesce_byte_windows(&pending_tensors, model_source, MAX_CHUNK_WINDOW_BYTES);
    let worker_count = if is_remote {
        windows.len().clamp(1, 4)
    } else {
        1
    };
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let results_mutex: Mutex<Vec<(String, u64, u64)>> = Mutex::new(Vec::new());

    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_idx in 0..worker_count {
            let cur_ref = &cursor;
            let win_ref = &windows;
            let res_ref = &results_mutex;
            handles.push(scope.spawn(move || -> Result<(), String> {
                loop {
                    let i = cur_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= win_ref.len() {
                        break;
                    }
                    let (ref win_shard, win_start, win_end, ref win_tensors) = win_ref[i];
                    let window_len = (win_end - win_start) as usize;
                    let window_buf = if is_remote {
                        fetch_http_byte_range(win_shard, win_start, win_end - 1)?
                    } else {
                        let mut f = std::fs::File::open(win_shard).map_err(|e| e.to_string())?;
                        let mut buf = vec![0u8; window_len];
                        f.seek(SeekFrom::Start(win_start))
                            .map_err(|e| e.to_string())?;
                        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
                        buf
                    };

                    let mut local_entries = Vec::with_capacity(win_tensors.len());
                    for t in win_tensors {
                        let rel_start = (t.absolute_file_offset - win_start) as usize;
                        let rel_end = rel_start + (t.byte_size as usize);
                        if rel_end > window_buf.len() {
                            return Err(format!(
                                "Window buffer underflow for tensor {}: expected {}..{}, got {}",
                                t.name,
                                rel_start,
                                rel_end,
                                window_buf.len()
                            ));
                        }
                        let tensor_slice = &window_buf[rel_start..rel_end];
                        let hash = ggml_rpc_fnv1a_hash(tensor_slice);
                        let out_path = cache_dir.join(format!("{:016x}", hash));

                        let exists = std::fs::metadata(&out_path)
                            .map(|m| m.len() == t.byte_size)
                            .unwrap_or(false);
                        if !exists {
                            let tmp_path = cache_dir.join(format!(
                                "{:016x}.{}.{}.tmp",
                                hash,
                                std::process::id(),
                                worker_idx
                            ));
                            std::fs::write(&tmp_path, tensor_slice).map_err(|e| e.to_string())?;
                            if std::fs::rename(&tmp_path, &out_path).is_err() {
                                let _ = std::fs::remove_file(&tmp_path);
                                if !std::fs::metadata(&out_path)
                                    .map(|m| m.len() == t.byte_size)
                                    .unwrap_or(false)
                                {
                                    std::fs::write(&out_path, tensor_slice)
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                        }
                        link_sparse_rpc_cache_entry(cache_dir, &out_path, &t.name, t.byte_size);
                        local_entries.push((t.name.clone(), hash, t.byte_size));
                    }

                    if let Ok(mut guard) = res_ref.lock() {
                        guard.extend(local_entries);
                    }
                }
                Ok(())
            }));
        }

        for h in handles {
            h.join()
                .map_err(|_| "shard pull worker thread panicked".to_string())??;
        }
        Ok(())
    })?;

    if let Ok(mut guard) = results_mutex.into_inner() {
        for (name, hash, bytes) in guard.drain(..) {
            known_hashes.insert(name.clone(), hash);
            report_hashes.insert(name, hash);
            cached_count += 1;
            downloaded_bytes += bytes;
        }
    }
    if let Ok(idx_json) = serde_json::to_string(&known_hashes) {
        let _ = std::fs::write(&index_path, idx_json);
    }

    let downloaded_mb = ((downloaded_bytes as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
    let bandwidth_saved_pct = if total_model_bytes > 0 {
        ((1.0 - (downloaded_bytes as f64 / total_model_bytes as f64)).max(0.0) * 1000.0).round()
            / 10.0
    } else {
        0.0
    };

    let report = PartialShardPullReport {
        model_source: model_source.to_string(),
        start_layer,
        end_layer_inclusive,
        include_non_block_tensors,
        cached_tensor_count: cached_count,
        downloaded_bytes,
        downloaded_mb,
        total_model_bytes,
        bandwidth_saved_pct,
        cache_dir: cache_dir.to_string_lossy().to_string(),
        tensor_hashes: report_hashes,
    };
    if let Ok(serialized) = serde_json::to_string_pretty(&report) {
        let _ = std::fs::write(&marker_path, serialized);
    }
    Ok(report)
}

/// Builds a **Zero-Layer-Download Sparse GGUF File** on a CPU-only Control Plane / Gateway node:
/// - Allocates a sparse file of exact virtual length `total_file_bytes` (`0 MB` physical disk blocks for holes).
/// - Writes the GGUF Header + Tokenizer Vocabulary (`0..hdr.tensor_data_offset`).
/// - Downloads ONLY CPU-resident `token_embd.weight` and tiny `< 64 KiB` norm/bias tensors.
/// - For all offloaded GPU layer tensors (`blk.0..N` and `output.weight` with `byte_size > 65536`),
///   writes ONLY the 64-byte [`ggml_rpc_sparse_tensor_tag`] at `t.absolute_file_offset`, leaving the remaining
///   97%+ of the model as zero-disk, zero-network Linux/NTFS sparse holes!
/// - When `model_source` is a multi-part split GGUF (`*-00001-of-0000N.gguf`), automatically builds all `N`
///   sibling sparse `.gguf` files in `dest_path.parent()` so `llama-server` natively mmaps the entire split set.
pub fn build_sparse_master_gguf(
    model_source: &str,
    hdr: &GgufHeader,
    dest_path: &Path,
) -> Result<u64, String> {
    build_sparse_master_gguf_for_local_layers(model_source, hdr, dest_path, &BTreeSet::new())
}

/// Materializes a Master `.gguf` where `token_embd`, `output`, `output_norm`, tensors `<= 1 MiB`,
/// AND any transformer layers in `local_layers` (Stage-0 + pinned MTP layers) contain real weights
/// (copied from local NVMe `rpc_cache_dir` with HTTP Range fallback), while remote layers stay
/// 64-byte `CUTMESH_SPARSE_V1:` sparse holes.
pub fn build_sparse_master_gguf_for_local_layers(
    model_source: &str,
    hdr: &GgufHeader,
    dest_path: &Path,
    local_layers: &BTreeSet<u16>,
) -> Result<u64, String> {
    let shards = crate::gguf::expand_split_gguf_sources(model_source);
    if shards.len() > 1 {
        let parent_dir = dest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut total_downloaded = 0u64;
        for (idx, shard_url) in shards.iter().enumerate() {
            let range_mib = if idx == 0 { 12 } else { 2 };
            let shard_hdr = fetch_single_gguf_header(shard_url, range_mib)?;
            let shard_fname = shard_url
                .rsplit('/')
                .next()
                .unwrap_or("shard.gguf")
                .split('?')
                .next()
                .unwrap_or("shard.gguf");
            let shard_dest = if idx == 0 {
                dest_path.to_path_buf()
            } else {
                parent_dir.join(shard_fname)
            };
            total_downloaded += build_single_sparse_master_gguf(
                shard_url,
                &shard_hdr,
                &shard_dest,
                dest_path,
                local_layers,
            )?;
        }
        return Ok(total_downloaded);
    }
    build_single_sparse_master_gguf(model_source, hdr, dest_path, dest_path, local_layers)
}

fn load_local_hash_index(primary_dest_path: &Path, rpc_cache_dir: &Path) -> BTreeMap<String, u64> {
    let manifest_candidate = PathBuf::from(format!("{}.hashes.json", primary_dest_path.to_string_lossy()));
    let mut local_hashes: BTreeMap<String, u64> = std::fs::read_to_string(&manifest_candidate)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    if let Ok(entries) = std::fs::read_dir(rpc_cache_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.starts_with(".cutmesh_tensor_hashes_") && fname.ends_with(".json") {
                if let Ok(raw) = std::fs::read_to_string(entry.path()) {
                    if let Ok(map) = serde_json::from_str::<BTreeMap<String, u64>>(&raw) {
                        local_hashes.extend(map);
                    }
                }
            }
        }
    }
    local_hashes
}

fn read_cached_tensor_bytes(
    t: &GgufTensorDescriptor,
    local_hashes: &BTreeMap<String, u64>,
    rpc_cache_dir: &Path,
) -> Option<Vec<u8>> {
    let h = *local_hashes.get(&t.name)?;
    let cached_path = rpc_cache_dir.join(format!("{:016x}", h));
    if std::fs::metadata(&cached_path)
        .map(|m| m.len() == t.byte_size)
        .unwrap_or(false)
    {
        return std::fs::read(&cached_path).ok();
    }
    None
}

fn fetch_source_byte_range(source: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>, String> {
    if source.starts_with("http://") || source.starts_with("https://") {
        return fetch_http_byte_range(source, start, end_inclusive);
    }
    let len = (end_inclusive.saturating_sub(start)).saturating_add(1) as usize;
    let mut f = std::fs::File::open(source).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; len];
    f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

fn should_materialize_tensor(t: &GgufTensorDescriptor, local_layers: &BTreeSet<u16>) -> bool {
    const UPSTREAM_RPC_HASH_THRESHOLD: u64 = 1_048_576;
    if t.byte_size <= UPSTREAM_RPC_HASH_THRESHOLD || t.layer_index.is_none() {
        return true;
    }
    t.layer_index.map(|l| local_layers.contains(&l)).unwrap_or(false)
}

fn build_single_sparse_master_gguf(
    model_source: &str,
    hdr: &GgufHeader,
    dest_path: &Path,
    primary_dest_path: &Path,
    local_layers: &BTreeSet<u16>,
) -> Result<u64, String> {
    let total_file_bytes = hdr
        .tensors
        .iter()
        .map(|t| t.absolute_file_offset + t.byte_size)
        .max()
        .unwrap_or(hdr.data_section_offset);

    let rpc_cache_dir = crate::gguf::slicer::default_ggml_rpc_cache_dir();
    let local_hashes = load_local_hash_index(primary_dest_path, &rpc_cache_dir);

    if let Ok(meta) = std::fs::metadata(dest_path) {
        if meta.len() == total_file_bytes {
            // Verify that inline RPC tensors, `output.weight`, and all `local_layers` tensors
            // are populated with real bytes rather than `CUTMESH_SPARSE_V1:` holes.
            let mut patch_tensors: Vec<GgufTensorDescriptor> = Vec::new();
            if let Ok(mut f) = std::fs::OpenOptions::new().read(true).write(true).open(dest_path) {
                for t in &hdr.tensors {
                    if t.byte_size > RPC_CACHE_HASH_THRESHOLD && should_materialize_tensor(t, local_layers) {
                        let mut probe = [0u8; 18];
                        if f.seek(SeekFrom::Start(t.absolute_file_offset)).is_ok()
                            && f.read_exact(&mut probe).is_ok()
                            && &probe == b"CUTMESH_SPARSE_V1:"
                        {
                            patch_tensors.push(t.clone());
                        }
                    }
                }
                if !patch_tensors.is_empty() {
                    tracing::info!(
                        "Patching {} tensori locali nello sparse GGUF {} (cache={}, source={})...",
                        patch_tensors.len(),
                        dest_path.display(),
                        local_hashes.len(),
                        model_source
                    );
                    let mut patched_bytes = 0u64;
                    let mut remote_missing: Vec<GgufTensorDescriptor> = Vec::new();
                    for t in patch_tensors {
                        if let Some(bytes) = read_cached_tensor_bytes(&t, &local_hashes, &rpc_cache_dir) {
                            f.seek(SeekFrom::Start(t.absolute_file_offset))
                                .map_err(|e| e.to_string())?;
                            f.write_all(&bytes).map_err(|e| e.to_string())?;
                        } else {
                            remote_missing.push(t);
                        }
                    }
                    remote_missing.sort_by_key(|t| t.absolute_file_offset);
                    const MAX_PATCH_WINDOW: u64 = 64 * 1024 * 1024;
                    for (win_src, win_start, win_end, win_tensors) in
                        coalesce_byte_windows(&remote_missing, model_source, MAX_PATCH_WINDOW)
                    {
                        let window_buf = fetch_source_byte_range(&win_src, win_start, win_end - 1)?;
                        patched_bytes += window_buf.len() as u64;
                        for t in &win_tensors {
                            let rel_start = (t.absolute_file_offset - win_start) as usize;
                            let rel_end = rel_start + (t.byte_size as usize);
                            if rel_end <= window_buf.len() {
                                f.seek(SeekFrom::Start(t.absolute_file_offset))
                                    .map_err(|e| e.to_string())?;
                                f.write_all(&window_buf[rel_start..rel_end])
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    let _ = f.flush();
                    tracing::info!(
                        "Patching GGUF sparso completato ({:.2} MB scaricati via HTTP Range)",
                        (patched_bytes as f64) / (1024.0 * 1024.0)
                    );
                    return Ok(patched_bytes);
                }
            }
            return Ok(0);
        }
    }

    if let Some(parent) = dest_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let part_path = dest_path.with_extension("gguf.sparse.part");
    let mut file = std::fs::File::create(&part_path).map_err(|e| e.to_string())?;
    file.set_len(total_file_bytes).map_err(|e| e.to_string())?;

    let mut actual_downloaded: u64 = 0;

    // 1. Write GGUF Header + Metadata + Vocabulary (`0..hdr.data_section_offset`)
    let hdr_end = hdr.data_section_offset.max(1);
    let hdr_bytes = fetch_source_byte_range(model_source, 0, hdr_end - 1)?;
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    file.write_all(&hdr_bytes).map_err(|e| e.to_string())?;
    actual_downloaded += hdr_bytes.len() as u64;

    // 2. Partition tensors: materialize (`<= 1 MiB`, `token_embd.weight`, `output.weight`,
    //    and any transformer block in `local_layers`), leave remote blocks sparse.
    let mut need_full: Vec<GgufTensorDescriptor> = Vec::new();
    for t in &hdr.tensors {
        if !should_materialize_tensor(t, local_layers) {
            let tag = ggml_rpc_sparse_tensor_tag(&t.name, t.byte_size);
            file.seek(SeekFrom::Start(t.absolute_file_offset))
                .map_err(|e| e.to_string())?;
            file.write_all(&tag).map_err(|e| e.to_string())?;
        } else if let Some(bytes) = read_cached_tensor_bytes(t, &local_hashes, &rpc_cache_dir) {
            file.seek(SeekFrom::Start(t.absolute_file_offset))
                .map_err(|e| e.to_string())?;
            file.write_all(&bytes).map_err(|e| e.to_string())?;
        } else {
            need_full.push(t.clone());
        }
    }

    need_full.sort_by_key(|t| t.absolute_file_offset);

    // 3. Coalesce and fetch ONLY remaining materialized tensors absent from local NVMe cache
    tracing::info!(
        "Materializzazione GGUF sparso locale ({} tensori da HTTP Range, cache={})...",
        need_full.len(),
        local_hashes.len()
    );
    const MAX_CHUNK_WINDOW_BYTES: u64 = 64 * 1024 * 1024;
    for (win_src, win_start, win_end, win_tensors) in
        coalesce_byte_windows(&need_full, model_source, MAX_CHUNK_WINDOW_BYTES)
    {
        let window_buf = fetch_source_byte_range(&win_src, win_start, win_end - 1)?;
        actual_downloaded += window_buf.len() as u64;

        for t in &win_tensors {
            let rel_start = (t.absolute_file_offset - win_start) as usize;
            let rel_end = rel_start + (t.byte_size as usize);
            if rel_end <= window_buf.len() {
                file.seek(SeekFrom::Start(t.absolute_file_offset))
                    .map_err(|e| e.to_string())?;
                file.write_all(&window_buf[rel_start..rel_end])
                    .map_err(|e| e.to_string())?;
            }
        }
    }

    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    std::fs::rename(&part_path, dest_path).map_err(|e| e.to_string())?;
    Ok(actual_downloaded)
}

/// Ensures the `models/<fname>.hashes.json` manifest (`tensor_name -> real_fnv1a_hash`, ~24 KB) is complete
/// and returns `(sparse_hash, real_hash)` translation pairs for `spawn_client_iroh_rpc_bridge`.
/// When Workers run the updated image, `worker_reported_hashes` already contains 100% of the hashes (`0 Bytes` streamed).
pub fn ensure_tensor_hash_manifest(
    model_source: &str,
    hdr: &GgufHeader,
    manifest_path: &Path,
    worker_reported_hashes: &BTreeMap<String, u64>,
) -> Result<Vec<(u64, u64)>, String> {
    let mut manifest: BTreeMap<String, u64> = std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let rpc_cache_hashes = load_local_hash_index(
        manifest_path,
        &crate::gguf::slicer::default_ggml_rpc_cache_dir(),
    );
    let mut dirty = false;
    for (k, v) in rpc_cache_hashes {
        if manifest.insert(k, v) != Some(v) {
            dirty = true;
        }
    }
    for (k, &v) in worker_reported_hashes {
        if manifest.insert(k.clone(), v) != Some(v) {
            dirty = true;
        }
    }

    let mut missing: Vec<GgufTensorDescriptor> = hdr
        .tensors
        .iter()
        .filter(|t| {
            t.byte_size > RPC_CACHE_HASH_THRESHOLD
                && (t.layer_index.is_some() || t.name == "output.weight")
                && !manifest.contains_key(&t.name)
        })
        .cloned()
        .collect();

    let missing_mb: f64 = missing.iter().map(|t| t.byte_size as f64).sum::<f64>() / (1024.0 * 1024.0);
    tracing::info!(
        "Manifest hash tensori: presenti={}, mancanti={} ({:.1} MB)",
        manifest.len(),
        missing.len(),
        missing_mb
    );

    if !missing.is_empty() {
        missing.sort_by(|a, b| {
            a.shard_source
                .cmp(&b.shard_source)
                .then(a.absolute_file_offset.cmp(&b.absolute_file_offset))
        });
        const MAX_CHUNK_WINDOW_BYTES: u64 = 64 * 1024 * 1024;
        let windows = coalesce_byte_windows(&missing, model_source, MAX_CHUNK_WINDOW_BYTES);

        // Stream missing hashes in parallel (8 threads) in memory without ever writing weights to disk
        let cursor = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let win_arc = std::sync::Arc::new(windows);
        let mut handles = Vec::new();
        for _ in 0..8.min(win_arc.len().max(1)) {
            let c_clone = cursor.clone();
            let w_clone = win_arc.clone();
            handles.push(std::thread::spawn(move || -> Result<Vec<(String, u64)>, String> {
                let mut local_out = Vec::new();
                loop {
                    let i = c_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= w_clone.len() {
                        break;
                    }
                    let (ref w_src, w_start, w_end, ref tensors) = w_clone[i];
                    let buf = fetch_http_byte_range(w_src, w_start, w_end - 1)?;
                    for t in tensors {
                        let rel_s = (t.absolute_file_offset - w_start) as usize;
                        let rel_e = rel_s + (t.byte_size as usize);
                        if rel_e <= buf.len() {
                            let h = ggml_rpc_fnv1a_hash(&buf[rel_s..rel_e]);
                            local_out.push((t.name.clone(), h));
                        }
                    }
                }
                Ok(local_out)
            }));
        }

        for h in handles {
            let pairs = h.join().map_err(|_| "hash worker thread panicked".to_string())??;
            for (name, hash) in pairs {
                manifest.insert(name, hash);
                dirty = true;
            }
        }
    }

    if dirty {
        if let Some(parent) = manifest_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&manifest) {
            let _ = std::fs::write(manifest_path, json);
        }
    }

    let mut translations = Vec::new();
    for t in &hdr.tensors {
        if let Some(&real_hash) = manifest.get(&t.name) {
            let sparse_hash = ggml_rpc_sparse_tensor_hash(&t.name, t.byte_size);
            translations.push((sparse_hash, real_hash));
        }
    }
    Ok(translations)
}

/// Resolves the canonical remote Hugging Face `.gguf` URL for a local model file so remote
/// Workers can autonomously pull their assigned layer shards directly from Hugging Face CDN
/// instead of streaming gigabytes of weights from the Gateway's home upload connection.
pub fn resolve_remote_gguf_url(model_path: &Path, explicit_url: Option<&str>) -> Option<String> {
    if let Some(u) = explicit_url {
        let trimmed = u.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Ok(env_url) = std::env::var("CUTMESH_MODEL_URL") {
        let trimmed = env_url.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let path_str = model_path.to_string_lossy();
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        return Some(path_str.to_string());
    }
    let sidecar = model_path.with_extension("gguf.url");
    if let Ok(raw) = std::fs::read_to_string(&sidecar) {
        let trimmed = raw.trim();
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            return Some(trimmed.to_string());
        }
    }

    let policy = crate::models::ModelPolicy::resolve(&path_str);
    if !policy.canonical_hf_url.is_empty() {
        return Some(policy.canonical_hf_url);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_two_layer_plus_mtp_test_gguf(path: &Path) {
        let mut buf = Vec::new();
        let push_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        let specs: [(&str, u8); 4] = [
            ("blk.0.ffn_gate.weight", 0x11),
            ("blk.1.ffn_gate.weight", 0x22),
            ("blk.2.ffn_gate.weight", 0x33),
            ("output.weight", 0x44),
        ];
        let tensor_bytes: u64 = 1_100_000; // > 1 MiB threshold
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        push_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut buf, "qwen35");

        let mut rel = 0u64;
        for (name, _) in specs {
            push_str(&mut buf, name);
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&275_000u64.to_le_bytes());
            buf.extend_from_slice(&1u64.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&rel.to_le_bytes());
            rel += tensor_bytes;
        }
        let rem = buf.len() % 32;
        if rem != 0 {
            buf.resize(buf.len() + (32 - rem), 0);
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
        for (_, fill) in specs {
            f.write_all(&vec![fill; tensor_bytes as usize]).unwrap();
        }
    }

    #[test]
    fn master_gguf_materializes_local_layers_from_cache_and_keeps_remote_sparse() {
        let dir = std::env::temp_dir().join(format!("cutmesh_rig_local_layers_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let llama_cache_root = dir.join("llama_cache");
        std::env::set_var("LLAMA_CACHE", &llama_cache_root);
        let cache_dir = crate::gguf::slicer::default_ggml_rpc_cache_dir();
        let _ = std::fs::create_dir_all(&cache_dir);

        let src_gguf = dir.join("src.gguf");
        let dst_gguf = dir.join("master.gguf");
        build_two_layer_plus_mtp_test_gguf(&src_gguf);
        let hdr = GgufHeader::parse_file(&src_gguf).unwrap();

        // Pre-populate blk.0 and blk.2 into rpc cache hash index (simulating prewarmed NVMe cache)
        let src_bytes = std::fs::read(&src_gguf).unwrap();
        let mut hash_map: BTreeMap<String, u64> = BTreeMap::new();
        for t in &hdr.tensors {
            if t.name.starts_with("blk.0.") || t.name.starts_with("blk.2.") {
                let slice = &src_bytes[t.absolute_file_offset as usize..(t.absolute_file_offset + t.byte_size) as usize];
                let h = ggml_rpc_fnv1a_hash(slice);
                std::fs::write(cache_dir.join(format!("{:016x}", h)), slice).unwrap();
                hash_map.insert(t.name.clone(), h);
            }
        }
        std::fs::write(
            cache_dir.join(".cutmesh_tensor_hashes_test.json"),
            serde_json::to_string(&hash_map).unwrap(),
        )
        .unwrap();

        let mut local_layers = BTreeSet::new();
        local_layers.insert(0);
        local_layers.insert(2);

        let downloaded = build_sparse_master_gguf_for_local_layers(
            &src_gguf.to_string_lossy(),
            &hdr,
            &dst_gguf,
            &local_layers,
        )
        .unwrap();

        // Only header + output.weight (~1.1 MB) needed copying from source; blk.0 + blk.2 came from cache!
        assert!(downloaded < 1_500_000, "unexpected download bytes: {}", downloaded);

        let out_bytes = std::fs::read(&dst_gguf).unwrap();
        for t in &hdr.tensors {
            let off = t.absolute_file_offset as usize;
            let prefix = &out_bytes[off..off + 18];
            if t.name == "blk.1.ffn_gate.weight" {
                assert_eq!(prefix, b"CUTMESH_SPARSE_V1:", "remote layer blk.1 must remain sparse");
            } else {
                assert_ne!(prefix, b"CUTMESH_SPARSE_V1:", "local tensor {} must be materialized", t.name);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stage0_master_election_prefers_16gb_headroom_over_8gb_card() {
        let gpu_8gb_fast_bw = GpuNodeCapabilities {
            node_name: "small-8gb".into(),
            endpoint_id: "node-8gb".into(),
            gpu_name: "RTX 3060 Ti".into(),
            compute_capability: "8.6".into(),
            sm_count: 38,
            cuda_cores: 4864,
            boost_clock_mhz: 1665,
            mem_clock_mhz: 7000,
            mem_bus_width_bits: 256,
            vram_total_mb: 8192,
            vram_free_mb: 7701,
            fp16_tflops_peak: 16.2,
            mem_bandwidth_gb_s_peak: 448.0,
            power_current_watts: 30.0,
            power_limit_watts: 200.0,
            temperature_c: 45,
            gpu_utilization_pct: 5,
            user_power_budget_pct: 100,
            allocated_power_budget_watts: 200.0,
            effective_available_tflops: 16.2,
            effective_available_bandwidth_gb_s: 448.0,
            direct_udp_reachable: true,
            measured_rtt_ms: 20.0,
            measured_link_mb_s: 12.0,
        };
        let gpu_16gb_lower_bw = GpuNodeCapabilities {
            node_name: "large-16gb".into(),
            endpoint_id: "node-16gb".into(),
            gpu_name: "RTX 4060 Ti 16GB".into(),
            compute_capability: "8.9".into(),
            sm_count: 34,
            cuda_cores: 4352,
            boost_clock_mhz: 2535,
            mem_clock_mhz: 9000,
            mem_bus_width_bits: 128,
            vram_total_mb: 16380,
            vram_free_mb: 15818,
            fp16_tflops_peak: 22.1,
            mem_bandwidth_gb_s_peak: 288.0,
            power_current_watts: 25.0,
            power_limit_watts: 165.0,
            temperature_c: 42,
            gpu_utilization_pct: 2,
            user_power_budget_pct: 100,
            allocated_power_budget_watts: 165.0,
            effective_available_tflops: 22.1,
            effective_available_bandwidth_gb_s: 288.0,
            direct_udp_reachable: true,
            measured_rtt_ms: 55.0,
            measured_link_mb_s: 8.0,
        };

        let s8 = compute_stage0_master_score(&gpu_8gb_fast_bw, 1536, 2011.4);
        let s16 = compute_stage0_master_score(&gpu_16gb_lower_bw, 1536, 2011.4);
        assert!(
            s16 > s8 * 5.0,
            "16 GB GPU (score={}) must strongly beat 8 GB GPU (score={}) for Stage-0 Master",
            s16,
            s8
        );
    }
}


