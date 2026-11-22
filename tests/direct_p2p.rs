//! End-to-end check of the direct P2P transport (no relay):
//! a worker endpoint is spawned locally and probed over a real QUIC connection.

use cutmesh::rpc_tunnel::{probe_worker_hardware_iroh, spawn_worker_iroh_rpc_server};

#[tokio::test(flavor = "multi_thread")]
async fn worker_is_reachable_directly_and_ticket_has_no_relay() {
    std::env::set_var("CUTMESH_SIGNAL_RELAY", "off");
    std::env::set_var("CUTMESH_AUTODETECT_PUBLIC_IP", "0");
    std::env::set_var("CUTMESH_LINK_PROBE_MB", "4");
    std::env::set_var("CUTMESH_PUBLIC_ADDR", "203.0.113.7:40495");

    // Two ephemeral workers must get distinct identities without passing any seed.
    let (ticket_a, _, _) =
        spawn_worker_iroh_rpc_server("test-worker-a", 0, "127.0.0.1:9", None, false, 100)
            .await
            .expect("worker A must start");
    let (ticket_b, _, _) =
        spawn_worker_iroh_rpc_server("test-worker-b", 0, "127.0.0.1:9", None, false, 100)
            .await
            .expect("worker B must start");
    assert_ne!(ticket_a.endpoint_addr.id, ticket_b.endpoint_addr.id);

    // Ticket: no relay URLs, advertised public address present.
    assert_eq!(ticket_a.endpoint_addr.relay_urls().count(), 0);
    let addrs = ticket_a.direct_addrs();
    assert!(
        addrs.iter().any(|a| a.to_string() == "203.0.113.7:40495"),
        "public addr missing from ticket: {:?}",
        addrs
    );

    // Real direct QUIC connection + throughput probe.
    let caps = probe_worker_hardware_iroh(&ticket_a, false)
        .await
        .expect("direct probe must succeed");
    assert!(caps.direct_udp_reachable);
    assert!(caps.measured_link_mb_s > 0.0, "link probe not measured");
    println!(
        "direct path OK: rtt={:.2} ms, link={:.1} MB/s, addrs={:?}",
        caps.measured_rtt_ms, caps.measured_link_mb_s, addrs
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn unreachable_worker_fails_fast_with_clear_error() {
    use cutmesh::rpc_tunnel::RpcNodeTicket;

    std::env::set_var("CUTMESH_SIGNAL_RELAY", "off");
    std::env::set_var("CUTMESH_AUTODETECT_PUBLIC_IP", "0");
    // Valid identity, but only an unroutable TEST-NET address and no relay.
    let (real, _, _) =
        spawn_worker_iroh_rpc_server("test-worker-c", 0, "127.0.0.1:9", None, false, 100)
            .await
            .expect("worker C must start");
    let only_unroutable = iroh::EndpointAddr::new(real.endpoint_addr.id)
        .with_ip_addr("203.0.113.9:40000".parse().unwrap());
    let ticket = RpcNodeTicket {
        endpoint_addr: only_unroutable,
    };

    let t0 = std::time::Instant::now();
    let err = probe_worker_hardware_iroh(&ticket, false)
        .await
        .expect_err("must not connect via relay or hang");
    assert!(t0.elapsed().as_secs() <= 20, "took too long: {:?}", t0.elapsed());
    assert!(
        err.contains("non raggiungibile") || err.contains("fallita"),
        "unexpected error: {}",
        err
    );
    println!("errore atteso: {}", err);
}

fn write_synthetic_gguf(path: &std::path::Path) {
    use std::io::Write;
    let mut buf = Vec::new();
    let push_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&14u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
    push_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    push_str(&mut buf, "qwen2");

    let suffixes = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    let tensor_bytes: u64 = 256 * 256 * 4; // 256 KiB per F32 tensor (> 64 KiB threshold)
    let mut rel_offset = 0u64;
    for layer in 0..2 {
        for suffix in suffixes {
            let name = format!("blk.{}.{}.weight", layer, suffix);
            push_str(&mut buf, &name);
            buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
            buf.extend_from_slice(&256u64.to_le_bytes());
            buf.extend_from_slice(&256u64.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
            buf.extend_from_slice(&rel_offset.to_le_bytes());
            rel_offset += tensor_bytes;
        }
    }
    let rem = buf.len() % 32;
    if rem != 0 {
        buf.resize(buf.len() + (32 - rem), 0);
    }
    let mut f = std::fs::File::create(path).expect("create temp gguf");
    f.write_all(&buf).expect("write gguf header");
    let data_chunk = vec![0x2Au8; tensor_bytes as usize];
    for _ in 0..14 {
        f.write_all(&data_chunk).expect("write gguf tensor");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_prewarms_shard_via_alpn_prewarm_shard() {
    use cutmesh::rpc_tunnel::{request_worker_prewarm_shard, PrewarmShardRequest};

    std::env::set_var("CUTMESH_SIGNAL_RELAY", "off");
    std::env::set_var("CUTMESH_AUTODETECT_PUBLIC_IP", "0");
    let pid = std::process::id();
    let tmp_cache = std::env::temp_dir().join(format!("cutmesh_test_rpc_cache_{}", pid));
    let tmp_gguf = std::env::temp_dir().join(format!("cutmesh_test_prewarm_{}.gguf", pid));
    let _ = std::fs::remove_dir_all(&tmp_cache);
    let _ = std::fs::remove_file(&tmp_gguf);
    std::env::set_var("LLAMA_CACHE", &tmp_cache);
    write_synthetic_gguf(&tmp_gguf);

    let (ticket, _stats, _caps) =
        spawn_worker_iroh_rpc_server("test-worker-prewarm", 0, "127.0.0.1:9", None, false, 100)
            .await
            .expect("worker must start");

    let req = PrewarmShardRequest {
        model_url: tmp_gguf.to_string_lossy().to_string(),
        start_layer: 0,
        end_layer_inclusive: 1,
        include_non_block_tensors: false,
    };

    let rep = request_worker_prewarm_shard(&ticket, false, &req)
        .await
        .expect("ALPN_PREWARM_SHARD must succeed");

    assert!(
        rep.cached_tensor_count >= 14,
        "expected >= 14 cached tensors (including Q/K/V/O > 64KB), got {}",
        rep.cached_tensor_count
    );
    assert!(
        rep.downloaded_mb >= 3.0,
        "expected >= 3.0 MB cached for 14 synthetic tensors, got {}",
        rep.downloaded_mb
    );

    // Second call must hit the fast completion marker in < 50 ms
    let t0 = std::time::Instant::now();
    let rep2 = request_worker_prewarm_shard(&ticket, false, &req)
        .await
        .expect("second ALPN_PREWARM_SHARD must succeed");
    assert_eq!(rep2.cached_tensor_count, rep.cached_tensor_count);
    println!(
        "ALPN_PREWARM_SHARD OK: {} tensori ({:.2} MB), secondo hit in {:?}",
        rep.cached_tensor_count,
        rep.downloaded_mb,
        t0.elapsed()
    );
    let _ = std::fs::remove_dir_all(&tmp_cache);
    let _ = std::fs::remove_file(&tmp_gguf);
}

#[tokio::test(flavor = "multi_thread")]
async fn synrpc02_fp16_bridge_roundtrips_over_direct_iroh_quic() {
    use cutmesh::rpc_tunnel::spawn_client_iroh_rpc_bridge;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    std::env::set_var("CUTMESH_SIGNAL_RELAY", "off");
    std::env::set_var("CUTMESH_AUTODETECT_PUBLIC_IP", "0");
    std::env::remove_var("CUTMESH_WIRE_FP16");

    // Mock local ggml-rpc-server speaking genuine HELLO + SET_TENSOR + GET_TENSOR framing
    let mock_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap().to_string();

    let server_task = tokio::spawn(async move {
        let (mut sock, _) = mock_listener.accept().await.unwrap();
        let mut stored_tensor = Vec::new();
        loop {
            let mut hdr = [0u8; 9];
            if sock.read_exact(&mut hdr).await.is_err() {
                break;
            }
            let cmd = hdr[0];
            let req_size = u64::from_le_bytes(hdr[1..9].try_into().unwrap()) as usize;
            let mut payload = vec![0u8; req_size];
            sock.read_exact(&mut payload).await.unwrap();
            match cmd {
                14 => {
                    // HELLO -> respond [rsp_size=4][1, 0, 0, 0]
                    sock.write_all(&4u64.to_le_bytes()).await.unwrap();
                    sock.write_all(&[1, 0, 0, 0]).await.unwrap();
                }
                6 => {
                    // SET_TENSOR -> store data slice (after 296B rpc_tensor + 1B cache_flag + 8B offset = 305)
                    stored_tensor = payload[305..].to_vec();
                }
                8 => {
                    // GET_TENSOR -> return stored F32 tensor
                    sock.write_all(&(stored_tensor.len() as u64).to_le_bytes()).await.unwrap();
                    sock.write_all(&stored_tensor).await.unwrap();
                }
                _ => {}
            }
        }
    });

    let (ticket, stats, _) =
        spawn_worker_iroh_rpc_server("test-worker-v2", 0, &mock_addr, None, false, 100)
            .await
            .expect("worker must start");

    let (bridge_addr, _client_stats) = spawn_client_iroh_rpc_bridge(ticket, false)
        .await
        .expect("bridge must start");

    let mut cli = TcpStream::connect(&bridge_addr).await.unwrap();

    // 1. HELLO
    let hello_payload = vec![0u8; 4];
    cli.write_all(&[14]).await.unwrap();
    cli.write_all(&(hello_payload.len() as u64).to_le_bytes()).await.unwrap();
    cli.write_all(&hello_payload).await.unwrap();
    let mut rsp_len_buf = [0u8; 8];
    cli.read_exact(&mut rsp_len_buf).await.unwrap();
    assert_eq!(u64::from_le_bytes(rsp_len_buf), 4);
    let mut hello_rsp = [0u8; 4];
    cli.read_exact(&mut hello_rsp).await.unwrap();

    // 2. SET_TENSOR with 5120 F32 floats (20,480 B -> compressed to 10,240 B over QUIC!)
    let mut act_bytes = Vec::with_capacity(5120 * 4);
    for i in 0..5120usize {
        let v = ((i as f32) - 2560.0) * 0.01;
        act_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut set_payload = vec![0u8; 296];
    set_payload[8..12].copy_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
    set_payload.push(0); // cache_flag = 0
    set_payload.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
    set_payload.extend_from_slice(&act_bytes);
    cli.write_all(&[6]).await.unwrap();
    cli.write_all(&(set_payload.len() as u64).to_le_bytes()).await.unwrap();
    cli.write_all(&set_payload).await.unwrap();

    // 3. GET_TENSOR 20,480 B F32
    let mut get_payload = vec![0u8; 296];
    get_payload[8..12].copy_from_slice(&0u32.to_le_bytes());
    get_payload.extend_from_slice(&0u64.to_le_bytes());
    get_payload.extend_from_slice(&20_480u64.to_le_bytes());
    cli.write_all(&[8]).await.unwrap();
    cli.write_all(&(get_payload.len() as u64).to_le_bytes()).await.unwrap();
    cli.write_all(&get_payload).await.unwrap();

    cli.read_exact(&mut rsp_len_buf).await.unwrap();
    assert_eq!(u64::from_le_bytes(rsp_len_buf), 20_480);
    let mut got = vec![0u8; 20_480];
    cli.read_exact(&mut got).await.unwrap();

    // Check values round-tripped closely via FP16 and QUIC bytes < raw 40,960 bytes
    for i in 0..5120usize {
        let expected = ((i as f32) - 2560.0) * 0.01;
        let actual = f32::from_le_bytes(got[i * 4..i * 4 + 4].try_into().unwrap());
        assert!((actual - expected).abs() <= 0.05, "mismatch at {}: {} vs {}", i, actual, expected);
    }
    let quic_rx = stats.bytes_rx.load(std::sync::atomic::Ordering::Relaxed);
    let quic_tx = stats.bytes_tx.load(std::sync::atomic::Ordering::Relaxed);
    assert!(quic_rx < 14_000, "expected compressed SET_TENSOR QUIC RX < 14KB, got {}", quic_rx);
    assert!(quic_tx < 14_000, "expected compressed GET_TENSOR QUIC TX < 14KB, got {}", quic_tx);

    drop(cli);
    server_task.abort();
}

