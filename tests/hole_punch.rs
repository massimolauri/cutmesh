//! Hole punching end-to-end: the client gets a ticket with **no IP addresses** (only the
//! signaling relay), exactly like a worker behind a home NAT. The connection must be
//! coordinated through the relay and then upgraded to a direct path before any data flows.
//!
//! Needs internet access to the signaling relay (n0 public relays by default, or
//! `CUTMESH_SIGNAL_RELAY`). Run with: `cargo test --test hole_punch -- --ignored --nocapture`

use cutmesh::rpc_tunnel::{
    probe_worker_hardware_iroh, spawn_worker_iroh_rpc_server, RpcNodeTicket,
};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "richiede internet (relay di segnalazione)"]
async fn relay_only_ticket_is_upgraded_to_direct_path() {
    std::env::set_var("CUTMESH_AUTODETECT_PUBLIC_IP", "0");
    std::env::set_var("CUTMESH_LINK_PROBE_MB", "4");
    std::env::remove_var("CUTMESH_PUBLIC_ADDR");

    let (full, _, _) =
        spawn_worker_iroh_rpc_server("nat-worker", 0, "127.0.0.1:9", None, false, 100)
            .await
            .expect("worker must start");
    let relays: Vec<iroh::RelayUrl> = full.endpoint_addr.relay_urls().cloned().collect();
    assert!(!relays.is_empty(), "worker did not register with a signaling relay");

    // Strip every IP address: the client only knows identity + signaling relay.
    let mut relay_only = iroh::EndpointAddr::new(full.endpoint_addr.id);
    for r in relays {
        relay_only = relay_only.with_relay_url(r);
    }
    let ticket = RpcNodeTicket {
        endpoint_addr: relay_only,
    };
    assert!(ticket.direct_addrs().is_empty());

    let caps = probe_worker_hardware_iroh(&ticket, false)
        .await
        .expect("hole punching must produce a direct path");
    assert!(caps.direct_udp_reachable, "data path is not direct");
    println!(
        "hole punching OK: direct path rtt={:.2} ms, link={:.1} MB/s (relay used only for signaling: {:?})",
        caps.measured_rtt_ms,
        caps.measured_link_mb_s,
        ticket.relay_urls()
    );
}
