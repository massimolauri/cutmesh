//! Zero-config networking & identity for P2P nodes.
//!
//! Goals:
//! - A container must start and be reachable **directly** (no relay) without manual flags.
//! - Every node gets a **unique, persistent** Iroh identity (no shared default seed).
//! - The public `ip:port` advertised in the ticket is the one peers can actually reach
//!   (e.g. the Vast.ai external UDP port mapped to the internal QUIC port).

use iroh::SecretKey;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{info, warn};

/// Directory where the node keeps its persistent state (identity key, pid files).
/// Override with `CUTMESH_DATA_DIR`.
pub fn cutmesh_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CUTMESH_DATA_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cutmesh")
}

/// Loads the node identity key, creating a new random one on first start.
///
/// - `legacy_seed = Some(n)`: deterministic key (tests only — two nodes with the same
///   seed get the same `EndpointId`).
/// - Otherwise the key is stored in `CUTMESH_KEY_FILE` or
///   `<data_dir>/node-<port>.key`, so multiple workers on the same host never collide.
pub fn load_or_create_node_key(bind_port: u16, legacy_seed: Option<u8>) -> Result<SecretKey, String> {
    if let Some(seed_byte) = legacy_seed {
        warn!(
            "Identità deterministica da --seed {} (solo per test): nodi con lo stesso seed avranno lo stesso EndpointId",
            seed_byte
        );
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = seed_byte.wrapping_add(i as u8).wrapping_mul(53);
        }
        return Ok(SecretKey::from_bytes(&seed));
    }

    let explicit_file = std::env::var("CUTMESH_KEY_FILE").ok();
    if bind_port == 0 && explicit_file.is_none() {
        // Random port = ephemeral node: a shared `node-0.key` would give every
        // ephemeral node on this host the same identity.
        return Ok(SecretKey::generate());
    }
    let path = explicit_file
        .map(PathBuf::from)
        .unwrap_or_else(|| cutmesh_data_dir().join(format!("node-{}.key", bind_port)));

    if let Ok(raw) = std::fs::read_to_string(&path) {
        let bytes = hex::decode(raw.trim())
            .map_err(|e| format!("Chiave nodo corrotta in {:?}: {}", path, e))?;
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| format!("Chiave nodo in {:?} non è lunga 32 byte", path))?;
        return Ok(SecretKey::from_bytes(&arr));
    }

    let key = SecretKey::generate();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Impossibile creare {:?}: {}", parent, e))?;
    }
    std::fs::write(&path, hex::encode(key.to_bytes()))
        .map_err(|e| format!("Impossibile salvare la chiave nodo in {:?}: {}", path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    info!("Nuova identità nodo generata e salvata in {:?}", path);
    Ok(key)
}

/// Public addresses (`ip:port`) on which this node's QUIC socket is reachable from the internet.
///
/// Resolution order:
/// 1. `CUTMESH_PUBLIC_ADDR` — explicit, comma separated `ip:port` list (wins over everything).
/// 2. Public IP: `CUTMESH_PUBLIC_IP` → `PUBLIC_IPADDR` (Vast.ai) → HTTP auto-detect
///    (disable with `CUTMESH_AUTODETECT_PUBLIC_IP=0`).
/// 3. Public port: `CUTMESH_PUBLIC_PORT` → `VAST_UDP_PORT_<internal_port>` (Vast.ai) → internal port.
pub async fn detect_public_addrs(internal_port: u16) -> Vec<SocketAddr> {
    let mut out = Vec::new();

    if let Ok(raw) = std::env::var("CUTMESH_PUBLIC_ADDR") {
        for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match part.parse::<SocketAddr>() {
                Ok(addr) => {
                    if !out.contains(&addr) {
                        out.push(addr);
                    }
                }
                Err(e) => warn!("CUTMESH_PUBLIC_ADDR: '{}' non valido ({}), ignorato", part, e),
            }
        }
    }

    let public_ip = match env_ip("CUTMESH_PUBLIC_IP").or_else(|| env_ip("PUBLIC_IPADDR")) {
        Some(ip) => Some(ip),
        None if std::env::var("CUTMESH_AUTODETECT_PUBLIC_IP").as_deref() != Ok("0") => {
            autodetect_public_ip().await
        }
        None => None,
    };

    let public_port = env_port("CUTMESH_PUBLIC_PORT")
        .or_else(|| env_port(&format!("VAST_UDP_PORT_{}", internal_port)))
        .unwrap_or(internal_port);

    if let Some(ip) = public_ip {
        if public_port != 0 {
            let auto_addr = SocketAddr::new(ip, public_port);
            if !out.contains(&auto_addr) {
                out.push(auto_addr);
            }
        }
    }
    out
}

fn env_ip(name: &str) -> Option<IpAddr> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().parse::<IpAddr>() {
        Ok(ip) => Some(ip),
        Err(_) => {
            warn!("{}='{}' non è un IP valido, ignorato", name, raw.trim());
            None
        }
    }
}

fn env_port(name: &str) -> Option<u16> {
    std::env::var(name).ok()?.trim().parse::<u16>().ok()
}

/// Best-effort public IPv4 discovery via plain HTTP (no extra dependencies).
async fn autodetect_public_ip() -> Option<IpAddr> {
    for host in ["api.ipify.org", "ipv4.icanhazip.com"] {
        match tokio::time::timeout(Duration::from_secs(3), http_get_body(host)).await {
            Ok(Ok(body)) => {
                if let Ok(ip) = body.trim().parse::<IpAddr>() {
                    info!("IP pubblico rilevato automaticamente via {}: {}", host, ip);
                    return Some(ip);
                }
            }
            Ok(Err(e)) => warn!("Rilevamento IP pubblico via {} fallito: {}", host, e),
            Err(_) => warn!("Rilevamento IP pubblico via {} in timeout", host),
        }
    }
    None
}

async fn http_get_body(host: &str) -> Result<String, String> {
    let mut stream = TcpStream::connect((host, 80))
        .await
        .map_err(|e| e.to_string())?;
    let req = format!(
        "GET / HTTP/1.1\r\nHost: {}\r\nUser-Agent: cutmesh\r\nConnection: close\r\n\r\n",
        host
    );
    stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf);
    let body = text
        .split("\r\n\r\n")
        .nth(1)
        .ok_or_else(|| "risposta HTTP senza body".to_string())?;
    Ok(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_seed_is_deterministic() {
        let a = load_or_create_node_key(0, Some(7)).unwrap();
        let b = load_or_create_node_key(0, Some(7)).unwrap();
        assert_eq!(a.public(), b.public());
    }

    #[test]
    fn persistent_key_is_reused_and_unique_per_file() {
        let dir = std::env::temp_dir().join(format!("cutmesh-key-test-{}", std::process::id()));
        let file_a = dir.join("a.key");
        let file_b = dir.join("b.key");
        std::env::set_var("CUTMESH_KEY_FILE", &file_a);
        let a1 = load_or_create_node_key(1, None).unwrap();
        let a2 = load_or_create_node_key(1, None).unwrap();
        std::env::set_var("CUTMESH_KEY_FILE", &file_b);
        let b1 = load_or_create_node_key(1, None).unwrap();
        std::env::remove_var("CUTMESH_KEY_FILE");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(a1.public(), a2.public());
        assert_ne!(a1.public(), b1.public());
    }
}
