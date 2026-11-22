//! Process hygiene for long-running nodes: single-instance locks and cleanup of
//! orphaned child processes left behind by a previous run (e.g. a `ggml-rpc-server`
//! still holding the RPC port after the worker was killed).

use std::path::PathBuf;
use tracing::{info, warn};

use crate::netcfg::cutmesh_data_dir;

fn pid_file(name: &str) -> PathBuf {
    cutmesh_data_dir().join(format!("{}.pid", name))
}

fn read_pid(name: &str) -> Option<u32> {
    std::fs::read_to_string(pid_file(name))
        .ok()?
        .trim()
        .parse::<u32>()
        .ok()
}

/// Records `pid` under `name` (e.g. `rpc-50052`).
pub fn write_pid(name: &str, pid: u32) {
    let path = pid_file(name);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, pid.to_string()) {
        warn!("Impossibile scrivere {:?}: {}", path, e);
    }
}

pub fn remove_pid(name: &str) {
    let _ = std::fs::remove_file(pid_file(name));
}

/// Returns the command line of a live process whose cmdline contains `needle`.
#[cfg(target_os = "linux")]
fn live_process_matching(pid: u32, needle: &str) -> Option<String> {
    let raw = std::fs::read(format!("/proc/{}/cmdline", pid)).ok()?;
    let cmd = String::from_utf8_lossy(&raw).replace('\0', " ");
    if cmd.contains(needle) {
        Some(cmd.trim().to_string())
    } else {
        None
    }
}

#[cfg(not(target_os = "linux"))]
fn live_process_matching(_pid: u32, _needle: &str) -> Option<String> {
    None
}

/// Fails if another live process matching `needle` already owns the lock `name`;
/// otherwise takes the lock with the current pid.
pub fn acquire_single_instance(name: &str, needle: &str) -> Result<(), String> {
    let me = std::process::id();
    if let Some(pid) = read_pid(name) {
        if pid != me {
            if let Some(cmd) = live_process_matching(pid, needle) {
                return Err(format!(
                    "Un'altra istanza è già attiva (pid {}: {}). Fermala oppure usa una porta diversa.",
                    pid, cmd
                ));
            }
        }
    }
    write_pid(name, me);
    Ok(())
}

/// Kills an orphaned child recorded under `name` if it is still alive and matches `needle`.
pub fn kill_orphan(name: &str, needle: &str) {
    let Some(pid) = read_pid(name) else {
        return;
    };
    if let Some(cmd) = live_process_matching(pid, needle) {
        warn!("Processo orfano trovato (pid {}: {}) -> lo termino", pid, cmd);
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
        std::thread::sleep(std::time::Duration::from_millis(300));
        info!("Processo orfano {} terminato", pid);
    }
    remove_pid(name);
}

/// True if a local TCP port is already bound by someone else.
pub fn tcp_port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
}

